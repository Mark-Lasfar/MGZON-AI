import os
import json
import logging
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from openai import OpenAI
from pydoc import html
from typing import List, Generator, Optional
import requests
from bs4 import BeautifulSoup
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

# تعريف نموذج البيانات للـ API
class QueryRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant capable of code generation, analysis, review, and more."
    history: Optional[List[dict]] = None
    temperature: float = 0.7  # خفضنا الـ temperature عشان ردود أكثر دقة
    max_new_tokens: int = 4096  # قللنا العدد عشان نضمن ردود مكتملة
    enable_browsing: bool = False

# تعريف LATEX_DELIMS
LATEX_DELIMS = [
    {"left": "$$",  "right": "$$",  "display": True},
    {"left": "$",   "right": "$",   "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحقق من الملفات في /app/
logger.info("Files in /app/: %s", os.listdir("/app"))

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
FALLBACK_API_ENDPOINT = "https://api-inference.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:fireworks-ai")
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "MGZON/Veltrix")
TERTIARY_MODEL_NAME = os.getenv("TERTIARY_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")

if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

# إعدادات الـ queue
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 80))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 20))

# دالة اختيار النموذج
def select_model(query: str) -> tuple[str, str]:
    query_lower = query.lower()
    mgzon_patterns = [
        r"\bmgzon\b", r"\bmgzon\s+(products|services|platform|features|mission|technology|solutions|oauth)\b",
        r"\bميزات\s+mgzon\b", r"\bخدمات\s+mgzon\b", r"\boauth\b"
    ]
    for pattern in mgzon_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.info(f"Selected {SECONDARY_MODEL_NAME} with endpoint {FALLBACK_API_ENDPOINT} for MGZon-related query: {query}")
            return SECONDARY_MODEL_NAME, FALLBACK_API_ENDPOINT
    logger.info(f"Selected {MODEL_NAME} with endpoint {API_ENDPOINT} for general query: {query}")
    return MODEL_NAME, API_ENDPOINT

# دالة بحث ويب محسنة
def web_search(query: str) -> str:
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if not google_api_key or not google_cse_id:
            logger.warning("GOOGLE_API_KEY or GOOGLE_CSE_ID not set, web search disabled.")
            return "Web search requires GOOGLE_API_KEY and GOOGLE_CSE_ID to be set."
        url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={query}+site:mgzon.com"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json().get("items", [])
        if not results:
            logger.info("No web results found for query: %s", query)
            return "No web results found."
        search_results = []
        for i, item in enumerate(results[:5]):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            try:
                page_response = requests.get(link, timeout=5)
                page_response.raise_for_status()
                soup = BeautifulSoup(page_response.text, "html.parser")
                paragraphs = soup.find_all("p")
                page_content = " ".join([p.get_text() for p in paragraphs][:1000])
            except Exception as e:
                logger.warning(f"Failed to fetch page content for {link}: {e}")
                page_content = snippet
            search_results.append(f"Result {i+1}:\nTitle: {title}\nLink: {link}\nContent: {page_content}\n")
        return "\n".join(search_results)
    except Exception as e:
        logger.exception("Web search failed for query: %s", query)
        return f"Web search error: {e}"

# دالة request_generation مع logging إضافي
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def request_generation(
    api_key: str,
    api_base: str,
    message: str,
    system_prompt: str,
    model_name: str,
    chat_history: Optional[List[dict]] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 4096,
    reasoning_effort: str = "off",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[str] = None,
    deep_search: bool = False,
) -> Generator[str, None, None]:
    logger.info(f"Requesting generation for model: {model_name}, endpoint: {api_base}, message: {message}")
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=60.0)
    task_type = "general"
    if "code" in message.lower() or "programming" in message.lower() or any(ext in message.lower() for ext in ["python", "javascript", "react", "django", "flask"]):
        task_type = "code"
        enhanced_system_prompt = f"{system_prompt}\nYou are an expert programmer. Provide accurate, well-commented code with examples and explanations. Support frameworks like React, Django, Flask, and others as needed."
    elif any(keyword in message.lower() for keyword in ["analyze", "analysis", "تحليل"]):
        task_type = "analysis"
        enhanced_system_prompt = f"{system_prompt}\nProvide detailed analysis with step-by-step reasoning, examples, and data-driven insights."
    elif any(keyword in message.lower() for keyword in ["review", "مراجعة"]):
        task_type = "review"
        enhanced_system_prompt = f"{system_prompt}\nReview the provided content thoroughly, identify issues, and suggest improvements with detailed explanations."
    elif any(keyword in message.lower() for keyword in ["publish", "نشر"]):
        task_type = "publish"
        enhanced_system_prompt = f"{system_prompt}\nPrepare content for publishing, ensuring clarity, professionalism, and adherence to best practices."
    else:
        enhanced_system_prompt = system_prompt

    logger.info(f"Task type detected: {task_type}")
    input_messages: List[dict] = [{"role": "system", "content": enhanced_system_prompt}]
    if chat_history:
        for msg in chat_history:
            clean_msg = {"role": msg.get("role"), "content": msg.get("content")}
            if clean_msg["content"]:
                input_messages.append(clean_msg)
    
    if deep_search:
        search_result = web_search(message)
        input_messages.append({"role": "user", "content": f"User query: {message}\nWeb search context: {search_result}"})
    else:
        input_messages.append({"role": "user", "content": message})

    tools = tools if tools and "gpt-oss" in model_name else []
    tool_choice = tool_choice if tool_choice in ["auto", "none", "any", "required"] and "gpt-oss" in model_name else "none"

    try:
        logger.debug(f"Sending request to {api_base} with model {model_name}")
        stream = client.chat.completions.create(
            model=model_name,
            messages=input_messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
        )

        reasoning_started = False
        reasoning_closed = False
        saw_visible_output = False
        last_tool_name = None
        last_tool_args = None
        buffer = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if content == "<|channel|>analysis<|message|>":
                    if not reasoning_started:
                        yield "analysis"
                        reasoning_started = True
                    continue
                if content == "<|channel|>final<|message|>":
                    if reasoning_started and not reasoning_closed:
                        yield "assistantfinal"
                        reasoning_closed = True
                    continue

                saw_visible_output = True
                buffer += content

                if "\n" in buffer or len(buffer) > 2000:
                    yield buffer
                    buffer = ""
                continue

            if chunk.choices[0].delta.tool_calls and "gpt-oss" in model_name:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                name = getattr(tool_call, "function", {}).get("name", None)
                args = getattr(tool_call, "function", {}).get("arguments", None)
                if name:
                    last_tool_name = name
                if args:
                    last_tool_args = args
                continue

            if chunk.choices[0].finish_reason in ("stop", "tool_calls", "error"):
                if buffer:
                    yield buffer
                    buffer = ""

                if reasoning_started and not reasoning_closed:
                    yield "assistantfinal"
                    reasoning_closed = True

                if not saw_visible_output:
                    msg = "I attempted to call a tool, but tools aren't executed in this environment, so no final answer was produced."
                    if last_tool_name:
                        try:
                            args_text = json.dumps(last_tool_args, ensure_ascii=False, default=str)
                        except Exception:
                            args_text = str(last_tool_args)
                        msg += f"\n\n• Tool requested: **{last_tool_name}**\n• Arguments: `{args_text}`"
                    yield msg

                if chunk.choices[0].finish_reason == "error":
                    yield f"Error: Unknown error"
                break

        if buffer:
            yield buffer
        logger.info(f"Generation completed successfully for model: {model_name}")

    except Exception as e:
        logger.exception(f"[Gateway] Streaming failed for model {model_name}: {e}")
        if model_name == MODEL_NAME:
            fallback_model = SECONDARY_MODEL_NAME
            fallback_endpoint = FALLBACK_API_ENDPOINT
            logger.info(f"Retrying with fallback model: {fallback_model} on {fallback_endpoint}")
            try:
                client = OpenAI(api_key=api_key, base_url=fallback_endpoint, timeout=60.0)
                stream = client.chat.completions.create(
                    model=fallback_model,
                    messages=input_messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    stream=True,
                    tools=[],
                    tool_choice="none",
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        if content == "<|channel|>analysis<|message|>":
                            if not reasoning_started:
                                yield "analysis"
                                reasoning_started = True
                            continue
                        if content == "<|channel|>final<|message|>":
                            if reasoning_started and not reasoning_closed:
                                yield "assistantfinal"
                                reasoning_closed = True
                            continue

                        saw_visible_output = True
                        buffer += content

                        if "\n" in buffer or len(buffer) > 2000:
                            yield buffer
                            buffer = ""
                        continue

                    if chunk.choices[0].finish_reason in ("stop", "error"):
                        if buffer:
                            yield buffer
                            buffer = ""

                        if reasoning_started and not reasoning_closed:
                            yield "assistantfinal"
                            reasoning_closed = True

                        if not saw_visible_output:
                            yield "No visible output produced."
                        if chunk.choices[0].finish_reason == "error":
                            yield f"Error: Unknown error with fallback model {fallback_model}"
                        break

                if buffer:
                    yield buffer
                logger.info(f"Fallback generation completed successfully for model: {fallback_model}")

            except Exception as e2:
                logger.exception(f"[Gateway] Streaming failed for fallback model {fallback_model}: {e2}")
                yield f"Error: Failed to load both models ({model_name} and {fallback_model}): {e2}"
                # تجربة النموذج الثالث
                try:
                    client = OpenAI(api_key=api_key, base_url=FALLBACK_API_ENDPOINT, timeout=60.0)
                    stream = client.chat.completions.create(
                        model=TERTIARY_MODEL_NAME,
                        messages=input_messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        stream=True,
                        tools=[],
                        tool_choice="none",
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            saw_visible_output = True
                            buffer += content
                            if "\n" in buffer or len(buffer) > 2000:
                                yield buffer
                                buffer = ""
                            continue
                        if chunk.choices[0].finish_reason in ("stop", "error"):
                            if buffer:
                                yield buffer
                                buffer = ""
                            if not saw_visible_output:
                                yield "No visible output produced."
                            if chunk.choices[0].finish_reason == "error":
                                yield f"Error: Unknown error with tertiary model {TERTIARY_MODEL_NAME}"
                            break
                    if buffer:
                        yield buffer
                    logger.info(f"Tertiary generation completed successfully for model: {TERTIARY_MODEL_NAME}")

                except Exception as e3:
                    logger.exception(f"[Gateway] Streaming failed for tertiary model {TERTIARY_MODEL_NAME}: {e3}")
                    yield f"Error: Failed to load all models: {e3}"
        else:
            yield f"Error: Failed to load model {model_name}: {e}"

# وظيفة التنسيق النهائي
def format_final(analysis_text: str, visible_text: str) -> str:
    reasoning_safe = html.escape((analysis_text or "").strip())
    response = (visible_text or "").strip()
    return (
        "<details><summary><strong>🤔 Analysis</strong></summary>\n"
        "<pre style='white-space:pre-wrap;'>"
        f"{reasoning_safe}"
        "</pre>\n</details>\n\n"
        "**💬 Response:**\n\n"
        f"{response}"
    )

# وظيفة التوليد مع محاكاة streaming
def generate(message, history, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens):
    if not message.strip():
        yield "Please enter a prompt."
        return

    model_name, api_endpoint = select_model(message)
    chat_history = []
    for h in history:
        if isinstance(h, dict):
            clean_msg = {"role": h.get("role"), "content": h.get("content")}
            if clean_msg["content"]:
                chat_history.append(clean_msg)
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            u, a = h
            if u: chat_history.append({"role": "user", "content": u})
            if a: chat_history.append({"role": "assistant", "content": a})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search_preview",
                "description": "Perform a web search to gather additional context",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "code_generation",
                "description": "Generate or modify code for various frameworks (React, Django, Flask, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Existing code to modify or empty for new code"},
                        "framework": {"type": "string", "description": "Framework (e.g., React, Django, Flask)"},
                        "task": {"type": "string", "description": "Task description (e.g., create a component, fix a bug)"},
                    },
                    "required": ["task"],
                },
            },
        }
    ] if "gpt-oss" in model_name else []
    tool_choice = "auto" if "gpt-oss" in model_name else "none"

    in_analysis = False
    in_visible = False
    raw_analysis = ""
    raw_visible = ""
    raw_started = False
    last_flush_len = 0

    def make_raw_preview() -> str:
        return (
            "```text\n"
            "Analysis (live):\n"
            f"{raw_analysis}\n\n"
            "Response (draft):\n"
            f"{raw_visible}\n"
            "```"
        )

    try:
        stream = request_generation(
            api_key=HF_TOKEN,
            api_base=api_endpoint,
            message=message,
            system_prompt=system_prompt,
            model_name=model_name,
            chat_history=chat_history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tools=tools,
            tool_choice=tool_choice,
            deep_search=enable_browsing,
        )

        for chunk in stream:
            if chunk == "analysis":
                in_analysis, in_visible = True, False
                if not raw_started:
                    raw_started = True
                    yield make_raw_preview()
                continue
            if chunk == "assistantfinal":
                in_analysis, in_visible = False, True
                if not raw_started:
                    raw_started = True
                    yield make_raw_preview()
                continue

            if in_analysis:
                raw_analysis += chunk
            elif in_visible:
                raw_visible += chunk
            else:
                raw_visible += chunk

            total_len = len(raw_analysis) + len(raw_visible)
            if total_len - last_flush_len >= 120 or "\n" in chunk:
                last_flush_len = total_len
                yield make_raw_preview()

        final_markdown = format_final(raw_analysis, raw_visible)
        if final_markdown.count("$") % 2:
            final_markdown += "$"
        yield final_markdown

    except Exception as e:
        logger.exception("Stream failed")
        yield f"❌ Error: {e}"

# إعداد CSS
css = """
.gradio-container { max-width: 800px; margin: auto; }
.chatbot { border: 1px solid #ccc; border-radius: 10px; }
.input-textbox { font-size: 16px; }
"""

# إعداد واجهة Gradio
chatbot_ui = gr.ChatInterface(
    fn=generate,
    type="messages",
    chatbot=gr.Chatbot(
        label="MGZon Chatbot",
        type="messages",
        height=600,
        latex_delimiters=LATEX_DELIMS,
    ),
    additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=True),
    additional_inputs=[
        gr.Textbox(label="System prompt", value="You are a helpful assistant capable of code generation, analysis, review, and more.", lines=2),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7),
        gr.Radio(label="Reasoning Effort", choices=["low", "medium", "high"], value="medium"),
        gr.Checkbox(label="Enable DeepSearch (web browsing)", value=True),
        gr.Slider(label="Max New Tokens", minimum=50, maximum=128000, step=50, value=4096),
    ],
    stop_btn="Stop",
    examples=[
        ["Explain the difference between supervised and unsupervised learning."],
        ["Generate a React component for a login form."],
        ["Review this Python code: print('Hello World')"],
        ["Analyze the performance of a Django REST API."],
        ["Tell me about MGZon products and services."],
        ["Create a Flask route for user authentication."],
        ["What are the latest trends in AI?"],
        ["Provide guidelines for publishing a technical blog post."],
        ["Who is the founder of MGZon?"],
    ],
    title="MGZon Chatbot",
    description="A versatile chatbot powered by GPT-OSS-20B and a fine-tuned model for MGZon queries. Supports code generation, analysis, review, web search, and MGZon-specific queries. Licensed under Apache 2.0. ***DISCLAIMER:*** Analysis may contain internal thoughts not suitable for final response.",
    theme="gradio/soft",
    css=css,
)

# دمج FastAPI مع Gradio
app = FastAPI(title="MGZon Chatbot API")

# ربط Gradio مع FastAPI
app = gr.mount_gradio_app(app, chatbot_ui, path="/gradio")

# إضافة endpoint للـ root
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>MGZon Chatbot - Powered by AI</title>
        <link href="https://cdn.jsdelivr.net/npm/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-color: #1e3c72;
                --secondary-color: #2a5298;
                --accent-color: #ff6f61;
                --text-color: #fff;
                --container-bg: rgba(255, 255, 255, 0.1);
                --container-border: rgba(255, 255, 255, 0.2);
            }
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: var(--text-color);
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                overflow-x: hidden;
            }
            .container {
                max-width: 900px;
                text-align: center;
                padding: 40px;
                background: var(--container-bg);
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(10px);
                margin: 20px;
            }
            h1 {
                font-size: 3.5rem;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            p {
                font-size: 1.2rem;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            a.cta-button {
                display: inline-block;
                padding: 15px 40px;
                background: var(--accent-color);
                color: var(--text-color);
                text-decoration: none;
                border-radius: 25px;
                font-size: 1.2rem;
                font-weight: bold;
                transition: background 0.3s, transform 0.2s;
            }
            a.cta-button:hover {
                background: #e55a50;
                transform: scale(1.05);
            }
            .features, .integration {
                margin-top: 40px;
                text-align: left;
            }
            .features h2, .integration h2 {
                font-size: 2rem;
                margin-bottom: 20px;
                text-align: center;
            }
            .features ul {
                list-style: none;
                padding: 0;
                background: rgba(0, 0, 0, 0.2);
                padding: 20px;
                border-radius: 10px;
            }
            .features li {
                margin-bottom: 15px;
                font-size: 1.1rem;
                display: flex;
                align-items: center;
            }
            .features li::before {
                content: '🚀';
                margin-right: 10px;
            }
            .integration pre {
                background: rgba(0, 0, 0, 0.3);
                padding: 20px;
                border-radius: 10px;
                font-family: 'Courier New', monospace;
                color: #c9e4ca;
                overflow-x: auto;
            }
            .footer {
                text-align: center;
                padding: 20px;
                background: rgba(0, 0, 0, 0.5);
                width: 100%;
                margin-top: 40px;
            }
            .footer a {
                color: var(--accent-color);
                text-decoration: none;
                margin: 0 10px;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .container > * {
                animation: fadeIn 0.5s ease-in-out;
            }
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            body {
                background-size: 200% 200%;
                animation: gradient 15s ease infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to MGZon Chatbot 🚀</h1>
            <p>
                MGZon Chatbot is your AI-powered assistant for code generation, analysis, and MGZon-specific queries. Built with Gradio and FastAPI, it supports multiple frameworks (React, Django, Flask) and languages (Python, JavaScript, and more). Ready to revolutionize your projects?
            </p>
            <a href="/gradio" class="cta-button">Launch Chatbot</a>
            <div class="features">
                <h2>Features</h2>
                <ul>
                    <li>💻 Generate clean, well-commented code for React, Django, Flask, and more.</li>
                    <li>🔍 Analyze code or data with step-by-step reasoning and insights.</li>
                    <li>🌐 Web search integration for real-time MGZon-related information.</li>
                    <li>🤖 Powered by GPT-OSS-20B and fine-tuned MGZon/Veltrix model.</li>
                    <li>📝 Review and improve your code with detailed suggestions.</li>
                </ul>
            </div>
            <div class="integration">
                <h2>Integrate with MGZon Chatbot</h2>
                <p>
                    Integrate our API into your projects with ease. Below are examples for Python, JavaScript, and cURL, supporting frameworks like React, Django, Flask, and more. Use it for web apps, mobile apps, or CLI tools.
                </p>
                <pre>
# Python Example (using gradio_client)
from gradio_client import Client
client = Client("https://mgzon-mgzon-app.hf.space/gradio")
result = client.predict(
    message="Generate a React component for a login form",
    system_prompt="You are a coding expert",
    temperature=0.7,
    max_new_tokens=4096,
    api_name="/api/chat"
)
print(result)

# JavaScript Example (using fetch)
fetch('https://mgzon-mgzon-app.hf.space/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: 'Generate a Flask route for user authentication',
        system_prompt: 'You are a coding expert',
        temperature: 0.7,
        max_new_tokens: 4096
    })
}).then(response => response.json()).then(data => console.log(data));

// Bash Example (using cURL)
curl -X POST https://mgzon-mgzon-app.hf.space/api/chat \
-H "Content-Type: application/json" \
-d '{"message":"Analyze the performance of a Django REST API","system_prompt":"You are an expert analyst","temperature":0.7,"max_new_tokens":4096}'

# Django Integration Example
# Add to your Django views.py
import requests
def chatbot_view(request):
    response = requests.post(
        'https://mgzon-mgzon-app.hf.space/api/chat',
        json={
            'message': 'Create a Django model for an e-commerce product',
            'system_prompt': 'You are a Django expert',
            'temperature': 0.7,
            'max_new_tokens': 4096
        }
    )
    return HttpResponse(response.json()['response'])

# React Integration Example
import React, { useState } from 'react';
function ChatbotComponent() {
    const [response, setResponse] = useState('');
    const fetchResponse = async () => {
        const res = await fetch('https://mgzon-mgzon-app.hf.space/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: 'Generate a React component for a dashboard',
                system_prompt: 'You are a React expert',
                temperature: 0.7,
                max_new_tokens: 4096
            })
        });
        const data = await res.json();
        setResponse(data.response);
    };
    return (
        <div>
            <button onClick={fetchResponse}>Generate Code</button>
            <pre>{response}</pre>
        </div>
    );
}
                </pre>
            </div>
        </div>
        <footer class="footer">
            <p>© 2025 MGZon AI. All rights reserved.</p>
            <p>
                <a href="https://mgzon.com/about">About</a> |
                <a href="https://mgzon.com/support">Support</a> |
                <a href="https://mgzon.com/docs">Documentation</a> |
                <a href="https://github.com/Mark-Lasfar/MGZon">GitHub</a>
            </p>
        </footer>
        <script>
            // إصلاح الـ redirect
            document.querySelector('.cta-button').addEventListener('click', (e) => {
                e.preventDefault();
                window.location.href = window.location.origin + '/gradio/';
            });
        </script>
    </body>
    </html>
    """

# API endpoints
@app.get("/api/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "secondary_model": SECONDARY_MODEL_NAME,
        "tertiary_model": TERTIARY_MODEL_NAME,
        "api_base": API_ENDPOINT,
        "status": "online"
    }

@app.get("/api/performance")
async def performance_stats():
    return {
        "queue_size": QUEUE_SIZE,
        "concurrency_limit": CONCURRENCY_LIMIT,
        "uptime": os.popen("uptime").read().strip()
    }

@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    model_name, api_endpoint = select_model(req.message)
    stream = request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message=req.message,
        system_prompt=req.system_prompt,
        model_name=model_name,
        chat_history=req.history,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
        deep_search=req.enable_browsing,
    )
    response = "".join(list(stream))
    return {"response": response}

@app.post("/api/code")
async def code_endpoint(req: dict):
    framework = req.get("framework")
    task = req.get("task")
    code = req.get("code", "")
    prompt = f"Generate code for task: {task} using {framework}. Existing code: {code}"
    model_name, api_endpoint = select_model(prompt)
    response = "".join(list(request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message=prompt,
        system_prompt="You are a coding expert.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=4096,
    )))
    return {"generated_code": response}

@app.post("/api/analysis")
async def analysis_endpoint(req: dict):
    message = req.get("text", "")
    model_name, api_endpoint = select_model(message)
    response = "".join(list(request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message=message,
        system_prompt="You are an expert analyst. Provide detailed analysis with step-by-step reasoning.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=4096,
    )))
    return {"analysis": response}

@app.get("/api/test-model")
async def test_model(model: str = MODEL_NAME, endpoint: str = API_ENDPOINT):
    try:
        client = OpenAI(api_key=HF_TOKEN, base_url=endpoint, timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=50
        )
        return {"status": "success", "response": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))