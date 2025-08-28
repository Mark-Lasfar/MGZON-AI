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
    temperature: float = 0.9
    max_new_tokens: int = 128000
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
            return "Web search requires GOOGLE_API_KEY and GOOGLE_CSE_ID to be set."
        url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={query}+site:mgzon.com"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json().get("items", [])
        if not results:
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
        logger.exception("Web search failed")
        return f"Web search error: {e}"

# دالة request_generation
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def request_generation(
    api_key: str,
    api_base: str,
    message: str,
    system_prompt: str,
    model_name: str,
    chat_history: Optional[List[dict]] = None,
    temperature: float = 0.9,
    max_new_tokens: int = 128000,
    reasoning_effort: str = "off",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[str] = None,
    deep_search: bool = False,
) -> Generator[str, None, None]:
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
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.9),
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
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: #fff;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                overflow-x: hidden;
            }
            .container {
                max-width: 800px;
                text-align: center;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(10px);
            }
            .footer {
    text-align: center;
    padding: 2rem 0;
    background-color: var(--primary-color);
    border-top: 1px solid var(--container-border);
}

.footer__copy {
    font-size: 0.9rem;
    color: var(--second-color);
}

.footer__links {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    justify-content: center;
    flex-wrap: wrap;
}

/* Footer-specific styles */
.animate-gradient-x {
  background: linear-gradient(270deg, #1f2937, #111827, #1f2937);
  background-size: 200% 200%;
  animation: gradient 15s ease infinite;
}

@keyframes gradient {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

.footer-card {
  background: #374151;
  padding: 1.5rem;
  border-radius: 0.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
}

.footer-card:hover {
  transform: scale(1.05);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
  background: #4b5563;
}
.footer__links .footer__icon {
    color: var(--text-color);
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    transition: color 0.3s;
}

.footer__links .footer__icon:hover {
    color: var(--first-color);
}

            h1 {
                font-size: 3rem;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            p {
                font-size: 1.2rem;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            a {
                display: inline-block;
                padding: 15px 30px;
                background: #ff6f61;
                color: #fff;
                text-decoration: none;
                border-radius: 25px;
                font-size: 1.1rem;
                transition: background 0.3s, transform 0.2s;
            }
            a:hover {
                background: #e55a50;
                transform: scale(1.05);
            }
            .features, .integration {
                margin-top: 40px;
                text-align: left;
            }
            .features h2, .integration h2 {
                font-size: 1.8rem;
                margin-bottom: 15px;
            }
            .features ul, .integration pre {
                background: rgba(0, 0, 0, 0.2);
                padding: 20px;
                border-radius: 10px;
                font-size: 1rem;
            }
            .features li {
                margin-bottom: 10px;
            }
            pre {
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                color: #c9e4ca;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .container > * {
                animation: fadeIn 0.5s ease-in-out;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to MGZon Chatbot 🚀</h1>
            <p>
                MGZon Chatbot is your AI-powered assistant for code generation, analysis, and MGZon-specific queries. Built with Gradio and FastAPI, it supports multiple frameworks and languages. Ready to explore?
            </p>
            <a href="/gradio" id="chatbot-link">Launch Chatbot</a>
            <div class="features">
                <h2>Features</h2>
                <ul>
                    <li>💻 Generate code for React, Django, Flask, and more.</li>
                    <li>🔍 Analyze and review code or data with detailed insights.</li>
                    <li>🌐 Web search integration for MGZon-related queries.</li>
                    <li>🤖 Powered by GPT-OSS-20B and fine-tuned MGZon/Veltrix model.</li>
                </ul>
            </div>
            <div class="integration">
                <h2>Integrate with MGZon Chatbot</h2>
                <p>Use our API to integrate the chatbot into your projects. Supports Python, JavaScript, and more.</p>
                <pre>
# Python Example (using gradio_client)
from gradio_client import Client
client = Client("https://mgzon-mgzon-app.hf.space/gradio")
result = client.predict(
    message="Generate a React component",
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
        message: 'Generate a React component',
        system_prompt: 'You are a coding expert',
        temperature: 0.7,
        max_new_tokens: 4096
    })
}).then(response => response.json()).then(data => console.log(data));

// Bash Example (using curl)
curl -X POST https://mgzon-mgzon-app.hf.space/api/chat \
-H "Content-Type: application/json" \
-d '{"message":"Generate a React component","system_prompt":"You are a coding expert","temperature":0.7,"max_new_tokens":4096}'
                </pre>
            </div>
        </div>
        <footer class="bg-gray-800 p-8 mt-12 animate-gradient-x">
  <div class="container mx-auto px-6">
    <div class="mb-12 text-center">
      <div class="containerDown">
        <div class="last">
          <div class="head">Total Website Visits</div>
          <div id="visit-count">1930537</div>
        </div>
      </div>

      <img
        src="https://raw.githubusercontent.com/Mark-Lasfar/MGZon/9a1b2149507ae61fec3bb7fb86d8d16c11852f3b/public/icons/mg.svg"
        alt="MGZon Logo" class="w-24 h-24 mx-auto mb-4">

      <p id="footer-company-desc" class="text-gray-300 max-w-2xl mx-auto text-lg">MGZon is a leading platform for
        e-commerce integrations and API solutions.</p>
    </div>
    <div class="mb-12">
      <h3 id="contact-title" class="text-3xl font-bold mb-8 text-center text-white">Contact Information</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="contact_email_details">
          <div class="flex justify-center mb-4"><i class="bx bx-mail-send text-4xl text-blue-400"></i></div>
          <h4 id="contact-email-title" class="text-xl font-semibold text-center text-white">Email Us</h4>
          <p id="contact-email-desc" class="text-gray-300 text-center mt-2">Reach out to our support team via email.
          </p>
          <p id="contact-email-details" class="text-gray-400 text-center text-sm mt-2 hidden">Contact our support team
            at support@mgzon.com for assistance with technical issues, account management, or general inquiries. We
            aim to respond within 24 hours during business days (Monday to Friday, 9 AM - 5 PM EST). For faster
            resolution, include your account ID, order number, or a detailed description of your issue. You can also
            submit a ticket via our support portal at mgzon.com/support.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="contact_phone_details">
          <div class="flex justify-center mb-4"><i class="bx bx-phone text-4xl text-blue-400"></i></div>
          <h4 id="contact-phone-title" class="text-xl font-semibold text-center text-white">Phone Support</h4>
          <p id="contact-phone-desc" class="text-gray-300 text-center mt-2">Call us for immediate assistance.</p>
          <p id="contact-phone-details" class="text-gray-400 text-center text-sm mt-2 hidden">Reach us at
            +1-800-123-4567 for immediate support. Our phone lines are open Monday to Friday, 9 AM - 5 PM EST. For
            urgent issues outside these hours, use our email support or live chat feature. Enterprise clients can
            access 24/7 dedicated support by contacting their account manager.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="contact_hours_details">
          <div class="flex justify-center mb-4"><i class="bx bx-group text-4xl text-blue-400"></i></div>
          <h4 id="contact-hours-title" class="text-xl font-semibold text-center text-white">Support Hours</h4>
          <p id="contact-hours-desc" class="text-gray-300 text-center mt-2">Available 9 AM - 6 PM, Monday to Friday.
          </p>
          <p id="contact-hours-details" class="text-gray-400 text-center text-sm mt-2 hidden">Our support team is
            available Monday to Friday, 9 AM - 5 PM EST. Outside these hours, explore our self-service options,
            including our comprehensive FAQ, community forums, and knowledge base. Enterprise clients can request 24/7
            support through their dedicated account manager or by emailing enterprise@mgzon.com.</p>
        </div>
      </div>
    </div>
    <div class="mb-12">
      <h3 id="resources-title" class="text-3xl font-bold mb-8 text-center text-white">Resources</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="resources_api_docs_details">
          <div class="flex justify-center mb-4"><i class="bx bx-code-alt text-4xl text-green-400"></i></div>
          <h4 id="resources-api-docs-title" class="text-xl font-semibold text-center text-white">API Documentation
          </h4>
          <p id="resources-api-docs-desc" class="text-gray-300 text-center mt-2">Explore our API endpoints and
            integration guides.</p>
          <p id="resources-api-docs-details" class="text-gray-400 text-center text-sm mt-2 hidden">Our API
            documentation provides detailed guides for integrating MGZon services into your applications. Learn about
            OAuth 2.0 authentication, available endpoints (e.g., /api/v1/clients, /api/products/import), rate limits,
            and error handling. Includes code examples in Python, JavaScript, and cURL. Visit our developer portal at
            mgzon.com/developers.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="resources_faq_details">
          <div class="flex justify-center mb-4"><i class="bx bx-help-circle text-4xl text-green-400"></i></div>
          <h4 id="resources-faq-title" class="text-xl font-semibold text-center text-white">FAQ</h4>
          <p id="resources-faq-desc" class="text-gray-300 text-center mt-2">Find answers to common questions.</p>
          <p id="resources-faq-details" class="text-gray-400 text-center text-sm mt-2 hidden">Find answers to common
            questions about account setup, billing, API usage, and troubleshooting in our FAQ. Updated weekly based on
            user feedback, it covers topics like OAuth scope management, seller integrations, and payment processing.
            If your question isn’t answered, submit a ticket or join our community forums at mgzon.com/community.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="resources_docs_details">
          <div class="flex justify-center mb-4"><img
              src="https://github.blog/wp-content/uploads/2024/07/Icon-Circle.svg" alt="Docs Icon"
              class="w-10 h-10 text-green-400"></div>
          <h4 id="resources-docs-title" class="text-xl font-semibold text-center text-white">Documentation</h4>
          <p id="resources-docs-desc" class="text-gray-300 text-center mt-2">Learn how to use MGZon with our detailed
            guides.</p>
          <p id="resources-docs-details" class="text-gray-400 text-center text-sm mt-2 hidden">Our documentation
            offers step-by-step guides on using MGZon’s platform, covering account management, product
            imports/exports, advertising campaigns, and analytics. Available in multiple languages (English, Arabic,
            etc.) with video tutorials and interactive demos. Check out advanced topics like webhook setup and
            real-time notifications at mgzon.com/docs.</p>
        </div>
      </div>
    </div>
    <div class="mb-12">
      <h3 id="support-title" class="text-3xl font-bold mb-8 text-center text-white">Support</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="support_contact_us_details">
          <div class="flex justify-center mb-4"><i class="bx bx-mail-send text-4xl text-yellow-400"></i></div>
          <h4 id="support-contact-us-title" class="text-xl font-semibold text-center text-white">Contact Support</h4>
          <p id="support-contact-us-desc" class="text-gray-300 text-center mt-2">Get in touch with our support team.
          </p>
          <p id="support-contact-us-details" class="text-gray-400 text-center text-sm mt-2 hidden">Our dedicated
            support team is here to assist with technical issues, billing inquiries, or feature requests. Reach us via
            live chat (available on mgzon.com/support), email (support@mgzon.com), or phone (+1-800-123-4567). For
            complex issues, submit a ticket through our portal for detailed tracking and resolution.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="support_community_details">
          <div class="flex justify-center mb-4"><i class="bx bx-group text-4xl text-yellow-400"></i></div>
          <h4 id="support-community-title" class="text-xl font-semibold text-center text-white">Community</h4>
          <p id="support-community-desc" class="text-gray-300 text-center mt-2">Join our community forums for help and
            discussions.</p>
          <p id="support-community-details" class="text-gray-400 text-center text-sm mt-2 hidden">Join our global
            community of developers and users to share knowledge, collaborate on projects, and get peer support.
            Participate in our forums at mgzon.com/community, attend virtual meetups, or contribute to our open-source
            projects on GitHub (github.com/Mark-Lasfar/MGZon). Monthly webinars and Q&A sessions are also available.
          </p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="support_tickets_details">
          <div class="flex justify-center mb-4"><i class="bx bx-help-circle text-4xl text-yellow-400"></i></div>
          <h4 id="support-tickets-title" class="text-xl font-semibold text-center text-white">Support Tickets</h4>
          <p id="support-tickets-desc" class="text-gray-300 text-center mt-2">Submit a ticket for personalized
            assistance.</p>
          <p id="support-tickets-details" class="text-gray-400 text-center text-sm mt-2 hidden">Submit a support
            ticket for personalized assistance with complex issues, such as API errors or integration challenges.
            Track your ticket status in real-time via our support portal (mgzon.com/support/tickets). Premium users
            and enterprise clients receive priority support with SLA-backed response times.</p>
        </div>
      </div>
    </div>
    <div class="mb-12">
      <h3 id="about-title" class="text-3xl font-bold mb-8 text-center text-white">About MGZon</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="about_mgzon_details">
          <div class="flex justify-center mb-4"><i class="bx bx-info-circle text-4xl text-purple-400"></i></div>
          <h4 id="about-mgzon-title" class="text-xl font-semibold text-center text-white">Our Story</h4>
          <p id="about-mgzon-desc" class="text-gray-300 text-center mt-2">Learn about MGZon's journey and vision.</p>
          <p id="about-mgzon-details" class="text-gray-400 text-center text-sm mt-2 hidden">MGZon is a leading
            platform empowering businesses and developers with innovative tools for e-commerce, advertising, and
            analytics. Founded in 2023, we’ve grown to support thousands of sellers globally. Learn about our history,
            values, and commitment to privacy and security at mgzon.com/about.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="about_team_details">
          <div class="flex justify-center mb-4"><i class="bx bx-group text-4xl text-purple-400"></i></div>
          <h4 id="about-team-title" class="text-xl font-semibold text-center text-white">Our Team</h4>
          <p id="about-team-desc" class="text-gray-300 text-center mt-2">Meet the team behind MGZon.</p>
          <p id="about-team-details" class="text-gray-400 text-center text-sm mt-2 hidden">Our team of developers,
            designers, and support specialists is dedicated to your success. Meet our leadership, including our CTO
            and product managers, who bring decades of experience in tech and e-commerce. Visit mgzon.com/team to
            learn more about our expertise and open roles.</p>
        </div>
        <div
          class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
          data-details="about_mission_details">
          <div class="flex justify-center mb-4"><img
              src="https://raw.githubusercontent.com/Mark-Lasfar/MGZon/9a1b2149507ae61fec3bb7fb86d8d16c11852f3b/public/icons/mg.svg"
              alt="MGZon Logo" class="w-10 h-10 text-purple-400"></div>
          <h4 id="about-mission-title" class="text-xl font-semibold text-center text-white">Our Mission</h4>
          <p id="about-mission-desc" class="text-gray-300 text-center mt-2">Discover our mission to empower
            e-commerce.</p>
          <p id="about-mission-details" class="text-gray-400 text-center text-sm mt-2 hidden">Our mission is to drive
            innovation by providing scalable, secure, and accessible technology solutions. We empower businesses and
            developers to succeed through tools like our OAuth-based API, real-time analytics, and seamless
            integrations with platforms like Shopify and ShipBob. Learn more at mgzon.com/mission.</p>
        </div>
      </div>
    </div>
    <div class="mb-12">
      <h3 id="media-title" class="text-3xl font-bold mb-8 text-center text-white">Media</h3>
      <div id="media-section" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Articles will be added dynamically here -->
      </div>
    </div>
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      <div
        class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
        data-details="media_videos_details">
        <div class="flex justify-center mb-4"><i class="bx bx-video text-4xl text-red-400"></i></div>
        <h4 id="media-videos-title" class="text-xl font-semibold text-center text-white">Videos</h4>
        <p id="media-videos-desc" class="text-gray-300 text-center mt-2">Watch tutorials and guides on YouTube.</p>
        <p id="media-videos-details" class="text-gray-400 text-center text-sm mt-2 hidden">Explore our video library
          at mgzon.com/videos, featuring tutorials on API integration, product demos for sellers, and webinars on
          e-commerce trends. New content is added weekly, including beginner guides and advanced topics like real-time
          inventory syncing. Subscribe to our YouTube channel for updates.</p>
      </div>
      <div
        class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
        data-details="media_books_details">
        <div class="flex justify-center mb-4"><i class="bx bx-book text-4xl text-red-400"></i></div>
        <h4 id="media-books-title" class="text-xl font-semibold text-center text-white">Books</h4>
        <p id="media-books-desc" class="text-gray-300 text-center mt-2">Explore recommended books for e-commerce and
          APIs.</p>
        <p id="media-books-details" class="text-gray-400 text-center text-sm mt-2 hidden">Discover our curated list of
          books and eBooks on e-commerce, software development, and innovation. Titles include 'API Design Patterns'
          and 'E-commerce Essentials.' Access free resources or purchase through our partners at
          mgzon.com/resources/books.</p>
      </div>
      <div
        class="footer-card bg-gray-700 p-6 rounded-lg shadow-lg hover:bg-gray-600 transition-all duration-300 cursor-pointer card"
        data-details="media_articles_details">
        <div class="flex justify-center mb-4"><i class="bx bx-news text-4xl text-red-400"></i></div>
        <h4 id="media-articles-title" class="text-xl font-semibold text-center text-white">Articles</h4>
        <p id="media-articles-desc" class="text-gray-300 text-center mt-2">Read our latest articles and blog posts.
        </p>
        <p id="media-articles-details" class="text-gray-400 text-center text-sm mt-2 hidden">Read our blog at
          mgzon.com/blog for insights on industry trends, technical tutorials, and MGZon updates. Recent posts cover
          OAuth best practices, seller advertising strategies, and performance optimization for APIs. Contribute your
          own articles via our community portal.</p>
      </div>
    </div>
    <div class="mt-12 text-center">
      <div class="flex justify-center space-x-6 mb-6">
        <a href="https://github.com/Mark-Lasfar/MGZon" class="text-gray-300 hover:text-blue-400"><i
            class="bx bxl-github text-3xl"></i></a>
        <a href="https://x.com/MGZon" class="text-gray-300 hover:text-blue-400"><i
            class="bx bxl-twitter text-3xl"></i></a>
        <a href="https://www.facebook.com/people/Mark-Al-Asfar/pfbid02GMisUQ8AqWkNZjoKtWFHH1tbdHuVscN1cjcFnZWy9HkRaAsmanBfT6mhySAyqpg4l/"
          class="text-gray-300 hover:text-blue-400"><i class="bx bxl-facebook text-3xl"></i></a>
      </div>
      <p id="footer Rights-reserved" class="text-gray-300">© 2025 Mark Al-Asfar & MGZon AI. All rights reserved.</p>
    </div>
    <div id="footer-modal" class="modal">
      <div id="modal-content" class="modal-content">
        <span id="close-btn" class="close-btn">&times;</span>
        <h3 id="modal-title" class="text-2xl font-bold mb-4"></h3>
        <p id="modal-details" class="text-gray-300"></p>
      </div>
    </div>
  </div>
</footer>
        <script>
            // Redirect to /gradio on button click
            document.getElementById('chatbot-link').addEventListener('click', (e) => {
                e.preventDefault();
                window.location.href = '/gradio';
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
        max_new_tokens=128000,
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
        max_new_tokens=128000,
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