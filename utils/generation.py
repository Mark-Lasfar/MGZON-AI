import os
import re
import json
from typing import List, Generator, Optional
from openai import OpenAI
from pydoc import html
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

# تعريف LATEX_DELIMS
LATEX_DELIMS = [
    {"left": "$$",  "right": "$$",  "display": True},
    {"left": "$",   "right": "$",   "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
FALLBACK_API_ENDPOINT = "https://api-inference.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:fireworks-ai")
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "MGZON/Veltrix")
TERTIARY_MODEL_NAME = os.getenv("TERTIARY_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")

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
    from utils.web_search import web_search  # تأخير الاستيراد
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
