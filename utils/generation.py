import os
import re
import json
from typing import List, Generator, Optional
from openai import OpenAI
from pydoc import html
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from cachetools import TTLCache
import hashlib
import requests
import pydub
import io
import torchaudio
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from utils.web_search import web_search  # استيراد مباشر

logger = logging.getLogger(__name__)

# إعداد Cache
cache = TTLCache(maxsize=100, ttl=600)

# تعريف LATEX_DELIMS
LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# إعداد العميل لـ Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
FALLBACK_API_ENDPOINT = "https://api-inference.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:together")
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:featherless-ai")
TERTIARY_MODEL_NAME = os.getenv("TERTIARY_MODEL_NAME", "openai/gpt-oss-120b:cerebras")
CLIP_BASE_MODEL = os.getenv("CLIP_BASE_MODEL", "openai/clip-vit-base-patch32")
CLIP_LARGE_MODEL = os.getenv("CLIP_LARGE_MODEL", "openai/clip-vit-large-patch14")
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "parler-tts/parler-tts-mini-v1")

def check_model_availability(model_name: str, api_base: str, api_key: str) -> tuple[bool, str]:
    try:
        response = requests.get(
            f"{api_base}/models/{model_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            return True, api_key
        elif response.status_code == 429 and BACKUP_HF_TOKEN and api_key != BACKUP_HF_TOKEN:
            logger.warning(f"Rate limit reached for token {api_key}. Switching to backup token.")
            return check_model_availability(model_name, api_base, BACKUP_HF_TOKEN)
        logger.error(f"Model {model_name} not available: {response.status_code}")
        return False, api_key
    except Exception as e:
        logger.error(f"Failed to check model availability for {model_name}: {e}")
        if BACKUP_HF_TOKEN and api_key != BACKUP_HF_TOKEN:
            logger.warning(f"Retrying with backup token for {model_name}")
            return check_model_availability(model_name, api_base, BACKUP_HF_TOKEN)
        return False, api_key

def select_model(query: str, input_type: str = "text") -> tuple[str, str]:
    query_lower = query.lower()
    if input_type == "audio" or any(keyword in query_lower for keyword in ["voice", "audio", "speech", "صوت", "تحويل صوت"]):
        logger.info(f"Selected {ASR_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for audio input")
        return ASR_MODEL, FALLBACK_API_ENDPOINT
    if any(keyword in query_lower for keyword in ["text-to-speech", "tts", "تحويل نص إلى صوت"]):
        logger.info(f"Selected {TTS_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for text-to-speech")
        return TTS_MODEL, FALLBACK_API_ENDPOINT
    if input_type == "image" or any(pattern in query_lower for pattern in [
        r"\bimage\b", r"\bpicture\b", r"\bphoto\b", r"\bvisual\b", r"\bصورة\b", r"\bتحليل\s+صورة\b",
        r"\bimage\s+analysis\b", r"\bimage\s+classification\b", r"\bimage\s+description\b"
    ]):
        logger.info(f"Selected {CLIP_BASE_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for image-related query: {query}")
        return CLIP_BASE_MODEL, FALLBACK_API_ENDPOINT
    logger.info(f"Selected {MODEL_NAME} with endpoint {API_ENDPOINT} for general query: {query}")
    return MODEL_NAME, API_ENDPOINT

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
def request_generation(
    api_key: str,
    api_base: str,
    message: str,
    system_prompt: str,
    model_name: str,
    chat_history: Optional[List[dict]] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 128000,
    reasoning_effort: str = "off",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[str] = None,
    deep_search: bool = False,
    input_type: str = "text",
    audio_data: Optional[bytes] = None,
    image_data: Optional[bytes] = None,
) -> Generator[bytes | str, None, None]:
    is_available, selected_api_key = check_model_availability(model_name, api_base, api_key)
    if not is_available:
        yield f"Error: Model {model_name} is not available. Please check the model endpoint or token."
        return

    cache_key = hashlib.md5(json.dumps({
        "message": message,
        "system_prompt": system_prompt,
        "model_name": model_name,
        "chat_history": chat_history,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens
    }, sort_keys=True).encode()).hexdigest()

    if cache_key in cache:
        logger.info(f"Cache hit for query: {message[:50]}...")
        for chunk in cache[cache_key]:
            yield chunk
        return

    client = OpenAI(api_key=selected_api_key, base_url=api_base, timeout=120.0)
    task_type = "general"
    enhanced_system_prompt = system_prompt

    if model_name == ASR_MODEL and audio_data is not None:
        task_type = "audio_transcription"
        try:
            audio_file = io.BytesIO(audio_data)
            audio = pydub.AudioSegment.from_file(audio_file)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio_file = io.BytesIO()
            audio.export(audio_file, format="wav")
            audio_file.name = "audio.wav"
            transcription = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                response_format="text"
            )
            yield transcription
            cache[cache_key] = [transcription]
            return
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            yield f"Error: Audio transcription failed: {e}"
            return

    if model_name == TTS_MODEL:
        task_type = "text_to_speech"
        try:
            model = ParlerTTSForConditionalGeneration.from_pretrained(model_name, token=selected_api_key)
            processor = AutoProcessor.from_pretrained(model_name, token=selected_api_key)
            inputs = processor(text=message, return_tensors="pt")
            audio = model.generate(**inputs)
            audio_file = io.BytesIO()
            torchaudio.save(audio_file, audio[0], sample_rate=22050, format="wav")
            audio_file.seek(0)
            yield audio_file.read()
            cache[cache_key] = [audio_file.read()]
            return
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            yield f"Error: Text-to-speech failed: {e}"
            return

    if model_name in [CLIP_BASE_MODEL, CLIP_LARGE_MODEL] and image_data is not None:
        task_type = "image_analysis"
        try:
            model = CLIPModel.from_pretrained(model_name, token=selected_api_key)
            processor = CLIPProcessor.from_pretrained(model_name, token=selected_api_key)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            inputs = processor(text=message, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            yield f"Image analysis result: {probs.tolist()}"
            cache[cache_key] = [f"Image analysis result: {probs.tolist()}"]
            return
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            yield f"Error: Image analysis failed: {e}"
            return

    if model_name in [CLIP_BASE_MODEL, CLIP_LARGE_MODEL]:
        task_type = "image"
        enhanced_system_prompt = f"{system_prompt}\nYou are an expert in image analysis and description. Provide detailed descriptions, classifications, or analysis of images based on the query."
    elif any(keyword in message.lower() for keyword in ["code", "programming", "python", "javascript", "react", "django", "flask"]):
        task_type = "code"
        enhanced_system_prompt = f"{system_prompt}\nYou are an expert programmer. Provide accurate, well-commented code with comprehensive examples and detailed explanations."
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
        enhanced_system_prompt = f"{system_prompt}\nFor general queries, provide comprehensive, detailed responses with examples and explanations where applicable."

    if len(message.split()) < 5:
        enhanced_system_prompt += "\nEven for short queries, provide a detailed, in-depth response with examples and context."

    logger.info(f"Task type detected: {task_type}")
    input_messages: List[dict] = [{"role": "system", "content": enhanced_system_prompt}]
    if chat_history:
        for msg in chat_history:
            clean_msg = {"role": msg.get("role"), "content": msg.get("content")}
            if clean_msg["content"]:
                input_messages.append(clean_msg)
    
    if deep_search:
        try:
            search_result = web_search(message)
            input_messages.append({"role": "user", "content": f"User query: {message}\nWeb search context: {search_result}"})
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            input_messages.append({"role": "user", "content": message})
    else:
        input_messages.append({"role": "user", "content": message})

    tools = tools if tools and model_name in [MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME] else []
    tool_choice = tool_choice if tool_choice in ["auto", "none", "any", "required"] and model_name in [MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME] else "none"

    cached_chunks = []
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
                        cached_chunks.append("analysis")
                        yield "analysis"
                        reasoning_started = True
                    continue
                if content == "<|channel|>final<|message|>":
                    if reasoning_started and not reasoning_closed:
                        cached_chunks.append("assistantfinal")
                        yield "assistantfinal"
                        reasoning_closed = True
                    continue

                saw_visible_output = True
                buffer += content

                if "\n" in buffer or len(buffer) > 5000:
                    cached_chunks.append(buffer)
                    yield buffer
                    buffer = ""
                continue

            if chunk.choices[0].delta.tool_calls and model_name in [MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME]:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                name = getattr(tool_call, "function", {}).get("name", None)
                args = getattr(tool_call, "function", {}).get("arguments", None)
                if name:
                    last_tool_name = name
                if args:
                    last_tool_args = args
                continue

            if chunk.choices[0].finish_reason in ("stop", "tool_calls", "error", "length"):
                if buffer:
                    cached_chunks.append(buffer)
                    yield buffer
                    buffer = ""

                if reasoning_started and not reasoning_closed:
                    cached_chunks.append("assistantfinal")
                    yield "assistantfinal"
                    reasoning_closed = True

                if not saw_visible_output:
                    msg = "I attempted to call a tool, but tools aren't executed in this environment."
                    if last_tool_name:
                        try:
                            args_text = json.dumps(last_tool_args, ensure_ascii=False, default=str)
                        except Exception:
                            args_text = str(last_tool_args)
                        msg += f"\n\n• Tool requested: **{last_tool_name}**\n• Arguments: `{args_text}`"
                    cached_chunks.append(msg)
                    yield msg

                if chunk.choices[0].finish_reason == "error":
                    cached_chunks.append(f"Error: Unknown error")
                    yield f"Error: Unknown error"
                elif chunk.choices[0].finish_reason == "length":
                    cached_chunks.append("Response truncated due to token limit. Please refine your query.")
                    yield "Response truncated due to token limit. Please refine your query."
                break

        if buffer:
            cached_chunks.append(buffer)
            yield buffer

        cache[cache_key] = cached_chunks

    except Exception as e:
        logger.exception(f"[Gateway] Streaming failed for model {model_name}: {e}")
        if selected_api_key != BACKUP_HF_TOKEN and BACKUP_HF_TOKEN:
            logger.warning(f"Retrying with backup token for model {model_name}")
            for chunk in request_generation(
                api_key=BACKUP_HF_TOKEN,
                api_base=api_base,
                message=message,
                system_prompt=system_prompt,
                model_name=model_name,
                chat_history=chat_history,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                reasoning_effort=reasoning_effort,
                tools=tools,
                tool_choice=tool_choice,
                deep_search=deep_search,
                input_type=input_type,
                audio_data=audio_data,
                image_data=image_data,
            ):
                yield chunk
            return
        for fallback_model in [SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME]:
            logger.info(f"Retrying with fallback model: {fallback_model}")
            try:
                is_available, selected_api_key = check_model_availability(fallback_model, FALLBACK_API_ENDPOINT, selected_api_key)
                if not is_available:
                    continue
                client = OpenAI(api_key=selected_api_key, base_url=FALLBACK_API_ENDPOINT, timeout=120.0)
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
                        saw_visible_output = True
                        buffer += content
                        if "\n" in buffer or len(buffer) > 5000:
                            cached_chunks.append(buffer)
                            yield buffer
                            buffer = ""
                        continue
                    if chunk.choices[0].finish_reason in ("stop", "error", "length"):
                        if buffer:
                            cached_chunks.append(buffer)
                            yield buffer
                            buffer = ""
                        if not saw_visible_output:
                            cached_chunks.append("No visible output produced.")
                            yield "No visible output produced."
                        if chunk.choices[0].finish_reason == "error":
                            cached_chunks.append(f"Error: Unknown error with fallback model {fallback_model}")
                            yield f"Error: Unknown error with fallback model {fallback_model}"
                        elif chunk.choices[0].finish_reason == "length":
                            cached_chunks.append("Response truncated due to token limit.")
                            yield "Response truncated due to token limit."
                        break
                if buffer:
                    cached_chunks.append(buffer)
                    yield buffer
                cache[cache_key] = cached_chunks
                return
            except Exception as e2:
                logger.exception(f"[Gateway] Streaming failed for fallback model {fallback_model}: {e2}")
                continue
        yield f"Error: Failed to load all models: Primary ({model_name}), Secondary ({SECONDARY_MODEL_NAME}), Tertiary ({TERTIARY_MODEL_NAME})."
        return

def format_final(analysis_text: str, visible_text: str) -> str:
    reasoning_safe = html.escape((analysis_text or "").strip())
    response = (visible_text or "").strip()
    if not reasoning_safe and not response:
        return "No response generated."
    return (
        "<details><summary><strong>🤔 Analysis</strong></summary>\n"
        "<pre style='white-space:pre-wrap;'>"
        f"{reasoning_safe}"
        "</pre>\n</details>\n\n"
        "**💬 Response:**\n\n"
        f"{response}" if response else "No final response available."
    )

def generate(message, history, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, input_type="text", audio_data=None, image_data=None):
    if not message.strip() and not audio_data and not image_data:
        yield "Please enter a prompt or upload a file."
        return

    model_name, api_endpoint = select_model(message, input_type=input_type)
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
                "description": "Generate or modify code for various frameworks",
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
        },
        {
            "type": "function",
            "function": {
                "name": "code_formatter",
                "description": "Format code for readability and consistency",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to format"},
                        "language": {"type": "string", "description": "Programming language (e.g., Python, JavaScript)"},
                    },
                    "required": ["code", "language"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_analysis",
                "description": "Analyze or describe an image based on the provided query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string", "description": "URL of the image to analyze"},
                        "task": {"type": "string", "description": "Task description (e.g., describe, classify)"},
                    },
                    "required": ["task"],
                },
            },
        }
    ] if model_name in [MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME] else []
    tool_choice = "auto" if model_name in [MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME] else "none"

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
            input_type=input_type,
            audio_data=audio_data,
            image_data=image_data,
        )

        for chunk in stream:
            if isinstance(chunk, bytes):
                yield chunk
                continue
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
