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
from utils.web_search import web_search
from huggingface_hub import snapshot_download
import torch
from diffusers import DiffusionPipeline
from utils.constants import MODEL_ALIASES, MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME, CLIP_BASE_MODEL, CLIP_LARGE_MODEL, ASR_MODEL, TTS_MODEL, IMAGE_GEN_MODEL, SECONDARY_IMAGE_GEN_MODEL

logger = logging.getLogger(__name__)

# إعداد Cache
cache = TTLCache(maxsize=int(os.getenv("QUEUE_SIZE", 100)), ttl=600)

# تعريف LATEX_DELIMS
LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# إعداد العميل لـ Hugging Face API
HF_TOKEN = os.getenv("HF_TOKEN")
BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
ROUTER_API_URL = os.getenv("ROUTER_API_URL", "https://router.huggingface.co")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
FALLBACK_API_ENDPOINT = os.getenv("FALLBACK_API_ENDPOINT", "https://api-inference.huggingface.co/v1")

# تحميل نموذج FLUX.1-dev مسبقًا إذا لزم الأمر
model_path = None
try:
    model_path = snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_type="model",
        ignore_patterns=["*.md", "*..gitattributes"],
        local_dir="FLUX.1-dev",
    )
except Exception as e:
    logger.error(f"Failed to download FLUX.1-dev: {e}")
    model_path = None

# تعطيل PROVIDER_ENDPOINTS لأننا بنستخدم Hugging Face فقط
PROVIDER_ENDPOINTS = {
    "huggingface": API_ENDPOINT
}

def check_model_availability(model_name: str, api_key: str) -> tuple[bool, str, str]:
    try:
        response = requests.get(
            f"{ROUTER_API_URL}/v1/models/{model_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )
        logger.debug(f"Checking model {model_name}: {response.status_code} - {response.text}")
        if response.status_code == 200:
            logger.info(f"Model {model_name} is available at {API_ENDPOINT}")
            return True, api_key, API_ENDPOINT
        elif response.status_code == 429 and BACKUP_HF_TOKEN and api_key != BACKUP_HF_TOKEN:
            logger.warning(f"Rate limit reached for token {api_key}. Switching to backup token.")
            return check_model_availability(model_name, BACKUP_HF_TOKEN)
        logger.error(f"Model {model_name} not available: {response.status_code} - {response.text}")
        return False, api_key, API_ENDPOINT
    except Exception as e:
        logger.error(f"Failed to check model availability for {model_name}: {e}")
        if BACKUP_HF_TOKEN and api_key != BACKUP_HF_TOKEN:
            logger.warning(f"Retrying with backup token for {model_name}")
            return check_model_availability(model_name, BACKUP_HF_TOKEN)
        return False, api_key, API_ENDPOINT

def select_model(query: str, input_type: str = "text", preferred_model: Optional[str] = None) -> tuple[str, str]:
    if preferred_model and preferred_model in MODEL_ALIASES:
        model_name = MODEL_ALIASES[preferred_model]
        is_available, _, endpoint = check_model_availability(model_name, HF_TOKEN)
        if is_available:
            logger.info(f"Selected preferred model {model_name} with endpoint {endpoint} for query: {query[:50]}...")
            return model_name, endpoint

    query_lower = query.lower()
    if input_type == "audio" or any(keyword in query_lower for keyword in ["voice", "audio", "speech", "صوت", "تحويل صوت"]):
        logger.info(f"Selected {ASR_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for audio input")
        return ASR_MODEL, FALLBACK_API_ENDPOINT
    if any(keyword in query_lower for keyword in ["text-to-speech", "tts", "تحويل نص إلى صوت"]) or input_type == "tts":
        logger.info(f"Selected {TTS_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for text-to-speech")
        return TTS_MODEL, FALLBACK_API_ENDPOINT
    image_patterns = [
        r"\bimage\b", r"\bpicture\b", r"\bphoto\b", r"\bvisual\b", r"\bصورة\b", r"\bتحليل\s+صورة\b",
        r"\bimage\s+analysis\b", r"\bimage\s+classification\b", r"\bimage\s+description\b"
    ]
    image_gen_patterns = [
        r"\bgenerate\s+image\b", r"\bcreate\s+image\b", r"\bimage\s+generation\b", r"\bصورة\s+توليد\b",
        r"\bimage\s+edit\b", r"\bتحرير\s+صورة\b"
    ]
    for pattern in image_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.info(f"Selected {CLIP_BASE_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for image-related query: {query[:50]}...")
            return CLIP_BASE_MODEL, FALLBACK_API_ENDPOINT
    for pattern in image_gen_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE) or input_type == "image_gen":
            logger.info(f"Selected {IMAGE_GEN_MODEL} with endpoint {FALLBACK_API_ENDPOINT} for image generation query: {query[:50]}...")
            return IMAGE_GEN_MODEL, FALLBACK_API_ENDPOINT
    available_models = [
        (MODEL_NAME, API_ENDPOINT),
        (SECONDARY_MODEL_NAME, FALLBACK_API_ENDPOINT),
        (TERTIARY_MODEL_NAME, API_ENDPOINT)
    ]
    for model_name, api_endpoint in available_models:
        is_available, _, endpoint = check_model_availability(model_name, HF_TOKEN)
        if is_available:
            logger.info(f"Selected {model_name} with endpoint {endpoint} for query: {query[:50]}...")
            return model_name, endpoint
    logger.error("No models available. Falling back to default.")
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
    max_new_tokens: int = 2048,
    reasoning_effort: str = "off",
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[str] = None,
    deep_search: bool = False,
    input_type: str = "text",
    audio_data: Optional[bytes] = None,
    image_data: Optional[bytes] = None,
    output_format: str = "text"
) -> Generator[bytes | str, None, None]:
    is_available, selected_api_key, selected_endpoint = check_model_availability(model_name, api_key)
    if not is_available:
        yield f"Error: Model {model_name} is not available. Please check the model endpoint or token."
        return

    cache_key = hashlib.md5(json.dumps({
        "message": message,
        "system_prompt": system_prompt,
        "model_name": model_name,
        "chat_history": chat_history,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "output_format": output_format
    }, sort_keys=True).encode()).hexdigest()

    if cache_key in cache:
        logger.info(f"Cache hit for query: {message[:50]}...")
        for chunk in cache[cache_key]:
            yield chunk
        return

    client = OpenAI(api_key=selected_api_key, base_url=selected_endpoint, timeout=120.0)
    task_type = "general"
    enhanced_system_prompt = system_prompt
    buffer = ""

    # معالجة الصوت
    if model_name == ASR_MODEL and audio_data:
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
            logger.debug(f"Transcription response: {transcription}")
            yield transcription
            cache[cache_key] = [transcription]
            return
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            yield f"Error: Audio transcription failed: {e}"
            return

    # معالجة تحويل النص إلى صوت
    if model_name == TTS_MODEL or output_format == "audio":
        task_type = "text_to_speech"
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=dtype).to(device)
            processor = AutoProcessor.from_pretrained(TTS_MODEL)
            inputs = processor(text=message, return_tensors="pt").to(device)
            audio = model.generate(**inputs)
            audio_file = io.BytesIO()
            torchaudio.save(audio_file, audio[0], sample_rate=22050, format="wav")
            audio_file.seek(0)
            audio_data = audio_file.read()
            logger.debug(f"Generated audio data of length: {len(audio_data)} bytes")
            yield audio_data
            cache[cache_key] = [audio_data]
            return
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            yield f"Error: Text-to-speech failed: {e}"
            return
        finally:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # معالجة تحليل الصور
    if model_name in [CLIP_BASE_MODEL, CLIP_LARGE_MODEL] and image_data:
        task_type = "image_analysis"
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            inputs = processor(text=message, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            result = f"Image analysis result: {probs.tolist()}"
            logger.debug(f"Image analysis result: {result}")
            if output_format == "audio":
                model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=dtype).to(device)
                processor = AutoProcessor.from_pretrained(TTS_MODEL)
                inputs = processor(text=result, return_tensors="pt").to(device)
                audio = model.generate(**inputs)
                audio_file = io.BytesIO()
                torchaudio.save(audio_file, audio[0], sample_rate=22050, format="wav")
                audio_file.seek(0)
                audio_data = audio_file.read()
                yield audio_data
            else:
                yield result
            cache[cache_key] = [result]
            return
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            yield f"Error: Image analysis failed: {e}"
            return
        finally:
            if 'model' in locals():
                del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # معالجة توليد الصور أو تحريرها
    if model_name in [IMAGE_GEN_MODEL, SECONDARY_IMAGE_GEN_MODEL] or input_type == "image_gen":
        task_type = "image_generation"
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if model_name == IMAGE_GEN_MODEL:
                pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype).to(device)
            else:
                pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype).to(device)

            polished_prompt = polish_prompt(message)
            image_params = {
                "prompt": polished_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            }
            if input_type == "image_gen" and image_data:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_params["image"] = image

            output = pipe(**image_params)
            image_file = io.BytesIO()
            output.images[0].save(image_file, format="PNG")
            image_file.seek(0)
            image_data = image_file.read()
            logger.debug(f"Generated image data of length: {len(image_data)} bytes")
            yield image_data
            cache[cache_key] = [image_data]
            return
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            yield f"Error: Image generation failed: {e}"
            return
        finally:
            if 'pipe' in locals():
                del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # معالجة النصوص
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
        enhanced_system_prompt += "\nEven for short or general queries, provide a detailed, in-depth response."

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
        payload = {
            "model": model_name,
            "messages": input_messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": True,
            "tools": tools,
            "tool_choice": tool_choice
        }
        logger.debug(f"Sending payload to {selected_endpoint}/chat/completions: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        stream = client.chat.completions.create(**payload)
        reasoning_started = False
        reasoning_closed = False
        saw_visible_output = False
        last_tool_name = None
        last_tool_args = None

        for chunk in stream:
            logger.debug(f"Received chunk: {chunk}")
            if chunk.choices and chunk.choices[0].delta.content:
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

            if chunk.choices and chunk.choices[0].delta.tool_calls and model_name in [MODEL_NAME, SECONDARY_MODEL_NAME, TERTIARY_MODEL_NAME]:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                name = getattr(tool_call, "function", {}).get("name", None)
                args = getattr(tool_call, "function", {}).get("arguments", None)
                if name:
                    last_tool_name = name
                if args:
                    last_tool_args = args
                continue

            if chunk.choices and chunk.choices[0].finish_reason in ("stop", "tool_calls", "error", "length"):
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
                    cached_chunks.append("Response truncated due to token limit.")
                    yield "Response truncated due to token limit."
                break

        if buffer:
            cached_chunks.append(buffer)
            yield buffer

        if output_format == "audio":
            try:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=dtype).to(device)
                processor = AutoProcessor.from_pretrained(TTS_MODEL)
                inputs = processor(text=buffer, return_tensors="pt").to(device)
                audio = model.generate(**inputs)
                audio_file = io.BytesIO()
                torchaudio.save(audio_file, audio[0], sample_rate=22050, format="wav")
                audio_file.seek(0)
                audio_data = audio_file.read()
                cached_chunks.append(audio_data)
                yield audio_data
            except Exception as e:
                logger.error(f"Text-to-speech conversion failed: {e}")
                yield f"Error: Text-to-speech conversion failed: {e}"
            finally:
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        cache[cache_key] = cached_chunks

    except Exception as e:
        logger.error(f"[Gateway] Streaming failed for model {model_name}: {e}")
        if selected_api_key != BACKUP_HF_TOKEN and BACKUP_HF_TOKEN:
            logger.warning(f"Retrying with backup token for {model_name}")
            for chunk in request_generation(
                api_key=BACKUP_HF_TOKEN,
                api_base=selected_endpoint,
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
                output_format=output_format,
            ):
                yield chunk
            return
        if model_name == MODEL_NAME:
            fallback_model = SECONDARY_MODEL_NAME
            fallback_endpoint = FALLBACK_API_ENDPOINT
            logger.info(f"Retrying with fallback model: {fallback_model} on {fallback_endpoint}")
            try:
                is_available, selected_api_key, selected_endpoint = check_model_availability(fallback_model, selected_api_key)
                if not is_available:
                    yield f"Error: Fallback model {fallback_model} is not available."
                    return
                client = OpenAI(api_key=selected_api_key, base_url=selected_endpoint, timeout=120.0)
                payload = {
                    "model": fallback_model,
                    "messages": input_messages,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                    "stream": True,
                    "tools": [],
                    "tool_choice": "none"
                }
                logger.debug(f"Sending payload to {selected_endpoint}/chat/completions: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                stream = client.chat.completions.create(**payload)
                buffer = ""
                for chunk in stream:
                    logger.debug(f"Received chunk from fallback: {chunk}")
                    if chunk.choices and chunk.choices[0].delta.content:
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

                    if chunk.choices and chunk.choices[0].finish_reason in ("stop", "error", "length"):
                        if buffer:
                            cached_chunks.append(buffer)
                            yield buffer
                            buffer = ""

                        if reasoning_started and not reasoning_closed:
                            cached_chunks.append("assistantfinal")
                            yield "assistantfinal"
                            reasoning_closed = True

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

                if buffer and output_format == "audio":
                    try:
                        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=dtype).to(device)
                        processor = AutoProcessor.from_pretrained(TTS_MODEL)
                        inputs = processor(text=buffer, return_tensors="pt").to(device)
                        audio = model.generate(**inputs)
                        audio_file = io.BytesIO()
                        torchaudio.save(audio_file, audio[0], sample_rate=22050, format="wav")
                        audio_file.seek(0)
                        audio_data = audio_file.read()
                        cached_chunks.append(audio_data)
                        yield audio_data
                    except Exception as e:
                        logger.error(f"Text-to-speech conversion failed: {e}")
                        yield f"Error: Text-to-speech conversion failed: {e}"
                    finally:
                        if 'model' in locals():
                            del model
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                cache[cache_key] = cached_chunks

            except Exception as e2:
                logger.error(f"[Gateway] Streaming failed for fallback model {fallback_model}: {e2}")
                try:
                    is_available, selected_api_key, selected_endpoint = check_model_availability(TERTIARY_MODEL_NAME, selected_api_key)
                    if not is_available:
                        yield f"Error: Tertiary model {TERTIARY_MODEL_NAME} is not available."
                        return
                    client = OpenAI(api_key=selected_api_key, base_url=selected_endpoint, timeout=120.0)
                    payload = {
                        "model": TERTIARY_MODEL_NAME,
                        "messages": input_messages,
                        "temperature": temperature,
                        "max_tokens": max_new_tokens,
                        "stream": True,
                        "tools": [],
                        "tool_choice": "none"
                    }
                    logger.debug(f"Sending payload to {selected_endpoint}/chat/completions: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                    stream = client.chat.completions.create(**payload)
                    buffer = ""
                    for chunk in stream:
                        logger.debug(f"Received chunk from tertiary: {chunk}")
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            saw_visible_output = True
                            buffer += content
                            if "\n" in buffer or len(buffer) > 5000:
                                cached_chunks.append(buffer)
                                yield buffer
                                buffer = ""
                            continue
                        if chunk.choices and chunk.choices[0].finish_reason in ("stop", "error", "length"):
                            if buffer:
                                cached_chunks.append(buffer)
                                yield buffer
                                buffer = ""
                            if not saw_visible_output:
                                cached_chunks.append("No visible output produced.")
                                yield "No visible output produced."
                            if chunk.choices[0].finish_reason == "error":
                                cached_chunks.append(f"Error: Unknown error with tertiary model {TERTIARY_MODEL_NAME}")
                                yield f"Error: Unknown error with tertiary model {TERTIARY_MODEL_NAME}"
                            elif chunk.choices[0].finish_reason == "length":
                                cached_chunks.append("Response truncated due to token limit.")
                                yield "Response truncated due to token limit."
                            break
                    if buffer and output_format == "audio":
                        try:
                            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=dtype).to(device)
                            processor = AutoProcessor.from_pretrained(TTS_MODEL)
                            inputs = processor(text=buffer, return_tensors="pt").to(device)
                            audio = model.generate(**inputs)
                            audio_file = io.BytesIO()
                            torchaudio.save(audio_file, audio[0], sample_rate=22050, format="wav")
                            audio_file.seek(0)
                            audio_data = audio_file.read()
                            cached_chunks.append(audio_data)
                            yield audio_data
                        except Exception as e:
                            logger.error(f"Text-to-speech conversion failed: {e}")
                            yield f"Error: Text-to-speech conversion failed: {e}"
                        finally:
                            if 'model' in locals():
                                del model
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    cache[cache_key] = cached_chunks
                except Exception as e3:
                    logger.error(f"[Gateway] Streaming failed for tertiary model {TERTIARY_MODEL_NAME}: {e3}")
                    yield f"Error: Failed to load all models: Primary ({model_name}), Secondary ({fallback_model}), Tertiary ({TERTIARY_MODEL_NAME})."
                    return
        else:
            yield f"Error: Failed to load model {model_name}: {e}"
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

def polish_prompt(original_prompt: str, image: Optional[Image.Image] = None) -> str:
    original_prompt = original_prompt.strip()
    system_prompt = "You are an expert in generating high-quality prompts for image generation. Rewrite the user input to be clear, descriptive, and optimized for creating visually appealing images."
    if any(0x0600 <= ord(char) <= 0x06FF for char in original_prompt):
        system_prompt += "\nRespond in Arabic with a polished prompt suitable for image generation."
    prompt = f"{system_prompt}\n\nUser Input: {original_prompt}\n\nRewritten Prompt:"
    magic_prompt = "Ultra HD, 4K, cinematic composition"
    try:
        client = OpenAI(api_key=HF_TOKEN, base_url=FALLBACK_API_ENDPOINT, timeout=120.0)
        polished_prompt = client.chat.completions.create(
            model=SECONDARY_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        ).choices[0].message.content.strip()
        polished_prompt = polished_prompt.replace("\n", " ")
    except Exception as e:
        logger.error(f"Error during prompt polishing: {e}")
        polished_prompt = original_prompt
    return polished_prompt + " " + magic_prompt

def generate(message, history, system_prompt, temperature, reasoning_effort, enable_browsing, max_new_tokens, input_type="text", audio_data=None, image_data=None, output_format="text"):
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
            """```text
Analysis (live):
{raw_analysis}

Response (draft):
{raw_visible}
```""".format(raw_analysis=raw_analysis, raw_visible=raw_visible)
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
            output_format=output_format,
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