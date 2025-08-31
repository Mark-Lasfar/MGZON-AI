import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import io
from openai import OpenAI
from api.models import QueryRequest
from utils.generation import request_generation, select_model
from utils.web_search import web_search

router = APIRouter()

HF_TOKEN = os.getenv("HF_TOKEN")
BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:cerebras")
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:featherless-ai")
TERTIARY_MODEL_NAME = os.getenv("TERTIARY_MODEL_NAME", "openai/gpt-oss-20b:together")

@router.get("/api/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "secondary_model": SECONDARY_MODEL_NAME,
        "tertiary_model": TERTIARY_MODEL_NAME,
        "clip_base_model": os.getenv("CLIP_BASE_MODEL", "openai/clip-vit-base-patch32"),
        "clip_large_model": os.getenv("CLIP_LARGE_MODEL", "openai/clip-vit-large-patch14"),
        "api_base": API_ENDPOINT,
        "status": "online"
    }

@router.get("/api/performance")
async def performance_stats():
    return {
        "queue_size": int(os.getenv("QUEUE_SIZE", 80)),
        "concurrency_limit": int(os.getenv("CONCURRENCY_LIMIT", 20)),
        "uptime": os.popen("uptime").read().strip()
    }

@router.post("/api/chat")
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
        output_format=req.output_format
    )
    if req.output_format == "audio":
        audio_data = b"".join([chunk for chunk in stream if isinstance(chunk, bytes)])
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    response = "".join([chunk for chunk in stream if isinstance(chunk, str)])
    return {"response": response}

@router.post("/api/audio-transcription")
async def audio_transcription_endpoint(file: UploadFile = File(...)):
    model_name, api_endpoint = select_model("transcribe audio", input_type="audio")
    audio_data = await file.read()
    response = "".join(list(request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message="Transcribe audio",
        system_prompt="Transcribe the provided audio using Whisper.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=128000,
        input_type="audio",
        audio_data=audio_data,
        output_format="text"
    )))
    return {"transcription": response}

@router.post("/api/text-to-speech")
async def text_to_speech_endpoint(req: dict):
    text = req.get("text", "")
    model_name, api_endpoint = select_model("text to speech", input_type="text")
    stream = request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message=text,
        system_prompt="Convert the provided text to speech using Parler-TTS.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=128000,
        input_type="text",
        output_format="audio"
    )
    audio_data = b"".join([chunk for chunk in stream if isinstance(chunk, bytes)])
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")

@router.post("/api/code")
async def code_endpoint(req: dict):
    framework = req.get("framework")
    task = req.get("task")
    code = req.get("code", "")
    output_format = req.get("output_format", "text")
    prompt = f"Generate code for task: {task} using {framework}. Existing code: {code}"
    model_name, api_endpoint = select_model(prompt)
    stream = request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message=prompt,
        system_prompt="You are a coding expert. Provide detailed, well-commented code with examples and explanations.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=128000,
        output_format=output_format
    )
    if output_format == "audio":
        audio_data = b"".join([chunk for chunk in stream if isinstance(chunk, bytes)])
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    response = "".join([chunk for chunk in stream if isinstance(chunk, str)])
    return {"generated_code": response}

@router.post("/api/analysis")
async def analysis_endpoint(req: dict):
    message = req.get("text", "")
    output_format = req.get("output_format", "text")
    model_name, api_endpoint = select_model(message)
    stream = request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message=message,
        system_prompt="You are an expert analyst. Provide detailed analysis with step-by-step reasoning and examples.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=128000,
        output_format=output_format
    )
    if output_format == "audio":
        audio_data = b"".join([chunk for chunk in stream if isinstance(chunk, bytes)])
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    response = "".join([chunk for chunk in stream if isinstance(chunk, str)])
    return {"analysis": response}

@router.post("/api/image-analysis")
async def image_analysis_endpoint(file: UploadFile = File(...)):
    output_format = "text"  # يمكن تعديله لدعم الصوت
    model_name, api_endpoint = select_model("analyze image", input_type="image")
    image_data = await file.read()
    stream = request_generation(
        api_key=HF_TOKEN,
        api_base=api_endpoint,
        message="Analyze this image",
        system_prompt="You are an expert in image analysis. Provide detailed descriptions or classifications based on the query.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=128000,
        input_type="image",
        image_data=image_data,
        output_format=output_format
    )
    if output_format == "audio":
        audio_data = b"".join([chunk for chunk in stream if isinstance(chunk, bytes)])
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    response = "".join([chunk for chunk in stream if isinstance(chunk, str)])
    return {"image_analysis": response}

@router.get("/api/test-model")
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
