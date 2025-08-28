import os
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from api.models import QueryRequest
from utils.generation import request_generation, select_model
from utils.web_search import web_search

router = APIRouter()

HF_TOKEN = os.getenv("HF_TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:fireworks-ai")

@router.get("/api/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "secondary_model": os.getenv("SECONDARY_MODEL_NAME", "MGZON/Veltrix"),
        "tertiary_model": os.getenv("TERTIARY_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
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
    )
    response = "".join(list(stream))
    return {"response": response}

@router.post("/api/code")
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

@router.post("/api/analysis")
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
