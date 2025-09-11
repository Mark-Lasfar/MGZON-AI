# api/endpoints.py
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, status, UploadFile, File
from fastapi.responses import StreamingResponse
from api.database import User, Conversation, Message
from api.models import QueryRequest, ConversationOut, ConversationCreate, UserUpdate
from api.auth import current_active_user
from api.database import get_db
from sqlalchemy.orm import Session
from utils.generation import request_generation, select_model, check_model_availability
from utils.web_search import web_search
import io
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import logging
from typing import List, Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# Check HF_TOKEN and BACKUP_HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in environment variables.")
    raise ValueError("HF_TOKEN is required for Inference API.")

BACKUP_HF_TOKEN = os.getenv("BACKUP_HF_TOKEN")
if not BACKUP_HF_TOKEN:
    logger.warning("BACKUP_HF_TOKEN is not set. Fallback to secondary model will not work if primary token fails.")

ROUTER_API_URL = os.getenv("ROUTER_API_URL", "https://router.huggingface.co")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://api.cerebras.ai/v1")  # تغيير الافتراضي لـ Cerebras
FALLBACK_API_ENDPOINT = os.getenv("FALLBACK_API_ENDPOINT", "https://api-inference.huggingface.co")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")  # النموذج الرئيسي
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
TERTIARY_MODEL_NAME = os.getenv("TERTIARY_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
CLIP_BASE_MODEL = os.getenv("CLIP_BASE_MODEL", "Salesforce/blip-image-captioning-large")
CLIP_LARGE_MODEL = os.getenv("CLIP_LARGE_MODEL", "openai/clip-vit-large-patch14")
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-large-v3")
TTS_MODEL = os.getenv("TTS_MODEL", "facebook/mms-tts-ara")

# Model alias mapping for user-friendly names
MODEL_ALIASES = {
    "advanced": MODEL_NAME,
    "standard": SECONDARY_MODEL_NAME,
    "light": TERTIARY_MODEL_NAME,
    "image_base": CLIP_BASE_MODEL,
    "image_advanced": CLIP_LARGE_MODEL,
    "audio": ASR_MODEL,
    "tts": TTS_MODEL
}

# MongoDB setup
MONGO_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["hager"]
session_message_counts = db["session_message_counts"]

# Helper function to handle sessions for non-logged-in users
async def handle_session(request: Request):
    if not hasattr(request, "session"):
        raise HTTPException(status_code=500, detail="Session middleware not configured")
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id
        await session_message_counts.insert_one({"session_id": session_id, "message_count": 0})
    
    session_doc = await session_message_counts.find_one({"session_id": session_id})
    if not session_doc:
        session_doc = {"session_id": session_id, "message_count": 0}
        await session_message_counts.insert_one(session_doc)
    
    message_count = session_doc["message_count"] + 1
    await session_message_counts.update_one(
        {"session_id": session_id},
        {"$set": {"message_count": message_count}}
    )
    if message_count > 4:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Message limit reached. Please log in to continue."
        )
    return session_id

# Helper function to enhance system prompt for Arabic language
def enhance_system_prompt(system_prompt: str, message: str, user: Optional[User] = None) -> str:
    enhanced_prompt = system_prompt
    # Check if the message is in Arabic
    if any(0x0600 <= ord(char) <= 0x06FF for char in message):
        enhanced_prompt += "\nRespond in Arabic with clear, concise, and accurate information tailored to the user's query."
    if user and user.additional_info:
        enhanced_prompt += f"\nUser Profile: {user.additional_info}\nConversation Style: {user.conversation_style or 'default'}"
    return enhanced_prompt

@router.get("/api/settings")
async def get_settings(user: User = Depends(current_active_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return {
        "available_models": [
            {"alias": "advanced", "description": "High-performance model for complex queries"},
            {"alias": "standard", "description": "Balanced model for general use"},
            {"alias": "light", "description": "Lightweight model for quick responses"}
        ],
        "conversation_styles": ["default", "concise", "analytical", "creative"],
        "user_settings": {
            "display_name": user.display_name,
            "preferred_model": user.preferred_model,
            "job_title": user.job_title,
            "education": user.education,
            "interests": user.interests,
            "additional_info": user.additional_info,
            "conversation_style": user.conversation_style
        }
    }

@router.get("/api/model-info")
async def model_info():
    return {
        "available_models": [
            {"alias": "advanced", "description": "High-performance model for complex queries"},
            {"alias": "standard", "description": "Balanced model for general use"},
            {"alias": "light", "description": "Lightweight model for quick responses"},
            {"alias": "image_base", "description": "Basic image analysis model"},
            {"alias": "image_advanced", "description": "Advanced image analysis model"},
            {"alias": "audio", "description": "Audio transcription model (default)"},
            {"alias": "tts", "description": "Text-to-speech model (default)"}
        ],
        "api_base": ROUTER_API_URL,
        "fallback_api_base": FALLBACK_API_ENDPOINT,
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
async def chat_endpoint(
    request: Request,
    req: QueryRequest,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Received chat request: {req}")
    
    if not user:
        await handle_session(request)
    
    conversation = None
    if user:
        title = req.title or (req.message[:50] + "..." if len(req.message) > 50 else req.message or "Untitled Conversation")
        conversation = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc()).first()
        if not conversation:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user.id,
                title=title
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        user_msg = Message(role="user", content=req.message, conversation_id=conversation.id)
        db.add(user_msg)
        db.commit()
    
    # Use user's preferred model if set
    preferred_model = user.preferred_model if user else None
    model_name, api_endpoint = select_model(req.message, input_type="text", preferred_model=preferred_model)
    
    # Check model availability
    is_available, api_key, selected_endpoint = check_model_availability(model_name, HF_TOKEN)
    if not is_available:
        logger.error(f"Model {model_name} is not available at {api_endpoint}")
        raise HTTPException(status_code=503, detail=f"Model {model_name} is not available. Please try another model.")
    
    system_prompt = enhance_system_prompt(req.system_prompt, req.message, user)
    
    stream = request_generation(
        api_key=api_key,
        api_base=selected_endpoint,
        message=req.message,
        system_prompt=system_prompt,
        model_name=model_name,
        chat_history=req.history,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens or 2048,
        deep_search=req.enable_browsing,
        input_type="text",
        output_format=req.output_format
    )
    
    if req.output_format == "audio":
        audio_chunks = []
        try:
            for chunk in stream:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                else:
                    logger.warning(f"Unexpected non-bytes chunk in audio stream: {chunk}")
            if not audio_chunks:
                logger.error("No audio data generated.")
                raise HTTPException(status_code=500, detail="No audio data generated.")
            audio_data = b"".join(audio_chunks)
            return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
    
    response_chunks = []
    try:
        for chunk in stream:
            if isinstance(chunk, str):
                response_chunks.append(chunk)
            else:
                logger.warning(f"Unexpected non-string chunk in text stream: {chunk}")
        response = "".join(response_chunks)
        if not response.strip():
            logger.error("Empty response generated.")
            raise HTTPException(status_code=500, detail="Empty response generated from model.")
        logger.info(f"Chat response: {response[:100]}...")  # Log first 100 chars
    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")
    
    if user and conversation:
        assistant_msg = Message(role="assistant", content=response, conversation_id=conversation.id)
        db.add(assistant_msg)
        db.commit()
        conversation.updated_at = datetime.utcnow()
        db.commit()
        return {
            "response": response,
            "conversation_id": conversation.conversation_id,
            "conversation_url": f"https://mgzon-mgzon-app.hf.space/chat/{conversation.conversation_id}",
            "conversation_title": conversation.title
        }
    
    return {"response": response}

@router.post("/api/audio-transcription")
async def audio_transcription_endpoint(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Received audio transcription request for file: {file.filename}")
    
    if not user:
        await handle_session(request)
    
    conversation = None
    if user:
        title = "Audio Transcription"
        conversation = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc()).first()
        if not conversation:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user.id,
                title=title
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        user_msg = Message(role="user", content="Audio message", conversation_id=conversation.id)
        db.add(user_msg)
        db.commit()
    
    model_name, api_endpoint = select_model("transcribe audio", input_type="audio")
    
    # Check model availability
    is_available, api_key, selected_endpoint = check_model_availability(model_name, HF_TOKEN)
    if not is_available:
        logger.error(f"Model {model_name} is not available at {api_endpoint}")
        raise HTTPException(status_code=503, detail=f"Model {model_name} is not available. Please try another model.")
    
    audio_data = await file.read()
    stream = request_generation(
        api_key=api_key,
        api_base=selected_endpoint,
        message="Transcribe audio",
        system_prompt="Transcribe the provided audio using Whisper. Ensure accurate transcription in the detected language.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=2048,
        input_type="audio",
        audio_data=audio_data,
        output_format="text"
    )
    response_chunks = []
    try:
        for chunk in stream:
            if isinstance(chunk, str):
                response_chunks.append(chunk)
            else:
                logger.warning(f"Unexpected non-string chunk in transcription stream: {chunk}")
        response = "".join(response_chunks)
        if not response.strip():
            logger.error("Empty transcription generated.")
            raise HTTPException(status_code=500, detail="Empty transcription generated from model.")
        logger.info(f"Audio transcription response: {response[:100]}...")
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")
    
    if user and conversation:
        assistant_msg = Message(role="assistant", content=response, conversation_id=conversation.id)
        db.add(assistant_msg)
        db.commit()
        conversation.updated_at = datetime.utcnow()
        db.commit()
        return {
            "transcription": response,
            "conversation_id": conversation.conversation_id,
            "conversation_url": f"https://mgzon-mgzon-app.hf.space/chat/{conversation.conversation_id}",
            "conversation_title": conversation.title
        }
    
    return {"transcription": response}

@router.post("/api/text-to-speech")
async def text_to_speech_endpoint(
    request: Request,
    req: dict,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        await handle_session(request)
    
    text = req.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required for text-to-speech.")
    
    model_name, api_endpoint = select_model("text to speech", input_type="tts")
    
    # Check model availability
    is_available, api_key, selected_endpoint = check_model_availability(model_name, HF_TOKEN)
    if not is_available:
        logger.error(f"Model {model_name} is not available at {api_endpoint}")
        raise HTTPException(status_code=503, detail=f"Model {model_name} is not available. Please try another model.")
    
    stream = request_generation(
        api_key=api_key,
        api_base=selected_endpoint,
        message=text,
        system_prompt="Convert the provided text to speech using a text-to-speech model. Ensure clear and natural pronunciation, especially for Arabic text.",
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=2048,
        input_type="tts",
        output_format="audio"
    )
    audio_chunks = []
    try:
        for chunk in stream:
            if isinstance(chunk, bytes):
                audio_chunks.append(chunk)
            else:
                logger.warning(f"Unexpected non-bytes chunk in TTS stream: {chunk}")
        if not audio_chunks:
            logger.error("No audio data generated for TTS.")
            raise HTTPException(status_code=500, detail="No audio data generated for text-to-speech.")
        audio_data = b"".join(audio_chunks)
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    except Exception as e:
        logger.error(f"Text-to-speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech generation failed: {str(e)}")

@router.post("/api/code")
async def code_endpoint(
    request: Request,
    req: dict,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        await handle_session(request)
    
    framework = req.get("framework")
    task = req.get("task")
    code = req.get("code", "")
    output_format = req.get("output_format", "text")
    if not task:
        raise HTTPException(status_code=400, detail="Task description is required.")
    
    prompt = f"Generate code for task: {task} using {framework}. Existing code: {code}"
    preferred_model = user.preferred_model if user else None
    model_name, api_endpoint = select_model(prompt, input_type="text", preferred_model=preferred_model)
    
    # Check model availability
    is_available, api_key, selected_endpoint = check_model_availability(model_name, HF_TOKEN)
    if not is_available:
        logger.error(f"Model {model_name} is not available at {api_endpoint}")
        raise HTTPException(status_code=503, detail=f"Model {model_name} is not available. Please try another model.")
    
    system_prompt = enhance_system_prompt(
        "You are a coding expert. Provide detailed, well-commented code with examples and explanations.",
        prompt, user
    )
    
    stream = request_generation(
        api_key=api_key,
        api_base=selected_endpoint,
        message=prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=2048,
        input_type="text",
        output_format=output_format
    )
    if output_format == "audio":
        audio_chunks = []
        try:
            for chunk in stream:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                else:
                    logger.warning(f"Unexpected non-bytes chunk in code audio stream: {chunk}")
            if not audio_chunks:
                logger.error("No audio data generated for code.")
                raise HTTPException(status_code=500, detail="No audio data generated for code.")
            audio_data = b"".join(audio_chunks)
            return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
        except Exception as e:
            logger.error(f"Code audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Code audio generation failed: {str(e)}")
    
    response_chunks = []
    try:
        for chunk in stream:
            if isinstance(chunk, str):
                response_chunks.append(chunk)
            else:
                logger.warning(f"Unexpected non-string chunk in code stream: {chunk}")
        response = "".join(response_chunks)
        if not response.strip():
            logger.error("Empty code response generated.")
            raise HTTPException(status_code=500, detail="Empty code response generated from model.")
        return {"generated_code": response}
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@router.post("/api/analysis")
async def analysis_endpoint(
    request: Request,
    req: dict,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        await handle_session(request)
    
    message = req.get("text", "")
    output_format = req.get("output_format", "text")
    if not message.strip():
        raise HTTPException(status_code=400, detail="Text input is required for analysis.")
    
    preferred_model = user.preferred_model if user else None
    model_name, api_endpoint = select_model(message, input_type="text", preferred_model=preferred_model)
    
    # Check model availability
    is_available, api_key, selected_endpoint = check_model_availability(model_name, HF_TOKEN)
    if not is_available:
        logger.error(f"Model {model_name} is not available at {api_endpoint}")
        raise HTTPException(status_code=503, detail=f"Model {model_name} is not available. Please try another model.")
    
    system_prompt = enhance_system_prompt(
        "You are an expert analyst. Provide detailed analysis with step-by-step reasoning and examples.",
        message, user
    )
    
    stream = request_generation(
        api_key=api_key,
        api_base=selected_endpoint,
        message=message,
        system_prompt=system_prompt,
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=2048,
        input_type="text",
        output_format=output_format
    )
    if output_format == "audio":
        audio_chunks = []
        try:
            for chunk in stream:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                else:
                    logger.warning(f"Unexpected non-bytes chunk in analysis audio stream: {chunk}")
            if not audio_chunks:
                logger.error("No audio data generated for analysis.")
                raise HTTPException(status_code=500, detail="No audio data generated for analysis.")
            audio_data = b"".join(audio_chunks)
            return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
        except Exception as e:
            logger.error(f"Analysis audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis audio generation failed: {str(e)}")
    
    response_chunks = []
    try:
        for chunk in stream:
            if isinstance(chunk, str):
                response_chunks.append(chunk)
            else:
                logger.warning(f"Unexpected non-string chunk in analysis stream: {chunk}")
        response = "".join(response_chunks)
        if not response.strip():
            logger.error("Empty analysis response generated.")
            raise HTTPException(status_code=500, detail="Empty analysis response generated from model.")
        return {"analysis": response}
    except Exception as e:
        logger.error(f"Analysis generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis generation failed: {str(e)}")

@router.post("/api/image-analysis")
async def image_analysis_endpoint(
    request: Request,
    file: UploadFile = File(...),
    output_format: str = "text",
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        await handle_session(request)
    
    conversation = None
    if user:
        title = "Image Analysis"
        conversation = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc()).first()
        if not conversation:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user.id,
                title=title
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        user_msg = Message(role="user", content="Image analysis request", conversation_id=conversation.id)
        db.add(user_msg)
        db.commit()
    
    preferred_model = user.preferred_model if user else None
    model_name, api_endpoint = select_model("analyze image", input_type="image", preferred_model=preferred_model)
    
    # Check model availability
    is_available, api_key, selected_endpoint = check_model_availability(model_name, HF_TOKEN)
    if not is_available:
        logger.error(f"Model {model_name} is not available at {api_endpoint}")
        raise HTTPException(status_code=503, detail=f"Model {model_name} is not available. Please try another model.")
    
    image_data = await file.read()
    system_prompt = enhance_system_prompt(
        "You are an expert in image analysis. Provide detailed descriptions or classifications based on the query.",
        "Analyze this image", user
    )
    
    stream = request_generation(
        api_key=api_key,
        api_base=selected_endpoint,
        message="Analyze this image",
        system_prompt=system_prompt,
        model_name=model_name,
        temperature=0.7,
        max_new_tokens=2048,
        input_type="image",
        image_data=image_data,
        output_format=output_format
    )
    if output_format == "audio":
        audio_chunks = []
        try:
            for chunk in stream:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                else:
                    logger.warning(f"Unexpected non-bytes chunk in image analysis audio stream: {chunk}")
            if not audio_chunks:
                logger.error("No audio data generated for image analysis.")
                raise HTTPException(status_code=500, detail="No audio data generated for image analysis.")
            audio_data = b"".join(audio_chunks)
            return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
        except Exception as e:
            logger.error(f"Image analysis audio generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image analysis audio generation failed: {str(e)}")
    
    response_chunks = []
    try:
        for chunk in stream:
            if isinstance(chunk, str):
                response_chunks.append(chunk)
            else:
                logger.warning(f"Unexpected non-string chunk in image analysis stream: {chunk}")
        response = "".join(response_chunks)
        if not response.strip():
            logger.error("Empty image analysis response generated.")
            raise HTTPException(status_code=500, detail="Empty image analysis response generated from model.")
        
        if user and conversation:
            assistant_msg = Message(role="assistant", content=response, conversation_id=conversation.id)
            db.add(assistant_msg)
            db.commit()
            conversation.updated_at = datetime.utcnow()
            db.commit()
            return {
                "image_analysis": response,
                "conversation_id": conversation.conversation_id,
                "conversation_url": f"https://mgzon-mgzon-app.hf.space/chat/{conversation.conversation_id}",
                "conversation_title": conversation.title
            }
        
        return {"image_analysis": response}
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@router.get("/api/test-model")
async def test_model(model: str = MODEL_NAME, endpoint: str = ROUTER_API_URL):
    try:
        is_available, api_key, selected_endpoint = check_model_availability(model, HF_TOKEN)
        if not is_available:
            logger.error(f"Model {model} is not available at {endpoint}")
            raise HTTPException(status_code=503, detail=f"Model {model} is not available.")
        
        client = OpenAI(api_key=api_key, base_url=selected_endpoint, timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=50
        )
        return {"status": "success", "response": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Test model failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test model failed: {str(e)}")

@router.post("/api/conversations", response_model=ConversationOut)
async def create_conversation(
    req: ConversationCreate,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    conversation_id = str(uuid.uuid4())
    conversation = Conversation(
        conversation_id=conversation_id,
        title=req.title or "Untitled Conversation",
        user_id=user.id
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return ConversationOut.from_orm(conversation)

@router.get("/api/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.add(conversation)
    db.commit()
    return ConversationOut.from_orm(conversation)

@router.get("/api/conversations", response_model=List[ConversationOut])
async def list_conversations(
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    conversations = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).all()
    return conversations

@router.put("/api/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    title: str,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.title = title
    conversation.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Conversation title updated", "title": conversation.title}

@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.query(Message).filter(Message.conversation_id == conversation.id).delete()
    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted successfully"}

@router.get("/users/me")
async def get_user_settings(user: User = Depends(current_active_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "preferred_model": user.preferred_model,
        "job_title": user.job_title,
        "education": user.education,
        "interests": user.interests,
        "additional_info": user.additional_info,
        "conversation_style": user.conversation_style,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser
    }

@router.put("/users/me")
async def update_user_settings(
    settings: UserUpdate,
    user: User = Depends(current_active_user),
    db: Session = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    
    # Validate preferred_model
    if settings.preferred_model and settings.preferred_model not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail="Invalid model alias")
    
    # Update user settings
    if settings.display_name is not None:
        user.display_name = settings.display_name
    if settings.preferred_model is not None:
        user.preferred_model = settings.preferred_model
    if settings.job_title is not None:
        user.job_title = settings.job_title
    if settings.education is not None:
        user.education = settings.education
    if settings.interests is not None:
        user.interests = settings.interests
    if settings.additional_info is not None:
        user.additional_info = settings.additional_info
    if settings.conversation_style is not None:
        user.conversation_style = settings.conversation_style
    
    db.commit()
    db.refresh(user)
    return {"message": "Settings updated successfully", "user": {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "preferred_model": user.preferred_model,
        "job_title": user.job_title,
        "education": user.education,
        "interests": user.interests,
        "additional_info": user.additional_info,
        "conversation_style": user.conversation_style,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser
    }}
