from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    message: str
    system_prompt: str = "You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, and image inputs. Transcribe audio using Whisper, convert text to speech using Parler-TTS, and analyze images using CLIP. Respond with text or audio based on input type. Continue until the query is fully addressed."
    history: Optional[List[dict]] = None
    temperature: float = 0.7
    max_new_tokens: int = 128000
    enable_browsing: bool = True
