from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    message: str
    system_prompt: str = "You are an expert assistant providing detailed, comprehensive, and well-structured responses. Support text, audio, image inputs. For audio, transcribe using Whisper. For text-to-speech, use Parler-TTS. For images, analyze using CLIP. Respond with voice output when requested. Continue until the query is fully addressed."
    history: Optional[List[dict]] = None
    temperature: float = 0.7
    max_new_tokens: int = 128000
    enable_browsing: bool = True
