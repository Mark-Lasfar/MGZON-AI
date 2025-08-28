from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant capable of code generation, analysis, review, and more."
    history: Optional[List[dict]] = None
    temperature: float = 0.9
    max_new_tokens: int = 128000
    enable_browsing: bool = False
