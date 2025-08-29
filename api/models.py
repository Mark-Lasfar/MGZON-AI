from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    message: str
    system_prompt: str = "You are an expert assistant providing detailed, comprehensive, and well-structured responses. For code, include comments, examples, and complete implementations. For image-related queries, provide detailed analysis or descriptions. For general queries, provide in-depth explanations with examples and additional context where applicable. Continue generating content until the query is fully addressed, leveraging the full capacity of the model."
    history: Optional[List[dict]] = None
    temperature: float = 0.7
    max_new_tokens: int = 128000
    enable_browsing: bool = False
