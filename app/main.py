from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import logging

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MGZON FLAN-T5 API")

# Environment Variable (لو هتحتاج Token)
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", None)

MODEL_NAME = "MGZON/mgzon-flan-t5-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HUGGING_FACE_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        use_auth_token=HUGGING_FACE_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"  # أو "cpu" لو مش فيه GPU
    )
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class RequestText(BaseModel):
    text: str
    max_length: int = 200

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/generate")
async def generate(req: RequestText):
    try:
        inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=req.max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": text}
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
