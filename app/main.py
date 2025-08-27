from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="MGZON FLAN-T5 API")

# تحميل النموذج من Hugging Face مباشرة
MODEL_NAME = "MGZON/mgzon-flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")

class RequestText(BaseModel):
    text: str
    max_length: int = 200

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/generate/")
async def generate(req: RequestText):
    inputs = tokenizer(req.text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=req.max_length)
    return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
