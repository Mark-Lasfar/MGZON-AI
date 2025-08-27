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
    # ترميز النص مع تحديد padding و truncation
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest"   # بدل max_length
    ).to(model.device)

    # توليد النص باستخدام معلمات sampling
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=req.max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,  # مهم لبعض النماذج
        num_return_sequences=1
    )

    # فك التشفير وإرجاع النص المولد
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
