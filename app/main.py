import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import threading

# -----------------------------
# تحميل النموذج
# -----------------------------
MODEL_NAME = "MGZON/mgzon-flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")

# -----------------------------
# إعداد FastAPI
# -----------------------------
app = FastAPI(title="MGZON FLAN-T5 API")

class RequestText(BaseModel):
    text: str
    max_length: int = 200

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/generate/")
async def generate_api(req: RequestText):
    try:
        inputs = tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=req.max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# إعداد Gradio
# -----------------------------
def generate_gradio(text, max_length=200):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=generate_gradio,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="اكتب النص هنا..."),
        gr.Slider(10, 512, value=200, step=1, label="Max Length")
    ],
    outputs="text",
    title="MGZON FLAN-T5 Text Generation",
    description="توليد النصوص باستخدام نموذج FLAN-T5 من MGZON"
)

# تشغيل Gradio في Thread منفصل حتى لا يوقف FastAPI
def launch_gradio():
    iface.launch(server_name="0.0.0.0", server_port=7861, share=True)  # share=True لتوليد رابط مباشر في Space

threading.Thread(target=launch_gradio, daemon=True).start()

# -----------------------------
# تشغيل FastAPI
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
