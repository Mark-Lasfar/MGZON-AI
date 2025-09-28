from transformers import pipeline
import whisper
import time
from gtts import gTTS
import os

# اختبار 1: DistilGPT2 (نصي)
def test_distilgpt2():
    print("\n=== اختبار DistilGPT2 ===")
    start_time = time.time()
    generator = pipeline('text-generation', model='distilgpt2')
    
    # اختبار بالإنجليزية
    prompt_en = "What's the capital of France?"
    result_en = generator(
        prompt_en,
        max_new_tokens=50,
        truncation=True,
        do_sample=True,
        top_p=0.9,
        top_k=50,  # إضافة top_k لتحسين التنوع
        temperature=0.7  # لتقليل العشوائية
    )
    print(f"الرد (إنجليزي): {result_en[0]['generated_text']}")
    
    # اختبار بالعربية
    prompt_ar = "ما عاصمة فرنسا؟"
    result_ar = generator(
        prompt_ar,
        max_new_tokens=50,
        truncation=True,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.7
    )
    print(f"الرد (عربي): {result_ar[0]['generated_text']}")
    
    print(f"الوقت: {time.time() - start_time:.2f} ثانية")

# اختبار 2: Qwen2-0.5B-Instruct (نصي، دعم أفضل للعربية)
def test_qwen2():
    print("\n=== اختبار Qwen2-0.5B-Instruct ===")
    start_time = time.time()
    generator = pipeline('text-generation', model='Qwen/Qwen2-0.5B-Instruct')
    
    # اختبار بالإنجليزية
    prompt_en = "What's the capital of France?"
    result_en = generator(
        prompt_en,
        max_new_tokens=50,
        truncation=True,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.7
    )
    print(f"الرد (إنجليزي): {result_en[0]['generated_text']}")
    
    # اختبار بالعربية
    prompt_ar = "ما عاصمة فرنسا؟"
    result_ar = generator(
        prompt_ar,
        max_new_tokens=50,
        truncation=True,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.7
    )
    print(f"الرد (عربي): {result_ar[0]['generated_text']}")
    
    print(f"الوقت: {time.time() - start_time:.2f} ثانية")

# اختبار 3: Whisper-tiny (صوتي)
def test_whisper_tiny():
    print("\n=== اختبار Whisper-tiny ===")
    start_time = time.time()
    
    # إنشاء ملف صوتي مؤقت باستخدام gTTS
    text = "Hello world, this is a test audio."
    tts = gTTS(text=text, lang='en')
    audio_file = "test_audio.wav"
    tts.save(audio_file)
    
    # تحميل واختبار Whisper-tiny
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_file, fp16=False)
    print(f"النص المترجم: {result['text']}")
    print(f"الوقت: {time.time() - start_time:.2f} ثانية")
    
    # تنظيف الملف المؤقت
    if os.path.exists(audio_file):
        os.remove(audio_file)

if __name__ == "__main__":
    test_distilgpt2()
    test_qwen2()
    test_whisper_tiny()
