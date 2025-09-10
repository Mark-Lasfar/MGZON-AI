# استخدام صورة PyTorch مع CUDA
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# تحديث pip
RUN pip install --upgrade pip

# تثبيت الـ dependencies المطلوبة لـ ffmpeg و chromium-driver
RUN apt-get update && apt-get install -y \
    chromium-driver \
    git \
    gcc \
    libc-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# تحديد مجلد العمل
WORKDIR /app

# نسخ requirements.txt
COPY requirements.txt .

# تثبيت الـ dependencies
RUN pip install --no-cache-dir -r requirements.txt

# إنشاء /data مع الصلاحيات
RUN mkdir -p /data && chmod -R 755 /data

# نسخ باقي الملفات
COPY . .

# التحقق من الملفات
RUN ls -R /app

# تعريض المنفذ 7860 لـ FastAPI
EXPOSE 7860

# تشغيل التطبيق
CMD ["python", "main.py"]
