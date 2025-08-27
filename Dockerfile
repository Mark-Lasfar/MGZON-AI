# Use Python 3.10 base image
FROM python:3.10

# Set working directory to /app
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main.py and __init__.py to /app
COPY app/main.py ./main.py
COPY app/__init__.py ./__init__.py

# Verify files in /app
RUN ls -R /app

# Expose port 7860 for FastAPI/Gradio
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]