# Use Python 3.10 base image
FROM python:3.10

# Set working directory to /app
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only main.py (no need for utils.py)
COPY ./app/main.py ./app/main.py

# Expose port 7860 for FastAPI/Gradio
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]