FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install chromium-driver and dependencies
RUN apt-get update && apt-get install -y chromium-driver && apt-get clean

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Verify files in /app
RUN ls -R /app

# Expose port 7860 for FastAPI/Gradio
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
