FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install chromium-driver and build dependencies
RUN apt-get update && apt-get install -y \
    chromium-driver \
    git \
    gcc \
    libc-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Verify files in /app
RUN ls -R /app

# Expose port 7860 for Gradio
EXPOSE 7860

# Run the Gradio app
CMD ["python", "main.py"]
