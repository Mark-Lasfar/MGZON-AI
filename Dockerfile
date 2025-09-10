FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium-driver \
    git \
    gcc \
    libc-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

RUN pip install packaging torch==2.4.1

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create /data directory with correct permissions
RUN mkdir -p /data && chmod -R 755 /data

# Copy all project files
COPY . .

# Verify files in /app
RUN ls -R /app

# Expose port 7860 for FastAPI
EXPOSE 7860

# Run the FastAPI app
CMD ["python", "main.py"]

