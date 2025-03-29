# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables (optional, can be set at runtime)
# ENV WEBSOCKET_PORT=8765
# ENV OPENAI_API_KEY="your_key_here"
# ENV VALID_BEARER_TOKENS="your_token1,your_token2" # For SimpleAuthService
# ENV LOG_LEVEL="INFO"

# Before pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Add build dependencies for opuslib if needed (check opuslib docs)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential libopus-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8765 available to the world outside this container (adjust if needed)
EXPOSE 8765

# Run main.py when the container launches
CMD ["python", "main.py"]