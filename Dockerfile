# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all source code first (needed for -e .)
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade dagshub

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
