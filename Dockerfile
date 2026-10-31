FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python requirements
RUN pip install --no-cache-dir -r web_app/requirements.txt

# Expose web port
EXPOSE 8080

ENV PORT=8080
ENV ADMIN_PIN=ACC2026

# Run app
WORKDIR /app/web_app
CMD ["python3", "app.py"]
