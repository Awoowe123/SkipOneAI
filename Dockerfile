FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p data logs

# Set environment
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.main"]
