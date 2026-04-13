# Base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]