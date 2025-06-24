FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Use python -m to avoid permission issues
CMD ["python", "-m", "gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1"]