FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code (the snapshot_download logic is inside app.py)
COPY . .

# Expose the port (informative only)
EXPOSE 7860

# Start the app on the correct HF port
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]