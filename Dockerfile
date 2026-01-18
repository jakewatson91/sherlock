FROM python:3.10-slim

# 1. Setup the low-privilege user required by Hugging Face
RUN useradd -m -u 1000 user
USER user

# 2. THE FIX: Add the hidden local bin folder to the system PATH
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# 3. Install dependencies as that user
# We use --chown to make sure 'user' owns the files, not 'root'
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 4. Copy the rest of your app
COPY --chown=user:user . .

# 5. Create the data folder for your PDFs
RUN mkdir -p data

# 6. Launch Chainlit on the HF-mandated port
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]