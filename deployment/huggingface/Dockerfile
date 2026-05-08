# Hugging Face Spaces Dockerfile for Heart Disease Prediction API
#
# HF Spaces requirements:
#   - container must run as non-root user with UID 1000
#   - default port is 7860 (configurable via README app_port)
#   - WORKDIR must be writable by the runtime user
FROM python:3.12-slim

# Create non-root user (HF Spaces runs containers as UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/home/user/app

WORKDIR $HOME/app

# Install Python dependencies (user install)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy application code and trained artefacts
COPY --chown=user src/ ./src/
COPY --chown=user models/ ./models/

EXPOSE 7860

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
