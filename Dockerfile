FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer across rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the already-trained model (no training happens here).
COPY api/ ./api/
COPY src/ ./src/
COPY model/ ./model/

# Cloud Run injects PORT at runtime; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

# Both models (model/standard/, model/extended/) are loaded at startup and
# served in parallel by this one process -- see api/main.py.

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
