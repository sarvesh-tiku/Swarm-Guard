# SwarmGuard on Hugging Face Spaces (Docker SDK). Spaces run the container as user 1000 on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    SWARMGUARD_CACHE=/home/user/.cache/swarmguard
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --user --no-deps -e .

# Pre-download the small sentence-embedding model so the first visit is faster (public model, ~90 MB).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Compute text embeddings for the bundled (public) German wiki export once, at build time, so the first
# visitor doesn't wait ~10 minutes for 20k texts to be embedded on CPU. AI Village is gated and is
# never pre-computed into the image; it is read at runtime with the HF_TOKEN secret.
RUN python -c "from swarmguard.data.german_wiki import load_dir; from swarmguard.embeddings.encoder import get_encoder; \
ev = load_dir('datasets/german_wiki').events; get_encoder('minilm').encode(ev['content'].fillna('').astype(str).tolist()); \
print('cached embeddings for', len(ev), 'wiki events')"

EXPOSE 7860
CMD ["streamlit", "run", "swarmguard/app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
