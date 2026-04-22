# syntax=docker/dockerfile:1.6
#
# OpenEnv server image for the self-improvement math environment.
#
# Target platform: Hugging Face Spaces (Docker SDK), but the same image
# runs fine under ``docker run`` locally.  HF Spaces enforces:
#   * non-root user with UID 1000 ("user")
#   * $PORT (default 7860) for the HTTP listener
#   * writable ``/data`` for persistent state (curriculum checkpoints)
#
# We pick a CUDA 12.4 runtime base because torch==2.5.1 from the
# pinned requirements.txt ships its own CUDA kernels - the image only
# needs the driver interface, not the full toolkit.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf \
    CURRICULUM_CHECKPOINT_DIR=/data/curriculum \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip

RUN useradd -m -u 1000 -s /bin/bash user
USER user
WORKDIR /home/user/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user deployment/requirements.txt ./requirements.txt
RUN pip install --user -r requirements.txt

COPY --chown=user:user src ./src
COPY --chown=user:user deployment/app.py ./app.py

# /data is the HF Spaces persistent volume.  We create it here so the
# curriculum manager can write checkpoints even on first boot.
USER root
RUN mkdir -p /data/hf /data/curriculum && chown -R user:user /data
USER user

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

EXPOSE 7860

# ``app.py`` is the canonical entrypoint HF Spaces looks for.  It just
# re-exports ``src.openenv.server:app`` and runs uvicorn on $PORT.
CMD ["python", "app.py"]
