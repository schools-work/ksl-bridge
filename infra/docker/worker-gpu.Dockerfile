# GPU inference/training image. Choose a CUDA runtime that matches the pinned torch build.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip ca-certificates \
    libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv
RUN useradd --create-home --uid 10001 ksl
WORKDIR /srv
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --extra ml --no-install-project || uv sync --no-dev --extra ml
COPY ksl_core ksl_core
COPY workers workers
COPY ml ml
COPY contracts contracts
USER ksl
CMD ["uv", "run", "--no-dev", "python", "-m", "workers.sign_worker"]
