# Multi-stage build: wheels are compiled in the builder, and only the runtime
# artefacts are copied into the final image. This keeps compilers and build
# headers out of the shipped container -- smaller image, smaller attack surface.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as an unprivileged user. A container process that does not need root
# should not have it, and this one only needs to read code and write to
# /data/storage.
RUN groupadd --system --gid 1001 securerag \
    && useradd --system --uid 1001 --gid securerag --create-home securerag

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=securerag:securerag . .

RUN mkdir -p /data/storage && chown -R securerag:securerag /data

USER securerag

EXPOSE 8000

# Migrations run at start rather than in the image build: the database does not
# exist at build time, and running them here means a fresh volume is always
# brought to the current schema before the app accepts traffic.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
