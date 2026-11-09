FROM python:3.11-slim AS base

WORKDIR /app

# System deps (libgomp1 required for LightGBM, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 curl unzip && rm -rf /var/lib/apt/lists/*

# Install AWS CLI (for S3 data sync at startup)
RUN curl -s "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws/

# Python deps (cached layer -- only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Bundle models for first-deploy bootstrap (copied to EFS on first run)
# Primary: v3_classifier_holdout  Fallback: v2_final
RUN mkdir -p models_bundled && \
    cp -r models/v3_classifier_holdout models_bundled/v3_classifier_holdout 2>/dev/null || true && \
    cp -r models/v2_final models_bundled/v2_final 2>/dev/null || true

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]
