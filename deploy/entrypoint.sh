#!/bin/bash
set -e

DATA_DIR="${EFS_MOUNT:-/data}"
PRIMARY_MODEL="v3_classifier_holdout"
FALLBACK_MODEL="v2_final"

# Ensure EFS directories exist
mkdir -p "$DATA_DIR/outputs" \
         "$DATA_DIR/models/$PRIMARY_MODEL" \
         "$DATA_DIR/models/$FALLBACK_MODEL"

# Symlink persistent paths so app code finds them at expected __file__-relative locations
for db in cache.db lines.db; do
    [ -f "$DATA_DIR/$db" ] && ln -sf "$DATA_DIR/$db" /app/$db
done

# Symlink outputs to EFS
for dir in outputs; do
    rm -rf /app/$dir
    ln -sf "$DATA_DIR/$dir" /app/$dir
done

# Models: symlink each model dir from EFS into the container
for mdir in "$PRIMARY_MODEL" "$FALLBACK_MODEL"; do
    if [ -d "$DATA_DIR/models/$mdir" ] && [ "$(ls -A "$DATA_DIR/models/$mdir" 2>/dev/null)" ]; then
        rm -rf "/app/models/$mdir"
        ln -sf "$DATA_DIR/models/$mdir" "/app/models/$mdir"
    fi
done

# Initialize cache.db if first run
if [ ! -f "$DATA_DIR/cache.db" ]; then
    touch "$DATA_DIR/cache.db"
    ln -sf "$DATA_DIR/cache.db" /app/cache.db
fi

# Always sync bundled models to EFS (ensures new models like steals/blocks get deployed)
if [ -d "/app/models_bundled/$PRIMARY_MODEL" ]; then
    echo "Syncing $PRIMARY_MODEL models to EFS..."
    cp -r "/app/models_bundled/$PRIMARY_MODEL/"* "$DATA_DIR/models/$PRIMARY_MODEL/"
fi

# Fallback model (v2_final)
if [ ! -f "$DATA_DIR/models/$FALLBACK_MODEL/training_meta.json" ]; then
    if [ -d "/app/models_bundled/$FALLBACK_MODEL" ]; then
        echo "Bootstrapping fallback $FALLBACK_MODEL models to EFS..."
        cp -r "/app/models_bundled/$FALLBACK_MODEL/"* "$DATA_DIR/models/$FALLBACK_MODEL/"
    fi
fi

# Pull training data from S3 if not on EFS (needed for feature builder)
TRAINING_DATA_DIR="$DATA_DIR/outputs"
if [ ! -f "$TRAINING_DATA_DIR/backtest_2023_24_data.json" ] && [ -n "$S3_BUCKET" ]; then
    echo "Downloading training data from S3..."
    aws s3 sync "s3://${S3_BUCKET}/training-data/" "$TRAINING_DATA_DIR/" \
        --exclude "*" \
        --include "backtest_*.json" \
        --quiet 2>/dev/null || echo "S3 training data sync failed (will use bundled if available)"
fi

# Fallback: copy from bundled image if S3 failed and data exists in image
if [ ! -f "$TRAINING_DATA_DIR/backtest_2023_24_data.json" ] && [ -f "/app/outputs_bundled/backtest_2023_24_data.json" ]; then
    echo "Using bundled training data..."
    cp /app/outputs_bundled/backtest_*.json "$TRAINING_DATA_DIR/" 2>/dev/null || true
fi

# Set AIBALL_MODELS_DIR to primary model if not already set
export AIBALL_MODELS_DIR="${AIBALL_MODELS_DIR:-$DATA_DIR/models/$PRIMARY_MODEL}"

# Validate required env vars for web mode
if [ "${1:-web}" = "web" ]; then
    missing=""
    [ -z "$PT_PASSWORD_HASH" ] && missing="$missing PT_PASSWORD_HASH"
    [ -z "$SESSION_SECRET" ] && missing="$missing SESSION_SECRET"
    if [ -n "$missing" ]; then
        echo "WARNING: Missing env vars:$missing -- login will not work"
    fi
fi

echo "Starting PropTerminal (mode: ${1:-web}, model: $AIBALL_MODELS_DIR)..."

case "${1:-web}" in
  web)
    exec uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 1
    ;;
  grade)
    exec python grade_predictions.py --last 1
    ;;
  retrain)
    python -m v2.train
    # Sync new models to S3 on success
    if [ -n "$S3_BUCKET" ]; then
        pip install -q boto3 2>/dev/null
        aws s3 sync "$DATA_DIR/models/" "s3://${S3_BUCKET}/models/" --quiet || echo "S3 sync failed (non-fatal)"
    fi
    ;;
  check-drift)
    exec python grade_predictions.py --last 7 --summary-only
    ;;
  analyze-all)
    exec python ball.py --today
    ;;
  *)
    exec "$@"
    ;;
esac
