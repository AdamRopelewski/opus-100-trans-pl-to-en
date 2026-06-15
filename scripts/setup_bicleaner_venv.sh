#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-bicleaner"

rm -rf "$VENV_DIR"
python -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel

# TensorFlow nightly GPU stack for RTX 5070 / Blackwell.
python -m pip install -U \
  "tf-nightly[and-cuda]==2.21.0.dev20260203" \
  "tf-keras-nightly==2.21.0.dev2026061509" \
  "keras-nightly==3.15.0.dev2026061505"

# Bicleaner deps manually, so pip does not replace tf-nightly with stable tensorflow.
python -m pip install -U \
  scikit-learn \
  PyYAML \
  pytest \
  toolwrapper \
  sentencepiece \
  bicleaner-ai-glove==0.2.1 \
  "transformers==4.57.*" \
  "huggingface-hub<1,>=0.30" \
  regex \
  tqdm \
  scipy

python -m pip install --no-deps "bicleaner-ai==3.6"

cat > "$VENV_DIR/bin/activate-bicleaner-cuda" <<'EOF'
VENV_SITE="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

CUDA_LIBS=(
  "$VENV_SITE/nvidia/cuda_runtime/lib"
  "$VENV_SITE/nvidia/cublas/lib"
  "$VENV_SITE/nvidia/cuda_cupti/lib"
  "$VENV_SITE/nvidia/cudnn/lib"
  "$VENV_SITE/nvidia/cufft/lib"
  "$VENV_SITE/nvidia/curand/lib"
  "$VENV_SITE/nvidia/cusolver/lib"
  "$VENV_SITE/nvidia/cusparse/lib"
  "$VENV_SITE/nvidia/nccl/lib"
  "$VENV_SITE/nvidia/nvjitlink/lib"
)

CUDA_LD_PATH=""
for path in "${CUDA_LIBS[@]}"; do
  if [ -d "$path" ]; then
    CUDA_LD_PATH="${CUDA_LD_PATH:+$CUDA_LD_PATH:}$path"
  fi
done

export LD_LIBRARY_PATH="${CUDA_LD_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=1
EOF

chmod +x "$VENV_DIR/bin/activate-bicleaner-cuda"

. "$VENV_DIR/bin/activate-bicleaner-cuda"

python - <<'PY'
import tensorflow as tf
print("tf:", tf.__version__)
print("gpu:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("ERROR: TensorFlow does not see GPU")
PY

python - <<'PY'
import bicleaner_ai
print("bicleaner-ai import: ok")
PY

echo "Bicleaner venv ready: $VENV_DIR"
echo "Use: source $VENV_DIR/bin/activate && source $VENV_DIR/bin/activate-bicleaner-cuda"