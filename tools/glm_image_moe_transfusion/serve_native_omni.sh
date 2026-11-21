#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/share/d00911077/glm-image
MODEL_CONFIG="${ROOT}/native_stages_v2/diffusion"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

source "${ROOT}/vllm-omni-venv/bin/activate"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export VLLM_OMNI_API_SERVER_TORCH_THREADS="${VLLM_OMNI_API_SERVER_TORCH_THREADS:-64}"
export PYTHONPATH="${ROOT}/vllm-omni:${PYTHONPATH:-}"
export GLM_IMAGE_MOE_CHECKPOINT="${ROOT}/g/dmd2_lyzsft10000_s2000"
export GLM_IMAGE_MOE_STAGE_CONFIG="${ROOT}/native_stages_v2/ar"
export GLM_IMAGE_MOE_TOKENIZER_PATH="${ROOT}/assets/glm_image_tokenizer"
export GLM_IMAGE_MOE_VAE_PATH="${ROOT}/assets/mnt/cogview_data_cloud/zwd/ckpt/flux2_vae_hf"
export GLM_IMAGE_MOE_VAE_DEVICE="${GLM_IMAGE_MOE_VAE_DEVICE:-cpu}"

exec vllm-omni serve "${MODEL_CONFIG}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --enforce-eager \
  "$@"
