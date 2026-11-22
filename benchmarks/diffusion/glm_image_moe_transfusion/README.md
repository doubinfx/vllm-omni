# GLM-Image MoE Transfusion Baselines

These tools record reproducible direct and thinking serving latencies and compare
paired outputs from a reference runtime with vLLM-Omni.

Start one serving mode at a time, then run the matching benchmark:

```bash
python benchmarks/diffusion/glm_image_moe_transfusion/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --mode direct \
  --prompt "一只在雪地里打滚的柴犬，浅景深，电影摄影" \
  --sizes 512 1024 --warmups 1 --repeats 2 \
  --output-dir /path/to/omni-direct
```

Use `--mode think` after restarting the service with the thinking topology. The
benchmark links the wait in each measured mode because GLM-Image selects its
topology at service startup.

Compare matching reference and candidate images:

```bash
python benchmarks/diffusion/glm_image_moe_transfusion/compare_images.py \
  --case no_think_512 /path/to/hygen/512.png /path/to/omni/512.png \
  --case no_think_1024 /path/to/hygen/1024.png /path/to/omni/1024.png \
  --output /path/to/quality.json
```

The comparison reports exact equality, mean/max absolute error, PSNR, SSIM and
basic color statistics that catch blank or constant images. Keep generated
images and full result JSON outside the source repository.

## A2 FlashAttentionScore vendor

On Ascend 910B, the diffusion attention adapter calls
`torch_npu.npu_fusion_attention`. If a matching custom operator package is
registered, its ACLNN `FlashAttentionScore` implementation replaces the stock
kernel. Set `GLM_IMAGE_MOE_A2_FA_VENDOR` to that package root so startup fails
when the expected 910B forward kernel is missing instead of silently selecting
another backend:

```bash
export GLM_IMAGE_MOE_A2_FA_VENDOR=/path/to/vendors/custom_transformer
source "${GLM_IMAGE_MOE_A2_FA_VENDOR}/bin/set_env.bash"
```

The custom operator path must be configured before `torch_npu` is imported.
