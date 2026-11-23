# GLM-Image MoE Transfusion Baselines

These tools record reproducible direct and thinking serving latencies and compare
paired outputs from a reference runtime with vLLM-Omni. The two-stage service
selects direct or thinking mode per request through `think`, so changing modes
does not require restarting the service.

Start the two-stage service, then run the benchmark:

```bash
python benchmarks/diffusion/glm_image_moe_transfusion/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --mode direct \
  --prompt "一只在雪地里打滚的柴犬，浅景深，电影摄影" \
  --sizes 512 1024 --warmups 1 --repeats 2 \
  --output-dir /path/to/omni-direct
```

Use `--mode think` for thinking requests. GLM-Image routes `think=false`
directly to diffusion and `think=true` through the AR stage before diffusion.

The native diffusion pipeline supports request-level batching. Concurrent
requests with compatible resolution, step count, guidance, and output settings
share one diffusion forward up to the stage's `max_num_seqs`. A short
`request_batch_max_wait_ms` can improve burst coalescing without changing the
OpenAI request format.

The deploy profile keeps `max_num_seqs=1` as the conservative default. On
four-card 910B, 512x512 benefited from batching, while 1024x1024 was slower than
serial execution at the current kernel shapes. Raise the value per workload
after validating latency, throughput, and memory.

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
