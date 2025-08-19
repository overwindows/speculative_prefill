from speculative_prefill.vllm_patch.executor.gpu_executor import (
    PatchedGPUExecutor, PatchedGPUExecutorAsync)


def patch_executor():
    # In vLLM 0.8.4+, gpu_executor was replaced with uniproc_executor
    try:
        from vllm.executor import gpu_executor
        gpu_executor.GPUExecutor = PatchedGPUExecutor
        gpu_executor.GPUExecutorAsync = PatchedGPUExecutorAsync
    except ImportError:
        # For vLLM 0.8.4+
        from vllm.executor import uniproc_executor
        uniproc_executor.UniProcExecutor = PatchedGPUExecutor
        uniproc_executor.UniProcExecutorAsync = PatchedGPUExecutorAsync