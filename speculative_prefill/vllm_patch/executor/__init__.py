from speculative_prefill.vllm_patch.executor.gpu_executor import PatchedUniProcExecutor


def patch_executor():
    # In vLLM 0.10.1.1, we need to patch UniProcExecutor to inject speculative workers
    from vllm.executor import uniproc_executor
    uniproc_executor.UniProcExecutor = PatchedUniProcExecutor
