import os
from typing import Callable, List, Optional, Tuple, Type, Union

from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase


class PatchedUniProcExecutor(UniProcExecutor):
    def _init_executor(self) -> None:
        """Override executor initialization to inject speculative worker when SPEC_MODEL is set."""
        
        # Check if we need to use speculative prefill worker
        if os.environ.get("SPEC_MODEL", None):
            # Update the worker class in the config to use our speculative worker
            self.vllm_config.parallel_config.worker_cls = "speculative_prefill.vllm_patch.worker.spec_prefill_worker.create_spec_worker"
        
        # Call the parent initialization with potentially modified config
        super()._init_executor()


# Maintain backward compatibility with old naming
PatchedGPUExecutor = PatchedUniProcExecutor

# Note: PatchedGPUExecutorAsync is no longer needed in vLLM 0.10.1.1
# since async functionality is built into UniProcExecutor