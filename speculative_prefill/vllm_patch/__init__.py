# Apply argparse compatibility patch FIRST, before any vLLM imports
import argparse
import sys

# Patch argparse for Python < 3.13 compatibility with vLLM 0.10.1.1
if sys.version_info < (3, 13):
    original_parser_add_argument = argparse.ArgumentParser.add_argument
    original_group_add_argument = argparse._ArgumentGroup.add_argument
    
    def patched_add_argument(self, *args, **kwargs):
        kwargs.pop('deprecated', None)
        return original_parser_add_argument(self, *args, **kwargs)
    
    def patched_group_add_argument(self, *args, **kwargs):
        kwargs.pop('deprecated', None)
        return original_group_add_argument(self, *args, **kwargs)
    
    argparse.ArgumentParser.add_argument = patched_add_argument
    argparse._ArgumentGroup.add_argument = patched_group_add_argument
    print(f"Applied argparse compatibility patch for Python {sys.version_info.major}.{sys.version_info.minor}")

import atexit
import os
from typing import Optional

import torch
import torch.distributed

from speculative_prefill.vllm_patch.config import init_spec_config
from speculative_prefill.vllm_patch.data import patch_data
from speculative_prefill.vllm_patch.executor import patch_executor
from speculative_prefill.vllm_patch.scheduler import patch_scheduler


def patch_platform_worker_selection():
    """
    Patch the platform's check_and_update_config to inject speculative worker
    when SPEC_MODEL environment variable is set.
    """
    from vllm.platforms.cuda import NvmlCudaPlatform
    import os
    
    # Store the original method
    original_check_and_update_config = NvmlCudaPlatform.check_and_update_config
    
    @classmethod
    def patched_check_and_update_config(cls, vllm_config):
        # Call the original method first
        original_check_and_update_config(vllm_config)
        
        # Override worker_cls if SPEC_MODEL is set
        if os.environ.get("SPEC_MODEL", None):
            print(f"Overriding worker_cls for speculative prefill: SPEC_MODEL={os.environ.get('SPEC_MODEL')}")
            vllm_config.parallel_config.worker_cls = "speculative_prefill.vllm_patch.worker.spec_prefill_worker.create_spec_worker"
    
    # Apply the patch
    NvmlCudaPlatform.check_and_update_config = patched_check_and_update_config
    print("Applied platform worker selection patch for CUDA")

_TITLE = """
|=========================================================================================|
|                                                                                         |
|  ███████╗██████╗ ███████╗ ██████╗██╗   ██╗██╗      █████╗ ████████╗██╗██╗   ██╗███████╗ |
|  ██╔════╝██╔══██╗██╔════╝██╔════╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝ |
|  ███████╗██████╔╝█████╗  ██║     ██║   ██║██║     ███████║   ██║   ██║██║   ██║█████╗   |
|  ╚════██║██╔═══╝ ██╔══╝  ██║     ██║   ██║██║     ██╔══██║   ██║   ██║╚██╗ ██╔╝██╔══╝   |
|  ███████║██║     ███████╗╚██████╗╚██████╔╝███████╗██║  ██║   ██║   ██║ ╚████╔╝ ███████╗ |
|  ╚══════╝╚═╝     ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝ |
|      ██████╗ ██████╗ ███████╗███████╗██╗██╗     ██╗     ██╗███╗   ██╗ ██████╗           |
|      ██╔══██╗██╔══██╗██╔════╝██╔════╝██║██║     ██║     ██║████╗  ██║██╔════╝           |
|      ██████╔╝██████╔╝█████╗  █████╗  ██║██║     ██║     ██║██╔██╗ ██║██║  ███╗          |
|      ██╔═══╝ ██╔══██╗██╔══╝  ██╔══╝  ██║██║     ██║     ██║██║╚██╗██║██║   ██║          |
|      ██║     ██║  ██║███████╗██║     ██║███████╗███████╗██║██║ ╚████║╚██████╔╝          |
|      ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝           |
|                                                                                         |
|=========================================================================================|
| Notes:                                                                                  |
|    - Currently support Llama and Qwen model as the base and speculator.                 |
|    - Currently does not support chunked prefill, use enable_chunked_prefill=False       |
|    - Recommend to set gpu_memory_utilization when using tensor_parallel_size > 1        |
|    - Please use enforce_eager=True, which makes long context task correct.              |
|=========================================================================================|
"""



def clean_up_fn():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def enable_prefill_spec(
    spec_model: str = 'meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path: Optional[str] = None
):
    print(_TITLE)
    print("Setting up environment vars...")
    os.environ.setdefault("SPEC_MODEL", spec_model)
    if spec_config_path is not None:
        os.environ.setdefault("SPEC_CONFIG_PATH", spec_config_path)
    
    # Force vLLM to use V0 engine since Speculative Prefill is designed for V0
    os.environ.setdefault("VLLM_USE_V1", "0")
    print("Forcing vLLM to use V0 engine (VLLM_USE_V1=0) for Speculative Prefill compatibility")

    init_spec_config()

    print("Applying speculative prefill vllm monkey patch...")
    # breakpoint()
    # assert False, "This is a temporary patch for vllm, please use the official vllm patch instead."
    patch_platform_worker_selection()
    patch_executor()
    patch_scheduler()
    patch_data()

    atexit.register(clean_up_fn)
