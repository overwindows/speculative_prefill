from cgi import print_directory
import math
from copy import deepcopy
from functools import partial
from types import MethodType
from typing import List, Optional, Tuple, Union

from speculative_prefill.vllm_patch.scheduler import logger
import torch
# from vllm.attention import AttentionMetadata  # Unused import
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.distributed.communication_op import (broadcast_tensor_dict,
                                               tensor_model_parallel_gather)
from vllm.model_executor.layers.sampler import SamplerOutput
try:  # Optional: llama classes (kept for type hints / backward compat)
    from vllm.model_executor.models.llama import (  # type: ignore
        LlamaAttention,  # noqa: F401
        LlamaDecoderLayer,  # noqa: F401
        LlamaForCausalLM,  # noqa: F401
    )
except Exception:  # pragma: no cover - absence is fine
    LlamaAttention = LlamaDecoderLayer = LlamaForCausalLM = None  # type: ignore

try:  # Optional: qwen (qwen2/qwen3) classes
    from vllm.model_executor.models.qwen import (  # type: ignore
        QwenAttention,  # noqa: F401
        QwenDecoderLayer,  # noqa: F401
        QwenForCausalLM,  # noqa: F401
    )
except Exception:  # pragma: no cover
    QwenAttention = QwenDecoderLayer = QwenForCausalLM = None  # type: ignore
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.worker.worker import Worker

from speculative_prefill.vllm_patch.config import get_spec_config
from speculative_prefill.vllm_patch.data.sequence import AugmentedSequenceData


def _forward_with_query_dump(
    self,  # model-specific attention module (llama / qwen / etc.)
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    query_buffer: List[List[torch.Tensor]],
    layer_idx: int,
) -> torch.Tensor:
    """Patched attention forward capturing query tensors.

    Updated for vLLM 0.10.1.1 where the attention forward signature is simplified.
    We capture the query tensors before calling the original attention forward.
    """
    # 1. Obtain q, k, v projections for query capture
    if hasattr(self, "qkv_proj"):
        # Llama (and many others in vLLM) fused path
        qkv, _ = self.qkv_proj(hidden_states)
        # Attribute names: q_size / kv_size are present in llama; fall back if absent
        q_size = getattr(self, "q_size", None)
        kv_size = getattr(self, "kv_size", None)
        if q_size is None or kv_size is None:
            # Derive assuming head counts exist
            num_q_heads = getattr(self, "num_heads", None) or getattr(self, "num_attention_heads", None)
            num_kv_heads = getattr(self, "num_kv_heads", None) or getattr(self, "num_key_value_heads", None) or num_q_heads
            head_dim = getattr(self, "head_size", None) or getattr(self, "head_dim", None)
            if num_q_heads and head_dim:
                q_size = num_q_heads * head_dim
            if num_kv_heads and head_dim:
                kv_size = num_kv_heads * head_dim
        if q_size is None or kv_size is None:
            raise RuntimeError("Unable to infer q_size/kv_size for fused qkv projection.")
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    else:
        # Separate projections (possible in some Qwen variants)
        if not all(hasattr(self, attr) for attr in ["q_proj", "k_proj", "v_proj"]):
            raise RuntimeError("Attention module missing both fused qkv_proj and separate q/k/v projections.")
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

    # 2. Rotary embeddings (if provided)
    if hasattr(self, "rotary_emb"):
        q, k = self.rotary_emb(positions, q, k)
    elif hasattr(self, "apply_rotary_pos_emb"):
        # Some implementations use a different helper
        q, k = self.apply_rotary_pos_emb(q, k, positions)
    # 3. Store query for speculative prefill (simplified for vLLM 0.10.1.1)
    # For prefill stage (iteration 0): only store the last token's query
    # For decode stages (iteration > 0): store all queries  
    # Prefill stage: extract and store only the last token's query
    # In prefill, q contains queries for all tokens in the sequence
    # We only want the query for the last token (which will generate subsequent tokens)
    print(q.shape, layer_idx)
    if q.shape[0] > 1:
        # Take the last token along the sequence dimension
        last_token_query = q[-1:] if q.shape[0] > 1 else q
        query_buffer[layer_idx].append(last_token_query)
    else:
        # Fallback: if tensor structure is unexpected, store as-is
        query_buffer[layer_idx].append(q)
    # 4. Call the original attention forward method
    # In vLLM 0.10.1.1, we delegate to the original implementation
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


class LookAheadSpecWorker(Worker):
    def load_model(self):
        super().load_model()
        logger.info("Loading model for look-ahead spec worker")
        # Support llama & qwen families (and potentially others) by duck-typing.
        model_cls_name = self.model_runner.model.__class__.__name__
        supported_markers = ["Llama", "Qwen"]
        if not any(marker in model_cls_name for marker in supported_markers):
            raise NotImplementedError(
                f"LookAheadSpecWorker currently supports models containing any of {supported_markers} in class name; got {model_cls_name}."
            )
        
        self.spec_config = get_spec_config()
        self.current_iteration = 0  # Track current speculation iteration

        self._prepare_query_buffer()

        # do patching here to allow model output attention
        for layer_idx, layer in enumerate(self.model_runner.model.model.layers):
            # Accept decoder layers with a self_attn attribute
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(
                    f"Layer {layer_idx} missing self_attn; unsupported architecture for look-ahead spec patching."
                )
            # Store the original forward method before patching
            layer.self_attn.forward = MethodType(
                partial(
                    _forward_with_query_dump,
                    query_buffer=self.query_buffer,
                    layer_idx=layer_idx,
                ),
                layer.self_attn,
            )
            
        cfg = self.model_runner.model.config
        # Fallback attribute names used across model families
        num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_kv_heads", None) or num_heads
        head_dim = getattr(cfg, "head_dim", None) or getattr(cfg, "hidden_size", None) // num_heads if num_heads and getattr(cfg, "hidden_size", None) else None
        if not all([num_heads, num_kv_heads, head_dim]):
            raise RuntimeError("Could not infer (num_heads, num_kv_heads, head_dim) for model config.")

        self._reshape_key = partial(
            torch.repeat_interleave,
            dim=1,
            repeats=(num_heads // num_kv_heads),
        )

        self._reshape_query = lambda x: x.reshape(-1, num_heads, head_dim)

    def _prepare_query_buffer(self):
        if not hasattr(self, "query_buffer"):
            self.query_buffer = [
                [] for _ in range(self._get_model_num_layers())
            ]
        else:
            # using clear instead of resetting prevents losing ref
            for qb in self.query_buffer:
                qb.clear()

    @torch.inference_mode
    def speculate(
        self, 
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[ExecuteModelRequest]:
        
        look_ahead_cnt = self.spec_config.look_ahead_cnt
        assert look_ahead_cnt > 0

        # this decides on the behavior
        is_driver = self.rank == 0

        if is_driver:
            assert execute_model_req.virtual_engine == 0

            self._raise_if_unsupported(execute_model_req)

            request, non_empty = self._extract_prompt_execute_model_req(
                execute_model_req=execute_model_req, 
                look_ahead_cnt=look_ahead_cnt
            )

            if not non_empty:
                raise ValueError("No prefill prompts")
        else:
            request = None

        self._prepare_query_buffer()
        # Reset iteration counter at the start of speculation
        self.current_iteration = 0

        model_outputs: List[SamplerOutput] = []
        for itr in range(look_ahead_cnt):
            # Update current iteration for query storage filtering
            self.current_iteration = itr
            
            if itr == 0:
                prefill_slot_mapping = self._get_prefill_slot_mapping(request)

            model_output: List[SamplerOutput] = self.execute_model(
                execute_model_req=request)
            assert len(model_output) in [0, 1], \
                "composing multistep workers not supported"
            
            if is_driver:
                model_output = model_output[0]
                print(itr, model_output)
                self._append_new_tokens(
                    model_output, 
                    request.seq_group_metadata_list, 
                    itr
                )
                model_outputs.append(model_output)

        if is_driver:
            # assume the same eos ids across requests
            actual_look_ahead_cnts = self._get_actual_look_ahead_cnts(
                model_outputs, 
                [128001, 128008, 128009] # hard coded for now
            )
        else:
            actual_look_ahead_cnts = None

        # basically aggregate over heads
        query_buffer = self._get_query_buffer()
        key_buffer = self._get_key_buffer(
            slot_mapping=prefill_slot_mapping, 
            kv_cache=self.kv_cache[0]
        )

        if not is_driver:
            return None

        # sample list of [num_layer, num_head, look_ahead_cnt, context_len]
        attn_scores = self._get_attention_scores(
            query_buffer=query_buffer, 
            key_buffer=key_buffer, 
            actual_look_ahead_cnts=actual_look_ahead_cnts
        )
        
        # a list of 1D tensor with size prefill len
        token_importance = self._token_importance_from_attn_scores(
            attn_scores=attn_scores
        )

        # get token indices that decide on what to keep
        kept_indices = self._get_kept_indices_from_token_importance(
            token_importance=token_importance
        )

        return self._reassemble_execute_model_req(
            execute_model_req=execute_model_req, 
            kept_indices=kept_indices
        )
    
    def _get_query_buffer(self) -> torch.Tensor | None:
        # turn the list of list of tensors into a tensor and gather
        # over the ranks
        layer_query_buffers = []
        for layer_qb in self.query_buffer:
            layer_query_buffers.append(torch.stack(layer_qb, dim=0))
        query_buffer = torch.stack(layer_query_buffers, dim=0)
        return tensor_model_parallel_gather(query_buffer, dst=0, dim=-1)

    def _get_key_buffer(
        self, 
        slot_mapping: Optional[List[torch.Tensor]], 
        kv_cache: List[torch.Tensor] 
    ) -> List[List[torch.Tensor]] | None:
        all_keys = []

        key_lens = None

        for layer_idx in range(self._get_model_num_layers()):
            # (num_blocks, block_size, num_kv_heads, head_size)
            key_cache = kv_cache[layer_idx][0]
            block_size = key_cache.shape[1]

            layer_keys = []
            
            for sm in slot_mapping:
                block_indices = sm // block_size
                block_pos = sm % block_size
                key = key_cache[block_indices, block_pos, :, :]
                layer_keys.append(key)

            if key_lens is None:
                key_lens = [
                    len(k) for k in layer_keys
                ]

            all_keys.append(torch.concat(layer_keys, dim=0))
        
        all_keys = torch.stack(all_keys, dim=0)
        
        all_keys = tensor_model_parallel_gather(all_keys, dst=0, dim=-2)

        if all_keys is not None:
            # split
            assert len(key_lens) > 0
            # first turn them into layer list
            all_keys = torch.split(
                all_keys, 
                split_size_or_sections=1, 
                dim=0
            )

            # split along sample dim
            all_keys = [
                torch.split(
                    lk.squeeze(0), 
                    split_size_or_sections=key_lens, 
                    dim=0
                ) for lk in all_keys
            ]
        
        return all_keys

    def _get_actual_look_ahead_cnts(
        self, 
        model_outputs: List[SamplerOutput], 
        stop_token_ids: List[int]
    ) -> List[int]:
        actual_look_ahead_cnts = [
            self.spec_config.look_ahead_cnt
        ] * len(model_outputs[0].outputs)

        if self.spec_config.ignore_eos:
            return actual_look_ahead_cnts

        for step, mo in enumerate(model_outputs):
            outputs = mo.outputs
            for idx, stop in enumerate([output.samples[0].output_token in stop_token_ids \
                for output in outputs]):
                if stop:
                    actual_look_ahead_cnts[idx] = min(
                        actual_look_ahead_cnts[idx], 
                        step + 1
                    )

        return actual_look_ahead_cnts

    def _reassemble_execute_model_req(
        self, 
        execute_model_req: ExecuteModelRequest, 
        kept_indices: List[torch.LongTensor]
    ) -> ExecuteModelRequest:
        # now we need to update the original request
        index_pos = 0
        new_seq_group_metadata_list = []
        for metadata in execute_model_req.seq_group_metadata_list:
            if metadata.is_prompt:
                cur_indices = kept_indices[index_pos].cpu()
                index_pos += 1

                # get the seq data
                assert len(metadata.seq_data) == 1
                seq_id = metadata.get_first_seq_id()
                seq_data = metadata.seq_data[seq_id]
                
                prompt_token_ids = seq_data._prompt_token_ids
                output_token_ids = seq_data._output_token_ids
                
                new_seq_data = AugmentedSequenceData.from_seqs_and_pos_ids(
                    prompt_token_ids=torch.LongTensor(prompt_token_ids)[cur_indices], 
                    position_ids=cur_indices.tolist(), 
                    output_token_ids=output_token_ids
                )

                metadata.seq_data[seq_id] = new_seq_data

            new_seq_group_metadata_list.append(metadata)

        assert index_pos == len(kept_indices)
        return execute_model_req.clone(
            seq_group_metadata_list=new_seq_group_metadata_list)
    
    def _get_kept_indices_from_token_importance(
        self, 
        token_importance: List[torch.Tensor]
    ) -> List[torch.LongTensor]:
        
        kept_indices = []
        for sample_ti in token_importance: 
            seq_len = len(sample_ti)
            
            percentage = self.spec_config.keep_kwargs.get("percentage", 1.0)

            if self.spec_config.keep_kwargs.get("chunk", False):
                chunk_size = self.spec_config.keep_kwargs.get("chunk_size", 32)
                chunk_ti = torch.split(sample_ti, chunk_size, dim=-1)
                chunk_ti = [cti.mean() for cti in chunk_ti]
                chunk_cnt = len(chunk_ti)
                
                keep_chunk_cnt = math.ceil(chunk_cnt * percentage)
                _, chunk_indices = torch.topk(torch.stack(chunk_ti), k=keep_chunk_cnt, dim=-1)
                indices = torch.split(torch.arange(seq_len), chunk_size, dim=-1)
                indices = torch.concat([indices[ci.item()] for ci in chunk_indices])
            else:
                topk = math.ceil(seq_len * percentage)
                _, indices = torch.topk(sample_ti, k=topk, dim=-1)

            kept_indices.append(torch.sort(indices)[0])

        return kept_indices

    def _token_importance_from_attn_scores(
        self, 
        attn_scores: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        token_importance: List[torch.Tensor] = []

        for attn in attn_scores:
            # [num_layer, num_head, look_ahead_cnt, context_len]
            
            # softmax
            original_dtype = attn.dtype
            attn = torch.nn.functional.softmax(
                attn, 
                dim=-1, 
                dtype=torch.float32
            ).to(original_dtype)

            # aggregate layer and heads
            attn = attn.flatten(0, 1)

            # smooth if we need to
            kernel_size = self.spec_config.pool_kernel_size
            if kernel_size:
                attn = torch.nn.functional.avg_pool1d(
                    attn, 
                    kernel_size=kernel_size, 
                    padding=kernel_size // 2,
                    stride=1
                )

            # max over heads and layers
            attn = attn.max(0)[0]
            # average over look ahead cnt
            attn = attn.mean(0)

            token_importance.append(attn)

        return token_importance

    def _get_model_num_layers(self) -> int:
        return getattr(
            self.model_runner.model_config.hf_text_config, "num_hidden_layers")

    def _get_attention_scores(
        self, 
        query_buffer: torch.Tensor, 
        # because each sample might have diff number of keys
        key_buffer: List[List[torch.Tensor]], 
        actual_look_ahead_cnts: List[int], 
    ) -> List[torch.Tensor]:
        """
            A caveat or potentially a good design here:
                For look ahead tokens, we also just attend to context, 
                not generated tokens
        """

        attn_weights = []

        # first lets get the queries out from query_buffer
        for layer_idx in range(len(query_buffer)):
            attn_weights.append([])

            keys = key_buffer[layer_idx]
            queries = query_buffer[layer_idx]
            queries = torch.split(
                queries, 
                split_size_or_sections=1, 
                dim=1)
            
            for q, k, c in zip(queries, keys, actual_look_ahead_cnts):
                # (len, num_heads, head_dim)
                query = self._reshape_query(q).transpose(0, 1)
                key = self._reshape_key(k).transpose(0, 1)
                
                # take off things if we hit EOS
                query = query[:, :c, :]

                attn = torch.matmul(
                    query, key.transpose(-1, -2)
                ) / math.sqrt(query.shape[-1])

                attn_weights[-1].append(attn)

        # reshape -> from [layer, sample, attn] to [sample, layer, attn]
        reshaped_attn_weights: List[torch.Tensor] = []
        for sample_idx in range(len(attn_weights[0])):
            reshaped_attn_weights.append(
                torch.stack(
                    [attn_weights[layer_idx][sample_idx] for layer_idx in range(len(attn_weights))], 
                    dim=0))

        return reshaped_attn_weights

    def _get_keys_from_slot_mapping(
        self, 
        slot_mapping: List[torch.Tensor], 
        layer_idx: int, 
        kv_cache: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        # (num_blocks, block_size, num_kv_heads, head_size)
        key_cache = kv_cache[layer_idx][0]
        block_size = key_cache.shape[1]

        keys = []
        
        for sm in slot_mapping:
            block_indices = sm // block_size
            block_pos = sm % block_size
            key = key_cache[block_indices, block_pos, :, :]
            keys.append(key)
            
        return keys

    def _get_prefill_slot_mapping(
        self, 
        execute_model_req: Optional[ExecuteModelRequest] = None, 
    ) -> Union[None, Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]]:
        model_input, _,  _ = self.prepare_input(
            execute_model_req
        )

        self.model_runner.attn_state.begin_forward(model_input)
        
        if model_input.attn_metadata is not None:
            # driver
            prefill_metadata: FlashAttentionMetadata = model_input.attn_metadata.prefill_metadata
            assert prefill_metadata is not None

            slot_mapping = list(torch.split(
                prefill_metadata.slot_mapping, 
                prefill_metadata.seq_lens_tensor.tolist()))
            broadcast_tensor_dict({
                "slot_mapping": slot_mapping
            }, src=0)
            return slot_mapping
        else:
            # non driver
            return broadcast_tensor_dict(src=0)["slot_mapping"]

    def _extract_prompt_execute_model_req(
        self, 
        execute_model_req: ExecuteModelRequest, 
        look_ahead_cnt: int
    ) -> Tuple[ExecuteModelRequest, bool]:
        prompt_seq_group_metadata_list = [
            deepcopy(sgm) for sgm in execute_model_req.seq_group_metadata_list \
            if sgm.is_prompt
        ]

        request = execute_model_req.clone(
            seq_group_metadata_list=prompt_seq_group_metadata_list, 
        )

        request.num_lookahead_slots = look_ahead_cnt

        return request, len(prompt_seq_group_metadata_list) > 0

    def _append_new_tokens(
        self,
        model_output: List[SamplerOutput],
        seq_group_metadata_list: List[SequenceGroupMetadata], 
        itr: int
    ) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done outside of the worker, but it is
        required if the worker is to perform multiple forward passes.
        """
        for seq_group_metadata, sequence_group_outputs in zip(
                seq_group_metadata_list, model_output):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                # NOTE: Beam search is not supported, so we can assume that
                # parent_seq_id == seq_id.
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                if itr == 0:
                    # prefill
                    new_token = seq.get_prompt_len()
                else:
                    # decode
                    new_token = 1
                seq.update_num_computed_tokens(new_token)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")
