import os
import time
from typing import Any, Dict
import ray
from typing import Any, AsyncGenerator, Optional
import json

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics

import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import ModelConfig, CacheConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
import queue
from fastapi.responses import JSONResponse, Response, StreamingResponse
import asyncio

class VllmClient(LLMClient):
    """Client for LiteLLM Completions API."""
   
    def __init__(self, model: str, num_clients: int):
        args = AsyncEngineArgs(
            model=model,
            max_num_seqs=256,
            max_seq_len_to_capture=131072,
            swap_space=0,
            kv_cache_dtype=os.getenv('KV_CACHE_DTYPE', 'auto'),
            dtype='half',
            gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=os.environ.get('ENFORCE_EAGER','0')=='1',
            trust_remote_code=True,
            max_num_batched_tokens=131072,
            disable_log_requests=True,
            enable_chunked_prefill=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.num_clients = num_clients
        self.num_request = 0
        self.queue = queue.Queue()
    
    def get_next_ready(self):
        if self.num_request >= self.num_clients:
            outs = []
            for _ in range(self.num_clients):
                out = self.queue.get()
            outs.append(out)
            self.num_request -= self.num_clients
            return outs
        return []

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt
        
        self.num_request += 1
        request_id = f'req-{self.num_request}'
        results_generator = self.engine.generate(
            inputs=prompt,
            sampling_params=SamplingParams(
                **request_config.sampling_params
            ),
            request_id=request_id
        )
        
        # Streaming case
        @asyncio.coroutine
        async def stream_results():
            metrics = {}

            metrics[common_metrics.ERROR_CODE] = None
            metrics[common_metrics.ERROR_MSG] = ""
            
            t = time.monotonic()
            ttft = 0
            time_to_next_token = []
            tokens_received = 0
            total_request_time = 0
            generated_text = ""
            async for request_output in results_generator:
                prompt = request_output.prompt
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
                # ret = {"text": text_outputs}
                # dump = (json.dumps(ret) + "\0").encode("utf-8")
                # print(dump)
                generated_text = text_outputs[0]
                interval = time.monotonic() - t
                if ttft == 0:
                    ttft = interval
                else:
                    time_to_next_token.append(interval)
                tokens_received += 1
                total_request_time += interval
                t = time.monotonic()
            
            metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
            metrics[common_metrics.TTFT] = ttft
            metrics[common_metrics.E2E_LAT] = total_request_time
            metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = len(time_to_next_token) / total_request_time
            metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
            metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
            metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
            result = (metrics, generated_text, request_config)
            self.queue.put(result)
            
            print('done')
            # while True:
            #     await asyncio.sleep(1)
        
        def loop_in_thread(loop):
            asyncio.set_event_loop(loop)
            asyncio.start(stream_results())

        loop = asyncio.get_event_loop()
        import threading
        t = threading.Thread(target=loop_in_thread, args=(loop,))
        t.start()
        
        return t