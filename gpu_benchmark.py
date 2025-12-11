#!/usr/bin/env python3
"""
GPU Benchmark Module
Runs LLM inference using LLM models for GPU performance testing
Supports CUDA (NVIDIA) and MPS (Apple Silicon)
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional
import tempfile
import os
import platform


# Hugging Face token for gated repos
HF_TOKEN = "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"


@dataclass
class GPUBenchmarkResult:
    """Results from GPU benchmark"""
    model_name: str
    prompt: str
    response: str
    total_tokens: int
    prompt_tokens: int
    generated_tokens: int
    total_time: float
    tokens_per_second: float
    time_to_first_token: float
    prefill_time: float
    generation_time: float
    gpu_info: dict
    memory_used_gb: float
    memory_total_gb: float
    success: bool
    error_message: str = ""
    backend: str = "unknown"


class LLMBenchmark:
    """GPU Benchmark using LLM inference"""
    
    # Only these two models - no fallback
    MODELS = [
        "google/gemma-3-27b-it",      # Gemma 3 27B - gated, needs license
        "Qwen/Qwen2.5-32B-Instruct",  # Qwen 2.5 32B - open
    ]
    
    BENCHMARK_PROMPT = (
        "Tell me if the Unix ideology of software and communism are "
        "related to each other. Provide a detailed analysis comparing "
        "the philosophical foundations of both."
    )
    
    def __init__(self, venv_path: Path, hf_token: Optional[str] = None):
        self.venv_path = venv_path
        self.hf_token = hf_token or HF_TOKEN
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[float], None]] = None
        
    def set_callbacks(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
            
    def _update_progress(self, progress: float):
        if self.progress_callback:
            self.progress_callback(progress)
    
    def get_python_path(self) -> str:
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")
    
    def run_benchmark(self, model_id: Optional[str] = None) -> GPUBenchmarkResult:
        """Run the GPU benchmark"""
        
        self._log("=" * 60)
        self._log("GPU BENCHMARK - LLM Inference")
        self._log("=" * 60)
        self._log("Models: Gemma 3 27B / Qwen 2.5 32B")
        self._log("Requires ~40-60GB VRAM")
        self._log("")
        
        self._update_progress(0.1)
        
        benchmark_script = self._create_benchmark_script(model_id)
        
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(benchmark_script)
            script_path = f.name
        
        try:
            self._log("Starting benchmark...")
            self._update_progress(0.2)
            
            env = os.environ.copy()
            env['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')
            env['TOKENIZERS_PARALLELISM'] = 'false'
            env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
            
            if self.hf_token:
                env['HF_TOKEN'] = self.hf_token
                env['HUGGING_FACE_HUB_TOKEN'] = self.hf_token
            
            process = subprocess.Popen(
                [self.get_python_path(), script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            stdout_lines = []
            
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue
                    
                stdout_lines.append(line)
                
                if line.startswith("PROGRESS:"):
                    try:
                        progress = float(line.split(":")[1])
                        self._update_progress(0.2 + progress * 0.7)
                    except:
                        pass
                elif line.startswith("LOG:"):
                    self._log(line[4:].strip())
                elif not line.startswith(("PROGRESS:", "LOG:", "RESULT:")):
                    if "error" in line.lower() or "warning" in line.lower():
                        self._log(f"  {line[:100]}")
            
            process.wait()
            self._update_progress(0.95)
            
            result_json = None
            for line in stdout_lines:
                if line.startswith("RESULT:"):
                    try:
                        result_json = json.loads(line[7:])
                        break
                    except json.JSONDecodeError:
                        pass
            
            if result_json and result_json.get("success"):
                result = GPUBenchmarkResult(
                    model_name=result_json.get("model_name", "Unknown"),
                    prompt=self.BENCHMARK_PROMPT,
                    response=result_json.get("response", ""),
                    total_tokens=result_json.get("total_tokens", 0),
                    prompt_tokens=result_json.get("prompt_tokens", 0),
                    generated_tokens=result_json.get("generated_tokens", 0),
                    total_time=result_json.get("total_time", 0),
                    tokens_per_second=result_json.get("tokens_per_second", 0),
                    time_to_first_token=result_json.get("time_to_first_token", 0),
                    prefill_time=result_json.get("prefill_time", 0),
                    generation_time=result_json.get("generation_time", 0),
                    gpu_info=result_json.get("gpu_info", {}),
                    memory_used_gb=result_json.get("memory_used_gb", 0),
                    memory_total_gb=result_json.get("memory_total_gb", 0),
                    success=True,
                    backend=result_json.get("backend", "unknown")
                )
                
                self._log("")
                self._log("=" * 60)
                self._log("RESULTS")
                self._log("=" * 60)
                self._log(f"Backend: {result.backend.upper()}")
                self._log(f"Model: {result.model_name}")
                self._log(f"Generated Tokens: {result.generated_tokens}")
                self._log(f"Total Time: {result.total_time:.2f}s")
                self._log(f"Tokens/Second: {result.tokens_per_second:.2f}")
                if result.memory_used_gb > 0:
                    self._log(f"Memory Used: {result.memory_used_gb:.2f} GB")
                self._log("=" * 60)
                
            else:
                error_msg = result_json.get("error", "Unknown error") if result_json else "Failed to run benchmark"
                self._log(f"✗ Benchmark failed: {error_msg}")
                
                result = GPUBenchmarkResult(
                    model_name="Unknown",
                    prompt=self.BENCHMARK_PROMPT,
                    response="",
                    total_tokens=0,
                    prompt_tokens=0,
                    generated_tokens=0,
                    total_time=0,
                    tokens_per_second=0,
                    time_to_first_token=0,
                    prefill_time=0,
                    generation_time=0,
                    gpu_info={},
                    memory_used_gb=0,
                    memory_total_gb=0,
                    success=False,
                    error_message=error_msg
                )
            
            self._update_progress(1.0)
            return result
            
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
    
    def _create_benchmark_script(self, model_id: Optional[str] = None) -> str:
        """Create the benchmark script"""
        
        models_str = json.dumps(self.MODELS)
        
        return f'''#!/usr/bin/env python3
"""GPU Benchmark Script"""

import json
import time
import sys
import gc
import platform
import warnings
import os

warnings.filterwarnings("ignore")

def log(msg):
    print(f"LOG: {{msg}}", flush=True)

def progress(val):
    print(f"PROGRESS: {{val}}", flush=True)

def result(data):
    print(f"RESULT: {{json.dumps(data)}}", flush=True)

def setup_hf_auth():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            log("Authenticated with Hugging Face")
            return token
        except Exception as e:
            log(f"Auth warning: {{str(e)[:50]}}")
            return token
    return None

def get_device_info():
    import torch
    
    info = {{
        "backend": "cpu",
        "device": "cpu",
        "name": "CPU",
        "memory_total": 0,
    }}
    
    if torch.cuda.is_available():
        info["backend"] = "cuda"
        info["device"] = "cuda"
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_version"] = torch.version.cuda
        
        # Check multi-GPU
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            info["gpu_count"] = gpu_count
            total_mem = sum(
                torch.cuda.get_device_properties(i).total_memory 
                for i in range(gpu_count)
            ) / (1024**3)
            info["memory_total"] = total_mem
            log(f"Detected {{gpu_count}} GPUs, total {{total_mem:.1f}}GB VRAM")
        
        return info
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["backend"] = "mps"
        info["device"] = "mps"
        info["name"] = "Apple Silicon (MPS)"
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info["memory_total"] = int(result.stdout.strip()) / (1024**3)
        except:
            pass
        return info
    
    info["name"] = platform.processor() or "CPU"
    return info

def load_model(model_id, device_info, hf_token=None):
    """Load model with appropriate settings for large models"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    backend = device_info["backend"]
    device = device_info["device"]
    available_memory = device_info["memory_total"]
    
    log(f"Loading {{model_id}}...")
    log(f"Available memory: {{available_memory:.1f}}GB")
    
    auth_kwargs = {{"token": hf_token}} if hf_token else {{}}
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        **auth_kwargs
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine loading strategy based on available memory
    load_kwargs = {{
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        **auth_kwargs
    }}
    
    if backend == "cuda":
        # For 27B/32B models, we need careful memory management
        if available_memory >= 80:
            # Plenty of VRAM - use float16
            log("Using float16 (full precision)")
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        elif available_memory >= 48:
            # Use bfloat16 if available
            log("Using bfloat16")
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        elif available_memory >= 32:
            # Use 8-bit quantization
            log("Using 8-bit quantization (load_in_8bit)")
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            load_kwargs["load_in_8bit"] = True
        else:
            # Use 4-bit quantization
            log("Using 4-bit quantization (load_in_4bit)")
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    
    elif backend == "mps":
        # Apple Silicon
        if available_memory >= 64:
            load_kwargs["torch_dtype"] = torch.float16
        else:
            log("Warning: May need 64GB+ unified memory for these models")
            load_kwargs["torch_dtype"] = torch.float16
    
    else:
        load_kwargs["torch_dtype"] = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    
    # Move to device if needed (MPS)
    if backend == "mps" and not hasattr(model, "hf_device_map"):
        model = model.to("mps")
    
    model.eval()
    
    return model, tokenizer

def run_generation(model, tokenizer, prompt, device, backend, max_new_tokens=256):
    import torch
    
    # Format prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            messages = [{{"role": "user", "content": prompt}}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to device
    if hasattr(model, "hf_device_map"):
        # Model is already distributed
        inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
        device = model.device
    else:
        inputs = {{k: v.to(device) for k, v in inputs.items()}}
    
    prompt_tokens = inputs["input_ids"].shape[1]
    
    gen_kwargs = {{
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }}
    
    # Sync before timing
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Sync after
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    
    end_time = time.perf_counter()
    
    generated_ids = outputs[0][prompt_tokens:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return {{
        "response": response,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
        "total_tokens": len(outputs[0]),
        "time": end_time - start_time
    }}


# Main execution
try:
    log("Importing libraries...")
    progress(0.1)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    torch.set_grad_enabled(False)
    
    progress(0.12)
    hf_token = setup_hf_auth()
    
    progress(0.15)
    device_info = get_device_info()
    backend = device_info["backend"]
    device = device_info["device"]
    
    log(f"Backend: {{backend.upper()}}")
    log(f"Device: {{device_info['name']}}")
    if device_info.get("cuda_version"):
        log(f"CUDA: {{device_info['cuda_version']}}")
    log(f"Memory: {{device_info['memory_total']:.1f}} GB")
    
    progress(0.2)
    
    # Models to try (only these two)
    models = {models_str}
    requested = {repr(model_id) if model_id else 'None'}
    
    if requested:
        models = [requested]
    
    model = None
    tokenizer = None
    used_model_id = None
    
    for model_id in models:
        try:
            log(f"Trying: {{model_id}}")
            model, tokenizer = load_model(model_id, device_info, hf_token)
            used_model_id = model_id
            log(f"Loaded: {{model_id}}")
            break
        except Exception as e:
            error_str = str(e)
            if "gated" in error_str.lower() or "access" in error_str.lower():
                log(f"GATED: Accept license at huggingface.co/{{model_id}}")
            elif "memory" in error_str.lower() or "oom" in error_str.lower():
                log(f"OUT OF MEMORY: Need more VRAM for {{model_id}}")
            else:
                log(f"Failed: {{error_str[:100]}}")
            continue
    
    if model is None:
        raise RuntimeError(
            "Could not load Gemma 3 27B or Qwen 2.5 32B. "
            "Ensure you have enough VRAM (32GB+ recommended) and "
            "have accepted the license for Gemma at huggingface.co"
        )
    
    progress(0.6)
    
    # Clear memory
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps":
        torch.mps.empty_cache()
    
    prompt = """{self.BENCHMARK_PROMPT}"""
    
    progress(0.7)
    
    # Warm-up
    log("Warming up...")
    try:
        _ = run_generation(model, tokenizer, "Hi", device, backend, max_new_tokens=5)
    except Exception as e:
        log(f"Warmup note: {{str(e)[:50]}}")
    
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    progress(0.75)
    
    # Benchmark
    log("Running benchmark...")
    gen_result = run_generation(model, tokenizer, prompt, device, backend, max_new_tokens=256)
    
    progress(0.9)
    
    total_time = gen_result["time"]
    generated_tokens = gen_result["generated_tokens"]
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
    
    memory_used = 0
    if backend == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)
    elif backend == "mps":
        try:
            memory_used = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        except:
            pass
    
    log(f"Complete: {{generated_tokens}} tokens in {{total_time:.2f}}s")
    log(f"Speed: {{tokens_per_second:.2f}} tok/s")
    
    progress(1.0)
    
    result({{
        "success": True,
        "model_name": used_model_id,
        "backend": backend,
        "response": gen_result["response"][:1500],
        "total_tokens": gen_result["total_tokens"],
        "prompt_tokens": gen_result["prompt_tokens"],
        "generated_tokens": generated_tokens,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": 0,
        "prefill_time": 0,
        "generation_time": total_time,
        "gpu_info": device_info,
        "memory_used_gb": memory_used,
        "memory_total_gb": device_info["memory_total"],
    }})

except Exception as e:
    import traceback
    error_msg = str(e)
    log(f"Error: {{error_msg}}")
    result({{
        "success": False,
        "error": error_msg
    }})
'''


if __name__ == "__main__":
    from pathlib import Path
    venv_path = Path("benchmark")
    if venv_path.exists():
        benchmark = LLMBenchmark(venv_path)
        result = benchmark.run_benchmark()
        print(f"Result: {result}")
