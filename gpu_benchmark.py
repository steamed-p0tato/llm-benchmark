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
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import tempfile
import os
import platform


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
    backend: str = "unknown"  # cuda, mps, or cpu


class LLMBenchmark:
    """GPU Benchmark using LLM inference"""
    
    # Models to try (in order of preference) - using well-tested models
    MODEL_OPTIONS = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - very compatible
        "microsoft/DialoGPT-medium",  # Smaller, very compatible
        "gpt2-medium",  # Fallback - always works
    ]
    
    LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # For systems with more VRAM
    
    BENCHMARK_PROMPT = (
        "Tell me if the Unix ideology of software and communism are "
        "related to each other. Provide a detailed analysis comparing "
        "the philosophical foundations of both."
    )
    
    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[float], None]] = None
        
    def set_callbacks(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """Set callback functions"""
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
        """Get the path to Python in the virtual environment"""
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")
    
    def run_benchmark(self, model_id: Optional[str] = None) -> GPUBenchmarkResult:
        """Run the GPU benchmark using LLM inference"""
        
        self._log("=" * 60)
        self._log("GPU BENCHMARK - LLM Inference")
        self._log("=" * 60)
        
        self._update_progress(0.1)
        
        # Create the benchmark script
        benchmark_script = self._create_benchmark_script(model_id)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(benchmark_script)
            script_path = f.name
        
        try:
            self._log("Detecting hardware acceleration...")
            self._log("Starting benchmark (this may take several minutes)...")
            self._log("Downloading model if not cached...")
            self._update_progress(0.2)
            
            # Run the benchmark script
            env = os.environ.copy()
            env['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')
            env['TOKENIZERS_PARALLELISM'] = 'false'
            env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
            
            process = subprocess.Popen(
                [self.get_python_path(), script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            # Collect output
            stdout_lines = []
            
            # Read output in real-time
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
                elif line.startswith("RESULT:"):
                    pass  # Will parse at the end
                elif not line.startswith(("PROGRESS:", "LOG:", "RESULT:")):
                    # Log other output for debugging
                    if "error" in line.lower() or "warning" in line.lower():
                        self._log(f"  {line[:100]}")
            
            process.wait()
            
            self._update_progress(0.95)
            
            # Parse results
            result_json = None
            for line in stdout_lines:
                if line.startswith("RESULT:"):
                    try:
                        result_json = json.loads(line[7:])
                        break
                    except json.JSONDecodeError as e:
                        self._log(f"Failed to parse result: {e}")
            
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
                error_msg = result_json.get("error", "Unknown error") if result_json else "Failed to parse results"
                
                # Show last few lines of output for debugging
                self._log(f"✗ Benchmark failed: {error_msg[:200]}")
                
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
            # Clean up temp file
            try:
                os.unlink(script_path)
            except:
                pass
    
    def _create_benchmark_script(self, model_id: Optional[str] = None) -> str:
        """Create the benchmark script to run in the venv"""
        
        model_options_str = json.dumps(self.MODEL_OPTIONS)
        llama_model = self.LLAMA_MODEL
        
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU Benchmark Script - Runs inside the benchmark venv"""

import json
import time
import sys
import gc
import platform
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def log(msg):
    print(f"LOG: {{msg}}", flush=True)

def progress(val):
    print(f"PROGRESS: {{val}}", flush=True)

def result(data):
    print(f"RESULT: {{json.dumps(data)}}", flush=True)

def get_device_info():
    """Detect available hardware acceleration"""
    import torch
    
    info = {{
        "backend": "cpu",
        "device": "cpu",
        "name": "CPU",
        "memory_total": 0,
    }}
    
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        info["backend"] = "cuda"
        info["device"] = "cuda"
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_version"] = torch.version.cuda
        return info
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["backend"] = "mps"
        info["device"] = "mps"
        info["name"] = "Apple Silicon (MPS)"
        # Get Mac memory info
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
    
    # CPU fallback
    info["name"] = platform.processor() or "CPU"
    return info

def try_load_model(model_id, device, backend):
    """Try to load a model with proper error handling"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    log(f"Attempting to load: {{model_id}}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype
    if backend == "mps":
        dtype = torch.float16
    elif backend == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load model configuration first to check compatibility
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model
    load_kwargs = {{
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }}
    
    # For CUDA, try device_map auto
    if backend == "cuda":
        load_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    
    # Move to device if needed (for MPS)
    if backend == "mps":
        model = model.to("mps")
    elif backend == "cpu":
        model = model.to("cpu")
    
    model.eval()
    
    return model, tokenizer

def run_generation(model, tokenizer, prompt, device, backend, max_new_tokens=256):
    """Run text generation with compatibility handling"""
    import torch
    
    # Format prompt for chat models if applicable
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            messages = [{{"role": "user", "content": prompt}}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    
    # Move to device
    inputs = {{k: v.to(device) for k, v in inputs.items()}}
    prompt_tokens = inputs["input_ids"].shape[1]
    
    # Generation config - use simple settings for compatibility
    gen_kwargs = {{
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # Enable KV cache
    }}
    
    # Synchronize before timing
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    
    # Generate
    start_time = time.perf_counter()
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        except Exception as e:
            # If cache fails, try without cache
            if "cache" in str(e).lower() or "seen_token" in str(e).lower():
                log("Cache issue detected, retrying without cache...")
                gen_kwargs["use_cache"] = False
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs
                )
            else:
                raise
    
    # Synchronize after generation
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    
    end_time = time.perf_counter()
    
    # Decode
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
    
    # Disable gradient computation globally
    torch.set_grad_enabled(False)
    
    progress(0.15)
    
    # Detect device
    device_info = get_device_info()
    backend = device_info["backend"]
    device = device_info["device"]
    
    log(f"Backend: {{backend.upper()}}")
    log(f"Device: {{device_info['name']}}")
    if device_info.get("cuda_version"):
        log(f"CUDA Version: {{device_info['cuda_version']}}")
    if device_info["memory_total"] > 0:
        log(f"Memory: {{device_info['memory_total']:.1f}} GB")
    
    progress(0.2)
    
    # Select model
    model_options = {model_options_str}
    requested_model = {repr(model_id) if model_id else 'None'}
    
    model = None
    tokenizer = None
    used_model_id = None
    
    # Try models in order
    models_to_try = []
    if requested_model:
        models_to_try.append(requested_model)
    
    # Add appropriate models based on backend and memory
    if backend == "cuda" and device_info["memory_total"] >= 16:
        models_to_try.append("{llama_model}")
    
    models_to_try.extend(model_options)
    
    for model_id in models_to_try:
        try:
            log(f"Trying model: {{model_id}}")
            model, tokenizer = try_load_model(model_id, device, backend)
            used_model_id = model_id
            log(f"Successfully loaded: {{model_id}}")
            break
        except Exception as e:
            log(f"Failed to load {{model_id}}: {{str(e)[:100]}}")
            continue
    
    if model is None:
        raise RuntimeError("Could not load any model")
    
    progress(0.6)
    log("Model loaded successfully")
    
    # Clear memory
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps":
        torch.mps.empty_cache()
    
    # Prepare prompt
    prompt = """{self.BENCHMARK_PROMPT}"""
    
    progress(0.7)
    
    # Warm-up run
    log("Warming up...")
    try:
        warmup_result = run_generation(model, tokenizer, "Hello", device, backend, max_new_tokens=10)
    except Exception as e:
        log(f"Warmup issue (continuing): {{str(e)[:50]}}")
    
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps":
        torch.mps.empty_cache()
    
    progress(0.75)
    
    # Actual benchmark
    log("Running benchmark generation...")
    gen_result = run_generation(model, tokenizer, prompt, device, backend, max_new_tokens=256)
    
    progress(0.9)
    
    # Calculate metrics
    total_time = gen_result["time"]
    generated_tokens = gen_result["generated_tokens"]
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
    
    # Get memory stats
    memory_used = 0
    memory_total = device_info["memory_total"]
    
    if backend == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)
    elif backend == "mps":
        try:
            memory_used = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        except:
            pass
    
    log(f"Generation complete: {{generated_tokens}} tokens in {{total_time:.2f}}s")
    log(f"Tokens/second: {{tokens_per_second:.2f}}")
    
    progress(1.0)
    
    result({{
        "success": True,
        "model_name": used_model_id,
        "backend": backend,
        "response": gen_result["response"][:1000],
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
        "memory_total_gb": memory_total,
    }})

except Exception as e:
    import traceback
    error_msg = str(e)
    tb = traceback.format_exc()
    result({{
        "success": False,
        "error": f"{{error_msg}}"
    }})
'''


if __name__ == "__main__":
    # Test run
    from pathlib import Path
    venv_path = Path("benchmark")
    if venv_path.exists():
        benchmark = LLMBenchmark(venv_path)
        result = benchmark.run_benchmark()
        print(f"Result: {result}")