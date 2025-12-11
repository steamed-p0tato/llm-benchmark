#!/usr/bin/env python3
"""
GPU Benchmark Module
Runs LLM inference using Llama 3.1 8B for GPU performance testing
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


class LLMBenchmark:
    """GPU Benchmark using LLM inference"""
    
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    # Alternative: Use a smaller model for testing
    FALLBACK_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    
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
    
    def run_benchmark(self, use_fallback: bool = False) -> GPUBenchmarkResult:
        """Run the GPU benchmark using LLM inference"""
        
        model_id = self.FALLBACK_MODEL_ID if use_fallback else self.MODEL_ID
        
        self._log("=" * 60)
        self._log("GPU BENCHMARK - LLM Inference")
        self._log("=" * 60)
        self._log(f"Model: {model_id}")
        self._log(f"Prompt: {self.BENCHMARK_PROMPT[:50]}...")
        self._log("")
        
        self._update_progress(0.1)
        
        # Create the benchmark script
        benchmark_script = self._create_benchmark_script(model_id)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(benchmark_script)
            script_path = f.name
        
        try:
            self._log("Starting benchmark (this may take several minutes)...")
            self._log("Downloading model if not cached...")
            self._update_progress(0.2)
            
            # Run the benchmark script
            env = os.environ.copy()
            env['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')
            
            process = subprocess.Popen(
                [self.get_python_path(), script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Collect output
            stdout_lines = []
            stderr_lines = []
            
            # Read output in real-time
            while True:
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line.strip())
                    if line.startswith("PROGRESS:"):
                        try:
                            progress = float(line.split(":")[1])
                            self._update_progress(0.2 + progress * 0.7)
                        except:
                            pass
                    elif line.startswith("LOG:"):
                        self._log(line[4:].strip())
                    elif line.startswith("RESULT:"):
                        # Parse final result
                        pass
                        
                if process.poll() is not None:
                    break
            
            # Get remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            stdout_lines.extend(remaining_stdout.strip().split('\n'))
            stderr_lines.append(remaining_stderr)
            
            self._update_progress(0.95)
            
            # Parse results
            result_json = None
            for line in stdout_lines:
                if line.startswith("RESULT:"):
                    try:
                        result_json = json.loads(line[7:])
                    except:
                        pass
            
            if result_json and result_json.get("success"):
                result = GPUBenchmarkResult(
                    model_name=model_id,
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
                    success=True
                )
                
                self._log("")
                self._log("=" * 60)
                self._log("RESULTS")
                self._log("=" * 60)
                self._log(f"Generated Tokens: {result.generated_tokens}")
                self._log(f"Total Time: {result.total_time:.2f}s")
                self._log(f"Tokens/Second: {result.tokens_per_second:.2f}")
                self._log(f"Time to First Token: {result.time_to_first_token:.2f}s")
                self._log(f"GPU Memory Used: {result.memory_used_gb:.2f} GB")
                self._log("=" * 60)
                
            else:
                error_msg = result_json.get("error", "Unknown error") if result_json else "Failed to parse results"
                self._log(f"✗ Benchmark failed: {error_msg}")
                
                # Check if we should try fallback model
                if not use_fallback and "CUDA" not in error_msg:
                    self._log("Trying fallback model...")
                    return self.run_benchmark(use_fallback=True)
                
                result = GPUBenchmarkResult(
                    model_name=model_id,
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
    
    def _create_benchmark_script(self, model_id: str) -> str:
        """Create the benchmark script to run in the venv"""
        
        return f'''#!/usr/bin/env python3
"""GPU Benchmark Script - Runs inside the benchmark venv"""

import json
import time
import sys
import gc

def log(msg):
    print(f"LOG: {{msg}}", flush=True)

def progress(val):
    print(f"PROGRESS: {{val}}", flush=True)

def result(data):
    print(f"RESULT: {{json.dumps(data)}}", flush=True)

try:
    log("Importing libraries...")
    progress(0.1)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    
    progress(0.2)
    
    # Check GPU
    if not torch.cuda.is_available():
        result({{"success": False, "error": "CUDA not available"}})
        sys.exit(1)
    
    gpu_info = {{
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{{torch.cuda.get_device_capability(0)[0]}}.{{torch.cuda.get_device_capability(0)[1]}}",
        "cuda_version": torch.version.cuda,
    }}
    
    log(f"GPU: {{gpu_info['name']}}")
    log(f"CUDA Version: {{gpu_info['cuda_version']}}")
    
    progress(0.3)
    
    # Model configuration for memory efficiency
    model_id = "{model_id}"
    
    log(f"Loading model: {{model_id}}")
    
    # Try to load with 4-bit quantization for memory efficiency
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as e:
        log(f"4-bit loading failed, trying without quantization: {{e}}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    progress(0.6)
    log("Model loaded successfully")
    
    # Prepare prompt
    prompt = """{self.BENCHMARK_PROMPT}"""
    
    # Format for chat models
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{{"role": "user", "content": prompt}}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    prompt_tokens = inputs.input_ids.shape[1]
    
    log(f"Prompt tokens: {{prompt_tokens}}")
    
    # Clear cache before benchmark
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    progress(0.7)
    
    # Benchmark generation
    log("Starting generation...")
    
    start_time = time.perf_counter()
    first_token_time = None
    generated_tokens = 0
    
    # Use streaming to measure time to first token
    with torch.no_grad():
        # Generate with timing
        generation_start = time.perf_counter()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    end_time = time.perf_counter()
    
    progress(0.9)
    
    # Calculate metrics
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - prompt_tokens
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
    
    # Decode response
    response = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
    
    # Get memory stats
    memory_used = torch.cuda.max_memory_allocated() / (1024**3)
    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    log(f"Generation complete: {{generated_tokens}} tokens in {{total_time:.2f}}s")
    log(f"Tokens/second: {{tokens_per_second:.2f}}")
    
    progress(1.0)
    
    result({{
        "success": True,
        "response": response[:1000],
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": 0,  # Would need streaming to measure accurately
        "prefill_time": 0,
        "generation_time": total_time,
        "gpu_info": gpu_info,
        "memory_used_gb": memory_used,
        "memory_total_gb": memory_total,
    }})

except Exception as e:
    import traceback
    result({{
        "success": False,
        "error": f"{{str(e)}}\\n{{traceback.format_exc()}}"
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
