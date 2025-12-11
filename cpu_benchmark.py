#!/usr/bin/env python3
"""
CPU Benchmark Module
Implements Mandelbrot set calculation for CPU performance testing
Optimized for high core count systems (up to 64+ cores)
"""

import time
import multiprocessing
import concurrent.futures
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
import platform

try:
    import psutil
except ImportError:
    psutil = None

# Try to import numba for JIT compilation (massive speedup)
try:
    from numba import jit, prange, set_num_threads
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    try:
        import numpy as np
        NUMPY_AVAILABLE = True
    except ImportError:
        NUMPY_AVAILABLE = False


@dataclass
class CPUBenchmarkResult:
    """Results from CPU benchmark"""
    total_time: float
    iterations: int
    width: int
    height: int
    max_iterations: int
    pixels_per_second: float
    score: float
    cpu_info: dict
    single_core_time: float
    multi_core_time: float
    core_count: int
    scaling_efficiency: float
    backend: str = "python"


# =============================================================================
# Pure Python Implementation (fallback)
# =============================================================================

def mandelbrot_escape_python(c_real: float, c_imag: float, max_iter: int) -> int:
    """Calculate escape iteration for a single point - Pure Python"""
    z_real, z_imag = 0.0, 0.0
    
    for i in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        
        if z_real_sq + z_imag_sq > 4.0:
            return i
            
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
        
    return max_iter


def calculate_chunk_python(args: Tuple) -> int:
    """Calculate a chunk of the Mandelbrot set - Pure Python
    Returns sum of iterations (checksum) to minimize data transfer.
    """
    start_y, end_y, width, max_iter, x_min, x_max, y_min, y_max, height = args
    
    checksum = 0
    x_scale = (x_max - x_min) / (width - 1)
    y_scale = (y_max - y_min) / (height - 1)
    
    for y in range(start_y, end_y):
        c_imag = y_min + y * y_scale
        for x in range(width):
            c_real = x_min + x * x_scale
            z_real, z_imag = 0.0, 0.0
            
            for i in range(max_iter):
                z_real_sq = z_real * z_real
                z_imag_sq = z_imag * z_imag
                
                if z_real_sq + z_imag_sq > 4.0:
                    checksum += i
                    break
                    
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = z_real_sq - z_imag_sq + c_real
            else:
                checksum += max_iter
    
    return checksum


# =============================================================================
# Numba JIT Implementation (10-100x faster)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True, fastmath=True)
    def mandelbrot_single_numba(
        width: int, height: int, max_iter: int,
        x_min: float, x_max: float,
        y_min: float, y_max: float
    ) -> int:
        """Calculate Mandelbrot set single-threaded - returns checksum"""
        checksum = 0
        x_scale = (x_max - x_min) / (width - 1)
        y_scale = (y_max - y_min) / (height - 1)
        
        for y in range(height):
            c_imag = y_min + y * y_scale
            for x in range(width):
                c_real = x_min + x * x_scale
                z_real, z_imag = 0.0, 0.0
                
                for i in range(max_iter):
                    z_real_sq = z_real * z_real
                    z_imag_sq = z_imag * z_imag
                    
                    if z_real_sq + z_imag_sq > 4.0:
                        checksum += i
                        break
                    
                    z_imag = 2.0 * z_real * z_imag + c_imag
                    z_real = z_real_sq - z_imag_sq + c_real
                else:
                    checksum += max_iter
        
        return checksum

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def mandelbrot_parallel_numba(
        width: int, height: int, max_iter: int,
        x_min: float, x_max: float,
        y_min: float, y_max: float
    ) -> int:
        """Calculate Mandelbrot set parallel - returns checksum"""
        # Use a results array for parallel reduction
        row_sums = np.zeros(height, dtype=np.int64)
        
        x_scale = (x_max - x_min) / (width - 1)
        y_scale = (y_max - y_min) / (height - 1)
        
        for y in prange(height):
            row_sum = 0
            c_imag = y_min + y * y_scale
            
            for x in range(width):
                c_real = x_min + x * x_scale
                z_real, z_imag = 0.0, 0.0
                
                for i in range(max_iter):
                    z_real_sq = z_real * z_real
                    z_imag_sq = z_imag * z_imag
                    
                    if z_real_sq + z_imag_sq > 4.0:
                        row_sum += i
                        break
                    
                    z_imag = 2.0 * z_real * z_imag + c_imag
                    z_real = z_real_sq - z_imag_sq + c_real
                else:
                    row_sum += max_iter
            
            row_sums[y] = row_sum
        
        return np.sum(row_sums)


# =============================================================================
# NumPy Vectorized Implementation
# =============================================================================

if NUMBA_AVAILABLE or NUMPY_AVAILABLE:
    def mandelbrot_numpy(
        width: int, height: int, max_iter: int,
        x_min: float, x_max: float,
        y_min: float, y_max: float
    ) -> int:
        """Calculate Mandelbrot using NumPy vectorization"""
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=np.int32)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            M[mask] = i + 1
        
        M[np.abs(Z) <= 2] = max_iter
        
        return int(np.sum(M))


# =============================================================================
# Main Benchmark Class
# =============================================================================

class MandelbrotBenchmark:
    """CPU Benchmark using Mandelbrot set calculations
    
    Optimized for high core count systems with multiple backends:
    - numba: JIT compiled, best performance (10-100x faster)
    - numpy: Vectorized, good performance (5-20x faster)
    - python: Pure Python, baseline performance
    """
    
    # Benchmark parameters scaled for different core counts
    PRESETS = {
        "quick": {"width": 1920, "height": 1080, "max_iter": 500},
        "standard": {"width": 3840, "height": 2160, "max_iter": 1000},
        "extended": {"width": 7680, "height": 4320, "max_iter": 2000},
        "extreme": {"width": 15360, "height": 8640, "max_iter": 4000},
    }
    
    # Scoring baseline
    BASELINE_SCORE = 1000.0
    BASELINE_PIXELS = 3840 * 2160
    BASELINE_TIME = 10.0
    
    def __init__(self, preset: str = "auto"):
        self.preset = preset
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[float], None]] = None
        self.backend = self._detect_best_backend()
        
    def _detect_best_backend(self) -> str:
        if NUMBA_AVAILABLE:
            return "numba"
        elif NUMPY_AVAILABLE:
            return "numpy"
        return "python"
    
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
    
    def _select_preset(self, core_count: int) -> dict:
        if self.preset != "auto":
            return self.PRESETS.get(self.preset, self.PRESETS["standard"])
        
        if core_count >= 32:
            return self.PRESETS["extreme"]
        elif core_count >= 16:
            return self.PRESETS["extended"]
        elif core_count >= 8:
            return self.PRESETS["standard"]
        else:
            return self.PRESETS["quick"]
    
    def run_single_core(
        self,
        width: int,
        height: int,
        max_iter: int
    ) -> Tuple[float, int]:
        """Run Mandelbrot calculation on a single core"""
        
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        
        start_time = time.perf_counter()
        
        if self.backend == "numba" and NUMBA_AVAILABLE:
            # Force single thread
            set_num_threads(1)
            checksum = mandelbrot_single_numba(
                width, height, max_iter, x_min, x_max, y_min, y_max
            )
            
        elif self.backend == "numpy" and (NUMBA_AVAILABLE or NUMPY_AVAILABLE):
            checksum = mandelbrot_numpy(
                width, height, max_iter, x_min, x_max, y_min, y_max
            )
            
        else:
            # Pure Python
            checksum = 0
            for y in range(height):
                c_imag = y_min + (y / (height - 1)) * (y_max - y_min)
                for x in range(width):
                    c_real = x_min + (x / (width - 1)) * (x_max - x_min)
                    checksum += mandelbrot_escape_python(c_real, c_imag, max_iter)
                
                if y % (height // 10) == 0:
                    self._update_progress((y / height) * 0.4)
        
        end_time = time.perf_counter()
        return end_time - start_time, checksum
    
    def run_multi_core(
        self,
        width: int,
        height: int,
        max_iter: int,
        num_cores: int
    ) -> Tuple[float, int]:
        """Run multi-core benchmark using best available method"""
        
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        
        if self.backend == "numba" and NUMBA_AVAILABLE:
            # Use numba's built-in parallelization - no subprocess issues
            set_num_threads(num_cores)
            
            start_time = time.perf_counter()
            checksum = mandelbrot_parallel_numba(
                width, height, max_iter, x_min, x_max, y_min, y_max
            )
            end_time = time.perf_counter()
            
            return end_time - start_time, checksum
        
        elif self.backend == "numpy" and (NUMBA_AVAILABLE or NUMPY_AVAILABLE):
            # NumPy is already somewhat parallelized via BLAS
            start_time = time.perf_counter()
            checksum = mandelbrot_numpy(
                width, height, max_iter, x_min, x_max, y_min, y_max
            )
            end_time = time.perf_counter()
            
            return end_time - start_time, checksum
        
        else:
            # Pure Python with ThreadPoolExecutor (safe for TUI)
            return self._run_multi_core_threads(
                width, height, max_iter, num_cores,
                x_min, x_max, y_min, y_max
            )
    
    def _run_multi_core_threads(
        self,
        width: int, height: int, max_iter: int, num_cores: int,
        x_min: float, x_max: float, y_min: float, y_max: float
    ) -> Tuple[float, int]:
        """Run using ThreadPoolExecutor - safe for TUI apps"""
        
        # Calculate chunks
        chunks_per_core = 4
        num_chunks = min(num_cores * chunks_per_core, height)
        rows_per_chunk = max(1, height // num_chunks)
        
        chunks = []
        for i in range(num_chunks):
            start_y = i * rows_per_chunk
            end_y = start_y + rows_per_chunk if i < num_chunks - 1 else height
            chunks.append((start_y, end_y, width, max_iter, x_min, x_max, y_min, y_max, height))
        
        start_time = time.perf_counter()
        
        # Use threads - works in TUI context
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(calculate_chunk_python, chunks))
        
        end_time = time.perf_counter()
        
        return end_time - start_time, sum(results)
    
    def get_cpu_info(self) -> dict:
        """Get CPU information"""
        info = {
            "processor": platform.processor() or "Unknown",
            "architecture": platform.machine(),
            "cores_physical": multiprocessing.cpu_count(),
            "cores_logical": multiprocessing.cpu_count(),
            "python_version": platform.python_version(),
        }
        
        if psutil:
            try:
                info["cores_physical"] = psutil.cpu_count(logical=False) or info["cores_physical"]
                info["cores_logical"] = psutil.cpu_count(logical=True) or info["cores_logical"]
                freq = psutil.cpu_freq()
                if freq:
                    info["frequency_mhz"] = freq.current
                    info["frequency_max_mhz"] = freq.max
            except Exception:
                pass
        
        # Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line.lower():
                            info["model_name"] = line.split(":")[1].strip()
                            break
            except Exception:
                pass
        
        # macOS
        elif platform.system() == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    info["model_name"] = result.stdout.strip()
            except Exception:
                pass
        
        # Windows
        elif platform.system() == "Windows":
            try:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, shell=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        info["model_name"] = lines[1].strip()
            except Exception:
                pass
                
        return info
    
    def run_benchmark(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_iter: Optional[int] = None,
    ) -> CPUBenchmarkResult:
        """Run the complete CPU benchmark"""
        
        cpu_info = self.get_cpu_info()
        core_count = cpu_info.get("cores_logical", multiprocessing.cpu_count())
        
        # Select preset if parameters not specified
        preset = self._select_preset(core_count)
        if width is None:
            width = preset["width"]
        if height is None:
            height = preset["height"]
        if max_iter is None:
            max_iter = preset["max_iter"]
        
        # Warm up numba JIT
        if self.backend == "numba" and NUMBA_AVAILABLE:
            self._log("Warming up Numba JIT compiler...")
            _ = mandelbrot_single_numba(10, 10, 10, -2.0, 1.0, -1.0, 1.0)
            _ = mandelbrot_parallel_numba(10, 10, 10, -2.0, 1.0, -1.0, 1.0)
        
        self._log("=" * 60)
        self._log("CPU BENCHMARK - Mandelbrot Set Calculation")
        self._log("=" * 60)
        self._log(f"Backend: {self.backend.upper()}")
        self._log(f"Resolution: {width}x{height} ({width*height:,} pixels)")
        self._log(f"Max Iterations: {max_iter}")
        self._log(f"CPU: {cpu_info.get('model_name', cpu_info.get('processor', 'Unknown'))}")
        self._log(f"Cores: {core_count} (physical: {cpu_info.get('cores_physical', 'N/A')})")
        self._log("")
        
        # Single-core benchmark (use smaller size to save time)
        self._log("Running single-core benchmark...")
        self._update_progress(0.0)
        
        single_width = min(width, 1920)
        single_height = min(height, 1080)
        single_max_iter = min(max_iter, 1000)
        single_time, _ = self.run_single_core(single_width, single_height, single_max_iter)
        
        # Scale single-core time to full resolution
        scale_factor = (width * height * max_iter) / (single_width * single_height * single_max_iter)
        estimated_single_time = single_time * scale_factor
        
        self._log(f"✓ Single-core time: {single_time:.2f}s (estimated full: {estimated_single_time:.2f}s)")
        self._update_progress(0.4)
        
        # Multi-core benchmark
        self._log(f"Running multi-core benchmark ({core_count} cores)...")
        
        multi_time, _ = self.run_multi_core(width, height, max_iter, core_count)
        
        self._log(f"✓ Multi-core time: {multi_time:.2f}s")
        self._update_progress(0.9)
        
        # Calculate metrics
        total_pixels = width * height
        pixels_per_second_multi = total_pixels / multi_time
        
        # Scaling efficiency
        theoretical_speedup = core_count
        actual_speedup = estimated_single_time / multi_time
        scaling_efficiency = (actual_speedup / theoretical_speedup) * 100
        
        # Normalized score
        work_factor = (total_pixels * max_iter) / (self.BASELINE_PIXELS * 1000)
        normalized_time = multi_time / work_factor
        score = (self.BASELINE_TIME / normalized_time) * self.BASELINE_SCORE
        
        self._update_progress(1.0)
        
        self._log("")
        self._log("=" * 60)
        self._log("RESULTS")
        self._log("=" * 60)
        self._log(f"Backend: {self.backend.upper()}")
        self._log(f"Multi-core: {pixels_per_second_multi:,.0f} pixels/sec")
        self._log(f"Speedup: {actual_speedup:.2f}x (theoretical: {theoretical_speedup}x)")
        self._log(f"Scaling Efficiency: {scaling_efficiency:.1f}%")
        self._log("")
        self._log(f"★ BENCHMARK SCORE: {score:,.0f}")
        self._log("=" * 60)
        
        return CPUBenchmarkResult(
            total_time=single_time + multi_time,
            iterations=max_iter,
            width=width,
            height=height,
            max_iterations=max_iter,
            pixels_per_second=pixels_per_second_multi,
            score=score,
            cpu_info=cpu_info,
            single_core_time=estimated_single_time,
            multi_core_time=multi_time,
            core_count=core_count,
            scaling_efficiency=scaling_efficiency,
            backend=self.backend
        )


def run_benchmark_standalone():
    """Run benchmark as standalone script"""
    print("Detecting best backend...")
    
    benchmark = MandelbrotBenchmark(preset="auto")
    print(f"Using backend: {benchmark.backend}")
    
    result = benchmark.run_benchmark()
    
    print(f"\nFinal Score: {result.score:,.0f}")
    print(f"Scaling Efficiency: {result.scaling_efficiency:.1f}%")
    
    return result


if __name__ == "__main__":
    run_benchmark_standalone()
