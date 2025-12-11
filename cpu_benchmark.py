#!/usr/bin/env python3
"""
CPU Benchmark Module
Implements Mandelbrot set calculation for CPU performance testing
"""

import time
import multiprocessing
import concurrent.futures
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import platform

try:
    import psutil
except ImportError:
    psutil = None


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


def mandelbrot_escape(c_real: float, c_imag: float, max_iter: int) -> int:
    """Calculate escape iteration for a single point"""
    z_real, z_imag = 0.0, 0.0
    
    for i in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        
        if z_real_sq + z_imag_sq > 4.0:
            return i
            
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
        
    return max_iter


def calculate_row(args: Tuple[int, int, int, float, float, float, float]) -> list:
    """Calculate a single row of the Mandelbrot set"""
    y, width, max_iter, x_min, x_max, y_min, y_max, height = args
    
    row_results = []
    c_imag = y_min + (y / (height - 1)) * (y_max - y_min)
    
    for x in range(width):
        c_real = x_min + (x / (width - 1)) * (x_max - x_min)
        iterations = mandelbrot_escape(c_real, c_imag, max_iter)
        row_results.append(iterations)
        
    return row_results


def calculate_chunk(args: Tuple[int, int, int, int, float, float, float, float, int]) -> list:
    """Calculate a chunk of rows of the Mandelbrot set"""
    start_y, end_y, width, max_iter, x_min, x_max, y_min, y_max, height = args
    
    chunk_results = []
    for y in range(start_y, end_y):
        row_results = []
        c_imag = y_min + (y / (height - 1)) * (y_max - y_min)
        
        for x in range(width):
            c_real = x_min + (x / (width - 1)) * (x_max - x_min)
            iterations = mandelbrot_escape(c_real, c_imag, max_iter)
            row_results.append(iterations)
        
        chunk_results.append(row_results)
        
    return chunk_results


class MandelbrotBenchmark:
    """CPU Benchmark using Mandelbrot set calculations"""
    
    # Benchmark parameters
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080
    DEFAULT_MAX_ITER = 1000
    
    # Scoring baseline (reference machine performance)
    BASELINE_SCORE = 1000.0  # Reference score for baseline machine
    BASELINE_TIME = 10.0     # Time in seconds for baseline machine
    
    def __init__(self):
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
    
    def run_single_core(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_iter: int = DEFAULT_MAX_ITER
    ) -> Tuple[float, list]:
        """Run Mandelbrot calculation on a single core"""
        
        # Mandelbrot set bounds (interesting region)
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        
        results = []
        start_time = time.perf_counter()
        
        for y in range(height):
            c_imag = y_min + (y / (height - 1)) * (y_max - y_min)
            row = []
            
            for x in range(width):
                c_real = x_min + (x / (width - 1)) * (x_max - x_min)
                iterations = mandelbrot_escape(c_real, c_imag, max_iter)
                row.append(iterations)
                
            results.append(row)
            
            # Update progress every 10%
            if y % (height // 10) == 0:
                progress = y / height
                self._update_progress(progress * 0.4)  # First 40% of total progress
                
        end_time = time.perf_counter()
        return end_time - start_time, results
    
    def run_multi_core_threaded(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_iter: int = DEFAULT_MAX_ITER,
        num_cores: Optional[int] = None
    ) -> Tuple[float, list]:
        """Run Mandelbrot calculation using multiple threads (for use in TUI)"""
        
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()
            
        # Mandelbrot set bounds
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        
        # Divide work into chunks
        chunk_size = max(1, height // num_cores)
        chunks = []
        for i in range(0, height, chunk_size):
            end_y = min(i + chunk_size, height)
            chunks.append((i, end_y, width, max_iter, x_min, x_max, y_min, y_max, height))
        
        start_time = time.perf_counter()
        
        # Use ThreadPoolExecutor for compatibility with Textual
        results = [None] * len(chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            future_to_idx = {executor.submit(calculate_chunk, chunk): idx 
                           for idx, chunk in enumerate(chunks)}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        
        # Flatten results
        flat_results = []
        for chunk_result in results:
            flat_results.extend(chunk_result)
            
        end_time = time.perf_counter()
        return end_time - start_time, flat_results
    
    def run_multi_core_process(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_iter: int = DEFAULT_MAX_ITER,
        num_cores: Optional[int] = None
    ) -> Tuple[float, list]:
        """Run Mandelbrot calculation using multiple processes (standalone mode)"""
        
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()
            
        # Mandelbrot set bounds
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.25, 1.25
        
        # Divide work into chunks
        chunk_size = max(1, height // num_cores)
        chunks = []
        for i in range(0, height, chunk_size):
            end_y = min(i + chunk_size, height)
            chunks.append((i, end_y, width, max_iter, x_min, x_max, y_min, y_max, height))
        
        start_time = time.perf_counter()
        
        # Use spawn context to avoid fork issues
        try:
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_cores,
                mp_context=ctx
            ) as executor:
                chunk_results = list(executor.map(calculate_chunk, chunks))
        except Exception:
            # Fallback to threading if multiprocessing fails
            chunk_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
                chunk_results = list(executor.map(calculate_chunk, chunks))
        
        # Flatten results
        flat_results = []
        for chunk_result in chunk_results:
            flat_results.extend(chunk_result)
            
        end_time = time.perf_counter()
        return end_time - start_time, flat_results
    
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
        
        # Try to get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line.lower():
                            info["model_name"] = line.split(":")[1].strip()
                            break
            except Exception:
                pass
        
        # Try to get CPU info on macOS
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
        
        # Try to get CPU info on Windows
        elif platform.system() == "Windows":
            try:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True
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
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_iter: int = DEFAULT_MAX_ITER,
        use_multiprocessing: bool = False
    ) -> CPUBenchmarkResult:
        """Run the complete CPU benchmark"""
        
        cpu_info = self.get_cpu_info()
        core_count = cpu_info.get("cores_logical", multiprocessing.cpu_count())
        
        self._log("=" * 60)
        self._log("CPU BENCHMARK - Mandelbrot Set Calculation")
        self._log("=" * 60)
        self._log(f"Resolution: {width}x{height}")
        self._log(f"Max Iterations: {max_iter}")
        self._log(f"CPU: {cpu_info.get('model_name', cpu_info.get('processor', 'Unknown'))}")
        self._log(f"Cores: {core_count}")
        self._log("")
        
        # Single-core benchmark
        self._log("Running single-core benchmark...")
        self._update_progress(0.0)
        single_time, _ = self.run_single_core(width, height, max_iter)
        self._log(f"✓ Single-core time: {single_time:.2f}s")
        self._update_progress(0.5)
        
        # Multi-core benchmark (use threading for TUI compatibility)
        self._log(f"Running multi-core benchmark ({core_count} threads)...")
        
        if use_multiprocessing:
            multi_time, _ = self.run_multi_core_process(width, height, max_iter, core_count)
        else:
            multi_time, _ = self.run_multi_core_threaded(width, height, max_iter, core_count)
        
        self._log(f"✓ Multi-core time: {multi_time:.2f}s")
        self._update_progress(0.9)
        
        # Calculate metrics
        total_pixels = width * height
        
        pixels_per_second_single = total_pixels / single_time
        pixels_per_second_multi = total_pixels / multi_time
        
        # Calculate scaling efficiency
        theoretical_speedup = core_count
        actual_speedup = single_time / multi_time
        scaling_efficiency = (actual_speedup / theoretical_speedup) * 100
        
        # Calculate score (higher is better)
        # Based on multi-core performance
        score = (self.BASELINE_TIME / multi_time) * self.BASELINE_SCORE
        
        self._update_progress(1.0)
        
        self._log("")
        self._log("=" * 60)
        self._log("RESULTS")
        self._log("=" * 60)
        self._log(f"Single-core: {pixels_per_second_single:,.0f} pixels/sec")
        self._log(f"Multi-core:  {pixels_per_second_multi:,.0f} pixels/sec")
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
            single_core_time=single_time,
            multi_core_time=multi_time,
            core_count=core_count,
            scaling_efficiency=scaling_efficiency
        )


def run_benchmark_standalone():
    """Run benchmark as standalone script"""
    benchmark = MandelbrotBenchmark()
    # Use multiprocessing when running standalone
    result = benchmark.run_benchmark(use_multiprocessing=True)
    return result


if __name__ == "__main__":
    # Set multiprocessing start method for standalone execution
    if platform.system() == "Darwin":  # macOS
        multiprocessing.set_start_method('spawn', force=True)
    
    run_benchmark_standalone()