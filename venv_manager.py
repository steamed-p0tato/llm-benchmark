#!/usr/bin/env python3
"""
Virtual Environment Manager Module
Handles creation and management of the benchmark virtual environment
"""

import os
import sys
import subprocess
import venv
import platform
import json
from pathlib import Path
from typing import Callable, Optional
import shutil


class VenvManager:
    """Manages the benchmark virtual environment"""
    
    VENV_NAME = "benchmark"
    
    # Required packages to verify
    REQUIRED_PACKAGES = ["torch", "transformers", "accelerate"]
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.venv_path = self.base_path / self.VENV_NAME
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[float], None]] = None
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = self.is_mac and platform.machine() == "arm64"
        
    def set_callbacks(
        self, 
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """Set callback functions for logging and progress updates"""
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def _log(self, message: str):
        """Log a message using callback or print"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
            
    def _update_progress(self, progress: float):
        """Update progress using callback"""
        if self.progress_callback:
            self.progress_callback(progress)
    
    def venv_exists(self) -> bool:
        """Check if the benchmark virtual environment exists"""
        if not self.venv_path.exists():
            return False
            
        # Check for key files that indicate a valid venv
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
            pip_path = self.venv_path / "bin" / "pip"
            
        return python_path.exists() and pip_path.exists()
    
    def get_python_path(self) -> Path:
        """Get the path to Python in the virtual environment"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"
    
    def get_pip_path(self) -> Path:
        """Get the path to pip in the virtual environment"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"
    
    def verify_installation(self) -> bool:
        """Verify that required packages are installed"""
        if not self.venv_exists():
            return False
        
        python_path = str(self.get_python_path())
        
        # Check if we can import required packages
        check_script = '''
import sys
try:
    import torch
    import transformers
    import accelerate
    print("OK")
except ImportError as e:
    print(f"MISSING: {e}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run(
                [python_path, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0 and "OK" in result.stdout
        except Exception:
            return False
    
    def create_venv(self) -> bool:
        """Create the virtual environment"""
        try:
            self._log(f"Creating virtual environment at: {self.venv_path}")
            self._update_progress(0.05)
            
            # Create the virtual environment
            builder = venv.EnvBuilder(
                system_site_packages=False,
                clear=True,
                with_pip=True,
                upgrade_deps=True
            )
            builder.create(self.venv_path)
            
            self._log("✓ Virtual environment created successfully")
            self._update_progress(0.1)
            return True
            
        except Exception as e:
            self._log(f"✗ Failed to create virtual environment: {e}")
            return False
    
    def install_packages(self) -> bool:
        """Install required packages in the virtual environment"""
        try:
            pip_path = str(self.get_pip_path())
            python_path = str(self.get_python_path())
            
            # Upgrade pip first
            self._log("Upgrading pip...")
            subprocess.run(
                [pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"],
                capture_output=True,
                text=True,
                check=True
            )
            self._update_progress(0.15)
            
            # Install PyTorch based on platform
            self._log("Installing PyTorch...")
            self._log("(This may take several minutes)")
            
            if self.is_apple_silicon:
                # Apple Silicon Mac - use MPS
                self._log("Detected Apple Silicon - installing PyTorch with MPS support")
                torch_cmd = [pip_path, "install", "torch", "torchvision", "torchaudio"]
            elif self.is_mac:
                # Intel Mac
                self._log("Detected Intel Mac - installing PyTorch CPU version")
                torch_cmd = [pip_path, "install", "torch", "torchvision", "torchaudio"]
            else:
                # Try CUDA version first (Linux/Windows)
                self._log("Attempting to install PyTorch with CUDA support...")
                torch_cmd = [
                    pip_path, "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ]
            
            torch_result = subprocess.run(torch_cmd, capture_output=True, text=True)
            
            # Fallback to CPU version if CUDA failed
            if torch_result.returncode != 0 and not self.is_mac:
                self._log("CUDA PyTorch not available, installing CPU version...")
                torch_result = subprocess.run(
                    [pip_path, "install", "torch", "torchvision", "torchaudio"],
                    capture_output=True,
                    text=True
                )
            
            if torch_result.returncode == 0:
                self._log("✓ PyTorch installed")
            else:
                self._log(f"⚠ PyTorch installation had issues")
                self._log(f"  {torch_result.stderr[:300] if torch_result.stderr else 'Unknown error'}")
                return False
            
            self._update_progress(0.4)
            
            # Verify PyTorch installed correctly
            verify_torch = subprocess.run(
                [python_path, "-c", "import torch; print(torch.__version__)"],
                capture_output=True,
                text=True
            )
            if verify_torch.returncode == 0:
                self._log(f"✓ PyTorch version: {verify_torch.stdout.strip()}")
            else:
                self._log(f"✗ PyTorch verification failed")
                return False
            
            self._update_progress(0.5)
            
            # Install transformers and accelerate
            self._log("Installing transformers and accelerate...")
            result = subprocess.run(
                [pip_path, "install", "transformers==4.40.0", "accelerate==0.29.0"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self._log("✓ Transformers installed")
            else:
                self._log(f"⚠ Transformers issue: {result.stderr[:200] if result.stderr else 'Unknown'}")
                # Try without version pinning
                self._log("Retrying with latest versions...")
                result = subprocess.run(
                    [pip_path, "install", "transformers", "accelerate"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    self._log(f"✗ Failed to install transformers")
                    return False
            
            self._update_progress(0.7)
            
            # Verify transformers installed
            verify_tf = subprocess.run(
                [python_path, "-c", "import transformers; print(transformers.__version__)"],
                capture_output=True,
                text=True
            )
            if verify_tf.returncode == 0:
                self._log(f"✓ Transformers version: {verify_tf.stdout.strip()}")
            else:
                self._log(f"✗ Transformers verification failed: {verify_tf.stderr}")
                return False
            
            # Install other packages
            other_packages = [
                "huggingface_hub>=0.20.0",
                "sentencepiece>=0.1.99",
                "protobuf>=4.25.0",
                "safetensors>=0.4.0",
                "tokenizers>=0.15.0",
                "numba>=0.59.0",
                "numpy>=1.24.0", 
                "psutil>=5.9.0",
            ]
            
            # Add bitsandbytes only for CUDA systems (not Mac)
            if not self.is_mac:
                other_packages.append("bitsandbytes>=0.42.0")
            
            total_packages = len(other_packages)
            for i, package in enumerate(other_packages):
                self._log(f"Installing {package}...")
                result = subprocess.run(
                    [pip_path, "install", package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self._log(f"✓ {package} installed")
                else:
                    self._log(f"⚠ Warning: {package} may have issues")
                
                progress = 0.7 + (0.25 * (i + 1) / total_packages)
                self._update_progress(progress)
            
            self._update_progress(1.0)
            self._log("✓ All packages installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log(f"✗ Package installation failed: {e}")
            self._log(f"  stderr: {e.stderr[:500] if e.stderr else 'No error output'}")
            return False
        except Exception as e:
            self._log(f"✗ Unexpected error during installation: {e}")
            import traceback
            self._log(traceback.format_exc())
            return False
    
    def setup_complete_environment(self) -> bool:
        """Create venv and install all packages"""
        if not self.create_venv():
            return False
        return self.install_packages()
    
    def get_installed_packages(self) -> list:
        """Get list of installed packages in the venv"""
        try:
            pip_path = str(self.get_pip_path())
            result = subprocess.run(
                [pip_path, "list", "--format=freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().split('\n')
        except Exception:
            return []
    
    def check_gpu_availability(self) -> dict:
        """Check GPU availability in the venv"""
        if not self.venv_exists():
            return {
                "cuda_available": False,
                "mps_available": False,
                "backend": "error",
                "error": "Virtual environment not found"
            }
        
        python_path = str(self.get_python_path())
        
        check_script = '''
import json
import platform

try:
    import torch
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": [],
        "backend": "cpu"
    }
    
    if info["cuda_available"]:
        info["backend"] = "cuda"
        info["gpu_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
            props = torch.cuda.get_device_properties(i)
            info["gpu_memory"].append(props.total_memory / (1024**3))
    elif info["mps_available"]:
        info["backend"] = "mps"
        info["gpu_count"] = 1
        info["gpu_names"].append("Apple Silicon (MPS)")
        # Try to get total memory
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                total_mem = int(result.stdout.strip()) / (1024**3)
                info["gpu_memory"].append(total_mem)
        except:
            info["gpu_memory"].append(0)
    
    print(json.dumps(info))
    
except ImportError as e:
    print(json.dumps({"error": f"Import error: {e}", "backend": "error"}))
except Exception as e:
    print(json.dumps({"error": str(e), "backend": "error"}))
'''
        
        try:
            result = subprocess.run(
                [python_path, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    return {
                        "cuda_available": False,
                        "mps_available": False,
                        "backend": "error",
                        "error": f"Invalid JSON: {result.stdout[:100]}"
                    }
            else:
                return {
                    "cuda_available": False,
                    "mps_available": False,
                    "backend": "error",
                    "error": result.stderr[:200] if result.stderr else "Unknown error"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "cuda_available": False,
                "mps_available": False,
                "backend": "error",
                "error": "Timeout checking GPU"
            }
        except Exception as e:
            return {
                "cuda_available": False,
                "mps_available": False,
                "backend": "error",
                "error": str(e)
            }
    
    def delete_venv(self) -> bool:
        """Delete the virtual environment"""
        try:
            if self.venv_path.exists():
                shutil.rmtree(self.venv_path)
                self._log("✓ Virtual environment deleted")
                return True
            return False
        except Exception as e:
            self._log(f"✗ Failed to delete venv: {e}")
            return False
