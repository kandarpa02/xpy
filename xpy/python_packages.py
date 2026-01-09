import importlib
import sys
import warnings
import subprocess
import platform
from typing import Optional, Tuple, Union, List

def import_numpy(numpy_version: Optional[str] = None) -> Optional[object]:
    """Import or install NumPy with optional version specification"""
    package_name = "numpy" if numpy_version is None else f"numpy=={numpy_version}"
    
    try:
        # Try to import without installing first
        numpy = importlib.import_module("numpy")
        current_version = numpy.__version__
        print(f"NumPy {current_version} already imported")
        
        # Check if specific version was requested
        if numpy_version and current_version != numpy_version:
            print(f"Warning: Requested NumPy {numpy_version}, but found {current_version}")
            print(f"Consider updating with: pip install numpy=={numpy_version}")
            
        return numpy
    except ImportError:
        print(f"NumPy not found. Installing {package_name}...")
        
        if install_package(package_name):
            try:
                numpy = importlib.import_module("numpy")
                print(f"Successfully installed and imported NumPy {numpy.__version__}")
                return numpy
            except ImportError:
                print(f"Failed to import NumPy after installation")
                return None
        else:
            print("Failed to install NumPy")
            return None

def import_cupy(
    cuda_version: Optional[str] = None,
    cupy_version: Optional[str] = None
) -> Optional[object]:
    """
    Import or install CuPy with version options
    
    Args:
        cuda_version: Specific CUDA version ("12", "11", "10.2", etc.)
                      or None for auto-detection
        cupy_version: Specific CuPy version string (e.g., "12.0.0")
                     or None for latest
    """
    
    # First check if CuPy is already available
    try:
        cupy = importlib.import_module("cupy")
        current_version = getattr(cupy, "__version__", "unknown")
        print(f"CuPy {current_version} already imported")
        return cupy
    except ImportError:
        pass
    
    # Determine which CuPy package to install
    packages_to_try = []
    
    # If specific CUDA version is requested
    if cuda_version:
        if cuda_version.startswith('12'):
            pkg = "cupy-cuda12x"
        elif cuda_version.startswith('11'):
            pkg = "cupy-cuda11x"
        elif cuda_version.startswith('10'):
            pkg = "cupy-cuda102"
        else:
            pkg = "cupy"  # Fallback to main package
        
        # Add version specifier if provided
        if cupy_version:
            pkg = f"{pkg}=={cupy_version}"
        
        packages_to_try = [pkg]
    else:
        # Auto-detect CUDA and try multiple options
        detected_cuda = detect_cuda_version()
        
        if detected_cuda:
            print(f"Detected CUDA version: {detected_cuda}")
            
            if detected_cuda.startswith('12'):
                base_packages = ["cupy-cuda12x", "cupy-cuda11x", "cupy"]
            elif detected_cuda.startswith('11'):
                base_packages = ["cupy-cuda11x", "cupy-cuda12x", "cupy"]
            elif detected_cuda.startswith('10'):
                base_packages = ["cupy-cuda102", "cupy"]
            else:
                base_packages = ["cupy"]
        else:
            print("No CUDA detected, trying CPU-compatible versions")
            base_packages = ["cupy"]
        
        # Add version specifiers if requested
        if cupy_version:
            packages_to_try = [f"{pkg}=={cupy_version}" for pkg in base_packages]
        else:
            packages_to_try = base_packages
    
    # Try each package in order
    for package in packages_to_try:
        print(f"Attempting to install {package}...")
        
        if install_package(package):
            try:
                cupy = importlib.import_module("cupy")
                version = getattr(cupy, "__version__", "unknown")
                print(f"Successfully installed and imported CuPy {version}")
                
                # Verify CUDA is accessible
                try:
                    from cupy import cuda as cp_cuda
                    device_count = cp_cuda.runtime.getDeviceCount()
                    if device_count > 0:
                        print(f"✓ CuPy detected {device_count} CUDA device(s)")
                    else:
                        print("⚠ CuPy installed but no CUDA devices found")
                except:
                    print("⚠ CuPy installed but CUDA not accessible")
                
                return cupy
            except ImportError:
                print(f"Failed to import CuPy after installing {package}")
                continue
        else:
            print(f"Failed to install {package}")
    
    # If we get here, all installations failed
    warnings.warn("Could not install CuPy. Proceeding without GPU acceleration.")
    return None

def detect_cuda_version() -> Optional[str]:
    """Detect CUDA version on the system"""
    
    # Method 1: Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            driver_version = result.stdout.strip().split('.')[0]
            # Map driver version to approximate CUDA version
            driver_mapping = {
                '535': '12.2', '530': '12.1', '525': '12.0',
                '520': '11.8', '515': '11.7', '510': '11.6',
                '495': '11.5', '470': '11.4', '465': '11.3',
                '460': '11.2', '450': '11.0', '440': '10.2'
            }
            for drv, cuda in driver_mapping.items():
                if int(driver_version) >= int(drv):
                    return cuda
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    
    # Method 2: Check common CUDA installation paths
    import os
    from pathlib import Path
    
    cuda_paths = []
    if platform.system() == "Windows":
        cuda_paths = [
            Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")),
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        ]
    else:
        cuda_paths = [
            Path("/usr/local/cuda"),
            Path.home() / ".cuda",
            Path("/opt/cuda")
        ]
    
    for cuda_path in cuda_paths:
        version_file = cuda_path / "version.txt"
        if version_file.exists():
            try:
                with open(version_file) as f:
                    content = f.read()
                    # Look for version pattern
                    import re
                    match = re.search(r"CUDA Version (\d+\.\d+)", content)
                    if match:
                        return match.group(1)
            except:
                continue
    
    return None

def install_package(package_spec: str) -> bool:
    """Install a package using pip with error handling"""
    
    # Parse package spec (handles version specifiers like "package==1.2.3")
    if "==" in package_spec:
        package_name = package_spec.split("==")[0]
    elif ">=" in package_spec:
        package_name = package_spec.split(">=")[0]
    elif "<=" in package_spec:
        package_name = package_spec.split("<=")[0]
    else:
        package_name = package_spec
    
    # First check if already installed
    try:
        importlib.import_module(package_name.replace("-", "_"))
        print(f"{package_name} is already installed")
        return True
    except ImportError:
        pass
    
    print(f"Installing {package_spec}...")
    
    # Try using subprocess (more reliable than pip._internal)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully installed {package_spec}")
            return True
        else:
            # Try without version specifier if versioned install failed
            if "==" in package_spec:
                print(f"Version {package_spec} not available, trying latest...")
                base_package = package_spec.split("==")[0]
                return install_package(base_package)
            
            print(f"✗ Failed to install {package_spec}")
            print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Installation timed out for {package_spec}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error installing {package_spec}: {e}")
        return False

def setup_gpu_support(
  numpy_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
    cupy_version: Optional[str] = None,
    require_cuda: bool = False
) -> Tuple[Optional[object], Optional[object], bool]:
    """
    Setup NumPy and CuPy with optional version specifications
    
    Args:
        numpy_version: Specific NumPy version (e.g., "1.24.0") or None for latest
        cuda_version: CUDA version for CuPy ("12", "11", "10.2", etc.) or None to auto-detect
        cupy_version: Specific CuPy version (e.g., "12.0.0") or None for latest
        require_cuda: If True, will raise error if CUDA is not available
    
    Returns:
        Tuple of (numpy_module, cupy_module, has_gpu)
    """
    
    print("=" * 60)
    print("Setting up NumPy and CuPy...")
    
    # Install/import NumPy
    numpy = import_numpy(numpy_version)
    if numpy is None:
        raise ImportError("Failed to import/install NumPy")
    
    # Install/import CuPy
    cupy = import_cupy(cuda_version, cupy_version)
    
    # Check if GPU is available
    has_gpu = False
    if cupy:
        try:
            from cupy import cuda as cp_cuda
            if cp_cuda.runtime.getDeviceCount() > 0:
                has_gpu = True
                print("GPU acceleration is available")
            else:
                print("CuPy installed but no GPU devices found")
        except:
            print("CuPy installed but GPU access failed")
    
    if require_cuda and not has_gpu:
        raise RuntimeError("CUDA is required but not available")
    
    print("=" * 60)
    return numpy, cupy, has_gpu

def install_with_versions(
    numpy_version: str|None = None,
    cuda_version: str|None = None,
    cupy_version: str|None = None
) -> dict:
    """
    Simplified installer with version specifications
    
    Returns dictionary with installed modules and status
    """
    results = {
        "numpy": None,
        "cupy": None,
        "has_gpu": False,
        "numpy_version": None,
        "cupy_version": None
    }
    
    try:
        # Install NumPy
        results["numpy"] = import_numpy(numpy_version)
        if results["numpy"]:
            results["numpy_version"] = results["numpy"].__version__
        
        # Install CuPy
        results["cupy"] = import_cupy(cuda_version, cupy_version)
        if results["cupy"]:
            results["cupy_version"] = getattr(results["cupy"], "__version__", "unknown")
            
            # Check GPU availability
            try:
                from cupy import cuda as cp_cuda
                results["has_gpu"] = cp_cuda.runtime.getDeviceCount() > 0
            except:
                pass
        
        return results
        
    except Exception as e:
        print(f"Installation failed: {e}")
        return results

#numpy 2.4.0
#cupy 13.6.0