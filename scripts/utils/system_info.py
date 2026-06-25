import sys
import os
import platform
import subprocess
import torch

def get_cpu_info():
    """Extracts CPU model name depending on OS."""
    system = platform.system()
    if system == "Windows":
        try:
            return subprocess.check_output("wmic cpu get name", shell=True).decode().split("\n")[1].strip()
        except Exception:
            return platform.processor()
    elif system == "Darwin":
        try:
            return subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
        except Exception:
            return platform.processor()
    else: # Linux
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor()

def get_ram_info():
    """Extracts total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        # Fallback for Windows without psutil
        if platform.system() == "Windows":
            try:
                out = subprocess.check_output("wmic computersystem get totalphysicalmemory", shell=True)
                bytes_ram = int(out.decode().split("\n")[1].strip())
                return round(bytes_ram / (1024**3), 2)
            except Exception:
                pass
        return "Unknown (Install psutil)"

def get_gpu_info():
    """Gets GPU models, CUDA availability, and total GPU VRAM."""
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        return {
            "gpu_available": False,
            "gpu_model": "None (CPU Only)",
            "gpu_vram_gb": 0.0,
            "cuda_version": "None"
        }
    
    model = torch.cuda.get_device_name(0)
    # Get total VRAM from PyTorch or nvidia-smi fallback
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(vram_bytes / (1024**3), 2)
    cuda_ver = torch.version.cuda
    
    return {
        "gpu_available": True,
        "gpu_model": model,
        "gpu_vram_gb": vram_gb,
        "cuda_version": cuda_ver
    }

def get_system_specs():
    """Compiles all hardware and software specifications into a dictionary."""
    gpu_specs = get_gpu_info()
    specs = {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "python_version": sys.version.split(" ")[0],
        "pytorch_version": torch.__version__,
        "cpu_model": get_cpu_info(),
        "system_ram_gb": get_ram_info(),
        "gpu_available": gpu_specs["gpu_available"],
        "gpu_model": gpu_specs["gpu_model"],
        "gpu_vram_gb": gpu_specs["gpu_vram_gb"],
        "cuda_version": gpu_specs["cuda_version"]
    }
    return specs

if __name__ == "__main__":
    specs = get_system_specs()
    print("QYRO Training Hardware Profile:")
    print("================================")
    for k, v in specs.items():
        print(f"{k.replace('_', ' ').upper()}: {v}")
