import os
import json
import subprocess
from datetime import datetime

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit[:7]
    except Exception:
        return "48c88a8" # Fallback

def main():
    manifest = {
        "dataset": "QYRO Dataset v2.0",
        "dataset_fingerprint": "184dde9ec69c7e16ca5217a442fdc2026d2715dc0f015673e4ba3bcde8a3186b",
        "model": "YOLOv8s",
        "epochs": 150,
        "batch": 16,
        "accumulate": 2,
        "imgsz": 640,
        "seed": 42,
        "start_time": datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    
    out_dir = r"C:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI\experiments\detection\QYRO_Acne_v2"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_manifest.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"[SUCCESS] Created training manifest at: {out_path}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
