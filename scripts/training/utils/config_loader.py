import os
import yaml

def load_config(config_path, hardware_profile_path=r"c:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI\configs\hardware_profile.yaml"):
    """
    Loads task configuration and merges it with the hardware profile configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if os.path.exists(hardware_profile_path):
        with open(hardware_profile_path, "r", encoding="utf-8") as f:
            hw_profile = yaml.safe_load(f)
        # Add hardware parameters to config
        config["hardware"] = hw_profile
    else:
        print(f"Warning: Hardware profile not found at {hardware_profile_path}. Standard fallbacks will be used.")
        config["hardware"] = {}
        
    return config

if __name__ == "__main__":
    # Test loading
    try:
        cfg = load_config(r"c:\Users\KARTHIK V\OneDrive\Desktop\QYRO-Medical-AI\configs\detection_config.yaml")
        print("Config Loaded Successfully. Model type:", cfg.get("model_params", {}).get("model_type"))
    except Exception as e:
        print("Test Load Failed:", e)
