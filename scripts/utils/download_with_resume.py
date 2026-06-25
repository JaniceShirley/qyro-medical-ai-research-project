import os
import sys
import time
import requests

def download_file(url, local_filename):
    """
    Downloads a file from a URL supporting HTTP Range Requests to resume downloads if interrupted.
    """
    temp_filename = local_filename + ".tmp"
    headers = {}
    
    if os.path.exists(temp_filename):
        temp_size = os.path.getsize(temp_filename)
        # Request only the remaining bytes
        headers['Range'] = f'bytes={temp_size}-'
        print(f"Resuming download of {local_filename} from byte {temp_size}...")
        mode = 'ab'
    else:
        temp_size = 0
        mode = 'wb'
        print(f"Starting fresh download of {local_filename}...")
        
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
    except Exception as e:
        print(f"Connection error: {e}")
        return False
        
    status = response.status_code
    
    if status == 206: # Partial Content (Range request accepted)
        content_range = response.headers.get('content-range', '')
        total_size = int(content_range.split('/')[-1]) if '/' in content_range else 0
        print("Server accepted resume request.")
    elif status == 200: # OK (Server didn't accept range, or starting fresh)
        total_size = int(response.headers.get('content-length', 0))
        if temp_size > 0:
            print("Server does not support resuming. Restarting download from scratch...")
            mode = 'wb'
            temp_size = 0
    else:
        print(f"Error: Server returned status code {status}")
        return False
        
    chunk_size = 1024 * 1024 # 1MB chunks
    bytes_downloaded = temp_size
    start_time = time.time()
    
    try:
        with open(temp_filename, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    bytes_downloaded += len(chunk)
                    # Print download status periodically
                    elapsed = time.time() - start_time
                    if elapsed > 10: # Print every 10 seconds
                        speed = (bytes_downloaded - temp_size) / (1024 * 1024 * elapsed)
                        pct = (bytes_downloaded / total_size * 100) if total_size > 0 else 0
                        print(f"  Downloaded: {bytes_downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.2f}%) | Speed: {speed:.2f} MB/s")
                        start_time = time.time()
                        temp_size = bytes_downloaded
    except Exception as e:
        print(f"Download interrupted: {e}")
        return False
        
    # Check if download finished
    actual_size = os.path.getsize(temp_filename)
    if total_size > 0 and actual_size < total_size:
        print(f"Download incomplete (got {actual_size} bytes, expected {total_size} bytes). Run the script again to resume.")
        return False
        
    # Rename temp file to final filename
    if os.path.exists(local_filename):
        os.remove(local_filename)
    os.rename(temp_filename, local_filename)
    print(f"Success! Saved file to {local_filename}")
    return True

if __name__ == "__main__":
    # URL configurations
    torch_url = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl"
    torchvision_url = "https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp310-cp310-win_amd64.whl"
    
    print("Downloading PyTorch GPU wheel...")
    t_ok = download_file(torch_url, "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl")
    
    if t_ok:
        print("\nDownloading Torchvision GPU wheel...")
        v_ok = download_file(torchvision_url, "torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl")
        if v_ok:
            print("\nAll downloads completed successfully!")
            sys.exit(0)
            
    sys.exit(1)
