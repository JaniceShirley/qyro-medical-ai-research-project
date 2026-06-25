"""
Robust chunked downloader for PyTorch CUDA wheel.
Handles connection resets by resuming from the last downloaded byte.
"""
import os
import sys
import time
import urllib.request
import urllib.error

URL = "https://download-r2.pytorch.org/whl/cu130/torch-2.12.0%2Bcu130-cp314-cp314-win_amd64.whl"
OUTPUT = "torch_cu130.whl"
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
MAX_RETRIES = 50
RETRY_DELAY = 5  # seconds

def get_file_size():
    """Get the current size of the partial download."""
    if os.path.exists(OUTPUT):
        return os.path.getsize(OUTPUT)
    return 0

def download_with_resume():
    """Download file with resume support."""
    retries = 0
    
    while retries < MAX_RETRIES:
        downloaded = get_file_size()
        
        try:
            req = urllib.request.Request(URL)
            if downloaded > 0:
                req.add_header('Range', f'bytes={downloaded}-')
                print(f"\n[Resuming from {downloaded / (1024**2):.1f} MB] Attempt {retries + 1}/{MAX_RETRIES}")
            else:
                print(f"\n[Starting download] Attempt {retries + 1}/{MAX_RETRIES}")
            
            response = urllib.request.urlopen(req, timeout=60)
            
            # Get total size
            content_range = response.headers.get('Content-Range', '')
            if content_range:
                total_size = int(content_range.split('/')[-1])
            else:
                total_size = int(response.headers.get('Content-Length', 0)) + downloaded
            
            total_mb = total_size / (1024**2)
            
            mode = 'ab' if downloaded > 0 else 'wb'
            with open(OUTPUT, mode) as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / total_size) * 100 if total_size else 0
                    print(f"\r  Progress: {downloaded / (1024**2):.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end='', flush=True)
            
            # Check if complete
            if downloaded >= total_size:
                print(f"\n\n✅ Download complete! {OUTPUT} ({downloaded / (1024**2):.1f} MB)")
                return True
            else:
                print(f"\n⚠️ Incomplete download ({downloaded}/{total_size} bytes). Retrying...")
                retries += 1
                time.sleep(RETRY_DELAY)
                
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as e:
            retries += 1
            downloaded_now = get_file_size()
            print(f"\n⚠️ Connection error at {downloaded_now / (1024**2):.1f} MB: {e}")
            print(f"   Retrying in {RETRY_DELAY}s... ({retries}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            retries += 1
            time.sleep(RETRY_DELAY)
    
    print(f"\n❌ Failed after {MAX_RETRIES} retries. Downloaded {get_file_size() / (1024**2):.1f} MB")
    return False

if __name__ == '__main__':
    print("=" * 60)
    print("PyTorch CUDA 13.0 Wheel Downloader (with resume)")
    print(f"Target: {URL.split('/')[-1]}")
    print(f"Output: {OUTPUT}")
    print("=" * 60)
    
    success = download_with_resume()
    sys.exit(0 if success else 1)
