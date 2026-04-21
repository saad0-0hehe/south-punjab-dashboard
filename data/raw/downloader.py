import requests
import os
import sys

def download_file(url, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Downloading {url}...")
    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded to {path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw")
    
    # Files identified earlier
    files_to_download = [
        ("https://www.pbs.gov.pk/wp-content/uploads/2020/07/PSLM_Report_2024-25-Social-1.pdf", "PSLM_Report_2024-25.pdf"),
        ("https://www.pbs.gov.pk/wp-content/uploads/2020/07/pcr_punjab.pdf", "Census_2017_Punjab_Report.pdf")
    ]
    
    for url, name in files_to_download:
        download_file(url, raw_dir, name)
