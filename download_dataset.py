import urllib.request
import zipfile
import os

url = "https://github.com/emanhamed/Houses-dataset/archive/refs/heads/master.zip"
zip_path = "Houses-dataset.zip"
extract_dir = "data_extracted"

print(f"Downloading {url} to {zip_path}...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete.")

print(f"Extracting {zip_path} to {extract_dir}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extraction complete.")
