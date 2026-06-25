import zipfile
import sys

zip_path = 'annotation_batches/batch_1_cvat_import.zip'
try:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print("ZIP is valid.")
        namelist = zf.namelist()
        print("Directory tree inside ZIP:")
        for name in sorted(namelist):
            print(f"- {name}")
except zipfile.BadZipFile:
    print("ZIP is corrupted.")
