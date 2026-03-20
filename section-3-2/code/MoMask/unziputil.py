import os
from zipfile import PyZipFile

BASE_PATH = "./checkpoints"
files = [("t2m", "humanml3d_models.zip"), ("kit", "kit_models.zip")]

for path in files:
    extract_path = os.path.join(BASE_PATH, path[0])
    zip_file = os.path.join(BASE_PATH, os.path.join(path[0], path[1]))

    pzf = PyZipFile(zip_file)
    pzf.extractall(extract_path)