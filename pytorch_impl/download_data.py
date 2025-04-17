import requests
import zipfile
import os
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "images_food101"

if image_path.is_dir():
    print(f"{image_path} already exists!")

else:
    print(f"creating {image_path} directory")
    image_path.mkdir(parents=True, exist_ok=True)

#Download the file 
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    requests = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print(f"Downloading image datasets...")
    f.write(requests.content)

#unzip the files 
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zipf:
    print("Unzipping the images")
    zipf.extractall(image_path)

print("Deleting the zip file")
os.remove(data_path / "pizza_steak_sushi.zip")


