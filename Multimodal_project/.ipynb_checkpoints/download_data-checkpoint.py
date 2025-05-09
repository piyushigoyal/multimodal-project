from datasets import load_dataset
from PIL import Image
import os, json

# 1) Load the test split (will download images and parquet behind the scenes)
ds = load_dataset("AI4Math/MathVista", split="testmini")

# 2) Make your output folders
output_dir = "mathvista_data/testmini"
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)
jsonl_path = os.path.join(output_dir, "data.jsonl")

# 3) Iterate, save the decoded_image, emit metadata
with open(jsonl_path, "w", encoding="utf-8") as out_f:
    for idx, item in enumerate(ds):
        img = item["decoded_image"]
        # Some HF versions give a NumPy array—convert if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        dst = os.path.join(images_dir, f"{idx:06d}.png")
        img.save(dst)

        # copy everything but the image fields
        record = {k: v for k, v in item.items() if k not in ("image", "decoded_image")}
        record["image_path"] = dst

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Dataset saved to {jsonl_path}")