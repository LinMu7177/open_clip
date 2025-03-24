import os
import json
from PIL import Image

input_image_dir = "/mnt/user_data/wenwen/data/CLEVR_Sample_Data/images_80000_199999"
input_json_dir = "/mnt/user_data/wenwen/data/CLEVR_Sample_Data/output_scene_dir"
output_dir = "/mnt/user_data/wenwen/data/CLEVR_Sample_Data/cc3m_style_dataset_80000_199999"
os.makedirs(output_dir, exist_ok=True)

json_files = sorted(os.listdir(input_json_dir))
valid_count = 0
start_index = 80000  # 在这里指定起始编号，想从080000开始就设为80000

for i, json_file in enumerate(json_files):
    with open(os.path.join(input_json_dir, json_file), "r") as f:
        data = json.load(f)

    # 检查 image_caption 是否存在
    if "image_caption" not in data:
        print(f"[!] Skipping {json_file} (no image_caption)")
        continue

    caption = data["image_caption"]
    img_filename = data["image_filename"]
    img_path = os.path.join(input_image_dir, img_filename)

    if not os.path.exists(img_path):
        print(f"[!] Missing image: {img_filename}")
        continue

    sample_id = f"{start_index + valid_count:06d}"
    output_img_path = os.path.join(output_dir, f"{sample_id}.jpg")
    output_txt_path = os.path.join(output_dir, f"{sample_id}.txt")

    try:
        with Image.open(img_path) as img:
            img.convert("RGB").save(output_img_path, "JPEG")
        with open(output_txt_path, "w") as f:
            f.write(caption)
        valid_count += 1
    except Exception as e:
        print(f"[X] Error processing {img_filename}: {e}")

    if valid_count % 1000 == 0 and valid_count > 0:
        print(f"[+] Processed {valid_count} valid samples")

print(f"✅ Step 1 complete: {valid_count} valid samples saved.")
