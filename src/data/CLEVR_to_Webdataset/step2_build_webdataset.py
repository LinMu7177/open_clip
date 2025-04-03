import os
import json
import webdataset as wds
from glob import glob

input_dir = "/mnt/user_data/wenwen/data/clevr/clevr_basic_100000/cc3m_style_dataset"
json_source_dir = "/mnt/user_data/wenwen/data/clevr/clevr_basic_100000/output_scene_qa"
output_dir = "/mnt/user_data/wenwen/data/clevr/clevr_basic_100000/webdataset_output"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted(glob(os.path.join(input_dir, "*.jpg")))
shard_size = 8000
count = 0  # 添加样本计数

# ✅ 注意这里是一个格式字符串，包含 %06d
shard_pattern = os.path.join(output_dir, "%05d.tar")
sink = wds.ShardWriter(shard_pattern, maxcount=shard_size)

for i, img_path in enumerate(image_files):
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(input_dir, base + ".txt")
    json_file = f"CLEVR_sample_{int(base):06d}.json"
    json_path = os.path.join(json_source_dir, json_file)

    if not (os.path.exists(txt_path) and os.path.exists(json_path)):
        print(f"[!] Skipping {base}, missing .txt or .json")
        continue

    with open(txt_path, "r") as f:
        caption = f.read()
    with open(json_path, "r") as f:
        metadata = f.read()

    sample = {
        "__key__": base,
        "jpg": open(img_path, "rb").read(),
        "txt": caption,
        "json": metadata
    }
    sink.write(sample)
    count += 1

sink.close()
print(f"✅ WebDataset with .json written successfully. Total samples written: {count}")
