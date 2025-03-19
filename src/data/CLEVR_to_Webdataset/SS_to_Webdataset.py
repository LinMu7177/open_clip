import os
import json
import webdataset as wds
from glob import glob
import uuid  # 用于生成唯一标识符

input_jsonl_file = '/home/yifei/code/Qwen_Labeler/output/SpatialSense.jsonl'
output_dir = '/mnt/shared/data/SpatialSense_webdataset'

os.makedirs(output_dir, exist_ok=True)

shard_size = 8000
count = 0  # 添加样本计数

shard_pattern = os.path.join(output_dir, "%05d.tar")
sink = wds.ShardWriter(shard_pattern, maxcount=shard_size)

with open(input_jsonl_file, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    data = json.loads(line)
    img_path = data['img']

    # 生成唯一标识符
    base = str(uuid.uuid4())

    txt = data['positive_sample']
    metadata = json.dumps(data).encode('utf-8')

    sample = {
        "__key__": base,
        "jpg": open(img_path, "rb").read(),
        "txt": txt.encode('utf-8'),
        "json": metadata
    }
    sink.write(sample)
    count += 1

sink.close()
print(f"✅ WebDataset with .json written successfully. Total samples written: {count}")
