import os
import json
import random
from tqdm import tqdm  # 进度条库

# ========== 你可以在这里修改 start_index 和 num_files_to_process ==========
start_index = 80000
num_files_to_process = 120000
# ========================================================================

# 设置 JSON 文件所在目录
input_dir = '/mnt/user_data/wenwen/data/CLEVR_Sample_Data/output_scene_dir'


# 函数：根据文件名解析出其数值索引
# 文件名格式假设为：CLEVR_sample_xxxxxx.json
# 比如：CLEVR_sample_000001.json -> index = 1
#       CLEVR_sample_080000.json -> index = 80000
def parse_index(filename):
    # 先去掉后缀 .json，然后以 '_' 分隔，取最后一个部分
    # 比如 "CLEVR_sample_080000.json" -> "CLEVR_sample_080000" -> 分割 ["CLEVR","sample","080000"]
    # 取最后一个"080000"再转int
    return int(filename.replace('.json', '').split('_')[-1])


def generate_caption(data):
    """
    根据 JSON 数据中 objects 的信息生成 image_caption
    逻辑：
      1. 统计 objects 数量
      2. 对每个 object，从 size, material, color 三个属性中随机选择两个作为定语
      3. 结合 object 的 shape 生成描述
      4. 使用多个模板随机生成完整的 caption
    """
    objects = data.get("objects", [])
    num_objects = len(objects)
    descriptions = []

    for obj in objects:
        shape = obj.get("shape", "object")
        size = obj.get("size", "")
        material = obj.get("material", "")
        color = obj.get("color", "")

        # 从 size, material, color 中随机选择两个作为定语
        options = []
        if size:
            options.append(size)
        if material:
            options.append(material)
        if color:
            options.append(color)
        if len(options) >= 2:
            adjectives = random.sample(options, 2)
        else:
            adjectives = options

        # 判断是否需要冠词 "an"
        article = "an" if shape[0].lower() in ['a', 'e', 'i', 'o', 'u'] else "a"
        description = f"{article} {' '.join(adjectives)} {shape}"
        descriptions.append(description)

    # 将多个对象描述组合成一句话
    if not descriptions:
        objects_sentence = "no objects"
    elif len(descriptions) == 1:
        objects_sentence = descriptions[0]
    else:
        objects_sentence = ", ".join(descriptions[:-1]) + " and " + descriptions[-1]

    # 定义多个 caption 模板
    templates = [
        "This image contains {num_objects} objects: {objects_sentence}.",
        "In this picture, there are {num_objects} items visible, including {objects_sentence}.",
        "The scene shows {num_objects} objects: {objects_sentence}.",
        "There are {num_objects} objects in the image, such as {objects_sentence}.",
        "We see {num_objects} objects here: {objects_sentence}."
    ]

    # 随机选择一个模板生成 caption
    template = random.choice(templates)
    caption = template.format(num_objects=num_objects, objects_sentence=objects_sentence)
    return caption


# ========== 筛选要处理的文件 ==========

# 1. 获取该文件夹下所有 .json 文件
all_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

# 2. 为每个文件提取 (filename, index) 元组
files_with_index = [(f, parse_index(f)) for f in all_files]

# 3. 按 index 升序排序
files_with_index.sort(key=lambda x: x[1])

# 4. 过滤出 index >= start_index 的文件
filtered_files = [f for f, idx in files_with_index if idx >= start_index]

# 5. 取前 num_files_to_process 个文件
json_files = filtered_files[:num_files_to_process]

# ========== 处理并写回文件 ==========

for filename in tqdm(json_files, desc="Processing files", unit="file"):
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 生成 image_caption
    caption = generate_caption(data)
    data["image_caption"] = caption

    # 将更新后的数据写回文件（格式化输出）
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
