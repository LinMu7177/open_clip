import json
import os
import glob
import random
from tqdm import tqdm

# ========== 1. 设置输入输出路径 ==========

INPUT_DIR = "/mnt/user_data/wenwen/data/clevr/clevr_basic_100000/output_scene"
OUTPUT_DIR = "/mnt/user_data/wenwen/data/clevr/clevr_basic_100000/output_scene_qa"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 2. 全局常量 & 函数 ==========

MAIN_ATTRIBUTES = ["shape", "size", "material", "color"]
DIRECTION_KEYS = ["front", "behind", "left", "right"]

# 可以先定义一组常量，用来识别每个属性值属于哪一类
SHAPE_SET = {"sphere", "cube", "cylinder"}  # 根据你们数据集可能还有别的形状
SIZE_SET = {"small", "large"}
MATERIAL_SET = {"rubber", "metal"}
COLOR_SET = {"red", "blue", "yellow", "cyan", "gray", "brown", "green", "purple"}  # 视数据集而定

ATTR_ORDER = ["size", "material", "color", "shape"]  # 固定想要的输出顺序
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_ORDER)}  # 用于排序时给属性类型下标


def get_attr_type(value):
    """ 根据属性值判断它属于 size/material/color/shape 中哪一种。 """
    if value in SIZE_SET:
        return "size"
    elif value in MATERIAL_SET:
        return "material"
    elif value in COLOR_SET:
        return "color"
    elif value in SHAPE_SET:
        return "shape"
    else:
        return None  # 万一出现意外值，可以返回 None


def sort_combo(combo):
    """
    对 combo 里的属性值按照 size->material->color->shape 的顺序排列，
    返回排好序的字符串，例如 "small metal red cube"。
    """
    tmp = []
    for val in combo:
        attr_type = get_attr_type(val)
        if attr_type is not None:
            tmp.append((attr_type, val))
        else:
            # 如果出现 combo 里有未知属性值，可以选择直接加到末尾或跳过。
            # 这里简单处理：直接 append (None, val)，后续排序让它在最后
            tmp.append((None, val))

    # 按照属性类型排序，没有识别到的(None)自动排在后面
    tmp_sorted = sorted(tmp, key=lambda x: ATTR_INDEX.get(x[0], 999))
    # 只把属性值本身连起来
    return " ".join(val for _, val in tmp_sorted)


# --- (A) 属性问答 ---
def generate_property_questions(scene_data):
    """
    从 scene_data["unique_combinations"] 中取前 2 个组合，分别寻找匹配的 object；
    然后在其剩余未使用的属性里，随机选一个来提问。
    """
    if "unique_combinations" not in scene_data:
        return []

    objects = scene_data["objects"]
    combos = scene_data["unique_combinations"]

    qa_pairs = []

    for combo in combos:
        matched_idx = -1
        for i, obj in enumerate(objects):
            obj_attr_values = {obj["shape"], obj["size"], obj["material"], obj["color"]}
            if set(combo).issubset(obj_attr_values):
                matched_idx = i
                break

        # 如果没找到匹配的物体，则跳过
        if matched_idx == -1:
            continue

        matched_obj = objects[matched_idx]

        # used_attrs：物体中哪些属性值实际上落在 combo 里
        used_attrs = {}
        for attr in MAIN_ATTRIBUTES:
            if matched_obj[attr] in combo:
                used_attrs[attr] = matched_obj[attr]

        # 剩余可问的属性
        leftover_attrs = [attr for attr in MAIN_ATTRIBUTES if attr not in used_attrs]
        if not leftover_attrs:
            # combo 已经囊括了所有4种属性，没有可问的
            continue

        # 随机选一个要提问的属性
        leftover_attr = random.choice(leftover_attrs)

        # 拼接物体描述，如 "large cyan cube"
        desc_parts = [used_attrs[a] for a in MAIN_ATTRIBUTES if a in used_attrs]

        object_phrase = sort_combo(desc_parts)

        question = f"What is the {leftover_attr} of the {object_phrase} in the image?"
        answer = matched_obj[leftover_attr]
        qa_pairs.append((question, answer))

    return qa_pairs


# --- (B) 数目问答 ---
def generate_counting_questions(scene_data):
    """
    只随机挑选 2 个属性（最多生成 2 个问题），然后对每个选中属性随机挑选一个属性值进行统计。
    """
    objects = scene_data["objects"]
    qa_pairs = []

    # 收集每种属性下可能的取值
    attribute_values = {attr: set() for attr in MAIN_ATTRIBUTES}
    for obj in objects:
        for attr in MAIN_ATTRIBUTES:
            attribute_values[attr].add(obj[attr])

    # 在 MAIN_ATTRIBUTES 中随机选 2 个属性来问（如果属性种类少于 2，则只选到可用数量）
    chosen_attrs = random.sample(MAIN_ATTRIBUTES, min(2, len(MAIN_ATTRIBUTES)))

    for attr in chosen_attrs:
        vals = list(attribute_values[attr])
        if not vals:
            continue
        # 随机挑选一个属性值
        chosen_val = random.choice(vals)
        count = sum(obj[attr] == chosen_val for obj in objects)

        if attr == "shape":
            question = f"How many {chosen_val}s are in the image?"
        else:
            question = f"How many {chosen_val} objects are in the image?"

        answer = str(count)
        qa_pairs.append((question, answer))

    return qa_pairs


def generate_spatial_relationship_questions(scene_data):
    """
    至多生成 2 个“空间关系问答”。
    逻辑：
      1) 先为每个物体生成一个描述 desc_i：
         - 若存在 combo ∈ unique_combinations，可完整匹配该物体属性，则用 combo 拼接描述
         - 否则用 "color + shape"
      2) 收集所有 (i, dir_, j)，其中 relationships[dir_][i] == [j]
      3) 不再随机打乱，直接取前 2 个生成问答
      4) 每次问随机属性（如果不想随机，可以改成固定属性）
    """
    if "relationships" not in scene_data:
        return []

    objects = scene_data["objects"]
    relationships = scene_data["relationships"]
    combos = scene_data.get("unique_combinations", [])

    # 1) 为每个物体生成描述
    desc_list = []
    for obj_i in objects:
        # 把物体的 4 大属性放进一个 set，方便与 combo 做子集判断
        obj_i_attr_set = {
            obj_i["shape"],
            obj_i["size"],
            obj_i["material"],
            obj_i["color"]
        }

        # 在 combos 中寻找一个能匹配该物体的组合
        matched_desc = None
        for combo in combos:
            if set(combo).issubset(obj_i_attr_set):
                matched_desc = sort_combo(combo)
                break

        # 如果找不到匹配的 combo，回退到简单描述
        if matched_desc is None:
            matched_desc = f"{obj_i['color']} {obj_i['shape']}"

        desc_list.append(matched_desc)

    # 2) 收集关系 (i, dir_, j, 描述i)
    candidates = []
    for i, obj_i in enumerate(objects):
        for dir_ in DIRECTION_KEYS:
            if dir_ in relationships:
                related_indices = relationships[dir_][i]
                # 原逻辑：当且仅当 related_indices 中恰好只有 1 个元素
                if len(related_indices) == 1:
                    j = related_indices[0]
                    candidates.append((i, dir_, j, desc_list[i]))

    # 3) 不再随机打乱，直接取前 2 个
    chosen = candidates[:2]

    # 4) 生成问答
    qa_pairs = []
    for (i, dir_, j, desc_i) in chosen:
        # 如果也不想随机属性，可以改成 attr = "color" 之类的
        attr = random.choice(MAIN_ATTRIBUTES)
        answer = objects[j][attr]
        question = f"What is the {attr} of the object {dir_} the {desc_i}?"
        qa_pairs.append((question, answer))

    return qa_pairs


from tqdm import tqdm  # 在文件顶部加上这个

def main():
    random.seed(2023)

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    # total_files = min(len(json_files), 5)  # 限制最多处理 5 个文件

    for i, json_file in enumerate(tqdm(json_files, desc="Processing scenes")):
        filename = os.path.basename(json_file)

        with open(json_file, "r") as f:
            scene_data = json.load(f)

        property_qas = generate_property_questions(scene_data)
        counting_qas = generate_counting_questions(scene_data)
        spatial_qas = generate_spatial_relationship_questions(scene_data)

        qa_dict = {
            "property": [{"question": q, "answer": a} for (q, a) in property_qas],
            "counting": [{"question": q, "answer": a} for (q, a) in counting_qas],
            "spatial": [{"question": q, "answer": a} for (q, a) in spatial_qas],
        }

        output_data = scene_data.copy()
        output_data["QA"] = qa_dict
        output_data["scene_filename"] = filename

        base_name = os.path.splitext(filename)[0]
        out_file_name = base_name + ".json"
        out_file_path = os.path.join(OUTPUT_DIR, out_file_name)

        with open(out_file_path, "w") as f:
            json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()