import json
import os
import glob
import random
from tqdm import tqdm

MAIN_ATTRIBUTES = ["shape", "size", "material", "color"]
DIRECTION_KEYS  = ["front", "behind", "left", "right"]

SHAPE_SET    = {"sphere", "cube", "cylinder"}
SIZE_SET     = {"small", "large"}
MATERIAL_SET = {"rubber", "metal"}
COLOR_SET    = {"red", "blue", "yellow", "cyan", "gray", "brown", "green", "purple"}

ATTR_SETS = {
    "shape":    SHAPE_SET,
    "size":     SIZE_SET,
    "material": MATERIAL_SET,
    "color":    COLOR_SET,
}

ATTR_ORDER = ["size", "material", "color", "shape"]
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_ORDER)}

# ========= 3. 辅助函数 =========
def get_attr_type(value: str):
    if value in SIZE_SET:
        return "size"
    if value in MATERIAL_SET:
        return "material"
    if value in COLOR_SET:
        return "color"
    if value in SHAPE_SET:
        return "shape"
    return None

def sort_combo(combo):
    """把若干属性值拼成固定顺序的人类可读描述"""
    tagged = [(get_attr_type(v), v) for v in combo]
    tagged.sort(key=lambda x: ATTR_INDEX.get(x[0], 999))
    return " ".join(v for _, v in tagged)

def random_negative(attr: str, correct_val: str):
    """给定属性类型，随机挑一个与正确值不同的取值"""
    candidates = list(ATTR_SETS[attr] - {correct_val})
    return random.choice(candidates) if candidates else correct_val

# ========= 4‑A. 单属性 P‑N =========
def generate_property_pns(scene):
    if "unique_combinations" not in scene:
        return []

    objects = scene["objects"]
    combos  = scene["unique_combinations"]
    pns = []

    for combo in combos:
        # 找到能完整匹配 combo 的第一个物体
        tgt_idx = next(
            (i for i, obj in enumerate(objects)
             if set(combo).issubset(
                 {obj["shape"], obj["size"], obj["material"], obj["color"]})),
            -1,
        )
        if tgt_idx == -1:
            continue

        obj = objects[tgt_idx]
        # combo 里已出现的属性
        used_attrs = {a: obj[a] for a in MAIN_ATTRIBUTES if obj[a] in combo}
        leftover_attrs = [a for a in MAIN_ATTRIBUTES if a not in used_attrs]
        if not leftover_attrs:
            continue

        ask_attr = random.choice(leftover_attrs)
        obj_phrase = sort_combo(list(used_attrs.values()))
        correct_val = obj[ask_attr]
        wrong_val   = random_negative(ask_attr, correct_val)

        pos = f"The {ask_attr} of the {obj_phrase} is {correct_val}."
        neg = f"The {ask_attr} of the {obj_phrase} is {wrong_val}."
        pns.append((pos, neg))

    return pns

# ========= 4‑B. 计数 P‑N =========
def generate_counting_pns(scene):
    objects = scene["objects"]
    pns = []

    # 汇总每种属性可能的取值
    attr_vals = {a: {obj[a] for obj in objects} for a in MAIN_ATTRIBUTES}
    chosen_attrs = random.sample(MAIN_ATTRIBUTES, min(2, len(MAIN_ATTRIBUTES)))

    for attr in chosen_attrs:
        vals = list(attr_vals[attr])
        if not vals:
            continue
        val = random.choice(vals)
        true_cnt = sum(obj[attr] == val for obj in objects)

        # 决定一个错误计数：±1 且不同于真值，且 >=0
        wrong_cnt = max(0, true_cnt + random.choice([-2, -1, 1, 2]))
        if wrong_cnt == true_cnt:
            wrong_cnt += 1

        noun = f"{val}s" if attr == "shape" else f"{val} objects"
        pos = f"There are {true_cnt} {noun} in the image."
        neg = f"There are {wrong_cnt} {noun} in the image."
        pns.append((pos, neg))

    return pns

# ========= 4‑C. 空间关系 P‑N =========
def generate_spatial_pns(scene, max_pairs: int = 2):
    """
    Positive : “<Obj‑A‑描述> <真实方位介词> <Obj‑B‑描述>.”
    Negative : 同一对物体，把方位换成错误的方向
    """
    if "relationships" not in scene:
        return []

    objs   = scene["objects"]
    rels   = scene["relationships"]
    combos = scene.get("unique_combinations", [])

    # 给每个物体做一句描述
    descs = []
    for obj in objs:
        attr_set = {obj[a] for a in MAIN_ATTRIBUTES}
        matched  = next((sort_combo(c) for c in combos
                         if set(c).issubset(attr_set)), None)
        descs.append(matched or f"{obj['color']} {obj['shape']}")

    # CLEVR ↔️ 文本里的介词映射
    DIR_PHRASE = {
        "front":  "in front of",
        "behind": "behind",
        "left":   "to the left of",
        "right":  "to the right of",
    }

    # 收集所有 (i, dir, j) 关系；若有多个邻居，随机挑一个
    triples = []
    for i, _ in enumerate(objs):
        for d in DIRECTION_KEYS:
            neigh = rels.get(d, [[]])[i]
            if neigh:                                    # 至少有 1 个邻居
                j = random.choice(neigh)                 # 随机挑 1 个
                triples.append((i, d, j))

    random.shuffle(triples)
    triples = triples[:max_pairs]

    pns = []
    for i, dir_true, j in triples:
        src_desc  = descs[i]
        tgt_desc  = descs[j]

        # 正例：真实方向
        pos = f"The {src_desc} is {DIR_PHRASE[dir_true]} the {tgt_desc}."

        # 负例：挑一个错误方向，确保(j) 不在该方向上
        wrong_dirs = [d for d in DIRECTION_KEYS
                      if d != dir_true and j not in rels.get(d, [[]])[i]]
        # 若极端情况全部方向都成立（几乎不会），就简单用不同于真值的方向
        if not wrong_dirs:
            wrong_dirs = [d for d in DIRECTION_KEYS if d != dir_true]

        dir_false = random.choice(wrong_dirs)
        neg       = f"The {src_desc} is {DIR_PHRASE[dir_false]} the {tgt_desc}."

        pns.append((pos, neg))

    return pns


def generate_pns(scene, neg_type):
    if neg_type == 'fix':
        properties = {
            "property_pos": scene["PN"]["property"][0]["Positive"],
            "property_neg": scene["PN"]["property"][0]["Negative"],
            "counting_pos": scene["PN"]["counting"][0]["Positive"],
            "counting_neg": scene["PN"]["counting"][0]["Negative"],
            "spatial_pos": scene["PN"]["spatial"][0]["Positive"],
            "spatial_neg": scene["PN"]["spatial"][0]["Negative"],
        }
    elif neg_type == 'dynamic':
        properties = {
            "property_pos": generate_property_pns(scene)[0][0],
            "property_neg": generate_property_pns(scene)[0][1],
            "counting_pos": generate_counting_pns(scene)[0][0],
            "counting_neg": generate_counting_pns(scene)[0][1],
            "spatial_pos": generate_spatial_pns(scene)[0][0],
            "spatial_neg": generate_spatial_pns(scene)[0][1],
        }
    return properties
