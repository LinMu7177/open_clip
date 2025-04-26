#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
from shutil import copy2
from tqdm import tqdm

def embed_edges(scene_dir: str,
                edges_dir: str,
                key_name: str = "edges",
                backup: bool = False):
    # 遍历所有 CLEVR_sample_XXXXX.json
    for fname in tqdm(sorted(os.listdir(scene_dir))):
        if not fname.endswith('.json'):
            continue

        scene_path = os.path.join(scene_dir, fname)
        stem = fname.replace('.json', '')
        pkl_path = os.path.join(edges_dir, f"{stem}_edges.pkl")
        if not os.path.exists(pkl_path):
            print(f"[Warning] 找不到 pkl: {pkl_path}")
            continue

        # 备份
        if backup:
            copy2(scene_path, scene_path + ".bak")

        # 1) 读取 scene JSON
        with open(scene_path, 'r') as f:
            scene = json.load(f)

        # 2) 读取 pkl & 处理 counts
        with open(pkl_path, 'rb') as f:
            edge_data = pickle.load(f)     # {'size':[H,W], 'counts': bytes|str}
        counts = edge_data['counts']
        if isinstance(counts, bytes):
            counts = counts.decode('ascii')    # COCO 推荐用 ascii 字符串
        scene[key_name] = {
            'size': edge_data['size'],
            'counts': counts
        }

        # 3) 覆盖写回
        with open(scene_path, 'w') as f:
            json.dump(scene, f)
    print("✅  All done – edges 已写入 scene JSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", default='/mnt/user_data/wenwen/data/clevr/clevr_40000/output_scene_pn',
                        help="存放 CLEVR_*_XXXX.json 的目录")
    parser.add_argument("--edges_dir", default='/mnt/shared/data/DINO_SAM2_Data/clevr/clevr_40000',
                        help="存放 *_edges.pkl 的目录")
    parser.add_argument("--key_name", default="edges",
                        help="写入 JSON 的字段名（默认 edges）")
    parser.add_argument("--backup", action='store_true',
                        help="写入前先复制 .json 为 .json.bak")
    args = parser.parse_args()
    embed_edges(**vars(args))
