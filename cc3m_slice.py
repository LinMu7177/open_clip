#!/usr/bin/env python3

import argparse
import os
import json
import math
import pandas as pd
import tarfile
import shutil

def format_index(index):
    """
    将数字转换为5位数的字符串格式，例如：0 -> '00000'
    """
    return f"{index:05d}"


def slice_parquet(parquet_path, output_dir, index_prefix, num_slices=10):
    """
    读取 parquet 文件并切分为 num_slices 个更小的文件。
    """
    print(f"Reading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    slice_size = math.ceil(total_rows / num_slices)
    print(f"Slice size: {slice_size}, total slices: {num_slices}")

    for i in range(num_slices):
        start_idx = i * slice_size
        end_idx = min((i + 1) * slice_size, total_rows)
        if start_idx >= total_rows:
            break
        df_slice = df.iloc[start_idx:end_idx]

        output_path = os.path.join(output_dir, f"{format_index(i)}.parquet")
        df_slice.to_parquet(output_path)
        print(f"Saved slice {i} to {output_path}")


def slice_json(json_path, output_dir, index_prefix, json_type, num_slices=10):
    """
    读取 JSON 文件并切分为 num_slices 个更小的文件。
    json_type: 'captions' 或 'stats'，用于区分文件类型
    """
    print(f"Reading {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported JSON data type: {type(data)}")

    total_items = len(items)
    print(f"Total JSON items: {total_items}")

    slice_size = math.ceil(total_items / num_slices)

    for i in range(num_slices):
        start_idx = i * slice_size
        end_idx = min((i + 1) * slice_size, total_items)
        if start_idx >= total_items:
            break

        if isinstance(data, dict):
            data_slice = dict(items[start_idx:end_idx])
        else:
            data_slice = items[start_idx:end_idx]

        output_path = os.path.join(output_dir, f"{format_index(i)}_{json_type}.json")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data_slice, f_out, ensure_ascii=False, indent=2)
        print(f"Saved {json_type} JSON slice {i} to {output_path}")


def slice_tar(tar_path, output_dir, index_prefix, num_slices=10):
    """
    切分 tar 文件为多个小的 tar 文件
    """
    print(f"Reading {tar_path} ...")

    # 创建临时目录用于解压
    temp_dir = os.path.join(output_dir, f"temp_{index_prefix}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 解压 tar 文件
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_dir)

        # 获取所有解压的文件
        files = []
        for root, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        total_files = len(files)
        print(f"Total files in tar: {total_files}")

        slice_size = math.ceil(total_files / num_slices)

        # 按切片创建新的 tar 文件
        for i in range(num_slices):
            start_idx = i * slice_size
            end_idx = min((i + 1) * slice_size, total_files)
            if start_idx >= total_files:
                break

            current_slice = files[start_idx:end_idx]
            output_tar = os.path.join(output_dir, f"{format_index(i)}.tar")

            with tarfile.open(output_tar, 'w') as new_tar:
                for file_path in current_slice:
                    # 获取相对路径，避免在tar中包含完整路径
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_tar.add(file_path, arcname=arcname)

            print(f"Saved tar slice {i} to {output_tar}")

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Slice CC3M files by given index.")
    parser.add_argument("--index", required=True, type=str,
                        help="Index of the CC3M file, e.g., 00000, 00047, etc.")
    parser.add_argument("--source_dir", required=True, type=str,
                        help="Path to the original CC3M directory.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory to save sliced files.")
    parser.add_argument("--num_slices", default=10, type=int,
                        help="Number of slices to split the files into. Default is 10.")

    args = parser.parse_args()

    index_str = args.index
    source_dir = args.source_dir
    output_dir = args.output_dir
    num_slices = args.num_slices

    # 创建专门的输出目录，避免文件混乱
    slice_output_dir = os.path.join(output_dir, index_str)
    os.makedirs(slice_output_dir, exist_ok=True)

    # 处理 parquet 文件
    parquet_file = os.path.join(source_dir, f"{index_str}.parquet")
    if os.path.isfile(parquet_file):
        slice_parquet(parquet_file, slice_output_dir, index_str, num_slices=num_slices)
    else:
        print(f"[WARNING] Parquet file {parquet_file} not found. Skip.")

    # 处理 captions.json 文件
    captions_file = os.path.join(source_dir, f"{index_str}_captions.json")
    if os.path.isfile(captions_file):
        slice_json(captions_file, slice_output_dir, index_str, "captions", num_slices=num_slices)
    else:
        print(f"[WARNING] Captions JSON {captions_file} not found. Skip.")

    # 处理 stats.json 文件
    stats_file = os.path.join(source_dir, f"{index_str}_stats.json")
    if os.path.isfile(stats_file):
        slice_json(stats_file, slice_output_dir, index_str, "stats", num_slices=num_slices)
    else:
        print(f"[WARNING] Stats JSON {stats_file} not found. Skip.")

    # 处理 tar 文件
    tar_file = os.path.join(source_dir, f"{index_str}.tar")
    if os.path.isfile(tar_file):
        slice_tar(tar_file, slice_output_dir, index_str, num_slices=num_slices)
    else:
        print(f"[WARNING] TAR file {tar_file} not found. Skip.")


if __name__ == "__main__":
    main()