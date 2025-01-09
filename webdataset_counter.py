import tarfile
import os

# 数据集路径
train_data_path = "/mnt/shared/data/cc3m/train/"
start_index = 0
end_index = 300


# 统计单个 tar 文件中的样本数量
def count_samples_in_tar(tar_path):
    count = 0
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if member.isfile():  # 只统计文件，不统计文件夹
                    count += 1
    except Exception as e:
        print(f"Error reading {tar_path}: {e}")
    return count


# 统计所有 tar 文件中的样本
total_samples = 0

for i in range(start_index, end_index + 1):
    tar_name = f"{str(i).zfill(5)}.tar"
    tar_path = os.path.join(train_data_path, tar_name)

    if os.path.exists(tar_path):
        sample_count = count_samples_in_tar(tar_path)
        print(f"{tar_name}: {sample_count} samples")
        total_samples += sample_count
    else:
        print(f"{tar_name} not found.")

print(f"Total samples: {total_samples}")
