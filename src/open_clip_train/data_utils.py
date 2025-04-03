"""
该模块包含从 data.py 中独立出来的关于 objects_sense 处理的相关函数，
以及该函数所依赖的工具和辅助方法，确保在迁移时所有必要的依赖都已导入。
"""
import os
import csv
import copy
import logging
import pickle
import warnings
import math

import numpy as np
import torch
from torch import Tensor
import webdataset as wds

from torchvision.transforms import Normalize
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

import pycocotools.mask as mask_util


class JointRandomResizedCrop(torch.nn.Module):
    """
    对图像和 objects_sense 同时应用随机裁剪，保证两者裁剪方式一致。
    """
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BILINEAR,
            antialias: bool = True,
            mask_interpolation=InterpolationMode.NEAREST,
            mask_antialias: bool = False
    ):
        super().__init__()
        self.size = self._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, (list, tuple)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (list, tuple)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = F._interpolation_modes_from_int(interpolation)
        if isinstance(mask_interpolation, int):
            mask_interpolation = F._interpolation_modes_from_int(mask_interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio
        self.mask_interpolation = mask_interpolation
        self.mask_antialias = mask_antialias

    @staticmethod
    def _setup_size(size, error_msg):
        if isinstance(size, (int, float)):
            return int(size), int(size)
        if isinstance(size, (list, tuple)) and len(size) == 1:
            return size[0], size[0]
        if len(size) != 2:
            raise ValueError(error_msg)
        return size

    @staticmethod
    def get_params(img: Tensor, scale, ratio):
        _, height, width = F.get_dimensions(img)
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, sample: dict) -> dict:
        img = sample["image"]
        msk = sample["objects_sense"]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        img = F.resized_crop(
            img, i, j, h, w,
            self.size,
            self.interpolation,
            antialias=self.antialias
        )

        msk = F.resized_crop(
            msk, i, j, h, w,
            self.size,
            self.mask_interpolation,
            antialias=self.mask_antialias
        )

        sample["image"] = img
        sample["objects_sense"] = msk
        return sample

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        mask_interpolate_str = self.mask_interpolation.value
        return (f"{self.__class__.__name__}(size={self.size}, "
                f"scale={tuple(round(s, 4) for s in self.scale)}, "
                f"ratio={tuple(round(r, 4) for r in self.ratio)}, "
                f"interpolation={interpolate_str}, "
                f"antialias={self.antialias}, "
                f"mask_interpolation={mask_interpolate_str}, "
                f"mask_antialias={self.mask_antialias})")


join_preprocess = JointRandomResizedCrop(
    size=(224, 224),
    scale=(0.9, 1.0),
    ratio=(0.75, 1.3333),
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
    mask_interpolation=InterpolationMode.NEAREST,
    mask_antialias=False
)

# 定义 objects_sense 的归一化
objects_sense_normalize = Normalize(mean=[0.5], std=[0.26])


def filter_no_caption_or_no_image(sample):
    """
    过滤掉没有 caption 或没有图像的数据样本。
    """
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """
    异常处理函数：捕获 webdataset 错误，输出警告信息，并继续处理。
    """
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def load_edges(edges_demo_path, image_shape):
    """
    尝试从 pickle 文件中加载边缘信息，如果不存在则返回默认的掩码。
    """
    if os.path.exists(edges_demo_path):
        with open(edges_demo_path, 'rb') as f:
            combined_edges = pickle.load(f)
        rle = {'size': combined_edges['size'], 'counts': combined_edges['counts']}
        mask = mask_util.decode(rle)
        return mask
    else:
        return np.ones(image_shape[:2], dtype=np.uint8)


def get_objects_sense(key, image, objects_sense_format, objects_data):
    if objects_sense_format == 'edges':
        objects_sense_path = os.path.join(objects_data, key + '_edges.pkl')
        edges = load_edges(objects_sense_path, image.size)
    # 将边缘信息转换为 tensor，并进行归一化
    edges = torch.as_tensor(edges).unsqueeze(0).half() * 255
    edges = objects_sense_normalize(edges)
    return edges


import random


def process_qa(qa_data, tokenizer):
    if not qa_data:
        return {}

    use_spatial = random.random() < 0.5

    if use_spatial and qa_data.get("spatial"):
        selected_category = "spatial"
    else:
        other_categories = [cat for cat, items in qa_data.items() if cat != "spatial" and items]
        if other_categories:
            selected_category = random.choice(other_categories)
        else:
            if qa_data.get("spatial"):
                selected_category = "spatial"
            else:
                return {}

    selected_item = random.choice(qa_data[selected_category])
    processed_item = {
        "question": tokenizer(selected_item["question"])[0],
        "answer": selected_item["answer"]
    }
    return processed_item


def build_objects_sense_pipeline(args, is_train, preprocess_img, tokenizer, negs_creator):
    """
    根据 args.objects_sense_format 与 is_train 的值构建相应的数据处理 pipeline 子流程，
    同时 QA 任务作为可选项（由 args.qa 控制）。

    参数：
        args: 参数对象，需包含如下属性：
              - objects_sense_format
              - objects_data
              - batch_size
              - vl_negs
              - qa           # 是否启用 QA 任务
        is_train: 布尔值，标识当前是否处于训练阶段。
        preprocess_img: 图像预处理对象，其 transforms 列表将根据训练或评估阶段进行调整。
        tokenizer: 用于文本处理的 tokenizer 函数或对象。
        negs_creator: 当 args.vl_negs 为 True 时，用于生成负样本的对象。

    返回：
        pipeline_steps: 一个包含数据处理步骤的列表，可通过 pipeline.extend() 添加到主数据处理流程中。
    """
    pipeline_steps = []
    if args.objects_sense_format:
        if is_train:
            preprocess_img.transforms = preprocess_img.transforms[1:]
            pipeline_steps.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                # 同时读取 __key__、图像、文本以及 json 数据
                wds.rename(key="__key__", image="jpg;png;jpeg;webp", text="txt", json="json"),
                (wds.map(lambda sample: {**sample, 'qa': process_qa(sample.get('json', {}).get('QA', {}), tokenizer)})
                 if args.qa else None),
                # 处理 objects_sense 数据
                wds.map(lambda sample: {**sample, 'objects_sense':
                    get_objects_sense(sample['key'], sample['image'], args.objects_sense_format, args.objects_data)}),
                # 对图像和 objects_sense 应用相同的随机裁剪
                wds.map(join_preprocess)
            ])

            pipeline_steps = [step for step in pipeline_steps if step is not None]

            if args.vl_negs:
                steps = [
                    wds.map(lambda sample: {**sample, 'negatives': negs_creator.create_negs(sample)}),
                    wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0],
                                 negatives=lambda negatives: tokenizer(negatives))
                ]
                # 根据是否启用 QA 任务选择输出 tuple 的字段
                if args.qa:
                    steps.append(wds.to_tuple("image", "text", "objects_sense", "qa", "negatives"))
                else:
                    steps.append(wds.to_tuple("image", "text", "objects_sense", "negatives"))
                steps.append(wds.batched(args.batch_size, partial=not is_train))
                pipeline_steps.extend(steps)
            else:
                steps = [wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0])]
                if args.qa:
                    steps.append(wds.to_tuple("image", "text", "objects_sense", "qa"))
                else:
                    steps.append(wds.to_tuple("image", "text", "objects_sense"))
                steps.append(wds.batched(args.batch_size, partial=not is_train))
                pipeline_steps.extend(steps)
        else:
            preprocess_objects_val = copy.deepcopy(preprocess_img)
            preprocess_objects_val.transforms = preprocess_objects_val.transforms[:2]
            pipeline_steps.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(key="__key__", image="jpg;png;jpeg;webp", text="txt"),
                wds.map(lambda sample: {**sample, 'objects_sense':
                    get_objects_sense(sample['key'], sample['image'], args.objects_sense_format, args.objects_data)}),
                wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0],
                             objects_sense=preprocess_objects_val),
                wds.to_tuple("image", "text", "objects_sense"),
                wds.batched(args.batch_size, partial=not is_train)
            ])
    else:
        pipeline_steps.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train)
        ])
    return pipeline_steps


"""Pending further modification"""
"""Pending further modification"""
"""Pending further modification"""

def build_mutil_objects_sense_pipeline(args, is_train, preprocess_img, tokenizer, negs_creator, input_shards):
    """
    构建 multi_wds_dataset 中的 objects_sense 数据处理 pipeline 子流程。
    如果 args.objects_sense_format 被设置，则构建 objects_sense 相关的处理流程；
    否则构建普通的图像-文本处理流程。

    参数：
        args: 参数对象，需包含下列属性：
              - objects_sense_format
              - objects_data
              - objects_add_data
              - batch_size
              - vl_negs
        is_train: 布尔值，表示当前是否为训练阶段。
        preprocess_img: 图像预处理对象，其 transforms 列表会根据训练或评估阶段进行调整。
        tokenizer: 文本分词器，用于处理文本。
        negs_creator: 当 args.vl_negs 为 True 时，用于生成负样本的对象。
        input_shards: 当前数据源，用于判断是否包含 "CLEVR"，从而选择合适的 objects_data。

    返回：
        pipeline_steps: 包含处理步骤的列表，可用于扩展主 pipeline。
    """
    pipeline_steps = []
    if args.objects_sense_format:
        # 根据当前数据源决定 objects_data 的使用
        objects_data = args.objects_data
        if "CLEVR" in input_shards:
            objects_data = args.objects_add_data

        if is_train:
            # 训练阶段：移除 preprocess_img.transforms 第一项
            preprocess_img.transforms = preprocess_img.transforms[1:]
            pipeline_steps.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(key="__key__", image="jpg;png;jpeg;webp", text="txt"),
                wds.map(lambda sample, objects_data=objects_data: {
                    **sample,
                    'objects_sense': get_objects_sense(
                        sample['key'],
                        sample['image'],
                        args.objects_sense_format,
                        objects_data
                    )
                }),
                wds.map(join_preprocess)
            ])

            if args.vl_negs:
                pipeline_steps.extend([
                    wds.map(lambda sample: {
                        **sample,
                        'negatives': negs_creator.create_negs(sample)
                    }),
                    wds.map_dict(
                        image=preprocess_img,
                        text=lambda text: tokenizer(text)[0],
                        negatives=lambda negatives: tokenizer(negatives)
                    ),
                    wds.to_tuple("image", "text", "objects_sense", "negatives"),
                    wds.batched(args.batch_size, partial=not is_train)
                ])
            else:
                pipeline_steps.extend([
                    wds.map_dict(
                        image=preprocess_img,
                        text=lambda text: tokenizer(text)[0]
                    ),
                    wds.to_tuple("image", "text", "objects_sense"),
                    wds.batched(args.batch_size, partial=not is_train)
                ])
        else:
            preprocess_objects_val = copy.deepcopy(preprocess_img)
            preprocess_objects_val.transforms = preprocess_objects_val.transforms[:2]
            pipeline_steps.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(key="__key__", image="jpg;png;jpeg;webp", text="txt"),
                wds.map(lambda sample, objects_data=objects_data: {
                    **sample,
                    'objects_sense': get_objects_sense(
                        sample['key'],
                        sample['image'],
                        args.objects_sense_format,
                        objects_data
                    )
                }),
                wds.map_dict(
                    image=preprocess_img,
                    text=lambda text: tokenizer(text)[0],
                    objects_sense=preprocess_objects_val
                ),
                wds.to_tuple("image", "text", "objects_sense"),
                wds.batched(args.batch_size, partial=not is_train)
            ])
    else:
        pipeline_steps.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(
                image=preprocess_img,
                text=lambda text: tokenizer(text)[0]
            ),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train)
        ])
    return pipeline_steps


def load_answers(csv_path: str):
    """
    从csv_path读取所有可能答案，返回一个答案列表。
    假设csv里一行一个答案文本。
    """
    answers = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # 假设row[0]就是答案文本
            answers.append(row[0].strip())
    return answers

def build_answer_mapping(answers):
    """
    给定一个字符串列表，生成 answer2idx, idx2answer 的双向映射。
    """
    answer2idx = {ans: i for i, ans in enumerate(answers)}
    idx2answer = {i: ans for i, ans in enumerate(answers)}
    return answer2idx, idx2answer

def get_answer_label(answer2idx, answer_str):
    if isinstance(answer_str, str):
        return answer2idx[answer_str]
    else:

        return [answer2idx[a] for a in answer_str]
