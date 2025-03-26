import ast
from collections import defaultdict
import json
import logging
import math
import os
import random
import numbers
import warnings
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import copy
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

import pickle
import pycocotools.mask as mask_util

from torchvision.transforms import Normalize
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
# from torchvision.transforms.transforms import JointRandomResizedCrop

from open_clip_train.svlc_learning.negs_and_pos import Negatives, NegativesLLM, ChunkSample, BothNegatives

from torch import Tensor
from collections.abc import Sequence
from typing import List, Tuple

import yaml

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class JointRandomResizedCrop(torch.nn.Module):
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
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
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
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
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
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): 包含 "image" 和 "mask" 的字典:
                sample["image"]: PIL.Image 或 Tensor
                sample["objects_sense"]:  PIL.Image 或 Tensor
            其他键值可以自行存放在这个字典里，本函数只会操作 "image" 和 "mask"。

        Returns:
            dict: 返回同一个字典，其中 "image" 和 "mask" 都经过随机裁剪 & resize。
        """
        # 取出图像与mask
        img = sample["image"]
        msk = sample["objects_sense"]

        # 1) 先得到随机裁剪参数
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # 2) 对图像进行随机裁剪+resize
        #   - 双线性/双三次插值可以使用 antialias=True
        img = F.resized_crop(
            img, i, j, h, w,
            self.size,
            self.interpolation,
            antialias=self.antialias
        )

        # 3) 对mask进行相同的随机裁剪+resize
        #   - 对mask通常用最近邻插值 (mask_interpolation) 并禁用 antialias
        #   - 避免将分类标签插值为非整数
        msk = F.resized_crop(
            msk, i, j, h, w,
            self.size,
            self.mask_interpolation,
            antialias=self.mask_antialias
        )

        # 4) 放回 sample
        sample["image"] = img
        sample["objects_sense"] = msk
        return sample

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        mask_interpolate_str = self.mask_interpolation.value
        format_string = (
            f"{self.__class__.__name__}(size={self.size}, "
            f"scale={tuple(round(s, 4) for s in self.scale)}, "
            f"ratio={tuple(round(r, 4) for r in self.ratio)}, "
            f"interpolation={interpolate_str}, "
            f"antialias={self.antialias}, "
            f"mask_interpolation={mask_interpolate_str}, "
            f"mask_antialias={self.mask_antialias})"
        )
        return format_string


join_preprocess = JointRandomResizedCrop(
    size=(224, 224),
    scale=(0.9, 1.0),
    ratio=(0.75, 1.3333),
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
    mask_interpolation=InterpolationMode.NEAREST,
    mask_antialias=False
)

objects_sense_normalize = Normalize(mean=[0.5], std=[0.26])


def choose_negs_function(args):
    if args.neg_type=='llm':
        return NegativesLLM(args)
    elif args.neg_type=='both':
        return BothNegatives(args)
    else:
        return Negatives(args)


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class JsonlDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, objects_sense_format=None, objects_data=None, is_train=False, num_negs=0):
        logging.debug(f'Loading jsonl data from {input_filename}.')
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line) for line in lines]
        self.dataset_name = os.path.basename(input_filename).split('.')[0]
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        self.objects_sense_format = objects_sense_format
        self.objects_data = objects_data
        self.is_train = is_train
        self.num_negs = num_negs

    def __len__(self):
        return len(self.data)
    
    def get_key(self, image_path):
        if self.dataset_name == 'SpatialSense':
            dirname = os.path.basename(os.path.dirname(image_path))
            filename = os.path.splitext(os.path.basename(image_path))[0]
            key = os.path.join(dirname, filename)
        elif self.dataset_name == 'CLEVR':
            key = os.path.splitext(os.path.basename(image_path))[0]
        return key

    def __getitem__(self, idx):
        image_path = self.data[idx]['img']
        images = Image.open(image_path)
        texts = self.tokenize([self.data[idx]['positive_sample']])[0]

        # Add objects_sense
        if self.objects_sense_format:
            key = self.get_key(image_path)
            objects_sense = get_objects_sense(key, images, self.objects_sense_format, self.objects_data)
            images = self.transforms(images)
            sample = join_preprocess({"image": images, "objects_sense": objects_sense})
            res = [sample["image"], texts, sample["objects_sense"]]
        else:
            res = [self.transforms(images), texts]

        # Add negatives
        if self.is_train and self.num_negs:
            negatives = self.data[idx]['negative_samples']
            if len(negatives) == 0:
                negatives = [""] * self.num_negs
            res.append(self.tokenize(negatives))
        return tuple(res)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if 'fname' not in filesample:
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed=None,
            deterministic=False,
            epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def load_edges(edges_demo_path, image_shape):
    if os.path.exists(edges_demo_path):
        try:
            with open(edges_demo_path, 'rb') as f:
                combined_edges = pickle.load(f)
            rle = {'size': combined_edges['size'], 'counts': combined_edges['counts']}
            mask = mask_util.decode(rle)
        except Exception as e:
            logging.error(f'Error loading edges from {edges_demo_path}: {e}')
            mask = np.ones(image_shape[:2], dtype=np.uint8)
        return mask
    else:
        return np.ones(image_shape[:2], dtype=np.uint8)


def get_objects_sense(key, image, objects_sense_format, objects_data):
    if objects_sense_format == 'edges':
        objects_sense_path = os.path.join(objects_data, key + '_edges.pkl')
        edges = load_edges(objects_sense_path, image.size)

    edges = torch.as_tensor(edges).unsqueeze(0).half() * 255
    edges = objects_sense_normalize(edges)
    return edges


def get_multi_wds_dataset(
    args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, negs_creator=None, num_workers=4
):
    """
    将多个 WebDataset 数据源按指定比例混合，返回一个 DataInfo 对象，
    其中 DataInfo 内含混合之后的 dataloader 以及 shared_epoch。
    """
    with open(args.dataset_info, 'r') as file:
        datasets_info = yaml.safe_load(file)
    
    pipelines, ratios = [], []
    total_num_samples = 0
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    shared_epoch = SharedEpoch(epoch=epoch)

    for name, info in datasets_info.items():
        input_shards = info['train_data'] if is_train else info['val_data']
        if resampled and is_train:
            _pipe = [
                ResampledShards2(
                    input_shards,
                    weights=None,
                    deterministic=True,
                    epoch=shared_epoch
                )
            ]
        else:
            _pipe = [wds.SimpleShardList(input_shards)]

        if is_train:
            if not resampled:
                _pipe.extend([
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ])
            _pipe.extend([
                tarfile_to_samples_nothrow,
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ])
        else:
            _pipe.extend([
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
            ])

        if args.objects_sense_format:
            objects_data = info['objects_data']
            if is_train:
                preprocess_img_train = copy.deepcopy(preprocess_img)
                preprocess_img_train.transforms = preprocess_img.transforms[1:]
                _pipe.extend([
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
                    _pipe.extend([
                        wds.map(lambda sample: {
                            **sample,
                            'negatives': negs_creator.create_negs(sample)
                        }),
                        wds.map_dict(
                            image=preprocess_img_train,
                            text=lambda text: tokenizer(text)[0],
                            negatives=lambda negatives: tokenizer(negatives)
                        ),
                        wds.to_tuple("image", "text", "objects_sense", "negatives"),
                        wds.batched(args.batch_size, partial=not is_train)
                    ])
                else:
                    _pipe.extend([
                        wds.map_dict(
                            image=preprocess_img_train,
                            text=lambda text: tokenizer(text)[0]
                        ),
                        wds.to_tuple("image", "text", "objects_sense"),
                        wds.batched(args.batch_size, partial=not is_train)
                    ])
            else:
                preprocess_objects_val = copy.deepcopy(preprocess_img)
                preprocess_objects_val.transforms = preprocess_objects_val.transforms[:2]
                _pipe.extend([
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
            _pipe.extend([
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

        # update
        pipelines.append(wds.DataPipeline(*_pipe))
        ratios.append(info['ratio'])
        total_num_samples += info['train_num_samples'] if is_train else info['val_num_samples']


    # TODO 选择不同的数据集混合方式
    # merged_pipeline = wds.RandomMix(pipelines, ratios, longest=True)
    merged_pipeline = wds.RoundRobin(pipelines, longest=True)
    # merged_pipeline = wds.ConcatMix(pipelines)


    if is_train:
        # 同样需要算总的 batch 数等，用 total_num_samples
        global_batch_size = args.batch_size * args.world_size
        round_fn = math.floor if floor else math.ceil
        num_batches = round_fn(total_num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        final_num_samples = num_batches * global_batch_size

        merged_pipeline = wds.DataPipeline(merged_pipeline)
        merged_pipeline = merged_pipeline.with_epoch(num_worker_batches)

    else:
        num_batches = math.ceil(total_num_samples / args.batch_size)
        final_num_samples = total_num_samples

    dataloader = wds.WebLoader(
        merged_pipeline,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = final_num_samples
    
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, negs_creator=None,
                    num_workers=4):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    if args.objects_sense_format:
        if is_train:
            preprocess_img.transforms = preprocess_img.transforms[1:]

            pipeline.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(key="__key__",image="jpg;png;jpeg;webp", text="txt"),
                wds.map(lambda sample: {**sample, 'objects_sense':
                    get_objects_sense(sample['key'], sample['image'], args.objects_sense_format, args.objects_data)}),
                # Apply the same random cropping to the image and objects sense
                wds.map(join_preprocess)
            ])

            if args.vl_negs:
                pipeline.extend([
                    wds.map(lambda sample: {**sample, 'negatives': negs_creator.create_negs(sample)}),
                    wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0],
                                 negatives=lambda negatives: tokenizer(negatives)),
                    wds.to_tuple("image", "text", "objects_sense", "negatives"),
                    wds.batched(args.batch_size, partial=not is_train)
                ])
            else:
                pipeline.extend([
                    wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
                    wds.to_tuple("image", "text", "objects_sense"),
                    wds.batched(args.batch_size, partial=not is_train)
                ])
        else:
            preprocess_objects_val = copy.deepcopy(preprocess_img)
            preprocess_objects_val.transforms = preprocess_objects_val.transforms[:2]
            pipeline.extend([
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(key="__key__", image="jpg;png;jpeg;webp", text="txt"),
                wds.map(lambda sample: {**sample, 'objects_sense':
                    get_objects_sense(sample['key'], sample['image'], args.objects_sense_format, args.objects_data)}),
                # Apply the same random cropping to the image and objects sense
                wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0],
                             objects_sense=preprocess_objects_val),
                wds.to_tuple("image", "text", "objects_sense"),
                wds.batched(args.batch_size, partial=not is_train)
            ])
    else:
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train)
        ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_jsonl_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, **kwargs):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonlDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        objects_sense_format=args.objects_sense_format,
        objects_data=args.objects_data,
        is_train=is_train,
        num_negs=args.num_negs
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "multi_webdataset":
        return get_multi_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "jsonl":
        return get_jsonl_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    negs_creator = choose_negs_function(args)

    if args.train_data or args.dataset_type == "synthetic" or args.dataset_type == "multi_webdataset":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer, negs_creator=negs_creator,
            num_workers=args.train_num_workers)

    if args.val_data or args.dataset_type == 'multi_webdataset':
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer, num_workers=args.val_num_workers)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data