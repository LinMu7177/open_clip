from open_clip_train.svlc_learning.color_list import color_list
from open_clip_train.svlc_learning.action_list import action_list
from open_clip_train.svlc_learning.material_list import material_list
from open_clip_train.svlc_learning.size_list import size_list
from open_clip_train.svlc_learning.state_list import state_list

import random
import numpy as np
from open_clip.tokenizer import tokenize
from torch.utils.data import Sampler, Dataset
from transformers import pipeline
import spacy
import string
from torch import distributed as dist

def is_positive(attr, texts):
    positives_in_cap = [c for c in attr if c in texts]
    return len(positives_in_cap) > 0

class BothNegatives(object):
    def __init__(self,args ) -> None:
        self.Negatives = Negatives(args)
        self.NegativesLLM = NegativesLLM(args)
        self.args = args

    def create_negs(self, caption):
        neg_type_curr = random.choice([0,1])
        if neg_type_curr == 0:
            negatives = self.Negatives.create_negs(caption)
        else:
            negatives = self.NegativesLLM.create_negs(caption)
        return negatives

class Negatives(object):
    def __init__(self,args ) -> None:
        self.dict_lists = {'color': color_list, 'action': action_list, 'size': size_list, 'state': state_list,
                      'material': material_list}
        self.args = args

    def create_negs(self,caption):
        negs = len(self.args.vl_neg_type) * [0]
        for ind, neg_type in enumerate(self.args.vl_neg_type):
            negs[ind] = int(
                is_positive(self.dict_lists[neg_type], caption))  # what attributes are in the text
        negs_possible_types = np.nonzero(negs)[0]  # what are the possible attributes types
        selected_type = random.choice(negs_possible_types) if len(negs_possible_types) != 0 else 0
        neg_type = self.args.vl_neg_type[
            selected_type]  # choose random attribute from the possible ones to be the positive type

        attributes = self.dict_lists[neg_type]
        negatives = [''] * self.args.num_negs
        positives_in_cap = [c for c in attributes if c in caption.split()]
        if len(positives_in_cap) > 0:
            neg_attr_text = []
            negative_attributes = list(set(attributes) - set(positives_in_cap))
            for i in range(self.args.num_negs):
                negative_attr = negative_attributes[random.randint(0, len(negative_attributes) - 1)]
                negative_attributes = list(set(negative_attributes) - {negative_attr})
                attr_to_change = positives_in_cap[random.randint(0, len(positives_in_cap) - 1)]
                neg_attr_text.append(caption.replace(attr_to_change, negative_attr))
                negatives = neg_attr_text

        return negatives

class NegativesLLM(object):
    def __init__(self,args ) -> None:
        self.classifier = pipeline("fill-mask",device=-1)
        self.args = args
        self.nlp = spacy.load("en_core_web_sm")

    def create_negs(self,caption):
        if len(caption) > 512:
            caption = caption[:100]
        neg_attr_text = []
        clean_caption = " ".join(caption.translate(str.maketrans('', '', string.punctuation)).split())
        doc = self.nlp(clean_caption)
        # Analyze syntax
        positives_in_cap = [token.text for token in doc if token.pos_ in self.args.llm_neg_types]
        positives_in_cap = list(set(positives_in_cap) - (set(positives_in_cap) - set(caption.split()))) #filter one letter and weird mistakes of spacy
        if len(positives_in_cap) > 0:
            neg_attr_text = []
            for i in range(self.args.num_negs):
                attr_to_change = positives_in_cap[random.randint(0, len(positives_in_cap) - 1)]
                pos_incides = np.nonzero([1 if (w == attr_to_change) else 0 for w in clean_caption.split()])[0]
                index_to_change = random.choice(pos_incides)
                list_clean_cap = clean_caption.split()
                list_clean_cap[index_to_change] = '<mask>'
                fill_mask_list = self.classifier(' '.join(list_clean_cap))
                # try:
                #     pos_incides = np.nonzero([1 if (w == attr_to_change) else 0 for w in clean_caption.split()])[0]
                #     index_to_change = random.choice(pos_incides)
                #     list_clean_cap = clean_caption.split()
                #     list_clean_cap[index_to_change] = '<mask>'
                #     fill_mask_list = self.classifier(' '.join(list_clean_cap))
                # except:
                #     print('fill-mask issue')
                try:
                    filttered_from_GT = [item for item in fill_mask_list if not (item["token_str"].strip(' ') == attr_to_change)]
                    negative_caption = filttered_from_GT[random.randint(0, len(filttered_from_GT) - 1)][
                        "sequence"]#[-1]
                    neg_attr_text.append(negative_caption)
                except:
                    print('post_process negs issue')
        if neg_attr_text != []:
            return neg_attr_text
        else:
            return [''] * self.args.num_negs


class ChunkSample(Sampler):
    def __init__(self, dataset: Dataset,
                 num_replicas: int = None,
                 rank: int = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:

        if num_replicas is None:
            if not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.num_replicas = num_replicas
        self.num_samples = None
        self.dataset = dataset
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        self.all_indices = range(len(self.dataset))
        self.shuffle = shuffle
        self.seed = seed
        self.indices =  self.all_indices[self.rank::self.num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

