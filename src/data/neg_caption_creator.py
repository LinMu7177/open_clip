import os
import random
import string
import numpy as np
import spacy
from transformers import pipeline, BertTokenizer, BertModel
import torch
import gensim
import webdataset as wds
from tqdm import tqdm
from typing import List, Dict


class NegativesLLM:
    def __init__(self, args, glove_model_path=None) -> None:
        """初始化NegativesLLM生成器，准备必要的工具和配置"""
        self.classifier = pipeline("fill-mask")
        self.args = args
        self.nlp = spacy.load("en_core_web_sm")
        self.pos_map = {
            "VERB": "verb",
            "NOUN": "noun",
            "ADP": "adp",
            "ADJ": "adj",
            "PROPN": "propn"
        }

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def get_word_embedding(self, word):
        """使用BERT获取词嵌入"""
        input_ids = self.tokenizer.encode(word, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids)
        embeddings = outputs.last_hidden_state
        return embeddings.mean(dim=1).squeeze()  # 获取句子的平均嵌入向量

    def calculate_similarity(self, word1, word2):
        embedding1 = self.get_word_embedding(word1)
        embedding2 = self.get_word_embedding(word2)
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return cos_sim.item()

    def create_negs(self, caption: str) -> List[Dict[str, str]]:
        """生成负例文本"""
        if len(caption) > 512:
            caption = caption[:100]

        clean_caption = self._clean_caption(caption)
        doc = self.nlp(clean_caption)

        neg_texts_list = []
        for pos_cat in self.args.llm_neg_types:
            tokens_of_pos = [token.text for token in doc if token.pos_ == pos_cat]
            if tokens_of_pos:
                neg_text = self._generate_negative_for_pos(clean_caption, tokens_of_pos)
                if neg_text:
                    key_name = f"neg_txt_{self.pos_map.get(pos_cat, pos_cat.lower())}"
                    neg_texts_list.append({key_name: neg_text})

        return neg_texts_list

    def _generate_negative_for_pos(self, clean_caption: str, tokens_of_pos: List[str]) -> str:
        """根据指定的词性生成负例文本"""
        try:
            if not tokens_of_pos:
                return ""

            # 随机选择一个词并替换为 <mask>
            attr_to_change = random.choice(tokens_of_pos)
            pos_indices = np.nonzero([1 if (w == attr_to_change) else 0 for w in clean_caption.split()])[0]

            if len(pos_indices) == 0:
                return ""

            index_to_change = random.choice(pos_indices)
            list_clean_cap = clean_caption.split()
            list_clean_cap[index_to_change] = "<mask>"

            # 使用 fill-mask 模型生成负例
            fill_mask_list = self.classifier(' '.join(list_clean_cap))

            # 过滤掉与原 token 相同的候选词
            filtered_from_gt = [
                item for item in fill_mask_list if item["token_str"].strip().lower() != attr_to_change.lower()
            ]

            # 根据相似度阈值排除过于相似的词
            filtered_from_gt = [
                item for item in filtered_from_gt
                if self.calculate_similarity(item["token_str"].strip().lower(), attr_to_change.lower()) < 0.6
            ]

            if not filtered_from_gt:
                return ""

            # 返回随机的一个负例
            return random.choice(filtered_from_gt)["sequence"]

        except Exception as e:
            print(f"Error generating negative for POS: {e}")
            return ""

    def _clean_caption(self, caption: str) -> str:
        """清理和去除标点符号的文本"""
        return " ".join(caption.translate(str.maketrans('', '', string.punctuation)).split())


class Args:
    def __init__(self):
        self.llm_neg_types = ['VERB', 'NOUN', 'ADP', 'ADJ', 'PROPN']  # 需要生成负例的词性类型
        self.num_negs = 1  # 每个样本生成的负例数


def sample_handler(e, src):
    """异常处理：处理出错的样本"""
    print(f"Warning: skipped sample due to error {e} from {src}")
    return None


def process_tar_file(input_tar_path: str, output_tar_path: str, negatives_generator: NegativesLLM):
    """处理单个tar文件并生成负例"""
    try:
        dataset = wds.WebDataset(input_tar_path, shardshuffle=False, handler=sample_handler)
        with wds.TarWriter(output_tar_path) as writer:
            for sample in tqdm(dataset, desc=f"Processing {input_tar_path}", unit="sample"):
                # 样本中必须包含 "txt" 字段
                if "txt" not in sample:
                    writer.write(sample)
                    continue

                caption_str = sample["txt"]
                if isinstance(caption_str, bytes):
                    caption_str = caption_str.decode("utf-8", errors="ignore")

                # 生成负例文本
                neg_captions = negatives_generator.create_negs(caption_str)

                # 将负例文本加入样本中
                for neg_item in neg_captions:
                    for k, v in neg_item.items():
                        sample[k] = v.encode("utf-8")

                writer.write(sample)
        print(f"Processed {input_tar_path} -> {output_tar_path}")

    except Exception as e:
        print(f"Error processing tar file {input_tar_path}: {e}")


def main():
    """主函数，加载数据，处理文件"""
    # 初始化参数和生成器
    args = Args()
    negatives_generator = NegativesLLM(args)

    # 输入 & 输出目录
    input_dir = "/mnt/shared/data/CC3M/cc3m"
    output_dir = "/mnt/shared/data/CC3M/cc3m_neg"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .tar 文件
    all_tar_files = [f for f in os.listdir(input_dir) if f.endswith(".tar")]

    # 处理每个 tar 文件
    for tarfile_name in all_tar_files:
        input_tar_path = os.path.join(input_dir, tarfile_name)
        output_tar_path = os.path.join(output_dir, tarfile_name)
        process_tar_file(input_tar_path, output_tar_path, negatives_generator)


if __name__ == "__main__":
    main()