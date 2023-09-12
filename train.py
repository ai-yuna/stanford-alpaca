#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# @dataclass 是 Python dataclasses 模块的一个装饰器，它可以自动生成特殊方法，如 __init__ 和 __repr__，使得类的定义更加简洁。
@dataclass
class ModelArguments:
    # 定义一个可选的字符串类型的属性model_name_or_path
    # 默认值为"facebook/opt-125m"
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    # 定义一个字符串类型的属性data_path
    # 默认值为None，metadata用于存储额外的信息，这里提供了一个帮助文本说明这个属性的用途
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 向分词器中添加特殊令牌，并返回新添加的令牌数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 调整模型的词嵌入大小以匹配新的分词器大小
    model.resize_token_embeddings(len(tokenizer))

    # 如果有新的令牌被添加
    if num_new_tokens > 0:
        # 获取模型的输入和输出词嵌入权重
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算除了新添加的令牌外的旧令牌的平均词嵌入
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 将新添加的令牌的词嵌入设置为平均值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # 对每条样本（example, source）使用分词器进行分词
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    # 从分词结果中提取input_ids
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # 计算每个input_ids的实际长度（即非填充部分的长度）
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]  #.item()：这是 PyTorch 张量的方法，它返回张量中的单个 Python 数字。因为上面的求和操作返回的是一个标量张量，所以 .item() 可以用来获取这个数值
    # 返回一个包含input_ids、labels、input_ids_lens和labels_lens的字典
    return dict(
        input_ids=input_ids,#这里是所有样本的input_ids
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:#分词
    """Preprocess the data by tokenizing."""
    # 将源文本和目标文本组合成一个完整的字符串
    examples = [s + t for s, t in zip(sources, targets)]
    # 对组合后的字符串和源文本进行分词  _tokenize_fn
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    # 从组合后的分词结果中提取input_ids
    input_ids = examples_tokenized["input_ids"]
    # 创建labels，并深拷贝input_ids的内容
    labels = copy.deepcopy(input_ids)

    # 使用IGNORE_INDEX值来替换labels中源文本部分的token id
    # 这样做的原因是在训练过程中，我们只关心预测目标文本部分的token
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # 返回数据集分词结果
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # 加载处理返回用于监督微调的数据集

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # 加载数据
        list_data_dict = utils.jload(data_path)

        #开始数据格式化
        logging.warning("Formatting inputs...")
        # 从PROMPT_DICT字典中获取prompt_input和prompt_no_input模板
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # 根据每个样本的内容生成源文本
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ] #format_map() 方法将使用 example 字典中的值来替换 prompt_input 中的占位符。
        # 生成目标文本，并在每个目标文本的末尾添加eos_token
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        # 使用preprocess函数进行处理,获取整个数据集的分词结果dict(input_ids=input_ids, labels=labels)
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 获取数据集的第i个元素
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    # 定义一个属性：tokenizer，这是一个预训练的分词器对象。
    tokenizer: transformers.PreTrainedTokenizer

    # 当你尝试调用一个对象时，Python会执行该对象的 __call__ 方法。
    # 这个方法接受一个参数：instances，这是一个包含多个字典的序列，每个字典代表一个样本。
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从每个样本中提取input_ids和labels，并将它们分别存储在两个列表中。
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # 使用torch的pad_sequence方法对input_ids进行填充，确保每个序列在同一批次中具有相同的长度。
        # batch_first=True表示输出张量的形状为(batch_size, sequence_length)。
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # 同样地，对labels进行填充。此处填充的值为IGNORE_INDEX。
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # 返回一个字典，包含填充后的input_ids、labels和一个计算得到的attention_mask。(一个batch_size数据的分词结果)
        # 注意力掩码用于指示哪些位置是真实的token，哪些位置是填充的。
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # 为监督式微调创建数据集和数据整理器
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # 返回一个包含训练数据集、评估数据集和数据整理器的字典
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


'''
伪代码：
1. 解析参数
2. 加载模型 transformers.AutoModelForCausalLM.from_pretrained()
3. 加载分词器 transformers.AutoTokenizer.from_pretrained()
4. 设置并更新分词器的特殊令牌，并调整模型的词嵌入大小
5. 加载训练数据集train_dataset和data_collator
    train_dataset:
        1. 加载数据，并prompt格式化，得到sources[str]、targets[str]结尾添加eos token
        2. 分词 获得整个数据集的分词结果：dict(input_ids=input_ids, labels=labels) 其中labels中的source部分设置为ignore_index -100 也就是只关注output部分
    data_collator:
        1. 返回batch size的数据集 dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
6. 初始化trainer：Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
7. 训练trainer.train()、保存训练状态trainer.save_state()、保存训练好的模型trainer.save_model(output_dir)
'''
def train():
    # 使用transformers库的HfArgumentParser解析命令行参数
    # ModelArguments, DataArguments, 和 TrainingArguments 是预先定义的数据类，用于存储不同类型的参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    #加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 初始化一个特殊令牌字典，用于存储分词器可能缺少的特殊令牌
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN #"[PAD]"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN #"</s>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN #"<s>"
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN #"<unk>"

    # 更新分词器的特殊令牌，并调整模型的词嵌入大小
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    # 数据 train_dataset data_collator eval_dataset=None
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 初始化trainer
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # 训练
    trainer.train()
    # 保存训练状态
    trainer.save_state()
    # 保存训练好的模型
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
