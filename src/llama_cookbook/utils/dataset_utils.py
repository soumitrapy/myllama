import torch

from llama_cookbook.datasets.chat_dataset import ChatDataset, chat_collate_fn
from llama_cookbook.utils.cloud_utils import get_cloud_path, LoadFromCloud
import random

def get_preprocessed_dataset(tokenizer, dataset_config):
    chats = LoadFromCloud(dataset_config.data_path)
    if dataset_config.split == "train":
        val_size = dataset_config.val_size
        if type(val_size) == float:
            val_size = int(len(chats)*val_size)
        random.shuffle(chats)
        chats, valchats = chats[val_size:], chats[:val_size]
        dataset = ChatDataset(
            chats=chats,
            tokenizer=tokenizer,
            max_length=dataset_config.max_length,
            chat_type=dataset_config.split
        )
        valds = ChatDataset(
            chats=valchats,
            tokenizer=tokenizer,
            max_length=dataset_config.max_length,
            chat_type=dataset_config.split
        )
        return dataset, valds
    elif dataset_config.split == "test":
        dataset = ChatDataset(
            chats=chats,
            tokenizer=tokenizer,
            #max_length=dataset_config.max_length,
            chat_type=dataset_config.split,
            test_labels = dataset_config.test_labels_available
        )
        return dataset
def get_data_collator(chat_type="train"):
    return lambda batch: chat_collate_fn(batch, chat_type=chat_type)
    