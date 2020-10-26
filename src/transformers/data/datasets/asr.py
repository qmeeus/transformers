import os
import json
import torch
import time
import pickle
from typing import Dict, List, Optional
from operator import itemgetter
from kaldiio import load_mat
from filelock import FileLock

from torch.utils.data import Dataset
from torch.nn import functional as F

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


class AsrDataset(Dataset):

    def __init__(
        self, 
        filepath, 
        tokenizer: PreTrainedTokenizer,
        longest_first=False,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):

        assert os.path.isfile(filepath), f"Input file path {filepath} not found"

        directory, filename = os.path.split(filepath)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_asr_{}_{}".format(
                tokenizer.__class__.__name__, 
                filename
            )
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    dataset = pickle.load(handle)
                
                self.features = dataset["features"]
                self.labels = dataset["labels"]

                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                with open(filepath, 'rb') as json_file:
                    metadata = json.load(json_file)

                assert metadata, "Empty JSON: {}".format(filepath)
                assert "utts" in metadata, "Malformed JSON: {}".format(filepath)

                if longest_first:
                    sort_key = lambda t: t[1]["input"][0]["shape"][0]
                else:
                    sort_key = itemgetter(0)
                
                keys, metadata = zip(*sorted(metadata["utts"].items(), key=itemgetter(0)))

                self.features = []
                self.labels = []
                for example in metadata:
                    self.features.append(load_mat(example["input"][0]["feat"]))
                    self.labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                        example["output"][0]["text"]
                    )))

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(
                        {"features": self.features, "labels": self.labels}, 
                        handle, 
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
                
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

        self.input_dim = self.features[0].shape[-1]
        self.output_dim = tokenizer.vocab_size
        

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.features[idx], dtype=torch.float)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
