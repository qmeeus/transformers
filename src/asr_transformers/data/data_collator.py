from dataclasses import dataclass
import torch
from typing import Dict, List, Optional, Union

from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase


@dataclass
class DataCollatorForAsr:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
              single sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
            >= 7.5 (Volta).
    """

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[float], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {
            key: [example[key] for example in features] for key in features[0]
        }
        del features
        input_ids = batch.pop("input_ids")
        maxlen = max(map(len, input_ids))
        batch["input_ids"] = pad_sequence(input_ids, batch_first=True)
        
        for key in ("attention_mask", "labels"):
            if key in batch:
                batch[key] = pad_sequence(
                    batch.pop(key),
                    batch_first=True,
                    padding_value=0 if key == "attention_mask" else -100
                )

        return batch
