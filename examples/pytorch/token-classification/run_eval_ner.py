import json
import numpy as np
import os
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from transformers import pipeline, HfArgumentParser
from typing import Optional
from ner_metrics import ner_scores


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    model_name: str = field(
        default=None, metadata={"help": "model for prediction"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "where we save the stuff"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split_name: Optional[str] = field(
        default=None, metadata={"help": "The split name of the dataset to use (via the datasets library)."}
    )
    text_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    aggregation_strategy: Optional[str] = field(
        default="first",
        metadata={"help": "Choices: none, simple, first, average, max"}
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    remove_spaces_in_labels: bool = field(
        default=False,
        metadata={"help": "Remove spaces before computing score (for XLM tokenizer)"},
    )
    device: Optional[int] = field(
        default=0, metadata={"help": "0 for cuda, -1 for CPU"}
    )
    no_eval: bool = field(
        default=False, metadata={"help": "Only predict, no evaluate the predictions"}
    )


def get_labels(dataset):
    entities = []
    for example in dataset:
        start, end = 0, 0
        entities.append([])
        for token, tag in zip(example["tokens"], example["tags"]):
            end = start + len(token)
            if tag.startswith("B-"):
                entities[-1].append({"entity_group": tag[2:], "word": token, "start": start, "end": end})
            if tag.startswith("I-"):
                entities[-1][-1]["word"] += token
                entities[-1][-1]["end"] = end
            start = end
    return entities


def main():

    parser = HfArgumentParser((DataTrainingArguments,))
    data_args, = parser.parse_args_into_dataclasses()

    do_eval = not(data_args.no_eval)

    ner = pipeline("ner", model=data_args.model_name, aggregation_strategy=data_args.aggregation_strategy, device=data_args.device)

    if data_args.dataset_name:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.dataset_split_name)

    elif data_args.text_file:
        with open(data_args.text_file) as f:
            dataset = Dataset.from_dict({"text": f.read().split("\n")})

    def data(dataset):
        for example in dataset:
            if type(example) is str:
                yield example
            elif "text" in example:
                yield example["text"]
            elif "tokens" in example:
                yield "".join(example["tokens"])
            else:
                raise ValueError(f"{type(example)}")


    predictions = list(ner(data(dataset)))
    if do_eval:
        labels = get_labels(dataset)


    def format_entity(entity):
        entity_type = entity["entity_group"].replace(" ", "_")
        entity_word = entity["word"].strip()
        if data_args.remove_spaces_in_labels:
            entity_word = entity_word.replace(" ", "")
        return entity_type, entity_word


    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return f"{obj:.6f}"
            return super().default(obj)


    os.makedirs(data_args.output_dir, exist_ok=True)
    filenames = [f"{data_args.output_dir}/predictions.jsonl"]
    iterators = (predictions,)
    if do_eval:
        filenames.append(f"{data_args.output_dir}/labels.jsonl")
        iterators += (labels,)

    outfiles = [open(filename, "w") for filename in filenames]
    for values in zip(*iterators):
        for value, ofile in zip(values, outfiles):
            print(json.dumps(value, cls=JSONEncoder), file=ofile)

    if do_eval:
        decoded_tags = [[format_entity(x) for x in entities] for entities in predictions]
        decoded_labels = [[format_entity(x) for x in entities] for entities in labels]
        metrics = dict(zip(["f1_score", "label_f1"], ner_scores(decoded_tags, decoded_labels)))
        print(metrics)


if __name__ == "__main__":
    main()
