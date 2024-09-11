"""
Script for computing label and entity F1 score given a reference file and a prediction file
This code was adapted from https://github.com/asappresearch/slue-toolkit

The reference and predcition files are expected to be tab-separated text files with header.
The following columns are expected in the reference file:
    - id: a unique identifier that correspond to an index in the prediction file
    - entities: a list of ground truth entity types and phrases as tuples (e.g. [("date", "Friday 13th")])
    - text (optional): reference text

The following columns are expected in the prediction file:
    - id: a unique identifier that correspond to an index in the reference file
    - ner: a list of predicted entity types and phrases as tuples (e.g. [("date", "Friday 13th")])
    - asr (optional): predicted transcription

The script takes the inner join of the indices of both files. If some indices do not exist, no error is raised.

Example usage:
python evaluate.py --refs references.tsv --hyps predictions.tsv > results.json
"""
import numpy as np
from collections import defaultdict
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from typing import Any, Dict, List, Tuple


NamedEntity = Tuple[str, str, int]
Json = Dict[str, Any]

whisper_norm = BasicTextNormalizer()


def compute_ner_scores(all_gt:List[NamedEntity], all_predictions:[List[NamedEntity]]) -> Json:
    """
    Evalutes per-label and overall (micro and macro) metrics of precision, recall, and fscore

    Input:
        all_gt/all_predictions:
            List of list of tuples: (label, phrase, identifier)
            Each list of tuples correspond to a sentence:
                label: entity tag
                phrase: entity phrase
                tuple_identifier: identifier to differentiate repeating (label, phrase) pairs

    Returns:
        Dictionary of metrics

    Example:
        List of GT (label, phrase) pairs of a sentence: [(GPE, "eu"), (DATE, "today"), (GPE, "eu")]
        all_gt: [(GPE, "eu", 0), (DATE, "today", 0), (GPE, "eu", 1)]
    """
    metrics = {}
    stats = get_ner_stats(all_gt, all_predictions)
    num_correct, num_gt, num_pred = 0, 0, 0
    prec_lst, recall_lst, fscore_lst = [], [], []
    for tag_name, tag_stats in stats.items():
        precision, recall, fscore = get_metrics(
            np.sum(tag_stats["tp"]),
            np.sum(tag_stats["gt_cnt"]),
            np.sum(tag_stats["pred_cnt"]),
        )
        _ = metrics.setdefault(tag_name, {})
        metrics[tag_name]["precision"] = precision
        metrics[tag_name]["recall"] = recall
        metrics[tag_name]["fscore"] = fscore

        num_correct += np.sum(tag_stats["tp"])
        num_pred += np.sum(tag_stats["pred_cnt"])
        num_gt += np.sum(tag_stats["gt_cnt"])

        prec_lst.append(precision)
        recall_lst.append(recall)
        fscore_lst.append(fscore)

    precision, recall, fscore = get_metrics(num_correct, num_gt, num_pred)
    metrics["overall_micro"] = {}
    metrics["overall_micro"]["precision"] = precision
    metrics["overall_micro"]["recall"] = recall
    metrics["overall_micro"]["fscore"] = fscore

    metrics["overall_macro"] = {}
    metrics["overall_macro"]["precision"] = np.mean(prec_lst)
    metrics["overall_macro"]["recall"] = np.mean(recall_lst)
    metrics["overall_macro"]["fscore"] = np.mean(fscore_lst)

    return metrics


def get_ner_stats(all_gt:List[NamedEntity], all_predictions:List[NamedEntity]) -> Json:
    stats = {}
    cnt = 0
    for gt, pred in zip(all_gt, all_predictions):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, entity_info1, entity_info2 in gt:
            entities_true[type_name].add((entity_info1, entity_info2))
        for type_name, entity_info1, entity_info2 in pred:
            entities_pred[type_name].add((entity_info1, entity_info2))
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        for tag_name in target_names:
            _ = stats.setdefault(tag_name, {})
            _ = stats[tag_name].setdefault("tp", [])
            _ = stats[tag_name].setdefault("gt_cnt", [])
            _ = stats[tag_name].setdefault("pred_cnt", [])
            entities_true_type = entities_true.get(tag_name, set())
            entities_pred_type = entities_pred.get(tag_name, set())
            stats[tag_name]["tp"].append(len(entities_true_type & entities_pred_type))
            stats[tag_name]["pred_cnt"].append(len(entities_pred_type))
            stats[tag_name]["gt_cnt"].append(len(entities_true_type))
    return stats


def safe_divide(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(x1, x2, where=x2 != 0)


def get_metrics(num_correct, num_gt, num_pred):
    precision = safe_divide([num_correct], [num_pred])
    recall = safe_divide([num_correct], [num_gt])
    fscore = safe_divide([2 * precision * recall], [(precision + recall)])
    return precision[0], recall[0], fscore[0][0]


def remap_entities_wrapper(normalize=True):

    def convert_entity(entity):
        if normalize:
            entity = whisper_norm(entity).strip()
        return entity

    def convert_tag(tag):
        tag = tag.replace(" ", "_").lower()
        return tag

    def remap_entities(entities):

        def _map(*entities):
            new_entities = []
            for tag, entity in entities:
                i = 0
                tag = convert_tag(tag)
                entity = convert_entity(entity)
                while True:
                    if (tag, entity, i) not in new_entities:
                        new_entities.append((convert_tag(tag), convert_entity(entity), i))
                        break
                    i += 1
            return new_entities

        if type(entities) is str:
            entities = eval(entities)

        if entities is None:
            entities = []

        if type(entities) is not list:
            raise TypeError(f"Unexpected type for entities {type(entities)}")

        return _map(*entities)

    return remap_entities

def f1_score(predictions, references, normalize=True):
    remap_entities = remap_entities_wrapper(normalize=normalize)
    all_gt = list(map(remap_entities, references))
    all_predictions = list(map(remap_entities, predictions))
    return compute_ner_scores(all_gt, all_predictions)["overall_micro"]["fscore"]


def label_f1(predictions, references, normalize=True):
    predictions = [[(tag, "DUMMY") for tag, _ in entities] for entities in predictions]
    references = [[(tag, "DUMMY") for tag, _ in entities] for entities in references]
    return f1_score(predictions, references, normalize=normalize)


def ner_scores(predictions, references, normalize=True):
    args = dict(predictions=predictions, references=references, normalize=normalize)
    return f1_score(**args), label_f1(**args)
