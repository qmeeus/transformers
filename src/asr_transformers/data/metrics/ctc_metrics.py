import numpy as np
import torch
from itertools import groupby
from operator import itemgetter
from editdistance import eval as editdistance
from torch.nn.utils.rnn import pad_sequence


def reduce_scores(scores, reduction="mean"):
    if reduction == "sum":
        return np.sum(scores)
    elif reduction == "mean":
        return np.mean(scores)
    return scores
    

def reduce_ctc_predictions(logits, blank_token_id=1):
    ctc_probs, ctc_ids = torch.softmax(logits.detach().cpu(), dim=-1).max(dim=-1)

    predictions = []
    probabilities = []

    for py, y in zip(ctc_probs, ctc_ids):

        groups = map(
            lambda group: (group[0], max(map(itemgetter(1), group[1]))), 
            groupby(
                zip(y.squeeze(0), py.squeeze(0)), 
                key=lambda x: x[0]
            )
        )

        y_hat, probs_hat = map(torch.tensor, zip(*groups))
        y_idx, = torch.nonzero(y_hat != 1, as_tuple=True)

        predictions.append(y_hat[y_idx].clone())
        probabilities.append(probs_hat[y_idx].clone())

    predictions = pad_sequence(predictions, batch_first=True, padding_value=-100)
    probabilities = pad_sequence(probabilities, batch_first=True, padding_value=1.)
    return predictions, probabilities


def compute_ter(predictions, labels, ignore_id=-100, reduction="mean"):
    device = predictions.device
    predictions = [[yi for yi in y if yi != ignore_id] for y in predictions.tolist()]
    labels = [[yi for yi in y if yi != ignore_id] for y in labels.tolist()]
    return torch.tensor(
        sequence_error_rate(
            predictions, labels, reduction=reduction
        ), device=device
    )


def batch_convert_token_ids_to_string(token_ids, tokenizer, ignore_id=-100):
    token_ids = [[yi for yi in y if yi != ignore_id] for y in token_ids.tolist()]
    sentences = list(map(tokenizer.decode, token_ids))
    return token_ids


def sequence_error_rate(predicted_sequences, true_sequences, reduction):
    return reduce_scores([
        editdistance(y_pred, y_true) / len(y_true) for
        y_pred, y_true in zip(predicted_sequences, true_sequences) 
    ], reduction=reduction)


def compute_wer(predictions, labels, tokenizer, ignore_id=-100, reduction="mean"):
    device = predictions.device

    def get_words(sequences):
        return list(map(str.split, batch_convert_token_ids_to_string(
            sequences, tokenizer, ignore_id=ignore_id
        )))

    predictions, targets = map(get_words, (predictions, labels))
    return torch.tensor(
        sequence_error_rate(
            predictions, targets, reduction=reduction
        ), device=device
    )


def compute_cer(predictions, labels, tokenizer, ignore_id=-100, reduction="mean"):
    device = predictions.device

    def get_chars(sequences):
        return list(map(lambda s: s.replace(" ", ""), batch_convert_token_ids_to_string(
            sequences, tokenizer, ignore_id=ignore_id
        )))

    predictions, targets = map(get_chars, (predictions, labels))
    return torch.tensor(
        sequence_error_rate(
            predictions, targets, reduction=reduction
        ), device=device
    )
