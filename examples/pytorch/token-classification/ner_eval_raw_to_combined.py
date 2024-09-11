from pathlib import Path
import sys

_utils = Path("~/.local/share/src/dotfiles/scripts/python/utils").expanduser()
sys.path.append(_utils.as_posix())
from pyutils import *
from ner_metrics import ner_scores

output_dir = sys.argv[1]
# output_dir = "/esat/spchtemp/scratch/qmeeus/repos/huggingface/whisper-for-slu/outputs/whisper-small/voxpopuli"

# Load predictions and labels
indices = {
    p.parent.name: [
        x.split()[0] for x in filter(bool, read_text(p).split("\n"))
    ] for p in Path(output_dir).rglob("**/rhyps")
}


class Loader:

    def __init__(self, fmt="lines"):
        self.fmt = fmt

    def __call__(self, *args, **kwargs):
        if self.fmt == "lines":
            return self.join_labels(self.load_lines(*args, **kwargs))
        elif self.fmt == "tsv":
            return self.load_tsv(*args, **kwargs)

        raise ValueError(fmt)

    def load_json_fmt(self, model_dir):
        predictions = {
            p.parent.name.split("_")[0]: read_json(p)
            for p in Path(model_dir).rglob("**/predictions/*_ws/predictions.jsonl")
        }
        model_dir = model_dir.split("/")[-1]

        predictions = {
            (lang, index): p
            for lang in predictions
            for p, index in zip(predictions[lang], indices[lang])
        }
        return pd.Series(predictions).rename(model_dir)

    def load_lines(self, *model_dirs):
        predictions = pd.concat([
            self.load_json_fmt(model)
            for model in model_dirs
        ], axis=1)
        return self.join_labels(predictions)


    def join_labels(self, predictions):
        labels = {
            p.name.split("-")[1]: read_json(p)
            for p in list(Path("/esat/audioslave/qmeeus/repos/spoken-ner/data").glob("transcript-*-ontonotes-v1.jsonl"))
            + [Path("/esat/audioslave/qmeeus/repos/spoken-ner/data/transcript-en-test-ann.jsonl")]
        }
        labels = pd.Series({
            (lang, x["id"]): x["annotation"]
            for lang in labels for x in labels[lang]
        }).rename("labels")

        return predictions.join(labels)

    def load_tsv(self, model_dir):
        predictions = pd.concat({
            p.parent.name: pd.read_csv(p, sep="\t").set_index("audio_id")
            for p in Path(model_dir).rglob("predictions/**/predictions.tsv")
        }, axis=0)["entities"].rename(model_dir.split("/")[-1]).map(eval)
        labels = pd.concat({
            p.parent.name: pd.read_csv(p, sep="\t").set_index("audio_id")
            for p in Path(model_dir).rglob("predictions/**/targets.tsv")
        }, axis=0)["entities"].rename("labels").map(eval)
        return pd.concat([predictions, labels], axis=1)

# # NLP models predictions
# predictions = Loader("lines")(".backup/roberta-msner", ".backup/slue_vp_en")


# WSLU-L
load = Loader("tsv")
predictions = load("/esat/audioslave/qmeeus/exp/whisper_slu/pipeline/whisper-large-spoken-ner")


# Convert raw to combined labels
predictions = predictions.dropna(how="any")
mapping = read_json("/esat/audioslave/qmeeus/repos/spoken-ner/data/slue_to_spoken-ner_mapping.json")
strip_entity = lambda x: (x.split("-")[1] if "-" in x else x).replace(" ",  "_")
mapping = dict(set(zip(map(strip_entity, mapping.values()), map(strip_entity, mapping.keys()))))
del mapping["O"]


raw_to_combined_tag_map = {
    "DATE": "WHEN",
    "TIME": "WHEN",
    "CARDINAL": "QUANT",
    "ORDINAL": "QUANT",
    "QUANTITY": "QUANT",
    "MONEY": "QUANT",
    "PERCENT": "QUANT",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "NORP": "NORP",
    "ORG": "ORG",
    "LAW": "LAW",
    "PERSON": "PERSON",
    "FAC": "DISCARD",
    "EVENT": "DISCARD",
    "WORK_OF_ART": "DISCARD",
    "PRODUCT": "DISCARD",
    "LANGUAGE": "DISCARD",
}

raw_to_combined_tag_map = {k: raw_to_combined_tag_map[v] for k, v in mapping.items()}

def get_annotations(format="raw"):
    assert format in ["raw", "combined"]

    def get_type(annot):
        if type(annot) is tuple:
            _type = annot[0]
        else:
            _type = annot["type" if "type" in annot else "entity_group"]
        _type = _type.replace(" ", "_")
        if format == "combined":
            _type = raw_to_combined_tag_map[_type]
        return _type

    def get_entity(annot):
        if type(annot) is tuple:
            entity = annot[1].replace(" ", "")
        else:
            entity = "".join(annot["entity"]).replace(" ", "") if "entity" in annot else annot["word"]
        return re.sub("\W", "", entity)

    def no_discard(x):
        return x[0] != "DISCARD"

    def dict_to_tuples(annotation):
        return list(filter(no_discard, [(get_type(annot), get_entity(annot)) for annot in annotation]))

    return dict_to_tuples

dict_to_tuples = get_annotations("combined")
model_names = [col for col in predictions.columns if col != "labels"]
*hyps_combined, refs_combined = (
    predictions[name].map(dict_to_tuples)
    for name in predictions.columns
)

for model, preds in zip(model_names, hyps_combined):
    print(model)
    for lang in refs_combined.index.get_level_values(0).unique():
        print(f"{lang} F1={{:.2%}} label-F1={{:.2%}}".format(*ner_scores(
            predictions=preds.loc[lang],
            references=refs_combined.loc[lang]
        )))

import ipdb; ipdb.set_trace()
