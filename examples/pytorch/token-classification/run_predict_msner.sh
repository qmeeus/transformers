# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -u
LANG=$1
ipdb run_ner.py \
  --model_name_or_path roberta-msner \
  --dataset_name qmeeus/MSNER-nlp \
  --dataset_config_name $LANG \
  --label_column_name tags \
  --text_column_name tokens \
  --output_dir roberta-msner/predict/$LANG \
  --do_train="false" \
  --do_eval="false" \
  --do_predict
