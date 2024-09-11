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

python run_ner.py \
  --model_name_or_path roberta-base \
  --dataset_name qmeeus/MSNER-nlp \
  --dataset_config_name en \
  --label_column_name tags \
  --text_column_name tokens \
  --label_all_tokens \
  --evaluation_strategy "steps" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 10 \
  --load_best_model_at_end \
  --metric_for_best_model "f1" \
  --greater_is_better "true" \
  --output_dir slue_vp_en \
  --do_train \
  --do_eval
