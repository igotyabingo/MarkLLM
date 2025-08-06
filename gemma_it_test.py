# Copyright 2024 THU-BPM MarkLLM.
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

# =================================================================
# assess_detectability.py
# Description: Assess the detectability of a watermarking algorithm
# =================================================================

import gc
import torch
import os
from dotenv import load_dotenv
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Gemma3ForCausalLM
from evaluation.tools.text_editor import TruncatePromptTextEditor, GemmaParaphraser, SynonymSubstitution
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_quality_analyzer import PPLCalculator


import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()

def assess_detectability(algorithm_name, dataset_path, model_path, do_sample):
    my_dataset = C4Dataset(f'dataset/{dataset_path}.jsonl')
    
    load_dotenv()
    access_token = os.getenv("access_token")

    model = Gemma3ForCausalLM.from_pretrained(model_path, token = access_token).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token = access_token)

    transformers_config = TransformersConfig(model=model,
                                             tokenizer=tokenizer,
                                             vocab_size=model.vocab_size,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=100,
                                             do_sample=False,
                                             eos_token_id=None, 
                                             no_repeat_ngram_size=4)

    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)

    # # Watermark-LLM generated -> 500 sample에 대한 z-score, detection boolean
    pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor()],
                                                 show_progress=True, return_type=DetectionPipelineReturnType.FULL) 

    # Human generated -> 500 sample에 대한 z-score, detection boolean
    pipeline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.FULL)

    # UnWatermark-LLM generated 
    pipeline3 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor()], text_source_mode='generated',
                                                show_progress=True, return_type=DetectionPipelineReturnType.FULL)

    result1, result2, result3 = pipeline1.evaluate(my_watermark), pipeline2.evaluate(my_watermark), pipeline3.evaluate(my_watermark)

    # 1. Detectability - (1) FN rate (pipeline1 result중 False의 비율), FP rate (pipeline2 result중 True의 비율) 계산
    bool_list1 = [result.detect_result['is_watermarked'] for result in result1]
    bool_list2 = [result.detect_result['is_watermarked'] for result in result2]

    fn_rate = bool_list1.count(False)/len(bool_list1)
    fp_rate = bool_list2.count(True)/len(bool_list2)

    print(f"FN rate: {fn_rate:.3f}, FP rate: {fp_rate:.3f}")

    # 2. Perplexity
    # result1, result3 각각의 edited text를 이용해 calculate
    calculator = PPLCalculator(model, tokenizer)
    w_ppl_sum = 0
    u_ppl_sum = 0

    for i in range(len(result1)):
        w_ppl_sum += calculator.analyze(result1[i].edited_text)
        u_ppl_sum += calculator.analyze(result3[i].edited_text)
    
    w_ppl_avg = w_ppl_sum / len(result1)
    u_ppl_avg = u_ppl_sum / len(result1)
    
    print(f"Average PPL for watermarked text: {w_ppl_avg:.3f}") #, Average PPL for unwatermarked text: {u_ppl_avg:.3f}")

if __name__ == '__main__':
    import argparse
    # argument: algorithm, dataset, model, sample
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument('--dataset', type=str, default='c4_realnews')
    parser.add_argument('--model', type=str, default='google/gemma-3-1b-it')
    parser.add_argument('--sample', type=bool, default=False)
    args = parser.parse_args()

    assess_detectability(args.algorithm, args.dataset, args.model, args.sample)