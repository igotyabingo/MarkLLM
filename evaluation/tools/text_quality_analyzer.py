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

# =======================================================
# text_quality_analyzer.py
# Description: Analyze text quality using various metrics
# =======================================================

import math
import torch
from exceptions.exceptions import CodeExecutionError, InvalidAnswerError


class TextQualityAnalyzer:
    """Base class for text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class DirectTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for direct text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass

class PPLCalculator(DirectTextQualityAnalyzer):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
            Initialize the perplexity calculator.

            Parameters:
                model: The language model for perplexity calculation.
                tokenizer: The tokenizer for the language model.
                device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, text: str):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)
        return ppl.item()