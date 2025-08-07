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

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text

class TruncatePromptTextEditor(TextEditor):
    """Truncate the prompt from the text."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the prompt (reference) part from the beginning of the text, preserving formatting."""
        if reference is not None:
            index = text.find(reference)
            if index != -1:
                # reference 다음 부분만 반환
                return text[index + len(reference):]
            else:
                # reference가 text 안에 없으면 원문 그대로 반환
                return text
        else:
            return text