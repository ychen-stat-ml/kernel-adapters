# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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

from typing import TYPE_CHECKING

from transformers.file_utils import (
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig"],
    "tokenization_roberta": ["RobertaTokenizer"],
}


from types import ModuleType
class _BaseLazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, import_structure):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])

    # Needed for autocompletion in an IDE
    def __dir__(self):
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> any:
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        raise NotImplementedError

if is_tokenizers_available():
    _import_structure["tokenization_roberta_fast"] = ["RobertaTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_roberta"] = [
        "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RobertaForCausalLM",
        "RobertaForMaskedLM",
        "RobertaForMultipleChoice",
        "RobertaForQuestionAnswering",
        "RobertaForSequenceClassification",
        "RobertaForTokenClassification",
        "RobertaModel",
        "RobertaPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_roberta"] = [
        "TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRobertaForMaskedLM",
        "TFRobertaForMultipleChoice",
        "TFRobertaForQuestionAnswering",
        "TFRobertaForSequenceClassification",
        "TFRobertaForTokenClassification",
        "TFRobertaMainLayer",
        "TFRobertaModel",
        "TFRobertaPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_roberta"] = [
        "FlaxRobertaForMaskedLM",
        "FlaxRobertaForMultipleChoice",
        "FlaxRobertaForQuestionAnswering",
        "FlaxRobertaForSequenceClassification",
        "FlaxRobertaForTokenClassification",
        "FlaxRobertaModel",
        "FlaxRobertaPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
    from .tokenization_roberta import RobertaTokenizer

    if is_tokenizers_available():
        from .tokenization_roberta_fast import RobertaTokenizerFast

    if is_torch_available():
        from .modeling_roberta import (
            ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            RobertaForCausalLM,
            RobertaForMaskedLM,
            RobertaForMultipleChoice,
            RobertaForQuestionAnswering,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
            RobertaModel,
            RobertaPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_roberta import (
            TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRobertaForMaskedLM,
            TFRobertaForMultipleChoice,
            TFRobertaForQuestionAnswering,
            TFRobertaForSequenceClassification,
            TFRobertaForTokenClassification,
            TFRobertaMainLayer,
            TFRobertaModel,
            TFRobertaPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_tf_roberta import (
            FlaxRobertaForMaskedLM,
            FlaxRobertaForMultipleChoice,
            FlaxRobertaForQuestionAnswering,
            FlaxRobertaForSequenceClassification,
            FlaxRobertaForTokenClassification,
            FlaxRobertaModel,
            FlaxRobertaPreTrainedModel,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
