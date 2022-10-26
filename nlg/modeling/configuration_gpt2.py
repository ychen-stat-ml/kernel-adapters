""" GPT2 model configuration """
from transformers.models.gpt2 import GPT2Config

class GPT2Config(GPT2Config):
    def __init__(self,
                 train_task_adapters=False,
                 prefix_tuning=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_task_adapters = train_task_adapters
        self.prefix_tuning = prefix_tuning
