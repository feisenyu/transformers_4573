from transformers_4573 import PreTrainedConfig


class CustomConfig(PreTrainedConfig):
    model_type = "custom"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)
