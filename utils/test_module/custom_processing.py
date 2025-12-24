from transformers_4573 import ProcessorMixin


class CustomProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
