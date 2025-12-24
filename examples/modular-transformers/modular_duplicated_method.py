from transformers_4573.models.llama.configuration_llama import LlamaConfig


class DuplicatedMethodConfig(LlamaConfig):
    @property
    def vocab_size(self):
        return 45

    @vocab_size.setter
    def vocab_size(self, value):
        self.vocab_size = value
