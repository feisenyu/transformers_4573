from transformers_4573.modeling_utils import AttentionInterface
from transformers_4573.models.llama.modeling_llama import LlamaAttention


def custom_flex(x, **kwargs):
    """Dummy function."""
    return x


ALL_ATTENTION_FUNCTIONS = AttentionInterface()
# This indexing statement and associated function should be exported correctly!
ALL_ATTENTION_FUNCTIONS["flex_attention"] = custom_flex


class GlobalIndexingAttention(LlamaAttention):
    pass
