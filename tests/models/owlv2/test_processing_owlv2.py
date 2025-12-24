import unittest

from transformers_4573 import Owlv2Processor
from transformers_4573.testing_utils import require_scipy

from ...test_processing_common import ProcessorTesterMixin


@require_scipy
class Owlv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Owlv2Processor
    model_id = "google/owlv2-base-patch16-ensemble"
