"""CPU-only Core ML regression check; no GPU/ANE or performance claim.

Runs on macOS with Core ML Tools installed. The standalone validator records
exact machine/runtime metadata for saved validation reports.
"""
import importlib.util
from pathlib import Path
import platform
import tempfile
import unittest

import numpy as np
import torch

from flashhead_coreml.export_dit import convert_component


@unittest.skipUnless(platform.system() == 'Darwin' and importlib.util.find_spec('coremltools'),
                     'Core ML prediction requires macOS and coremltools')
class CoreMLPrecisionTests(unittest.TestCase):
    def test_attention_stays_float32_and_matches_reference(self):
        import coremltools as ct
        class Attention(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.manual_seed(73)
        torch.set_num_threads(2)
        module = Attention().eval()
        inputs = {name: torch.randn(1, 2, 1024, 64) for name in ('q', 'k', 'v')}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'attention.mlpackage'
            convert_component(module, inputs, 'out', destination, precision='mixed')
            model = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
            operations = [op for function in model.get_spec().mlProgram.functions.values()
                          for block in function.block_specializations.values() for op in block.operations]
            attention = [op for op in operations if op.type == 'scaled_dot_product_attention']
            self.assertEqual(len(attention), 1)
            self.assertEqual(attention[0].outputs[0].type.tensorType.dataType, ct.proto.MIL_pb2.FLOAT32)
            arrays = {name: tensor.numpy().astype(np.float16) for name, tensor in inputs.items()}
            with torch.no_grad():
                expected = module(*(torch.from_numpy(value).float() for value in arrays.values())).numpy()
            actual = model.predict(arrays)['out'].astype(np.float32)
            error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            self.assertLess(error, .002)


if __name__ == '__main__':
    unittest.main()
