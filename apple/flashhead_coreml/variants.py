"""Named transformer variants for CPU A/B rendering and Core ML export.

A variant name is one or more ``+``-joined parts, applied in order after strict
checkpoint loading and before Apple weight compression:

``baseline``  the original PRO transformer (all 30 blocks, dense FFN)
``fold``      exact cross-attention output reassociation (dit.fold_cross_output)
``ffnNN``     NN% of FFN hidden units removed per block, e.g. ffn25, ffn12.5 (ffn_pruning.py)
``ffnkK``     per layer, the largest fitted removal within K x that layer's INT8 error
``ffnmapL=F,...`` explicit per-layer removal fractions; unlisted layers stay dense

Every transform returns a JSON-serialisable description of what it changed.
"""
from __future__ import annotations


def _fold(runtime_or_blocks):
    blocks = getattr(runtime_or_blocks, 'blocks', runtime_or_blocks)
    blocks.fold_cross_output()
    return {'transform': 'fold_cross_attention_output', 'exact_in_real_arithmetic': True,
            'precision_note': 'probabilities stay FP32; the value and output contractions run in FP16 under the mixed policy'}


def build_transforms(name: str):
    if name == 'baseline':
        return [], 'Original PRO transformer (all 30 blocks, dense 8,960-wide FFN)'
    transforms, descriptions = [], []
    for part in name.split('+'):
        if part == 'fold':
            transforms.append(_fold)
            descriptions.append('cross-attention output projection folded into the 32 audio keys '
                                '(exact in real arithmetic; FP16 contraction order differs)')
        elif part.startswith('ffn'):
            from .ffn_pruning import variant_transform
            extra, text = variant_transform(part)
            transforms.extend(extra)
            descriptions.append(text)
        else:
            raise ValueError(f'Unknown variant part: {part}')
    return transforms, '; '.join(descriptions)
