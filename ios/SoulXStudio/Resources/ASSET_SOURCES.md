# Bundled avatar sources

The repository owner confirmed the **Indian-man portrait** and the repository's
`examples/girl.png` as the two requested avatars, labeled **Indian** and **Asian**
in the app. The app defaults to Indian.

Both PNGs are copied byte-for-byte from the verified source files.

| Bundle path | Source and provenance | SHA-256 |
| --- | --- | --- |
| `Avatars/indian/portrait.png` | Local MuseTalk `benchmarks/same-avatar/shared.png`; exact source hash matches SoulX `benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/results.json` → `inputs.source`. SoulX `benchmarks/distance_lipsync/make_indian_male_notebook.py` explicitly identifies that shared source as the Indian male portrait. | `cb7f390757c2d1a5e782a5fe0a0660d5d0fc848d54fc1f6eed72e4a3757feac9` |
| `Avatars/girl/portrait.png` | Original SoulX checkout `examples/girl.png`, also the default portrait in its inference scripts and Gradio app. | `384abd202f7dae45fdbbab3b6a7777cd7c8a64625fb19217a45d056f2162c624` |

The Indian asset is the verified **original shared source**, not the unavailable
1.50× zoom fixture whose hash is `a0a2317609afe70ed6c3fa3c620654dc56b123981429b2316dc6f7cc8222d676`.
Any cached reference must be computed from the exact bundled PNG and validated
against its raw source checksum and model profile. Cache metadata also records
the preprocessing and normalized-input checksum used during export.
The existing source asset provenance and licensing continue to apply.
