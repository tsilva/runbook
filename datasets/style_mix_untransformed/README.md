# Style Mix Untransformed Dataset

This is a small SFT JSONL mix for a minimum viable Unsloth experiment on
`meta-llama/Llama-3.1-8B-Instruct`.

It contains 1,000 untransformed chat examples from `HuggingFaceTB/smoltalk2`
`SFT`:

- 700 rows with `restyle: false` to preserve normal assistant behavior.
- 300 rows with `restyle: true` to rewrite before training.

Keep the `restyle: false` rows unchanged. Rewrite only the assistant message
content in `restyle: true` rows, preserving the user request and the JSONL
shape.

Regenerate with:

```bash
python3 scripts/make_style_mix_dataset.py
```

