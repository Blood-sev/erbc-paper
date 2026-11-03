# Running the experiments

[Overview](README.md) | [Input formats](INPUT_FORMATS.md) | [Implementation map](IMPLEMENTATION.md)

## Environment and checks

Use Python 3.10 or later. The scorer, split utility, archive extraction, and API runner use the standard library. Install the CPU head dependencies before running the full test suite:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python verify_files.py
python prepare_sources.py --output reference_source
```

Numerical Qwen inference additionally uses `requirements-models.txt` and a PyTorch build appropriate for the target CUDA installation. The recorded experiment environment used PyTorch `2.13.0a0+8145d630e8.nv26.06`, Transformers `5.16.1`, and PEFT `0.21.0`. These are the experiment versions, not a cross-platform installation lock.

## Final correction head

This example uses invented evidence states. It exercises three stages and two seeds for all three trainable conditions; it is a software example, not a paper result.

```bash
python train_erbc.py --data example_training.json --config example_training_config.json --output demo_training
python predict_erbc.py --input example_inference.json --checkpoint demo_training/joint_20261020/stage3.safetensors --output demo_predictions.json
python score_predictions.py --predictions demo_predictions.json --labels example_labels.json --output demo_metrics.json
```

For the paper's final-stage experiment, use the 708 fitting units and `training_config.json`. It fixes responsibility-only, bias-only, and joint conditions; seeds 20261020/20261021; three stages of four epochs; batch size 8; learning rates 0.0005/0.0001/0.0001; AdamW with betas 0.9/0.95 and weight decay 0.01; and gradient norm clipping at 1. This gives 356 updates per stage, 1,068 per head, and 6,408 across six heads.

Each stage starts from its own preceding stage with a fresh optimizer. The teacher is a frozen copy of that condition and seed's preceding head. The last stage reduces the two bias-retention coefficients from 0.02 to 0.005. Every fixed final checkpoint is saved; the command does not select a winner.

The training bundle contains paired reviewed-view and predicted-view states. Only the predicted view contributes task-level MAE. Local supervision and reviewed states are not inputs to the prediction command. For the common frozen reference:

```bash
python predict_erbc.py --input evidence_states.json --frozen-reference --output frozen_predictions.json
```

Use `--arm q_only` or `--arm bias_only` with the corresponding checkpoint.

## Numerical Qwen and NTL readout

The numerical runner reads final-position logits for the ten digit tokens, normalizes those ten logits, and takes the expectation over anchors 0/9 through 9/9. Omit `--adapter` for the untuned checkpoint.

```bash
python run_numerical.py infer --input c_tokens.json --model /models/Qwen3-4B-Instruct-2507 --adapter /models/ntl_a --output ntl_a.json
python run_numerical.py infer --input c_tokens.json --model /models/Qwen3-4B-Instruct-2507 --adapter /models/ntl_b --output ntl_b.json
python run_numerical.py ensemble --left ntl_a.json --right ntl_b.json --output ntl_ensemble.json
```

The ensemble requires identical token bindings. In the B-to-C protocol, compute the 50/50 B estimate first, then render both C inputs using that same B estimate. Other baselines use their own B. The input token sequences must contain the complete recorded prompt; the runner performs no truncation or prompt construction.

TRACT first generates a rationale and then scores it. Its generation, training, and scoring functions are in the reference module listed in `IMPLEMENTATION.md`; direct scoring of an ordinary C prompt is not the TRACT procedure.

## API comparisons

`api_config.json` records model IDs, sampling settings, and fixed output instructions. Set the endpoint there and set `ERBC_API_KEY` in the terminal environment. Keep credentials out of configuration files.

Validate input without contacting the endpoint:

```bash
python run_api.py --mode component --input example_api_component.json --config api_config.json --output api_component
python run_api.py --mode pipeline --input example_api_pipeline.json --config api_config.json --output api_pipeline
```

Use prepared evaluation inputs and append `--execute` to send requests. Temperature is 0 and the output budget is 4,096 tokens. Original prompt constants use Unicode escapes to preserve their decoded values.

The component protocol changes only the scalar-output instruction. The pipeline inserts each model's own B result into its C request, preserving the blueprint, operations, and evidence. Completed responses are reused only when their request and input bindings match. Invalid B output prevents the dependent C call. Transport errors, truncated responses, and unexpected model identities stop execution rather than becoming model penalties. Re-running an unchanged command reuses completed responses and retries unfinished requests.

The output `predictions.json` maps model IDs to prediction bundles. Select one model's bundle and pass it to `score_predictions.py` with the matching labels. Labels are loaded only for scoring.

## Split and score

```bash
python split_recipe.py records.jsonl membership.jsonl
python evaluate.py synthetic_scores.json
```

The split precedes knowledge admission. Scoring requires fixed evaluation membership, rejects missing/duplicate records, and assigns the protocol's 100 pp penalty to upstream blocks and completed invalid outputs. Execution failures stay unfinished. Penalized and valid-only metrics are both reported. To score CS, provide its fixed membership and matching records.
