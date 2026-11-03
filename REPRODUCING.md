# Running the code

[Back to overview](README.md)

## Quick checks

Python 3.10+; the root utilities use only the standard library.

```bash
python -m unittest discover -v
python evaluate.py synthetic_scores.json
python verify_files.py
```

The synthetic example returns pMAE 40.0 pp, pRMSE 58.8784 pp, valid-only MAE 10.0 pp, and valid-only RMSE 14.1421 pp. It is a test fixture, not a reported experiment.

## Evaluation input

`evaluate.py` accepts a JSON object with `expected_ids` and `records`. Each expected identity must appear exactly once. Each record has `id`, `status`, and a finite `target` in [0,1]; valid records also have `prediction` in [0,1].

| Status | Treatment |
| :--- | :--- |
| `valid` | Absolute prediction error |
| `blocked` | Protocol-defined upstream block: 100 pp penalty |
| `invalid` | Completed invalid model output: 100 pp penalty |
| `execution_error` | Stop scoring; not a model penalty |

Supply the fixed evaluation membership rather than deriving it from completed outputs. Missing or duplicate identities stop scoring. To evaluate CS, provide its fixed membership and corresponding records separately. The scorer computes pMAE/pRMSE using all expected units, and valid-only metrics using valid predictions.

## Group split

```bash
python split_recipe.py records.jsonl membership.jsonl
```

Input records contain `unit_id`, `metadata.subject`, and `metadata.student_id`. The output file must not already exist. Group hashing is applied before knowledge admission. For the paper's original pool, the split is 927/158 before admission and 925/156 after two exclusions per side.

## Paper implementation modules

Extract `erbc_source_snapshot.zip` and use `SOURCE_INDEX.json` to locate and verify modules. The source paths are preserved so functions can be traced to the experimental implementation. The archive is organized as follows:

| Source group | Paper task |
| :--- | :--- |
| Data preparation and schemas | Build and validate linked records |
| Blueprint and operation processing | Structured upstream A/B evidence |
| Numerical B/C modules | NTL, R2CCP, RAFT, and TRACT |
| ERBC model, feature, and objective modules | Frozen structured/evidence chain and final correction |
| `factorial_training`, `factorial_evaluation` | Correction-condition models and training objectives |

The model modules operate on prepared records, model/checkpoint paths, or frozen features, as specified in their function signatures. Configure these inputs for your environment before invoking them. The root commands above run the standalone utilities; they are not model-training launch commands.

The recorded model runtime uses PyTorch `2.13.0a0+8145d630e8.nv26.06`, Transformers `5.16.1`, and PEFT `0.21.0`, with matching CUDA and tokenizer resources. The root utilities were tested separately without these model dependencies. Results and input-protocol distinctions are documented in [RESULTS.md](RESULTS.md).
