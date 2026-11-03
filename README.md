# ERBC

**Evidence-guided Responsibility and Bias Correction**  
Supporting material for *ERBC: Evidence-Guided Estimation of AI Responsibility in Educational Interactions*.

[Method](METHOD_GUIDE.md) · [Results](RESULTS.md) · [Data protocol](DATA_PROTOCOL.md) · [Run the experiments](REPRODUCING.md)

## From recorded activity to responsibility estimates

ERBC links task requirements, operation records, and source evidence. Its final estimator combines a frozen evidence state with a responsibility update and a separate scalar correction.

![ERBC architecture](architecture.png)

## Paper materials

The repository includes the final correction head, its three-stage training objective, numerical and API inference interfaces, fixed-membership evaluation, and the recorded group-split rule. Reference modules cover data preparation, operation extraction, baseline training, and the retained evidence chain. Examples connect training, checkpoint loading, prediction, and scoring.

## Quick start

Python 3.10 or later; no third-party packages or GPU required.

```bash
python -m unittest test_evaluate test_split_recipe test_runners.RunnerContracts -v
python evaluate.py synthetic_scores.json
python verify_files.py
```

The example contains invented records. For a complete CPU example of the final head, install `requirements.txt` and follow [training, prediction, and scoring](REPRODUCING.md). Synthetic examples check the software and are not reported experiment results.

## Reported results

Final C pMAE in percentage points (lower is better).

| Protocol | NTL-Ens: Overall / CS | ERBC: Overall / CS |
| :--- | ---: | ---: |
| 148-unit C component | 17.5520 / 27.2767 | **15.1457 / 23.0870** |
| 148-unit B-to-C pipeline | 17.1158 / 26.5058 | **14.9537 / 22.7527** |
| 162-unit frozen pipeline check | 17.8538 / 26.5647 | **15.5594 / 22.8563** |

The 148-unit collection was used during development and evaluation. The separately stored 162-unit collection was evaluated with the frozen configuration. CS is a subset of the overall collection. See [Results](RESULTS.md) for the API baseline additions and protocol details.

## Files

| File | Purpose |
| :--- | :--- |
| [Source snapshot](erbc_source_snapshot.zip) | 125 implementation and runtime modules, indexed in [SOURCE_INDEX.json](SOURCE_INDEX.json) |
| [Paper figures](FIGURES.md) | English rendered figures |
| `train_erbc.py`, `predict_erbc.py` | Train the final head and apply fixed checkpoints |
| `run_numerical.py`, `run_api.py` | Numerical readout and API comparison interfaces |
| `evaluate.py`, `score_predictions.py` | Join labels and score fixed evaluation collections |
| [Input formats](INPUT_FORMATS.md), [implementation map](IMPLEMENTATION.md) | Schemas, commands, and source locations |
| `split_recipe.py` | Reconstruct the recorded pre-admission group split from authorized metadata |
| `test_*.py` | Synthetic tests of metric and splitting behavior |
| `synthetic_scores.json` | Small runnable example with expected scores |
| `split_counts.json` | Aggregate admission and collection counts |
| [Result tables](RESULT_TABLES.md) | Three main comparisons from the manuscript |
| `SHA256SUMS`, `verify_files.py` | Release-file integrity checks |

## Use and attribution

The standalone original Python utilities at the repository root are licensed under MIT; see [LICENSE](LICENSE). The source archive retains existing attribution and license notices. The root utility license does not replace the terms of third-party components or archived source.
