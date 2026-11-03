# Implementation map

| Paper component | Entry or source |
| :--- | :--- |
| Final head and readout | `erbc_head.py`, `predict_erbc.py` |
| Three-stage own-chain training | `train_erbc.py`, `erbc_losses.py`, `training_config.json` |
| Frozen-state and supervision checks | `erbc_io.py` |
| Numerical readout and fixed ensemble | `run_numerical.py` |
| API component and own-B pipeline | `run_api.py`, `api_config.json` |
| Fixed-membership metrics | `evaluate.py`, `score_predictions.py` |
| Student-group split | `split_recipe.py` |

After extraction, reference implementations are grouped under `historical_source`:

| Directory | Contents |
| :--- | :--- |
| `data_preparation` | Linked-record construction and loading |
| `structured_records/runtime` | Upstream routing, gate/extract processing, prompts, retrieval, schema validation |
| `numerical_b/package232` | Numerical B training and required execution helpers |
| `numerical_c/NTL` | NTL objective, input preparation, DDP training |
| `numerical_c/ntl_b` | Second NTL configuration and fusion |
| `numerical_c/R2CCP` | Distributional regression objective and training |
| `numerical_c/RAFT` | Expected-score regression and training |
| `numerical_c/TRACT` | Rationale generation, rationale training, and numerical scoring |
| `current_project/outputs/packages` | Structured readout, transport, evidence features, retained correction layers |

The archive also contains `factorial_training` and `factorial_evaluation`. The root head and loss modules preserve the reference equations, parameter shapes, initialization, and loss coefficients. Source paths and string serialization are normalized for this English release; reference content hashes are in `SOURCE_INDEX.json`.

Runtime helpers remain alongside their callers. GPU allocation, corpus paths, tokenizer paths, frozen-state inputs, and checkpoint paths belong to the experiment's local configuration. Root command interfaces and JSON formats are documented in [Running the experiments](REPRODUCING.md).
