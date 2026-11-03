# Results and evaluation protocols

[Back to overview](README.md) | [Full main-result tables](RESULT_TABLES.md)

## Frozen pipeline check on 162 units

The fixed configuration was evaluated on the separately stored 162-unit collection, including blueprint generation, structured operation extraction, and numerical B-to-C inference. Values below are final C pMAE in percentage points. NTL-a and NTL-b correspond to historical run names NTL and 232; ERBC corresponds to OURS.

| Method | Overall ↓ | CS ↓ |
| :--- | ---: | ---: |
| Qwen3-4B-Instruct | 31.0898 | 41.4688 |
| NTL-a | 17.9194 | 26.6750 |
| NTL-b | 17.9014 | 26.6389 |
| NTL-Ens | 17.8538 | 26.5647 |
| R2CCP | 23.7739 | 30.9647 |
| RAFT | 18.9284 | 26.6956 |
| TRACT | 19.2640 | 28.8216 |
| **ERBC** | **15.5594** | **22.8563** |

Using unrounded results, ERBC reduces error by 2.2943 pp overall and 3.7084 pp on CS relative to NTL-Ens. This check adds evidence from another collection under a frozen configuration; it does not change the historical development use of the 148-unit collection.

## Two interfaces on 148 units

| Setting | NTL-Ens · Overall / CS | ERBC · Overall / CS |
| :--- | ---: | ---: |
| Shared-upstream C component | 17.5520 / 27.2767 | 15.1457 / 23.0870 |
| Retained-operation B-to-C pipeline | 17.1158 / 26.5058 | 14.9537 / 22.7527 |

Both experiments use the same evaluation identities but different C inputs. The pipeline retains common A and operation evidence and injects each method's own numerical B; ERBC uses NTL-Ens B. Differences between rows do not isolate the causal contribution of B.

## Stage-wise accounting and factorial conditions

Under the shared-upstream component protocol, overall pMAE changes from 17.5520 for NTL-Ens to 16.1994 for the retained structured/evidence chain and then to 15.1457 for the final joint head. The respective stage differences are 1.3526 and 1.0537 pp. This accounting does not isolate the effects of individual upstream modules.

The final factorial study has four conditions: frozen reference, responsibility-only correction, bias-only correction, and joint correction. Each trainable condition uses two seeds and its own three-stage training chain. The frozen reference is not trained. The source archive separates factorial model and objective modules; inference-time switches are not presented as retrained ablations.

## Denominators and blocked cases

| Collection | Overall / CS | Valid · Overall / CS | Blocked · Overall / CS |
| :--- | ---: | ---: | ---: |
| 148 units | 148 / 85 | 141 / 80 | 7 / 5 |
| 162 units | 162 / 100 | 154 / 93 | 8 / 7 |

pMAE uses the fixed collection denominator and assigns a 100 pp error to upstream-blocked units. Valid-only metrics exclude these units and must be labeled separately. Execution failures or unfinished inference are not interchangeable with protocol-defined blocked cases.

Resampling intervals on the 148-unit collection are descriptive and do not remove selection effects from adaptive evaluation. Annotation reliability is a separate question: model–reference role agreement is not human inter-annotator agreement.

## API baselines added on November 1, 2026

These models received no task-specific fine-tuning or demonstrations. Values are final C pMAE (Overall / CS). The API additions used fixed settings after the original frozen check; they are not described as part of its earlier archived protocol.

| Protocol | GPT-6 (Sol) | GLM-5.3 |
| :--- | ---: | ---: |
| 148-unit C component | 23.6016 / 29.3707 | 20.7889 / 28.2443 |
| 148-unit B-to-C pipeline | 25.9296 / 33.8382 | 22.5032 / 29.4625 |
| 162-unit B-to-C pipeline | 26.3345 / 29.2722 | 22.1262 / 27.1478 |

Names denote the requested/returned gateway model identifiers, not an independent verification of the backend. Published values follow the manuscript tables.
