# Input formats

Task-level values are fractions in [0, 1]; reported errors are percentage points.

## Frozen evidence state

| Field | Shape | Meaning |
| :--- | :--- | :--- |
| `latent` | N x 32 | Frozen operation features, float32 |
| `responsibility` | N x 3 | Student/shared/AI probabilities, float64 |
| `mass` | N | Nonnegative contribution weights summing to one |
| `prediction` | Scalar | Frozen evidence reference |
| `gate` | Scalar | Retained gate in [-1, 1] |

N is the operation count. Responsibility rows sum to one. `example_inference.json` shows a label-free bundle, including a blocked unit with null state. `expected_ids` must match the records exactly.

## Training

`example_training.json` contains the complete schema. Each row has `id`, `target`, `clean`, `predicted`, and `supervision`. A blocked predicted view is null. Local supervision has an operation-level Boolean `mask` and three-class `responsibility` distributions for each view; null means no local supervision.

Include only prescribed fitting identities. Development, internal-audit, and evaluation identities do not enter the optimizer. The paper configuration expects 708 fitting records; the synthetic example expects four.

## Numerical readout

Each bundle has `expected_ids` and `records`. Each row has `id`, `status`, and `input_ids`. A valid row carries token IDs produced by the checkpoint tokenizer and recorded prompt renderer. A blocked row carries null input IDs. No target is accepted in an inference row.

## API inputs

`example_api_component.json` and `example_api_pipeline.json` contain invented system/user pairs. Each valid row has `id`, `status`, and `C_messages`; a pipeline row also has `B_messages`. Preserve the original numerical prompt before the runner replaces its output instruction.

The C pipeline user message is a JSON object with `task`, `subject`, `question_segments`, `confirmed_blueprint`, `confirmed_operations`, `evidence`, and `evidence_policy`. The runner inserts the numerical B prediction itself.

## Labels and predictions

Keep labels in a separate object with a `records` list of `id` and `target`, as shown in `example_labels.json`. Prediction bundles use `valid`, `blocked`, or `invalid`; valid records also carry `prediction`. Execution errors must not be relabeled as blocked or invalid. `score_predictions.py` joins labels by exact identity before calling the metric implementation.
