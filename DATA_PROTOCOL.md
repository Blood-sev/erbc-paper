# Data and annotation protocol

[Back to overview](README.md)

## Collection flow

The study uses materials from StudyChat and AI4EDU. Collection counts are listed in `split_counts.json`.

```text
1,403 annotated units
  └─ exclude 8 overlength units → 1,395
       ├─ 162 separate test units → frozen pipeline check
       ├─ 148 original validation units → reported evaluation collection
       └─ 1,085 original training-pool units
            └─ within-subject student-group split → 927 / 158
                 └─ exclude 2 per side outside the knowledge candidate set
                      → 925 training / 156 development

Original C collections: 925 + 156 + 148 = 1,229
```

The four knowledge-candidate exclusions occur after group splitting. The original 1,085-unit pool is therefore not the final training set. The separate 162 units are not part of the 1,229-unit total.

## Recorded split rule

`split_recipe.py` preserves the historical within-subject student-group rule. It combines subject and student identifier into a group key, orders groups by SHA-256 with the fixed prefix `C_COMMON_20261014_V1|`, and assigns approximately 15% of each subject's groups to development. Rounding and small-group handling are fixed in the script; members of a group remain together.

The rule requires authorized original metadata. It cannot recover true membership without those records. Changing group identifiers changes hash ordering, so arbitrary identifier replacement is not equivalent to reproducing the original split.

Generate a local membership file without overwriting an existing output:

```bash
python split_recipe.py authorized_records.jsonl local_split.jsonl
```

Keep this output in the controlled environment, not in the repository. The script reproduces the grouping step; knowledge-candidate admission still requires the original rule and resources.

## Annotation scope

Linked annotations connect blueprints, operations, KSAs, and evidence references. Responsibility labels distinguish student, shared, and AI contributions. Model-assisted candidates underwent human review. Project records confirm participation by two reviewers; this does not establish independent double coding of every unit.

An independent reliability audit should retain individual submissions, sampling scope, and adjudication records. Unfinished human-agreement analysis is not reported as measured IAA, and model–reference κ is not a substitute.

