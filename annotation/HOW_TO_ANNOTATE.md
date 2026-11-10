# Human Error Annotation

## Prepare the annotation sheet

After completing the model runs, run the scoring script from the project directory:

```sh
python3 reproduce_all.py
```

Copy `annotation/sheet_blank.csv` to `annotation/final_consensus.csv`. For each row, enter a non-negative integer in each error column. Enter `0` explicitly when there are no errors of that type. A blank cell means the row is still unannotated. Keep the `blind_id`, source text, reference translation, and model output intact.

| Column | What to count |
|---|---|
| `morph` | Errors in affixes, case, grammatical number, possession, or vowel harmony. |
| `syntax` | Errors in sentence structure or grammatical agreement. |
| `lexical` | Incorrect word choices or inappropriate borrowings. |
| `semantic` | Omissions, additions, or distortions of meaning. |

## Annotate independently and resolve disagreements

`key_PRIVATE.csv` maps anonymous IDs to models and identifies rows marked `double=1`. Keep this file private until independent annotation is complete.

Two annotators should label each row marked `double=1` independently in separate copies. They should then review disagreements and record the agreed counts in `final_consensus.csv`. Retain both independent copies for the agreement analysis.

## Calculate annotation statistics

After completing `final_consensus.csv`, run:

```sh
python3 reproduce_all.py
```

The script calculates statistics from the human consensus annotations. It treats blank cells as unannotated. The kappa statistic remains pending until the independent annotations are checked separately.
