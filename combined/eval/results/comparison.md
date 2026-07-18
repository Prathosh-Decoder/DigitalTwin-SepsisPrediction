# Sepsis model comparison — your model vs Tung vs combined

> **Fairness caveat — read this first.** Tung's model was trained on **all 40,336 PhysioNet patients, including the 4,034 held-out test patients scored here**, so every Tung and combined number in the *in-sample* tables below is optimistic. Your model never saw these patients. There is no leakage-free comparison set on this data (Tung saw every patient) — so the fair reference for Tung is its authors' own out-of-fold score.

## The FAIR comparison (this is the real answer)

Tung trained on our test patients, so its score on this test set is in-sample. The honest reference is Tung's **own authors' out-of-fold** number (from its metadata) vs your held-out number:

| Model | AUROC | AUPRC |
| --- | ---: | ---: |
| Your model (held-out) | 0.8465 | 0.1252 |
| Tung (authors' out-of-fold, honest) | 0.8490 | 0.1260 |

**They are a statistical tie.** The two models are essentially equivalent on honest, leakage-free footing. The much larger Tung numbers in the table below are the *in-sample* scores — the inflation from Tung having memorized these patients, not a real advantage.

## In-sample tables (Tung memorized these patients — read with the caveat)

Evaluated on **156,319 test rows** (2,813 positive).

- **Val-tuned weight** (maximizing validation AUPRC) = `0.00` on your model. It collapsed to **pure Tung** — because Tung is in-sample on the validation patients too, the optimizer discards your model. So *combined-tuned ≈ Tung*.
- **Combined 50/50** = `0.50·your_prob + 0.50·tung_prob` — a genuine blend of both models (this is what the live app's Combined tab uses). Alarm at ensemble probability ≥ `0.1173`.

## Threshold-free ranking (the fair headline)

AUROC/AUPRC don't depend on a threshold — the cleanest 'which model ranks patients better'.

| Model | AUROC | AUPRC |
| --- | ---: | ---: |
| Your model | 0.8465 | 0.1252 |
| Tung (in-sample) | 0.8989 | 0.2341 |
| Combined tuned (in-sample) | 0.8989 | 0.2341 |
| Combined 50/50 (in-sample) | 0.8792 | 0.1756 |

## Precision at matched recall (apples-to-apples precision)

Raw precision at each model's own threshold isn't comparable (different recall), so we fix recall and read precision.

| Model | Precision @ recall 0.67 | Precision @ recall 0.7 |
| --- | ---: | ---: |
| Your model | 0.0799 | 0.0739 |
| Tung (in-sample) | 0.1319 | 0.1205 |
| Combined tuned (in-sample) | 0.1319 | 0.1205 |
| Combined 50/50 (in-sample) | 0.1070 | 0.0968 |

## At each model's native operating point

Your model @ raw≥0.01; Tung @ prob≥0.023; combined @ prob≥0.1173 (F1-tuned on validation).

| Model | Precision | Recall | F1 | Alarm rate | Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Your model | 0.0800 | 0.6694 | 0.1429 | 0.1506 | 0.4554 |
| Tung (in-sample) | 0.0823 | 0.7853 | 0.1490 | 0.1717 | 0.5400 |
| Combined tuned (in-sample) | 0.2719 | 0.3590 | 0.3095 | 0.0238 | 0.3323 |
| Combined 50/50 (in-sample) | 0.2458 | 0.3185 | 0.2775 | 0.0233 | 0.2921 |

## Bottom line

- **Honestly, the two models are tied.** Your held-out AUPRC (~0.125) ≈ Tung's own out-of-fold AUPRC (~0.126); AUROC ~0.847 ≈ ~0.849. Neither model is meaningfully better.
- Tung's apparent lead in the in-sample tables (AUPRC ~0.23) is **leakage**, not skill — it trained on these patients. Its authors' own honest number matches yours.
- The **50/50 blend** is what the app's Combined tab serves; on honest footing you would expect a real ensemble of two comparable models to match or slightly beat either alone.
