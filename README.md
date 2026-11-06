# RepoBugClassifier: GitHub Issue Triage with Verified Labels

Open-source maintainers read every new issue to find the real bugs among questions, feature requests and user mistakes.
This project builds two classical ML models that help them do it faster:

- **Model A, bug or not?** Classifies a new issue as a bug report or not (feature, docs, question).
- **Model B, will it get fixed?** Ranks bug reports so the ones most likely to lead to a fix come first.

Most of the work went into one question: **can we trust the labels?** The "bug" label is often added automatically by the issue form when the reporter picks "Bug report", and "closed as completed" is just a button. So before training anything, the labels were checked against evidence.

---

## Results

### Model A: bug or not bug (tested once, on data never used for training)

| Model | Future issues (2026), macro-F1 | Unseen projects (pytest, black), macro-F1 |
|---|---|---|
| Keyword rule (baseline) | 0.80 | 0.67 |
| Embeddings + Logistic Regression | 0.85 | 0.82 |
| TF-IDF + Logistic Regression | 0.86 | 0.83 |
| **TF-IDF + embeddings + Logistic Regression** | **0.87** | **0.85** |

Best model, in detail:

| Exam | Accuracy | Bug precision | Bug recall | Non-bug recall | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Future issues (1,453) | 0.90 | 0.95 | 0.92 | 0.85 | 0.87 | 0.95 |
| Unseen projects (581) | 0.85 | 0.84 | 0.90 | 0.78 | 0.85 | 0.91 |

The model was chosen on the validation set (late 2025), not on the test set.
Macro-F1 is the main metric because about 70% of issues are bugs: always answering "bug" already gives 71% accuracy.

### Model B: will this bug report get fixed? (ranking, judged inside each project)

| Project | AUC | Fixed in the model's top 20% | Normal fix rate |
|---|---|---|---|
| electron | 0.71 | 65% | 37% |
| numpy | 0.74 | 76% | 56% |
| pandas | 0.61 | 94% | 79% |
| black (never seen in training) | 0.63 | 63% | 46% |

AUC of 0.5 means random order. A simple rule ("has an error message + has code") scored about 0.5, so the wording of the report matters more than its format.

---

## What the data showed

1. **Template leakage.** Issue forms put text like `BUG:` in the title and headings like `### Describe the bug` in the body. In pandas, the `BUG:` prefix appeared on 1,087 bugs and only 39 non-bugs, so a model could cheat by reading it. Section-aware cleaning keeps only what the reporter wrote: the most common remaining line dropped from up to 99% of issues to 2% or less.
2. **Only about 39% of closed bug reports had a linked fix.** Each closed issue was checked with the GitHub GraphQL API for a merged pull request or commit that closed it.
3. **Linking a fix is a project habit, not a bug signal.** Non-bugs were linked to merged PRs almost as often as bugs (for example 52% vs 59% in pandas, 18% vs 22% in electron).
4. **Maintainers rarely say it in words.** In about 34,600 comments, a clear maintainer confirmation appeared on about 6% of bug reports (9 of 10 correct when checked by hand) and a clear rejection on about 4%. Confirmed bugs were fixed 69% of the time, rejected ones 22%.
5. **Repo effects are real.** Some patterns ("has code") looked strong overall but mostly disappeared inside each project, so all results are reported per project and on unseen projects.
6. **Mislabeled issues.** Issues the model strongly disagreed with were often questions or feature requests filed through the bug form ("How to…", "Support…"), mostly in Electron.

From this evidence, every bug report got a **trust level** (strong bug, strong not-bug, uncertain). Training only on the trusted labels used half the data and caught more non-bugs (89% vs 85% on future issues).

Model B learned that clear, technical reports ("fails", "exception", "should") and reporters who have contributed before are more likely to be fixed, while vague reports ("my application", "unable to") are less likely.

---

## Data

- 13,252 issues from 2024 to 2026 from scikit-learn, pandas, numpy, React, Electron, pytest and black (GitHub REST API)
- 7,247 issues with a clear type after mapping each project's labels and removing bots, spam and untriaged issues
- Time-based split: train (2024 to mid 2025), validation (late 2025), test (2026), plus pytest and black kept only for testing
- Project names are replaced with `PROJECT`, so the model cannot guess from the project

---

## Demo

```
streamlit run demo.py
```

Paste an issue to see whether it is a bug, its chance of being fixed compared with other reports in the same project, and the words that drove the decision.
The demo uses the TF-IDF version by default so it runs on a small laptop. Set `USE_EMBEDDINGS = True` in `step13_save_models.py` to use the full model.

---

## How to run

1. Create a `.env` file with `GITHUB_TOKEN=your_token`
2. `pip install -r requirements.txt`
3. Run the steps in order:

| File | What it does |
|---|---|
| `step1_fetch.py` | download issues |
| `step1b_fetch_comments.py` | download comments |
| `step2_prepare.py` | map labels, check for merged fixes, clean text, split |
| `step1c_fix_nonbugs.py` | check non-bugs for merged fixes |
| `step3_check.py` | check for leftover template text and look at patterns |
| `step6_signals.py` | find maintainer confirmations and rejections in comments |
| `step8_trust.py` | combine the evidence into trust levels |
| `step10_suspicious.py` | list issues that look mislabeled |
| `step11_model_b.py` | Model B (will it get fixed?) |
| `step12_evaluate.py` | all metrics for Model A and B (`FEATURES` = `tfidf`, `embeddings` or `both`) |
| `step13_save_models.py` | train the final models for the demo |
| `demo.py` | Streamlit website |

Embeddings (`all-MiniLM-L6-v2`) were created once on Google Colab and saved to `data/embeddings.parquet`.

---

## Limitations

- Labels are still partly noisy: a merged fix confirms a bug but its absence does not prove the opposite.
- Model B's test set for pytest and scikit-learn is small (55 to 69 bugs), so those numbers vary a lot.
- Closed 2026 issues are biased toward quickly fixed bugs, which raises Model B's fix rates in the test set.
- Comments are only used to build labels, never as model input, since a new issue has no comments yet.

---

`v1_old/` contains the first version of this project (TF-IDF + XGBoost trained directly on labels), kept to show what changed.
