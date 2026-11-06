# step4_train.py - first models for "bug or not bug", measured on val

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')
train = data[data['split'] == 'train']
val = data[data['split'] == 'val']        # test_time and test_repo stay locked for the end
print('train:', len(train), 'issues | val:', len(val), 'issues\n')


def measure(true, predicted, name):
    return {
        'model': name,
        'accuracy': round(accuracy_score(true, predicted), 3),
        'macro_F1': round(f1_score(true, predicted, average='macro'), 3),      # main number
        'bug_recall': round(recall_score(true, predicted, pos_label=1), 3),    # bugs we caught
        'bug_precision': round(precision_score(true, predicted, pos_label=1, zero_division=0), 3),
        'nonbug_recall': round(recall_score(true, predicted, pos_label=0), 3), # non-bugs we caught
    }


results = []

# ---------- baseline 0: always say "bug" ----------
results.append(measure(val['is_bug'], [1] * len(val), 'always bug'))

# ---------- baseline 1: count bug words vs non-bug words ----------
BUG_WORDS = r'\b(?:crash\w*|fail\w*|error|exception|raises?|segfault|wrong|broken|bug|regression)\b'
NONBUG_WORDS = r'\b(?:add|implement|propos\w*|support|improv\w*|docs?|documentation|docstring|typo|feature|useful)\b'


def keyword_rule(text):
    text = text.lower()
    bug_hits = len(re.findall(BUG_WORDS, text))
    nonbug_hits = len(re.findall(NONBUG_WORDS, text))
    if bug_hits >= nonbug_hits:
        return 1
    return 0


results.append(measure(val['is_bug'], val['text'].apply(keyword_rule), 'keyword rule'))

# ---------- model 1: TF-IDF + Logistic Regression ----------
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
X_train = tfidf.fit_transform(train['text'])     # learn words from TRAIN only
X_val = tfidf.transform(val['text'])

model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, train['is_bug'])
predictions = model.predict(X_val)
results.append(measure(val['is_bug'], predictions, 'TF-IDF + LogReg'))

print('RESULTS ON VAL')
print(pd.DataFrame(results).to_string(index=False))

# ---------- results per repo ----------
val = val.copy()
val['prediction'] = predictions
per_repo = []
for repo in val['repo'].unique():
    part = val[val['repo'] == repo]
    row = measure(part['is_bug'], part['prediction'], repo)
    row['issues'] = len(part)
    per_repo.append(row)
print('\nPER REPO')
print(pd.DataFrame(per_repo).to_string(index=False))

# ---------- what did the model learn? ----------
learned = pd.DataFrame({'word': tfidf.get_feature_names_out(), 'weight': model.coef_[0]})
learned = learned.sort_values('weight')
print('\nSTRONGEST "NOT BUG" WORDS')
print(learned.head(20).to_string(index=False))
print('\nSTRONGEST "BUG" WORDS')
print(learned.tail(20).to_string(index=False))
