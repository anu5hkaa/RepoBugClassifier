# step9_final_test.py - the final exam (run it ONCE, after all choices are made)

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')
data = data.merge(pd.read_parquet(DATA_FOLDER + '/trust.parquet'), on=['repo', 'number'])
NAMES = r'\b(?:electron|pandas|numpy|scikit-learn|sklearn|pytest)\b'
data['masked'] = data['text'].str.replace(NAMES, 'PROJECT', case=False, regex=True)

# learn from ALL the past now (train + val); choices were already made on val
learn = data[data['split'].isin(['train', 'val'])]

# ---------- the 3 contestants ----------
BUG_WORDS = r'\b(?:crash\w*|fail\w*|error|exception|raises?|segfault|wrong|broken|bug|regression)\b'
NONBUG_WORDS = r'\b(?:add|implement|propos\w*|support|improv\w*|docs?|documentation|docstring|typo|feature|useful)\b'


def keyword_rule(texts):
    return [1 if len(re.findall(BUG_WORDS, t.lower())) >= len(re.findall(NONBUG_WORDS, t.lower())) else 0
            for t in texts]


def train_model(rows, label):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(tfidf.fit_transform(rows['masked']), rows[label])
    return lambda texts: model.predict(tfidf.transform(texts))


models = {
    'keyword rule': keyword_rule,
    'B: all labels': train_model(learn, 'is_bug'),
    'C: strong labels': train_model(learn[learn['trust'] != 'uncertain'], 'answer'),
}

# ---------- the exams: never seen before ----------
tests = {
    '1. future (2026)': data[data['split'] == 'test_time'],
    '2. new repos (both)': data[data['split'] == 'test_repo'],
    '   - pytest only': data[data['repo'] == 'pytest-dev/pytest'],
    '   - black only': data[data['repo'] == 'psf/black'],
}

rows = []
for test_name, test in tests.items():
    trusted = test[test['trust'] != 'uncertain']
    for judge, part, label in [('all labels', test, 'is_bug'), ('trusted only', trusted, 'answer')]:
        for model_name, predict in models.items():
            pred = predict(part['masked'])
            rows.append({'test': test_name, 'judged by': judge, 'model': model_name, 'issues': len(part),
                         'macro_F1': f1_score(part[label], pred, average='macro'),
                         'nonbug_recall': recall_score(part[label], pred, pos_label=0, zero_division=0)})

results = pd.DataFrame(rows)
print('MACRO-F1  (main score)')
print(results.pivot_table(index=['test', 'judged by', 'issues'], columns='model', values='macro_F1').round(3))
print('\nNON-BUGS CAUGHT  (how well it spots questions / features / docs)')
print(results.pivot_table(index=['test', 'judged by'], columns='model', values='nonbug_recall').round(3))