
FEATURES = 'both'

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')
NAMES = r'\b(?:electron|pandas|numpy|scikit-learn|sklearn|pytest)\b'
data['masked'] = data['text'].str.replace(NAMES, 'PROJECT', case=False, regex=True)
if FEATURES != 'tfidf':
    emb = pd.read_parquet(DATA_FOLDER + '/embeddings.parquet').set_index(['repo', 'number'])


def fit_text(fit_rows):
    tfidf = None
    if FEATURES in ('tfidf', 'both'):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
        tfidf.fit(fit_rows['masked'])

    def to_numbers(rows):
        parts = []
        if tfidf is not None:
            parts.append(tfidf.transform(rows['masked']))
        if FEATURES in ('embeddings', 'both'):
            parts.append(csr_matrix(emb.loc[list(zip(rows['repo'], rows['number']))].values))
        return hstack(parts).tocsr()
    return to_numbers


def new_lr():
    return LogisticRegression(max_iter=3000, class_weight='balanced')


def model_a(fit_rows, test_rows):
    to_numbers = fit_text(fit_rows)
    model = new_lr().fit(to_numbers(fit_rows), fit_rows['is_bug'])
    prob = model.predict_proba(to_numbers(test_rows))[:, 1]     
    pred = (prob >= 0.5).astype(int)
    y = test_rows['is_bug']
    metrics = {
        'issues': len(y),
        'accuracy': accuracy_score(y, pred),
        'bug_precision': precision_score(y, pred),
        'bug_recall': recall_score(y, pred),
        'nonbug_recall': recall_score(y, pred, pos_label=0),
        'macro_F1': f1_score(y, pred, average='macro'),
        'ROC_AUC': roc_auc_score(y, prob),
    }
    matrix = pd.DataFrame(confusion_matrix(y, pred),
                          index=['real: not bug', 'real: bug'], columns=['said: not bug', 'said: bug'])
    return metrics, matrix


past = data[data['split'].isin(['train', 'val'])]
exams = {
    'val (choose models here)': (data[data['split'] == 'train'], data[data['split'] == 'val']),
    'test: future 2026': (past, data[data['split'] == 'test_time']),
    'test: new projects': (past, data[data['split'] == 'test_repo']),
}
table_a = {}
print('=' * 70, f'\nMODEL A  ({FEATURES} + Logistic Regression)')
for name, (fit_rows, test_rows) in exams.items():
    table_a[name], matrix = model_a(fit_rows, test_rows)
    print(f'\nConfusion matrix - {name}')
    print(matrix)
table_a = pd.DataFrame(table_a).T.round(3)
print('\nMODEL A METRICS')
print(table_a)

bugs = data[(data['is_bug'] == 1) & data['group'].isin(['true_bug', 'unclear', 'not_fixed'])
            & ~data['author_role'].isin(['OWNER', 'MEMBER', 'COLLABORATOR'])].copy()
bugs['fixed'] = (bugs['group'] == 'true_bug').astype(int)
TRAIN_REPOS = sorted(bugs.loc[~bugs['is_test_repo'], 'repo'].unique())


def extra_facts(rows):
    facts = pd.DataFrame({'has_code': rows['has_code'], 'has_error': rows['has_error'],
                          'log_words': np.log1p(rows['n_words']),
                          'contributor': rows['author_role'] == 'CONTRIBUTOR'})
    for repo in TRAIN_REPOS:                          # absorbs each project's fixing habit
        facts[repo] = rows['repo'] == repo
    return csr_matrix(facts.astype(float).values)


def model_b(fit_rows, test_rows):
    to_numbers = fit_text(fit_rows)
    X = lambda rows: hstack([to_numbers(rows), extra_facts(rows)]).tocsr()
    model = new_lr().fit(X(fit_rows), fit_rows['fixed'])
    test_rows = test_rows.copy()
    test_rows['score'] = model.predict_proba(X(test_rows))[:, 1]
    rows = []
    for repo, part in test_rows.groupby('repo'):         
        if part['fixed'].nunique() < 2:
            continue
        top = max(1, int(len(part) * 0.2))
        rows.append({'repo': repo, 'bugs': len(part),
                     'AUC': roc_auc_score(part['fixed'], part['score']),
                     'normal_fix_%': 100 * part['fixed'].mean(),
                     'top20_fix_%': 100 * part.sort_values('score', ascending=False)['fixed'].head(top).mean()})
    return pd.DataFrame(rows).round(2)


bugs_past = bugs[bugs['split'].isin(['train', 'val'])]
b_val = model_b(bugs[bugs['split'] == 'train'], bugs[bugs['split'] == 'val'])
b_test = model_b(bugs_past, bugs[bugs['split'].isin(['test_time', 'test_repo'])])
print('\n' + '=' * 70, f'\nMODEL B  ({FEATURES} + Logistic Regression) - VAL (choose models here)')
print(b_val.to_string(index=False))
print('\nMODEL B - TEST (future 2026 + new projects)')
print(b_test.to_string(index=False))


print('\n' + '=' * 70)
print(f"SUMMARY ({FEATURES}):  Model A val macro-F1 = {table_a.iloc[0]['macro_F1']:.3f}"
      f"   |   Model B val mean AUC = {b_val['AUC'].mean():.3f}")
table_a.to_csv(DATA_FOLDER + f'/results_A_{FEATURES}.csv')
pd.concat([b_val.assign(exam='val'), b_test.assign(exam='test')]).to_csv(
    DATA_FOLDER + f'/results_B_{FEATURES}.csv', index=False)
print(f'Saved results_A_{FEATURES}.csv and results_B_{FEATURES}.csv')