
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from settings import DATA_FOLDER

USE_EMBEDDINGS = False      

data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')
if USE_EMBEDDINGS:
    emb = pd.read_parquet(DATA_FOLDER + '/embeddings.parquet').set_index(['repo', 'number'])
NAMES = r'\b(?:electron|pandas|numpy|scikit-learn|sklearn|pytest)\b'
data['masked'] = data['text'].str.replace(NAMES, 'PROJECT', case=False, regex=True)


def vectors(rows):                                   
    if not USE_EMBEDDINGS:
        return csr_matrix((len(rows), 0))
    return csr_matrix(emb.loc[list(zip(rows['repo'], rows['number']))].values)


def new_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)


def new_lr():
    return LogisticRegression(max_iter=3000, class_weight='balanced')


tfidf_a = new_tfidf().fit(data['masked'])
model_a = new_lr().fit(hstack([tfidf_a.transform(data['masked']), vectors(data)]), data['is_bug'])

bugs = data[(data['is_bug'] == 1) & data['group'].isin(['true_bug', 'unclear', 'not_fixed'])
            & ~data['author_role'].isin(['OWNER', 'MEMBER', 'COLLABORATOR'])].copy()
bugs['fixed'] = (bugs['group'] == 'true_bug').astype(int)
REPOS = sorted(bugs['repo'].unique())


def facts(rows):
    f = pd.DataFrame({'has_code': rows['has_code'], 'has_error': rows['has_error'],
                      'log_words': np.log1p(rows['n_words']),
                      'contributor': rows['author_role'] == 'CONTRIBUTOR'})
    for repo in REPOS:
        f[repo] = rows['repo'] == repo
    return csr_matrix(f.astype(float).values)


tfidf_b = new_tfidf().fit(bugs['masked'])
X_b = hstack([tfidf_b.transform(bugs['masked']), vectors(bugs), facts(bugs)])
model_b = new_lr().fit(X_b, bugs['fixed'])

bugs['score'] = model_b.predict_proba(X_b)[:, 1]
levels = {repo: (part['score'].quantile(0.4), part['score'].quantile(0.8)) for repo, part in bugs.groupby('repo')}
levels['other'] = (bugs['score'].quantile(0.4), bugs['score'].quantile(0.8))

os.makedirs('final_models', exist_ok=True)
joblib.dump({'tfidf_a': tfidf_a, 'model_a': model_a, 'tfidf_b': tfidf_b, 'model_b': model_b,
             'repos': REPOS, 'levels': levels, 'names': NAMES,
             'use_embeddings': USE_EMBEDDINGS}, 'final_models/triage.joblib')
print('Saved final_models/triage.joblib  (Model A:', len(data), 'issues, Model B:', len(bugs), 'bugs)')