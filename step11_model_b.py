
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')

bugs = data[(data['is_bug'] == 1)
            & data['group'].isin(['true_bug', 'unclear', 'not_fixed'])      # closed, not duplicate
            & ~data['author_role'].isin(['OWNER', 'MEMBER', 'COLLABORATOR'])].copy()


bugs['fixed'] = (bugs['group'] == 'true_bug').astype(int)


NAMES = r'\b(?:electron|pandas|numpy|scikit-learn|sklearn|pytest)\b'
bugs['masked'] = bugs['text'].str.replace(NAMES, 'PROJECT', case=False, regex=True)
TRAIN_REPOS = sorted(bugs.loc[~bugs['is_test_repo'], 'repo'].unique())


def extra_facts(rows):
    facts = pd.DataFrame({
        'has_code': rows['has_code'],
        'has_error': rows['has_error'],
        'log_words': np.log1p(rows['n_words']),
        'contributor': rows['author_role'] == 'CONTRIBUTOR',
    })
    for repo in TRAIN_REPOS:                     
        facts[repo] = rows['repo'] == repo
    return csr_matrix(facts.astype(float).values)


learn = bugs[bugs['split'].isin(['train', 'val'])]
test = bugs[bugs['split'].isin(['test_time', 'test_repo'])].copy()
print('learn:', len(learn), 'bugs,', round(100 * learn['fixed'].mean()), '% fixed')
print('test: ', len(test), 'bugs,', round(100 * test['fixed'].mean()), '% fixed\n')

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
X_learn = hstack([tfidf.fit_transform(learn['masked']), extra_facts(learn)])
X_test = hstack([tfidf.transform(test['masked']), extra_facts(test)])
model = LogisticRegression(max_iter=3000, class_weight='balanced').fit(X_learn, learn['fixed'])

test['random'] = np.random.RandomState(0).rand(len(test))           # today: no order
test['simple_rule'] = test['has_error'].astype(int) + test['has_code'].astype(int)
test['model_B'] = model.predict_proba(X_test)[:, 1]


rows = []
for repo, part in test.groupby('repo'):
    row = {'repo': repo, 'bugs': len(part), 'normal_fix_%': round(100 * part['fixed'].mean())}
    top = max(1, int(len(part) * 0.2))                              
    for name in ['random', 'simple_rule', 'model_B']:
        row[name + '_AUC'] = round(roc_auc_score(part['fixed'], part[name]), 2)
        row[name + '_top20_%'] = round(100 * part.sort_values(name, ascending=False)['fixed'].head(top).mean())
    rows.append(row)
results = pd.DataFrame(rows)
print('AUC: 0.5 = random order, higher = fixed bugs ranked higher')
print(results[['repo', 'bugs', 'random_AUC', 'simple_rule_AUC', 'model_B_AUC']].to_string(index=False))
print('\nTOP 20%: % fixed among the top fifth of the list (compare with normal_fix_%)')
print(results[['repo', 'normal_fix_%', 'random_top20_%', 'simple_rule_top20_%', 'model_B_top20_%']].to_string(index=False))


n_words = len(tfidf.get_feature_names_out())
words = pd.DataFrame({'word': tfidf.get_feature_names_out(),
                      'weight': model.coef_[0][:n_words]}).sort_values('weight')
print('\nWords -> MORE likely fixed: ', words.tail(15)['word'].tolist())
print('Words -> LESS likely fixed: ', words.head(15)['word'].tolist())
facts = ['has_code', 'has_error', 'log_words', 'contributor']
weights = model.coef_[0][n_words:n_words + len(facts)]
print('Extra facts (+ helps, - hurts):', {f: round(float(w), 2) for f, w in zip(facts, weights)})