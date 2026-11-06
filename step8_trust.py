

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')
data = data.merge(pd.read_parquet(DATA_FOLDER + '/signals.parquet'), on=['repo', 'number'])


def trust(row):
    if row['is_bug'] == 0:
        return 'strong_not_bug'                                         
    yes = row['fixed'] or row['maint_confirm'] or row['maint_already_fixed']  
    no = row['maint_reject']                                            
    if yes and not no:
        return 'strong_bug'
    if no and not yes:
        return 'strong_not_bug'
    return 'uncertain'                                                    


data['trust'] = data.apply(trust, axis=1)
data['answer'] = (data['trust'] == 'strong_bug').astype(int)              
print('TRUST LEVELS PER REPO')
print(pd.crosstab(data['repo'], data['trust'], margins=True))


NAMES = r'\b(?:electron|pandas|numpy|scikit-learn|sklearn|pytest)\b'
data['masked'] = data['text'].str.replace(NAMES, 'PROJECT', case=False, regex=True)


train = data[data['split'] == 'train']
val = data[data['split'] == 'val']
val_strong = val[val['trust'] != 'uncertain']       


def train_and_test(name, rows, label, text):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(tfidf.fit_transform(rows[text]), rows[label])
    result = {'model': name, 'trained_on': len(rows)}
    for test_name, test, test_label in [('val_all', val, 'is_bug'), ('val_strong', val_strong, 'answer')]:
        pred = model.predict(tfidf.transform(test[text]))
        result[test_name + '_F1'] = round(f1_score(test[test_label], pred, average='macro'), 3)
    return result, tfidf, model


results = []
r, _, _ = train_and_test('A: all labels', train, 'is_bug', 'text')
results.append(r)
r, _, _ = train_and_test('B: all labels, names hidden', train, 'is_bug', 'masked')
results.append(r)
r, tfidf, model = train_and_test('C: strong labels, names hidden', train[train['trust'] != 'uncertain'], 'answer', 'masked')
results.append(r)

print('\nRESULTS (macro-F1)')
print('val_all    = tested on all labels (as before)')
print('val_strong = tested only on issues we trust')
print(pd.DataFrame(results).to_string(index=False))


words = pd.DataFrame({'word': tfidf.get_feature_names_out(), 'weight': model.coef_[0]}).sort_values('weight')
print('\nMODEL C - strongest BUG words')
print(words.tail(15)['word'].tolist())
print('MODEL C - strongest NOT-BUG words')
print(words.head(15)['word'].tolist())

data[['repo', 'number', 'trust', 'answer']].to_parquet(DATA_FOLDER + '/trust.parquet')
print('\nSaved trust.parquet')