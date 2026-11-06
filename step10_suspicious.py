
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from settings import DATA_FOLDER

data = pd.read_parquet(DATA_FOLDER + '/model_ready.parquet')
data = data.merge(pd.read_parquet(DATA_FOLDER + '/trust.parquet'), on=['repo', 'number'])
NAMES = r'\b(?:electron|pandas|numpy|scikit-learn|sklearn|pytest)\b'
data['masked'] = data['text'].str.replace(NAMES, 'PROJECT', case=False, regex=True)


def new_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, class_weight='balanced'))


strong = data[data['trust'] != 'uncertain'].copy()
uncertain = data[data['trust'] == 'uncertain'].copy()


strong['p_bug'] = cross_val_predict(new_model(), strong['masked'], strong['answer'],
                                    cv=5, method='predict_proba')[:, 1]


model = new_model().fit(strong['masked'], strong['answer'])
uncertain['p_bug'] = model.predict_proba(uncertain['masked'])[:, 1]

nonbugs = strong[strong['is_bug'] == 0]
print('NON-BUG labels the model is sure are bugs (p > 0.8):',
      (nonbugs['p_bug'] > 0.8).sum(), 'of', len(nonbugs))
print('UNCERTAIN bug labels the model is sure are NOT bugs (p < 0.2):',
      (uncertain['p_bug'] < 0.2).sum(), 'of', len(uncertain))


disguised = nonbugs.sort_values('p_bug', ascending=False).head(15).assign(kind='non-bug label, looks like a BUG')
doubtful = uncertain.sort_values('p_bug').head(15).assign(kind='bug label, looks like NOT a bug')
check = pd.concat([disguised, doubtful])[['kind', 'repo', 'type', 'p_bug', 'url', 'clean_title']]
check['p_bug'] = check['p_bug'].round(2)
check['my_check'] = ''                     # you write: wrong / right / unsure
check.to_csv(DATA_FOLDER + '/suspicious.csv', index=False, encoding='utf-8-sig')

for kind, part in check.groupby('kind'):
    print('\n' + kind.upper())
    for _, r in part.head(5).iterrows():
        print(f"  {r['p_bug']:.2f}  [{r['type']}] {r['clean_title'][:70]}\n        {r['url']}")
print('\nAll 30 saved to data/suspicious.csv')