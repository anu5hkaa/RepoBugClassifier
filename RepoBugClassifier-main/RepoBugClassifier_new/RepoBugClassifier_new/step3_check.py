# step3_check.py - are the form cheats gone? what do real bugs look like?

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
dfa = pd.read_parquet(DATA_FOLDER + '/model_a_data.parquet')


# ---------- check 1: are there still repeated template lines? ----------
print('CHECK 1: most common lines in each repo (should be only a few %)')
for repo in dfa['repo'].unique():
    part = dfa[dfa['repo'] == repo]
    line_count = {}
    for text in part['clean_text']:
        lines_in_this_issue = set(line.strip() for line in text.splitlines() if line.strip())
        for line in lines_in_this_issue:
            line_count[line] = line_count.get(line, 0) + 1

    top_lines = sorted(line_count.items(), key=lambda pair: pair[1], reverse=True)[:8]
    print('\n' + repo)
    print(' in %  bug %  line')
    for line, n in top_lines:
        with_line = part[part['clean_text'].str.contains(line, regex=False)]
        print(f"{100 * n / len(part):5.0f}  {100 * with_line['is_bug'].mean():5.0f}  {line[:70]}")


# ---------- check 2: simple features in each group ----------
train = dfa[dfa['is_test_repo'] == False].copy()
text = (train['clean_title'] + ' ' + train['clean_text']).str.lower()

train['f_error'] = train['has_error']
train['f_code'] = train['has_code']
train['f_expected'] = text.str.contains(r'\bexpect')
train['f_not_work'] = text.str.contains(r"\b(?:not|doesn't|does not|don't|can't|cannot|fails?|failed|broken|wrong|crash\w*)\b")
train['f_used_to'] = text.str.contains(r'used to work|worked in|regression|after upgrad|since version')
train['f_version'] = text.str.contains(r'\b\d+\.\d+(?:\.\d+)?\b')
train['f_how_to'] = text.str.contains(r'\bhow (?:do|can|to|should)\b|is (?:it|there) (?:possible|a way)')
train['f_feature'] = text.str.contains(r'would be (?:nice|great|useful)|\bpropose|\badd (?:a|an|support)|\ballow\b|\boption to\b')
train['f_docs'] = text.str.contains(r'\bdocs?\b|documentation|docstring|typo')

features = ['f_error', 'f_code', 'f_expected', 'f_not_work', 'f_used_to',
            'f_version', 'f_how_to', 'f_feature', 'f_docs']
table = train.groupby('group')[features].mean() * 100
table['median_words'] = train.groupby('group')['n_words'].median()
table['count'] = train['group'].value_counts()
print('\nCHECK 2: % of issues with each feature')
print(table.round().reindex(['true_bug', 'unclear', 'not_fixed', 'duplicate', 'open', 'non-bug']))


# ---------- check 3: the same inside each repo ----------
bugs = train[train['group'].isin(['true_bug', 'unclear', 'not_fixed'])]
print('\nCHECK 3: code and error % inside each repo')
print((bugs.groupby(['repo', 'group'])[['f_code', 'f_error']].mean() * 100).round().unstack())


# ---------- check 4: words typical for real bugs vs non-bugs ----------
sure = train[train['group'].isin(['true_bug', 'non-bug'])]
counter = CountVectorizer(binary=True, min_df=20)
X = counter.fit_transform((sure['clean_title'] + ' ' + sure['clean_text']).str.lower())
is_true_bug = (sure['group'] == 'true_bug').values

words = pd.DataFrame({'word': counter.get_feature_names_out()})
words['in_true_bugs_%'] = (X[is_true_bug].mean(axis=0).A1 * 100).round(1)
words['in_non_bugs_%'] = (X[~is_true_bug].mean(axis=0).A1 * 100).round(1)
words['ratio'] = ((words['in_true_bugs_%'] + 1) / (words['in_non_bugs_%'] + 1)).round(2)

print('\nCHECK 4: words typical for REAL BUGS')
print(words.sort_values('ratio', ascending=False).head(20).to_string(index=False))
print('\nwords typical for NON-BUGS')
print(words.sort_values('ratio').head(20).to_string(index=False))
