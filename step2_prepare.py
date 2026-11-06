# step2_prepare.py - labels, proof of fix, cleaning, split -> model_ready.parquet

import os
import json
import pandas as pd
from settings import (DATA_FOLDER, REPOS, TEST_REPOS, WAITING_LABEL, TYPE_RULES,
                      TRAIN_END, VAL_END)
from cleaning import clean_title, clean_body


# ---------- 1. load all issues into one table ----------
rows = []
for repo in REPOS:
    path = DATA_FOLDER + '/raw/' + repo.replace('/', '__') + '/issues.json'
    with open(path, encoding='utf-8') as f:
        for line in f:
            issue = json.loads(line)
            labels = []
            for label in issue['labels']:
                labels.append(label['name'])
            rows.append({
                'repo': repo,
                'number': issue['number'],
                'title': issue['title'],
                'body': issue['body'] or '',
                'labels': labels,
                'author_type': issue['user']['type'],           # User or Bot
                'author_role': issue['author_association'],     # NONE, CONTRIBUTOR, MEMBER...
                'created': pd.to_datetime(issue['created_at']),
                'state': issue['state'],                        # open or closed
                'close_reason': issue['state_reason'],          # completed, not_planned, duplicate
                'url': issue['html_url'],
            })

df = pd.DataFrame(rows)
df['is_test_repo'] = df['repo'].isin(TEST_REPOS)
print('All issues:', len(df))


# ---------- 2. give every issue a type ----------
def get_type(row):
    # bots and spam -> removed
    if row['author_type'] == 'Bot' or 'spam / ai' in row['labels']:
        return 'removed'
    # not checked by maintainers yet -> unknown
    if WAITING_LABEL.get(row['repo']) in row['labels']:
        return 'unknown'
    # look up the labels in our rules
    rules = TYPE_RULES[row['repo']]
    types = set()
    for label in row['labels']:
        if label in rules:
            types.add(rules[label])
    if len(types) == 1:
        return types.pop()
    if len(types) == 0:
        return 'unknown'
    return 'mixed'


df['type'] = df.apply(get_type, axis=1)
print(pd.crosstab(df['repo'], df['type'], margins=True))


# ---------- 3. data for Model A (clear type, no react) ----------
keep = df['type'].isin(['bug', 'feature', 'docs', 'question']) & (df['repo'] != 'facebook/react')
dfa = df[keep].copy()
dfa['is_bug'] = (dfa['type'] == 'bug').astype(int)     # 1 = bug, 0 = not bug
print('\nModel A data:', len(dfa), 'issues')


# ---------- 4. was the bug fixed by a pull request? (asks GitHub, saved to a file) ----------
fix_file = DATA_FOLDER + '/fix_check.json'
if os.path.exists(fix_file):
    with open(fix_file) as f:
        fixed = json.load(f)
else:
    fixed = {}

closed_bugs = dfa[(dfa['is_bug'] == 1) & (dfa['state'] == 'closed')]
to_check = []
for repo, number in zip(closed_bugs['repo'], closed_bugs['number']):
    key = repo + '#' + str(number)
    if key not in fixed:
        to_check.append((repo, number))
print(len(closed_bugs), 'closed bugs,', len(to_check), 'still to check on GitHub')

if to_check:
    from github_helper import was_fixed
    for i, (repo, number) in enumerate(to_check, start=1):
        fixed[repo + '#' + str(number)] = was_fixed(repo, number)
        if i % 50 == 0 or i == len(to_check):          # save every 50
            with open(fix_file, 'w') as f:
                json.dump(fixed, f)
            print(i, '/', len(to_check), 'checked', end='\r')
    print()


def get_group(row):
    if row['is_bug'] == 0:
        return 'non-bug'
    if row['state'] == 'open':
        return 'open'
    if fixed.get(row['repo'] + '#' + str(row['number'])):
        return 'true_bug'                 # fixed by a merged PR
    if row['close_reason'] == 'duplicate':
        return 'duplicate'
    if row['close_reason'] == 'not_planned':
        return 'not_fixed'
    return 'unclear'                      # closed as completed, but no fix found


dfa['group'] = dfa.apply(get_group, axis=1)
print(pd.crosstab(dfa['repo'], dfa['group']))


# ---------- 5. clean the text ----------
clean_titles, clean_texts, codes, has_errors = [], [], [], []
for title, body in zip(dfa['title'], dfa['body']):
    text, code, has_error = clean_body(body)
    clean_titles.append(clean_title(title))
    clean_texts.append(text)
    codes.append(code)
    has_errors.append(has_error)

dfa['clean_title'] = clean_titles
dfa['clean_text'] = clean_texts
dfa['code'] = codes
dfa['has_code'] = dfa['code'].str.len() > 0
dfa['has_error'] = has_errors
dfa['n_words'] = dfa['clean_text'].str.split().str.len()
print('\nCleaned', len(dfa), 'issues')


# ---------- 6. final table + split ----------
columns = ['repo', 'number', 'url', 'created', 'author_role',
           'clean_title', 'clean_text', 'code', 'has_code', 'has_error', 'n_words',
           'type', 'is_bug', 'group', 'is_test_repo']
model_df = dfa[columns].copy()
model_df['text'] = model_df['clean_title'] + '. ' + model_df['clean_text']


def get_split(row):
    if row['is_test_repo']:
        return 'test_repo'            # pytest, black
    if row['created'] < pd.Timestamp(TRAIN_END, tz='UTC'):
        return 'train'                # 2024 -> mid 2025
    if row['created'] < pd.Timestamp(VAL_END, tz='UTC'):
        return 'val'                  # second half of 2025
    return 'test_time'                # 2026


model_df['split'] = model_df.apply(get_split, axis=1)
print('\nSplit sizes:')
print(pd.crosstab(model_df['repo'], model_df['split'], margins=True))
print('\n% bugs in each split:')
print((model_df.groupby('split')['is_bug'].mean() * 100).round())


# ---------- 7. save ----------
dfa.to_parquet(DATA_FOLDER + '/model_a_data.parquet')
model_df.to_parquet(DATA_FOLDER + '/model_ready.parquet')
print('\nSaved model_a_data.parquet and model_ready.parquet')
