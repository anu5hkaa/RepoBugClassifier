# step6_signals.py - what did maintainers say? (smarter version: sentence by sentence)

import re
import json
import pandas as pd
from settings import DATA_FOLDER

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
dfa = pd.read_parquet(DATA_FOLDER + '/model_a_data.parquet')
fixed = json.load(open(DATA_FOLDER + '/fix_check.json'))

MAINTAINERS = ['OWNER', 'MEMBER', 'COLLABORATOR']

# checked in this order: the first one that matches a sentence wins
ALREADY_FIXED = r"(?:can'?t|cannot|unable to|not able to) reproduce (?:\w+ ){0,3}(?:on|in|with) (?:the )?(?:main|master|latest|dev|nightly)|already (?:been )?fixed"
REJECT = r"\b(?:(?:is|that's|this is) (?:the )?expected behaviou?r|not a bug|work(?:s|ing) as (?:intended|expected|designed)|by design|can'?t reproduce|cannot reproduce|unable to reproduce|not able to reproduce|user error|usage question)\b"
CONFIRM = r"\b(?:i can reproduce|i could reproduce|i was able to reproduce|can confirm|confirmed (?:on|in|with)|good catch|this is a bug|is indeed a bug|looks like a bug|this is a regression)\b"


def sentences(text):
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)          # drop code
    text = re.sub(r'^\s*>.*$', ' ', text, flags=re.M)           # drop quoted text (often the reporter's words)
    return re.split(r'(?<=[.!?])\s+|\n+', text.lower())


# ---------- 1. read maintainer comments, sentence by sentence ----------
found = {}     # (repo, number) -> {'confirm': sentence, 'reject': sentence, 'already_fixed': sentence}
for repo in dfa['repo'].unique():
    path = DATA_FOLDER + '/raw/' + repo.replace('/', '__') + '/comments.json'
    for line in open(path, encoding='utf-8'):
        c = json.loads(line)
        is_bot = (c.get('user') or {}).get('type') == 'Bot'
        if c['author_association'] not in MAINTAINERS or is_bot:
            continue
        key = (repo, c['issue_number'])
        found.setdefault(key, {})
        for s in sentences(c['body'] or ''):
            s = s.strip()
            if not s or s.endswith('?'):                         # skip empty lines and questions
                continue
            if re.search(ALREADY_FIXED, s):
                found[key].setdefault('already_fixed', s)
            elif re.search(REJECT, s):
                found[key].setdefault('reject', s)
            elif re.search(CONFIRM, s):
                found[key].setdefault('confirm', s)

# ---------- 2. the signals ----------
keys = list(zip(dfa['repo'], dfa['number']))
dfa['maint_replied'] = [k in found for k in keys]
for name in ['confirm', 'reject', 'already_fixed']:
    dfa[name + '_sentence'] = [found.get(k, {}).get(name, '') for k in keys]
    dfa['maint_' + name] = dfa[name + '_sentence'] != ''
dfa['fixed'] = [bool(fixed.get(f'{r}#{n}', False)) for r, n in keys]
dfa['label'] = dfa['is_bug'].map({1: 'bug label', 0: 'non-bug label'})
signals = ['maint_replied', 'maint_confirm', 'maint_reject', 'maint_already_fixed', 'fixed']

# ---------- 3. each signal on its own ----------
print('SIGNAL ALONE: % of issues with each signal')
print((dfa.groupby('label')[signals].mean() * 100).round(1))

bugs = dfa[dfa['is_bug'] == 1]
print('\nPER REPO (bug labels only): % with each signal')
print((bugs.groupby('repo')[signals].mean() * 100).round(1))

# ---------- 4. overlap with "fixed" (closed bug labels only) ----------
closed = bugs[bugs['state'] == 'closed']
print('\nOVERLAP: closed bug labels -> how many were fixed?')
print(closed.groupby(['maint_confirm', 'maint_reject'])['fixed'].agg(issues='count', pct_fixed='mean').round(2))

# ---------- 5. check by hand: link + the exact sentence ----------
for name in ['confirm', 'reject', 'already_fixed']:
    part = dfa[dfa['maint_' + name]].sample(frac=1, random_state=1).head(10)
    print(f'\nCHECK BY HAND - {name.upper()} (is the sentence really a {name}?)')
    for _, r in part.iterrows():
        print(f"  [{r['type']}] {r['url']}\n      \"{r[name + '_sentence'][:150]}\"")

# ---------- 6. save ----------
dfa[['repo', 'number'] + signals].to_parquet(DATA_FOLDER + '/signals.parquet')
print('\nSaved signals.parquet')