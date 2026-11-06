# check_comments.py - did we get the comments?

import json
import pandas as pd
from settings import DATA_FOLDER

used = pd.read_parquet(DATA_FOLDER + '/model_a_data.parquet')

for repo in used['repo'].unique():
    folder = DATA_FOLDER + '/raw/' + repo.replace('/', '__')

    expected = {}                                   # issue -> comments GitHub says it has
    for line in open(folder + '/issues.json', encoding='utf-8'):
        issue = json.loads(line)
        expected[issue['number']] = issue['comments']

    got = {}                                        # issue -> comments we saved
    for line in open(folder + '/comments.json', encoding='utf-8'):
        n = json.loads(line)['issue_number']
        got[n] = got.get(n, 0) + 1

    numbers = used[used['repo'] == repo]['number']
    should = sum(expected.get(n, 0) for n in numbers)
    have = sum(got.get(n, 0) for n in numbers)
    print(f'{repo:28} comments saved: {have:6}   expected: {should:6}')