# step1b_fetch_comments.py - download comments for the issues we use

import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from settings import DATA_FOLDER

load_dotenv()
HEADERS = {'Authorization': 'Bearer ' + os.environ['GITHUB_TOKEN']}

used = pd.read_parquet(DATA_FOLDER + '/model_a_data.parquet')   # the 7,247 issues

for repo in used['repo'].unique():
    folder = DATA_FOLDER + '/raw/' + repo.replace('/', '__')
    path = folder + '/comments.json'
    if os.path.exists(path):
        print(repo, 'already done')
        continue

    # how many comments each issue has (from the raw issues file)
    n_comments = {}
    with open(folder + '/issues.json', encoding='utf-8') as f:
        for line in f:
            issue = json.loads(line)
            n_comments[issue['number']] = issue['comments']

    numbers = [int(n) for n in used[used['repo'] == repo]['number']]
    comments = []

    for i, number in enumerate(numbers, start=1):
        if n_comments.get(number, 0) == 0:          # no comments -> nothing to fetch
            continue

        url = f'https://api.github.com/repos/{repo}/issues/{number}/comments?per_page=100'
        while url:
            r = requests.get(url, headers=HEADERS)
            if r.status_code in (403, 429):          # too many requests -> wait
                print('Too many requests - waiting 60 seconds')
                time.sleep(60)
                continue
            if r.status_code != 200:                 # deleted issue -> skip
                break
            for comment in r.json():
                comment['issue_number'] = number     # remember which issue it belongs to
                comments.append(comment)
            url = r.links.get('next', {}).get('url')

        print(repo, i, '/', len(numbers), end='\r')

    # save: one comment per line
    with open(path, 'w', encoding='utf-8') as f:
        for comment in comments:
            f.write(json.dumps(comment) + '\n')
    print(repo, 'saved', len(comments), 'comments')