# step1_fetch.py - download issues from GitHub and save them

import os
import json
import time
import requests
from dotenv import load_dotenv
from settings import DATA_FOLDER, REPOS, START_DATE, MAX_ISSUES

load_dotenv()
HEADERS = {'Authorization': 'Bearer ' + os.environ['GITHUB_TOKEN']}

for repo in REPOS:
    folder = DATA_FOLDER + '/raw/' + repo.replace('/', '__')
    path = folder + '/issues.json'

    # already downloaded? skip it
    if os.path.exists(path):
        print(repo, 'already done')
        continue

    issues = []
    failed = False
    # first page: 100 issues, newest first
    url = f'https://api.github.com/repos/{repo}/issues?state=all&sort=created&direction=desc&per_page=100'

    while url:
        response = requests.get(url, headers=HEADERS)

        # too many requests -> wait and try the same page again
        if response.status_code in (403, 429):
            print('Too many requests - waiting 60 seconds')
            time.sleep(60)
            continue

        # any other problem -> show it and stop
        if response.status_code != 200:
            print('Error', response.status_code, response.text[:200])
            failed = True
            break

        for item in response.json():
            if item['created_at'] < START_DATE:     # older than 2024 -> stop
                url = None
                break
            if 'pull_request' in item:              # pull request, not an issue -> skip
                continue
            issues.append(item)
            if len(issues) == MAX_ISSUES:           # enough -> stop
                url = None
                break

        print(repo, len(issues), 'issues', end='\r')

        # go to the next page (GitHub gives us the link)
        if url:
            url = response.links.get('next', {}).get('url')

    # don't save half a repo -> it will be tried again next time
    if failed:
        print(repo, 'not saved because of the error')
        continue

    # save: one issue per line
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for issue in issues:
            f.write(json.dumps(issue) + '\n')
    print(repo, 'saved', len(issues), 'issues')
