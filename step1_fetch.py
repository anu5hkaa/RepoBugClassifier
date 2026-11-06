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

    if os.path.exists(path):
        print(repo, 'already done')
        continue

    issues = []
    failed = False
    url = f'https://api.github.com/repos/{repo}/issues?state=all&sort=created&direction=desc&per_page=100'

    while url:
        response = requests.get(url, headers=HEADERS)

        if response.status_code in (403, 429):
            print('Too many requests - waiting 60 seconds')
            time.sleep(60)
            continue

        if response.status_code != 200:
            print('Error', response.status_code, response.text[:200])
            failed = True
            break

        for item in response.json():
            if item['created_at'] < START_DATE:
                url = None
                break
            if 'pull_request' in item:
                continue
            issues.append(item)
            if len(issues) == MAX_ISSUES:
                url = None
                break

        print(repo, len(issues), 'issues', end='\r')

        if url:
            url = response.links.get('next', {}).get('url')

    if failed:
        print(repo, 'not saved because of the error')
        continue

    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for issue in issues:
            f.write(json.dumps(issue) + '\n')
    print(repo, 'saved', len(issues), 'issues')