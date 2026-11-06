# github_helper.py - asks GitHub "was this issue fixed by a merged pull request?"

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()   # read GITHUB_TOKEN from the .env file
HEADERS = {'Authorization': 'Bearer ' + os.environ['GITHUB_TOKEN']}

# the question we ask GitHub: "what closed this issue, and was the fix merged?"
QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      timelineItems(itemTypes: [CLOSED_EVENT], last: 1) {
        nodes { ... on ClosedEvent { closer { __typename ... on PullRequest { merged } } } }
      }
      closedByPullRequestsReferences(first: 5, includeClosedPrs: true) { nodes { merged } }
    }
  }
}
"""


def was_fixed(repo, number):
    # returns True if a merged pull request (or a commit) fixed this issue
    owner, name = repo.split('/')
    question = {'query': QUERY,
                'variables': {'owner': owner, 'name': name, 'number': int(number)}}

    while True:
        try:
            response = requests.post('https://api.github.com/graphql',
                                     headers=HEADERS, json=question, timeout=30)
            if response.status_code == 200:
                break
            print('GitHub said', response.status_code, '- waiting 60 seconds...')
        except requests.RequestException:
            print('Network problem, waiting 60 seconds...')
        time.sleep(60)

    data = response.json().get('data') or {}
    issue = (data.get('repository') or {}).get('issue')
    if issue is None:
        return False

    # proof 1: the issue was closed by a merged PR or by a commit
    closed_by_code = False
    events = issue['timelineItems']['nodes']
    if events and events[0].get('closer'):
        closer = events[0]['closer']
        if closer['__typename'] == 'Commit' or closer.get('merged') is True:
            closed_by_code = True

    # proof 2: a PR linked as "fixes this issue" was merged
    linked_pr_merged = False
    for pr in issue['closedByPullRequestsReferences']['nodes']:
        if pr['merged']:
            linked_pr_merged = True

    return closed_by_code or linked_pr_merged
