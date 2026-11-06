import json
import pandas as pd
from settings import DATA_FOLDER
from github_helper import was_fixed

dfa = pd.read_parquet(DATA_FOLDER + '/model_a_data.parquet')
fix_file = DATA_FOLDER + '/fix_check.json'
fixed = json.load(open(fix_file))

todo = dfa[(dfa['is_bug'] == 0) & (dfa['state'] == 'closed')]
print(len(todo), 'closed non-bugs to check')

for i, (repo, number) in enumerate(zip(todo['repo'], todo['number']), start=1):
    key = repo + '#' + str(number)
    if key in fixed:
        continue
    fixed[key] = was_fixed(repo, number)
    if i % 50 == 0:
        json.dump(fixed, open(fix_file, 'w'))
        print(i, '/', len(todo), 'checked', end='\r')

json.dump(fixed, open(fix_file, 'w'))

todo = todo.copy()
todo['fixed'] = [fixed[r + '#' + str(n)] for r, n in zip(todo['repo'], todo['number'])]
print('\n% of closed non-bugs fixed by code:')
print((todo.groupby('repo')['fixed'].mean() * 100).round())