# settings.py - all settings in one place

# where data is saved
DATA_FOLDER = 'data'

# repos we use
REPOS = [
    'scikit-learn/scikit-learn',
    'pandas-dev/pandas',
    'numpy/numpy',
    'facebook/react',
    'electron/electron',
    'pytest-dev/pytest',
    'psf/black',
]

# these 2 repos are only for the final test (never for training)
TEST_REPOS = ['pytest-dev/pytest', 'psf/black']

# fetching
START_DATE = '2024-01-01'   # only issues from this date
MAX_ISSUES = 4000           # max issues per repo

# the dates for splitting the data
TRAIN_END = '2025-07-01'    # train = before this date
VAL_END = '2026-01-01'      # val = before this date, test_time = after

# the bug label of each repo
BUG_LABEL = {
    'scikit-learn/scikit-learn': 'Bug',
    'pandas-dev/pandas': 'Bug',
    'numpy/numpy': '00 - Bug',
    'facebook/react': 'Type: Bug',
    'electron/electron': 'bug :beetle:',
    'pytest-dev/pytest': 'type: bug',
    'psf/black': 'T: bug',
}

# "not checked yet" labels -> we don't trust the label
WAITING_LABEL = {
    'scikit-learn/scikit-learn': 'Needs Triage',
    'pandas-dev/pandas': 'Needs Triage',
    'facebook/react': 'Status: Unconfirmed',
}

# which label means which type
TYPE_RULES = {
    'scikit-learn/scikit-learn': {
        'Bug': 'bug', 'New Feature': 'feature', 'Enhancement': 'feature',
        'Documentation': 'docs'},
    'pandas-dev/pandas': {
        'Bug': 'bug', 'Regression': 'bug', 'Enhancement': 'feature', 'Docs': 'docs'},
    'numpy/numpy': {
        '00 - Bug': 'bug', '01 - Enhancement': 'feature',
        '04 - Documentation': 'docs', '33 - Question': 'question'},
    'facebook/react': {
        'Type: Bug': 'bug', 'Type: Feature Request': 'feature',
        'Type: Question': 'question', 'Type: Discussion': 'question'},
    'electron/electron': {
        'bug :beetle:': 'bug', 'enhancement :sparkles:': 'feature'},
    'pytest-dev/pytest': {
        'type: bug': 'bug', 'type: regression': 'bug', 'type: proposal': 'feature',
        'type: enhancement': 'feature', 'type: docs': 'docs', 'type: question': 'question'},
    'psf/black': {
        'T: bug': 'bug', 'C: crash': 'bug', 'T: enhancement': 'feature',
        'T: style': 'feature', 'T: documentation': 'docs'},
}
