# cleaning.py - keep what the person wrote, remove what the form wrote

import re

# ---------- 1. title ----------
# words that forms put at the start of titles, like "BUG: ..." or "ENH: ..."
PREFIXES = ['BUG', 'ENH', 'DOC', 'DOCS', 'PERF', 'TYP', 'API', 'BLD', 'CI', 'TST',
            'MAINT', 'QST', 'REGR', 'FEAT', 'FEATURE', 'ENHANCEMENT', 'QUESTION']


def clean_title(title):
    title = title.strip()
    while True:
        old = title
        # remove "[Bug]:" or "[Feature Request]" at the start
        title = re.sub(r'^\[[^\]]*\]\s*:?\s*', '', title)
        # remove "BUG:" at the start (only words from our list)
        match = re.match(r'^([A-Za-z]+)\s*:\s*', title)
        if match and match.group(1).upper() in PREFIXES:
            title = title[match.end():]
        if title == old:        # nothing changed -> done
            break
    return title.strip()


# ---------- 2. form noise ----------
# sentences written by forms, templates or bots
TEMPLATE_PHRASES = [
    "I don't know how to test",
    'This issue is not yet ready for a PR',
    '[!WARNING]',
    'The resulting error is:',
    'And run it with these arguments:',
    'For example, take this code:',
    'Take this code:',
    "If you're new contributor, please check",
    'Please check the links below. If any ASVs',
    'Unable to find junit file',
    '_Originally posted by',
]

# lines that are only a short form answer
SHORT_ANSWERS = ['yes', 'no', 'n/a', 'na', 'none', '-', '---', 'cc']


def remove_form_noise(body):
    body = re.sub(r'<!--.*?-->', ' ', body, flags=re.S)              # hidden comments
    body = re.sub(r'<details>.*?</details>', ' ', body, flags=re.S)   # folded version lists
    body = re.sub(r'^\s*[-*] \[[ xX]\].*$', ' ', body, flags=re.M)    # checklists "- [x] ..."
    body = body.replace('_No response_', ' ')                        # empty form fields
    body = re.sub(r'!\[[^\]]*\]\([^)]*\)?', ' ', body)               # image links

    good_lines = []
    for line in body.splitlines():
        text = line.strip()
        is_template = False
        for phrase in TEMPLATE_PHRASES:
            if phrase in text:
                is_template = True
        if is_template:
            continue
        if text.lower().strip('.') in SHORT_ANSWERS:
            continue
        good_lines.append(line)
    return '\n'.join(good_lines)


# ---------- 3. sections ----------
# if a heading has one of these words, we drop that whole section
DROP_WORDS = ['version', 'environment', 'system', 'arch', 'checklist', 'preflight',
              'installed', 'runtime', 'gist', 'last known working',
              'introduce yourself', 'interest in fixing', 'checks']


def keep_good_sections(body):
    good_lines = []
    keep = True          # text before the first heading is kept
    in_code = False      # are we inside a ``` code block?

    for line in body.splitlines():
        text = line.strip()
        if text.startswith('```'):
            in_code = not in_code

        # a heading looks like "### Something" or "**Something**"
        is_heading = False
        if not in_code:
            if text.startswith('#') or re.fullmatch(r'\*\*[^*]+\*\*:?', text):
                is_heading = True

        if is_heading:
            heading = text.strip('#* :').lower()
            keep = True
            for word in DROP_WORDS:
                if word in heading:
                    keep = False
            continue                 # never keep the heading line itself

        if keep:
            good_lines.append(line)
    return '\n'.join(good_lines)


# ---------- 4. code, personal info, tidy ----------
ERROR_PATTERN = r'Traceback \(most recent call last\)|\b\w+(Error|Exception)\b:'


def split_code(text):
    code_blocks = re.findall(r'```.*?```', text, flags=re.S)   # everything between ```
    text_without_code = re.sub(r'```.*?```', ' ', text, flags=re.S)
    return text_without_code, '\n'.join(code_blocks)


def remove_personal(text):
    text = re.sub(r'https?://\S+', ' ', text)        # links
    text = re.sub(r'@[A-Za-z0-9-]+', ' ', text)      # usernames like @john
    text = re.sub(r'(/[\w.-]+){2,}', ' ', text)      # file paths like /home/john/x.py
    return text


def tidy(text):
    text = re.sub(r'[ \t]+', ' ', text)              # many spaces -> one
    text = re.sub(r'\n\s*\n+', '\n', text)           # many empty lines -> one
    return text.strip()


# ---------- 5. all steps for one issue ----------
def clean_body(body):
    body = remove_form_noise(body)
    body = keep_good_sections(body)
    text, code = split_code(body)
    text = tidy(remove_personal(text))
    has_error = re.search(ERROR_PATTERN, body) is not None
    return text, code, has_error
