

import re
import joblib
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from cleaning import clean_title, clean_body


@st.cache_resource                                   
def load():
    models = joblib.load('final_models/triage.joblib')
    encoder = None
    if models['use_embeddings']:                     
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return models, encoder


m, encoder = load()

st.title('🐞 GitHub Issue Triage')
st.caption('Is it a real bug? Will it get fixed? Trained on 7,000+ verified issues from '
           'scikit-learn, pandas, numpy, electron, pytest and black.')

title = st.text_input('Issue title')
body = st.text_area('Issue description', height=250)
col1, col2 = st.columns(2)
project = col1.selectbox('Project', m['repos'] + ['other'])
author = col2.selectbox('Author', ['new user', 'has contributed before'])

if st.button('Analyze', type='primary') and title.strip():
    
    text, code, has_error = clean_body(body)
    masked = re.sub(m['names'], 'PROJECT', clean_title(title) + '. ' + text, flags=re.I)
    meaning = csr_matrix((1, 0)) if encoder is None else csr_matrix(encoder.encode([masked], normalize_embeddings=True))

    
    words_a = m['tfidf_a'].transform([masked])
    p_bug = m['model_a'].predict_proba(hstack([words_a, meaning]))[0, 1]
    if p_bug >= 0.5:
        st.success(f'🐞 **Bug** ({p_bug:.0%} likely)')
    else:
        st.info(f'💡 **Probably not a bug**, more like a feature, question or docs issue ({1 - p_bug:.0%} likely)')

    
    if p_bug >= 0.5:
        facts = [len(code) > 0, has_error, np.log1p(len(text.split())), author == 'has contributed before']
        facts += [project == repo for repo in m['repos']]
        x_b = hstack([m['tfidf_b'].transform([masked]), meaning, csr_matrix(np.array([facts], dtype=float))])
        score = m['model_b'].predict_proba(x_b)[0, 1]
        low, high = m['levels'][project]
        level = '🟢 High' if score >= high else ('🔴 Low' if score < low else '🟡 Medium')
        st.write(f'**Chance it gets fixed:** {level} (compared with other bug reports in this project)')

    
    n_words = words_a.shape[1]
    effect = words_a.toarray()[0] * m['model_a'].coef_[0][:n_words]
    vocab = m['tfidf_a'].get_feature_names_out()
    used = np.nonzero(effect)[0]
    toward_bug = [vocab[i] for i in used[np.argsort(-effect[used])] if effect[i] > 0][:6]
    toward_not = [vocab[i] for i in used[np.argsort(effect[used])] if effect[i] < 0][:6]
    st.write('**Words pointing to bug:** ' + (', '.join(toward_bug) or '–'))
    st.write('**Words pointing to not a bug:** ' + (', '.join(toward_not) or '–'))