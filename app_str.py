import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import json
import logging

model_id = "paraphrase-multilingual-mpnet-base-v2"

# Loading Pinecone API from the credentials.json file in the same directory
# this file is not included in the repo, but you can create it yourself
with open('credentials.json') as file:
    credentials = json.load(file)
PINECONE_KEY = credentials['PINECONE_KEY_' + model_id]
INDEX_ID = 'med-sem-' + model_id.lower()


@st.cache_resource
def init_pinecone():
    pinecone.init(api_key=PINECONE_KEY, environment="us-central1-gcp")
    return pinecone.Index(INDEX_ID)
    
@st.cache_resource
def init_retriever():
    return SentenceTransformer(model_id)

def make_query(query, retriever, top_k=10, include_values=True, include_metadata=True, filter=None):
    xq = retriever.encode([query]).tolist()
    logging.info(f"Query: {query}")
    attempt = 0
    while attempt < 3:
        try:
            xc = st.session_state.index.query(
                xq,
                top_k=top_k,
                include_values=include_values,
                include_metadata=include_metadata,
                filter=filter
            )
            matches = xc['matches']
            break
        except:
            # force reload
            pinecone.init(api_key=PINECONE_KEY, environment="us-west1-gcp")
            st.session_state.index = pinecone.Index(INDEX_ID)
            attempt += 1
            matches = []
    if len(matches) == 0:
        logging.error(f"Query failed")
    return matches

st.session_state.index = init_pinecone()
retriever = init_retriever()

def card(title: str, context: str):
    html = f"""
    <div class="container-fluid">
        <div class="row align-items-start">
            <div class="col-md-4 col-sm-4">
                <div class="position-relative">
                </div>
            </div>
            <div  class="col-md-8 col-sm-8">
                <h2>Report id: {title}</h2>
            </div>
        <div>
            {context}
    <br><br>
    """
    return st.markdown(html, unsafe_allow_html=True)
    # return st.markdown(context, unsafe_allow_html=True)

    
st.write("""
# Medical Semantic Text Search
""")

st.info("""
* Multilingual DL model: trained on parallel data for 50+ languages (ar, bg, ca, cs, da, de, el, en, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw).
* Source code available on [GitHub](https://github.com/hwasiti/medical_semantic_text_search)
* Contact for more information: Haider Alwasiti at: [haider.alwasiti@helsinki.fi](mailto:haider.alwasiti@helsinki.fi)
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

# with st.expander("Advanced Options"):
#     channel_options = st.select(
#         'Models',
#         ['paraphrase-multilingual-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1'],
#         ['paraphrase-multilingual-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1']
#     )

if query != "":
    # channels = [channel_map[name] for name in channel_options]
    print(f"query: {query}")
    matches = make_query(
        query, retriever, top_k=5
        # filter={
        #     'media': {'$in': 'medical data'}
        # }
    )
    
    # now display cards
    for match in matches:
        card(
            title=match['id'],
            context=match['metadata']['context']
        )