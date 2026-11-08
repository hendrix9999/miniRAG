"""Interface web : streamlit run app.py"""

import os
import tempfile

import streamlit as st

from rag import LLM_MODEL, MiniRAG

st.set_page_config(page_title="miniRAG", page_icon="📄")
st.title("📄 Discute avec tes PDF")
st.caption(f"100 % local · modèle : {LLM_MODEL}")


@st.cache_resource
def load_rag():
    return MiniRAG()


rag = load_rag()

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Ajoute des PDF", type="pdf", accept_multiple_files=True)
    if files and st.button("Indexer"):
        for f in files:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.getbuffer())
                path = tmp.name
            with st.spinner(f"Indexation de {f.name}..."):
                n = rag.add_pdf(path, name=f.name)
            os.remove(path)
            st.success(f"{f.name} : {n} morceaux")

    indexed = rag.sources()
    if indexed:
        st.write("Déjà indexés :")
        for s in indexed:
            st.write(f"- {s}")
        if st.button("Vider la base"):
            rag.reset()
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Aucun document pour l'instant.")

    k = st.slider("Nombre de passages", 1, 10, 4)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Pose ta question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche et rédaction..."):
                answer, hits = rag.ask(question, k)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        st.markdown(answer)
        if hits:
            with st.expander("Sources"):
                for i, h in enumerate(hits, start=1):
                    st.markdown(f"**[{i}] {h['source']} — page {h['page']}** (score {h['score']})")
                    st.caption(h["text"])
    st.session_state.messages.append({"role": "assistant", "content": answer})
