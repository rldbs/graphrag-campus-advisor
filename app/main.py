import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import streamlit as st

from src.document_loader import load_documents_from_folder, make_chunks
from src.vector_store import load_embedding_model, build_index, search
from src.guardrails import check_query_safety, mask_personal_info
from src.generator import generate_extractive_answer

DATA_DIR = Path("data/raw")


st.set_page_config(
    page_title="GraphRAG Campus Advisor",
    page_icon="📚",
    layout="wide",
)


st.title("📚 GraphRAG Campus Advisor")
st.caption("학과 문서 기반 신뢰형 AI 질의응답 시스템 MVP")


@st.cache_resource(show_spinner="문서 인덱스를 생성하는 중입니다...")
def load_search_index():
    documents = load_documents_from_folder(DATA_DIR)
    chunks = make_chunks(documents)

    if not chunks:
        return [], None, None

    model = load_embedding_model()
    embeddings = build_index(chunks, model)

    return chunks, embeddings, model


st.sidebar.header("프로젝트 상태")
st.sidebar.write("현재 단계: Vector RAG 검색 MVP")
st.sidebar.write("다음 단계: LLM 답변 생성 + Guardrail 강화 + GraphRAG")

pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    st.warning("`data/raw` 폴더에 PDF 파일을 1개 이상 넣어주세요.")
    st.stop()

st.sidebar.subheader("로드된 PDF")
for pdf in pdf_files:
    st.sidebar.write(f"- {pdf.name}")

chunks, embeddings, model = load_search_index()

if not chunks:
    st.error("PDF에서 텍스트를 추출하지 못했습니다. 스캔 PDF가 아닌 텍스트 PDF를 사용해보세요.")
    st.stop()

st.success(f"문서 chunk {len(chunks)}개를 인덱싱했습니다.")

query = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 졸업요건은 어떻게 되나요?",
)

top_k = st.slider("검색할 근거 개수", min_value=1, max_value=8, value=4)

if st.button("질문하기", type="primary"):
    if not query.strip():
        st.warning("질문을 입력해주세요.")
        st.stop()

    is_safe, reason = check_query_safety(query)

    if not is_safe:
        st.error(reason)
        st.stop()

    results = search(query, chunks, embeddings, model, top_k=top_k)

    if not results:
        st.warning("관련 문서를 찾지 못했습니다.")
        st.stop()

    st.subheader("답변 초안")

    draft_answer = generate_extractive_answer(query, results)
    draft_answer = mask_personal_info(draft_answer)

    st.markdown(draft_answer)

    st.subheader("검색된 근거 문서")

    for result in results:
        with st.expander(
            f"{result['rank']}위 | {result['source']} p.{result['page']} chunk.{result['chunk_id']} | score={result['score']:.4f}"
        ):
            st.write(mask_personal_info(result["text"]))