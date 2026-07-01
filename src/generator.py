import re


KOREAN_STOPWORDS = {
    "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로",
    "와", "과", "의", "도", "만", "및", "또는", "그리고",
    "어떻게", "뭐야", "무엇", "알려줘", "설명해줘", "되나요",
    "받아", "받는", "하는", "있나요", "관련"
}


def split_sentences(text: str) -> list[str]:
    """긴 텍스트를 문장 단위로 나눈다."""
    text = re.sub(r"\s+", " ", text).strip()

    # 마침표/물음표/느낌표/괄호 뒤 기준으로 분리
    sentences = re.split(r"(?<=[.!?。])\s+|(?<=\))\s+", text)

    # 너무 긴 문장은 쉼표 기준으로 한 번 더 나눔
    refined = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 250:
            refined.extend([s.strip() for s in sent.split(",") if s.strip()])
        elif sent:
            refined.append(sent)

    return refined


def extract_keywords(query: str) -> set[str]:
    """질문에서 간단한 키워드를 추출한다."""
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query.lower())

    keywords = set()
    for token in tokens:
        if len(token) >= 2 and token not in KOREAN_STOPWORDS:
            keywords.add(token)

    return keywords


def sentence_score(sentence: str, keywords: set[str]) -> int:
    """문장이 질문 키워드를 얼마나 포함하는지 점수화한다."""
    lowered = sentence.lower()
    score = 0

    for keyword in keywords:
        if keyword in lowered:
            score += 3

    # 너무 짧거나 너무 긴 문장은 약간 감점
    if len(sentence) < 20:
        score -= 1
    if len(sentence) > 350:
        score -= 1

    return score


def generate_extractive_answer(query: str, search_results: list[dict], max_sentences: int = 4) -> str:
    """
    LLM 없이 검색된 문서에서 질문과 관련 있는 문장만 뽑아 답변 초안을 만든다.
    """
    keywords = extract_keywords(query)

    if not keywords:
        return search_results[0]["text"][:500]

    candidates = []

    for result in search_results:
        sentences = split_sentences(result["text"])

        for sentence in sentences:
            score = sentence_score(sentence, keywords)

            if score > 0:
                candidates.append(
                    {
                        "sentence": sentence,
                        "score": score + result.get("score", 0),
                        "source": result["source"],
                        "page": result["page"],
                    }
                )

    if not candidates:
        return (
            "질문과 직접적으로 일치하는 문장을 찾지 못했습니다. "
            "아래의 검색된 근거 문서를 확인해 주세요."
        )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected = candidates[:max_sentences]

    answer_lines = []
    for item in selected:
        answer_lines.append(f"- {item['sentence']}")

    return "\n".join(answer_lines)