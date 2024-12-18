import streamlit as st
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# KoBART 모델 및 토크나이저 로드
model_name = 'gogamza/kobart-summarization'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 뉴스 요약 함수
def summarize_article(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=130,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit 애플리케이션
st.title("뉴스 요약 웹사이트")
st.write("뉴스 기사를 입력해주세요")

# 사용자 입력
user_input = st.text_area("News Article", height=300, placeholder="Enter the news article here...")

if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Summarizing..."):
            summary = summarize_article(user_input)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.error("Please enter a valid article.")
