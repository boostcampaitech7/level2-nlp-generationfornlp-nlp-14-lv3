import pandas as pd
import streamlit as st

DATA_DIR = '/data/ephemeral/level2-nlp-generationfornlp-nlp-14-lv3/data/'
wiki = pd.read_csv(DATA_DIR + 'filtered_wikipedia.csv')

#각 text 길이 추가하여 wiki_data로 복사
wiki["text_length"] = wiki["text"].apply(len)
wiki_data = wiki.copy()

def wiki_streamlit():
    # Streamlit 앱 작성
    st.set_page_config(layout="wide")  # WIDE 형태로 변경
    st.title("Wiki 데이터 분석")

    # 기본 데이터프레임 표시
    st.subheader("원본 데이터")
    st.dataframe(wiki.head())

    # 텍스트 길이 분포 확인
    st.subheader("텍스트 길이 분포")
    bins = list(range(0, wiki["text_length"].max() + 500, 500))
    hist_data = pd.cut(wiki["text_length"], bins=bins).value_counts().sort_index()

    # 구간 레이블 문자열로 변환
    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in hist_data.index]
    hist_data.index = bin_labels

    st.bar_chart(hist_data)

    # 모든 텍스트 출력
    st.subheader("모든 텍스트 출력")
    sorted_data = wiki_data.sort_values(by="text_length").reset_index(drop=True)

    # 페이지 설정
    st.write("### 전체 텍스트 출력 (한 번에 5개씩, 5개의 열로 표시)")
    total_pages = len(sorted_data) // 5 + (len(sorted_data) % 5 > 0)  # 2개씩 표시
    page = st.number_input("페이지 번호", min_value=1, max_value=total_pages, value=1, step=1)

    # 현재 페이지 데이터
    start_idx = (page - 1) * 5
    end_idx = start_idx + 5
    current_data = sorted_data.iloc[start_idx:end_idx]

    # 현재 페이지 데이터를 두 열로 표시
    cols = st.columns(5)
    for idx, (col, row) in enumerate(zip(cols, current_data.iterrows())):
        _, row = row
        with col:
            st.write(f"**ID**: {row['id']}")
            st.write(f"**Title**: {row['title']}")
            st.write(f"**Text Length**: {row['text_length']}")
            st.text_area(f"**Text (ID: {row['id']})**", row['text'], height=200, key=f"sorted_text_{start_idx + idx}")
            st.write("---")

    # 선택한 구간에 따른 필터링
    st.subheader("구간별 데이터 필터링")
    selected_bin = st.selectbox("텍스트 길이 구간 선택", options=bin_labels)

    # 선택된 구간의 데이터 필터링
    selected_range = selected_bin.split('-')
    min_length, max_length = int(selected_range[0]), int(selected_range[1])
    filtered_data = sorted_data[(sorted_data["text_length"] > min_length) & (sorted_data["text_length"] <= max_length)].reset_index(drop=True)

    # 페이지 설정
    st.write(f"### 선택된 구간: **{selected_bin}** (한 번에 5개씩, 5개의 열로 표시)")
    filtered_total_pages = len(filtered_data) // 5 + (len(filtered_data) % 5 > 0)
    filtered_page = st.number_input(
        "구간 페이지 번호", min_value=1, max_value=filtered_total_pages, value=1, step=1, key="filtered_page"
    )

    # 현재 페이지 데이터
    filtered_start_idx = (filtered_page - 1) * 5
    filtered_end_idx = filtered_start_idx + 5
    filtered_current_data = filtered_data.iloc[filtered_start_idx:filtered_end_idx]

    # 현재 페이지 데이터를 두 열로 표시
    cols = st.columns(5)
    for idx, (col, row) in enumerate(zip(cols, filtered_current_data.iterrows())):
        _, row = row
        with col:
            st.write(f"**ID**: {row['id']}")
            st.write(f"**Title**: {row['title']}")
            st.write(f"**Text Length**: {row['text_length']}")
            st.text_area(f"**Text (ID: {row['id']})**", row['text'], height=200, key=f"filtered_text_{filtered_start_idx + idx}")
            st.write("---")
    
wiki_streamlit()