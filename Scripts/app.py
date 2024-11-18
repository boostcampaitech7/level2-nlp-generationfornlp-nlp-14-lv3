import streamlit as st
import pandas as pd

def create_error_analysis_app(dataframes, names, common_indices):
    """
    Streamlit 앱으로 공통 오답을 분석합니다.
    
    Parameters:
    dataframes (list): DataFrame들의 리스트
    names (list): 각 DataFrame의 이름 리스트
    common_indices (list): 공통적으로 틀린 문제들의 인덱스
    """
    st.title("모델 공통 오답 분석")
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = common_indices[0]
    # 사이드바에 문제 선택 옵션 추가
    st.sidebar.header("문제 선택")
    selected_index = st.sidebar.selectbox(
        "분석할 문제 인덱스를 선택하세요",
        options=common_indices,
        format_func=lambda x: f"문제 #{x}",
        key='selected_problem',
        index=common_indices.index(st.session_state.current_index)
    )
    st.session_state.current_index = selected_index
    
    # 메인 화면에 선택된 문제 정보 표시
    st.header(f"문제 #{selected_index} 분석")
    
    # 선택된 문제의 정보 가져오기
    base_df = dataframes[0]
    
    # 문제 정보를 카드 형태로 표시
    with st.container():
        st.subheader("문제 정보")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 질문")
            st.write(base_df.loc[selected_index, 'question'])
            
            st.markdown("### ✏ ID")
            st.write(base_df.loc[selected_index, 'id'])
            
            if 'paragraph' in base_df.columns:
                st.markdown("### 📖 지문")
                st.write(base_df.loc[selected_index, 'paragraph'])
            
            if 'choices' in base_df.columns:
                st.markdown("### 🔍 선택지")
                st.write(base_df.loc[selected_index, 'choices'])
        
        with col2:
            st.markdown("### ✅ 정답")
            st.info(base_df.loc[selected_index, 'label'])
    
    # 각 모델의 예측 결과를 표로 표시
    st.subheader("모델별 예측 결과")
    predictions_data = []
    for df, name in zip(dataframes, names):
        pred = df.loc[selected_index, 'answer']
        correct = "✅" if pred == base_df.loc[selected_index, 'label'] else "❌"
        predictions_data.append({
            "모델": name,
            "예측": pred,
            "정답 여부": correct
        })
    
    predictions_df = pd.DataFrame(predictions_data)
    st.table(predictions_df)
    
    # 버튼 
    st.sidebar.markdown("---")
    current_idx = common_indices.index(selected_index)
    col1, col2 = st.sidebar.columns(2)
    def prev_page():
        current_idx = common_indices.index(st.session_state.current_index)
        if current_idx > 0:
            st.session_state.current_index = common_indices[current_idx - 1]
    
    def next_page():
        current_idx = common_indices.index(st.session_state.current_index)
        if current_idx < len(common_indices) - 1:
            st.session_state.current_index = common_indices[current_idx + 1]
            
    if current_idx > 0:
        col1.button("⬅️ 이전 문제", on_click=prev_page)
    
    if current_idx < len(common_indices) - 1:
        col2.button("다음 문제 ➡️", on_click=next_page)
    # 통계 정보

# 앱 실행을 위한 예시 코드
if __name__ == "__main__":  
    df_gemma = pd.read_csv('./data/train_output_gemma.csv')
    df_gemma_len = pd.read_csv('./data/train_output_gemma_length.csv')
    df_aya = pd.read_csv('./data/train_output_aya.csv')
    df_aya_len = pd.read_csv('./data/train_output_aya_length.csv')

    dataframes = [df_aya, df_aya_len, df_gemma, df_gemma_len]
    names = ['AYA', 'AYA_LEN', 'GEMMA', 'GEMMA_LEN']
    common_indices = [3, 8, 12, 14, 15, 18, 22, 23, 24, 28, 31, 35, 36, 40, 44, 47, 48, 53, 57, 58, 59, 61, 68, 72, 87, 90, 94, 98, 106, 108, 110, 124, 130, 131, 132, 134, 144, 148, 149, 151, 156, 157, 161, 172, 185, 201, 202, 208, 212, 213, 217, 219, 221, 230, 233, 237, 238, 240, 241, 242, 250, 263, 264, 265, 267, 274, 277, 279, 280, 284, 285, 286, 289, 290, 291, 292, 295, 296, 300, 303, 305, 308, 310, 315, 316, 325, 328, 330, 331, 344, 357, 359, 364, 368, 374, 375, 382, 386, 389, 413, 414, 415, 422, 430, 431, 438, 440, 447, 454, 456, 459, 461, 465, 472, 473, 479, 480, 489, 491, 492, 494, 501, 511, 541, 548, 553, 555, 559, 560, 561, 566, 567, 568, 571, 572, 576, 577, 579, 583, 585, 586, 587, 591, 592, 593, 596, 624, 637, 638, 645, 648, 653, 655, 657, 661, 662, 669, 670, 678, 679, 687, 689, 693, 697, 705, 709, 711, 722, 747, 761, 766, 767, 891, 926, 962, 977, 1021, 1136, 1206, 1212, 1285, 1359, 1427, 1633, 1646, 1726, 1731, 1752, 1775, 1777, 1779, 1803, 1818, 1849, 1905, 1908, 1920, 1940, 1941, 1947, 1969, 2013]
    
    create_error_analysis_app(dataframes, names, common_indices)