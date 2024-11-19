import pandas as pd
import streamlit as st


def create_error_analysis_app():
    """
    Streamlit app for analyzing common errors among models.
    """
    dataframes = st.session_state.dataframes
    names = st.session_state.names
    common_indices = st.session_state.common_indices
    df_train = st.session_state.df_train

    st.title("모델 공통 오답 분석")

    # Determine correctness for each question
    # Key: question ID, Value: True (correct) or False (incorrect)
    correctness_dict = {}
    for idx in common_indices:
        question_row = df_train[df_train["id"] == idx].iloc[0]
        correct_answer = question_row["answer"]
        is_correct = True
        for df in dataframes:
            pred_row = df[df["id"] == idx]
            if not pred_row.empty:
                pred = pred_row["answer"].values[0]
                if pred != correct_answer:
                    is_correct = False
                    break
            else:
                is_correct = False
                break
        correctness_dict[idx] = is_correct

    # Add filter option in the sidebar
    st.sidebar.header("필터 옵션")
    filter_option = st.sidebar.selectbox(
        "질문 분류 기준을 선택하세요",
        options=["모든 질문", "맞춘 질문", "틀린 질문"],
        index=0,
    )

    # Filter common_indices based on correctness
    if filter_option == "모든 질문":
        filtered_indices = common_indices
    elif filter_option == "맞춘 질문":
        filtered_indices = [idx for idx in common_indices if correctness_dict[idx]]
    else:  # "틀린 질문"
        filtered_indices = [idx for idx in common_indices if not correctness_dict[idx]]

    if not filtered_indices:
        st.warning("선택한 기준에 해당하는 질문이 없습니다.")
        return

    # Sorting indices numerically
    sorted_indices = filtered_indices

    if (
        "current_index" not in st.session_state
        or st.session_state.current_index not in sorted_indices
    ):
        st.session_state.current_index = sorted_indices[0]

    # Sidebar for question navigation
    st.sidebar.header("문제 선택")

    selected_index = st.sidebar.selectbox(
        "분석할 문제 인덱스를 선택하세요",
        options=sorted_indices,
        format_func=lambda x: f"문제 #{int(x.split('-')[-1])}",
        key="selected_problem",
        index=sorted_indices.index(st.session_state.current_index),
    )

    st.session_state.current_index = selected_index

    # Main screen: Display selected question details
    st.header(f"문제 #{selected_index} 분석")

    # Fetch question details from df_train
    question_row = df_train[df_train["id"] == selected_index].iloc[0]
    with st.container():
        st.subheader("문제 정보")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📝 질문")
            st.write(question_row["question"])

            st.markdown("### ✏ ID")
            st.write(question_row["id"])

            if "paragraph" in df_train.columns:
                st.markdown("### 📖 지문")
                st.write(question_row["paragraph"])

            if "choices" in df_train.columns:
                st.markdown("### 🔍 선택지")
                st.write(question_row["choices"])

        with col2:
            st.markdown("### ✅ 정답")
            st.info(question_row["answer"])

    # Display model predictions
    st.subheader("모델별 예측 결과")
    predictions_data = []
    for df, name in zip(dataframes, names):
        pred_row = df[df["id"] == selected_index]
        if not pred_row.empty:
            pred = pred_row["answer"].values[0]
            correct = "✅" if pred == question_row["answer"] else "❌"
            predictions_data.append({"모델": name, "예측": pred, "정답 여부": correct})
        else:
            predictions_data.append(
                {"모델": name, "예측": "데이터 없음", "정답 여부": "❓"}
            )

    predictions_df = pd.DataFrame(predictions_data)
    st.table(predictions_df)

    # Navigation buttons
    st.sidebar.markdown("---")
    current_idx = sorted_indices.index(selected_index)
    col1, col2 = st.sidebar.columns(2)

    def prev_page():
        current_idx = sorted_indices.index(st.session_state.current_index)
        if current_idx > 0:
            st.session_state.current_index = sorted_indices[current_idx - 1]

    def next_page():
        current_idx = sorted_indices.index(st.session_state.current_index)
        if current_idx < len(sorted_indices) - 1:
            st.session_state.current_index = sorted_indices[current_idx + 1]

    if current_idx > 0:
        col1.button("⬅️ 이전 문제", on_click=prev_page)
    if current_idx < len(sorted_indices) - 1:
        col2.button("다음 문제 ➡️", on_click=next_page)


# Main Streamlit App
if __name__ == "__main__":
    # Initialize session state variables
    if "uploaded_dataframes" not in st.session_state:
        st.session_state.uploaded_dataframes = {}
        st.session_state.uploaded_names = []
        st.session_state.dataframes = []
        st.session_state.names = []
        st.session_state.common_indices = []
        st.session_state.df_train = None

    st.title("CSV File Analysis")

    # File uploader to allow multiple uploads
    uploaded_files = st.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True
    )

    # Process uploaded files immediately and store DataFrames
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_dataframes:
                try:
                    # Read the file into a DataFrame
                    df = pd.read_csv(uploaded_file)  # Assuming 'id' column exists
                    st.session_state.uploaded_dataframes[uploaded_file.name] = df
                    st.session_state.uploaded_names.append(uploaded_file.name)
                    st.write(f"Uploaded {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

    # Display the list of uploaded files
    if st.session_state.uploaded_names:
        st.subheader("Uploaded Files:")
        for name in st.session_state.uploaded_names:
            st.write(name)

    # START button to trigger analysis
    if st.button("START"):
        df_train = pd.read_csv("../data/train_flatten.csv")
        st.session_state.df_train = df_train

        if not st.session_state.uploaded_dataframes:
            st.error("No files uploaded. Please upload files before starting.")
        else:
            # Store dataframes and names in session state
            st.session_state.dataframes = list(
                st.session_state.uploaded_dataframes.values()
            )
            st.session_state.names = list(st.session_state.uploaded_dataframes.keys())

            dataframes = st.session_state.dataframes
            names = st.session_state.names

            # Now compute common_indices
            train_ids = set(df_train["id"])
            common_indices = train_ids

            for df in dataframes:
                ids = set(df["id"])
                common_indices = common_indices.intersection(ids)

            # Sort common_indices numerically
            common_indices = sorted(
                list(common_indices), key=lambda x: int(x.split("-")[-1])
            )
            st.session_state.common_indices = common_indices

            if common_indices:
                st.session_state.current_index = common_indices[0]
                create_error_analysis_app()
            else:
                st.error(
                    "No common indices found between the uploaded files and train data."
                )

    # If analysis is already started, continue showing the app
    elif "common_indices" in st.session_state and st.session_state.common_indices:
        create_error_analysis_app()
