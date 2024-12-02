import pandas as pd
import random
import ast  # 안전한 eval 대체

# 데이터 불러오기
data = pd.read_csv("../../data/train_data_aistages.csv")

# answer가 1인 경우 랜덤하게 바꾸는 함수
def shuffle_answer(problem):
    # answer가 1이면 랜덤하게 변경
    if problem["answer"] == 1:
        new_answer = random.choice([2, 3, 4])
        # choices에서 기존 정답과 새로운 정답 스왑
        problem["choices"][0], problem["choices"][new_answer - 1] = problem["choices"][new_answer - 1], problem["choices"][0]
        problem["answer"] = new_answer

    return problem

# 각 행을 처리하는 함수
def process_row(row):
    # problems 필드 파싱 (문자열 -> 리스트[딕셔너리])
    problems = ast.literal_eval(row["problems"])

    # 다시 문자열로 변환하여 저장
    row["problems"] = str(shuffle_answer(problems))
    return row

# 데이터프레임에 함수 적용
data = data.apply(process_row, axis=1)

# 결과 저장
data.to_csv("../data/shuffled_train/train_no_1_answer.csv", index=False)
