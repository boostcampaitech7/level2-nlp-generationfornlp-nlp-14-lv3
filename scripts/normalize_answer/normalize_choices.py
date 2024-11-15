import pandas as pd
import ast
import random
import csv
import os

CUR_DIR = os.getcwd()
DATA_DIR = os.path.join(CUR_DIR, "../../data")
OUTPUT_DIR = os.path.join(DATA_DIR, "shuffled_train")


data = pd.read_csv(os.path.join(DATA_DIR, "train_data_aistages.csv"))

# choice를 셔플하는 함수
def normalize_choices(problem, add_choice = False):
    problem_dict = ast.literal_eval(problem)
    
    # choice 개수가 4개인 경우 임의의 choice 추가
    if add_choice and len(problem_dict['choices']) == 4:
        additional_choice = "<IGNORE>"
        problem_dict['choices'].append(additional_choice)
    
    # choice를 셔플한 후 정답 수정
    original_answer = problem_dict['choices'][problem_dict['answer'] - 1]
    choices = problem_dict['choices']
    random.shuffle(choices)
    new_answer = choices.index(original_answer) + 1
    
    # problem_dict 업데이트
    problem_dict['choices'] = choices
    problem_dict['answer'] = new_answer
    
    return problem_dict

# 데이터프레임에 함수 적용
data['problems'] = data['problems'].apply(lambda x: normalize_choices(x))

# 결과를 CSV로 저장
output_file = os.path.join(OUTPUT_DIR, "normalized_train_v4.csv")
data.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

print(f"Processed data saved to {output_file}")
