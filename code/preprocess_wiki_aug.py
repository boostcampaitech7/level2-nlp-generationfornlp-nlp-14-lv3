import json
import pandas as pd

# 기존 JSON 데이터를 불러오기
with open("../data/gpt_augmentation_final.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

print(list(json_data[0].keys()))

# 변환된 데이터를 저장할 리스트
valid_data = []
missing_question = []
missing_choices = []
missing_answer = []

# JSON 데이터 변환
for item in json_data:
    has_question = "question" in item
    has_choices = "choices" in item
    has_answer = "correct_answer" in item

    # 문제 항목 만들기
    transformed_item = {
        "id": item.get("id", ""),
        "paragraph": item.get("paragraph", ""),
        "problems": json.dumps({
            "question": item.get("question", ""),  # question 없으면 빈 문자열
            "choices": item.get("choices", []),    # choices 없으면 빈 리스트
            "answer": item.get("correct_answer", "")  # answer 없으면 빈 문자열
        }, ensure_ascii=False),
        "question_plus": ""  # 비어있는 column
    }

    # 조건에 따라 데이터 분류
    if not has_question:
        missing_question.append(transformed_item)
    elif not has_choices:
        missing_choices.append(transformed_item)
    elif not has_answer:
        missing_answer.append(transformed_item)
    else:
        valid_data.append(transformed_item)

# 유효한 데이터 저장
if valid_data:
    valid_df = pd.DataFrame(valid_data)
    valid_df.to_csv("valid_data.csv", index=False, encoding="utf-8")

# question 누락된 데이터 저장
if missing_question:
    missing_question_df = pd.DataFrame(missing_question)
    missing_question_df.to_csv("missing_question.csv", index=False, encoding="utf-8")

# choices 누락된 데이터 저장
if missing_choices:
    missing_choices_df = pd.DataFrame(missing_choices)
    missing_choices_df.to_csv("missing_choices.csv", index=False, encoding="utf-8")

# correct_answer 누락된 데이터 저장
if missing_answer:
    missing_answer_df = pd.DataFrame(missing_answer)
    missing_answer_df.to_csv("missing_answer.csv", index=False, encoding="utf-8")


### 기존 Train Data와 형식 맞추기 ###
# CSV 파일 읽기
df = pd.DataFrame(valid_data)

# 수정된 데이터를 저장할 리스트
updated_data = []

# 데이터 처리
for _, row in df.iterrows():
    # paragraph에서 "와 '를 수정
    paragraph = row['paragraph'].replace('"', '“').replace("'", '‘')

    # problems 컬럼 파싱
    problems = json.loads(row['problems'])

    # question, choices, answer 수정
    problems['question'] = problems.get('question', '').replace('"', '“').replace("'", '‘')
    problems['answer'] = problems.get('answer', '')

    # choices 안의 값들을 수정
    problems['choices'] = [choice.replace('"', '“').replace("'", '‘') for choice in problems.get('choices', [])]

    # 새로운 형태의 데이터 추가
    updated_item = {
        "id":row["id"],
        "paragraph": paragraph,
        "problems": {'question': problems['question'], 'choices': problems['choices'], 'answer': problems['answer']},
        "question_plus": row['question_plus'],
    }

    updated_data.append(updated_item)

# 수정된 데이터를 DataFrame으로 변환
updated_df = pd.DataFrame(updated_data)

# 수정된 데이터를 새로운 CSV로 저장
updated_df.to_csv('updated_valid_data.csv', index=False, encoding='utf-8')
