from openai import OpenAI
import json
import pandas as pd
from tqdm import tqdm

OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)

# 데이터프레임 로드
df = pd.read_csv("../data/filtered_wikipedia_v2.csv")

# 결과 저장용 리스트
results = []

# 총 데이터 개수
total_rows = len(df)

for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows"):
    df_id = row["id"]
    paragraph = row["text"]

    # GPT API 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Please generate a exam question in Korean. Provide a question, 5 answer choices, and the correct answer in JSON format."
            },
            {
                "role": "user",
                "content": f"Create a question based on the following paragraph:\n\n{paragraph}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "exam_question_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "description": "This is the question that the user needs to answer.",
                            "type": "string"
                        },
                        "choices": {
                            "description": "These are the 5 possible answer choices for the question.",
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 5,
                            "maxItems": 5
                        },
                        "correct_answer": {
                            "description": "The index (1-5) of the correct answer from the 'choices' array.",
                            "type": "integer"
                        }
                    },
                    "required": ["question", "choices", "correct_answer"],
                    "additionalProperties": False
                }
            }
        }
    )

    # 응답 파싱
    response_json = json.loads(response.choices[0].message.content)
    response_json["id"] = df_id
    response_json["paragraph"] = paragraph

    results.append(response_json)

    # 10% 단위로 저장
    if (index + 1) % (total_rows // 10) == 0:
        percentage = (index + 1) // (total_rows // 10) * 10
        file_name = f"gpt_augmentation_{percentage}.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"{percentage}% 데이터 저장 완료: {file_name}")


# 최종 결과 저장
with open("gpt_augmentation_final.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print("최종 데이터 저장 완료: gpt_augmentation_final.json")
