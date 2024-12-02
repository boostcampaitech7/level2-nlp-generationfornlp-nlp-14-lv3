# 수능형 문제 풀이 모델 생성 프로젝트

요즘 AI의 발전 속도가 놀라울 정도로 빠릅니다. 특히 대규모 언어 모델(LLM)은 다양한 시험에서 우수한 성과를 내고 있습니다. 사법고시나 의사 시험뿐만 아니라 수능에서도 높은 점수를 기록했다는 소식이 심심찮게 들려옵니다. 이제 AI가 박사 수준의 지식을 가지고 있다고 말하기도 합니다. 이렇게 AI 기술은 눈부시게 발전하고 있지만, 작은 규모의 모델들은 아직 GPT나 Claude 같은 대형 모델에 비해 성능이 부족한 것이 사실입니다.

이번 대회에서는 '한국어'와 '시험'이라는 주제에 맞춰서 작은 모델들로 수능 시험을 풀어보는 도전을 시작해보려 합니다.

대부분의 대형 모델들은 한국어에 완벽히 최적화되지 않았음에도 불구하고 수능에서 꽤 높은 성적을 기록하고 있습니다. 그렇다면, 작은 모델로도 같은 성적을 낼 수 있을까요? 혹은, 우리가 알고 있는 한국어의 특성과 수능 시험의 특징을 바탕으로 수능에 특화된 우리만의 AI 모델을 만들어볼 수 있을까요?

수능에 최적화된 모델을 만들어, GPT, Claude, Gemini 같은 대형 모델들을 뛰어넘어 봅시다!

- input: 입력 데이터는 수능 국어와 사회 과목의 지문 형태를 따릅니다. 'id', 'paragraph', 'problems', 'question_plus'의 형태로 되어 있고, 각각은 id, 지문, 문제, 보기를 의미합니다. problems에는 'question', 'choices', 'answer'로 구성되어 있고 각각 질문, 선택지, 정답을 의미합니다. (자세한 내용은 Data 탭을 참고해주세요.)

- Output: 주어진 선택지 중에서 정답을 맞추어야 합니다. 정답은 지정된 submission 형식에 맞게 csv 파일로 작성하여 제출해야 합니다. (submission 형식에 대한 자세한 내용은 평가 방법을 참고해주세요.)

## 평가방법

![accuracy formulation](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe865flhpA80bQPm3hVtRHjlwTFPZuhZQQR7Vm99aQ1pX8_X_xjRwYEdZPLxEAOU40fAaIlVTfcB4CVRY8AG4Tx2Ul1Ek_S44vWouPZklT5x6LIjTaZbsQxDPZOM8LhAMbqh9aeIqcRQb_5XFgpHYBRmK4?key=oHHognsfvjnUPTMndpBxDg)

## 규칙

- 외부 데이터셋 규정
  - 수능(전체 영역), 평가원 모의고사, 수능 관련 문제집(EBS/사설 교재 등) 등 수능과 관련된 데이터는 일절 사용이 금지되며, 그 외 저작권상 활용 가능한 모든 외부 데이터는 사용을 허용합니다.

- 데이터 증강 관련 규정
  - 제공된 학습 데이터와 저작권상 활용 가능한 외부 데이터를 기반으로 데이터 증강이 가능하며, 이 과정에서 AI를 활용한 유료 API 사용을 허용합니다. 단, 수능 관련 데이터 및 테스트셋을 시드 데이터로 활용하거나 이를 기반으로 사람이 직접 데이터를 제작하는 것은 불가합니다.

- 기학습 가중치 사용
  - 데이터셋 규정과 유사하게 Ko-MMLU, Multilingual MMLU, KLUE-MRC, 수능 데이터로 학습된 기학습 가중치 사용은 금지합니다. 가중치는 모두 public에 공개되어 있고 저작권 문제 없이 누구나 사용 가능해야 합니다. 사용하는 기학습 가중치는 공지 게시판의 '기학습 가중치 사용 공지' 게시글에 댓글로 모델 이름 및 접근 가능한 링크를 반드시 공유합니다. 이미 공유되어 있을 경우 추가로 공유주실 필요는 없습니다.

- 테스트셋 활용 가능 여부
  - 참가자는 모델 학습에 테스트셋을 활용하거나, 테스트셋의 정보를 학습에 활용하여 최종 결과를 낼 수 없습니다. 테스트셋을 분석하고 사용(학습)하는 행위 역시 본 대회에서는 금지합니다. (눈으로 직접 판별 후 라벨링 하는 행위 포함)

- 데이터셋 저작권
  - 대회 데이터셋은 '캠프 교육용 라이선스' 아래 사용 가능합니다. 저작권 관련 세부 내용은 부스트코스 공지사항을 반드시 참고해주세요.

## Quickstart

### Train

```bash
git clone https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-14-lv3.git
cd level2-nlp-generationfornlp-nlp-14-lv3
cp sample_args.json args.json
cd src
pip install -r requirements.txt
wandb login
pre-commit install
python main.py
```

- 모든 학습 결과는 experiments 디렉토리 아래에 학습을 실행한 일시로 생성되는 디렉토리에 저장됩니다.
- 프로젝트 루트의 args.json을 통해 학습 인자를 바꿀 수 있습니다.


### Arguments
