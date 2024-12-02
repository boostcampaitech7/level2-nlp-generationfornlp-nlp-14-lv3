# CSAT-Style Problem Solving Model Generation Project

The pace of AI development is remarkably fast these days. Large Language Models (LLMs) in particular are showing excellent performance in various tests. We frequently hear news about them achieving high scores not only in bar exams and medical licensing exams but also in the CSAT (Korean College Scholastic Ability Test). Some even say that AI now possesses doctorate-level knowledge. While AI technology is developing brilliantly, it's true that smaller models still lack performance compared to large models like GPT or Claude.

In this competition, we will start the challenge of solving CSAT problems with smaller models, focusing on the themes of 'Korean language' and 'examinations'.

Most large models achieve quite high scores on the CSAT despite not being perfectly optimized for Korean. So, can we achieve the same results with a smaller model? Or, can we create our own AI model specialized for CSAT based on our understanding of Korean language characteristics and CSAT test features?

Let's create a CSAT-optimized model and try to surpass large models like GPT, Claude, and Gemini!

- input: The input data follows the format of CSAT Korean language and Social Studies passages. It consists of 'id', 'paragraph', 'problems', and 'question_plus', which represent ID, passage, problems, and additional information respectively. Problems contain 'question', 'choices', and 'answer', representing the question, multiple choices, and correct answer. (Please refer to the Data tab for detailed information.)

- output: You must predict the correct answer among the given choices. The answer must be submitted as a CSV file according to the specified submission format. (Please refer to the evaluation method for detailed information about the submission format.)

## Evaluation Method

![accuracy formulation](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe865flhpA80bQPm3hVtRHjlwTFPZuhZQQR7Vm99aQ1pX8_X_xjRwYEdZPLxEAOU40fAaIlVTfcB4CVRY8AG4Tx2Ul1Ek_S44vWouPZklT5x6LIjTaZbsQxDPZOM8LhAMbqh9aeIqcRQb_5XFgpHYBRmK4?key=oHHognsfvjnUPTMndpBxDg)

## Rules

- External Dataset Regulations
  - Use of any CSAT-related data (all sections), KICE mock tests, CSAT-related workbooks (EBS/private materials, etc.) is strictly prohibited. All other external data that can be used under copyright laws is permitted.

- Data Augmentation Regulations
  - Data augmentation based on provided training data and copyright-cleared external data is allowed, including the use of paid AI APIs. However, using CSAT-related data or test sets as seed data, or having humans directly create data based on these, is not allowed.

- Pre-trained Weight Usage
  - Similar to dataset regulations, using pre-trained weights from Ko-MMLU, Multilingual MMLU, KLUE-MRC, or CSAT data is prohibited. Weights must be publicly available and usable by anyone without copyright issues. Used pre-trained weights must be shared in the comments of the 'Pre-trained Weight Usage Notice' post on the announcement board with model name and accessible link. No need to share if already shared by others.

- Test Set Usage
  - Participants cannot use the test set for model training or utilize test set information for final results. Analyzing and using (training on) the test set is also prohibited in this competition. (Including direct visual inspection and labeling)

- Dataset Copyright
  - The competition dataset is available under the 'Camp Educational License'. Please refer to the Boostcourse announcements for detailed copyright information.

## Quickstart

```bash
git clone https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-14-lv3.git
cd level2-nlp-generationfornlp-nlp-14-lv3/src
pip install -r requirements.txt
pre-commit install
python main.py
```
