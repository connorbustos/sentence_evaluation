from openai import OpenAI
import pandas as pd
import time
import re
from sentence_transformers import SentenceTransformer, util

client = OpenAI(api_key="key")

def chat(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

file_path = 'gpt_automated_test_cases.csv'  # Update this with the actual file path
data = pd.read_csv(file_path)
questions = expected_solution_array = data['Question'].fillna('No Solution').to_numpy()
expected_solution_array = data['Expected Solution'].fillna('No Solution').to_numpy()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pass_count, fail_count = 0, 0

for idx, (question, expected_output) in enumerate(zip(questions, expected_solution_array)):
    # Ensure that the inputs are strings
    expected_output = str(expected_output)
    print("Calling GPT for case " + str(idx + 1) + " with question: " + question)
    gpt_output = str(chat("Solve this question. " + question + ". When giving the answer, format your response by returning `The answer to (question) is (answer)`. For the answer, you CANT display it using latex. It needs to be in regular text format."))
    print("GPT Successfully Answered with", gpt_output)
    # Find the last occurrence of "is" or "="
    last_is_index = gpt_output.rfind(" is ")
    last_equals_index = gpt_output.rfind(" = ")
    if last_is_index != -1 or last_equals_index != -1:
        if last_is_index > last_equals_index:
            delimiter_length = 4  # Length of " is "
            last_index = last_is_index
        else:
            delimiter_length = 3  # Length of " = "
            last_index = last_equals_index
        # Extract everything after "is" or "=" for the answer
        extracted_answer = gpt_output[last_index + delimiter_length:].strip()
        # Extract everything before "is" or "=" for the method and approach
        method_and_approach = gpt_output[:last_index].strip()
    else:
        extracted_answer = 'No Solution'
        method_and_approach = gpt_output  # Use whole output as method if no delimiter found

    embedding_1 = model.encode(extracted_answer, convert_to_tensor=True)
    embedding_2 = model.encode(expected_output, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    if cosine_similarity > 0.7:
        print(f'Case {idx + 1}: PASS')
        pass_count += 1
    else:
        print(f'Case {idx + 1}: FAIL')
        fail_count += 1
    print(f'Case {idx + 1} - Question: "{question}", Extracted Answer: "{extracted_answer}", Expected: "{expected_output}", Similarity: {cosine_similarity.item()}')
    print(f'Method and Approach: {method_and_approach}\n')

total = pass_count + fail_count

print(f'Passed: {pass_count}, Failed: {fail_count}')
print("Pass Rate for Answer: {:.2f}%".format(pass_count / total * 100))