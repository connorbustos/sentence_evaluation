import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

file_path = 'limits_automated_cases.csv'  # Update this with the actual file path
data = pd.read_csv(file_path)
expected_solution_array = data['Expected Solution'].fillna('No Solution').to_numpy()
gpt_solution_array = data['GPT Solution'].fillna('No Solution').to_numpy()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


pass_count, fail_count = 0, 0

for idx, (expected_output, gpt_output) in enumerate(zip(expected_solution_array, gpt_solution_array)):
    # Ensure that the inputs are strings
    expected_output = str(expected_output)
    gpt_output = str(gpt_output)

    last_is_index = gpt_output.rfind(" is ")
    if last_is_index != -1:
        # Extract everything after "is" for the answer
        extracted_answer = gpt_output[last_is_index + 4:].strip()
        # Extract everything before "is" for the method and approach
        method_and_approach = gpt_output[:last_is_index].strip()
    else:
        extracted_answer = 'No Solution'
        method_and_approach = gpt_output

    # Encode and compute similarity between extracted answer and expected output
    embedding_1 = model.encode(extracted_answer, convert_to_tensor=True)
    embedding_2 = model.encode(expected_output, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    if cosine_similarity > 0.7:
        print(f'Case {idx + 1}: PASS')
        pass_count += 1
    else:
        print(f'Case {idx + 1}: FAIL')
        fail_count += 1
    print(f'Case {idx + 1} - Extracted Answer: "{extracted_answer}", Expected: "{expected_output}", Similarity: {cosine_similarity.item()}')
    print(f'Method and Approach: {method_and_approach}\n')

total = pass_count + fail_count

print(f'Passed: {pass_count}, Failed: {fail_count}')
print("Pass Rate for Answer: {:.2f}%".format(pass_count / total * 100))