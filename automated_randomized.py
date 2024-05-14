from openai import OpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import nlpaug.augmenter.word as naw

client = OpenAI(api_key="")
def chat(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

file_path = 'automated_limits_demo.csv'
data = pd.read_csv(file_path)
questions = expected_solution_array = data['Question'].fillna('No Solution').to_numpy()
expected_solution_array = data['Expected Solution'].fillna('No Solution').to_numpy()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

results = []
pass_count, fail_count = 0, 0
threshold = 0.725 # found from analyzing test results for when GPT answer should be a pass or fail

# Data/Text Augmentation
aug = naw.SynonymAug(aug_src='wordnet')

for idx, (question, expected_output) in enumerate(zip(questions, expected_solution_array)):
    augmented_question = str(aug.augment(question)) # Original Question Augmented
    expected_output = str(expected_output)
    if not expected_output:
        continue
    print("----------------------------------------------------------------------------------------------------------------")
    print("Calling GPT for case: " + str(idx + 1) + " for: " + augmented_question)
    gpt_output = str(chat("Solve this question. " + augmented_question + ". When giving the answer, format your response by returning `The answer to (question) is (answer)`. You have to format the answer in that way because I am extracting the answer in your last sentence after the substring ` is `"))
    print("GPT Answer:", gpt_output)
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
        method_and_approach = gpt_output 
    embedding_1 = model.encode(extracted_answer, convert_to_tensor=True)
    embedding_2 = model.encode(expected_output, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    pass_fail = 'PASS' if cosine_similarity > threshold else 'FAIL'
    if cosine_similarity > threshold:
        print(f'Case {idx + 1}: PASS')
        pass_count += 1
    else:
        print(f'Case {idx + 1}: FAIL')
        fail_count += 1
    results.append([question, augmented_question, gpt_output, extracted_answer, expected_output, cosine_similarity, pass_fail])
    print(f'Extracted Answer: "{extracted_answer}"')
    print(f'Expected: "{expected_output}"\nSimilarity: {cosine_similarity.item()}')
    print("----------------------------------------------------------------------------------------------------------------")

total = pass_count + fail_count

print(f'Passed: {pass_count}, Failed: {fail_count}')
print("Pass Rate for Answer: {:.2f}%".format(pass_count / total * 100))

df_results = pd.DataFrame(results, columns=['Question', 'Augmented Question', 'GPT Answer', 'Extracted Answer','Expected Answer', 'Cosine Similarity', 'Pass/Fail'])
df_results.to_csv('test_case_results.csv', index=False)

labels = ['Passed', 'Failed']
sizes = [pass_count, fail_count]
colors = ['green', 'red']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Test Case Results')
plt.show()