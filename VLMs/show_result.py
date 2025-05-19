

import json 


# load result json from path 
file_path = ''

print(file_path)

# Open and load the JSON file
# Open the .jsonl file and read line by line
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse each line as a separate JSON object and append to the list
        data.append(json.loads(line))


for entry in data:
    question = entry["prompt"]
    answer_choices = entry["text"]
    
    print("Câu hỏi:")
    print(question)
    print("\nCâu trả lời:")
    print(answer_choices)
    print("="*50)  # Phân cách giữa các câu hỏi