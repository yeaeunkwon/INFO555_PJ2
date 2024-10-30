import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
import json
import argparse
import json
import re
rows=[]
with open("./output/e5_fever_relevant_documents_top2.jsonl", 'r') as f:
    for line in f:
        row=json.loads(line)
        rows.append(row)
        
my_token=MY_KEY
login(token = my_token)
check=set()
with open("./output/e5_fever_QA_top_k_sent.jsonl",'r') as f:
    for line in f:
        row=json.loads(line)
        check.add(row['statement'])    
         
model_id = "meta-llama/Llama-3.2-1B-Instruct"
for row in rows:
    query=row['statement']
    question1=row['Question 1']
    documents1=' '.join(row['relevant_docs_Q1'][0])
    question2=row['Question 2']
    documents2=' '.join(row['relevant_docs_Q2'][0])
    instruction="You are given a question and relevant documents to the question. Generate the correct answer to the question and the concise and pointed evidence based on the documents.\n\
    Don't generate unrelated explanations."
    prompt1=f"Question: {question1} Relevant documents: {documents1}"
    prompt2=f"Question: {question2} Relevant documents: {documents2}"  

    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
    pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            
        )
    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt1},
        ]
    outputs = pipe(
            messages,
            max_new_tokens=100,
        )
    pred1=outputs[0]["generated_text"][-1]
    print(pred1)
    
    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt2},
        ]
    outputs = pipe(
            messages,
            max_new_tokens=100,
        )
    
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
    pred2=outputs[0]["generated_text"][-1]
    print(pred2)
    answers={'id':row['id'],
             'statement':row['statement'],
         'Question_1':row['Question 1'],
         'Answer_1': pred1['content'],
         'Question_2': row['Question 2'],
         'Answer_2': pred2['content'],
         'label':row['label']}
    print(answers)
    with open("./output/e5_fever_QA_llama_top_k_sent.jsonl",'a') as f:
        f.write(json.dumps(answers)+'\n')