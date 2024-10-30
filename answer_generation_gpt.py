from openai import OpenAI
import json
import re
rows=[]
with open("./output/e5_fever_relevant_documents_top2.jsonl", 'r') as f:
    for line in f:
        row=json.loads(line)
        rows.append(row)

API_KEY=MY_KEY
client=OpenAI(api_key=API_KEY)
for row in rows:
    query=row['statement']
    question1=row['Question 1']
    documents1=' '.join(row['relevant_docs_Q1'])
    question2=row['Question 2']
    documents2=' '.join(row['relevant_docs_Q2'])
    instruction="You are given a question and relevant documents to the question. Generate the correct answer to the question and the concise and pointed evidence based on the documents.\n\
    Don't generate any unrelated explanations.\n"
    prompt=f"Question: {question1} Relevant documents: {documents1}"
    
    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)
    response=client.chat.completions.create(model="gpt-3.5-turbo", 
                                                    messages=[{"role":"user","content":instruction+prompt}],
                                                    max_tokens=128,
                                                    temperature=0.01,
                                                    top_p=1.0,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0)
    pred1=response.choices[0].message.content
    prompt2=f"Question: {question2} Relevant documents: {documents2}\n"
    response=client.chat.completions.create(model="gpt-3.5-turbo", 
                                                    messages=[{"role":"user","content":instruction+prompt2}],
                                                    max_tokens=128,
                                                    temperature=0.01,
                                                    top_p=1.0,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0)
    
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)
    
    pred2=response.choices[0].message.content
    answers={'statement':row['statement'],
         'Question_1':row['Question 1'],
         'Answer_1': pred1,
         'Question_2': row['Question 2'],
         'Answer_2': pred2,
         'label':row['label']}
    print(answers)
    with open("./output/e5_fever_QA_gpt_top2_long.jsonl",'a') as f:
        f.write(json.dumps(answers)+'\n')
