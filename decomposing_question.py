from datasets import load_dataset
from openai import OpenAI
import argparse
import os 
import json


parser=argparse.ArgumentParser()
parser.add_argument('--output_dir',default="./output/",type=str)
args = parser.parse_args()

API_KEY=MY_KEY
client=OpenAI(api_key=API_KEY)
dataset=load_dataset("fever/fever",'v1.0')
if os.path.exists(args.output_dir)==False:
    os.mkdir(args.output_dir)
count_label={"REFUTES":0,"SUPPORTS":0}  
ids=set() 
for i,data in enumerate(dataset['labelled_dev']):
    if data['id'] in ids or data['label']=="NOT ENOUGH INFO":
        continue
    ids.add(data['id'])
    query=data['claim']
    prompt=f"""
    You are a professional fact checker. To fact-check the following claim, generate two questions that should be answered.\
    You should decompose this claim to two questions to check its factuality. Don't generate the unrelevant explanations.\
    Claim: {query}\
    Follow the specific JSON format:\
    '''
        {{
                "questions":["question1","question2"]
        }}
    '''"""
    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)
    response=client.chat.completions.create(model="gpt-3.5-turbo", 
                                                messages=[{"role":"user","content":prompt}],
                                                max_tokens=200,
                                                temperature=0.01,
                                                top_p=1.0,
                                                frequency_penalty=0.0,
                                                presence_penalty=0.0)

    pred=response.choices[0].message.content
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)
    pred=pred.strip().strip("'''")
    print(pred)
    
    
    count_label[data['label']]+=1
    try:
        pred_json=json.loads(pred)  
    except json.JSONDecodeError as e:
        continue
    
    row={'id':data['id'],
        'label':data['label'],
        'statement':query,
        'Question 1':pred_json["questions"][0],
        'Question 2':pred_json["questions"][1]}
    with open(args.output_dir+"fever_subquestions.jsonl",'a') as f:
        f.write(json.dumps(row)+"\n")
   
    
   