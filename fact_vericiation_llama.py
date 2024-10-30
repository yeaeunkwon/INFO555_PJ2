import torch
from transformers import pipeline
from huggingface_hub import login
import json
import argparse
import json


parser=argparse.ArgumentParser()
parser.add_argument('--output_dir',default="./output/",type=str)
args = parser.parse_args()

rows=[]
with open(args.output_dir+"e5_fever_QA_top_k_sent.jsonl", 'r') as f:
   for line in f:
        row=json.loads(line)
        rows.append(row)
my_token=MY_KEY
login(token = my_token)

predictions=[]
true_labels=[]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
for row in rows:
    query=row['statement']
    question1=row['Question_1']
    answer1=row['Answer_1']
    question2=row['Question_2']
    answer2=row['Answer_2']
    instruction=f"""
    You are a professional fact checker. You are provided with question-anwer pairs regarding the following claim.\
        Based on strictly the claim and the question-answers provided, you have to provide the rating of the following claim.\
        You must choose one of the following classes to rate the claim.\n
        -Rating: The rating for claim should be one of "supports" if and only if the question Answer pairs specifically support the claim,\
            "refutes" if and only if the Question Answer pairs specifically refutes the claim or "not enough info" if there is not enough information\
                 to answer the claim appropriately.\n
    """
    
    prompt= f"""
        Claim : {query}\n
        Question-Answer pairs:\n 
        Question1: {question1}\n
        Answer1: {answer1}\n
        Question2: {question2}\n
        Answer2: {answer2}
    Does the Question-Answer pairs support,refute, or provide not enough information?
    Generated only one rating without any explanations.Follow the JSON format like the example below:\n
    {{
        "prediction": "refutes"
    }}
    """

    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
    pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
            
        )
    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
    outputs = pipe(
            messages,
            max_new_tokens=50,
            temperature=0.1
    )
    pred=outputs[0]["generated_text"][-1]['content']
    
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
    try:
        pred_json=json.loads(pred.strip()) 
        pred_t=pred_json["prediction"]
    except:
        pred_t=pred
    print(pred_t)
    row={'statement':query,
         'true_label':row['label'],
        'prediction':pred_t
    }
    with open(args.output_dir+"fever_llama_top_k_predictions.jsonl",'a') as f:
        f.write(json.dumps(row)+"\n")