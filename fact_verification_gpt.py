import json
import argparse
import json
from openai import OpenAI

parser=argparse.ArgumentParser()
parser.add_argument('--output_dir',default="./output/",type=str)
args = parser.parse_args()

rows=[]
with open(args.output_dir+"e5_fever_QA_top_k_sent.jsonl", 'r') as f:
   for line in f:
        row=json.loads(line)
        rows.append(row)
        
API_KEY=MY_KEY
client=OpenAI(api_key=API_KEY)

predictions=[]
true_labels=[]
for row in rows:
    query=row['statement']
    question1=row['Question_1']
    answer1=row['Answer_1']
    question2=row['Question_2']
    answer2=row['Answer_2']
    prompt=f"""
     You are a professional fact checker. You are provided with question-anwer pairs regarding the following claim: {query}\
        Question-Answer pairs: 
        Question1: {question1}\n
        Answer1: {answer1}\n
        Question2: {question2}\n
        Answer2: {answer2}
        Based on strictly the claim and the question-answers provided, you have to provide the rating of the following claim.\
        You must choose one of the following classes to rate the claim.\n
        -Rating: The rating for claim should be one of "supports" if and only if the Question Answer pairs specifically support the claim,\
            "refutes" if and only if the Question Answer pairs specifically refutes the claim or "not enough info" if there is not enough information\
                 to answer the claim.\n

    
    Generated only one rating without any explanations.Follow the JSON format like the example below:\n
    {{
        "prediction": "refutes"
    }}
    """
    ### START OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)
    response=client.chat.completions.create(model="gpt-3.5-turbo", 
                                                    messages=[{"role":"user","content":prompt}],
                                                    max_tokens=10,
                                                    temperature=0.01,
                                                    top_p=1.0,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0)
    pred=response.choices[0].message.content
    ### END OF CODE FROM EXTERNAL SOURCE (URL: https://platform.openai.com/docs/guides/text-generation/building-prompts)
    
    try:
        pred_json=json.loads(pred.strip()) 
        pred_t=pred_json["prediction"]
    except:
        pred_t=pred
    print(pred_t)
    row={'true_label':row['label'],
        'statement':query,
        'prediction':pred_t
    }
    with open(args.output_dir+"fever_gpt_top_k_predictions.jsonl",'a') as f:
        f.write(json.dumps(row)+"\n")
