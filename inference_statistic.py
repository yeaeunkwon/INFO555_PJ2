import json
import numpy as np
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--base_file',default="./output/fever_llama_base_predictions_accuracy.jsonl",type=str)
parser.add_argument('--exp_file',default="./output/fever_llama_predictions_accuracy.jsonl",type=str)
args = parser.parse_args()

base_rows=[]
base_path=args.base_file
with open(base_path,'r') as f:
    for line in f:
        row=json.loads(line)
        base_rows.append(row)
base_rows=sorted(base_rows,key=lambda x: x['id'])

rows=[]
experimental_path=args.exp_file
with open(experimental_path,'r') as f:
    for line in f:
        row=json.loads(line)
        rows.append(row)
rows=sorted(rows,key=lambda x: x['id'])


difference_score=[]
for base, exp in zip(base_rows,rows):
    if base['statement'] != exp['statement']:
        print("Error: different statement")
    score=exp['accuracy']-base['accuracy']
    difference_score.append(score)
    
sample=len(difference_score)
n=1000
better=0
not_better=0
for i in range(n):
    sampling=np.random.choice(difference_score,sample)
    if np.sum(sampling)>0:
        better+=1
    else:
        not_better+=1
        
p_value=not_better/n
print(p_value)
