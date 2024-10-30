from datasets import load_dataset
import json
import random
        
def extract_not_enough_info(row):
    pred=row['prediction']
    pred=pred.lower()
    if pred=="not enough info":
        return 1
    #"{\n    \"prediction\": \"not enough info\"\n"}
    elif pred[0]=="{":
        pred=pred.strip("{}").strip().split('"prediction":')[1]
        pred=pred.strip().strip('"')
        if pred=="not enough info":
            print(pred)
            return 1
        else:
            return 0
    else:
        return 0


if __name__=="__main__":
    rows=[]
    ids=[]
    file="fever_gpt_base_predictions_accuracy"
    with open("./output/"+file+".jsonl", 'r') as f:
        for line in f:
                row=json.loads(line)
                rows.append(row)
                if extract_not_enough_info(row)==1:
                    ids.append(row['id'])
    
    sampling_id=random.sample(ids,10)
    for row in rows:
        if row['id'] in sampling_id:
            with open("./output/"+file+"erranalysis.jsonl", 'a') as f:
                f.write(json.dumps(row)+"\n")