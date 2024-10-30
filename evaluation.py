from rouge import Rouge
from bert_score import score
import json

def metrics(label,path):
    tp=tn=fp=fn=0
    label=label.lower()
    with open(path,'r') as f:
        for line in f:
            row=json.loads(line)
            
            if row['true_label'].lower()==label:
                if row['accuracy']==1:
                    tp+=1
                    #print(f"tp : {row['true_label']}")
                else:
                    fn+=1
                    #print(f"fn : {row['true_label']},{row['prediction']}")
            else:
                if row['prediction'].lower()==label:
                    fp+=1
                    #print(f"fp : {row['true_label']},{row['prediction']}")
                elif row['prediction'][0]=="{":
                    pred=row['prediction'].strip("{}").strip().split('"prediction":')[1]
                    pred=pred.strip().strip('"')
                    if pred.lower()==label:
                        fp+=1
                        #print(f"fp : {row['true_label']},{row['prediction']}")
                    else:
                        tn+=1
                else:
                    tn+=1
                    #print(f"tn : {row['true_label']},{row['prediction']}")
    return tp,tn,fp,fn

def accuracy(pred,true):
    print(pred,true)
    true=true.lower()
    pred=pred.lower()
    if pred==true:
        return 1
    
    #to check this format of prediction "{\n    \"prediction\": \"not enough info\"\n"}
    
    elif pred[0]=="{":
        pred=pred.strip("{}").strip().split('"prediction":')[1]
        pred=pred.strip().strip('"')
    print(pred,true)
    if pred==true:
        return 1
    else:
        return 0

def rouge_score(row):
    predicted= row['Question_1']+row['Question_2']
    rouge=Rouge()
    scores=rouge.get_scores(row['statement'],predicted)
    
    return scores[0]['rouge-1']['f'],scores[0]['rouge-2']['f'],scores[0]['rouge-l']['f']

def bert_score(row):
    predicted= row['Question_1']+row['Question_2']
    p,r,f1=score([row['statement']],[predicted],lang="en")
    return p,r,f1

def counting_notenoutinfo(path):
    
    class_dict={"refutes":0,"supports":0}
    with open(path,'r') as f:
        for line in f:
            row=json.loads(line)
            if row['prediction']=="not enough info":
                class_dict[row['true_label'].lower()]+=1
            elif row['prediction'][0]=="{":
                    pred=row['prediction'].strip("{}").strip().split('"prediction":')[1]
                    pred=pred.strip().strip('"')
                    if pred.lower()=="not enough info":
                        class_dict[row['true_label'].lower()]+=1
    return class_dict

if __name__=="__main__":
    
    rows=[]
    path="./output/fever_llama_base_predictions_accuracy.jsonl"
    print(counting_notenoutinfo(path))
   
    # precision,recall,f1 
    classes=["SUPPORTS","REFUTES"]
    for c in classes:
        tp,tn,fp,fn=metrics(c,path)
        print(tp,tn,fp,fn)
        recall=tp/(tp+fn)
        precision=tp/(tp+fp)
        f1=(2*recall*precision)/(precision+recall)
        print(f"class {c} | recall: {recall}, precision: {precision}, f1: {f1}")
        
    
    #accuracy
    rows=[]
    acc=0
    with open(path,'r') as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            rows.append(row)
            row['accuracy']=accuracy(row['prediction'],row['true_label'])
            acc+=row['accuracy']
            with open(path,'a') as f:
                f.write(json.dumps(row)+"\n")
    print(acc/len(rows))
  
    #Question and claim relevance check
    file_path="./output/e5_fever_QA_gpt_top2_long.jsonl"
    dup_check=[]       
    r={"rouge1":0,"rouge2":0,"rougel":0}
    b=0
    with open(file_path,'r') as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            if row['statement'] in dup_check:
                continue
            dup_check.append(row['statement'])
            rouge1,rouge2,rougel=rouge_score(row)
            bertscore=bert_score(row)
            r["rouge1"]+=rouge1
            r["rouge2"]+=rouge2
            r["rougel"]+=rougel
            print(bertscore[2])
            b+=bertscore[2]
            
                
    print(b/len(dup_check),r["rouge1"]/len(dup_check),r["rouge2"]/len(dup_check),r["rougel"]/len(dup_check))
 