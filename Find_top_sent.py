from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from nltk.tokenize import sent_tokenize

        
def save_answers(top_sent_q1,top_sent_q2,row):
    answers={'statement':row['statement'],
         'Question_1':row['Question 1'],
         'Answer_1': top_sent_q1,
         'Question_2': row['Question 2'],
         'Answer_2': top_sent_q2,
         'label':row['label']}
    
    print(answers)
    with open("./output/e5_fever_QA_top_k_sent.jsonl",'a') as f:
        f.write(json.dumps(answers)+'\n')

def top_embeddings(doc,claim,k=5):
    
    model=SentenceTransformer("intfloat/e5-large")
    doc.append(claim)
    embed=model.encode(doc,normalize_embeddings=True)
   
    cos=cosine_similarity(embed,embed)
    cos_list=[(i,cos) for i,cos in enumerate(cos[-1][:-1])]
    cos_list=sorted(cos_list,key=lambda x: x[1],reverse=True)
    topk=cos_list[:k]
    top_sent=' '.join([doc[idx[0]] for idx in topk])     
    return top_sent

def get_top_doc(row,k=5):
    q1_top=row['relevant_docs_Q1'][0]
    q2_top=row['relevant_docs_Q2'][0]
    
    q1_sent=sent_tokenize(q1_top)
    q2_sent=sent_tokenize(q2_top)
    top_sent_q1=top_embeddings(q1_sent,row['statement']+" "+row['Question 1'],k)
    top_sent_q2=top_embeddings(q2_sent,row['statement']+" "+row['Question 2'],k)
    
    save_answers(top_sent_q1,top_sent_q2,row)
    
if __name__=="__main__":
    data=[]
    with open("./output/e5_fever_relevant_documents_top2.jsonl",'r') as f:
        for line in f:
            row=json.loads(line)
            get_top_doc(row)
        