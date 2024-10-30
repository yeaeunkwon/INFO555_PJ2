from duckduckgo_search import DDGS
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article
import json

parser=argparse.ArgumentParser()
parser.add_argument('--k',default=2,type=int)
parser.add_argument('--output_dir',default="./output/",type=str)
parser.add_argument('--file_name',default="e5_fever_base_relevant_documents_top2.jsonl",type=str)

args = parser.parse_args()

def save_json(row):
   with open(args.output_dir+args.file_name,'a') as f:
      f.write(json.dumps(row)+"\n")
      
def web_scrap(results):
   texts=[]
   links=[]
   for result in results:
      try:
         page=Article(result['href'])
         page.download()
         page.parse()
         text=page.text
         texts.append(result['title']+text)
         links.append(result['href'])
         print(len(result['title']+text))
      except Exception as e:
         print(f"Error in scraping: {e}")
         texts.append(result['title']+result['body'])
         links.append(result['href'])
   return texts,links

def embedding(documents,query):
   model=SentenceTransformer("intfloat/e5-large")
   
   embeddings=model.encode(documents,normalize_embeddings=True)
   embeddings_q=model.encode(query,normalize_embeddings=True)
   print(embeddings.shape)
   
   return embeddings,embeddings_q

def find_topk_query(documents,claim,sources,k): 
   embeddings_doc,embeddings_claim=embedding(documents,claim)
   print(embeddings_doc[0].shape,embeddings_claim.reshape(1,-1).shape)
   cos_scores=[cosine_similarity(doc.reshape(1,-1),embeddings_claim.reshape(1,-1)) for doc in embeddings_doc]
   flattened_scores = [(i,score.item()) for i,score in enumerate(cos_scores)]
   sorted_scores = sorted(flattened_scores, key=lambda x: x[1], reverse=True)
   topk = sorted_scores[:k]
   print(topk)
   top_docs=[documents[k[0]][:4000] for k in topk]
   top_docs_source=[sources[k[0]][:4000] for k in topk]
   return top_docs,top_docs_source

def find_topk_question(documents,question,sources,k):
   embeddings_doc,embeddings_query=embedding(documents,question)
   print(embeddings_doc[0].shape,embeddings_query.reshape(1,-1).shape)
   cos_scores=[cosine_similarity(doc.reshape(1,-1),embeddings_query.reshape(1,-1)) for doc in embeddings_doc]
   flattened_scores = [(i,score.item()) for i,score in enumerate(cos_scores)]
   sorted_scores = sorted(flattened_scores, key=lambda x: x[1], reverse=True)
   topk = sorted_scores[:k]
   print(topk)
   top_docs=[documents[k[0]][:4000] for k in topk]
   top_docs_source=[sources[k[0]][:4000] for k in topk]
   return top_docs,top_docs_source

def search_api_question(row,max_results=5):
   questions=[row['Question 1'],row['Question 2']]
   for i,question in enumerate(questions):
      print(f"question {i+1}")
      results=list(DDGS().text(question,region='us-en',safesearch='off',max_results=max_results))
     
      text_results,sources=web_scrap(results)
      top_docs,top_sources=find_topk_question(text_results,question,sources,args.k)
      row[f'relevant_docs_Q{i+1}']=top_docs
      row[f'sources_Q{i+1}']=top_sources
   save_json(row) 
   return  

def search_api_claim(row,max_results=5):
  
   results=list(DDGS().text(row['statement'],region='us-en',safesearch='off',max_results=max_results))
   
   text_results,sources=web_scrap(results)
   top_docs,top_sources=find_topk_query(text_results,row['statement'],sources,args.k)
   new_row={'id':row['id'],
        'label':row['label'],
        'statement':row['statement'],
        'relevant_docs':top_docs,
         'sources':top_sources
         }
  
   save_json(new_row) 
   return 


   
if __name__=="__main__":
   rows=[]
   with open("./output/fever_subquestions.jsonl",'r') as f:
      for line in f:
         row=json.loads(line)
         rows.append(row)
         
   for i,row in enumerate(rows):
      search_api_question(row)   
         
   for i,row in enumerate(rows):
      search_api_claim(row) 