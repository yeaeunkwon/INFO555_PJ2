# Misinformation dectection with question-answering and RAG

This project explores misinformation detection using question-answering and the Retrieval-Augmented Generation (RAG) technique, structured in three main phases. In the Decomposition Phase, claims are broken down into two specific questions through a prompting strategy. Next, in the Retrieval Phase, relevant documents are retrieved from the external DuckDuckGo API and are used as evidence of each claim for baseline models. For experimental models, answers are generated from these documents either through prompting or by extracting the top five most similar sentences to each claim and question. Finally, in the Fact-Verification Phase, the claim is verified based on the generated answers and the relevant documents.

![pj2_system](https://github.com/user-attachments/assets/78a2bca8-0f78-4f27-9470-fc4a2414b698)


## Dataset
{FEVER Dataset}(https://huggingface.co/datasets/fever/fever)
