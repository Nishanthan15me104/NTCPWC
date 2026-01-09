
# Context
1. valuation and Deliverables
2. setup instructions

## Evaluation and Deliverables

### 1.1.	A working Python script or Jupyter Notebook containing the full RAG implementation and setup instructions.

- code is in same repo.

#### clone instruction
```
git clone https://github.com/Nishanthan15me104/NTCPWC.git
cd MARITIME_RAG/maritime_rag
```

### 2.Screenshots of the RAG system's responses to the 5 sample queries below.

**Result screenshot** 

![RAG Query Results](img/queryresult_screenshot.png)


**Actual output**
```bash
 | Time (s)  
-----------------------------------------------------------------
1    | What is the main focus of the Amrit Kaal Vi   | 2.77      
RESPONSE:
The main focus of the Amrit Kaal Vision 2047 is to strengthen India's maritime sector, with key themes including leading the world in safe, sustainable, and green maritime sector, promoting ocean, coastal, and river cruise sector, enhancing modal share of coastal shipping and inland waterway transport, and strengthening India's global maritime presence (Page 13). The vision aims to achieve this through various initiatives, including technological innovation, port development, and international cooperation (Page 12).

2    | Who provided messages in the document, and    | 0.82      
RESPONSE:
There is no information in the provided context about the Prime Minister providing a message in the document. The context only mentions various reports, policies, and initiatives related to the maritime sector, but it does not mention any message from the Prime Minister. (No specific page number can be cited for this as the information is not present)

3    | Describe the cover image of the document.     | 0.11      
RESPONSE:
Information not available in the document.

4    | What are the 11 key themes identified in th   | 1.01
RESPONSE:
The 11 key themes identified in the Amrit Kaal Vision 2047 are not explicitly listed in the provided context. However, two of the key themes are mentioned:

1. Lead The World In Safe, Sustainable & Green Maritime Sector
2. Promote Ocean, Coastal

The context only provides a partial list of the key themes, and the complete list of 11 key themes is not available in the given text. (Page 9)

5    | What visual elements are in the Executive S   | 1.02      
RESPONSE:
On page 8, there are two visual elements: img_8_1.png and img_8_2.png, which appear to be table contents for the table of contents (page 8). However, the Executive Summary itself starts on page 9, and there is no information provided about the visual elements on page 9. Therefore, it is unclear what visual elements are present in the Executive Summary.

(.venv) PS C:\Documents\MARITIME_RAG\maritime_rag> 
```

### 3.	A table summarizing the performance metrics.


| Query No. | Text Retrieval Time (s) | Image Retrieval Time (s) | LLM Response Time (s) | Total Response Time (s) |
|----------|-------------------------|--------------------------|-----------------------|-------------------------|
| 1 | 0.217 | 0.000 | 2.722 | 2.939 |
| 2 | 0.052 | 0.000 | 0.540 | 0.593 |
| 3 | 0.050 | 0.035 | 0.000 | 0.086 |
| 4 | 0.047 | 0.000 | 0.692 | 0.740 |
| 5 | 0.054 | 0.034 | 0.835 | 0.923 |

**performance screenshot**

![performance_screenshot](img/performance.png)


### 4.	Evaluate the multi modal rag application with the standard metrics.

| Query No. | Question                                                                                    | Context Precision | Context Recall |
| --------- | ------------------------------------------------------------------------------------------- | ----------------- | -------------- |
| 1         | What is the main focus of the Amrit Kaal Vision 2047?                                       | 1.000             | 0.883          |
| 2         | Who provided messages in the document, and what is the key message from the Prime Minister? | 1.000             | 0.718          |
| 3         | Describe the cover image of the document.                                                   | 0.000             | 0.000          |
| 4         | What are the 11 key themes identified in the Amrit Kaal Vision 2047?                        | 1.000             | 0.772          |
| 5         | What visual elements are in the Executive Summary page?                                     | 1.000             | 0.731          |

> The evaluation shows consistently high context precision, indicating that retrieved documents are highly relevant to user queries. Lower recall for visual-only queries reflects the current text-dominant nature of the multimodal pipeline, where images are retrieved but not semantically captioned.



# setup instructions: 
## Project Structure Overview

```bash
MARITIME_RAG/
│
├── maritime_rag/
│   ├── data/                  # Extracted text chunks & image metadata
│   ├── img/                   # Extracted images
│   ├── output/                # Optional outputs / logs
│   ├── src/
│   │   ├── processor.py       # PDF processing (text + image extraction)
│   │   ├── build_text_index.py
│   │   ├── build_image_index.py
│   │   ├── retriever.py       # Hybrid retriever (text + image)
│   │   ├── generator.py       # LLM answer generation
│   │   └── __init__.py
│   ├── main.py                # Query answering pipeline
│   ├── ragas_eval.py          # Evaluation script
│   ├── Maritime_AKV_Vision_2047.pdf
│   ├── .env
│   ├── requirements.txt
│   └── readme.md

```
## Step 1: Document Processing (Text + Images)
```bash
cd C:\Documents\MARITIME_RAG\maritime_rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Create a .env file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
## Step 1: Document Processing (Text + Images)

```
python src/processor.py

```
output:
```
data/metadata.json

img/ (extracted images)
```
## Step 2: Build Text FAISS Index
```
python src/build_text_index.py
```
output:
```
data/faiss/text/
├── index.faiss
└── index.pkl
```
## Step 3: Build Image FAISS Index
```
python src/build_image_index.py
```
output:
```
data/faiss/images/
├── index.faiss
└── index.pkl
```
## Step 4: Run the RAG Question–Answering Pipeline
```
python main.py
```
## Step 5: Evaluate the RAG System without ragas
```
python ragas_eval.py
```
## execution order
```
1. python src/processor.py
2. python src/build_text_index.py
3. python src/build_image_index.py
4. python main.py
5. python ragas_eval.py
```


# Deviations from Original Task Requirements

- Image Embedding Model (CLIP not used,Used BLIP for image captioning,Used BAAI/bge-small-en-v1.5 for embedding image captions instead of CLIP)

- Chunk Size & Overlap (800 / 100 instead of 500 / 50

- Hybrid Orchestration (LangChain used instead of LlamaIndex)

 - Hybrid Fusion & Reranking

 # improvements 

 - 1: Proper Hybrid Reranking
 - 2: Clearer & Safer Prompting (Anti-Hallucination)
 - 3: Display Images in Output (Very Important for Multimodal)
 - 4: True Multimodal Triggering (Better than keyword list)
 - 5: Multimodal Score Fusion
 - 6: Evaluation Metrics (need to do ragas)