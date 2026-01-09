## End-to-end data flow
```
Query
 ↓
Hybrid Retriever
   ├─ Text FAISS
   └─ (Optional) Image FAISS
 ↓
Retrieved Documents
 ↓
Generator
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

COMPLETE MULTIMODAL RAG DATA FLOW (START → END)
STAGE 0 — RAW INPUT
Input
PDF file (e.g. Amrit_Kaal_Vision_2047.pdf)

Nature

Unstructured

Mixed modalities:

Text

Images

Layout

LLMs cannot consume this directly.

STAGE 1 — MULTIMODAL PROCESSOR (PDF → STRUCTURED DATA)
Component

MultimodalProcessor

Responsibility

Convert PDF → reusable structured state

1.1 TEXT EXTRACTION (PyMuPDF / fitz)

Each page is read independently.

Data produced (in memory)
[
  {
    "text": "Full text of page 1 ...",
    "page_num": 1
  },
  {
    "text": "Full text of page 2 ...",
    "page_num": 2
  }
]


✔ Page number preserved
✔ Text still raw, not chunked

1.2 IMAGE EXTRACTION (pdf2image + OpenCV)
Step-by-step transformation (per page)
PDF page
 → PIL Image
 → NumPy Array
 → OpenCV Image
 → Grayscale
 → Threshold
 → Contour Detection
 → Bounding Boxes
 → Cropped Images

Filter logic
Ignore small contours (noise)
Keep large regions → likely figures/diagrams

1.3 IMAGE FILE OUTPUT (Disk)

Each detected image is saved as:

data/extracted/images/img_<page>_<n>.png


Example:

img_47_3.png

1.4 IMAGE METADATA (in memory)
{
  "image_id": "img_47_3",
  "page_num": 47,
  "path": "data/extracted/images/img_47_3.png"
}

1.5 FINAL METADATA STATE (Saved to Disk)

📁 data/extracted/metadata.json

{
  "text": [
    {
      "text": "Text from page 1 ...",
      "page_num": 1
    },
    ...
  ],
  "images": [
    {
      "image_id": "img_47_3",
      "page_num": 47,
      "path": "data/extracted/images/img_47_3.png"
    },
    ...
  ]
}


🧠 This is the SINGLE SOURCE OF TRUTH for the entire system

STAGE 2 — IMAGE → TEXT (CAPTIONING PIPELINE)
Component

BLIP Image Captioning

Why?

Vector databases work on text, not pixels.

2.1 Image Loading

Each .png file is loaded as:

PIL.Image → RGB

2.2 Caption Generation (BLIP)
Image
 → Vision Encoder
 → Language Decoder
 → Caption


Example caption:

"A cover page showing a ship at sea with government branding"

2.3 IMAGE DOCUMENT OBJECT

Each image becomes a LangChain Document:

{
  "page_content": "A cover page showing a ship at sea...",
  "metadata": {
    "page_num": 1,
    "image": "img_1_1.png"
  }
}


📌 Important:

Image is now semantic text

Page alignment is preserved

2.4 IMAGE EMBEDDING

Caption → Vector (via bge-small-en-v1.5)

Caption text → 384-dim embedding

2.5 IMAGE FAISS INDEX (Disk)

📁 data/faiss/images/

Stores:

Vector

Caption

Page number

Image filename

STAGE 3 — TEXT CHUNKING & INDEXING
Component

Text Index Builder

3.1 Load Text from Metadata
{
  "text": [
    { "text": "...", "page_num": 1 }
  ]
}

3.2 TEXT DOCUMENT OBJECT

Each page becomes:

{
  "page_content": "Full page text...",
  "metadata": {
    "page_num": 1
  }
}

3.3 CHUNKING (RecursiveCharacterTextSplitter)

Parameters:

chunk_size = 800
chunk_overlap = 100

Result
Page
 → Chunk 1
 → Chunk 2
 → Chunk 3


Each chunk keeps:

{
  "page_num": 1
}

3.4 TEXT EMBEDDING

Each chunk → vector using same embedding model

🧠 Why same model?

Shared vector space

Enables cross-modal reasoning

3.5 TEXT FAISS INDEX (Disk)

📁 data/faiss/text/

Stores:

Chunk vectors

Chunk text

Page metadata

STAGE 4 — QUERY-TIME RETRIEVAL
Component

MaritimeHybridRetriever

4.1 User Query

Example:

"Describe the cover image of the document"

4.2 QUERY EMBEDDING
Query text → vector


Same embedding model → same vector space

4.3 TEXT RETRIEVAL (Always)
Query vector
 → FAISS (text)
 → Top 5 chunks


Output:

[
  {
    "page_content": "...",
    "metadata": { "page_num": 1 }
  }
]

4.4 VISUAL INTENT DETECTION

If query contains:

image, visual, cover, figure, diagram...


➡ Trigger image retrieval

4.5 PAGE ALIGNMENT LOGIC

From text results:

Relevant pages = {1, 2}


This prevents unrelated image hallucinations.

4.6 IMAGE RETRIEVAL (Conditional)
Query vector
 → FAISS (images)
 → Top 10 captions
 → Filter by relevant pages


Output:

{
  "page_content": "A cover page showing a ship...",
  "metadata": {
    "page_num": 1,
    "image": "img_1_1.png"
  }
}

4.7 FINAL RETRIEVAL OUTPUT
[Text chunks] + [Image captions]

STAGE 5 — CONTEXT CONSTRUCTION
Component

MaritimeGenerator

5.1 CONTEXT FORMAT
[Text | Page 1]: ...
[Visual | Page 1 | img_1_1.png]: ...


🧠 Why?

Explicit modality labeling

Forces grounded reasoning

STAGE 6 — LLM ANSWER GENERATION
Model
llama-3.3-70b-versatile

Prompt Structure
Use ONLY the context below.
CONTEXT:
...
Question:
Answer:

Output
Final grounded answer

STAGE 7 — METRICS & EVALUATION
Measured

Text retrieval time

Image retrieval time

LLM response time

Total latency

Stored as
{
  "Query": 3,
  "Text Retrieval (s)": 0.21,
  "Image Retrieval (s)": 0.34,
  "LLM Response (s)": 1.82,
  "Total Time (s)": 2.45
}

🧠 FINAL ONE-LINE FLOW (MEMORIZE THIS)

PDF → Structured Metadata → Text & Image Indices → Conditional Multimodal Retrieval → Grounded Prompt → LLM Answer