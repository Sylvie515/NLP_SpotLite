# NLP_SpotLite  

A System for Personalized, Aspect-Based Restaurant Recommendation  



---  

## 🗂️ **Project Structure**  
 
SpotLite/  
│  
├── src/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── keywords.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(# main processing script)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── food.csv&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(# seeds for food aspect)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── aspect_seeds.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(# auto-growing seed dictionary)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── seed_candidates_review.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(# pending seed keywords for manual review)  
│  
├── data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(# input review JSON files)  
│  
├── output/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(# generated output files)  
│  
└── README.md  

---  

## 🧰 **Installation & Requirements**  

### **1. Install Dependencies**  

```bash
pip install pyabsa keybert sentence-transformers scikit-learn emoji transformers llama-cpp-python
```  

### **2. Required Versions**  

python==3.10  
pyabsa==2.4.3  
keybert==0.8.3  
sentence-transformers==2.2.2  
scikit-learn==1.4.1  
emoji==2.12.1  
llama-cpp-python==0.2.27  
torch==2.1.2  

---  

## 🖥️ **Local LLM Download (for Summary Rewrite)**  

1. Download the following model from Hugging Face  
🔗 [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main)  
2. Required files  
qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf  
qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf  
3. Place them here (Windows example)  
C:\hf_cache\models\qwen2.5-7b-instruct-q4_k_m\  
4. Ensure your script points to the first file  
LOCAL_QWEN = r"C:\hf_cache\models\qwen2.5-7b-instruct-q4_k_m\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

---  

## 🔁 **Processing Pipeline**  

### **Pipeline Overview**  

- **PyABSA (ATEPC)**: Candidate extraction  
  Extracts aspect–opinion–sentiment triplets from each review and outputs candidate "compound phrases" with sentiment.  

- **SBERT prototype matching**: Aspect assignment  
  Each candidate phrase is embedded and compared (cosine similarity) to per-aspect prototype vectors (mean of seed embeddings).  
  Phrases with similarity ≥ 0.5 are assigned to that aspect.  

- **KeyBERT**: Top keyword extraction  
  After grouping candidate phrases by assigned aspect, KeyBERT extracts representative n-gram keywords (1–3 tokens) from that group.  

Finally, the extracted keywords and sentiment scores are used to generate aspect-level summaries and structured outputs.  

### **1. Text Cleaning & Google Metadata Parsing**  

•	Removes emojis, repeated whitespace, URLs  
•	Removes Google attributes such as:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Food: 4  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Service: 3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Meal Type: Dinner  
•	Extracts available price range and stores it.  

### **2. Aspect Candidate Extraction & Sentiment Detection (PyABSA)**  

•	🔗 [PyABSA](https://github.com/yangheng95/PyABSA)  
•	We use PyABSA (ATEPC) to extract candidate aspect–opinion–sentiment triplets from each review.  
•	Example extraction:  
| Review Text | Extracted |  
| ------ | ------ |  
| "Amazing broth but slow service." | "broth → food → positive", "service → negative" |  

•	These results form the "compound phrases" (opinion + aspect) and the associated sentiment labels will be further processed.  

### **3. Semantic Aspect Assignment (SBERT prototype matching)**  

•	For each aspect, we compute a prototype embedding (the mean embedding of current seed keywords).  
•	Each candidate phrase (from PyABSA) is encoded using the SentenceTransformer embedder and compared to all prototype embeddings via cosine similarity.  
•	If cosine similarity ≥ 0.50, the phrase is assigned to that aspect.  

### **4. Keyword Filtering & Ranking**  

•	For each aspect, we run KeyBERT on the grouped candidate phrases to extract top candidate n-grams (1–3 tokens).  
•	For each candidate phrase, we compute:  
| Metric | Meaning |  
| ------ | ------ |  
| Overall % | % of all reviews mentioning the phrase |  
| Aspect Coverage % | % of reviews related to that aspect mentioning the phrase |  
| Relevance Score | 0.7 * KeyBERT relevance + 0.3 * cosine similarity |  

Only keywords passing noise filters and coverage thresholds (at least 5% aspect_review) appear in the final output.  
	
### **5. Aspect Sentiment Scoring**  

•	For each aspect, sentiment strength is computed as: score = (positive - negative) / (positive + negative)
| Range | Label |  
| ------ | ------ |  
| score > 0.35 | positive |  
| -0.35 < score < 0.35 | mixed |  
| score < -0.35 | negative |  
| positive + negative = 0 | neutral |  

### **6. Keyword-Driven Structured Summary Generation**  
	
Summaries use templated natural-language statements based on sentiment distributions.  
	
### **7. LLM Rewrite to Natural Summary**  

A local Qwen 2.5-7B-Instruct model rewrites the bullet structure summary into a smooth summary.  

### **8. Self-Growing Aspect Seed**  
 
•	The system expands the aspect seed dictionary automatically based on coverage and semantic similarity.  
•	If a keyword appears frequently and similarity ≥ 0.60, it is auto-added.  
•	If the similarity is between 0.50 and 0.59, the keyword will be stored in seed_candidates_review.json for manual approval.  

### **9. Recommended Dishes Extraction**  
 
•	The system automatically identifies top-5 praised dish names based on real user reviews.  
•	Automatically selected from positive food aspect keywords, ranked by aspect-specific mention coverage and relevance score.  
•	Removes generic words (e.g., “food”, “meal”, “dish”)  

---  

## ⚙️ **Usage**  

•	Process a Single File  
```bash
python keywords.py --input "./data/HIMALAYAN_House_reviews.json"
```  

•	Process an Entire Folder  
 ```bash
python keywords.py --input "./data"
```  

---  

## 📌 **Example Output (JSON)**  

 ```bash
{"food": [{"score": 0.034, "sentiment": "mixed"},
          {"positive": [["keyword1", 28.57, 62.5, 0.6123], ["keyword2", 22.45, 50.0, 0.5981]],
           "negative": [......],
           "neutral": [......]}],
 "price": [{......}, {......}],
 "environment": [{……}, {......}],
 "service": [{......}, {......}],
 "waiting_time": [{......}, {......}],
 "price_per_person": "10-20",
 "review_cnt": 40,
 "summary": "......",
 "recommended_dishes": ["spicy beef broth",  "pork dumplings", "handmade noodles"]}
```  

•	Each aspect receives an overall sentiment score (ranging from -1 to +1) and a corresponding label: positive, negative, mixed, or neutral.  
•	For each extracted keyword, the output format looks like ["keyword", 28.57, 62.5, 0.6123] and represents:  
| Position | Meaning |  
| ------ | ------ |  
| 28.57 | The percentage of all reviews that mention this keyword. |  
| 62.5	 | The percentage of reviews related to the same aspect that contain this keyword. |  
| 0.6123 | A weighted relevance score measuring how strongly the keyword represents this aspect. |  

•	The final relevance score ranges from 0 to 1 and is computed using a weighted combination of:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Cosine similarity between the keyword embedding and the aspect prototype embedding.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	KeyBERT relevance score based on contextual keyword importance.  
This produces a Weighted Phrase Importance Score, helping determine whether the phrase is strongly representative of the aspect or only weakly associated.  
