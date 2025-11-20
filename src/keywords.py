# PyABSA: https://github.com/yangheng95/PyABSA
# KeyBERT
# pip install pyabsa keybert sentence-transformers scikit-learn emoji transformers
import os
import json
import re
import math
import random
import argparse
from collections import defaultdict
import emoji
from llama_cpp import Llama
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pyabsa import AspectTermExtraction as ATEPC



# path config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # SpotLite/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # SpotLite/
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")  # SpotLite/output



# pretrained model (English): extract aspect / sentiment / opinion
aspect_extractor = ATEPC.AspectExtractor(checkpoint = "english", auto_device = True, save_mode = False, **{"overwrite_cache": True})
# aspect assignment by semantic similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# keyBERT model
keyword_model = KeyBERT(model = embedder)

# similarity threshold
SIM_THRESHOLD = 0.5  # semantic threshold (similarity) for aspect assignment
AUTO_SIM_THRESHOLD = 0.6  # >= 0.6 => auto add in seeds
CANDIDATE_SIM_THRESHOLD = 0.5  # 0.5~0.6 => candidate  for manual review
MIN_ASPECT_COVERAGE = 5.0  # at least 5% aspect_review

# aspect seed grows along with # of restaurants
DEFAULT_SEEDS = {"food": ["food", "taste", "flavor", "ingredients", "spicy", "sweet", "fresh", "delicious", "broth", "soup", "noodle", "meat", "vegetable", "meal", "dish", "dessert", "drink", "alcohol", "cocktail", "appetizer"],
                 "price": ["price","cost","value","expensive","cheap","affordable","worth"],
                 "environment": ["environment", "atmosphere", "ambience", "clean", "dirty", "crowded", "space", "noisy", "loud", "quiet", "cozy", "music", "lighting", "parking"],
                 "service": ["service", "staff", "waiter", "cashier", "attitude", "friendly", "rude", "helpful", "explain", "speed", "delay", "slow", "fast", "quick"],
                 "waiting_time": ["waiting", "wait", "time", "queue", "line", "delay", "slow", "fast", "quick"]}
# aspect keywords
SEED_FILE = os.path.join(SCRIPT_DIR, "aspect_seeds.json")
def load_seeds():
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, "r", encoding = "utf-8") as f:
            return json.load(f)
    # if aspect_seeds.json not exists
    return None

def save_seeds(seeds: dict):
    with open(SEED_FILE, "w", encoding = "utf-8") as f:
        json.dump(seeds, f, indent = 2, ensure_ascii = False)
# seed candidates for manual review
CANDIDATE_FILE = os.path.join(SCRIPT_DIR, "seed_candidates_review.json")

# aspect score: return numeric score (-1 to +1) and sentiment label
def get_aspect_score(aspect_data):
    pos_list = aspect_data.get("positive", [])
    neg_list = aspect_data.get("negative", [])
    # weight = aspect_percentage
    pos_weight = sum(item[2] for item in pos_list)
    neg_weight = sum(item[2] for item in neg_list)
    if pos_weight + neg_weight == 0:
        score = 0.0
    else:
        score = (pos_weight - neg_weight) / (pos_weight + neg_weight)
    # sentiment mapping based on score threshold
    if pos_weight + neg_weight == 0:
        label = "neutral"
    elif score > 0.35:
        label = "positive"
    elif score < -0.35:
        label = "negative"
    else:
        label = "mixed"

    return round(score, 3), label

# AI summary: structured, statistics-aware summary aligned with aspect sentiment scores
def build_summary(final_output):
    review_cnt = max(1, int(final_output.get("review_cnt", 50)))
    aspect_labels = {"food": "food quality", "price": "pricing", "environment": "environment and atmosphere", "service": "service", "waiting_time": "waiting time",}
    parts = []
    # sort aspects by importance: stronger sentiment + higher mention count first
    ordered_aspects = sorted([aspect for aspect in aspect_labels.keys() if aspect in final_output],
                             key = lambda aspect: abs(final_output[aspect][0]["score"]) * (len(final_output[aspect][1].get("positive", [])) + len(final_output[aspect][1].get("negative", []))),
                             reverse = True)
    # tone variation templates
    positive_templates = ["Most reviewers enjoyed the {label}, often mentioning {example}.",
                          "Many people appreciated the {label}, with highlights like {example}.",
                          "The {label} received mostly positive reactions, especially regarding {example}."]
    negative_templates = ["Several reviewers were unhappy with the {label}, commonly citing {example}.",
                          "The {label} received mixed to negative feedback, particularly issues with {example}.",
                          "Some diners expressed frustration with the {label}, especially mentioning {example}."]
    mixed_templates = ["Opinions on the {label} were mixed — some liked {pos}, while others complained about {neg}.",
                       "Feedback on the {label} varied; a number of people enjoyed {pos}, but {neg} also came up."]
    neutral_templates = ["The {label} was mentioned occasionally but without strong positive or negative reactions.",
                         "There weren't many strong opinions about the {label}."]
    for aspect in ordered_aspects:
        label = aspect_labels[aspect]
        if aspect not in final_output:
            continue
        aspect_overall = final_output[aspect][0]  # {"score": x, "sentiment": y}
        keywords = final_output[aspect][1]
        score = aspect_overall["score"]
        sentiment = aspect_overall["sentiment"]
        # pick keywords only if it appears often enough relative to restaurant size
        def top_terms(bucket, review_cnt):
            if not bucket:
                return None
            # dynamic threshold based on review size
            adaptive_threshold = min(5, max(2, round(math.log10(review_cnt) * 3, 1)))
            # bucket row structure: (phrase, overall%, aspect%, relevance score)
            filtered = [row for row in bucket if row[1] >= adaptive_threshold]
            # fallback to original: still pick best relevance phrase if nothing passes threshold
            if not filtered:
                filtered = bucket
            sorted_bucket = sorted(filtered, key=lambda x: (-x[2], -x[3]))  # aspect% then relevance
            # skip keywords identical to aspect name (ex: "food", "service")
            for phrase, *_ in sorted_bucket:
                if phrase.lower() != aspect.lower():
                    return phrase
            # still return first keyword if nothing qualifies
            return sorted_bucket[0][0]
        pos_example = top_terms(keywords.get("positive", []), review_cnt)
        neg_example = top_terms(keywords.get("negative", []), review_cnt)
        # generate phrasing rules based on sentiment score:
        if sentiment == "positive":
            if pos_example:
                sentence = random.choice(positive_templates).format(label = label, example = pos_example)
            else:
                sentence = f"People generally had a positive impression of the {label}."
        elif sentiment == "negative":
            if neg_example:
                sentence = random.choice(negative_templates).format(label = label, example = neg_example)
            else:
                sentence = f"Some reviewers had complaints about the {label}."
        elif sentiment == "mixed":
            if pos_example and neg_example:
                sentence = random.choice(mixed_templates).format(label = label, pos = pos_example, neg = neg_example)
            elif pos_example:
                sentence = (f"Feedback was mixed — some enjoyed {pos_example}, though there were complaints.")
            elif neg_example:
                sentence = (f"Feedback was mixed — common concerns were {neg_example}.")
            else:
                sentence = f"Opinions varied on the {label}."
        else:  # neutral
            sentence = random.choice(neutral_templates).format(label = label)
        parts.append(sentence)
    # summary (no price)
    summary_body = " ".join(parts).strip()
    # price
    price_info = final_output.get("price_per_person")
    if price_info:
        bullets = []
        if price_info:
            bullets.append(f"• Price per person: ${price_info}")
        for sentence in parts:
            bullets.append(f"• {sentence.rstrip('.')}")
        return "\n".join(bullets)
    else:
        return summary_body

# local LlamaCpp rewrite model (Qwen)
LOCAL_QWEN = r"C:\hf_cache\models\qwen2.5-7b-instruct-q4_k_m\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
rewriter = Llama(model_path =  LOCAL_QWEN, n_ctx =  4096,  temperature = 0.6, top_p = 0.9, repeat_penalty = 1.15)
def rewrite_summary(summary_text: str) -> str:
    if not summary_text or len(summary_text.split()) < 4:
        return summary_text

    summary_text = re.sub(r"(with highlights like|especially regarding|mentioning)\s+\b(?:food|service|price|environment|waiting)\b", "", summary_text, flags=re.IGNORECASE)
    summary_text = re.sub(r"\b(\w+)\s+\1\b", r"\1", summary_text)

    prompt = f"""
Rewrite the following bullet-point insights into a single natural human-like restaurant review summary.
- Keep meaning accurate.
- Avoid repetition and filler.
- 5–9 sentences.
- Tone: helpful, smooth, natural (similar to a good Yelp / TripAdvisor summary).
- Do NOT invent facts.

Bullet Points:
{summary_text}

Final Summary:"""

    response = rewriter(f"<|im_start|>system You summarize restaurant reviews clearly and naturally.<|im_end|>\n"
                        f"<|im_start|>user {prompt}<|im_end|>\n"
                        f"<|im_start|>assistant",
                        max_tokens = 300)

    text = response["choices"][0]["text"].strip()
    return text.split("<|im_end|>")[0].strip()



def process_file(input_file):

    # aspect keywords
    seeds_from_file = load_seeds()
    ASPECT_SEEDS = seeds_from_file or DEFAULT_SEEDS



    def clean_google_review(raw_text: str):
        """
        1. extract Price per person $20–30
        2. remove Google review default format (Food: 5, Service: 5, Meal type: Lunch ...)
        return: (cleaned_text, price_range_or_None)
        """
        if not raw_text:
            return "", None
        text = raw_text

        # price per person
        price_pattern = r"Price per person[: ]\s*\$?(\d+)\s*[–-]\s*\$?(\d+)"
        price_ranges = set()
        for m in re.finditer(price_pattern, text):
            low = m.group(1)
            high = m.group(2)
            price_ranges.add(f"{low}-{high}")
        # if price range appear more than 1 time
        per_review_price = None
        if price_ranges:
            per_review_price = sorted(price_ranges)[0]

        # remove Google review default format
        TAG_KEYS = ["Food", "Service", "Atmosphere", "Meal type", "Price per person", "Noise level", "Seating type", "Group size", "Reservation", "Recommendation for vegetarians", "Parking space", "Parking options", "Special events"]
        # remove "Food: 5", "Service: 3", "Atmosphere: 4", "Meal type: Lunch"
        tag_pattern = r"\b(" + "|".join(re.escape(k) for k in TAG_KEYS) + r")\s*:\s*[^\n]*"
        text = re.sub(tag_pattern, " ", text)
        # no ":"
        no_colon_pattern = r"\b(" + "|".join(re.escape(k) for k in TAG_KEYS) + r")\s+(?!was|is|were|are)[A-Za-z][^\n]*"
        text = re.sub(no_colon_pattern, " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        return text, per_review_price

    # clean review
    def clean_text(text: str) -> str:
        t = text or ""
        t = t.lower()
        t = re.sub(r'http\S+|www\.\S+', ' ', t)
        t = emoji.replace_emoji(t, replace=' ')
        t = re.sub(r'…\s*more$', ' ', t.strip())
        t = re.sub(r'[^0-9a-z\s\.\,\-\'"]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    # read input file (one restaurant's reviews in JSON)
    with open(input_file, "r", encoding = "utf-8") as f:
        data = json.load(f)
    # clean reviews + price range
    raw_texts = [d.get("text", "") for d in data]
    clean_reviews = []
    price_counts = defaultdict(int)
    for raw in raw_texts:
        stripped, price_range = clean_google_review(raw)
        cleaned = clean_text(stripped)
        clean_reviews.append(cleaned)
        if price_range is not None:
            price_counts[price_range] += 1
    n_reviews = len(clean_reviews)



    ## ATEPC extraction

    # run PyABSA model, extract aspect / sentiment / opinion
        # result: {"aspect": "food", "sentiment": "Positive", "opinions": ["amazing"]}
    ATEPC_results = aspect_extractor.extract_aspect(inference_source = clean_reviews, pred_sentiment = True, save_result = False, print_result = False, overwrite_cache = True)
    # list of {aspect: str, sentiment: str, opinion: str}
    records = []
    for review_index, r in enumerate(ATEPC_results):
        # column name: aspect(s), sentiment(s), opinion(s)
        aspects = r.get("aspect") or r.get("aspects") or []
        sentiments = r.get("sentiment") or r.get("sentiments") or []
        opinions = r.get("opinion") or r.get("opinions") or []
        # convert to list
        def as_list(x):
            if x is None: return []
            if isinstance(x, list): return x
            return [x]
        aspects, sentiments, opinions = map(as_list, (aspects, sentiments, opinions))
        # padding to the same length
        L = max(len(aspects), len(sentiments), len(opinions))
        aspects += [""]*(L-len(aspects))
        sentiments += [""]*(L-len(sentiments))
        opinions += [""]*(L-len(opinions))
        for a, s, o in zip(aspects, sentiments, opinions):
            a = (a or "").strip()
            s = (s or "").strip()
            o = (o or "").strip()
            if not (a or o):  # extractor result is empty: skip empty pair
                continue
            # o = horrible, a = taste => oa = horrible taste
            phrase = f"{o} {a}".strip() if o and a else (o or a)
            # normalize duplicates ("food food" → "food")
            phrase = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', phrase).strip()
            records.append({"review_idx": review_index, 
                            "compound_phrase": phrase,
                            "sentiment": s  # Positive / Negative / Neutral
                            })



    ## aspect assignment by semantic similarity: if this phrase belongs to the aspect

    # compute prototype embeddings
    # build prototype vector for each aspect (mean of seed embeddings vector)
    seed_vectors = {aspect: embedder.encode(v, normalize_embeddings = True) for aspect, v in ASPECT_SEEDS.items()}
    prototype_vectors = {aspect: seed_vectors[aspect].mean(axis = 0, keepdims = True) for aspect in seed_vectors}

    # calculate cosine similarity
    def get_aspect(term: str):
        if not term: 
            return None, 0.0
        v = embedder.encode([term], normalize_embeddings = True)
        best_aspect, best_similarity = None, -1.0
        for aspect, prototype_vector in prototype_vectors.items():
            similarity = float(cosine_similarity(v, prototype_vector)[0,0])
            if similarity > best_similarity:
                best_similarity, best_aspect = similarity, aspect
        if best_similarity >= SIM_THRESHOLD:
            # return (best_aspect, best_similarity)
            return best_aspect, best_similarity
        # return (None, best_similarity) if below threshold
        return None, best_similarity

    # collect items per aspect & sentiment
    by_aspect_sentiment = defaultdict(lambda: {"positive": [], "negative": []})
    aspect_review_ids = defaultdict(set)  # reviews that belong to an aspect
    # classified each record (aspect or opinion) as 1 of the 5 aspects
    for record in records:
        candidate_term = record["compound_phrase"]
        aspect, similarity = get_aspect(candidate_term)
        if aspect is None:
            continue
        sentiment = record["sentiment"].lower()
        if sentiment.startswith("pos"):
            by_aspect_sentiment[aspect]["positive"].append((candidate_term, similarity, record["review_idx"]))
        elif sentiment.startswith("neg"):
            by_aspect_sentiment[aspect]["negative"].append((candidate_term, similarity, record["review_idx"]))
        elif sentiment.startswith("neu") or sentiment.startswith("neutr"):
            by_aspect_sentiment[aspect].setdefault("neutral", []).append((candidate_term, similarity, record["review_idx"]))
        else:
            continue
        # any sentiment that is mapped to this aspect should count
        aspect_review_ids[aspect].add(record["review_idx"])



    ## KeyBERT candidates (contextual embedding + n-gram) + review coverage percentages (%): if this phrase is important (top keyword) in the aspect

    # word-boundary regex
    def review_contains(phrase, text):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        return re.search(pattern, text) is not None

    # candidate new aspect seeds
    new_seeds_auto = {aspect: set() for aspect in ASPECT_SEEDS.keys()}  # auto-add (>=0.8 similarity)
    new_seeds_review = {aspect: set() for aspect in ASPECT_SEEDS.keys()}  # review list (0.6-0.79)

    def top10_per_aspect(aspect):
        pos_items = by_aspect_sentiment[aspect]["positive"]
        neg_items = by_aspect_sentiment[aspect]["negative"]
        neu_items = by_aspect_sentiment[aspect].get("neutral", [])
        # for KeyBERT score
        pos_text = " ".join(x[0] for x in pos_items)
        neg_text = " ".join(x[0] for x in neg_items)
        neu_text = " ".join(x[0] for x in neu_items)

        # gather candidates from both sides
        candidate_all = []  # (phrase, score, sentiment)
        def extract_side(txt: str, sentiment: str):
            if not txt.strip():
                return
            keywords = keyword_model.extract_keywords(txt, top_n = 50, keyphrase_ngram_range = (1,3), stop_words = "english", use_mmr = True, diversity = 0.7)
            for phrase, score in keywords:
                candidate_all.append((phrase, float(score), sentiment))
        extract_side(pos_text, "positive")
        extract_side(neg_text, "negative")
        extract_side(neu_text, "neutral")

        # remove duplcates to avoid sub-phrase duplication
        # prefer longer first (EX: "spicy hot pot" > "spicy"), then higher score
        seen = set()
        dedup = []
        for phrase, score, sentiment in sorted(candidate_all, key = lambda x: (-len(x[0]), -x[1])):
            # normalize repeated tokens (food food → food)
            phrase = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', phrase).strip()
            if not phrase:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            dedup.append((phrase, score, sentiment))

        # compute both percentages: overall and aspect-only
        overall_denom = max(1, n_reviews)
        aspect_ids = aspect_review_ids.get(aspect, set())
        aspect_denom = max(1, len(aspect_ids))
        review_pool_overall = range(n_reviews)
        review_pool_aspect = aspect_ids if aspect_ids else []
        # (phrase, overall_pct, aspect_pct, score, sentiment)
        phrase_stats = []
        for phrase, score, sentiment in dedup:
            # hits over all reviews
            overall_hits = 0
            for i in review_pool_overall:
                if review_contains(phrase, clean_reviews[i]):
                    overall_hits += 1
            # hits over aspect reviews
            aspect_hits = 0
            for i in review_pool_aspect:
                if review_contains(phrase, clean_reviews[i]):
                    aspect_hits += 1

            if overall_hits == 0 and aspect_hits == 0:
                continue

            overall_percentage = round(100.0 * overall_hits / overall_denom, 2)
            aspect_percentage  = round(100.0 * aspect_hits / aspect_denom, 2) if aspect_review_ids[aspect] else 0.0
            # KeyBERT score: relevance: TF-IDF + Transformer embedding MMR
            # phrase cosine similarity to prototype vector
            phrase_similarity = float(cosine_similarity(embedder.encode([phrase], normalize_embeddings = True), prototype_vectors[aspect])[0,0])
            final_score = 0.7 * score + 0.3 * phrase_similarity
            # noise filter: if phrase has low coverage and not very similar
            if aspect_percentage < 2 and phrase_similarity < 0.35:
                continue
            phrase_stats.append((phrase, overall_percentage, aspect_percentage, round(final_score, 4), sentiment))
            
            # semi-auto seeds growth
                # only consider phrases that appear frequently enough (aspect_coverage >= 5%) & 1~3 token phrase
            if aspect_percentage >= MIN_ASPECT_COVERAGE and 1 <= len(phrase.split()) <= 3:
                # if already in seed
                if phrase.lower() in {s.lower() for s in ASPECT_SEEDS[aspect]}:
                    continue
                # similarity >= AUTO_SIM_THRESHOLD: auto-add to aspect seeds
                if phrase_similarity >= AUTO_SIM_THRESHOLD:
                    new_seeds_auto[aspect].add(phrase)
                # AUTO_SIM_THRESHOLD >= similarity >=  CANDIDATE_SIM_THRESHOLD: candidates for human review
                elif phrase_similarity >= CANDIDATE_SIM_THRESHOLD:
                    new_seeds_review[aspect].add(phrase)
        
        # split pos / neg vs neutral
        pos_neg = [row for row in phrase_stats if row[4] in ("positive", "negative")]
        neu = [row for row in phrase_stats if row[4] == "neutral"]
        # sort by (aspect_percentage desc, score desc)
        pos_neg.sort(key = lambda x: (-x[2], -x[3]))
        neu.sort(key = lambda x: (-x[2], -x[3]))
        # take top-10 merged (pos + neg)
        top_pos_neg = pos_neg[:10]
        # take independent top-5 (neu)
        top_neu = neu[:5]

        # split back into positive / negative buckets for output
        output = {"positive": [], "negative": [], "neutral": []}
        for row in top_pos_neg:
            output[row[4]].append((row[0], row[1], row[2], row[3]))
        for row in top_neu:
            output["neutral"].append((row[0], row[1], row[2], row[3]))
        return output

    final_output = {}
    for aspect in ASPECT_SEEDS.keys():
        keywords = top10_per_aspect(aspect)
        score, label = get_aspect_score(keywords)
        final_output[aspect] = [{"score": score, "sentiment": label}, keywords]
    final_output["review_cnt"] = n_reviews
    if price_counts:
        best_range, _ = max(price_counts.items(), key = lambda x: x[1])
        final_output["price_per_person"] = best_range
    raw_summary = build_summary(final_output)
    final_output["summary"] = rewrite_summary(raw_summary)



    # output: {aspect: sentiment: [[keyword, overall_percentage, aspect_percentage, KeyBERT score], ......], ......}
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    base = os.path.basename(input_file)
    base_name = os.path.splitext(base)[0]
    output_name = os.path.join(OUTPUT_DIR, f"{base_name}_keywords.json")

    with open(output_name, "w", encoding = "utf-8") as f:
        json.dump(final_output, f, indent = 2, ensure_ascii = False)

    '''
        {"food": [{"score": 0.72,  "sentiment": "positive"},
                {"positive": [["rich broth", 28.57, 62.5, 0.6123], ["fresh ingredients", 22.45, 50.0, 0.5981]],
                "negative": [["too salty", 6.12, 10.0, 0.5510]],
                "neutral": [["......", 5.58, 12.3, 0.2856]]}],
        "price": [{"score": 0.05,  "sentiment": "negative"}, {......},]
        "environment": [{"score": 0.52,  "sentiment": "..."}, {......},]
        "service": [{"score": 0.25,  "sentiment": "..."}, {......},]
        "waiting_time": [{"score": 0.70,  "sentiment": "..."}, {......},]
        "review_cnt": 140,
        "price_per_person": 20-30,
        "summary": "......"}
    '''

    # update aspect_seeds.json (auto-only): merge new_seeds to ASPECT_SEEDS, remove duplicates (lowercase)
    updated_seeds = {}
    for aspect, seed_list in ASPECT_SEEDS.items():
        existing = {s.lower(): s.lower() for s in seed_list}
        # auto-accepted seeds only
        for candidate in new_seeds_auto[aspect]:
            key = candidate.lower()
            if key not in existing:
                existing[key] = candidate
        updated_seeds[aspect] = sorted(existing.values(), key = str.lower)
    save_seeds(updated_seeds)
    
    # save candidate list for manual review (accumulated across restaurants)
    # load previous candidate file if exists
    if os.path.exists(CANDIDATE_FILE):
        with open(CANDIDATE_FILE, "r", encoding = "utf-8") as f:
            old_candidates = json.load(f)
    else:
        old_candidates = {aspect: [] for aspect in ASPECT_SEEDS.keys()}
    # merge new candidates
    for aspect, phrases in new_seeds_review.items():
        merged = set(old_candidates.get(aspect, [])) | set(phrases)
        old_candidates[aspect] = sorted(list(merged), key = str.lower)
    # save back
    with open(CANDIDATE_FILE, "w", encoding = "utf-8") as f:
        json.dump(old_candidates, f, indent = 2, ensure_ascii = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Aspect-based sentiment keyword extractor")
    parser.add_argument("--input", required = True, help = "Filename inside /data OR full path")
    args = parser.parse_args()

    input_file = args.input.strip()
    if not os.path.isabs(input_file):
        path = os.path.abspath(os.path.join(PROJECT_ROOT, input_file))
    else:
        path = input_file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File or folder not found: {path}")

    # folder mode
    if os.path.isdir(path):
        json_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
        for f in json_files:
            print(f" → Processing: {os.path.basename(f)}")
            process_file(f)
    # file mode
    else:
        process_file(path)
