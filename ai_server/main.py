# ai_server/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional,Dict ,Any
import uvicorn
import numpy as np
import pandas as pd
import math
import os
import re
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
# local imports
from services.data_loader import (
    load_posts,
    load_domain_scores,
    load_author_fingerprints,
    load_top_url_cascade,
)
from services.ai_client import summarize_posts_with_openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------------------------

app = FastAPI(title="REDDIT ANALYZER")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # add deployed frontend origin here later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Return a datetime64 series. Try epoch seconds first, then generic parse."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    # Try numeric epoch seconds
    try:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            parsed = pd.to_datetime(s_num, unit="s", errors="coerce")
            if parsed.notna().any():
                return parsed
    except Exception:
        pass
    # Fallback generic parse
    return pd.to_datetime(s, errors="coerce")


def sanitize_df_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Make DataFrame safe for JSON: datetimes -> ISO, NaN/inf -> None, numpy -> python."""
    df2 = df.copy()
    for col in df2.columns:
        col_vals = df2[col]
        # datetime columns
        if pd.api.types.is_datetime64_any_dtype(col_vals):
            df2[col] = col_vals.dt.tz_localize(None).apply(
                lambda x: x.isoformat() if pd.notnull(x) else None
            )
        else:
            # replace inf with NaN, then NaN -> None
            df2[col] = col_vals.replace([np.inf, -np.inf], np.nan)
            df2[col] = df2[col].where(pd.notnull(df2[col]), None)

            def to_py(x):
                if x is None:
                    return None
                if isinstance(x, (np.generic,)):
                    try:
                        return x.item()
                    except Exception:
                        return x
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    return None
                return x

            df2[col] = df2[col].apply(to_py)
    return df2


def top_values(series: Optional[pd.Series], topn: int):
    if series is None:
        return []
    s = series.fillna("UNKNOWN")
    vc = s.value_counts().head(topn)
    return [{"key": str(k), "count": int(v)} for k, v in zip(vc.index.tolist(), vc.values.tolist())]


# -------------------------------------------------------------------
# Load data once on startup
# -------------------------------------------------------------------

_posts_df = load_posts()
_domain_scores_df = load_domain_scores()
_author_df = load_author_fingerprints()
_top_cascade = load_top_url_cascade()
# ---------------------------------------------------------
# INSERT THIS AFTER LOADING DATA (Before 'Vectorizing text')
# ---------------------------------------------------------

# 1. Setup NLTK VADER
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

# 2. Calculate Sentiment for ALL posts
print("Running VADER Sentiment Analysis on dataset...")

# Helper to get score (-1 to 1)
def get_sentiment_score(text):
    if not isinstance(text, str) or len(text) < 2:
        return 0.0
    # Analyze first 500 chars for speed
    return sid.polarity_scores(text[:500])['compound']

# Prepare text if not already done
if "full_text" not in _posts_df.columns:
    _posts_df["full_text"] = (
        _posts_df["title"].fillna("") + " " + _posts_df["selftext"].fillna("")
    ).astype(str)

# Apply to DataFrame (Creates the 'sentiment' column)
_posts_df['sentiment'] = _posts_df['full_text'].apply(get_sentiment_score)

print("Sentiment scoring complete.")
# coerce created_utc at startup
if "created_utc" in _posts_df.columns:
    _posts_df["created_utc"] = ensure_datetime_series(_posts_df["created_utc"])

# text convenience
_posts_df["title"] = _posts_df.get("title", "").fillna("").astype(str)
_posts_df["selftext"] = _posts_df.get("selftext", "").fillna("").astype(str)
_posts_df["full_text"] = _posts_df["title"] + " " + _posts_df["selftext"]
# Precompute TF-IDF corpus
_corpus = (_posts_df["title"] + " " + _posts_df["selftext"]).tolist()
_vectorizer = TfidfVectorizer(max_features=5000).fit(_corpus)
_corpus_vectors = _vectorizer.fit_transform(_posts_df["full_text"].tolist())
print("Server Ready.")

# -------------------------------------------------------------------
# Topic definitions (simple keyword groups) for topic time-series
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Topic definitions
# -------------------------------------------------------------------

# 1. Define your Standard Topics (The "Base")
BASE_KEYWORDS = {
    "election": ["election", "vote", "ballot", "voting", "polls"],
    "protest": ["protest", "riot", "march", "demonstration", "police"],
    "economy": ["economy", "inflation", "jobs", "unemployment", "tax"],
    "war": ["war", "invasion", "conflict", "military", "gaza", "ukraine"],
}

# 2. Define your Deep-Dive Topics (The "Additions")
EXTRA_KEYWORDS = {
    "misinformation": [
        "fake news", "hoax", "conspiracy", "misinformation", "disinformation", 
        "propaganda", "deepfake", "rumor"
    ],
    "foreign influence": [
        "russia", "china", "iran", "cia", "fbi", "kgb", "espionage", "sanctions", "geopolitics"
    ],
    "health": [
        "covid", "vaccine", "pandemic", "virus", "cure", "doctor", "healthcare"
    ],
    "crime": [
        "crime", "corruption", "scam", "fraud", "violence", "murder", "abuse", "drug", "trafficking"
    ],
    "technology": [
        "ai", "artificial intelligence", "machine learning", "privacy", "surveillance",
        "data breach", "cyberattack", "hacking", "encryption"
    ],
    "climate": [
        "climate change", "global warming", "environment", "pollution", "carbon", "renewable", "disaster"
    ],
    "rights": [
        "human rights", "freedom", "inequality", "gender", "lgbtq", "women rights", "racism"
    ]
}

# 3. MERGE THEM (This adds "Base" to "Topics")
# The '|' operator merges two dictionaries (Python 3.9+)
TOPIC_KEYWORDS = BASE_KEYWORDS | EXTRA_KEYWORDS

# --- PASTE THIS INTO ai_server/main.py ---

@app.get("/api/time-series")
def time_series(
    freq: str = Query("D"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    subreddit: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
):
    """
    Time series of post counts aggregated by freq (H/D/W/M...).
    """
    # Safety Check
    if '_posts_df' not in globals() or _posts_df.empty:
        return []

    df = _posts_df.copy()
    
    # 1. Clean Dates
    if "created_utc" not in df.columns:
        return []
    
    # Use helper if available, else standard pandas
    if 'ensure_datetime_series' in globals():
        df["created_utc"] = ensure_datetime_series(df["created_utc"])
    else:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    # 2. Filters
    if start:
        try:
            start_ts = pd.to_datetime(start)
            df = df[df["created_utc"] >= start_ts]
        except: pass
    if end:
        try:
            end_ts = pd.to_datetime(end)
            df = df[df["created_utc"] <= end_ts]
        except: pass
    if subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == subreddit.lower()]
    if keyword:
        kw = keyword.lower()
        df = df[
            (df["title"].fillna("").str.lower().str.contains(kw)) |
            (df["selftext"].fillna("").str.lower().str.contains(kw))
        ]

    df = df.dropna(subset=["created_utc"])
    if df.empty:
        return []

    # 3. Grouping
    try:
        df["date"] = df["created_utc"].dt.to_period(freq).dt.to_timestamp()
    except:
        df["date"] = df["created_utc"].dt.to_period("D").dt.to_timestamp()

    ts = df.groupby("date").size().reset_index(name="count")

    out = []
    for _, row in ts.iterrows():
        out.append({"date": str(row["date"]), "count": int(row["count"])})
    return out

@app.get("/api/top-lists")
def top_lists(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    subreddit: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    per_list: int = Query(10, ge=1, le=100),
):
    """
    KPIs + top lists (Communities, Authors, Domains) for current filter range.
    """
    # Safety check
    if '_posts_df' not in globals() or _posts_df.empty:
        return {"total_posts": 0, "avg_per_day": 0, "top_subreddits": []}

    df = _posts_df.copy()
    
    # 1. Clean Dates
    if "created_utc" in df.columns:
        if 'ensure_datetime_series' in globals():
            df["created_utc"] = ensure_datetime_series(df["created_utc"])
        else:
            df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    # 2. Apply Filters 
    if start:
        try:
            start_ts = pd.to_datetime(start)
            df = df[df["created_utc"] >= start_ts]
        except: pass
    if end:
        try:
            end_ts = pd.to_datetime(end)
            df = df[df["created_utc"] <= end_ts]
        except: pass
    if subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == subreddit.lower()]
    if keyword:
        kw = keyword.lower()
        df = df[
            (df["title"].fillna("").str.lower().str.contains(kw)) |
            (df["selftext"].fillna("").str.lower().str.contains(kw))
        ]

    # 3. Calculate Stats
    total_posts = int(len(df))

    # Avg per day
    avg_per_day = 0
    if "created_utc" in df.columns and not df["created_utc"].dropna().empty:
        min_date = df["created_utc"].min()
        max_date = df["created_utc"].max()
        days = max(1, (max_date - min_date).days or 1)
        avg_per_day = round(total_posts / days, 2)

    # Growth 7d
    growth_7d = None
    if "created_utc" in df.columns and not df["created_utc"].dropna().empty:
        latest = df["created_utc"].max()
        last7_start = latest - pd.Timedelta(days=7)
        prev7_start = latest - pd.Timedelta(days=14)
        last7_count = df[df["created_utc"] > last7_start].shape[0]
        prev7_count = df[
            (df["created_utc"] > prev7_start) & (df["created_utc"] <= last7_start)
        ].shape[0]
        if prev7_count > 0:
            growth_7d = int(round(((last7_count - prev7_count) / prev7_count) * 100))

    # 4. Generate Top Lists (This feeds the Pie Chart)
    top_subreddits = top_values(df["subreddit"] if "subreddit" in df.columns else None, per_list)
    top_authors = top_values(df["author"] if "author" in df.columns else None, per_list)
    top_domains = top_values(df["domain"] if "domain" in df.columns else None, per_list)

    return {
        "total_posts": total_posts,
        "avg_per_day": avg_per_day,
        "growth_7d": growth_7d,
        "top_subreddits": top_subreddits,
        "top_authors": top_authors,
        "top_domains": top_domains,
    }
# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    subreddit: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
# --- Add this to your SCHEMAS section ---

class SummarizeRequest(BaseModel):
    mode: str = "range"  # "range" | "posts"
    post_ids: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    subreddit: Optional[str] = None
    keyword: Optional[str] = None
    length: str = "short"
import openai
# --- REPLACE THE chat_endpoint FUNCTION IN ai_server/main.py ---

# ai_server/main.py

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    query = req.query
    print(f"\n--- [DEBUG] Incoming Query: {query} ---")

    # 1. Initialization Check
    if '_vectorizer' not in globals() or _posts_df.empty:
        return {"answer": "System is initializing.", "sources": []}

    # 2. Vector Search (Find relevant posts)
    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _corpus_vectors).flatten()
    top_indices = np.argsort(-sims)[:10]
    relevant_posts = _posts_df.iloc[top_indices]
    
    if req.subreddit:
        sub_mask = relevant_posts["subreddit"].str.lower() == req.subreddit.lower()
        if sub_mask.any():
            relevant_posts = relevant_posts[sub_mask]

    # --- 3. LOOKUP: DOMAINS & AUTHORS ---
    
    context_data = ""
    query_terms = set(re.findall(r'\w+', query.lower())) # Split query into words

    # A. AUTHOR LOOKUP
    if '_author_df' in globals() and not _author_df.empty:
        for term in query_terms:
            if len(term) < 3: continue
            
            # Exact match for authors is safer (case-insensitive)
            match = _author_df[_author_df['author'].str.lower() == term]
            
            if not match.empty:
                row = match.iloc[0]
                # -- Calculate Author Score Inline --
                score = 100
                flags = []
                
                # Penalties
                if row.get('duplicate_text_ratio', 0) > 0.5:
                    score -= 30; flags.append("Repetitive content")
                if row.get('percent_link_posts', 0) > 0.9:
                    score -= 20; flags.append("High link ratio")
                if row.get('avg_time_between_posts_sec', 999) < 120 and row.get('post_count', 0) > 5:
                    score -= 40; flags.append("Bot-like frequency")
                
                score = max(0, min(100, score))
                
                line = (f"- Author: {row['author']} | Trust Score: {score}/100 | "
                        f"Activity: {row['post_count']} posts | Flags: {', '.join(flags)}\n")
                print(f"[FOUND AUTHOR] {line.strip()}")
                context_data += line

    # B. DOMAIN LOOKUP
    if '_domain_scores_df' in globals() and not _domain_scores_df.empty:
        for term in query_terms:
            if len(term) < 3: continue
            
            # Partial match for domains (e.g. "bbc" matches "bbc.co.uk")
            matches = _domain_scores_df[_domain_scores_df['domain'].str.contains(term, case=False, na=False)]
            
            for _, row in matches.head(2).iterrows():
                d_name = row['domain']
                raw_score = row.get('score', 0)
                reasons = row.get('reasons', '')
                
                # Normalize Score (-2 to +2  ->  10 to 95)
                norm_score = 50 + (raw_score * 15)
                norm_score = int(max(10, min(95, norm_score)))
                
                line = f"- Domain: {d_name} | Trust Score: {norm_score}/100 | Notes: {reasons}\n"
                if line not in context_data:
                    print(f"[FOUND DOMAIN] {line.strip()}")
                    context_data += line

    # 4. Build Post Text Context
    posts_context = ""
    sources = []
    for _, row in relevant_posts.iterrows():
        title = row['title']
        body = str(row['selftext'])[:200].replace("\n", " ")
        sub = row['subreddit']
        posts_context += f"- [r/{sub}] {title}: {body}\n"
        sources.append({
            "id": str(row.get("id", "")),
            "title": title,
            "subreddit": sub,
            "url": str(row.get("url", ""))
        })

    # 5. LLM Call
    system_prompt = (
        "You are ANALYZER AI. Answer using the provided Data Context when relevant.\n"
        "The Context includes 'Trust Scores' for specific Authors or Domains.\n"
        "If the user asks about credibility, trust, or if a user is a bot, YOU MUST CITE these scores.\n"
        "If the data section is empty, admit you don't know."
    )
    
    user_prompt = (
        f"Data & Trust Scores (High Priority):\n{context_data}\n\n"
        f"Recent Reddit Posts:\n{posts_context}\n\n"
        f"User Question: {query}"
    )

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"answer": "Error: OPENAI_API_KEY missing.", "sources": sources}

        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        answer = "I'm having trouble connecting to OpenAI."

    return {
        "answer": answer,
        "sources": sources
    }
# -------------------------------------------------------------------
# Basic endpoints
# -------------------------------------------------------------------

# In ai_server/main.py

@app.get("/api/posts")
def get_posts(
    limit: int = Query(100, ge=1, le=10000),
    subreddit: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    author: Optional[str] = Query(None), # <--- NEW: Author Parameter
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    min_ups: Optional[int] = Query(None),
):
    """
    Get a list of individual posts with extensive filtering (Sub, Author, Keyword, Date).
    """
    # Safety check
    if '_posts_df' not in globals() or _posts_df.empty:
        return []

    df = _posts_df.copy()

    # 1. Ensure Dates are Parsed
    if "created_utc" in df.columns:
        if 'ensure_datetime_series' in globals():
            df["created_utc"] = ensure_datetime_series(df["created_utc"])
        else:
            df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    # 2. Apply Filters
    if subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == subreddit.lower()]
    
    if author: # <--- NEW: Filter by Author
        df = df[df["author"].fillna("").str.lower() == author.lower()]

    if keyword:
        kw = keyword.lower()
        # Search in both title and body
        df = df[
            (df["title"].fillna("").str.lower().str.contains(kw)) |
            (df["selftext"].fillna("").str.lower().str.contains(kw))
        ]

    if start:
        try:
            start_ts = pd.to_datetime(start)
            df = df[df["created_utc"] >= start_ts]
        except: pass

    if end:
        try:
            end_ts = pd.to_datetime(end)
            df = df[df["created_utc"] <= end_ts]
        except: pass

    if min_ups is not None and "ups" in df.columns:
        try:
            df = df[df["ups"] >= min_ups]
        except: pass

    # 3. Sort & Limit
    # Sort by newest first if date exists
    if "created_utc" in df.columns:
        df = df.sort_values("created_utc", ascending=False)
    
    df = df.head(limit)
    
    # 4. Sanitize for JSON (Handle NaN, Infinity, NaT)
    df_safe = sanitize_df_for_json(df)
    
    # Returns all columns (including 'sentiment' if it was calculated at startup)
    return df_safe.to_dict(orient="records")

@app.get("/api/domain-scores")
def domain_scores():
    df_safe = sanitize_df_for_json(_domain_scores_df)
    return df_safe.to_dict(orient="records")


@app.get("/api/author-fingerprints")
def author_fingerprints(limit: int = 100):
    df = _author_df.head(limit)
    df_safe = sanitize_df_for_json(df)
    return df_safe.to_dict(orient="records")


@app.get("/api/top-url-cascade")
def top_url_cascade():
    df_safe = sanitize_df_for_json(_top_cascade)
    return df_safe.to_dict(orient="records")

# -------------------------------------------------------------------
# CHATBOT ENDPOINT (The functionality you asked for)
# -------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    raw_query = req.query
    query_lower = raw_query.lower().strip()
    
    # --- 1. INSTANT GREETING (No DB Search) ---
    # Regex matches "hi", "hello", "hi bot" but not "high prices"
    if re.match(r"^(hi|hello|hey|greetings|sup|yo)\b", query_lower):
        return {
            "answer": "Hello! I am your Reddit Data Analyst. I can help you find trends, check credibility scores (e.g., 'worst domains'), or summarize specific topics.",
            "sources": []
        }

    # 2. Init Check
    if '_vectorizer' not in globals() or _posts_df.empty:
        return {"answer": "System is initializing. Please wait.", "sources": []}

    # --- 3. INTELLIGENT QUERY CLEANING ---
    # If user asks "Why is there a spike in technology", we want to search for "technology", NOT "spike".
    # We remove "analysis words" to find the actual content.
    stop_words = ["why", "is", "there", "a", "spike", "in", "trend", "about", "suddenly", "happening", "with", "show", "me", "list", "the"]
    search_terms = [w for w in query_lower.split() if w not in stop_words]
    clean_query = " ".join(search_terms)
    
    if not clean_query: 
        clean_query = raw_query # Fallback if everything was removed

    # --- 4. BUILD CONTEXT (Trust + Content) ---
    trust_context = ""
    posts_context = ""
    sources = []

    # A. TRUST DATA (Domains/Authors)
    if '_domain_scores_df' in globals() and not _domain_scores_df.empty:
        # Ensure scores are numbers
        df_scores = _domain_scores_df.copy()
        df_scores['score'] = pd.to_numeric(df_scores['score'], errors='coerce').fillna(0)

        # Logic: "Most Credible" / "Best"
        if any(x in query_lower for x in ["most", "best", "highest", "top", "credible", "trust"]):
            # Filter out self-posts to show real news sites
            best = df_scores[~df_scores['domain'].str.startswith('self.', na=False)]
            best = best.sort_values("score", ascending=False).head(5)
            trust_context += "\n[TOP RATED DOMAINS]\n"
            for _, row in best.iterrows():
                s = int(max(10, min(95, 50 + (row['score'] * 15))))
                trust_context += f"- {row['domain']} (Score: {s}/100)\n"

        # Logic: "Least Credible" / "Worst"
        elif any(x in query_lower for x in ["least", "worst", "lowest", "bad", "spam", "fake"]):
            worst = df_scores.sort_values("score", ascending=True).head(5)
            trust_context += "\n[LOWEST RATED DOMAINS]\n"
            for _, row in worst.iterrows():
                s = int(max(10, min(95, 50 + (row['score'] * 15))))
                trust_context += f"- {row['domain']} (Score: {s}/100) - Flags: {row.get('reasons', 'Unknown')}\n"

        # Logic: Specific Domain Check (e.g. "Is bbc reliable?")
        for term in search_terms:
            if len(term) < 3: continue
            match = df_scores[df_scores['domain'].str.contains(term, case=False)]
            if not match.empty:
                row = match.iloc[0]
                s = int(max(10, min(95, 50 + (row['score'] * 15))))
                trust_context += f"\n[Domain Info: {row['domain']}] Score: {s}/100 ({row.get('reasons', '')})\n"

    # B. POST CONTENT (Vector Search with Cleaned Query)
    query_vec = _vectorizer.transform([clean_query])
    sims = cosine_similarity(query_vec, _corpus_vectors).flatten()
    top_indices = np.argsort(-sims)[:10]
    relevant_posts = _posts_df.iloc[top_indices]
    
    # Filter posts by subreddit if UI filter is active
    if req.subreddit:
        sub_mask = relevant_posts["subreddit"].str.lower() == req.subreddit.lower()
        if sub_mask.any():
            relevant_posts = relevant_posts[sub_mask]

    for _, row in relevant_posts.iterrows():
        title = row['title']
        date = str(row['created_utc']).split(" ")[0] # Important for "Trend" questions
        sub = row['subreddit']
        posts_context += f"- [{date}] [r/{sub}] {title}\n"
        sources.append({"id": str(row.get("id","")), "title": title, "subreddit": sub, "url": str(row.get("url",""))})

    # --- 5. SYSTEM PROMPT (The Brain) ---
    system_prompt = (
        "You are SimPPL AI. You interpret Reddit data for a user.\n"
        "DATA SOURCES:\n"
        "1. [TRUST DATA]: Lists domains and their credibility scores (0-100). Use this for 'best/worst/credibility' questions.\n"
        "2. [POSTS]: A list of recent post titles with dates. Use this for 'trends/spikes/topic' questions.\n\n"
        "RULES:\n"
        "- If asked about a 'spike' or 'trend', look at the [POSTS] dates and titles to explain WHAT is being discussed.\n"
        "- If asked about 'credibility' or 'best domains', ONLY use the [TRUST DATA].\n"
        "- If the contexts are empty, apologize and say you need more data."
    )
    
    user_prompt = f"--- TRUST DATA ---\n{trust_context}\n\n--- POSTS ---\n{posts_context}\n\n--- USER QUESTION ---\n{raw_query}"

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: return {"answer": "API Key Missing", "sources": []}
        
        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        answer = "I'm having trouble connecting to OpenAI."

    return {"answer": answer, "sources": sources}
    # ---------------------------------------------------------
    # PART B: TRUST DATA LOOKUP (For "Credibility", "Bots")
    # ---------------------------------------------------------
    trust_context = ""
    
    # 1. Ranking Queries ("Worst", "Best")
    if '_domain_scores_df' in globals() and not _domain_scores_df.empty:
        if any(x in query_lower for x in ["least", "worst", "lowest", "bad", "spam"]):
            worst = _domain_scores_df.sort_values("score", ascending=True).head(5)
            trust_context += "\n[System Data: Lowest Credibility Domains]\n"
            for _, row in worst.iterrows():
                s = int(max(10, min(95, 50 + (row.get('score', 0) * 15))))
                trust_context += f"- {row['domain']} (Score: {s}/100)\n"
        
        elif any(x in query_lower for x in ["most", "best", "highest", "top", "good", "trust"]):
            best = _domain_scores_df.sort_values("score", ascending=False).head(5)
            trust_context += "\n[System Data: Highest Credibility Domains]\n"
            for _, row in best.iterrows():
                s = int(max(10, min(95, 50 + (row.get('score', 0) * 15))))
                trust_context += f"- {row['domain']} (Score: {s}/100)\n"

    # 2. Specific Lookup (Domains/Authors mentioned in query)
    query_terms = set(re.findall(r'\w+', query_lower))
    
    # Check Authors
    if '_author_df' in globals() and not _author_df.empty:
        for term in query_terms:
            if len(term) < 3: continue
            match = _author_df[_author_df['author'].str.lower() == term]
            if not match.empty:
                row = match.iloc[0]
                trust_context += f"\n[Author Data: {row['author']}]\n"
                trust_context += f"- Post Count: {row['post_count']}\n"
                trust_context += f"- Link Ratio: {round(row.get('percent_link_posts', 0)*100)}% (High links = sus)\n"

    # Check Domains
    if '_domain_scores_df' in globals() and not _domain_scores_df.empty:
        for term in query_terms:
            if len(term) < 3: continue
            match = _domain_scores_df[_domain_scores_df['domain'].str.contains(term, case=False)]
            if not match.empty:
                row = match.iloc[0]
                s = int(max(10, min(95, 50 + (row.get('score', 0) * 15))))
                trust_context += f"\n[Domain Data: {row['domain']}]\n- Credibility Score: {s}/100\n- Flags: {row.get('reasons', 'None')}\n"

    # ---------------------------------------------------------
    # PART C: THE FIX (Better Prompting)
    # ---------------------------------------------------------
    if not trust_context:
        trust_context = "No specific credibility scores relevant to this query."

    system_prompt = (
        "You are SimPPL AI. You have access to two types of data:\n"
        "1. 'REDDIT POSTS': The actual content discussion.\n"
        "2. 'TRUST DATA': Credibility scores for domains/authors.\n\n"
        "INSTRUCTIONS:\n"
        "- If the user asks about TRENDS, EVENTS, or SPIKES: Ignore the Trust Data and synthesize the 'REDDIT POSTS'.\n"
        "- If the user asks about CREDIBILITY or TRUST: Use the 'TRUST DATA'.\n"
        "- Never say 'I don't have data' if the 'REDDIT POSTS' section has content. Use that content to answer."
    )
    
    user_prompt = (
        f"--- TRUST DATA ---\n{trust_context}\n\n"
        f"--- REDDIT POSTS (Context) ---\n{posts_context}\n\n"
        f"--- USER QUESTION ---\n{query}"
    )

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"answer": "Error: OPENAI_API_KEY missing.", "sources": sources}

        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        answer = "I'm having trouble connecting to OpenAI."

    return {
        "answer": answer,
        "sources": sources
    }
# -------------------------------------------------------------------
# Topic time-series (rubric b)
# -------------------------------------------------------------------

@app.get("/api/topic-time-series")
def topic_time_series(
    freq: str = Query("D"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    subreddit: Optional[str] = Query(None),
):
    """
    Time series of key topics/themes based on simple keyword groups.

    Returns:
    [
      { "name": "election", "data": [{ "date": "2025-02-01", "count": 12 }, ...] },
      ...
    ]
    """
    df = _posts_df.copy()
    if "created_utc" not in df.columns:
        raise HTTPException(status_code=500, detail="created_utc column missing")

    df["created_utc"] = ensure_datetime_series(df["created_utc"])

    # filters
    if start:
        try:
            start_ts = pd.to_datetime(start)
            df = df[df["created_utc"] >= start_ts]
        except Exception:
            raise HTTPException(status_code=400, detail=f"invalid start date: {start}")
    if end:
        try:
            end_ts = pd.to_datetime(end)
            df = df[df["created_utc"] <= end_ts]
        except Exception:
            raise HTTPException(status_code=400, detail=f"invalid end date: {end}")
    if subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == subreddit.lower()]

    df = df.dropna(subset=["created_utc"])
    if df.empty:
        return []

    # text column for matching
    df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.lower()

    allowed_freqs = {"H", "D", "W", "M", "Q", "Y"}
    if freq.upper() not in allowed_freqs:
        raise HTTPException(status_code=400, detail=f"freq must be one of {sorted(allowed_freqs)}")

    df["date"] = df["created_utc"].dt.to_period(freq).dt.to_timestamp()

    results = []
    for topic_name, keywords in BASE_KEYWORDS.items():
        # build regex like (word1|word2|word3)
        pattern = "|".join(re.escape(k.lower()) for k in keywords)
        mask = df["text"].str.contains(pattern, regex=True)
        df_topic = df[mask]
        if df_topic.empty:
            continue
        ts = df_topic.groupby("date").size().reset_index(name="count")
        topic_data = []
        for _, row in ts.iterrows():
            topic_data.append({
                "date": str(row["date"]),
                "count": int(row["count"]),
            })
        results.append({
            "name": topic_name,
            "data": topic_data,
        })

    return results


# -------------------------------------------------------------------
# Top-lists / KPIs
# -------------------------------------------------------------------

@app.get("/api/top-lists")
def top_lists(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    subreddit: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    per_list: int = Query(10, ge=1, le=100),
):
    """
    KPIs + top lists for current filter range:
    { total_posts, avg_per_day, growth_7d, top_subreddits, top_authors, top_domains }
    """
    df = _posts_df.copy()

    if "created_utc" in df.columns:
        df["created_utc"] = ensure_datetime_series(df["created_utc"])

    if start:
        try:
            start_ts = pd.to_datetime(start)
            df = df[df["created_utc"] >= start_ts]
        except Exception:
            raise HTTPException(status_code=400, detail=f"invalid start date: {start}")
    if end:
        try:
            end_ts = pd.to_datetime(end)
            df = df[df["created_utc"] <= end_ts]
        except Exception:
            raise HTTPException(status_code=400, detail=f"invalid end date: {end}")
    if subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == subreddit.lower()]
    if keyword:
        kw = keyword.lower()
        df = df[
            (df["title"].str.lower().str.contains(kw)) |
            (df["selftext"].str.lower().str.contains(kw))
        ]

    total_posts = int(len(df))

    if "created_utc" in df.columns and not df["created_utc"].dropna().empty:
        min_date = df["created_utc"].min()
        max_date = df["created_utc"].max()
        days = max(1, (max_date - min_date).days or 1)
        avg_per_day = round(total_posts / days, 2)
    else:
        avg_per_day = None

    growth_7d = None
    if "created_utc" in df.columns and not df["created_utc"].dropna().empty:
        latest = df["created_utc"].max()
        last7_start = latest - pd.Timedelta(days=7)
        prev7_start = latest - pd.Timedelta(days=14)
        last7_count = df[df["created_utc"] > last7_start].shape[0]
        prev7_count = df[
            (df["created_utc"] > prev7_start) & (df["created_utc"] <= last7_start)
        ].shape[0]
        if prev7_count > 0:
            growth_7d = int(round(((last7_count - prev7_count) / prev7_count) * 100))
        else:
            growth_7d = None

    top_subreddits = top_values(df["subreddit"] if "subreddit" in df.columns else None, per_list)
    top_authors = top_values(df["author"] if "author" in df.columns else None, per_list)
    top_domains = top_values(df["domain"] if "domain" in df.columns else None, per_list)

    return {
        "total_posts": total_posts,
        "avg_per_day": avg_per_day,
        "growth_7d": growth_7d,
        "top_subreddits": top_subreddits,
        "top_authors": top_authors,
        "top_domains": top_domains,
    }


# -------------------------------------------------------------------
# Semantic search (your existing stub)
# -------------------------------------------------------------------

@app.post("/api/semantic-search")
def semantic_search(query: str, k: int = 5):
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _corpus_vectors).flatten()
    idxs = np.argsort(-sims)[:k]
    results = []
    for i in idxs:
        row = _posts_df.iloc[i]
        results.append({
            "id": row.get("id", ""),
            "author": row.get("author", ""),
            "score": float(sims[i]),
            "title": row.get("title", ""),
            "selftext": row.get("selftext", ""),
            "created_utc": str(row.get("created_utc", "")),
        })
    return {"results": results}


# -------------------------------------------------------------------
# Summarize for Explain selection (rubric AI feature)
# -------------------------------------------------------------------

# --- PASTE THIS INTO ai_server/main.py (Replacing the old summarize function) ---

@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    """
    Summarizes the dataset by combining HARD STATS (quantitative) 
    with CONTENT SAMPLES (qualitative) for a true data insight.
    """
    # 1. Safety Check
    if '_posts_df' not in globals() or _posts_df.empty:
        return {
            "headline": "Dataset Empty",
            "summary": "The dataset is currently empty or not loaded.",
            "evidence": []
        }

    df = _posts_df.copy()
    
    # 2. Apply Filters
    if "created_utc" in df.columns:
        if 'ensure_datetime_series' in globals():
            df["created_utc"] = ensure_datetime_series(df["created_utc"])
        else:
            df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    if req.start:
        try: df = df[df["created_utc"] >= pd.to_datetime(req.start)]
        except: pass
    if req.end:
        try: df = df[df["created_utc"] <= pd.to_datetime(req.end)]
        except: pass
    if req.subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == req.subreddit.lower()]
    if req.keyword:
        kw = req.keyword.lower()
        df = df[
            (df["title"].fillna("").str.lower().str.contains(kw)) |
            (df["selftext"].fillna("").str.lower().str.contains(kw))
        ]

    # 3. Check if empty
    total_matches = len(df)
    if df.empty:
        return {
            "headline": "No Data Found",
            "summary": "No posts match your current filters.",
            "evidence": []
        }

    # ---------------------------------------------------------
    # 4. CALCULATE HARD STATS (The "Insight" Engine)
    # ---------------------------------------------------------
    
    # Top Communities
    top_subs = df['subreddit'].value_counts().head(3)
    subs_str = ", ".join([f"r/{k} ({v})" for k, v in top_subs.items()])
    
    # Top Domains (Sources)
    if 'domain' in df.columns:
        top_domains = df['domain'].value_counts().head(3)
        domains_str = ", ".join([f"{k} ({v})" for k, v in top_domains.items()])
    else:
        domains_str = "N/A"

    # Engagement Stats
    avg_score = int(df['ups'].mean()) if 'ups' in df.columns else 0
    
    # ---------------------------------------------------------
    # 5. SMART SAMPLING (Content Context)
    # ---------------------------------------------------------
    # Get highest engagement posts to see what people care about
    if "ups" in df.columns:
        df_sample = df.sort_values("ups", ascending=False).head(25)
    else:
        df_sample = df.head(25)

    sample_texts = []
    for _, row in df_sample.iterrows():
        sub = str(row.get('subreddit', ''))
        title = str(row.get('title', ''))
        text_entry = f"[r/{sub}] {title}"
        sample_texts.append(text_entry)

    # ---------------------------------------------------------
    # 6. CONSTRUCT DATA SCIENTIST PROMPT
    # ---------------------------------------------------------
    
    prompt_template = (
        f"Act as a Senior Data Scientist analyzing a Reddit dataset. "
        f"Do NOT just summarize the news. Analyze the BEHAVIOR and TRENDS of the dataset.\n\n"
        
        f"--- DATASET STATISTICS ---\n"
        f"Total Posts: {total_matches}\n"
        f"Top Communities: {subs_str}\n"
        f"Top Sources/Domains: {domains_str}\n"
        f"Avg Upvotes: {avg_score}\n\n"
        
        f"--- TOP POST SAMPLES ---\n"
        f"{chr(10).join(sample_texts)}\n\n"
        
        "--- TASK ---\n"
        "Provide a high-level strategic insight JSON object:\n"
        "1. 'headline': A 5-word analytical title (e.g., 'High Polarization in Political Subs').\n"
        "2. 'summary': A technical summary explaining the dataset's focus. Mention which communities are dominating the conversation and if the sentiment seems organic or coordinated.\n"
        "3. 'evidence': 3 bullet points highlighting distinct data patterns (e.g., 'r/politics dominates volume', 'Heavy reliance on twitter.com links', 'Focus shifts from economy to foreign policy')."
    )

    # 7. Call AI Client
    try:
        result = summarize_posts_with_openai([], prompt_template) # We pass empty list because we embedded text in prompt
        
        if not isinstance(result, dict):
             return {
                 "headline": "Dataset Analysis",
                 "summary": str(result),
                 "evidence": ["See summary"]
             }
             
        return {
            "headline": result.get("headline", "Data Insights"),
            "summary": result.get("summary", "Analysis complete."),
            "evidence": result.get("evidence", [])
        }

    except Exception as e:
        print(f"Summarization Error: {e}")
        return {
            "headline": "Statistical Overview",
            "summary": f"This dataset contains {total_matches} posts, primarily driven by {subs_str}.",
            "evidence": [f"Dominant Sub: {top_subs.index[0] if not top_subs.empty else 'None'}", f"Avg Score: {avg_score}"]
        }
# -------------------------------------------------------------------
# Network endpoint (rubric d)
# -------------------------------------------------------------------

@app.get("/api/network")
def network(
    subreddit: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    limit: int = Query(300, ge=10, le=1000),
):
    """
    Simple author–domain network:
    - nodes: authors + domains
    - edges: AUTHOR --SHARED--> DOMAIN based on posts
    """
    df = _posts_df.copy()
    if "author" not in df.columns or "domain" not in df.columns:
        raise HTTPException(status_code=500, detail="author/domain columns missing from posts")

    # filters
    if subreddit:
        df = df[df["subreddit"].fillna("").str.lower() == subreddit.lower()]
    if keyword:
        kw = keyword.lower()
        df = df[
            (df["title"].str.lower().str.contains(kw)) |
            (df["selftext"].str.lower().str.contains(kw))
        ]

    df = df.dropna(subset=["author", "domain"])
    if df.empty:
        return {"nodes": [], "edges": []}

    # top authors & domains
    author_counts = df["author"].value_counts()
    domain_counts = df["domain"].value_counts()

    max_authors = 300
    max_domains = 100


    top_authors = author_counts.head(max_authors).index.tolist()
    top_domains = domain_counts.head(max_domains).index.tolist()
    self_domains = df[df['domain'].str.startswith('self.', na=False)]['domain'].unique().tolist()
    target_domains = list(set(top_domains + self_domains))
    df_sub = df[df["author"].isin(top_authors) & df["domain"].isin(target_domains)]
    if df_sub.shape[0] > limit * 3:
        df_sub = df_sub.sample(limit * 3, random_state=42)

    # ---- domain score lookup from CSV ----
    # assumes _domain_scores_df has columns: 'domain', 'score'
    domain_score_lookup = {
        str(row["domain"]).lower(): float(row["score"])
        for _, row in _domain_scores_df.iterrows()
        if pd.notna(row.get("domain"))
    }

    # build nodes
    nodes = []
    node_ids = set()

    def add_node(node_id, label, node_type):
        if node_id in node_ids:
            return
        node_ids.add(node_id)

        node = {
            "id": node_id,
            "label": str(label),
            "type": node_type,
        }

        # only domains get credibility score
        if node_type == "domain":
            key = str(label).lower()
            node["credScore"] = domain_score_lookup.get(key)

        nodes.append(node)

    for a in top_authors:
        add_node(f"author:{a}", a, "author")
    for d in top_domains:
        add_node(f"domain:{d}", d, "domain")

    # build edges
    edges = []
    seen_edges = set()
    for _, row in df_sub.iterrows():
        a = str(row["author"])
        d = str(row["domain"])
        src = f"author:{a}"
        tgt = f"domain:{d}"
        key = (src, tgt)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({
            "source": src,
            "target": tgt,
            "rel": "SHARED",
        })
        if len(edges) >= limit:
            break

    return {"nodes": nodes, "edges": edges}



# -------------------------------------------------------------------
# Existing simple AI summary endpoint (per post_ids)
# -------------------------------------------------------------------

@app.post("/api/ai-summary")
def ai_summary(post_ids: List[str], prompt_template: str = "Summarize these posts"):
    df = _posts_df[_posts_df["id"].isin(post_ids)]
    sample_texts = (df["title"].fillna("") + " " + df["selftext"].fillna("")).tolist()
    result = summarize_posts_with_openai(sample_texts, prompt_template)
    return result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("ai_server.main:app", host="0.0.0.0", port=8000, reload=True)
