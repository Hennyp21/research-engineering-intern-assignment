import json, csv, hashlib
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict, Counter


RAW_JSONL = Path("data/raw/reddit_data.jsonl")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = PROCESSED_DIR / "clean_reddit_10cols.csv"
DOMAIN_SCORES = PROCESSED_DIR / "domain_scores.csv"
AUTHOR_FINGERPRINTS = PROCESSED_DIR / "author_fingerprints.csv"
TOP_URL_CASCADE = PROCESSED_DIR / "top_url_cascade.csv"
AI_PROMPT_FILE = PROCESSED_DIR / "ai_summary_prompt.txt"

# --------------------------------------------------------

wanted = ["id","subreddit","author","title","selftext","created_utc","ups","num_comments","domain","url"]

# Write cleaned CSV streaming
with open(OUT_CSV, "w", encoding="utf-8", newline='') as outf:
    writer = csv.DictWriter(outf, fieldnames=wanted)
    writer.writeheader()

    domain_counts = Counter()
    url_counts = Counter()
    author_stats = defaultdict(lambda: {"count":0, "times":[], "link_posts":0, "texts_counter":Counter()})
    total_rows = 0

    with open(RAW_JSONL, "r", encoding="utf-8") as inf:
        for raw in inf:
            line = raw.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                data = obj.get("data", obj)
            except:
                continue

            row = {k: data.get(k, "") for k in wanted}

            # Clean timestamp
            try:
                if row.get("created_utc") not in ("", None):
                    row["created_utc"] = int(float(row["created_utc"]))
                else:
                    row["created_utc"] = ""
            except:
                row["created_utc"] = ""

            # Infer domain
            if not row.get("domain") and row.get("url"):
                try:
                    row["domain"] = urlparse(row["url"]).netloc or ""
                except:
                    row["domain"] = ""

            writer.writerow(row)
            total_rows += 1

            # Aggregations
            domain_counts[row["domain"]] += 1
            url_counts[row["url"]] += 1
            auth = row.get("author","")

            if auth:
                author_stats[auth]["count"] += 1
                if row["created_utc"] != "":
                    author_stats[auth]["times"].append(row["created_utc"])
                if row["url"]:
                    author_stats[auth]["link_posts"] += 1

                text = (row["title"] or "") + " " + (row["selftext"] or "")
                h = hashlib.md5(text.strip().encode()).hexdigest()
                author_stats[auth]["texts_counter"][h] += 1

# Domain scoring
#1. Exact Trusted Domains (Automatic High Score)
trusted_domains = {
    "fortune.com", "economist.com", "wsj.com", "bloomberg.com",
    "ft.com", "reuters.com", "apnews.com", "npr.org", "pbs.org",
    "bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "usatoday.com", "politico.com", "thehill.com",
    "wired.com", "techcrunch.com", "theverge.com", "arstechnica.com",
    "nature.com", "sciencemag.org", "scientificamerican.com",
    "foreignpolicy.com", "theatlantic.com", "newyorker.com",
    "propublica.org", "pewresearch.org", "snopes.com"
}

# 2. Positive Keywords (Partial Match)
# If the domain contains these, it gets a boost
news_keywords = [
    "news", "times", "chronicle", "gazette", "post", "herald",
    "tribune", "journal", "press", "daily", "observer", "independent",
    "media", "report", "insider", "monitor", "telegraph", "broadcast",
    "gov", "edu", "abc", "nbc", "cbs", "cnn", "fox", "msnbc"
]

# 3. Negative Keywords / Patterns
spam_keywords = ["crypto", "bestbuy", "cheap", "free", "promo", "xxx", "porn"]

with open(DOMAIN_SCORES, "w", encoding="utf-8", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["domain", "count", "score", "reasons"])
    writer.writeheader()

    for dom, cnt in domain_counts.most_common():
        dom_low = dom.lower()
        score = 0
        reasons = []

        # --- SCORING RULES ---

        # Rule A: Trusted Extensions (+2)
        if dom_low.endswith(".gov") or dom_low.endswith(".edu") or dom_low.endswith(".mil"):
            score += 2
            reasons.append("Gov/Edu/Mil (+2)")

        # Rule B: Exact Trusted List (+3) - Takes priority
        if dom_low in trusted_domains or f"www.{dom_low}" in trusted_domains:
            score += 3
            reasons.append("Trusted Source (+3)")
        
        # Rule C: News Keyword Match (+1) - Only if not already trusted
        elif any(kw in dom_low for kw in news_keywords):
            score += 1
            reasons.append("News Keyword (+1)")

        # Rule D: Spam/Suspicious (-2)
        if any(kw in dom_low for kw in spam_keywords):
            score -= 2
            reasons.append("Spam Keyword (-2)")

        # Rule E: URL Shorteners (-1)
        if dom_low in ("bit.ly", "t.co", "goo.gl", "tinyurl.com", "is.gd"):
            score -= 2
            reasons.append("URL Shortener (-2)")

        # Rule F: Extremely Long Domains (-1)
        # Valid news sites are usually short. "best-crypto-news-daily-update.com" is suspicious.
        if len(dom_low) > 35 and "blogspot" not in dom_low and "wordpress" not in dom_low:
            score -= 1
            reasons.append("Long Domain Name (-1)")

        # --- FINAL SCORE CLAMPING ---
        # We cap the score between -2 (Bad) and +3 (Excellent)
        # In your graph: 
        #   -1 or lower = RED
        #   0           = ORANGE (Neutral/Unknown)
        #   1 or higher = GREEN
        
        final_score = max(-2, min(3, score))

        writer.writerow({
            "domain": dom,
            "count": cnt,
            "score": final_score,
            "reasons": "; ".join(reasons)
        })

# Author fingerprints
with open(AUTHOR_FINGERPRINTS, "w", encoding="utf-8", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "author","post_count","avg_time_between_posts_sec",
        "percent_link_posts","duplicate_text_ratio"
    ])
    writer.writeheader()

    for auth, stats in sorted(author_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        cnt = stats["count"]
        times = sorted(stats["times"])
        avg_delta = ""

        if len(times) >= 2:
            deltas = [t2 - t1 for t1,t2 in zip(times,times[1:]) if (t2 - t1) > 0]
            if deltas:
                avg_delta = sum(deltas)/len(deltas)

        pct_links = stats["link_posts"]/cnt if cnt > 0 else 0
        total_texts = sum(stats["texts_counter"].values())
        unique_texts = len(stats["texts_counter"])
        dup_ratio = 1 - (unique_texts/total_texts) if total_texts else 0

        writer.writerow({
            "author": auth,
            "post_count": cnt,
            "avg_time_between_posts_sec": avg_delta,
            "percent_link_posts": pct_links,
            "duplicate_text_ratio": dup_ratio
        })

# Top URL cascade
if url_counts:
    top_url = url_counts.most_common(1)[0][0]

    with open(TOP_URL_CASCADE, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id","author","created_utc","title","selftext"])
        writer.writeheader()

        with open(RAW_JSONL, "r", encoding="utf-8") as inf:
            for raw in inf:
                data = json.loads(raw).get("data", {})
                if data.get("url") == top_url:
                    writer.writerow({
                        "id": data.get("id",""),
                        "author": data.get("author",""),
                        "created_utc": data.get("created_utc",""),
                        "title": data.get("title",""),
                        "selftext": data.get("selftext","")
                    })

# AI prompt template
with open(AI_PROMPT_FILE, "w") as f:
    f.write("You are an analyst... (summary prompt here)")
