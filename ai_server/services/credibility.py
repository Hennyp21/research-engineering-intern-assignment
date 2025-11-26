import pandas as pd
import numpy as np

def load_data():
    """Loads the processed CSVs."""
    try:
        authors_df = pd.read_csv("data/processed/author_fingerprints.csv")
        domains_df = pd.read_csv("data/processed/domain_scores.csv")
        return authors_df, domains_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run the processing script first.")
        return None, None

def get_author_credibility(author_name, authors_df):
    """
    Calculates a credibility score (0-100) for a Reddit author.
    
    Factors:
    - Account activity (Too high frequency = suspicious)
    - Content originality (High duplicate text = spam)
    - Link spamming (100% links = likely bot/farmer)
    """
    # 1. Find the author
    row = authors_df[authors_df['author'] == author_name]
    
    if row.empty:
        return None, "Author not found in database."
    
    row = row.iloc[0]
    
    # 2. Extract metrics
    post_count = row['post_count']
    avg_time = row['avg_time_between_posts_sec']
    percent_links = row['percent_link_posts']
    dup_ratio = row['duplicate_text_ratio']
    
    # 3. Base Score
    score = 100
    reasons = []

    # -- Penalty: Bot-like Speed --
    # If avg time between posts is < 5 minutes (300s) and they have posted a lot
    if pd.notna(avg_time) and post_count > 5:
        if avg_time < 60: # 1 minute
            score -= 50
            reasons.append("Extremely high frequency posting (Bot-like)")
        elif avg_time < 300: # 5 minutes
            score -= 20
            reasons.append("Very high frequency posting")

    # -- Penalty: Content Farm / Reposting --
    # If more than 50% of their content text is duplicated
    if dup_ratio > 0.5:
        penalty = int(dup_ratio * 40) # Max 40 points penalty
        score -= penalty
        reasons.append(f"High repetition of content ({int(dup_ratio*100)}% duplicate)")

    # -- Penalty: Link Spam --
    # If they only post links (no self-text discussions) and have posted multiple times
    if percent_links > 0.9 and post_count > 3:
        score -= 20
        reasons.append("Almost exclusively posts links")

    # -- Bonus: Engagement (Implicit) --
    # If they have posted a moderate amount (showing they are active but not spamming)
    if 5 <= post_count <= 50:
        score += 5 # Slight trust for established history
    
    # Clamp score between 0 and 100
    final_score = max(0, min(100, score))
    
    return final_score, reasons

def get_domain_credibility(domain_name, domains_df):
    """
    Calculates a credibility score (0-100) for a domain based on heuristic scoring.
    """
    # 1. Find the domain
    row = domains_df[domains_df['domain'] == domain_name]
    
    if row.empty:
        return 50, ["Unknown domain (Neutral start)"] # Default for unknown
    
    row = row.iloc[0]
    raw_score = row['score'] # This is from your previous script (-2 to +2 usually)
    
    # 2. Convert Raw Score to 0-100 Scale
    # We assume raw_score ranges roughly from -3 (bad) to +3 (good)
    
    base_score = 50 # Start neutral
    
    # Each point of raw score is worth 15 points on the 100 scale
    final_score = base_score + (raw_score * 15)
    
    # Extract reasons string from CSV
    reasons = [row['reasons']] if pd.notna(row['reasons']) and row['reasons'] else []
    
    # Cap limits
    final_score = max(10, min(95, final_score)) # Never give 0 or 100 purely on heuristics
    
    return int(final_score), reasons