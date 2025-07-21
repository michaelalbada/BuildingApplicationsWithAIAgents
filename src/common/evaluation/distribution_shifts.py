import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Example 1: Kolmogorov-Smirnov (KS) Test for numerical feature drift (e.g., query lengths)
def detect_ks_drift(historical_data: np.ndarray, current_data: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Detects distribution shift using KS test.
    Returns True if significant drift (statistic > threshold).
    """
    ks_stat, p_value = stats.ks_2samp(historical_data, current_data)
    print(f"KS Statistic: {ks_stat}, p-value: {p_value}")
    return ks_stat > threshold

# Sample usage
historical_lengths = np.array([len(q) for q in ["What is the weather?", "Book a flight to Paris", "Recommend a book"] * 100])
current_lengths = np.array([len(q) for q in ["Query about latest AI news", "Longer user input with details"] * 150])
if detect_ks_drift(historical_lengths, current_lengths):
    print("Drift detected: Review input changes.")

# Example 2: Kullback-Leibler (KL) Divergence for probability shifts (e.g., token distributions)
def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Computes KL divergence between two probability distributions.
    Add epsilon to avoid division by zero.
    """
    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

# Sample usage (token frequency histograms)
historical_tokens = np.bincount([ord(c) for q in ["hello world"] * 100 for c in q])  # Simplified token counts
current_tokens = np.bincount([ord(c) for q in ["hola mundo"] * 100 for c in q])
kl_score = kl_divergence(historical_tokens, current_tokens)
print(f"KL Divergence: {kl_score}")
if kl_score > 0.5:
    print("Concept drift detected: Possible language shift.")

# Example 3: Population Stability Index (PSI) for categorical metrics (e.g., tool usage)
def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Computes PSI for categorical or binned continuous data.
    """
    expected_percents = expected / np.sum(expected)
    actual_percents = actual / np.sum(actual)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)

# Sample usage (tool invocation counts)
historical_tools = np.array([50, 30, 20])  # e.g., counts for 'refund', 'cancel', 'modify'
current_tools = np.array([40, 40, 20])
psi = calculate_psi(historical_tools, current_tools)
print(f"PSI: {psi}")
if psi > 0.1:
    print("Minor drift in tool usage.") 
elif psi > 0.25:
    print("Major drift: Intervention required.")

# Example 4: Embedding-based Similarity for Query Drift
def detect_embedding_drift(historical_queries: list, current_queries: list, threshold: float = 0.8):
    """
    Computes average cosine similarity between query embeddings.
    Uses TF-IDF for simplicity; replace with sentence transformers for better semantics.
    """
    vectorizer = TfidfVectorizer()
    all_queries = historical_queries + current_queries
    embeddings = vectorizer.fit_transform(all_queries)
    hist_emb = embeddings[:len(historical_queries)]
    curr_emb = embeddings[len(historical_queries):]
    similarities = cosine_similarity(curr_emb, hist_emb)
    mean_sim = np.mean(similarities)
    print(f"Mean Cosine Similarity: {mean_sim}")
    return mean_sim < threshold

# Sample usage
historical = ["Refund my order", "Cancel shipment", "Change address"] * 50
current = ["Return damaged item", "Stop delivery now", "Update shipping info"] * 50
if detect_embedding_drift(historical, current):
    print("Query drift detected: Retrain or adapt prompts.")