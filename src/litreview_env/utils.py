"""Utility functions for text similarity, JSON parsing, and validation.

Provides multiple similarity metrics designed to handle paraphrased content
from LLMs rather than requiring exact character-level matches.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Common academic stopwords to ignore in similarity
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that", "this",
    "which", "what", "who", "whom", "these", "those", "it", "its", "we",
    "they", "them", "their", "our", "your", "his", "her", "he", "she",
    "also", "about", "up", "down",
})


def safe_parse_json(text: str) -> Optional[dict[str, Any]]:
    """Attempt to parse JSON from text, handling common formatting issues.

    Tries to extract a JSON object from text that might contain markdown
    code fences, trailing commas, etc.

    Args:
        text: Raw text that may contain a JSON object.

    Returns:
        Parsed dict, or None if parsing fails.
    """
    # Strip whitespace
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try removing trailing commas (common LLM issue)
    if brace_start != -1 and brace_end != -1:
        candidate = text[brace_start : brace_end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_content_words(text: str, min_len: int = 2) -> list[str]:
    """Extract content words (non-stopwords) from normalized text."""
    words = normalize_text(text).split()
    return [w for w in words if len(w) >= min_len and w not in _STOPWORDS]


def _get_ngrams(words: list[str], n: int) -> list[str]:
    """Generate n-grams from a word list."""
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


# ---------------------------------------------------------------------------
# Core similarity metrics
# ---------------------------------------------------------------------------

def sequence_similarity(text_a: str, text_b: str) -> float:
    """Character-level SequenceMatcher ratio."""
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def word_jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity using content words only."""
    words_a = set(_get_content_words(text_a))
    words_b = set(_get_content_words(text_b))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def word_recall(predicted: str, reference: str) -> float:
    """Fraction of reference content words found in predicted text.

    This is more lenient than Jaccard — it doesn't penalize the model
    for adding extra relevant information.
    """
    ref_words = set(_get_content_words(reference))
    pred_words = set(_get_content_words(predicted))
    if not ref_words:
        return 1.0
    if not pred_words:
        return 0.0
    return len(ref_words & pred_words) / len(ref_words)


def rouge_l(predicted: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence of content words.

    Much better than SequenceMatcher for paraphrased content because it
    works at word level and allows word reordering within the LCS.
    """
    pred_words = _get_content_words(predicted)
    ref_words = _get_content_words(reference)

    if not pred_words or not ref_words:
        return 0.0

    # Compute LCS length using DP
    m, n = len(ref_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """N-gram overlap F1 score (like ROUGE-N)."""
    words_a = _get_content_words(text_a)
    words_b = _get_content_words(text_b)
    ngrams_a = Counter(_get_ngrams(words_a, n))
    ngrams_b = Counter(_get_ngrams(words_b, n))

    if not ngrams_a or not ngrams_b:
        return 0.0

    overlap = sum((ngrams_a & ngrams_b).values())
    precision = overlap / sum(ngrams_a.values()) if sum(ngrams_a.values()) else 0
    recall = overlap / sum(ngrams_b.values()) if sum(ngrams_b.values()) else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_similarity_tfidf(text_a: str, text_b: str) -> float:
    """Simple TF-based cosine similarity on content words."""
    words_a = _get_content_words(text_a)
    words_b = _get_content_words(text_b)

    if not words_a or not words_b:
        return 0.0

    tf_a = Counter(words_a)
    tf_b = Counter(words_b)
    all_words = set(tf_a.keys()) | set(tf_b.keys())

    dot_product = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v * v for v in tf_a.values()))
    mag_b = math.sqrt(sum(v * v for v in tf_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Composite similarity (main function used by graders)
# ---------------------------------------------------------------------------

def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute a robust text similarity score between 0 and 1.

    Uses a weighted blend of multiple metrics to handle paraphrasing:
    - ROUGE-L (word-level LCS) — captures meaning despite word reordering
    - Word recall — doesn't penalize adding extra information
    - Cosine similarity — captures term frequency overlap
    - Sequence similarity — character-level backup

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity score in [0, 1].
    """
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0

    rl = rouge_l(text_a, text_b)
    wr = word_recall(text_a, text_b)
    cs = cosine_similarity_tfidf(text_a, text_b)
    ss = sequence_similarity(text_a, text_b)
    bi = ngram_overlap(text_a, text_b, n=2)

    # Weighted blend: prioritize word-level metrics over character-level
    score = (0.30 * rl + 0.25 * wr + 0.20 * cs + 0.15 * bi + 0.10 * ss)
    return min(1.0, score)


def keyword_overlap(predicted: str, reference: str, min_word_len: int = 3) -> float:
    """Compute keyword overlap between predicted and reference text.

    Uses recall-oriented scoring: what fraction of reference keywords
    appear in the prediction.

    Args:
        predicted: The agent's text.
        reference: The reference text.
        min_word_len: Minimum word length to consider as a keyword.

    Returns:
        Overlap score in [0, 1].
    """
    pred_words = set(
        w for w in _get_content_words(predicted) if len(w) >= min_word_len
    )
    ref_words = set(
        w for w in _get_content_words(reference) if len(w) >= min_word_len
    )
    if not ref_words:
        return 1.0 if not pred_words else 0.0

    # Use recall (what fraction of reference keywords are in prediction)
    # rather than Jaccard (which penalizes extra good info)
    recall = len(pred_words & ref_words) / len(ref_words)

    # Also compute Jaccard but weight recall more
    union = pred_words | ref_words
    jaccard = len(pred_words & ref_words) / len(union) if union else 0.0

    return 0.7 * recall + 0.3 * jaccard


def list_coverage(predicted: list[str], reference: list[str], threshold: float = 0.35) -> float:
    """Score how well a predicted list covers a reference list.

    For each reference item, finds the best-matching predicted item.
    Uses a blend of binary coverage (above threshold) and continuous
    similarity for a smoother gradient.

    Args:
        predicted: The agent's list of items.
        reference: The ground-truth list.
        threshold: Minimum similarity to count as a match.

    Returns:
        Coverage score in [0, 1].
    """
    if not reference:
        return 1.0
    if not predicted:
        return 0.0

    total_sim = 0.0
    covered = 0
    for ref_item in reference:
        best_sim = max(
            (semantic_similarity(pred_item, ref_item) for pred_item in predicted),
            default=0.0,
        )
        # Blend: binary coverage (50%) + continuous similarity (50%)
        if best_sim >= threshold:
            covered += 1
        total_sim += best_sim

    binary_score = covered / len(reference)
    continuous_score = total_sim / len(reference)

    # Blend binary coverage with continuous similarity for smoother gradient
    return 0.5 * binary_score + 0.5 * continuous_score


def check_paper_ids_grounded(
    text: str, valid_ids: list[str]
) -> tuple[float, list[str]]:
    """Check that paper ID references in text are grounded in the corpus.

    Args:
        text: The agent's response text.
        valid_ids: The list of valid paper IDs from the corpus.

    Returns:
        Tuple of (grounding_score, hallucinated_ids).
    """
    # Find all paper ID references in the text
    mentioned_ids = set(re.findall(r"P\d{3}", text))
    valid_set = set(valid_ids)

    if not mentioned_ids:
        return 0.5, []  # No references is neither good nor bad

    hallucinated = [pid for pid in mentioned_ids if pid not in valid_set]

    grounding = 1.0 - (len(hallucinated) / len(mentioned_ids))
    return grounding, hallucinated
