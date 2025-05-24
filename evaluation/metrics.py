import json, pathlib, importlib, argparse, statistics as stats
from typing import Dict, List
import sacrebleu, torch
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer("all-MiniLM-L6-v2")

def parse_weights(w_pairs: List[str]) -> Dict[str, float]:
    out = {}
    for pair in w_pairs:
        k, v = pair.split("=", 1)
        out[k.lower()] = float(v)
    return out

def accuracy(match, exp): 
    return float(match.strip() == exp.strip())

def bleu(pred, ref):
    return sacrebleu.corpus_bleu([pred], [[ref]]).score / 100.0

def bert(pred, ref):
    P, R, F = bert_score([pred], [ref], lang="en", rescale_with_baseline=True)
    return F.mean().item()          # value between 0-1

def cosine(pred, ref):
    # returns a value between –1 … 1
    # Can also be used with any embedding model
    emb = st_model.encode([pred, ref], convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb[0], emb[1]).item()

METRICS = {"accuracy": accuracy, "bleu": bleu, "bertscore": bert, "cosine": cosine}
