import numpy as np

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def search_top_k(query_vec, all_vecs, k=5):
    sims = [cosine_similarity(query_vec, v) for v in all_vecs]
    idx = np.argsort(sims)[::-1][:k]
    return idx, [sims[i] for i in idx]
