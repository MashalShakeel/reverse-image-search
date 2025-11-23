import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from embed import compute_embedding
from search import search_top_k
from db import IMAGE_DIR, save_embedding, load_all_embeddings

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"**/*{e}"), recursive=True))
    return sorted(files)

def build_database(model_type="resnet50"):
    paths = list_images(IMAGE_DIR)
    print("Found images:", len(paths))

    for p in paths:
        img = Image.open(p).convert("RGB")
        vec = compute_embedding(img, model_type=model_type)
        save_embedding(p, vec)

    print(f"Database updated! (Embeddings using {model_type})")


def search_demo(query_image_path, model_type="resnet50", top_k=5):
    q_img = Image.open(query_image_path).convert("RGB")
    q_vec = compute_embedding(q_img, model_type=model_type)

    docs = load_all_embeddings()
    all_vecs = [np.array(d["embedding"], dtype="float32") for d in docs]
    all_paths = [d["path"] for d in docs]

    idxs, scores = search_top_k(q_vec, all_vecs, k=top_k)
    result_paths = [all_paths[i] for i in idxs]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, top_k, 1)
    plt.imshow(q_img)
    plt.title("Query")
    plt.axis("off")

    for i, p in enumerate(result_paths, start=2):
        plt.subplot(2, top_k, i)
        plt.imshow(Image.open(p))
        plt.title(f"Match")
        plt.axis("off")

    plt.show()
    print("Results:", list(zip(result_paths, scores)))


if __name__ == "__main__":
    while True:
        model_type = input("Select embedding model ('resnet50' or 'hist'): ").strip().lower()
        if model_type in ("resnet50", "hist"):
            break
        print("Invalid choice. Please type 'resnet50' or 'hist'.")

    build_database(model_type=model_type)

    all_images = list_images(IMAGE_DIR)
    from collections import defaultdict
    cat_dict = defaultdict(list)
    for path in all_images:
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        prefix = ''.join(c for c in base if not c.isdigit()).rstrip('_.')
        cat_dict[prefix.lower()].append(path)

    categories = sorted(cat_dict.keys())
    print("\nAvailable categories:")
    for cat in categories:
        print(f"- {cat} ({len(cat_dict[cat])} images)")

    while True:
        chosen_cat = input("\nEnter a category for the query image: ").strip().lower()
        if chosen_cat in cat_dict:
            query_path = cat_dict[chosen_cat][0]
            print(f"Selected query image: {os.path.basename(query_path)}")
            break
        print("Category not found. Please enter one of the available categories.")

    search_demo(query_path, model_type=model_type, top_k=5)
