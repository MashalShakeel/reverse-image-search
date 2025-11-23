# main.py
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

# Build DB on first run
def build_database():
    paths = list_images(IMAGE_DIR)
    print("Found images:", len(paths))

    for p in paths:
        img = Image.open(p).convert("RGB")
        vec = compute_embedding(img)
        save_embedding(p, vec)

    print("Database updated!")

# Query demo
def search_demo(query_image_path, top_k=5):
    q_img = Image.open(query_image_path).convert("RGB")
    q_vec = compute_embedding(q_img)

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
    # Build the database
    build_database()

    # List all images
    all_images = list_images(IMAGE_DIR)
    print("\nImages available in folder:")
    for i, img_path in enumerate(all_images, start=1):
        print(f"{i}. {os.path.basename(img_path)}")

    # Ask user to pick one by name
    while True:
        img_name = input("\nEnter the filename of the query image: ").strip()
        query_path = os.path.join(IMAGE_DIR, img_name)
        if os.path.exists(query_path):
            break
        print("File not found. Please enter a valid filename from the list above.")

    # Run the search
    search_demo(query_path, top_k=5)

