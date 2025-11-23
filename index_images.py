import os
import glob
from PIL import Image
from embed import compute_embedding
from db import IMAGE_DIR, save_embedding, load_all_embeddings

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"**/*{e}"), recursive=True))
    return sorted(files)

def build_database():
    paths = list_images(IMAGE_DIR)
    print("Found images:", len(paths))

    # load existing paths from MongoDB
    docs = load_all_embeddings()
    existing_paths = set(d["path"] for d in docs)

    # only indexing new images saves time.
    new_paths = [p for p in paths if p not in existing_paths]
    print("New images to index:", len(new_paths))

    for p in new_paths:
        img = Image.open(p).convert("RGB")
        vec = compute_embedding(img)
        save_embedding(p, vec)

    print("Database updated!")

if __name__ == "__main__":
    build_database()

