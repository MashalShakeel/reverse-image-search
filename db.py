from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "image_search_db"
COLLECTION_NAME = "images"
IMAGE_DIR = "./images"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def save_embedding(path, embedding):
    doc = {
        "path": path,
        "embedding": embedding.tolist()
    }
    collection.update_one({"path": path}, {"$set": doc}, upsert=True)

def load_all_embeddings():
    return list(collection.find({}))
