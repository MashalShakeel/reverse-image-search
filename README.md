**Mashal Shakeel**  
**FA23-BCS-187**  
**Lab task**

---

## Reverse Image Search

The lab task implements a simple reverse image search system. It allows user to query an image and find the most visually similar images in a dataset.  
The system uses:

- **PIL** to load images  
- **NumPy** to compute image embeddings (color histogram or modern CNN embeddings)  
- **MongoDB** to store and fetch image embeddings  
- **Matplotlib** to display results  
- **Cosine similarity** (pure NumPy) for similarity search  

---

## Features

1. Build and store embeddings of all images in a folder to MongoDB.
2. Query by example: input a **category name** (like `cat`, `car`, `tree`) and find similar images.
3. Displays query and top-K matches in a neat grid.
4. Flexible: works with color histograms (simple) or pre-trained embedding (modern).

---

## Project Structure

reverse-image-search/

│

├─ images/ 

├─ main.py 

├─ index_images.py

├─ embed.py

├─ search.py

├─ db.py

└─ README.md 

---

## How to Run

### 1. Install dependencies:

```bash
pip install pillow numpy matplotlib pymongo torch torchvision tqdm
```

If you already have pymongo installed, make sure it is updated:
```bash
pip install --upgrade pymongo
```

- Start MongoDB (make sure it's running locally).
- Add images to the images/ folder.
- Index images to MongoDB:

```commandline
python index_images.py
```
- Run the main search program:

```commandline
python main.py
```
Follow the prompts:
- Select an embedding model
- The script will show available categories.
- Enter a category name to select a query image.

Top-K similar images will be displayed.

## Notes
- Images are stored and retrieved from MongoDB to avoid recomputing embeddings every run.
- The first run of the main program builds the database automatically.
- To change the query image, simply select and type a different category.

## Quality Check (Embeddings Comparison)

**Query:** car

### 1. ResNet50 embeddings (deep features):
- **Observation:** The network captures semantic similarity, so all top matches are cars, even if the color, angle, or background differs.
- **Interpretation:** ResNet50 embeddings successfully ignore irrelevant objects and focus on the main object in the image.

### 2. HSV Color Histogram embeddings (color-based):
- **Observation:** Some top matches are not cars (e.g., cat1.jpg, tree1.jpg) because color distribution is similar.
- **Interpretation:** Histogram embeddings only capture color similarity, not semantic content. 
Thus, objects of a different category but similar colors appear as matches.

### Overall Analysis:
- ResNet50 embeddings provide accurate semantic matching and are suitable for real reverse image search.
- Histogram embeddings are useful as a lightweight baseline, but results can include false positives from unrelated objects with similar colors.
- **ResNet50 vs Histogram:** ResNet50 is better.