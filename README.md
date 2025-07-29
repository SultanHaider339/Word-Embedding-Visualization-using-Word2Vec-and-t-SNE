# 🧠 Word Embedding Visualization using Word2Vec and t-SNE

This project demonstrates the use of **Word2Vec** for learning word embeddings from raw text and visualizing those embeddings in **3D space** using **t-SNE** and **Plotly**.

---

## 📂 Dataset

- **Input**: A plain text file (`Assignmentdataset.txt`) containing raw sentences.
- **Loading**: The text is read linearly and split by full stops (`.`) into individual sentences.
- **Preprocessing**:
  - Whitespace trimming
  - Removal of numeric and special characters
  - Tokenization using `nltk.word_tokenize`

---

## 🔧 Preprocessing Steps

The preprocessing ensures the text is cleaned and tokenized for Word2Vec training.

### Cleaning & Tokenization

```python
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return word_tokenize(text.lower())
