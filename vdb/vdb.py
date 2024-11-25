import os
import re
import spacy
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")


def filter_text(text):
    text = re.sub(r"==.*?==", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def split_into_sentences_spacy(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    combined_sentences = [
        " ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)
    ]
    return combined_sentences


def read_and_split_txt_files(directory):
    sentences = []
    sentence_to_file_map = []
    
    # 使用 tqdm 包裹 os.listdir 以显示文件处理进度
    for filename in tqdm(os.listdir(directory), desc="Processing files", unit="file"):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                file_sentences = split_into_sentences_spacy(filter_text(text))
                sentences.extend(file_sentences)
                sentence_to_file_map.extend([filename] * len(file_sentences))
    
    return sentences, sentence_to_file_map


def build_faiss_index(sentences, model_name="all-MiniLM-L6-v2"):
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(model_name)
    
    print("Calculating sentence embeddings...")
    # Use tqdm to show progress during embedding computation
    embeddings = []
    batch_size = 64  # Adjust this based on available memory
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    print(f"Computed embeddings for {len(sentences)} sentences.")
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    
    return index, model, embeddings  # 返回 embeddings


def search_faiss_index(query, index, model, sentences, sentence_to_file_map, k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "sentence": sentences[idx],
            "file": sentence_to_file_map[idx],
            "distance": dist
        })
    return results


def save_faiss_index(index, embeddings, sentence_to_file_map, sentences, index_file="faiss_index.index", metadata_file="metadata.pkl"):
    faiss.write_index(index, index_file)
    print(f"FAISS index saved in {index_file}")
    
    metadata = {
        "embeddings": embeddings,
        "sentence_to_file_map": sentence_to_file_map,
        "sentences": sentences
    }
    with open(metadata_file, "wb") as file:
        pickle.dump(metadata, file)
    print(f"Metadata saved in {metadata_file}")


def load_faiss_index(index_file="faiss_index.index", metadata_file="metadata.pkl"):
    index = faiss.read_index(index_file)
    print(f"FAISS index loaded from {index_file}")
    
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)
    print(f"Metadata loaded from {metadata_file}")
    
    return index, metadata


if __name__ == "__main__":
    directory_path = "wiki"
    index_file = "faiss_index3.index"
    metadata_file = "metadata3.pkl"
    
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        # 如果文件存在，则加载索引和元数据
        index, metadata = load_faiss_index(index_file, metadata_file)
        sentences = metadata["sentences"]
        sentence_to_file_map = metadata["sentence_to_file_map"]
        model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        # 如果文件不存在，则重新构建索引
        sentences, sentence_to_file_map = read_and_split_txt_files(directory_path)
        index, model, embeddings = build_faiss_index(sentences)  # 修复解包
        save_faiss_index(index, embeddings, sentence_to_file_map, sentences, index_file, metadata_file)
    
    query = "HIV rash"
    results = search_faiss_index(query, index, model, sentences, sentence_to_file_map)
    for result in results:
        print(f"Sentence: {result['sentence']}")
        print(f"File: {result['file']}")
        print(f"Distance: {result['distance']}")
        print("---")
