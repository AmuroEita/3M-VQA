import os
import re
import argparse
import pandas as pd
import base64
import torch
from io import BytesIO
from PIL import Image
from caption.load_caption import load_caption
from rag.vdb import load_faiss_index, search_faiss_index
from sentence_transformers import SentenceTransformer
from inference import question_prompt1, question_prompt2, question_prompt3, question_prompt4, extract_abcd_or_default, get_answer
from transformers import MllamaForConditionalGeneration, AutoProcessor
from utils.gpt import get_option
from google.cloud import translate_v2 as translate

client = translate.Client()

datasets_map = {
    'data/brazil_local_processed.tsv': 0,
    'data/israel_local_processed.tsv': 1,
    'data/japan_local_processed.tsv': 2,
    'data/spain_local_processed.tsv': 3
}

index_file = "faiss_index.index"
metadata_file = "metadata.pkl"

question_text = question_prompt1 + question_prompt4
ds_name = ""

model_path = "Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)
print(f"{model_path} loaded ")

index, metadata = None, None
if os.path.exists(index_file) and os.path.exists(metadata_file):
    index, metadata = load_faiss_index(index_file, metadata_file)
    sentences = metadata["sentences"]
    sentence_to_file_map = metadata["sentence_to_file_map"]
    st_model = SentenceTransformer("all-MiniLM-L6-v2")


def translate_text(text, target_language="en"):
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]
  

def load_results(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()


def display_caption(dataset, idx):
    captions = load_caption()
    file_idx = datasets_map[dataset]
    global question_text
    question_text += captions[file_idx][idx]
    print(f"\nImage Analysis : {captions[file_idx][idx]}")
    
    
def clean_text(text):
    text = re.sub(r"\s*==.*?==\s*", "", text)
    text = re.sub(r" {3,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    lines = text.splitlines()
    merged_sentences = []
    current_sentence = ""
    
    for line in lines:
        line = line.strip()  
        if not line:  
            continue
        if re.search(r'[.!?]$', line):
            current_sentence += " " + line
            merged_sentences.append(current_sentence.strip())
            current_sentence = ""  
        else:
            current_sentence += " " + line  

    if current_sentence:
        merged_sentences.append(current_sentence.strip())
    
    cleaned_text = "\n".join(merged_sentences)
    return cleaned_text.strip()


def display_rag(text):
    results = search_faiss_index(text, index, st_model, sentences, sentence_to_file_map)
    
    rag_text = ""
    for result in results:
        rag_text += clean_text(result['sentence'])
    global question_text
    question_text += rag_text
    print(f"\nBackground Knowledge : {rag_text}")


def display_question(df, question_idx, dataset):
    """Display details of a specific question"""
    filtered_df = df[df["index"] == question_idx]
    if filtered_df.empty:
        print(f"No question found with index {question_idx}.")
        return

    row = filtered_df.iloc[0]
    display_rag("row['question']")
    display_caption(dataset, question_idx)
    global question_text
    question_text += translate_text(row['question'])
    print(f"\nQuestion {question_idx}: {row['question']}")
    print(f"\nOptions:")
    for option in ['A', 'B', 'C', 'D']:
        if pd.notna(row.get(option)):
            print(f"{option}: {row[option]}")
            question_text += translate_text(f"{option}: {row[option]}")
    print(f"\nCorrect Answer: {row['correct_option']}")

    if "image" in row and pd.notna(row["image"]):
        try:
            image_data = base64.b64decode(row["image"])
            image = Image.open(BytesIO(image_data))
            image.show(title=f"Image for Question {question_idx}")
        except Exception as e:
            print(f"Failed to display the image: {e}")
            
    print("Start inference ! ")
    print("*****************************************************")
    res = get_answer(image, question_text, model, processor)
    
    opt = get_option(res)
    
    print("*****************************************************")
    print(f"\nFinish inference and answer from LLaMA : {opt}")


def main():
    parser = argparse.ArgumentParser(description="Medical Q&A Command Line Interface")
    parser.add_argument("--results", type=str, help="Path to inference results file (e.g., result_all.txt)")
    parser.add_argument("--confirm", type=str, help="Path to the file with questions needing confirmation (e.g., need_confirm.txt)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to TSV format dataset file")
    parser.add_argument("--question_idx", type=int, help="Specify the question index to display")
    parser.add_argument("--list", action="store_true", help="List all question indices")

    args = parser.parse_args()

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Loaded succeed ! ")
    print("*****************************************************")
    
    df = pd.read_csv(args.dataset, sep='\t')
    if args.list:
        print("\nAll question indices in the dataset:")
        print(df["index"].tolist())
        return

    # Display a specific question
    if args.question_idx is not None:
        display_question(df, args.question_idx, args.dataset)

    # Display inference results
    if args.results:
        print("\nInference Results:")
        print(load_results(args.results))

    # Display questions needing confirmation
    if args.confirm:
        print("\nQuestions Needing Confirmation:")
        print(load_results(args.confirm))


if __name__ == "__main__":
    main()
