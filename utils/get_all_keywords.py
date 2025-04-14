import pandas as pd
from google.cloud import translate_v2 as translate
from transformers import AutoTokenizer, LlamaForCausalLM
from keyword_utils import get_keywords_from_text, extract_keywords

def translate_text(text, target_language="en"):
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]

if __name__ == "__main__": 
    client = translate.Client()

    model_path = "Llama-3.1-Tulu-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = LlamaForCausalLM.from_pretrained(model_path).to("cuda")
    
    tsv_files = [
        "data/brazil_local_processed.tsv",
        "data/israel_local_processed.tsv",
        "data/japan_local_processed.tsv",
        "data/spain_local_processed.tsv"
    ]
        
    all_keywords = []
        
    datasets = {}
    for file_path in tsv_files:
        print(file_path)
        datasets[file_path] = pd.read_csv(file_path, sep='\t')
    
    for df_name, df in datasets.items():
        index_list = df.index.tolist()
        
        for idx in index_list:
            filtered_df = df[df["index"] == idx]
            if filtered_df.empty:
                continue
            
            row = df[df["index"] == idx].iloc[0]
            question = translate_text(row["question"])
            
            options = ['A', 'B', 'C', 'D']
            correct_opt = ""
            formatted_options = []
            for option in options:
                if pd.isna(row.get(option)):
                    continue
                opt = translate_text(row[option])
                question += f"{option}. {opt} \n"
                   
            print("Starting fetching keywords ...")
            
            raw_text = get_keywords_from_text(model, tokenizer, question)
            all_keywords.extend(extract_keywords(raw_text))
            
    file_path = "keywords.txt"

    with open(file_path, 'w') as file:
        for keyword in all_keywords:
            file.write(keyword + '\n')

    print(f"Keywords wrote in {file_path}")
          
            
    
    
