import os
import re
import time
import torch
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from google.cloud import translate_v2 as translate
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, MllamaForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from vdb import load_faiss_index, search_faiss_index
from caption import load_caption
from gpt import get_option

question_prompt1 = """
Diagnostic Procedure: 
Identify the main diagnosis or condition. """

question_prompt2 = """
Background Knowledge: """

question_prompt3 = """
Use differential diagnosis and clinical guidelines to eliminate clearly wrong options. 
Choose the most appropriate answer based on the remaining options. 
Given the background knowledge above and standard diagnostic practices, analyze the following multiple-choice question STEP BY STEP following the below instruction: 
Review the Question and Options: Understand the key medical concepts and terms. 
Use Background Knowledge: Integrate and understand the provided background knowledge and your own internal knowledge to assess the options. 
Evaluate Each Option: For each choice, consider if it aligns with medical facts and standard diagnostic procedures. Eliminate incorrect options based on logical reasoning. 
If Background Knowledge is Insufficient: Use elimination and reasoning to narrow down the choices. 
Final Decision: Choose the correct answer and justify your selection. 
Finally provided your answer at the end of the whole process with only the option letter (eg. A) so that I can collect your answer."""

question_prompt4 = """
Image Caption: """

def trans_base64_image(encoded_string, width=None, height=None):
    image_data = base64.b64decode(encoded_string)
    image = Image.open(BytesIO(image_data))

    if width or height:
        original_width, original_height = image.size

        if width and not height:
            height = int((width / original_width) * original_height)
        elif height and not width:
            width = int((height / original_height) * original_width)

        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        image_path = "temp_image.jpg"
        image.save(image_path)


def translate_text(text, target_language="en"):
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]


def get_answer(image, text, model, processor):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text}
        ]}
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=1200)
    print(processor.decode(output[0]))
    return processor.decode(output[0])


def extract_abcd_or_default(text):
    pattern = r"([A-D])(?=[^A-Z]*<\|eot_id\|>)"
    match = re.search(pattern, text)
    return match.group(1) if match else 'Z'


if __name__ == "__main__": 
    
    model_path = "Llama-3.2-90B-Vision-Instruct"
    logging.set_verbosity_debug() 
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"{model_path} loaded ")
    
    index_file = "faiss_index2.index"
    metadata_file = "metadata2.pkl"
    
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        # 如果文件存在，则加载索引和元数据
        index, metadata = load_faiss_index(index_file, metadata_file)
        sentences = metadata["sentences"]
        sentence_to_file_map = metadata["sentence_to_file_map"]
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    client = translate.Client()
  
    temp_img = "temp_image.jpg"
    
    captions = load_caption()
    
    tsv_files = [
        "data/brazil_local_processed.tsv",
        "data/israel_local_processed.tsv",
        "data/japan_local_processed.tsv",
        "data/spain_local_processed.tsv"
    ]
        
    datasets = {}
    for file_path in tsv_files:
        print(file_path)
        datasets[file_path] = pd.read_csv(file_path, sep='\t')
    
    file_idx = 0
    for df_name, df in datasets.items():

        index_list = df.index.tolist()
        
        print(index_list)
        correct_cnt = 0
        
        for idx in index_list:
              
            filtered_df = df[df["index"] == idx]
            if filtered_df.empty:
                continue

            base64_image = df[df["index"] == idx]["image"].values[0]
            
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            if image.mode in ('RGBA', 'P'):  
                image = image.convert('RGB')

            image.save(temp_img, format='JPEG')
            
            row = df[df["index"] == idx].iloc[0]
            
            question = translate_text(row["question"])
            question_text = f"{question_prompt3} Question {idx}: {question} \n"
            
            options = ['A', 'B', 'C', 'D']
            correct_opt = ""
            formatted_options = []
            for option in options:
                if pd.isna(row.get(option)):
                    continue
                if option == row['correct_option']:
                    opt = translate_text(row[option])
                    question_text += f"{option}. {opt} \n"
                    correct_opt = option
                else:
                    opt = translate_text(row[option])
                    question_text += f"{option}. {opt} \n"
                    
            question = question_text
            image_file = temp_img
               
            img = Image.open(image_file)
            
            results = search_faiss_index(question, index, st_model, sentences, sentence_to_file_map)
            
            rag_text = question_prompt2
            for result in results:
                rag_text += result['sentence']
                
            input = question_prompt1 + question_prompt4 + captions[file_idx][idx] + rag_text + question_text
            
            print(f"Start inference on question {idx} .... in dataset {df_name}")
            start_time = time.time()
            res = get_answer(img, input, model, processor)
            end_time = time.time()
            
            opt = extract_abcd_or_default(res)
            if opt == "Z":
                opt = get_option(res)
            
            if opt == "Z":
                with open("need_confirm.txt", "a", encoding="utf-8") as file:
                    file.write(f"\n {idx} in {df_name}, correct option {correct_opt}")
                    file.write("\n" + res)
                    file.write("\n **************************************************")
            
            print(f"End inference on question {idx} .... in dataset {df_name}," 
                  f"answer is {opt}, correct answer is {correct_opt}")
            
            if opt == correct_opt:
                correct_cnt += 1
            
        correct_rate = correct_cnt / len(index_list)
        file_idx += 1
        
        with open('result_all.txt', 'a') as file:
            file.write(f'\n{df_name} : {correct_rate}, ({correct_cnt}/{len(index_list)})')
            
    
    
