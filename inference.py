import time
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from google.cloud import translate_v2 as translate
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

question_prompt = """Background Knowledge: 
Diagnostic Procedure: 
Identify the main diagnosis or condition. 
Use differential diagnosis and clinical guidelines to eliminate clearly wrong options. 
Choose the most appropriate answer based on the remaining options. 
Given the background knowledge above and standard diagnostic practices, analyze the following multiple-choice question STEP BY STEP following the below instruction: 
Review the Question and Options: Understand the key medical concepts and terms. 
Use Background Knowledge: Integrate and understand the provided background knowledge and your own internal knowledge to assess the options. 
Evaluate Each Option: For each choice, consider if it aligns with medical facts and standard diagnostic procedures. Eliminate incorrect options based on logical reasoning. 
If Background Knowledge is Insufficient: Use elimination and reasoning to narrow down the choices. 
Final Decision: Choose the correct answer and justify your selection. 
Finally provided your answer at the end of the whole process with only the option letter (eg. A) so that I can collect your answer."""

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


def get_answer(img, text, model, tokenizer):
    text_inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")

    inputs = {
        "input_ids": text_inputs["input_ids"],            
        "attention_mask": text_inputs["attention_mask"],  
        "vision_inputs": img,                 
    }
    
    outputs = model.generate(**inputs, max_length=300)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text + "\n\n")
    return generated_text


if __name__ == "__main__": 
    
    model_path = "Llama-3.2-3B-Instruct-uncensored-GGUF"
    logging.set_verbosity_debug() 
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
    print(f"{model_path} loaded ")
    
    client = translate.Client()
  
    temp_img = "temp_image.jpg"
    
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
            question_text = f"{question_prompt} Question {idx}: {question} \n"
            
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
                    correct_opt = option
                else:
                    opt = translate_text(row[option])
                    question_text += f"{option}. {opt} \n"
                    
            question = question_text
            image_file = temp_img
               
            img = Image.open(image_file)
            
            print(f"Start inference on question {idx} .... in dataset {df_name}")
            start_time = time.time()
            get_answer(img, question, model, tokenizer)
            end_time = time.time()
            print(f"End inference on question {idx} .... in dataset {df_name}")
            
        correct_rate = correct_cnt / len(index_list)
        
        with open('result_all.txt', 'a') as file:
            file.write(f'\n{df_name} : {correct_rate}, ({correct_cnt}/{len(index_list)})')
            
    
    
