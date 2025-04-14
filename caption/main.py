from openai import OpenAI
from PIL import Image
import pandas as pd
import base64
import requests
import os
import json
from abc import ABC
#client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

LANGUAGES = ['brazil', 'israel', 'japan', 'spain']

DATA_LOCATIONS = {
    'brazil':'./brazil_english_processed.tsv',
    'israel':'./israel_english_processed.tsv',
    'japan':'./japan_english_processed.tsv',
    'spain':'./spain_english_processed.tsv'
}

CAPTION_PATHS_wq = {
    'brazil': './brazil_caption.csv',
    'israel': './israel_caption.csv',
    'japan': './japan_caption.csv',
    'spain': './spain_caption.csv'
}

CAPTION_PATHS = {
    'brazil': './brazil_caption_img_only.csv',
    'israel': './israel_caption_img_only.csv',
    'japan': './japan_caption_img_only.csv',
    'spain': './spain_caption_img_only.csv'
}

CAPTION_PATHS_MINI = {
    'brazil': './brazil_caption_wq_4o_mini.csv',
    'israel': './israel_caption_wq_4o_mini.csv',
    'japan': './japan_caption_wq_4o_mini.csv',
    'spain': './spain_caption_wq_4o_mini.csv'
}

class GPTVision(ABC):
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        self.provider = "openai"

        self.history = []
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def basic_request(self, system_prompt, prompt, image_path, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        base64_images = image_path
        
        if base64_images=='':
            data = {
                **kwargs,
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ]
                    }
                ],
            }
        else:
            data = {
                **kwargs,
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_images}"
                                }
                            }
                        ]
                    }
                ],
                # "max_tokens": 300
            }
            
        response = requests.post(self.base_url, headers=headers, json=data)
        try:
            response = response.json()
        except:
            response = ''

        return response
        

if __name__ == "__main__":
    #gpt4o = GPTVision(model='gpt-4o', api_key="Your OpenAI API key")
    
    for language in LANGUAGES:
        #if language=='brazil' or language=='israel':
        #    continue
        df_eng = pd.read_csv(DATA_LOCATIONS[language], sep='\t')
        df_cap = pd.read_csv(CAPTION_PATHS[language])
        
        #system_prompt = "You are a helpful assistant."
        system_prompt = "You are a helpful assistant to answer Multiple-choice questions with biomedical image."
        #system_prompt = "You are a helpful assistant to answer Multiple-choice questions in biomedical setting."
        #system_prompt = "You are a helpful assistant to give captions to biomedical images."
        
        num_data = len(df_eng)
        print(language)
        print(num_data)
        res = {'id':[i+1 for i in range(num_data)]}
        output = []
        for i in range(num_data):
            print(i)        
            '''
            # Image, Question -> answer
            txt_prompt = '<image> \n' + 'Question: ' + df_eng.iloc[i,2] + '\n'
            #txt_prompt = 'Question: ' + df_eng.iloc[i,2] + '\n'
            txt_prompt = txt_prompt + 'A. ' + df_eng.iloc[i,3] + '\n'
            txt_prompt = txt_prompt + 'B. ' + df_eng.iloc[i,4] + '\n'
            txt_prompt = txt_prompt + 'C. ' + df_eng.iloc[i,5] + '\n'
            txt_prompt = txt_prompt + 'D. ' + df_eng.iloc[i,6] + '\n'
            txt_prompt = txt_prompt + "Please answer the question with the exact correct option without explaining the reason."
            image_path = df_eng.iloc[i,1]
            #image_path = ''
            '''
            
            # Get caption with question
            #txt_prompt = '<image> \n' + 'Question: ' + df_eng.iloc[i,2] + '\n'
            #txt_prompt = txt_prompt + "Describe the contents of the biomedical image with the help of the given question."
            
            # Get caption w/o question
            #txt_prompt = '<image> ' + "Describe the contents of the biomedical image from " + language + "."
            #image_path = df_eng.iloc[i,1]
            
            '''
            # Caption Question Option -> Answer
            txt_prompt = 'Context: ' + df_cap['caption'][i] + '\n'
            txt_prompt = txt_prompt + 'Question: ' + df_eng.iloc[i,2] + '\n' + 'Options:\n'
            txt_prompt = txt_prompt + 'A. ' + df_eng.iloc[i,3] + '\n'
            txt_prompt = txt_prompt + 'B. ' + df_eng.iloc[i,4] + '\n'
            txt_prompt = txt_prompt + 'C. ' + df_eng.iloc[i,5] + '\n'
            txt_prompt = txt_prompt + 'D. ' + df_eng.iloc[i,6] + '\n'
            txt_prompt = txt_prompt + 'Given the context (image caption), please answer the question with the exact correct option without explaining the reason.'
            image_path = ''
            '''
            
            # Image, Caption, Question -> answer
            txt_prompt = '<image> \n' + 'Context: ' + df_cap['caption'][i] + '\n'
            txt_prompt = txt_prompt + 'Question: ' + df_eng.iloc[i,2] + '\n' + 'Options:\n'
            txt_prompt = txt_prompt + 'A. ' + df_eng.iloc[i,3] + '\n'
            txt_prompt = txt_prompt + 'B. ' + df_eng.iloc[i,4] + '\n'
            txt_prompt = txt_prompt + 'C. ' + df_eng.iloc[i,5] + '\n'
            txt_prompt = txt_prompt + 'D. ' + df_eng.iloc[i,6] + '\n'
            txt_prompt = txt_prompt + 'Given the biomedical image and the context (image caption), please answer the question with the exact correct option without explaining the reason.'
            image_path = df_eng.iloc[i,1]
            '''
            # Reasoning
            txt_prompt = '<image> \n' + 'Instruction: Given the biomedical image from the medical examination of the following countries Brazil, Israel, Japan, Spain, \
                        try to understand the image with different language setting and the question thoroughly. Think step by step to answer the question with the exact correct option. \
                        If you cannot find a correct answer, please select the most possible one. \n'
            txt_prompt = txt_prompt + 'Question: ' + df_eng.iloc[i,2] + '\n' + 'Options:\n'
            txt_prompt = txt_prompt + 'A. ' + df_eng.iloc[i,3] + '\n'
            txt_prompt = txt_prompt + 'B. ' + df_eng.iloc[i,4] + '\n'
            txt_prompt = txt_prompt + 'C. ' + df_eng.iloc[i,5] + '\n'
            txt_prompt = txt_prompt + 'D. ' + df_eng.iloc[i,6] + '\n'
            image_path = df_eng.iloc[i,1]
            '''
            
            answer = gpt4o.basic_request(system_prompt, txt_prompt, image_path)
            #print(answer)
            ans='X'
            if answer!='':
                ans = answer['choices'][0]['message']['content']
            output.append(ans)
            
        
        res['answer'] = output
        df_res = pd.DataFrame(res)
        df_res.to_csv(language+'_image_caption_question_prediction_4o.csv', index=False)
        