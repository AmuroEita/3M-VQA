import pandas as pd
from openai import OpenAI
import json
import requests
from abc import ABC

DATA_LOCATIONS = {
    'brazil':'./brazil_image_instruction_4o_mini.csv',
    'israel':'./israel_image_instruction_4o_mini.csv',
    'japan':'./japan_image_instruction_4o_mini.csv',
    'spain':'./spain_image_instruction_4o_mini.csv'
}

DATA_CLEAN = {
    'brazil':'./brazil_image_instruction_4o_mini_clean.csv',
    'israel':'./israel_image_instruction_4o_mini_clean.csv',
    'japan':'./japan_image_instruction_4o_mini_clean.csv',
    'spain':'./spain_image_instruction_4o_mini_clean.csv'
}

LANGUAGES = ['brazil', 'israel', 'japan', 'spain']

class GPTText(ABC):
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        self.provider = "openai"

        self.history = []
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def basic_request(self, system_prompt, prompt, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
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
        response = requests.post(self.base_url, headers=headers, json=data)
        try:
            response = response.json()
        except:
            response = ''

        return response

#gpt4o = GPTText(model='gpt-4o', api_key="API KEY")

for language in LANGUAGES:
    if language=='brazil' or language=='israel' or language=='japan':
        continue
    df = pd.read_csv(DATA_LOCATIONS[language])
    num_data = len(df)
    print(language)
    print(num_data)
    res = {'id':[i+1 for i in range(num_data)]}
    output = []
    system_prompt = "You are a helpful assistant to understand some texts and find the key information in it."
    
    for i in range(num_data):
        print(i)
        txt_prompt = "Instruction: Given the following answer to a multiple choice question with options A B C D, please find the final selected option of the answer and output exactly the option. \
                    You should only output one alphabet from A or B or C or D. If the answer does not contain the option please random output one from A or B or C or D. \n"
        txt_prompt = txt_prompt + "Answer: " + df['answer'][i]
        
        answer = gpt4o.basic_request(system_prompt, txt_prompt)
        #print(answer)
        ans='X'
        if answer!='':
            try:
                ans = answer['choices'][0]['message']['content']
            except:
                ans='X'
        output.append(ans)
    res['answer'] = output
    df_res = pd.DataFrame(res)
    df_res.to_csv(DATA_CLEAN[language], index=False)