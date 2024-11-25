import openai

# Your Open API Key
openai.api_key = ''

client = openai.OpenAI(api_key=openai.api_key)

def get_option(text):
    response = client.chat.completions.create(
    model="gpt-4",  
    messages=[
            {"role": "user", "content": f"Please give me the correct option (Just give me the option, no other text in your answer) of the following text : {text}"}
        ],
        temperature=0.7,
        max_tokens=1200
    )

    print(response.choices[0].message.content)
    
    return response.choices[0].message.content