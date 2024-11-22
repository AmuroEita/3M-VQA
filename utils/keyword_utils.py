import re
from transformers import AutoTokenizer, LlamaForCausalLM

keyword_prompt = """You are a professional information searcher. 
Your task is to identify top 5 most effective multi-word keywords 
(e.g., short phrases or combinations of two or three words) 
for a Wikipedia search to solve the given multi-choice question. 
Carefully analyze the question and all provided options. 
Focus on capturing critical terms or phrases (e.g., 'ventricular tachycardia', 'synchronized shock'). 
Combine terms when they are contextually linked and can yield more precise search results. 
Avoid splitting keywords that are more useful when combined. 
Provide the keywords in a Python list format at the end of your answer, ensuring clarity and easy extraction."""

def get_keywords_from_text(model, tokenizer, input_text):
  input_text = keyword_prompt + input_text
  inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
  
  outputs = model.generate(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      max_length=2000
  )
  
  generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
  generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
  print(generated_text)
  
  return generated_text


def extract_keywords(text):
    return re.findall(r"'(.*?)'", text)
  
