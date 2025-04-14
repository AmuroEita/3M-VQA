from openai import OpenAI
import os

from torch.utils.data import Dataset

from tqdm import tqdm
import json
import re
import random
import pandas as pd

API_KEY = 'your_key'
MODEL="gpt-4o-mini"
CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", API_KEY))
SYSTEM_TEXT = ("You are an expert medical assistant skilled in diagnosing diseases. "
               "Your task is to analyze the provided information, including an image description, a medical question, and options. "
               "Generate a detailed reasoning process explaining which answer is correct, why it is the best choice and why the other options are incorrect. "
               "To help you out, we have provided examples of reasoning processes for different questions.")

NUM_EXAMPLES = 2
DATA_LOCATIONS = {
    'brazil':'./data/brazil_english_processed.tsv',
    'israel':'./data/israel_english_processed.tsv',
    'japan':'./data/japan_english_processed.tsv',
    'spain':'./data/spain_english_processed.tsv'
}

REASON_PATHS = {
    'brazil': './reasoning/gpt-4o_brazil_english_reasoning.json',
    'israel': './reasoning/gpt-4o_israel_english_reasoning.json',
    'japan': './reasoning/gpt-4o_japan_english_reasoning.json',
    'spain': './reasoning/gpt-4o_spain_english_reasoning.json'
}

class ReasoningDataset(Dataset):
    """
    PyTorch Dataset for LLaVa with support for in-context learning examples.
    """
    def __init__(
        self,
        dataset_name_or_path: str,
        reasoning_path: str = None,
        num_examples: int = 2  # Number of in-context examples to include
    ):
        super().__init__()
        self.data_pd = pd.read_csv(dataset_name_or_path, sep='\t')
        self.data_length = len(self.data_pd)
        # self.load_non_english()
        self.num_examples = num_examples  # Number of examples to provide as context

        self.reasoning = None
        if reasoning_path:
            self.reasoning = json.load(open(reasoning_path, 'r'))

    def __len__(self) -> int:
        return self.data_length
    
    def create_instruction_set(self, idx, sample):
        image = sample['image']
        question = sample['question']
        answer = sample['correct_option']

        prompt = f"Question: {question}\n\nOptions:\n"

        alphabets = ['A', 'B', 'C', 'D']
        for i in range(len(alphabets)):
            letter = alphabets[i]
            prompt += f"Option {letter}: {sample[letter]}\n"

        reason = None
        if self.reasoning:
            reason = self.reasoning[idx]

        return image, prompt, answer, reason

    def generate_list(self, idx, numsamples):
        numbers = list(range(self.__len__()))
        numbers.remove(idx)
        unique_numbers = random.sample(numbers, numsamples)

        unique_numbers.append(idx)
        
        return unique_numbers
        
    def __getitem__(self, idx: int):
        samples = []
        # Collect in-context examples
        idx_choices = self.generate_list(idx, self.num_examples)
        
        for idx in idx_choices:
            example_sample = self.data_pd.iloc[idx]
            base_64image, example_text, example_answer, reason = self.create_instruction_set(idx, example_sample)
            samples.append((base_64image, example_text, example_answer, reason))
        
        return samples
    
def create_n_shot_prompt(samples, system_text):
    messages = [
        # Task definition
        {"role": "system", "content": system_text}
    ]

    # Add few-shot examples
    for sample in samples[:-1]:
        image = sample[0]
        prompt = sample[1]

        messages.append(
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                {"type": "text", "text": prompt}
            ]}
        )

        if sample[3] is not None:
            reasoning, _ = sample[3]
            messages.append({"role": "assistant", "content": reasoning})

    # Add the example to be predicted
    image = samples[-1][0]
    prompt = sample[-1][1]
    answer = samples[-1][2]

    messages.append(
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            {"type": "text", "text": prompt}
        ]}
    )

    return messages, answer

def run_single_inference(model, client, samples, system_text):
    messages, answer = create_n_shot_prompt(samples, system_text)
    completion = client.chat.completions.create(
        model=model,
        messages = messages
    )
    return completion, answer

def process_data(datasets):
    # Run inference
    predictions = {}
    for language, dataset in datasets.items():
        completions = []
        for i in tqdm(range(len(dataset))):
            samples = dataset[i]
            completion, answer = run_single_inference(MODEL, CLIENT, samples, SYSTEM_TEXT)
            completions.append((completion.choices[0].message.content, answer))

        predictions[language] = completions

    save_data(predictions)

def save_data(predictions):
    for lang, result in predictions.items():
        with open(f"./results/{MODEL}_COT_{lang}_predictions.json", "w") as file:
            json.dump(result, file)

def load_data():
    predictions = {}
    for lang in DATA_LOCATIONS.keys():
        with open(f"./results/{MODEL}_COT_{lang}_predictions_1.json", "r") as file:
            predictions[lang] = json.load(file)

    return predictions

if __name__ == '__main__':
    # Load data
    datasets = {}
    for language, file_path in DATA_LOCATIONS.items():
        datasets[language] = ReasoningDataset(file_path, num_examples=NUM_EXAMPLES, 
                                              reasoning_path=REASON_PATHS[language])

    # process_data(datasets)
    predictions = load_data()

    results = {}
    for lang, data in predictions.items():
        total = 0
        correct = 0
        for i, (pred, ground_truth) in enumerate(data):
            total += 1
            lines = pred.splitlines()[-1]

            matches = re.findall(r"Option [A-D]", lines)

            if not matches:
                continue
            
            matches = matches[0][-1]
            if matches == ground_truth:
                correct += 1

        results[lang] = (correct, total)

    for lang, data in results.items():
        print(lang, data)
        print(f"Language: {lang}, Accuracy: {data[0]/data[1]:.2f}%")