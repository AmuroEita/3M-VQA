# Med-VQA
M3-VQA, a novel pipeline for multilingual and multimodal biomedical VQA. M3-VQA leverages translation for multilingual inputs, retrieval augmented generation (RAG) for knowledge grounding, and in-context learning (ICL) with Chain-of-Thought prompting for accurate reasoning. 

## Getting Started

### Prerequisites

1. Get a free API Key for Google Translate at [https://cloud.google.com/translate/docs/reference/rest/](https://cloud.google.com/translate/docs/reference/rest/)
2. Clone the repo
   ```sh
   git clone https://github.com/AmuroEita/M3-VQA.git && cd M3-VQA
   ```
3. Install required Python packages
   ```sh
   pip install -r requirements.txt
   ```
4. Enter your GPT API in `utils/GPT-API.txt`
   ```js
   echo "${Your GPT API Key}" > utils/GPT-API.txt

   ```
5. Prepare the datasets 
   ```sh
   cd data && sh download.sh
   ```
6. Download the model via hugging face
   ```sh
   huggingface-cli login
   huggingface-cli download --resume-download unsloth/Llama-3.2-11B-Vision-Instruct --local-dir Llama-3.2-11B-Vision-Instruct
   ```


### Usage

#### Specify a Question for Testing
Use this mode to provide a specific question for Med-VQA to answer. The following example demonstrates how to test the 11th question in the israel_local_processed.tsv dataset. The process and results will be displayed directly in the command line.
```sh
export GOOGLE_APPLICATION_CREDENTIALS="/your_path_to/google_translate.json" && python3 demo.py --dataset data/israel_local_processed.tsv --question_idx 11
```


#### Evaluate on a Dataset
Run on the entire dataset to compute accuracy. Results will be saved in the results folder for further analysis.
```sh
export GOOGLE_APPLICATION_CREDENTIALS="/your_path_to/google_translate.json" && python3 inference.py
```
