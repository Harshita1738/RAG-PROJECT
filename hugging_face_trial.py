from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline

#set hugging face api key
HUGGING_FACE_API_KEY='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'

hugging_face_model='TinyLlama/TinyLlama-1.1B-Chat-v1.0'

required_files=[
    "special_tokens_map.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "eval_results.json",
    "tokenizer.model",
    "tokenizer.json",
    "config.json"
]

# download model files
for filename in required_files:
    download_location=hf_hub_download(
        repo_id=hugging_face_model,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )
    print(f"FILE DOWNLOADED TO: {download_location}")

# load the tokenizer and the model
model=AutoModelForCausalLM.from_pretrained(hugging_face_model)
tokenizer=AutoTokenizer.from_pretrained(hugging_face_model)

# create a pipeline for text generation
text_generation_pipeline=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1000
)

#example query
response = text_generation_pipeline("Tell me a joke.")
print(response)

'''hf_hub_download ->	Downloads specific model files from Hugging Face Model Hub.
AutoTokenizer ->	Automatically loads the appropriate tokenizer for a model.
AutoModelForCausalLM ->	Loads a causal language model (for tasks like text generation).
pipeline ->	Creates a ready-to-use NLP pipeline, like text generation, summarization, etc.'''

