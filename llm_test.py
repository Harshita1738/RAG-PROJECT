from ctransformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", hf=True)
tokenizer = AutoTokenizer.from_pretrained(model)