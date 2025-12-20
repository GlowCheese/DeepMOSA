# pip3 install transformers
# python3 deepseek_tokenizer.py

from transformers import AutoTokenizer

chat_tokenizer_dir = "data/deepseek"
tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)
