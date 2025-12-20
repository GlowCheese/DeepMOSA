# pip3 install transformers
# python3 deepseek_tokenizer.py

from transformers import AutoTokenizer, LlamaTokenizerFast

chat_tokenizer_dir = "data/deepseek"
tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)
