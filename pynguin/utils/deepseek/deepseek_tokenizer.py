# pip3 install transformers
# python3 deepseek_tokenizer.py

from pathlib import Path

from transformers import AutoTokenizer, LlamaTokenizerFast

chat_tokenizer_dir = (Path(__file__).parent).resolve() / "data"

tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)
