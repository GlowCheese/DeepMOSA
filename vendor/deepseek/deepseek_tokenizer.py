# pip3 install transformers
# python3 deepseek_tokenizer.py

import io
import contextlib

from pathlib import Path


chat_tokenizer_dir = (Path(__file__).parent).resolve()


with (
    contextlib.redirect_stdout(io.StringIO()) as stdout,
    contextlib.redirect_stderr(io.StringIO()) as stderr
):
    from transformers import AutoTokenizer
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()


tokenizer = AutoTokenizer.from_pretrained( 
    chat_tokenizer_dir, trust_remote_code=True
)
