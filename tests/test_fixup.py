import ast


def fixup_result(result: str):
    """
    In case we aborted generation early (due to running out of tokens), remove
    any lingering syntax errors that prevent parsing by the `ast` module.
    (There may still be syntax errors when actually running the code)

    Args:
        result: some natural language source code

    Returns:
        source code that parses with ast.pasrse
    """
    result = result.lstrip()

    if result.startswith("```"):
        result = "\n".join(result.split("\n")[1:])

    try:
        ast.parse(result)
        return result
    except SyntaxError as e:
        line_to_rm = e.lineno
        lines = result.split("\n")
        print(line_to_rm)
        if line_to_rm is None or line_to_rm >= len(lines):
            return fixup_result("\n".join(lines[:-1]))
        else:
            return fixup_result("\n".join(lines[:line_to_rm]))


source = """
```python
def test_load_method():
    tokenizer = Tokenizer()
    model_file = "test_model.model"
    with open(model_file, 'w') as f:
        f.write("minbpe v1\n")
        f.write("test_pattern\n")
        f.write("1\n")
        f.write("<|endoftext|> 100257\n")
        f.write("0 1\n")
    tokenizer.load(model_file)
    assert tokenizer.pattern == "test_pattern"
    assert tokenizer.special_tokens == {"<|endoftext|>": 100257}
    assert tokenizer.merges == {(0, 1): 256}
    assert tokenizer.vocab[256] == tokenizer.vocab[0] + tokenizer.vocab[1]
"""

print(fixup_result(source))
