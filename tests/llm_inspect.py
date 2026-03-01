import json
from pprint import pprint

import openai

from pynguin import environ
from pynguin.configuration import config
from pynguin.llm.api_errors import APIContentFilterError, APIRefusalError
from pynguin.utils.deepseek import tokenizer

# ==============================================
#   system prompt + user prompt
# ==============================================

sys_prompt = """
Do NOT import pytest and unittest when writting test cases.
A good unit test should only contains variable assignments, assertions and function/method/constructor calls (i.e. without any custom class or function definition or control structure like `if`, `for`, `while`, `match`, `with`, ... statements).
All test cases should starts with: `def test_[test case's name]():`.
Your response should only contain the test case itself without any additional text or information.
"""

user_prompt = '''
@dataclasses.dataclass(frozen=True)
class File:
    stream: TextIO
    path: Path
    encoding: str

    @staticmethod
    def detect_encoding(filename: str | Path, readline: Callable[[], bytes]) -> str:
        try:
            raise UnsupportedEncoding(filename)

    @staticmethod
    def from_contents(contents: str, filename: str) -> "File":
        encoding = File.detect_encoding(filename, BytesIO(contents.encode("utf-8")).readline)
    @property
    def extension(self) -> str:
    @staticmethod
    def _open(filename: str | Path) -> TextIOWrapper:
        """Open a file in read only mode using the encoding detected by
        detect_encoding().
        """
        buffer = open(filename, "rb")
        try:
            encoding = File.detect_encoding(filename, buffer.readline)
            buffer.seek(0)
            text = TextIOWrapper(buffer, encoding, line_buffering=True, newline="")
            text.mode = "r"  # type: ignore
            buffer.close()
            raise

    @staticmethod
    @contextmanager
    def read(filename: str | Path) -> Iterator["File"]:
        file_path = Path(filename).resolve()
        stream = None
        try:
            stream = File._open(file_path)
            yield File(stream=stream, path=file_path, encoding=stream.encoding)
        finally:
            if stream is not None:
                stream.close()

```
Write unit test to ensure that the predicate at line 5 evaluates to False.
```
1 |     @staticmethod
2 |     def detect_encoding(filename: str | Path, readline: Callable[[], bytes]) -> str:
3 |         try:
6 |             raise UnsupportedEncoding(filename)
```
'''

# ==============================================
#   messages = [sys_prompt, user_prompt]
# ==============================================

messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_prompt},
]

# ==============================================
#   sending messages to deepseek
# ==============================================

config.llm.model = "deepseek/deepseek-chat"

client = openai.OpenAI(api_key=environ.OPENAI_API_KEY, base_url=config.llm.base_url)
print("==============================================\n")
print(
    f"Sending query to model: {config.llm.model} (temp={config.llm.temperature}, max_tokens={config.llm.max_tokens})"
)
num_chars = sum(len(m["content"]) for m in messages)
num_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
print("\nQuery used:")
pprint(messages)
print(f"\nQuery size: {num_chars} characters (~{num_tokens} tokens)")

query = client.chat.completions.create(
    messages=messages,  # type: ignore
    model=config.llm.model,
    temperature=config.llm.temperature,
    stream=False,
    stop="\n```",
    max_tokens=config.llm.max_tokens,
)

# ==============================================
#   inspecting query
# ==============================================

response = query.choices[0]
if response.finish_reason == "content_filter":
    raise APIContentFilterError()
if response.message.refusal is not None:
    raise APIRefusalError(response.message.refusal)

print("\n\n==============================================\n")
print("Query usage:")
print(json.dumps(query.usage.model_dump(), indent=2))
# assert query.usage is not None

# self._num_llm_calls += 1
# self._time_calling_llm += time.time() - query_at
# self._input_tokens_cnt += query.usage.prompt_tokens
# self._output_tokens_cnt += query.usage.completion_tokens

# _logger.info("Output size: %s tokens", query.usage.completion_tokens)

# stat.track_output_variable(RuntimeVariable.LLMCalls, self._num_llm_calls)
# stat.track_output_variable(RuntimeVariable.LLMQueryTime, self._time_calling_llm)
# stat.track_output_variable(RuntimeVariable.LLMInputTokens, self._input_tokens_cnt)
# stat.track_output_variable(RuntimeVariable.LLMOutputTokens, self._output_tokens_cnt)

print("\nLLM Response:")
print(json.dumps(response.model_dump(), indent=2))
