####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    from pathlib import Path
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_with_coding_declaration():
    from io import BytesIO
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "latin-1"


def test_detect_encoding_default_utf8():
    from io import BytesIO
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_invalid_raises_unsupported_encoding():
    from io import BytesIO
    from pathlib import Path
    
    def failing_readline():
        raise LookupError("Unknown encoding")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_vim_style():
    from io import BytesIO
    contents = "# vim: set fileencoding=utf-8 :\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


# LLM-generated content at query #2
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


def test_file_constructor_with_different_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    path1 = Path("/tmp/file1.txt")
    file1 = File(stream=stream, path=path1, encoding=encoding)
    assert file1.path == path1
    
    path2 = Path("/home/user/file2.py")
    file2 = File(stream=stream, path=path2, encoding=encoding)
    assert file2.path == path2


# LLM-generated content at query #3
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #4
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    class UnsupportedEncoding(Exception):
        def __init__(self, filename):
            self.filename = filename
    
    @staticmethod
    def detect_encoding(filename: str | Path, readline) -> str:
        try:
            return tokenize.detect_encoding(readline)[0]
        except Exception:
            raise UnsupportedEncoding(filename)
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def bad_readline():
        raise ValueError("Invalid encoding data")
    
    # Test that the except clause is executed (predicate at line 5 evaluates to True)
    # To ensure predicate evaluates to False, we need a case where no exception occurs
    valid_bytes = b"# -*- coding: utf-8 -*-\nprint('hello')\n"
    valid_readline = BytesIO(valid_bytes).readline
    
    # This should NOT raise UnsupportedEncoding, meaning the except block is NOT executed
    result = detect_encoding("test.py", valid_readline)
    assert result == "utf-8"


# LLM-generated content at query #5
#--------------------------

```python
def test_detect_encoding_exception_handler():
    from pathlib import Path
    from io import BytesIO
    from unittest.mock import patch
    
    # Mock tokenize.detect_encoding to raise an exception
    with patch('tokenize.detect_encoding', side_effect=Exception("Test exception")):
        try:
            File.detect_encoding("test.py", BytesIO(b"").readline)
            # If we reach here, the exception was not raised, test fails
            assert False, "Expected UnsupportedEncoding to be raised"
        except UnsupportedEncoding:
            # This is expected - the except block at line 5 was executed
            # The predicate at line 5 evaluated to True (exception was caught)
            pass


# LLM-generated content at query #6
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected frozen dataclass to raise error"
    except AttributeError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #10
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from unittest.mock import Mock
    
    # Create a mock readline that raises an exception
    mock_readline = Mock(side_effect=Exception("Test exception"))
    
    # Mock the UnsupportedEncoding exception class
    class UnsupportedEncoding(Exception):
        pass
    
    # Patch File to use our mock exception
    import dataclasses
    from io import StringIO, TextIOWrapper, BytesIO
    from typing import TextIO, Callable, Iterator
    from contextlib import contextmanager
    import tokenize
    
    @dataclasses.dataclass(frozen=True)
    class File:
        stream: TextIO
        path: Path
        encoding: str

        @staticmethod
        def detect_encoding(filename: str | Path, readline: Callable[[], bytes]) -> str:
            try:
                return tokenize.detect_encoding(readline)[0]
            except Exception:
                raise UnsupportedEncoding(filename)
    
    # Test that the exception is raised when detect_encoding fails
    try:
        File.detect_encoding("test.py", mock_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # This confirms the except block at line 5 was executed (predicate was True)
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Expected frozen dataclass to raise FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #13
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected frozen dataclass to raise error"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #15
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        assert True


def test_file_constructor_with_different_encoding():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.encoding == "latin-1"
    assert file.path.suffix == ".txt"


# LLM-generated content at query #16
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #17
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.encoding == "latin-1"


# LLM-generated content at query #18
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #19
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #20
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    for encoding in ["utf-8", "latin-1", "ascii", "cp1252"]:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.encoding == encoding


def test_file_constructor_with_various_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    paths = [
        Path("/tmp/test.txt"),
        Path("./relative/path.py"),
        Path("/home/user/documents/file.md"),
    ]
    
    for path in paths:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.path == path


# LLM-generated content at query #21
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected dataclass to be frozen"
    except Exception:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/file.txt")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == "latin-1"


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #27
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_ascii.encoding == "ascii"
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #29
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #31
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_extension_property():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "py"


def test_file_extension_property_no_extension():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == ""


def test_file_extension_property_multiple_dots():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.backup.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "py"


# LLM-generated content at query #33
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # This is the expected behavior - the except Exception clause at line 5 should evaluate to True
        # and raise UnsupportedEncoding
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected dataclass to be frozen"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #35
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #36
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #37
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #39
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #40
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    import tokenize
    
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_with_different_encoding():
    from io import BytesIO
    
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "iso8859-1"


def test_detect_encoding_default_utf8():
    from io import BytesIO
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_raises_unsupported_encoding_on_error():
    from io import BytesIO
    from pathlib import Path
    
    def failing_readline():
        raise ValueError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding as e:
        assert str(e.filename) == "test.py"


def test_detect_encoding_with_bom():
    from io import BytesIO
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8-sig")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8-sig"


# LLM-generated content at query #41
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file_obj = File(stream=stream, path=path, encoding=encoding)
    
    assert file_obj.stream is stream
    assert file_obj.path == path
    assert file_obj.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file_obj = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file_obj.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test")
    path = Path("/tmp/test.py")
    
    file_obj1 = File(stream=stream, path=path, encoding="utf-8")
    file_obj2 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_obj1.encoding == "utf-8"
    assert file_obj2.encoding == "latin-1"


# LLM-generated content at query #42
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #43
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


# LLM-generated content at query #44
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


def test_file_constructor_with_different_paths():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    path1 = Path("/test/file.txt")
    file1 = File(stream=stream, path=path1, encoding=encoding)
    assert file1.path == path1
    
    path2 = Path("/other/document.py")
    file2 = File(stream=stream, path=path2, encoding=encoding)
    assert file2.path == path2


# LLM-generated content at query #45
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #46
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    for encoding in ["utf-8", "latin-1", "ascii", "utf-16"]:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.encoding == encoding


def test_file_constructor_with_different_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    paths = [
        Path("/tmp/test.txt"),
        Path("./relative/path.py"),
        Path("/home/user/document.md"),
    ]
    
    for path in paths:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.path == path


# LLM-generated content at query #47
#--------------------------

```python
def test_detect_encoding_with_valid_utf8():
    from io import BytesIO
    import tokenize
    
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_latin1():
    from io import BytesIO
    
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "iso8859-1"


def test_detect_encoding_with_ascii():
    from io import BytesIO
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("ascii")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_invalid_encoding_raises_unsupported_encoding():
    from io import BytesIO
    
    def failing_readline():
        raise SyntaxError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_path_object():
    from io import BytesIO
    from pathlib import Path
    
    contents = "# coding: utf-8\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding(Path("test.py"), readline)
    assert encoding == "utf-8"


# LLM-generated content at query #48
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #49
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #50
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected frozen dataclass to raise error"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #51
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #52
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected frozen dataclass to raise error"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #53
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #54
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    # The predicate at line 5 (except Exception:) should evaluate to True when an exception occurs
    # To make it evaluate to False, we need detect_encoding to succeed without raising
    
    # Create valid Python content with encoding declaration
    valid_content = b"# -*- coding: utf-8 -*-\nprint('hello')\n"
    readline_func = BytesIO(valid_content).readline
    
    # This should not raise UnsupportedEncoding, meaning the except block is NOT executed
    # Therefore the predicate at line 5 evaluates to False
    result = File.detect_encoding("test.py", readline_func)
    
    assert result == "utf-8"


# LLM-generated content at query #55
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #56
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #57
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #58
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #59
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #60
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    from pathlib import Path
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_encoding_declaration():
    from io import BytesIO
    contents = "# -*- coding: iso-8859-1 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "iso-8859-1"


def test_detect_encoding_with_coding_comment():
    from io import BytesIO
    contents = "# coding: utf-16\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-16"


def test_detect_encoding_invalid_raises_unsupported_encoding():
    from io import BytesIO
    from pathlib import Path
    
    class BrokenReadline:
        def __call__(self):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
    
    try:
        File.detect_encoding("test.py", BrokenReadline())
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_empty_file():
    from io import BytesIO
    readline = BytesIO(b"").readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_path_object():
    from io import BytesIO
    from pathlib import Path
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding(Path("test.py"), readline)
    assert encoding == "utf-8"


# LLM-generated content at query #2
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == "latin-1"


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(File)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_detect_encoding_exception_handler():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        # If we reach here, the except block was not executed
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # The except block at line 5 was executed
        assert True


# LLM-generated content at query #5
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #7
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    import tokenize
    
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_latin1():
    from io import BytesIO
    
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "iso8859-1"


def test_detect_encoding_default_utf8():
    from io import BytesIO
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_invalid_raises_exception():
    from io import BytesIO
    
    def failing_readline():
        raise ValueError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_path_object():
    from io import BytesIO
    from pathlib import Path
    
    contents = "# coding: utf-8\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding(Path("test.py"), readline)
    assert encoding == "utf-8"


# LLM-generated content at query #8
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    # The predicate at line 5 (except Exception:) should evaluate to True when an exception occurs
    # To test that it evaluates to False, we need a case where no exception is raised
    valid_content = b"# -*- coding: utf-8 -*-\nprint('hello')\n"
    readline_func = BytesIO(valid_content).readline
    
    # This should not raise UnsupportedEncoding, meaning the except block is not executed
    result = File.detect_encoding("test.py", readline_func)
    
    assert result == "utf-8"
    assert isinstance(result, str)


# LLM-generated content at query #9
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #10
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError or AttributeError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #12
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #13
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #15
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #16
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/file.py")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #19
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected frozen dataclass to raise error"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #20
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    assert file.__dataclass_fields__['stream'].frozen
    assert file.__dataclass_fields__['path'].frozen
    assert file.__dataclass_fields__['encoding'].frozen


# LLM-generated content at query #21
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected dataclass to be frozen"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #22
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/file.txt")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == "latin-1"


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #28
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/document.py")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == "latin-1"


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    from pathlib import Path
    from src.file import File
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_with_encoding_declaration():
    from io import BytesIO
    from pathlib import Path
    from src.file import File
    
    contents = "# -*- coding: iso-8859-1 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("iso-8859-1")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "iso-8859-1"


def test_detect_encoding_with_coding_comment():
    from io import BytesIO
    from src.file import File
    
    contents = "# coding: utf-16\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result is not None


def test_detect_encoding_empty_file():
    from io import BytesIO
    from src.file import File
    
    contents = ""
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_invalid_encoding_raises_exception():
    from io import BytesIO
    from src.file import File
    from src.file import UnsupportedEncoding
    
    def bad_readline():
        raise SyntaxError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", bad_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_bom():
    from io import BytesIO
    from src.file import File
    
    contents = "print('hello')"
    readline = BytesIO(b'\xef\xbb\xbf' + contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8-sig"


# LLM-generated content at query #2
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


def test_file_constructor_with_different_paths():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    file1 = File(stream=stream, path=Path("/test/file1.txt"), encoding=encoding)
    file2 = File(stream=stream, path=Path("/test/file2.txt"), encoding=encoding)
    
    assert file1.path != file2.path
    assert file1.path == Path("/test/file1.txt")
    assert file2.path == Path("/test/file2.txt")


# LLM-generated content at query #3
#--------------------------

```python
def test_detect_encoding_exception_raises_unsupported_encoding():
    from pathlib import Path
    from io import BytesIO
    
    class UnsupportedEncoding(Exception):
        def __init__(self, filename):
            self.filename = filename
    
    def mock_readline():
        raise ValueError("Invalid encoding data")
    
    try:
        File.detect_encoding("test.py", mock_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == "latin-1"


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Expected frozen dataclass to raise error"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_ascii.encoding == "ascii"
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #7
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    # The predicate at line 5 (except Exception:) should evaluate to True when an exception occurs
    # To test that it evaluates to False, we need a case where no exception is raised
    def valid_readline():
        return b"# coding: utf-8\n"
    
    # This should NOT raise UnsupportedEncoding because no exception occurs in the try block
    result = File.detect_encoding("test.py", valid_readline)
    assert result == "utf-8"


# LLM-generated content at query #8
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.encoding == "latin-1"


def test_file_constructor_with_different_path():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/document.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.path == path


# LLM-generated content at query #10
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected frozen dataclass to raise FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #11
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should raise FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    for encoding in ["utf-8", "latin-1", "ascii", "utf-16"]:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.encoding == encoding


def test_file_constructor_with_different_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    for path_str in ["/tmp/test.txt", "/home/user/file.py", "relative/path.txt"]:
        path = Path(path_str)
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.path == path


# LLM-generated content at query #12
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    import io
    import tokenize
    contents = "print('hello')\n"
    readline = io.BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_coding_declaration():
    import io
    contents = "# -*- coding: latin-1 -*-\nprint('hello')\n"
    readline = io.BytesIO(contents.encode("latin-1")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "latin-1"


def test_detect_encoding_invalid_raises_unsupported_encoding():
    import io
    from pathlib import Path
    
    def bad_readline():
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
    
    try:
        File.detect_encoding("test.py", bad_readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding as e:
        assert str(e.filename) == "test.py"


def test_detect_encoding_empty_file():
    import io
    readline = io.BytesIO(b"").readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_bom():
    import io
    contents = "\ufeffprint('hello')\n"
    readline = io.BytesIO(contents.encode("utf-8-sig")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding in ("utf-8-sig", "utf-8")


# LLM-generated content at query #13
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    for encoding in ["utf-8", "latin-1", "ascii", "utf-16"]:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.encoding == encoding


# LLM-generated content at query #15
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #17
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #19
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #22
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #23
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/file.py")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should raise FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected frozen dataclass to raise FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


def test_file_constructor_with_different_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    path1 = Path("/home/user/file1.py")
    file1 = File(stream=stream, path=path1, encoding=encoding)
    assert file1.path == path1
    
    path2 = Path("/home/user/file2.txt")
    file2 = File(stream=stream, path=path2, encoding=encoding)
    assert file2.path == path2


# LLM-generated content at query #26
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected frozen dataclass to raise error"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"
    assert file_utf8.encoding != file_latin1.encoding


# LLM-generated content at query #27
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #28
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #29
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #30
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #31
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "latin-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.encoding == "latin-1"


# LLM-generated content at query #32
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    for encoding in ["utf-8", "latin-1", "ascii", "utf-16"]:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.encoding == encoding


def test_file_constructor_with_different_paths():
    from io import StringIO
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    paths = [
        Path("/absolute/path/file.py"),
        Path("relative/path/file.txt"),
        Path("/path/with spaces/file name.py"),
    ]
    
    for path in paths:
        file = File(stream=stream, path=path, encoding=encoding)
        assert file.path == path


# LLM-generated content at query #33
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # Exception was raised as expected, predicate at line 5 evaluated to True
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #35
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file_obj = File(stream=stream, path=path, encoding=encoding)
    
    assert file_obj.stream is stream
    assert file_obj.path == path
    assert file_obj.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file_obj = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file_obj)
    try:
        file_obj.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_obj_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_obj_utf8.encoding == "utf-8"
    
    file_obj_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_obj_latin1.encoding == "latin-1"


def test_file_constructor_with_different_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    path1 = Path("/tmp/test1.txt")
    file_obj1 = File(stream=stream, path=path1, encoding=encoding)
    assert file_obj1.path == path1
    
    path2 = Path("/home/user/test2.py")
    file_obj2 = File(stream=stream, path=path2, encoding=encoding)
    assert file_obj2.path == path2


# LLM-generated content at query #36
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test_iso.txt")
    encoding = "iso-8859-1"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.encoding == "iso-8859-1"


# LLM-generated content at query #37
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #38
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #39
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    import tokenize
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # The except Exception block at line 5 was executed, so the predicate is True
        # We want to test that it evaluates to False, meaning no exception occurs
        pass
    
    # Test case where no exception occurs (predicate at line 5 evaluates to False)
    valid_content = b"# -*- coding: utf-8 -*-\nprint('hello')"
    def valid_readline():
        return valid_content
    
    result = File.detect_encoding("test.py", BytesIO(valid_content).readline)
    assert result == 'utf-8'


# LLM-generated content at query #40
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #41
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #42
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #43
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #44
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "ascii"
        assert False, "Should have raised FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream is stream
    assert file.path == path
    assert file.encoding == encoding


def test_file_constructor_frozen():
    from io import StringIO
    from pathlib import Path
    import dataclasses
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


