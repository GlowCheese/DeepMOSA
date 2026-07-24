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


def test_detect_encoding_with_explicit_encoding():
    from io import BytesIO
    
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "latin-1"


def test_detect_encoding_default_utf8():
    from io import BytesIO
    
    contents = "print('hello')\n"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_with_vim_encoding():
    from io import BytesIO
    
    contents = "# vim: set fileencoding=iso-8859-1 :\nprint('hello')"
    readline = BytesIO(contents.encode("iso-8859-1")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "iso-8859-1"


def test_detect_encoding_raises_unsupported_encoding_on_invalid_readline():
    from io import BytesIO
    
    def broken_readline():
        raise Exception("Broken readline")
    
    try:
        File.detect_encoding("test.py", broken_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_detect_encoding_exception_handler_evaluates_to_false():
    from io import BytesIO
    from pathlib import Path
    
    # Create a valid UTF-8 encoded content that won't raise an exception
    valid_content = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline_func = BytesIO(valid_content.encode("utf-8")).readline
    
    # Call detect_encoding with valid input - should NOT raise exception
    result = File.detect_encoding("test.py", readline_func)
    
    # Assert that the function returns successfully (exception handler not triggered)
    assert result == "utf-8"
    assert isinstance(result, str)


# LLM-generated content at query #3
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
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #5
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


def test_file_extension_property():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "py"


def test_file_extension_property_no_extension():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == ""


def test_file_extension_property_multiple_dots():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.tar.gz")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "gz"


# LLM-generated content at query #6
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    readline = BytesIO(b"# -*- coding: utf-8 -*-\nprint('hello')").readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_latin1():
    from io import BytesIO
    readline = BytesIO(b"# coding: latin-1\nprint('hello')").readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "latin-1"


def test_detect_encoding_default():
    from io import BytesIO
    readline = BytesIO(b"print('hello')\nprint('world')").readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding in ("utf-8", "utf8")


def test_detect_encoding_unsupported_raises_exception():
    from io import BytesIO
    from pathlib import Path
    
    def failing_readline():
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_path_object():
    from io import BytesIO
    from pathlib import Path
    readline = BytesIO(b"# coding: utf-8\n").readline
    encoding = File.detect_encoding(Path("test.py"), readline)
    assert encoding == "utf-8"


# LLM-generated content at query #7
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
    assert file_utf8.encoding == "utf-8"
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #8
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
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #9
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
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # Exception handler at line 5 was executed (predicate evaluated to True)
        pass
    
    # Test case where no exception occurs (predicate evaluates to False)
    valid_readline = BytesIO(b"# -*- coding: utf-8 -*-\nprint('hello')\n").readline
    result = File.detect_encoding("test.py", valid_readline)
    assert result == "utf-8"


# LLM-generated content at query #10
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
        assert False, "Expected frozen dataclass to raise an error"
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
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #12
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
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


def test_file_constructor_with_various_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    path1 = Path("/absolute/path/file.txt")
    file1 = File(stream=stream, path=path1, encoding=encoding)
    assert file1.path == path1
    
    path2 = Path("relative/path/file.txt")
    file2 = File(stream=stream, path=path2, encoding=encoding)
    assert file2.path == path2


# LLM-generated content at query #13
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
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


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


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/file.py")
    encoding = "iso-8859-1"
    
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
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except Exception:
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


def test_file_extension_property_hidden_file():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/.gitignore")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "gitignore"


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


# LLM-generated content at query #18
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
    except (AttributeError, dataclasses.FrozenInstanceError):
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


# LLM-generated content at query #20
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
    path = Path("/test/file.txt")
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
        file.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #21
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


def test_file_extension_property():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "py"


def test_file_extension_property_no_extension():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/Makefile")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == ""


def test_file_extension_property_multiple_dots():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.tar.gz")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "gz"


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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
    
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.encoding = "latin-1"
        assert False, "Expected dataclass to be frozen"
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
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


def test_file_constructor_with_different_paths():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    encoding = "utf-8"
    
    path1 = Path("/home/user/file.py")
    file1 = File(stream=stream, path=path1, encoding=encoding)
    assert file1.path == path1
    
    path2 = Path("./relative/path.txt")
    file2 = File(stream=stream, path=path2, encoding=encoding)
    assert file2.path == path2


# LLM-generated content at query #25
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
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, Exception):
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
        file.encoding = "latin-1"
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
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_file_constructor():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file_obj = File(stream=stream, path=path, encoding=encoding)
    
    assert file_obj.stream is stream
    assert file_obj.path == path
    assert file_obj.encoding == encoding


def test_file_constructor_frozen():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file_obj = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file_obj.encoding = "latin-1"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
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


# LLM-generated content at query #31
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from io import BytesIO
    from pathlib import Path
    
    def failing_readline():
        raise ValueError("Test error")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        assert True


# LLM-generated content at query #32
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
        assert False, "Expected frozen dataclass to raise error"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #33
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
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"
    assert file_ascii.encoding == "ascii"


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


# LLM-generated content at query #35
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
    
    assert dataclasses.is_dataclass(File)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


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
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #37
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
        assert False, "Expected frozen dataclass to raise FrozenInstanceError"
    except AttributeError:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_detect_encoding_exception_handling():
    from pathlib import Path
    from io import BytesIO
    
    # Create a readline function that will cause tokenize.detect_encoding to raise an exception
    def failing_readline():
        raise SyntaxError("Invalid encoding declaration")
    
    filename = "test.py"
    
    try:
        File.detect_encoding(filename, failing_readline)
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        # This is the expected path - the except Exception clause at line 5 evaluates to True
        # and raises UnsupportedEncoding
        assert True


# LLM-generated content at query #39
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


# LLM-generated content at query #40
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


# LLM-generated content at query #41
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
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #42
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream == stream
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


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
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
    except AttributeError:
        pass


def test_file_constructor_with_different_encodings():
    from pathlib import Path
    from io import StringIO
    
    stream = StringIO("test")
    path = Path("/test/file.py")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"


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
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    from pathlib import Path
    from flake8_plugin_utils import File
    
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_coding_declaration():
    from io import BytesIO
    from pathlib import Path
    from flake8_plugin_utils import File
    
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "iso8859-1"


def test_detect_encoding_default_utf8():
    from io import BytesIO
    from flake8_plugin_utils import File
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_invalid_raises_exception():
    from io import BytesIO
    from flake8_plugin_utils import File, UnsupportedEncoding
    
    readline = BytesIO(b"").readline
    try:
        File.detect_encoding("test.py", readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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
        assert False, "Expected dataclass to be frozen"
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
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #3
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    from pathlib import Path
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_valid_latin1():
    from io import BytesIO
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding in ("utf-8", "latin-1")


def test_detect_encoding_no_explicit_encoding():
    from io import BytesIO
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_invalid_raises_exception():
    from io import BytesIO
    
    class FailingReadline:
        def __call__(self):
            raise Exception("readline failed")
    
    try:
        File.detect_encoding("test.py", FailingReadline())
        assert False, "Expected UnsupportedEncoding to be raised"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_path_object():
    from io import BytesIO
    from pathlib import Path
    contents = "# coding: utf-8\nprint('test')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding(Path("test.py"), readline)
    assert encoding == "utf-8"


def test_detect_encoding_empty_file():
    from io import BytesIO
    contents = ""
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


# LLM-generated content at query #4
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_with_coding_declaration():
    from io import BytesIO
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding in ["latin-1", "iso8859-1"]


def test_detect_encoding_default_utf8():
    from io import BytesIO
    contents = "print('hello world')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_invalid_raises_unsupported_encoding():
    from io import BytesIO
    def failing_readline():
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_vim_style_encoding():
    from io import BytesIO
    contents = "# vim: set fileencoding=utf-8 :\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


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
    
    stream = StringIO("test content")
    path = Path("/tmp/test.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    try:
        file.stream = StringIO("new content")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, dataclasses.FrozenInstanceError):
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
        file.stream = StringIO("new content")
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
    result = File.detect_encoding("test.py", readline)
    assert isinstance(result, str)
    assert result == "utf-8"


def test_detect_encoding_with_latin1():
    from io import BytesIO
    
    contents = "# coding: latin-1\nprint('hello')"
    readline = BytesIO(contents.encode("latin-1")).readline
    result = File.detect_encoding("test.py", readline)
    assert isinstance(result, str)


def test_detect_encoding_default():
    from io import BytesIO
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert isinstance(result, str)


def test_detect_encoding_invalid_raises_exception():
    from io import BytesIO
    
    def failing_readline():
        raise ValueError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding:
        pass


def test_detect_encoding_with_bom():
    from io import BytesIO
    
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8-sig")).readline
    result = File.detect_encoding("test.py", readline)
    assert isinstance(result, str)


# LLM-generated content at query #8
#--------------------------

```python
def test_detect_encoding_valid_utf8():
    from io import BytesIO
    from pathlib import Path
    
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_with_latin1():
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


def test_detect_encoding_with_vim_encoding():
    from io import BytesIO
    
    contents = "# vim: set fileencoding=utf-8 :\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    result = File.detect_encoding("test.py", readline)
    assert result == "utf-8"


def test_detect_encoding_invalid_raises_exception():
    from io import BytesIO
    
    def failing_readline():
        raise ValueError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", failing_readline)
        assert False, "Should raise UnsupportedEncoding"
    except UnsupportedEncoding:
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
    assert file_utf8.encoding == "utf-8"
    
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
        file.encoding = "latin-1"
        assert False, "Expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #11
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


# LLM-generated content at query #12
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


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.stream == stream
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


# LLM-generated content at query #15
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


# LLM-generated content at query #17
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
        assert False, "Should have raised FrozenInstanceError"
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


# LLM-generated content at query #19
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
        assert False, "Expected dataclass to be frozen"
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


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/home/user/file.py")
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
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


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


def test_file_extension_property():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/test.py")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "py"


def test_file_extension_property_no_extension():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/Makefile")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == ""


def test_file_extension_property_dot_file():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/tmp/.gitignore")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert file.extension == "gitignore"


# LLM-generated content at query #22
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
        file.encoding = "ascii"
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, Exception):
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
    
    assert file.stream == stream
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


# LLM-generated content at query #24
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
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"


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
        file.stream = StringIO("new content")
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


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
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, TypeError):
        pass


# LLM-generated content at query #27
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


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
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
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    try:
        file.encoding = "ascii"
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #28
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
        file.stream = StringIO("new content")
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


# LLM-generated content at query #30
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


def test_file_constructor_with_different_encoding():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test content")
    path = Path("/test/file.py")
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
    path = Path("/test/file.txt")
    encoding = "utf-8"
    
    file = File(stream=stream, path=path, encoding=encoding)
    
    assert dataclasses.is_dataclass(file)
    assert file.__dataclass_fields__['stream'].frozen == True


# LLM-generated content at query #31
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
    
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    assert file_ascii.encoding == "ascii"


# LLM-generated content at query #32
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
    assert file_utf8.encoding == "utf-8"
    
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    assert file_latin1.encoding == "latin-1"


# LLM-generated content at query #33
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
    contents = "# -*- coding: latin-1 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding in ("utf-8", "latin-1")


def test_detect_encoding_empty_file():
    from io import BytesIO
    contents = ""
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


def test_detect_encoding_invalid_raises_unsupported_encoding():
    from io import BytesIO
    import sys
    
    class FailingReadline:
        def __call__(self):
            raise SyntaxError("Invalid encoding")
    
    try:
        File.detect_encoding("test.py", FailingReadline())
        assert False, "Should have raised UnsupportedEncoding"
    except UnsupportedEncoding as e:
        assert "test.py" in str(e) or e.args[0] == "test.py"


def test_detect_encoding_with_bom():
    from io import BytesIO
    contents = "print('hello')"
    readline = BytesIO(contents.encode("utf-8-sig")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding in ("utf-8-sig", "utf-8")


def test_detect_encoding_multiple_lines():
    from io import BytesIO
    contents = "# coding: utf-8\nprint('hello')\nprint('world')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding("test.py", readline)
    assert encoding == "utf-8"


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
    except (AttributeError, dataclasses.FrozenInstanceError):
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


def test_file_constructor_with_different_encodings():
    from io import StringIO
    from pathlib import Path
    
    stream = StringIO("test")
    path = Path("/tmp/test.txt")
    
    file_utf8 = File(stream=stream, path=path, encoding="utf-8")
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"
    assert file_utf8.encoding != file_latin1.encoding


# LLM-generated content at query #36
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
        assert False, "Should have raised FrozenInstanceError"
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


# LLM-generated content at query #37
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
    file_latin1 = File(stream=stream, path=path, encoding="latin-1")
    file_ascii = File(stream=stream, path=path, encoding="ascii")
    
    assert file_utf8.encoding == "utf-8"
    assert file_latin1.encoding == "latin-1"
    assert file_ascii.encoding == "ascii"


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


