####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_detect_encoding_with_valid_utf8():
    filename = "test.py"
    contents = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nprint('Hello, world!')"
    readline = BytesIO(contents.encode("utf-8")).readline
    assert File.detect_encoding(filename, readline) == "utf-8"

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    contents = "invalid encoding content"
    readline = BytesIO(contents.encode("latin-1")).readline
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding(filename, readline)


# LLM-generated content at query #2
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.txt", lambda: b"invalid")


# LLM-generated content at query #3
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #4
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #5
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #6
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.py", lambda: b"invalid")


# LLM-generated content at query #7
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with raises(UnsupportedEncoding):
        File.detect_encoding("test.txt", lambda: b"invalid")


# LLM-generated content at query #8
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #9
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #10
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #11
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #12
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #13
#--------------------------

```python
def test_file_constructor_initialization():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #15
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #16
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #17
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #18
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #19
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #20
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #21
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #22
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)

    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #23
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #24
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #25
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #26
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #27
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt").resolve()
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #28
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #29
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #30
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_file_constructor_creates_immutable_instance():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.stream = StringIO("new content")
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.path = Path("/new/path.txt")
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.encoding = "ascii"


# LLM-generated content at query #2
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #3
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #4
#--------------------------

```python
def test_detect_encoding_with_valid_utf8_file():
    filename = "test.py"
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    assert File.detect_encoding(filename, readline) == "utf-8"

def test_detect_encoding_with_valid_ascii_file():
    filename = "test.py"
    contents = "print('hello')"
    readline = BytesIO(contents.encode("ascii")).readline
    assert File.detect_encoding(filename, readline) == "utf-8"

def test_detect_encoding_with_unsupported_encoding():
    filename = "test.py"
    contents = "# -*- coding: invalid-encoding -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    try:
        File.detect_encoding(filename, readline)
        assert False, "Expected UnsupportedEncoding exception"
    except UnsupportedEncoding as e:
        assert str(e) == filename


# LLM-generated content at query #5
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #6
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.txt", lambda: b"invalid")


# LLM-generated content at query #7
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #8
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #9
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #10
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #11
#--------------------------

```python
def test_file_constructor_initialization():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #12
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with raises(UnsupportedEncoding):
        File.detect_encoding("test.txt", lambda: b"invalid")


# LLM-generated content at query #13
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #15
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #16
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #17
#--------------------------

```python
def test_detect_encoding_with_valid_utf8_file():
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    encoding = File.detect_encoding("test.py", BytesIO(contents.encode("utf-8")).readline)
    assert encoding == "utf-8"

def test_detect_encoding_with_valid_ascii_file():
    contents = "print('hello')"
    encoding = File.detect_encoding("test.py", BytesIO(contents.encode("ascii")).readline)
    assert encoding == "utf-8"

def test_detect_encoding_with_unsupported_encoding():
    contents = "# -*- coding: invalid-encoding -*-\nprint('hello')"
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.py", BytesIO(contents.encode("utf-8")).readline)


# LLM-generated content at query #18
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #19
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #20
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #21
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #22
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #23
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #24
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #25
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #26
#--------------------------

```python
def test_file_constructor_creates_immutable_instance():
    stream = StringIO("test content")
    path = Path("/test/path")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.stream = StringIO("new content")
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.path = Path("/new/path")
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.encoding = "latin-1"


# LLM-generated content at query #27
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #28
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #29
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #30
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


