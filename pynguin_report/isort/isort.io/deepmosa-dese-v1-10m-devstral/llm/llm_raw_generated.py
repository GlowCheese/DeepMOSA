####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_detect_encoding_with_valid_utf8_file():
    filename = "test.py"
    contents = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nprint('Hello, world!')"
    encoding = File.detect_encoding(filename, BytesIO(contents.encode("utf-8")).readline)
    assert encoding == "utf-8"

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    contents = b"invalid encoding content"
    try:
        File.detect_encoding(filename, BytesIO(contents).readline)
        assert False, "Expected UnsupportedEncoding exception"
    except UnsupportedEncoding:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_detect_encoding_with_valid_utf8_file():
    filename = "test.py"
    contents = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nprint('Hello, world!')"
    encoding = File.detect_encoding(filename, BytesIO(contents.encode("utf-8")).readline)
    assert encoding == "utf-8"

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    contents = b"\x00\x01\x02\x03"
    try:
        File.detect_encoding(filename, BytesIO(contents).readline)
    except UnsupportedEncoding as e:
        assert str(e) == filename
    else:
        assert False, "Expected UnsupportedEncoding exception"


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
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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
    with pytest.raises(FrozenInstanceError):
        file.stream = StringIO("new content")
    with pytest.raises(FrozenInstanceError):
        file.path = Path("/new/path")
    with pytest.raises(FrozenInstanceError):
        file.encoding = "latin-1"


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
def test_detect_encoding_with_valid_utf8_file():
    contents = "# -*- coding: utf-8 -*-\nprint('Hello, world!')"
    filename = "test.py"
    encoding = File.detect_encoding(filename, BytesIO(contents.encode("utf-8")).readline)
    assert encoding == "utf-8"

def test_detect_encoding_with_valid_ascii_file():
    contents = "print('Hello, world!')"
    filename = "test.py"
    encoding = File.detect_encoding(filename, BytesIO(contents.encode("ascii")).readline)
    assert encoding == "ascii"

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    try:
        File.detect_encoding(filename, BytesIO(b"invalid encoding").readline)
        assert False, "Expected UnsupportedEncoding exception"
    except UnsupportedEncoding:
        pass


# LLM-generated content at query #10
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
    assert File.detect_encoding(filename, readline) == "ascii"

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    contents = b"\x80invalid"
    readline = BytesIO(contents).readline
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding(filename, readline)


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
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.py")
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
    path = Path("test.txt")
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
    path = Path("/test/path")
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
    path = Path("/test/path.txt")
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
    path = Path("test.txt").resolve()
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
def test_file_constructor_creates_immutable_instance():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding
    with pytest.raises(FrozenInstanceError):
        file.stream = StringIO("new content")
    with pytest.raises(FrozenInstanceError):
        file.path = Path("/new/path.txt")
    with pytest.raises(FrozenInstanceError):
        file.encoding = "ascii"


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
    path = Path("test.txt").resolve()
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


# LLM-generated content at query #31
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


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.txt", lambda: b"invalid")


# LLM-generated content at query #34
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


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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
        file.encoding = "new-encoding"


# LLM-generated content at query #37
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.txt", lambda: b"invalid")


# LLM-generated content at query #38
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


# LLM-generated content at query #39
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


# LLM-generated content at query #40
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


# LLM-generated content at query #41
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


# LLM-generated content at query #42
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


# LLM-generated content at query #43
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


# LLM-generated content at query #44
#--------------------------

```python
def test_detect_encoding_with_valid_utf8():
    filename = "test.py"
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    assert File.detect_encoding(filename, readline) == "utf-8"

def test_detect_encoding_with_valid_ascii():
    filename = "test.py"
    contents = "print('hello')"
    readline = BytesIO(contents.encode("ascii")).readline
    assert File.detect_encoding(filename, readline) == "ascii"

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    contents = b"\x80invalid"
    readline = BytesIO(contents).readline
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding(filename, readline)


# LLM-generated content at query #45
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


# LLM-generated content at query #46
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


# LLM-generated content at query #47
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


# LLM-generated content at query #48
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


# LLM-generated content at query #49
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


# LLM-generated content at query #50
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


# LLM-generated content at query #51
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


# LLM-generated content at query #52
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.py")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #53
#--------------------------

```python
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    assert_raises(
        UnsupportedEncoding,
        File.detect_encoding,
        "test.py",
        lambda: (_ for _ in ()).throw(Exception)
    )


# LLM-generated content at query #54
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
        file.encoding = "new-encoding"


# LLM-generated content at query #55
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


# LLM-generated content at query #56
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


# LLM-generated content at query #57
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_detect_encoding_with_valid_utf8_file():
    filename = "test.py"
    contents = "# -*- coding: utf-8 -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    encoding = File.detect_encoding(filename, readline)
    assert encoding == "utf-8"

def test_detect_encoding_with_valid_ascii_file():
    filename = "test.py"
    contents = "print('hello')"
    readline = BytesIO(contents.encode("ascii")).readline
    encoding = File.detect_encoding(filename, readline)
    assert encoding == "ascii"

def test_detect_encoding_with_unsupported_encoding():
    filename = "test.py"
    contents = "# -*- coding: invalid-encoding -*-\nprint('hello')"
    readline = BytesIO(contents.encode("utf-8")).readline
    try:
        File.detect_encoding(filename, readline)
        assert False, "Expected UnsupportedEncoding exception"
    except UnsupportedEncoding as e:
        assert str(e) == filename


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
def test_detect_encoding_raises_unsupported_encoding_on_exception():
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding("test.py", lambda: b"invalid")


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
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("test.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding


# LLM-generated content at query #7
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
def test_file_constructor_creates_immutable_instance():
    stream = StringIO("test content")
    path = Path("/test/path.txt")
    encoding = "utf-8"
    file = File(stream=stream, path=path, encoding=encoding)
    assert file.stream == stream
    assert file.path == path
    assert file.encoding == encoding
    with pytest.raises(FrozenInstanceError):
        file.stream = StringIO("new content")
    with pytest.raises(FrozenInstanceError):
        file.path = Path("/new/path.txt")
    with pytest.raises(FrozenInstanceError):
        file.encoding = "ascii"


# LLM-generated content at query #10
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

def test_detect_encoding_with_invalid_encoding():
    filename = "test.py"
    contents = b"\x80\x81"
    readline = BytesIO(contents).readline
    with pytest.raises(UnsupportedEncoding):
        File.detect_encoding(filename, readline)


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
def test_detect_encoding_raises_unsupported_encoding():
    assert_raises(UnsupportedEncoding, File.detect_encoding, "test.txt", lambda: b"invalid encoding")


# LLM-generated content at query #13
#--------------------------

```python
def test_file_constructor_with_valid_args():
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
    path = Path("test.txt").resolve()
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
    path = Path("test.txt")
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
        file.encoding = "latin-1"


# LLM-generated content at query #23
#--------------------------

```python
def test_file_constructor():
    stream = StringIO("test content")
    path = Path("/test/path.py")
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
    path = Path("test.txt")
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


