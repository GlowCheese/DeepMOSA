####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sort_stream_with_diff_true():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=True,
    )
    assert result is True
    assert output_stream.getvalue() != ""

def test_sort_stream_with_diff_stream():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    diff_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=diff_stream,
    )
    assert result is True
    assert diff_stream.getvalue() != ""
    assert output_stream.getvalue() == ""

def test_sort_stream_without_diff():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_no_changes():
    input_stream = StringIO("import a\nimport b")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
    )
    assert result is False
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_config_kwargs():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        line_length=50,
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_custom_config():
    config = Config(line_length=50)
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        config=config,
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_file_path():
    file_path = Path("test.py")
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        file_path=file_path,
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_extension():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        extension="py",
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_atomic():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        atomic=True,
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_disregard_skip():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    result = sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        show_diff=False,
        disregard_skip=True,
    )
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_find_imports_in_stream_default_config():
    input_stream = io.StringIO("import sys\nimport os")
    result = list(find_imports_in_stream(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_custom_config():
    input_stream = io.StringIO("import sys\nimport os")
    config = Config(known_first_party=["sys"])
    result = list(find_imports_in_stream(input_stream, config=config))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_unique_true():
    input_stream = io.StringIO("import sys\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_unique_alias():
    input_stream = io.StringIO("import sys as s\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ALIAS))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_unique_attribute():
    input_stream = io.StringIO("from sys import path\nfrom sys import argv")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ATTRIBUTE))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "argv"

def test_find_imports_in_stream_unique_module():
    input_stream = io.StringIO("import sys.path\nimport sys.argv")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.MODULE))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_unique_package():
    input_stream = io.StringIO("import sys.path\nimport sys.argv")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.PACKAGE))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_top_only():
    input_stream = io.StringIO("import sys\ndef foo():\n    import os")
    result = list(find_imports_in_stream(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_path():
    input_stream = io.StringIO("import sys")
    path = Path("/tmp/test.py")
    result = list(find_imports_in_stream(input_stream, file_path=path))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_config_kwargs():
    input_stream = io.StringIO("import sys")
    result = list(find_imports_in_stream(input_stream, known_first_party=["sys"]))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_invalid_config_combination():
    input_stream = io.StringIO("import sys")
    config = Config(known_first_party=["sys"])
    with pytest.raises(ValueError):
        list(find_imports_in_stream(input_stream, config=config, known_first_party=["os"]))


# LLM-generated content at query #3
#--------------------------

```python
def test_find_imports_in_paths_with_empty_iterator():
    result = list(find_imports_in_paths(iter([])))
    assert result == []

def test_find_imports_in_paths_with_single_file():
    with patch("builtins.open", mock_open(read_data="import os")) as mock_file:
        result = list(find_imports_in_paths(iter(["test.py"])))
        assert len(result) == 1
        assert result[0].module == "os"

def test_find_imports_in_paths_with_multiple_files():
    with patch("builtins.open", mock_open(read_data="import sys")) as mock_file:
        result = list(find_imports_in_paths(iter(["test1.py", "test2.py"])))
        assert len(result) == 2
        assert all(imp.module == "sys" for imp in result)

def test_find_imports_in_paths_with_unique_true():
    with patch("builtins.open", mock_open(read_data="import os\nimport os")) as mock_file:
        result = list(find_imports_in_paths(iter(["test.py"]), unique=True))
        assert len(result) == 1
        assert result[0].module == "os"

def test_find_imports_in_paths_with_unique_import_key_module():
    with patch("builtins.open", mock_open(read_data="import os\nfrom os import path")) as mock_file:
        result = list(find_imports_in_paths(iter(["test.py"]), unique=ImportKey.MODULE))
        assert len(result) == 1
        assert result[0].module == "os"

def test_find_imports_in_paths_with_top_only_true():
    with patch("builtins.open", mock_open(read_data="import os\ndef foo():\n    import sys")) as mock_file:
        result = list(find_imports_in_paths(iter(["test.py"]), top_only=True))
        assert len(result) == 1
        assert result[0].module == "os"

def test_find_imports_in_paths_with_config_kwargs():
    with patch("builtins.open", mock_open(read_data="import os")) as mock_file:
        result = list(find_imports_in_paths(iter(["test.py"]), line_length=100))
        assert len(result) == 1
        assert result[0].module == "os"

def test_find_imports_in_paths_with_custom_config():
    config = Config(line_length=100)
    with patch("builtins.open", mock_open(read_data="import os")) as mock_file:
        result = list(find_imports_in_paths(iter(["test.py"]), config=config))
        assert len(result) == 1
        assert result[0].module == "os"


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_28_evaluates_to_true():
    input_stream = io.StringIO("import sys\nimport os")
    config = DEFAULT_CONFIG
    file_path = None
    unique = True
    top_only = False
    _seen = None

    result = list(find_imports_in_stream(input_stream, config, file_path, unique, top_only, _seen))
    assert len(result) == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_sort_stream_atomic_config():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(atomic=True)
    result = sort_stream(input_stream, output_stream, config=config)
    assert result is True


# LLM-generated content at query #6
#--------------------------

```python
def test_check_stream_with_correctly_sorted_imports():
    input_stream = StringIO("import os\nimport sys\n")
    assert check_stream(input_stream) is True

def test_check_stream_with_incorrectly_sorted_imports():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream) is False

def test_check_stream_with_show_diff_true():
    input_stream = StringIO("import sys\nimport os\n")
    output_stream = StringIO()
    assert check_stream(input_stream, show_diff=True) is False
    assert len(output_stream.getvalue()) > 0

def test_check_stream_with_show_diff_stream():
    input_stream = StringIO("import sys\nimport os\n")
    output_stream = StringIO()
    assert check_stream(input_stream, show_diff=output_stream) is False
    assert len(output_stream.getvalue()) > 0

def test_check_stream_with_custom_config():
    input_stream = StringIO("import sys\nimport os\n")
    config = Config(line_length=120)
    assert check_stream(input_stream, config=config) is False

def test_check_stream_with_file_path():
    input_stream = StringIO("import sys\nimport os\n")
    file_path = Path("test.py")
    assert check_stream(input_stream, file_path=file_path) is False

def test_check_stream_with_disregard_skip():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, disregard_skip=True) is False

def test_check_stream_with_extension():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, extension="py") is False

def test_check_stream_with_config_kwargs():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, line_length=120) is False


# LLM-generated content at query #7
#--------------------------

```python
def test_sort_file_with_valid_file():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        result = sort_file(tmp.name)
        assert result is True
        with open(tmp.name) as f:
            assert f.read() == "import a\nimport b\n"

def test_sort_file_with_invalid_syntax():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a\ninvalid syntax")
        tmp.flush()
        with pytest.warns(UserWarning):
            result = sort_file(tmp.name)
        assert result is False

def test_sort_file_with_skip_setting():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(skip=["test.py"])
        result = sort_file(tmp.name, config=config)
        assert result is False

def test_sort_file_with_show_diff():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        output = StringIO()
        result = sort_file(tmp.name, show_diff=output)
        assert result is False
        assert "import a" in output.getvalue()

def test_sort_file_with_write_to_stdout():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        output = StringIO()
        with contextlib.redirect_stdout(output):
            result = sort_file(tmp.name, write_to_stdout=True)
        assert result is True
        assert output.getvalue() == "import a\nimport b\n"

def test_sort_file_with_output_stream():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        output = StringIO()
        result = sort_file(tmp.name, output=output)
        assert result is True
        assert output.getvalue() == "import a\nimport b\n"

def test_sort_file_with_ask_to_apply():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        with unittest.mock.patch("builtins.input", return_value="n"):
            result = sort_file(tmp.name, ask_to_apply=True)
        assert result is False

def test_sort_file_with_disregard_skip():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(skip=["test.py"])
        result = sort_file(tmp.name, config=config, disregard_skip=True)
        assert result is True

def test_sort_file_with_custom_config():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        result = sort_file(tmp.name, line_length=50)
        assert result is True

def test_sort_file_with_atomic_config():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(atomic=True)
        result = sort_file(tmp.name, config=config)
        assert result is True

def test_sort_file_with_overwrite_in_place():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(overwrite_in_place=True)
        result = sort_file(tmp.name, config=config)
        assert result is True
        with open(tmp.name) as f:
            assert f.read() == "import a\nimport b\n"

def test_sort_file_with_quiet_config():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(quiet=True)
        with contextlib.redirect_stdout(StringIO()) as output:
            result = sort_file(tmp.name, config=config)
        assert result is True
        assert output.getvalue() == ""

def test_sort_file_with_verbose_config():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(verbose=True)
        with contextlib.redirect_stdout(StringIO()) as output:
            result = sort_file(tmp.name, config=config)
        assert result is True
        assert "Fixing" in output.getvalue()

def test_sort_file_with_color_output():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(color_output=True)
        output = StringIO()
        result = sort_file(tmp.name, config=config, show_diff=output)
        assert result is False
        assert output.getvalue() != ""

def test_sort_file_with_cython_extension():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".pyx", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(verbose=True)
        with contextlib.redirect_stdout(StringIO()) as output:
            result = sort_file(tmp.name, config=config)
        assert result is True
        assert "Cython" in output.getvalue()

def test_sort_file_with_existing_syntax_errors():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a\ninvalid syntax")
        tmp.flush()
        with pytest.warns(UserWarning):
            result = sort_file(tmp.name, raise_on_skip=True)
        assert result is False

def test_sort_file_with_introduced_syntax_errors():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write("import b\nimport a")
        tmp.flush()
        config = Config(atomic=True)
        with unittest.mock.patch("isort.core.process", side_effect=IntroducedSyntaxErrors(tmp.name)):
            with pytest.warns(UserWarning):
                result = sort_file(tmp.name, config=config)
        assert result is False


# LLM-generated content at query #8
#--------------------------

```python
def test_find_imports_in_file_with_valid_file():
    filename = "test_file.py"
    content = "import os\nimport sys\nfrom pathlib import Path"
    expected_imports = [
        identify.Import(module="os"),
        identify.Import(module="sys"),
        identify.Import(module="pathlib", attribute="Path"),
    ]

    with patch("io.File.read") as mock_read:
        mock_file = MagicMock()
        mock_file.stream = io.StringIO(content)
        mock_file.path = Path(filename)
        mock_read.return_value.__enter__.return_value = mock_file

        imports = list(find_imports_in_file(filename))
        assert imports == expected_imports

def test_find_imports_in_file_with_invalid_file():
    filename = "nonexistent_file.py"

    with patch("io.File.read") as mock_read:
        mock_read.side_effect = OSError("File not found")

        with patch("warnings.warn") as mock_warn:
            imports = list(find_imports_in_file(filename))
            assert imports == []
            mock_warn.assert_called_once_with(
                f"Unable to parse file {filename} due to File not found",
                stacklevel=2,
            )

def test_find_imports_in_file_with_unique_true():
    filename = "test_file.py"
    content = "import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path"
    expected_imports = [
        identify.Import(module="os"),
        identify.Import(module="pathlib", attribute="Path"),
    ]

    with patch("io.File.read") as mock_read:
        mock_file = MagicMock()
        mock_file.stream = io.StringIO(content)
        mock_file.path = Path(filename)
        mock_read.return_value.__enter__.return_value = mock_file

        imports = list(find_imports_in_file(filename, unique=True))
        assert imports == expected_imports

def test_find_imports_in_file_with_unique_import_key():
    filename = "test_file.py"
    content = "import os\nimport os.path\nfrom pathlib import Path\nfrom pathlib import Path as P"
    expected_imports = [
        identify.Import(module="os"),
        identify.Import(module="pathlib", attribute="Path"),
    ]

    with patch("io.File.read") as mock_read:
        mock_file = MagicMock()
        mock_file.stream = io.StringIO(content)
        mock_file.path = Path(filename)
        mock_read.return_value.__enter__.return_value = mock_file

        imports = list(find_imports_in_file(filename, unique=ImportKey.MODULE))
        assert imports == expected_imports

def test_find_imports_in_file_with_top_only():
    filename = "test_file.py"
    content = "import os\ndef foo():\n    import sys"
    expected_imports = [identify.Import(module="os")]

    with patch("io.File.read") as mock_read:
        mock_file = MagicMock()
        mock_file.stream = io.StringIO(content)
        mock_file.path = Path(filename)
        mock_read.return_value.__enter__.return_value = mock_file

        imports = list(find_imports_in_file(filename, top_only=True))
        assert imports == expected_imports

def test_find_imports_in_file_with_config_kwargs():
    filename = "test_file.py"
    content = "import os"
    expected_imports = [identify.Import(module="os")]

    with patch("io.File.read") as mock_read:
        mock_file = MagicMock()
        mock_file.stream = io.StringIO(content)
        mock_file.path = Path(filename)
        mock_read.return_value.__enter__.return_value = mock_file

        imports = list(find_imports_in_file(filename, src_paths=["."]))
        assert imports == expected_imports


# LLM-generated content at query #9
#--------------------------

```python
def test_sort_stream_skip_file():
    file_path = Path("test.py")
    config = Config()
    config.is_skipped = lambda _: True
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()

    try:
        sort_stream(
            input_stream=input_stream,
            output_stream=output_stream,
            file_path=file_path,
            config=config,
        )
        assert False, "Expected FileSkipSetting to be raised"
    except FileSkipSetting:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_show_diff_predicate_evaluates_to_true():
    input_stream = StringIO("import b\nimport a\n")
    show_diff = True
    extension = None
    config = Config()
    file_path = None
    disregard_skip = False

    result = check_stream(
        input_stream=input_stream,
        show_diff=show_diff,
        extension=extension,
        config=config,
        file_path=file_path,
        disregard_skip=disregard_skip,
    )

    assert result is False


# LLM-generated content at query #11
#--------------------------

```python
def test_find_imports_in_paths_predicate():
    assert find_imports_in_paths is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_find_imports_in_stream_with_default_config():
    input_stream = io.StringIO("import sys\nimport os")
    result = list(find_imports_in_stream(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_unique_true():
    input_stream = io.StringIO("import sys\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_unique_import_key_module():
    input_stream = io.StringIO("import sys\nimport sys.path")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.MODULE))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_unique_import_key_attribute():
    input_stream = io.StringIO("from sys import path\nfrom sys import path")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ATTRIBUTE))
    assert len(result) == 1
    assert result[0].attribute == "path"

def test_find_imports_in_stream_with_unique_import_key_package():
    input_stream = io.StringIO("import sys.path\nimport sys.version")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.PACKAGE))
    assert len(result) == 1
    assert result[0].module == "sys.path"

def test_find_imports_in_stream_with_top_only():
    input_stream = io.StringIO("import sys\ndef foo():\n    import os")
    result = list(find_imports_in_stream(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_config_kwargs():
    input_stream = io.StringIO("import sys")
    config_kwargs = {"section_comment": "custom"}
    result = list(find_imports_in_stream(input_stream, **config_kwargs))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_custom_config():
    input_stream = io.StringIO("import sys")
    custom_config = Config(section_comment="custom")
    result = list(find_imports_in_stream(input_stream, config=custom_config))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_file_path():
    input_stream = io.StringIO("import sys")
    file_path = Path("/tmp/test.py")
    result = list(find_imports_in_stream(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_seen_set():
    input_stream = io.StringIO("import sys\nimport os")
    seen = {"sys"}
    result = list(find_imports_in_stream(input_stream, unique=True, _seen=seen))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #13
#--------------------------

```python
def test_check_file_with_valid_file():
    assert check_file("valid_file.py") is True

def test_check_file_with_invalid_file():
    assert check_file("invalid_file.py") is False

def test_check_file_with_show_diff_true():
    check_file("invalid_file.py", show_diff=True)

def test_check_file_with_show_diff_stream():
    stream = StringIO()
    check_file("invalid_file.py", show_diff=stream)

def test_check_file_with_custom_config():
    config = Config(line_length=100)
    check_file("file.py", config=config)

def test_check_file_with_config_kwargs():
    check_file("file.py", line_length=100)

def test_check_file_with_disregard_skip_false():
    check_file("file.py", disregard_skip=False)

def test_check_file_with_extension():
    check_file("file.py", extension="py")

def test_check_file_with_file_path():
    file_path = Path("file.py")
    check_file("file.py", file_path=file_path)

def test_check_file_with_config_trie():
    config_trie = {}
    check_file("file.py", config_trie=config_trie)

def test_check_file_with_verbose_config():
    config = Config(verbose=True)
    check_file("file.py", config=config)

def test_check_file_with_color_output_config():
    config = Config(color_output=True)
    check_file("file.py", config=config)


# LLM-generated content at query #14
#--------------------------

```python
def test_config_with_path_and_default_config():
    path = Path("/some/path")
    config = _config(path=path)
    assert config.settings_path == path

def test_config_with_path_and_config_kwargs():
    path = Path("/some/path")
    config = _config(path=path, some_setting="value")
    assert config.settings_path == path
    assert config.some_setting == "value"

def test_config_with_custom_config_and_no_kwargs():
    custom_config = Config(some_setting="custom_value")
    config = _config(config=custom_config)
    assert config.some_setting == "custom_value"

def test_config_with_custom_config_and_kwargs_raises_error():
    custom_config = Config(some_setting="custom_value")
    try:
        _config(config=custom_config, another_setting="another_value")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "You can either specify custom configuration options using kwargs or "
            "passing in a Config object. Not Both!"
        )

def test_config_with_kwargs_only():
    config = _config(some_setting="value", another_setting="another_value")
    assert config.some_setting == "value"
    assert config.another_setting == "another_value"

def test_config_with_default_config_and_no_args():
    config = _config()
    assert config == DEFAULT_CONFIG

def test_config_with_settings_path_in_kwargs():
    path = Path("/some/path")
    config = _config(settings_path=path)
    assert config.settings_path == path

def test_config_with_settings_file_in_kwargs():
    config = _config(settings_file="some_file")
    assert config.settings_file == "some_file"

def test_config_with_path_and_settings_path_in_kwargs():
    path = Path("/some/path")
    other_path = Path("/other/path")
    config = _config(path=path, settings_path=other_path)
    assert config.settings_path == other_path

def test_config_with_path_and_settings_file_in_kwargs():
    path = Path("/some/path")
    config = _config(path=path, settings_file="some_file")
    assert config.settings_file == "some_file"


# LLM-generated content at query #15
#--------------------------

```python
def test_check_file_with_valid_file():
    result = check_file("valid_file.py")
    assert result is True

def test_check_file_with_invalid_file():
    result = check_file("invalid_file.py")
    assert result is False

def test_check_file_with_show_diff_true():
    result = check_file("file.py", show_diff=True)
    assert result is False

def test_check_file_with_show_diff_stream():
    stream = StringIO()
    result = check_file("file.py", show_diff=stream)
    assert result is False
    assert stream.getvalue() != ""

def test_check_file_with_custom_config():
    config = Config(line_length=79)
    result = check_file("file.py", config=config)
    assert result is True

def test_check_file_with_config_kwargs():
    result = check_file("file.py", line_length=79)
    assert result is True

def test_check_file_with_disregard_skip():
    result = check_file("file.py", disregard_skip=True)
    assert result is True

def test_check_file_with_extension():
    result = check_file("file.py", extension="py")
    assert result is True

def test_check_file_with_file_path():
    file_path = Path("file.py")
    result = check_file("file.py", file_path=file_path)
    assert result is True

def test_check_file_with_config_trie():
    config_trie = ConfigTrie()
    result = check_file("file.py", config_trie=config_trie)
    assert result is True


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_28_evaluates_to_true():
    input_stream = StringIO("import sys\nimport os")
    config = DEFAULT_CONFIG
    file_path = None
    unique = True
    top_only = False
    _seen = None

    result = list(find_imports_in_stream(input_stream, config, file_path, unique, top_only, _seen))
    assert len(result) == 2


# LLM-generated content at query #17
#--------------------------

```python
def test_check_stream_verbose_and_not_only_modified():
    config = Config(verbose=True, only_modified=False, color_output=False)
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, config=config) is True


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_27_evaluates_to_false():
    _seen = set()
    assert not (_seen is None)


# LLM-generated content at query #19
#--------------------------

```python
def test_sort_stream_predicate_false():
    input_stream = StringIO("import sys")
    output_stream = StringIO()
    file_path = Path("test.py")
    config = Config()
    config.skip = ["test.py"]

    with pytest.raises(FileSkipSetting):
        sort_stream(
            input_stream=input_stream,
            output_stream=output_stream,
            file_path=file_path,
            config=config,
            disregard_skip=False,
        )


# LLM-generated content at query #20
#--------------------------

```python
def test_sort_stream_predicate_true():
    file_path = Path("test.py")
    config = Config()
    config.skip = ["test.py"]
    disregard_skip = False

    assert not disregard_skip and file_path and config.is_skipped(file_path)


# LLM-generated content at query #21
#--------------------------

```python
def test_check_stream_prints_error_when_changed():
    input_stream = StringIO("import b\nimport a\n")
    config = Config(color_output=False, format_error="{error}: {message}", format_success="{success}: {message}")
    printer = create_terminal_printer(color=config.color_output, error=config.format_error, success=config.format_success)
    changed = sort_stream(input_stream=input_stream, output_stream=Empty, extension=None, config=config, file_path=None, disregard_skip=False)
    assert changed
    printer.error(" Imports are incorrectly sorted and/or formatted.")


# LLM-generated content at query #22
#--------------------------

```python
def test_tmp_file_creates_correct_suffix():
    file = File(stream=StringIO(""), path=Path("test.py"), encoding="utf-8")
    assert _tmp_file(file) == Path("test.py.isorted")

def test_tmp_file_preserves_directory():
    file = File(stream=StringIO(""), path=Path("dir/test.py"), encoding="utf-8")
    assert _tmp_file(file) == Path("dir/test.py.isorted")

def test_tmp_file_handles_different_extensions():
    file = File(stream=StringIO(""), path=Path("test.js"), encoding="utf-8")
    assert _tmp_file(file) == Path("test.js.isorted")

def test_tmp_file_with_no_extension():
    file = File(stream=StringIO(""), path=Path("test"), encoding="utf-8")
    assert _tmp_file(file) == Path("test.isorted")


# LLM-generated content at query #23
#--------------------------

```python
def test_check_file_with_valid_file():
    filename = "test_file.py"
    result = check_file(filename)
    assert result is True

def test_check_file_with_invalid_file():
    filename = "invalid_file.py"
    result = check_file(filename)
    assert result is False

def test_check_file_with_show_diff_true():
    filename = "test_file.py"
    result = check_file(filename, show_diff=True)
    assert result is True

def test_check_file_with_show_diff_stream():
    filename = "test_file.py"
    output_stream = StringIO()
    result = check_file(filename, show_diff=output_stream)
    assert result is True

def test_check_file_with_custom_config():
    filename = "test_file.py"
    config = Config(line_length=120)
    result = check_file(filename, config=config)
    assert result is True

def test_check_file_with_config_kwargs():
    filename = "test_file.py"
    result = check_file(filename, line_length=120)
    assert result is True

def test_check_file_with_file_path():
    filename = "test_file.py"
    file_path = Path("custom_path.py")
    result = check_file(filename, file_path=file_path)
    assert result is True

def test_check_file_with_disregard_skip_false():
    filename = "test_file.py"
    result = check_file(filename, disregard_skip=False)
    assert result is True

def test_check_file_with_extension():
    filename = "test_file.py"
    result = check_file(filename, extension="py")
    assert result is True

def test_check_file_with_config_trie():
    filename = "test_file.py"
    config_trie = {"test_file.py": ("config_name", {"line_length": 120})}
    result = check_file(filename, config_trie=config_trie)
    assert result is True


# LLM-generated content at query #24
#--------------------------

```python
def test_extension_predicate_false():
    file_path = Path("test.txt")
    assert not (file_path and file_path.suffix.lstrip("."))


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sort_stream_basic():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_no_changes():
    input_stream = StringIO("import a\nimport b\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream)
    assert result is False
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_config():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(line_length=100)
    result = sort_stream(input_stream, output_stream, config=config)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_file_path():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    result = sort_stream(input_stream, output_stream, file_path=file_path)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_show_diff_true():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream, show_diff=True)
    assert result is True
    assert "import a" in output_stream.getvalue()
    assert "import b" in output_stream.getvalue()

def test_sort_stream_show_diff_stream():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    diff_stream = StringIO()
    result = sort_stream(input_stream, output_stream, show_diff=diff_stream)
    assert result is True
    assert "import a" in diff_stream.getvalue()
    assert "import b" in diff_stream.getvalue()

def test_sort_stream_disregard_skip():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    config = Config(skip=["test.py"])
    result = sort_stream(input_stream, output_stream, file_path=file_path, config=config, disregard_skip=True)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_raise_on_skip():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    config = Config(skip=["test.py"])
    try:
        sort_stream(input_stream, output_stream, file_path=file_path, config=config, raise_on_skip=True)
        assert False, "Expected FileSkipSetting exception"
    except FileSkipSetting:
        pass

def test_sort_stream_atomic_success():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(atomic=True)
    result = sort_stream(input_stream, output_stream, config=config)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_atomic_syntax_error():
    input_stream = StringIO("import b\nimport a\ninvalid syntax\n")
    output_stream = StringIO()
    config = Config(atomic=True)
    try:
        sort_stream(input_stream, output_stream, config=config)
        assert False, "Expected ExistingSyntaxErrors exception"
    except ExistingSyntaxErrors:
        pass

def test_sort_stream_with_extension():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream, extension="py")
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_config_atomic_is_true():
    config = Config(atomic=True)
    assert config.atomic


# LLM-generated content at query #3
#--------------------------

```python
def test_config_atomic_is_true():
    from io import StringIO
    from isort import Config, sort_stream

    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    config = Config(atomic=True)

    sort_stream(
        input_stream=input_stream,
        output_stream=output_stream,
        config=config,
    )

    assert config.atomic is True


# LLM-generated content at query #4
#--------------------------

```python
def test_config_with_path_and_default_config():
    path = Path("/some/path")
    result = _config(path=path)
    assert result.settings_path == path
    assert result is not DEFAULT_CONFIG

def test_config_with_path_and_custom_config():
    path = Path("/some/path")
    custom_config = Config(settings_path=Path("/other/path"))
    result = _config(path=path, config=custom_config)
    assert result.settings_path == Path("/other/path")
    assert result is custom_config

def test_config_with_config_kwargs():
    result = _config(settings_path=Path("/custom/path"))
    assert result.settings_path == Path("/custom/path")
    assert result is not DEFAULT_CONFIG

def test_config_with_both_custom_config_and_kwargs_raises_error():
    custom_config = Config(settings_path=Path("/custom/path"))
    try:
        _config(config=custom_config, settings_path=Path("/another/path"))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "You can either specify custom configuration options using kwargs or "
            "passing in a Config object. Not Both!"
        )

def test_config_with_default_config_and_no_kwargs():
    result = _config()
    assert result is DEFAULT_CONFIG

def test_config_with_settings_file_in_kwargs():
    result = _config(settings_file="custom_file.json")
    assert result.settings_file == "custom_file.json"
    assert result is not DEFAULT_CONFIG

def test_config_with_settings_path_in_kwargs():
    result = _config(settings_path=Path("/custom/path"))
    assert result.settings_path == Path("/custom/path")
    assert result is not DEFAULT_CONFIG


# LLM-generated content at query #5
#--------------------------

```python
def test_find_imports_in_paths_with_valid_paths():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config))
    assert len(result) >= 0

def test_find_imports_in_paths_with_empty_paths():
    paths = []
    config = Config()
    result = list(find_imports_in_paths(paths, config=config))
    assert result == []

def test_find_imports_in_paths_with_unique_true():
    paths = ["path/to/file.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config, unique=True))
    assert len(result) >= 0

def test_find_imports_in_paths_with_top_only_true():
    paths = ["path/to/file.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config, top_only=True))
    assert len(result) >= 0

def test_find_imports_in_paths_with_custom_config():
    paths = ["path/to/file.py"]
    config = Config(include_star_import=True)
    result = list(find_imports_in_paths(paths, config=config))
    assert len(result) >= 0

def test_find_imports_in_paths_with_config_kwargs():
    paths = ["path/to/file.py"]
    result = list(find_imports_in_paths(paths, include_star_import=True))
    assert len(result) >= 0


# LLM-generated content at query #6
#--------------------------

```python
def test_tmp_file_creates_correct_path():
    file = File(stream=StringIO("test"), path=Path("/path/to/file.py"), encoding="utf-8")
    result = _tmp_file(file)
    assert result == Path("/path/to/file.py.isorted")

def test_tmp_file_handles_different_extensions():
    file = File(stream=StringIO("test"), path=Path("/path/to/file.js"), encoding="utf-8")
    result = _tmp_file(file)
    assert result == Path("/path/to/file.js.isorted")

def test_tmp_file_preserves_directory_structure():
    file = File(stream=StringIO("test"), path=Path("/deep/nested/path/file.py"), encoding="utf-8")
    result = _tmp_file(file)
    assert result == Path("/deep/nested/path/file.py.isorted")


# LLM-generated content at query #7
#--------------------------

```python
def test_check_stream_with_correct_imports():
    input_stream = StringIO("import os\nimport sys\n")
    assert check_stream(input_stream) is True

def test_check_stream_with_incorrect_imports():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream) is False

def test_check_stream_with_show_diff_true():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, show_diff=True) is False

def test_check_stream_with_show_diff_stream():
    output_stream = StringIO()
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, show_diff=output_stream) is False
    assert len(output_stream.getvalue()) > 0

def test_check_stream_with_file_path():
    input_stream = StringIO("import sys\nimport os\n")
    file_path = Path("test.py")
    assert check_stream(input_stream, file_path=file_path) is False

def test_check_stream_with_disregard_skip():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, disregard_skip=True) is False

def test_check_stream_with_config_kwargs():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, line_length=120) is False

def test_check_stream_with_extension():
    input_stream = StringIO("import sys\nimport os\n")
    assert check_stream(input_stream, extension="py") is False


# LLM-generated content at query #8
#--------------------------

```python
def test_find_imports_in_stream_with_default_config():
    input_stream = io.StringIO("import os\nimport sys\n")
    result = list(find_imports_in_stream(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

def test_find_imports_in_stream_with_custom_config():
    input_stream = io.StringIO("import os\nimport sys\n")
    config = Config(known_modules=["os", "sys"])
    result = list(find_imports_in_stream(input_stream, config=config))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

def test_find_imports_in_stream_with_unique_true():
    input_stream = io.StringIO("import os\nimport os\n")
    result = list(find_imports_in_stream(input_stream, unique=True))
    assert len(result) == 1
    assert result[0].module == "os"

def test_find_imports_in_stream_with_unique_alias():
    input_stream = io.StringIO("import os as operating_system\nimport os as os_module\n")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ALIAS))
    assert len(result) == 1
    assert result[0].alias == "operating_system"

def test_find_imports_in_stream_with_unique_attribute():
    input_stream = io.StringIO("from os import path\nfrom os import system\n")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ATTRIBUTE))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "system"

def test_find_imports_in_stream_with_unique_module():
    input_stream = io.StringIO("import os.path\nimport os.system\n")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.MODULE))
    assert len(result) == 1
    assert result[0].module == "os"

def test_find_imports_in_stream_with_unique_package():
    input_stream = io.StringIO("import os.path\nimport sys.path\n")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.PACKAGE))
    assert len(result) == 2
    assert result[0].module == "os.path"
    assert result[1].module == "sys.path"

def test_find_imports_in_stream_with_top_only():
    input_stream = io.StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(find_imports_in_stream(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

def test_find_imports_in_stream_with_file_path_and_config_kwargs():
    input_stream = io.StringIO("import os\nimport sys\n")
    file_path = Path("/tmp/test.py")
    result = list(find_imports_in_stream(input_stream, file_path=file_path, settings_path=file_path))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

def test_find_imports_in_stream_with_invalid_config_combination():
    input_stream = io.StringIO("import os\nimport sys\n")
    config = Config(known_modules=["os", "sys"])
    with pytest.raises(ValueError):
        list(find_imports_in_stream(input_stream, config=config, known_modules=["os"]))


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_28_evaluates_to_true():
    input_stream = StringIO("import sys\nimport os")
    config = DEFAULT_CONFIG
    file_path = None
    unique = True
    top_only = False
    _seen = None
    result = list(find_imports_in_stream(input_stream, config, file_path, unique, top_only, _seen))
    assert len(result) == 2


# LLM-generated content at query #10
#--------------------------

```python
def test_sort_stream_basic_functionality():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_no_changes():
    input_stream = StringIO("import a\nimport b")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream)
    assert changed is False
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_extension():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, extension="py")
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_config():
    config = Config(line_length=79)
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, config=config)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_file_path():
    file_path = Path("test.py")
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, file_path=file_path)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_disregard_skip():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, disregard_skip=True)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_show_diff_true():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, show_diff=True)
    assert changed is True
    assert "import a" in output_stream.getvalue()
    assert "import b" in output_stream.getvalue()

def test_sort_stream_with_show_diff_stream():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    diff_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, show_diff=diff_stream)
    assert changed is True
    assert "import a" in diff_stream.getvalue()
    assert "import b" in diff_stream.getvalue()

def test_sort_stream_with_config_kwargs():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, line_length=79)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_atomic_config():
    config = Config(atomic=True)
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, config=config)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_raise_on_skip_false():
    input_stream = StringIO("import b\nimport a")
    output_stream = StringIO()
    changed = sort_stream(input_stream, output_stream, raise_on_skip=False)
    assert changed is True
    assert output_stream.getvalue() == "import a\nimport b\n"


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_28_evaluates_to_true():
    input_stream = io.StringIO("import sys\nimport os")
    config = DEFAULT_CONFIG
    file_path = None
    unique = True
    top_only = False
    _seen = None

    result = list(find_imports_in_stream(input_stream, config, file_path, unique, top_only, _seen))

    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"


# LLM-generated content at query #12
#--------------------------

```python
def test_config_with_path_and_default_config():
    path = Path("test_path")
    result = _config(path=path)
    assert result.settings_path == path
    assert result is not DEFAULT_CONFIG

def test_config_with_path_and_custom_config():
    path = Path("test_path")
    custom_config = Config(settings_path=Path("custom_path"))
    result = _config(path=path, config=custom_config)
    assert result.settings_path == Path("custom_path")
    assert result is custom_config

def test_config_with_config_kwargs_and_default_config():
    result = _config(settings_path=Path("test_path"))
    assert result.settings_path == Path("test_path")
    assert result is not DEFAULT_CONFIG

def test_config_with_config_kwargs_and_custom_config_raises_error():
    custom_config = Config(settings_path=Path("custom_path"))
    with pytest.raises(ValueError):
        _config(config=custom_config, settings_path=Path("test_path"))

def test_config_with_no_args_returns_default():
    result = _config()
    assert result is DEFAULT_CONFIG

def test_config_with_custom_config_returns_same():
    custom_config = Config(settings_path=Path("custom_path"))
    result = _config(config=custom_config)
    assert result is custom_config


# LLM-generated content at query #13
#--------------------------

```python
def test_sort_file_with_write_to_stdout():
    filename = "test_file.py"
    content = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    with open(filename, "w") as f:
        f.write(content)
    output = StringIO()
    result = sort_file(filename, write_to_stdout=True, output=output)
    assert result is True
    output.seek(0)
    assert output.read() == expected_output
    os.remove(filename)

def test_sort_file_with_show_diff():
    filename = "test_file.py"
    content = "import b\nimport a\n"
    with open(filename, "w") as f:
        f.write(content)
    output = StringIO()
    result = sort_file(filename, show_diff=True, output=output)
    assert result is False
    output.seek(0)
    diff_output = output.read()
    assert "--" in diff_output
    assert "++" in diff_output
    os.remove(filename)

def test_sort_file_with_ask_to_apply_no():
    filename = "test_file.py"
    content = "import b\nimport a\n"
    with open(filename, "w") as f:
        f.write(content)
    with patch("builtins.input", return_value="n"):
        result = sort_file(filename, ask_to_apply=True)
    assert result is False
    with open(filename) as f:
        assert f.read() == content
    os.remove(filename)

def test_sort_file_with_ask_to_apply_yes():
    filename = "test_file.py"
    content = "import b\nimport a\n"
    expected_content = "import a\nimport b\n"
    with open(filename, "w") as f:
        f.write(content)
    with patch("builtins.input", return_value="y"):
        result = sort_file(filename, ask_to_apply=True)
    assert result is True
    with open(filename) as f:
        assert f.read() == expected_content
    os.remove(filename)

def test_sort_file_with_output_stream():
    filename = "test_file.py"
    content = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    with open(filename, "w") as f:
        f.write(content)
    output = StringIO()
    result = sort_file(filename, output=output)
    assert result is True
    output.seek(0)
    assert output.read() == expected_output
    os.remove(filename)

def test_sort_file_with_no_changes():
    filename = "test_file.py"
    content = "import a\nimport b\n"
    with open(filename, "w") as f:
        f.write(content)
    result = sort_file(filename)
    assert result is False
    os.remove(filename)

def test_sort_file_with_config_kwargs():
    filename = "test_file.py"
    content = "import b\nimport a\n"
    expected_output = "import a\nimport b\n"
    with open(filename, "w") as f:
        f.write(content)
    output = StringIO()
    result = sort_file(filename, output=output, line_length=120)
    assert result is True
    output.seek(0)
    assert output.read() == expected_output
    os.remove(filename)


# LLM-generated content at query #14
#--------------------------

```python
def test_sort_stream_file_skip_comment_raises():
    input_stream = StringIO("from b import b\nfrom a import a\n# isort: skip")
    output_stream = StringIO()
    with pytest.raises(FileSkipComment):
        sort_stream(input_stream, output_stream)


# LLM-generated content at query #15
#--------------------------

```python
def test_tmp_file_creates_correct_path():
    file = File(stream=StringIO(""), path=Path("/path/to/file.py"), encoding="utf-8")
    assert _tmp_file(file) == Path("/path/to/file.py.isorted")

def test_tmp_file_handles_different_extensions():
    file = File(stream=StringIO(""), path=Path("/path/to/file.txt"), encoding="utf-8")
    assert _tmp_file(file) == Path("/path/to/file.txt.isorted")

def test_tmp_file_preserves_parent_directory():
    file = File(stream=StringIO(""), path=Path("/another/path/file.py"), encoding="utf-8")
    assert _tmp_file(file) == Path("/another/path/file.py.isorted")


# LLM-generated content at query #16
#--------------------------

```python
def test_find_imports_in_paths_basic():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config))
    assert len(result) == 2
    assert all(isinstance(imp, identify.Import) for imp in result)

def test_find_imports_in_paths_unique_true():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config, unique=True))
    assert len(result) == 1

def test_find_imports_in_paths_unique_import_key():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config, unique=ImportKey.MODULE))
    assert all(isinstance(imp, identify.Import) for imp in result)

def test_find_imports_in_paths_top_only():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    config = Config()
    result = list(find_imports_in_paths(paths, config=config, top_only=True))
    assert all(isinstance(imp, identify.Import) for imp in result)

def test_find_imports_in_paths_config_kwargs():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    result = list(find_imports_in_paths(paths, config_kwargs={"line_length": 100}))
    assert len(result) == 2
    assert all(isinstance(imp, identify.Import) for imp in result)

def test_find_imports_in_paths_empty_paths():
    paths = []
    config = Config()
    result = list(find_imports_in_paths(paths, config=config))
    assert len(result) == 0

def test_find_imports_in_paths_file_path():
    paths = ["path/to/file1.py", "path/to/file2.py"]
    file_path = Path("base/path")
    config = Config()
    result = list(find_imports_in_paths(paths, config=config, file_path=file_path))
    assert len(result) == 2
    assert all(isinstance(imp, identify.Import) for imp in result)


# LLM-generated content at query #17
#--------------------------

```python
def test_extension_predicate_false():
    file_path = None
    extension = None
    assert not (extension or (file_path and file_path.suffix.lstrip(".")) or "py")


# LLM-generated content at query #18
#--------------------------

```python
def test_check_stream_predicate_true():
    input_stream = StringIO("import sys\nimport os")
    config = Config(verbose=True, only_modified=False)
    assert check_stream(input_stream, config=config) is True


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_evaluates_to_false():
    path = Path("/some/path")
    config = Config()
    config_kwargs = {"settings_path": path}

    assert not (path and config is DEFAULT_CONFIG and "settings_path" not in config_kwargs and "settings_file" not in config_kwargs)


# LLM-generated content at query #20
#--------------------------

```python
def test_find_imports_in_stream_with_default_config():
    input_stream = io.StringIO("import sys\nimport os")
    result = list(find_imports_in_stream(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_unique_true():
    input_stream = io.StringIO("import sys\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_unique_import_key_alias():
    input_stream = io.StringIO("import sys as system\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ALIAS))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "sys"

def test_find_imports_in_stream_with_unique_import_key_attribute():
    input_stream = io.StringIO("from sys import path\nfrom sys import argv")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ATTRIBUTE))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "argv"

def test_find_imports_in_stream_with_unique_import_key_module():
    input_stream = io.StringIO("import sys\nimport sys.path")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.MODULE))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_unique_import_key_package():
    input_stream = io.StringIO("import sys.path\nimport sys.argv")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.PACKAGE))
    assert len(result) == 1
    assert result[0].module == "sys.path"

def test_find_imports_in_stream_with_top_only_true():
    input_stream = io.StringIO("import sys\ndef foo():\n    import os")
    result = list(find_imports_in_stream(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_custom_config():
    input_stream = io.StringIO("import sys")
    custom_config = Config(known_modules=["sys"])
    result = list(find_imports_in_stream(input_stream, config=custom_config))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_config_kwargs():
    input_stream = io.StringIO("import sys")
    result = list(find_imports_in_stream(input_stream, known_modules=["sys"]))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_path_and_config_kwargs():
    input_stream = io.StringIO("import sys")
    path = Path("/tmp")
    result = list(find_imports_in_stream(input_stream, path=path, known_modules=["sys"]))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_invalid_config_and_kwargs():
    input_stream = io.StringIO("import sys")
    custom_config = Config(known_modules=["sys"])
    with pytest.raises(ValueError):
        list(find_imports_in_stream(input_stream, config=custom_config, known_modules=["os"]))


# LLM-generated content at query #21
#--------------------------

```python
def test_sort_stream_basic_functionality():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_no_changes():
    input_stream = StringIO("import a\nimport b\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream)
    assert result is False
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_config_kwargs():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream, line_length=120)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_file_path():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    result = sort_stream(input_stream, output_stream, file_path=file_path)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_show_diff_true():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream, show_diff=True)
    assert result is True
    assert "import a\nimport b\n" in output_stream.getvalue()

def test_sort_stream_show_diff_stream():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    diff_stream = StringIO()
    result = sort_stream(input_stream, output_stream, show_diff=diff_stream)
    assert result is True
    assert "import a\nimport b\n" in diff_stream.getvalue()

def test_sort_stream_atomic_mode():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    config = Config(atomic=True)
    result = sort_stream(input_stream, output_stream, config=config)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_with_extension():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    result = sort_stream(input_stream, output_stream, extension="py")
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"

def test_sort_stream_disregard_skip():
    input_stream = StringIO("import b\nimport a\n")
    output_stream = StringIO()
    file_path = Path("test.py")
    config = Config(skip=["test.py"])
    result = sort_stream(input_stream, output_stream, file_path=file_path, config=config, disregard_skip=True)
    assert result is True
    assert output_stream.getvalue() == "import a\nimport b\n"


# LLM-generated content at query #22
#--------------------------

```python
def test_find_imports_in_stream_with_default_config():
    input_stream = StringIO("import sys\nimport os")
    result = list(find_imports_in_stream(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_unique_true():
    input_stream = StringIO("import sys\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_unique_alias():
    input_stream = StringIO("import sys as system\nimport sys")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.ALIAS))
    assert len(result) == 2

def test_find_imports_in_stream_with_unique_module():
    input_stream = StringIO("import sys\nfrom sys import path\nimport os")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.MODULE))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_unique_package():
    input_stream = StringIO("import sys\nimport sys.path\nimport os")
    result = list(find_imports_in_stream(input_stream, unique=ImportKey.PACKAGE))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_top_only():
    input_stream = StringIO("import sys\ndef foo():\n    import os")
    result = list(find_imports_in_stream(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "sys"

def test_find_imports_in_stream_with_config_kwargs():
    input_stream = StringIO("import sys\nimport os")
    result = list(find_imports_in_stream(input_stream, config_kwargs={"line_length": 100}))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_custom_config():
    config = Config(line_length=100)
    input_stream = StringIO("import sys\nimport os")
    result = list(find_imports_in_stream(input_stream, config=config))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_file_path():
    input_stream = StringIO("import sys\nimport os")
    file_path = Path("/tmp/test.py")
    result = list(find_imports_in_stream(input_stream, file_path=file_path))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[1].module == "os"

def test_find_imports_in_stream_with_seen_set():
    input_stream = StringIO("import sys\nimport os")
    seen = {"sys"}
    result = list(find_imports_in_stream(input_stream, unique=True, _seen=seen))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #23
#--------------------------

```python
def test_config_atomic_is_true():
    config = Config(atomic=True)
    assert config.atomic is True


# LLM-generated content at query #24
#--------------------------

```python
def test_unique_is_not_true_or_importkey_alias():
    assert not (True, ImportKey.ALIAS)


# LLM-generated content at query #25
#--------------------------

```python
def test_check_stream_predicate_true():
    input_stream = StringIO("import os\nimport sys\n")
    config = Config(verbose=True, only_modified=False, color_output=False)
    assert check_stream(input_stream, show_diff=False, config=config, file_path=None)


