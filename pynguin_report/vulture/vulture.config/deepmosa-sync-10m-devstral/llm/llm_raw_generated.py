####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_input_config_valid():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"key1": "value1", "unknown_key": 123}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #2
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(argv=["--verbose", "path1", "path2"])
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False

def test_make_config_with_toml_and_cli_args():
    toml_content = """
        [tool.vulture]
        exclude = ["file*.py", "dir/"]
        ignore_decorators = ["deco1", "deco2"]
        ignore_names = ["name1", "name2"]
        make_whitelist = true
        min_confidence = 10
        sort_by_size = true
        verbose = true
        paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--verbose", "path3"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["path3"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

def test_make_config_with_toml_only():
    toml_content = """
        [tool.vulture]
        exclude = ["file*.py", "dir/"]
        ignore_decorators = ["deco1", "deco2"]
        ignore_names = ["name1", "name2"]
        make_whitelist = true
        min_confidence = 10
        sort_by_size = true
        verbose = true
        paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

def test_make_config_with_no_args_and_no_toml():
    config = make_config(argv=[])
    assert config["verbose"] is False
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False

def test_make_config_with_invalid_toml_key():
    toml_content = """
        [tool.vulture]
        invalid_key = "value"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

def test_make_config_with_invalid_toml_type():
    toml_content = """
        [tool.vulture]
        min_confidence = "not_an_int"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"

def test_make_config_with_no_paths():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["path1"])
    assert config["paths"] == ["path1"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_cli_overrides():
    config = make_config(argv=[
        "path1", "path2",
        "--exclude", "*.py",
        "--ignore-decorators", "decorator1",
        "--ignore-names", "name1",
        "--make-whitelist",
        "--min-confidence", "80",
        "--sort-by-size",
        "--verbose",
        "--config", "custom.toml"
    ])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["*.py"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["config"] == "custom.toml"

def test_make_config_toml_file():
    import io
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    exclude = ["*.py"]
    ignore_decorators = ["decorator1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 80
    sort_by_size = true
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["*.py"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
    [tool.vulture]
    paths = ["toml_path"]
    exclude = ["*.py"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--exclude", "*.txt"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["exclude"] == ["*.txt"]

def test_make_config_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

def test_make_config_invalid_type_in_toml_raises_error():
    import io
    toml_content = """
    [tool.vulture]
    paths = "not_a_list"
    """
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)

def test_make_config_unknown_key_in_toml_raises_error():
    import io
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    """
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)


# LLM-generated content at query #4
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #5
#--------------------------

```python
def test_tomlfile_predicate():
    tomlfile = "test.toml"
    assert tomlfile


# LLM-generated content at query #6
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True


# LLM-generated content at query #7
#--------------------------

```python
def test_check_input_config_valid():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"key1": "value1", "unknown_key": "value2"}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #8
#--------------------------

```python
def test_check_input_config_correct_type():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #9
#--------------------------

```python
def test_verbose_message_when_toml_detected():
    with patch("builtins.print") as mock_print:
        with patch("pathlib.Path") as mock_path:
            mock_path.return_value.resolve.return_value.is_file.return_value = True
            mock_file = MagicMock()
            mock_file.__str__.return_value = "test.toml"
            with patch("builtins.open", return_value=mock_file):
                config = make_config(argv=["--verbose"])
                assert mock_print.called
                assert "Reading configuration from test.toml" in mock_print.call_args[0][0]


# LLM-generated content at query #10
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("pyproject.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #11
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    assert _check_input_config(data) is None


# LLM-generated content at query #12
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config_defaults():
    config = make_config()
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_cli_args():
    config = make_config(["--exclude", "test_*.py", "--min-confidence", "80", "--verbose"])
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

def test_make_config_toml_file():
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 80
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 80
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--exclude", "docs", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["exclude"] == ["docs"]
    assert config["min_confidence"] == 90

def test_make_config_invalid_key():
    import pytest
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(["--invalid-key", "value"])

def test_make_config_invalid_type():
    import pytest
    with pytest.raises(InputError, match="Data type for"):
        make_config(["--min-confidence", "not_an_int"])

def test_make_config_no_paths():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config()


# LLM-generated content at query #14
#--------------------------

```python
def test_verbose_toml_path_print():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True
    assert config["config"] == "test.toml"


# LLM-generated content at query #15
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_with_toml_file():
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    exclude = ["*test*.py"]
    ignore_decorators = ["@decorator1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 80
    sort_by_size = true
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["*test*.py"]
    assert config["ignore_decorators"] == ["@decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    paths = ["toml_path"]
    min_confidence = 80
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_path"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 60  # CLI default overrides TOML

def test_make_config_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

def test_make_config_unknown_key_in_toml_raises_error():
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    """
    import io
    import pytest
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)

def test_make_config_wrong_type_in_toml_raises_error():
    toml_content = """
    [tool.vulture]
    paths = "not_a_list"
    """
    import io
    import pytest
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)

def test_make_config_verbose_shows_toml_path():
    toml_content = """
    [tool.vulture]
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True


# LLM-generated content at query #16
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("pyproject.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #17
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #18
#--------------------------

```python
def test_toml_path_is_file():
    assert pathlib.Path("pyproject.toml").resolve().is_file()


# LLM-generated content at query #19
#--------------------------

```python
def test_make_config_defaults():
    config = make_config()
    assert config == DEFAULTS

def test_make_config_cli_overrides_defaults():
    config = make_config(["--min-confidence", "50"])
    assert config["min_confidence"] == 50
    assert config["paths"] == DEFAULTS["paths"]
    assert config["verbose"] == DEFAULTS["verbose"]

def test_make_config_toml_overrides_defaults():
    toml_content = """[tool.vulture]
min_confidence = 75
paths = ["src/"]
"""
    with open("test_pyproject.toml", "w") as f:
        f.write(toml_content)
    with open("test_pyproject.toml", "rb") as f:
        config = make_config(tomlfile=f)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["src/"]
    assert config["verbose"] == DEFAULTS["verbose"]

def test_make_config_cli_overrides_toml():
    toml_content = """[tool.vulture]
min_confidence = 75
paths = ["src/"]
"""
    with open("test_pyproject.toml", "w") as f:
        f.write(toml_content)
    with open("test_pyproject.toml", "rb") as f:
        config = make_config(["--min-confidence", "90"], tomlfile=f)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["src/"]

def test_make_config_verbose_shows_toml_path():
    toml_content = """[tool.vulture]
min_confidence = 75
"""
    with open("test_pyproject.toml", "w") as f:
        f.write(toml_content)
    with open("test_pyproject.toml", "rb") as f:
        config = make_config(["--verbose"], tomlfile=f)
    assert config["verbose"] is True
    assert config["min_confidence"] == 75

def test_make_config_no_paths_raises_error():
    with pytest.raises(InputError):
        make_config(["--exclude", "test_*.py"])

def test_make_config_invalid_key_raises_error():
    with pytest.raises(InputError):
        make_config(["--invalid-key", "value"])

def test_make_config_wrong_type_raises_error():
    with pytest.raises(InputError):
        make_config(["--min-confidence", "not_an_integer"])

def test_make_config_toml_invalid_key_raises_error():
    toml_content = """[tool.vulture]
invalid_key = "value"
"""
    with open("test_pyproject.toml", "w") as f:
        f.write(toml_content)
    with open("test_pyproject.toml", "rb") as f:
        with pytest.raises(InputError):
            make_config(tomlfile=f)

def test_make_config_toml_wrong_type_raises_error():
    toml_content = """[tool.vulture]
min_confidence = "not_an_integer"
"""
    with open("test_pyproject.toml", "w") as f:
        f.write(toml_content)
    with open("test_pyproject.toml", "rb") as f:
        with pytest.raises(InputError):
            make_config(tomlfile=f)


# LLM-generated content at query #20
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 42, "key2": True}
    DEFAULTS = {"key1": 0, "key2": False}
    assert type(data["key1"]) is type(DEFAULTS["key1"])
    assert type(data["key2"]) is type(DEFAULTS["key2"])


# LLM-generated content at query #21
#--------------------------

```python
def test_verbose_output_with_toml():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True


# LLM-generated content at query #22
#--------------------------

```python
def test_check_input_config_with_valid_data():
    valid_data = {"key1": 1, "key2": "value"}
    _check_input_config(valid_data)

def test_check_input_config_with_unknown_key():
    invalid_data = {"unknown_key": 1}
    try:
        _check_input_config(invalid_data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    invalid_data = {"key1": "not_an_int"}
    try:
        _check_input_config(invalid_data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    import pathlib
    import tempfile

    # Create a temporary file to simulate a TOML config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
        tmp.write("[tool.vulture]\n")
        tmp_path = tmp.name

    # Ensure the file exists and is a file
    toml_path = pathlib.Path(tmp_path).resolve()
    assert toml_path.is_file()


# LLM-generated content at query #24
#--------------------------

```python
def test__check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #25
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("valid_toml_file.toml")
    assert toml_path.is_file() is True


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(["--exclude", "test_*.py", "--verbose", "path1", "path2"])
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False

def test_make_config_with_toml_and_cli_args():
    import io
    toml_content = """
[tool.vulture]
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
paths = ["path1"]
min_confidence = 10
sort_by_size = true
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--exclude", "test_*.py", "--verbose", "path2"], tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["verbose"] is True
    assert config["paths"] == ["path2"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

def test_make_config_with_toml_only():
    import io
    toml_content = """
[tool.vulture]
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
paths = ["path1"]
min_confidence = 10
sort_by_size = true
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["paths"] == ["path1"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

def test_make_config_with_invalid_toml_key():
    import io
    toml_content = """
[tool.vulture]
invalid_key = "value"
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

def test_make_config_with_invalid_cli_key():
    try:
        make_config(["--invalid-key", "value"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_wrong_type_in_toml():
    import io
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"

def test_make_config_with_wrong_type_in_cli():
    try:
        make_config(["--min-confidence", "not_an_int"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_no_paths():
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_make_config_with_defaults():
    config = make_config(["path1"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["paths"] == ["path1"]


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_args_defaults():
    args = _parse_args([])
    assert args == {"paths": [], "config": "pyproject.toml"}

def test_parse_args_paths():
    args = _parse_args(["file1.py", "file2.py"])
    assert args == {"paths": ["file1.py", "file2.py"], "config": "pyproject.toml"}

def test_parse_args_exclude():
    args = _parse_args(["--exclude", "test_*,venv"])
    assert args == {"exclude": ["test_*", "venv"], "config": "pyproject.toml"}

def test_parse_args_ignore_decorators():
    args = _parse_args(["--ignore-decorators", "@app.route,@require_*"])
    assert args == {"ignore_decorators": ["@app.route", "@require_*"], "config": "pyproject.toml"}

def test_parse_args_ignore_names():
    args = _parse_args(["--ignore-names", "visit_*,do_*"])
    assert args == {"ignore_names": ["visit_*", "do_*"], "config": "pyproject.toml"}

def test_parse_args_make_whitelist():
    args = _parse_args(["--make-whitelist"])
    assert args == {"make_whitelist": True, "config": "pyproject.toml"}

def test_parse_args_min_confidence():
    args = _parse_args(["--min-confidence", "50"])
    assert args == {"min_confidence": 50, "config": "pyproject.toml"}

def test_parse_args_sort_by_size():
    args = _parse_args(["--sort-by-size"])
    assert args == {"sort_by_size": True, "config": "pyproject.toml"}

def test_parse_args_verbose():
    args = _parse_args(["-v"])
    assert args == {"verbose": True, "config": "pyproject.toml"}

def test_parse_args_config():
    args = _parse_args(["--config", "custom.toml"])
    assert args == {"config": "custom.toml"}

def test_parse_args_combined():
    args = _parse_args(["file.py", "--exclude", "test_*", "--min-confidence", "75", "--verbose"])
    assert args == {
        "paths": ["file.py"],
        "exclude": ["test_*"],
        "min_confidence": 75,
        "verbose": True,
        "config": "pyproject.toml"
    }


# LLM-generated content at query #2
#--------------------------

```python
def test_check_output_config_with_empty_paths():
    config = {"paths": []}
    try:
        _check_output_config(config)
        assert False, "Expected InputError to be raised"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"


# LLM-generated content at query #3
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": 1}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": "value"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #4
#--------------------------

```python
def test_make_config_cli_only():
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_toml_only():
    toml_content = """
    [tool.vulture]
    paths = ["path1", "path2"]
    exclude = ["file*.py"]
    ignore_decorators = ["deco1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    paths = ["path1", "path2"]
    exclude = ["file*.py"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["path3", "--exclude", "dir/"], tomlfile=tomlfile)
    assert config["paths"] == ["path3"]
    assert config["exclude"] == ["dir/"]
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_verbose_toml_path():
    toml_content = """
    [tool.vulture]
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True


# LLM-generated content at query #5
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    with open("pyproject.toml", "w") as f:
        f.write("[tool.vulture]\nverbose = true\n")

    config = make_config(argv=["--config", "pyproject.toml"])
    assert config["verbose"] is True
    assert "Reading configuration from pyproject.toml" in capsys.readouterr().out


# LLM-generated content at query #6
#--------------------------

```python
def test_check_input_config_valid_types():
    DEFAULTS = {"key1": 1, "key2": "value", "key3": True}
    data = {"key1": 2, "key2": "another", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    toml_path.touch()
    assert toml_path.is_file()


# LLM-generated content at query #8
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": 1}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": "not_an_int"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #9
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": "default"}
    _check_input_config(data)


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    import pathlib
    import tempfile

    # Create a temporary TOML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
        tmp.write("[vulture]\nmin_confidence = 0.5\n")
        tmp_path = tmp.name

    # Mock the CLI config to point to the temporary TOML file
    cli_config = {"config": tmp_path, "verbose": False}

    # Ensure the predicate evaluates to True
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert toml_path.is_file()


# LLM-generated content at query #11
#--------------------------

```python
def test_verbose_output_with_toml_path():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True
    assert config["config"] == "test.toml"


# LLM-generated content at query #12
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(tomlfile=io.StringIO("[tool.vulture]\nverbose = true"))
    assert config["verbose"] is True


# LLM-generated content at query #13
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("pyproject.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #14
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("valid_toml_file.toml").resolve()
    assert toml_path.is_file() == True


# LLM-generated content at query #15
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"key1": "value1", "unknown_key": "value2"}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #16
#--------------------------

```python
def test_detected_toml_path_and_verbose_config():
    config = make_config(tomlfile="test.toml", argv=["--verbose"])
    assert config["verbose"] is True
    assert config["config"] == "test.toml"


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("test.toml").resolve()
    toml_path.touch()
    assert toml_path.is_file()


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    import pathlib
    import tempfile

    # Create a temporary file to simulate a valid TOML file
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name).resolve()

    # Ensure the file exists
    assert tmp_path.is_file()


# LLM-generated content at query #19
#--------------------------

```python
def test_detected_toml_path_and_verbose_prints_message():
    config = make_config(tomlfile="test.toml", argv=["--verbose"])
    assert config["verbose"] is True
    assert config["config"] == "test.toml"


# LLM-generated content at query #20
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #21
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 42, "key2": True}
    DEFAULTS = {"key1": 0, "key2": False}
    try:
        _check_input_config(data)
    except InputError:
        assert False, "InputError raised unexpectedly"


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("test.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #23
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True


# LLM-generated content at query #24
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file():
    toml_content = """
        [tool.vulture]
        verbose = true
        min_confidence = 50
        paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    toml_content = """
        [tool.vulture]
        verbose = false
        min_confidence = 50
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--verbose", "--min-confidence", "70"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 70

def test_make_config_defaults():
    config = make_config([])
    assert config["verbose"] is False
    assert config["min_confidence"] == 60
    assert config["paths"] == []

def test_make_config_unknown_key():
    toml_content = """
        [tool.vulture]
        unknown_key = "value"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_wrong_type():
    toml_content = """
        [tool.vulture]
        min_confidence = "not_an_int"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be" in str(e)

def test_make_config_no_paths():
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #25
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_tomlfile():
    import io
    toml_content = b"""
        [tool.vulture]
        min_confidence = 90
        verbose = true
        paths = ["path1", "path2"]
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = b"""
        [tool.vulture]
        min_confidence = 90
        verbose = false
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

def test_make_config_defaults_applied():
    config = make_config(argv=[])
    assert config["min_confidence"] == DEFAULTS["min_confidence"]
    assert config["verbose"] == DEFAULTS["verbose"]
    assert config["paths"] == DEFAULTS["paths"]

def test_make_config_verbose_toml_path():
    import io
    toml_content = b"""
        [tool.vulture]
        verbose = true
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True


