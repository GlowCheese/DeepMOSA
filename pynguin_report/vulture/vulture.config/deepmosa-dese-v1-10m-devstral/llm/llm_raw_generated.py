####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config_cli_only():
    config = make_config(["--min-confidence", "80", "--verbose", "src/"])
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_toml_only():
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src/"]
    min_confidence = 80
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src/"]
    min_confidence = 80
    verbose = false
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "90", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_defaults():
    config = make_config([])
    assert config["min_confidence"] == DEFAULTS["min_confidence"]
    assert config["verbose"] == DEFAULTS["verbose"]
    assert config["paths"] == DEFAULTS["paths"]

def test_make_config_verbose_toml_output():
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src/"]
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_invalid_cli_key():
    try:
        make_config(["--invalid-key", "value"])
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_invalid_toml_key():
    import io
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_no_paths():
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_check_input_config_valid():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 456}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
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


# LLM-generated content at query #3
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    tomlfile = pathlib.Path("pyproject.toml").open("rb")
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["config"] == str(tomlfile)


# LLM-generated content at query #4
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True


# LLM-generated content at query #5
#--------------------------

```python
def test_check_input_config_correct_type():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    assert type(data["key1"]) is type(DEFAULTS["key1"])
    assert type(data["key2"]) is type(DEFAULTS["key2"])


# LLM-generated content at query #6
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        toml_path = pathlib.Path(tmp.name).resolve()
        assert toml_path.is_file()


# LLM-generated content at query #7
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {key: type(DEFAULTS[key])() for key in DEFAULTS}
    _check_input_config(data)


# LLM-generated content at query #8
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("valid_file.toml").resolve()
    assert toml_path.is_file() == True


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--exclude", "test_*.py", "--min-confidence", "50", "path1", "path2"])
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False
    assert config["sort_by_size"] is False

def test_make_config_with_toml_file():
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 50
    paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False
    assert config["sort_by_size"] is False

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 50
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--exclude", "other_*.py", "--min-confidence", "70"], tomlfile=tomlfile)
    assert config["exclude"] == ["other_*.py"]
    assert config["min_confidence"] == 70

def test_make_config_defaults():
    config = make_config()
    assert config["exclude"] == []
    assert config["min_confidence"] == 60
    assert config["paths"] == []
    assert config["verbose"] is False
    assert config["sort_by_size"] is False

def test_make_config_verbose_with_toml():
    toml_content = """
    [tool.vulture]
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True

def test_make_config_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config(["--exclude", "test_*.py"])


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #11
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    with open("pyproject.toml", "w") as f:
        f.write("[tool.vulture]\nverbose = true\n")

    config = make_config(["--config", "pyproject.toml"])
    assert config["verbose"] is True


# LLM-generated content at query #12
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
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


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("test.toml").resolve()
    toml_path.touch()
    assert toml_path.is_file()


# LLM-generated content at query #14
#--------------------------

```python
def test_check_input_config_valid_data():
    valid_data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(valid_data)

def test_check_input_config_unknown_key():
    invalid_data = {"unknown_key": 1}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(invalid_data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    invalid_data = {"key1": "not_an_int"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(invalid_data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"

def test_check_input_config_bool_vs_int():
    invalid_data = {"key1": True}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(invalid_data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #15
#--------------------------

```python
def test_check_input_config_valid_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #16
#--------------------------

```python
def test_verbose_toml_path_printed():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True
    assert config["config"] == "test.toml"


# LLM-generated content at query #17
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
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    toml_content = """
        [tool.vulture]
        min_confidence = 10
        verbose = false
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "50", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

def test_make_config_with_no_paths():
    import pytest
    with pytest.raises(InputError):
        make_config([])

def test_make_config_with_invalid_toml_key():
    toml_content = """
        [tool.vulture]
        invalid_key = "value"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)

def test_make_config_with_invalid_toml_type():
    toml_content = """
        [tool.vulture]
        min_confidence = "not_an_int"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)


# LLM-generated content at query #18
#--------------------------

```python
def test_verbose_with_toml_path():
    assert make_config(tomlfile="test.toml", argv=["--verbose"])["verbose"] is True


# LLM-generated content at query #19
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(argv=["--verbose", "path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_and_cli_args():
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py"]
    ignore_decorators = ["deco1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "20", "path2"], tomlfile=tomlfile)
    assert config["paths"] == ["path2"]
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 20
    assert config["sort_by_size"] is True
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_only():
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py"]
    ignore_decorators = ["deco1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["path1"]
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["config"] == "pyproject.toml"

def test_make_config_with_defaults():
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["verbose"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"

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


# LLM-generated content at query #20
#--------------------------

```python
def test_check_input_config_type_matching():
    DEFAULTS = {"key1": 1, "key2": "value"}
    data = {"key1": 2, "key2": "another"}
    _check_input_config(data)


# LLM-generated content at query #21
#--------------------------

```python
def test_verbose_toml_path_print():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True
    assert config["config"] == "test.toml"


# LLM-generated content at query #22
#--------------------------

```python
def test_check_input_config_valid_type():
    data = {"key": 1}
    DEFAULTS = {"key": 1}
    assert not _check_input_config(data)


# LLM-generated content at query #23
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("pyproject.toml").resolve()
    toml_path.touch()
    assert toml_path.is_file()


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("test.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #25
#--------------------------

```python
def test_check_input_config_valid_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file():
    from io import StringIO
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path"]
    """
    config = make_config(tomlfile=StringIO(toml_content))
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path"]

def test_make_config_cli_overrides_toml():
    from io import StringIO
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    """
    config = make_config(argv=["--min-confidence", "50"], tomlfile=StringIO(toml_content))
    assert config["min_confidence"] == 50

def test_make_config_defaults():
    config = make_config(argv=[])
    assert config["min_confidence"] == DEFAULTS["min_confidence"]
    assert config["paths"] == []

def test_make_config_verbose_toml_output():
    from io import StringIO
    import sys
    toml_content = """
    [tool.vulture]
    verbose = true
    """
    config = make_config(tomlfile=StringIO(toml_content))
    assert config["verbose"] is True

def test_make_config_missing_paths_raises_error():
    try:
        make_config(argv=[])
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_check_output_config_with_empty_paths():
    config = {"paths": []}
    try:
        _check_output_config(config)
        assert False, "Expected InputError to be raised"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"


# LLM-generated content at query #2
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 456}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
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


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(["--verbose", "path1", "path2"])
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False

def test_make_config_with_toml_and_cli_args():
    toml_data = """
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
    tomlfile = io.StringIO(toml_data)
    config = make_config(["--verbose", "path3"], tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["path3"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

def test_make_config_with_invalid_toml_key():
    toml_data = """
        [tool.vulture]
        invalid_key = "value"
    """
    tomlfile = io.StringIO(toml_data)
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config([], tomlfile)

def test_make_config_with_invalid_toml_type():
    toml_data = """
        [tool.vulture]
        min_confidence = "not_an_int"
    """
    tomlfile = io.StringIO(toml_data)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config([], tomlfile)

def test_make_config_with_no_paths():
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([])


# LLM-generated content at query #4
#--------------------------

```python
def test_check_input_config_with_correct_types():
    DEFAULTS = {"key1": 1, "key2": "value", "key3": True}
    data = {"key1": 2, "key2": "another_value", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #5
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": "value1", "key2": 42, "key3": True}
    DEFAULTS = {"key1": "default1", "key2": 0, "key3": False}
    _check_input_config(data)


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["--help"])
    assert config == {}

def test_make_config_cli_only():
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

def test_make_config_toml_only():
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
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    verbose = false
    paths = ["path1"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path2"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path2"]

def test_make_config_unknown_key():
    import pytest
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--unknown-key", "value"])

def test_make_config_wrong_type():
    import pytest
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])

def test_make_config_no_paths():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_verbose_toml_output():
    import io
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    import sys
    from unittest.mock import patch
    with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        config = make_config(tomlfile=tomlfile)
        assert "Reading configuration from" in mock_stdout.getvalue()


# LLM-generated content at query #7
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True


# LLM-generated content at query #8
#--------------------------

```python
def test_parse_args_defaults():
    args = _parse_args([])
    assert args == {}

def test_parse_args_paths():
    args = _parse_args(["file1.py", "file2.py"])
    assert args == {"paths": ["file1.py", "file2.py"]}

def test_parse_args_exclude():
    args = _parse_args(["--exclude", "*.py,test_*.py"])
    assert args == {"exclude": ["*.py", "test_*.py"]}

def test_parse_args_ignore_decorators():
    args = _parse_args(["--ignore-decorators", "@app.route,@require_*"])
    assert args == {"ignore_decorators": ["@app.route", "@require_*"]}

def test_parse_args_ignore_names():
    args = _parse_args(["--ignore-names", "visit_*,do_*"])
    assert args == {"ignore_names": ["visit_*", "do_*"]}

def test_parse_args_make_whitelist():
    args = _parse_args(["--make-whitelist"])
    assert args == {"make_whitelist": True}

def test_parse_args_min_confidence():
    args = _parse_args(["--min-confidence", "80"])
    assert args == {"min_confidence": 80}

def test_parse_args_sort_by_size():
    args = _parse_args(["--sort-by-size"])
    assert args == {"sort_by_size": True}

def test_parse_args_config():
    args = _parse_args(["--config", "custom.toml"])
    assert args == {"config": "custom.toml"}

def test_parse_args_verbose():
    args = _parse_args(["-v"])
    assert args == {"verbose": True}

def test_parse_args_multiple_options():
    args = _parse_args([
        "file.py",
        "--exclude", "*.py",
        "--ignore-decorators", "@app.route",
        "--min-confidence", "90",
        "--verbose"
    ])
    assert args == {
        "paths": ["file.py"],
        "exclude": ["*.py"],
        "ignore_decorators": ["@app.route"],
        "min_confidence": 90,
        "verbose": True
    }

def test_parse_args_invalid_key():
    with pytest.raises(InputError):
        _parse_args(["--invalid-key", "value"])

def test_parse_args_wrong_type():
    with pytest.raises(InputError):
        _parse_args(["--min-confidence", "not_a_number"])


# LLM-generated content at query #9
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("valid.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #10
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="pyproject.toml")
    assert config["verbose"] is True


# LLM-generated content at query #11
#--------------------------

```python
def test_check_input_config_valid_types():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["--version"])
    assert config == {}

def test_make_config_cli_only():
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

def test_make_config_toml_only():
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
    from io import StringIO
    tomlfile = StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    toml_content = """
[tool.vulture]
min_confidence = 10
verbose = false
"""
    from io import StringIO
    tomlfile = StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "50", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

def test_make_config_no_paths_error():
    from vulture.core import InputError
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_make_config_unknown_key_error():
    from vulture.core import InputError
    try:
        make_config(argv=["--unknown-key", "value"])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_make_config_wrong_type_error():
    from vulture.core import InputError
    try:
        make_config(argv=["--min-confidence", "not_a_number"])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"


# LLM-generated content at query #13
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("test.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #14
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == DEFAULTS["exclude"]
    assert config["ignore_decorators"] == DEFAULTS["ignore_decorators"]
    assert config["ignore_names"] == DEFAULTS["ignore_names"]
    assert config["make_whitelist"] == DEFAULTS["make_whitelist"]
    assert config["min_confidence"] == DEFAULTS["min_confidence"]
    assert config["sort_by_size"] == DEFAULTS["sort_by_size"]
    assert config["verbose"] == DEFAULTS["verbose"]

def test_make_config_cli_overrides():
    config = make_config(argv=[
        "--exclude", "test_*,*.pyc",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "--make-whitelist",
        "--min-confidence", "50",
        "--sort-by-size",
        "--verbose",
        "path1", "path2"
    ])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test_*", "*.pyc"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_toml_overrides():
    toml_content = """
        [tool.vulture]
        exclude = ["test_*", "*.pyc"]
        ignore_decorators = ["deco1", "deco2"]
        ignore_names = ["name1", "name2"]
        make_whitelist = true
        min_confidence = 50
        sort_by_size = true
        verbose = true
        paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test_*", "*.pyc"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    toml_content = """
        [tool.vulture]
        exclude = ["test_*", "*.pyc"]
        ignore_decorators = ["deco1", "deco2"]
        ignore_names = ["name1", "name2"]
        make_whitelist = true
        min_confidence = 50
        sort_by_size = true
        verbose = true
        paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(
        argv=[
            "--exclude", "override_*,*.pyc",
            "--ignore-decorators", "override_deco",
            "--ignore-names", "override_name",
            "--min-confidence", "90",
            "override_path1", "override_path2"
        ],
        tomlfile=tomlfile
    )
    assert config["paths"] == ["override_path1", "override_path2"]
    assert config["exclude"] == ["override_*", "*.pyc"]
    assert config["ignore_decorators"] == ["override_deco"]
    assert config["ignore_names"] == ["override_name"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 90
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_empty_paths_raises():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

def test_make_config_unknown_key_raises():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--unknown-key", "value"])

def test_make_config_wrong_type_raises():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"])


# LLM-generated content at query #15
#--------------------------

```python
def test_check_input_config_type_mismatch():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default_value"}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"
    else:
        assert False, "Expected InputError was not raised"


# LLM-generated content at query #16
#--------------------------

```python
def test_verbose_toml_path_print():
    config = make_config(argv=["--verbose"], tomlfile=io.StringIO("[tool.vulture]\nverbose = true"))
    assert config["verbose"] is True
    assert config["config"] == "pyproject.toml"


# LLM-generated content at query #17
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    toml_path = pathlib.Path("pyproject.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #18
#--------------------------

```python
def test_check_input_config_valid():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 123}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
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


# LLM-generated content at query #19
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="pyproject.toml")
    assert config["verbose"] is True


# LLM-generated content at query #20
#--------------------------

```python
def test_check_input_config_type_matching():
    data = {"key1": 42, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #21
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile

    # Create a temporary file to simulate a TOML file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name).resolve()

    # Ensure the file exists
    assert tmp_path.is_file()


# LLM-generated content at query #22
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--exclude", "test_*.py", "--verbose", "src/"])
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]
    assert "min_confidence" in config
    assert "sort_by_size" in config

def test_make_config_with_toml_file():
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    verbose = true
    paths = ["src/"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    verbose = false
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--exclude", "temp_*.py", "--verbose"], tomlfile=tomlfile)
    assert config["exclude"] == ["temp_*.py"]
    assert config["verbose"] is True

def test_make_config_defaults():
    config = make_config(["src/"])
    assert config["paths"] == ["src/"]
    assert config["min_confidence"] == DEFAULTS["min_confidence"]
    assert config["sort_by_size"] == DEFAULTS["sort_by_size"]

def test_make_config_empty_paths_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config([])


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #24
#--------------------------

```python
def test_check_input_config_valid_types():
    DEFAULTS = {"key1": 1, "key2": "value", "key3": True}
    data = {"key1": 2, "key2": "another", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #25
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] is True


# LLM-generated content at query #26
#--------------------------

```python
def test_check_input_config_valid_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #27
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile

    # Create a temporary file to simulate a TOML file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name).resolve()

    # Ensure the path exists and is a file
    assert tmp_path.is_file()


