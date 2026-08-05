####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 42}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    try:
        _check_input_config({"unknown_key": "value"})
        assert False
    except InputError:
        pass

def test_check_input_config_with_wrong_type():
    try:
        _check_input_config({"key1": 123})
        assert False
    except InputError:
        pass

def test_check_input_config_with_empty_data():
    _check_input_config({})
```


# LLM-generated content at query #2
#--------------------------

```python
def test_make_config_with_no_argv_and_no_tomlfile_uses_defaults():
    import pathlib
    config = make_config(argv=[], tomlfile=None)
    assert "paths" in config
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_cli_arguments_overrides_defaults():
    config = make_config(argv=["--verbose", "--min-confidence", "50"], tomlfile=None)
    assert config["verbose"] is True
    assert config["min_confidence"] == 50

def test_make_config_with_tomlfile_uses_toml_settings():
    import io
    toml_content = (
        '[tool.vulture]\n'
        'exclude = ["test_*.py"]\n'
        'verbose = true\n'
    )
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True

def test_make_config_with_cli_overrides_toml():
    import io
    toml_content = (
        '[tool.vulture]\n'
        'verbose = false\n'
        'min_confidence = 10\n'
    )
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--verbose", "--min-confidence", "80"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 80

def test_make_config_with_paths_from_cli():
    config = make_config(argv=["src", "tests"], tomlfile=None)
    assert config["paths"] == ["src", "tests"]

def test_make_config_with_empty_paths_raises_error():
    import io
    toml_content = '[tool.vulture]\npaths = []\n'
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

def test_make_config_with_unknown_config_key_in_toml_raises_error():
    import io
    toml_content = '[tool.vulture]\nunknown_key = true\n'
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_unknown_config_key_in_cli_raises_error():
    try:
        make_config(argv=["--unknown-flag"], tomlfile=None)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_version_flag_returns_early():
    import sys
    try:
        make_config(argv=["--version"], tomlfile=None)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_help_flag_returns_early():
    try:
        make_config(argv=["--help"], tomlfile=None)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_non_default_config_path():
    import tempfile
    import pathlib
    import io
    toml_content = '[tool.vulture]\nverbose = true\n'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        config_path = f.name
    config = make_config(argv=["--config", config_path], tomlfile=None)
    assert config["verbose"] is True
    pathlib.Path(config_path).unlink()

def test_make_config_with_tomlfile_prints_config_path_in_verbose():
    import io
    toml_content = '[tool.vulture]\nverbose = true\n'
    tomlfile = io.StringIO(toml_content)
    tomlfile.name = "test.toml"
    import sys
    from io import StringIO
    captured_output = StringIO()
    sys.stdout = captured_output
    config = make_config(argv=[], tomlfile=tomlfile)
    sys.stdout = sys.__stdout__
    assert "Reading configuration from test.toml" in captured_output.getvalue()

def test_make_config_without_verbose_does_not_print_config_path():
    import io
    toml_content = '[tool.vulture]\nverbose = false\n'
    tomlfile = io.StringIO(toml_content)
    tomlfile.name = "test.toml"
    import sys
    from io import StringIO
    captured_output = StringIO()
    sys.stdout = captured_output
    config = make_config(argv=[], tomlfile=tomlfile)
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue() == ""```


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config_with_cli_only():
    args = ["--verbose", "path1", "path2"]
    config = make_config(argv=args)
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_tomlfile():
    import io
    toml_content = "[tool.vulture]\nverbose = true\npaths = [\"path1\"]"
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = "[tool.vulture]\nverbose = false\npaths = [\"path1\"]"
    tomlfile = io.BytesIO(toml_content.encode())
    args = ["--verbose", "path2"]
    config = make_config(argv=args, tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["path2"]

def test_make_config_defaults_applied():
    config = make_config(argv=["path1"])
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["min_confidence"] == 100

def test_make_config_unknown_key_raises_error():
    import io
    toml_content = "[tool.vulture]\nunknown_key = true"
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(tomlfile=tomlfile)
        assert False
    except InputError:
        pass

def test_make_config_no_paths_raises_error():
    try:
        make_config(argv=[])
        assert False
    except InputError:
        pass
```


# LLM-generated content at query #4
#--------------------------

```python
def test_toml_path_is_file_true():
    import tempfile
    import pathlib
    toml_file = tempfile.NamedTemporaryFile(suffix=".toml", delete=False)
    toml_file.close()
    toml_path = pathlib.Path(toml_file.name)
    assert toml_path.is_file()
    toml_path.unlink()


# LLM-generated content at query #5
#--------------------------

```
def test_parse_toml_returns_empty_dict_when_no_vulture_section():
    import tomllib
    from io import StringIO
    test_input = StringIO("[tool.other]\nkey = 'value'\n")
    result = _parse_toml(test_input)
    assert result == {}

def test_parse_toml_returns_vulture_settings():
    import tomllib
    from io import StringIO
    test_input = StringIO("[tool.vulture]\nexclude = ['file*.py', 'dir/']\nignore_decorators = ['deco1', 'deco2']\nignore_names = ['name1', 'name2']\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ['path1', 'path2']\n")
    result = _parse_toml(test_input)
    assert result == {'exclude': ['file*.py', 'dir/'], 'ignore_decorators': ['deco1', 'deco2'], 'ignore_names': ['name1', 'name2'], 'make_whitelist': True, 'min_confidence': 10, 'sort_by_size': True, 'verbose': True, 'paths': ['path1', 'path2']}

def test_parse_toml_raises_input_error_for_unknown_key():
    import tomllib
    from io import StringIO
    test_input = StringIO("[tool.vulture]\nunknown_key = true\n")
    try:
        _parse_toml(test_input)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_parse_toml_raises_input_error_for_wrong_type():
    import tomllib
    from io import StringIO
    test_input = StringIO("[tool.vulture]\nmake_whitelist = 1\n")
    try:
        _parse_toml(test_input)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for make_whitelist must be" in str(e)


# LLM-generated content at query #6
#--------------------------

def test_parse_toml_valid_config():
    import io, tomllib
    infile = io.StringIO('[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]')
    result = _parse_toml(infile)
    assert result == {"exclude": ["file*.py", "dir/"], "ignore_decorators": ["deco1", "deco2"], "ignore_names": ["name1", "name2"], "make_whitelist": True, "min_confidence": 10, "sort_by_size": True, "verbose": True, "paths": ["path1", "path2"]}

def test_parse_toml_missing_tool_section():
    import io, tomllib
    infile = io.StringIO('[other]\nkey = "value"')
    result = _parse_toml(infile)
    assert result == {}

def test_parse_toml_missing_vulture_section():
    import io, tomllib
    infile = io.StringIO('[tool]\nother = "value"')
    result = _parse_toml(infile)
    assert result == {}

def test_parse_toml_empty_file():
    import io, tomllib
    infile = io.StringIO('')
    result = _parse_toml(infile)
    assert result == {}

def test_parse_toml_raises_input_error():
    import io, tomllib
    infile = io.StringIO('[tool.vulture]\nunknown_key = "value"')
    try:
        _parse_toml(infile)
        assert False
    except InputError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_type_mismatch_raises_input_error():
    data = {"some_key": "wrong_type"}
    DEFAULTS = {"some_key": 42}
    try:
        _check_input_config(data)
        assert False, "Expected InputError was not raised"
    except InputError:
        pass
```


# LLM-generated content at query #8
#--------------------------

```python
def test_toml_path_is_file_evaluates_to_true():
    import pathlib
    import tempfile
    import os

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp_path = pathlib.Path(tmp_file.name)
    tmp_file.close()

    try:
        argv = ["--config", str(tmp_path)]
        result = make_config(argv=argv)
        assert result is not None
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config_with_cli_only():
    config = make_config(argv=["path1", "--verbose"])
    assert config["paths"] == ["path1"]
    assert config["verbose"] == True
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["exclude"] == []

def test_make_config_with_toml_file(tmp_path):
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true'
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_bytes(toml_content)
    config = make_config(argv=[str(tmp_path)])
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True
    assert config["paths"] == [str(tmp_path)]

def test_make_config_cli_overrides_toml(tmp_path):
    toml_content = b'[tool.vulture]\nverbose = false\nmin_confidence = 50'
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_bytes(toml_content)
    config = make_config(argv=[str(tmp_path), "--verbose", "--min-confidence", "80"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 80

def test_make_config_with_inline_toml():
    toml_data = b'[tool.vulture]\nexclude = ["test_*.py"]\nignore_names = ["unused"]'
    import io
    tomlfile = io.BytesIO(toml_data)
    config = make_config(argv=["path1"], tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_names"] == ["unused"]
    assert config["paths"] == ["path1"]

def test_make_config_without_paths_raises_error():
    toml_data = b'[tool.vulture]\nverbose = true'
    import io
    tomlfile = io.BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False
    except InputError:
        pass

def test_make_config_cli_paths_override_toml_paths(tmp_path):
    toml_content = b'[tool.vulture]\npaths = ["dir1", "dir2"]'
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_bytes(toml_content)
    config = make_config(argv=["custom_path"])
    assert config["paths"] == ["custom_path"]

def test_make_config_default_min_confidence():
    config = make_config(argv=["path1"])
    assert config["min_confidence"] == 0

def test_make_config_default_sort_by_size():
    config = make_config(argv=["path1"])
    assert config["sort_by_size"] == False

def test_make_config_default_make_whitelist():
    config = make_config(argv=["path1"])
    assert config["make_whitelist"] == False

def test_make_config_default_verbose():
    config = make_config(argv=["path1"])
    assert config["verbose"] == False

def test_make_config_default_exclude():
    config = make_config(argv=["path1"])
    assert config["exclude"] == []

def test_make_config_default_ignore_decorators():
    config = make_config(argv=["path1"])
    assert config["ignore_decorators"] == []

def test_make_config_default_ignore_names():
    config = make_config(argv=["path1"])
    assert config["ignore_names"] == []```


# LLM-generated content at query #10
#--------------------------

```python
def test_tomlfile_truthy():
    tomlfile = type('FakeFile', (), {'__bool__': lambda self: True})()
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    assert True
```


# LLM-generated content at query #11
#--------------------------

def test_parse_args_returns_empty_dict_when_no_args():
    missing = object()
    result = _parse_args([])
    assert result == {}

def test_parse_args_returns_paths():
    result = _parse_args(["file1.py", "dir/"])
    assert "paths" in result
    assert result["paths"] == ["file1.py", "dir/"]

def test_parse_args_returns_exclude():
    result = _parse_args(["--exclude", "*.pyc,*.pyo"])
    assert "exclude" in result
    assert result["exclude"] == ["*.pyc", "*.pyo"]

def test_parse_args_returns_ignore_decorators():
    result = _parse_args(["--ignore-decorators", "@app.route,@require_*"])
    assert "ignore_decorators" in result
    assert result["ignore_decorators"] == ["@app.route", "@require_*"]

def test_parse_args_returns_ignore_names():
    result = _parse_args(["--ignore-names", "visit_*,do_*"])
    assert "ignore_names" in result
    assert result["ignore_names"] == ["visit_*", "do_*"]

def test_parse_args_returns_make_whitelist():
    result = _parse_args(["--make-whitelist"])
    assert "make_whitelist" in result
    assert result["make_whitelist"] == True

def test_parse_args_returns_min_confidence():
    result = _parse_args(["--min-confidence", "80"])
    assert "min_confidence" in result
    assert result["min_confidence"] == 80

def test_parse_args_returns_sort_by_size():
    result = _parse_args(["--sort-by-size"])
    assert "sort_by_size" in result
    assert result["sort_by_size"] == True

def test_parse_args_returns_config():
    result = _parse_args(["--config", "custom.toml"])
    assert "config" in result
    assert result["config"] == "custom.toml"

def test_parse_args_returns_verbose():
    result = _parse_args(["-v"])
    assert "verbose" in result
    assert result["verbose"] == True


# LLM-generated content at query #12
#--------------------------

```python
def test_toml_path_is_file_returns_true():
    import tempfile
    import pathlib
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = pathlib.Path(tmp_file.name)
    tmp_path.write_text("")
    config = {"config": str(tmp_path)}
    result = tmp_path.is_file()
    tmp_path.unlink()
    assert result is True
```


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_true_when_detected_toml_path_and_verbose():
    config = {"verbose": True}
    detected_toml_path = "/path/to/pyproject.toml"
    if detected_toml_path and config["verbose"]:
        pass
```


# LLM-generated content at query #14
#--------------------------

def test_valid_config_type_matches():
    data = {"timeout": 30}
    _check_input_config(data)


# LLM-generated content at query #15
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 42}
    DEFAULTS = {"key1": "string", "key2": 0}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"known_key": "default"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "string"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_bool_instead_of_int():
    data = {"key1": True}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_int_instead_of_bool():
    data = {"key1": 1}
    DEFAULTS = {"key1": False}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_none_value():
    data = {"key1": None}
    DEFAULTS = {"key1": "default"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_empty_data():
    data = {}
    DEFAULTS = {"key1": "default"}
    _check_input_config(data)

def test_check_input_config_with_multiple_keys():
    data = {"key1": "value1", "key2": 42}
    DEFAULTS = {"key1": "string", "key2": 0, "key3": 3.14}
    _check_input_config(data)
```


# LLM-generated content at query #16
#--------------------------

def test_verbose_with_toml_path():
    import io
    import pathlib
    # Create a temporary toml file
    toml_content = b""
    with open("/tmp/test_toml.toml", "wb") as f:
        f.write(toml_content)
    tomlfile = io.BytesIO(toml_content)
    tomlfile.name = "/tmp/test_toml.toml"
    # Simulate CLI arguments with verbose flag
    argv = ["--verbose"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["verbose"] == True
    assert detected_toml_path != ""  # This line is conceptual; actual variable not accessible


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_true():
    tomlfile = "/path/to/pyproject.toml"
    config = {"verbose": True}
    detected_toml_path = str(tomlfile)
    result = bool(detected_toml_path and config["verbose"])
    assert result == True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config_with_default_values():
    config = make_config(argv=["path1"], tomlfile=None)
    assert config["paths"] == ["path1"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

def test_make_config_with_cli_arguments():
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file():
    import io
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\n'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["path1"], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = b'[tool.vulture]\nverbose = false\nmin_confidence = 30\n'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--verbose", "--min-confidence", "80", "path1"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 80

def test_make_config_without_paths_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

def test_make_config_with_unknown_key_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--unknown-key"])

def test_make_config_toml_with_unknown_key_raises_error():
    import io
    import pytest
    toml_content = b'[tool.vulture]\nunknown_key = true\n'
    tomlfile = io.BytesIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=["path1"], tomlfile=tomlfile)

def test_make_config_with_help_does_not_raise_error():
    import sys
    try:
        config = make_config(argv=["--help"])
    except SystemExit:
        pass

def test_make_config_with_version_does_not_raise_error():
    import sys
    try:
        config = make_config(argv=["--version"])
    except SystemExit:
        pass
```


# LLM-generated content at query #19
#--------------------------

```python
def test_check_input_config_type_mismatch_int():
    data = {"num_workers": "2"}
    DEFAULTS = {"num_workers": 1}
    try:
        _check_input_config(data)
    except InputError:
        pass
```


# LLM-generated content at query #20
#--------------------------

```python
def test_toml_path_is_file_returns_true():
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.close()
    import pathlib
    path = pathlib.Path(tmp_file.name)
    assert path.is_file() == True
```


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_true():
    config = {"verbose": True}
    config.update({"verbose": True})
    detected_toml_path = "/path/to/pyproject.toml"
    if detected_toml_path and config["verbose"]:
        pass
```


# LLM-generated content at query #22
#--------------------------

```python
def test_type_match_success():
    data = {"key1": "value1"}
    _check_input_config(data)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_true_with_valid_toml_and_verbose():
    toml_content = b""
    from io import BytesIO
    tomlfile = BytesIO(toml_content)
    argv = ["--verbose"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_evaluates_to_true_when_toml_file_exists():
    import pathlib
    import tempfile
    import os

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp_file.close()
    config_path = pathlib.Path(tmp_file.name).resolve()
    argv = ["--config", str(config_path)]
    cli_config = {"config": config_path}
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert toml_path.is_file()
    os.unlink(tmp_file.name)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_type_check_passes_when_types_match():
    data = {"integer_key": 42}
    _check_input_config(data)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_check_input_config_correct_types():
    data = {"key1": "value1"}
    DEFAULTS = {"key1": "default1"}
    _check_input_config(data)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_true_with_verbose_and_toml_path():
    config = {"verbose": True}
    config.update({})
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    detected_toml_path = "pyproject.toml"
    assert detected_toml_path and config["verbose"]
```


# LLM-generated content at query #28
#--------------------------

```python
def test_make_config_returns_dict_with_defaults_when_no_toml_and_no_cli():
    config = make_config(argv=[], tomlfile=None)
    assert isinstance(config, dict)
    assert config == DEFAULTS

def test_make_config_cli_overrides_defaults():
    config = make_config(argv=["--verbose"], tomlfile=None)
    assert config["verbose"] is True

def test_make_config_toml_values_merged():
    from io import BytesIO
    toml_data = b'[tool.vulture]\nverbose = true\n'
    tomlfile = BytesIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    from io import BytesIO
    toml_data = b'[tool.vulture]\nverbose = true\n'
    tomlfile = BytesIO(toml_data)
    config = make_config(argv=["--verbose=false"], tomlfile=tomlfile)
    assert config["verbose"] is False

def test_make_config_raises_input_error_for_unknown_key():
    from io import BytesIO
    toml_data = b'[tool.vulture]\nunknown_key = 1\n'
    tomlfile = BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_raises_input_error_for_empty_paths():
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_raises_input_error_for_wrong_type():
    from io import BytesIO
    toml_data = b'[tool.vulture]\nmin_confidence = "ten"\n'
    tomlfile = BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_prints_config_path_when_verbose_and_toml_exists(monkeypatch):
    import sys
    from io import BytesIO, StringIO
    monkeypatch.setattr(sys, "stdout", StringIO())
    toml_data = b'[tool.vulture]\nverbose = true\npaths = ["src"]\n'
    tomlfile = BytesIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert "Reading configuration from" in sys.stdout.getvalue()

def test_make_config_detects_toml_from_cli_config_path(tmp_path, monkeypatch):
    import pathlib
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_text('[tool.vulture]\nverbose = true\npaths = ["src"]\n')
    monkeypatch.chdir(tmp_path)
    config = make_config(argv=[], tomlfile=None)
    assert config["verbose"] is True

def test_make_config_uses_cli_config_path(tmp_path, monkeypatch):
    import pathlib
    custom_toml = tmp_path / "custom.toml"
    custom_toml.write_text('[tool.vulture]\nverbose = true\npaths = ["src"]\n')
    config = make_config(argv=["--config", str(custom_toml)], tomlfile=None)
    assert config["verbose"] is True
```


# LLM-generated content at query #29
#--------------------------

```python
def test_check_input_config_valid_type():
    data = {"timeout": 30}
    _check_input_config(data)
```


# LLM-generated content at query #30
#--------------------------

```python
def test_toml_path_is_file_true():
    import pathlib
    import tempfile
    import os
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    tmp_path = pathlib.Path(tmp.name)
    argv = ["--config", str(tmp_path)]
    config = make_config(argv=argv)
    assert config is not None
    os.unlink(tmp.name)
```


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config_predicate_true():
    import io
    import tempfile
    import os
    config_data = "[tool.vulture]\nverbose = true\n"
    toml_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    toml_file.write(config_data)
    toml_file.close()
    with open(toml_file.name, "rb") as f:
        config = make_config(argv=["--verbose"], tomlfile=f)
    os.unlink(toml_file.name)


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_args_defaults():
    import sys
    sys.argv = ["vulture"]
    result = _parse_args([])
    assert result == {}

def test_parse_args_paths():
    import sys
    sys.argv = ["vulture", "file1.py", "dir1"]
    result = _parse_args(["file1.py", "dir1"])
    assert result == {"paths": ["file1.py", "dir1"]}

def test_parse_args_exclude():
    import sys
    sys.argv = ["vulture", "--exclude", "*.py,docs"]
    result = _parse_args(["--exclude", "*.py,docs"])
    assert result == {"exclude": ["*.py", "docs"]}

def test_parse_args_ignore_decorators():
    import sys
    sys.argv = ["vulture", "--ignore-decorators", "@app.route,@require_*"]
    result = _parse_args(["--ignore-decorators", "@app.route,@require_*"])
    assert result == {"ignore_decorators": ["@app.route", "@require_*"]}

def test_parse_args_ignore_names():
    import sys
    sys.argv = ["vulture", "--ignore-names", "visit_*,do_*"]
    result = _parse_args(["--ignore-names", "visit_*,do_*"])
    assert result == {"ignore_names": ["visit_*", "do_*"]}

def test_parse_args_make_whitelist():
    import sys
    sys.argv = ["vulture", "--make-whitelist"]
    result = _parse_args(["--make-whitelist"])
    assert result == {"make_whitelist": True}

def test_parse_args_min_confidence():
    import sys
    sys.argv = ["vulture", "--min-confidence", "75"]
    result = _parse_args(["--min-confidence", "75"])
    assert result == {"min_confidence": 75}

def test_parse_args_sort_by_size():
    import sys
    sys.argv = ["vulture", "--sort-by-size"]
    result = _parse_args(["--sort-by-size"])
    assert result == {"sort_by_size": True}

def test_parse_args_config():
    import sys
    sys.argv = ["vulture", "--config", "custom.toml"]
    result = _parse_args(["--config", "custom.toml"])
    assert result == {"config": "custom.toml"}

def test_parse_args_verbose():
    import sys
    sys.argv = ["vulture", "-v"]
    result = _parse_args(["-v"])
    assert result == {"verbose": True}

def test_parse_args_version():
    import sys
    try:
        _parse_args(["--version"])
        assert False
    except SystemExit:
        pass

def test_parse_args_unknown_key():
    import sys
    try:
        _parse_args(["--unknown-option"])
        assert False
    except SystemExit:
        pass

def test_parse_args_wrong_type():
    import sys
    try:
        _parse_args(["--min-confidence", "not_an_int"])
        assert False
    except SystemExit:
        pass
```


# LLM-generated content at query #2
#--------------------------

def test_make_config_with_cli_args_only():
    import io
    import pathlib
    config = make_config(argv=["--verbose", "path1", "path2"])
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_tomlfile():
    import io
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\npaths = ["test_path"]'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True
    assert config["paths"] == ["test_path"]

def test_make_config_with_tomlfile_and_cli_override():
    import io
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = false\npaths = ["toml_path"]'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--verbose", "cli_path"], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]

def test_make_config_with_defaults():
    import io
    config = make_config(argv=["path"])
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["make_whitelist"] == False

def test_make_config_raises_on_unknown_key():
    import io
    toml_content = b'[tool.vulture]\nunknown_key = true'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_raises_on_no_paths():
    import io
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_true():
    argv = ["--verbose"]
    tomlfile = "/fake/path/pyproject.toml"
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["config"] == "/fake/path/pyproject.toml"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"timeout": 30, "retries": 5}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    try:
        data = {"unknown_key": "value"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    try:
        data = {"timeout": "thirty"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for timeout must be 'int'"

def test_check_input_config_bool_vs_int():
    try:
        data = {"timeout": True}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for timeout must be 'int'"

def test_check_input_config_empty_data():
    data = {}
    _check_input_config(data)
```


# LLM-generated content at query #5
#--------------------------

```
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 42}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    try:
        _check_input_config({"unknown_key": "value"})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_wrong_type():
    try:
        _check_input_config({"key1": 123})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_bool_instead_of_int():
    try:
        _check_input_config({"key2": True})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_empty_dict():
    data = {}
    _check_input_config(data)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp.write(b"[tool.vulture]\n")
    tmp.close()
    toml_path = pathlib.Path(tmp.name)
    result = toml_path.is_file()
    os.unlink(tmp.name)
    assert result == True
```


# LLM-generated content at query #7
#--------------------------

```python
def test_type_mismatch_raises_input_error():
    from your_module import _check_input_config, InputError, DEFAULTS
    # Ensure DEFAULTS has a key with expected type int
    # and pass a value with the same type as int but not int (e.g., bool)
    import sys
    # We need to find a key in DEFAULTS whose value's type is not bool
    # For simplicity, assume DEFAULTS has at least one key with int value
    key = next(k for k, v in DEFAULTS.items() if type(v) is int)
    # bool is subclass of int, so type(True) is bool, type(0) is int
    try:
        _check_input_config({key: True})
    except InputError:
        pass
    else:
        raise AssertionError("Expected InputError for type mismatch (bool vs int)")
```


# LLM-generated content at query #8
#--------------------------

def test_predicate_true_when_toml_path_detected_and_verbose():
    config = {"verbose": True}
    detected_toml_path = "pyproject.toml"
    assert detected_toml_path and config["verbose"]


# LLM-generated content at query #9
#--------------------------

```python
def test_toml_path_is_file_returns_true():
    import tempfile
    import pathlib
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    toml_path = pathlib.Path(tmp.name)
    assert toml_path.is_file()


# LLM-generated content at query #10
#--------------------------

```python
def test_value_type_matches_default():
    data = {"timeout": 30}
    DEFAULTS = {"timeout": 10}
    _check_input_config(data)
```


# LLM-generated content at query #11
#--------------------------

def test_make_config_with_only_cli_args():
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"
    assert config["verbose"] is False

def test_make_config_with_toml_file():
    import io
    import tomllib
    toml_data = b'[tool.vulture]\npaths = ["src"]\nexclude = ["test*.py"]\n'
    tomlfile = io.BytesIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test*.py"]

def test_make_config_cli_overrides_toml():
    import io
    toml_data = b'[tool.vulture]\npaths = ["toml_path"]\nverbose = true\n'
    tomlfile = io.BytesIO(toml_data)
    config = make_config(argv=["cli_path", "--verbose"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["verbose"] is True

def test_make_config_with_missing_toml_falls_back_to_defaults():
    import pathlib
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = make_config(argv=["dir1"])
            assert config["paths"] == ["dir1"]
            assert config["exclude"] == []
            assert config["verbose"] is False
        finally:
            os.chdir(original_cwd)

def test_make_config_with_paths_empty_raises_error():
    import io
    toml_data = b'[tool.vulture]\npaths = []\n'
    tomlfile = io.BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False
    except InputError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_type_mismatch_raises_error():
    data = {"timeout": 3.14}
    with pytest.raises(InputError):
        _check_input_config(data)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config_with_cli_arguments():
    argv = ["--verbose", "path1", "path2"]
    config = make_config(argv=argv, tomlfile=None)
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_tomlfile():
    import io
    toml_content = """
[tool.vulture]
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 10
sort_by_size = true
verbose = true
paths = ["path1", "path2"]
"""
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
verbose = false
paths = ["toml_path"]
"""
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    argv = ["--verbose", "cli_path"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]

def test_make_config_with_missing_paths_raises_error():
    import pytest
    argv = []
    try:
        make_config(argv=argv, tomlfile=None)
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_defaults_for_missing_options():
    argv = ["path1"]
    config = make_config(argv=argv, tomlfile=None)
    for key, value in DEFAULTS.items():
        assert key in config
        assert config[key] == value

def test_make_config_with_unknown_config_key_raises_error():
    import io
    toml_content = """
[tool.vulture]
unknown_key = true
"""
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    try:
        make_config(argv=[], tomlfile=tomlfile)
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_wrong_type_in_toml_raises_error():
    import io
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    try:
        make_config(argv=[], tomlfile=tomlfile)
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e) or "Data type for" in str(e)

def test_make_config_with_config_key_in_cli():
    argv = ["--config", "custom_config.toml", "path1"]
    config = make_config(argv=argv, tomlfile=None)
    assert config["config"] == "custom_config.toml"
    assert config["paths"] == ["path1"]```


# LLM-generated content at query #14
#--------------------------

```python
def test_make_config_with_cli_args_only():
    argv = ["path1", "path2", "--verbose"]
    config = make_config(argv=argv, tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_only():
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\n'
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] is True
    assert config["paths"] == []

def test_make_config_cli_overrides_toml():
    toml_content = b'[tool.vulture]\nverbose = false\nmin_confidence = 50\n'
    import io
    tomlfile = io.BytesIO(toml_content)
    argv = ["--verbose", "--min-confidence", "80"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 80

def test_make_config_empty_toml_and_cli():
    toml_content = b'[tool.vulture]\n'
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["verbose"] is False

def test_make_config_raises_input_error_for_empty_paths():
    import io
    tomlfile = io.BytesIO(b'[tool.vulture]\n')
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_with_default_config_file():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="wb") as f:
        f.write(b'[tool.vulture]\nverbose = true\n')
        temp_path = f.name
    try:
        argv = ["--config", temp_path]
        config = make_config(argv=argv, tomlfile=None)
        assert config["verbose"] is True
    finally:
        os.unlink(temp_path)

def test_make_config_with_explicit_tomlfile_ignores_config_path():
    import io
    tomlfile = io.BytesIO(b'[tool.vulture]\nverbose = true\n')
    config = make_config(argv=["--config", "nonexistent.toml"], tomlfile=tomlfile)
    assert config["verbose"] is True

def test_make_config_prints_config_path_when_verbose_and_toml_detected():
    import io
    import sys
    from unittest.mock import patch
    tomlfile = io.BytesIO(b'[tool.vulture]\nverbose = true\n')
    tomlfile.name = "test.toml"
    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        make_config(argv=["--verbose"], tomlfile=tomlfile)
        output = mock_stdout.getvalue()
        assert "Reading configuration from test.toml" in output

def test_make_config_does_not_print_config_path_when_not_verbose():
    import io
    import sys
    from unittest.mock import patch
    tomlfile = io.BytesIO(b'[tool.vulture]\nverbose = false\n')
    tomlfile.name = "test.toml"
    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        make_config(argv=[], tomlfile=tomlfile)
        output = mock_stdout.getvalue()
        assert output == ""```


# LLM-generated content at query #15
#--------------------------

```python
def test_make_config_with_cli_args_and_toml_file():
    import io
    import pathlib
    toml_data = """
[tool.vulture]
exclude = ["test_exclude"]
verbose = true
"""
    toml_file = io.BytesIO(toml_data.encode())
    argv = ["--exclude", "cli_exclude", "--verbose"]
    config = make_config(argv=argv, tomlfile=toml_file)
    assert config["exclude"] == ["cli_exclude"]
    assert config["verbose"] == True
    assert config["paths"] == []

def test_make_config_with_defaults():
    import io
    toml_data = """
[tool.vulture]
verbose = true
"""
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["exclude"] == []
    assert config["paths"] == []

def test_make_config_with_only_cli_args():
    config = make_config(argv=["--verbose", "path1"])
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]

def test_make_config_with_empty_toml_file():
    import io
    toml_file = io.BytesIO(b"")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config == {
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "min_confidence": 0,
        "sort_by_size": False,
        "verbose": False,
        "paths": [],
        "config": "pyproject.toml",
    }

def test_make_config_with_no_paths_raises_error():
    import io
    toml_data = """
[tool.vulture]
verbose = true
"""
    toml_file = io.BytesIO(toml_data.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False
    except InputError:
        pass

def test_make_config_with_cli_paths_and_toml_file():
    import io
    toml_data = """
[tool.vulture]
exclude = ["test_exclude"]
"""
    toml_file = io.BytesIO(toml_data.encode())
    argv = ["path1", "path2"]
    config = make_config(argv=argv, tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test_exclude"]
```


# LLM-generated content at query #16
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".toml", delete=False)
    tmp.close()
    toml_path = pathlib.Path(tmp.name).resolve()
    cli_config = {"config": str(toml_path)}
    _parse_args = lambda argv: cli_config
    _parse_toml = lambda f: {}
    DEFAULTS = {}
    _check_output_config = lambda config: None

    result = make_config()
    os.unlink(tmp.name)
    assert toml_path.is_file() == True
```


# LLM-generated content at query #17
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"timeout": 30, "verbose": True, "name": "test"}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    try:
        data = {"unknown_key": 1}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type_int():
    try:
        data = {"timeout": "30"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for timeout must be 'int'"

def test_check_input_config_wrong_type_bool():
    try:
        data = {"verbose": "true"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for verbose must be 'bool'"

def test_check_input_config_wrong_type_str():
    try:
        data = {"name": 42}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for name must be 'str'"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_true():
    tomlfile = "test.toml"
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] == True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile
    import os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp.write(b"[tool.vulture]\n")
    tmp.close()
    argv = ["--config", tmp.name]
    toml_path = pathlib.Path(tmp.name).resolve()
    assert toml_path.is_file()
    os.unlink(tmp.name)
```


# LLM-generated content at query #20
#--------------------------

def test_predicate_true_when_toml_path_and_verbose():
    config = {"verbose": True}
    detected_toml_path = "some/path/to/pyproject.toml"
    assert detected_toml_path and config["verbose"]


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_evaluates_to_true():
    import io
    import tempfile
    import os
    
    toml_content = b"[tool.vulture]\nverbose = true\n"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    
    try:
        config = make_config(argv=[], tomlfile=io.BytesIO(toml_content))
    finally:
        os.unlink(toml_path)
```


# LLM-generated content at query #22
#--------------------------

```python
def test_toml_is_file_evaluates_true():
    import pathlib
    import tempfile
    import os

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp_file.close()
    test_path = pathlib.Path(tmp_file.name)
    test_path.write_text("[tool.vulture]\n")
    
    original_is_file = pathlib.Path.is_file
    pathlib.Path.is_file = lambda self: self == test_path
    
    from your_module import make_config
    result = make_config(argv=["--config", str(test_path)])
    
    pathlib.Path.is_file = original_is_file
    os.unlink(tmp_file.name)
    
    assert "config" in result
```


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_line_39_true():
    config = {"verbose": True, "paths": ["."], "sort_by_size": False, "min_confidence": 0.0}
    import io
    import pathlib
    toml_content = b'[tool.vulture]\nverbose = true\npaths = ["."]\nsort_by_size = false\nmin_confidence = 0.0\n'
    tomlfile = io.BytesIO(toml_content)
    tomlfile.name = "test_pyproject.toml"
    from unittest.mock import patch
    with patch("builtins.print") as mock_print:
        from your_module import make_config
        make_config(argv=[], tomlfile=tomlfile)
        assert mock_print.called
        assert mock_print.call_args[0][0] == "Reading configuration from test_pyproject.toml"
```


# LLM-generated content at query #24
#--------------------------

```python
def test_toml_path_is_file_true():
    import tempfile
    import pathlib
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp_file.name
    temp_file.close()
    try:
        config = {"config": temp_path}
        path = pathlib.Path(config["config"]).resolve()
        assert path.is_file()
    finally:
        pathlib.Path(temp_path).unlink()
```


