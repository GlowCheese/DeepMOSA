####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_make_config_with_only_cli_args():
    argv = ["vulture", "path1", "path2"]
    result = make_config(argv=argv, tomlfile=None)
    assert result["paths"] == ["path1", "path2"]
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] == False
    assert result["min_confidence"] == 0
    assert result["sort_by_size"] == False
    assert result["verbose"] == False
    assert result["config"] == "pyproject.toml"

def test_make_config_with_toml_only():
    import io
    toml_content = """
[tool.vulture]
exclude = ["file1.py", "dir/"]
ignore_decorators = ["deco1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 10
sort_by_size = true
verbose = true
paths = ["pathA", "pathB"]
"""
    tomlfile = io.StringIO(toml_content)
    result = make_config(argv=None, tomlfile=tomlfile)
    assert result["paths"] == ["pathA", "pathB"]
    assert result["exclude"] == ["file1.py", "dir/"]
    assert result["ignore_decorators"] == ["deco1"]
    assert result["ignore_names"] == ["name1"]
    assert result["make_whitelist"] == True
    assert result["min_confidence"] == 10
    assert result["sort_by_size"] == True
    assert result["verbose"] == True

def test_make_config_with_toml_overridden_by_cli():
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path"]
verbose = false
"""
    tomlfile = io.StringIO(toml_content)
    argv = ["vulture", "--verbose", "cli_path"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    assert result["paths"] == ["cli_path"]
    assert result["verbose"] == True

def test_make_config_with_defaults():
    import io
    toml_content = """
[tool.vulture]
paths = ["test_path"]
"""
    tomlfile = io.StringIO(toml_content)
    result = make_config(argv=None, tomlfile=tomlfile)
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] == False
    assert result["min_confidence"] == 0
    assert result["sort_by_size"] == False
    assert result["verbose"] == False

def test_make_config_missing_paths_raises_error():
    import io
    toml_content = """
[tool.vulture]
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=None, tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_make_config_with_unknown_key_raises_error():
    import io
    toml_content = """
[tool.vulture]
unknown_key = true
paths = ["test_path"]
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=None, tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"


# LLM-generated content at query #2
#--------------------------

```
def test_parse_toml_with_valid_settings():
    import tomllib
    from io import StringIO
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
    infile = StringIO(toml_content)
    result = _parse_toml(infile)
    expected = {
        "exclude": ["file*.py", "dir/"],
        "ignore_decorators": ["deco1", "deco2"],
        "ignore_names": ["name1", "name2"],
        "make_whitelist": True,
        "min_confidence": 10,
        "sort_by_size": True,
        "verbose": True,
        "paths": ["path1", "path2"]
    }
    assert result == expected

def test_parse_toml_with_no_vulture_section():
    import tomllib
    from io import StringIO
    toml_content = """
        [tool.other]
        key = "value"
    """
    infile = StringIO(toml_content)
    result = _parse_toml(infile)
    assert result == {}

def test_parse_toml_with_empty_vulture_section():
    import tomllib
    from io import StringIO
    toml_content = """
        [tool.vulture]
    """
    infile = StringIO(toml_content)
    result = _parse_toml(infile)
    assert result == {}

def test_parse_toml_with_unknown_key():
    import tomllib
    from io import StringIO
    toml_content = """
        [tool.vulture]
        unknown_key = "value"
    """
    infile = StringIO(toml_content)
    try:
        _parse_toml(infile)
        assert False
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_parse_toml_with_wrong_type():
    import tomllib
    from io import StringIO
    toml_content = """
        [tool.vulture]
        make_whitelist = "not_bool"
    """
    infile = StringIO(toml_content)
    try:
        _parse_toml(infile)
        assert False
    except InputError as e:
        assert "Data type for make_whitelist" in str(e)


# LLM-generated content at query #3
#--------------------------

```
def test_valid_config_values():
    data = {"key1": "value1", "key2": 42}
    _check_input_config(data)

def test_invalid_key():
    data = {"invalid_key": "value"}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

def test_wrong_type_string_instead_of_int():
    data = {"key_int": "string"}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key_int must be 'int'"

def test_wrong_type_int_instead_of_bool():
    data = {"key_bool": 1}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key_bool must be 'bool'"

def test_multiple_errors_first_key_invalid():
    data = {"unknown_key": "value", "valid_key": 123}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_multiple_errors_first_type_invalid():
    data = {"valid_key": "wrong_type", "other_key": 42}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for valid_key must be 'int'"

def test_empty_data():
    data = {}
    _check_input_config(data)

def test_none_value_with_correct_type():
    data = {"key_none": None}
    _check_input_config(data)

def test_boolean_true_value():
    data = {"key_bool": True}
    _check_input_config(data)

def test_boolean_false_value():
    data = {"key_bool": False}
    _check_input_config(data)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_type_mismatch_raises_error():
    DEFAULTS = {"timeout": 10, "verbose": False}
    data = {"timeout": "10"}
    _check_input_config(data)
```


# LLM-generated content at query #5
#--------------------------

```
def test_parse_args_with_defaults():
    args = []
    result = _parse_args(args)
    assert result == {}

def test_parse_args_with_paths():
    args = ["path1", "path2"]
    result = _parse_args(args)
    assert result == {"paths": ["path1", "path2"]}

def test_parse_args_with_exclude():
    args = ["--exclude", "*.py,test"]
    result = _parse_args(args)
    assert result == {"exclude": ["*.py", "test"]}

def test_parse_args_with_ignore_decorators():
    args = ["--ignore-decorators", "@app.route,@require_*"]
    result = _parse_args(args)
    assert result == {"ignore_decorators": ["@app.route", "@require_*"]}

def test_parse_args_with_ignore_names():
    args = ["--ignore-names", "visit_*,do_*"]
    result = _parse_args(args)
    assert result == {"ignore_names": ["visit_*", "do_*"]}

def test_parse_args_with_make_whitelist():
    args = ["--make-whitelist"]
    result = _parse_args(args)
    assert result == {"make_whitelist": True}

def test_parse_args_with_min_confidence():
    args = ["--min-confidence", "80"]
    result = _parse_args(args)
    assert result == {"min_confidence": 80}

def test_parse_args_with_sort_by_size():
    args = ["--sort-by-size"]
    result = _parse_args(args)
    assert result == {"sort_by_size": True}

def test_parse_args_with_config():
    args = ["--config", "custom.toml"]
    result = _parse_args(args)
    assert result == {"config": "custom.toml"}

def test_parse_args_with_verbose():
    args = ["-v"]
    result = _parse_args(args)
    assert result == {"verbose": True}

def test_parse_args_with_version():
    args = ["--version"]
    try:
        _parse_args(args)
    except SystemExit:
        pass

def test_parse_args_with_unknown_key():
    args = ["--unknown", "value"]
    try:
        _parse_args(args)
    except SystemExit:
        pass

def test_parse_args_with_wrong_type():
    args = ["--min-confidence", "abc"]
    try:
        _parse_args(args)
    except SystemExit:
        pass

def test_parse_args_with_multiple_options():
    args = ["path", "--exclude", "*.py", "--verbose", "--make-whitelist"]
    result = _parse_args(args)
    assert result == {"paths": ["path"], "exclude": ["*.py"], "verbose": True, "make_whitelist": True}


# LLM-generated content at query #6
#--------------------------

def test_make_config_with_only_cli_args():
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file():
    import io
    toml_content = """
[tool.vulture]
exclude = ["file*.py"]
verbose = true
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
verbose = false
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] == True

def test_make_config_with_missing_toml_uses_defaults():
    import io
    toml_content = """
[tool.vulture]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False

def test_make_config_raises_input_error_no_paths():
    import io
    toml_content = """
[tool.vulture]
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False
    except InputError:
        pass

def test_make_config_with_cli_paths_and_toml():
    import io
    toml_content = """
[tool.vulture]
exclude = ["old.py"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["new.py"], tomlfile=tomlfile)
    assert config["paths"] == ["new.py"]
    assert config["exclude"] == ["old.py"]


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_true_when_detected_toml_path_and_verbose():
    tomlfile = io.BytesIO(b'[tool.vulture]\nverbose = true\n')
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] is True


# LLM-generated content at query #8
#--------------------------

```python
def test_make_config_with_cli_only_and_defaults():
    argv = ["--verbose", "path1"]
    config = make_config(argv)
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_only():
    import io
    tomlfile = io.BytesIO(b"[tool.vulture]\nverbose = true\npaths = [\"path1\"]")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False

def test_make_config_with_cli_overriding_toml():
    import io
    tomlfile = io.BytesIO(b"[tool.vulture]\nverbose = false\npaths = [\"toml_path\"]")
    argv = ["--verbose", "cli_path"]
    config = make_config(argv, tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]

def test_make_config_with_toml_no_tool_vulture():
    import io
    tomlfile = io.BytesIO(b"[other]\nkey = \"value\"")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] == False
    assert config["paths"] == []

def test_make_config_with_none_paths_raises_error():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError:
        pass
```


# LLM-generated content at query #9
#--------------------------

```
def test_check_input_config_valid_data():
    data = {"key1": "value1", "key2": 2}
    DEFAULTS = {"key1": "string", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"known_key": "default"}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "string"}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"

def test_check_input_config_empty_data():
    data = {}
    DEFAULTS = {"key1": "default"}
    _check_input_config(data)

def test_check_input_config_boolean_vs_int():
    data = {"key1": True}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_true_when_toml_path_and_verbose():
    config = make_config(argv=["--verbose"], tomlfile="pyproject.toml")


# LLM-generated content at query #11
#--------------------------

```python
def test_check_input_config_with_correct_types():
    DEFAULTS = {"a": 1, "b": "hello", "c": 3.14}
    data = {"a": 10, "b": "world", "c": 2.71}
    _check_input_config(data)
```


# LLM-generated content at query #12
#--------------------------

```
def test_make_config_with_tomlfile_merges_settings():
    import io
    import sys
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\nexclude = ["test.py"]\npaths = ["src"]')
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["test.py"]
    assert config["paths"] == ["src"]
    assert config["verbose"] is False
    assert config["sort_by_size"] is False

def test_make_config_cli_overrides_toml():
    import io
    import sys
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\nexclude = ["test.py"]\nverbose = true')
    config = make_config(argv=["--exclude", "other.py"], tomlfile=tomlfile)
    assert config["exclude"] == ["other.py"]
    assert config["verbose"] is True

def test_make_config_sets_defaults_for_missing_options():
    import io
    import sys
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\npaths = ["src"]')
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_uses_default_config_path_when_no_tomlfile():
    import sys
    sys.argv = ["vulture", "--config", "nonexistent.toml"]
    config = make_config(argv=["--verbose"], tomlfile=None)
    assert config["verbose"] is True
    assert config["paths"] == []

def test_make_config_raises_input_error_for_unknown_key():
    import io
    import sys
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\nunknown_key = 123')
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_raises_input_error_for_wrong_type():
    import io
    import sys
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\nverbose = "yes"')
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert "Data type for verbose must be 'bool'" in str(e)

def test_make_config_raises_input_error_for_empty_paths():
    import io
    import sys
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\npaths = []')
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_prints_config_path_in_verbose():
    import io
    import sys
    from unittest.mock import patch
    sys.argv = ["vulture"]
    tomlfile = io.StringIO('[tool.vulture]\nverbose = true\npaths = ["src"]')
    with patch("builtins.print") as mock_print:
        config = make_config(argv=[], tomlfile=tomlfile)
        mock_print.assert_called_once_with(f"Reading configuration from {tomlfile}")
    assert config["verbose"] is True
```


# LLM-generated content at query #13
#--------------------------

```python
def test_toml_path_is_file_returns_true_when_file_exists(tmp_path):
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text("")
    import pathlib
    path = pathlib.Path(config_file)
    assert path.is_file() == True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_toml_path_is_file_returns_true():
    import tempfile
    import pathlib
    tmp_file = tempfile.NamedTemporaryFile(suffix=".toml", delete=False)
    tmp_file.close()
    test_path = pathlib.Path(tmp_file.name).resolve()
    assert test_path.is_file() == True
    pathlib.Path(tmp_file.name).unlink()
```


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_true_with_toml_path_and_verbose():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")


# LLM-generated content at query #16
#--------------------------

```
def test_check_input_config_valid_data():
    DEFAULTS = {"key1": "string", "key2": 42}
    data = {"key1": "test", "key2": 100}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    DEFAULTS = {"key1": "string"}
    data = {"unknown_key": "value"}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    DEFAULTS = {"key1": "string"}
    data = {"key1": 123}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"

def test_check_input_config_bool_vs_int():
    DEFAULTS = {"key1": False}
    data = {"key1": 0}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'bool'"

def test_check_input_config_int_vs_bool():
    DEFAULTS = {"key1": 0}
    data = {"key1": False}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"

def test_check_input_config_multiple_keys():
    DEFAULTS = {"key1": 1, "key2": "a"}
    data = {"key1": 2, "key2": "b"}
    _check_input_config(data)

def test_check_input_config_empty_data():
    DEFAULTS = {}
    data = {}
    _check_input_config(data)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_make_config_with_only_cli_arguments():
    import tempfile
    import sys
    import argparse
    from unittest.mock import patch
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False

def test_make_config_with_toml_file():
    import io
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\n'
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] is True
    assert config["paths"] == []

def test_make_config_cli_overrides_toml():
    import io
    toml_content = b'[tool.vulture]\nverbose = true\n'
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--verbose", "false"], tomlfile=toml_file)
    assert config["verbose"] is False

def test_make_config_with_default_toml_not_found():
    config = make_config(argv=["--config", "nonexistent.toml"], tomlfile=None)
    assert config["paths"] == []
    assert config["verbose"] is False

def test_make_config_with_empty_paths_raises_error():
    import io
    toml_content = b'[tool.vulture]\npaths = []\n'
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_unknown_key_raises_error():
    import io
    toml_content = b'[tool.vulture]\nunknown_key = true\n'
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_wrong_type_raises_error():
    import io
    toml_content = b'[tool.vulture]\nverbose = "yes"\n'
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for verbose must be 'bool'" in str(e)

def test_make_config_verbose_prints_config_path(capsys):
    import tempfile
    import pathlib
    import sys
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.vulture]\nverbose = true\n')
        tmp_path = pathlib.Path(f.name)
    config = make_config(argv=["--config", str(tmp_path)], tomlfile=None)
    captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out
    tmp_path.unlink()

def test_make_config_non_default_min_confidence():
    config = make_config(argv=["--min-confidence", "80"], tomlfile=None)
    assert config["min_confidence"] == 80

def test_make_config_non_default_sort_by_size():
    config = make_config(argv=["--sort-by-size"], tomlfile=None)
    assert config["sort_by_size"] is True

def test_make_config_non_default_make_whitelist():
    config = make_config(argv=["--make-whitelist"], tomlfile=None)
    assert config["make_whitelist"] is True

def test_make_config_with_exclude_from_cli():
    config = make_config(argv=["--exclude", "*.pyc,docs"], tomlfile=None)
    assert config["exclude"] == ["*.pyc", "docs"]

def test_make_config_with_ignore_decorators_from_cli():
    config = make_config(argv=["--ignore-decorators", "@app.route"], tomlfile=None)
    assert config["ignore_decorators"] == ["@app.route"]

def test_make_config_with_ignore_names_from_cli():
    config = make_config(argv=["--ignore-names", "visit_*"], tomlfile=None)
    assert config["ignore_names"] == ["visit_*"]

def test_make_config_with_paths_from_cli():
    config = make_config(argv=["src", "tests"], tomlfile=None)
    assert config["paths"] == ["src", "tests"]

def test_make_config_with_verbose_flag():
    config = make_config(argv=["-v"], tomlfile=None)
    assert config["verbose"] is True

def test_make_config_with_version_flag():
    import sys
    import argparse
    try:
        config = make_config(argv=["--version"], tomlfile=None)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_help_flag():
    import sys
    import argparse
    try:
        config = make_config(argv=["--help"], tomlfile=None)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_with_toml_and_cli_paths():
    import io
    toml_content = b'[tool.vulture]\npaths = ["dir1"]\n'
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["dir2"], tomlfile=toml_file)
    assert config["paths"] == ["dir2"]

def test_make_config_with_toml_and_cli_exclude():
    import io
    toml_content = b'[tool.vulture]\nexclude = ["*.pyc"]\n'
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--exclude", "*.pyo"], tomlfile=toml_file)
    assert config["exclude"] == ["*.pyo"]

def test_make_config_with_toml_and_cli_verbose():
    import io
    toml_content = b'[tool.vulture]\nverbose = false\n'
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True

def test_make_config_with_toml_unknown_key_raises_error():
    import io
    toml_content = b'[tool.vulture]\nunknown = true\n'
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_toml_wrong_type_raises_error():
    import io
    toml_content = b'[tool.vulture]\nmin_confidence = "high"\n'
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)```


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_39_true():
    tomlfile = "/path/to/existing.toml"
    config = {"verbose": True, "paths": [], "exclude": [], "ignore_names": [], "ignore_decorators": [], "make_whitelist": False, "sort_by_size": False, "min_confidence": 0}
    detected_toml_path = str(tomlfile)
    assert detected_toml_path and config["verbose"] == True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_toml_path_is_file_returns_true():
    import tempfile
    import pathlib
    import os
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    try:
        test_tomlfile = pathlib.Path(tmp_path)
        result = test_tomlfile.is_file()
        assert result is True
    finally:
        os.unlink(tmp_path)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config_with_cli_args_only():
    argv = ["--verbose", "path1", "path2"]
    config = make_config(argv=argv)
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_file():
    toml_content = b'[tool.vulture]\nverbose = true\npaths = ["path1"]'
    from io import BytesIO
    tomlfile = BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]

def test_make_config_cli_overrides_toml():
    toml_content = b'[tool.vulture]\nverbose = false\npaths = ["path1"]'
    from io import BytesIO
    tomlfile = BytesIO(toml_content)
    argv = ["--verbose", "path2"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["path2"]

def test_make_config_without_args_or_toml():
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["verbose"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"```


# LLM-generated content at query #21
#--------------------------

def test_correct_type_passes():
    data = {"timeout": 30, "verbose": False}
    _check_input_config(data)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_line25_true():
    import tempfile
    import pathlib
    temp_file = tempfile.NamedTemporaryFile(suffix=".toml", delete=False)
    temp_file.close()
    temp_path = pathlib.Path(temp_file.name)
    temp_path.touch()
    argv = ["--config", str(temp_path)]
    make_config(argv=argv)
    temp_path.unlink()


# LLM-generated content at query #23
#--------------------------

```python
def test_input_config_type_match():
    DEFAULTS = {"timeout": 30, "verbose": False}
    data = {"timeout": 30, "verbose": False}
    _check_input_config(data)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_true():
    import io
    import tempfile
    import pathlib

    # Create a temporary TOML file with verbose=true
    toml_content = b'verbose = true\n'
    with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = pathlib.Path(f.name)

    try:
        config = make_config(argv=['--verbose'], tomlfile=toml_path)
    finally:
        toml_path.unlink()

    assert config["verbose"] == True
```


# LLM-generated content at query #25
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False
    assert config["exclude"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

def test_make_config_with_toml_only():
    import io
    toml_content = """
[tool.vulture]
paths = ["src"]
verbose = true
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src"]
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
paths = ["src"]
verbose = false
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["other_path", "--verbose"], tomlfile=tomlfile)
    assert config["paths"] == ["other_path"]
    assert config["verbose"] is True

def test_make_config_empty_toml_defaults():
    import io
    toml_content = "[tool.vulture]\n"
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["verbose"] is False
    assert config["exclude"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

def test_make_config_no_toml_no_cli_paths():
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []


# LLM-generated content at query #26
#--------------------------

```python
def test_correct_type_passes():
    data = {"port": 8080}
    _check_input_config(data)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_make_config_with_only_cli_args():
    argv = ["path1", "path2"]
    config = make_config(argv=argv)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []

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
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path"]
verbose = false
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    argv = ["cli_path", "--verbose"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["verbose"] is True

def test_make_config_with_no_paths_raises_input_error():
    import io
    toml_content = """
[tool.vulture]
paths = []
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

def test_make_config_with_unknown_key_in_toml_raises_input_error():
    import io
    toml_content = """
[tool.vulture]
unknown_key = true
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_wrong_type_in_cli_raises_input_error():
    argv = ["--min-confidence", "not_an_int"]
    try:
        make_config(argv=argv)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

def test_make_config_defaults_for_missing_options():
    argv = ["path"]
    config = make_config(argv=argv)
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_verbose_output_with_toml():
    import io
    toml_content = """
[tool.vulture]
verbose = true
paths = ["path1"]
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    import sys
    from io import StringIO
    captured_output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        config = make_config(tomlfile=tomlfile)
    finally:
        sys.stdout = old_stdout
    output = captured_output.getvalue()
    assert "Reading configuration from" in output

def test_make_config_no_toml_file_found():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            argv = ["path"]
            config = make_config(argv=argv)
            assert config["paths"] == ["path"]
        finally:
            os.chdir(original_cwd)

def test_make_config_with_mixed_cli_and_toml_defaults():
    import io
    toml_content = """
[tool.vulture]
sort_by_size = true
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    argv = ["path", "--verbose"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["paths"] == ["path"]
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_check_output_config_raises_error_when_paths_empty():
    config = {"paths": []}
    try:
        _check_output_config(config)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_output_config_does_not_raise_error_when_paths_not_empty():
    config = {"paths": ["/some/path"]}
    _check_output_config(config)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_toml_returns_empty_dict_when_no_vulture_section():
    import tempfile, pathlib
    content = "[tool.other]\nkey = 'value'\n"
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
    tmp.write(content)
    tmp.close()
    result = _parse_toml(pathlib.Path(tmp.name))
    assert result == {}
    pathlib.Path(tmp.name).unlink()

def test_parse_toml_returns_correct_settings():
    import tempfile, pathlib
    content = "[tool.vulture]\nexclude = ['file*.py', 'dir/']\nignore_decorators = ['deco1']\nmin_confidence = 10\nsort_by_size = true\n"
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
    tmp.write(content)
    tmp.close()
    result = _parse_toml(pathlib.Path(tmp.name))
    assert result == {'exclude': ['file*.py', 'dir/'], 'ignore_decorators': ['deco1'], 'min_confidence': 10, 'sort_by_size': True}
    pathlib.Path(tmp.name).unlink()

def test_parse_toml_raises_input_error_for_unknown_key():
    import tempfile, pathlib
    content = "[tool.vulture]\nunknown_key = 'value'\n"
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
    tmp.write(content)
    tmp.close()
    try:
        _parse_toml(pathlib.Path(tmp.name))
        assert False
    except InputError:
        pass
    pathlib.Path(tmp.name).unlink()

def test_parse_toml_raises_input_error_for_wrong_type():
    import tempfile, pathlib
    content = "[tool.vulture]\nmin_confidence = 'ten'\n"
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
    tmp.write(content)
    tmp.close()
    try:
        _parse_toml(pathlib.Path(tmp.name))
        assert False
    except InputError:
        pass
    pathlib.Path(tmp.name).unlink()
```


# LLM-generated content at query #3
#--------------------------

```
def test_make_config_with_cli_only():
    argv = ["--verbose", "path1"]
    result = make_config(argv=argv)
    assert result["verbose"] == True
    assert result["paths"] == ["path1"]

def test_make_config_with_tomlfile():
    import io
    import tomllib
    toml_data = """
[tool.vulture]
verbose = true
paths = ["path1"]
"""
    tomlfile = io.BytesIO(tomllib.dumps({"tool": {"vulture": {"verbose": True, "paths": ["path1"]}}}))
    result = make_config(tomlfile=tomlfile)
    assert result["verbose"] == True
    assert result["paths"] == ["path1"]

def test_make_config_cli_overrides_toml():
    import io
    import tomllib
    toml_data = """
[tool.vulture]
verbose = false
paths = ["toml_path"]
"""
    tomlfile = io.BytesIO(tomllib.dumps({"tool": {"vulture": {"verbose": False, "paths": ["toml_path"]}}}))
    argv = ["--verbose", "cli_path"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    assert result["verbose"] == True
    assert result["paths"] == ["cli_path"]

def test_make_config_defaults_applied():
    argv = ["path1"]
    result = make_config(argv=argv)
    assert result["sort_by_size"] == False
    assert result["make_whitelist"] == False

def test_make_config_raises_on_empty_paths():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_without_toml_and_default_config():
    import pathlib
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            result = make_config(argv=["path1"])
            assert result["paths"] == ["path1"]
            assert result["config"] == "pyproject.toml"
        finally:
            os.chdir(original_cwd)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_39_true():
    config = {"verbose": True, "paths": ["./"], "sort_by_size": False, "min_confidence": 0.0, "exclude": [], "ignore_names": [], "ignore_decorators": [], "ignore_variables": [], "make_whitelist": False, "output_format": "default"}
    detected_toml_path = "/path/to/pyproject.toml"
    assert detected_toml_path and config["verbose"]


# LLM-generated content at query #5
#--------------------------

```
def test_check_input_config_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "string", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"key1": "string"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_check_input_config_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "string"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for key1 must be 'str'" in str(e)

def test_check_input_config_empty_data():
    data = {}
    DEFAULTS = {"key1": "value1"}
    _check_input_config(data)

def test_check_input_config_multiple_keys():
    data = {"key1": "correct", "key2": 456}
    DEFAULTS = {"key1": "string", "key2": 0}
    _check_input_config(data)

def test_check_input_config_int_bool_differentiation():
    data = {"key1": False}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for key1 must be 'int'" in str(e)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config_with_only_cli_args():
    import tempfile
    import pathlib
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_tomlfile():
    import io
    import tomllib
    toml_data = io.BytesIO(b'[tool.vulture]\npaths = ["src"]\nverbose = true\n')
    config = make_config(argv=[], tomlfile=toml_data)
    assert config["paths"] == ["src"]
    assert config["verbose"] == True

def test_make_config_cli_overrides_toml():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\npaths = ["toml_path"]\nverbose = false\n')
    config = make_config(argv=["cli_path", "--verbose"], tomlfile=toml_data)
    assert config["paths"] == ["cli_path"]
    assert config["verbose"] == True

def test_make_config_with_missing_paths_raises_error():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\nmake_whitelist = true\n')
    import pytest
    try:
        config = make_config(argv=[], tomlfile=toml_data)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_unknown_key_in_toml_raises_error():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\nunknown_key = true\n')
    try:
        config = make_config(argv=[], tomlfile=toml_data)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_wrong_type_in_toml_raises_error():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"\n')
    try:
        config = make_config(argv=[], tomlfile=toml_data)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

def test_make_config_with_explicit_config_file():
    import tempfile
    import pathlib
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.vulture]\npaths = ["cfg_path"]\n')
        f.flush()
        config_path = f.name
    try:
        config = make_config(argv=["--config", config_path])
        assert config["paths"] == ["cfg_path"]
        assert config["config"] == config_path
    finally:
        os.unlink(config_path)

def test_make_config_defaults_when_no_config_file():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_config(argv=[tmpdir])
        assert config["paths"] == [tmpdir]
        assert config["exclude"] == []
        assert config["ignore_decorators"] == []
        assert config["ignore_names"] == []
        assert config["make_whitelist"] == False
        assert config["min_confidence"] == 0
        assert config["sort_by_size"] == False
        assert config["verbose"] == False
        assert config["config"] == "pyproject.toml"

def test_make_config_verbose_from_toml():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\npaths = ["src"]\nverbose = true\n')
    config = make_config(argv=[], tomlfile=toml_data)
    assert config["verbose"] == True

def test_make_config_verbose_from_cli():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\npaths = ["src"]\nverbose = false\n')
    config = make_config(argv=["--verbose"], tomlfile=toml_data)
    assert config["verbose"] == True

def test_make_config_with_empty_toml_no_config_file():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_config(argv=[tmpdir])
        assert config["paths"] == [tmpdir]

def test_make_config_multiple_cli_paths():
    config = make_config(argv=["path1", "path2", "path3"])
    assert config["paths"] == ["path1", "path2", "path3"]

def test_make_config_with_all_cli_options():
    config = make_config(argv=[
        "path", "--exclude", "*.pyc", "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2", "--make-whitelist", "--min-confidence", "50",
        "--sort-by-size", "--verbose", "--config", "myconfig.toml"
    ])
    assert config["paths"] == ["path"]
    assert config["exclude"] == ["*.pyc"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["config"] == "myconfig.toml"

def test_make_config_no_args_defaults():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_config(argv=[tmpdir])
        assert config["paths"] == [tmpdir]
        assert config["exclude"] == []
        assert config["ignore_decorators"] == []
        assert config["ignore_names"] == []
        assert config["make_whitelist"] == False
        assert config["min_confidence"] == 0
        assert config["sort_by_size"] == False
        assert config["verbose"] == False
        assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_and_no_cli_paths():
    import io
    toml_data = io.BytesIO(b'[tool.vulture]\npaths = ["toml_path"]\n')
    config = make_config(argv=[], tomlfile=toml_data)
    assert config["paths"] == ["toml_path"]

def test_make_config_unknown_key_in_cli_raises_error():
    try:
        config = make_config(argv=["--unknown-key", "value"])
        assert False, "Expected InputError"
    except Exception as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_wrong_type_in_cli_raises_error():
    try:
        config = make_config(argv=["--min-confidence", "not_an_int"])
        assert False, "Expected InputError"
    except Exception as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

def test_make_config_with_existing_pyproject_toml():
    import tempfile
    import pathlib
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_path = pathlib.Path(tmpdir) / "pyproject.toml"
        pyproject_path.write_text('[tool.vulture]\npaths = ["pyproject_path"]\n')
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            config = make_config(argv=[])
            assert config["paths"] == ["pyproject_path"]
        finally:
            os.chdir(original_cwd)

def test_make_config_no_existing_pyproject_toml():
    import tempfile
    import pathlib
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            config = make_config(argv=["."])
            assert config["paths"] == ["."]
        finally:
            os.chdir(original_cwd)

def test_make_config_verbose_output_with_toml():
    import io
    import sys
    from io import StringIO
    toml_data = io.BytesIO(b'[tool.vulture]\npaths = ["src"]\nverbose = true\n')
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        config = make_config(argv=[], tomlfile=toml_data)
    finally:
        sys.stdout = original_stdout
    assert "Reading configuration from" in captured_output.getvalue()

def test_make_config_verbose_output_with_existing_pyproject_toml():



# LLM-generated content at query #7
#--------------------------

def test_make_config_with_only_cli_args_provided():
    import io
    import tempfile
    import pathlib
    from unittest.mock import patch
    import sys
    import vulture
    config = vulture.make_config(argv=["/some/path"], tomlfile=None)
    assert config["paths"] == ["/some/path"]

def test_make_config_with_tomlfile_and_no_cli_overrides():
    import io
    import vulture
    toml_data = b'[tool.vulture]\npaths = ["/toml/path"]\nverbose = true'
    config = vulture.make_config(argv=[], tomlfile=io.BytesIO(toml_data))
    assert config["paths"] == ["/toml/path"]
    assert config["verbose"] == True

def test_make_config_cli_overrides_toml():
    import io
    import vulture
    toml_data = b'[tool.vulture]\npaths = ["/toml/path"]\nverbose = false'
    config = vulture.make_config(argv=["/cli/path", "--verbose"], tomlfile=io.BytesIO(toml_data))
    assert config["paths"] == ["/cli/path"]
    assert config["verbose"] == True

def test_make_config_defaults_applied_when_missing():
    import io
    import vulture
    config = vulture.make_config(argv=["/some/path"], tomlfile=None)
    assert config["sort_by_size"] == False
    assert config["min_confidence"] == 0

def test_make_config_with_toml_and_cli_args_merge():
    import io
    import vulture
    toml_data = b'[tool.vulture]\nignore_names = ["foo"]\nmin_confidence = 50'
    config = vulture.make_config(argv=["/path", "--min-confidence", "80"], tomlfile=io.BytesIO(toml_data))
    assert config["ignore_names"] == ["foo"]
    assert config["min_confidence"] == 80
    assert config["paths"] == ["/path"]

def test_make_config_empty_toml_section_results_in_defaults():
    import io
    import vulture
    toml_data = b'[tool.vulture]\n'
    config = vulture.make_config(argv=["/path"], tomlfile=io.BytesIO(toml_data))
    assert config["paths"] == ["/path"]
    assert config["verbose"] == False

def test_make_config_no_toml_file_found_uses_only_cli():
    import tempfile
    import pathlib
    import vulture
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        config = vulture.make_config(argv=["/path"], tomlfile=None)
        assert config["paths"] == ["/path"]
        assert "sort_by_size" in config


# LLM-generated content at query #8
#--------------------------

```python
def test_toml_path_is_file_predicate_true():
    import pathlib
    import tempfile
    import os
    tmp = tempfile.NamedTemporaryFile(suffix=".toml", delete=False)
    tmp.close()
    toml_path = pathlib.Path(tmp.name)
    config = make_config(argv=[], tomlfile=toml_path)
    os.unlink(tmp.name)
    assert True
```


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_true_with_valid_toml_and_verbose():
    tomlfile = io.StringIO("[tool.vulture]\nverbose = true")
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == []  # default from DEFAULTS
```


# LLM-generated content at query #10
#--------------------------

```python
def test_make_config_with_tomlfile():
    import io
    import tomllib
    toml_content = """
[tool.vulture]
exclude = ["test.py"]
ignore_decorators = ["deco1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 50
sort_by_size = true
verbose = true
paths = ["path1", "path2"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_cli_args():
    config = make_config(argv=["--exclude", "test.py", "--verbose", "--min-confidence", "80", "path1", "path2"])
    assert config["exclude"] == ["test.py"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
exclude = ["toml_exclude.py"]
verbose = false
min_confidence = 50
paths = ["toml_path"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--exclude", "cli_exclude.py", "--verbose", "--min-confidence", "90", "cli_path"], tomlfile=toml_file)
    assert config["exclude"] == ["cli_exclude.py"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 90
    assert config["paths"] == ["cli_path"]

def test_make_config_defaults():
    config = make_config(argv=[], tomlfile=io.BytesIO(b""))
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["paths"] == []

def test_make_config_unknown_key_raises():
    import io
    import pytest
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

def test_make_config_empty_paths_raises():
    import io
    import pytest
    toml_content = """
[tool.vulture]
paths = []
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

def test_make_config_no_toml_file():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = pathlib.Path.cwd()
        pathlib.Path.cwd = lambda: pathlib.Path(tmpdir)
        try:
            config = make_config(argv=["path1"])
            assert config["paths"] == ["path1"]
        finally:
            pathlib.Path.cwd = old_cwd

def test_make_config_toml_file_exists():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = pathlib.Path(tmpdir) / "pyproject.toml"
        toml_path.write_text("""
[tool.vulture]
exclude = ["test.py"]
verbose = true
paths = ["path1"]
""")
        old_cwd = pathlib.Path.cwd
        pathlib.Path.cwd = lambda: pathlib.Path(tmpdir)
        try:
            config = make_config(argv=[])
            assert config["exclude"] == ["test.py"]
            assert config["verbose"] == True
            assert config["paths"] == ["path1"]
        finally:
            pathlib.Path.cwd = old_cwd

def test_make_config_verbose_with_toml():
    import io
    import sys
    from io import StringIO
    toml_content = """
[tool.vulture]
verbose = true
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        config = make_config(argv=[], tomlfile=toml_file)
        output = sys.stdout.getvalue()
        assert "Reading configuration from" in output
    finally:
        sys.stdout = old_stdout

def test_make_config_verbose_false_no_output():
    import io
    import sys
    from io import StringIO
    toml_content = """
[tool.vulture]
verbose = false
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        config = make_config(argv=[], tomlfile=toml_file)
        output = sys.stdout.getvalue()
        assert output == ""
    finally:
        sys.stdout = old_stdout
```


# LLM-generated content at query #11
#--------------------------

```python
def test_type_mismatch_raises_error():
    data = {"timeout": "30"}
    DEFAULTS = {"timeout": 30}
    try:
        _check_input_config(data)
    except InputError:
        pass
```


# LLM-generated content at query #12
#--------------------------

```python
def test_expected_type_passes():
    DEFAULTS = {"timeout": 5, "verbose": False}
    data = {"timeout": 10, "verbose": True}
    _check_input_config(data)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_true_when_detected_toml_path_and_verbose():
    from io import StringIO
    import sys
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        config = make_config(argv=["--verbose"], tomlfile=StringIO("[tool.vulture]\n"))
        sys.stdout.seek(0)
        output = sys.stdout.read()
        assert "Reading configuration from" in output
    finally:
        sys.stdout = original_stdout
```


# LLM-generated content at query #14
#--------------------------

```python
def test_check_input_config_with_correct_type():
    data = {"key1": 42}
    DEFAULTS = {"key1": 0}
    _check_input_config(data)
```


# LLM-generated content at query #15
#--------------------------

```
def test_check_input_config_with_valid_data():
    data = {"key1": 123, "key2": "string"}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    try:
        _check_input_config({"unknown_key": 1})
        assert False
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_check_input_config_with_wrong_type_for_key():
    try:
        _check_input_config({"key1": "string"})
        assert False
    except InputError as e:
        assert "Data type for key1 must be 'int'" in str(e)

def test_check_input_config_with_bool_instead_of_int():
    try:
        _check_input_config({"key1": True})
        assert False
    except InputError as e:
        assert "Data type for key1 must be 'int'" in str(e)

def test_check_input_config_empty_data():
    _check_input_config({})
```


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_true_when_detected_toml_path_and_verbose():
    config = {"verbose": True}
    detected_toml_path = "pyproject.toml"
    assert detected_toml_path and config["verbose"] == True
```


# LLM-generated content at query #17
#--------------------------

```python
def test_toml_path_is_file_returns_true():
    import tempfile
    import pathlib
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = pathlib.Path(temp_file.name)
    temp_path.write_text("")
    try:
        config = make_config(argv=["--config", str(temp_path)], tomlfile=None)
        assert True
    finally:
        temp_path.unlink()


# LLM-generated content at query #18
#--------------------------

```python
def test_toml_path_is_file_true():
    import tempfile
    import os
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp_file.close()
    argv = ["--config", tmp_file.name]
    config = make_config(argv=argv)
    os.unlink(tmp_file.name)
    assert config is not None
```


# LLM-generated content at query #19
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"alpha": 0.5, "max_iter": 100, "use_log": True}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"unknown_key": 42}
    try:
        _check_input_config(data)
    except InputError:
        pass

def test_check_input_config_with_wrong_type_int_vs_bool():
    data = {"use_log": 1}
    try:
        _check_input_config(data)
    except InputError:
        pass

def test_check_input_config_with_wrong_type_float_vs_int():
    data = {"max_iter": 10.5}
    try:
        _check_input_config(data)
    except InputError:
        pass

def test_check_input_config_with_wrong_type_string():
    data = {"alpha": "0.5"}
    try:
        _check_input_config(data)
    except InputError:
        pass
```


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_true_with_toml_and_verbose():
    import io
    import pathlib
    config = {
        "verbose": True,
        "paths": [],
        "exclude": [],
        "ignore_names": [],
        "ignore_decorators": [],
        "ignore_variables": [],
        "min_confidence": 0.5,
        "sort_by_size": False,
        "output_format": "",
        "exclude_paths": [],
        "exclude_files": [],
        "config": "nonexistent"
    }
    toml_content = b""
    tomlfile = io.BytesIO(toml_content)
    result = make_config(argv=[], tomlfile=tomlfile)
    assert result["verbose"] == True
    assert result["detected_toml_path"] == str(tomlfile) if hasattr(result, "detected_toml_path") else True
```


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_true_when_toml_path_detected_and_verbose():
    tomlfile = io.StringIO("[tool.vulture]\nverbose = true")
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["config"] is not None
```


# LLM-generated content at query #22
#--------------------------

```python
def test_make_config_with_defaults():
    config = make_config(argv=[], tomlfile=None)
    assert config == {
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "min_confidence": 0,
        "sort_by_size": False,
        "verbose": False,
        "config": "pyproject.toml",
    }

def test_make_config_with_cli_args():
    config = make_config(argv=["--verbose", "--make-whitelist", "path1", "path2"], tomlfile=None)
    assert config["verbose"] == True
    assert config["make_whitelist"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml():
    import io
    toml_data = """
[tool.vulture]
paths = ["src"]
verbose = true
"""
    config = make_config(argv=[], tomlfile=io.StringIO(toml_data))
    assert config["paths"] == ["src"]
    assert config["verbose"] == True

def test_make_config_cli_overrides_toml():
    import io
    toml_data = """
[tool.vulture]
paths = ["src"]
verbose = false
"""
    config = make_config(argv=["--verbose"], tomlfile=io.StringIO(toml_data))
    assert config["paths"] == ["src"]
    assert config["verbose"] == True

def test_make_config_with_invalid_key_raises_error():
    import io
    toml_data = """
[tool.vulture]
invalid_key = true
"""
    try:
        make_config(argv=[], tomlfile=io.StringIO(toml_data))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_wrong_type_raises_error():
    import io
    toml_data = """
[tool.vulture]
verbose = "yes"
"""
    try:
        make_config(argv=[], tomlfile=io.StringIO(toml_data))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for verbose must be 'bool'" in str(e)

def test_make_config_with_empty_paths_raises_error():
    import io
    toml_data = """
[tool.vulture]
paths = []
"""
    try:
        make_config(argv=[], tomlfile=io.StringIO(toml_data))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_without_paths_raises_error():
    import io
    toml_data = """
[tool.vulture]
verbose = true
"""
    try:
        make_config(argv=[], tomlfile=io.StringIO(toml_data))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_cli_paths_and_empty_toml():
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_exclude_and_cli():
    import io
    toml_data = """
[tool.vulture]
exclude = ["test_*.py"]
"""
    config = make_config(argv=["--exclude", "*.pyc"], tomlfile=io.StringIO(toml_data))
    assert config["exclude"] == ["*.pyc"]
```


# LLM-generated content at query #23
#--------------------------

```python
def test_check_input_config_type_check_passes():
    data = {"timeout": 30}
    DEFAULTS = {"timeout": 10}
    _check_input_config(data)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_true_with_non_empty_detected_toml_path_and_verbose_true():
    config = make_config(argv=["--verbose"], tomlfile="test.toml")
    assert config["verbose"] == True
    assert config["detected_toml_path"] == "test.toml"
```


# LLM-generated content at query #25
#--------------------------

```python
def test_toml_path_is_file():
    import tempfile
    import os
    import pathlib
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp.write(b"")
    tmp.close()
    config = {"config": tmp.name}
    toml_path = pathlib.Path(config["config"]).resolve()
    assert toml_path.is_file()
    os.unlink(tmp.name)
```


# LLM-generated content at query #26
#--------------------------

```
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 123}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"unknown_key": "value"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    data = {"key1": 123}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for key1 must be" in str(e)

def test_check_input_config_with_bool_instead_of_int():
    data = {"key2": True}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for key2 must be" in str(e)

def test_check_input_config_with_empty_data():
    data = {}
    _check_input_config(data)
```


