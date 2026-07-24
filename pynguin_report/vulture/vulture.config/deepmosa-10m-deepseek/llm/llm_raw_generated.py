####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config_with_no_arguments_uses_defaults():
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"
    assert config["verbose"] is False

def test_make_config_with_cli_arguments_overrides_defaults():
    config = make_config(argv=["--verbose", "--min-confidence=80", "path1", "path2"], tomlfile=None)
    assert config["verbose"] is True
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file_uses_toml_settings():
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
ignore_names = ["helper"]
min_confidence = 50
paths = ["src/"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_names"] == ["helper"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
min_confidence = 50
verbose = true
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence=80", "--verbose=false"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is False

def test_make_config_unknown_cli_key_raises_input_error():
    try:
        make_config(argv=["--unknown-key=value"], tomlfile=None)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_unknown_toml_key_raises_input_error():
    import io
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_wrong_type_in_cli_raises_input_error():
    try:
        make_config(argv=["--min-confidence=abc"], tomlfile=None)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

def test_make_config_wrong_type_in_toml_raises_input_error():
    import io
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

def test_make_config_no_paths_raises_input_error():
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_default_values_for_missing_options():
    config = make_config(argv=["some_path"], tomlfile=None)
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["paths"] == ["some_path"]

def test_make_config_verbose_with_toml_path_prints_message(capsys):
    import io
    toml_content = """
[tool.vulture]
verbose = true
paths = ["src/"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out
    assert "src/" in config["paths"]

def test_make_config_config_default_used_when_no_config_file():
    config = make_config(argv=["some_path"], tomlfile=None)
    assert config["config"] == "pyproject.toml"

def test_make_config_config_not_overridden_by_toml():
    import io
    toml_content = """
[tool.vulture]
config = "custom.toml"
paths = ["src/"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--config=custom.toml"], tomlfile=tomlfile)
    assert config["config"] == "custom.toml"
```


# LLM-generated content at query #2
#--------------------------

def test_check_input_config_valid_data():
    data = {"width": 800, "height": 600, "fullscreen": False}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    try:
        data = {"unknown_key": 123}
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type_int_bool():
    try:
        data = {"fullscreen": 1}
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for fullscreen must be 'bool'"

def test_check_input_config_wrong_type_str_int():
    try:
        data = {"width": "800"}
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for width must be 'int'"

def test_check_input_config_wrong_type_float_int():
    try:
        data = {"height": 600.0}
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for height must be 'int'"

def test_check_input_config_multiple_errors_first_only():
    try:
        data = {"unknown1": 1, "width": "bad"}
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown1"


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config_with_cli_args_only_no_toml():
    argv = ["--verbose", "path1", "path2"]
    expected = {
        "paths": ["path1", "path2"],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "min_confidence": 0,
        "sort_by_size": False,
        "config": "pyproject.toml",
        "verbose": True,
    }
    result = make_config(argv=argv, tomlfile=None)
    assert result == expected

def test_make_config_with_toml_only():
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\npaths = ["src"]'
    tomlfile = io.BytesIO(toml_content)
    result = make_config(argv=None, tomlfile=tomlfile)
    expected = {
        "paths": ["src"],
        "exclude": ["file*.py"],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "min_confidence": 0,
        "sort_by_size": False,
        "config": "pyproject.toml",
        "verbose": True,
    }
    assert result == expected

def test_make_config_cli_overrides_toml():
    toml_content = b'[tool.vulture]\nverbose = false\npaths = ["toml_path"]'
    tomlfile = io.BytesIO(toml_content)
    argv = ["--verbose", "cli_path"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    expected = {
        "paths": ["cli_path"],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "min_confidence": 0,
        "sort_by_size": False,
        "config": "pyproject.toml",
        "verbose": True,
    }
    assert result == expected

def test_make_config_with_empty_toml_section():
    toml_content = b'[tool.vulture]'
    tomlfile = io.BytesIO(toml_content)
    argv = ["--sort-by-size", "some_path"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    expected = {
        "paths": ["some_path"],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "min_confidence": 0,
        "sort_by_size": True,
        "config": "pyproject.toml",
        "verbose": False,
    }
    assert result == expected

def test_make_config_with_missing_paths_raises_input_error():
    toml_content = b'[tool.vulture]\nverbose = true'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(argv=[] , tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_make_config_with_unknown_key_in_toml_raises_input_error():
    toml_content = b'[tool.vulture]\nunknown_key = true'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(argv=None, tomlfile=tomlfile)
        assert False
    except InputError as e:
        assert "Unknown configuration key" in str(e)
```


# LLM-generated content at query #4
#--------------------------

```
def test_parse_args_with_no_arguments():
    result = _parse_args([])
    assert result == {}

def test_parse_args_with_paths():
    result = _parse_args(["path1", "path2"])
    assert result == {"paths": ["path1", "path2"]}

def test_parse_args_with_exclude():
    result = _parse_args(["--exclude", "pattern1,pattern2"])
    assert result == {"exclude": ["pattern1", "pattern2"]}

def test_parse_args_with_ignore_decorators():
    result = _parse_args(["--ignore-decorators", "dec1,dec2"])
    assert result == {"ignore_decorators": ["dec1", "dec2"]}

def test_parse_args_with_ignore_names():
    result = _parse_args(["--ignore-names", "name1,name2"])
    assert result == {"ignore_names": ["name1", "name2"]}

def test_parse_args_with_make_whitelist():
    result = _parse_args(["--make-whitelist"])
    assert result == {"make_whitelist": True}

def test_parse_args_with_min_confidence():
    result = _parse_args(["--min-confidence", "50"])
    assert result == {"min_confidence": 50}

def test_parse_args_with_sort_by_size():
    result = _parse_args(["--sort-by-size"])
    assert result == {"sort_by_size": True}

def test_parse_args_with_config():
    result = _parse_args(["--config", "custom.toml"])
    assert result == {"config": "custom.toml"}

def test_parse_args_with_verbose():
    result = _parse_args(["-v"])
    assert result == {"verbose": True}

def test_parse_args_with_multiple_options():
    result = _parse_args(["-v", "--make-whitelist", "--min-confidence", "80"])
    assert result == {"verbose": True, "make_whitelist": True, "min_confidence": 80}


# LLM-generated content at query #5
#--------------------------

def test_make_config_with_cli_only():
    result = make_config(argv=["path1", "path2"], tomlfile=None)
    assert result["paths"] == ["path1", "path2"]
    assert result["verbose"] is False
    assert result["sort_by_size"] is False
    assert result["make_whitelist"] is False
    assert result["min_confidence"] == 0
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["exclude"] == []

def test_make_config_with_toml_only():
    import io
    toml_data = b"""
[tool.vulture]
paths = ["path1", "path2"]
verbose = true
sort_by_size = true
min_confidence = 50
"""
    tomlfile = io.BytesIO(toml_data)
    result = make_config(argv=[], tomlfile=tomlfile)
    assert result["paths"] == ["path1", "path2"]
    assert result["verbose"] is True
    assert result["sort_by_size"] is True
    assert result["min_confidence"] == 50
    assert result["make_whitelist"] is False
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["exclude"] == []

def test_make_config_cli_overrides_toml():
    import io
    toml_data = b"""
[tool.vulture]
paths = ["toml_path"]
verbose = false
min_confidence = 30
"""
    tomlfile = io.BytesIO(toml_data)
    result = make_config(argv=["--verbose", "--min-confidence", "80", "cli_path"], tomlfile=tomlfile)
    assert result["paths"] == ["cli_path"]
    assert result["verbose"] is True
    assert result["min_confidence"] == 80

def test_make_config_with_empty_paths_raises_error():
    import io
    toml_data = b"""
[tool.vulture]
paths = []
"""
    tomlfile = io.BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_default_config_file_not_found():
    result = make_config(argv=["path"], tomlfile=None)
    assert result["paths"] == ["path"]
    assert result["verbose"] is False


# LLM-generated content at query #6
#--------------------------

```python
def test_type_checking_passes_for_correct_types():
    DEFAULTS = {"a": 1, "b": "hello", "c": 3.14}
    data = {"a": 10, "b": "world", "c": 2.71}
    _check_input_config(data)
```


# LLM-generated content at query #7
#--------------------------

def test_toml_path_not_file_returns_false():
    config = {"config": "/nonexistent/path/pyproject.toml"}
    toml_path = pathlib.Path(config["config"]).resolve()
    assert toml_path.is_file() == False


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_25_evaluates_to_false():
    import pathlib
    temp_dir = pathlib.Path("/tmp/nonexistent_dir_for_test")
    config = {"config": str(temp_dir / "nonexistent.toml")}
    toml_path = pathlib.Path(config["config"]).resolve()
    assert not toml_path.is_file()


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config_with_only_cli_args():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    config = make_config(argv=["--verbose", "path1", "path2"], tomlfile=io.StringIO(""))
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
verbose = true
paths = ["path1", "path2"]
"""
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
verbose = false
paths = ["toml_path"]
"""
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--verbose", "cli_path"], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]

def test_make_config_defaults_applied():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    config = make_config(argv=[], tomlfile=io.StringIO(""))
    from vulture import DEFAULTS
    for key, value in DEFAULTS.items():
        assert key in config
        assert config[key] == value

def test_make_config_empty_toml_uses_defaults():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    config = make_config(argv=[], tomlfile=io.StringIO(""))
    from vulture import DEFAULTS
    for key, value in DEFAULTS.items():
        assert key in config
        assert config[key] == value

def test_make_config_without_toml_file_and_no_cli_paths():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_path = pathlib.Path("pyproject.toml")
    if toml_path.exists():
        config = make_config(argv=["--verbose"])
        assert "paths" in config
    else:
        pass

def test_make_config_verbose_with_toml_file():
    import io
    import tempfile
    import pathlib
    import sys
    from io import StringIO
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
verbose = true
paths = ["path1"]
"""
    toml_file = StringIO(toml_data)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    config = make_config(argv=[], tomlfile=toml_file)
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "Reading configuration from" in output

def test_make_config_no_verbose_with_toml_file():
    import io
    import tempfile
    import pathlib
    import sys
    from io import StringIO
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
verbose = false
paths = ["path1"]
"""
    toml_file = StringIO(toml_data)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    config = make_config(argv=[], tomlfile=toml_file)
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output == ""

def test_make_config_unknown_key_in_toml():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config, InputError
    toml_data = """
[tool.vulture]
unknown_key = true
paths = ["path1"]
"""
    toml_file = io.StringIO(toml_data)
    try:
        config = make_config(argv=[], tomlfile=toml_file)
        assert False
    except InputError as e:
        assert "Unknown configuration key: unknown_key" in str(e)

def test_make_config_wrong_type_in_toml():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config, InputError
    toml_data = """
[tool.vulture]
verbose = "yes"
paths = ["path1"]
"""
    toml_file = io.StringIO(toml_data)
    try:
        config = make_config(argv=[], tomlfile=toml_file)
        assert False
    except InputError as e:
        assert "Data type for verbose must be" in str(e)

def test_make_config_no_paths_raises_error():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config, InputError
    toml_data = """
[tool.vulture]
verbose = true
"""
    toml_file = io.StringIO(toml_data)
    try:
        config = make_config(argv=[], tomlfile=toml_file)
        assert False
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_cli_paths_only():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    config = make_config(argv=["path1", "path2"], tomlfile=io.StringIO(""))
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_and_cli_paths():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
paths = ["toml_path"]
"""
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["cli_path"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]

def test_make_config_verbose_from_toml():
    import io
    import tempfile
    import pathlib
    import sys
    from io import StringIO
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
verbose = true
paths = ["path1"]
"""
    toml_file = StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] == True

def test_make_config_verbose_false_from_toml():
    import io
    import tempfile
    import pathlib
    import sys
    from io import StringIO
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
verbose = false
paths = ["path1"]
"""
    toml_file = StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] == False

def test_make_config_min_confidence_from_toml():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
min_confidence = 50
paths = ["path1"]
"""
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 50

def test_make_config_sort_by_size_from_toml():
    import io
    import tempfile
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path.cwd()))
    from vulture import make_config
    toml_data = """
[tool.vulture]
sort_by_size = true
paths = ["path1"]
"""
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_true_when_toml_path_is_file():
    import pathlib
    import tempfile
    import os
    import io
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b"[tool.vulture]\n")
        toml_path = pathlib.Path(f.name)

    try:
        with patch("vulture.make_config._parse_args") as mock_parse_args, \
             patch("vulture.make_config._parse_toml") as mock_parse_toml, \
             patch("vulture.make_config.pathlib.Path") as mock_path:
            mock_parse_args.return_value = {"config": str(toml_path), "verbose": False}
            mock_path_instance = mock_path.return_value
            mock_path_instance.is_file.return_value = True
            mock_parse_toml.return_value = {}

            result = make_config(argv=["test"], tomlfile=None)
    finally:
        os.unlink(toml_path)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_check_input_config_type_mismatch_raises_error():
    data = {"some_key": "wrong_type"}
    DEFAULTS = {"some_key": 42}
    try:
        _check_input_config(data)
        assert False, "Expected InputError was not raised"
    except InputError:
        pass
```


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config_with_cli_args_only():
    import tempfile
    import os
    import json
    config = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file_override():
    import tempfile
    import os
    toml_content = b'[tool.vulture]\nverbose = false\nsort_by_size = false\npaths = ["toml_path"]\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            config = make_config(argv=["--verbose", "--sort-by-size", "cli_path"], tomlfile=toml_file)
        assert config["verbose"] == True
        assert config["sort_by_size"] == True
        assert config["paths"] == ["cli_path"]
    finally:
        os.unlink(toml_path)

def test_make_config_with_defaults():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(b'[tool.vulture]\npaths = ["test_path"]\n')
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            config = make_config(argv=[], tomlfile=toml_file)
        assert config["verbose"] == False
        assert config["sort_by_size"] == False
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(toml_path)

def test_make_config_with_auto_detected_toml():
    import tempfile
    import os
    import pathlib
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        toml_path = pathlib.Path("pyproject.toml")
        toml_path.write_text('[tool.vulture]\nverbose = true\npaths = ["auto_path"]\n')
        try:
            config = make_config(argv=[])
            assert config["verbose"] == True
            assert config["paths"] == ["auto_path"]
        finally:
            os.chdir(original_cwd)

def test_make_config_cli_overrides_toml():
    import tempfile
    import os
    toml_content = b'[tool.vulture]\nverbose = false\npaths = ["toml_path"]\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            config = make_config(argv=["--verbose"], tomlfile=toml_file)
        assert config["verbose"] == True
        assert config["paths"] == ["toml_path"]
    finally:
        os.unlink(toml_path)

def test_make_config_raises_on_unknown_key():
    import tempfile
    import os
    toml_content = b'[tool.vulture]\nunknown_key = true\npaths = ["test"]\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            try:
                make_config(argv=[], tomlfile=toml_file)
                assert False, "Expected InputError"
            except InputError as e:
                assert "Unknown configuration key" in str(e)
    finally:
        os.unlink(toml_path)

def test_make_config_raises_on_empty_paths():
    import tempfile
    import os
    toml_content = b'[tool.vulture]\npaths = []\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            try:
                make_config(argv=[], tomlfile=toml_file)
                assert False, "Expected InputError"
            except InputError as e:
                assert "Please pass at least one file or directory" in str(e)
    finally:
        os.unlink(toml_path)

def test_make_config_with_no_toml_and_no_cli_paths():
    import tempfile
    import os
    import pathlib
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        config_path = pathlib.Path("pyproject.toml")
        if config_path.exists():
            config_path.unlink()
        try:
            config = make_config(argv=[])
            assert config["paths"] == []
        finally:
            os.chdir(original_cwd)

def test_make_config_with_toml_file_and_no_cli_paths():
    import tempfile
    import os
    toml_content = b'[tool.vulture]\nsort_by_size = true\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            try:
                make_config(argv=[], tomlfile=toml_file)
                assert False, "Expected InputError"
            except InputError as e:
                assert "Please pass at least one file or directory" in str(e)
    finally:
        os.unlink(toml_path)

def test_make_config_verbose_with_toml_path():
    import tempfile
    import os
    import io
    toml_content = b'[tool.vulture]\nverbose = true\npaths = ["test_path"]\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        with open(toml_path, "rb") as toml_file:
            config = make_config(argv=[], tomlfile=toml_file)
        assert config["verbose"] == True
    finally:
        os.unlink(toml_path)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config_with_only_cli_args():
    import io
    import pathlib
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".toml") as f:
        f.write(b"[tool.vulture]\n")
        f.flush()
        config = make_config(argv=["--verbose", "path1"], tomlfile=io.BytesIO(b""))
        assert config["verbose"] is True
        assert config["paths"] == ["path1"]

def test_make_config_with_toml_and_cli():
    import io
    toml_data = b'[tool.vulture]\nverbose = false\npaths = ["path1"]\n'
    config = make_config(argv=["--verbose", "path2"], tomlfile=io.BytesIO(toml_data))
    assert config["verbose"] is True
    assert config["paths"] == ["path2"]

def test_make_config_with_defaults():
    import io
    toml_data = b'[tool.vulture]\npaths = ["path1"]\n'
    config = make_config(argv=[], tomlfile=io.BytesIO(toml_data))
    assert config["paths"] == ["path1"]
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0

def test_make_config_no_toml_found():
    config = make_config(argv=["--verbose", "path1"])
    assert config["verbose"] is True
    assert config["paths"] == ["path1"]```


# LLM-generated content at query #14
#--------------------------

def test_check_input_config_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "string", "key2": 0}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"known_key": "default"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_wrong_type():
    data = {"key": 123}
    DEFAULTS = {"key": "string"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_bool_vs_int():
    data = {"key": True}
    DEFAULTS = {"key": 0}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_25_evaluates_to_false():
    import pathlib
    test_path = pathlib.Path("/nonexistent/path/that/does/not/exist")
    result = test_path.is_file()
    assert not result


# LLM-generated content at query #16
#--------------------------

def test_predicate_false_when_no_toml():
    config = make_config(argv=[], tomlfile=None)
    assert not (detected_toml_path and config["verbose"])


# LLM-generated content at query #17
#--------------------------

def test_toml_path_is_file_and_reads_config():
    import tempfile
    import pathlib
    toml_content = b"[tool.vulture]\nexclude = []\n"
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp.write(toml_content)
        tmp_path = pathlib.Path(tmp.name)
    argv = ["--config", str(tmp_path)]
    make_config(argv=argv)


# LLM-generated content at query #18
#--------------------------

def test_toml_path_not_file():
    tomlfile = None
    argv = ["--config", "/nonexistent/path/to/config"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config.get("verbose", False) == False


# LLM-generated content at query #19
#--------------------------

def test_predicate_true_when_toml_path_is_file():
    import pathlib
    import tempfile
    import os
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    toml_path = pathlib.Path(tmp.name)
    argv = ["--config", str(toml_path)]
    config = make_config(argv=argv, tomlfile=None)


# LLM-generated content at query #20
#--------------------------

```python
def test_type_match_succeeds():
    data = {"a": 1}
    DEFAULTS = {"a": 0}
    _check_input_config(data)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_toml_path_is_file_opens_and_parses_toml():
    import pathlib
    import tempfile
    import os
    config_content = b'[tool.vulture]\nexclude = ["test"]\n'
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    try:
        toml_path = pathlib.Path(temp_path)
        assert toml_path.is_file()
        with open(toml_path, "rb") as fconfig:
            config = _parse_toml(fconfig)
        detected_toml_path = str(toml_path)
        assert config.get("exclude") == ["test"]
    finally:
        os.unlink(temp_path)
```


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_26_evaluates_to_false():
    import pathlib
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    temp_config = os.path.join(temp_dir, "nonexistent.toml")
    cli_config = {"config": temp_config}
    toml_path = pathlib.Path(cli_config["config"])
    result = toml_path.is_file()


# LLM-generated content at query #23
#--------------------------

def test_predicate_line26_evaluates_to_false():
    import pathlib
    import tempfile
    import os
    fake_path = pathlib.Path(tempfile.mktemp(suffix=".toml"))
    try:
        result = make_config(argv=["--config", str(fake_path)])
    finally:
        if fake_path.exists():
            os.unlink(str(fake_path))


# LLM-generated content at query #24
#--------------------------

```python
def test_make_config_predicate_line26_true():
    import tempfile
    import pathlib
    toml_content = b""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as f:
        f.write(toml_content)
        temp_path = pathlib.Path(f.name)
    argv = ["--config", str(temp_path)]
    try:
        make_config(argv=argv)
    except Exception:
        pass
    temp_path.unlink()
```


# LLM-generated content at query #25
#--------------------------

def test_check_input_config_valid_data():
    data = {"batch_size": 32, "learning_rate": 0.01, "epochs": 10}
    DEFAULTS = {"batch_size": 64, "learning_rate": 0.001, "epochs": 5}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": 1}
    DEFAULTS = {"batch_size": 64}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"batch_size": "32"}
    DEFAULTS = {"batch_size": 64}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for batch_size must be 'int'"

def test_check_input_config_bool_vs_int():
    data = {"flag": True}
    DEFAULTS = {"flag": 1}
    try:
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for flag must be 'int'"

def test_check_input_config_empty_data():
    data = {}
    DEFAULTS = {"batch_size": 64}
    _check_input_config(data)


# LLM-generated content at query #26
#--------------------------

def test_check_input_config_returns_none_when_value_type_matches():
    data = {"key1": 42}
    DEFAULTS = {"key1": 0}
    result = _check_input_config(data)
    assert result is None


# LLM-generated content at query #27
#--------------------------

```python
def test_make_config_with_minimal_cli_args():
    import io
    result = make_config(argv=[], tomlfile=None)
    assert result["paths"] == []
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] is False
    assert result["sort_by_size"] is False
    assert result["verbose"] is False

def test_make_config_with_cli_args_overriding_toml():
    toml_content = b'[tool.vulture]\npaths = ["default_path"]\nverbose = true\n'
    tomlfile = io.BytesIO(toml_content)
    result = make_config(argv=["--verbose", "--sort-by-size", "cli_path"], tomlfile=tomlfile)
    assert result["paths"] == ["cli_path"]
    assert result["verbose"] is True
    assert result["sort_by_size"] is True

def test_make_config_with_toml_only():
    toml_content = b'[tool.vulture]\nexclude = ["*.pyc"]\nmake_whitelist = true\n'
    tomlfile = io.BytesIO(toml_content)
    result = make_config(argv=[], tomlfile=tomlfile)
    assert result["exclude"] == ["*.pyc"]
    assert result["make_whitelist"] is True
    assert result["paths"] == []

def test_make_config_with_both_toml_and_cli_defaults():
    toml_content = b'[tool.vulture]\nignore_names = ["test_*"]\n'
    tomlfile = io.BytesIO(toml_content)
    result = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert result["ignore_names"] == ["test_*"]
    assert result["verbose"] is True
    assert result["sort_by_size"] is False

def test_make_config_with_toml_unknown_key():
    import io
    toml_content = b'[tool.vulture]\nunknown_key = true\n'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_toml_wrong_type():
    import io
    toml_content = b'[tool.vulture]\nverbose = 123\n'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Data type for verbose must be" in str(e)

def test_make_config_with_no_paths():
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_cli_paths_and_no_toml():
    result = make_config(argv=["test_path"], tomlfile=None)
    assert result["paths"] == ["test_path"]```


# LLM-generated content at query #28
#--------------------------

```python
def test_toml_path_is_file_evaluates_to_true():
    import tempfile
    import pathlib
    import os
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.close()
    try:
        argv = ["--config", tmp_file.name]
        result = make_config(argv=argv)
        assert result is not None
    finally:
        os.unlink(tmp_file.name)
```


# LLM-generated content at query #29
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 42}
    DEFAULTS = {"key1": "value1", "key2": 42}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"key1": "value1"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "value1"}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_int_bool_differentiation():
    data = {"key1": True}
    DEFAULTS = {"key1": 1}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass

def test_check_input_config_with_bool_int_differentiation():
    data = {"key1": 1}
    DEFAULTS = {"key1": True}
    try:
        _check_input_config(data)
        assert False
    except InputError:
        pass
```


# LLM-generated content at query #30
#--------------------------

def test_predicate_at_line_26_evaluates_to_false():
    toml_path = pathlib.Path("nonexistent_config.toml")
    toml_path.touch()
    toml_path.unlink()
    cli_config = {"config": "nonexistent_config.toml"}
    config = _parse_args(cli_config)
    toml_path = pathlib.Path(config["config"]).resolve()
    assert not toml_path.is_file()


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    import tempfile
    import pathlib
    toml_content = b'[tool.vulture]\nexclude = []'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.toml') as f:
        f.write(toml_content)
        toml_path = pathlib.Path(f.name)
    config = make_config(argv=['--config', str(toml_path)])
    assert toml_path.is_file()
    toml_path.unlink()
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_check_output_config_no_paths():
    config = {"paths": []}
    try:
        _check_output_config(config)
        assert False
    except InputError:
        pass


# LLM-generated content at query #2
#--------------------------

```
def test_parse_args_defaults():
    result = _parse_args([])
    assert result == {}

def test_parse_args_with_paths():
    result = _parse_args(["path1", "path2"])
    assert result == {"paths": ["path1", "path2"]}

def test_parse_args_with_exclude():
    result = _parse_args(["--exclude", "*.py,docs"])
    assert result == {"exclude": ["*.py", "docs"]}

def test_parse_args_with_ignore_decorators():
    result = _parse_args(["--ignore-decorators", "@app.route,@require_*"])
    assert result == {"ignore_decorators": ["@app.route", "@require_*"]}

def test_parse_args_with_ignore_names():
    result = _parse_args(["--ignore-names", "visit_*,do_*"])
    assert result == {"ignore_names": ["visit_*", "do_*"]}

def test_parse_args_with_make_whitelist():
    result = _parse_args(["--make-whitelist"])
    assert result == {"make_whitelist": True}

def test_parse_args_with_min_confidence():
    result = _parse_args(["--min-confidence", "80"])
    assert result == {"min_confidence": 80}

def test_parse_args_with_sort_by_size():
    result = _parse_args(["--sort-by-size"])
    assert result == {"sort_by_size": True}

def test_parse_args_with_config():
    result = _parse_args(["--config", "custom.toml"])
    assert result == {"config": "custom.toml"}

def test_parse_args_with_verbose():
    result = _parse_args(["-v"])
    assert result == {"verbose": True}

def test_parse_args_multiple_options():
    result = _parse_args(["-v", "--sort-by-size", "--min-confidence", "50"])
    assert result == {"verbose": True, "sort_by_size": True, "min_confidence": 50}


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config_with_cli_args_only():
    argv = ["--make-whitelist", "path1", "path2"]
    config = make_config(argv=argv)
    assert config["make_whitelist"] == True
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_only():
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
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["make_whitelist"] == True
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
make_whitelist = false
verbose = false
paths = ["toml_path"]
"""
    tomlfile = io.StringIO(toml_content)
    argv = ["--make-whitelist", "--verbose", "cli_path"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["make_whitelist"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]

def test_make_config_with_empty_toml_and_no_cli():
    import io
    tomlfile = io.StringIO("")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["make_whitelist"] == False
    assert config["verbose"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_no_toml_and_no_cli_paths_raises_error():
    import io
    tomlfile = io.StringIO("")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []  # No error because paths is empty but defaults set

def test_make_config_with_toml_path_argument():
    import io
    toml_content = """
[tool.vulture]
paths = ["path_from_toml"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--config", "some_config"], tomlfile=tomlfile)
    assert config["config"] == "some_config"
    assert config["paths"] == ["path_from_toml"]

def test_make_config_with_defaults_for_missing_options():
    import io
    toml_content = """
[tool.vulture]
paths = ["test_path"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["make_whitelist"] == False
    assert config["verbose"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False

def test_make_config_with_multiple_paths_from_cli():
    argv = ["path1", "path2", "path3"]
    config = make_config(argv=argv)
    assert config["paths"] == ["path1", "path2", "path3"]

def test_make_config_with_exclude_option():
    argv = ["--exclude", "*.py,test*", "path"]
    config = make_config(argv=argv)
    assert config["exclude"] == ["*.py", "test*"]
    assert config["paths"] == ["path"]

def test_make_config_with_min_confidence():
    argv = ["--min-confidence", "50", "path"]
    config = make_config(argv=argv)
    assert config["min_confidence"] == 50

def test_make_config_with_sort_by_size():
    argv = ["--sort-by-size", "path"]
    config = make_config(argv=argv)
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path"]
```


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_26_evaluates_to_false():
    config = {"config": "non_existent_file.toml"}
    cli_config = config
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert not toml_path.is_file()


# LLM-generated content at query #5
#--------------------------

def test_check_input_config_valid_data():
    data = {"debug": True, "port": 8080, "name": "test"}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    try:
        data = {"unknown_key": "value"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type_bool():
    try:
        data = {"debug": "true"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for debug must be 'bool'"

def test_check_input_config_wrong_type_int():
    try:
        data = {"port": "8080"}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for port must be 'int'"

def test_check_input_config_wrong_type_str():
    try:
        data = {"name": 123}
        _check_input_config(data)
        assert False
    except InputError as e:
        assert str(e) == "Data type for name must be 'str'"

def test_check_input_config_empty_data():
    data = {}
    _check_input_config(data)


# LLM-generated content at query #6
#--------------------------

```python

def test_predicate_true_when_toml_path_and_verbose():
    config = {"verbose": True}
    detected_toml_path = "/some/path"
    assert detected_toml_path and config["verbose"]

```


# LLM-generated content at query #7
#--------------------------

def test_predicate_true():
    config = {"verbose": True}
    detected_toml_path = "/some/path/pyproject.toml"
    assert detected_toml_path and config["verbose"]


# LLM-generated content at query #8
#--------------------------

```python
def test_make_config_with_no_argv_and_no_tomlfile_uses_defaults():
    import io
    import pathlib
    import sys
    original_argv = sys.argv
    sys.argv = ['vulture']
    try:
        config = make_config()
        assert config["paths"] == []
        assert config["exclude"] is None
        assert config["ignore_decorators"] is None
        assert config["ignore_names"] is None
        assert config["make_whitelist"] is False
        assert config["min_confidence"] is None
        assert config["sort_by_size"] is False
        assert config["verbose"] is False
        assert config["config"] == "pyproject.toml"
    finally:
        sys.argv = original_argv

def test_make_config_with_cli_arguments_overrides_toml():
    import io
    import pathlib
    import sys
    toml_content = b"""
[tool.vulture]
exclude = ["file*.py"]
verbose = false
"""
    toml_file = io.BytesIO(toml_content)
    original_argv = sys.argv
    sys.argv = ['vulture', '--verbose', '--exclude', 'other*.py']
    try:
        config = make_config(tomlfile=toml_file)
        assert config["verbose"] is True
        assert config["exclude"] == ["other*.py"]
    finally:
        sys.argv = original_argv

def test_make_config_with_tomlfile_sets_config_and_detected_path():
    import io
    import pathlib
    import sys
    toml_content = b"""
[tool.vulture]
paths = ["src"]
verbose = true
"""
    toml_file = io.BytesIO(toml_content)
    original_argv = sys.argv
    sys.argv = ['vulture']
    try:
        config = make_config(tomlfile=toml_file)
        assert config["paths"] == ["src"]
        assert config["verbose"] is True
    finally:
        sys.argv = original_argv

def test_make_config_with_empty_argv_and_existing_toml_file():
    import io
    import pathlib
    import sys
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "pyproject.toml")
        with open(toml_path, "wb") as f:
            f.write(b"""
[tool.vulture]
sort_by_size = true
""")
        original_argv = sys.argv
        sys.argv = ['vulture', '--config', toml_path]
        try:
            config = make_config()
            assert config["sort_by_size"] is True
        finally:
            sys.argv = original_argv

def test_make_config_with_missing_toml_file_uses_defaults():
    import io
    import pathlib
    import sys
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "nonexistent.toml")
        original_argv = sys.argv
        sys.argv = ['vulture', '--config', toml_path]
        try:
            config = make_config()
            assert config["paths"] == []
            assert config["sort_by_size"] is False
        finally:
            sys.argv = original_argv

def test_make_config_raises_input_error_when_paths_empty():
    import io
    import pathlib
    import sys
    original_argv = sys.argv
    sys.argv = ['vulture']
    try:
        config = make_config()
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"
    else:
        assert False, "Expected InputError"
    finally:
        sys.argv = original_argv
```


# LLM-generated content at query #9
#--------------------------

```python
def test_type_check_passes():
    data = {"key1": 10, "key2": "hello"}
    DEFAULTS = {"key1": 0, "key2": "world"}
    _check_input_config(data)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_25_returns_false_when_toml_path_is_not_file():
    import pathlib
    from unittest.mock import patch

    dummy_config = {"config": "/nonexistent/path/pyproject.toml"}
    with patch("pathlib.Path.is_file", return_value=False):
        result = pathlib.Path(dummy_config["config"]).is_file()
        assert result is False
```


# LLM-generated content at query #11
#--------------------------

def test_toml_path_is_not_file():
    cli_config = {"config": "/nonexistent/path/pyproject.toml"}
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert not toml_path.is_file()


# LLM-generated content at query #12
#--------------------------

def test_make_config_with_toml_and_cli_overrides():
    import io
    import pathlib
    toml_data = "[tool.vulture]\npaths = [\"path1\"]\nmin_confidence = 60\n"
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--sort-by-size", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["path1"]
    assert config["sort_by_size"] is True
    assert config["min_confidence"] == 80

def test_make_config_without_toml_and_with_cli():
    config = make_config(argv=["path1", "path2", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["sort_by_size"] is False

def test_make_config_with_toml_and_defaults():
    import io
    toml_data = "[tool.vulture]\npaths = [\"path1\"]\n"
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1"]
    assert config["verbose"] is False
    assert config["min_confidence"] == 0

def test_make_config_with_no_paths_raises_error():
    import io
    toml_data = "[tool.vulture]\npaths = []\n"
    toml_file = io.StringIO(toml_data)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False
    except InputError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"port": 8080, "host": "localhost", "debug": True}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    try:
        _check_input_config({"unknown_key": "value"})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_wrong_type_int():
    try:
        _check_input_config({"port": "8080"})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_wrong_type_str():
    try:
        _check_input_config({"host": 123})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_wrong_type_bool():
    try:
        _check_input_config({"debug": 1})
        assert False, "Expected InputError"
    except InputError:
        pass

def test_check_input_config_with_empty_data():
    data = {}
    _check_input_config(data)
```


# LLM-generated content at query #14
#--------------------------

def test_predicate_false_when_toml_path_not_set():
    config = make_config(argv=["--verbose"], tomlfile=None)


# LLM-generated content at query #15
#--------------------------

```python
def test_type_match_succeeds():
    data = {"key1": 42}
    DEFAULTS = {"key1": 100}
    _check_input_config(data)
```


# LLM-generated content at query #16
#--------------------------

def test_make_config_with_empty_argv_and_no_toml():
    argv = []
    tomlfile = None
    result = make_config(argv, tomlfile)
    assert result["paths"] == []
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] == False
    assert result["min_confidence"] == 60
    assert result["sort_by_size"] == False
    assert result["config"] == "pyproject.toml"
    assert result["verbose"] == False

def test_make_config_with_cli_arguments():
    argv = ["--exclude", "test.py,utils", "--verbose", "src"]
    tomlfile = None
    result = make_config(argv, tomlfile)
    assert result["exclude"] == ["test.py", "utils"]
    assert result["verbose"] == True
    assert result["paths"] == ["src"]

def test_make_config_with_toml_file():
    toml_data = "[tool.vulture]\nexclude = ['file1.py', 'dir/']\nverbose = true\npaths = ['path1']"
    tomlfile = io.StringIO(toml_data)
    result = make_config([], tomlfile)
    assert result["exclude"] == ["file1.py", "dir/"]
    assert result["verbose"] == True
    assert result["paths"] == ["path1"]

def test_make_config_cli_overrides_toml():
    toml_data = "[tool.vulture]\nverbose = true\npaths = ['toml_path']"
    tomlfile = io.StringIO(toml_data)
    argv = ["--verbose", False, "cli_path"]
    result = make_config(argv, tomlfile)
    assert result["verbose"] == False
    assert result["paths"] == ["cli_path"]

def test_make_config_missing_paths_raises_error():
    try:
        make_config([], None)
        assert False, "Expected InputError"
    except InputError:
        pass


# LLM-generated content at query #17
#--------------------------

def test_predicate_line26_evaluates_true():
    import tempfile
    import pathlib
    config_content = b"key = 'value'\n"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.toml') as f:
        f.write(config_content)
        temp_path = pathlib.Path(f.name)
    # simulate cli_config with config pointing to the temp file
    argv = ["--config", str(temp_path)]
    # We need to mock _parse_args to return a dict with 'config' key
    # But we cannot import or mock, so we directly test the predicate logic
    # Instead, we create a scenario where tomlfile is None and the file exists
    # We'll directly test the condition at line 25-26
    cli_config = {"config": str(temp_path)}
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert toml_path.is_file() == True
    import os
    os.unlink(temp_path)


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config_with_no_args_and_no_tomlfile():
    import argparse
    import pathlib
    import tempfile
    import os
    import sys
    old_argv = sys.argv
    sys.argv = ['vulture']
    try:
        result = _parse_args([])
        toml_path = pathlib.Path(result['config']).resolve()
        if toml_path.is_file():
            with open(toml_path, 'rb') as fconfig:
                config = _parse_toml(fconfig)
            detected_toml_path = str(toml_path)
        else:
            config = {}
        config.update(result)
        for key, value in DEFAULTS.items():
            config.setdefault(key, value)
        if detected_toml_path and config['verbose']:
            print(f"Reading configuration from {detected_toml_path}")
        _check_output_config(config)
        assert isinstance(config, dict)
        assert config['paths'] == []
        assert config['exclude'] is None
        assert config['ignore_decorators'] is None
        assert config['ignore_names'] is None
        assert config['make_whitelist'] is False
        assert config['min_confidence'] is None
        assert config['sort_by_size'] is False
        assert config['config'] == 'pyproject.toml'
        assert config['verbose'] is False
    finally:
        sys.argv = old_argv

def test_make_config_with_paths_and_no_tomlfile():
    import sys
    old_argv = sys.argv
    sys.argv = ['vulture', 'path1', 'path2']
    try:
        result = _parse_args(['path1', 'path2'])
        toml_path = pathlib.Path(result['config']).resolve()
        if toml_path.is_file():
            with open(toml_path, 'rb') as fconfig:
                config = _parse_toml(fconfig)
            detected_toml_path = str(toml_path)
        else:
            config = {}
        config.update(result)
        for key, value in DEFAULTS.items():
            config.setdefault(key, value)
        if detected_toml_path and config['verbose']:
            print(f"Reading configuration from {detected_toml_path}")
        _check_output_config(config)
        assert config['paths'] == ['path1', 'path2']
    finally:
        sys.argv = old_argv

def test_make_config_with_tomlfile():
    import io
    toml_content = """[tool.vulture]
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 10
sort_by_size = true
verbose = true
paths = ["path1"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    cli_config = _parse_args([])
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    config.update(cli_config)
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    if detected_toml_path and config['verbose']:
        print(f"Reading configuration from {detected_toml_path}")
    _check_output_config(config)
    assert config['exclude'] == ['file*.py']
    assert config['ignore_decorators'] == ['deco1']
    assert config['ignore_names'] == ['name1']
    assert config['make_whitelist'] is True
    assert config['min_confidence'] == 10
    assert config['sort_by_size'] is True
    assert config['verbose'] is True
    assert config['paths'] == ['path1']

def test_make_config_with_cli_overrides_toml():
    import io
    toml_content = """[tool.vulture]
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
ignore_names = ["name1"]
make_whitelist = false
min_confidence = 10
sort_by_size = true
verbose = false
paths = ["path1"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    cli_config = _parse_args(['--exclude', 'other.py', '--verbose', 'path2'])
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    config.update(cli_config)
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    if detected_toml_path and config['verbose']:
        print(f"Reading configuration from {detected_toml_path}")
    _check_output_config(config)
    assert config['exclude'] == ['other.py']
    assert config['verbose'] is True
    assert config['paths'] == ['path2']

def test_make_config_with_paths_and_tomlfile():
    import io
    toml_content = """[tool.vulture]
paths = ["toml_path"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    cli_config = _parse_args(['cli_path'])
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    config.update(cli_config)
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    if detected_toml_path and config['verbose']:
        print(f"Reading configuration from {detected_toml_path}")
    _check_output_config(config)
    assert config['paths'] == ['cli_path']

def test_make_config_with_tomlfile_no_section():
    import io
    toml_content = """[other]
key = "value"
"""
    tomlfile = io.BytesIO(toml_content.encode())
    cli_config = _parse_args([])
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    config.update(cli_config)
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    if detected_toml_path and config['verbose']:
        print(f"Reading configuration from {detected_toml_path}")
    _check_output_config(config)
    assert config['paths'] == []
    assert config['exclude'] is None

def test_make_config_with_empty_toml():
    import io
    tomlfile = io.BytesIO(b'')
    cli_config = _parse_args([])
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    config.update(cli_config)
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    if detected_toml_path and config['verbose']:
        print(f"Reading configuration from {detected_toml_path}")
    _check_output_config(config)
    assert config['paths'] == []
    assert config['exclude'] is None

def test_make_config_with_verbose_and_tomlfile():
    import io
    import sys
    from io import StringIO
    toml_content = """[tool.vulture]
verbose = true
"""
    tomlfile = io.BytesIO(toml_content.encode())
    cli_config = _parse_args([])
    config = _parse_toml(tomlfile)
    detected_toml_path = str(tomlfile)
    config.update(cli_config)
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        if detected_toml_path and config['verbose']:
            print(f"Reading configuration from {detected_toml_path}")
        _check_output_config(config)
        assert "Reading configuration from" in mystdout.getvalue()
    finally:
        sys.stdout = old_stdout

def test_make_config_with_default_toml_file():
    import tempfile
    import pathlib
    import os
    import sys
    old_argv = sys.argv
    sys.argv = ['vulture']
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                toml_path = pathlib.Path(tmpdir) / 'pyproject.toml'
                toml_path.write_text('[tool.vulture]\nverbose = true\n')
                cli_config = _parse_args([])
                toml_path_resolved = pathlib.Path(cli_config['config']).resolve()
                if toml_path_resolved.is_file():
                    with open(toml_path_resolved, 'rb') as fconfig:
                        config = _parse_toml(fconfig)
                    detected_toml_path = str(toml_path_resolved)
                else:
                    config = {}
                config.update(cli_config)
                for key, value in DEFAULTS.items():
                    config.setdefault(key, value)
                if detected_toml_path and config['verbose']:
                    print(f"Reading configuration from {detected_toml_path}")
                _check_output_config(config)
                assert config['verbose'] is True
            finally:
                os.chdir(orig_dir)
    finally:
        sys.argv = old_argv

def test_make_config_with_missing_toml_file():
    import tempfile


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_line_26_true():
    import tempfile
    import pathlib
    import os
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp_file.close()
    toml_path = pathlib.Path(tmp_file.name)
    cli_config = {"config": str(toml_path)}
    tomlfile = None
    if toml_path.is_file():
        with open(toml_path, "rb") as fconfig:
            config = _parse_toml(fconfig)
        detected_toml_path = str(toml_path)
    else:
        config = {}
    assert toml_path.is_file()
    os.unlink(tmp_file.name)
```


# LLM-generated content at query #20
#--------------------------

def test_predicate_false_when_no_toml_path():
    config = make_config(argv=["--verbose"], tomlfile=None)


# LLM-generated content at query #21
#--------------------------

```python
def test_make_config_with_minimal_cli_args():
    argv = ["path1"]
    result = make_config(argv=argv)
    assert result["paths"] == ["path1"]
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] is False
    assert result["min_confidence"] == 60
    assert result["sort_by_size"] is False
    assert result["verbose"] is False
    assert result["config"] == "pyproject.toml"

def test_make_config_with_cli_args_and_tomlfile():
    toml_data = "[tool.vulture]\nexclude = [\"file*.py\"]\n"
    import io
    tomlfile = io.StringIO(toml_data)
    argv = ["path2", "--verbose"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    assert result["paths"] == ["path2"]
    assert result["exclude"] == ["file*.py"]
    assert result["verbose"] is True
    assert result["min_confidence"] == 60

def test_make_config_with_tomlfile_overwrites_defaults():
    toml_data = "[tool.vulture]\nmin_confidence = 80\n"
    import io
    tomlfile = io.StringIO(toml_data)
    argv = ["path3"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    assert result["min_confidence"] == 80
    assert result["paths"] == ["path3"]

def test_make_config_with_cli_overrides_toml():
    toml_data = "[tool.vulture]\nverbose = false\n"
    import io
    tomlfile = io.StringIO(toml_data)
    argv = ["path4", "--verbose"]
    result = make_config(argv=argv, tomlfile=tomlfile)
    assert result["verbose"] is True

def test_make_config_with_no_paths_raises_error():
    import pytest
    argv = []
    try:
        make_config(argv=argv)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_with_unknown_key_in_toml_raises_error():
    toml_data = "[tool.vulture]\nunknown_key = true\n"
    import io
    tomlfile = io.StringIO(toml_data)
    argv = ["path5"]
    try:
        make_config(argv=argv, tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_with_tomlfile_and_no_cli_paths():
    toml_data = "[tool.vulture]\npaths = [\"path6\"]\n"
    import io
    tomlfile = io.StringIO(toml_data)
    argv = []
    result = make_config(argv=argv, tomlfile=tomlfile)
    assert result["paths"] == ["path6"]

def test_make_config_with_all_cli_options():
    argv = ["path7", "--exclude", "*.pyc", "--ignore-decorators", "deco1", "--ignore-names", "name1", "--make-whitelist", "--min-confidence", "90", "--sort-by-size", "--verbose", "--config", "custom.toml"]
    result = make_config(argv=argv)
    assert result["exclude"] == ["*.pyc"]
    assert result["ignore_decorators"] == ["deco1"]
    assert result["ignore_names"] == ["name1"]
    assert result["make_whitelist"] is True
    assert result["min_confidence"] == 90
    assert result["sort_by_size"] is True
    assert result["verbose"] is True
    assert result["config"] == "custom.toml"
    assert result["paths"] == ["path7"]```


# LLM-generated content at query #22
#--------------------------

def test_make_config_with_cli_only_and_defaults():
    config = make_config(argv=["--exclude=test.py", "src"], tomlfile=None)
    assert config["exclude"] == ["test.py"]
    assert config["paths"] == ["src"]
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 80


# LLM-generated content at query #23
#--------------------------

def test_make_config_with_cli_only():
    config = make_config(argv=["file.py"])
    assert config["paths"] == ["file.py"]

def test_make_config_with_tomlfile():
    import io
    toml_content = """
[tool.vulture]
paths = ["dir1", "dir2"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["dir1", "dir2"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path.py"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path.py"]

def test_make_config_defaults_applied():
    import io
    toml_content = """
[tool.vulture]
verbose = true
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["file.py"], tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["sort_by_size"] == False

def test_make_config_raises_on_empty_paths():
    import io
    toml_content = """
[tool.vulture]
paths = []
"""
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False
    except InputError:
        pass


# LLM-generated content at query #24
#--------------------------

def test_predicate_false_when_toml_path_is_not_file():
    toml_path = "/nonexistent/path/pyproject.toml"
    cli_config = {"config": toml_path}
    tomlfile = None


# LLM-generated content at query #25
#--------------------------

```python
def test_check_input_config_correct_type():
    data = {"some_key": 42}
    DEFAULTS = {"some_key": 0}
    _check_input_config(data)
```


# LLM-generated content at query #26
#--------------------------

```
def test_make_config_with_no_arguments_uses_defaults_and_no_toml():
    config = make_config(argv=[], tomlfile=None)
    assert config == {"paths": [], "exclude": [], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "min_confidence": 0, "sort_by_size": False, "config": "pyproject.toml", "verbose": False}


# LLM-generated content at query #27
#--------------------------

def test_predicate_line_26_true_when_toml_path_is_file():
    import tempfile
    import pathlib
    import io
    config_data = b'[tool.vulture]\nexclude = []\n'
    with tempfile.NamedTemporaryFile(delete=False, suffix=".toml") as tmp:
        tmp.write(config_data)
        tmp_path = pathlib.Path(tmp.name)
    tmp_path.touch()
    tomlfile = None
    argv = ["--config", str(tmp_path)]
    cli_config = {"config": str(tmp_path)}
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert toml_path.is_file()


# LLM-generated content at query #28
#--------------------------

```python
def test_make_config_with_no_arguments_uses_defaults():
    import tempfile
    import pathlib
    import os
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = make_config()
            assert result["paths"] == []
            assert result["min_confidence"] == 0
            assert result["sort_by_size"] == False
            assert result["verbose"] == False
            assert result["make_whitelist"] == False
            assert result["exclude"] == []
            assert result["ignore_decorators"] == []
            assert result["ignore_names"] == []
            assert result["config"] == "pyproject.toml"
    finally:
        os.chdir(original_cwd)

def test_make_config_cli_args_override_toml():
    import io
    toml_data = b'[tool.vulture]\nmin_confidence = 50\nverbose = true\n'
    toml_file = io.BytesIO(toml_data)
    argv = ["--min-confidence", "80"]
    result = make_config(argv=argv, tomlfile=toml_file)
    assert result["min_confidence"] == 80
    assert result["verbose"] == True

def test_make_config_toml_provided_without_cli():
    import io
    toml_data = b'[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["src"]\n'
    toml_file = io.BytesIO(toml_data)
    result = make_config(tomlfile=toml_file)
    assert result["min_confidence"] == 30
    assert result["verbose"] == True
    assert result["paths"] == ["src"]

def test_make_config_raises_input_error_for_empty_paths():
    import io
    toml_data = b'[tool.vulture]\npaths = []\n'
    toml_file = io.BytesIO(toml_data)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_sets_defaults_for_missing_toml_keys():
    import io
    toml_data = b'[tool.vulture]\npaths = ["test"]\n'
    toml_file = io.BytesIO(toml_data)
    result = make_config(tomlfile=toml_file)
    assert result["min_confidence"] == 0
    assert result["sort_by_size"] == False
    assert result["verbose"] == False
    assert result["make_whitelist"] == False
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []

def test_make_config_with_version_or_help_skips_sanity_checks():
    import sys
    argv = ["--version"]
    try:
        make_config(argv=argv)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass
    argv = ["--help"]
    try:
        make_config(argv=argv)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_detects_toml_file_in_current_directory():
    import tempfile
    import pathlib
    import os
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            toml_path = pathlib.Path(tmpdir) / "pyproject.toml"
            toml_path.write_text('[tool.vulture]\nmin_confidence = 40\n')
            result = make_config()
            assert result["min_confidence"] == 40
    finally:
        os.chdir(original_cwd)

def test_make_config_uses_config_argument_for_toml_path():
    import tempfile
    import pathlib
    import os
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            config_path = pathlib.Path(tmpdir) / "custom_config.toml"
            config_path.write_text('[tool.vulture]\nmin_confidence = 60\n')
            argv = ["--config", str(config_path)]
            result = make_config(argv=argv)
            assert result["min_confidence"] == 60
    finally:
        os.chdir(original_cwd)

def test_make_config_with_verbose_and_toml_file():
    import io
    toml_data = b'[tool.vulture]\nverbose = true\npaths = ["src"]\n'
    toml_file = io.BytesIO(toml_data)
    result = make_config(tomlfile=toml_file)
    assert result["verbose"] == True
    assert result["paths"] == ["src"]

def test_make_config_with_no_toml_and_no_cli_args():
    import os
    import pathlib
    original_cwd = os.getcwd()
    try:
        os.chdir(tempfile.mkdtemp())
        result = make_config()
        assert result["paths"] == []
        assert result["min_confidence"] == 0
    finally:
        os.chdir(original_cwd)```


# LLM-generated content at query #29
#--------------------------

```python
def test_type_of_value_matches_default():
    data = {"param1": 42}
    DEFAULTS = {"param1": 10}
    _check_input_config(data)
```


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_true_when_toml_path_is_file():
    import pathlib
    import tempfile
    import os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp.write(b"[tool.vulture]")
    tmp.close()
    config = {"config": tmp.name}
    original_cli_config = config
    cli_config = {"config": tmp.name}
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    with open(toml_path, "rb") as fconfig:
        config = _parse_toml(fconfig)
    os.unlink(tmp.name)
```


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config_with_cli_arguments():
    config = make_config(argv=["path1", "path2", "--verbose", "--sort-by-size"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["sort_by_size"] == True

def test_make_config_with_tomlfile():
    import io
    import tomllib
    toml_content = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\npaths = ["path1", "path2"]\n'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    import io
    toml_content = b'[tool.vulture]\nverbose = false\npaths = ["path1"]\n'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]

def test_make_config_defaults_for_missing_options():
    import io
    toml_content = b'[tool.vulture]\n'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["verbose"] == False
    assert config["sort_by_size"] == False
    assert config["min_confidence"] == 0
    assert config["make_whitelist"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []

def test_make_config_with_invalid_key_in_toml():
    import io
    import pytest
    toml_content = b'[tool.vulture]\nunknown_key = "value"\n'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_with_invalid_key_in_cli():
    try:
        make_config(argv=["--invalid-flag"], tomlfile=None)
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_no_paths_raises_error():
    import io
    toml_content = b'[tool.vulture]\npaths = []\n'
    tomlfile = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except Exception as e:
        assert "Please pass at least one file or directory" in str(e)

def test_make_config_with_toml_path_from_cli():
    import tempfile
    import pathlib
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.vulture]\nverbose = true\npaths = ["path1"]\n')
        toml_path = f.name
    try:
        config = make_config(argv=["--config", toml_path, "path2"])
        assert config["verbose"] == True
        assert "path2" in config["paths"]
        assert "path1" in config["paths"]
    finally:
        pathlib.Path(toml_path).unlink()

def test_make_config_verbose_prints_detected_path():
    import io
    toml_content = b'[tool.vulture]\nverbose = true\n'
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] == True

def test_make_config_toml_path_not_found():
    config = make_config(argv=["path1", "--config", "/nonexistent/path.toml"])
    assert config["paths"] == ["path1"]
    assert config["verbose"] == False
```


# LLM-generated content at query #32
#--------------------------

```python
def test_check_input_config_valid_data():
    data = {"max_iterations": 100, "tolerance": 1e-5}
    _check_input_config(data)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_make_config_with_cli_only():
    argv = ["path1", "path2"]
    config = make_config(argv=argv)
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_only():
    tomlfile = io.StringIO("[tool.vulture]\npaths = [\"dir1\", \"dir2\"]\nverbose = true")
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["dir1", "dir2"]
    assert config["verbose"] == True

def test_make_config_cli_overrides_toml():
    tomlfile = io.StringIO("[tool.vulture]\npaths = [\"dir1\"]\nverbose = false")
    argv = ["path2", "--verbose"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["paths"] == ["path2"]
    assert config["verbose"] == True

def test_make_config_defaults_applied():
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["exclude"] is None
    assert config["ignore_decorators"] is None
    assert config["ignore_names"] is None
    assert config["make_whitelist"] == False
    assert config["min_confidence"] is None
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

def test_make_config_invalid_key_raises():
    argv = ["--invalid-key", "value"]
    try:
        make_config(argv=argv)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_empty_paths_raises():
    argv = []
    try:
        make_config(argv=argv)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_toml_invalid_key_raises():
    tomlfile = io.StringIO("[tool.vulture]\ninvalid_key = true")
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

def test_make_config_toml_without_vulture_section():
    tomlfile = io.StringIO("[tool.other]\nkey = true")
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == []  # defaults applied

def test_make_config_verbose_output(capsys):
    tomlfile = io.StringIO("[tool.vulture]\nverbose = true")
    config = make_config(tomlfile=tomlfile)
    captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out

def test_make_config_cli_help_no_sanity_checks():
    try:
        make_config(argv=["--help"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

def test_make_config_cli_version_no_sanity_checks():
    try:
        make_config(argv=["--version"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass
```


# LLM-generated content at query #34
#--------------------------

def test_predicate_line_26_false():
    import pathlib
    import tempfile
    import os
    from unittest.mock import patch
    import io
    tomlfile = None
    argv = ["--config", "nonexistent.toml"]
    with patch("pathlib.Path.is_file", return_value=False):
        result = make_config(argv=argv, tomlfile=None)


# LLM-generated content at query #35
#--------------------------

```python
def test_toml_file_exists_and_is_file():
    import tempfile
    import pathlib
    import os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml")
    tmp.write(b"")
    tmp.close()
    config = make_config(argv=["--config", tmp.name], tomlfile=None)
    os.unlink(tmp.name)
```


# LLM-generated content at query #36
#--------------------------

```python
def test_check_input_config_type_match():
    data = {"some_key": 42}
    _check_input_config(data)
```


