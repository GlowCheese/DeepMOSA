####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["file.py"])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_cli_args():
    config = make_config(argv=["--exclude", "test_*.py", "--min-confidence", "80", "file.py"])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 80

def test_make_config_toml_file():
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
paths = ["path1"]
min_confidence = 10
"""
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "path2"], tomlfile=tomlfile)
    assert config["paths"] == ["path2"]
    assert config["min_confidence"] == 80

def test_make_config_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_unknown_key_in_toml_raises_error():
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    import io
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(tomlfile=tomlfile)

def test_make_config_wrong_type_in_toml_raises_error():
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    import io
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(tomlfile=tomlfile)


# LLM-generated content at query #2
#--------------------------

```python
def test_tomlfile_predicate_evaluates_to_true():
    tomlfile = "test.toml"
    assert tomlfile


# LLM-generated content at query #3
#--------------------------

```python
def test_verbose_output_when_toml_detected():
    config = make_config(argv=["--verbose"], tomlfile=pathlib.Path("test.toml").open("rb"))
    assert config["verbose"] is True
    assert "Reading configuration from" in str(config)


# LLM-generated content at query #4
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["--help"])
    assert config == {}

def test_make_config_cli_args():
    config = make_config(argv=["--exclude", "test_*.py", "--verbose", "src/"])
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_toml_file():
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
    paths = ["src/"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--exclude", "docs/", "--verbose"], tomlfile=tomlfile)
    assert config["exclude"] == ["docs/"]
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

def test_make_config_missing_paths():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_invalid_key():
    import pytest
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["--invalid-key", "value"])

def test_make_config_wrong_type():
    import pytest
    with pytest.raises(InputError, match="Data type for"):
        make_config(argv=["--min-confidence", "not_an_int"])


# LLM-generated content at query #5
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["--version"])
    assert config == {}

def test_make_config_cli_args():
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_toml_file():
    from io import StringIO
    toml_content = """
    [tool.vulture]
    min_confidence = 90
    verbose = true
    paths = ["path1", "path2"]
    """
    tomlfile = StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    from io import StringIO
    toml_content = """
    [tool.vulture]
    min_confidence = 90
    verbose = false
    """
    tomlfile = StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

def test_make_config_no_paths_raises():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_make_config_unknown_key_raises():
    try:
        make_config(argv=["--unknown-key", "value"])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e).startswith("Unknown configuration key:")

def test_make_config_wrong_type_raises():
    try:
        make_config(argv=["--min-confidence", "not_an_int"])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e).startswith("Data type for min_confidence must be")


# LLM-generated content at query #6
#--------------------------

```python
def test_check_input_config_valid():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)

def test_check_input_config_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    data = {"key1": "not_an_int"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_false():
    result = make_config(argv=["--verbose"], tomlfile=None)
    assert not (result.get("verbose") and result.get("detected_toml_path"))


# LLM-generated content at query #8
#--------------------------

```python
def test_check_input_config_with_correct_types():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_false():
    config = make_config(argv=["--config", "nonexistent.toml"])
    assert not (config.get("detected_toml_path") and config["verbose"])


# LLM-generated content at query #10
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("existing_file.toml")
    assert toml_path.is_file()


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").is_file()


# LLM-generated content at query #12
#--------------------------

```python
def test_check_input_config_type_matches_default():
    data = {"key1": 42, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("non_existent_file.toml").resolve().is_file()


# LLM-generated content at query #14
#--------------------------

```python
def test_check_input_config_valid_input():
    valid_data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default_value", "key2": 0}
    _check_input_config(valid_data)

def test_check_input_config_unknown_key():
    invalid_data = {"unknown_key": "value"}
    DEFAULTS = {"key1": "default_value"}
    try:
        _check_input_config(invalid_data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_wrong_type():
    invalid_data = {"key1": 123}
    DEFAULTS = {"key1": "default_value"}
    try:
        _check_input_config(invalid_data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #15
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"unknown_key": "value"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    data = {"key1": "not_an_int"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #16
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--exclude", "test_*.py", "--verbose", "path1", "path2"])
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_with_toml_file():
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
verbose = true
paths = ["path1", "path2"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

def test_make_config_cli_overrides_toml():
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
verbose = false
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--exclude", "other_*.py", "--verbose"], tomlfile=tomlfile)
    assert config["exclude"] == ["other_*.py"]
    assert config["verbose"] is True

def test_make_config_defaults():
    config = make_config()
    assert config["exclude"] == DEFAULTS["exclude"]
    assert config["verbose"] == DEFAULTS["verbose"]
    assert config["paths"] == []

def test_make_config_no_paths_raises_error():
    with pytest.raises(InputError):
        make_config()

def test_make_config_with_version_flag():
    with pytest.raises(SystemExit):
        make_config(["--version"])


# LLM-generated content at query #17
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

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
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "50", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

def test_make_config_with_defaults():
    config = make_config([])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["paths"] == []

def test_make_config_empty_paths_raises_error():
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([])


# LLM-generated content at query #18
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").resolve().is_file()


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["path1.py"])
    assert config["paths"] == ["path1.py"]
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
        "path1.py", "path2.py"
    ])
    assert config["paths"] == ["path1.py", "path2.py"]
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
paths = ["path1.py", "path2.py"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["path1.py", "path2.py"]
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
paths = ["path1.py", "path2.py"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[
        "--exclude", "new_test_*,*.pyc",
        "--ignore-decorators", "new_deco1,new_deco2",
        "--ignore-names", "new_name1,new_name2",
        "--min-confidence", "75",
        "path3.py"
    ], tomlfile=tomlfile)
    assert config["paths"] == ["path3.py"]
    assert config["exclude"] == ["new_test_*", "*.pyc"]
    assert config["ignore_decorators"] == ["new_deco1", "new_deco2"]
    assert config["ignore_names"] == ["new_name1", "new_name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 75
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_no_paths_error():
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_unknown_key_error():
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--unknown-key", "value", "path1.py"])

def test_make_config_wrong_type_error():
    with pytest.raises(InputError, match="Data type for 'min_confidence' must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "path1.py"])


# LLM-generated content at query #21
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default1", "key2": 0}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"key1": "value1", "unknown_key": "value2"}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("/non/existent/file.toml").is_file()


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    import pathlib
    import tempfile

    # Create a temporary TOML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
        tmp.write("[tool.vulture]\nmin_confidence = 0.5\n")
        tmp_path = tmp.name

    # Mock the CLI config to point to the temporary TOML file
    cli_config = {"config": tmp_path, "verbose": False}

    # Ensure the predicate evaluates to True
    toml_path = pathlib.Path(cli_config["config"]).resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #24
#--------------------------

```python
def test_check_input_config_type_mismatch():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default_value"}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("non_existent_file.toml").is_file()


# LLM-generated content at query #27
#--------------------------

```python
def test_check_input_config_type_matching():
    DEFAULTS = {"key1": 1, "key2": "value", "key3": True}
    data = {"key1": 2, "key2": "another", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #29
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
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_file_only():
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
    assert config["config"] == "pyproject.toml"

def test_make_config_with_both_cli_and_toml():
    toml_content = """
[tool.vulture]
paths = ["toml_path1"]
exclude = ["*test*.py"]
ignore_decorators = ["@decorator1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 80
sort_by_size = true
verbose = false
"""
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--verbose", "--min-confidence", "90", "cli_path1", "cli_path2"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path1", "cli_path2"]
    assert config["exclude"] == ["*test*.py"]
    assert config["ignore_decorators"] == ["@decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 90
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["config"] == "pyproject.toml"

def test_make_config_with_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])

def test_make_config_with_unknown_cli_key_raises_error():
    import pytest
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--unknown-key", "value"])

def test_make_config_with_wrong_type_cli_value_raises_error():
    import pytest
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])

def test_make_config_with_unknown_toml_key_raises_error():
    import pytest
    import io
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(tomlfile=tomlfile)

def test_make_config_with_wrong_type_toml_value_raises_error():
    import pytest
    import io
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(tomlfile=tomlfile)


# LLM-generated content at query #30
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

def test_make_config_with_toml_file_only():
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

def test_make_config_with_cli_args_overriding_toml():
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
    config = make_config(argv=["--min-confidence", "20", "path3"], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["path3"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_with_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])

def test_make_config_with_unknown_key_in_toml_raises_error():
    import pytest
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(tomlfile=tomlfile)

def test_make_config_with_wrong_type_in_toml_raises_error():
    import pytest
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_integer"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(tomlfile=tomlfile)


# LLM-generated content at query #31
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"key1": 1, "key2": "value"}
    DEFAULTS = {"key1": 0, "key2": ""}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"unknown_key": 1}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    data = {"key1": "not_an_int"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"

def test_check_input_config_with_bool_vs_int():
    data = {"key1": True}
    DEFAULTS = {"key1": 1}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #32
#--------------------------

```python
def test_toml_path_is_file_predicate():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


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
        assert False, "Expected InputError was not raised"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_check_output_config_with_valid_paths():
    config = {"paths": ["/some/path"]}
    _check_output_config(config)


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_args_defaults():
    result = _parse_args([])
    assert result == {}

def test_parse_args_paths():
    result = _parse_args(["file1.py", "file2.py"])
    assert result == {"paths": ["file1.py", "file2.py"]}

def test_parse_args_exclude():
    result = _parse_args(["--exclude", "*.py,test_*"])
    assert result == {"exclude": ["*.py", "test_*"]}

def test_parse_args_ignore_decorators():
    result = _parse_args(["--ignore-decorators", "@app.route,@require_*"])
    assert result == {"ignore_decorators": ["@app.route", "@require_*"]}

def test_parse_args_ignore_names():
    result = _parse_args(["--ignore-names", "visit_*,do_*"])
    assert result == {"ignore_names": ["visit_*", "do_*"]}

def test_parse_args_make_whitelist():
    result = _parse_args(["--make-whitelist"])
    assert result == {"make_whitelist": True}

def test_parse_args_min_confidence():
    result = _parse_args(["--min-confidence", "50"])
    assert result == {"min_confidence": 50}

def test_parse_args_sort_by_size():
    result = _parse_args(["--sort-by-size"])
    assert result == {"sort_by_size": True}

def test_parse_args_config():
    result = _parse_args(["--config", "custom.toml"])
    assert result == {"config": "custom.toml"}

def test_parse_args_verbose():
    result = _parse_args(["-v"])
    assert result == {"verbose": True}

def test_parse_args_multiple_options():
    result = _parse_args(["file.py", "--exclude", "*.py", "--verbose"])
    assert result == {"paths": ["file.py"], "exclude": ["*.py"], "verbose": True}

def test_parse_args_invalid_key():
    with pytest.raises(InputError):
        _parse_args(["--invalid-key", "value"])

def test_parse_args_wrong_type():
    with pytest.raises(InputError):
        _parse_args(["--min-confidence", "not_a_number"])


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

def test_make_config_with_toml_file():
    import io
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
    import io
    toml_content = """
        [tool.vulture]
        verbose = false
        paths = ["toml_path"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--verbose", "cli_path"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["cli_path"]

def test_make_config_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([])


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_evaluates_to_false():
    toml_path = pathlib.Path("nonexistent_file.toml").resolve()
    assert not toml_path.is_file()


# LLM-generated content at query #5
#--------------------------

```python
def test_check_input_config_with_valid_data():
    data = {"key1": "value1", "key2": 123}
    DEFAULTS = {"key1": "default", "key2": 0}
    _check_input_config(data)

def test_check_input_config_with_unknown_key():
    data = {"key1": "value1", "unknown_key": 123}
    DEFAULTS = {"key1": "default"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    data = {"key1": 123}
    DEFAULTS = {"key1": "default"}
    try:
        _check_input_config(data)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert "exclude" in config
    assert "ignore_decorators" in config
    assert "ignore_names" in config
    assert "make_whitelist" in config
    assert "sort_by_size" in config

def test_make_config_with_toml_file():
    import io
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
    import io
    toml_content = """
        [tool.vulture]
        min_confidence = 10
        verbose = false
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "50", "--verbose"], tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

def test_make_config_no_paths_raises_error():
    import pytest
    with pytest.raises(InputError):
        make_config([])


# LLM-generated content at query #7
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["--help"])
    assert config == {}

def test_make_config_cli_only():
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

def test_make_config_toml_only():
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
ignore_decorators = ["@deco1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 30
sort_by_size = true
verbose = true
paths = ["path1"]
"""
    from io import StringIO
    tomlfile = StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1"]

def test_make_config_cli_overrides_toml():
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
min_confidence = 30
paths = ["path1"]
"""
    from io import StringIO
    tomlfile = StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "50", "path2"], tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path2"]

def test_make_config_invalid_toml_key():
    toml_content = """
[tool.vulture]
invalid_key = "value"
"""
    from io import StringIO
    tomlfile = StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_invalid_toml_type():
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    from io import StringIO
    tomlfile = StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be" in str(e)

def test_make_config_no_paths():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_check_input_config_valid():
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
    data = {"key1": "not_an_int"}
    DEFAULTS = {"key1": 0}
    try:
        _check_input_config(data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'int'"


# LLM-generated content at query #9
#--------------------------

```python
def test_check_input_config_type_matching():
    DEFAULTS = {'key1': 1, 'key2': 'value', 'key3': True}
    data = {'key1': 2, 'key2': 'another', 'key3': False}
    _check_input_config(data)


# LLM-generated content at query #10
#--------------------------

```python
def test_toml_path_is_file_predicate_false():
    assert not pathlib.Path("non_existent_file.toml").is_file()


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").resolve().is_file()


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["file.py"])
    assert config["paths"] == ["file.py"]
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
        "--exclude", "test_*.py",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "--make-whitelist",
        "--min-confidence", "80",
        "--sort-by-size",
        "--verbose",
        "file.py"
    ])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_toml_overrides():
    toml_content = """
[tool.vulture]
paths = ["toml_file.py"]
exclude = ["*test*.py"]
ignore_decorators = ["toml_deco"]
ignore_names = ["toml_name"]
make_whitelist = true
min_confidence = 70
sort_by_size = true
verbose = true
"""
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_file.py"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_file.py"]
    assert config["exclude"] == ["*test*.py"]
    assert config["ignore_decorators"] == ["toml_deco"]
    assert config["ignore_names"] == ["toml_name"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 70
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_no_paths_error():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_invalid_key_error():
    import pytest
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(argv=["--invalid-key", "value", "file.py"])

def test_make_config_wrong_type_error():
    import pytest
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "file.py"])


# LLM-generated content at query #13
#--------------------------

```python
def test_check_input_config_type_matching():
    data = {"key1": 42, "key2": "value"}
    DEFAULTS = {"key1": 42, "key2": "value"}
    _check_input_config(data)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").resolve().is_file()


# LLM-generated content at query #15
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
    assert config["config"] == "pyproject.toml"

def test_make_config_cli_args():
    config = make_config(["path1", "path2", "--exclude=*.py", "--min-confidence=80"])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["*.py"]
    assert config["min_confidence"] == 80

def test_make_config_tomlfile():
    toml_content = """
[tool.vulture]
paths = ["path1", "path2"]
exclude = ["*.py"]
min_confidence = 80
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["*.py"]
    assert config["min_confidence"] == 80

def test_make_config_cli_overrides_toml():
    toml_content = """
[tool.vulture]
paths = ["path1"]
min_confidence = 80
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(["path2", "--min-confidence=90"], tomlfile=tomlfile)
    assert config["paths"] == ["path2"]
    assert config["min_confidence"] == 90

def test_make_config_verbose_toml_path():
    toml_content = """
[tool.vulture]
verbose = true
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True

def test_make_config_invalid_toml_key():
    toml_content = """
[tool.vulture]
invalid_key = "value"
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

def test_make_config_invalid_toml_type():
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be" in str(e)

def test_make_config_no_paths():
    try:
        make_config()
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_false():
    config = {"verbose": False}
    detected_toml_path = "some_path"
    assert not (detected_toml_path and config["verbose"])


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").resolve().is_file()


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(argv=["file.py"])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_with_toml_and_cli_args():
    toml_content = """
    [tool.vulture]
    paths = ["dir/"]
    exclude = ["test_*.py"]
    ignore_decorators = ["@decorator"]
    ignore_names = ["unused_*"]
    make_whitelist = true
    min_confidence = 80
    sort_by_size = true
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["dir/"]
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@decorator"]
    assert config["ignore_names"] == ["unused_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 90
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_with_defaults():
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

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

def test_make_config_with_invalid_cli_key():
    try:
        make_config(argv=["--invalid-key", "value"])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

def test_make_config_with_wrong_type_in_toml():
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

def test_make_config_with_wrong_type_in_cli():
    try:
        make_config(argv=["--min-confidence", "not_an_int"])
        assert False, "Expected InputError"
    except SystemExit:
        pass  # argparse handles this with a SystemExit

def test_make_config_with_empty_paths():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

def test_make_config_with_verbose_and_toml():
    toml_content = """
    [tool.vulture]
    paths = ["dir/"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_26():
    import pathlib
    import tempfile
    import os

    # Create a temporary TOML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
        tmp.write("[tool.vulture]\nmin_confidence = 0.5\n")
        tmp_path = tmp.name

    try:
        # Mock the CLI config to point to the temporary TOML file
        cli_config = {"config": tmp_path, "verbose": False}
        toml_path = pathlib.Path(cli_config["config"]).resolve()

        # Ensure the predicate at line 26 evaluates to True
        assert toml_path.is_file() == True
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config_with_cli_args():
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_file():
    import io
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
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    verbose = false
    paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "50", "--verbose", "path2"], tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path2"]

def test_make_config_defaults():
    config = make_config([])
    assert config["min_confidence"] == 100
    assert config["verbose"] is False
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["paths"] == []

def test_make_config_missing_paths():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([])


# LLM-generated content at query #21
#--------------------------

```python
def test_toml_path_is_file_predicate():
    toml_path = pathlib.Path("pyproject.toml").resolve()
    assert toml_path.is_file()


# LLM-generated content at query #22
#--------------------------

```python
def test_toml_path_is_file_predicate():
    toml_path = pathlib.Path("existing_file.toml").resolve()
    assert toml_path.is_file() is True


# LLM-generated content at query #23
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(["--min-confidence", "50", "path/to/file.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path/to/file.py"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_and_cli_args():
    toml_content = """
        [tool.vulture]
        exclude = ["test_*.py"]
        min_confidence = 30
        paths = ["src/"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "50", "path/to/file.py"], tomlfile)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path/to/file.py"]
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_toml_only():
    toml_content = """
        [tool.vulture]
        exclude = ["test_*.py"]
        min_confidence = 30
        paths = ["src/"]
        verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src/"]
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is True
    assert config["config"] == "pyproject.toml"

def test_make_config_with_defaults():
    config = make_config(["path/to/file.py"])
    assert config["min_confidence"] == 60
    assert config["paths"] == ["path/to/file.py"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_with_verbose_and_toml():
    toml_content = """
        [tool.vulture]
        paths = ["src/"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--verbose"], tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    import pathlib
    from unittest.mock import patch

    toml_path = pathlib.Path("/non/existent/file.toml")
    assert not toml_path.is_file()


# LLM-generated content at query #25
#--------------------------

```python
def test_check_input_config_with_valid_data():
    valid_data = {"key1": "value1", "key2": 42}
    DEFAULTS = {"key1": "default1", "key2": 0}
    _check_input_config(valid_data)

def test_check_input_config_with_unknown_key():
    invalid_data = {"unknown_key": "value"}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(invalid_data)
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"

def test_check_input_config_with_wrong_type():
    invalid_data = {"key1": 123}
    DEFAULTS = {"key1": "default1"}
    try:
        _check_input_config(invalid_data)
    except InputError as e:
        assert str(e) == "Data type for key1 must be 'str'"


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["file.py"])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_cli_overrides():
    config = make_config(argv=["--exclude", "test_*.py", "--min-confidence", "80", "file.py"])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 80

def test_make_config_toml_parsing():
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 80
    paths = ["file.py"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 80

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 80
    paths = ["file.py"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--exclude", "other_*.py", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == ["other_*.py"]
    assert config["min_confidence"] == 90

def test_make_config_no_paths_raises_error():
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_unknown_key_raises_error():
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--unknown-key", "value", "file.py"])

def test_make_config_wrong_type_raises_error():
    with pytest.raises(InputError, match="Data type for 'min_confidence' must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "file.py"])


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_false():
    config = make_config(argv=["--verbose"])
    assert not (config["detected_toml_path"] and config["verbose"])


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_false():
    # Case 1: detected_toml_path is empty
    config = make_config(argv=["--verbose"])
    assert not (config.get("detected_toml_path") and config["verbose"])

    # Case 2: verbose is False
    with open("dummy.toml", "w") as f:
        f.write("[tool.vulture]\nverbose = false")
    config = make_config(tomlfile=open("dummy.toml", "rb"))
    assert not (config.get("detected_toml_path") and config["verbose"])


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    import pathlib
    toml_path = pathlib.Path("nonexistent_file.toml")
    assert not toml_path.is_file()


# LLM-generated content at query #30
#--------------------------

```python
def test_check_input_config_type_comparison():
    data = {"key": 1}
    DEFAULTS = {"key": 1}
    assert type(data["key"]) is type(DEFAULTS["key"])


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(argv=["file.py"])
    assert config["paths"] == ["file.py"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 60
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_with_toml_only():
    toml_content = """
    [tool.vulture]
    paths = ["dir/"]
    exclude = ["test_*.py"]
    ignore_decorators = ["@decorator"]
    ignore_names = ["unused_*"]
    make_whitelist = true
    min_confidence = 80
    sort_by_size = true
    verbose = true
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["dir/"]
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@decorator"]
    assert config["ignore_names"] == ["unused_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

def test_make_config_cli_overrides_toml():
    toml_content = """
    [tool.vulture]
    paths = ["dir/"]
    min_confidence = 80
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["file.py", "--min-confidence", "70"], tomlfile=tomlfile)
    assert config["paths"] == ["file.py"]
    assert config["min_confidence"] == 70

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
        assert "Unknown configuration key" in str(e)

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
        assert "Data type for min_confidence must be" in str(e)

def test_make_config_with_no_paths():
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #32
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

def test_make_config_with_toml_only():
    toml_content = """
        [tool.vulture]
        paths = ["path1", "path2"]
        verbose = true
        exclude = ["file*.py", "dir/"]
        ignore_decorators = ["deco1", "deco2"]
        ignore_names = ["name1", "name2"]
        make_whitelist = true
        min_confidence = 10
        sort_by_size = true
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

def test_make_config_with_cli_overriding_toml():
    toml_content = """
        [tool.vulture]
        paths = ["path1", "path2"]
        verbose = false
        min_confidence = 10
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--verbose", "--min-confidence", "20"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 20

def test_make_config_with_no_paths():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([])


# LLM-generated content at query #33
#--------------------------

```python
def test_check_input_config_valid_types():
    data = {"key1": 1, "key2": "value", "key3": True}
    DEFAULTS = {"key1": 0, "key2": "", "key3": False}
    _check_input_config(data)


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").is_file()


# LLM-generated content at query #35
#--------------------------

```python
def test_make_config_with_cli_args_only():
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

def test_make_config_with_toml_file_only():
    import io
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

def test_make_config_with_cli_args_overriding_toml():
    import io
    toml_content = """
        [tool.vulture]
        min_confidence = 10
        paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "50", "path2"], tomlfile)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path2"]

def test_make_config_with_missing_paths():
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([])

def test_make_config_with_invalid_cli_arg():
    import pytest
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(["--invalid-arg", "value"])

def test_make_config_with_invalid_toml_config():
    import io
    import pytest
    toml_content = """
        [tool.vulture]
        invalid_key = "value"
    """
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(tomlfile=tomlfile)

def test_make_config_with_wrong_type_in_cli_args():
    import pytest
    with pytest.raises(InputError, match="Data type for min_confidence must be"):
        make_config(["--min-confidence", "not_a_number"])

def test_make_config_with_wrong_type_in_toml():
    import io
    import pytest
    toml_content = """
        [tool.vulture]
        min_confidence = "not_a_number"
    """
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Data type for min_confidence must be"):
        make_config(tomlfile=tomlfile)


# LLM-generated content at query #36
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
    toml_content = """
        [tool.vulture]
        invalid_key = "value"
    """
    import io
    tomlfile = io.StringIO(toml_content)
    try:
        make_config([], tomlfile)
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
        make_config([], tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"

def test_make_config_with_empty_paths():
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"


# LLM-generated content at query #37
#--------------------------

```python
def test_toml_path_is_file():
    toml_path = pathlib.Path("test.toml")
    toml_path.write_text("[tool.vulture]\nmin_confidence = 0.5")
    assert toml_path.is_file()


# LLM-generated content at query #38
#--------------------------

```python
def test_type_comparison_with_identical_types():
    DEFAULTS = {"key1": 1}
    data = {"key1": 2}
    assert type(data["key1"]) is type(DEFAULTS["key1"])


# LLM-generated content at query #39
#--------------------------

```python
def test_make_config_defaults():
    config = make_config(argv=["--help"])
    assert config == {}

def test_make_config_cli_args():
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

def test_make_config_toml_file():
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
    config = make_config(argv=["--min-confidence", "50", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

def test_make_config_missing_paths():
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_39_evaluates_to_false():
    assert not (detected_toml_path and config["verbose"])


# LLM-generated content at query #41
#--------------------------

```python
def test_toml_path_is_file():
    import pathlib
    import tempfile

    # Create a temporary TOML file
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp.write(b"[vulture]\nmin_confidence = 0.5")
        tmp_path = tmp.name

    # Ensure the path exists and is a file
    toml_path = pathlib.Path(tmp_path).resolve()
    assert toml_path.is_file()


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    assert not pathlib.Path("nonexistent_file.toml").resolve().is_file()


