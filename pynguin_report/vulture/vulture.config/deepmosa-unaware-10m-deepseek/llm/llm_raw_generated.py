####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["new_path", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["new_path"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with missing paths (should raise InputError)
    import pytest
    with pytest.raises(InputError):
        config = make_config(argv=[], tomlfile=None)

    # Test with invalid TOML config
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(invalid_toml)
    with pytest.raises(InputError):
        config = make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML config
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    toml_file = io.StringIO(wrong_type_toml)
    with pytest.raises(InputError):
        config = make_config(argv=[], tomlfile=toml_file)

    # Test with config file path from CLI
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path], tomlfile=None)
        assert config["min_confidence"] == 10
        assert config["verbose"] is True
    finally:
        os.unlink(temp_path)

    # Test with non-existent config file
    config = make_config(argv=["--config", "nonexistent.toml", "path1"], tomlfile=None)
    assert config["paths"] == ["path1"]
    assert config["min_confidence"] == 0
```


# LLM-generated content at query #2
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    
    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 30
    exclude = ["test*.py"]
    """
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test*.py"]
    
    # Test that CLI arguments take precedence over TOML
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["cli_path", "--min-confidence", "80"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80
    
    # Test with invalid configuration key
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path"]
    """)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong data type
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)
    
    # Test with no paths provided
    try:
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\n"))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config():
    # Test with no CLI arguments and no TOML file - should raise InputError
    with pytest.raises(InputError):
        make_config([])
    
    # Test with CLI arguments only
    config = make_config(["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["test*.py", "temp/"]
    min_confidence = 50
    paths = ["src/", "main.py"]
    verbose = true
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(["extra.py"], tomlfile=tomlfile)
    assert config["paths"] == ["extra.py"]  # CLI overrides TOML
    assert config["exclude"] == ["test*.py", "temp/"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    
    # Test CLI overrides TOML
    toml_data = """
    [tool.vulture]
    min_confidence = 50
    paths = ["toml_path.py"]
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80", "cli_path.py"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path.py"]
    
    # Test with all CLI options
    config = make_config([
        "--exclude", "test*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--min-confidence", "70",
        "--sort-by-size",
        "--verbose",
        "--config", "custom.toml",
        "path1.py", "path2.py"
    ])
    assert config["exclude"] == ["test*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 70
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["path1.py", "path2.py"]
    
    # Test with TOML file and no CLI paths
    toml_data = """
    [tool.vulture]
    paths = ["src/"]
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config([], tomlfile=tomlfile)
    assert config["paths"] == ["src/"]
    
    # Test that missing paths raises InputError
    toml_data = """
    [tool.vulture]
    exclude = ["test*.py"]
    """
    tomlfile = io.StringIO(toml_data)
    with pytest.raises(InputError):
        make_config([], tomlfile=tomlfile)


# LLM-generated content at query #4
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
    config = make_config(argv=[])
    assert config == DEFAULTS
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 0

    # Test with toml file
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["src"]
    exclude = ["test_*.py"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 10
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test_*.py"]

    # Test CLI overrides TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "20", "other"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["other"]

    # Test with actual pyproject.toml file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 30
        paths = ["lib"]
        """)
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["min_confidence"] == 30
        assert config["paths"] == ["lib"]
    finally:
        os.unlink(temp_path)

    # Test with invalid config key
    toml_content = """
    [tool.vulture]
    invalid_key = true
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type
    toml_content = """
    [tool.vulture]
    min_confidence = "high"
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with no paths
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test verbose output with toml file
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(AssertionError):
        make_config(argv=[], tomlfile=toml_file, verbose=True)

    # Test that CLI arguments with multiple values are parsed correctly
    config = make_config(argv=["--exclude", "a.py,b.py", "--ignore-decorators", "dec1,dec2", "--ignore-names", "name1,name2"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test that defaults are set for missing options
    config = make_config(argv=["path1"])
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
```


# LLM-generated content at query #5
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["path_from_toml"]
    verbose = true
    """
    config = make_config([], tomlfile=io.StringIO(toml_data))
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path_from_toml"]
    assert config["verbose"] is True

    # Test CLI arguments override TOML values
    config = make_config(["--min-confidence", "80"], tomlfile=io.StringIO(toml_data))
    assert config["min_confidence"] == 80

    # Test with invalid TOML key
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["test"]
    """
    import pytest
    with pytest.raises(InputError):
        make_config([], tomlfile=io.StringIO(invalid_toml))

    # Test with no paths
    with pytest.raises(InputError):
        make_config([])

    # Test with path provided but no TOML file exists
    config = make_config(["test_file.py"])
    assert config["paths"] == ["test_file.py"]
    assert config["config"] == "pyproject.toml"

    # Test with explicit config file that doesn't exist
    config = make_config(["--config", "nonexistent.toml", "test.py"])
    assert config["paths"] == ["test.py"]

    # Test with all defaults applied
    config = make_config(["test.py"], tomlfile=io.StringIO("[tool.vulture]\n"))
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
```


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
[tool.vulture]
exclude = ["file*.py", "dir/"]
min_confidence = 10
sort_by_size = true
paths = ["path1", "path2"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config([], tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override TOML settings
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "80"], tomlfile)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]

    # Test with --exclude CLI argument
    config = make_config(["--exclude", "file1.py,file2.py", "path"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with --ignore-decorators CLI argument
    config = make_config(["--ignore-decorators", "@app.route,@require_*", "path"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names CLI argument
    config = make_config(["--ignore-names", "visit_*,do_*", "path"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --make-whitelist
    config = make_config(["--make-whitelist", "path"])
    assert config["make_whitelist"] is True

    # Test with --sort-by-size
    config = make_config(["--sort-by-size", "path"])
    assert config["sort_by_size"] is True

    # Test with no paths should raise InputError
    import pytest
    with pytest.raises(InputError):
        make_config([])

    # Test with unknown configuration key in TOML
    bad_toml = """
[tool.vulture]
unknown_key = "value"
"""
    tomlfile = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile)

    # Test with wrong type in TOML
    bad_type_toml = """
[tool.vulture]
min_confidence = "high"
"""
    tomlfile = io.StringIO(bad_type_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile)

    # Test with --config pointing to existing file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        temp_path = f.name
    try:
        config = make_config(["--config", temp_path])
        assert config["min_confidence"] == 10
        assert config["sort_by_size"] is True
    finally:
        os.unlink(temp_path)

    # Test that default config path is used when no --config is given
    config = make_config([], None)
    assert config["config"] == "pyproject.toml"

    # Test verbose output when loading from TOML file
    tomlfile = io.StringIO("""
[tool.vulture]
paths = ["path"]
""")
    config = make_config(["--verbose"], tomlfile)
    assert config["verbose"] is True

    # Test that paths from CLI are preserved even if TOML has paths
    tomlfile = io.StringIO("""
[tool.vulture]
paths = ["toml_path"]
""")
    config = make_config(["cli_path"], tomlfile)
    assert config["paths"] == ["cli_path"]
```


# LLM-generated content at query #7
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 75
    paths = ["toml_path1"]
    exclude = ["test_*.py"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["toml_path1"]
    assert config["exclude"] == ["test_*.py"]

    # Test CLI overrides toml
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "90"], tomlfile)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["toml_path1"]

    # Test with multiple CLI options
    config = make_config([
        "--exclude", "file1.py,file2.py",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--config", "custom_config.toml"
    ])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["config"] == "custom_config.toml"

    # Test error when no paths provided
    try:
        config = make_config([])
        # Should raise InputError
        assert False, "Expected InputError for empty paths"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile
    import io
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
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config([], tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override tomlfile
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(["--min-confidence", "80", "--verbose"], tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    assert config["exclude"] == ["file*.py", "dir/"]

    # Test with missing paths raises InputError
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with unknown configuration key
    toml_data = """
    [tool.vulture]
    unknown_key = true
    paths = ["path1"]
    """
    tomlfile = io.BytesIO(toml_data.encode())
    try:
        make_config([], tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key: unknown_key" in str(e)

    # Test with wrong type
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    tomlfile = io.BytesIO(toml_data.encode())
    try:
        make_config([], tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with --version
    import sys
    try:
        make_config(["--version"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test with --help
    try:
        make_config(["--help"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test with --exclude
    config = make_config(["--exclude", "file1.py,file2.py", "path1"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with --ignore-decorators
    config = make_config(["--ignore-decorators", "@app.route,@require_*", "path1"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names
    config = make_config(["--ignore-names", "visit_*,do_*", "path1"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --make-whitelist
    config = make_config(["--make-whitelist", "path1"])
    assert config["make_whitelist"] == True

    # Test with --sort-by-size
    config = make_config(["--sort-by-size", "path1"])
    assert config["sort_by_size"] == True

    # Test with --config
    config = make_config(["--config", "custom.toml", "path1"])
    assert config["config"] == "custom.toml"


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override toml config
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "20", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml

    # Test with invalid config key in toml
    bad_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in toml
    bad_type_toml = """
    [tool.vulture]
    min_confidence = "high"
    """
    toml_file = io.StringIO(bad_type_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with no paths (should raise InputError)
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with paths provided via toml
    toml_file = io.StringIO("""
    [tool.vulture]
    paths = ["test.py"]
    """)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["test.py"]

    # Test with --version (should not raise InputError)
    try:
        make_config(argv=["--version"])
    except SystemExit:
        pass  # argparse exits on --version

    # Test with --help (should not raise InputError)
    try:
        make_config(argv=["--help"])
    except SystemExit:
        pass  # argparse exits on --help
```


# LLM-generated content at query #10
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50

    # Test with toml file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--verbose", "--min-confidence", "75"], tomlfile=toml_file)
    assert config["verbose"] is True
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML still applies

    # Test with invalid config key in toml
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(invalid_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(wrong_type_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "must be" in str(e)

    # Test with no paths
    try:
        make_config(argv=[], tomlfile=io.StringIO(""))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with paths from CLI
    config = make_config(argv=["test.py"])
    assert config["paths"] == ["test.py"]

    # Test with paths from toml
    toml_file = io.StringIO("[tool.vulture]\npaths = ['test.py']")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["test.py"]
```


# LLM-generated content at query #11
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 10
    paths = ["path_from_toml"]
    exclude = ["exclude1", "exclude2"]
    """
    config = make_config(argv=[], tomlfile=io.StringIO(toml_data))
    assert config["min_confidence"] == 10
    assert config["paths"] == ["path_from_toml"]
    assert config["exclude"] == ["exclude1", "exclude2"]
    assert config["verbose"] == False

    # Test CLI overrides TOML
    toml_data = """
    [tool.vulture]
    min_confidence = 10
    paths = ["path_from_toml"]
    """
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=io.StringIO(toml_data))
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test with invalid config key in TOML
    toml_data = """
    [tool.vulture]
    invalid_key = "value"
    """
    try:
        make_config(argv=[], tomlfile=io.StringIO(toml_data))
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    try:
        make_config(argv=[], tomlfile=io.StringIO(toml_data))
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths specified
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should raise InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (uses defaults)
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override toml file
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "75"], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["make_whitelist"] is True  # from toml

    # Test with invalid config key in toml
    toml_file = io.StringIO("[tool.vulture]\ninvalid_key = 1\n")
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    toml_file = io.StringIO("[tool.vulture]\nmin_confidence = 'not_an_int'\n")
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths (should raise InputError)
    try:
        make_config(argv=["--min-confidence", "50"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths but no toml file
    config = make_config(argv=["test_file.py"])
    assert config["paths"] == ["test_file.py"]
    assert config["min_confidence"] == 0  # default

    # Test with explicit config file path in CLI that doesn't exist
    config = make_config(argv=["test_file.py", "--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["test_file.py"]
    assert config["min_confidence"] == 0  # defaults used

    # Test with help argument
    try:
        make_config(argv=["--help"])
        assert False, "Should have exited"
    except SystemExit:
        pass

    # Test with version argument
    try:
        make_config(argv=["--version"])
        assert False, "Should have exited"
    except SystemExit:
        pass
```


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    result = make_config(argv=[], tomlfile=None)
    assert result["config"] == "pyproject.toml"
    assert result["min_confidence"] == 0
    assert result["paths"] == []
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] == False
    assert result["sort_by_size"] == False
    assert result["verbose"] == False

    # Test with CLI arguments
    cli_args = ["--min-confidence", "50", "--verbose", "test_file.py"]
    result = make_config(argv=cli_args, tomlfile=None)
    assert result["min_confidence"] == 50
    assert result["verbose"] == True
    assert result["paths"] == ["test_file.py"]

    # Test with tomlfile
    toml_content = """
[tool.vulture]
min_confidence = 20
exclude = ["test_*.py"]
"""
    toml_file = io.StringIO(toml_content)
    result = make_config(argv=[], tomlfile=toml_file)
    assert result["min_confidence"] == 20
    assert result["exclude"] == ["test_*.py"]

    # Test that CLI args override tomlfile
    toml_file = io.StringIO(toml_content)
    result = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert result["min_confidence"] == 80

    # Test with missing paths raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test that make_whitelist and sort_by_size are set correctly
    result = make_config(argv=["--make-whitelist", "--sort-by-size"], tomlfile=None)
    assert result["make_whitelist"] == True
    assert result["sort_by_size"] == True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (should use defaults and fail due to no paths)
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test with paths provided via CLI
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["make_whitelist"] is False
    
    # Test with tomlfile
    import io
    toml_content = '''
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    '''
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "20", "cli_path"], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["cli_path"]
    
    # Test with invalid config key in TOML
    tomlfile = io.BytesIO(b'[tool.vulture]\ninvalid_key = true\npaths = ["path"]')
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with wrong type in TOML
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = "10"\npaths = ["path"]')
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with no paths in TOML
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = 10')
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with missing config file and CLI paths
    config = make_config(argv=["--config", "nonexistent.toml", "path1"])
    assert config["paths"] == ["path1"]
    assert config["config"] == "nonexistent.toml"
```


# LLM-generated content at query #15
#--------------------------

```python
def test_make_config():
    # Test with no args and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI args
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with toml file
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
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI args taking precedence over toml
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "90", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["verbose"] == True

    # Test with invalid toml config
    import pytest
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.StringIO(invalid_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with unknown CLI arg
    with pytest.raises(SystemExit):
        make_config(argv=["--unknown-arg"], tomlfile=None)

    # Test with no paths (should raise InputError)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with paths provided
    config = make_config(argv=["test.py"], tomlfile=None)
    assert config["paths"] == ["test.py"]

    # Test with exclude as comma-separated string
    config = make_config(argv=["test.py", "--exclude", "file1.py,file2.py"], tomlfile=None)
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with ignore-decorators as comma-separated string
    config = make_config(argv=["test.py", "--ignore-decorators", "deco1,deco2"], tomlfile=None)
    assert config["ignore_decorators"] == ["deco1", "deco2"]

    # Test with ignore-names as comma-separated string
    config = make_config(argv=["test.py", "--ignore-names", "name1,name2"], tomlfile=None)
    assert config["ignore_names"] == ["name1", "name2"]
```


# LLM-generated content at query #16
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
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
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = 10
    """)
    config = make_config(argv=["--min-confidence", "20", "path"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["path"]

    # Test defaults are applied
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    paths = ["test_path"]
    """)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test error with unknown configuration key
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    unknown_key = true
    paths = ["test"]
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test error with wrong type
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test"]
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)


# LLM-generated content at query #17
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config()
    assert config == DEFAULTS

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0

    # Test with tomlfile
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 50
    paths = ["path1", "path2"]
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["cli_path.py"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 50

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "20"])

    # Test with invalid config key in TOML
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path.py"]
    """)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)

    # Test with invalid config key in CLI
    with pytest.raises(InputError):
        make_config(argv=["--invalid-option"])

    # Test verbose mode prints configuration path
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config():
    # Test with valid TOML file and no CLI args
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
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI args overriding TOML
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py"]
    min_confidence = 10
    paths = ["path1"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(
        argv=["--min-confidence", "20", "--verbose", "path3"],
        tomlfile=toml_file
    )
    
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["paths"] == ["path3"]  # CLI overrides TOML
    assert config["exclude"] == ["file*.py"]  # TOML value preserved

    # Test with no TOML and CLI args only
    config = make_config(argv=["--min-confidence", "50", "test.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test.py"]
    assert config["config"] == "pyproject.toml"  # default value
    assert config["exclude"] == []
    assert config["verbose"] is False

    # Test with empty TOML and no CLI args raises error
    toml_file = io.BytesIO(b"")
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e.message)

    # Test with invalid config key in TOML
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test with wrong type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "10"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)

    # Test with make_whitelist and sort_by_size defaults
    config = make_config(argv=["test.py"])
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test with verbose output when TOML file is used
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    config = make_config(argv=[], tomlfile=toml_file)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    assert "Reading configuration from" in output
    assert "test.py" in output
    assert config["verbose"] is True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "80", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with TOML file
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test CLI arguments override TOML settings
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "95", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 95
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with invalid config key in TOML
    invalid_toml = """
    [tool.vulture]
    invalid_key = true
    paths = ["path1"]
    """
    tomlfile = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    tomlfile = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with CLI argument that has wrong type
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"], tomlfile=None)

    # Test with unknown CLI argument
    with pytest.raises(SystemExit):
        make_config(argv=["--unknown-arg"], tomlfile=None)

    # Test with --help
    with pytest.raises(SystemExit):
        make_config(argv=["--help"], tomlfile=None)

    # Test with --version
    with pytest.raises(SystemExit):
        make_config(argv=["--version"], tomlfile=None)

    # Test with custom config file path
    config = make_config(argv=["--config", "custom.toml"], tomlfile=None)
    assert config["config"] == "custom.toml"

    # Test with config file that doesn't exist
    config = make_config(argv=["--config", "nonexistent.toml", "path1"], tomlfile=None)
    assert config["paths"] == ["path1"]
```


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["-v", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
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
    config = make_config([], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test TOML with CLI override
    tomlfile = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(["--min-confidence", "50"])


# LLM-generated content at query #21
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config == DEFAULTS.copy()
    
    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    
    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    paths = ["path1", "path2"]
    min_confidence = 75
    exclude = ["file*.py"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] is True
    
    # Test CLI arguments override TOML settings
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["path1", "path2"]
    
    # Test defaults are applied for missing options
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    
    # Test validation errors
    import pytest
    
    # Test no paths provided
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "50"], tomlfile=io.BytesIO(b""))
    
    # Test invalid TOML config type
    bad_toml = """
    [tool.vulture]
    paths = "not_a_list"
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(bad_toml.encode()))
    
    # Test unknown configuration key
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path1"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(bad_toml.encode()))
    
    # Test CLI arguments with invalid type
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "abc", "path1"])
    
    # Test with paths from CLI but no TOML file
    config = make_config(argv=["path1"])
    assert config["paths"] == ["path1"]
    
    # Test with actual pyproject.toml file
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write("""
            [tool.vulture]
            paths = ["src"]
            min_confidence = 80
            """)
        config = make_config(argv=["--config", pyproject_path])
        assert config["paths"] == ["src"]
        assert config["min_confidence"] == 80
```


# LLM-generated content at query #22
#--------------------------

```python
def test_make_config():
    # Test with default values (no TOML file, no CLI args)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file and CLI arguments (CLI takes precedence)
    toml_content = b"""
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path"]
    verbose = true
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "70"], tomlfile=toml_file)
    assert config["min_confidence"] == 70
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] == True

    # Test with only TOML file
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] == True

    # Test with empty paths raises InputError
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test with invalid config key in TOML
    invalid_toml = b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path"]
    """
    toml_file = io.BytesIO(invalid_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    wrong_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    toml_file = io.BytesIO(wrong_type_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with --help (should not raise InputError)
    # This would normally print help and exit, but we can't test that easily
    # Instead, test that --version doesn't raise InputError
    config = make_config(argv=["--version"])
    assert config is not None

    # Test with verbose and TOML file (should print message)
    import contextlib
    import sys
    from io import StringIO
    toml_file = io.BytesIO(toml_content)
    captured_output = StringIO()
    with contextlib.redirect_stdout(captured_output):
        config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert "Reading configuration from" in captured_output.getvalue()
```


# LLM-generated content at query #23
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with tomlfile
    import io
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
    config = make_config([], io.StringIO(toml_data))
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments take precedence over tomlfile
    config = make_config(["cli_path", "--min-confidence", "90"], io.StringIO(toml_data))
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90

    # Test unknown configuration key raises InputError
    bad_toml = """
[tool.vulture]
unknown_option = true
"""
    try:
        make_config([], io.StringIO(bad_toml))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test wrong type in toml config raises InputError
    wrong_type_toml = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    try:
        make_config([], io.StringIO(wrong_type_toml))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test empty paths raises InputError
    try:
        make_config([], io.StringIO("[tool.vulture]\n"))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with existing pyproject.toml file
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write('[tool.vulture]\nmin_confidence = 25\npaths = ["test.py"]\n')
        config = make_config([], None)
        assert config["min_confidence"] == 25
        assert config["paths"] == ["test.py"]

    # Test --version and --help don't raise errors
    try:
        make_config(["--version"])
        make_config(["--help"])
    except SystemExit:
        pass  # argparse exits with SystemExit for --version and --help
```


# LLM-generated content at query #24
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config == DEFAULTS
    
    # Test with CLI arguments overriding defaults
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    exclude = ["file*.py"]
    ignore_decorators = ["deco1"]
    verbose = true
    paths = ["toml_path1"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config([], toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["verbose"] is True
    assert config["paths"] == ["toml_path1"]
    
    # Test CLI arguments override TOML settings
    toml_file = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "80"], toml_file)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py"]
    
    # Test with invalid TOML key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config([], toml_file)
        assert False, "Should raise InputError"
    except InputError:
        pass
    
    # Test with wrong type in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config([], toml_file)
        assert False, "Should raise InputError"
    except InputError:
        pass
    
    # Test with missing paths
    try:
        make_config(["--min-confidence", "10"])
        assert False, "Should raise InputError"
    except InputError:
        pass
    
    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 25
        paths = ["custom_path"]
        """)
        temp_path = f.name
    try:
        config = make_config(["--config", temp_path])
        assert config["min_confidence"] == 25
        assert config["paths"] == ["custom_path"]
    finally:
        os.unlink(temp_path)
    
    # Test with --version argument
    try:
        make_config(["--version"])
        assert False, "Should exit"
    except SystemExit:
        pass
    
    # Test with --help argument
    try:
        make_config(["--help"])
        assert False, "Should exit"
    except SystemExit:
        pass
    
    # Test with comma-separated values
    config = make_config(["--exclude", "file1.py,file2.py", "path"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    
    # Test with empty paths and config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        verbose = true
        """)
        temp_path = f.name
    try:
        try:
            make_config(["--config", temp_path])
            assert False, "Should raise InputError"
        except InputError:
            pass
    finally:
        os.unlink(temp_path)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_make_config():
    # Test with empty argv and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    
    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    
    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 10
    sort_by_size = true
    paths = ["path1", "path2"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test CLI precedence over toml
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90
    
    # Test with invalid config key in toml
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should raise InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong type in toml
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should raise InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)
    
    # Test with no paths
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should raise InputError for no paths"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
    
    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 30
        paths = ["test_path"]
        """)
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path], tomlfile=None)
        assert config["min_confidence"] == 30
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)
    
    # Test verbose output with toml file
    toml_content = """
    [tool.vulture]
    paths = ["test_path"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    import contextlib
    import io as io_module
    output = io_module.StringIO()
    with contextlib.redirect_stdout(output):
        config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert "Reading configuration from" in output.getvalue()


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments overriding TOML config
    import io
    import sys
    from unittest.mock import patch

    # Test 1: Basic TOML config with CLI override
    toml_data = b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["path1.py", "path2.py"]
    verbose = true
    """
    tomlfile = io.BytesIO(toml_data)
    
    with patch('vulture.config._parse_args') as mock_parse_args:
        mock_parse_args.return_value = {"min_confidence": 20}
        config = make_config(tomlfile=tomlfile)
        
        assert config["min_confidence"] == 20  # CLI overrides TOML
        assert config["paths"] == ["path1.py", "path2.py"]
        assert config["verbose"] is True
        assert config["exclude"] == []
        assert config["ignore_decorators"] == []
        assert config["ignore_names"] == []
        assert config["make_whitelist"] is False
        assert config["sort_by_size"] is False

    # Test 2: No TOML file, only CLI args
    with patch('vulture.config._parse_args') as mock_parse_args, \
         patch('pathlib.Path.is_file', return_value=False):
        mock_parse_args.return_value = {"paths": ["test_file.py"]}
        config = make_config(argv=["test_file.py"])
        
        assert config["paths"] == ["test_file.py"]
        assert config["min_confidence"] == 0
        assert config["config"] == "pyproject.toml"
        assert config["verbose"] is False

    # Test 3: TOML file detected from CLI config path
    toml_file_path = "custom_pyproject.toml"
    with patch('vulture.config._parse_args') as mock_parse_args, \
         patch('pathlib.Path.resolve') as mock_resolve, \
         patch('pathlib.Path.is_file', return_value=True), \
         patch('builtins.open', unittest.mock.mock_open(read_data=b"""
         [tool.vulture]
         min_confidence = 5
         paths = ["file1.py"]
         """)) as mock_open:
        mock_parse_args.return_value = {"config": toml_file_path}
        mock_resolve.return_value = pathlib.Path(toml_file_path)
        
        config = make_config(argv=["--config", toml_file_path])
        
        assert config["min_confidence"] == 5
        assert config["paths"] == ["file1.py"]
        assert config["config"] == toml_file_path

    # Test 4: InputError when no paths specified
    with patch('vulture.config._parse_args') as mock_parse_args:
        mock_parse_args.return_value = {}
        try:
            make_config(argv=[])
            assert False, "Should have raised InputError"
        except InputError as e:
            assert str(e) == "Please pass at least one file or directory"

    # Test 5: InputError for unknown configuration key
    toml_data = b"""
    [tool.vulture]
    invalid_key = true
    paths = ["test.py"]
    """
    tomlfile = io.BytesIO(toml_data)
    with patch('vulture.config._parse_args') as mock_parse_args:
        mock_parse_args.return_value = {"paths": ["test.py"]}
        try:
            make_config(tomlfile=tomlfile)
            assert False, "Should have raised InputError"
        except InputError as e:
            assert "Unknown configuration key" in str(e)

    # Test 6: InputError for wrong data type
    toml_data = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    tomlfile = io.BytesIO(toml_data)
    with patch('vulture.config._parse_args') as mock_parse_args:
        mock_parse_args.return_value = {"paths": ["test.py"]}
        try:
            make_config(tomlfile=tomlfile)
            assert False, "Should have raised InputError"
        except InputError as e:
            assert "Data type" in str(e)

    # Test 7: Verbose output with detected TOML file
    toml_data = b"""
    [tool.vulture]
    paths = ["test.py"]
    verbose = true
    """
    tomlfile = io.BytesIO(toml_data)
    with patch('vulture.config._parse_args') as mock_parse_args, \
         patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        mock_parse_args.return_value = {}
        config = make_config(tomlfile=tomlfile)
        
        assert "Reading configuration from" in mock_stdout.getvalue()
        assert config["verbose"] is True

    # Test 8: No verbose output when verbose is False
    toml_data = b"""
    [tool.vulture]
    paths = ["test.py"]
    verbose = false
    """
    tomlfile = io.BytesIO(toml_data)
    with patch('vulture.config._parse_args') as mock_parse_args, \
         patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
        mock_parse_args.return_value = {}
        config = make_config(tomlfile=tomlfile)
        
        assert "Reading configuration from" not in mock_stdout.getvalue()
        assert config["verbose"] is False

    # Test 9: Default values are set correctly
    with patch('vulture.config._parse_args') as mock_parse_args:
        mock_parse_args.return_value = {"paths": ["test.py"]}
        config = make_config(argv=["test.py"])
        
        assert config["min_confidence"] == 0
        assert config["exclude"] == []
        assert config["ignore_decorators"] == []
        assert config["ignore_names"] == []
        assert config["make_whitelist"] is False
        assert config["sort_by_size"] is False
        assert config["verbose"] is False

    # Test 10: CLI paths override TOML paths
    toml_data = b"""
    [tool.vulture]
    paths = ["toml_path.py"]
    """
    tomlfile = io.BytesIO(toml_data)
    with patch('vulture.config._parse_args') as mock_parse_args:
        mock_parse_args.return_value = {"paths": ["cli_path.py"]}
        config = make_config(tomlfile=tomlfile)
        
        assert config["paths"] == ["cli_path.py"]


# LLM-generated content at query #27
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments and toml file
    import io
    toml_content = """
[tool.vulture]
min_confidence = 30
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
ignore_names = ["name1"]
make_whitelist = true
sort_by_size = true
paths = ["toml_path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["toml_path1"]

    # Test that CLI arguments override toml file
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py"]  # from toml

    # Test with exclude as comma-separated string
    config = make_config(argv=["--exclude", "file1.py,file2.py", "path"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with ignore_decorators and ignore_names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "--ignore-names", "visit_*,do_*", "path"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test that missing paths raises InputError
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test with custom config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
min_confidence = 90
""")
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path, "test_path"])
        assert config["min_confidence"] == 90
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)

    # Test with invalid config key in toml
    bad_toml = io.BytesIO(b"""
[tool.vulture]
invalid_key = 5
paths = ["test"]
""")
    try:
        make_config(argv=[], tomlfile=bad_toml)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
```


# LLM-generated content at query #28
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    sort_by_size = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == []
    assert config["verbose"] is False

    # Test that CLI arguments override TOML settings
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80

    # Test that TOML settings are kept when CLI doesn't override
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["path3"], tomlfile=tomlfile)
    assert config["paths"] == ["path3"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

    # Test with non-existent config file
    config = make_config(argv=["--config", "nonexistent.toml"], tomlfile=None)
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == []

    # Test with InputError when no paths are provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)
```


# LLM-generated content at query #29
#--------------------------

```python
def test_make_config(tmp_path, capsys):
    # Test with no TOML file and no CLI args
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    
    # Test with CLI args
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test with TOML file
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
    toml_file = tmp_path / "test_config.toml"
    toml_file.write_text(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config([], f)
    
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test TOML + CLI precedence
    with open(toml_file, "rb") as f:
        config = make_config(["--min-confidence", "30"], f)
    
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    
    # Test verbose output
    with open(toml_file, "rb") as f:
        config = make_config(["--verbose"], f)
    captured = capsys.readouterr()
    assert f"Reading configuration from {toml_file}" in captured.out
    
    # Test error when no paths
    try:
        make_config([], tmp_path / "nonexistent.toml")
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e.message)
    
    # Test with default TOML file discovery
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
    [tool.vulture]
    min_confidence = 20
    paths = ["src"]
    """)
    
    config = make_config(["--config", str(pyproject)])
    assert config["min_confidence"] == 20
    assert config["paths"] == ["src"]


# LLM-generated content at query #30
#--------------------------

```python
def test_make_config():
    # Test with no arguments (defaults)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments take precedence over TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = 10
    """)
    config = make_config(argv=["--min-confidence", "20"], tomlfile=toml_file)
    assert config["min_confidence"] == 20

    # Test with missing paths (should raise InputError)
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "10"])

    # Test with unknown configuration key
    toml_file = io.StringIO("""
    [tool.vulture]
    unknown_key = "value"
    """)
    with pytest.raises(InputError):
        make_config(argv=["path"], tomlfile=toml_file)

    # Test with invalid data type
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError):
        make_config(argv=["path"], tomlfile=toml_file)
```


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["path1", "path2", "--min-confidence", "75", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 75
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    paths = ["src"]
    exclude = ["test*"]
    min_confidence = 50
    verbose = true
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config([], tomlfile)
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test*"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test TOML overridden by CLI
    tomlfile2 = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "90"], tomlfile2)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["src"]

    # Test with config file specified
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_data)
        fname = f.name
    try:
        config = make_config(["--config", fname])
        assert config["paths"] == ["src"]
        assert config["min_confidence"] == 50
    finally:
        os.unlink(fname)

    # Test InputError for unknown key
    import pytest
    bad_toml = """
    [tool.vulture]
    unknown_key = true
    """
    tomlfile3 = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile3)

    # Test InputError for wrong type
    bad_toml2 = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile4 = io.StringIO(bad_toml2)
    with pytest.raises(InputError):
        make_config([], tomlfile4)

    # Test InputError for no paths
    with pytest.raises(InputError):
        make_config([], io.StringIO(""))
```


# LLM-generated content at query #32
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding toml
    toml_data = b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    verbose = true
    """
    import io
    config = make_config(argv=["--min-confidence", "90"], tomlfile=io.BytesIO(toml_data))
    assert config["min_confidence"] == 90
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] is True

    # Test with toml file only
    config = make_config(argv=[], tomlfile=io.BytesIO(toml_data))
    assert config["min_confidence"] == 10
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] is True

    # Test with empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid config key in toml
    invalid_toml = b"""
    [tool.vulture]
    invalid_key = true
    paths = ["path"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(invalid_toml))

    # Test with wrong type in toml
    wrong_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(wrong_type_toml))

    # Test with wrong type in CLI
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"])


# LLM-generated content at query #33
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1.py", "path2.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test with toml file
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10

    # Test with CLI overriding toml
    toml_file = io.BytesIO(toml_content)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with invalid config key
    toml_content = b"""
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(toml_content)
    try:
        config = make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong config value type
    toml_content = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_content)
    try:
        config = make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with no paths
    try:
        config = make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths from toml
    toml_content = b"""
    [tool.vulture]
    paths = ["path1.py", "path2.py"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test with paths from CLI
    config = make_config(["test.py"])
    assert config["paths"] == ["test.py"]

    # Test with exclude as comma-separated string
    config = make_config(["--exclude", "file1.py,file2.py", "test.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with verbose flag
    config = make_config(["--verbose", "test.py"])
    assert config["verbose"] is True

    # Test with make-whitelist flag
    config = make_config(["--make-whitelist", "test.py"])
    assert config["make_whitelist"] is True

    # Test with sort-by-size flag
    config = make_config(["--sort-by-size", "test.py"])
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #34
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50

    # Test with a TOML file
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 75
    verbose = true
    paths = ["src"]
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(tomlfile=tomlfile)
    assert config["min_confidence"] == 75
    assert config["verbose"] is True
    assert config["paths"] == ["src"]

    # Test that CLI arguments override TOML options
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "90"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90

    # Test with invalid configuration key
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = true
    """)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong data type
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths
    try:
        make_config()
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with empty paths list
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
```


# LLM-generated content at query #35
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile providing configuration
    toml_content = b"""
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
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "-v"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test that paths are required
    try:
        make_config(argv=[])
        assert False, "Expected InputError for empty paths"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with config file path (create temporary pyproject.toml)
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "pyproject.toml")
        with open(toml_path, "wb") as f:
            f.write(toml_content)
        config = make_config(argv=["--config", toml_path])
        assert config["min_confidence"] == 10
        assert config["verbose"] is True

    # Test invalid configuration key raises InputError
    invalid_toml = b"""
    [tool.vulture]
    invalid_key = true
    """
    toml_file = io.BytesIO(invalid_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
```


# LLM-generated content at query #36
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["test.py", "--verbose"], tomlfile=None)
    assert config["paths"] == ["test.py"]
    assert config["verbose"] is True

    # Test with tomlfile
    import io
    import tomli_w
    toml_data = {
        "tool": {
            "vulture": {
                "exclude": ["file*.py", "dir/"],
                "ignore_decorators": ["deco1", "deco2"],
                "ignore_names": ["name1", "name2"],
                "make_whitelist": True,
                "min_confidence": 10,
                "sort_by_size": True,
                "verbose": True,
                "paths": ["path1", "path2"]
            }
        }
    }
    tomlfile = io.BytesIO(tomli_w.dumps(toml_data).encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments take precedence over tomlfile
    tomlfile = io.BytesIO(tomli_w.dumps(toml_data).encode())
    config = make_config(argv=["test.py", "--min-confidence", "50"], tomlfile=tomlfile)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 50

    # Test with invalid tomlfile
    invalid_toml = io.BytesIO(b"invalid toml content")
    with pytest.raises(Exception):
        make_config(argv=[], tomlfile=invalid_toml)

    # Test with no paths
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with unknown configuration key in tomlfile
    bad_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=bad_toml)

    # Test with wrong type in tomlfile
    wrong_type_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "not_an_int"')
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=wrong_type_toml)

    # Test with wrong type in CLI arguments
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"], tomlfile=None)

    # Test with config file that exists
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.vulture]\nmin_confidence = 20\n')
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path], tomlfile=None)
        assert config["min_confidence"] == 20
    finally:
        os.unlink(temp_path)

    # Test with verbose and tomlfile
    tomlfile = io.BytesIO(tomli_w.dumps(toml_data).encode())
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #37
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == DEFAULTS.copy()
    
    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    
    # Test with toml file
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src"]
    exclude = ["test_*.py"]
    min_confidence = 80
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 80
    
    # Test CLI arguments override toml file
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["custom_path", "--min-confidence", "90"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 90
    
    # Test with make-whitelist flag
    config = make_config(argv=["path", "--make-whitelist"])
    assert config["make_whitelist"] is True
    
    # Test with verbose flag and toml file
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True
    
    # Test with sort-by-size flag
    config = make_config(argv=["path", "--sort-by-size"])
    assert config["sort_by_size"] is True
    
    # Test with ignore decorators and names
    config = make_config(argv=["path", "--ignore-decorators", "dec1,dec2", "--ignore-names", "name1,name2"])
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test with exclude patterns
    config = make_config(argv=["path", "--exclude", "file1.py,file2.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    
    # Test with config file path
    config = make_config(argv=["path", "--config", "custom.toml"])
    assert config["config"] == "custom.toml"
    
    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
        make_config(argv=["--verbose"])
        make_config(tomlfile=io.BytesIO(b""))
```


# LLM-generated content at query #38
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
    config = make_config([])
    assert config == DEFAULTS
    
    # Test with CLI arguments
    config = make_config(['--min-confidence', '50', 'path1', 'path2'])
    assert config['min_confidence'] == 50
    assert config['paths'] == ['path1', 'path2']
    
    # Test with CLI arguments overriding defaults
    config = make_config(['--verbose', '--sort-by-size', 'test.py'])
    assert config['verbose'] is True
    assert config['sort_by_size'] is True
    assert config['paths'] == ['test.py']
    
    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 10
    paths = ["path1", "path2"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config['exclude'] == ["file*.py", "dir/"]
    assert config['min_confidence'] == 10
    assert config['paths'] == ["path1", "path2"]
    
    # Test with TOML file and CLI arguments (CLI should take precedence)
    toml_content = b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["path1", "path2"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(['--min-confidence', '90', 'cli_path'], tomlfile=toml_file)
    assert config['min_confidence'] == 90
    assert config['paths'] == ['cli_path']
    
    # Test with TOML file and CLI arguments for different keys
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py"]
    min_confidence = 10
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(['--sort-by-size', 'test.py'], tomlfile=toml_file)
    assert config['exclude'] == ["file*.py"]
    assert config['min_confidence'] == 10
    assert config['sort_by_size'] is True
    assert config['paths'] == ['test.py']
    
    # Test with no paths (should raise InputError)
    import pytest
    with pytest.raises(InputError):
        make_config([])
    
    # Test with paths but no other arguments
    config = make_config(['test.py'])
    assert config['paths'] == ['test.py']
    assert config['min_confidence'] == 0
    assert config['exclude'] == []
    assert config['ignore_decorators'] == []
    assert config['ignore_names'] == []
    assert config['make_whitelist'] is False
    assert config['sort_by_size'] is False
    assert config['verbose'] is False
```


# LLM-generated content at query #39
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src"]
    exclude = ["test*.py"]
    min_confidence = 20
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test*.py"]
    assert config["min_confidence"] == 20

    # Test CLI overrides toml
    toml_content = """
    [tool.vulture]
    paths = ["src"]
    min_confidence = 20
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["src"]
    assert config["min_confidence"] == 80

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with invalid config key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with actual pyproject.toml file
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("""
            [tool.vulture]
            paths = ["src"]
            exclude = ["test*.py"]
            """)
        config = make_config(argv=["--config", config_path])
        assert config["paths"] == ["src"]
        assert config["exclude"] == ["test*.py"]

    # Test verbose output with toml file
    toml_content = """
    [tool.vulture]
    paths = ["src"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    import contextlib
    from io import StringIO
    output = StringIO()
    with contextlib.redirect_stdout(output):
        config = make_config(argv=[], tomlfile=toml_file)
    assert "Reading configuration from" in output.getvalue()
```


# LLM-generated content at query #40
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with TOML file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = 10
    verbose = false
    """)
    config = make_config(argv=["--min-confidence", "50", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with invalid config key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "high"
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths (should raise error)
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with empty paths list in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    paths = []
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths provided via TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    paths = ["src"]
    """)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src"]
```


# LLM-generated content at query #41
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "80", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with tomlfile
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
    import io
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override TOML settings
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["custom.py", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["custom.py"]
    assert config["min_confidence"] == 90

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO(""))

    # Test with invalid config key in toml
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\ninvalid_key = true"))

    # Test with wrong type in toml
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\nverbose = 'yes'"))

    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("[tool.vulture]\nmin_confidence = 50\n")
        temp_path = f.name
    try:
        config = make_config(argv=[], )
        # This test will use the default pyproject.toml if it exists in cwd
        # For testing, we can't easily mock the file system, so just verify
        # that the function works with a temporary toml file
        config = make_config(argv=["--config", temp_path])
        assert config["min_confidence"] == 50
    finally:
        os.unlink(temp_path)

    # Test with invalid config file path
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\npaths = []"))
```


# LLM-generated content at query #42
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with tomlfile
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path1", "toml_path2"]
min_confidence = 30
exclude = ["exclude1", "exclude2"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["exclude1", "exclude2"]

    # Test CLI arguments override tomlfile
    toml_content = """
[tool.vulture]
paths = ["toml_path"]
min_confidence = 30
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "80"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80

    # Test CLI arguments with comma-separated values
    config = make_config(argv=["--exclude", "file1.py,file2.py", "--ignore-decorators", "deco1,deco2"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]

    # Test with missing paths raises InputError
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with unknown configuration key in toml
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=["path"], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=["path"], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test verbose output when reading from toml file
    import contextlib
    import io as io_module
    output = io_module.StringIO()
    toml_content = """
[tool.vulture]
paths = ["path"]
verbose = true
"""
    tomlfile = io.StringIO(toml_content)
    with contextlib.redirect_stdout(output):
        config = make_config(argv=[], tomlfile=tomlfile)
    assert "Reading configuration from" in output.getvalue()
```


# LLM-generated content at query #43
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (should use defaults and fail due to no paths)
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test with paths provided via CLI
    config = make_config(argv=["test_file.py"])
    assert config["paths"] == ["test_file.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    
    # Test with tomlfile provided
    import io
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
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["test_file.py"], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["test_file.py"]  # CLI takes precedence
    
    # Test that CLI arguments override TOML settings
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["test_file.py", "--min-confidence", "50"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    
    # Test with invalid config key in TOML
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(invalid_toml.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=toml_file)
    
    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(wrong_type_toml.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=toml_file)
    
    # Test with no paths in config
    no_paths_toml = """
    [tool.vulture]
    verbose = true
    """
    toml_file = io.BytesIO(no_paths_toml.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test with --version argument
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])
    
    # Test with --help argument
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])
```


# LLM-generated content at query #44
#--------------------------

```python
def test_make_config():
    # Test 1: Basic configuration with defaults
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test 2: Configuration from CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test 3: Configuration from TOML file
    toml_content = b'''
    [tool.vulture]
    paths = ["src", "tests"]
    min_confidence = 80
    exclude = ["*.pyc"]
    ignore_decorators = ["@app.route"]
    ignore_names = ["_private"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    '''
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src", "tests"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["*.pyc"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["_private"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test 4: CLI arguments have precedence over TOML
    toml_content = b'''
    [tool.vulture]
    paths = ["src"]
    min_confidence = 80
    '''
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test 5: Missing paths raises InputError
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError:
        pass

    # Test 6: Unknown configuration key in TOML
    toml_content = b'''
    [tool.vulture]
    invalid_key = "value"
    paths = ["src"]
    '''
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test 7: Wrong data type in TOML
    toml_content = b'''
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["src"]
    '''
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)

    # Test 8: No paths in TOML but paths in CLI
    toml_content = b'''
    [tool.vulture]
    min_confidence = 80
    '''
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["some_path"], tomlfile=toml_file)
    assert config["paths"] == ["some_path"]

    # Test 9: Default config file detection
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a pyproject.toml in temp directory
        toml_path = os.path.join(tmpdir, "pyproject.toml")
        with open(toml_path, "w") as f:
            f.write('[tool.vulture]\npaths = ["test_path"]\nmin_confidence = 30\n')
        
        # Change to temp directory and test
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = make_config(argv=[])
            assert config["paths"] == ["test_path"]
            assert config["min_confidence"] == 30
        finally:
            os.chdir(old_cwd)

    # Test 10: Custom config path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_toml = os.path.join(tmpdir, "custom.toml")
        with open(custom_toml, "w") as f:
            f.write('[tool.vulture]\npaths = ["custom_path"]\n')
        
        config = make_config(argv=["--config", custom_toml])
        assert config["paths"] == ["custom_path"]
```


# LLM-generated content at query #45
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
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
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["custom_path", "--min-confidence", "80"], tomlfile=tomlfile)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value retained

    # Test with invalid config key in TOML
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.StringIO(invalid_toml)
    try:
        make_config(argv=["path"], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.StringIO(wrong_type_toml)
    try:
        make_config(argv=["path"], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test missing paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
```


# LLM-generated content at query #46
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
    toml_content = """
[tool.vulture]
min_confidence = 30
exclude = ["test_*.py"]
paths = ["src"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src"]

    # Test CLI arguments override toml settings
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "70"], tomlfile=toml_file)
    assert config["min_confidence"] == 70

    # Test with paths from toml and CLI combined
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]

    # Test with unknown configuration key in toml
    invalid_toml = """
[tool.vulture]
invalid_key = true
"""
    toml_file = io.BytesIO(invalid_toml.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    wrong_type_toml = """
[tool.vulture]
min_confidence = "high"
"""
    toml_file = io.BytesIO(wrong_type_toml.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths given
    try:
        make_config(argv=[])
        assert False, "Should raise InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)


# LLM-generated content at query #47
#--------------------------

```python
def test_make_config():
    # Test with TOML file (via io.StringIO)
    import io
    toml_data = '''
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    '''
    tomlfile = io.StringIO(toml_data)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    tomlfile = io.StringIO("""
    [tool.vulture]
    paths = ["toml_path"]
    min_confidence = 5
    """)
    config = make_config(argv=["--min-confidence", "50", "cli_path"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 50

    # Test defaults are applied for missing options
    config = make_config(argv=["test_path"])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == ["test_path"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test InputError when paths is empty
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test InputError for invalid configuration key
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    with pytest.raises(InputError):
        make_config(argv=["test_path"], tomlfile=tomlfile)

    # Test InputError for wrong type
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError):
        make_config(argv=["test_path"], tomlfile=tomlfile)

    # Test that CLI args with missing sentinel value are not included
    config = make_config(argv=["--verbose", "test_path"])
    assert config["verbose"] is True
    assert "config" not in config  # config uses default value, not CLI

    # Test that --config is respected
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.toml') as f:
        f.write("""
        [tool.vulture]
        paths = ["from_config_file"]
        """)
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["paths"] == ["from_config_file"]
    finally:
        os.unlink(temp_path)
```


# LLM-generated content at query #48
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--verbose", "--min-confidence", "50", "--paths", "test1.py", "test2.py"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test1.py", "test2.py"]

    # Test with tomlfile
    import io
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
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override tomlfile
    tomlfile = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]

    # Test with empty tomlfile
    tomlfile = io.StringIO("")
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test with invalid configuration key in tomlfile
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    """
    tomlfile = io.StringIO(invalid_toml)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in tomlfile
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "invalid"
    paths = ["test.py"]
    """
    tomlfile = io.StringIO(wrong_type_toml)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with --version argument (should not raise InputError)
    import sys
    config = make_config(["--version"])
    # The version action exits the program, so we shouldn't reach here
    # If we do, something went wrong
    assert False, "Should have exited with version" 
```


# LLM-generated content at query #49
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file only
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 80
    paths = ["toml_path1", "toml_path2"]
    exclude = ["test_*.py"]
    sort_by_size = true
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["test_*.py"]
    assert config["sort_by_size"] is True

    # Test with both toml and CLI (CLI should override)
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90
    assert config["exclude"] == ["test_*.py"]
    assert config["sort_by_size"] is True

    # Test with invalid config key in toml
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config(argv=["path"], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config(argv=["path"], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "must be" in str(e)

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with config file path that doesn't exist
    config = make_config(argv=["path", "--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["path"]

    # Test with various CLI options
    config = make_config(argv=[
        "path",
        "--exclude", "test_*.py,*.bak",
        "--ignore-decorators", "@app.route,@decorator",
        "--ignore-names", "private_*,_internal",
        "--make-whitelist",
        "--sort-by-size",
        "--min-confidence", "75"
    ])
    assert config["exclude"] == ["test_*.py", "*.bak"]
    assert config["ignore_decorators"] == ["@app.route", "@decorator"]
    assert config["ignore_names"] == ["private_*", "_internal"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["min_confidence"] == 75

    # Test with verbose output
    import io
    tomlfile = io.StringIO("""
    [tool.vulture]
    verbose = true
    paths = ["test_path"]
    """)
    import contextlib
    import sys
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        config = make_config(argv=[], tomlfile=tomlfile)
    assert "Reading configuration from" in stdout_capture.getvalue()

    # Test default config path detection
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Create a pyproject.toml
            with open("pyproject.toml", "w") as f:
                f.write("""
                [tool.vulture]
                min_confidence = 30
                paths = ["from_default_config"]
                """)
            config = make_config(argv=[])
            assert config["min_confidence"] == 30
            assert config["paths"] == ["from_default_config"]
        finally:
            os.chdir(old_cwd)

    # Test that CLI overrides default config
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("pyproject.toml", "w") as f:
                f.write("""
                [tool.vulture]
                min_confidence = 30
                paths = ["from_default_config"]
                """)
            config = make_config(argv=["cli_path", "--min-confidence", "45"])
            assert config["min_confidence"] == 45
            assert config["paths"] == ["cli_path"]
        finally:
            os.chdir(old_cwd)


# LLM-generated content at query #50
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults and fail on paths)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with CLI arguments overriding defaults
    config = make_config(argv=[
        "--min-confidence", "50",
        "--exclude", "test_*.py,venv",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "path1.py"
    ])
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py"]
    
    # Test with toml file
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path1.py", "toml_path2.py"]
min_confidence = 10
exclude = ["excluded*.py"]
ignore_decorators = ["deco1", "deco2"]
ignore_names = ["name1", "name2"]
make_whitelist = true
sort_by_size = true
verbose = true
"""
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path1.py", "toml_path2.py"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["excluded*.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    
    # Test with toml file and CLI args overriding toml settings
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "20", "cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["excluded*.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    
    # Test with invalid toml config (unknown key)
    toml_file = io.StringIO("""
[tool.vulture]
invalid_key = "value"
paths = ["path.py"]
""")
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test with invalid toml config (wrong type)
    toml_file = io.StringIO("""
[tool.vulture]
min_confidence = "not_an_int"
paths = ["path.py"]
""")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test with invalid CLI config (wrong type)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "path.py"])
    
    # Test with unknown CLI key
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["--unknown-option", "path.py"])
```


# LLM-generated content at query #51
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "75", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 75
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 50
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/"]

    # Test CLI overrides TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["src/"]

    # Test with make_whitelist flag
    config = make_config(argv=["--make-whitelist"])
    assert config["make_whitelist"] is True

    # Test with sort_by_size flag
    config = make_config(argv=["--sort-by-size"])
    assert config["sort_by_size"] is True

    # Test with ignore_decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with ignore_names
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with exclude
    config = make_config(argv=["--exclude", "*settings.py,docs,*/test_*.py,venv"])
    assert config["exclude"] == ["*settings.py", "docs", "*/test_*.py", "venv"]

    # Test that paths are required
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid config key in TOML
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=toml_file)


# LLM-generated content at query #52
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding toml file
    toml_content = b'''
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    verbose = false
    '''
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "75"], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] is False

    # Test with toml file only
    toml_file2 = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file2)
    assert config["min_confidence"] == 10
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] is False

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test boolean flags in toml file
    toml_content_bool = b'''
    [tool.vulture]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    paths = ["path"]
    '''
    toml_file3 = io.BytesIO(toml_content_bool)
    config = make_config(argv=[], tomlfile=toml_file3)
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with exclude and ignore patterns
    toml_content_patterns = b'''
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["@app.route", "@require_*"]
    ignore_names = ["visit_*", "do_*"]
    paths = ["path"]
    '''
    toml_file4 = io.BytesIO(toml_content_patterns)
    config = make_config(argv=[], tomlfile=toml_file4)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with CLI exclude
    config = make_config(argv=["--exclude", "file1.py,file2.py", "path"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["path"]

    # Test with CLI ignore decorators and names
    config = make_config(argv=["--ignore-decorators", "@dec1,@dec2", "--ignore-names", "name1,name2", "path"])
    assert config["ignore_decorators"] == ["@dec1", "@dec2"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test with unknown config key in toml raises InputError
    toml_content_unknown = b'''
    [tool.vulture]
    unknown_key = "value"
    paths = ["path"]
    '''
    toml_file5 = io.BytesIO(toml_content_unknown)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file5)

    # Test with wrong type in toml raises InputError
    toml_content_wrong_type = b'''
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    '''
    toml_file6 = io.BytesIO(toml_content_wrong_type)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file6)


# LLM-generated content at query #53
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    paths = ["path1", "path2"]
    min_confidence = 75
    exclude = ["test_*.py", "docs"]
    verbose = true
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["verbose"] is True

    # Test that CLI arguments override TOML settings
    toml_content = b"""
    [tool.vulture]
    paths = ["toml_path"]
    min_confidence = 10
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90

    # Test with invalid TOML configuration
    toml_content = b"""
    [tool.vulture]
    invalid_key = true
    """
    toml_file = io.BytesIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with no paths raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with custom config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b"""
        [tool.vulture]
        paths = ["custom_path"]
        """)
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["paths"] == ["custom_path"]
    finally:
        os.unlink(temp_path)

    # Test with --exclude as comma-separated values
    config = make_config(argv=["path", "--exclude", "a.py,b.py"])
    assert config["exclude"] == ["a.py", "b.py"]

    # Test with --ignore-decorators and --ignore-names
    config = make_config(argv=["path", "--ignore-decorators", "@deco1,@deco2", 
                               "--ignore-names", "name1,name2"])
    assert config["ignore_decorators"] == ["@deco1", "@deco2"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test with --make-whitelist and --sort-by-size flags
    config = make_config(argv=["path", "--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #54
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config()
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_data = io.StringIO("""
[tool.vulture]
min_confidence = 30
sort_by_size = true
paths = ["src", "tests"]
""")
    config = make_config(tomlfile=toml_data)
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src", "tests"]

    # Test CLI overrides TOML
    toml_data = io.StringIO("""
[tool.vulture]
min_confidence = 30
""")
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_data)
    assert config["min_confidence"] == 80

    # Test with invalid TOML key
    toml_data = io.StringIO("""
[tool.vulture]
invalid_key = 10
""")
    try:
        make_config(tomlfile=toml_data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    toml_data = io.StringIO("""
[tool.vulture]
min_confidence = "high"
""")
    try:
        make_config(tomlfile=toml_data)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with empty paths
    try:
        make_config(argv=["--exclude", "foo"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with csv conversion
    config = make_config(argv=["--exclude", "a,b,c", "path"])
    assert config["exclude"] == ["a", "b", "c"]

    # Test verbose with TOML file
    toml_data = io.StringIO("""
[tool.vulture]
verbose = true
paths = ["src"]
""")
    config = make_config(tomlfile=toml_data)
    assert config["verbose"] is True
```


# LLM-generated content at query #55
#--------------------------

```python
def test_make_config():
    # Test with default config file path (no tomlfile passed)
    import tempfile
    import os
    
    # Test 1: No CLI args, no tomlfile - should use defaults
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    
    # Test 2: CLI args override defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    
    # Test 3: CLI args with paths
    config = make_config(argv=["test_file.py", "another_file.py"])
    assert config["paths"] == ["test_file.py", "another_file.py"]
    
    # Test 4: TOML config file
    toml_content = b"""
[tool.vulture]
paths = ["path1", "path2"]
min_confidence = 30
verbose = true
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        toml_path = f.name
    
    try:
        with open(toml_path, "rb") as toml_file:
            config = make_config(argv=[], tomlfile=toml_file)
        assert config["paths"] == ["path1", "path2"]
        assert config["min_confidence"] == 30
        assert config["verbose"] == True
    finally:
        os.unlink(toml_path)
    
    # Test 5: CLI args override TOML config
    toml_content = b"""
[tool.vulture]
min_confidence = 30
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        toml_path = f.name
    
    try:
        with open(toml_path, "rb") as toml_file:
            config = make_config(argv=["--min-confidence", "70"], tomlfile=toml_file)
        assert config["min_confidence"] == 70
    finally:
        os.unlink(toml_path)
    
    # Test 6: Error when no paths are provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test 7: Error with unknown configuration key in TOML
    toml_content = b"""
[tool.vulture]
unknown_key = "value"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        toml_path = f.name
    
    try:
        with open(toml_path, "rb") as toml_file:
            with pytest.raises(InputError):
                make_config(argv=[], tomlfile=toml_file)
    finally:
        os.unlink(toml_path)
    
    # Test 8: Error with wrong type in TOML
    toml_content = b"""
[tool.vulture]
min_confidence = "not_an_int"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        toml_path = f.name
    
    try:
        with open(toml_path, "rb") as toml_file:
            with pytest.raises(InputError):
                make_config(argv=[], tomlfile=toml_file)
    finally:
        os.unlink(toml_path)
    
    # Test 9: Custom config file path
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_config = os.path.join(tmpdir, "custom.toml")
        with open(custom_config, "w") as f:
            f.write("[tool.vulture]\nmin_confidence = 25\n")
        
        config = make_config(argv=["--config", custom_config, "test.py"])
        assert config["min_confidence"] == 25
        assert config["paths"] == ["test.py"]
    
    # Test 10: Test with all CLI options
    config = make_config(argv=[
        "test.py",
        "--exclude", "*.pyc,test_*.py",
        "--ignore-decorators", "@app.route,@login_required",
        "--ignore-names", "private_*,_internal",
        "--make-whitelist",
        "--min-confidence", "75",
        "--sort-by-size",
        "--verbose"
    ])
    assert config["paths"] == ["test.py"]
    assert config["exclude"] == ["*.pyc", "test_*.py"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    assert config["ignore_names"] == ["private_*", "_internal"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 75
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    
    # Test 11: Test that verbose prints configuration message when TOML is used
    import contextlib
    import io
    
    toml_content = b"""
[tool.vulture]
paths = ["test_path"]
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        toml_path = f.name
    
    try:
        with open(toml_path, "rb") as toml_file:
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                config = make_config(argv=["--verbose"], tomlfile=toml_file)
            assert "Reading configuration from" in stdout_capture.getvalue()
    finally:
        os.unlink(toml_path)


# LLM-generated content at query #56
#--------------------------

```python
def test_make_config():
    # Test with minimal arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "path1", "path2", "--verbose", "--sort-by-size"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["sort_by_size"] is True

    # Test with TOML file input
    import io
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
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding TOML settings
    toml_data = """
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]

    # Test that missing paths raises an error
    try:
        make_config(argv=[])
        assert False, "Expected InputError for missing paths"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with unknown configuration key in TOML
    toml_data = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError for unknown key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong data type in TOML
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with --version (should not raise InputError even without paths)
    config = make_config(argv=["--version"])
    assert config["paths"] == []

    # Test with --help (should not raise InputError even without paths)
    config = make_config(argv=["--help"])
    assert config["paths"] == []
```


# LLM-generated content at query #57
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }
    
    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    
    # Test with TOML file
    toml_content = b"""
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
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["verbose"] is True  # from TOML
    
    # Test with missing paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)
    
    # Test with unknown config key in TOML
    bad_toml = b"""
    [tool.vulture]
    unknown_key = true
    paths = ["path1"]
    """
    tomlfile = io.BytesIO(bad_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
```


# LLM-generated content at query #58
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with TOML file provided
    toml_content = b'''
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    '''
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "50", "--paths", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["cli_path"]
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value preserved

    # Test with CLI arguments only
    config = make_config(argv=["--min-confidence", "30", "path1", "path2"])
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []

    # Test with invalid TOML key
    invalid_toml = b'''
    [tool.vulture]
    invalid_key = "value"
    paths = ["path"]
    '''
    toml_file = io.BytesIO(invalid_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with no paths when paths is required
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError for no paths"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
```


# LLM-generated content at query #59
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file (should use defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "file1.py", "dir/"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["file1.py", "dir/"]

    # Test with toml file
    toml_data = b'''
[tool.vulture]
paths = ["src/"]
min_confidence = 75
exclude = ["test_*", "docs"]
ignore_decorators = ["@app.route"]
ignore_names = ["internal_*"]
make_whitelist = true
sort_by_size = true
verbose = true
'''
    import io
    tomlfile = io.BytesIO(toml_data)
    config = make_config([], tomlfile=tomlfile)
    assert config["paths"] == ["src/"]
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test_*", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["internal_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test CLI overrides toml
    tomlfile = io.BytesIO(b'[tool.vulture]\npaths = ["src/"]\nmin_confidence = 75\n')
    config = make_config(["--min-confidence", "25", "myfile.py"], tomlfile=tomlfile)
    assert config["paths"] == ["myfile.py"]
    assert config["min_confidence"] == 25

    # Test with unknown configuration key in toml
    tomlfile = io.BytesIO(b'[tool.vulture]\ninvalid_key = true\n')
    try:
        make_config([], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"\n')
    try:
        make_config([], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with config file specified but not found
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = make_config(["--config", "nonexistent.toml", "file.py"])
            assert config["paths"] == ["file.py"]
            assert config["min_confidence"] == 0
        finally:
            os.chdir(old_cwd)

    # Test with config file found
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("pyproject.toml", "wb") as f:
                f.write(b'[tool.vulture]\nmin_confidence = 90\n')
            config = make_config(["file.py"])
            assert config["min_confidence"] == 90
            assert config["paths"] == ["file.py"]
        finally:
            os.chdir(old_cwd)
```


# LLM-generated content at query #60
#--------------------------

```python
def test_make_config():
    # Test with no CLI args and no TOML file (should raise InputError due to no paths)
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e.message)

    # Test with CLI args providing paths and no TOML file
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file content
    import io
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 50
    exclude = ["exclude1", "exclude2"]
    ignore_decorators = ["dec1", "dec2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["exclude1", "exclude2"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test that CLI args override TOML values
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "75"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["exclude1", "exclude2"]

    # Test with invalid TOML config key
    toml_content_invalid = """
    [tool.vulture]
    paths = ["test"]
    invalid_key = "value"
    """
    toml_file = io.StringIO(toml_content_invalid)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key: invalid_key" in str(e.message)

    # Test with wrong type in TOML config
    toml_content_wrong_type = """
    [tool.vulture]
    paths = ["test"]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(toml_content_wrong_type)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)

    # Test with --version flag (should not raise InputError)
    config = make_config(argv=["--version"], tomlfile=None)
    assert config is not None

    # Test with --help flag (should not raise InputError)
    config = make_config(argv=["--help"], tomlfile=None)
    assert config is not None

    # Test with non-existent config file
    config = make_config(argv=["test.py", "--config", "nonexistent.toml"], tomlfile=None)
    assert config["paths"] == ["test.py"]
    assert config["config"] == "nonexistent.toml"
```


# LLM-generated content at query #61
#--------------------------

```python
def test_make_config():
    # Test with no arguments - should use defaults and detect pyproject.toml
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(["--min-confidence", "50", "--verbose", "file1.py", "file2.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["file1.py", "file2.py"]

    # Test with TOML file
    import io
    import tomli_w
    
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    exclude = ["test_*.py", "docs"]
    ignore_decorators = ["@app.route"]
    ignore_names = ["private_*"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    paths = ["src", "lib"]
    """
    
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["private_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["src", "lib"]

    # Test that CLI arguments override TOML values
    toml_file = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80", "--paths", "custom_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["custom_path"]

    # Test that paths in TOML are used when no CLI paths given
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["paths"] == ["src", "lib"]

    # Test error when no paths provided
    with pytest.raises(InputError):
        config = make_config(["--min-confidence", "50"])

    # Test error with unknown configuration key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = true
    paths = ["test"]
    """
    toml_file = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        config = make_config([], tomlfile=toml_file)

    # Test error with wrong type in TOML
    bad_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test"]
    """
    toml_file = io.StringIO(bad_type_toml)
    with pytest.raises(InputError):
        config = make_config([], tomlfile=toml_file)

    # Test with specific config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_data)
        temp_path = f.name
    
    try:
        config = make_config(["--config", temp_path])
        assert config["min_confidence"] == 30
        assert config["paths"] == ["src", "lib"]
    finally:
        os.unlink(temp_path)

    # Test verbose output when config file is detected
    toml_file = io.StringIO(toml_data)
    config = make_config(["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True
```


# LLM-generated content at query #62
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["cli_path"], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["cli_path"]  # CLI overrides TOML

    # Test CLI overrides TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(
        argv=["cli_path", "--min-confidence", "20", "--verbose"],
        tomlfile=toml_file,
    )
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value preserved

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid TOML key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1"]
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with no TOML file and default config
    config = make_config(argv=["path1"])
    assert config["config"] == "pyproject.toml"
    assert config["paths"] == ["path1"]


# LLM-generated content at query #63
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override toml file
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = 10
    paths = ["path1", "path2"]
    """)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with invalid input config
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "string"
    """)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with no paths
    try:
        make_config(["--min-confidence", "50"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #64
#--------------------------

```python
def test_make_config():
    # Test basic functionality with no arguments
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(["--exclude", "test.py,foo.py", "--min-confidence", "50"])
    assert config["exclude"] == ["test.py", "foo.py"]
    assert config["min_confidence"] == 50

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 30
    paths = ["src", "tests"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src", "tests"]

    # Test CLI overrides TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with missing paths
    try:
        make_config([])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with unknown config key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(bad_toml)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "high"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(wrong_type_toml)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with paths provided via CLI
    config = make_config(["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test with verbose mode
    config = make_config(["test.py", "--verbose"])
    assert config["verbose"] == True

    # Test with make_whitelist
    config = make_config(["test.py", "--make-whitelist"])
    assert config["make_whitelist"] == True

    # Test with sort_by_size
    config = make_config(["test.py", "--sort-by-size"])
    assert config["sort_by_size"] == True

    # Test with ignore_decorators and ignore_names
    config = make_config(["--ignore-decorators", "@app.route,@require_*", 
                          "--ignore-names", "visit_*,do_*", "test.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["test.py"]

    # Test with invalid config file path
    config = make_config(["--config", "nonexistent.toml", "test.py"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
```


# LLM-generated content at query #65
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 20
    exclude = ["file*.py", "dir/"]
    paths = ["src/"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["paths"] == ["src/"]
    assert config["verbose"] is False

    # Test that CLI args override tomlfile
    toml_content = """
    [tool.vulture]
    min_confidence = 20
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80

    # Test with invalid configuration key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong data type
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

    # Test with no paths (should raise InputError)
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths provided
    config = make_config(argv=["test_file.py"], tomlfile=None)
    assert config["paths"] == ["test_file.py"]
```


# LLM-generated content at query #66
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file (should use defaults)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    exclude = ["file1.py", "file2.py"]
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["src"]

    # Test with both TOML and CLI (CLI should override)
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["src"]

    # Test with invalid configuration key
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with no paths (should raise error)
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths provided
    config = make_config(argv=["some_file.py"])
    assert config["paths"] == ["some_file.py"]
```


# LLM-generated content at query #67
#--------------------------

```python
def test_make_config():
    # Test with no configuration file and no CLI arguments
    config = make_config(argv=[], tomlfile=None)
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with a TOML file
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
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding TOML
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "50", "script.py"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["script.py"]
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value preserved

    # Test with CLI arguments only
    config = make_config(argv=["--verbose", "--sort-by-size", "test.py"], tomlfile=None)
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0  # default

    # Test with a custom config file path
    tmp_path = pathlib.Path("temp_config.toml")
    tmp_path.write_text(toml_content)
    config = make_config(argv=["--config", str(tmp_path)], tomlfile=None)
    assert config["min_confidence"] == 10
    tmp_path.unlink()

    # Test error when no paths provided
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test error with unknown configuration key
    tomlfile = io.StringIO("[tool.vulture]\nunknown_key = 5\npaths = ['test.py']")
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: unknown_key"
```


# LLM-generated content at query #68
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with TOML file
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
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["--exclude", "test1.py,test2.py", "--min-confidence", "50"], tomlfile=toml_file)
    assert config["exclude"] == ["test1.py", "test2.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments without TOML
    config = make_config(argv=["--exclude", "test1.py,test2.py", "--min-confidence", "50"], tomlfile=None)
    assert config["exclude"] == ["test1.py", "test2.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == []

    # Test with paths argument
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with invalid TOML key
    toml_content_with_bad_key = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(toml_content_with_bad_key.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_content_with_bad_type = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_content_with_bad_type.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in CLI args
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"], tomlfile=None)
```


# LLM-generated content at query #69
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--verbose", "--min-confidence", "80", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments take precedence over toml
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "50"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with missing paths raises InputError
    try:
        make_config(argv=["--verbose"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with unknown configuration key in toml
    toml_data_invalid = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_data_invalid)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    toml_data_wrong_type = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_data_wrong_type)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with --version flag (should not raise InputError)
    config = make_config(argv=["--version"])
    assert config["paths"] == []

    # Test with --help flag (should not raise InputError)
    config = make_config(argv=["--help"])
    assert config["paths"] == []
```


# LLM-generated content at query #70
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with paths provided via CLI
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=[
        "--exclude", "*.py,test_*.py",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--min-confidence", "50",
        "--sort-by-size",
        "--verbose",
        "path1", "path2"
    ])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["*.py", "test_*.py"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with toml file
    toml_content = b"""
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
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test TOML overridden by CLI
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    paths = ["toml_path"]
    min_confidence = 10
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(
        argv=["--exclude", "*.py", "--min-confidence", "20", "cli_path"],
        tomlfile=toml_file
    )
    assert config["paths"] == ["cli_path"]
    assert config["exclude"] == ["*.py"]
    assert config["min_confidence"] == 20

    # Test with invalid TOML config key
    toml_content = b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1"]
    """
    toml_file = io.BytesIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with invalid TOML config value type
    toml_content = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    toml_file = io.BytesIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with invalid CLI argument type
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int", "path1"])

    # Test with unknown CLI argument
    with pytest.raises(SystemExit):
        make_config(argv=["--unknown-arg", "path1"])

    # Test with invalid config file path
    config = make_config(argv=["--config", "non_existent.toml", "path1"])
    assert config["paths"] == ["path1"]

    # Test with no paths (should raise InputError)
    with pytest.raises(InputError):
        make_config(argv=[])
```


# LLM-generated content at query #71
#--------------------------

```python
def test_make_config():
    # Test with empty CLI arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments and toml file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 20
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    paths = ["toml_path1", "toml_path2"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80  # CLI takes precedence
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test with CLI override of toml options
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--exclude", "custom*.py", "--no-make-whitelist"], tomlfile=tomlfile)
    assert config["exclude"] == ["custom*.py"]
    assert config["make_whitelist"] is False

    # Test with invalid toml configuration
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.StringIO(invalid_toml)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test with wrong type in toml
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.StringIO(wrong_type_toml)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e.message)

    # Test with no paths
    try:
        make_config(argv=[], tomlfile=io.StringIO(""))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e.message)

    # Test with paths from CLI
    config = make_config(argv=["/some/path"])
    assert config["paths"] == ["/some/path"]

    # Test with paths from toml
    tomlfile = io.StringIO("[tool.vulture]\npaths = ['/toml/path']")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["/toml/path"]

    # Test with --version (should not raise error)
    try:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        make_config(argv=["--version"])
        sys.stdout = old_stdout
    except SystemExit:
        pass  # Expected exit for --version
    finally:
        sys.stdout = old_stdout

    # Test with --help (should not raise error)
    try:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        make_config(argv=["--help"])
        sys.stdout = old_stdout
    except SystemExit:
        pass  # Expected exit for --help
    finally:
        sys.stdout = old_stdout
```


# LLM-generated content at query #72
#--------------------------

```python
def test_make_config():
    # Test with minimal CLI arguments (no config file)
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(
        argv=[
            "path1",
            "--exclude", "test_*.py,docs",
            "--ignore-decorators", "@app.route,@require_*",
            "--ignore-names", "visit_*,do_*",
            "--make-whitelist",
            "--min-confidence", "50",
            "--sort-by-size",
            "--verbose",
        ]
    )
    assert config["paths"] == ["path1"]
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with a TOML file
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
    import io
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(
        argv=["cli_path", "--min-confidence", "90"],
        tomlfile=tomlfile
    )
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90
    assert config["exclude"] == ["file*.py", "dir/"]  # from TOML

    # Test missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "10"])

    # Test unknown configuration key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path"]
    """
    tomlfile = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test wrong type in TOML
    bad_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    tomlfile = io.BytesIO(bad_type_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test config file detection from CLI
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "pyproject.toml")
        with open(toml_path, "w") as f:
            f.write("""
            [tool.vulture]
            min_confidence = 25
            paths = ["toml_path"]
            """)
        config = make_config(argv=["--config", toml_path])
        assert config["min_confidence"] == 25
        assert config["paths"] == ["toml_path"]

    # Test default config file in current directory (if exists)
    import pathlib
    if pathlib.Path("pyproject.toml").exists():
        config = make_config(argv=[])
        assert "paths" in config
        assert config["paths"] != []  # from the actual pyproject.toml

    # Test verbose output with TOML detection
    tomlfile = io.BytesIO(toml_data.encode())
    import contextlib
    import sys
    with contextlib.redirect_stdout(io.StringIO()) as output:
        config = make_config(argv=["--verbose"], tomlfile=tomlfile)
        assert "Reading configuration from" in output.getvalue()
```


# LLM-generated content at query #73
#--------------------------

```python
def test_make_config():
    # Test basic config with no arguments
    config = make_config()
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "80", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 50
    paths = ["src/"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/"]

    # Test CLI arguments override TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["src/"]

    # Test with --exclude comma-separated values
    config = make_config(argv=["--exclude", "a.py,b.py,c.py"])
    assert config["exclude"] == ["a.py", "b.py", "c.py"]

    # Test with --ignore-decorators comma-separated values
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names comma-separated values
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --make-whitelist
    config = make_config(argv=["--make-whitelist"])
    assert config["make_whitelist"] is True

    # Test with --sort-by-size
    config = make_config(argv=["--sort-by-size"])
    assert config["sort_by_size"] is True

    # Test with --config option
    config = make_config(argv=["--config", "custom.toml"])
    assert config["config"] == "custom.toml"
```


# LLM-generated content at query #74
#--------------------------

```python
def test_make_config():
    # Test with no arguments (empty CLI args)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["src", "tests"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["src", "tests"]

    # Test CLI arguments override TOML settings
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src", "tests"]

    # Test with missing paths raises InputError
    try:
        make_config(argv=["--min-confidence", "50"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with unknown configuration key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = true
    paths = ["src"]
    """
    toml_file = io.BytesIO(bad_toml.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    bad_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["src"]
    """
    toml_file = io.BytesIO(bad_type_toml.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)
```


# LLM-generated content at query #75
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (uses defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "test.py", "dir/"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["test.py", "dir/"]

    # Test with TOML file content
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override TOML options
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "30", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["verbose"] == True

    # Test with no paths (should raise InputError)
    try:
        make_config([])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with unknown configuration key
    tomlfile = io.BytesIO(b'[tool.vulture]\nunknown_key = "value"\n')
    try:
        make_config([], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type for configuration value
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = "not_an_int"\n')
    try:
        make_config([], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)


# LLM-generated content at query #76
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (should fail due to no paths)
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test with valid paths via CLI
    config = make_config(argv=["test_file.py"])
    assert config["paths"] == ["test_file.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] is False
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    
    # Test with CLI arguments overriding defaults
    config = make_config(argv=[
        "test_file.py",
        "--min-confidence", "50",
        "--exclude", "test1.py,test2.py",
        "--verbose",
        "--make-whitelist",
        "--sort-by-size",
        "--ignore-decorators", "@deco1,@deco2",
        "--ignore-names", "name1,name2"
    ])
    assert config["paths"] == ["test_file.py"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test1.py", "test2.py"]
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["ignore_decorators"] == ["@deco1", "@deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["config.py", "settings.py"]
    min_confidence = 30
    verbose = true
    paths = ["path1.py", "path2.py"]
    """
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["config.py", "settings.py"]
    assert config["min_confidence"] == 30
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py", "path2.py"]
    
    # Test that CLI arguments take precedence over tomlfile
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    paths = ["path1.py"]
    verbose = false
    """
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(argv=["cli_path.py", "--min-confidence", "70", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 70
    assert config["paths"] == ["cli_path.py"]
    assert config["verbose"] is True
    
    # Test with invalid TOML key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1.py"]
    """
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with wrong type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1.py"]
    """
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with wrong type in CLI
    with pytest.raises(InputError):
        make_config(argv=["test.py", "--min-confidence", "not_an_int"])
    
    # Test with unknown CLI argument
    with pytest.raises(SystemExit):
        make_config(argv=["test.py", "--unknown-flag"])
    
    # Test that missing paths raises error even with valid toml
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    """
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with empty paths list in TOML
    toml_content = """
    [tool.vulture]
    paths = []
    """
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
```


# LLM-generated content at query #77
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config == DEFAULTS
    
    # Test with TOML file provided
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["test*.py"]
    min_confidence = 50
    """
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == []
    
    # Test with CLI arguments overriding TOML
    toml_data = """
    [tool.vulture]
    min_confidence = 50
    sort_by_size = true
    """
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["--min-confidence", "80", "path/to/file.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path/to/file.py"]
    
    # Test with CLI arguments only
    config = make_config(argv=["--exclude", "test*.py,*.bak", "src"])
    assert config["exclude"] == ["test*.py", "*.bak"]
    assert config["paths"] == ["src"]
    
    # Test with default config file
    config = make_config(argv=["--config", "nonexistent.toml", "file.py"])
    assert config["paths"] == ["file.py"]
    
    # Test with invalid config key
    toml_data = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong type
    toml_data = """
    [tool.vulture]
    min_confidence = "high"
    """
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)
    
    # Test with no paths
    toml_data = """
    [tool.vulture]
    exclude = ["test*.py"]
    """
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
    
    # Test verbose output
    toml_data = """
    [tool.vulture]
    verbose = true
    """
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["file.py"], tomlfile=toml_file)
    assert config["verbose"] is True
```


# LLM-generated content at query #78
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "10", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 10
    assert config["verbose"] is True

    # Test with tomlfile only
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with both CLI and tomlfile (CLI takes precedence)
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path.py", "--min-confidence", "20"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] is True

    # Test error when no paths are provided
    try:
        make_config(argv=[], tomlfile=io.BytesIO(b""))
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with unknown configuration key in toml
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["test.py"]
    """
    try:
        make_config(argv=[], tomlfile=io.BytesIO(bad_toml.encode()))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    bad_toml_type = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    try:
        make_config(argv=[], tomlfile=io.BytesIO(bad_toml_type.encode()))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)
```


# LLM-generated content at query #79
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["paths"] == []

    # Test with CLI arguments
    config = make_config(argv=["test.py", "--min-confidence", "50"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 50

    # Test with tomlfile
    import io
    toml_data = '''
    [tool.vulture]
    paths = ["test1.py", "test2.py"]
    min_confidence = 30
    exclude = ["test_exclude.py"]
    '''
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["test1.py", "test2.py"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test_exclude.py"]

    # Test CLI overrides TOML
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["cli.py", "--min-confidence", "70"], tomlfile=tomlfile)
    assert config["paths"] == ["cli.py"]
    assert config["min_confidence"] == 70

    # Test defaults are applied
    config = make_config(argv=["test.py"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test unknown configuration key in TOML
    toml_data = '''
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    '''
    tomlfile = io.StringIO(toml_data)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test wrong type in TOML
    toml_data = '''
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    '''
    tomlfile = io.StringIO(toml_data)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test wrong type in CLI args
    with pytest.raises(InputError):
        make_config(argv=["test.py", "--min-confidence", "not_an_int"])

    # Test verbose prints config path
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)

    # Test with actual pyproject.toml file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('''
        [tool.vulture]
        paths = ["test.py"]
        min_confidence = 25
        ''')
        temp_path = f.name
    config = make_config(argv=["--config", temp_path])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 25
    import os
    os.unlink(temp_path)

    # Test with empty config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('''
        [tool.vulture]
        paths = ["test.py"]
        ''')
        temp_path = f.name
    config = make_config(argv=["--config", temp_path])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    os.unlink(temp_path)

    # Test all CLI options
    config = make_config(argv=[
        "test.py",
        "--exclude", "file1.py,file2.py",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "--make-whitelist",
        "--min-confidence", "40",
        "--sort-by-size",
        "--verbose"
    ])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 40
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
```


# LLM-generated content at query #80
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["--verbose", "--min-confidence", "50", "file1.py", "file2.py"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["file1.py", "file2.py"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py", "temp/"]
    min_confidence = 30
    sort_by_size = true
    paths = ["src/", "lib/"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py", "temp/"]
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src/", "lib/"]

    # Test CLI overrides TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "custom.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["custom.py"]
    assert config["exclude"] == ["test_*.py", "temp/"]  # TOML values preserved
    assert config["sort_by_size"] is True

    # Test with unknown configuration key raises InputError
    import pytest
    toml_file = io.StringIO("""
    [tool.vulture]
    unknown_key = true
    """)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)

    # Test with wrong type raises InputError
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)

    # Test with no paths raises InputError
    with pytest.raises(InputError):
        make_config(argv=["--verbose"])
```


# LLM-generated content at query #81
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([], None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "test.py"], None)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 75
    paths = ["src", "tests"]
    exclude = ["*.pyc"]
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["src", "tests"]
    assert config["exclude"] == ["*.pyc"]

    # Test that CLI arguments override TOML
    tomlfile = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "90"], tomlfile)
    assert config["min_confidence"] == 90

    # Test with paths from CLI overriding TOML paths
    tomlfile = io.StringIO(toml_data)
    config = make_config(["custom.py"], tomlfile)
    assert config["paths"] == ["custom.py"]

    # Test with multiple CLI arguments
    config = make_config(
        ["--exclude", "file1.py,file2.py", "--ignore-decorators", "@deco1,@deco2",
         "--ignore-names", "name1,name2", "--make-whitelist", "--sort-by-size",
         "--min-confidence", "30", "--verbose", "path1", "path2"],
        None
    )
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["@deco1", "@deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["min_confidence"] == 30
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config([], None)

    # Test that invalid TOML config raises InputError
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    with pytest.raises(InputError):
        make_config([], tomlfile)
```


# LLM-generated content at query #82
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML config file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["toml_path1", "toml_path2"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test CLI arguments override TOML config
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["cli_path", "--min-confidence", "20", "--verbose"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML values preserved

    # Test defaults are applied when neither CLI nor TOML provides values
    config = make_config(argv=[], tomlfile=io.StringIO(""))
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO(""))

    # Test error for unknown configuration key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = true
    paths = ["path1"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO(bad_toml))

    # Test error for wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO(wrong_type_toml))


# LLM-generated content at query #83
#--------------------------

```python
def test_make_config():
    # Test with no CLI args and no toml file (should use defaults and fail on empty paths)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test with paths provided via CLI
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with CLI args overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "--make-whitelist", "path"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    assert config["paths"] == ["path"]
    
    # Test with toml file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    paths = ["toml_path1", "toml_path2"]
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["paths"] == ["toml_path1", "toml_path2"]
    
    # Test CLI args override toml values
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "90", "cli_path"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["cli_path"]
    assert config["exclude"] == ["file*.py", "dir/"]  # toml values preserved
    
    # Test with invalid toml key
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    paths = ["test"]
    """)
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with wrong type in toml
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test"]
    """)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        paths = ["config_file_path"]
        """)
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["paths"] == ["config_file_path"]
    finally:
        os.unlink(temp_path)
```


# LLM-generated content at query #84
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with TOML file
    toml_data = b"""
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
    tomlfile = io.BytesIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_data = b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["from_toml"]
    """
    tomlfile = io.BytesIO(toml_data)
    config = make_config(argv=["--min-confidence", "50", "from_cli"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["from_cli"]

    # Test CLI arguments without TOML
    config = make_config(argv=["--min-confidence", "50", "--verbose", "test.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["test.py"]

    # Test with --config pointing to non-existent file
    config = make_config(argv=["--config", "nonexistent.toml", "test.py"])
    assert config["paths"] == ["test.py"]

    # Test that InputError is raised when no paths are given
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError:
        pass

    # Test with invalid config key in TOML
    toml_data = b"""
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_data)
    try:
        make_config(argv=["test.py"], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass
```


# LLM-generated content at query #85
#--------------------------

```python
def test_make_config():
    # Test with no arguments (should use defaults and fail due to no paths)
    import pytest
    with pytest.raises(InputError):
        make_config([])
    
    # Test with CLI arguments only
    config = make_config(["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with CLI arguments overriding defaults
    config = make_config([
        "--min-confidence", "50",
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "path1.py"
    ])
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py"]
    
    # Test with TOML file
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
paths = ["path1.py", "path2.py"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py", "path2.py"]
    
    # Test with TOML file and CLI overrides
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "80", "custom_path.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["custom_path.py"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] is True
    
    # Test with invalid TOML config key
    bad_toml = """
[tool.vulture]
invalid_key = true
paths = ["path1.py"]
"""
    toml_file = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)
    
    # Test with invalid TOML config type
    bad_type_toml = """
[tool.vulture]
min_confidence = "high"
paths = ["path1.py"]
"""
    toml_file = io.BytesIO(bad_type_toml.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)
    
    # Test with no paths in config
    no_paths_toml = """
[tool.vulture]
min_confidence = 10
"""
    toml_file = io.BytesIO(no_paths_toml.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)
    
    # Test with custom config path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as tmp:
        tmp.write(toml_content.encode())
        tmp_path = tmp.name
    try:
        config = make_config(["--config", tmp_path])
        assert config["min_confidence"] == 10
        assert config["paths"] == ["path1.py", "path2.py"]
    finally:
        os.unlink(tmp_path)
    
    # Test verbose output
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True
    
    # Test with non-existent config file
    config = make_config(["--config", "non_existent.toml", "test_path.py"])
    assert config["paths"] == ["test_path.py"]
    assert config["min_confidence"] == 0
```


# LLM-generated content at query #86
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (uses defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(["--min-confidence", "50", "--verbose", "file1.py", "file2.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["file1.py", "file2.py"]

    # Test with tomlfile providing configuration
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 75
    exclude = ["test_*.py"]
    verbose = true
    paths = ["src/"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile=tomlfile)
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

    # Test that CLI arguments take precedence over tomlfile
    toml_content = """
    [tool.vulture]
    min_confidence = 75
    paths = ["src/"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "90"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["src/"]

    # Test with all options set via CLI
    config = make_config([
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--config", "custom.toml",
        "path1", "path2"
    ])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["path1", "path2"]

    # Test that missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(["--min-confidence", "10"])

    # Test that unknown configuration keys in tomlfile raise InputError
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["file.py"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=tomlfile)

    # Test that wrong types in tomlfile raise InputError
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["file.py"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=tomlfile)
```


# LLM-generated content at query #87
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config == DEFAULTS
    with pytest.raises(InputError):
        make_config(argv=["--config", "nonexistent.toml"])

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "test_path"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test_path"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []

    # Test with tomlfile
    from io import StringIO
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
    tomlfile = StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    tomlfile = StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with invalid toml config
    bad_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with wrong data types in toml
    bad_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = StringIO(bad_type_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test no paths provided
    with pytest.raises(InputError):
        make_config(argv=["--verbose"])


# LLM-generated content at query #88
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (should use defaults)
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["make_whitelist"] is False

    # Test with tomlfile provided
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

    # Test CLI arguments take precedence over tomlfile
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] is True

    # Test with missing paths raises InputError
    try:
        make_config(argv=["--min-confidence", "50"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with unknown configuration key in tomlfile
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path1"]
    """
    tomlfile = io.StringIO(bad_toml)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in tomlfile
    bad_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    tomlfile = io.StringIO(bad_toml)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type" in str(e)
```


# LLM-generated content at query #89
#--------------------------

```python
def test_make_config():
    # Test with no arguments (should use defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]

    # Test with TOML file and missing options get defaults
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with unknown configuration key raises InputError
    toml_data = """
    [tool.vulture]
    unknown_key = "value"
    """
    toml_file = io.StringIO(toml_data)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type raises InputError
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(toml_data)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths raises InputError
    try:
        make_config([])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths from TOML
    toml_data = """
    [tool.vulture]
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["paths"] == ["path1"]


# LLM-generated content at query #90
#--------------------------

```python
def test_make_config(tmp_path):
    # Test with no arguments and no toml file (should use defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 10
    sort_by_size = true
    paths = ["toml_path1", "toml_path2"]
    """
    config = make_config([], tomlfile=__import__('io').BytesIO(toml_content))
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test that CLI arguments take precedence over toml
    toml_content = b"""
    [tool.vulture]
    min_confidence = 10
    """
    config = make_config(["--min-confidence", "80"], tomlfile=__import__('io').BytesIO(toml_content))
    assert config["min_confidence"] == 80

    # Test reading from actual pyproject.toml file
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
    [tool.vulture]
    min_confidence = 30
    """)
    config = make_config(["--config", str(pyproject)])
    assert config["min_confidence"] == 30

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config([])

    # Test with unknown config key in toml
    toml_content = b"""
    [tool.vulture]
    unknown_key = true
    """
    with pytest.raises(InputError):
        make_config([], tomlfile=__import__('io').BytesIO(toml_content))
```


# LLM-generated content at query #91
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile - should use defaults
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with tomlfile
    import io
    toml_content = """
[tool.vulture]
min_confidence = 75
exclude = ["test*.py", "docs/"]
ignore_decorators = ["@app.route"]
ignore_names = ["private_*"]
make_whitelist = true
sort_by_size = true
verbose = true
paths = ["src", "tests"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test*.py", "docs/"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["private_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["src", "tests"]

    # Test CLI arguments override TOML settings
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "20", "--verbose", "path3"], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["paths"] == ["path3"]
    assert config["exclude"] == ["test*.py", "docs/"]  # from TOML

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with invalid TOML config raises InputError
    invalid_toml = """
[tool.vulture]
min_confidence = "invalid"
"""
    tomlfile = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with unknown configuration key in TOML
    unknown_key_toml = """
[tool.vulture]
unknown_key = true
"""
    tomlfile = io.BytesIO(unknown_key_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)


# LLM-generated content at query #92
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    with pytest.raises(InputError):
        make_config(argv=["--verbose"], tomlfile=None)

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False

    # Test with toml file
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
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["config"] == "pyproject.toml"

    # Test CLI overrides toml
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with invalid toml config
    invalid_toml = """
    [tool.vulture]
    invalid_key = 5
    """
    tomlfile = io.StringIO(invalid_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test missing paths
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\n"))

    # Test with actual pyproject.toml file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("[tool.vulture]\npaths = ['test.py']\n")
        fname = f.name
    config = make_config(argv=["--config", fname])
    assert config["paths"] == ["test.py"]
    import os
    os.unlink(fname)

    # Test with non-existent pyproject.toml
    config = make_config(argv=["--config", "non_existent_file.toml"])
    assert config == DEFAULTS

    # Test --version and --help don't raise errors
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])
```


# LLM-generated content at query #93
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with tomlfile
    toml_content = b"""
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
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding toml
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = 10\npaths = ["toml_path"]')
    argv = ["--min-confidence", "20", "--verbose", "cli_path.py"]
    config = make_config(argv=argv, tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["paths"] == ["cli_path.py"]

    # Test with CLI arguments only
    argv = ["--exclude", "test_*.py,docs", "--sort-by-size", "file.py"]
    config = make_config(argv=argv)
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["sort_by_size"] is True
    assert config["paths"] == ["file.py"]

    # Test with custom config file path
    tmp_path = pathlib.Path(tmpdir)
    config_file = tmp_path / "custom_config.toml"
    config_file.write_text('[tool.vulture]\nmin_confidence = 50\n')
    argv = ["--config", str(config_file), "some_path.py"]
    config = make_config(argv=argv)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["some_path.py"]

    # Test error when no paths provided
    try:
        make_config(argv=["--verbose"])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test error with unknown config key in toml
    tomlfile = io.BytesIO(b'[tool.vulture]\nunknown_key = true\n')
    try:
        make_config(tomlfile=tomlfile, argv=["path.py"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test error with wrong type in toml
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = "10"\n')
    try:
        make_config(tomlfile=tomlfile, argv=["path.py"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)


# LLM-generated content at query #94
#--------------------------

```python
def test_make_config():
    # Test with TOML file
    import io
    import tempfile
    import os
    
    # Test basic TOML config
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 50
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src"]
    assert config["verbose"] is True
    
    # Test CLI args override TOML
    toml_file2 = io.StringIO(toml_content)
    config2 = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file2)
    assert config2["min_confidence"] == 80
    assert config2["verbose"] is True
    
    # Test defaults are set
    config3 = make_config(argv=["test_path"], tomlfile=io.StringIO(""))
    assert config3["exclude"] == []
    assert config3["make_whitelist"] is False
    assert config3["sort_by_size"] is False
    assert config3["verbose"] is False
    assert config3["ignore_decorators"] == []
    assert config3["ignore_names"] == []
    
    # Test with actual file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        paths = ["test_dir"]
        ignore_names = ["_private"]
        """)
        temp_path = f.name
    
    try:
        config4 = make_config(argv=[])
        # Since temp file is not named pyproject.toml, it won't be auto-detected
        assert config4["paths"] == []
        
        # Test with explicit config path
        config5 = make_config(argv=["--config", temp_path])
        assert config5["paths"] == ["test_dir"]
        assert config5["ignore_names"] == ["_private"]
    finally:
        os.unlink(temp_path)
    
    # Test error for no paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e.message)
    
    # Test csv parsing for excludes
    config6 = make_config(argv=["--exclude", "file1.py,file2.py", "src"])
    assert config6["exclude"] == ["file1.py", "file2.py"]
    
    # Test unknown configuration key
    toml_bad = io.StringIO("""
    [tool.vulture]
    unknown_key = true
    """)
    try:
        make_config(argv=["test_path"], tomlfile=toml_bad)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)


# LLM-generated content at query #95
#--------------------------

```python
def test_make_config(monkeypatch, tmp_path):
    # Test with explicit tomlfile and argv
    toml_content = b"""
    [tool.vulture]
    exclude = ["test_pattern"]
    min_confidence = 50
    paths = ["test_path"]
    """
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(argv=["--verbose"], tomlfile=f)
    
    assert config["exclude"] == ["test_pattern"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test_path"]
    assert config["verbose"] is True
    
    # Test CLI precedence over TOML
    with open(toml_file, "rb") as f:
        config = make_config(argv=["--min-confidence", "75", "--verbose"], tomlfile=f)
    
    assert config["min_confidence"] == 75
    assert config["verbose"] is True
    
    # Test defaults applied for missing keys
    with open(toml_file, "rb") as f:
        config = make_config(argv=[], tomlfile=f)
    
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    
    # Test with no tomlfile and no config file present
    config = make_config(argv=["test_path"])
    assert config["paths"] == ["test_path"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    
    # Test InputError for missing paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test InputError for unknown config key
    bad_toml = b"""
    [tool.vulture]
    nonexistent_key = true
    paths = ["test_path"]
    """
    bad_file = tmp_path / "bad_config.toml"
    bad_file.write_bytes(bad_toml)
    with open(bad_file, "rb") as f:
        with pytest.raises(InputError):
            make_config(argv=[], tomlfile=f)


# LLM-generated content at query #96
#--------------------------

```python
def test_make_config():
    # Test that CLI args take precedence over TOML config
    toml_data = b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["path1.py"]
    verbose = false
    """
    import io
    toml_file = io.BytesIO(toml_data)
    
    config = make_config(argv=["--min-confidence", "20", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py"]
    
    # Test that defaults are applied for missing options
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test that paths are required
    import pytest
    with pytest.raises(InputError) as exc_info:
        make_config(argv=[])
    assert "at least one file or directory" in str(exc_info.value)
    
    # Test that invalid config key raises error
    invalid_toml = b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(invalid_toml)
    with pytest.raises(InputError) as exc_info:
        make_config(argv=["--verbose"], tomlfile=toml_file)
    assert "Unknown configuration key" in str(exc_info.value)
    
    # Test that wrong type raises error
    wrong_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(wrong_type_toml)
    with pytest.raises(InputError) as exc_info:
        make_config(argv=["--verbose"], tomlfile=toml_file)
    assert "Data type for min_confidence must be 'int'" in str(exc_info.value)
    
    # Test that verbose prints config path
    import io as io_module
    toml_file = io_module.BytesIO(b"[tool.vulture]\nverbose = true\npaths = ['test.py']")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]


# LLM-generated content at query #97
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile - should use defaults
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with tomlfile providing configuration
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
    # Note: tomllib needs binary mode, so wrap in BytesIO
    import io
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override tomlfile
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "50", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml

    # Test CLI arguments without tomlfile
    config = make_config(["--paths", "test.py", "--min-confidence", "80"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 80

    # Test with invalid config file (should raise InputError)
    import pytest
    tomlfile = io.BytesIO(b"unknown_key = 1")
    with pytest.raises(InputError):
        make_config([], tomlfile=tomlfile)

    # Test with no paths (should raise InputError)
    with pytest.raises(InputError):
        make_config([])

    # Test with wrong type in toml file
    tomlfile = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'")
    with pytest.raises(InputError):
        make_config([], tomlfile=tomlfile)
```


# LLM-generated content at query #98
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
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
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["override_path", "--min-confidence", "30"], tomlfile=toml_file)
    assert config["paths"] == ["override_path"]
    assert config["min_confidence"] == 30

    # Test with invalid config key
    toml_file = io.BytesIO(b"[tool.vulture]\ninvalid_key = 5\n")
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

    # Test with wrong type
    toml_file = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'high'\n")
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"

    # Test with no paths
    try:
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\n"))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"
```


# LLM-generated content at query #99
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "80", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 50
    paths = ["src", "tests"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src", "tests"]

    # Test CLI overrides TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["custom_path", "--min-confidence", "90"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 90

    # Test with invalid TOML config
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test wrong type in TOML config
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with missing paths
    try:
        make_config(argv=["--min-confidence", "50"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
```


# LLM-generated content at query #100
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 75
    exclude = ["test_*.py"]
    paths = ["src/"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src/"]

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "30"], tomlfile=tomlfile)
    assert config["min_confidence"] == 30

    # Test with --help raises SystemExit
    import pytest
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])

    # Test with --version raises SystemExit
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])

    # Test with invalid config key in toml
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with wrong type in toml
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "high"
    """
    tomlfile = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with no paths raises InputError
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with paths in both toml and CLI
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path.py"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path.py"]

    # Test with verbose and tomlfile prints message
    import contextlib
    import sys
    verbose_toml = """
    [tool.vulture]
    verbose = true
    paths = ["src/"]
    """
    tomlfile = io.BytesIO(verbose_toml.encode())
    with contextlib.redirect_stdout(io.StringIO()) as output:
        config = make_config(argv=[], tomlfile=tomlfile)
    assert "Reading configuration from" in output.getvalue()

    # Test with sort_by_size and make_whitelist
    config = make_config(argv=["--sort-by-size", "--make-whitelist", "file.py"])
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True

    # Test with exclude and ignore patterns
    config = make_config(
        argv=["--exclude", "test_*.py,docs", "--ignore-decorators", "@app.route",
              "--ignore-names", "visit_*", "file.py"]
    )
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
```


# LLM-generated content at query #101
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["test.py"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["test.py"], tomlfile=tomlfile)
    assert config["paths"] == ["test.py"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["test.py", "--min-confidence", "50"], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with no paths raises error
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with TOML that has no paths
    tomlfile = io.BytesIO(b"[tool.vulture]\nmin_confidence = 20\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with invalid config key in TOML
    tomlfile = io.BytesIO(b"[tool.vulture]\ninvalid_key = 1\n")
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=tomlfile)

    # Test with wrong type in TOML
    tomlfile = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'high'\n")
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=tomlfile)

    # Test with wrong type in CLI
    with pytest.raises(InputError):
        make_config(argv=["test.py", "--min-confidence", "high"])

    # Test with unknown CLI argument
    with pytest.raises(SystemExit):
        make_config(argv=["test.py", "--unknown-arg"])

    # Test with verbose and TOML file (should print message)
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["test.py", "--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #102
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50

    # Test with toml file
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    exclude = ["file1.py", "file2.py"]
    paths = ["src", "tests"]
    """
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["src", "tests"]

    # Test with CLI arguments overriding toml file
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "70"], tomlfile=tomlfile)
    assert config["min_confidence"] == 70
    assert config["paths"] == ["src", "tests"]

    # Test with invalid toml config key
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(invalid_toml.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError:
        pass

    # Test with invalid toml config type
    invalid_type_toml = """
    [tool.vulture]
    min_confidence = "string"
    """
    tomlfile = io.BytesIO(invalid_type_toml.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError:
        pass

    # Test with no paths
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError:
        pass

    # Test with verbose and toml file
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["-v"], tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #103
#--------------------------

```python
def test_make_config(tmp_path, monkeypatch, capsys):
    # Test with no arguments and no toml file
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] is False
    assert config["sort_by_size"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "test_file.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["test_file.py"]

    # Test with toml file
    toml_content = """
    [tool.vulture]
    exclude = ["exclude1.py", "exclude2.py"]
    min_confidence = 30
    paths = ["path1.py", "path2.py"]
    """
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text(toml_content)
    with open(toml_file, "rb") as f:
        config = make_config([], tomlfile=f)
    assert config["exclude"] == ["exclude1.py", "exclude2.py"]
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test CLI args override toml
    with open(toml_file, "rb") as f:
        config = make_config(["--min-confidence", "80"], tomlfile=f)
    assert config["min_confidence"] == 80

    # Test verbose output with toml
    with open(toml_file, "rb") as f:
        config = make_config(["--verbose"], tomlfile=f)
        captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out

    # Test with config file path in CLI
    config_file = tmp_path / "custom_config.toml"
    config_file.write_text("[tool.vulture]\nmin_confidence = 45\n")
    config = make_config(["--config", str(config_file)])
    assert config["min_confidence"] == 45

    # Test error when no paths given
    with pytest.raises(InputError):
        make_config([])

    # Test error with invalid config key
    toml_content_bad = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file_bad = tmp_path / "bad_config.toml"
    toml_file_bad.write_text(toml_content_bad)
    with open(toml_file_bad, "rb") as f:
        with pytest.raises(InputError):
            make_config(["--config", str(toml_file_bad)], tomlfile=f)

    # Test error with wrong data type
    toml_content_wrong_type = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file_wrong = tmp_path / "wrong_type.toml"
    toml_file_wrong.write_text(toml_content_wrong_type)
    with open(toml_file_wrong, "rb") as f:
        with pytest.raises(InputError):
            make_config(["--config", str(toml_file_wrong)], tomlfile=f)

    # Test default values are set correctly
    config = make_config(["test.py"])
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with paths in CLI
    config = make_config(["path1.py", "path2.py", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["verbose"] is True
```


# LLM-generated content at query #104
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file (uses defaults)
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file only
    toml_content = b'''
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    '''
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments take precedence over TOML
    toml_file = io.BytesIO(b'''
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    ''')
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nmin_confidence = 10'))

    # Test with unknown configuration key raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nunknown_key = 10'))

    # Test with wrong data type raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nmin_confidence = "not_an_int"'))

    # Test with --version (should not raise even without paths)
    config = make_config(argv=["--version"], tomlfile=None)
    assert config == {}

    # Test with --help (should not raise even without paths)
    config = make_config(argv=["--help"], tomlfile=None)
    assert config == {}

    # Test with a real TOML file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.vulture]\nmin_confidence = 30\npaths = ["test_path"]')
        temp_path = f.name
    try:
        config = make_config(argv=[], tomlfile=None)
        # Note: This will use the default config file path, not the temp file
        # So we need to pass the config path explicitly
        config = make_config(argv=["--config", temp_path], tomlfile=None)
        assert config["min_confidence"] == 30
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)

    # Test with config file that doesn't exist
    config = make_config(argv=["--config", "nonexistent.toml"], tomlfile=None)
    assert config["config"] == "nonexistent.toml"
    # Should use defaults
    assert config["min_confidence"] == 0
    assert config["paths"] == []

    # Test with empty paths and no config file
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with paths provided in CLI but not in TOML
    config = make_config(argv=["cli_path"], tomlfile=io.BytesIO(b'[tool.vulture]\nmin_confidence = 5'))
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 5

    # Test with multiple CLI arguments
    config = make_config(argv=["path1", "path2", "--exclude", "*.py,test_*", "--ignore-names", "foo,bar",
                               "--ignore-decorators", "@deco1,@deco2", "--make-whitelist", "--sort-by-size"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["*.py", "test_*"]
    assert config["ignore_names"] == ["foo", "bar"]
    assert config["ignore_decorators"] == ["@deco1", "@deco2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (uses defaults)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    sort_by_size = true
    paths = ["path1", "path2"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test with invalid configuration key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong data type
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths provided
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with config file path that doesn't exist
    config = make_config(argv=["path1", "--config", "nonexistent.toml"])
    assert config["paths"] == ["path1"]
    assert config["config"] == "nonexistent.toml"

    # Test with --exclude and --ignore-decorators CSV parsing
    config = make_config(argv=["path1", "--exclude", "a.py,b.py", "--ignore-decorators", "@app.route,@require_*"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --make-whitelist and --sort-by-size flags
    config = make_config(argv=["path1", "--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #2
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test with TOML file
    import io
    import tempfile
    
    toml_content = """
    [tool.vulture]
    paths = ["src", "tests"]
    min_confidence = 80
    exclude = ["*.pyc"]
    ignore_decorators = ["@staticmethod"]
    ignore_names = ["_private"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    """
    
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src", "tests"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["*.pyc"]
    assert config["ignore_decorators"] == ["@staticmethod"]
    assert config["ignore_names"] == ["_private"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test CLI arguments override TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "30"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["*.pyc"]
    assert config["make_whitelist"] is True

    # Test with actual file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        tmp_path = f.name
    
    config = make_config(argv=[], tomlfile=open(tmp_path, 'rb'))
    assert config["paths"] == ["src", "tests"]
    assert config["min_confidence"] == 80
    
    import os
    os.unlink(tmp_path)

    # Test error when no paths provided
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should raise InputError for no paths"
    except InputError as e:
        assert "at least one file or directory" in str(e.message)

    # Test with custom config file path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        tmp_path = f.name
    
    config = make_config(argv=["--config", tmp_path], tomlfile=None)
    assert config["paths"] == ["src", "tests"]
    assert config["min_confidence"] == 80
    
    os.unlink(tmp_path)

    # Test with invalid config key
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(invalid_toml)
    try:
        make_config(argv=["path"], tomlfile=toml_file)
        assert False, "Should raise InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test with wrong type
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "high"
    """
    toml_file = io.StringIO(wrong_type_toml)
    try:
        make_config(argv=["path"], tomlfile=toml_file)
        assert False, "Should raise InputError for wrong type"
    except InputError as e:
        assert "Data type" in str(e.message)

    # Test with CSV arguments
    config = make_config(argv=["path", "--exclude", "*.pyc,*.pyo", "--ignore-decorators", "@a,@b"], tomlfile=None)
    assert config["exclude"] == ["*.pyc", "*.pyo"]
    assert config["ignore_decorators"] == ["@a", "@b"]

    # Test verbose output
    import contextlib
    import io as io_module
    
    verbose_toml = """
    [tool.vulture]
    paths = ["src"]
    verbose = true
    """
    toml_file = io.StringIO(verbose_toml)
    output = io_module.StringIO()
    with contextlib.redirect_stdout(output):
        config = make_config(argv=[], tomlfile=toml_file)
    assert "Reading configuration from" in output.getvalue()
```


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding TOML settings
    toml_content = b"""
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
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["path3", "--min-confidence", "30"], tomlfile=toml_file)
    assert config["paths"] == ["path3"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with TOML settings only
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with no paths - should raise InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with unknown configuration key in TOML
    bad_toml_content = b"""
    [tool.vulture]
    unknown_key = "value"
    paths = ["path1"]
    """
    toml_file = io.BytesIO(bad_toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML
    bad_type_content = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    toml_file = io.BytesIO(bad_type_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with verbose mode and TOML file detection
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "pyproject.toml")
        with open(toml_path, "wb") as f:
            f.write(toml_content)
        config = make_config(argv=["--verbose"])
        assert config["verbose"] is True
        assert config["paths"] == ["path1", "path2"]  # from TOML file
```


# LLM-generated content at query #4
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["exclude"] == ["file*.py", "dir/"]

    # Test with invalid configuration key
    toml_file = io.StringIO('[tool.vulture]\ninvalid_key = "value"')
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong data type
    toml_file = io.StringIO('[tool.vulture]\nmin_confidence = "high"')
    try:
        make_config(tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths from TOML file but no CLI paths
    toml_file = io.StringIO('[tool.vulture]\npaths = ["test.py"]')
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["test.py"]

    # Test with --version (should not raise InputError)
    import pytest
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])
```


# LLM-generated content at query #5
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
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
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override toml file
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "75"], tomlfile=toml_file)
    assert config["min_confidence"] == 75

    # Test with missing paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid type in toml
    invalid_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)

    # Test with unknown key in toml
    unknown_key_toml = """
    [tool.vulture]
    unknown_key = "value"
    """
    toml_file = io.BytesIO(unknown_key_toml.encode())
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)

    # Test with CLI arguments that have invalid types
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int", "path1"])

    # Test config file detection from pyproject.toml
    # Create temporary pyproject.toml
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdirname:
        pyproject_path = os.path.join(tmpdirname, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write(toml_content)
        
        # Change working directory
        old_cwd = os.getcwd()
        os.chdir(tmpdirname)
        try:
            config = make_config()
            assert config["min_confidence"] == 10
            assert config["verbose"] is True
        finally:
            os.chdir(old_cwd)


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    exclude = ["test_*.py"]
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src"]

    # Test with both TOML and CLI (CLI takes precedence)
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "70"], tomlfile=toml_file)
    assert config["min_confidence"] == 70

    # Test with unknown configuration key in TOML
    toml_content_invalid = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content_invalid)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    toml_content_wrong_type = """
    [tool.vulture]
    min_confidence = "high"
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content_wrong_type)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths (should raise error)
    try:
        make_config(argv=["--min-confidence", "10"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths provided via TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src"]

    # Test with verbose mode and TOML file (should print message)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        toml_file = io.StringIO(toml_content)
        config = make_config(argv=["--verbose"], tomlfile=toml_file)
        output = sys.stdout.getvalue()
        assert "Reading configuration from" in output
        assert config["verbose"] is True
    finally:
        sys.stdout = old_stdout

    # Test with paths provided in both CLI and TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]  # CLI takes precedence

    # Test with exclude option
    config = make_config(argv=["--exclude", "test_*.py,venv", "path1"])
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["paths"] == ["path1"]

    # Test with ignore_decorators and ignore_names
    config = make_config(
        argv=["--ignore-decorators", "@app.route", "--ignore-names", "helper_*", "path1"]
    )
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["helper_*"]

    # Test with make_whitelist and sort_by_size
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "path1"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #7
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
    config = make_config([])
    assert config == DEFAULTS
    with pytest.raises(InputError):
        config = make_config([])
    
    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test with toml file
    toml_data = b"""
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path1", "toml_path2"]
    exclude = ["test_*.py"]
    """
    import io
    toml_file = io.BytesIO(toml_data)
    config = make_config([], toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["test_*.py"]
    
    # Test CLI overrides toml
    toml_data = b"""
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path1"]
    """
    toml_file = io.BytesIO(toml_data)
    config = make_config(["--min-confidence", "80", "cli_path"], toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]
    
    # Test unknown configuration key
    toml_data = b"""
    [tool.vulture]
    invalid_key = "test"
    """
    toml_file = io.BytesIO(toml_data)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config([], toml_file)
    
    # Test wrong type
    toml_data = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_data)
    with pytest.raises(InputError, match="Data type for min_confidence"):
        make_config([], toml_file)
    
    # Test no paths provided
    with pytest.raises(InputError, match="at least one file or directory"):
        make_config([])
    
    # Test verbose output
    toml_data = b"""
    [tool.vulture]
    verbose = true
    paths = ["test_path"]
    """
    toml_file = io.BytesIO(toml_data)
    config = make_config([], toml_file)
    assert config["verbose"] is True
    
    # Test csv parsing for exclude
    config = make_config(["--exclude", "a.py,b.py,c.py", "path"])
    assert config["exclude"] == ["a.py", "b.py", "c.py"]
    
    # Test boolean flags
    config = make_config(["--make-whitelist", "--sort-by-size", "path"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    
    # Test default values are set
    config = make_config(["--min-confidence", "10", "path"])
    assert config["config"] == "pyproject.toml"
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
```


# LLM-generated content at query #8
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "20", "custom_path.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["custom_path.py"]

    # Test defaults are applied
    toml_file = io.StringIO("[tool.vulture]\npaths = ['test.py']")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test error for unknown configuration key
    toml_file = io.StringIO("[tool.vulture]\nunknown_key = true\npaths = ['test.py']")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    paths = ["path1", "path2"]
    """
    import io
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML options
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80

    # Test with invalid configuration key in TOML
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths provided
    config = make_config(argv=["some_path"])
    assert config["paths"] == ["some_path"]

    # Test with config file path
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        temp_path = f.name
    config = make_config(argv=["--config", temp_path, "some_path"])
    assert config["paths"] == ["some_path"]
    import os
    os.unlink(temp_path)


# LLM-generated content at query #10
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file providing configuration
    import io
    toml_content = """
[tool.vulture]
min_confidence = 20
exclude = ["test_*.py", "docs"]
ignore_decorators = ["@app.route"]
ignore_names = ["helper_*"]
make_whitelist = true
sort_by_size = true
verbose = true
paths = ["src", "lib"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["helper_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["src", "lib"]

    # Test with CLI arguments overriding TOML configuration
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["exclude"] == ["test_*.py", "docs"]  # TOML value preserved
    assert config["paths"] == ["src", "lib"]  # TOML value preserved

    # Test with custom config path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        f.write(toml_content.encode())
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["min_confidence"] == 20
        assert config["paths"] == ["src", "lib"]
    finally:
        os.unlink(temp_path)

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--min-confidence", "50"], tomlfile=None)

    # Test with invalid TOML configuration key
    toml_content_invalid = """
[tool.vulture]
invalid_key = "value"
paths = ["src"]
"""
    toml_file = io.BytesIO(toml_content_invalid.encode())
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong data type in TOML
    toml_content_wrong_type = """
[tool.vulture]
min_confidence = "high"
paths = ["src"]
"""
    toml_file = io.BytesIO(toml_content_wrong_type.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_file)

    # Test with CSV parsing for exclude
    config = make_config(argv=["--exclude", "test_*.py,docs", "path"])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["paths"] == ["path"]

    # Test with --make-whitelist and --sort-by-size flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "path"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path"]
```


# LLM-generated content at query #11
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "path3"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path3"]

    # Test that missing config key raises InputError
    import pytest
    bad_toml = """
[tool.vulture]
invalid_key = "value"
"""
    toml_file = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test that empty paths raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO(""))

    # Test that wrong type raises InputError
    bad_type_toml = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    toml_file = io.StringIO(bad_type_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
    toml_data = """
    [tool.vulture]
    paths = ["src"]
    exclude = ["tests"]
    min_confidence = 30
    verbose = true
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["tests"]
    assert config["min_confidence"] == 30
    assert config["verbose"] is True

    # Test CLI overrides toml
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80

    # Test unknown configuration key raises InputError
    toml_data = """
    [tool.vulture]
    unknown_key = "value"
    """
    tomlfile = io.StringIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test wrong type raises InputError
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.StringIO(toml_data)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test empty paths raises InputError
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "-v", "file1.py", "dir/"], tomlfile=None)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["file1.py", "dir/"]

    # Test with tomlfile providing configuration
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override tomlfile values
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "20", "custom_path.py"], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["custom_path.py"]
    assert config["exclude"] == ["file*.py", "dir/"]  # from tomlfile

    # Test with custom config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as tmp:
        tmp.write(toml_content.encode())
        tmp_path = tmp.name
    try:
        config = make_config(argv=["--config", tmp_path], tomlfile=None)
        assert config["min_confidence"] == 10
        assert config["verbose"] is True
    finally:
        os.unlink(tmp_path)

    # Test with non-existent config file
    config = make_config(argv=["--config", "nonexistent.toml"], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test with empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nverbose = true\n"))

    # Test with unknown configuration key raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nunknown_key = 1\n"))

    # Test with wrong data type raises InputError
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'\n"))

    # Test with CLI arguments that have wrong types
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"], tomlfile=None)

    # Test with verbose output
    import contextlib
    import sys
    tomlfile = io.BytesIO(toml_content.encode())
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        config = make_config(argv=["-v"], tomlfile=tomlfile)
    assert "Reading configuration from" in stdout.getvalue()
```


# LLM-generated content at query #14
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    
    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "test.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]
    
    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 25
    paths = ["src/"]
    exclude = ["tests/"]
    ignore_names = ["unused_*"]
    make_whitelist = true
    sort_by_size = true
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 25
    assert config["paths"] == ["src/"]
    assert config["exclude"] == ["tests/"]
    assert config["ignore_names"] == ["unused_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is False
    
    # Test that CLI arguments override TOML settings
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "75"], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["src/"]
    
    # Test with invalid config key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong type
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)
    
    # Test with no paths provided
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
    
    # Test with multiple paths
    config = make_config(argv=["path1.py", "path2.py", "dir/"])
    assert config["paths"] == ["path1.py", "path2.py", "dir/"]
    
    # Test with exclude patterns
    config = make_config(argv=["--exclude", "*.pyc,test_*.py", "src/"])
    assert config["exclude"] == ["*.pyc", "test_*.py"]
    
    # Test with ignore decorators and names
    config = make_config(
        argv=[
            "--ignore-decorators", "@app.route,@require_*",
            "--ignore-names", "visit_*,do_*",
            "src/"
        ]
    )
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test default config value
    config = make_config(argv=["--config", "custom.toml", "src/"])
    assert config["config"] == "custom.toml"
```


# LLM-generated content at query #15
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no TOML file
    config = make_config(argv=[])
    assert config == DEFAULTS
    
    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["verbose"] is True
    
    # Test with TOML file
    toml_data = b"""
    [tool.vulture]
    paths = ["toml_path.py"]
    min_confidence = 50
    exclude = ["test*.py"]
    """
    import io
    toml_file = io.BytesIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path.py"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test*.py"]
    
    # Test CLI takes precedence over TOML
    toml_file = io.BytesIO(toml_data)
    config = make_config(argv=["cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 50
    
    # Test missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test unknown configuration key in TOML
    bad_toml = b"""
    [tool.vulture]
    unknown_key = "value"
    """
    toml_file = io.BytesIO(bad_toml)
    with pytest.raises(InputError):
        make_config(argv=["path.py"], tomlfile=toml_file)
    
    # Test wrong type in TOML
    bad_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(bad_type_toml)
    with pytest.raises(InputError):
        make_config(argv=["path.py"], tomlfile=toml_file)


# LLM-generated content at query #16
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file (should use defaults)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    sort_by_size = true
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True

    # Test CLI arguments override TOML settings
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value preserved

    # Test with paths from TOML
    toml_content_paths = """
    [tool.vulture]
    paths = ["path1", "path2"]
    min_confidence = 20
    """
    toml_file = io.StringIO(toml_content_paths)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 20

    # Test error when no paths provided
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with --config pointing to a non-existent file
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "nonexistent.toml")
        config = make_config(argv=["--config", config_path, "test.py"])
        assert config["paths"] == ["test.py"]

    # Test with --help should not raise InputError
    try:
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            config = make_config(argv=["--help"])
        assert exc_info.value.code == 0
    except ImportError:
        pass  # pytest not available, skip this test
```


# LLM-generated content at query #17
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1.py", "path2.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test with TOML file
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
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(argv=["--min-confidence", "20", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML values still present

    # Test invalid TOML config raises InputError
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(invalid_toml.encode("utf-8"))
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

    # Test invalid CLI arguments
    try:
        make_config(argv=["--invalid-option"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test _check_output_config raises InputError for empty paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])  # paths is empty by default

    # Test with valid paths but no paths in CLI
    config = make_config(argv=["test.py"])
    assert config["paths"] == ["test.py"]

    # Test with TOML file that has no paths
    toml_no_paths = """
    [tool.vulture]
    min_confidence = 5
    """
    tomlfile = io.BytesIO(toml_no_paths.encode("utf-8"))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with --config option
    config = make_config(argv=["--config", "custom.toml", "test.py"])
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["test.py"]

    # Test with --exclude as comma-separated string
    config = make_config(argv=["--exclude", "a.py,b.py", "test.py"])
    assert config["exclude"] == ["a.py", "b.py"]

    # Test with --ignore-decorators and --ignore-names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", 
                               "--ignore-names", "visit_*,do_*", "test.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --make-whitelist and --sort-by-size
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "test.py"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments only
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file only
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] is True

    # Test with both TOML and CLI (CLI takes precedence)
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nmin_confidence = 50\n"))

    # Test with invalid TOML configuration key
    toml_data = """
    [tool.vulture]
    invalid_key = "test"
    paths = ["path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML configuration
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with verbose output
    toml_data = """
    [tool.vulture]
    paths = ["path"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] is True
```


# LLM-generated content at query #19
#--------------------------

```python
def test_make_config():
    # Test with default arguments (no CLI args, no TOML file)
    import io
    import tempfile
    import os
    from unittest.mock import patch
    
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"
    
    # Test with TOML file
    toml_content = b'''
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    '''
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    
    # Test that CLI arguments take precedence over TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    
    # Test with actual TOML file on disk
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_path = f.name
    
    try:
        config = make_config(argv=[f"--config={temp_path}", "test_path"])
        assert config["exclude"] == ["file*.py", "dir/"]
        assert config["min_confidence"] == 0  # CLI default overrides TOML
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)
    
    # Test with nonexistent config file (defaults to defaults)
    config = make_config(argv=["path"], config_arg=None)
    # Note: This test assumes make_config can be called with a config_arg parameter
    # If not, we'd need to mock the file system
    
    # Test error handling for missing paths
    with patch('vulture.config._parse_args', return_value={}):
        with patch('vulture.config._parse_toml', return_value={}):
            try:
                make_config(argv=[], tomlfile=io.BytesIO(b''))
                assert False, "Should have raised InputError"
            except InputError as e:
                assert "Please pass at least one file or directory" in str(e)
    
    # Test verbose mode with TOML file (should print message)
    toml_file = io.BytesIO(b'''
    [tool.vulture]
    verbose = true
    paths = ["test_path"]
    ''')
    with patch('builtins.print') as mock_print:
        config = make_config(argv=["--verbose"], tomlfile=toml_file)
        mock_print.assert_called_once()
    
    # Test with unknown configuration key
    toml_file = io.BytesIO(b'''
    [tool.vulture]
    unknown_key = "value"
    paths = ["test_path"]
    ''')
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError for unknown key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong data type
    toml_file = io.BytesIO(b'''
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test_path"]
    ''')
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (should use defaults and fail because no paths)
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "50"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments including flags
    config = make_config(argv=["path.py", "--exclude", "test*.py,venv", "--ignore-decorators", "app.route,require_*", 
                               "--ignore-names", "visit_*,do_*", "--make-whitelist", "--sort-by-size", "--verbose"])
    assert config["paths"] == ["path.py"]
    assert config["exclude"] == ["test*.py", "venv"]
    assert config["ignore_decorators"] == ["app.route", "require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with tomlfile
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1.py", "path2.py"]
    """
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test CLI arguments override toml settings
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["override.py", "--min-confidence", "100"], tomlfile=tomlfile)
    assert config["paths"] == ["override.py"]
    assert config["min_confidence"] == 100
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml
    assert config["make_whitelist"] is True  # from toml

    # Test defaults are applied for missing options
    tomlfile = io.BytesIO(b"[tool.vulture]\npaths = ['test.py']\n")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with a real pyproject.toml file
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "pyproject.toml")
        with open(toml_path, "w") as f:
            f.write('[tool.vulture]\npaths = ["test.py"]\nmin_confidence = 20\n')
        config = make_config(argv=[], tomlfile=None)
        # Since the default config path is "pyproject.toml" and we're in tmpdir,
        # we need to specify the config path
        config = make_config(argv=["--config", toml_path])
        assert config["paths"] == ["test.py"]
        assert config["min_confidence"] == 20

    # Test error when no paths are provided
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test error with unknown configuration key in toml
    tomlfile = io.BytesIO(b"[tool.vulture]\nunknown_key = 'value'\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test error with wrong type in toml
    tomlfile = io.BytesIO(b"[tool.vulture]\npaths = ['test.py']\nmin_confidence = 'not_int'\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test error with wrong type in CLI
    with pytest.raises(InputError):
        make_config(argv=["test.py", "--min-confidence", "not_int"])

    # Test that config file path is respected
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        toml_path = os.path.join(tmpdir, "custom.toml")
        with open(toml_path, "w") as f:
            f.write('[tool.vulture]\npaths = ["test.py"]\n')
        config = make_config(argv=["--config", toml_path])
        assert config["paths"] == ["test.py"]

    # Test verbose output with toml detection
    import io
    tomlfile = io.BytesIO(b"[tool.vulture]\npaths = ['test.py']\nverbose = true\n")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #21
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments only
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
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
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override TOML values
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    """)
    config = make_config(["--min-confidence", "80", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]

    # Test config file detection
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "wb") as f:
            f.write(b"""
            [tool.vulture]
            min_confidence = 20
            paths = ["detected_path"]
            """)
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = make_config([])
            assert config["min_confidence"] == 20
            assert config["paths"] == ["detected_path"]
        finally:
            os.chdir(old_cwd)

    # Test error when no paths are provided
    try:
        make_config([], tomlfile=io.BytesIO(b"[]"))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test error for unknown configuration key
    try:
        make_config([], tomlfile=io.BytesIO(b"""
        [tool.vulture]
        unknown_key = "value"
        paths = ["test"]
        """))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test error for wrong data type
    try:
        make_config([], tomlfile=io.BytesIO(b"""
        [tool.vulture]
        min_confidence = "not_an_int"
        paths = ["test"]
        """))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test verbose output
    import io as io_module
    captured_output = io_module.StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        config = make_config(["-v"], tomlfile=io.BytesIO(b"""
        [tool.vulture]
        paths = ["test"]
        """))
    finally:
        sys.stdout = old_stdout
    assert "Reading configuration from" in captured_output.getvalue()
```


# LLM-generated content at query #22
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1.py", "path2.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    min_confidence = 30
    exclude = ["test_*.py"]
    paths = ["src/"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config([], toml_file)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src/"]

    # Test CLI arguments take precedence over TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(["--min-confidence", "70"], toml_file)
    assert config["min_confidence"] == 70

    # Test with --exclude and --ignore-decorators
    config = make_config([
        "--exclude", "*.pyc,venv",
        "--ignore-decorators", "@app.route,@login_required",
        "--ignore-names", "helper_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "src/",
    ])
    assert config["exclude"] == ["*.pyc", "venv"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    assert config["ignore_names"] == ["helper_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with config file that doesn't exist - should use defaults
    config = make_config(["--config", "nonexistent.toml", "path.py"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["path.py"]
    assert config["min_confidence"] == 0

    # Test InputError when no paths provided
    import pytest as pt
    with pt.raises(InputError):
        make_config(["--min-confidence", "10"])

    # Test InputError for unknown configuration key in TOML
    invalid_toml = b"""
    [tool.vulture]
    invalid_option = true
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(invalid_toml)
    with pt.raises(InputError):
        make_config([], toml_file)

    # Test InputError for wrong type in TOML
    invalid_type_toml = b"""
    [tool.vulture]
    min_confidence = "high"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(invalid_type_toml)
    with pt.raises(InputError):
        make_config([], toml_file)

    # Test with empty paths in TOML
    empty_paths_toml = b"""
    [tool.vulture]
    paths = []
    """
    toml_file = io.BytesIO(empty_paths_toml)
    with pt.raises(InputError):
        make_config([], toml_file)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test with tomlfile
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path1", "toml_path2"]
    exclude = ["exclude1", "exclude2"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config([], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["exclude1", "exclude2"]
    assert config["verbose"] is False
    
    # Test with CLI arguments overriding toml values
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(["--min-confidence", "80", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]
    
    # Test with invalid configuration key
    toml_data = """
    [tool.vulture]
    invalid_key = "test"
    """
    toml_file = io.BytesIO(toml_data.encode())
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong type
    toml_data = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(toml_data.encode())
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)
    
    # Test with no paths
    try:
        make_config([], tomlfile=io.BytesIO(b"[tool.vulture]\n"))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #24
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
    toml_data = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 75
    verbose = true
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 75
    assert config["verbose"] is True

    # Test that CLI arguments take precedence over toml
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90

    # Test with --exclude option
    config = make_config(argv=["--exclude", "*.py,test_*.py"])
    assert config["exclude"] == ["*.py", "test_*.py"]

    # Test with --ignore-decorators option
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names option
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with boolean flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True

    # Test that empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with config file path
    config = make_config(argv=["--config", "custom_config.toml"])
    assert config["config"] == "custom_config.toml"

    # Test with toml file containing unknown keys
    tomlfile = io.StringIO("""
    [tool.vulture]
    unknown_key = "value"
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with toml file containing wrong types
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config()
    assert config == DEFAULTS

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    assert config["config"] == "pyproject.toml"

    # Test with tomlfile
    toml_content = b"""
    [tool.vulture]
    min_confidence = 30
    paths = ["path3"]
    exclude = ["test_*.py"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path3"]
    assert config["exclude"] == ["test_*.py"]

    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path3"]

    # Test with all CLI arguments
    config = make_config([
        "--exclude", "file1.py,dir/",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "--min-confidence", "75",
        "--config", "custom.toml",
        "path1", "path2"
    ])
    assert config["exclude"] == ["file1.py", "dir/"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["min_confidence"] == 75
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["path1", "path2"]

    # Test with empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(["--min-confidence", "20"])

    # Test with invalid config key in toml
    invalid_toml = b"""
    [tool.vulture]
    invalid_key = "test"
    paths = ["test_path"]
    """
    toml_file = io.BytesIO(invalid_toml)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)

    # Test with wrong type in toml
    wrong_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test_path"]
    """
    toml_file = io.BytesIO(wrong_type_toml)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config():
    # Test with no arguments (should use defaults and raise InputError for no paths)
    try:
        make_config()
        assert False, "Expected InputError for no paths"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test with paths provided via CLI
    config = make_config(["test.py"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test CLI overrides TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(["override.py", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["override.py"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value preserved

    # Test with CLI arguments overriding defaults
    config = make_config(["test.py", "--exclude", "*.pyc,__pycache__", "--verbose"])
    assert config["exclude"] == ["*.pyc", "__pycache__"]
    assert config["verbose"] is True

    # Test with invalid config key in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = true
    paths = ["test.py"]
    """)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError for unknown config key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with invalid CLI argument type
    try:
        make_config(["test.py", "--min-confidence", "not_an_int"])
        assert False, "Expected InputError for wrong CLI type"
    except (InputError, SystemExit) as e:
        # argparse will raise SystemExit for invalid int
        pass

    # Test verbose output when reading from TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    paths = ["test.py"]
    verbose = true
    """)
    import contextlib
    import io as io_module
    output = io_module.StringIO()
    with contextlib.redirect_stdout(output):
        config = make_config(tomlfile=toml_file)
    assert "Reading configuration from" in output.getvalue()

    # Test with non-existent config file
    config = make_config(["test.py", "--config", "nonexistent.toml"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0  # default preserved
```


# LLM-generated content at query #27
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 20
    paths = ["dir1", "dir2"]
    exclude = ["test_*.py"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["dir1", "dir2"]
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] == True

    # Test CLI arguments override TOML
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with invalid TOML configuration
    import pytest
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=toml_file)
    assert "Unknown configuration key" in str(excinfo.value)

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "high"
    """
    toml_file = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=toml_file)
    assert "Data type for min_confidence" in str(excinfo.value)

    # Test with no paths
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[])
    assert "Please pass at least one file or directory" in str(excinfo.value)
```


# LLM-generated content at query #28
#--------------------------

```python
def test_make_config():
    # Test basic functionality with no arguments
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_data = io.BytesIO(b"""
        [tool.vulture]
        exclude = ["file*.py", "dir/"]
        min_confidence = 10
        paths = ["path1", "path2"]
    """)
    config = make_config(argv=[], tomlfile=toml_data)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 10
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    toml_data = io.BytesIO(b"""
        [tool.vulture]
        min_confidence = 10
        paths = ["toml_path"]
    """)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_data)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test invalid config key in TOML
    toml_data = io.BytesIO(b"""
        [tool.vulture]
        invalid_key = "value"
    """)
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=toml_data)
    assert "Unknown configuration key" in str(excinfo.value.message)

    # Test invalid config key in CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--invalid-option", "value"])
    assert "Unknown configuration key" in str(excinfo.value.message)

    # Test wrong type in TOML
    toml_data = io.BytesIO(b"""
        [tool.vulture]
        min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=toml_data)
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value.message)

    # Test empty paths raises InputError
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[])
    assert "Please pass at least one file or directory" in str(excinfo.value.message)

    # Test with paths from TOML
    toml_data = io.BytesIO(b"""
        [tool.vulture]
        paths = ["some_path"]
    """)
    config = make_config(argv=[], tomlfile=toml_data)
    assert config["paths"] == ["some_path"]

    # Test with exclude from CLI (comma-separated)
    config = make_config(argv=["path", "--exclude", "file1.py,file2.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with ignore_decorators from CLI
    config = make_config(argv=["path", "--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with ignore_names from CLI
    config = make_config(argv=["path", "--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with make_whitelist
    config = make_config(argv=["path", "--make-whitelist"])
    assert config["make_whitelist"] is True

    # Test with sort_by_size
    config = make_config(argv=["path", "--sort-by-size"])
    assert config["sort_by_size"] is True

    # Test verbose output
    toml_data = io.BytesIO(b"""
        [tool.vulture]
        paths = ["some_path"]
        verbose = true
    """)
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        config = make_config(argv=[], tomlfile=toml_data)
    assert "Reading configuration from" in stdout.getvalue()

    # Test with custom config path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
            [tool.vulture]
            paths = ["custom_path"]
            min_confidence = 75
        """)
        temp_path = f.name

    try:
        config = make_config(argv=["--config", temp_path])
        assert config["paths"] == ["custom_path"]
        assert config["min_confidence"] == 75
    finally:
        os.unlink(temp_path)

    # Test that missing config file still works
    config = make_config(argv=["--config", "nonexistent.toml", "some_path"])
    assert config["paths"] == ["some_path"]
    assert config["min_confidence"] == 0


# LLM-generated content at query #29
#--------------------------

```python
def test_make_config(tmp_path, capsys):
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
    toml_content = b"""
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
    toml_file = tmp_path / "test_config.toml"
    toml_file.write_bytes(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(argv=[], tomlfile=f)
    
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test that CLI arguments override TOML config
    with open(toml_file, "rb") as f:
        config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=f)
    
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]  # from TOML

    # Test verbose output
    with open(toml_file, "rb") as f:
        config = make_config(argv=["--verbose"], tomlfile=f)
    
    captured = capsys.readouterr()
    assert f"Reading configuration from {toml_file}" in captured.out

    # Test with invalid config key
    invalid_toml = b"""
[tool.vulture]
invalid_key = "value"
"""
    toml_file_invalid = tmp_path / "invalid_config.toml"
    toml_file_invalid.write_bytes(invalid_toml)
    
    with open(toml_file_invalid, "rb") as f:
        try:
            make_config(argv=[], tomlfile=f)
            assert False, "Should have raised InputError"
        except InputError as e:
            assert "Unknown configuration key" in str(e)

    # Test with wrong type
    wrong_type_toml = b"""
[tool.vulture]
min_confidence = "not_an_int"
"""
    toml_file_wrong = tmp_path / "wrong_type.toml"
    toml_file_wrong.write_bytes(wrong_type_toml)
    
    with open(toml_file_wrong, "rb") as f:
        try:
            make_config(argv=[], tomlfile=f)
            assert False, "Should have raised InputError"
        except InputError as e:
            assert "Data type for min_confidence" in str(e)

    # Test missing paths
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with config file that doesn't exist
    config = make_config(argv=["path", "--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["path"]
```


# LLM-generated content at query #30
#--------------------------

```python
def test_make_config():
    # Test with no arguments (default config file not found)
    config = make_config([])
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(["--verbose", "--min-confidence", "50", "src"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src"]

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI args take precedence over TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = 10
    verbose = false
    """)
    config = make_config(["--min-confidence", "90", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config([])

    # Test with invalid TOML key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    paths = ["src"]
    """)
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["src"]
    """)
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test with explicit config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 42
        paths = ["some_path"]
        """)
        temp_path = f.name
    
    try:
        config = make_config(["--config", temp_path])
        assert config["min_confidence"] == 42
        assert config["paths"] == ["some_path"]
    finally:
        os.unlink(temp_path)

    # Test with CSV arguments
    config = make_config(["--exclude", "a.py,b.py", "--ignore-names", "foo,bar"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_names"] == ["foo", "bar"]
```


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path1"]
    verbose = true
    """
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path1"]
    assert config["verbose"] is True

    # Test CLI arguments override tomlfile
    config = make_config(["--min-confidence", "70"], tomlfile)
    assert config["min_confidence"] == 70
    assert config["paths"] == ["toml_path1"]

    # Test with invalid config key
    invalid_toml = """
    [tool.vulture]
    invalid_key = 10
    """
    invalid_tomlfile = io.BytesIO(invalid_toml.encode())
    try:
        make_config([], invalid_tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test with wrong data type
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    wrong_type_tomlfile = io.BytesIO(wrong_type_toml.encode())
    try:
        make_config([], wrong_type_tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)

    # Test with missing paths
    try:
        make_config(["--min-confidence", "10"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e.message)

    # Test with paths from toml
    toml_paths = """
    [tool.vulture]
    paths = ["some_path"]
    """
    toml_paths_file = io.BytesIO(toml_paths.encode())
    config = make_config([], toml_paths_file)
    assert config["paths"] == ["some_path"]

    # Test with exclude and ignore patterns
    config = make_config([
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "private_*",
        "main.py"
    ])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["private_*"]

    # Test with make_whitelist and sort_by_size
    config = make_config(["--make-whitelist", "--sort-by-size", "file.py"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #32
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    import tempfile
    import os

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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments take precedence over TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML values still apply

    # Test with a temporary pyproject.toml file
    temp_dir = tempfile.mkdtemp()
    pyproject_path = os.path.join(temp_dir, "pyproject.toml")
    with open(pyproject_path, "w") as f:
        f.write(toml_content)
    
    config = make_config(argv=["--config", pyproject_path], tomlfile=None)
    assert config["min_confidence"] == 10
    assert config["paths"] == ["path1", "path2"]

    # Test that missing paths raises error
    import pytest
    with pytest.raises(InputError):
        config = make_config(argv=[], tomlfile=io.StringIO(""))

    # Test with invalid config key in TOML
    with pytest.raises(InputError):
        invalid_toml = """
        [tool.vulture]
        invalid_key = "value"
        paths = ["path"]
        """
        make_config(argv=[], tomlfile=io.StringIO(invalid_toml))

    # Test with invalid type in TOML
    with pytest.raises(InputError):
        invalid_toml = """
        [tool.vulture]
        min_confidence = "not_an_int"
        paths = ["path"]
        """
        make_config(argv=[], tomlfile=io.StringIO(invalid_toml))

    # Test with invalid type in CLI
    with pytest.raises(SystemExit):
        make_config(argv=["--min-confidence", "not_an_int"], tomlfile=None)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (uses default config path)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with tomlfile
    import io
    import tomllib
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
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "90", "custom_path"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["custom_path"]
    assert config["verbose"] is True  # TOML value preserved

    # Test with unknown configuration key in TOML
    toml_data_invalid = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_data_invalid.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

    # Test with wrong type in TOML
    toml_data_wrong_type = """
    [tool.vulture]
    min_confidence = "high"
    """
    tomlfile = io.BytesIO(toml_data_wrong_type.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"

    # Test with no paths (should raise InputError)
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test with paths specified
    config = make_config(argv=["some_path"])
    assert config["paths"] == ["some_path"]
```


# LLM-generated content at query #34
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 20
    exclude = ["file1.py", "file2.py"]
    paths = ["src"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["src"]

    # Test CLI arguments override tomlfile
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "80"], tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["src"]

    # Test with empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config([])

    # Test with unknown config key in tomlfile
    bad_toml = """
    [tool.vulture]
    unknown_key = 5
    paths = ["src"]
    """
    with pytest.raises(InputError):
        make_config([], io.StringIO(bad_toml))

    # Test with wrong type in tomlfile
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["src"]
    """
    with pytest.raises(InputError):
        make_config([], io.StringIO(wrong_type_toml))


# LLM-generated content at query #35
#--------------------------

```python
def test_make_config():
    # Test with default values (no CLI args, no tomlfile)
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 30
    exclude = ["exclude1", "exclude2"]
    """
    tomlfile = io.StringIO(toml_data)
    # Convert StringIO to BytesIO since tomllib requires binary
    toml_bytes = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_bytes)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["exclude1", "exclude2"]

    # Test that CLI args override TOML config
    toml_bytes = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["cli_path", "--min-confidence", "80"], tomlfile=toml_bytes)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80

    # Test with missing paths (should raise InputError)
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid TOML key
    bad_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    bad_toml_bytes = io.BytesIO(bad_toml.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=["test_path"], tomlfile=bad_toml_bytes)

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    wrong_type_bytes = io.BytesIO(wrong_type_toml.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=["test_path"], tomlfile=wrong_type_bytes)

    # Test with --version flag (should not raise InputError for missing paths)
    config = make_config(argv=["--version"])
    assert config["paths"] == []

    # Test with --help flag
    config = make_config(argv=["--help"])
    assert config["paths"] == []
```


# LLM-generated content at query #36
#--------------------------

```python
def test_make_config():
    # Test with no arguments (should use defaults)
    config = make_config([])
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    min_confidence = 80
    exclude = ["test_*.py", "docs"]
    paths = ["src", "tests"]
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["paths"] == ["src", "tests"]
    assert config["verbose"] is False

    # Test with TOML file and CLI arguments (CLI should override)
    toml_content = b"""
    [tool.vulture]
    min_confidence = 80
    paths = ["src"]
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(["--min-confidence", "30", "cli_path"], tomlfile)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["cli_path"]

    # Test with invalid TOML key
    toml_content = b"""
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_content)
    with pytest.raises(InputError):
        make_config([], tomlfile)

    # Test with wrong type in TOML
    toml_content = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.BytesIO(toml_content)
    with pytest.raises(InputError):
        make_config([], tomlfile)

    # Test with no paths provided
    with pytest.raises(InputError):
        make_config(["--min-confidence", "50"])

    # Test with config file path that doesn't exist
    config = make_config(["--config", "nonexistent.toml", "some_path"])
    assert config["paths"] == ["some_path"]
    assert config["min_confidence"] == 0

    # Test with config file path that exists
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b"""
        [tool.vulture]
        min_confidence = 90
        paths = ["test_path"]
        """)
        temp_path = f.name
    try:
        config = make_config(["--config", temp_path])
        assert config["min_confidence"] == 90
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)

    # Test with all CLI options
    config = make_config([
        "--exclude", "file1.py,file2.py",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "--min-confidence", "75",
        "path1", "path2"
    ])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["min_confidence"] == 75
    assert config["paths"] == ["path1", "path2"]
```


# LLM-generated content at query #37
#--------------------------

```python
def test_make_config():
    # Test with TOML file and no CLI args
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["test_exclude"]
    min_confidence = 50
    paths = ["test_path"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test_exclude"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test_path"]
    assert config["verbose"] is False

    # Test with CLI args overriding TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["test_exclude"]

    # Test with CLI args only
    config = make_config(argv=["--min-confidence", "30", "path1", "path2"])
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["verbose"] is False

    # Test with CLI args for boolean flags
    config = make_config(argv=["--verbose", "--make-whitelist", "my_path"])
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    assert config["paths"] == ["my_path"]

    # Test with default values
    config = make_config(argv=["my_path"])
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []

    # Test with --exclude as comma-separated
    config = make_config(argv=["--exclude", "a.py,b.py,c.py", "path"])
    assert config["exclude"] == ["a.py", "b.py", "c.py"]

    # Test with --ignore-decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "path"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names
    config = make_config(argv=["--ignore-names", "visit_*,do_*", "path"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --sort-by-size
    config = make_config(argv=["--sort-by-size", "path"])
    assert config["sort_by_size"] is True

    # Test with --config pointing to non-existent file
    config = make_config(argv=["--config", "nonexistent.toml", "path"])
    assert config["paths"] == ["path"]
    assert config["min_confidence"] == 0

    # Test with TOML file and CLI paths
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["cli_path"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 50

    # Test with --version (should exit with SystemExit)
    try:
        make_config(argv=["--version"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test with --help (should exit with SystemExit)
    try:
        make_config(argv=["--help"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test with missing paths
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #38
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with custom CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    min_confidence = 25
    exclude = ["test_*.py", "venv"]
    paths = ["src"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["min_confidence"] == 25
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["paths"] == ["src"]

    # Test CLI arguments override TOML settings
    toml_content = b"""
    [tool.vulture]
    min_confidence = 25
    verbose = false
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "75", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["verbose"] is True

    # Test with make-whitelist and sort-by-size flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "path1"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1"]

    # Test with exclude, ignore-decorators, and ignore-names
    config = make_config(argv=["--exclude", "a.py,b.py", "--ignore-decorators", "deco1,deco2", 
                              "--ignore-names", "name1,name2", "path"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["paths"] == ["path"]

    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as tmp:
        tmp.write(b'[tool.vulture]\nmin_confidence = 80\n')
        tmp_path = tmp.name
    try:
        config = make_config(argv=["--config", tmp_path])
        assert config["min_confidence"] == 80
    finally:
        os.unlink(tmp_path)

    # Test InputError when no paths are provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "10"])
```


# LLM-generated content at query #39
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["-v", "--min-confidence", "50", "test.py"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test.py"]

    # Test with TOML file content
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py", "venv"]
    min_confidence = 20
    sort_by_size = true
    paths = ["src", "main.py"]
    """
    import io
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["min_confidence"] == 20
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src", "main.py"]

    # Test CLI arguments override TOML
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True  # from TOML

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(["--min-confidence", "10"])

    # Test with invalid config key in TOML
    invalid_toml = """
    [tool.vulture]
    invalid_key = true
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "high"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test with explicit config file path
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
        [tool.vulture]
        verbose = true
        paths = ["test.py"]
        """)
        temp_path = f.name
    
    config = make_config(["--config", temp_path])
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]
    
    import os
    os.unlink(temp_path)
```


# LLM-generated content at query #40
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 75
    exclude = ["exclude1", "exclude2"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["exclude1", "exclude2"]

    # Test CLI arguments override tomlfile
    toml_content = """
    [tool.vulture]
    paths = ["toml_path"]
    min_confidence = 10
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90

    # Test with --exclude, --ignore-decorators, --ignore-names
    config = make_config(argv=["--exclude", "a.py,b.py", "--ignore-decorators", "dec1,dec2", "--ignore-names", "name1,name2"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test with --make-whitelist and --sort-by-size
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test with custom config file path
    config = make_config(argv=["--config", "custom.toml"])
    assert config["config"] == "custom.toml"

    # Test that InputError is raised when no paths are provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test that InputError is raised for unknown configuration key
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    """
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)

    # Test that InputError is raised for wrong data type
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=tomlfile)
```


# LLM-generated content at query #41
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    import io
    import tempfile
    import os
    
    # Test with tomlfile parameter
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    paths = ["src/"]
    exclude = ["test_*.py"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/"]
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is False
    assert config["make_whitelist"] is False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with CLI arguments overriding tomlfile
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "75", "path1", "path2"], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test_*.py"]
    
    # Test with only CLI arguments
    config = make_config(argv=["--verbose", "--sort-by-size", "myfile.py"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["myfile.py"]
    assert config["min_confidence"] == 0
    
    # Test with pyproject.toml file in current directory
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("pyproject.toml", "w") as f:
                f.write("[tool.vulture]\nmin_confidence = 30\npaths = [\"dir/\"]\n")
            config = make_config()
            assert config["min_confidence"] == 30
            assert config["paths"] == ["dir/"]
        finally:
            os.chdir(old_cwd)
    
    # Test with custom config path
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "custom_config.toml")
        with open(config_path, "w") as f:
            f.write("[tool.vulture]\nmin_confidence = 40\npaths = [\"src/\"]\n")
        config = make_config(argv=["--config", config_path])
        assert config["min_confidence"] == 40
        assert config["paths"] == ["src/"]
    
    # Test with empty paths - should raise InputError
    import pytest
    with pytest.raises(InputError):
        config = make_config(argv=[])
    
    # Test with invalid config key
    bad_toml = io.StringIO("[tool.vulture]\ninvalid_key = 5\npaths = [\"x\"]\n")
    with pytest.raises(InputError):
        config = make_config(tomlfile=bad_toml)
    
    # Test with wrong type
    wrong_type_toml = io.StringIO("[tool.vulture]\nmin_confidence = \"not_int\"\npaths = [\"x\"]\n")
    with pytest.raises(InputError):
        config = make_config(tomlfile=wrong_type_toml)
    
    # Test with no paths in tomlfile
    no_paths_toml = io.StringIO("[tool.vulture]\nmin_confidence = 10\n")
    with pytest.raises(InputError):
        config = make_config(tomlfile=no_paths_toml)
```


# LLM-generated content at query #42
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50

    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 30
    ignore_names = ["test_*"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 30
    assert config["ignore_names"] == ["test_*"]

    # Test that CLI arguments override tomlfile settings
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "80"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with invalid toml configuration
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path"]
    """
    tomlfile = io.StringIO(invalid_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with wrong type in toml
    wrong_type_toml = """
    [tool.vulture]
    paths = "not_a_list"
    """
    tomlfile = io.StringIO(wrong_type_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with exclude as comma-separated string
    toml_exclude = """
    [tool.vulture]
    paths = ["path"]
    exclude = ["file1.py,file2.py"]
    """
    tomlfile = io.StringIO(toml_exclude)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file1.py,file2.py"]

    # Test verbose mode prints configuration path
    import contextlib
    import sys
    from io import StringIO
    tomlfile = io.StringIO(toml_content)
    stdout = StringIO()
    with contextlib.redirect_stdout(stdout):
        config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert "Reading configuration from" in stdout.getvalue()
```


# LLM-generated content at query #43
#--------------------------

```python
def test_make_config():
    # Test with TOML file and no CLI args
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
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with CLI args overriding TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "20", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True

    # Test with only CLI args
    config = make_config(argv=["--min-confidence", "30", "path1", "path2"])
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with no paths raises error
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e.message)

    # Test with invalid config key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1"]
    """)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test with wrong type
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)
```


# LLM-generated content at query #44
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "80", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] == True

    # Test with toml file
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
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override toml file
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["override.py", "--min-confidence", "50"], tomlfile=tomlfile)
    assert config["paths"] == ["override.py"]
    assert config["min_confidence"] == 50

    # Test with invalid config key in toml
    toml_invalid = """
[tool.vulture]
invalid_key = "value"
"""
    tomlfile = io.StringIO(toml_invalid)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in toml
    toml_wrong_type = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.StringIO(toml_wrong_type)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with paths provided
    config = make_config(argv=["test.py"])
    assert config["paths"] == ["test.py"]

    # Test with --version flag
    try:
        make_config(argv=["--version"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test with --help flag
    try:
        make_config(argv=["--help"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass
```


# LLM-generated content at query #45
#--------------------------

```python
def test_make_config():
    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding TOML
    toml_file = io.StringIO(toml_content)
    argv = ["--min-confidence", "20", "--verbose"]
    config = make_config(argv=argv, tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["paths"] == ["path1", "path2"]

    # Test with only CLI arguments
    argv = ["path1", "path2", "--exclude", "file*.py,dir/"]
    config = make_config(argv=argv)
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with empty paths raises InputError
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test with invalid configuration key
    toml_content_invalid = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path"]
    """
    toml_file = io.StringIO(toml_content_invalid)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Unknown configuration key: invalid_key"

    # Test with wrong type
    toml_content_wrong_type = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    toml_file = io.StringIO(toml_content_wrong_type)
    try:
        make_config(tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Data type for min_confidence must be 'int'"
```


# LLM-generated content at query #46
#--------------------------

```python
def test_make_config():
    # Test with no arguments (no toml file)
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
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

    # Test CLI overrides TOML
    tomlfile = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "75"], tomlfile=tomlfile)
    assert config["min_confidence"] == 75

    # Test with invalid config key
    tomlfile = io.StringIO("[tool.vulture]\ninvalid_key = true\npaths = ['test']")
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    tomlfile = io.StringIO("[tool.vulture]\nmin_confidence = 'not_an_int'\npaths = ['test']")
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test no paths provided
    try:
        make_config([])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
```


# LLM-generated content at query #47
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    paths = ["src", "tests"]
    min_confidence = 80
    exclude = ["*.pyc", "venv"]
    """
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src", "tests"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["*.pyc", "venv"]
    assert config["make_whitelist"] is False

    # Test CLI overrides TOML
    toml_content = b"""
    [tool.vulture]
    paths = ["src"]
    min_confidence = 80
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["custom_path", "--min-confidence", "30"], tomlfile=tomlfile)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 30

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\n"))

    # Test with --version or --help (should not raise error)
    import sys
    if "--version" in sys.argv or "--help" in sys.argv:
        pass  # These would normally exit, but we skip in test
    else:
        # Test that --version is handled
        try:
            make_config(argv=["--version"])
        except SystemExit:
            pass  # Expected behavior

    # Test with invalid config key in TOML
    toml_content = b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["src"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(toml_content))

    # Test with wrong type in TOML
    toml_content = b"""
    [tool.vulture]
    min_confidence = "high"
    paths = ["src"]
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(toml_content))
```


# LLM-generated content at query #48
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "--sort-by-size", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
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
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML values
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "90", "--verbose", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    assert config["paths"] == ["cli_path"]

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "10"], tomlfile=io.BytesIO(b"[tool.vulture]\n"))

    # Test with unknown configuration key in TOML
    toml_file = io.BytesIO(b"[tool.vulture]\nunknown_key = true\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)
```


# LLM-generated content at query #49
#--------------------------

```python
def test_make_config(tmp_path, capsys):
    # Test with CLI args only
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
    toml_content = """
[tool.vulture]
exclude = ["exclude1", "exclude2"]
min_confidence = 75
paths = ["path_a", "path_b"]
verbose = true
"""
    toml_file = tmp_path / "test_config.toml"
    toml_file.write_text(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(tomlfile=f)
    
    assert config["exclude"] == ["exclude1", "exclude2"]
    assert config["min_confidence"] == 75
    assert config["paths"] == ["path_a", "path_b"]
    assert config["verbose"] is True
    assert config["sort_by_size"] is False
    
    # Test CLI overrides TOML
    with open(toml_file, "rb") as f:
        config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=f)
    
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90
    assert config["exclude"] == ["exclude1", "exclude2"]  # TOML value preserved

    # Test with no paths raises error
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "10"])

    # Test verbose output with TOML file
    with open(toml_file, "rb") as f:
        config = make_config(argv=["--verbose"], tomlfile=f)
    
    captured = capsys.readouterr()
    assert f"Reading configuration from {toml_file}" in captured.out

    # Test with default config file discovery
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.vulture]
min_confidence = 30
paths = ["auto_detected_path"]
""")
    
    config = make_config(argv=["--config", str(pyproject)])
    assert config["min_confidence"] == 30
    assert config["paths"] == ["auto_detected_path"]
    
    # Test defaults are set for missing options
    config = make_config(argv=["test_path"])
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
```


# LLM-generated content at query #50
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    exclude = ["test_*.py", "docs/"]
    ignore_decorators = ["decorator1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 50
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py", "docs/"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "cli_path.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["cli_path.py"]

    # Test with missing paths raises InputError
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with invalid config key
    toml_content_invalid = b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content_invalid)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    toml_content_wrong_type = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content_wrong_type)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "must be" in str(e)

    # Test with config file path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "wb") as f:
            f.write(b"""
            [tool.vulture]
            min_confidence = 30
            paths = ["test.py"]
            """)
        config = make_config(argv=["--config", config_path])
        assert config["min_confidence"] == 30
        assert config["paths"] == ["test.py"]

    # Test with verbose and TOML file prints message
    toml_file = io.BytesIO(toml_content)
    import contextlib
    import io as io_module
    output = io_module.StringIO()
    with contextlib.redirect_stdout(output):
        config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert "Reading configuration from" in output.getvalue()


# LLM-generated content at query #51
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    result = make_config([])
    assert result["config"] == "pyproject.toml"
    assert result["min_confidence"] == 0
    assert result["paths"] == []
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] is False
    assert result["sort_by_size"] is False
    assert result["verbose"] is False

    # Test with CLI arguments
    result = make_config(["--min-confidence", "80", "--verbose", "path1", "path2"])
    assert result["min_confidence"] == 80
    assert result["verbose"] is True
    assert result["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test*.py"]
    min_confidence = 50
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    result = make_config([], toml_file)
    assert result["exclude"] == ["test*.py"]
    assert result["min_confidence"] == 50
    assert result["paths"] == ["src"]

    # Test CLI arguments override toml values
    toml_file = io.BytesIO(toml_content.encode())
    result = make_config(["--min-confidence", "90"], toml_file)
    assert result["min_confidence"] == 90
    assert result["exclude"] == ["test*.py"]

    # Test with all CLI arguments
    result = make_config([
        "--exclude", "file1.py,dir/",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "--make-whitelist",
        "--sort-by-size",
        "--config", "custom.toml",
        "path1", "path2"
    ])
    assert result["exclude"] == ["file1.py", "dir/"]
    assert result["ignore_decorators"] == ["deco1", "deco2"]
    assert result["ignore_names"] == ["name1", "name2"]
    assert result["make_whitelist"] is True
    assert result["sort_by_size"] is True
    assert result["config"] == "custom.toml"
    assert result["paths"] == ["path1", "path2"]

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config([])  # This will fail because paths is empty

    # Test with invalid config key in toml
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config([], toml_file)

    # Test with wrong type in toml
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError):
        make_config([], toml_file)
```


# LLM-generated content at query #52
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
    config = make_config(argv=[])
    assert config == DEFAULTS
    assert config["paths"] == []
    
    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "test.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test.py"]
    
    # Test with toml file providing configuration
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    paths = ["path1", "path2"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["paths"] == ["path1", "path2"]
    
    # Test CLI arguments take precedence over toml file
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "path2"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path2"]
    
    # Test with unknown configuration key in toml file
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong data type in toml file
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)
    
    # Test with no paths provided
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
    
    # Test with boolean flags
    config = make_config(argv=["--verbose", "--make-whitelist", "test.py"])
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    
    # Test with comma-separated values
    config = make_config(argv=["--exclude", "file1.py,file2.py", "test.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    
    # Test with default config file path (no toml file exists)
    config = make_config(argv=["test.py"], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    
    # Test with verbose flag and toml file (should print message)
    import io
    import sys
    from contextlib import redirect_stdout
    
    toml_content = """
    [tool.vulture]
    paths = ["path1"]
    """
    tomlfile = io.StringIO(toml_content)
    f = io.StringIO()
    with redirect_stdout(f):
        config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert "Reading configuration from" in f.getvalue()
    
    # Test with sort_by_size flag
    config = make_config(argv=["--sort-by-size", "test.py"])
    assert config["sort_by_size"] is True
    
    # Test with ignore_names
    config = make_config(argv=["--ignore-names", "name1,name2", "test.py"])
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test with ignore_decorators
    config = make_config(argv=["--ignore-decorators", "deco1,deco2", "test.py"])
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    
    # Test that defaults are filled in for missing options
    config = make_config(argv=["test.py"])
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
```


# LLM-generated content at query #53
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with tomlfile
    import io
    toml_content = b"""
    [tool.vulture]
    exclude = ["file1.py", "file2.py"]
    min_confidence = 20
    verbose = true
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["min_confidence"] == 20
    assert config["verbose"] == True

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80

    # Test with invalid config key in TOML
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert e.message == "Unknown configuration key: invalid_key"

    # Test with wrong type in TOML
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert e.message == "Data type for min_confidence must be 'int'"

    # Test with no paths
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert e.message == "Please pass at least one file or directory"
```


# LLM-generated content at query #54
#--------------------------

```python
def test_make_config():
    # Test with no arguments
    config = make_config([])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    assert config["paths"] == []

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
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
    config = make_config([], tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML file
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = 10
    verbose = true
    paths = ["toml_path"]
    """)
    config = make_config(["--min-confidence", "70", "cli_path"], tomlfile)
    assert config["min_confidence"] == 70
    assert config["verbose"] is True
    assert config["paths"] == ["cli_path"]

    # Test with invalid TOML configuration
    tomlfile = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1"]
    """)
    try:
        make_config([], tomlfile)
        assert False, "Expected InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test with wrong data type in TOML
    tomlfile = io.StringIO("""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """)
    try:
        make_config([], tomlfile)
        assert False, "Expected InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)

    # Test with no paths
    try:
        make_config([])
        assert False, "Expected InputError for no paths"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e.message)

    # Test with invalid CLI arguments
    try:
        make_config(["--invalid-option"])
        assert False, "Expected SystemExit for invalid CLI option"
    except SystemExit:
        pass

    # Test with --help (should not raise InputError)
    try:
        make_config(["--help"])
        assert False, "Expected SystemExit for --help"
    except SystemExit:
        pass

    # Test with --version (should not raise InputError)
    try:
        make_config(["--version"])
        assert False, "Expected SystemExit for --version"
    except SystemExit:
        pass
```


# LLM-generated content at query #55
#--------------------------

```python
def test_make_config():
    # Test 1: No arguments, no TOML file (default config)
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["path1.py", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test 3: TOML file provides configuration
    toml_data = b"""
    [tool.vulture]
    min_confidence = 75
    exclude = ["test_*.py"]
    paths = ["src/"]
    """
    import io
    toml_file = io.BytesIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src/"]
    assert config["verbose"] is False

    # Test 4: CLI arguments override TOML configuration
    toml_data = b"""
    [tool.vulture]
    min_confidence = 75
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_data)
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["src/"]

    # Test 5: Missing paths raises InputError
    toml_data = b"""
    [tool.vulture]
    min_confidence = 75
    """
    toml_file = io.BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test 6: Invalid configuration key in TOML
    toml_data = b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test 7: Invalid configuration value type in TOML
    toml_data = b"""
    [tool.vulture]
    min_confidence = "high"
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_data)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test 8: Empty exclude list in CLI
    config = make_config(argv=["path.py", "--exclude", ""], tomlfile=None)
    assert config["exclude"] == [""]

    # Test 9: Multiple paths in CLI
    config = make_config(argv=["path1.py", "path2.py", "dir/"], tomlfile=None)
    assert config["paths"] == ["path1.py", "path2.py", "dir/"]

    # Test 10: Make whitelist and sort by size flags
    config = make_config(argv=["path.py", "--make-whitelist", "--sort-by-size"], tomlfile=None)
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #56
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with no arguments
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test 2: Config from TOML file
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test 3: CLI arguments override TOML config
    toml_data = """
    [tool.vulture]
    min_confidence = 10
    verbose = true
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--min-confidence", "50", "path2", "path3"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True  # Not overridden
    assert config["paths"] == ["path2", "path3"]

    # Test 4: Invalid TOML config raises InputError
    toml_data = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(toml_data)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test 5: No paths raises InputError
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test 6: Invalid CLI argument type
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "invalid"])

    # Test 7: Unknown CLI argument
    with pytest.raises(SystemExit):
        make_config(argv=["--unknown-arg"])

    # Test 8: Version flag
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])

    # Test 9: Help flag
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])

    # Test 10: Verbose with TOML file
    toml_data = """
    [tool.vulture]
    verbose = true
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] is True

    # Test 11: Config path from CLI
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 20
        paths = ["test_path"]
        """)
        config_path = f.name
    config = make_config(argv=["--config", config_path])
    assert config["min_confidence"] == 20
    assert config["paths"] == ["test_path"]
    os.unlink(config_path)

    # Test 12: Non-existent config file
    config = make_config(argv=["--config", "nonexistent.toml", "path1"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["path1"]
```


# LLM-generated content at query #57
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2", "--min-confidence", "80"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == []
    assert config["verbose"] is False

    # Test with TOML file and CLI arguments (CLI takes precedence)
    import io
    import tempfile
    toml_content = """
[tool.vulture]
min_confidence = 50
paths = ["toml_path1"]
sort_by_size = true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        toml_file = f.name
    
    with open(toml_file, "rb") as f:
        config = make_config(argv=["cli_path", "--min-confidence", "90"], tomlfile=f)
    
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 90
    assert config["sort_by_size"] is True
    assert config["exclude"] == []

    # Test with invalid config key in TOML
    invalid_toml = """
[tool.vulture]
invalid_key = "value"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(invalid_toml)
        invalid_file = f.name
    
    with open(invalid_file, "rb") as f:
        try:
            make_config(argv=[], tomlfile=f)
            assert False, "Should have raised InputError"
        except InputError as e:
            assert "Unknown configuration key" in str(e)

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with default config file (pyproject.toml)
    # Create a temporary directory with a pyproject.toml file
    import os
    temp_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    pyproject_content = """
[tool.vulture]
min_confidence = 30
paths = ["default_path"]
"""
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    config = make_config(argv=[])
    assert config["min_confidence"] == 30
    assert config["paths"] == ["default_path"]
    
    os.chdir(old_cwd)
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    # Test verbose mode with TOML file
    verbose_toml = """
[tool.vulture]
verbose = true
paths = ["test_path"]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(verbose_toml)
        verbose_file = f.name
    
    with open(verbose_file, "rb") as f:
        import contextlib
        import io as io_module
        output = io_module.StringIO()
        with contextlib.redirect_stdout(output):
            config = make_config(argv=[], tomlfile=f)
    
    assert "Reading configuration from" in output.getvalue()
    assert config["verbose"] is True

    # Clean up temporary files
    os.unlink(toml_file)
    os.unlink(invalid_file)
    os.unlink(verbose_file)
```


# LLM-generated content at query #58
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config == DEFAULTS
    with pytest.raises(InputError):
        make_config([])

    # Test with CLI arguments only
    config = make_config(["--min-confidence", "50", "path1.py", "path2.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["verbose"] is False

    # Test with TOML file only
    toml_content = b"""
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
    toml_file = io.BytesIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments take precedence over TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(["--min-confidence", "90", "cli_path.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["cli_path.py"]
    assert config["exclude"] == ["file*.py", "dir/"]  # from TOML

    # Test with actual pyproject.toml file
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_path = pathlib.Path(tmpdir) / "pyproject.toml"
        pyproject_path.write_bytes(toml_content)
        config = make_config([], tomlfile=None)
        assert config["min_confidence"] == 10
        assert config["paths"] == ["path1", "path2"]

    # Test error on missing paths
    with pytest.raises(InputError):
        make_config(["--min-confidence", "50"])

    # Test error on unknown config key in TOML
    bad_toml = b"""
    [tool.vulture]
    unknown_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(bad_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test error on wrong type in TOML
    bad_type_toml = b"""
    [tool.vulture]
    min_confidence = "high"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(bad_type_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test error on wrong type in CLI
    with pytest.raises(InputError):
        make_config(["--min-confidence", "not_a_number", "test.py"])

    # Test verbose output
    toml_file = io.BytesIO(toml_content)
    captured = capsys.readouterr()
    config = make_config([], tomlfile=toml_file)
    captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out
```


# LLM-generated content at query #59
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["test*.py"]
    ignore_decorators = ["deco1"]
    min_confidence = 75
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["min_confidence"] == 75
    assert config["paths"] == ["src"]

    # Test CLI arguments take precedence over toml
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "100"], tomlfile=toml_file)
    assert config["min_confidence"] == 100
    assert config["exclude"] == ["test*.py"]

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with unknown configuration key in toml
    toml_file = io.BytesIO(b"[tool.vulture]\nunknown_key = true\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in toml
    toml_file = io.BytesIO(b"[tool.vulture]\nmin_confidence = '50'\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with paths provided but empty list
    with pytest.raises(InputError):
        make_config(argv=["--exclude", "test.py"])

    # Test verbose output
    toml_file = io.BytesIO(b"[tool.vulture]\nverbose = true\n")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] is True
```


# LLM-generated content at query #60
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    result = make_config(argv=["path1", "path2"])
    assert result["paths"] == ["path1", "path2"]
    assert result["min_confidence"] == 0
    assert result["exclude"] == []
    assert result["ignore_decorators"] == []
    assert result["ignore_names"] == []
    assert result["make_whitelist"] is False
    assert result["sort_by_size"] is False
    assert result["verbose"] is False
    assert result["config"] == "pyproject.toml"

    # Test with TOML file
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
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    result = make_config(argv=[], tomlfile=tomlfile)
    assert result["exclude"] == ["file*.py", "dir/"]
    assert result["ignore_decorators"] == ["deco1", "deco2"]
    assert result["ignore_names"] == ["name1", "name2"]
    assert result["make_whitelist"] is True
    assert result["min_confidence"] == 10
    assert result["sort_by_size"] is True
    assert result["verbose"] is True
    assert result["paths"] == ["path1", "path2"]

    # Test with CLI arguments overriding TOML
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    result = make_config(argv=["--min-confidence", "20", "--verbose"], tomlfile=tomlfile)
    assert result["min_confidence"] == 20
    assert result["verbose"] is True
    assert result["exclude"] == ["file*.py", "dir/"]
    assert result["paths"] == ["path1", "path2"]

    # Test with CLI arguments only, overriding defaults
    result = make_config(argv=["--min-confidence", "15", "--verbose", "--sort-by-size", "--make-whitelist", "--exclude", "test.py,test2.py", "--ignore-decorators", "deco1", "--ignore-names", "name1", "path1"])
    assert result["min_confidence"] == 15
    assert result["verbose"] is True
    assert result["sort_by_size"] is True
    assert result["make_whitelist"] is True
    assert result["exclude"] == ["test.py", "test2.py"]
    assert result["ignore_decorators"] == ["deco1"]
    assert result["ignore_names"] == ["name1"]
    assert result["paths"] == ["path1"]

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid config key
    tomlfile = io.BytesIO(b"[tool.vulture]\ninvalid_key = 1\n")
    with pytest.raises(InputError):
        make_config(argv=["path1"], tomlfile=tomlfile)

    # Test with wrong type
    tomlfile = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'invalid'\n")
    with pytest.raises(InputError):
        make_config(argv=["path1"], tomlfile=tomlfile)
```


# LLM-generated content at query #61
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "80", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    assert config["config"] == "pyproject.toml"

    # Test with tomlfile
    import io
    toml_data = """
[tool.vulture]
exclude = ["file*.py", "dir/"]
ignore_decorators = ["deco1", "deco2"]
ignore_names = ["name1", "name2"]
make_whitelist = true
min_confidence = 10
sort_by_size = true
verbose = true
paths = ["toml_path1.py", "toml_path2.py"]
"""
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["toml_path1.py", "toml_path2.py"]

    # Test that CLI arguments override TOML settings
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=["cli_path.py", "--min-confidence", "50"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml
    assert config["verbose"] == True  # from toml

    # Test with invalid config key
    tomlfile = io.StringIO("[tool.vulture]\ninvalid_key = true\n")
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    tomlfile = io.StringIO("[tool.vulture]\nmin_confidence = 'not_an_int'\n")
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test that no paths raises error
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths from CLI only
    config = make_config(argv=["some_path.py"])
    assert config["paths"] == ["some_path.py"]

    # Test with --make-whitelist and --sort-by-size flags
    config = make_config(argv=["path.py", "--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test with --exclude, --ignore-decorators, --ignore-names
    config = make_config(argv=["path.py", "--exclude", "a.py,b.py", "--ignore-decorators", "dec1,dec2", "--ignore-names", "name1,name2"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]
```


# LLM-generated content at query #62
#--------------------------

```python
def test_make_config():
    # Test with no arguments (defaults)
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    toml_content = b'''
    [tool.vulture]
    exclude = ["file*.py"]
    ignore_decorators = ["deco1"]
    min_confidence = 50
    paths = ["src", "tests"]
    '''
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src", "tests"]

    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "80"])

    # Test unknown configuration key in TOML
    bad_toml = b'''
    [tool.vulture]
    unknown_key = true
    paths = ["test"]
    '''
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(bad_toml))

    # Test invalid data type in TOML
    bad_type_toml = b'''
    [tool.vulture]
    min_confidence = "high"
    paths = ["test"]
    '''
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(bad_type_toml))

    # Test verbose output prints config path
    capsys = pytest.CaptureFixture()
    with capsys.disabled():
        config = make_config(argv=["--verbose"], tomlfile=io.BytesIO(toml_content))
    captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out
```


# LLM-generated content at query #63
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file
    config = make_config([])
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    toml_data = b"""
    [tool.vulture]
    min_confidence = 80
    paths = ["src/"]
    verbose = true
    """
    import io
    toml_file = io.BytesIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["src/"]
    assert config["verbose"] is True

    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_data)
    config = make_config(["--min-confidence", "30"], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src/"]

    # Test with multiple CLI arguments
    config = make_config(["--exclude", "test*.py,venv", "--ignore-names", "foo,bar", "--sort-by-size"])
    assert config["exclude"] == ["test*.py", "venv"]
    assert config["ignore_names"] == ["foo", "bar"]
    assert config["sort_by_size"] is True

    # Test with make-whitelist
    config = make_config(["--make-whitelist"])
    assert config["make_whitelist"] is True

    # Test with invalid config key in TOML
    bad_toml = b"""
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(bad_toml)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    bad_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(bad_type_toml)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with no paths
    try:
        make_config([], tomlfile=io.BytesIO(b"[]"))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)
```


# LLM-generated content at query #64
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI args override toml config
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["custom.py", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["custom.py"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml
    assert config["verbose"] is True  # from toml

    # Test with empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--config", "nonexistent.toml"])

    # Test with invalid config file raises InputError
    with pytest.raises(InputError):
        make_config(argv=["--config", "nonexistent.toml", "--min-confidence", "invalid"])

    # Test with unknown config key in toml
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(invalid_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in toml config
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not an int"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(wrong_type_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with --version
    config = make_config(argv=["--version"])
    assert config == DEFAULTS

    # Test with --help
    config = make_config(argv=["--help"])
    assert config == DEFAULTS
```


# LLM-generated content at query #65
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    assert config["paths"] == []
    
    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"], tomlfile=None)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test with toml file
    toml_content = b"""
    [tool.vulture]
    min_confidence = 20
    paths = ["toml_path1"]
    verbose = true
    """
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["toml_path1"]
    assert config["verbose"] is True
    
    # Test CLI overrides toml
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    
    # Test defaults are set for missing options
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    paths = ["test_path"]
    """)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    
    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "50"], tomlfile=None)
    
    # Test error with unknown config key in toml
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    invalid_key = "value"
    paths = ["test"]
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
    
    # Test error with wrong type in toml
    tomlfile = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test"]
    """)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)
```


# LLM-generated content at query #66
#--------------------------

```python
def test_make_config():
    # Test with a TOML file providing configuration
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src"]
    min_confidence = 50
    exclude = ["test_*.py"]
    ignore_decorators = ["@app.route"]
    ignore_names = ["private_*"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["private_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    
    # Test with CLI arguments overriding TOML
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "src"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["src"]
    
    # Test with no TOML file and no CLI args
    config = make_config(argv=["test_file.py"])
    assert config["paths"] == ["test_file.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    
    # Test with CLI args only
    config = make_config(argv=["--exclude", "test_*.py,docs", "--sort-by-size", "src"])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src"]
    
    # Test that paths are required
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError:
        pass
    
    # Test with invalid config key in TOML
    toml_content_invalid = """
    [tool.vulture]
    paths = ["src"]
    invalid_key = "value"
    """
    tomlfile = io.StringIO(toml_content_invalid)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)


# LLM-generated content at query #67
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config == {
        "config": "pyproject.toml",
        "min_confidence": 0,
        "paths": [],
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }
    
    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test with tomlfile
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 20
    exclude = ["file1.py", "file2.py"]
    paths = ["src"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile)
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["src"]
    
    # Test CLI overrides toml
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "80"], tomlfile)
    assert config["min_confidence"] == 80
    
    # Test with invalid config key in toml
    tomlfile = io.BytesIO(b'[tool.vulture]\ninvalid_key = true\n')
    try:
        make_config([], tomlfile)
        assert False, "Should have raised InputError"
    except InputError:
        pass
    
    # Test with wrong type in toml
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"\n')
    try:
        make_config([], tomlfile)
        assert False, "Should have raised InputError"
    except InputError:
        pass
    
    # Test with no paths (should raise InputError)
    try:
        make_config([])
        assert False, "Should have raised InputError"
    except InputError:
        pass
    
    # Test with paths in CLI
    config = make_config(["test_file.py"])
    assert config["paths"] == ["test_file.py"]
    
    # Test with exclude as comma-separated string
    config = make_config(["--exclude", "file1.py,file2.py", "test_file.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    
    # Test with ignore_decorators
    config = make_config(["--ignore-decorators", "@app.route,@require", "test_file.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require"]
    
    # Test with ignore_names
    config = make_config(["--ignore-names", "visit_*,do_*", "test_file.py"])
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test with make_whitelist flag
    config = make_config(["--make-whitelist", "test_file.py"])
    assert config["make_whitelist"] is True
    
    # Test with sort_by_size flag
    config = make_config(["--sort-by-size", "test_file.py"])
    assert config["sort_by_size"] is True
    
    # Test with verbose flag
    config = make_config(["--verbose", "test_file.py"])
    assert config["verbose"] is True
    
    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b'[tool.vulture]\nmin_confidence = 30\n')
        temp_path = f.name
    try:
        config = make_config(["--config", temp_path, "test_file.py"])
        assert config["min_confidence"] == 30
    finally:
        os.unlink(temp_path)
```


# LLM-generated content at query #68
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    cli_args = ["path1", "path2", "--min-confidence", "50", "--verbose"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    paths = ["path1", "path2"]
    min_confidence = 30
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    sort_by_size = true
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test precedence: CLI arguments override TOML settings
    toml_content = """
    [tool.vulture]
    paths = ["toml_path"]
    min_confidence = 30
    """
    toml_file = io.BytesIO(toml_content.encode())
    cli_args = ["cli_path", "--min-confidence", "80"]
    config = make_config(argv=cli_args, tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80

    # Test that defaults are set for missing values
    toml_content = """
    [tool.vulture]
    paths = ["test_path"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["min_confidence"] == 0

    # Test that InputError is raised when no paths are provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "10"], tomlfile=io.BytesIO(b""))
```


# LLM-generated content at query #69
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
    toml_content = """
[tool.vulture]
min_confidence = 30
exclude = ["test*.py", "docs"]
paths = ["src"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test*.py", "docs"]
    assert config["paths"] == ["src"]

    # Test CLI overrides TOML
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["src"]

    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
[tool.vulture]
min_confidence = 40
paths = ["src"]
""")
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["min_confidence"] == 40
        assert config["paths"] == ["src"]
    finally:
        os.unlink(temp_path)

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with custom tomlfile that has no paths
    tomlfile = io.StringIO("""
[tool.vulture]
min_confidence = 20
""")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with verbose flag and toml file
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True

    # Test with sort-by-size flag
    config = make_config(argv=["--sort-by-size", "src"])
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src"]
```


# LLM-generated content at query #70
#--------------------------

```python
def test_make_config(tmp_path, capsys):
    # Test 1: Default config with no paths should raise InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test 2: Config from TOML file with valid paths
    toml_content = b"""
    [tool.vulture]
    paths = ["path1", "path2"]
    min_confidence = 10
    """
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_bytes(toml_content)
    
    with open(toml_path, "rb") as f:
        config = make_config(argv=[], tomlfile=f)
    
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == []

    # Test 3: CLI arguments override TOML settings
    with open(toml_path, "rb") as f:
        config = make_config(argv=["--min-confidence", "20", "cli_path"], tomlfile=f)
    
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test 4: Default config file detection
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_bytes(b"""
    [tool.vulture]
    paths = ["default_path"]
    verbose = true
    """)
    
    # Change working directory to tmp_path
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = make_config(argv=[])
        assert config["paths"] == ["default_path"]
        assert config["verbose"] is True
    finally:
        os.chdir(old_cwd)

    # Test 5: Verbose mode with TOML file prints message
    with open(toml_path, "rb") as f:
        config = make_config(argv=["--verbose", "verbose_path"], tomlfile=f)
    
    captured = capsys.readouterr()
    assert "Reading configuration from" in captured.out
    assert str(toml_path) in captured.out
    assert config["paths"] == ["verbose_path"]

    # Test 6: Check all default values are set when not provided
    with open(toml_path, "rb") as f:
        config = make_config(argv=["only_path"], tomlfile=f)
    
    for key, default in DEFAULTS.items():
        assert key in config
        assert config[key] == default

    # Test 7: Error on unknown configuration key in TOML
    bad_toml_content = b"""
    [tool.vulture]
    unknown_key = "value"
    paths = ["path"]
    """
    bad_toml_path = tmp_path / "bad_pyproject.toml"
    bad_toml_path.write_bytes(bad_toml_content)
    
    with open(bad_toml_path, "rb") as f:
        with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
            make_config(argv=[], tomlfile=f)

    # Test 8: Error on wrong type in TOML
    bad_type_toml = b"""
    [tool.vulture]
    paths = "not_a_list"
    """
    bad_type_path = tmp_path / "bad_type.toml"
    bad_type_path.write_bytes(bad_type_toml)
    
    with open(bad_type_path, "rb") as f:
        with pytest.raises(InputError, match="Data type for paths must be 'list'"):
            make_config(argv=[], tomlfile=f)

    # Test 9: Config file path from CLI
    config_file = tmp_path / "custom_config.toml"
    config_file.write_bytes(b"""
    [tool.vulture]
    paths = ["custom_path"]
    """)
    
    config = make_config(argv=["--config", str(config_file)])
    assert config["paths"] == ["custom_path"]

    # Test 10: Non-existent config file falls back to defaults
    config = make_config(argv=["--config", str(tmp_path / "nonexistent.toml"), "fallback_path"])
    assert config["paths"] == ["fallback_path"]
    assert config["min_confidence"] == 0
```


# LLM-generated content at query #71
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with toml file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 10
    sort_by_size = true
    paths = ["path3", "path4"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path3", "path4"]

    # Test CLI overrides toml
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["custom_path", "--min-confidence", "75"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml
    assert config["sort_by_size"] is True  # from toml

    # Test invalid input raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test unknown config key raises InputError
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["path"]
    """
    toml_file = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test wrong type raises InputError
    bad_toml_type = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    toml_file = io.BytesIO(bad_toml_type.encode())
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)
```


# LLM-generated content at query #72
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test with tomlfile
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override tomlfile
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test with invalid config key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    tomlfile = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths from toml
    toml_content = """
    [tool.vulture]
    paths = ["path1"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["path1"]


# LLM-generated content at query #73
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] == True

    # Test with invalid TOML configuration
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = "value"
    """)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with no paths raises error
    try:
        make_config(["--config", "nonexistent.toml"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with default TOML file in working directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("pyproject.toml", "w") as f:
                f.write("""
                [tool.vulture]
                min_confidence = 30
                """)
            config = make_config(["test.py"])
            assert config["min_confidence"] == 30
            assert config["paths"] == ["test.py"]
        finally:
            os.chdir(original_dir)
```


# LLM-generated content at query #74
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "--verbose", "test_file.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["test_file.py"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test*.py"]
    min_confidence = 30
    paths = ["path1", "path2"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["test*.py"]
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file overridden by CLI
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["path1", "path2"]

    # Test with invalid config key in TOML
    bad_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.BytesIO(bad_toml.encode())
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Unknown configuration key" in e.message

    # Test with no paths
    try:
        make_config([], tomlfile=io.BytesIO(b"[tool.vulture]\\n".encode()))
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in e.message

    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.BytesIO(wrong_type_toml.encode())
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should raise InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in e.message
```


# LLM-generated content at query #75
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test with tomlfile containing valid configuration
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test*.py", "docs"]
    ignore_decorators = ["@app.route"]
    ignore_names = ["private_*"]
    make_whitelist = true
    min_confidence = 75
    sort_by_size = true
    verbose = false
    paths = ["src", "tests"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["test*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["private_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 75
    assert config["sort_by_size"] is True
    assert config["verbose"] is False
    assert config["paths"] == ["src", "tests"]
    assert config["config"] == "pyproject.toml"

    # Test CLI arguments override TOML values
    config = make_config(argv=["--min-confidence", "90", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    assert config["exclude"] == ["test*.py", "docs"]  # from TOML
    assert config["paths"] == ["src", "tests"]  # from TOML

    # Test error when no paths are provided
    try:
        make_config(argv=[], tomlfile=io.BytesIO(b""))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test error for unknown configuration key in TOML
    bad_toml = b"""
    [tool.vulture]
    unknown_key = true
    """
    try:
        make_config(argv=["path"], tomlfile=io.BytesIO(bad_toml))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test error for wrong type in TOML
    bad_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    try:
        make_config(argv=["path"], tomlfile=io.BytesIO(bad_type_toml))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with --help and --version
    import contextlib
    import io as io_module
    
    # Test --version
    with contextlib.redirect_stdout(io_module.StringIO()):
        config = make_config(argv=["--version"], tomlfile=None)
    
    # Test --help (should raise SystemExit)
    try:
        with contextlib.redirect_stdout(io_module.StringIO()):
            make_config(argv=["--help"], tomlfile=None)
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass
```


# LLM-generated content at query #76
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["path1", "path2", "--min-confidence", "50"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    exclude = ["test*.py"]
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test*.py"]
    assert config["verbose"] is True

    # Test TOML file overridden by CLI arguments
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "70"], tomlfile=tomlfile)
    assert config["min_confidence"] == 70

    # Test with paths provided in TOML
    toml_content = """
    [tool.vulture]
    paths = ["src", "tests"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src", "tests"]

    # Test with empty paths raises InputError
    try:
        make_config(argv=[], tomlfile=io.StringIO(""))
        assert False, "Expected InputError"
    except InputError:
        pass

    # Test with invalid config key in TOML
    toml_content = """
    [tool.vulture]
    invalid_key = true
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass

    # Test with wrong type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "high"
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError"
    except InputError:
        pass
```


# LLM-generated content at query #77
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no pyproject.toml file (use temp directory)
    import tempfile
    import os
    
    # Test with temporary directory to avoid finding actual pyproject.toml
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Test with paths provided via CLI
            config = make_config(argv=['test_file.py'])
            assert config['paths'] == ['test_file.py']
            assert config['min_confidence'] == 0
            assert config['exclude'] == []
            assert config['ignore_decorators'] == []
            assert config['ignore_names'] == []
            assert config['make_whitelist'] is False
            assert config['sort_by_size'] is False
            assert config['verbose'] is False
            assert config['config'] == 'pyproject.toml'
        finally:
            os.chdir(original_cwd)
    
    # Test with TOML file input
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config['paths'] == ['path1', 'path2']
    assert config['min_confidence'] == 10
    assert config['exclude'] == ['file*.py', 'dir/']
    assert config['ignore_decorators'] == ['deco1', 'deco2']
    assert config['ignore_names'] == ['name1', 'name2']
    assert config['make_whitelist'] is True
    assert config['sort_by_size'] is True
    assert config['verbose'] is True
    
    # Test CLI overrides TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=['--min-confidence', '20', 'cli_file.py'], tomlfile=toml_file)
    assert config['paths'] == ['cli_file.py']
    assert config['min_confidence'] == 20
    
    # Test with --exclude as comma-separated string
    config = make_config(argv=['--exclude', 'file1.py,file2.py', 'test.py'])
    assert config['exclude'] == ['file1.py', 'file2.py']
    
    # Test with --ignore-decorators
    config = make_config(argv=['--ignore-decorators', '@app.route,@require_*', 'test.py'])
    assert config['ignore_decorators'] == ['@app.route', '@require_*']
    
    # Test with --ignore-names
    config = make_config(argv=['--ignore-names', 'visit_*,do_*', 'test.py'])
    assert config['ignore_names'] == ['visit_*', 'do_*']
    
    # Test with --make-whitelist
    config = make_config(argv=['--make-whitelist', 'test.py'])
    assert config['make_whitelist'] is True
    
    # Test with --sort-by-size
    config = make_config(argv=['--sort-by-size', 'test.py'])
    assert config['sort_by_size'] is True
    
    # Test with --verbose
    config = make_config(argv=['--verbose', 'test.py'])
    assert config['verbose'] is True
    
    # Test with --config and missing file
    config = make_config(argv=['--config', 'nonexistent.toml', 'test.py'])
    assert config['paths'] == ['test.py']
    
    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test error with unknown configuration key in TOML
    invalid_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(invalid_toml)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test error with wrong type in TOML
    invalid_toml_type = """
    [tool.vulture]
    min_confidence = "ten"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(invalid_toml_type)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)
```


# LLM-generated content at query #78
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    import io
    import sys
    from unittest.mock import patch
    
    # Test with empty TOML file
    toml_content = """
    [tool.vulture]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    
    # Test with TOML file containing settings
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
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    
    # Test with CLI arguments overriding TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_content)
    argv = ["--min-confidence", "20", "path2"]
    config = make_config(argv=argv, tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["path2"]
    
    # Test with CLI arguments only (no TOML)
    argv = ["--min-confidence", "30", "--verbose", "path3"]
    config = make_config(argv=argv)
    assert config["min_confidence"] == 30
    assert config["verbose"] == True
    assert config["paths"] == ["path3"]
    
    # Test with no paths raises error
    toml_content = """
    [tool.vulture]
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)
    
    # Test with invalid config key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)
    
    # Test with wrong type
    toml_content = """
    [tool.vulture]
    min_confidence = "invalid"
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)
    
    # Test verbose output when TOML file is detected
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_content)
    with patch('builtins.print') as mock_print:
        config = make_config(tomlfile=toml_file)
        mock_print.assert_called_once()
        assert "Reading configuration from" in mock_print.call_args[0][0]


# LLM-generated content at query #79
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    paths = ["src"]
    min_confidence = 30
    exclude = ["test_*.py"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src"]
    assert config["min_confidence"] == 30
    assert config["exclude"] == ["test_*.py"]

    # Test TOML overridden by CLI
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["custom_path", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["test_*.py"]

    # Test with missing paths raises error
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\nmin_confidence = 10\n"))

    # Test with invalid TOML data type
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\nmin_confidence = 'invalid'\n"))

    # Test with unknown configuration key
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.StringIO("[tool.vulture]\nunknown_key = 10\npaths = ['test']\n"))

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--sort-by-size", "--make-whitelist"])
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True

    # Test with exclude as comma-separated string
    config = make_config(argv=["--exclude", "file1.py,file2.py", "path"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["path"]

    # Test with ignore decorators and names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "--ignore-names", "visit_*,do_*", "path"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["path"]
```


# LLM-generated content at query #80
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1.py", "path2.py"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["src"]
    exclude = ["test_*.py"]
    verbose = true
    """
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True

    # Test CLI arguments override TOML settings
    toml_file = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["src"]

    # Test with invalid configuration key
    toml_file = io.StringIO("""
    [tool.vulture]
    invalid_key = true
    """)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    toml_file = io.StringIO("""
    [tool.vulture]
    min_confidence = "high"
    """)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test with no paths
    try:
        make_config([], tomlfile=io.StringIO(""))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths from CLI
    config = make_config(["somefile.py"])
    assert config["paths"] == ["somefile.py"]

    # Test with --exclude and --ignore-decorators and --ignore-names
    config = make_config(
        ["--exclude", "test_*.py,docs", "--ignore-decorators", "@app.route", "--ignore-names", "private_*", "file.py"]
    )
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["private_*"]
    assert config["paths"] == ["file.py"]

    # Test with boolean flags
    config = make_config(["--make-whitelist", "--sort-by-size", "--verbose", "file.py"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with --config argument pointing to non-existent file
    config = make_config(["--config", "nonexistent.toml", "file.py"])
    assert config["paths"] == ["file.py"]
    assert config["config"] == "nonexistent.toml"


# LLM-generated content at query #81
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 30
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src"]

    # Test CLI arguments override TOML settings
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with exclude pattern from CLI
    config = make_config(argv=["--exclude", "file1.py,file2.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with ignore decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with ignore names
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with make-whitelist
    config = make_config(argv=["--make-whitelist"])
    assert config["make_whitelist"] is True

    # Test with sort-by-size
    config = make_config(argv=["--sort-by-size"])
    assert config["sort_by_size"] is True

    # Test with no paths should raise InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid config key in TOML
    toml_content_invalid = """
    [tool.vulture]
    invalid_key = 10
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content_invalid.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_content_wrong_type = """
    [tool.vulture]
    min_confidence = "high"
    paths = ["src"]
    """
    toml_file = io.BytesIO(toml_content_wrong_type.encode('utf-8'))
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with explicit config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("[tool.vulture]\nmin_confidence = 42\npaths = ['test_path']\n")
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["min_confidence"] == 42
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)

    # Test with verbose and TOML file (should print message)
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True
```


# LLM-generated content at query #82
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py"]
    min_confidence = 30
    verbose = true
    """
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py"]
    assert config["min_confidence"] == 30
    assert config["verbose"] is True

    # Test that CLI overrides TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with mixed TOML and CLI
    toml_file = io.StringIO(toml_data)
    config = make_config(["--make-whitelist", "path"], tomlfile=toml_file)
    assert config["make_whitelist"] is True
    assert config["paths"] == ["path"]
    assert config["exclude"] == ["file*.py"]

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config([])


# LLM-generated content at query #83
#--------------------------

```python
def test_make_config():
    # Test with default values
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "-v"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    min_confidence = 20
    exclude = ["test_*.py"]
    paths = ["src"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src"]

    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80

    # Test with missing paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b""))

    # Test with unknown configuration key in TOML
    bad_toml = b"""
    [tool.vulture]
    unknown_key = "value"
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(bad_toml))

    # Test with wrong type in TOML
    bad_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(bad_type_toml))
```


# LLM-generated content at query #84
#--------------------------

```python
def test_make_config(monkeypatch, tmp_path):
    # Test with no CLI args and no toml file (should use defaults)
    config = make_config([])
    assert config == DEFAULTS
    
    # Test with CLI args overriding defaults
    cli_args = ["--min-confidence", "50", "--verbose", "test.py"]
    config = make_config(cli_args)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]
    
    # Test with toml file
    toml_content = b"""
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
    toml_file = tmp_path / "test.toml"
    toml_file.write_bytes(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config([], tomlfile=f)
    
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test CLI args take precedence over toml
    toml_content = b"""
    [tool.vulture]
    min_confidence = 10
    paths = ["path1"]
    """
    toml_file = tmp_path / "precedence.toml"
    toml_file.write_bytes(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(["--min-confidence", "80", "cli_path"], tomlfile=f)
    
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]
    
    # Test with pyproject.toml auto-detection
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
    [tool.vulture]
    min_confidence = 20
    paths = ["auto_path"]
    """)
    
    monkeypatch.chdir(tmp_path)
    config = make_config([])
    assert config["min_confidence"] == 20
    assert config["paths"] == ["auto_path"]
    
    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(["--min-confidence", "30"])
    
    # Test with config file that has invalid key
    bad_toml = b"""
    [tool.vulture]
    invalid_key = true
    paths = ["test.py"]
    """
    bad_file = tmp_path / "bad.toml"
    bad_file.write_bytes(bad_toml)
    
    with open(bad_file, "rb") as f:
        with pytest.raises(InputError):
            make_config([], tomlfile=f)


# LLM-generated content at query #85
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test that CLI arguments override TOML settings
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]  # from TOML
    assert config["verbose"] is True  # from TOML

    # Test error for no paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with invalid configuration key
    toml_file = io.StringIO("[tool.vulture]\ninvalid_key = true\npaths = ['test.py']")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type
    toml_file = io.StringIO("[tool.vulture]\nmin_confidence = 'not_an_int'\npaths = ['test.py']")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with --version
    with pytest.raises(SystemExit) as exc_info:
        make_config(argv=["--version"])
    assert exc_info.value.code == 0

    # Test with --help
    with pytest.raises(SystemExit) as exc_info:
        make_config(argv=["--help"])
    assert exc_info.value.code == 0
```


# LLM-generated content at query #86
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file (should use defaults)
    config = make_config()
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50

    # Test with toml file
    import io
    toml_content = b"""
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
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override toml settings
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True  # from toml

    # Test missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test invalid toml key raises InputError
    bad_toml = b"""
    [tool.vulture]
    invalid_key = true
    paths = ["test.py"]
    """
    with pytest.raises(InputError):
        make_config(tomlfile=io.BytesIO(bad_toml))

    # Test wrong type in toml raises InputError
    wrong_type_toml = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    with pytest.raises(InputError):
        make_config(tomlfile=io.BytesIO(wrong_type_toml))

    # Test config file path resolution
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "wb") as f:
            f.write(toml_content)
        config = make_config(argv=["--config", config_path])
        assert config["min_confidence"] == 10
        assert config["paths"] == ["path1", "path2"]

    # Test verbose output when reading from toml file
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["verbose"] is True

    # Test that make_whitelist and sort_by_size work correctly
    tomlfile = io.BytesIO(toml_content)
    config = make_config(tomlfile=tomlfile)
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #87
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file
    config = make_config([])
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1"]
    min_confidence = 25
    """
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config([], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["min_confidence"] == 25

    # Test CLI arguments override toml file
    tomlfile = io.BytesIO(toml_content)
    config = make_config(["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1"]

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config([])

    # Test with unknown configuration key in toml
    bad_toml = io.BytesIO(b"""
    [tool.vulture]
    invalid_key = "value"
    """)
    with pytest.raises(InputError):
        make_config([], tomlfile=bad_toml)

    # Test with wrong type in toml
    bad_toml_type = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError):
        make_config([], tomlfile=bad_toml_type)

    # Test with verbose and toml file
    tomlfile = io.BytesIO(toml_content)
    config = make_config(["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 25

    # Test exclude as comma-separated string
    config = make_config(["--exclude", "file1.py,file2.py", "path"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["paths"] == ["path"]

    # Test default config file path
    config = make_config(["--config", "custom.toml", "path"])
    assert config["config"] == "custom.toml"

    # Test make_whitelist and sort_by_size flags
    config = make_config(["--make-whitelist", "--sort-by-size", "path"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #88
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file content
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
    toml_file = io.StringIO(toml_content)
    toml_file.name = "test_toml.toml"
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with both TOML and CLI (CLI takes precedence)
    config = make_config(argv=["cli_path.py", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    assert config["verbose"] is True  # From TOML

    # Test missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "50"])  # No paths provided

    # Test invalid config key raises InputError
    invalid_toml = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path.py"]
    """
    invalid_file = io.StringIO(invalid_toml)
    invalid_file.name = "invalid.toml"
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=invalid_file)

    # Test wrong type raises InputError
    wrong_type_toml = """
    [tool.vulture]
    paths = ["path.py"]
    min_confidence = "not_an_int"
    """
    wrong_type_file = io.StringIO(wrong_type_toml)
    wrong_type_file.name = "wrong_type.toml"
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=wrong_type_file)
```


# LLM-generated content at query #89
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file (uses defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(["--min-confidence", "50", "path1", "path2"])
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test*.py", "temp/"]
    ignore_decorators = ["deco1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 75
    sort_by_size = true
    verbose = true
    paths = ["src/"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["test*.py", "temp/"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 75
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

    # Test CLI arguments override toml settings
    toml_file = io.StringIO(toml_content)
    config = make_config(["--min-confidence", "90", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    assert config["exclude"] == ["test*.py", "temp/"]  # from toml

    # Test with unknown configuration key raises InputError
    import pytest
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    """
    toml_file = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test with wrong data type raises InputError
    bad_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(bad_toml)
    with pytest.raises(InputError):
        make_config([], tomlfile=toml_file)

    # Test with no paths raises InputError
    with pytest.raises(InputError):
        make_config([], tomlfile=io.StringIO(""))

    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        config_file = f.name
    try:
        config = make_config(["--config", config_file])
        assert config["min_confidence"] == 75
        assert config["verbose"] is True
    finally:
        os.unlink(config_file)

    # Test with non-existent config file uses defaults
    config = make_config(["--config", "nonexistent.toml", "path"])
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["paths"] == ["path"]

    # Test with comma-separated exclude
    config = make_config(["--exclude", "a.py,b.py,c.py", "path"])
    assert config["exclude"] == ["a.py", "b.py", "c.py"]
```


# LLM-generated content at query #90
#--------------------------

```python
def test_make_config():
    # Test with no CLI arguments and no TOML file should raise InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with paths provided via CLI
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["path1", "--min-confidence", "50", "--verbose", "--make-whitelist"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["make_whitelist"] == True

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file and CLI overrides
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] == True

    # Test with invalid TOML configuration key
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path1"]
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with no paths in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_file)

    # Test with --version (should not raise InputError)
    config = make_config(argv=["--version"])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
```


# LLM-generated content at query #91
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no TOML file
    config = make_config([])
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with CLI arguments
    cli_args = ["path1", "path2", "--verbose", "--min-confidence", "50"]
    config = make_config(cli_args)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50

    # Test with TOML file
    toml_content = b"""
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 70
    verbose = true
    """
    config = make_config([], tomlfile=io.BytesIO(toml_content))
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 70
    assert config["verbose"] is True

    # Test TOML + CLI override (CLI takes precedence)
    config = make_config(
        ["--min-confidence", "80"], tomlfile=io.BytesIO(toml_content)
    )
    assert config["min_confidence"] == 80
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test with missing paths raises InputError
    try:
        make_config([])
    except InputError as e:
        assert "at least one file or directory" in str(e)
    else:
        assert False, "Expected InputError for missing paths"

    # Test with unknown TOML key raises InputError
    bad_toml_content = b"""
    [tool.vulture]
    unknown_key = "value"
    paths = ["test.py"]
    """
    try:
        make_config([], tomlfile=io.BytesIO(bad_toml_content))
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    else:
        assert False, "Expected InputError for unknown key"

    # Test with wrong type in TOML
    wrong_type_toml = b"""
    [tool.vulture]
    paths = ["test.py"]
    min_confidence = "high"
    """
    try:
        make_config([], tomlfile=io.BytesIO(wrong_type_toml))
    except InputError as e:
        assert "Data type" in str(e)
    else:
        assert False, "Expected InputError for wrong type"

    # Test with wrong type in CLI
    try:
        make_config(["--min-confidence", "not_a_number"])
    except SystemExit:
        pass  # argparse will exit with error
    else:
        assert False, "Expected SystemExit for invalid CLI type"

    # Test with config file path that doesn't exist
    config = make_config(["--config", "nonexistent.toml", "test.py"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0  # default applied

    # Test with make_whitelist and sort_by_size CLI flags
    config = make_config(
        ["--make-whitelist", "--sort-by-size", "test.py"]
    )
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True

    # Test with exclude, ignore_decorators, ignore_names
    config = make_config(
        [
            "--exclude", "*.pyc,test_*.py",
            "--ignore-decorators", "@app.route,@require_*",
            "--ignore-names", "visit_*,do_*",
            "main.py"
        ]
    )
    assert config["exclude"] == ["*.pyc", "test_*.py"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["main.py"]
```


# LLM-generated content at query #92
#--------------------------

```python
def test_make_config():
    # Test basic config with no arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50

    # Test with TOML file
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML settings
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value retained
    
    # Test that empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])  # No paths provided

    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        config_path = f.name
    try:
        config = make_config(argv=["--config", config_path, "test_path"])
        assert config["min_confidence"] == 10
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(config_path)

    # Test unknown configuration key raises InputError
    tomlfile = io.BytesIO(b"[tool.vulture]\nunknown_key = true\n")
    with pytest.raises(InputError):
        make_config(argv=["test"], tomlfile=tomlfile)

    # Test wrong data type raises InputError
    tomlfile = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'invalid'\n")
    with pytest.raises(InputError):
        make_config(argv=["test"], tomlfile=tomlfile)
```


# LLM-generated content at query #93
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[])
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(argv=["--verbose", "--min-confidence", "50", "src"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src"]
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []

    # Test with TOML file
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
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI precedence over TOML
    toml_file = io.BytesIO(b"[tool.vulture]\nmin_confidence = 10\n")
    config = make_config(argv=["--min-confidence", "75"], tomlfile=toml_file)
    assert config["min_confidence"] == 75

    # Test with no paths
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with invalid config key
    toml_file = io.BytesIO(b"[tool.vulture]\ninvalid_key = 1\n")
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type
    toml_file = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'invalid'\n")
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("[tool.vulture]\nmin_confidence = 42\n")
        temp_name = f.name
    try:
        config = make_config(argv=["--config", temp_name, "test.py"])
        assert config["min_confidence"] == 42
        assert config["paths"] == ["test.py"]
    finally:
        os.unlink(temp_name)

    # Test with --version (should not raise InputError)
    config = make_config(argv=["--version"])
    assert isinstance(config, dict)

    # Test with --help (should not raise InputError)
    config = make_config(argv=["--help"])
    assert isinstance(config, dict)
```


# LLM-generated content at query #94
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file (should use defaults)
    config = make_config(argv=[], tomlfile=None)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "50", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file providing config values
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["toml_path1", "toml_path2"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test that CLI arguments take precedence over TOML values
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["cli_path.py", "--min-confidence", "90"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 90
    assert config["exclude"] == ["file*.py", "dir/"]  # from TOML

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)

    # Test with invalid config key
    with pytest.raises(InputError):
        make_config(argv=["--invalid-option"], tomlfile=None)

    # Test with wrong data type
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"], tomlfile=None)

    # Test with config file that doesn't exist (should use defaults)
    config = make_config(argv=["path.py", "--config", "nonexistent.toml"], tomlfile=None)
    assert config["paths"] == ["path.py"]
    assert config["config"] == "nonexistent.toml"
    assert config["min_confidence"] == 0

    # Test with --make-whitelist flag
    config = make_config(argv=["path.py", "--make-whitelist"], tomlfile=None)
    assert config["make_whitelist"] is True

    # Test with --sort-by-size flag
    config = make_config(argv=["path.py", "--sort-by-size"], tomlfile=None)
    assert config["sort_by_size"] is True

    # Test with --exclude option
    config = make_config(argv=["path.py", "--exclude", "file1.py,file2.py"], tomlfile=None)
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test with --ignore-decorators option
    config = make_config(argv=["path.py", "--ignore-decorators", "@app.route,@require_*"], tomlfile=None)
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names option
    config = make_config(argv=["path.py", "--ignore-names", "visit_*,do_*"], tomlfile=None)
    assert config["ignore_names"] == ["visit_*", "do_*"]
```


# LLM-generated content at query #95
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
    import io
    config = make_config(argv=[])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 50

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "20"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20

    # Test defaults are applied for missing keys
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test paths are required
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test unknown configuration key
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=["path"], tomlfile=toml_file)

    # Test wrong data type
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(argv=["path"], tomlfile=toml_file)


# LLM-generated content at query #96
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--min-confidence", "50"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 50

    # Test with tomlfile
    import io
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
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80

    # Test defaults are filled
    tomlfile = io.BytesIO(b"[tool.vulture]\npaths = ['path1']")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b""))
```


# LLM-generated content at query #97
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(
        argv=["--min-confidence", "50", "--verbose", "path1", "path2"],
        tomlfile=None
    )
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
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
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments override TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(
        argv=["--min-confidence", "20", "--verbose"],
        tomlfile=toml_file
    )
    assert config["min_confidence"] == 20
    assert config["verbose"] is True

    # Test with --exclude and --ignore-decorators as comma-separated values
    config = make_config(
        argv=["--exclude", "file1.py,file2.py", "--ignore-decorators", "deco1,deco2"],
        tomlfile=None
    )
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]

    # Test with --make-whitelist and --sort-by-size flags
    config = make_config(
        argv=["--make-whitelist", "--sort-by-size", "path"],
        tomlfile=None
    )
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True

    # Test with --config pointing to a non-existent file
    config = make_config(
        argv=["--config", "nonexistent.toml", "path"],
        tomlfile=None
    )
    assert config["paths"] == ["path"]

    # Test with invalid config key in TOML
    bad_toml = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(bad_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong type in TOML
    bad_toml = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(bad_toml)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError for wrong type"
    except InputError as e:
        assert "Data type" in str(e)

    # Test with no paths
    try:
        make_config(argv=[], tomlfile=None)
        assert False, "Expected InputError for no paths"
    except InputError as e:
        assert "at least one file or directory" in str(e)
```


# LLM-generated content at query #98
#--------------------------

```python
def test_make_config(tmp_path, capsys):
    # Test 1: No config file, no CLI args (should raise InputError for no paths)
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test 2: CLI args only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test 3: TOML file with settings
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
    toml_file = tmp_path / "test_config.toml"
    toml_file.write_text(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(argv=[], tomlfile=f)
    
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test 4: CLI overrides TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    paths = ["toml_path.py"]
    """
    toml_file = tmp_path / "override_config.toml"
    toml_file.write_text(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(argv=["cli_path.py", "--min-confidence", "20"], tomlfile=f)
    
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 20

    # Test 5: Verbose output with TOML file
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["path.py"]
    """
    toml_file = tmp_path / "verbose_config.toml"
    toml_file.write_text(toml_content)
    
    with open(toml_file, "rb") as f:
        config = make_config(argv=[], tomlfile=f)
    
    captured = capsys.readouterr()
    assert f"Reading configuration from {toml_file}" in captured.out

    # Test 6: Config file auto-detection from pyproject.toml
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
    [tool.vulture]
    paths = ["auto_detected.py"]
    """)
    
    config = make_config(argv=["--config", str(pyproject)])
    assert config["paths"] == ["auto_detected.py"]

    # Test 7: Invalid config key
    with pytest.raises(InputError):
        _check_input_config({"invalid_key": "value"})

    # Test 8: Invalid config value type
    with pytest.raises(InputError):
        _check_input_config({"min_confidence": "not_an_int"})
```


# LLM-generated content at query #99
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile (should use defaults)
    config = make_config([])
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments overriding defaults
    config = make_config(["--min-confidence", "50", "--verbose", "test.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]

    # Test with tomlfile
    import io
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
    toml_file = io.StringIO(toml_data)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI arguments take precedence over toml
    toml_file = io.StringIO(toml_data)
    config = make_config(["--min-confidence", "20", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml

    # Test with invalid toml config
    toml_data = """
    [tool.vulture]
    invalid_key = "value"
    """
    toml_file = io.StringIO(toml_data)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with no paths (should raise InputError)
    try:
        make_config([], tomlfile=io.StringIO(""))
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)

    # Test with paths provided via CLI
    config = make_config(["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]

    # Test with config file specified via CLI
    import tempfile
    import pathlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 30
        paths = ["test_path"]
        """)
        temp_path = f.name
    config = make_config(["--config", temp_path])
    assert config["min_confidence"] == 30
    assert config["paths"] == ["test_path"]
    pathlib.Path(temp_path).unlink()

    # Test with --version flag (should not raise InputError)
    try:
        make_config(["--version"])
    except SystemExit:
        pass  # argparse exits with SystemExit

    # Test with --help flag (should not raise InputError)
    try:
        make_config(["--help"])
    except SystemExit:
        pass  # argparse exits with SystemExit
```


# LLM-generated content at query #100
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (uses defaults)
    result = make_config([])
    assert result == DEFAULTS
    
    # Test with CLI arguments overriding defaults
    cli_args = ["-v", "--min-confidence", "50", "path1.py", "path2.py"]
    result = make_config(cli_args)
    assert result["verbose"] == True
    assert result["min_confidence"] == 50
    assert result["paths"] == ["path1.py", "path2.py"]
    assert result["config"] == "pyproject.toml"
    
    # Test with toml file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 80
    verbose = true
    paths = ["src", "tests"]
    """
    toml_file = io.StringIO(toml_content)
    result = make_config([], toml_file)
    assert result["exclude"] == ["test_*.py"]
    assert result["min_confidence"] == 80
    assert result["verbose"] == True
    assert result["paths"] == ["src", "tests"]
    
    # Test that CLI arguments take precedence over toml
    toml_file = io.StringIO(toml_content)
    result = make_config(["--min-confidence", "30"], toml_file)
    assert result["min_confidence"] == 30
    
    # Test with invalid config key in toml
    toml_content = """
    [tool.vulture]
    invalid_key = true
    paths = ["test.py"]
    """
    toml_file = io.StringIO(toml_content)
    try:
        make_config([], toml_file)
        assert False, "Should raise InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)
    
    # Test with wrong type in toml
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    toml_file = io.StringIO(toml_content)
    try:
        make_config([], toml_file)
        assert False, "Should raise InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)
    
    # Test with no paths provided
    try:
        make_config([])
        assert False, "Should raise InputError when no paths provided"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #101
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config([])
    assert config == DEFAULTS

    # Test with CLI arguments
    cli_args = ["--min-confidence", "50", "path1", "path2"]
    config = make_config(cli_args)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 30
    paths = ["toml_path1", "toml_path2"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config([], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test CLI overrides TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path1"]
    """
    toml_file = io.StringIO(toml_content)
    cli_args = ["--min-confidence", "80", "cli_path"]
    config = make_config(cli_args, tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]

    # Test with flags
    cli_args = ["--make-whitelist", "--sort-by-size", "--verbose", "path"]
    config = make_config(cli_args)
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test with exclude pattern
    cli_args = ["--exclude", "file1.py,file2.py", "path"]
    config = make_config(cli_args)
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test error when no paths provided
    try:
        make_config(["--verbose"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"

    # Test error with unknown configuration key
    toml_content = """
    [tool.vulture]
    invalid_key = true
    """
    toml_file = io.StringIO(toml_content)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test error with wrong type
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    """
    toml_file = io.StringIO(toml_content)
    try:
        make_config([], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)
```


# LLM-generated content at query #102
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file (should use defaults)
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path1", "toml_path2"]
    exclude = ["excluded_file.py"]
    verbose = true
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["excluded_file.py"]
    assert config["verbose"] is True

    # Test that CLI arguments take precedence over TOML
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    paths = ["toml_path"]
    """
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["cli_path", "--min-confidence", "80"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80

    # Test with empty paths raises InputError
    try:
        make_config(argv=[])
        assert False, "Expected InputError for empty paths"
    except InputError as e:
        assert "at least one file or directory" in str(e)

    # Test with invalid configuration key in TOML
    toml_content = """
    [tool.vulture]
    invalid_key = "value"
    paths = ["path"]
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError for invalid key"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test with wrong data type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["path"]
    """
    tomlfile = io.StringIO(toml_content)
    try:
        make_config(argv=[], tomlfile=tomlfile)
        assert False, "Expected InputError for wrong type"
    except InputError as e:
        assert "Data type for min_confidence" in str(e)

    # Test with TOML file that doesn't exist (should use defaults)
    config = make_config(argv=["--config", "nonexistent.toml", "path"])
    assert config["paths"] == ["path"]
    assert config["min_confidence"] == 0
```


# LLM-generated content at query #103
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=None)  # Should raise InputError due to empty paths

    # Test with paths provided via CLI
    config = make_config(argv=["path1", "path2"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # Test with CLI arguments
    config = make_config(
        argv=[
            "--min-confidence", "50",
            "--exclude", "file1.py,file2.py",
            "--ignore-decorators", "deco1,@deco2",
            "--ignore-names", "name1,name2",
            "--make-whitelist",
            "--sort-by-size",
            "--verbose",
            "path1",
        ],
        tomlfile=None
    )
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["deco1", "@deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1"]

    # Test with TOML file
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
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file and CLI overrides
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(
        argv=["--min-confidence", "80", "--verbose", "extra_path"],
        tomlfile=tomlfile
    )
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2", "extra_path"]

    # Test with invalid TOML config
    invalid_toml = """
[tool.vulture]
invalid_key = true
"""
    tomlfile = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=["path"], tomlfile=tomlfile)

    # Test with wrong types in TOML
    wrong_type_toml = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError):
        make_config(argv=["path"], tomlfile=tomlfile)

    # Test with custom config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_path = f.name
    try:
        config = make_config(argv=["--config", temp_path, "extra_path"])
        assert config["min_confidence"] == 10
        assert config["paths"] == ["path1", "path2", "extra_path"]
    finally:
        os.unlink(temp_path)

    # Test with non-existent config file
    config = make_config(argv=["--config", "nonexistent.toml", "path"])
    assert config["paths"] == ["path"]
    assert config["min_confidence"] == 0

    # Test verbose output with TOML file
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose"], tomlfile=tomlfile)
    assert config["verbose"] is True
```


# LLM-generated content at query #104
#--------------------------

```python
def test_make_config():
    # Test with no arguments (no pyproject.toml in current directory)
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--min-confidence", "50", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test with TOML file (mock a simple TOML content)
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
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content.encode("utf-8"))
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    # TOML values should still be present for other keys
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["paths"] == ["path1", "path2"]

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\n"))

    # Test with invalid config key
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\ninvalid_key = 1\n"))

    # Test with wrong type
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_int'\n"))
```


