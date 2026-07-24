####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config():
    # Test with CLI args only
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["exclude"] == []
    assert config["config"] == "pyproject.toml"

    # Test with CLI args overriding defaults
    config = make_config(argv=[
        "--verbose",
        "--sort-by-size",
        "--make-whitelist",
        "--min-confidence", "50",
        "--exclude", "test_*.py,venv",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "visit_*",
        "path1",
    ])
    assert config["paths"] == ["path1"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    assert config["exclude"] == ["test_*.py", "venv"]

    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    paths = ["path1", "path2"]
    min_confidence = 30
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 30
    assert config["verbose"] is True

    # Test TOML with CLI override
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test error with unknown key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    """
    bad_file = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(tomlfile=bad_file)

    # Test error with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    verbose = "string instead of bool"
    """
    wrong_type_file = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(tomlfile=wrong_type_file)

    # Test error with wrong type from CLI
    with pytest.raises(SystemExit):  # argparse handles type errors
        make_config(argv=["--min-confidence", "not_an_int"])


# LLM-generated content at query #2
#--------------------------

```python
def test_make_config():
    # Test with CLI args only (no TOML file)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] is False

    # Test with TOML file (using StringIO)
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["test_*.py", "venv"]
    min_confidence = 50
    verbose = true
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["src"]

    # Test CLI args override TOML
    toml_file = io.StringIO(toml_data)
    config = make_config(argv=["--verbose", "mypath.py"], tomlfile=toml_file)
    assert config["verbose"] is True
    assert config["paths"] == ["mypath.py"]
    assert config["min_confidence"] == 50  # Still from TOML

    # Test with no paths (should raise InputError)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with --help (should not raise error about paths)
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])

    # Test with --version (should not raise error about paths)
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config():
    # Test 1: Default config when no arguments and no toml file
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["path1", "path2", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False

    # Test 3: TOML file configuration
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path1", "toml_path2"]
min_confidence = 50
verbose = true
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test 4: CLI arguments override TOML configuration
    toml_content = """
[tool.vulture]
paths = ["toml_path1"]
min_confidence = 50
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path1", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path1"]
    assert config["min_confidence"] == 80

    # Test 5: Exclude patterns from CLI
    config = make_config(argv=["path1", "--exclude", "test_*.py,venv"])
    assert "test_*.py" in config["exclude"]
    assert "venv" in config["exclude"]

    # Test 6: Ignore decorators and names from CLI
    config = make_config(argv=["path1", "--ignore-decorators", "@app.route,@require_*", "--ignore-names", "visit_*,do_*"])
    assert "@app.route" in config["ignore_decorators"]
    assert "@require_*" in config["ignore_decorators"]
    assert "visit_*" in config["ignore_names"]
    assert "do_*" in config["ignore_names"]

    # Test 7: Make whitelist and sort by size from CLI
    config = make_config(argv=["path1", "--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True

    # Test 8: Custom config file path (simulating non-existent file)
    config = make_config(argv=["path1", "--config", "nonexistent.toml"])
    assert config["paths"] == ["path1"]
    assert config["config"] == "nonexistent.toml"

    # Test 9: Unknown configuration key in TOML raises InputError
    toml_content = """
[tool.vulture]
unknown_key = "value"
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file)

    # Test 10: Wrong type in TOML configuration raises InputError
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_file)

    # Test 11: Missing paths in config raises InputError
    toml_content = """
[tool.vulture]
verbose = true
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=toml_file)


# LLM-generated content at query #4
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with CLI paths only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path.py"])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["path.py"]

    # Test 3: TOML file config with CLI override
    import io
    toml_data = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 30
    verbose = true
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src/"]

    # Test 4: TOML file config only
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    sort_by_size = true
    make_whitelist = true
    """)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == True
    assert config["paths"] == []

    # Test 5: Empty config raises error (no paths)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 6: CLI config with all options
    config = make_config(argv=[
        "path1.py",
        "path2.py",
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--min-confidence", "75",
        "--sort-by-size",
        "--verbose",
        "--config", "custom.toml"
    ])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 75
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["config"] == "custom.toml"

    # Test 7: TOML with unknown key raises error
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    unknown_key = true
    """)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["path.py"], tomlfile=toml_file)

    # Test 8: TOML with wrong type raises error
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path.py"], tomlfile=toml_file)

    # Test 9: CLI with wrong type raises error
    with pytest.raises(SystemExit):
        make_config(argv=["--min-confidence", "not_an_int", "path.py"])


# LLM-generated content at query #5
#--------------------------

```python
def test_make_config():
    """Test make_config with CLI arguments and TOML file."""
    # Test 1: Basic CLI arguments only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    
    # Test 2: CLI arguments with options
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path.py"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path.py"]
    
    # Test 3: TOML file overrides defaults, CLI overrides TOML
    toml_content = b"""
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 30
    verbose = true
    paths = ["src/"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "custom.py"], tomlfile=toml_file)
    assert config["paths"] == ["custom.py"]  # CLI overrides TOML
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["exclude"] == ["test_*.py"]  # From TOML
    assert config["verbose"] is True  # From TOML
    
    # Test 4: Empty CLI paths should raise InputError
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test 5: Invalid config key should raise InputError
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["test.py", "--invalid-option"])
    
    # Test 6: Boolean options work correctly
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "test.py"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    
    # Test 7: CSV options are properly parsed
    config = make_config(argv=["--exclude", "file1.py,file2.py,dir/", "--ignore-decorators", "@app.route,@login_required", "test.py"])
    assert config["exclude"] == ["file1.py", "file2.py", "dir/"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    
    # Test 8: TOML file with all options
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py"]
    ignore_decorators = ["@deco1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 90
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["@deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 90
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
```


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] == False
    assert config["sort_by_size"] == False
    assert config["make_whitelist"] == False
    assert config["config"] == "pyproject.toml"

    # Test 2: Config with TOML file (simulated)
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    ignore_names = ["test_*"]
    verbose = true
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["extra_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["extra_path.py"]  # CLI overrides TOML
    assert config["min_confidence"] == 50
    assert config["ignore_names"] == ["test_*"]
    assert config["verbose"] == True
    
    # Test 3: CLI arguments override TOML
    toml_content2 = """
    [tool.vulture]
    min_confidence = 30
    verbose = false
    """
    toml_file2 = io.BytesIO(toml_content2.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path.py"], tomlfile=toml_file2)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    assert config["paths"] == ["path.py"]

    # Test 4: Config with multiple CLI options
    config = make_config(argv=[
        "--exclude", "test_*.py,*.pyc",
        "--ignore-decorators", "@app.route,@login_required",
        "--ignore-names", "helper_*,internal_*",
        "--make-whitelist",
        "--sort-by-size",
        "--min-confidence", "75",
        "file1.py", "dir/"
    ])
    assert config["exclude"] == ["test_*.py", "*.pyc"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    assert config["ignore_names"] == ["helper_*", "internal_*"]
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["min_confidence"] == 75
    assert config["paths"] == ["file1.py", "dir/"]

    # Test 5: Config with custom config file path (but no actual file exists)
    config = make_config(argv=["--config", "custom_config.toml", "path.py"])
    assert config["config"] == "custom_config.toml"
    assert config["paths"] == ["path.py"]

    # Test 6: Empty TOML file (no vulture section)
    toml_file_empty = io.BytesIO(b"")
    config = make_config(argv=["path.py"], tomlfile=toml_file_empty)
    assert config["paths"] == ["path.py"]
    assert config["min_confidence"] == 0

    # Test 7: TOML file with empty vulture section
    toml_file_empty_vulture = io.BytesIO(b"[tool.vulture]\n")
    config = make_config(argv=["path.py"], tomlfile=toml_file_empty_vulture)
    assert config["paths"] == ["path.py"]
    assert config["min_confidence"] == 0

    # Test 8: Verify all default values are present
    config = make_config(argv=["test.py"])
    for key in DEFAULTS:
        assert key in config, f"Missing key: {key}"


# LLM-generated content at query #7
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with no arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["sort_by_size"] == False
    assert config["make_whitelist"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test 3: CLI arguments with values
    config = make_config(argv=["--min-confidence", "50", "--exclude", "test*.py,venv"])
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test*.py", "venv"]

    # Test 4: TOML config via tomlfile parameter
    import io
    toml_content = """
[tool.vulture]
verbose = true
min_confidence = 30
paths = ["src", "tests"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src", "tests"]

    # Test 5: CLI arguments override TOML config
    toml_content2 = """
[tool.vulture]
verbose = false
min_confidence = 10
"""
    toml_file2 = io.BytesIO(toml_content2.encode())
    config = make_config(argv=["--verbose", "--min-confidence", "80"], tomlfile=toml_file2)
    assert config["verbose"] == True  # CLI overrides TOML
    assert config["min_confidence"] == 80  # CLI overrides TOML

    # Test 6: TOML config with list values
    toml_content3 = """
[tool.vulture]
exclude = ["*.pyc", "__pycache__"]
ignore_decorators = ["@login_required"]
ignore_names = ["test_*"]
"""
    toml_file3 = io.BytesIO(toml_content3.encode())
    config = make_config(argv=[], tomlfile=toml_file3)
    assert config["exclude"] == ["*.pyc", "__pycache__"]
    assert config["ignore_decorators"] == ["@login_required"]
    assert config["ignore_names"] == ["test_*"]

    # Test 7: Custom config file path via CLI
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        toml_content4 = """
[tool.vulture]
verbose = true
"""
        f.write(toml_content4.encode())
        temp_path = f.name
    
    config = make_config(argv=["--config", temp_path])
    assert config["verbose"] == True
    os.unlink(temp_path)

    # Test 8: InputError for empty paths
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 9: InputError for unknown configuration key
    toml_content5 = """
[tool.vulture]
unknown_key = true
"""
    toml_file5 = io.BytesIO(toml_content5.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file5)

    # Test 10: InputError for wrong data type
    toml_content6 = """
[tool.vulture]
verbose = "not_a_boolean"
"""
    toml_file6 = io.BytesIO(toml_content6.encode())
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=toml_file6)


# LLM-generated content at query #8
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Default config with no CLI args and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"
    
    # Test 2: CLI args override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path1", "path2"]
    
    # Test 3: TOML file config
    import io
    toml_content = """
    [tool.vulture]
    verbose = true
    min_confidence = 50
    paths = ["src/", "tests/"]
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/", "tests/"]
    
    # Test 4: CLI args override TOML config
    tomlfile.seek(0)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True  # From TOML
    
    # Test 5: CSV parsing for exclude
    config = make_config(argv=["--exclude", "file1.py,file2.py,dir/"])
    assert config["exclude"] == ["file1.py", "file2.py", "dir/"]
    
    # Test 6: CSV parsing for ignore-decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    
    # Test 7: CSV parsing for ignore-names
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test 8: make-whitelist flag
    config = make_config(argv=["--make-whitelist"])
    assert config["make_whitelist"] == True
    
    # Test 9: Custom config file path
    config = make_config(argv=["--config", "custom_config.toml"])
    assert config["config"] == "custom_config.toml"
    
    # Test 10: Empty paths should raise InputError
    try:
        make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e.message) == "Please pass at least one file or directory"
    
    # Test 11: Unknown configuration key should raise InputError
    try:
        make_config(argv=["--unknown-key", "value"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)
    
    # Test 12: Wrong type should raise InputError
    try:
        make_config(argv=["--min-confidence", "not_an_int"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)
    
    # Test 13: Multiple paths as positional arguments
    config = make_config(argv=["path1.py", "path2.py", "path3/"])
    assert config["paths"] == ["path1.py", "path2.py", "path3/"]
    
    # Test 14: All options combined
    toml_content = """
    [tool.vulture]
    sort_by_size = true
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(
        argv=["--verbose", "--min-confidence", "75", "--exclude", "test_*", "src/"],
        tomlfile=tomlfile
    )
    assert config["verbose"] == True
    assert config["min_confidence"] == 75
    assert config["exclude"] == ["test_*"]
    assert config["sort_by_size"] == True  # From TOML
    assert config["paths"] == ["src/"]
```


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
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
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with toml file
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py", "venv"]
ignore_decorators = ["decorator1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 30
sort_by_size = true
verbose = true
paths = ["src"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["src"]

    # Test that CLI overrides toml
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    assert config["exclude"] == ["test_*.py", "venv"]  # from toml

    # Test that empty paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with custom config file path
    config = make_config(argv=["--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == []  # defaults

    # Test with paths provided
    config = make_config(argv=["file1.py", "file2.py"])
    assert config["paths"] == ["file1.py", "file2.py"]
```


# LLM-generated content at query #10
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
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file (simulated)
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
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test CLI arguments override TOML values
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    # Other values should still come from TOML
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["paths"] == ["path1", "path2"]

    # Test with custom config file path that doesn't exist
    config = make_config(argv=["--config", "nonexistent.toml", "test_path"])
    assert config["paths"] == ["test_path"]
    assert config["config"] == "nonexistent.toml"
    # Should use defaults for missing options
    assert config["min_confidence"] == 0
    assert config["verbose"] == False

    # Test exception when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test exception with unknown config key in TOML
    tomlfile = io.BytesIO(b"[tool.vulture]\nunknown_key = true")
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=tomlfile)

    # Test exception with wrong type in CLI
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])


# LLM-generated content at query #11
#--------------------------

```python
def test_make_config():
    # Test 1: Basic configuration with CLI args only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["sort_by_size"] == False

    # Test 2: CLI args override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "test.py"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["test.py"]

    # Test 3: TOML file configuration
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
min_confidence = 50
verbose = true
paths = ["src/"]
"""
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["src/"]

    # Test 4: CLI args override TOML settings
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "other.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["exclude"] == ["test_*.py"]  # Still from TOML
    assert config["paths"] == ["other.py"]  # CLI overrides TOML paths

    # Test 5: Empty TOML with CLI args
    empty_toml = io.StringIO("[tool.vulture]\n")
    config = make_config(argv=["--verbose", "file.py"], tomlfile=empty_toml)
    assert config["verbose"] == True
    assert config["paths"] == ["file.py"]
    assert config["min_confidence"] == 0

    # Test 6: Default values when nothing specified
    config = make_config(argv=["test.py"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False

    # Test 7: CSV parsing for exclude
    config = make_config(argv=["--exclude", "*.pyc,__pycache__", "test.py"])
    assert config["exclude"] == ["*.pyc", "__pycache__"]

    # Test 8: CSV parsing for ignore-decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@login_required", "test.py"])
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]

    # Test 9: CSV parsing for ignore-names
    config = make_config(argv=["--ignore-names", "private_*,helper_*", "test.py"])
    assert config["ignore_names"] == ["private_*", "helper_*"]

    # Test 10: InputError on unknown config key
    import pytest
    bad_toml = io.StringIO("[tool.vulture]\nunknown_key = true")
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["test.py"], tomlfile=bad_toml)

    # Test 11: InputError on wrong type
    bad_type_toml = io.StringIO("[tool.vulture]\nmin_confidence = \"not_an_int\"")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["test.py"], tomlfile=bad_type_toml)

    # Test 12: InputError when no paths given
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 13: Config with all options set
    import io
    full_toml = io.StringIO("""
[tool.vulture]
exclude = ["excluded.py"]
ignore_decorators = ["@decorator1"]
ignore_names = ["name1"]
make_whitelist = true
min_confidence = 75
sort_by_size = true
verbose = true
paths = ["path1", "path2"]
""")
    config = make_config(argv=[], tomlfile=full_toml)
    assert config["exclude"] == ["excluded.py"]
    assert config["ignore_decorators"] == ["@decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 75
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test 14: --config argument
    config = make_config(argv=["--config", "custom.toml", "test.py"])
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["test.py"]

    # Test 15: Multiple paths
    config = make_config(argv=["file1.py", "file2.py", "dir1/"])
    assert config["paths"] == ["file1.py", "file2.py", "dir1/"]

    # Test 16: --make-whitelist flag
    config = make_config(argv=["--make-whitelist", "test.py"])
    assert config["make_whitelist"] == True

    # Test 17: --min-confidence with value
    config = make_config(argv=["--min-confidence", "100", "test.py"])
    assert config["min_confidence"] == 100

    # Test 18: TOML with empty section
    empty_section_toml = io.StringIO("[tool.other]\nkey = 'value'\n[tool.vulture]\n")
    config = make_config(argv=["test.py"], tomlfile=empty_section_toml)
    assert config["paths"] == ["test.py"]
    assert config["exclude"] == []

    # Test 19: TOML with no [tool.vulture] section
    no_vulture_toml = io.StringIO("[tool.other]\nkey = 'value'\n")
    config = make_config(argv=["test.py"], tomlfile=no_vulture_toml)
    assert config["paths"] == ["test.py"]

    # Test 20: Verify defaults are set for missing keys
    partial_toml = io.StringIO("[tool.vulture]\nverbose = true\npaths = ['test.py']\n")
    config = make_config(argv=[], tomlfile=partial_toml)
    assert config["verbose"] == True
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
```


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config():
    # Test basic config with no args and no toml
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS

    # Test config with CLI args
    config = make_config(argv=["path1", "path2", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 0

    # Test config with CLI args overriding defaults
    config = make_config(argv=["--min-confidence", "50", "--sort-by-size"], tomlfile=None)
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True

    # Test config with TOML file
    import io
    toml_content = b"""
    [tool.vulture]
    verbose = true
    min_confidence = 30
    paths = ["toml_path1", "toml_path2"]
    """
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 30
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test that CLI args override TOML config
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path_cli"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path_cli"]  # CLI paths override TOML paths

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    
    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "path1", "path2"], tomlfile=None)
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    
    # Test with tomlfile
    import io
    toml_data = """
[tool.vulture]
min_confidence = 50
paths = ["toml_path"]
verbose = true
"""
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 50
    assert config["paths"] == ["toml_path"]
    assert config["verbose"] == True
    
    # Test CLI overrides toml
    toml_data = """
[tool.vulture]
min_confidence = 50
paths = ["toml_path"]
"""
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]
    
    # Test with non-existent config file
    config = make_config(argv=["--config", "nonexistent.toml"], tomlfile=None)
    assert config == DEFAULTS
    
    # Test that InputError is raised when no paths are provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"], tomlfile=None)


# LLM-generated content at query #14
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no tomlfile
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments only
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile only
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    verbose = true
    paths = ["path3", "path4"]
    """
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["verbose"] == True
    assert config["paths"] == ["path3", "path4"]

    # Test CLI overrides tomlfile
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    verbose = false
    """
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True

    # Test with exclude, ignore_decorators, ignore_names
    config = make_config(argv=[
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*"
    ])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --make-whitelist and --sort-by-size
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test default values are set when not provided
    config = make_config(argv=["--verbose"])
    assert config["min_confidence"] == 0
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False

    # Test InputError when no paths provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test InputError with unknown key in toml
    toml_data = """
    [tool.vulture]
    unknown_key = true
    paths = ["test.py"]
    """
    tomlfile = io.BytesIO(toml_data.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=tomlfile)


# LLM-generated content at query #15
#--------------------------

```python
def test_make_config():
    # Test with CLI args only (no TOML file)
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test with CLI args overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path"]

    # Test with TOML file
    import io
    toml_data = io.BytesIO(b"""
[tool.vulture]
min_confidence = 30
ignore_decorators = ["deco1", "deco2"]
paths = ["toml_path1", "toml_path2"]
""")
    config = make_config(argv=[], tomlfile=toml_data)
    assert config["min_confidence"] == 30
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["verbose"] is False

    # Test CLI args override TOML
    toml_data = io.BytesIO(b"""
[tool.vulture]
min_confidence = 30
paths = ["toml_path"]
""")
    config = make_config(argv=["--min-confidence", "80", "cli_path"], tomlfile=toml_data)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path"]

    # Test InputError for unknown key in TOML
    toml_data = io.BytesIO(b"""
[tool.vulture]
unknown_key = "value"
""")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_data)

    # Test InputError for no paths
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test InputError for wrong type in CLI
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int"])

    # Test with boolean flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "path"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True


# LLM-generated content at query #16
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Default config when no arguments provided and no toml file
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
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test 3: TOML config with CLI override
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    verbose = false
    paths = ["toml_path"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["verbose"] is True  # CLI overrides TOML
    assert config["paths"] == ["toml_path"]
    
    # Test 4: TOML config with lists
    toml_data = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test 5: TOML config with booleans
    toml_data = """
    [tool.vulture]
    make_whitelist = true
    sort_by_size = true
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    
    # Test 6: Empty TOML config
    toml_data = ""
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
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
    
    # Test 7: TOML config without tool.vulture section
    toml_data = """
    [tool.other]
    value = 42
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
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
    
    # Test 8: CLI arguments with lists
    config = make_config(argv=["--exclude", "file1.py,file2.py", 
                                "--ignore-decorators", "dec1,dec2",
                                "--ignore-names", "name1,name2"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test 9: CLI boolean flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    
    # Test 10: Error when no paths provided
    try:
        make_config(argv=["--verbose"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e) == "Please pass at least one file or directory"


# LLM-generated content at query #17
#--------------------------

```python
def test_make_config():
    """Test make_config with various combinations of CLI args and TOML config."""
    
    # Test 1: Basic CLI args only (no TOML file)
    argv = ["file1.py", "file2.py"]
    config = make_config(argv=argv, tomlfile=None)
    assert config["paths"] == ["file1.py", "file2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"
    
    # Test 2: TOML config only (no CLI args)
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src/", "tests/"]
    exclude = ["*.pyc", "__pycache__"]
    min_confidence = 80
    verbose = true
    sort_by_size = true
    """
    toml_file = io.StringIO(toml_content)
    # Need to convert to bytes-like for tomllib
    toml_bytes = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_bytes)
    assert config["paths"] == ["src/", "tests/"]
    assert config["exclude"] == ["*.pyc", "__pycache__"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    
    # Test 3: CLI args override TOML config
    toml_content2 = """
    [tool.vulture]
    paths = ["toml_path/"]
    min_confidence = 50
    verbose = false
    """
    toml_bytes2 = io.BytesIO(toml_content2.encode('utf-8'))
    argv = ["cli_path.py", "--verbose", "--min-confidence", "90"]
    config = make_config(argv=argv, tomlfile=toml_bytes2)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 90
    assert config["verbose"] is True
    
    # Test 4: CLI args with various options
    argv = [
        "path1.py", "path2.py",
        "--exclude", "test_*,docs",
        "--ignore-decorators", "@staticmethod,@classmethod",
        "--ignore-names", "private_*,internal_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "--min-confidence", "75",
        "--config", "custom_config.toml"
    ]
    config = make_config(argv=argv, tomlfile=None)
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["exclude"] == ["test_*", "docs"]
    assert config["ignore_decorators"] == ["@staticmethod", "@classmethod"]
    assert config["ignore_names"] == ["private_*", "internal_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["min_confidence"] == 75
    assert config["config"] == "custom_config.toml"
    
    # Test 5: Empty paths should raise InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)
    
    # Test 6: TOML with unknown key should raise InputError
    toml_content3 = """
    [tool.vulture]
    paths = ["test.py"]
    unknown_key = "value"
    """
    toml_bytes3 = io.BytesIO(toml_content3.encode('utf-8'))
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=toml_bytes3)
    
    # Test 7: TOML with wrong type should raise InputError
    toml_content4 = """
    [tool.vulture]
    paths = ["test.py"]
    verbose = "not_a_bool"
    """
    toml_bytes4 = io.BytesIO(toml_content4.encode('utf-8'))
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=toml_bytes4)
    
    # Test 8: CLI args with wrong type should raise InputError
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["test.py", "--min-confidence", "not_an_int"], tomlfile=None)
    
    # Test 9: Empty TOML config (no tool.vulture section)
    toml_content5 = """
    [tool.other_tool]
    some_setting = true
    """
    toml_bytes5 = io.BytesIO(toml_content5.encode('utf-8'))
    config = make_config(argv=["test.py"], tomlfile=toml_bytes5)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    
    # Test 10: Complete TOML config with all options
    toml_content6 = """
    [tool.vulture]
    paths = ["module1/", "module2/"]
    exclude = ["old_*"]
    ignore_decorators = ["@deprecated"]
    ignore_names = ["unused_*"]
    make_whitelist = true
    min_confidence = 100
    sort_by_size = true
    verbose = true
    """
    toml_bytes6 = io.BytesIO(toml_content6.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_bytes6)
    assert config["paths"] == ["module1/", "module2/"]
    assert config["exclude"] == ["old_*"]
    assert config["ignore_decorators"] == ["@deprecated"]
    assert config["ignore_names"] == ["unused_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 100
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
```


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config():
    # Test 1: Default config with no arguments
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 50

    # Test 3: TOML file config (using StringIO to simulate file)
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    """
    toml_file = io.StringIO(toml_content)
    # Need to encode to bytes for tomllib
    toml_bytes = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_bytes)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test 4: CLI arguments override TOML config
    toml_content2 = """
    [tool.vulture]
    min_confidence = 10
    verbose = false
    """
    toml_bytes2 = io.BytesIO(toml_content2.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_bytes2)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True

    # Test 5: Missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test 6: Custom config file path via CLI
    config = make_config(argv=["--config", "custom.toml", "test_path"])
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["test_path"]

    # Test 7: Paths provided
    config = make_config(argv=["src", "tests"])
    assert config["paths"] == ["src", "tests"]

    # Test 8: Edge case - empty paths with TOML providing paths
    toml_content3 = """
    [tool.vulture]
    paths = ["src", "lib"]
    """
    toml_bytes3 = io.BytesIO(toml_content3.encode())
    config = make_config(argv=[], tomlfile=toml_bytes3)
    assert config["paths"] == ["src", "lib"]

    # Test 9: Edge case - ignore_names and ignore_decorators as CSV
    config = make_config(argv=["--ignore-names", "name1,name2", "--ignore-decorators", "dec1,dec2", "test_path"])
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]


# LLM-generated content at query #19
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with no arguments
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

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test 3: TOML file configuration
    import io
    toml_content = b"""
[tool.vulture]
exclude = ["file*.py", "dir/"]
verbose = true
min_confidence = 20
"""
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 20

    # Test 4: CLI arguments override TOML settings
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    assert config["verbose"] == True  # From TOML

    # Test 5: Empty TOML file
    tomlfile = io.BytesIO(b"")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test 6: TOML with empty vulture section
    tomlfile = io.BytesIO(b"[tool.vulture]")
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test 7: CLI arguments with comma-separated values
    config = make_config(argv=["--exclude", "file*.py,dir/", "--ignore-decorators", "@app.route,@require_*"])
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test 8: make_whitelist and sort_by_size flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test 9: Custom config file path (should not find it)
    config = make_config(argv=["--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    # Should still work with defaults since file doesn't exist

    # Test 10: Multiple paths as positional arguments
    config = make_config(argv=["path1.py", "path2.py", "dir/"])
    assert config["paths"] == ["path1.py", "path2.py", "dir/"]

    # Test 11: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 12: Error with unknown configuration key in TOML
    tomlfile = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["path1"], tomlfile=tomlfile)

    # Test 13: Error with wrong type in TOML
    tomlfile = io.BytesIO(b'[tool.vulture]\nmin_confidence = "not_an_int"')
    with pytest.raises(InputError, match="Data type for min_confidence must be"):
        make_config(argv=["path1"], tomlfile=tomlfile)


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    import io
    import tempfile
    import os

    # Test 1: Basic TOML config with all defaults
    toml_content = b"""
    [tool.vulture]
    paths = ["src/"]
    """
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        toml_path = f.name

    config = make_config(tomlfile=io.BytesIO(toml_content))
    assert config["paths"] == ["src/"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] is False

    # Test 2: CLI arguments override TOML config
    config = make_config(
        argv=["--verbose", "src/"],
        tomlfile=io.BytesIO(b"[tool.vulture]\nverbose = false")
    )
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

    # Test 3: CLI arguments without TOML file
    config = make_config(argv=["--verbose", "src/"])
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

    # Test 4: TOML file from pyproject.toml in current directory
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("pyproject.toml", "wb") as f:
                f.write(b"[tool.vulture]\nmin_confidence = 50")
            config = make_config(argv=[])
            assert config["min_confidence"] == 50
            assert config["paths"] == []
        finally:
            os.chdir(original_dir)

    # Test 5: CSV parsing for exclude, ignore_decorators, ignore_names
    config = make_config(
        argv=["--exclude", "*.pyc,test_*", "src/"]
    )
    assert config["exclude"] == ["*.pyc", "test_*"]
    assert config["paths"] == ["src/"]

    # Test 6: Boolean flags
    config = make_config(
        argv=["--make-whitelist", "--sort-by-size", "src/"]
    )
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True

    # Test 7: Custom config file path
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(b"[tool.vulture]\nmin_confidence = 75")
        custom_config = f.name
    config = make_config(argv=["--config", custom_config, "src/"])
    assert config["min_confidence"] == 75
    os.unlink(custom_config)

    # Test 8: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 9: TOML file not found, using CLI args only
    config = make_config(argv=["--verbose", "src/"])
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

    # Test cleanup
    if os.path.exists(toml_path):
        os.unlink(toml_path)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_make_config():
    # Test with default values (no TOML file, no CLI args)
    config = make_config(argv=[])
    assert config["paths"] == DEFAULTS["paths"]
    assert config["min_confidence"] == DEFAULTS["min_confidence"]
    assert config["exclude"] == DEFAULTS["exclude"]
    assert config["ignore_decorators"] == DEFAULTS["ignore_decorators"]
    assert config["ignore_names"] == DEFAULTS["ignore_names"]
    assert config["make_whitelist"] == DEFAULTS["make_whitelist"]
    assert config["sort_by_size"] == DEFAULTS["sort_by_size"]
    assert config["verbose"] == DEFAULTS["verbose"]
    assert config["config"] == DEFAULTS["config"]

    # Test CLI arguments override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test TOML config file is used when provided
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    min_confidence = 50
    verbose = true
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]

    # Test CLI arguments override TOML config
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["exclude"] == ["test_*.py"]  # From TOML
    assert config["verbose"] is True  # From TOML

    # Test InputError is raised when paths is empty
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with --make-whitelist flag
    config = make_config(argv=["--make-whitelist", "file.py"])
    assert config["make_whitelist"] is True
    assert config["paths"] == ["file.py"]

    # Test with --ignore-decorators and --ignore-names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "--ignore-names", "visit_*,do_*", "file.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["file.py"]

    # Test verbose output when TOML config is detected
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["verbose"] is True
    assert config["paths"] == ["test.py"]


# LLM-generated content at query #22
#--------------------------

```python
def test_make_config():
    # Test 1: Default config (no args, no toml file)
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test 2: CLI arguments override defaults
    config = make_config(
        argv=["--verbose", "--make-whitelist", "--sort-by-size", "path1", "path2"],
        tomlfile=None
    )
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test 3: TOML file configuration
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
ignore_decorators = ["@app.route"]
ignore_names = ["visit_*"]
min_confidence = 50
sort_by_size = true
paths = ["src/"]
verbose = true
"""
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src/"]
    assert config["verbose"] is True

    # Test 4: CLI arguments override TOML configuration
    toml_file2 = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(
        argv=["--verbose", "--min-confidence=80", "new_path"],
        tomlfile=toml_file2
    )
    assert config["verbose"] is True
    assert config["min_confidence"] == 80
    assert config["paths"] == ["new_path"]
    assert config["exclude"] == ["test_*.py"]  # from toml
    assert config["sort_by_size"] is True  # from toml

    # Test 5: Empty paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test 6: Invalid config key raises InputError
    toml_invalid = io.BytesIO(b"[tool.vulture]\ninvalid_key = true")
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(argv=[], tomlfile=toml_invalid)

    # Test 7: Wrong data type in TOML raises InputError
    toml_wrong_type = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_wrong_type)


# LLM-generated content at query #23
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

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file configuration
    import io
    toml_data = """
    [tool.vulture]
    verbose = true
    min_confidence = 75
    paths = ["toml_path1", "toml_path2"]
    """
    toml_file = io.StringIO(toml_data)
    toml_file.buffer = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file.buffer)
    assert config["verbose"] == True
    assert config["min_confidence"] == 75
    assert config["paths"] == ["toml_path1", "toml_path2"]

    # Test with CLI arguments overriding TOML configuration
    toml_data = """
    [tool.vulture]
    verbose = false
    min_confidence = 10
    """
    toml_file = io.StringIO(toml_data)
    toml_file.buffer = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--verbose", "--min-confidence", "90"], tomlfile=toml_file.buffer)
    assert config["verbose"] == True
    assert config["min_confidence"] == 90

    # Test with --exclude CLI argument
    config = make_config(argv=["--exclude", "test_*.py,*.pyc"])
    assert config["exclude"] == ["test_*.py", "*.pyc"]

    # Test with --ignore-decorators CLI argument
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with --ignore-names CLI argument
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with --make-whitelist and --sort-by-size
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test that InputError is raised when no paths are given
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with custom config file path that doesn't exist
    config = make_config(argv=["--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == []  # Still empty, will raise InputError
```


# LLM-generated content at query #24
#--------------------------

```python
def test_make_config():
    import io
    import tempfile
    
    # Test 1: Default configuration with no arguments
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
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test 3: TOML file configuration
    toml_content = b"""
    [tool.vulture]
    min_confidence = 30
    verbose = true
    paths = ["toml_path1", "toml_path2"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["verbose"] == True
    assert config["paths"] == ["toml_path1", "toml_path2"]
    
    # Test 4: CLI arguments override TOML configuration
    toml_content = b"""
    [tool.vulture]
    min_confidence = 30
    verbose = false
    paths = ["toml_path"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]
    
    # Test 5: TOML file with exclude, ignore_decorators, ignore_names
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    ignore_names = ["name1", "name2"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test 6: TOML file with make_whitelist and sort_by_size
    toml_content = b"""
    [tool.vulture]
    make_whitelist = true
    sort_by_size = true
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    
    # Test 7: Unknown configuration key in TOML raises InputError
    toml_content = b"""
    [tool.vulture]
    unknown_key = "test"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)
    
    # Test 8: Wrong data type in TOML raises InputError
    toml_content = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test.py"]
    """
    toml_file = io.BytesIO(toml_content)
    try:
        make_config(argv=[], tomlfile=toml_file)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)
    
    # Test 9: No paths provided raises InputError
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e.message)
    
    # Test 10: CLI arguments with multiple values
    config = make_config(argv=[
        "--exclude", "file1.py,file2.py",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "path1", "path2"
    ])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["paths"] == ["path1", "path2"]
    
    # Test 11: Config file from CLI argument
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b"""
        [tool.vulture]
        min_confidence = 75
        paths = ["toml_path"]
        """)
        temp_path = f.name
    
    config = make_config(argv=["--config", temp_path])
    assert config["min_confidence"] == 75
    assert config["paths"] == ["toml_path"]
    
    # Test 12: Non-existent config file falls back to defaults
    config = make_config(argv=["--config", "nonexistent.toml", "test_path"])
    assert config["min_confidence"] == 0
    assert config["paths"] == ["test_path"]


# LLM-generated content at query #25
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
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "path1.py", "path2.py"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0

    # Test with CLI arguments and toml file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    verbose = true
    exclude = ["test_*.py"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--sort-by-size"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["exclude"] == ["test_*.py"]

    # Test that CLI arguments override toml
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True

    # Test with all CLI arguments
    config = make_config(argv=[
        "--exclude", "dir1,dir2",
        "--ignore-decorators", "deco1,deco2",
        "--ignore-names", "name1,name2",
        "--make-whitelist",
        "--min-confidence", "100",
        "--sort-by-size",
        "-v",
        "path1.py", "path2.py"
    ])
    assert config["exclude"] == ["dir1", "dir2"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 100
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test that InputError is raised when no paths are given
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--verbose"])

    # Test that InputError is raised for unknown configuration key
    with pytest.raises(InputError):
        toml_file = io.BytesIO(b"""
        [tool.vulture]
        unknown_key = true
        paths = ["test.py"]
        """)
        make_config(argv=[], tomlfile=toml_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config():
    # Test basic configuration with no arguments
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

    # Test CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test TOML configuration
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

    # Test CLI arguments override TOML configuration
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path3"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["paths"] == ["path3"]

    # Test error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test error with unknown configuration key
    toml_content = """
[tool.vulture]
unknown_key = true
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path1"], tomlfile=toml_file)

    # Test error with wrong data type
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path1"], tomlfile=toml_file)

    # Test that config argument is used for TOML file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
verbose = true
paths = ["test_path"]
""")
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["verbose"] is True
        assert config["paths"] == ["test_path"]
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #27
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False

    # Test with TOML file
    import io
    toml_data = """
    [tool.vulture]
    paths = ["src/"]
    min_confidence = 50
    verbose = true
    """
    toml_file = io.StringIO(toml_data)
    # Convert to bytes since tomllib expects bytes
    toml_bytes = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_bytes)
    assert config["paths"] == ["src/"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test CLI overrides TOML
    toml_bytes = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["--min-confidence", "80", "custom_path.py"], tomlfile=toml_bytes)
    assert config["paths"] == ["custom_path.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test with empty TOML and no CLI paths raises InputError
    empty_toml = io.BytesIO(b"[tool.vulture]\n")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=empty_toml)

    # Test with unknown config key in TOML
    bad_toml = io.BytesIO(b"[tool.vulture]\nunknown_key = true\npaths = ['test.py']")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=bad_toml)

    # Test with wrong type in TOML
    wrong_type_toml = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'\npaths = ['test.py']")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=wrong_type_toml)


# LLM-generated content at query #28
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with no arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "file1.py", "file2.py"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["file1.py", "file2.py"]
    
    # Test 3: Config from TOML file
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py"]
    ignore_decorators = ["deprecated"]
    verbose = true
    paths = ["src/"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["deprecated"]
    assert config["verbose"] is True
    assert config["paths"] == ["src/"]
    
    # Test 4: CLI arguments override TOML config
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True  # Both set to True
    
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML default
    
    # Test 5: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test 6: Custom config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        paths = ["custom_path.py"]
        """)
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["paths"] == ["custom_path.py"]
    finally:
        os.unlink(temp_path)
    
    # Test 7: Various CLI options
    config = make_config(argv=[
        "--exclude", "test_*.py,dir1",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "helper_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "path1.py", "path2.py"
    ])
    assert config["exclude"] == ["test_*.py", "dir1"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["helper_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1.py", "path2.py"]
    
    # Test 8: Default values are set for missing options
    config = make_config(argv=["file.py"])
    for key, value in DEFAULTS.items():
        if key != "paths":  # paths is provided
            assert config[key] == value


# LLM-generated content at query #29
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with no arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["sort_by_size"] == False
    assert config["make_whitelist"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"

    # Test 2: Config with CLI arguments
    config = make_config(argv=["path1.py", "path2.py", "--verbose", "--sort-by-size"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["verbose"] == True
    assert config["sort_by_size"] == True

    # Test 3: Config with TOML file (using StringIO)
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    make_whitelist = true
    min_confidence = 10
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10

    # Test 4: CLI arguments override TOML config
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "50"], tomlfile=toml_file)
    assert config["min_confidence"] == 50  # CLI overrides TOML

    # Test 5: Empty TOML file
    empty_toml = io.BytesIO(b"")
    config = make_config(argv=[], tomlfile=empty_toml)
    assert config["paths"] == []
    assert config["min_confidence"] == 0

    # Test 6: Config with all CLI arguments
    config = make_config(argv=[
        "path1.py", "path2.py",
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "visit_*",
        "--make-whitelist",
        "--min-confidence", "80",
        "--sort-by-size",
        "--verbose"
    ])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test 7: Custom config file path
    config = make_config(argv=["--config", "custom_config.toml"])
    assert config["config"] == "custom_config.toml"

    # Test 8: Paths argument with multiple files
    config = make_config(argv=["file1.py", "file2.py", "file3.py"])
    assert len(config["paths"]) == 3
    assert config["paths"] == ["file1.py", "file2.py", "file3.py"]

    # Test 9: Verify that empty paths raises InputError
    import pytest
    with pytest.raises(InputError) as exc_info:
        make_config(argv=["--make-whitelist"])
    assert "Please pass at least one file or directory" in str(exc_info.value)


# LLM-generated content at query #30
#--------------------------

```python
def test_make_config():
    # Test with CLI args only (no TOML file)
    config = make_config(argv=["path1", "path2", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] == False

    # Test with TOML file provided
    import io
    toml_content = b"""
[tool.vulture]
paths = ["src", "tests"]
verbose = true
min_confidence = 50
"""
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src", "tests"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 50

    # Test CLI args override TOML settings
    toml_file.seek(0)
    config = make_config(argv=["custom_path", "--min-confidence=80"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path"]
    assert config["min_confidence"] == 80
    assert config["verbose"] == True  # from TOML

    # Test with empty CLI args and no TOML
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False

    # Test defaults when no config provided
    config = make_config(argv=["test.py"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False

    # Test InputError raised when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test with minimal CLI arguments (paths only)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=[
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "visit_*",
        "--make-whitelist",
        "--min-confidence", "50",
        "--sort-by-size",
        "--verbose",
        "path1.py"
    ])
    assert config["paths"] == ["path1.py"]
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

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
    config = make_config(argv=["path3.py"], tomlfile=toml_file)
    assert config["paths"] == ["path3.py"]  # CLI overrides TOML
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test with TOML file and no CLI paths
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]

    # Test with empty TOML file (only tool.vulture section missing)
    toml_file = io.BytesIO(b"[tool.other]\nkey = 'value'")
    config = make_config(argv=["path1.py"], tomlfile=toml_file)
    assert config["paths"] == ["path1.py"]
    assert config["min_confidence"] == 0

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with unknown configuration key in TOML
    toml_file = io.BytesIO(b"[tool.vulture]\nunknown_key = 'value'")
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path1.py"], tomlfile=toml_file)

    # Test with wrong data type in TOML
    toml_file = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path1.py"], tomlfile=toml_file)


# LLM-generated content at query #32
#--------------------------

```python
def test_make_config():
    # Test 1: Default configuration with no arguments
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

    # Test 2: CLI arguments override defaults
    config = make_config(
        argv=["--verbose", "--min-confidence=50", "src/"],
        tomlfile=None
    )
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/"]

    # Test 3: TOML configuration is loaded and merged
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
ignore_decorators = ["decorator1"]
sort_by_size = true
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == False  # default

    # Test 4: CLI arguments override TOML configuration
    toml_content = """
[tool.vulture]
verbose = true
min_confidence = 30
    """
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(
        argv=["--verbose", "--min-confidence=80"],
        tomlfile=tomlfile
    )
    assert config["verbose"] == True
    assert config["min_confidence"] == 80

    # Test 5: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\n"))

    # Test 6: Error with invalid config key
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(
            argv=[],
            tomlfile=io.BytesIO(b'[tool.vulture]\ninvalid_key = "value"'),
        )

    # Test 7: Error with wrong data type
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(
            argv=[],
            tomlfile=io.BytesIO(b'[tool.vulture]\nverbose = "yes"'),
        )


# LLM-generated content at query #33
#--------------------------

```python
def test_make_config():
    """Test make_config with various scenarios."""
    # Test 1: Basic config with no arguments
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["sort_by_size"] == False
    assert config["make_whitelist"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"

    # Test 2: Config with CLI arguments
    config = make_config(argv=[
        "--verbose",
        "--min-confidence", "50",
        "--sort-by-size",
        "--make-whitelist",
        "--exclude", "test_*.py,*.pyc",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "path1", "path2"
    ], tomlfile=None)
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == True
    assert config["exclude"] == ["test_*.py", "*.pyc"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["path1", "path2"]

    # Test 3: Config with TOML file (using StringIO)
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
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test 4: CLI arguments override TOML settings
    toml_content = """
[tool.vulture]
min_confidence = 10
verbose = false
paths = ["toml_path1", "toml_path2"]
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[
        "--min-confidence", "80",
        "--verbose",
        "cli_path1"
    ], tomlfile=tomlfile)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True  # CLI overrides TOML
    assert config["paths"] == ["cli_path1"]  # CLI overrides TOML

    # Test 5: Missing paths should raise InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"], tomlfile=None)

    # Test 6: Config with custom config file path
    config = make_config(argv=["--config", "custom.toml"], tomlfile=None)
    assert config["config"] == "custom.toml"

    # Test 7: Empty TOML config (no [tool.vulture] section)
    toml_content = """
[other_tool]
some_option = true
"""
    tomlfile = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["path1"], tomlfile=tomlfile)
    assert config["paths"] == ["path1"]
    assert config["verbose"] == False

    # Test 8: Default values are set for missing options
    config = make_config(argv=["path1"], tomlfile=None)
    for key, value in DEFAULTS.items():
        if key != "config":  # config has special handling
            assert key in config
            if key != "paths":  # paths is provided
                assert config[key] == value


# LLM-generated content at query #34
#--------------------------

```python
def test_make_config():
    """Test make_config with various scenarios."""
    import io
    
    # Test with CLI args only (no toml file)
    result = make_config(argv=["path1.py", "path2.py"])
    assert result["paths"] == ["path1.py", "path2.py"]
    assert result["min_confidence"] == 0
    assert result["verbose"] is False
    assert result["sort_by_size"] is False
    
    # Test with toml file
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    verbose = true
    paths = ["toml_path.py"]
    """
    toml_file = io.StringIO(toml_content)
    toml_file.name = "test_config"
    # Convert StringIO to BytesIO for binary mode
    toml_bytes = io.BytesIO(toml_content.encode('utf-8'))
    toml_bytes.name = "test_config"
    result = make_config(argv=["cli_path.py"], tomlfile=toml_bytes)
    # CLI should override toml
    assert result["paths"] == ["cli_path.py"]
    assert result["min_confidence"] == 50
    assert result["verbose"] is True
    
    # Test with CLI overriding toml
    toml_bytes.seek(0)
    result = make_config(argv=["--min-confidence=80", "cli_path.py"], tomlfile=toml_bytes)
    assert result["min_confidence"] == 80
    assert result["paths"] == ["cli_path.py"]
    
    # Test with empty toml content
    empty_toml = io.BytesIO(b"[tool.vulture]\n")
    empty_toml.name = "empty_config"
    result = make_config(argv=["test.py"], tomlfile=empty_toml)
    assert result["paths"] == ["test.py"]
    assert result["min_confidence"] == 0
    
    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test with unknown key in toml
    bad_toml = io.BytesIO(b"""
    [tool.vulture]
    unknown_key = 123
    """)
    bad_toml.name = "bad_config"
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["test.py"], tomlfile=bad_toml)
    
    # Test with wrong type in toml
    wrong_type_toml = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    wrong_type_toml.name = "wrong_type_config"
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["test.py"], tomlfile=wrong_type_toml)
    
    # Test make_whitelist flag
    result = make_config(argv=["--make-whitelist", "test.py"])
    assert result["make_whitelist"] is True
    
    # Test sort_by_size flag
    result = make_config(argv=["--sort-by-size", "test.py"])
    assert result["sort_by_size"] is True
    
    # Test exclude option
    result = make_config(argv=["--exclude=*.pyc,__pycache__", "test.py"])
    assert result["exclude"] == ["*.pyc", "__pycache__"]
    
    # Test ignore_decorators
    result = make_config(argv=["--ignore-decorators=@app.route,@login_required", "test.py"])
    assert result["ignore_decorators"] == ["@app.route", "@login_required"]
    
    # Test ignore_names
    result = make_config(argv=["--ignore-names=visit_*,do_*", "test.py"])
    assert result["ignore_names"] == ["visit_*", "do_*"]
    
    # Test config option with non-existent file (should use defaults)
    result = make_config(argv=["--config=nonexistent.toml", "test.py"])
    assert result["paths"] == ["test.py"]
    assert result["config"] == "nonexistent.toml"


# LLM-generated content at query #35
#--------------------------

```python
def test_make_config():
    # Test with CLI args only (no TOML file)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["sort_by_size"] == False

    # Test with TOML file
    import io
    toml_content = """
[tool.vulture]
paths = ["src/"]
min_confidence = 50
verbose = true
"""
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["src/"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test CLI args override TOML settings
    toml_content = """
[tool.vulture]
paths = ["src/"]
min_confidence = 50
"""
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "custom_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path.py"]
    assert config["min_confidence"] == 80

    # Test defaults are applied for missing options
    config = make_config(argv=["test.py"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["make_whitelist"] == False

    # Test InputError for wrong type in CLI args
    import pytest
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int", "test.py"])

    # Test InputError for unknown config key in TOML
    toml_content = """
[tool.vulture]
unknown_key = "value"
paths = ["test.py"]
"""
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)

    # Test InputError when no paths given
    with pytest.raises(InputError):
        make_config(argv=[])


# LLM-generated content at query #36
#--------------------------

```python
def test_make_config():
    # Test with default config (no args, no tomlfile)
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile providing config
    import io
    toml_data = """
    [tool.vulture]
    min_confidence = 30
    paths = ["path_from_toml"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["paths"] == ["path_from_toml"]
    assert config["verbose"] == True

    # Test CLI args override tomlfile values
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["paths"] == ["path_from_toml"]  # TOML value preserved

    # Test with make_whitelist flag
    config = make_config(argv=["--make-whitelist", "test.py"])
    assert config["make_whitelist"] == True
    assert config["paths"] == ["test.py"]

    # Test with sort_by_size flag
    config = make_config(argv=["--sort-by-size", "test.py"])
    assert config["sort_by_size"] == True

    # Test with exclude patterns
    config = make_config(argv=["--exclude", "test_*.py,*.pyc", "src"])
    assert "test_*.py" in config["exclude"]
    assert "*.pyc" in config["exclude"]
    assert config["paths"] == ["src"]

    # Test with ignore_decorators and ignore_names
    config = make_config(argv=[
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "src"
    ])
    assert "@app.route" in config["ignore_decorators"]
    assert "@require_*" in config["ignore_decorators"]
    assert "visit_*" in config["ignore_names"]
    assert "do_*" in config["ignore_names"]

    # Test empty paths raises InputError
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test unknown config key in toml
    bad_toml = io.BytesIO("""
    [tool.vulture]
    unknown_key = "test"
    paths = ["test.py"]
    """.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=bad_toml)

    # Test config file path from CLI
    config = make_config(argv=["--config", "nonexistent.toml", "test.py"])
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0  # default preserved

    # Test version flag doesn't raise error
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])


# LLM-generated content at query #37
#--------------------------

```python
def test_make_config():
    # Test basic config with paths from CLI
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    
    # Test config with TOML file
    import io
    toml_content = b"""
    [tool.vulture]
    min_confidence = 50
    verbose = true
    paths = ["toml_path1", "toml_path2"]
    """
    toml_file = io.BytesIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["toml_path1", "toml_path2"]
    
    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "75", "cli_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["cli_path"]
    
    # Test boolean flags
    config = make_config(argv=["--verbose", "--sort-by-size", "path"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    
    # Test exclude with comma-separated values
    config = make_config(argv=["--exclude", "*.pyc,__pycache__,test_*", "path"])
    assert config["exclude"] == ["*.pyc", "__pycache__", "test_*"]
    
    # Test ignore_decorators and ignore_names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", 
                                "--ignore-names", "visit_*,do_*", "path"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test make_whitelist
    config = make_config(argv=["--make-whitelist", "path"])
    assert config["make_whitelist"] is True
    
    # Test default config file path
    config = make_config(argv=["path"])
    assert config["config"] == "pyproject.toml"
    
    # Test custom config file path
    config = make_config(argv=["--config", "custom.toml", "path"])
    assert config["config"] == "custom.toml"
    
    # Test default values are set when not provided
    config = make_config(argv=["path"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    
    # Test InputError when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test InputError for invalid config key in TOML
    toml_content_invalid = b"""
    [tool.vulture]
    invalid_key = true
    paths = ["test"]
    """
    toml_file = io.BytesIO(toml_content_invalid)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(tomlfile=toml_file)
    
    # Test InputError for wrong type in TOML
    toml_content_wrong_type = b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    paths = ["test"]
    """
    toml_file = io.BytesIO(toml_content_wrong_type)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(tomlfile=toml_file)


# LLM-generated content at query #38
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with no arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "test.py"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert "test.py" in config["paths"]
    
    # Test 3: TOML config file is parsed correctly
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    min_confidence = 50
    verbose = true
    paths = ["path1", "path2"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    
    # Test 4: CLI arguments override TOML config
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "75", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 75  # Override from CLI
    assert config["verbose"] == True
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    
    # Test 5: Default values are set for missing options
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    paths = ["test.py"]
    """)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 0  # Default
    assert config["sort_by_size"] == False  # Default
    assert config["make_whitelist"] == False  # Default
    
    # Test 6: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])
    
    # Test 7: TOML file with unknown key raises error
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    unknown_key = "value"
    paths = ["test.py"]
    """)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test 8: TOML file with wrong type raises error
    toml_file = io.BytesIO(b"""
    [tool.vulture]
    verbose = "not_a_boolean"
    paths = ["test.py"]
    """)
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test 9: CLI argument with wrong type raises error
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "test.py"])


# LLM-generated content at query #39
#--------------------------

```python
def test_make_config():
    # Test with only CLI arguments
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path.py"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path.py"]

    # Test with TOML file
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path.py"]
min_confidence = 30
verbose = true
"""
    toml_file = io.StringIO(toml_content)
    toml_file_binary = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file_binary)
    assert config["paths"] == ["toml_path.py"]
    assert config["min_confidence"] == 30
    assert config["verbose"] is True

    # Test CLI arguments override TOML
    toml_file_binary = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path.py"], tomlfile=toml_file_binary)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path.py"]

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with unknown config key in TOML
    toml_content_bad = """
[tool.vulture]
unknown_key = true
paths = ["test.py"]
"""
    toml_file_binary = io.BytesIO(toml_content_bad.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file_binary)

    # Test with wrong type in TOML
    toml_content_type = """
[tool.vulture]
min_confidence = "not_an_int"
paths = ["test.py"]
"""
    toml_file_binary = io.BytesIO(toml_content_type.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_file_binary)

    # Test with CLI argument --exclude
    config = make_config(argv=["--exclude", "test_*.py,dir", "path.py"])
    assert config["exclude"] == ["test_*.py", "dir"]

    # Test with CLI argument --ignore-decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "path.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test with CLI argument --ignore-names
    config = make_config(argv=["--ignore-names", "visit_*,do_*", "path.py"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with CLI argument --make-whitelist
    config = make_config(argv=["--make-whitelist", "path.py"])
    assert config["make_whitelist"] is True

    # Test with CLI argument --sort-by-size
    config = make_config(argv=["--sort-by-size", "path.py"])
    assert config["sort_by_size"] is True

    # Test with CLI argument --config
    config = make_config(argv=["--config", "custom.toml", "path.py"])
    assert config["config"] == "custom.toml"


# LLM-generated content at query #40
#--------------------------

```python
def test_make_config():
    # Test with no arguments
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments
    config = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path1", "path2"]

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

    # Test CLI overrides toml
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose", "--min-confidence", "50"], tomlfile=tomlfile)
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["file*.py", "dir/"]  # from toml

    # Test with --help (should not raise)
    import sys
    try:
        make_config(argv=["--help"])
    except SystemExit:
        pass

    # Test with --version (should not raise)
    try:
        make_config(argv=["--version"])
    except SystemExit:
        pass

    # Test InputError for missing paths
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nmin_confidence = 10"))

    # Test InputError for unknown key in toml
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nunknown_key = true"))

    # Test InputError for wrong type
    with pytest.raises(InputError, match="Data type for paths must be 'list'"):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\npaths = 'single_path'"))

    # Test with config file path
    config = make_config(argv=["--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == []


# LLM-generated content at query #41
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Basic config with minimal CLI args
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
    
    # Test 2: CLI args override defaults
    config = make_config(argv=[
        "--min-confidence", "50",
        "--verbose",
        "--make-whitelist",
        "--sort-by-size",
        "--exclude", "test_*.py,doc",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "path1"
    ])
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["exclude"] == ["test_*.py", "doc"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["path1"]
    
    # Test 3: TOML config file (via StringIO)
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
    
    # Test 4: CLI args override TOML config
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "path3"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] is True
    assert config["paths"] == ["path3"]  # CLI overrides TOML
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    
    # Test 5: Empty config raises InputError (no paths)
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test 6: Invalid config key raises InputError
    toml_file = io.StringIO("[tool.vulture]\ninvalid_key = true")
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)
    
    # Test 7: Wrong type for config value raises InputError
    toml_file = io.StringIO("[tool.vulture]\nverbose = 123")
    with pytest.raises(InputError):
        make_config(tomlfile=toml_file)


# LLM-generated content at query #42
#--------------------------

```python
def test_make_config():
    # Test 1: Default config when no arguments and no toml file
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test 3: TOML config file with settings
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
    config = make_config(tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test 4: CLI arguments override TOML config
    toml_data = """
[tool.vulture]
min_confidence = 10
verbose = false
paths = ["toml_path"]
"""
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "90", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 90
    assert config["verbose"] == True
    assert config["paths"] == ["toml_path"]

    # Test 5: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 6: Empty TOML config (only tool.vulture section empty)
    toml_data = """
[tool.vulture]
"""
    tomlfile = io.BytesIO(toml_data.encode())
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(tomlfile=tomlfile)

    # Test 7: CLI arguments with paths should work
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]

    # Test 8: Custom config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
paths = ["custom_path"]
verbose = true
""")
        temp_path = f.name
    config = make_config(argv=["--config", temp_path])
    assert config["paths"] == ["custom_path"]
    assert config["verbose"] == True
    os.unlink(temp_path)

    # Test 9: TOML config with paths and CLI override
    toml_data = """
[tool.vulture]
paths = ["toml_path1", "toml_path2"]
verbose = true
"""
    tomlfile = io.BytesIO(toml_data.encode())
    config = make_config(argv=["cli_path"], tomlfile=tomlfile)
    assert config["paths"] == ["cli_path"]
    assert config["verbose"] == True

    # Test 10: Test that config file path is read from CLI
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
paths = ["custom_path"]
""")
        temp_path = f.name
    config = make_config(argv=["--config", temp_path, "cli_path"])
    assert config["paths"] == ["cli_path"]
    os.unlink(temp_path)


# LLM-generated content at query #43
#--------------------------

```python
def test_make_config():
    """Test make_config with various scenarios."""
    # Test with CLI args only (no toml file)
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is False
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test with CLI args overriding defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "path"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path"]
    
    # Test with CLI args and toml file
    import io
    toml_data = """
[tool.vulture]
paths = ["toml_path1", "toml_path2"]
verbose = true
min_confidence = 50
"""
    config = make_config(argv=[], tomlfile=io.StringIO(toml_data))
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    
    # Test CLI args override toml values
    config = make_config(
        argv=["--min-confidence", "80", "cli_path"],
        tomlfile=io.StringIO(toml_data)
    )
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    
    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])
    
    # Test with empty toml (no vulture section)
    config = make_config(argv=["path"], tomlfile=io.StringIO(""))
    assert config["paths"] == ["path"]
    assert config["min_confidence"] == 0


# LLM-generated content at query #44
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Default configuration (no CLI args, no TOML file)
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test 2: CLI arguments override defaults
    config = make_config(
        argv=[
            "path1", "path2",
            "--exclude", "test*,docs",
            "--ignore-decorators", "@app.route,@require_*",
            "--ignore-names", "visit_*,do_*",
            "--make-whitelist",
            "--min-confidence", "50",
            "--sort-by-size",
            "--verbose",
            "--config", "custom_config.toml"
        ],
        tomlfile=None
    )
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test*", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["config"] == "custom_config.toml"

    # Test 3: TOML file configuration with CLI override
    toml_content = """
    [tool.vulture]
    paths = ["toml_path1", "toml_path2"]
    exclude = ["toml_exclude*"]
    ignore_decorators = ["toml_decorator"]
    ignore_names = ["toml_name"]
    make_whitelist = true
    min_confidence = 30
    sort_by_size = true
    verbose = true
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path1", "toml_path2"]
    assert config["exclude"] == ["toml_exclude*"]
    assert config["ignore_decorators"] == ["toml_decorator"]
    assert config["ignore_names"] == ["toml_name"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test 4: CLI overrides TOML configuration
    toml_file = io.StringIO(toml_content)
    config = make_config(
        argv=[
            "--min-confidence", "80",
            "--verbose",
            "cli_path"
        ],
        tomlfile=toml_file
    )
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True
    assert config["paths"] == ["cli_path"]  # CLI overrides TOML paths

    # Test 5: Partial TOML configuration with defaults
    partial_toml = """
    [tool.vulture]
    min_confidence = 20
    verbose = true
    """
    toml_file = io.StringIO(partial_toml)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 20
    assert config["verbose"] == True
    # Defaults for missing keys
    assert config["paths"] == []
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False

    # Test 6: Error when paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=io.StringIO(""))

    # Test 7: Error for unknown configuration key in TOML
    invalid_toml = """
    [tool.vulture]
    unknown_key = "value"
    """
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=io.StringIO(invalid_toml))

    # Test 8: Error for wrong type in TOML configuration
    wrong_type_toml = """
    [tool.vulture]
    min_confidence = "not_an_integer"
    """
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=io.StringIO(wrong_type_toml))

    # Note: Test 3-8 require io.StringIO to be imported
    # The tests assume that a non-existent pyproject.toml file will not be found
    # (no actual file system interaction)


# LLM-generated content at query #45
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Default configuration (no CLI args, no toml file)
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
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test 3: TOML configuration with CLI overrides
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 30
    verbose = true
    paths = ["toml_path1"]
    exclude = ["test_*.py"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True  # From TOML
    assert config["paths"] == ["toml_path1"]  # From TOML
    assert config["exclude"] == ["test_*.py"]  # From TOML
    
    # Test 4: TOML configuration only
    toml_content = """
    [tool.vulture]
    sort_by_size = true
    make_whitelist = true
    ignore_decorators = ["decorator1"]
    ignore_names = ["name1", "name2"]
    """
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == True
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test 5: Error when no paths provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test 6: Error for unknown configuration key in TOML
    toml_content = """
    [tool.vulture]
    unknown_key = "value"
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["path1"], tomlfile=toml_file)
    
    # Test 7: Error for wrong data type in TOML
    toml_content = """
    [tool.vulture]
    min_confidence = "not_an_integer"
    """
    toml_file = io.StringIO(toml_content)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path1"], tomlfile=toml_file)
    
    # Test 8: CLI arguments with multiple values
    config = make_config(argv=[
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "path1", "path2"
    ])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["path1", "path2"]
    
    # Test 9: Boolean flags from CLI
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "--verbose", "path1"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    
    # Test 10: Custom config file path (without actual file)
    config = make_config(argv=["--config", "custom_config.toml", "path1"])
    assert config["config"] == "custom_config.toml"
    # Should use defaults since custom_config.toml doesn't exist
    assert config["min_confidence"] == 0
    assert config["verbose"] == False


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_make_config():
    # Test 1: Default config when no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["path1", "path2", "--verbose", "--min-confidence", "50"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 50

    # Test 3: TOML file configuration
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["test_*.py", "build/"]
    ignore_decorators = ["decorator1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 30
    sort_by_size = true
    verbose = false
    paths = ["src/", "tests/"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py", "build/"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] == True
    assert config["verbose"] == False
    assert config["paths"] == ["src/", "tests/"]

    # Test 4: CLI arguments override TOML configuration
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["test_*.py", "build/"]

    # Test 5: Custom config path argument
    config = make_config(argv=["--config", "custom.toml"])
    assert config["config"] == "custom.toml"

    # Test 6: CSV parsing for exclude, ignore_decorators, ignore_names
    config = make_config(argv=["--exclude", "file1.py,file2.py", "--ignore-decorators", "dec1,dec2"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]

    # Test 7: Empty arguments
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["verbose"] == False

    # Test 8: Missing paths should raise InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])

    # Test 9: Invalid configuration key in TOML should raise InputError
    invalid_toml = """
    [tool.vulture]
    nonexistent_key = true
    """
    toml_file = io.BytesIO(invalid_toml.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["path1"], tomlfile=toml_file)

    # Test 10: Wrong type in CLI should raise InputError
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path1", "--min-confidence", "not_an_int"])

    # Test 11: Both TOML and CLI with paths
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["custom_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path.py"]  # CLI overrides TOML
    assert config["exclude"] == ["test_*.py", "build/"]  # TOML still applies
```


# LLM-generated content at query #2
#--------------------------

```python
def test_make_config():
    """Test make_config with various scenarios including TOML and CLI args."""
    import io
    import tempfile
    import os

    # Test 1: Basic CLI args only
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test 2: CLI args with options
    config = make_config(argv=["--verbose", "--sort-by-size", "path.py"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path.py"]

    # Test 3: TOML file with settings
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    min_confidence = 10
    sort_by_size = true
    paths = ["path1.py", "path2.py"]
    """
    toml_file = io.StringIO(toml_content)
    toml_file.name = "test_config.toml"
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test 4: CLI args override TOML settings
    toml_content = """
    [tool.vulture]
    min_confidence = 10
    verbose = false
    paths = ["path1.py"]
    """
    toml_file = io.StringIO(toml_content)
    toml_file.name = "test_config.toml"
    config = make_config(argv=["--min-confidence", "50", "--verbose", "path2.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["paths"] == ["path2.py"]

    # Test 5: Empty TOML with CLI args
    toml_content = """
    [tool.vulture]
    """
    toml_file = io.StringIO(toml_content)
    toml_file.name = "test_config.toml"
    config = make_config(argv=["--verbose", "path.py"], tomlfile=toml_file)
    assert config["verbose"] is True
    assert config["paths"] == ["path.py"]
    assert config["min_confidence"] == 0

    # Test 6: TOML file from filesystem
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        exclude = ["test_*.py"]
        paths = ["path1.py"]
        """)
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path])
        assert config["exclude"] == ["test_*.py"]
        assert config["paths"] == ["path1.py"]
    finally:
        os.unlink(temp_path)

    # Test 7: Default values when nothing is specified
    config = make_config(argv=["path.py"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test 8: Edge case - no paths should raise InputError
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 9: Test with all CLI options
    config = make_config(argv=[
        "--exclude", "test_*.py,venv",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "--min-confidence", "75",
        "path1.py", "path2.py"
    ])
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["min_confidence"] == 75
    assert config["paths"] == ["path1.py", "path2.py"]


# LLM-generated content at query #3
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Default config with no arguments and no toml file
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

    # Test 2: CLI arguments override defaults
    config = make_config(argv=[
        "--verbose",
        "--min-confidence=50",
        "path1", "path2"
    ], tomlfile=None)
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test 3: TOML config with settings
    import io
    toml_content = """
[tool.vulture]
exclude = ["file*.py", "dir/"]
ignore_decorators = ["deco1", "deco2"]
make_whitelist = true
min_confidence = 10
verbose = true
paths = ["path1", "path2"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]

    # Test 4: CLI arguments override TOML settings
    toml_content = """
[tool.vulture]
min_confidence = 10
verbose = false
paths = ["path1", "path2"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[
        "--min-confidence=50",
        "--verbose",
        "path3"
    ], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["path3"]

    # Test 5: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test 6: TOML file with unknown configuration key
    toml_content = """
[tool.vulture]
unknown_key = "value"
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file)

    # Test 7: TOML file with wrong data type
    toml_content = """
[tool.vulture]
verbose = "not_bool"
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=toml_file)

    # Test 8: Config with sort_by_size and make_whitelist
    config = make_config(argv=[
        "--sort-by-size",
        "--make-whitelist",
        "path1"
    ], tomlfile=None)
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == True
    assert config["paths"] == ["path1"]

    # Test 9: Config with exclude, ignore_decorators, and ignore_names
    config = make_config(argv=[
        "--exclude=*.pyc,__pycache__",
        "--ignore-decorators=@app.route,@login_required",
        "--ignore-names=helper_*,utility_*",
        "path1"
    ], tomlfile=None)
    assert config["exclude"] == ["*.pyc", "__pycache__"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    assert config["ignore_names"] == ["helper_*", "utility_*"]

    # Test 10: Config with custom config file path (non-existent)
    config = make_config(argv=[
        "--config=nonexistent.toml",
        "path1"
    ], tomlfile=None)
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["path1"]


# LLM-generated content at query #4
#--------------------------

```python
def test_make_config():
    # Test with default values (no toml file, no args)
    import io
    import tempfile
    import os
    
    # Test 1: Basic config with no arguments and no toml file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    
    # Test 2: CLI arguments override toml
    toml_content = b"""
[tool.vulture]
paths = ["path1", "path2"]
min_confidence = 50
verbose = false
"""
    toml_file = io.BytesIO(tml_content)
    config = make_config(argv=["--verbose", "--min-confidence", "75"], tomlfile=toml_file)
    assert config["verbose"] == True  # CLI overrides toml
    assert config["min_confidence"] == 75  # CLI overrides toml
    assert config["paths"] == ["path1", "path2"]  # From toml
    
    # Test 3: Test with paths from CLI
    config = make_config(argv=["path/to/file.py", "--verbose"], tomlfile=None)
    assert config["paths"] == ["path/to/file.py"]
    assert config["verbose"] == True
    
    # Test 4: Test with exclude from CLI
    config = make_config(argv=["--exclude", "test_*,docs"], tomlfile=None)
    assert config["exclude"] == ["test_*", "docs"]
    
    # Test 5: Test with ignore-decorators from CLI
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"], tomlfile=None)
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    
    # Test 6: Test with ignore-names from CLI
    config = make_config(argv=["--ignore-names", "visit_*,do_*"], tomlfile=None)
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test 7: Test with make-whitelist flag
    config = make_config(argv=["--make-whitelist"], tomlfile=None)
    assert config["make_whitelist"] == True
    
    # Test 8: Test with sort-by-size flag
    config = make_config(argv=["--sort-by-size"], tomlfile=None)
    assert config["sort_by_size"] == True
    
    # Test 9: Test with config file path
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
        f.write(b"""
[tool.vulture]
verbose = true
min_confidence = 30
""")
        temp_path = f.name
    
    config = make_config(argv=["--config", temp_path])
    assert config["verbose"] == True
    assert config["min_confidence"] == 30
    os.unlink(temp_path)
    
    # Test 10: Test that InputError is raised when no paths given
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\n"))
    
    # Test 11: Test that unknown config key raises error
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=io.BytesIO(b"[tool.vulture]\nunknown_key = true"))
    
    # Test 12: Test that wrong type raises error
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])


# LLM-generated content at query #5
#--------------------------

```python
def test_make_config():
    # Test with CLI args only (no toml file)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["verbose"] == False
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI args overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "test.py"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test.py"]

    # Test with toml file (using StringIO)
    import io
    import tomllib
    toml_content = """
    [tool.vulture]
    paths = ["toml_path.py"]
    verbose = true
    min_confidence = 30
    """
    toml_file = io.StringIO(toml_content)
    # Convert to bytes since tomllib expects bytes
    toml_bytes = io.BytesIO(toml_content.encode())
    config = make_config(argv=["another_path.py"], tomlfile=toml_bytes)
    # CLI args should override toml
    assert config["paths"] == ["another_path.py"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 30

    # Test with toml only (no CLI paths)
    toml_bytes = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_bytes)
    assert config["paths"] == ["toml_path.py"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 30

    # Test with make_whitelist and sort_by_size
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "test.py"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test with exclude, ignore_decorators, ignore_names
    config = make_config(argv=["--exclude", "*.pyc,__pycache__", "--ignore-decorators", "@staticmethod,@classmethod", "--ignore-names", "helper_*,internal_*", "test.py"])
    assert config["exclude"] == ["*.pyc", "__pycache__"]
    assert config["ignore_decorators"] == ["@staticmethod", "@classmethod"]
    assert config["ignore_names"] == ["helper_*", "internal_*"]

    # Test that missing config key raises error
    try:
        config = make_config(argv=["--unknown-option", "test.py"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test that wrong type raises error
    try:
        config = make_config(argv=["--min-confidence", "not_an_int", "test.py"])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e)

    # Test that empty paths raises error
    try:
        config = make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    # Test 1: Basic config with default values (no TOML file)
    import io
    
    # Create mock TOML content
    toml_content = """
    [tool.vulture]
    paths = ["test.py"]
    min_confidence = 50
    """
    
    # Test with TOML file provided
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 50
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"
    
    # Test 2: CLI arguments override TOML config
    toml_file2 = io.StringIO(toml_content)
    config2 = make_config(
        argv=["--min-confidence", "75", "--verbose", "test_cli.py"],
        tomlfile=toml_file2
    )
    assert config2["paths"] == ["test_cli.py"]  # CLI path overrides TOML path
    assert config2["min_confidence"] == 75  # CLI value overrides TOML
    assert config2["verbose"] == True  # CLI value
    
    # Test 3: Multiple paths from CLI
    config3 = make_config(
        argv=["path1.py", "path2.py", "path3.py"],
        tomlfile=io.StringIO("")
    )
    assert config3["paths"] == ["path1.py", "path2.py", "path3.py"]
    
    # Test 4: Exclude patterns from CLI
    config4 = make_config(
        argv=["--exclude", "*.pyc,__pycache__,*.egg", "."],
        tomlfile=io.StringIO("")
    )
    assert "*.pyc" in config4["exclude"]
    assert "__pycache__" in config4["exclude"]
    assert "*.egg" in config4["exclude"]
    assert config4["paths"] == ["."]
    
    # Test 5: Ignore decorators and names
    config5 = make_config(
        argv=["--ignore-decorators", "@app.route,@require_*",
              "--ignore-names", "visit_*,do_*", "."],
        tomlfile=io.StringIO("")
    )
    assert "@app.route" in config5["ignore_decorators"]
    assert "@require_*" in config5["ignore_decorators"]
    assert "visit_*" in config5["ignore_names"]
    assert "do_*" in config5["ignore_names"]
    
    # Test 6: Boolean flags
    config6 = make_config(
        argv=["--make-whitelist", "--sort-by-size", "."],
        tomlfile=io.StringIO("")
    )
    assert config6["make_whitelist"] == True
    assert config6["sort_by_size"] == True
    
    # Test 7: Custom config file path
    config7 = make_config(
        argv=["--config", "custom_config.toml", "."],
        tomlfile=io.StringIO("")
    )
    assert config7["config"] == "custom_config.toml"
    
    # Test 8: InputError when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=io.StringIO(""))
    
    # Test 9: InputError for unknown configuration key
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(
            argv=["--unknown-option", "."],
            tomlfile=io.StringIO("[tool.vulture]\nunknown_key = true")
        )
    
    # Test 10: InputError for wrong data type
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(
            argv=["--min-confidence", "not_an_int", "."],
            tomlfile=io.StringIO("")
        )


# LLM-generated content at query #7
#--------------------------

```python
def test_make_config():
    # Test with only CLI arguments (no TOML file)
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
        "path.py",
        "--min-confidence", "50",
        "--verbose",
        "--sort-by-size",
        "--make-whitelist",
        "--exclude", "test*,docs",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "helper_*",
    ])
    assert config["paths"] == ["path.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True
    assert config["exclude"] == ["test*", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["helper_*"]

    # Test with TOML file (using io.StringIO)
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path1.py", "toml_path2.py"]
min_confidence = 30
verbose = true
exclude = ["venv", "__pycache__"]
"""
    toml_file = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path1.py", "toml_path2.py"]
    assert config["min_confidence"] == 30
    assert config["verbose"] is True
    assert config["exclude"] == ["venv", "__pycache__"]
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False

    # Test with TOML file overridden by CLI arguments
    toml_file.seek(0)
    config = make_config(
        argv=["cli_path.py", "--min-confidence", "80"],
        tomlfile=toml_file
    )
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True  # From TOML, not overridden

    # Test with empty CLI arguments (should raise InputError due to no paths)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with --version (should not raise InputError about paths)
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])

    # Test with --help (should not raise InputError about paths)
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])

    # Test with custom config path that doesn't exist
    config = make_config(argv=["path.py", "--config", "nonexistent.toml"])
    assert config["config"] == "nonexistent.toml"
    assert config["paths"] == ["path.py"]

    # Test with TOML file containing all options
    toml_content_full = """
[tool.vulture]
paths = ["p1.py", "p2.py"]
min_confidence = 100
verbose = true
sort_by_size = true
make_whitelist = true
exclude = ["ex1", "ex2"]
ignore_decorators = ["dec1", "dec2"]
ignore_names = ["name1", "name2"]
"""
    toml_file_full = io.BytesIO(toml_content_full.encode('utf-8'))
    config = make_config(argv=[], tomlfile=toml_file_full)
    assert config["paths"] == ["p1.py", "p2.py"]
    assert config["min_confidence"] == 100
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True
    assert config["exclude"] == ["ex1", "ex2"]
    assert config["ignore_decorators"] == ["dec1", "dec2"]
    assert config["ignore_names"] == ["name1", "name2"]


# LLM-generated content at query #8
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
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1.py"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1.py"]

    # Test TOML config without CLI args
    toml_content = b"""
    [tool.vulture]
    min_confidence = 30
    verbose = true
    paths = ["test_path.py"]
    """
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["verbose"] == True
    assert config["paths"] == ["test_path.py"]

    # Test CLI args override TOML config
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] == True
    assert config["paths"] == ["test_path.py"]

    # Test with all CLI arguments
    config = make_config(argv=[
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--sort-by-size",
        "--config", "custom_config.toml",
        "path1.py", "path2.py"
    ])
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["config"] == "custom_config.toml"
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test InputError for unknown config key in TOML
    bad_toml = io.BytesIO(b"""
    [tool.vulture]
    invalid_key = true
    """)
    try:
        make_config(argv=[], tomlfile=bad_toml)
        assert False, "Expected InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e)

    # Test InputError for wrong type in TOML
    bad_type_toml = io.BytesIO(b"""
    [tool.vulture]
    min_confidence = "not_an_int"
    """)
    try:
        make_config(argv=[], tomlfile=bad_type_toml)
        assert False, "Expected InputError"
    except InputError as e:
        assert "must be" in str(e)

    # Test InputError for missing paths
    try:
        make_config(argv=["--make-whitelist"])
        assert False, "Expected InputError"
    except InputError as e:
        assert "at least one file or directory" in str(e)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test with default values (no CLI args, no TOML file)
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    
    # Test with CLI arguments only
    config = make_config(argv=["--verbose", "path1", "path2"], tomlfile=None)
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    
    # Test with TOML file content
    import io
    toml_content = b"""
[tool.vulture]
paths = ["test_path"]
verbose = true
min_confidence = 50
"""
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["test_path"]
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    
    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["test_path"]
    
    # Test with exclude patterns
    config = make_config(argv=["--exclude", "test_*.py,venv", "src"], tomlfile=None)
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["paths"] == ["src"]
    
    # Test with ignore decorators and names
    config = make_config(
        argv=["--ignore-decorators", "@app.route", "--ignore-names", "visit_*", "src"],
        tomlfile=None
    )
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    
    # Test with boolean flags
    config = make_config(
        argv=["--make-whitelist", "--sort-by-size", "src"],
        tomlfile=None
    )
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    
    # Test that --version doesn't raise error (but returns early)
    # Note: argparse will exit with SystemExit(0) on --version
    try:
        config = make_config(argv=["--version"], tomlfile=None)
    except SystemExit:
        pass  # Expected behavior
    
    # Test InputError when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)
    
    # Test with invalid config key in TOML
    toml_content = b"""
[tool.vulture]
invalid_key = true
paths = ["test"]
"""
    toml_file = io.BytesIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=toml_file)
    
    # Test with wrong type in TOML
    toml_content = b"""
[tool.vulture]
min_confidence = "not_an_int"
paths = ["test"]
"""
    toml_file = io.BytesIO(toml_content)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_file)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_make_config():
    # Test 1: Default configuration (no CLI args, no TOML file)
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS.copy()

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--make-whitelist", "file1.py", "file2.py"], tomlfile=None)
    assert config["verbose"] == True
    assert config["make_whitelist"] == True
    assert config["paths"] == ["file1.py", "file2.py"]

    # Test 3: TOML file configuration
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
ignore_decorators = ["@app.route"]
min_confidence = 50
verbose = true
paths = ["src/"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["file.py"], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["paths"] == ["file.py"]  # CLI overrides TOML

    # Test 4: CLI overrides TOML
    toml_content = """
[tool.vulture]
verbose = true
min_confidence = 80
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose", "--min-confidence", "30", "path.py"], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["min_confidence"] == 30  # CLI overrides TOML
    assert config["paths"] == ["path.py"]

    # Test 5: Unknown config key raises InputError
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    toml_file = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=["test.py"], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test 6: Wrong type raises InputError
    toml_content = """
[tool.vulture]
verbose = "not_a_bool"
"""
    toml_file = io.BytesIO(toml_content.encode())
    try:
        make_config(argv=["test.py"], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for verbose must be 'bool'" in str(e.message)

    # Test 7: No paths raises InputError
    try:
        make_config(argv=[], tomlfile=None)
    except InputError as e:
        assert "Please pass at least one file or directory" in str(e.message)
    
    # Test 8: Paths with whitespace-only strings
    config = make_config(argv=["  ", "test.py"], tomlfile=None)
    assert "  " in config["paths"]
    assert "test.py" in config["paths"]

    # Test 9: CLI arguments with dashes are converted to underscores
    config = make_config(argv=["--sort-by-size", "test.py"], tomlfile=None)
    assert config["sort_by_size"] == True

    # Test 10: Multiple values in CLI args
    config = make_config(argv=["--exclude", "file1.py,file2.py", "--ignore-names", "name1,name2", "test.py"], tomlfile=None)
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test 11: Version argument does not trigger InputError for missing paths
    # This requires mocking sys.exit, so we'll skip this for simplicity

    # Test 12: Help argument does not trigger InputError for missing paths
    # This requires mocking sys.exit, so we'll skip this for simplicity

    # Test 13: Config file path from CLI
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
min_confidence = 90
paths = ["temp_test.py"]
""")
        temp_path = f.name
    
    config = make_config(argv=["--config", temp_path, "test.py"], tomlfile=None)
    assert config["min_confidence"] == 90
    assert config["paths"] == ["test.py"]  # CLI overrides TOML paths
    import os
    os.unlink(temp_path)


# LLM-generated content at query #11
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with paths from CLI
    config = make_config(argv=["file1.py", "file2.py"])
    assert config["paths"] == ["file1.py", "file2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "file.py"])
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["file.py"]

    # Test 3: TOML config with CLI override
    import io
    toml_data = """
[tool.vulture]
paths = ["toml_path.py"]
min_confidence = 50
verbose = true
"""
    config = make_config(
        argv=["file.py", "--min-confidence", "80"],
        tomlfile=io.StringIO(toml_data)
    )
    assert config["paths"] == ["file.py"]  # CLI overrides TOML
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True

    # Test 4: TOML config only (no CLI args)
    config = make_config(
        argv=[],
        tomlfile=io.StringIO(toml_data)
    )
    assert config["paths"] == ["toml_path.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test 5: CSV parsing for exclude, ignore_decorators, ignore_names
    config = make_config(
        argv=["--exclude", "*.pyc,__pycache__",
              "--ignore-decorators", "@app.route,@require_*",
              "--ignore-names", "visit_*,do_*",
              "file.py"]
    )
    assert config["exclude"] == ["*.pyc", "__pycache__"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test 6: make-whitelist flag
    config = make_config(argv=["--make-whitelist", "file.py"])
    assert config["make_whitelist"] == True

    # Test 7: Missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 8: Unknown config key in TOML raises InputError
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(
            argv=[],
            tomlfile=io.StringIO("[tool.vulture]\ninvalid_key = true")
        )

    # Test 9: Wrong type in TOML config raises InputError
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(
            argv=[],
            tomlfile=io.StringIO("[tool.vulture]\nmin_confidence = 'not_an_int'")
        )

    # Test 10: Custom config path (when file doesn't exist, no TOML loaded)
    config = make_config(argv=["--config", "nonexistent.toml", "file.py"])
    assert config["paths"] == ["file.py"]
    assert config["config"] == "nonexistent.toml"

    # Test 11: Verbose mode prints detected config file
    config = make_config(
        argv=["--verbose"],
        tomlfile=io.StringIO("[tool.vulture]\npaths = ['file.py']")
    )
    # The print should have happened during make_config call
    # We can't easily capture stdout here, but the config should be correct
    assert config["verbose"] == True
    assert config["paths"] == ["file.py"]

    # Test 12: All defaults when no config provided
    config = make_config(argv=["test.py"])
    for key, value in DEFAULTS.items():
        if key == "paths":
            assert config[key] == ["test.py"]
        elif key == "config":
            assert config[key] == "pyproject.toml"
        else:
            assert config[key] == value, f"Default mismatch for {key}"


# LLM-generated content at query #12
#--------------------------

```python
def test_make_config():
    # Test 1: No arguments, no toml file
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
    
    # Test 2: CLI arguments override defaults
    config = make_config(["--verbose", "--sort-by-size", "path1", "path2"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test 3: TOML file provides configuration
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    exclude = ["test_*.py", "docs"]
    verbose = true
    paths = ["src"]
    """
    toml_file = io.StringIO(toml_content)
    toml_file.seek(0)
    config = make_config([], tomlfile=io.BytesIO(toml_content.encode()))
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["verbose"] is True
    assert config["paths"] == ["src"]
    
    # Test 4: CLI arguments override TOML file settings
    toml_content2 = """
    [tool.vulture]
    min_confidence = 30
    verbose = false
    """
    config = make_config(["--min-confidence", "80", "--verbose"], 
                         tomlfile=io.BytesIO(toml_content2.encode()))
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    
    # Test 5: Custom config path via CLI
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('''
        [tool.vulture]
        min_confidence = 90
        paths = ["custom_path"]
        ''')
        temp_path = f.name
    
    config = make_config(["--config", temp_path, "extra_path"])
    assert config["min_confidence"] == 90
    assert config["paths"] == ["custom_path", "extra_path"]
    os.unlink(temp_path)
    
    # Test 6: InputError when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config([], tomlfile=io.BytesIO(b"[tool.vulture]\nmin_confidence = 5"))
    
    # Test 7: InputError for unknown configuration key
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config([], tomlfile=io.BytesIO(b"[tool.vulture]\nunknown_key = true"))
    
    # Test 8: CSV parsing for exclude, ignore_decorators, ignore_names
    config = make_config(["--exclude", "a.py,b.py", "--ignore-decorators", "@dec1,@dec2", 
                         "--ignore-names", "name1,name2", "test_path"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_decorators"] == ["@dec1", "@dec2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test 9: make_whitelist flag
    config = make_config(["--make-whitelist", "some_path"])
    assert config["make_whitelist"] is True
    
    # Test 10: Min confidence integer parsing
    config = make_config(["--min-confidence", "42", "test_path"])
    assert config["min_confidence"] == 42
```


# LLM-generated content at query #13
#--------------------------

```python
def test_make_config():
    # Test with empty argv and no tomlfile
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(
        argv=["--min-confidence", "50", "--verbose", "--sort-by-size", "path1", "path2"],
        tomlfile=None
    )
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile
    import io
    toml_content = """
[tool.vulture]
min_confidence = 30
verbose = true
exclude = ["test_*.py"]
paths = ["src/"]
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["min_confidence"] == 30
    assert config["verbose"] is True
    assert config["exclude"] == ["test_*.py"]
    assert config["paths"] == ["src/"]

    # Test CLI overrides toml
    toml_content = """
[tool.vulture]
min_confidence = 30
verbose = false
"""
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test with exclude and ignore_decorators
    config = make_config(
        argv=["--exclude", "dir1,dir2", "--ignore-decorators", "@app.route,@login_required"],
        tomlfile=None
    )
    assert config["exclude"] == ["dir1", "dir2"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]

    # Test with ignore_names
    config = make_config(
        argv=["--ignore-names", "helper_,internal_"],
        tomlfile=None
    )
    assert config["ignore_names"] == ["helper_", "internal_"]

    # Test with make_whitelist
    config = make_config(argv=["--make-whitelist"], tomlfile=None)
    assert config["make_whitelist"] is True

    # Test with config option
    config = make_config(argv=["--config", "custom_config.toml"], tomlfile=None)
    assert config["config"] == "custom_config.toml"

    # Test that paths must be provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)


# LLM-generated content at query #14
#--------------------------

```python
def test_make_config():
    """Test make_config with various combinations of CLI args and TOML file."""
    
    # Test 1: Basic case with only paths from CLI
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []

    # Test 2: CLI args override TOML settings
    toml_data = b"""
    [tool.vulture]
    paths = ["toml_path.py"]
    min_confidence = 50
    verbose = true
    """
    config = make_config(
        argv=["--min-confidence", "80", "cli_path.py"],
        tomlfile=BytesIO(toml_data)
    )
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] is True

    # Test 3: TOML settings applied when no CLI override
    toml_data = b"""
    [tool.vulture]
    exclude = ["test_*.py"]
    ignore_decorators = ["@app.route"]
    make_whitelist = true
    """
    config = make_config(
        argv=["some_path.py"],
        tomlfile=BytesIO(toml_data)
    )
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 0  # default

    # Test 4: Empty TOML, only CLI args
    toml_data = b"[tool.vulture]\n"
    config = make_config(
        argv=["--verbose", "--sort-by-size", "path.py"],
        tomlfile=BytesIO(toml_data)
    )
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path.py"]

    # Test 5: Multiple CLI options
    config = make_config(
        argv=[
            "--exclude", "dir/,test_*.py",
            "--ignore-names", "helper*,utils",
            "--make-whitelist",
            "main.py", "utils.py"
        ]
    )
    assert config["exclude"] == ["dir/", "test_*.py"]
    assert config["ignore_names"] == ["helper*", "utils"]
    assert config["make_whitelist"] is True
    assert config["paths"] == ["main.py", "utils.py"]

    # Test 6: Error when no paths provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 7: Error with unknown config key in TOML
    toml_data = b"""
    [tool.vulture]
    unknown_key = true
    paths = ["test.py"]
    """
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=BytesIO(toml_data))

    # Test 8: Error with wrong type in CLI
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "path.py"])

    # Test 9: TOML with all options
    toml_data = b"""
    [tool.vulture]
    paths = ["path1.py", "path2.py"]
    exclude = ["venv", "*.pyc"]
    ignore_decorators = ["@staticmethod", "@classmethod"]
    ignore_names = ["__init__"]
    make_whitelist = false
    min_confidence = 30
    sort_by_size = true
    verbose = false
    """
    config = make_config(tomlfile=BytesIO(toml_data))
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["exclude"] == ["venv", "*.pyc"]
    assert config["ignore_decorators"] == ["@staticmethod", "@classmethod"]
    assert config["ignore_names"] == ["__init__"]
    assert config["make_whitelist"] is False
    assert config["min_confidence"] == 30
    assert config["sort_by_size"] is True
    assert config["verbose"] is False

    # Test 10: Proper merging - CLI overrides specific TOML settings
    toml_data = b"""
    [tool.vulture]
    paths = ["toml_only.py"]
    min_confidence = 10
    verbose = true
    exclude = ["test_*"]
    """
    config = make_config(
        argv=["--min-confidence", "90", "cli_only.py"],
        tomlfile=BytesIO(toml_data)
    )
    assert config["paths"] == ["cli_only.py"]  # CLI overrides
    assert config["min_confidence"] == 90  # CLI overrides
    assert config["verbose"] is True  # From TOML
    assert config["exclude"] == ["test_*"]  # From TOML

    # Test 11: Config path from CLI (but we use tomlfile for testing)
    config = make_config(
        argv=["--config", "custom.toml", "test.py"],
        tomlfile=BytesIO(b"[tool.vulture]\nmin_confidence = 50")
    )
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test.py"]

    # Test 12: CSV parsing for list arguments
    config = make_config(
        argv=["--exclude", "a,b,c", "--ignore-decorators", "d,e,f", "path.py"]
    )
    assert config["exclude"] == ["a", "b", "c"]
    assert config["ignore_decorators"] == ["d", "e", "f"]


# LLM-generated content at query #15
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only (no tomlfile)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=[
        "path.py",
        "--verbose",
        "--sort-by-size",
        "--make-whitelist",
        "--min-confidence", "50",
        "--exclude", "test_*,docs",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "helper_*",
        "--config", "custom.toml"
    ])
    assert config["paths"] == ["path.py"]
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test_*", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["helper_*"]
    assert config["config"] == "custom.toml"

    # Test with tomlfile (BytesIO simulating pyproject.toml content)
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
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test that CLI arguments override TOML settings
    toml_file = io.BytesIO(toml_content)
    config = make_config(argv=["override.py", "--min-confidence", "90"], tomlfile=toml_file)
    assert config["paths"] == ["override.py"]  # CLI overrides TOML
    assert config["min_confidence"] == 90  # CLI overrides TOML
    assert config["verbose"] is True  # From TOML
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML

    # Test error when no paths provided
    try:
        make_config(argv=[])
        assert False, "Expected InputError"
    except InputError as e:
        assert str(e.message) == "Please pass at least one file or directory"

    # Test error with unknown config key
    try:
        make_config(argv=["path.py", "--invalid-arg"], tomlfile=io.BytesIO(b""))
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test error with wrong type in CLI
    try:
        make_config(argv=["path.py", "--min-confidence", "not_a_number"])
        assert False, "Expected SystemExit"
    except SystemExit:
        pass

    # Test error with wrong type in TOML
    bad_toml = b"""
    [tool.vulture]
    min_confidence = "not_a_number"
    paths = ["path.py"]
    """
    try:
        make_config(tomlfile=io.BytesIO(bad_toml))
        assert False, "Expected InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)


# LLM-generated content at query #16
#--------------------------

```python
def test_make_config():
    # Test basic config with no arguments
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["sort_by_size"] == False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["config"] == "pyproject.toml"

    # Test CLI arguments override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "--min-confidence", "50"], tomlfile=None)
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["min_confidence"] == 50

    # Test CLI arguments with paths
    config = make_config(argv=["path1.py", "path2.py"], tomlfile=None)
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test TOML file config
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
    toml_file = io.BytesIO(toml_data.encode('utf-8'))
    config = make_config(argv=["--min-confidence", "90", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 90
    assert config["verbose"] == True
    assert config["exclude"] == ["file*.py", "dir/"]  # TOML value preserved

    # Test missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test with paths provided
    config = make_config(argv=["path.py"], tomlfile=None)
    assert config["paths"] == ["path.py"]

    # Test config parameter
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
min_confidence = 20
""")
        config_path = f.name
    config = make_config(argv=["--config", config_path, "test.py"], tomlfile=None)
    assert config["min_confidence"] == 20
    assert config["paths"] == ["test.py"]
    os.unlink(config_path)


# LLM-generated content at query #17
#--------------------------

```python
def test_make_config():
    # Test with TOML file and no CLI args
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src"]
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test CLI args override TOML
    toml_file.seek(0)
    config = make_config(argv=["--verbose", "--min-confidence", "50"], tomlfile=toml_file)
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src"]

    # Test CLI args with paths
    config = make_config(argv=["path1", "path2"])
    assert config["paths"] == ["path1", "path2"]

    # Test CLI args with exclude
    config = make_config(argv=["--exclude", "test_*.py,*.pyc"])
    assert config["exclude"] == ["test_*.py", "*.pyc"]

    # Test defaults when no config file and no CLI args
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test ignore_decorators from CLI
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "test.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["paths"] == ["test.py"]

    # Test ignore_names from CLI
    config = make_config(argv=["--ignore-names", "visit_*,do_*", "test.py"])
    assert config["ignore_names"] == ["visit_*,do_*"]

    # Test make_whitelist
    config = make_config(argv=["--make-whitelist", "test.py"])
    assert config["make_whitelist"] is True

    # Test sort_by_size
    config = make_config(argv=["--sort-by-size", "test.py"])
    assert config["sort_by_size"] is True

    # Test verbose
    config = make_config(argv=["--verbose", "test.py"])
    assert config["verbose"] is True

    # Test config file path from CLI
    toml_file.seek(0)
    config = make_config(argv=["--config", "custom.toml"], tomlfile=toml_file)
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["src"]  # From TOML

    # Test that TOML with unknown keys raises error
    bad_toml = """
    [tool.vulture]
    unknown_key = "value"
    paths = ["test"]
    """
    bad_file = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=bad_file)

    # Test that TOML with wrong types raises error
    wrong_type_toml = """
    [tool.vulture]
    verbose = "yes"
    paths = ["test"]
    """
    wrong_file = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=wrong_file)


# LLM-generated content at query #18
#--------------------------

```python
def test_make_config():
    """Test make_config with various inputs."""
    # Test 1: Minimal CLI arguments (paths provided)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] == False

    # Test 2: CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path.py"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path.py"]

    # Test 3: TOML file provided via tomlfile parameter
    import io
    toml_data = """
[tool.vulture]
paths = ["toml_path.py"]
min_confidence = 30
verbose = true
"""
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["toml_path.py"]
    assert config["min_confidence"] == 30
    assert config["verbose"] == True

    # Test 4: CLI arguments override TOML settings
    toml_data = """
[tool.vulture]
paths = ["toml_path.py"]
min_confidence = 30
"""
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]  # CLI overrides TOML
    assert config["min_confidence"] == 80      # CLI overrides TOML

    # Test 5: No paths provided raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 6: Unknown configuration key in TOML raises InputError
    toml_data = """
[tool.vulture]
unknown_key = "value"
paths = ["path.py"]
"""
    toml_file = io.BytesIO(toml_data.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(tomlfile=toml_file)

    # Test 7: Wrong data type in CLI arguments raises InputError
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "path.py"])

    # Test 8: Sorting by size and make whitelist
    config = make_config(argv=["--sort-by-size", "--make-whitelist", "path.py"])
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == True

    # Test 9: Exclude patterns with commas
    config = make_config(argv=["--exclude", "file1.py,file2.py", "path.py"])
    assert config["exclude"] == ["file1.py", "file2.py"]

    # Test 10: Ignore decorators and names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", 
                               "--ignore-names", "visit_*,do_*", "path.py"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]


# LLM-generated content at query #19
#--------------------------

```python
def test_make_config():
    # Test with default values when no arguments and no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["make_whitelist"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file
    import io
    toml_content = """
[tool.vulture]
verbose = true
min_confidence = 30
paths = ["src/"]
"""
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 30
    assert config["paths"] == ["src/"]

    # Test CLI arguments overriding TOML settings
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "other/"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["other/"]
    assert config["verbose"] is True  # from TOML

    # Test with missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test with invalid config key in TOML
    toml_content_invalid = """
[tool.vulture]
invalid_key = true
"""
    tomlfile = io.BytesIO(toml_content_invalid.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["path1"], tomlfile=tomlfile)

    # Test with wrong data type in TOML
    toml_content_wrong_type = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    tomlfile = io.BytesIO(toml_content_wrong_type.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path1"], tomlfile=tomlfile)


# LLM-generated content at query #20
#--------------------------

```python
def test_make_config():
    """Test make_config with various scenarios."""
    
    # Test 1: Default config with no arguments
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

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1.py"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1.py"]

    # Test 3: TOML file with valid configuration
    toml_content = b"""
[tool.vulture]
exclude = ["file*.py"]
ignore_decorators = ["deco1"]
verbose = true
"""
    import io
    toml_file = io.BytesIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["file*.py"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["verbose"] == True
    assert config["paths"] == []  # Not set in TOML, should be empty

    # Test 4: CLI arguments override TOML settings
    toml_file = io.BytesIO(b"[tool.vulture]\nverbose = true\nmin_confidence = 10\n")
    config = make_config(argv=["--min-confidence", "90"], tomlfile=toml_file)
    assert config["min_confidence"] == 90  # CLI overrides TOML
    assert config["verbose"] == True  # From TOML

    # Test 5: Paths from CLI arguments
    config = make_config(argv=["file1.py", "file2.py", "dir/"])
    assert config["paths"] == ["file1.py", "file2.py", "dir/"]

    # Test 6: CSV parsing for exclude patterns
    config = make_config(argv=["--exclude", "file*.py,dir/,test_*.py"])
    assert config["exclude"] == ["file*.py", "dir/", "test_*.py"]

    # Test 7: CSV parsing for ignore_decorators
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test 8: CSV parsing for ignore_names
    config = make_config(argv=["--ignore-names", "visit_*,do_*"])
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test 9: Boolean flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test 10: Custom config file path
    config = make_config(argv=["--config", "custom_config.toml"])
    assert config["config"] == "custom_config.toml"

    # Test 11: Error when no paths provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 12: Error with unknown configuration key in TOML
    bad_toml = io.BytesIO(b"[tool.vulture]\nunknown_key = true\n")
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(tomlfile=bad_toml)

    # Test 13: Error with wrong data type in TOML
    bad_type_toml = io.BytesIO(b"[tool.vulture]\nverbose = 123\n")
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(tomlfile=bad_type_toml)

    # Test 14: Error with wrong data type in CLI
    with pytest.raises(InputError, match="Data type for make_whitelist must be 'bool'"):
        make_config(argv=["--make-whitelist", "not_bool"])

    # Test 15: Empty TOML section
    empty_toml = io.BytesIO(b"[tool.vulture]\n")
    config = make_config(tomlfile=empty_toml)
    assert config["paths"] == []  # Should fail output check
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(tomlfile=empty_toml)

    # Test 16: Partial TOML configuration
    partial_toml = io.BytesIO(b"[tool.vulture]\nmin_confidence = 75\n")
    config = make_config(argv=["file.py"], tomlfile=partial_toml)
    assert config["min_confidence"] == 75
    assert config["paths"] == ["file.py"]
    assert config["verbose"] == False  # Default value

    # Test 17: Multiple paths in TOML
    paths_toml = io.BytesIO(b"[tool.vulture]\npaths = [\"src/\", \"tests/\"]\n")
    config = make_config(tomlfile=paths_toml)
    assert config["paths"] == ["src/", "tests/"]

    # Test 18: Verbose output with TOML detection
    verbose_toml = io.BytesIO(b"[tool.vulture]\nverbose = true\npaths = [\"file.py\"]\n")
    config = make_config(tomlfile=verbose_toml)
    assert config["verbose"] == True
    assert config["paths"] == ["file.py"]


# LLM-generated content at query #21
#--------------------------

```python
def test_make_config():
    # Test with only CLI args
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test with TOML file
    import io
    toml_content = """
[tool.vulture]
paths = ["toml_path1.py", "toml_path2.py"]
min_confidence = 50
verbose = true
"""
    toml_file = io.StringIO(toml_content)
    config = make_config(tomlfile=toml_file)
    assert config["paths"] == ["toml_path1.py", "toml_path2.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test CLI args override TOML
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 50

    # Test with TOML file and CLI verbose override
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=["path.py", "--verbose", "false"], tomlfile=toml_file)
    assert config["paths"] == ["path.py"]
    assert config["verbose"] is False

    # Test defaults are applied
    config = make_config(argv=["test.py"])
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False

    # Test with various CLI options
    config = make_config(
        argv=[
            "path.py",
            "--exclude", "test.py,*.txt",
            "--ignore-decorators", "@decorator1,@decorator2",
            "--ignore-names", "name1,name2",
            "--make-whitelist",
            "--min-confidence", "80",
            "--sort-by-size",
            "--verbose",
        ]
    )
    assert config["exclude"] == ["test.py", "*.txt"]
    assert config["ignore_decorators"] == ["@decorator1", "@decorator2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 80
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # Test InputError raised for missing paths
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])


# LLM-generated content at query #22
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    import io
    import tempfile
    import os
    
    # Test 1: No arguments, default config file doesn't exist
    config = make_config(argv=[])
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=[
        "path1", "path2",
        "--exclude", "test*,docs",
        "--verbose",
        "--make-whitelist",
        "--sort-by-size",
        "--min-confidence", "50",
        "--ignore-decorators", "decor1,decor2",
        "--ignore-names", "name1,name2"
    ])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test*", "docs"]
    assert config["verbose"] == True
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["min_confidence"] == 50
    assert config["ignore_decorators"] == ["decor1", "decor2"]
    assert config["ignore_names"] == ["name1", "name2"]
    
    # Test 3: TOML config file provided as IO
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
    
    # Test 4: CLI arguments override TOML config
    toml_file = io.StringIO(toml_content)
    config = make_config(argv=[
        "--min-confidence", "80",
        "--verbose",  # This is False in TOML, but we don't provide it
    ], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True  # From TOML
    
    # Test 5: Actual pyproject.toml file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
[tool.vulture]
paths = ["test_path"]
min_confidence = 25
""")
        temp_path = f.name
    
    config = make_config(argv=["--config", temp_path])
    assert config["paths"] == ["test_path"]
    assert config["min_confidence"] == 25
    
    # Clean up temp file
    os.unlink(temp_path)
    
    # Test 6: No paths provided raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test 7: Unknown configuration key in TOML raises InputError
    bad_toml = io.StringIO("""
[tool.vulture]
unknown_key = "value"
paths = ["test"]
""")
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=bad_toml)
    
    # Test 8: Wrong type in TOML raises InputError
    bad_type_toml = io.StringIO("""
[tool.vulture]
min_confidence = "not_an_integer"
paths = ["test"]
""")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=bad_type_toml)


# LLM-generated content at query #23
#--------------------------

```python
def test_make_config():
    # Test with empty CLI args and no TOML file
    result = make_config(argv=[])
    assert result == DEFAULTS.copy()
    
    # Test with CLI arguments overriding defaults
    result = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"])
    assert result["verbose"] is True
    assert result["sort_by_size"] is True
    assert result["paths"] == ["path1", "path2"]
    
    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    verbose = true
    paths = ["toml_path1", "toml_path2"]
    min_confidence = 50
    """
    toml_file = io.BytesIO(toml_content.encode())
    result = make_config(argv=[], tomlfile=toml_file)
    assert result["verbose"] is True
    assert result["paths"] == ["toml_path1", "toml_path2"]
    assert result["min_confidence"] == 50
    
    # Test CLI overrides TOML
    toml_file.seek(0)
    result = make_config(argv=["--min-confidence", "75"], tomlfile=toml_file)
    assert result["min_confidence"] == 75  # CLI overrides TOML
    assert result["verbose"] is True  # From TOML
    
    # Test with --make-whitelist flag
    result = make_config(argv=["--make-whitelist", "path1"])
    assert result["make_whitelist"] is True
    assert result["paths"] == ["path1"]
    
    # Test with --exclude
    result = make_config(argv=["--exclude", "test_*.py,*.bak", "path1"])
    assert result["exclude"] == ["test_*.py", "*.bak"]
    assert result["paths"] == ["path1"]
    
    # Test with --ignore-decorators
    result = make_config(argv=["--ignore-decorators", "@app.route,@require_*", "path1"])
    assert result["ignore_decorators"] == ["@app.route", "@require_*"]
    
    # Test with --ignore-names
    result = make_config(argv=["--ignore-names", "visit_*,do_*", "path1"])
    assert result["ignore_names"] == ["visit_*", "do_*"]
    
    # Test InputError raised when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test InputError raised for unknown configuration key in TOML
    toml_content_invalid = """
    [tool.vulture]
    unknown_key = true
    paths = ["path1"]
    """
    toml_file_invalid = io.BytesIO(toml_content_invalid.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file_invalid)


# LLM-generated content at query #24
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: No arguments provided, no TOML file
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "test_file.py"])
    assert config["verbose"] is True
    assert config["sort_by_size"] is True
    assert "test_file.py" in config["paths"]
    
    # Test 3: TOML file configuration
    import io
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
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
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["paths"] == ["path1", "path2"]
    
    # Test 4: CLI arguments override TOML configuration
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "50", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["sort_by_size"] is True  # from TOML
    assert config["exclude"] == ["file*.py", "dir/"]  # from TOML
    
    # Test 5: CLI arguments override TOML with paths
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["custom_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path.py"]
    
    # Test 6: Empty TOML file
    empty_toml = io.BytesIO(b"")
    config = make_config(argv=["test.py"], tomlfile=empty_toml)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False
    
    # Test 7: TOML file with only tool section but no vulture
    partial_toml = io.BytesIO(b"[tool]\nother = true")
    config = make_config(argv=["test.py"], tomlfile=partial_toml)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    
    # Test 8: Verify defaults are set for missing options
    minimal_toml = io.BytesIO(b"[tool.vulture]\npaths = ['test.py']")
    config = make_config(argv=[], tomlfile=minimal_toml)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"
    
    # Test 9: Test with multiple CLI paths
    config = make_config(argv=["path1.py", "path2.py", "path3.py"])
    assert config["paths"] == ["path1.py", "path2.py", "path3.py"]
    
    # Test 10: Test with --exclude comma-separated values
    config = make_config(argv=["--exclude", "file1.py,file2.py,dir/", "test.py"])
    assert config["exclude"] == ["file1.py", "file2.py", "dir/"]
    assert config["paths"] == ["test.py"]


# LLM-generated content at query #25
#--------------------------

```python
def test_make_config():
    """Test make_config function with various configurations."""
    
    # Test 1: Default configuration with no arguments
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

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test 3: TOML file configuration
    import io
    toml_content = """
[tool.vulture]
min_confidence = 30
verbose = true
exclude = ["test_*.py", "docs"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 30
    assert config["verbose"] == True
    assert config["exclude"] == ["test_*.py", "docs"]

    # Test 4: CLI arguments override TOML configuration
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True  # From TOML
    assert config["exclude"] == ["test_*.py", "docs"]  # From TOML

    # Test 5: CSV parsing for list arguments
    config = make_config(argv=["--exclude", "file1.py,file2.py,dir1", 
                               "--ignore-decorators", "@app.route,@require_*",
                               "--ignore-names", "visit_*,do_*"])
    assert config["exclude"] == ["file1.py", "file2.py", "dir1"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test 6: Boolean flags
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "--verbose"])
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test 7: Custom config file path
    config = make_config(argv=["--config", "custom.toml"])
    assert config["config"] == "custom.toml"

    # Test 8: Path arguments
    config = make_config(argv=["src/main.py", "tests/"])
    assert config["paths"] == ["src/main.py", "tests/"]

    # Test 9: Error when no paths provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 10: Error for unknown configuration key
    import io
    bad_toml = """
[tool.vulture]
unknown_key = "value"
"""
    bad_file = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=bad_file)

    # Test 11: Error for wrong data type
    bad_type_toml = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    bad_type_file = io.BytesIO(bad_type_toml.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=bad_type_file)

    # Test 12: Empty TOML file
    import io
    empty_toml = io.BytesIO(b"")
    config = make_config(argv=["--verbose", "test_path"], tomlfile=empty_toml)
    assert config["verbose"] == True
    assert config["paths"] == ["test_path"]


# LLM-generated content at query #26
#--------------------------

```python
def test_make_config():
    # Test with no arguments (no toml file)
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

    # Test with some CLI arguments
    config = make_config(argv=["path1", "path2", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True

    # Test with TOML file
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
verbose = true
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True
    assert config["paths"] == []

    # Test CLI overrides TOML
    toml_content = """
[tool.vulture]
verbose = false
min_confidence = 50
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["verbose"] is True  # CLI overrides TOML
    assert config["min_confidence"] == 50  # TOML value kept

    # Test with minimum paths required
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])

    # Test with --version (should not raise InputError)
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])

    # Test with --help (should not raise InputError)
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])

    # Test with unknown configuration key
    import io
    toml_content = """
[tool.vulture]
unknown_key = "value"
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(tomlfile=toml_file)

    # Test with wrong type in TOML
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(tomlfile=toml_file)


# LLM-generated content at query #27
#--------------------------

```python
def test_make_config():
    # Test basic configuration with CLI args
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"
    
    # Test with custom CLI options
    config = make_config(argv=[
        "path.py",
        "--min-confidence", "50",
        "--verbose",
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@deco1,@deco2",
        "--ignore-names", "helper_,_internal",
        "--make-whitelist",
        "--sort-by-size",
        "--config", "custom_config.toml"
    ])
    assert config["paths"] == ["path.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@deco1", "@deco2"]
    assert config["ignore_names"] == ["helper_", "_internal"]
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["config"] == "custom_config.toml"
    
    # Test with TOML configuration (using StringIO as file-like object)
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
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    
    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path.py"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["cli_path.py"]
    assert config["exclude"] == ["file*.py", "dir/"]
    
    # Test missing paths raises error
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test with empty TOML
    empty_toml = io.BytesIO(b"")
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(tomlfile=empty_toml)
    
    # Test with TOML having only tool section but no paths
    toml_no_paths = io.BytesIO(b"[tool.vulture]\nverbose = true")
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(tomlfile=toml_no_paths)
    
    # Test invalid config key in TOML
    invalid_toml = io.BytesIO(b"[tool.vulture]\ninvalid_key = 123")
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(tomlfile=invalid_toml)
    
    # Test invalid config key in CLI
    # Note: This would normally fail at argparse level, but we test the _check_input_config
    # through the TOML parsing path
    pass


# LLM-generated content at query #28
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
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
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with CSV arguments
    config = make_config(argv=["--exclude", "file1.py,dir1", "--ignore-decorators", "deco1,deco2"])
    assert config["exclude"] == ["file1.py", "dir1"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]

    # Test with boolean flags
    config = make_config(argv=["--sort-by-size", "--make-whitelist"])
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True

    # Test with toml file (simulated as StringIO)
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

    # Test CLI overrides toml
    tomlfile = io.StringIO(toml_content)
    config = make_config(argv=["--min-confidence", "80", "--verbose", "false"], tomlfile=tomlfile)
    assert config["min_confidence"] == 80
    assert config["verbose"] is False  # CLI overrides toml

    # Test with empty toml file
    tomlfile = io.StringIO("")
    config = make_config(argv=["path1"], tomlfile=tomlfile)
    assert config["paths"] == ["path1"]
    assert config["min_confidence"] == 0  # Default value

    # Test InputError for missing paths
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])

    # Test InputError for invalid config key
    tomlfile = io.StringIO("[tool.vulture]\ninvalid_key = true")
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(argv=[], tomlfile=tomlfile)

    # Test InputError for wrong type
    tomlfile = io.StringIO("[tool.vulture]\nverbose = 123")
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=tomlfile)

    # Test with --version (should not raise InputError for missing paths)
    with pytest.raises(SystemExit):
        make_config(argv=["--version"])

    # Test with --help (should not raise InputError for missing paths)
    with pytest.raises(SystemExit):
        make_config(argv=["--help"])
```


# LLM-generated content at query #29
#--------------------------

```python
def test_make_config():
    # Test with no arguments (default config)
    config = make_config(argv=[], tomlfile=None)
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
    config = make_config(
        argv=["--verbose", "--min-confidence", "50", "path1", "path2"],
        tomlfile=None
    )
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file (using StringIO)
    import io
    toml_data = """
[tool.vulture]
paths = ["src/"]
exclude = ["test_*.py"]
verbose = true
"""
    config = make_config(argv=[], tomlfile=io.StringIO(toml_data))
    assert config["paths"] == ["src/"]
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] == True

    # Test CLI overrides TOML
    toml_data = """
[tool.vulture]
paths = ["src/"]
verbose = false
"""
    config = make_config(
        argv=["--verbose", "custom_path"],
        tomlfile=io.StringIO(toml_data)
    )
    assert config["verbose"] == True
    assert config["paths"] == ["custom_path"]

    # Test with exclude as comma-separated list
    config = make_config(
        argv=["--exclude", "*.pyc,__pycache__"],
        tomlfile=None
    )
    assert config["exclude"] == ["*.pyc", "__pycache__"]

    # Test with ignore-decorators and ignore-names
    config = make_config(
        argv=["--ignore-decorators", "@app.route,@require_*",
              "--ignore-names", "visit_*,do_*"],
        tomlfile=None
    )
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test make-whitelist and sort-by-size
    config = make_config(
        argv=["--make-whitelist", "--sort-by-size"],
        tomlfile=None
    )
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True

    # Test InputError raised when no paths given
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test InputError raised for unknown key
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["--unknown-key"], tomlfile=None)


# LLM-generated content at query #30
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with CLI args only (no TOML file)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test 2: CLI args override TOML config
    import io
    toml_data = """
[tool.vulture]
paths = ["toml_path.py"]
min_confidence = 50
verbose = true
"""
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=["--min-confidence", "80", "cli_path.py"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path.py"]
    assert config["min_confidence"] == 80
    assert config["verbose"] == True

    # Test 3: TOML config provides defaults when no CLI args given
    toml_file = io.BytesIO(toml_data.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["toml_path.py"]
    assert config["min_confidence"] == 50
    assert config["verbose"] == True

    # Test 4: Empty TOML section uses all defaults
    toml_file = io.BytesIO(b"[tool.vulture]\n")
    config = make_config(argv=["test.py"], tomlfile=toml_file)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    assert config["make_whitelist"] == False

    # Test 5: Multiple CLI args with various types
    config = make_config(argv=[
        "--exclude", "test_*,venv",
        "--ignore-decorators", "@app.route,@login_required",
        "--ignore-names", "helper_*,temp_*",
        "--make-whitelist",
        "--sort-by-size",
        "--verbose",
        "path1.py", "path2.py"
    ])
    assert config["exclude"] == ["test_*", "venv"]
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    assert config["ignore_names"] == ["helper_*", "temp_*"]
    assert config["make_whitelist"] == True
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1.py", "path2.py"]

    # Test 6: Raises InputError for missing paths
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])

    # Test 7: Raises InputError for invalid config key in TOML
    toml_file = io.BytesIO(b"""
[tool.vulture]
invalid_key = true
""")
    with pytest.raises(InputError):
        make_config(argv=["test.py"], tomlfile=toml_file)

    # Test 8: Raises InputError for wrong type in CLI
    with pytest.raises(InputError):
        make_config(argv=["--min-confidence", "not_an_int", "test.py"])


# LLM-generated content at query #31
#--------------------------

```python
def test_make_config():
    # Test with default values when no args and no toml file
    import tempfile
    import os
    
    # Test 1: Basic case with paths provided via CLI
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
    
    # Test 2: CLI arguments override defaults
    config = make_config(argv=[
        "--min-confidence", "50",
        "--verbose",
        "--sort-by-size",
        "--make-whitelist",
        "--exclude", "test_*.py,venv",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "path1", "path2"
    ])
    assert config["min_confidence"] == 50
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["make_whitelist"] == True
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["paths"] == ["path1", "path2"]
    
    # Test 3: TOML file with config values
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = f.name
    
    try:
        with open(toml_path, 'rb') as toml_file:
            config = make_config(argv=[], tomlfile=toml_file)
        
        assert config["exclude"] == ["file*.py", "dir/"]
        assert config["ignore_decorators"] == ["deco1", "deco2"]
        assert config["ignore_names"] == ["name1", "name2"]
        assert config["make_whitelist"] == True
        assert config["min_confidence"] == 10
        assert config["sort_by_size"] == True
        assert config["verbose"] == True
        assert config["paths"] == ["path1", "path2"]
    finally:
        os.unlink(toml_path)
    
    # Test 4: CLI arguments override TOML values
    toml_content = """
[tool.vulture]
min_confidence = 10
paths = ["toml_path"]
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = f.name
    
    try:
        with open(toml_path, 'rb') as toml_file:
            config = make_config(argv=["--min-confidence", "80", "cli_path"], tomlfile=toml_file)
        
        # CLI values should override TOML values
        assert config["min_confidence"] == 80
        assert config["paths"] == ["cli_path"]
    finally:
        os.unlink(toml_path)
    
    # Test 5: Error when no paths provided
    try:
        config = make_config(argv=[])
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e.message) == "Please pass at least one file or directory"
    
    # Test 6: Error with unknown configuration key
    toml_content = """
[tool.vulture]
unknown_key = "value"
paths = ["path1"]
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = f.name
    
    try:
        with open(toml_path, 'rb') as toml_file:
            config = make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)
    finally:
        os.unlink(toml_path)
    
    # Test 7: Wrong data type in TOML
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
paths = ["path1"]
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = f.name
    
    try:
        with open(toml_path, 'rb') as toml_file:
            config = make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)
    finally:
        os.unlink(toml_path)


# LLM-generated content at query #32
#--------------------------

```python
def test_make_config():
    # Test basic config with no arguments
    config = make_config(argv=[], tomlfile=None)
    assert config == DEFAULTS.copy()
    
    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--sort-by-size", "path1", "path2"], tomlfile=None)
    assert config["verbose"] == True
    assert config["sort_by_size"] == True
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 0
    assert config["make_whitelist"] == False
    
    # Test with TOML file
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    exclude = ["file*.py"]
    verbose = true
    paths = ["path1"]
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["file*.py"]
    assert config["verbose"] == True
    assert config["paths"] == ["path1"]
    
    # Test CLI arguments override TOML
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80", "custom_path"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["paths"] == ["custom_path"]
    
    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)
    
    # Test with unknown config key in TOML
    bad_toml = """
    [tool.vulture]
    unknown_key = true
    paths = ["test.py"]
    """
    bad_file = io.BytesIO(bad_toml.encode())
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=bad_file)
    
    # Test with wrong type in TOML
    wrong_type_toml = """
    [tool.vulture]
    verbose = "yes"
    paths = ["test.py"]
    """
    wrong_type_file = io.BytesIO(wrong_type_toml.encode())
    with pytest.raises(InputError, match="Data type for verbose must be 'bool'"):
        make_config(argv=[], tomlfile=wrong_type_file)


# LLM-generated content at query #33
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test basic config with no arguments
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
    
    # Test config with CLI arguments
    config = make_config(
        argv=["--verbose", "--min-confidence", "50", "path1", "path2"],
        tomlfile=None
    )
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]
    
    # Test config with TOML file
    from io import StringIO
    toml_content = """
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    ignore_decorators = ["deco1", "deco2"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    paths = ["path1", "path2"]
    """
    toml_file = StringIO(toml_content)
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["paths"] == ["path1", "path2"]
    
    # Test CLI overrides TOML
    toml_file.seek(0)
    config = make_config(
        argv=["--min-confidence", "80", "--verbose"],
        tomlfile=toml_file
    )
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    
    # Test config with --version (should not raise InputError)
    import sys
    try:
        config = make_config(argv=["--version"], tomlfile=None)
    except SystemExit:
        pass  # Expected behavior for --version
    
    # Test config with --help (should not raise InputError)
    try:
        config = make_config(argv=["--help"], tomlfile=None)
    except SystemExit:
        pass  # Expected behavior for --help
    
    # Test config with --make-whitelist
    config = make_config(
        argv=["--make-whitelist", "test.py"],
        tomlfile=None
    )
    assert config["make_whitelist"] == True
    assert config["paths"] == ["test.py"]
    
    # Test config with --ignore-names
    config = make_config(
        argv=["--ignore-names", "visit_*,do_*", "test.py"],
        tomlfile=None
    )
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test config with --exclude
    config = make_config(
        argv=["--exclude", "*settings.py,docs,*/test_*.py", "test.py"],
        tomlfile=None
    )
    assert config["exclude"] == ["*settings.py", "docs", "*/test_*.py"]


# LLM-generated content at query #34
#--------------------------

```python
def test_make_config():
    # Test with only CLI arguments (no TOML file)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] == False
    assert config["sort_by_size"] == False
    assert config["verbose"] == False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=[
        "path.py",
        "--exclude=test.py,*.txt",
        "--ignore-decorators=decorator1,decorator2",
        "--ignore-names=name1,name2",
        "--make-whitelist",
        "--min-confidence=50",
        "--sort-by-size",
        "--verbose",
    ])
    assert config["paths"] == ["path.py"]
    assert config["exclude"] == ["test.py", "*.txt"]
    assert config["ignore_decorators"] == ["decorator1", "decorator2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test with TOML file (simulated)
    import io
    toml_content = """
    [tool.vulture]
    paths = ["src/", "tests/"]
    exclude = ["*.pyc"]
    ignore_decorators = ["deco1"]
    ignore_names = ["name1"]
    make_whitelist = true
    min_confidence = 10
    sort_by_size = true
    verbose = true
    """
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src/", "tests/"]
    assert config["exclude"] == ["*.pyc"]
    assert config["ignore_decorators"] == ["deco1"]
    assert config["ignore_names"] == ["name1"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test that CLI arguments override TOML configuration
    toml_file.seek(0)
    config = make_config(
        argv=["custom_path.py", "--min-confidence=80", "--verbose"],
        tomlfile=toml_file
    )
    assert config["paths"] == ["custom_path.py"]  # CLI overrides TOML paths
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] == True  # CLI overrides TOML
    assert config["exclude"] == ["*.pyc"]  # TOML value preserved
    assert config["ignore_decorators"] == ["deco1"]  # TOML value preserved

    # Test with no paths (should raise InputError)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with empty paths in TOML (should raise InputError)
    empty_toml = b"[tool.vulture]\npaths = []\n"
    empty_file = io.BytesIO(empty_toml)
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=empty_file)

    # Test with unknown configuration key in TOML
    bad_toml = b"[tool.vulture]\nunknown_key = true\npaths = ['test.py']\n"
    bad_file = io.BytesIO(bad_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=bad_file)

    # Test with wrong type in TOML
    wrong_type_toml = b"[tool.vulture]\nmin_confidence = 'string'\npaths = ['test.py']\n"
    wrong_type_file = io.BytesIO(wrong_type_toml)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=wrong_type_file)
```


# LLM-generated content at query #35
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only (no TOML file)
    config = make_config(argv=["path1", "path2", "--verbose"])
    assert config["paths"] == ["path1", "path2"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["exclude"] == []
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    assert config["make_whitelist"] is False
    assert config["config"] == "pyproject.toml"

    # Test with TOML file (using StringIO to simulate file)
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

    # Test CLI overrides TOML
    tomlfile = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose", "--min-confidence=50"], tomlfile=tomlfile)
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with no paths raises error
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with empty TOML (no tool.vulture section)
    tomlfile = io.BytesIO(b"[tool.other]\nkey = 'value'")
    config = make_config(argv=["path1"], tomlfile=tomlfile)
    assert config["paths"] == ["path1"]
    assert config["verbose"] is False

    # Test with unknown config key in TOML
    tomlfile = io.BytesIO(b"[tool.vulture]\nunknown_key = 'value'")
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path1"], tomlfile=tomlfile)

    # Test with wrong type in TOML
    tomlfile = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path1"], tomlfile=tomlfile)


# LLM-generated content at query #36
#--------------------------

```python
def test_make_config():
    # Test with CLI arguments only (no TOML file)
    config = make_config(argv=["path1.py", "path2.py"])
    assert config["paths"] == ["path1.py", "path2.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] == False
    
    # Test with TOML file override
    import io
    toml_content = """
    [tool.vulture]
    min_confidence = 50
    verbose = true
    paths = ["toml_path.py"]
    """
    toml_file = io.StringIO(toml_content)
    # Need to encode for tomllib
    toml_bytes = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path.py"], tomlfile=toml_bytes)
    assert config["paths"] == ["cli_path.py"]  # CLI overrides TOML
    assert config["min_confidence"] == 50  # From TOML
    assert config["verbose"] == False  # CLI default overrides TOML
    
    # Test with empty TOML (no [tool.vulture] section)
    empty_toml = io.BytesIO(b"[build-system]\nrequires = ['setuptools']")
    config = make_config(argv=["test.py"], tomlfile=empty_toml)
    assert config["paths"] == ["test.py"]
    assert config["min_confidence"] == 0
    
    # Test that defaults are applied correctly
    config = make_config(argv=["test.py"])
    assert config["sort_by_size"] == False
    assert config["make_whitelist"] == False
    assert config["ignore_decorators"] == []
    assert config["ignore_names"] == []
    
    # Test that InputError is raised when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test with explicit config file path
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write("""
        [tool.vulture]
        min_confidence = 80
        """)
        temp_path = f.name
    
    try:
        config = make_config(argv=["--config", temp_path, "test.py"])
        assert config["min_confidence"] == 80
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_make_config():
    # Test with empty argv and no tomlfile (should use defaults)
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
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with tomlfile (using StringIO)
    import io
    toml_data = """
    [tool.vulture]
    paths = ["src"]
    exclude = ["test_*.py"]
    verbose = true
    """
    tomlfile = io.StringIO(toml_data)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["paths"] == ["src"]
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] is True

    # Test CLI arguments override TOML settings
    toml_data2 = """
    [tool.vulture]
    paths = ["src"]
    verbose = false
    """
    tomlfile2 = io.StringIO(toml_data2)
    config = make_config(argv=["--verbose", "custom_path"], tomlfile=tomlfile2)
    assert config["verbose"] is True  # CLI overrides TOML
    assert config["paths"] == ["custom_path"]  # CLI overrides TOML

    # Test with --version (should not raise error)
    import sys
    config = make_config(argv=["--version"])
    # Should exit with 0, but we can't easily test that without capturing SystemExit

    # Test with --help (should not raise error)
    config = make_config(argv=["--help"])
    # Similar to --version

    # Test that missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--verbose"])

    # Test with exclude parameter
    config = make_config(argv=["--exclude", "dir1,dir2", "path1"])
    assert config["exclude"] == ["dir1", "dir2"]
    assert config["paths"] == ["path1"]

    # Test with ignore_decorators and ignore_names
    config = make_config(argv=["--ignore-decorators", "@app.route,@login_required", 
                               "--ignore-names", "visit_*,do_*", "path1"])
    assert config["ignore_decorators"] == ["@app.route", "@login_required"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test with make_whitelist and sort_by_size
    config = make_config(argv=["--make-whitelist", "--sort-by-size", "path1"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #38
#--------------------------

```python
def test_make_config():
    # Test with no arguments and no toml file (should use defaults)
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

    # Test with CLI arguments overriding defaults
    config = make_config(
        argv=["--verbose", "--min-confidence", "50", "test_path"],
        tomlfile=None
    )
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["test_path"]

    # Test with toml file settings
    toml_content = b"""
    [tool.vulture]
    exclude = ["file*.py", "dir/"]
    verbose = true
    """
    import io
    tomlfile = io.BytesIO(toml_content)
    config = make_config(argv=[], tomlfile=tomlfile)
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] == True
    assert config["paths"] == []  # Default

    # Test CLI override of toml settings
    tomlfile2 = io.BytesIO(b'[tool.vulture]\nverbose = true')
    config = make_config(
        argv=["--verbose", "--exclude", "test_*.py,dir2"],
        tomlfile=tomlfile2
    )
    assert config["verbose"] == True
    assert config["exclude"] == ["test_*.py", "dir2"]  # CLI overrides toml

    # Test that paths are required (should raise InputError if empty)
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nmin_confidence = 10'))

    # Test with paths provided via toml
    tomlfile3 = io.BytesIO(b'[tool.vulture]\npaths = ["src/", "tests/"]')
    config = make_config(argv=[], tomlfile=tomlfile3)
    assert config["paths"] == ["src/", "tests/"]

    # Test unknown configuration key raises error
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nunknown_key = true'))

    # Test wrong type for configuration value
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nmin_confidence = "not_an_int"'))

    # Test that missing toml file falls back to defaults and CLI args
    config = make_config(argv=["test_path", "--verbose"], tomlfile=None)
    assert config["paths"] == ["test_path"]
    assert config["verbose"] == True
    assert config["exclude"] == []  # Default
```


# LLM-generated content at query #39
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Default config with no arguments
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

    # Test 2: CLI arguments override defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test 3: TOML file configuration
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

    # Test 4: CLI arguments override TOML settings
    toml_file.seek(0)
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] is True  # CLI overrides TOML
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML

    # Test 5: Empty TOML file
    empty_toml = io.BytesIO(b"")
    config = make_config(argv=["path1"], tomlfile=empty_toml)
    assert config["paths"] == ["path1"]

    # Test 6: TOML without vulture section
    no_vulture_toml = io.BytesIO(b"[tool.other]\nkey = 'value'")
    config = make_config(argv=["path1"], tomlfile=no_vulture_toml)
    assert config["paths"] == ["path1"]

    # Test 7: InputError for missing paths
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 8: InputError for wrong type in TOML
    wrong_type_toml = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'string'")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(tomlfile=wrong_type_toml)

    # Test 9: InputError for unknown key in TOML
    unknown_key_toml = io.BytesIO(b"[tool.vulture]\nunknown_key = 'value'")
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(tomlfile=unknown_key_toml)

    # Test 10: CLI with version flag should not raise InputError for missing paths
    with pytest.raises(SystemExit) as exc_info:
        make_config(argv=["--version"])
    assert exc_info.value.code == 0

    # Test 11: CLI with help flag should not raise InputError for missing paths
    with pytest.raises(SystemExit) as exc_info:
        make_config(argv=["--help"])
    assert exc_info.value.code == 0

    # Test 12: Config with specific config path
    config = make_config(argv=["--config", "custom.toml", "path1"])
    assert config["config"] == "custom.toml"
    assert config["paths"] == ["path1"]

    # Test 13: Make whitelist flag
    config = make_config(argv=["--make-whitelist", "path1"])
    assert config["make_whitelist"] is True

    # Test 14: Sort by size flag
    config = make_config(argv=["--sort-by-size", "path1"])
    assert config["sort_by_size"] is True

    # Test 15: Ignore decorators and names
    config = make_config(argv=["--ignore-decorators", "@app.route,@require_*", 
                               "--ignore-names", "visit_*,do_*", "path1"])
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]

    # Test 16: Exclude patterns
    config = make_config(argv=["--exclude", "*settings.py,docs,*/test_*.py,venv", "path1"])
    assert config["exclude"] == ["*settings.py", "docs", "*/test_*.py", "venv"]


# LLM-generated content at query #40
#--------------------------

```python
def test_make_config():
    # Test 1: Basic config with CLI paths only
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

    # Test 2: Config with CLI options overriding defaults
    config = make_config(argv=[
        "path1", "path2",
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route",
        "--ignore-names", "visit_*",
        "--make-whitelist",
        "--min-confidence", "50",
        "--sort-by-size",
        "--verbose",
        "--config", "custom_config.toml"
    ])
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["config"] == "custom_config.toml"

    # Test 3: Config with TOML file (simulated)
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
    assert config["paths"] == ["path1", "path2"]
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] == True
    assert config["verbose"] == True

    # Test 4: CLI arguments override TOML settings
    toml_content = """
[tool.vulture]
min_confidence = 10
verbose = false
paths = ["toml_path"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["cli_path", "--min-confidence", "20", "--verbose"], tomlfile=toml_file)
    assert config["paths"] == ["cli_path"]
    assert config["min_confidence"] == 20
    assert config["verbose"] == True

    # Test 5: Empty TOML file
    toml_content = ""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["path1"], tomlfile=toml_file)
    assert config["paths"] == ["path1"]
    assert config["min_confidence"] == 0
    assert config["verbose"] == False

    # Test 6: TOML file with no vulture section
    toml_content = "[tool.other]\nkey = 'value'"
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["path1"], tomlfile=toml_file)
    assert config["paths"] == ["path1"]

    # Test 7: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 8: Error with unknown config key in TOML
    toml_content = """
[tool.vulture]
unknown_key = true
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file)

    # Test 9: Error with wrong type in TOML
    toml_content = """
[tool.vulture]
min_confidence = "not_an_int"
paths = ["path1"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=toml_file)


# LLM-generated content at query #41
#--------------------------

```python
def test_make_config():
    # Test with no arguments (default config)
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

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "path1", "path2"])
    assert config["verbose"] == True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["path1", "path2"]

    # Test with TOML file provided
    import io
    toml_content = """
[tool.vulture]
paths = ["src/"]
exclude = ["test_*.py"]
verbose = true
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["paths"] == ["src/"]
    assert config["exclude"] == ["test_*.py"]
    assert config["verbose"] == True

    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--verbose"], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["paths"] == ["src/"]

    # Test with CSV arguments
    config = make_config(argv=["--exclude", "file1.py,file2.py", "--ignore-decorators", "@app.route,@require_*"])
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]

    # Test InputError for missing paths
    try:
        config = make_config(argv=[], tomlfile=None)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert str(e.message) == "Please pass at least one file or directory"

    # Test InputError for unknown configuration key
    import io
    bad_toml = """
[tool.vulture]
unknown_key = true
"""
    try:
        toml_file = io.BytesIO(bad_toml.encode())
        config = make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Unknown configuration key" in str(e.message)

    # Test InputError for wrong data type
    bad_toml = """
[tool.vulture]
min_confidence = "not_an_int"
"""
    try:
        toml_file = io.BytesIO(bad_toml.encode())
        config = make_config(argv=[], tomlfile=toml_file)
        assert False, "Should have raised InputError"
    except InputError as e:
        assert "Data type for min_confidence must be 'int'" in str(e.message)


# LLM-generated content at query #42
#--------------------------

```python
def test_make_config():
    # Test with CLI args only (no TOML file)
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

    # Test with CLI args overriding defaults
    config = make_config(argv=[
        "path.py",
        "--exclude=test_*.py,docs",
        "--ignore-decorators=@app.route",
        "--ignore-names=visit_*",
        "--make-whitelist",
        "--min-confidence=50",
        "--sort-by-size",
        "--verbose",
        "--config=custom.toml"
    ])
    assert config["paths"] == ["path.py"]
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route"]
    assert config["ignore_names"] == ["visit_*"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    assert config["config"] == "custom.toml"

    # Test with TOML file (using io.StringIO to simulate file)
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
    # Note: _parse_toml expects a binary file, so we need to adjust
    # For testing, we'll use BytesIO instead
    toml_bytes = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["extra_path.py"], tomlfile=toml_bytes)
    # TOML values should be present
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["min_confidence"] == 10
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    # CLI args should override TOML values
    assert config["paths"] == ["extra_path.py"]

    # Test with TOML file and CLI override
    toml_bytes = io.BytesIO(toml_content.encode('utf-8'))
    config = make_config(argv=["--verbose=False", "extra_path.py"], tomlfile=toml_bytes)
    # CLI should override TOML verbose value
    assert config["verbose"] is False

    # Test with no paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test with invalid config key in TOML
    bad_toml = io.BytesIO(b'[tool.vulture]\ninvalid_key = true')
    with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
        make_config(argv=["test.py"], tomlfile=bad_toml)

    # Test with wrong type in TOML
    bad_type_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "not_an_int"')
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["test.py"], tomlfile=bad_type_toml)


# LLM-generated content at query #43
#--------------------------

```python
def test_make_config():
    # Test with minimal arguments
    config = make_config(argv=["file1.py"])
    assert config["paths"] == ["file1.py"]
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] == False
    
    # Test with all CLI arguments
    config = make_config(argv=[
        "file1.py", "dir1/",
        "--exclude", "test_*.py,venv",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
        "--make-whitelist",
        "--min-confidence", "50",
        "--sort-by-size",
        "--verbose",
        "--config", "custom.toml"
    ])
    assert config["paths"] == ["file1.py", "dir1/"]
    assert config["exclude"] == ["test_*.py", "venv"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    assert config["make_whitelist"] == True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] == True
    assert config["verbose"] == True
    assert config["config"] == "custom.toml"
    
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
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["verbose"] == True
    
    # Test CLI overrides TOML
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["custom_path.py", "--min-confidence", "80"], tomlfile=toml_file)
    assert config["paths"] == ["custom_path.py"]
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    
    # Test empty config raises InputError
    import pytest
    with pytest.raises(InputError):
        make_config(argv=[])
    
    # Test with no paths in TOML
    toml_no_paths = io.BytesIO(b"[tool.vulture]\nexclude = []")
    with pytest.raises(InputError):
        make_config(argv=[], tomlfile=toml_no_paths)
    
    # Test verbose with TOML detection
    toml_file = io.BytesIO(b"[tool.vulture]\nverbose = true\npaths = ['test.py']")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] == True
    assert config["paths"] == ["test.py"]


# LLM-generated content at query #44
#--------------------------

```python
def test_make_config():
    """Test make_config function with various scenarios."""
    
    # Test 1: Basic config with CLI paths
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
    
    # Test 2: Config with CLI options
    config = make_config(argv=[
        "path1.py",
        "--verbose",
        "--min-confidence", "50",
        "--sort-by-size",
        "--make-whitelist",
        "--exclude", "test_*.py,docs",
        "--ignore-decorators", "@app.route,@require_*",
        "--ignore-names", "visit_*,do_*",
    ])
    assert config["paths"] == ["path1.py"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["make_whitelist"] is True
    assert config["exclude"] == ["test_*.py", "docs"]
    assert config["ignore_decorators"] == ["@app.route", "@require_*"]
    assert config["ignore_names"] == ["visit_*", "do_*"]
    
    # Test 3: Config with TOML file (using StringIO)
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
    assert config["paths"] == ["path1", "path2"]
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["file*.py", "dir/"]
    assert config["ignore_decorators"] == ["deco1", "deco2"]
    assert config["ignore_names"] == ["name1", "name2"]
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True
    
    # Test 4: CLI overrides TOML
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80  # CLI overrides TOML
    assert config["verbose"] is True  # CLI overrides TOML
    assert config["exclude"] == ["file*.py", "dir/"]  # From TOML
    
    # Test 5: Error when no paths provided
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])
    
    # Test 6: Error with unknown config key
    bad_toml = io.BytesIO(b"[tool.vulture]\nunknown_key = true")
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=bad_toml)
    
    # Test 7: Error with wrong type in config
    bad_type_toml = io.BytesIO(b"[tool.vulture]\nmin_confidence = 'not_an_int'")
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=[], tomlfile=bad_type_toml)
    
    # Test 8: Config with custom config file path
    config = make_config(argv=["--config", "custom_config.toml", "path.py"])
    assert config["config"] == "custom_config.toml"
    assert config["paths"] == ["path.py"]
    
    # Test 9: Config with multiple paths
    config = make_config(argv=["dir1", "dir2", "file1.py", "file2.py"])
    assert config["paths"] == ["dir1", "dir2", "file1.py", "file2.py"]
    
    # Test 10: Partial TOML config with defaults
    partial_toml = io.BytesIO(b"[tool.vulture]\npaths = [\"src\"]")
    config = make_config(argv=["--verbose"], tomlfile=partial_toml)
    assert config["paths"] == ["src"]
    assert config["verbose"] is True
    assert config["min_confidence"] == 0  # Default value
    assert config["sort_by_size"] is False  # Default value


# LLM-generated content at query #45
#--------------------------

```python
def test_make_config():
    # Test with no arguments, no toml file (should use defaults)
    config = make_config(argv=[], tomlfile=None)
    assert config["paths"] == []
    assert config["min_confidence"] == 0
    assert config["exclude"] == []
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

    # Test with CLI arguments overriding defaults
    config = make_config(argv=["--verbose", "--min-confidence", "50", "src/"])
    assert config["verbose"] is True
    assert config["min_confidence"] == 50
    assert config["paths"] == ["src/"]

    # Test with toml file providing configuration
    import io
    toml_content = """
[tool.vulture]
exclude = ["test_*.py"]
ignore_decorators = ["decorator1"]
sort_by_size = true
paths = ["src/"]
"""
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["exclude"] == ["test_*.py"]
    assert config["ignore_decorators"] == ["decorator1"]
    assert config["sort_by_size"] is True
    assert config["paths"] == ["src/"]

    # Test CLI arguments override toml configuration
    toml_file = io.BytesIO(toml_content.encode())
    config = make_config(argv=["--min-confidence", "80", "--verbose"], tomlfile=toml_file)
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert config["exclude"] == ["test_*.py"]  # from toml

    # Test verbose output when loading from toml
    toml_file = io.BytesIO(b"""
[tool.vulture]
verbose = true
paths = ["src/"]
""")
    config = make_config(argv=[], tomlfile=toml_file)
    assert config["verbose"] is True

    # Test missing paths raises InputError
    import pytest
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=None)

    # Test with specific toml file path (simulating existing pyproject.toml)
    # Note: This test requires a temporary file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b"""
[tool.vulture]
paths = ["src/"]
min_confidence = 30
""")
        temp_path = f.name
    
    try:
        config = make_config(argv=[f"--config={temp_path}"])
        assert config["min_confidence"] == 30
        assert config["paths"] == ["src/"]
    finally:
        os.unlink(temp_path)


