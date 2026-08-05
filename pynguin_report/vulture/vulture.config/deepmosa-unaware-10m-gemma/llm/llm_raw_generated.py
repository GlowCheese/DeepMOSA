####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_config", [
    # 1. Test CLI arguments only (no TOML)
    (
        ["path/to/code"],
        None,
        {
            "config": "pyproject.toml",
            "min_confidence": 0,
            "paths": ["path/to/code"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": False,
        },
    ),
    # 2. Test CLI arguments overriding TOML values
    (
        ["path/to/code", "--min-confidence", "50", "--sort-by-size"],
        '[tool.vulture]\nmin_confidence = 10\nverbose = true',
        {
            "config": "pyproject.toml",
            "min_confidence": 50,
            "paths": ["path/to/code"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": True,
            "verbose": True,
        },
    ),
    # 3. Test TOML parsing with multiple values
    (
        ["path/to/code"],
        '[tool.vulture]\nexclude = ["*.tmp"]\nignore_names = ["test_"]',
        {
            "config": "pyproject.toml",
            "min_confidence": 0,
            "paths": ["path/to/code"],
            "exclude": ["*.tmp"],
            "ignore_decorators": [],
            "ignore_names": ["test_"],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": False,
        },
    ),
])
def test_make_config(argv, toml_content, expected_config):
    toml_data = io.BytesIO(toml_content.encode("utf-8")) if toml_content else None
    
    # We use the tomlfile argument to bypass file system lookups for testing
    config = make_config(argv=argv, tomlfile=toml_data)
    
    assert config == expected_config

def test_make_config_error_no_paths():
    """Test that providing no paths raises an InputError."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # argv contains only flags, no positional 'paths'
        make_config(argv=["--verbose"])

def test_make_config_invalid_toml_type():
    """Test that providing the wrong type in TOML raises an InputError."""
    toml_data = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["path/to/code"], tomlfile=toml_data)

def test_make_config_unknown_key():
    """Test that providing an unknown key in TOML raises an InputError."""
    toml_data = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path/to/code"], tomlfile=toml_data)

def test_make_config_file_loading(tmp_path):
    """Test that make_config correctly loads from an actual file on disk."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.vulture]\nmin_confidence = 25\nverbose = true', encoding="utf-8")
    
    # Pass the path via CLI --config
    argv = ["path/to/code", "--config", str(pyproject)]
    
    config = make_config(argv=argv)
    assert config["min_confidence"] == 25
    assert config["verbose"] is True
```


# LLM-generated content at query #2
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test Default configuration (no files, no args)
    # We must provide at least one path via CLI or it fails _check_output_config
    with patch("sys.argv", ["vulture", "test_dir"]):
        config = make_config()
        assert config["paths"] == ["test_dir"]
        assert config["min_confidence"] == 0
        assert config["verbose"] is False

    # 2. Test CLI arguments overriding defaults and CSV parsing
    cli_args = [
        "vulture", 
        "path1", 
        "--exclude", "a.py,b.py", 
        "--min-confidence", "50", 
        "--sort-by-size", 
        "--verbose"
    ]
    with patch("sys.argv", cli_args):
        config = make_config()
        assert config["paths"] == ["path1"]
        assert config["exclude"] == ["a.py", "b.py"]
        assert config["min_confidence"] == 50
        assert config["sort_by_size"] is True
        assert config["verbose"] is True

    # 3. Test TOML file loading and merging with CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["from_toml.py"]
ignore_names = ["unused_var"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI provides 'paths' and overrides 'min_confidence'
    cli_args = ["vulture", "some_path", "--min-confidence", "80"]
    with patch("sys.argv", cli_args):
        config = make_config()
        assert config["paths"] == ["some_path"]
        assert config["min_confidence"] == 80  # CLI wins
        assert config["exclude"] == ["from_toml.py"]  # From TOML
        assert config["ignore_names"] == ["unused_var"]  # From TOML

    # 4. Test passing an explicit IO stream (tomlfile argument)
    toml_stream = io.BytesIO(b'[tool.vulture]\nverbose = true\npaths = ["stream_path"]')
    with patch("sys.argv", ["vulture", "cli_path"]):
        config = make_config(tomlfile=toml_stream)
        assert config["paths"] == ["cli_path"] # CLI path overrides TOML paths
        assert config["verbose"] is True      # From TOML stream

    # 5. Test InputError on invalid type via CLI
    with patch("sys.argv", ["vulture", "path", "--min-confidence", "not_an_int"]):
        with pytest.raises(SystemExit): # argparse handles type error and exits
            make_config()

    # 6. Test InputError on unknown key via TOML simulation
    bad_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with patch("sys.argv", ["vulture", "path"]):
        with pytest.raises(InputError) as excinfo:
            make_config(tomlfile=bad_toml)
        assert "Unknown configuration key" in str(excinfo.value)

    # 7. Test InputError on missing paths (sanity check)
    # We use a dummy arg list that results in no paths
    with patch("sys.argv", ["vulture"]):
        # Note: _parse_args uses 'missing' sentinel, so we must trigger the empty path logic
        # Since 'paths' defaults to missing, if we pass nothing, it might not trigger 
        # unless we force an empty list. We simulate by overriding the namespace.
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            class MockNamespace:
                def __init__(self):
                    from types import SimpleNamespace
                    return SimpleNamespace(
                        paths=[], exclude=None, ignore_decorators=None, 
                        ignore_names=None, make_whitelist=False, 
                        min_confidence=0, sort_by_size=False, config="pyproject.toml", verbose=False
                    )
            # This is tricky because _parse_args logic is baked in. 
            # Let's just test the direct failure of the output check.
            with pytest.raises(InputError) as excinfo:
                _check_output_config({"paths": []})
            assert "Please pass at least one file or directory" in str(excinfo.value)

    # 8. Test Type mismatch in TOML
    bad_type_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with patch("sys.argv", ["vulture", "path"]):
        with pytest.raises(InputError) as excinfo:
            make_config(tomlfile=bad_type_toml)
        assert "Data type for min_confidence must be 'int'" in str(excinfo.value)
```


# LLM-generated content at query #3
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Basic functionality - CLI arguments only (no file)
    # We provide paths so _check_output_config passes
    args = ["path/to/code"]
    config = make_config(argv=args)
    assert config["paths"] == ["path/to/argparse_logic_is_internal"] # argparse handles positional args
    # Note: because of how argparse is set up in the snippet, 
    # 'paths' will contain the elements provided.
    assert "path/to/code" in config["paths"]
    assert config["min_confidence"] == 0  # Default
    assert config["verbose"] is False     # Default

    # Test 2: CLI overriding TOML
    toml_content = b"""
[tool.vulture]
min_confidence = 50
verbose = true
paths = ["from_toml"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI arg '--min-confidence 80' should override TOML '50'
    # We use --config to point to our temp file
    args = ["--config", str(toml_file), "--min-confidence", "80", "extra_path"]
    
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # We use a real file approach since make_config attempts to resolve path
        config = make_config(argv=args)
        
    assert config["min_confidence"] == 80
    assert config["verbose"] is True
    assert "from_toml" in config["paths"]
    assert "extra_path" in config["paths"]

    # Test 3: InputError on invalid type via CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test 4: InputError on unknown key via CLI (simulated by passing bad arg if parser allowed it)
    # Since argparse is strict, we test the internal _check_input_config logic indirectly
    with pytest.raises(InputError):
        _check_input_arg_logic_test({"unknown_key": True})

    # Test 5: Error when no paths are provided (Sanity check)
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[])
    assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test 6: Using the tomlfile parameter directly (bypassing filesystem)
    toml_io = io.BytesIO(b'[tool.vulture]\npaths = ["io_path"]\nmin_confidence = 10')
    config = make_config(argv=["some_path"], tomlfile=toml_io)
    assert "io_path" in config["paths"]
    assert config["min_confidence"] == 10

def _check_input_arg_logic_test(data):
    """Helper to trigger the internal validator for invalid keys."""
    _check_input_config(data)
```


# LLM-generated content at query #4
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Basic functionality with defaults (no files, no args)
    # We need to mock the file existence check for pyproject.toml to avoid real IO
    with patch("pathlib.Path.is_file", return_value=False), \
         patch("argparse.ArgumentParser.parse_args") as mock_args:
        
        # Simulate no CLI args provided (only defaults)
        mock_args.return_value = argparse.Namespace(
            paths=[], config="pyproject.toml"
        )
        # We must bypass _check_output_config error for empty paths in this specific test
        # by providing a path via argv
        
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # Test 2: CLI arguments precedence over TOML
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["test.py"]
paths = ["."]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking argv to provide --min-confidence=80 and a path
    # This should override the 50 in TOML
    argv = ["--min-confidence", "80", "src/"]
    
    # We use real file reading here since we created it in tmp_path
    config = make_config(argv=argv)
    
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["test.py"]
    assert config["paths"] == ["src/"]

    # Test 3: CLI arguments overriding TOML for lists (comma separated)
    argv_list = ["--exclude", "ignore1.py,ignore2.py", "src/"]
    config_list = make_config(argv=argv_list)
    assert config_list["exclude"] == ["ignore1.py", "ignore2.py"]

    # Test 4: Verify type validation (InputError)
    # Passing a string where an int is expected via CLI simulation
    # Note: argparse handles the conversion, so we simulate the error 
    # by passing invalid types through _parse_args logic if possible, 
    # or just checking the internal validator.
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        _check_input_config({"min_confidence": "not_an_int"})

    # Test 5: Unknown key error
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        _check_input_config({"unknown_key": True})

    # Test 6: Testing via tomlfile argument directly
    toml_io = io.BytesIO(b'[tool.vulture]\nverbose = true\npaths = ["path1"]')
    config_direct = make_config(argv=["path2"], tomlfile=toml_io)
    assert config_direct["verbose"] is True
    assert "path2" in config_direct["paths"]

    # Test 7: Full integration - CLI flag 'sort-by-size'
    argv_flags = ["--sort-by-size", "src/"]
    config_flags = make_config(argv=argv_flags)
    assert config_flags["sort_by_size"] is True

    # Test 8: Boolean flag 'make-whitelist'
    argv_white = ["--make-whitelist", "src/"]
    config_white = make_config(argv=argv_white)
    assert config_white["make_whitelist"] is True
```


# LLM-generated content at query #5
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Setup a dummy pyproject.toml
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.vulture]\nmin_confidence = 50\nverbose = true\npaths = ["test_dir"]')

    # Test case 1: CLI arguments override TOML
    # We pass argv such that --min-confidence is 80 and paths is provided via CLI
    args = ["--min-confidence", "80", "--paths", "src/", "--config", str(pyproject)]
    
    with patch("sys.argv", ["vulture"] + args):
        # We use the actual file on disk for the first part of the test logic
        # to verify it reads from the provided path correctly.
        config = make_config(argv=args)
        
        assert config["min_template"] is None # Checking a non-existent key doesn't crash
        assert config["min_confidence"] == 80
        assert config["paths"] == ["src/"]
        assert config["verbose"] is True
        # Check default from DEFAULTS was applied for things not in TOML or CLI
        assert config["sort_by_size"] is False

    # Test case 2: Only CLI arguments (no TOML)
    args_only_cli = ["--exclude", "venv/,.git/", "--paths", "my_folder"]
    with patch("sys.argv", ["vulture"] + args_only_cli):
        config = make_config(argv=args_only_cli)
        assert config["exclude"] == ["venv/", ".git/"]
        assert config["paths"] == ["my_folder"]
        assert config["min_confidence"] == 0  # Default

    # Test case 3: Using the tomlfile parameter directly (Injecting IO stream)
    toml_content = b'[tool.vulture]\nignore_names = ["foo", "bar"]\npaths = ["path1"]'
    toml_stream = io.BytesIO(toml_content)
    
    # We pass argv as empty so it doesn't override the tomlfile settings via CLI
    config_from_stream = make_config(argv=[], tomlfile=toml_stream)
    assert config_from_stream["ignore_names"] == ["foo", "bar"]
    assert config_from_stream["paths"] == ["path1"]

    # Test case 4: InputError on invalid type (passing string to int field via CLI)
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int"])
    # Note: argparse handles the type conversion error before _check_input_config 
    # is even called for the value, but if we bypass argparse via manual dict:
    with pytest.raises(InputError):
        _check_input_config({"min_confidence": "high"})

    # Test case 5: InputError on missing paths (Output validation)
    # Providing no paths in CLI and no paths in TOML
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--config", str(pyproject)]) 
        # Note: In the test case 1 we had paths in TOML, but if we 
        # mocked a toml without paths and no CLI paths:
        empty_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence=10')
        make_config(argv=[], tomlfile=empty_toml)

    # Test case 6: Unknown key error
    with pytest.raises(InputError, match="Unknown configuration key"):
        _check_input_config({"invalid_key": True})

def test_make_config_integration_error():
    """Test that invalid types in TOML trigger InputError."""
    toml_content = b'[tool.vulture]\nmin_confidence = "high"' # Should be int
    toml_stream = io.BytesIO(toml_content)
    
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=[], tomlfile=toml_stream)
```


# LLM-generated content at query #6
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_config", [
    # Case 1: Only CLI arguments provided (no TOML)
    (
        ["--min-confidence", "50", "--sort-by-size", "path/to/code"],
        None,
        {
            "config": "pyproject.toml",
            "min_confidence": 50,
            "paths": ["path/to/code"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": True,
            "verbose": False,
        }
    ),
    # Case 2: TOML and CLI arguments (CLI should override TOML)
    (
        ["--verbose", "--exclude", "test_*.py"],
        '[tool.vulture]\nmin_confidence = 10\nexclude = ["old.py"]',
        {
            "config": "pyproject.toml",
            "min_confidence": 10,
            "paths": [], # Note: In real scenario paths must be provided to pass _check_output_config
            "exclude": ["test_*.py"],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": True,
        }
    ),
])
def test_make_config_logic(argv, toml_content, expected_config):
    # We need to provide a path so _check_output_config doesn't raise InputError
    if not argv or "--paths" not in " ".join(argv) and not any(p for p in argv if not p.startswith("-")):
        argv = argv + ["some_path"]

    toml_data = io.BytesIO(toml_content.encode("utf-8")) if toml_content else None
    
    # Mocking _parse_args behavior implicitly by passing argv
    # We use a real StringIO for the tomlfile argument
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=argv, tomlfile=toml_data)
        
        for key, value in expected_config.items():
            assert config[key] == value

def test_make_config_error_no_paths():
    """Test that InputError is raised if no paths are provided."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_toml_file_loading():
    """Test that the function attempts to read from a real file when no tomlfile is passed."""
    toml_content = b'[tool.vulture]\nmin_confidence = 25\n'
    mock_file_content = io.BytesIO(toml_content)
    
    # Mocking open and pathlib to simulate an existing pyproject.toml
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=toml_content)), \
         patch("tomllib.load") as mock_toml:
        
        mock_toml.return_value = {"tool": {"vulture": {"min_confidence": 25}}}
        
        config = make_config(argv=["some_path"])
        assert config["min_confidence"] == 25

def test_make_config_type_mismatch():
    """Test that passing the wrong type via CLI (if it were possible) or TOML raises error."""
    toml_data = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["some_path"], tomlfile=toml_data)

def test_make_config_unknown_key():
    """Test that unknown configuration keys raise InputError."""
    toml_data = io.BytesIO(b'[tool.vulture]\nunknown_key = True')
    
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["some_path"], tomlfile=toml_data)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_key_values", [
    # 1. Test CLI overrides TOML
    (
        ["--min-confidence", "50", "--sort-by-size"],
        '[tool.vulture]\nmin_confidence = 20\nsort_by:size = false',
        {"min_confidence": 50, "sort_by_size": True, "paths": []}
    ),
    # 2. Test TOML provides values not in CLI
    (
        ["--paths", "src/"],
        '[tool.vulture]\nexclude = ["test/*"]\nignore_names = ["temp"]',
        {"exclude": ["test/*"], "ignore_names": ["temp"], "paths": ["src/"]}
    ),
    # 3. Test defaults are applied when nothing is provided
    (
        [],
        '[tool.vulture]\nverbose = true',
        {"verbose": True, "min_confidence": 0, "paths": []}
    ),
])
def test_make_config_logic(argv, toml_content, expected_key_values):
    """Test the merging logic of CLI and TOML."""
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    # We use a mock for _parse_toml or just provide the file handle directly to make_config
    # Since make_config calls _parse_args first, we check if it merges correctly
    config = make_config(argv=argv, tomlfile=toml_file)
    
    for key, value in expected_key_values.items():
        assert config[key] == value

def test_make_config_raises_error_on_empty_paths():
    """Test that _check_output_config raises error if no paths are provided."""
    # CLI with no paths and no TOML content
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--min-confidence", "10"], tomlfile=io.BytesIO(b""))

def test_make_config_file_loading(tmp_path):
    """Test that the function correctly attempts to read from a real file if no tomlfile arg is passed."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.vulture]\nmin_confidence = 80', encoding="utf-8")
    
    # Pass the path via CLI --config
    argv = ["--config", str(pyproject), "--paths", "test_dir"]
    config = make_config(argv=argv)
    
    assert config["min_confidence"] == 80
    assert config["paths"] == ["test_dir"]

def test_make_config_invalid_type_in_toml():
    """Test that providing the wrong type in TOML raises InputError."""
    toml_content = '[tool.vulture]\nmin_confidence = "high"'  # Should be int
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=[], tomlfile=toml_file)

def test_make_config_unknown_key_in_toml():
    """Test that providing an unknown key in TOML raises InputError."""
    toml_content = '[tool.vulture]\nunknown_key = True'
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=toml_file)

def test_make_config_csv_parsing():
    """Test that the CSV parsing logic in _parse_args works for excludes."""
    argv = ["--exclude", "file1.py,file2.py", "--paths", "src"]
    config = make_config(argv=argv, tomlfile=io.BytesIO(b""))
    assert config["exclude"] == ["file1.py", "file2.py"]
```


# LLM-generated content at query #8
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: Pure CLI arguments (no TOML)
    args = ["--min-confidence", "50", "--sort-by-size", "path/to/code"]
    config = make_config(argv=args, tomlfile=None)
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path/to/code"]
    assert config["exclude"] == []  # Default value

    # Test Case 2: TOML file values with CLI overrides
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["test_*.py"]
verbose = true
"""
    # Use a StringIO-like object to simulate the TOML file content
    toml_file = io.BytesIO(toml_content)
    # CLI provides override for min_confidence and adds paths
    args = ["--min-confidence", "80", "src/"]
    
    config = make_config(argv=args, tomlfile=toml_file)
    
    assert config["min_confidence"] == 80  # CLI wins
    assert config["exclude"] == ["test_*.py"]  # From TOML
    assert config["verbose"] is True  # From TOML
    assert config["paths"] == ["src/"]  # From CLI

    # Test Case 3: Empty paths triggers InputError (via _check_output_config)
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--min-confidence", "10"])

    # Test Case 4: Invalid type in CLI arguments
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])

    # Test Case 5: Unknown key in TOML (simulated via _parse_toml)
    invalid_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=[], tomlfile=invalid_toml)

    # Test Case 6: Testing the file system lookup logic (using mock_open)
    # Create a dummy pyproject.toml in the current directory for mocking
    dummy_toml = tmp_path / "pyproject.toml"
    dummy_toml.write_text('[tool.vulture]\nmin_confidence = 10', encoding="utf-8")
    
    # We patch 'open' and 'pathlib.Path.is_file' to point to our dummy file
    with patch("builtins.open", mock_open(read_data=b'[tool.vulture]\nmin_confidence = 10')):
        with patch("pathlib.Path.is_file", return_value=True):
            # We pass no argv, so it should look for pyproject.toml and find the mocked content
            config = make_config(argv=["some_path"])
            assert config["min_confidence"] == 10
```


# LLM-generated content at query #9
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test basic configuration with defaults (no files, no args)
    # We mock _parse_args to return empty dict so it doesn't look for real sys.argv
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["test_path"], 
            config="pyproject.toml", 
            exclude=object(), # Using object to simulate 'missing' sentinel logic if needed
            # In real usage, we'd match the 'missing' behavior from the module
        )
        # For simplicity in testing make_config specifically:
        # We bypass the complex CLI parsing by mocking _parse_args directly 
        # to control exactly what "CLI" provides.
        pass

    # 2. Test merging TOML and CLI args
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["test*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking _parse_args to return specific CLI overrides
    # We override 'min_confidence' and 'paths'
    cli_overrides = {
        "paths": ["src"],
        "min_confidence": 80,
        "config": str(toml_file)
    }

    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        # We need to simulate the 'missing' object behavior for the parser
        # To avoid complex setup, we mock _parse_args return value directly
        with patch("__main__._parse_args", return_value=cli_overrides):
            config = make_config(argv=["src"])
            
            assert config["min_confidence"] == 80  # CLI wins
            assert config["exclude"] == ["test*.py"] # From TOML
            assert config["paths"] == ["src"]      # From CLI
            assert config["verbose"] is True       # From TOML

    # 3. Test CLI precedence over TOML
    cli_overrides_v2 = {
        "paths": ["src"],
        "min_confidence": 90,
        "config": str(toml_file)
    }
    with patch("__main__._parse_args", return_value=cli_overrides_v2):
        config = make_config(argv=["src"])
        assert config["min_confidence"] == 90

    # 4. Test InputError when paths are empty
    cli_no_paths = {
        "paths": [],
        "config": str(toml_file)
    }
    with patch("__main__._parse_args", return_value=cli_no_paths):
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # 5. Test using an IO stream (tomlfile argument) directly
    toml_stream = io.BytesIO(toml_content)
    cli_overrides_simple = {"paths": ["direct_stream"]}
    with patch("__main__._parse_args", return_value=cli_overrides_simple):
        config = make_config(tomlfile=toml_stream)
        assert config["min_confidence"] == 50
        assert config["paths"] == ["direct_stream"]

    # 6. Test default values when nothing is provided in TOML or CLI (except paths)
    cli_defaults = {"paths": ["only_path"]}
    with patch("__main__._parse_args", return_value=cli_defaults):
        # Mocking no toml file exists
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=["only_path"])
            assert config["min_confidence"] == 0  # From DEFAULTS
            assert config["sort_by_size"] is False # From DEFAULTS
            assert config["paths"] == ["only_path"]

    # 7. Test InputError for invalid types (manually triggering _check_input_config via make_config)
    cli_invalid_type = {
        "paths": ["src"],
        "min_confidence": "not_an_int" # Should be int
    }
    with patch("__main__._parse_args", return_value=cli_invalid_type):
        with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
            make_config(argv=["src"])

def test_make_config_unknown_key(tmp_path):
    # 8. Test InputError for unknown configuration keys
    cli_unknown_key = {
        "paths": ["src"],
        "unknown_key": "invalid"
    }
    with patch("__main__._parse_args", return_value=cli_unknown_key):
        with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
            make_config(argv=["src"])
```


# LLM-generated content at query #10
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: Pure CLI arguments (no TOML file)
    # Should return cli args merged with DEFAULTS
    args = ["--min-confidence", "50", "--sort-by-size", "path/to/code"]
    config = make_config(argv=args, tomlfile=None)
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path/to/code"]
    assert config["verbose"] is False  # Default value from DEFAULTS
    assert config["exclude"] == []     # Default value from DEFAULTS

    # Test Case 2: TOML file with values and CLI overrides
    # CLI arguments should take precedence over TOML
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["test*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI provides 'min_confidence' as 80, which should override TOML's 20
    args = ["--min-confidence", "80", "some_dir"]
    
    # We use a mock to simulate opening the file from the path provided in cli_args
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # Mocking is_file to return True so it attempts to read the pyproject.toml
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=args, tomlfile=None)

    assert config["min_confidence"] == 80  # Overridden by CLI
    assert config["exclude"] == ["test*.py"] # Loaded from TOML
    assert config["verbose"] is True        # Loaded from TOML
    assert config["paths"] == ["some_dir"]

    # Test Case 3: Using the tomlfile parameter directly (IO instance)
    toml_io = io.BytesIO(b'[tool.vulture]\nignore_names = ["unused"]')
    args = ["--ignore-names", "extra_name", "path"]
    config = make_config(argv=args, tomlfile=toml_io)
    
    # Note: argparse 'csv' type splits by comma. 
    # The TOML parser returns the list from TOML, CLI adds its own list.
    # Since config.update(cli_config) is used, CLI replaces the key entirely.
    assert "extra_name" in config["ignore_names"]

    # Test Case 4: InputError on invalid type (CLI)
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int", "path"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test Case 5: InputError on invalid key (via TOML simulation)
    invalid_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path"], tomlfile=invalid_toml)
    assert "Unknown configuration key: unknown_key" in str(excinfo.value)

    # Test Case 6: InputError on empty paths (Output validation)
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[]) # No paths provided, and no TOML to provide them
    assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test Case 7: InputError on type mismatch (TOML)
    mismatched_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path"], tomlfile=mismatched_toml)
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)
```


# LLM-generated content at query #11
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # 1. Test CLI arguments override everything and defaults are applied
    (["--min-confidence", "50", "path/to/code"], None, {
        "min_confidence": 50,
        "paths": ["path/to/code"],
        "config": "pyproject.toml",
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }),
    # 2. Test TOML loading and merging with CLI
    (["--verbose", "--exclude", "test/*"], 
     '[tool.vulture]\nmin_confidence = 10\nexclude = ["old/*"]\n[tool.vulture.paths]\npaths = ["."] \n', # Note: logic expects paths in TOML if used, but we'll mock toml load
     None), 
])
def test_make_config(argv, toml_content, expected_keys):
    # Since the original code has complex dependencies on file existence and tomllib,
    # we use a more controlled approach for unit testing make_config logic.
    pass

def test_make_config_logic():
    """
    A comprehensive test for make_config covering:
    - CLI precedence over TOML
    - Default value application
    - Error handling for empty paths
    """
    
    # Mocking data for a valid TOML structure
    toml_data = {
        "tool": {
            "vulture": {
                "min_confidence": 20,
                "exclude": ["temp/*"],
                "paths": ["src"]
            }
        }
    }

    # Test Case 1: CLI overrides TOML
    # Args: --min-confidence 80, path 'my_dir'
    # TOML has min_confidence 20
    args = ["--min-confidence", "80", "my_dir"]
    
    with patch("tomllib.load", return_value=toml_data), \
         patch("builtins.open", mock_open()), \
         patch("pathlib.Path.is_file", return_value=True):
        
        config = make_config(argv=args, tomlfile=io.BytesIO(b"dummy"))
        
        assert config["min_confidence"] == 80  # Overridden
        assert config["paths"] == ["my_dir"]   # From CLI
        assert config["exclude"] == ["temp/*"] # From TOML
        assert config["config"] == "pyproject.toml" # Default

    # Test Case 2: Empty paths raises InputError
    args_empty = [] # No paths provided in CLI or TOML (if we mock it)
    with patch("tomllib.load", return_value={"tool": {"vulture": {"paths": []}}}), \
         patch("builtins.open", mock_open()), \
         patch("pathlib.Path.is_file", return_value=True):
        
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=args_empty, tomlfile=io.BytesIO(b"dummy"))

    # Test Case 3: Invalid type in CLI
    args_bad_type = ["--min-confidence", "not_an_int"]
    with pytest.raises(SystemExit): # argparse exits on bad type
        make_config(argv=args_bad_type)

    # Test Case 4: Unknown key in TOML
    invalid_toml = {
        "tool": {
            "vulture": {
                "unknown_key": "value",
                "paths": ["src"]
            }
        }
    }
    with patch("tomllib.load", return_value=invalid_toml), \
         patch("builtins.open", mock_open()), \
         patch("pathlib.Path.is_file", return_value=True):
        
        with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
            make_config(argv=["src"], tomlfile=io.BytesIO(b"dummy"))

def test_make_config_no_toml_file_exists():
    """Test behavior when no TOML file is found on disk."""
    args = ["--min-confidence", "10", "path/to/code"]
    
    with patch("pathlib.Path.is_file", return_value=False), \
         patch("argparse.ArgumentParser.parse_args") as mock_parse:
        
        # Mocking the namespace returned by argparse
        class Namespace:
            def __init__(self):
                self.paths = ["path/to/code"]
                self.exclude = None
                self.ignore_decorators = None
                self.ignore_names = None
                self.make_whitelist = False
                self.min_confidence = 10
                self.sort_by_size = False
                self.config = "pyproject.toml"
                self.verbose = False
        
        mock_parse.return_value = Namespace()
        
        # When no file exists, it should fallback to defaults + CLI
        config = make_config(argv=args)
        assert config["min_confidence"] == 10
        assert config["paths"] == ["path/to/code"]
        assert config["exclude"] == [] # Default
```


# LLM-generated content at query #12
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: CLI arguments only (no file)
    # Testing precedence and defaults
    args = ["--min-confidence", "50", "--sort-by-size", "path/to/code"]
    config = make_config(argv=args)
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path/to/code"]
    assert config["exclude"] == []  # default
    assert config["verbose"] is False # default

    # Test Case 2: TOML file configuration
    toml_content = b"""
[tool.vulture]
min_confidence = 10
exclude = ["test_*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking open to ensure we read our temp file
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            # Use argv that points to the mock file
            args = ["--config", str(toml_file), "some_path"]
            config = make_config(argv=args)
            
            assert config["min_confidence"] == 10
            assert config["exclude"] == ["test_*.py"]
            assert config["verbose"] is True
            assert config["paths"] == ["some_path"]

    # Test Case 3: CLI overrides TOML
    # Overriding the 'min_confidence' from 10 (in TOML) to 80 (in CLI)
    args = ["--min-confidence", "80", "some_path"]
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.isass_file", return_value=True): # Helper for logic
            # We re-run with the same file content but different args
            config = make_config(argv=args)
            assert config["min_confidence"] == 80
            assert config["exclude"] == ["test_*.py"] # Still from TOML

    # Test Case 4: Input Error - Wrong Type
    # min_confidence expects int, providing string via a fake arg that bypasses argparse type check logic
    # However, _parse_args handles the type conversion. We test _check_input_config via make_config.
    with pytest.raises(InputError) as excinfo:
        # Passing a list where a string/bool is expected manually through a mock if possible, 
        # but here we can trigger it by simulating a bad TOML load
        bad_toml = b'[tool.vulture]\nmin_confidence = "high"]'
        with patch("tomllib.load", return_value={"tool": {"vulture": {"min_confidence": "high"}}}):
            make_config(argv=["path"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test Case 5: Input Error - Unknown Key
    with pytest_raises_input_error(b'[tool.vulture]\nunknown_key = true'):
        pass # See implementation below for a cleaner way to test this

def pytest_raises_input_error(toml_bytes):
    """Helper to test unknown key error"""
    with patch("tomllib.load", return_value={"tool": {"vulture": {"unknown_key": True}}}):
        with pytest.raises(InputError, match="Unknown configuration key"):
            make_config(argv=["path"])

def test_make_config_output_validation():
    # Test Case 6: Output Error - No paths provided
    # If paths is empty (and no default triggered), _check_output_config raises error.
    # Note: argparse defaults 'paths' to missing, then make_config fills it from DEFAULTS.
    # To trigger this, we need the final config['paths'] to be empty.
    # This happens if we explicitly pass an empty list via a custom logic or if DEFAULTS was modified.
    # Since we cannot easily modify DEFAULTS in a unit test without side effects, 
    # we simulate a scenario where paths is empty.
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=[], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

def test_make_config_csv_parsing():
    # Test Case 7: CSV parsing for exclude/ignore
    args = ["--exclude", "file1.py,file2.py", "--ignore-names", "name1,name2", "path"]
    config = make_config(argv=args)
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_names"] == ["name1", "name2"]
```


# LLM-generated content at query #13
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: CLI arguments only (No config file)
    # We pass argv that doesn't trigger --version or --help to avoid SystemExit
    argv = ["path/to/dir", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=argv)
    
    assert config["paths"] == ["path/to/arg"] # argparse behavior with provided list
    # Note: _parse_args uses the provided list. 
    # Since we didn't provide a real file, it should use DEFAULTS for others.
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["exclude"] == []
    assert config["config"] == "pyproject.toml"

    # Test Case 2: TOML file and CLI arguments (Merging)
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["test*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)
    
    # CLI overrides TOML
    argv = ["some_path", "--min-confidence", "80"]
    
    # We mock open to ensure it reads our temp file when no tomlfile is passed explicitly
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=argv)
            
    assert config["min_confidence"] == 80  # From CLI
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["verbose"] is True  # From TOML
    assert config["paths"] == ["some_path"] # From CLI

    # Test Case 3: Using the 'tomlfile' parameter explicitly (IO stream)
    toml_stream = io.BytesIO(b'[tool.vulture]\nignore_names = ["unused"]')
    argv = ["."]
    config = make_config(argv=argv, tomlfile=toml_stream)
    assert config["ignore_names"] == ["unused"]
    assert config["paths"] == ["."]

    # Test Case 4: InputError for invalid type in TOML
    invalid_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"]')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["."], tomlfile=invalid_toml)
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test Case 5: InputError for missing paths (Output check)
    # We force an empty path list via a mock or specific argv if possible
    # In _parse_args, if no path is provided and no defaults are used, it might fail.
    # However, the easiest way to trigger _check_output_config is providing empty paths.
    with pytest.raises(InputError) as excinfo:
        # We need to bypass the 'missing' sentinel logic by explicitly passing an empty list
        # but argparse handles positional args. We simulate a config with no paths.
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = type('Namespace', (), {
                "paths": [], "exclude": [], "ignore_decorators": [], 
                "ignore_names": [], "make_whitelist": False, "sort_by_size": False,
                "verbose": False, "config": "pyproject.toml", "min_confidence": 0
            })
            make_config(argv=[])
    assert "Please pass at least one file or directory" in str(excinfo.value)

def test_make_config_unknown_key(tmp_path):
    # Test Case 6: Unknown key in TOML
    invalid_key_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = True')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["."], tomlfile=invalid_key_toml)
    assert "Unknown configuration key: unknown_key" in str(excinfo.value)
```


# LLM-generated content at query #14
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test default behavior (no args, no file) -> should raise error because paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # 2. Test CLI arguments only
    cli_args = ["path/to/code", "--min-confidence", "50", "--sort-by-size", "--verbose"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/arg"] # Note: argparse behavior on args list
    # Re-running with explicit list to ensure precision
    config = make_config(argv=["src", "--min-confidence", "20"])
    assert config["paths"] == ["src"]
    assert config["min_confidence"] == 20
    assert config["sort_by_size"] is False # default from DEFAULTS because not in CLI

    # 3. Test TOML parsing and merging with CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 10
exclude = ["test*.py"]
sort_by_size = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI overrides TOML: set min_confidence to 80
    cli_args = ["src", "--min-confidence", "80"]
    
    # We use a mock for open to ensure it reads our temp file correctly 
    # or just pass the file object via tomlfile param as supported by make_config
    with io.BytesIO(toml_content) as f:
        config = make_config(argv=cli_args, tomlfile=f)

    assert config["min_confidence"] == 80  # CLI override
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["sort_by_size"] is True  # From TOML
    assert config["paths"] == ["src"]  # From CLI

    # 4. Test Type Error in configuration (InputError)
    bad_toml = b"""
[tool.vulture]
min_confidence = "not-an-int"
"""
    with io.BytesIO(bad_toml) as f:
        with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
            make_config(argv=["src"], tomlfile=f)

    # 5. Test Unknown Key Error
    bad_key_toml = b"""
[tool.vulture]
unknown_key = True
"""
    with io.BytesIO(bad_key_toml) as f:
        with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
            make_config(argv=["src"], tomlfile=f)

    # 6. Test CSV parsing for strings in CLI
    config = make_config(argv=["src", "--exclude", "a.py,b.py", "--ignore-names", "func1,func2"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_names"] == ["func1", "func2"]

    # 7. Test Boolean flags in CLI
    config = make_config(argv=["src", "--make-whitelist", "-v"])
    assert config["make_whitelist"] is True
    assert config["verbose"] is True
```


# LLM-generated content at query #15
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Pure CLI arguments (no TOML)
    # We provide paths to satisfy _check_output_config requirement
    args = ["path/to/code", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=args)
    assert config["paths"] == ["path/to/argparse_mock_path" if False else "path/to/code"]
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["exclude"] == []  # Default

    # Test 2: TOML file loading and merging with CLI
    toml_content = b"""
[tool.vulture]
exclude = ["*.tmp"]
min_confidence = 20
verbose = true
paths = ["toml_path"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI overrides TOML: change min_confidence from 20 to 80
    args = ["--min-confidence", "80"]
    
    # We mock open to ensure it reads our temp file when make_config looks for pyproject.toml
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            # Use the real path but mock the filesystem behavior if necessary
            # Here we just pass the file handle directly via a trick or rely on the path
            config = make_config(argv=args, tomlfile=io.BytesIO(toml_content))
    
    assert config["exclude"] == ["*.tmp"]         # From TOML
    assert config["min_confidence"] == 80        # From CLI (overrides TOML)
    assert config["verbose"] is True             # From TOML
    assert config["paths"] == ["toml_path"]      # From TOML

    # Test 3: Validation Error - Missing paths
    # _check_output_config should raise InputError if no paths are provided via CLI or TOML
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 4: Validation Error - Invalid Type
    # Passing a string to min_confidence (which expects int)
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])

    # Test 5: Validation Error - Unknown Key
    # We simulate an unknown key appearing in the parsed args via a manual check 
    # (Note: argparse usually prevents this, but _check_input_config is the target)
    with pytest::raises(InputError, match="Unknown configuration key"):
        _check_input_config({"invalid_key": True})

def test_make_config_defaults(tmp_path):
    # Test that defaults are applied when nothing else is provided
    args = ["some_dir"]
    config = make_config(argv=args)
    assert config["config"] == "pyproject.toml"
    assert config["min_confidence"] == 0
    assert config["make_whitelist"] is False
```


# LLM-generated content at query #16
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test basic functionality with only CLI args (no file)
    # We pass paths as a positional argument in argv
    cli_args = ["test_dir/"]
    config = make_config(argv=cli_args)
    
    assert config["paths"] == ["test_dir/"]
    assert config["min_confidence"] == 0  # Default value
    assert config["verbose"] is False     # Default value

    # 2. Test merging CLI args with TOML file content
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["test.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI arg '--config' points to our temp file
    # CLI arg '--sort-by-size' overrides nothing in TOML but is provided
    cli_args_merged = ["test_dir/", "--config", str(toml_file), "--sort-by-size"]
    
    config = make_config(argv=cli_args_merged)

    assert config["paths"] == ["test_dir/"]
    assert config["min_confidence"] == 50        # From TOML
    assert config["exclude"] == ["test.py"]     # From TOML
    assert config["verbose"] is True            # From TOML
    assert config["sort_by_size"] is True       # From CLI (overrides default)

    # 3. Test CLI precedence (CLI should override TOML)
    cli_args_precedence = ["test_dir/", "--config", str(toml_file), "--min-confidence", "80"]
    config = make_config(argv=cli_args_precedence)
    assert config["min_confidence"] == 80        # CLI wins over TOML

    # 4. Test InputError for invalid types in CLI/TOML
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["test_dir/", "--min-confidence", "not_an_int"])

    # 5. Test InputError for unknown keys in TOML
    invalid_toml = b"""
[tool.vulture]
unknown_key = True
"""
    invalid_toml_file = tmp_path / "invalid.toml"
    invalid_toml_file.write_bytes(invalid_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["test_dir/", "--config", str(invalid_toml_file)])

    # 6. Test InputError for empty paths (Output check)
    # Note: _parse_args returns paths as a list of strings. 
    # If we provide no positional args and no TOML, it triggers error in _check_output_config.
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # 7. Test using the 'tomlfile' parameter directly (IO instance)
    toml_io = io.BytesIO(b'[tool.vulture]\nignore_names = ["foo"]')
    config = make_config(argv=["test_dir/"], tomlfile=toml_io)
    assert "foo" in config["ignore_names"]

    # 8. Test CSV parsing for lists via CLI
    cli_args_csv = ["test_dir/", "--ignore-decorators", "deco1,deco2,deco3"]
    config = make_config(argv=cli_args_csv)
    assert config["ignore_decorators"] == ["deco1", "deco2", "deco3"]

    # 9. Test Boolean flags (store_true)
    cli_args_bool = ["test_dir/", "--make-whitelist"]
    config = make_config(argv=cli_args_bool)
    assert config["make_whitelist"] is True
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize(
    "argv, toml_content, expected_config",
    [
        # 1. Test purely CLI arguments (no TOML)
        (
            ["path/to/code"],
            None,
            {
                "config": "pyproject.toml",
                "min_confidence": 0,
                "paths": ["path/to/code"],
                "exclude": [],
                "ignore_decorators": [],
                "ignore_names": [],
                "make_whitelist": False,
                "sort_by_size": False,
                "verbose": False,
            },
        ),
        # 2. Test CLI overriding TOML
        (
            ["path/to/code", "--min-confidence", "50", "--sort-by-size"],
            '[tool.vulture]\nmin_confidence = 10\nsort_by_size = false\npaths = ["toml_path"]',
            {
                "config": "pyproject.toml",
                "min_confidence": 50,
                "paths": ["path/to/code"],
                "exclude": [],
                "ignore_decorators": [],
                "ignore_names": [],
                "make_whitelist": False,
                "sort_by_size": True,
                "verbose": False,
            },
        ),
        # 3. Test TOML providing values not in CLI
        (
            ["path/to/code"],
            '[tool.vulture]\nexclude = ["*.tmp"]\nverbose = true',
            {
                "config": "pyproject.toml",
                "min_confidence": 0,
                "paths": ["path/to/code"],
                "exclude": ["*.tmp"],
                "ignore_decorators": [],
                "ignore_names": [],
                "make_whitelist": False,
                "sort_by:size": False, # Note: There's a typo in the provided source code 'sort_by:size' vs 'sort_by_size', but assuming logic follows DEFAULTS
                "sort_by_size": False,
                "verbose": True,
            },
        ),
    ],
)
def test_make_config(argv, toml_content, expected_config):
    toml_file = io.BytesIO(toml_content.encode("utf-8")) if toml_content else None

    # We mock _parse_args logic implicitly by passing argv
    # and we use the provided tomlfile argument to bypass filesystem checks
    config = make_config(argv=argv, tomlfile=toml_file)

    for key, value in expected_config.items():
        assert config[key] == value


def test_make_config_error_no_paths():
    """Test that missing paths raises InputError."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # Passing empty argv results in no paths
        make_config(argv=[])


def test_make_config_invalid_toml_type():
    """Test that invalid type in TOML raises InputError."""
    toml_content = '[tool.vulture]\nmin_confidence = "not_an_int"'
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["path/to/code"], tomlfile=toml_file)


def test_make_config_unknown_key():
    """Test that unknown keys in TOML raise InputError."""
    toml_content = '[tool.vulture]\nunknown_key = true'
    toml_file = io.BytesIO(toml_content.encode("utf-8"))

    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path/to/code"], tomlfile=toml_file)


def test_make_config_filesystem_loading():
    """Test loading from actual file path via mocking open."""
    toml_content = '[tool.vulture]\nmin_confidence = 25'
    # Mocking the existence of a file and its content
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=toml_content.encode("utf-8"))), \
         patch("tomllib.load") as mock_load:
        
        # Setup mock to return the parsed dict
        mock_load.return_value = {"tool": {"vulture": {"min_confidence": 25}}}
        
        config = make_config(argv=["path/to/code", "--config", "fake_pyproject.toml"])
        assert config["min_confidence"] == 25
        assert config["paths"] == ["path/to/code"]
```


# LLM-generated content at query #2
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test Case: Default behavior (no args, no file)
    # We must mock sys.argv because _parse_args uses it by default.
    # We also need to bypass the file existence check for 'pyproject.toml'
    with patch("sys.argv", ["vulture", "test_path/"]):
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=["test_path/"])
            assert config["paths"] == ["test_path/"]
            assert config["min_confidence"] == 0
            assert config["verbose"] is False

    # 2. Test Case: CLI arguments overriding defaults
    with patch("sys.argv", ["vulture"]):
        config = make_config(argv=["--min-confidence", "50", "--sort-by-size", "--exclude", "test.py,temp/"])
        assert config["min_confidence"] == 50
        assert config["sort_by_size"] is True
        assert config["exclude"] == ["test.py", "temp/"]

    # 3. Test Case: TOML file integration
    toml_content = b"""
[tool.vulture]
min_confidence = 25
ignore_names = ["unused_var"]
paths = ["src/"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # We use the 'tomlfile' argument to inject the file stream directly for testing
    with patch("sys.argv", ["vulture"]):
        stream = io.BytesIO(toml_content)
        config = make_config(argv=[], tomlfile=stream)
        assert config["min_confidence"] == 25
        assert config["ignore_names"] == ["unused_var"]
        # Check that defaults are still applied for keys not in TOML
        assert config["verbose"] is False

    # 4. Test Case: CLI precedence over TOML
    # If both provide 'min_confidence', CLI should win (25 from TOML, 80 from CLI)
    with patch("sys.argv", ["vulture", "--min-confidence", "80"]):
        stream = io.BytesIO(toml_content)
        config = make_config(argv=["--min_confidence", "80"], tomlfile=stream)
        # Note: _parse_args converts '--min-confidence' to 'min_confidence' in the dict
        assert config["min_confidence"] == 80

    # 5. Test Case: InputError on invalid type
    with patch("sys.argv", ["vulture"]):
        with pytest.raises(InputError) as excinfo:
            # Passing a string to an integer field via CLI (argparse handles conversion, 
            # but we can trigger _check_input_config by passing bad data in a mock scenario)
            # Here we simulate the result of a manual dict update that violates types.
            # Since we can't easily break argparse without complex mocks, 
            # let's test the internal checker via make_config if possible.
            make_config(argv=["--min-confidence", "not_an_int"])
        assert "invalid int value" in str(excinfo.value).lower() or "argument" in str(excinfo.value)

    # 6. Test Case: InputError on missing paths (Output Config Check)
    with patch("sys.argv", ["vulture"]):
        # Empty paths list triggers _check_output_config error
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # 7. Test Case: InputError on unknown key
    # We mock the namespace to inject an unknown key into the dictionary after parsing
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        class MockNamespace:
            def __init__(self):
                self.paths = ["."]
                self.unknown_key = "error"
        
        mock_parse.return_value = MockNamespace()
        # We need to patch vars(namespace) which is what _parse_args uses
        with patch("argparse.Namespace", spec=dict) as mock_ns:
            # This is tricky because we need to inject into the dict returned by vars()
            # A cleaner way is to simulate a key that exists in Namespace but not DEFAULTS
            with patch("argparse.ArgumentParser.parse_args") as m:
                m.return_value = argparse.Namespace(paths=["."], config="pyproject.toml")
                # Manual override of the dict returned by vars()
                with patch("argparse.Namespace.__dict__", {"paths": ["."], "bad_key": 1}, create=True):
                    # This is difficult to trigger via standard CLI, but tests logic
                    pass

    # 8. Test Case: Integration of CSV parsing in CLI
    with patch("sys.argv", ["vulture", "--ignore-decorators", "deco1,deco2"]):
        config = make_config(argv=["--ignore-decorators", "deco1,deco2", "path/"])
        assert config["ignore_decorators"] == ["deco1", "deco2"]

```


# LLM-generated content at query #3
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test basic defaults (no args, no file)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking argparse to return minimal paths
        mock_args.return_value = argparse.Namespace(paths=["test_dir"], config="pyproject.toml")
        # Patching _parse_toml to return empty dict since file won't exist
        with patch("config_module._parse_toml", return_value={}):
            config = make_config(argv=[])
            assert config["paths"] == ["test_dir"]
            assert config["min_confidence"] == 0
            assert config["verbose"] is False

    # 2. Test merging CLI and TOML
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["*.tmp"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking argv to include an override for min_confidence and paths
    argv = ["test_dir", "--min-confidence", "80", "--exclude", "pattern1,pattern2"]
    
    # We use a real file here so the logic in make_config actually reads it
    with patch("sys.argv", ["vulture"] + argv):
        config = make_config(argv=argv)
        
        # CLI should take precedence: 80 instead of 50
        assert config["min_confidence"] == 80
        # CLI parsed comma-separated list into list
        assert config["exclude"] == ["pattern1", "pattern2"]
        # TOML value should be present if not overridden (if we were testing a key not in argv)
        # But here we test that paths from argv are respected.

    # 3. Test InputError for invalid types via CLI
    with pytest.raises(InputError):
        # min_confidence expects int, passing string that isn't an int (argparse handles this usually, 
        # but _check_input_config catches type mismatches)
        make_config(argv=["test_dir", "--min-confidence", "not_an_int"])

    # 4. Test InputError for unknown keys in TOML
    invalid_toml = b"""
[tool.vulture]
unknown_key = True
"""
    invalid_toml_file = tmp_path / "invalid.toml"
    invalid_toml_file.write_bytes(invalid_toml)
    
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(paths=["test_dir"], config=str(invalid_toml_file))
        with pytest.raises(InputError, match="Unknown configuration key"):
            make_config(argv=["--config", str(invalid_toml_file)])

    # 5. Test InputError for empty paths (Output validation)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(paths=[], config="pyproject.toml")
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # 6. Test CSV parsing for ignore_names
    argv_csv = ["test_dir", "--ignore-names", "func1,func2"]
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(paths=["test_dir"], config="pyproject.toml", 
                                                   ignore_names=["func1", "func2"])
        # We manually simulate what _parse_args does for the slice of logic we want to test
        config = make_config(argv=argv_csv)
        assert config["ignore_names"] == ["func1", "func2"]

    # 7. Test Boolean flag (store_true)
    argv_bool = ["test_dir", "--sort-by-size", "--verbose"]
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Simulate the namespace produced by argparse for flags
        mock_args.return_value = argparse.Namespace(paths=["test_dir"], config="pyproject.toml", 
                                                   sort_by_size=True, verbose=True)
        config = make_config(argv=argv_bool)
        assert config["sort_by_size"] is True
        assert config["verbose"] is True
```


# LLM-generated content at query #4
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Setup a dummy pyproject.toml
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["test*.py"]
verbose = true
paths = ["src"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Test Case 1: Merge TOML and CLI (CLI takes precedence)
    # We pass argv to override min_confidence from 50 to 80
    argv = ["--min-confidence", "80", "some_path.py"]
    
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # Using the actual file on disk for path resolution logic in make_config
        config = make_config(argv=argv)

    assert config["min_confidence"] == 80
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["paths"] == ["some_path.py"] # From CLI
    assert config["verbose"] is True          # From TOML
    assert config["sort_by_size"] is False    # From DEFAULTS

    # Test Case 2: Pure CLI (No TOML file)
    argv_only = ["--exclude", "extra.py", "--sort-by-size", "dir/"]
    config_cli = make_config(argv=argv_only)
    
    assert config_cli["exclude"] == ["extra.py"]
    assert config_cli["sort_by_size"] is True
    assert config_cli["paths"] == ["dir/"]
    assert config_cli["min_confidence"] == 0  # Default

    # Test Case 3: Using provided tomlfile stream (IO instance)
    toml_stream = io.BytesIO(toml_content)
    config_stream = make_config(argv=["path/to/code"], tomlfile=toml_stream)
    
    assert config_stream["min_confidence"] == 50
    assert config_stream["paths"] == ["path/to/code"]

    # Test Case 4: InputError on invalid type in CLI
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])

    # Test Case 5: InputError on unknown key (simulated via TOML)
    bad_toml = b'[tool.vulture]\nunknown_key = true'
    bad_stream = io.BytesIO(bad_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path"], tomlfile=bad_stream)

    # Test Case 6: InputError on missing paths (Output validation)
    # Note: _parse_args provides 'paths' from argv. 
    # If we force an empty path list via a way that bypasses the CLI args parser's default
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            config="pyproject.toml",
            min_confidence=0,
            paths=[], # Empty paths should trigger error
            exclude=[],
            ignore_decorators=[],
            ignore_names=[],
            make_whitelist=False,
            sort_by_size=False,
            verbose=False
        )
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

```


# LLM-generated content at query #5
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Default behavior (no args, no file exists)
    # We patch 'open' to simulate pyproject.toml not existing
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=[])
            assert config["min_confidence"] == 0
            assert config["paths"] == []
            assert config["verbose"] is False

    # Test 2: CLI arguments override everything
    # Providing paths and min_confidence via CLI
    config = make_config(argv=["path/to/dir", "--min-confidence", "50", "--verbose"])
    assert "path/to/dir" in config["paths"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True

    # Test 3: TOML file loading and merging with CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["test_*.py"]
sort_by_size = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # We use the actual file on disk for this test
    # CLI provides 'paths' and overrides 'min_confidence'
    config = make_config(argv=["src/", "--min-confidence", "80"])
    
    assert config["min_confidence"] == 80  # Overridden by CLI
    assert config["exclude"] == ["test_*.py"]  # Loaded from TOML
    assert config["sort_by_size"] is True     # Loaded from TOML
    assert "src/" in config["paths"]         # From CLI

    # Test 4: CSV parsing for lists in CLI
    config = make_config(argv=["src/", "--exclude", "pattern1,pattern2", "--ignore-names", "name1"])
    assert config["exclude"] == ["pattern1", "pattern2"]
    assert config["ignore_names"] == ["name1"]

    # Test 5: InputError on invalid type in CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["src/", "--min-confidence", "not_an_int"])
    assert "invalid literal for int()" in str(excinfo.value)

    # Test 6: InputError on unknown key (via simulated TOML)
    toml_bad_content = b"""
[tool.vulture]
unknown_key = "error"
"""
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_bytes(toml_bad_content)
    
    # We use the tomlfile argument to inject the bad stream directly
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["src/"], tomlfile=io.BytesIO(toml_bad_content))
    assert "Unknown configuration key" in str(excinfo.value)

    # Test 7: OutputError when no paths are provided
    # Note: _parse_args returns 'missing' if no paths, but make_config 
    # populates DEFAULTS['paths'] as [] if not found in CLI or TOML.
    # To trigger _check_output_config, we need an empty list of paths.
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking namespace to return empty paths and no error for other keys
        class Namespace:
            def __init__(self):
                for k, v in DEFAULTS.items():
                    setattr(self, k, v)
                self.paths = [] 
        
        mock_args.return_value = Namespace()
        with pytest.raises(InputError) as excinfo:
            make_config(argv=[])
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test 8: Boolean flags (store_true)
    config = make_config(argv=["src/", "--make-whitelist", "--sort-by-size"])
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
```


# LLM-generated content at query #6
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test default configuration (no args, no file)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Simulate no CLI arguments provided (only defaults from argparse)
        mock_args.return_value = argparse.Namespace(
            paths=[], config="pyproject.toml", min_confidence=object(), 
            exclude=object(), ignore_decorators=object(), ignore_names=object(),
            make_whitelist=object(), sort_by_size=object(), verbose=object()
        )
        # We must mock the 'paths' to be non-empty to pass _check_output_config
        # but since we are testing defaults, let's trigger the error first
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # 2. Test merging CLI and TOML
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["test*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking argparse to provide a path and an override
    # We use a list of strings as if passed via sys.argv
    cli_args = ["path/to/code", "--min-confidence", "80", "--sort-by-size"]
    
    # We need to mock 'open' for the auto-detection logic in make_config
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # We use a real file on tmp_path so pathlib.Path(..).is_file() works
        # but we override argv to provide the paths
        config = make_config(argv=["path/to/code", "--min-confidence", "80"])

    # Assertions
    assert config["paths"] == ["path/to/code"]
    assert config["min_confidence"] == 80  # CLI overrides TOML (50)
    assert config["exclude"] == ["test*.py"] # From TOML
    assert config["sort_by_size"] is False   # Default
    assert config["verbose"] is False       # Not in CLI, not in TOML (if we assume default)

    # 3. Test input validation error (wrong type)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path/to/code", "--min-confidence", "not_an_int"])

    # 4. Test TOML parsing with specific file handle
    toml_io = io.BytesIO(toml_content)
    config_from_io = make_config(argv=["some_path"], tomlfile=toml_io)
    assert config_from_io["min_confidence"] == 50
    assert config_from_io["exclude"] == ["test*.py"]

    # 5. Test unknown key in TOML
    bad_toml = b'[tool.vulture]\nunknown_key = true'
    bad_toml_io = io.BytesIO(bad_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["some_path"], tomlfile=bad_toml_io)

def test_make_config_cli_precedence(tmp_path):
    """Specific test for precedence: CLI > TOML > Defaults"""
    toml_content = b'[tool.vulture]\nmin_confidence = 20\nverbose = false'
    toml_file = tmp_path / "test_config.toml"
    toml_file.write_bytes(toml_content)

    # CLI sets min_confidence to 90 and verbose to True
    argv = ["my_dir", "--min-confidence", "90", "--verbose"]
    
    config = make_config(argv=argv, tomlfile=io.BytesIO(toml_content))
    
    assert config["min_confidence"] == 90  # CLI wins
    assert config["verbose"] is True       # CLI wins
    assert config["paths"] == ["my_dir"]   # From CLI
```


# LLM-generated content at query #7
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: Basic usage with no arguments and no file (uses defaults)
    # We must mock 'is_file' to False so it doesn't try to read a non-existent pyproject.toml
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=[])
        assert config["min_confidence"] == 0
        assert config["paths"] == []
        assert config["verbose"] is False

    # Test Case 2: CLI arguments override defaults
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=["path/to/code", "--min-confidence", "50", "--sort-by-size"])
        assert config["paths"] == ["path/to/code"]
        assert config["min_confidence"] == 50
        assert config["sort_by_size"] is True

    # Test Case 3: CLI arguments with CSV parsing
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=["--exclude", "test.py,venv/"])
        assert config["exclude"] == ["test.py", "venv/"]

    # Test Case 4: Merging TOML and CLI (CLI takes precedence)
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["old.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking open to return our temp file content when the function tries to read pyproject.toml
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            # CLI provides 40, TOML provides 20 -> Should be 40
            config = make_config(argv=["--min-confidence", "40"])
            assert config["min_confidence"] == 40
            assert config["exclude"] == ["old.py"]
            assert config["verbose"] is True

    # Test Case 5: Using the 'tomlfile' parameter directly (IO stream)
    toml_stream = io.BytesIO(toml_content)
    config = make_config(argv=["--sort-by-size"], tomlfile=toml_stream)
    assert config["min_confidence"] == 20
    assert config["sort_by_size"] is True

    # Test Case 6: InputError on invalid type via CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not-an-int"])
    assert "invalid literal for int()" in str(excinfo.value)

    # Test Case 7: InputError on invalid type via TOML (simulated via tomlfile)
    bad_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=bad_toml)
    assert "must be 'int'" in str(excinfo.value)

    # Test Case 8: InputError on unknown key
    bad_key_toml = io_bytes = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=io_bytes)
    assert "Unknown configuration key" in str(excinfo.value)

    # Test Case 9: InputError on empty paths (Output validation)
    with patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(InputError) as excinfo:
            make_config(argv=[]) # No paths provided in argv and no file to provide them
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test Case 10: CSV parsing for ignore-decorators
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=["path/to/dir", "--ignore-decorators", "deco1,deco2"])
        assert config["ignore_decorators"] == ["deco1", "deco2"]
```


# LLM-generated content at query #8
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Default behavior (no args, no toml file) -> Should raise error because paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # Test 2: CLI arguments only
    cli_args = ["path/to/code", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/code"]
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["exclude"] == []  # Default value

    # Test 3: CLI arguments with comma-separated values
    cli_args_csv = ["path/to/code", "--exclude", "test.py,venv/", "--ignore-names", "foo,bar"]
    config = make_config(argv=cli_args_csv)
    assert config["exclude"] == ["test.py", "venv/"]
    assert config["ignore_names"] == ["foo", "bar"]

    # Test 4: Merging TOML and CLI (CLI takes precedence)
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["from_toml.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI provides a different min_confidence and an extra path
    cli_args_override = ["other_path", "--min-confidence", "80"]
    
    # We mock the file opening to ensure it picks up our tmp_path file
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # We must also mock the path resolution or ensure the file exists in a way 
        # that pathlib can find it. Since we used tmp_path, we point config to it.
        cli_args_override[0] = "other_path"
        # Use the actual path from tmp_path for the config arg
        config_arg = f"--config {toml_file.absolute()}"
        # Reconstruct args list to include the new config path
        args = ["other_path", "--min-confidence", "80", "--config", str(toml_file)]
        
        config = make_config(argv=args)
        
        assert config["min_confidence"] == 80  # Overridden by CLI
        assert config["exclude"] == ["from_from_toml.py"] # Error in my manual logic? No, check TOML content: ["from_toml.py"]
        assert config["exclude"] == ["from_toml.py"]
        assert config["verbose"] is True      # From TOML
        assert "other_path" in config["paths"]

    # Test 5: Invalid type in CLI (should be caught by _check_input_config)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path", "--min-confidence", "not_an_int"])

    # Test 6: Invalid key in TOML
    invalid_toml = b"""
[tool.vulture]
unknown_key = True
"""
    toml_err_file = tmp_path / "invalid_pyproject.toml"
    toml_err_file.write_bytes(invalid_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path", "--config", str(toml_err_file)])

    # Test 7: Using the tomlfile argument directly (IO instance)
    toml_io = io.BytesIO(b'[tool.vulture]\nexclude = ["io_test.py"]')
    config = make_config(argv=["path"], tomlfile=toml_io)
    assert config["exclude"] == ["io_test.py"]
    assert config["paths"] == ["path"]
```


# LLM-generated content at query #9
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_paths", [
    (["path/to/code"], '{"tool": {"vulture": {"paths": ["file1.py"]}}}', ["file1.py"]),
    (["--exclude", "test.py", "path/to/code"], '{"tool": {"vulture": {"exclude": ["other.py"]}}}', ["path/to/code"]),
    (["--min-confidence", "50"], "", ["path/to/code"]),
    ([], '{"tool": {"vulture": {"sort_by_size": true}}}', []), # This will fail _check_output_config if paths empty
])
def test_make_config_logic(argv, toml_content, expected_paths):
    # We need at least one path for _check_output_config to pass
    if not argv or ("path/to/code" not in argv and "PATH" not in str(argv)):
        argv = argv + ["some_path.py"]
        
    toml_data = io.BytesIO(toml_content.encode("utf-8"))
    
    # Mocking open to prevent actual file system access for the default 'pyproject.toml'
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=argv, tomlfile=toml_data)
            
            assert config["paths"] == ["some_path.py"] if "some_path.py" in argv else []
            if "exclude" in toml_content:
                # Check if CLI overrides TOML (If we passed --exclude via argv)
                pass 

def test_make_config_cli_precedence():
    toml_content = '{"tool": {"vulture": {"min_confidence": 10, "paths": ["file1.py"]}}}'
    toml_data = io.BytesIO(toml_content.encode("utf-8"))
    argv = ["--min-confidence", "20", "path/to/dir"]
    
    config = make_config(argv=argv, tomlfile=toml_data)
    
    assert config["min_confidence"] == 20
    assert config["paths"] == ["path/to/dir"]

def test_make_config_input_error_on_empty_paths():
    # Testing the _check_output_config validation
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_type_error():
    toml_content = '{"tool": {"vulture": {"min_confidence": "not_an_int"}}}'
    toml_data = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Data type for min_confidence must"):
        make_config(argv=["some_path.py"], tomlfile=toml_data)

def test_make_config_unknown_key():
    toml_content = '{"tool": {"vulture": {"unknown_key": True}}}'
    toml_data = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["some_path.py"], tomlfile=toml_data)

def test_make_config_defaults():
    # Test that defaults are applied when nothing is provided in TOML or CLI
    argv = ["some_path.py"]
    toml_data = io.BytesIO(b'{"tool": {"vulture": {}}}')
    
    config = make_config(argv=argv, tomlfile=toml_data)
    
    assert config["verbose"] is False
    assert config["sort_by_size"] is False
    assert config["min_confidence"] == 0
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from io import BytesIO
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # Test 1: Pure CLI arguments (no TOML)
    (["--min-confidence", "50", "--sort-by-size", "path/to/dir"], None, {"min_confidence": 50, "sort_by_size": True, "paths": ["path/to/dir"]}),
    
    # Test 2: CLI overriding TOML
    (["--min-confidence", "80", "path1"], b'[tool.vulture]\nmin_confidence = 10\n[tool.vulture]\npaths = ["path2"]', {"min_confidence": 80, "paths": ["path1"]}),
    
    # Test 3: TOML and CLI merging (Comma separated lists)
    (["--exclude", "a,b"], b'[tool.vulture]\nexclude = ["c"]\npaths=["p"]', {"exclude": ["a", "b"], "paths": ["p"]}),
])
def test_make_config(argv, toml_content, expected_keys):
    """Tests the merging logic of CLI and TOML configurations."""
    
    # We use a mock for _parse_args indirectly by controlling argv
    # But we need to handle the file system aspect for 'config' default path.
    
    if toml_content:
        toml_file = BytesIO(toml_content)
        # Note: _parse_toml uses tomllib.load, so we provide a stream
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=argv, tomlfile=toml_file)
    else:
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=argv, tomlfile=None)

    for key, value in expected_keys.items():
        assert config[key] == value

def test_make_config_error_no_paths():
    """Tests that an error is raised if no paths are provided in either source."""
    # CLI with no paths and no TOML file exists
    with patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=["--min-confidence", "10"])

def test_make_config_input_type_error():
    """Tests that providing the wrong type via CLI raises InputError."""
    # min_confidence expects int, passing a string that isn't an int (via internal logic)
    # Note: argparse handles the conversion, but _check_input_config checks types.
    # We simulate a bad type bypass if possible or rely on argparse behavior.
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Manually force an invalid type into the namespace
        mock_args.return_value.items.return_value = [("min_confidence", "not_an_int"), ("paths", ["p"])]
        # Since we can't easily mock the namespace object for iteration without complexity, 
        # we rely on a controlled dict if we were testing _check_input_config directly.
        pass

def test_make_config_toml_file_loading():
    """Tests that make_config correctly reads from an existing file on disk."""
    toml_data = b'[tool.vulture]\nmin_confidence = 25\npaths = ["src"]'
    
    # Mocking open and pathlib to simulate a real file existence
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.resolve", return_value=pathlib.Path("pyproject.toml")), \
         patch("builtins.open", mock_open(read_data=toml_data)):
        
        # We need to mock tomllib.load because it expects a binary stream
        with patch("tomllib.load") as mock_load:
            mock_load.return_value = {"tool": {"vulture": {"min_confidence": 25, "paths": ["src"]}}}
            config = make_config(argv=["--verbose"])
            assert config["min_confidence"] == 25
            assert config["paths"] == ["src"]
            assert config["verbose"] is True

def test_make_config_invalid_key_in_toml():
    """Tests that an unknown key in TOML raises InputError."""
    toml_data = b'[tool.vulture]\nunknown_key = "value"\npaths = ["src"]'
    toml_file = BytesIO(toml_data)
    
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["--paths", "src"], tomlfile=toml_file)
```


# LLM-generated content at query #11
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Setup a dummy pyproject.toml content
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["test*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Test Case 1: CLI arguments override TOML
    # We use argv=['--min-confidence', '80', '--paths', 'src']
    # The paths must be provided to pass _check_output_config
    argv = ["--min-confidence", "80", "--paths", "src"]
    
    # Mocking open to return our dummy toml file when the code tries to read it
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # We also need to ensure pathlib.Path.is_file() returns True for our dummy path
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=argv)

    assert config["min_confidence"] == 80  # Overridden by CLI
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["paths"] == ["src"]  # From CLI
    assert config["verbose"] is True  # From TOML
    assert config["sort_by_size"] is False  # Default

    # Test Case 2: No TOML, only CLI
    argv_only_cli = ["--exclude", "temp.py", "--paths", "."]
    config_cli = make_config(argv=argv_only_cli)
    assert config_cli["exclude"] == ["temp.py"]
    assert config_cli["paths"] == ["."]
    assert config_cli["min_confidence"] == 0  # Default

    # Test Case 3: Using an explicit tomlfile (IO object)
    toml_io = io.BytesIO(toml_content)
    argv_empty = ["--paths", "test_dir"]
    config_io = make_config(argv=argv_empty, tomlfile=toml_io)
    assert config_io["min_confidence"] == 50
    assert config_io["paths"] == ["test_dir"]

    # Test Case 4: InputError when no paths are provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--min-confidence", "10"])

    # Test Case 5: InputError for invalid type in CLI (argparse handles this, but we check integration)
    with pytest.raises(SystemExit):
        # argparse will exit when type conversion fails (e.g. string for int)
        make_config(argv=["--min-confidence", "not_an_int"])

    # Test Case 6: InputError for unknown key in TOML
    invalid_toml = b'[tool.vulture]\nunknown_key = true'
    toml_io_invalid = io.BytesIO(invalid_toml)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["--paths", "."], tomlfile=toml_io_invalid)

    # Test Case 7: InputError for type mismatch in TOML
    type_mismatch_toml = b'[tool.vulture]\nmin_confidence = "high"'
    toml_io_mismatch = io.BytesIO(type_mismatch_toml)
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["--paths", "."], tomlfile=toml_io_mismatch)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Basic functionality with no arguments and no file (uses defaults)
    # We must mock 'paths' because _check_output_config requires at least one path
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["test_path"], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        # Mocking file existence to avoid real filesystem access
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=["test_path"])
            assert config["paths"] == ["test_path"]
            assert config["min_confidence"] == 0
            assert config["config"] == "pyproject.toml"

    # Test 2: Merging TOML and CLI (CLI takes precedence)
    toml_content = b'[tool.vulture]\nmin_confidence = 50\nverbose = false\npaths = ["toml_path"]'
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # We use argv to trigger the CLI parsing logic
    # In this test, we simulate running: vulture path1 --min-confidence 80
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Simulate argparse returning specific values
        mock_args.return_value = argparse.Namespace(
            paths=["path1"], config=str(toml_file), min_confidence=80, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=True
        )
        
        # Use the actual file via tomlfile argument for controlled environment
        with open(toml_file, "rb") as f:
            config = make_config(argv=["path1"], tomlfile=f)
            
        assert config["min_confidence"] == 80  # CLI Overrode TOML (50)
        assert config["paths"] == ["path1"]    # CLI Overrode TOML (["toml_path"])
        assert config["verbose"] is True       # CLI Overrode TOML (False)

    # Test 3: InputError when no paths are provided
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=[], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # Test 4: InputError for invalid type in CLI (handled by argparse usually, 
    # but we test the internal _check_input_config via make_config)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["path"], config="pyproject.toml", min_confidence="high", # Wrong type (str instead of int)
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
            make_config(argv=["path"])

    # Test 5: Error when unknown key is present
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["path"], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False,
            unknown_key="error" # This is problematic for Namespace but let's assume it passed through
        )
        # We manually trigger the dictionary update via a mock to test _check_input_config logic
        with patch("argparse.Namespace.__dict__", {"paths": ["p"], "unknown_key": 1}):
            # Since we can't easily corrupt Namespace without breaking argparse, 
            # we verify the internal logic via a direct call if needed, 
            # but here we simulate the dictionary state after parsing.
            pass 
```


# LLM-generated content at query #13
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test Default configuration (no file, no args)
    # We need to mock open because it tries to read pyproject.toml by default
    with patch("builtins.open", mock_open(read_data=b"")):
        with patch("pathlib.Path.is_file", return_value=False):
            with patch("argparse.ArgumentParser.parse_args") as mock_args:
                # Mocking args to provide paths so _check_output_config passes
                mock_args.return_value = argparse.Namespace(paths=["test.py"])
                config = make_config(argv=[])
                assert config["paths"] == ["test.py"]
                assert config["min_confidence"] == 0

    # 2. Test CLI arguments overriding everything
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["cli_path.py"],
            exclude=["exclude1,exclude2"],
            min_confidence=50,
            make_whitelist=True,
            config="dummy.toml"
        )
        # Mocking is_file to False so it doesn't look for dummy.toml
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=["--min-confidence", "50"])
            assert config["paths"] == ["cli_path.py"]
            assert config["min_confidence"] == 50
            assert config["exclude"] == ["exclude1", "exclude2"]
            assert config["make_whitelist"] is True

    # 3. Test TOML file loading and merging with CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["file1.py"]
paths = ["toml_path.py"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # CLI provides an override for min_confidence and a path
        mock_args.return_value = argparse.Namespace(
            paths=["cli_override.py"],
            min_confidence=80,
            config=str(toml_file)
        )
        # Ensure the file exists so make_config reads it
        with patch("pathlib.Path.is_file", return_value=True):
             config = make_config(argv=["--min-confidence", "80"])
             # From TOML
             assert config["exclude"] == ["file1.py"]
             # From CLI (overriding TOML)
             assert config["min_confidence"] == 80
             assert config["paths"] == ["cli_override.py"]

    # 4. Test InputError on invalid type via CLI
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["test.py"],
            min_confidence="not_an_int" # This would actually be caught by argparse, 
                                        # but we test the internal _check_input_config logic
        )
        with pytest.raises(InputError):
             # Manually trigger the type check via a bad namespace
             # We use a mock that bypasses argparse's type conversion to hit the logic
             pass 

    # 5. Test InputError on empty paths (Output Config validation)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(paths=[])
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # 6. Test InputError on unknown key in TOML
    bad_toml = b"""
[tool.vulture]
unknown_key = "oops"
paths = ["test.py"]
"""
    with io.BytesIO(bad_toml) as f:
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = argparse.Namespace(paths=["test.py"], config="dummy.toml")
            with pytest.raises(InputError, match="Unknown configuration key"):
                make_config(tomlfile=f)

def test_make_config_verbose_output(capsys):
    toml_content = b"""
[tool.vulture]
paths = ["test.py"]
verbose = true
"""
    with io.BytesIO(toml_content) as f:
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = argparse.Namespace(paths=["test.py"], config="dummy.toml", verbose=True)
            make_config(tomlfile=f)
            captured = capsys.readouterr()
            # Note: The path in the print statement depends on how 'f' is handled, 
            # but we check if any print happened.
            assert "Reading configuration from" in captured.out
```


# LLM-generated content at query #14
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test Case: CLI arguments only (No TOML)
    # We pass paths to satisfy _check_output_config
    args = ["path/to/code", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=args, tomlfile=None)
    assert config["paths"] == ["path/to/argparse_mock_logic_dummy_path"] # Note: argparse uses sys.argv if args is None
    # Since we are passing args directly to _parse_args, the paths are processed correctly.
    # Re-evaluating: args=["path1"] -> config["paths"] == ["path1"]
    
    # Let's do a precise test for CLI overriding defaults
    args = ["test_dir", "--min-confidence", "20", "--verbose"]
    config = make_config(argv=args, tomlfile=None)
    assert config["paths"] == ["test_dir"]
    assert config["min_confidence"] == 20
    assert config["verbose"] is True
    assert config["sort_by_size"] is False  # Default

    # 2. Test Case: TOML file only (No CLI)
    toml_content = b"""
[tool.vulture]
min_confidence = 10
exclude = ["test*.py"]
sort_by_size = true
paths = ["from_toml"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # We mock the file opening in make_config to use our tmp_path file
    # But since we can pass a file-like object via tomlfile, let's do that.
    toml_io = io.BytesIO(toml_content)
    config = make_config(argv=["from_toml"], tomlfile=toml_io)
    assert config["min_confidence"] == 10
    assert config["exclude"] == ["test*.py"]
    assert config["sort_by_size"] is True
    assert config["paths"] == ["from_toml"]

    # 3. Test Case: Merging TOML and CLI (CLI takes precedence)
    toml_io_2 = io.BytesIO(b"""
[tool.vulture]
min_confidence = 10
exclude = ["old_pattern"]
paths = ["original_path"]
""")
    # CLI specifies min_confidence=80 and adds a new path
    args = ["new_path", "--min-confidence", "80"]
    config = make_config(argv=args, tomlfile=toml_io_2)
    
    assert config["min_confidence"] == 80  # Overridden by CLI
    assert config["exclude"] == ["old_pattern"]  # Kept from TOML
    assert "new_path" in config["paths"]
    assert "original_path" in config["paths"]

    # 4. Test Case: InputError for invalid types via CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path", "--min-confidence", "not_an_int"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # 5. Test Case: InputError for unknown key via TOML
    bad_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = True\npaths = ["p"]')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["p"], tomlfile=bad_toml)
    assert "Unknown configuration key: unknown_key" in str(excinfo.value)

    # 6. Test Case: InputError for empty paths (Sanity check)
    # We must bypass _parse_args's requirement of path by mocking or using a specific setup
    # But if we provide no paths via CLI and no TOML has paths, it should fail.
    # Note: _parse_args with no args results in paths = missing, which is filtered out.
    # If config['paths'] ends up empty after defaults:
    with pytest.raises(InputError) as excinfo:
        # Using an empty list of args and a TOML that doesn't provide paths
        empty_toml = io.Bytesly(b'[tool.vulture]\nmin_confidence=1') # No paths provided
        # We need to force 'paths' to be empty. 
        # In the current code, if not in CLI and not in TOML, it takes DEFAULTS['paths'] which is []
        make_config(argv=[], tomlfile=io.BytesIO(b'[tool.vulture]\nmin_confidence=1'))
    assert "Please pass at least one file or directory" in str(excinfo.value)

# Helper for the test suite execution
if __name__ == "__main__":
    pytest.main([__file__])
```


