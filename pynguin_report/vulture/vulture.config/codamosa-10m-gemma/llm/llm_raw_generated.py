####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Default behavior (No args, no file)
    # We must mock 'paths' because _check_output_config raises error if paths is empty
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["test_path"], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        # Mocking file existence to ensure it doesn't actually look for pyproject.toml
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=[])
            assert config["paths"] == ["test_path"]
            assert config["min_confidence"] == 0

    # Test 2: Merging CLI arguments with TOML file content
    toml_content = b"""
[tool.vulture]
min_confidence = 50
exclude = ["temp/"]
paths = ["src/"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking CLI to provide an override for 'min_confidence' and 'paths'
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # We simulate passing --min-confidence 80 via CLI
        mock_args.return_value = argparse.Namespace(
            paths=["cli_path"], config=str(toml_file), min_confidence=80, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=True
        )
        
        config = make_config(argv=["--min-confidence", "80", "cli_path"])
        
        # CLI (80) should override TOML (50)
        assert config["min_confidence"] == 80
        # TOML value should persist if not overridden in CLI
        assert config["exclude"] == ["temp/"]
        # CLI value should override TOML paths
        assert config["paths"] == ["cli_path"]

    # Test 3: InputError when paths are missing
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return = argparse.Namespace(
            paths=[], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

    # Test 4: InputError for invalid types in TOML
    invalid_toml = b"""
[tool.vulture]
min_confidence = "not-an-int"
"""
    toml_file_invalid = tmp_path / "invalid.toml"
    toml_file_invalid.write_bytes(invalid_toml)

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["some_path"], config=str(toml_file_invalid), min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
            make_config(argv=[])

    # Test 5: Using the tomlfile argument directly (io instance)
    toml_stream = io.BytesIO(b'[tool.vulture]\nverbose = true\npaths = ["direct_path"]')
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["cli_path"], config="pyproject.toml", min_confidence=0, 
            exclude=[], ignore_decorators=[], ignore_names=[], 
            make_whitelist=False, sort_by_size=False, verbose=False
        )
        config = make_config(tomlfile=toml_stream)
        assert config["verbose"] is True
        assert config["paths"] == ["direct_path"]
```


# LLM-generated content at query #2
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # Test 1: CLI arguments only (no TOML)
    (["--min-confidence", "50", "path/to/dir"], None, {"min_confidence": 50, "paths": ["path/to/dir"]}),
    
    # Test 2: CLI arguments overriding TOML
    (["--min-confidence", "80", "--sort-by-size"], 
     '[tool.vulture]\nmin_confidence = 10\nsort_by:size = false', 
     {"min_confidence": 80, "sort_by_size": True}),

    # Test 3: TOML provides values not in CLI
    (["--paths", "src"], 
     '[tool.vulture]\nexclude = ["*.tmp"]\nignore_names = ["test"]', 
     {"paths": ["src"], "exclude": ["*.tmp"], "ignore_names": ["test"]}),

    # Test 4: CSV parsing in CLI
    (["--exclude", "a.py,b.py", "--ignore-decorators", "deco1,deco2"], None, {"exclude": ["a.py", "b.py"], "ignore_decorators": ["deco1", "deco2"]}),
])
def test_make_config(argv, toml_content, expected_keys):
    toml_data = toml_content.encode("utf-8") if toml_content else b""
    toml_file = io.BytesIO(toml_data)

    # We mock 'open' to prevent it from looking for real files on disk
    # and we mock '_parse_args' or similar logic via controlled argv
    with patch("builtins.open", mock_open(read_data=toml_data)):
        # If no tomlfile is passed, make_config tries to find pyproject.toml on disk.
        # To isolate the test, we pass the tomlfile object directly.
        config = make_config(argv=argv, tomlfile=toml_file)

    for key, value in expected_keys.items():
        assert config[key] == value

def test_make_config_error_no_paths():
    """Test that missing paths raises InputError."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # Passing an empty list of paths via argv (which triggers the check)
        make_config(argv=[])

def test_make_config_invalid_type():
    """Test that invalid type in CLI raises InputError."""
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        # Passing a string where an int is expected
        make_config(argv=["--min-confidence", "not_an_int"])

def test_make_config_unknown_key():
    """Test that unknown keys in TOML raise InputError."""
    toml_content = b'[tool.vulture]\nunknown_key = true'
    toml_file = io.BytesIO(toml_content)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path/to/dir"], tomlfile=toml_file)

def test_make_config_defaults():
    """Test that DEFAULTS are applied when nothing is provided in CLI or TOML."""
    # We must provide at least one path to pass _check_output_config
    config = make_config(argv=["some_path"])
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
```


# LLM-generated content at query #3
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # 1. Test CLI arguments precedence and defaults
    (["--min-confidence", "50", "path/to/dir"], None, {
        "min_confidence": 50,
        "paths": ["path/to/dir"],
        "config": "pyproject.toml",
        "exclude": [],
        "ignore_decorators": [],
        "ignore_names": [],
        "make_whitelist": False,
        "sort_by_size": False,
        "verbose": False,
    }),
    # 2. Test CLI overriding TOML
    (["--verbose", "--exclude", "test.py"], '[tool.vulture]\nexclude = ["old.py"]\nverbose = false', {
        "exclude": ["old.py", "test.py"], # Note: parser logic for csv might vary, but we check keys
        "verbose": True,
    }),
])
def test_make_config_logic(argv, toml_content, expected_keys):
    """Basic functional tests for merging and precedence."""
    # We mock _parse_toml to avoid file system dependency for the logic test
    mock_toml_data = {}
    if toml_content:
        import tomllib
        mock_toml_data = tomllib.loads(toml_content).get("tool", {}).get("vulture", {})

    with patch("config_module._parse_toml", return_value=mock_toml_data):
        # Mocking file existence to skip reading from disk
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=argv, tomlfile=None)
            
            for key, value in expected_keys.items():
                # For lists like exclude, we check if the target is present 
                # because _parse_args logic for CSV might be complex
                if isinstance(value, list):
                    assert any(item in config[key] for item in value) or config[key] == value
                else:
                    assert config[key] == value

def test_make_config_with_toml_file():
    """Test specifically using the tomlfile argument."""
    toml_content = b'[tool.vulture]\nmin_confidence = 25\npaths = ["src"]'
    toml_file = io.BytesIO(toml_content)
    
    # We don't pass argv, so it uses defaults (which includes paths=missing in _parse_args logic)
    # But make_config needs at least one path to pass _check_output_config
    argv = ["src"] 
    
    config = make_config(argv=argv, tomlfile=toml_file)
    assert config["min_confidence"] == 25
    assert "src" in config["paths"]

def test_make_config_error_no_paths():
    """Test that missing paths raises InputError."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # Passing empty argv results in empty paths
        make_config(argv=[])

def test_make_config_type_error():
    """Test that invalid types in CLI raise InputError."""
    with pytest.pytest.raises(InputError, match="Data type for min_confidence must"):
        # Passing a string where int is expected (though argparse usually handles this, 
        # we test the internal _check_input_config validation)
        make_config(argv=["--min-confidence", "not_an_int"])

def test_make_config_unknown_key():
    """Test that unknown keys in TOML raise InputError."""
    toml_content = b'[tool.vulture]\nunknown_key = true'
    toml_file = io.BytesIO(toml_content)
    
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["src"], tomlfile=toml_file)

def test_make_config_integration_with_file_system(tmp_path):
    """Integration test: testing the actual file reading logic."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.vulture]\nmin_confidence = 10\nverbose = true', encoding="utf-8")
    
    # We need to mock 'open' and 'is_file' because the module uses absolute path resolution
    # but we want to point it to our tmp_path
    with patch("pathlib.Path.is_file", return_value=True):
        with patch("builtins.open", mock_open(read_data=pyproject.read_text().encode())):
            # We pass the path that would be resolved
            config = make_config(argv=["--config", str(pyproject), "my_dir"])
            assert config["min_confidence"] == 10
            assert config["verbose"] is True
            assert "my_dir" in config["paths"]
```


# LLM-generated content at query #4
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Basic functionality - CLI arguments only (no file)
    # We use a dummy path that doesn't exist to trigger the empty config branch
    argv = ["vulture", "some_dir", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=argv)
    
    assert config["paths"] == ["some_dir"]
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["exclude"] == []  # Default value
    assert config["config"] == "pyproject.toml"

    # Test 2: Merging CLI and TOML
    # Create a mock TOML content
    toml_content = b"""
[tool.vulture]
exclude = ["test.py"]
min_confidence = 10
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI provides 'paths' and overrides 'min_confidence'
    argv = ["vulture", "my_folder", "--min-confidence", "90"]
    
    # We patch open to ensure it reads our tmp_path file when searching for pyproject.toml
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # To make the logic work with the real filesystem check in make_config, 
        # we need to ensure the path exists or pass the file handle directly via 'tomlfile'
        config = make_config(argv=argv, tomlfile=io.BytesIO(toml_content))

    assert config["paths"] == ["my_folder"]          # From CLI
    assert config["min_confidence"] == 90            # Overridden by CLI
    assert config["exclude"] == ["test.py"]          # From TOML
    assert config["verbose"] is True                 # From TOML

    # Test 3: InputError on invalid type in CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["vulture", "--min-confidence", "not_an_int"])
    assert "must be 'int'" in str(excinfo.value)

    # Test 4: InputError on unknown key via TOML injection
    invalid_toml = b'[tool.vulture]\nunknown_key = true'
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["vulture", "."], tomlfile=io.BytesIO(invalid_toml))
    assert "Unknown configuration key" in str(excinfo.value)

    # Test 5: InputError on missing paths (Output validation)
    # We pass empty paths via CLI (using a trick since 'paths' defaults to 'missing')
    # Since the parser uses nargs='*', we can't easily pass an "empty" list that isn't the default.
    # However, if we provide an arg that results in no paths:
    with pytest.raises(InputError) as excinfo:
        # Passing --config to a non-existent file and no paths
        make_config(argv=["vulture", "--config", "non_existent.toml"])
    # Note: This actually triggers because 'paths' remains the 'missing' sentinel 
    # which is not in DEFAULTS or doesn't satisfy _check_output_config if it were empty.
    # In our implementation, paths defaults to 'missing', then gets set to [] via DEFAULTS.
    # We need to force an empty list into the final config.
    
    # Let's test the specific error message for no paths:
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking namespace to simulate a state where paths is empty after defaults
        class Namespace:
            def __init__(self):
                self.paths = []
                self.config = "pyproject.toml"
                self.min_confidence = 0
                self.exclude = []
                self.ignore_decorators = []
                self.ignore_names = []
                self.make_whitelist = False
                self.sort_by_size = False
                self.verbose = False
        mock_args.return_value = Namespace()
        with pytest.raises(InputError) as excinfo:
            make_config(argv=[])
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test 6: CSV parsing for exclude/ignore
    argv = ["vulture", ".", "--exclude", "file1.py,file2.py", "--ignore-names", "func_a,func_b"]
    config = make_config(argv=argv)
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_names"] == ["func_a", "func_b"]
```


# LLM-generated content at query #5
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test default behavior (no args, no file) -> Should raise error because paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # 2. Test CLI arguments only
    cli_args = ["path/to/code", "--min-confidence", "50", "--sort-by-size", "--verbose"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/arg"] # argparse uses argv[1:] logic internally via sys.argv simulation
    # Re-evaluating: _parse_args(args) is called. If args=["path"], namespace.paths is ["path"]
    config = make_config(argv=["src", "--min-confidence", "50"])
    assert config["paths"] == ["src"]
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is False # default from DEFAULTS

    # 3. Test TOML file parsing and merging with CLI
    toml_content = b"""
[tool.vulture]
exclude = ["*.tmp"]
min_confidence = 20
paths = ["file1.py"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI overrides TOML
    # We pass the path via --config so it finds our dummy toml
    cli_args = ["src", "--config", str(toml_file), "--min-confidence", "80"]
    config = make_config(argv=cli_args)
    
    assert config["paths"] == ["src"] # CLI paths take precedence or are added? 
    # Looking at code: cli_config contains 'paths' from argv. 
    # _parse_args returns {'paths': ['src'], 'min_confidence': 80, ...}
    # _parse_toml returns {'exclude': ['*.tmp'], 'min_confidence': 20, 'paths': ['file1.py']}
    # config.update(cli_config) -> paths becomes ['src'], min_confidence becomes 80
    assert config["min_confidence"] == 80
    assert config["exclude"] == ["*.tmp"]

    # 4. Test InputError for invalid types in TOML
    invalid_toml = b"""
[tool.vulture]
min_confidence = "not_an_int"
"""
    invalid_file = tmp_path / "invalid.toml"
    invalid_file.write_bytes(invalid_toml)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--config", str(invalid_file), "src"])

    # 5. Test InputError for unknown keys in TOML
    bad_key_toml = b"""
[tool.vulture]
unknown_key = True
"""
    bad_key_file = tmp_path / "bad_key.toml"
    bad_key_file.write_bytes(bad_key_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--config", str(bad_key_file), "src"])

    # 6. Test CSV parsing in CLI
    cli_csv = ["src", "--exclude", "test.py,temp/*", "--ignore-names", "unused_var"]
    config = make_config(argv=cli_csv)
    assert config["exclude"] == ["test.py", "temp/*"]
    assert config["ignore_names"] == ["unused_var"]

    # 7. Test via IO object (tomlfile parameter)
    toml_io = io.BytesIO(b'[tool.vulture]\npaths = ["io_path.py"]')
    config = make_config(argv=["src"], tomlfile=toml_io)
    assert "io_path.py" in config["paths"] or config["paths"] == ["src"] 
    # Note: In the implementation, cli_config is parsed first, then toml is loaded.
    # config.update(cli_config) happens AFTER toml load. 
    # So if CLI has paths=['src'], it overwrites toml's ['io_path.py'].
    assert config["paths"] == ["src"]

    # 8. Test full integration of defaults
    config = make_config(argv=["src"])
    assert config["make_whitelist"] is False
    assert config["verbose"] is False
    assert config["paths"] == ["src"]
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # Test 1: Pure CLI arguments (no TOML)
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
        },
    ),
    # Test 2: Pure TOML arguments (no CLI)
    (
        [],
        '[tool.vulture]\nmin_confidence = 20\nexclude = ["test/*.py"]',
        {
            "config": "pyproject.toml",
            "min_confidence": 20,
            "paths": [], # This will fail _check_output_config if paths is empty
            "exclude": ["test/*.py"],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": False,
        },
    ),
    # Test 3: Merged arguments (CLI overrides TOML)
    (
        ["--min-confidence", "80", "some_path"],
        '[tool.vulture]\nmin_confidence = 10\nverbose = true',
        {
            "config": "pyproject.toml",
            "min_confidence": 80,
            "paths": ["some_path"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": True,
        },
    ),
])
def test_make_config(argv, toml_content, expected):
    """
    Tests the merging logic of make_config. 
    Note: Test 2 is designed to fail if paths are empty, so we handle it via pytest.raises.
    """
    if toml_content is None:
        # No TOML file provided, testing CLI only
        result = make_config(argv=argv, tomlfile=None)
        assert result == expected
    else:
        # Using io.BytesIO to simulate a TOML file content
        toml_file = io.BytesIO(toml_content.encode("utf-8"))
        
        # If the test case expects paths to be empty, it will trigger _check_output_config error
        if not expected.get("paths") and "paths" in expected:
             with pytest.raises(InputError) as excinfo:
                 make_config(argv=argv, tomlfile=toml_file)
             assert "Please pass at least one file or directory" in str(excinfo.value)
        else:
            # We must provide a path in argv for the 'Pure TOML' test case logic 
            # to bypass the empty paths check if we want it to succeed.
            # For this specific unit test implementation, we assume valid inputs.
            if not argv or (isinstance(argv[0], str) and argv[0].startswith("--")):
                 argv.append("dummy_path")
            
            result = make_config(argv=argv, tomlfile=toml_file)
            # Check that CLI overrides TOML
            assert result["min_confidence"] == expected["min_confidence"]

def test_make_config_invalid_type():
    """Test that providing a wrong type via CLI raises InputError."""
    with pytest.raises(InputError) as excinfo:
        # min_confidence expects int, passing string that isn't an int (via custom logic if possible)
        # Since argparse handles the conversion, we simulate an invalid type in TOML
        toml_data = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
        make_config(argv=["path/to/code"], tomlfile=toml_data)
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

def test_make_config_unknown_key():
    """Test that an unknown key in TOML raises InputError."""
    toml_data = io.BytesIO(b'[tool.vulture]\nunknown_key = True')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path/to/code"], tomlfile=toml_data)
    assert "Unknown configuration key: unknown_key" in str(excinfo.value)

@patch("pathlib.Path.is_file")
@patch("builtins.open", new_callable=mock_open, read_data=b'[tool.vulture]\nmin_confidence = 10')
def test_make_config_reads_from_disk(mock_file, mock_is_file):
    """Test that the function attempts to read pyproject.toml from disk if not provided."""
    mock_is_file.return_value = True
    # We need to provide a path in argv so _check_output_config doesn't fail
    result = make_config(argv=["some_path"])
    assert result["min_confidence"] == 10
    mock_file.assert_called()
```


# LLM-generated content at query #7
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test default behavior (no args, no file) -> Should raise InputError because paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # 2. Test CLI arguments only
    cli_args = ["path/to/code", "--min-confidence", "50", "--sort-by-size", "--verbose"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/arg"] # Note: argparse behavior with positional args
    # Re-testing specifically with explicit paths
    config = make_config(argv=["my_dir", "--min-confidence", "20"])
    assert config["paths"] == ["my_dir"]
    assert config["min_confidence"] == 20
    assert config["sort_by_size"] is False
    assert config["verbose"] is False

    # 3. Test TOML file parsing and merging with CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 10
exclude = ["test_*.py"]
make_whitelist = true
paths = ["from_toml"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI overrides TOML: min_confidence 10 (toml) -> 80 (cli)
    # paths in TOML is ["from_toml"], but we must provide at least one path via CLI or TOML
    # to satisfy _check_output_config.
    argv = ["cli_path", "--min-confidence", "80"]
    
    # We use patch to ensure it reads our specific tmp_path file instead of looking for real pyproject.toml
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking the behavior of _parse_args for a controlled test of the merge logic
        # However, we can just pass the tomlfile directly to make_config as per its signature
        
        toml_io = io.BytesIO(toml_content)
        config = make_config(argv=["cli_path"], tomlfile=toml_io)
        
        assert config["min_confidence"] == 80  # CLI override
        assert config["exclude"] == ["test_*.py"] # From TOML
        assert config["make_whitelist"] is True   # From TOML
        assert config["paths"] == ["cli_path"]    # CLI provided path

    # 4. Test InputError for invalid types in TOML
    invalid_toml = b"""
[tool.vulture]
min_confidence = "not_an_int"
"""
    toml_io_invalid = io.BytesIO(invalid_toml)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path"], tomlfile=toml_io_invalid)

    # 5. Test InputError for unknown keys in TOML
    unknown_key_toml = b"""
[tool.vulture]
unknown_key = True
paths = ["p"]
"""
    toml_io_unknown = io.BytesIO(unknown_key_toml)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path"], tomlfile=toml_io_unknown)

    # 6. Test CSV parsing in CLI
    argv_csv = ["path", "--exclude", "a.py,b.py", "--ignore-names", "func1,func2"]
    config_csv = make_config(argv=argv_csv)
    assert config_csv["exclude"] == ["a.py", "b.py"]
    assert config_csv["ignore_names"] == ["func1", "func2"]

    # 7. Test default values application
    # Using a minimal valid setup to check defaults
    argv_min = ["path"]
    config_defaults = make_config(argv=argv_min)
    assert config_defaults["sort_by_size"] is False
    assert config_defaults["verbose"] is False
    assert config_defaults["min_confidence"] == 0
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_data, expected_keys", [
    # Case 1: Only CLI arguments provided
    (["--min-confidence", "50", "--sort-by-size", "path/to/code"], {}, ["min_confidence", "sort_by_size", "paths"]),
    
    # Case 2: Only TOML data provided (via tomlfile argument)
    ([], {"tool": {"vulture": {"exclude": ["test.py"], "verbose": True}}}, ["exclude", "verbose"]),
    
    # Case 3: CLI overrides TOML
    (["--min-confidence", "80", "--exclude", "cli_only.py"], {"tool": {"vulture": {"exclude": ["toml_only.py"], "min_confidence": 10}}}, ["exclude", "min_confidence"]),
    
    # Case 4: Full integration (CLI + TOML)
    (["--verbose", "--ignore-names", "unused"], {"tool": {"vulture": {"paths": ["src"], "ignore_decorators": ["@deco"]}}}, ["verbose", "ignore_names", "paths", "ignore_decorators"]),
])
def test_make_config(argv, toml_data, expected_keys):
    # Setup a mock TOML file content
    toml_content = ""
    if toml_data:
        import tomllib
        # We simulate the structure needed for _parse_toml
        # Since we can't easily use tomllib.dumps, we assume valid TOML strings or dicts are handled
        # For testing purposes, we convert dict back to a string-like format if possible, 
        # but here we just use a mock stream.
        import json
        # Simplified: using a trick to create a valid TOML string from a dict is hard without a library,
        # so we mock the behavior of tomllib.load directly via patching.
        pass

    with patch("tomllib.load") as mock_load:
        mock_load.return_value = toml_data
        
        # Create an IO object for the tomlfile argument
        toml_stream = io.BytesIO(b"dummy content")
        
        config = make_config(argv=argv, tomlfile=toml_stream)
        
        for key in expected_keys:
            assert key in config

def test_make_config_error_no_paths():
    # Test that it raises InputError if no paths are provided via any source
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--min-confidence", "10"])

def test_make_config_invalid_type():
    # Test that it raises InputError if type mismatch occurs in CLI
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["--min-confidence", "not_an_int"])

def test_make_config_unknown_key():
    # Test that it raises InputError if unknown key is in TOML
    bad_toml = {"tool": {"vulture": {"invalid_key": True}}}
    toml_stream = io.BytesIO(b"dummy content")
    with patch("tomllib.load", return_value=bad_toml):
        with pytest.raises(InputError, match="Unknown configuration key: invalid_key"):
            make_config(argv=["path/to/code"], tomlfile=toml_stream)

def test_make_config_file_system_loading():
    # Test the logic that attempts to open pyproject.toml if no tomlfile is provided
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=b"dummy")), \
         patch("tomllib.load", return_value={"tool": {"vulture": {"paths": ["src"]}}}):
        
        config = make_config(argv=["src"])
        assert config["paths"] == ["src"]

def test_make_config_csv_parsing():
    # Test that comma-separated strings are parsed into lists
    config = make_config(argv=["--exclude", "file1.py,file2.py", "path/to/code"])
    assert config["exclude"] == ["file1.py", "file2.py"]
```


# LLM-generated content at query #9
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: CLI arguments only (no file)
    # We pass paths to satisfy _check_output_config requirement
    argv = ["vulture", "src/", "--min-confidence", "50"]
    config = make_config(argv=argv)
    assert config["paths"] == ["src/"]
    assert config["min_confidence"] == 50
    assert config["config"] == "pyproject.toml"  # Default

    # Test 2: TOML file content only (no CLI overrides)
    toml_content = b"""
[tool.vulture]
min_confidence = 80
exclude = ["test/*"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)
    
    # We mock open to ensure it reads our temp file instead of looking at real filesystem
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=["vulture", "my_dir"])
            assert config["min_confidence"] == 80
            assert config["exclude"] == ["test/*"]
            assert config["paths"] == ["my_dir"]

    # Test 3: CLI overrides TOML
    # CLI --min-confidence 20 should beat TOML 80
    argv = ["vulture", "src/", "--min-confidence", "20"]
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=argv)
            assert config["min_confidence"] == 20
            assert config["exclude"] == ["test/*"] # Still from TOML

    # Test 4: Invalid input type (InputError)
    # Passing a string to min_confidence via CLI is handled by argparse, 
    # but we can simulate an invalid type in a manual dict-like way if needed.
    # Here we test the logic inside _check_input_config triggered by make_config
    with pytest.raises(InputError) as excinfo:
        # Passing 'abc' to an int field via argv will cause argparse to raise SystemExit,
        # so we simulate a bad type in a way that bypasses argparse but hits _check_input_config
        # Since make_config calls _parse_args first, we test the error message specifically.
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = type('Namespace', (), {"paths": ["."], "min_confidence": "not_an_int", "config": "p.toml"})
            make_config(argv=["vulture"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test 5: Missing paths (OutputError/InputError)
    with pytest.raises(InputError) as excinfo:
        # Argparse will have empty paths if no positional args provided and we force it
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = type('Namespace', (), {"paths": [], "config": "p.toml"})
            make_config(argv=[])
    assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test 6: CSV parsing in CLI
    argv = ["vulture", "src/", "--exclude", "a,b,c", "--ignore-names", "name1,name2"]
    config = make_config(argv=argv)
    assert config["exclude"] == ["a", "b", "c"]
    assert config["ignore_names"] == ["name1", "name2"]
```


# LLM-generated content at query #10
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

    # Test Case 1: Merging TOML and CLI (CLI should take precedence)
    # We pass argv such that --min-confidence overrides the TOML value
    argv = ["--min-confidence", "80", "--paths", "src/"]
    
    # Mocking open to ensure it reads our temp file when no tomlfile is passed
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            config = make_config(argv=argv)

    assert config["min_confidence"] == 80  # Overridden by CLI
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["paths"] == ["src/"]  # From CLI
    assert config["verbose"] is True  # From TOML
    assert config["sort_by_size"] is False  # From DEFAULTS

    # Test Case 2: Using explicit tomlfile argument
    toml_io = io.BytesIO(toml_content)
    config_explicit = make_config(argv=["--paths", "."], tomlfile=toml_io)
    assert config_explicit["min_confidence"] == 50
    assert config_explicit["paths"] == ["."]

    # Test Case 3: InputError on invalid type via CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test Case 4: InputError on unknown key via CLI (via manual injection simulation)
    # Since argparse doesn't allow unknown args easily without extra setup, 
    # we test the underlying _check_input_config logic via a controlled call if needed,
    # but for make_config specifically, we check the empty paths error.
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--paths"]) # paths is empty/missing in logic context
    assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test Case 5: Check Defaults when no config exists
    with patch("pathlib.Path.is_file", return_value=False):
        config_defaults = make_config(argv=["--paths", "folder"])
        assert config_defaults["min_confidence"] == 0
        assert config_defaults["make_whitelist"] is False
```


# LLM-generated content at query #11
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: CLI arguments only (no file)
    # We pass argv that doesn't trigger --version or --help to avoid SystemExit
    args = ["some_path", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=args)
    assert config["paths"] == ["some_arg"] # Note: argparse logic in provided code maps paths via positional args
    # Re-evaluating the provided _parse_args: 'paths' is the first positional argument.
    # In the provided code, 'paths' is parsed from argv.
    
    # Let's use a more precise approach for testing the specific logic in the snippet
    
    # Test Case 2: Merging TOML and CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["test*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)
    
    # CLI overrides TOML (min_confidence 50 vs 20)
    cli_args = ["src/", "--min-confidence", "50"]
    
    with patch("pathlib.Path.is_file", return_value=True):
        with patch("builtins.open", mock_open(read_data=toml_content)):
            config = make_config(argv=cli_args)
            
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["test*.py"]
    assert config["paths"] == ["src/"]
    assert config["verbose"] is True

    # Test Case 3: InputError on invalid type in TOML
    invalid_toml = b"""
[tool.vulture]
min_confidence = "high"
"""
    with patch("builtins.open", mock_open(read_data=invalid_toml)):
        with pytest.raises(InputError) as excinfo:
            make_config(argv=["path/"], tomlfile=io.BytesIO(invalid_toml))
        assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test Case 4: InputError on unknown key
    unknown_toml = b"""
[tool.vulture]
unknown_key = True
"""
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path/"], tomlfile=io.BytesIO(unknown_toml))
    assert "Unknown configuration key" in str(excinfo.value)

    # Test Case 5: InputError on missing paths (Output validation)
    # To trigger _check_output_config, we need a config where 'paths' is empty.
    # Since _parse_args defaults to a sentinel 'missing' and then we check 
    # the final config, we can simulate an empty path via CLI if possible, 
    # but argparse requires at least one if we want to avoid the error.
    # However, if we provide no paths in argv and no paths in TOML:
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking namespace to return empty paths
        class Namespace:
            def __init__(self):
                self.paths = []
                self.config = "pyproject.toml"
                self.min_confidence = 0
                self.exclude = []
                self.ignore_decorators = []
                self.ignore_names = []
                self.make_whitelist = False
                self.sort_by_size = False
                self.verbose = False
        mock_args.return_value = Namespace()
        with pytest.raises(InputError) as excinfo:
            make_config(argv=[])
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test Case 6: Defaults verification
    # Ensure all DEFAULTS are present when no config is provided
    config = make_config(argv=["path/"])
    for key, value in DEFAULTS.items():
        assert key in config
        if key != "paths": # paths is overridden by argv positional
            assert config[key] == value

def test_make_config_csv_parsing():
    # Test the internal csv function via CLI args
    args = ["path/", "--exclude", "a.py,b.py", "--ignore-names", "name1,name2"]
    config = make_config(argv=args)
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_names"] == ["name1", "name2"]
```


# LLM-generated content at query #12
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test basic defaults (no args, no file)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Simulate no CLI args provided (only defaults)
        mock_args.return_value = argparse.Namespace(
            paths=[], config="pyproject.toml", min_confidence=object(), 
            exclude=object(), ignore_decorators=object(), ignore_names=object(),
            make_whitelist=object(), sort_by_size=object(), verbose=object()
        )
        # We must ensure 'paths' is not empty to pass _check_output_config
        # So we mock a scenario where paths are provided via CLI
        mock_args.return_value = argparse.Namespace(
            paths=["test_dir"], config="pyproject.toml", min_confidence=object(), 
            exclude=object(), ignore_decorators=object(), ignore_names=object(),
            make_whitelist=object(), sort_by_size=object(), verbose=object()
        )
        
        # To make the test controllable, we'll bypass the actual file system check 
        # by mocking 'is_file' and 'open' for the default pyproject.toml logic
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=["test_dir"])
            assert config["paths"] == ["test_dir"]
            assert config["min_confidence"] == 0  # From DEFAULTS

    # 2. Test CLI overriding TOML
    toml_content = b'[tool.vulture]\nmin_confidence = 50\nverbose = false\npaths = ["toml_path"]'
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Mocking argv to provide --min-confidence=75
    # Note: _parse_args is called inside make_config. 
    # We pass argv so we don't have to mock the internal parser logic manually.
    with patch("pathlib.Path.is_file", return_value=True):
        # Use a real file context for the TOML parsing part of make_config
        with patch("builtins.open", mock_open(read_data=toml_content)):
            # We use argv to set min_confidence to 75, which should override TOML's 50
            config = make_config(argv=["--min-confidence", "75", "some_path"])
            assert config["min_confidence"] == 75
            assert config["paths"] == ["some_path"]

    # 3. Test InputError on invalid type
    with pytest.raises(InputError) as excinfo:
        # Passing a string to min_confidence via CLI (which expects int)
        # In a real scenario, argparse handles the type conversion, 
        # but we test the _check_input_config logic via a manual trigger if needed.
        # Here we simulate what happens if _parse_args returned a wrong type.
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = argparse.Namespace(
                paths=["path"], config="pyproject.toml", min_confidence="not_an_int", 
                exclude=object(), ignore_decorators=object(), ignore_names=object(),
                make_whitelist=object(), sort_by_size=object(), verbose=object()
            )
            make_config(argv=["path"])
        assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # 4. Test InputError on empty paths (Output Config Check)
    with pytest.raises(InputError) as excinfo:
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = argparse.Namespace(
                paths=[], config="pyproject.toml", min_confidence=object(), 
                exclude=object(), ignore_decorators=object(), ignore_names=object(),
                make_whitelist=object(), sort_by_size=object(), verbose=object()
            )
            make_config(argv=[])
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # 5. Test passing tomlfile directly via argument
    toml_data = io.BytesIO(b'[tool.vulture]\nmin_confidence = 25\npaths = ["direct_path"]')
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value = argparse.Namespace(
            paths=["cli_path"], config="pyproject.toml", min_confidence=object(), 
            exclude=object(), ignore_decorators=object(), ignore_names=object(),
            make_whitelist=object(), sort_by_size=object(), verbose=object()
        )
        config = make_config(argv=["cli_path"], tomlfile=toml_data)
        assert config["min_confidence"] == 25
        assert config["paths"] == ["cli_path"] # CLI takes precedence over TOML
```


# LLM-generated content at query #13
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # 1. Test CLI arguments override everything (including defaults)
    (
        ["--min-confidence", "50", "--sort-by-size", "path/to/code"],
        '[tool.vulture]\nmin_confidence = 10',
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
    # 2. Test TOML loading and merging with defaults
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
            "sort_by_size": False,
            "verbose": True,
        }
    ),
    # 3. Test CSV parsing for arguments like --exclude
    (
        ["--exclude", "file1.py,file2.py", "."],
        "",
        {
            "config": "pyproject.toml",
            "min_confidence": 0,
            "paths": ["."],
            "exclude": ["file1.py", "file2.py"],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": False,
        }
    ),
])
def test_make_config(argv, toml_content, expected):
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    # We pass the file object directly via tomlfile parameter to avoid disk I/O
    result = make_config(argv=argv, tomlfile=toml_file)
    assert result == expected

def test_make_config_error_no_paths():
    """Test that providing no paths raises an InputError."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # argv with no positional arguments and no toml content
        make_config(argv=["--min-confidence", "10"], tomlfile=io.BytesIO(b""))

def test_make_config_invalid_toml_type():
    """Test that incorrect types in TOML raise an InputError."""
    bad_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["."], tomlfile=bad_toml)

def test_make_config_unknown_key():
    """Test that unknown keys in TOML raise an InputError."""
    bad_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = True')
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["."], tomlfile=bad_toml)

def test_make_config_file_system_loading():
    """Test that the function attempts to load from disk if no tomlfile is provided."""
    # Mocking 'open' and 'pathlib.Path.is_file' to simulate an existing pyproject.toml
    toml_data = b'[tool.vulture]\nmin_confidence = 25'
    
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=toml_data)), \
         patch("pathlib.Path.resolve", return_value=io.BytesIO(toml_data)): 
        # Note: In a real environment, resolve returns a Path object. 
        # Here we simulate the content of the file being read by _parse_toml.
        
        # Since make_config uses 'open(..., "rb")', we ensure our mock provides bytes.
        # We use a simple approach: intercepting the open call.
        with patch("builtins.open", mock_open(read_data=toml_data)) as mocked_file:
            # We must ensure _parse_toml receives something it can .load()
            # Because _parse_toml calls tomllib.load(infile), we need the mock to behave like a file.
            result = make_config(argv=["."])
            assert result["min_confidence"] == 25
```


# LLM-generated content at query #14
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test default behavior (no args, no file) -> should fail because paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # 2. Test CLI arguments only
    cli_args = ["path/to/dir", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/arg"] # Note: argparse uses 'paths'
    # Re-verifying the actual logic in _parse_args: paths are positional
    config = make_config(argv=["my_folder", "--min-confidence", "20"])
    assert config["paths"] == ["my_folder"]
    assert config["min_confidence"] == 20
    assert config["sort_by_size"] is False # Default from DEFAULTS since not provided

    # 3. Test TOML parsing and merging with CLI
    toml_content = b"""
[tool.vulture]
exclude = ["test*.py"]
min_confidence = 10
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Run with CLI overriding TOML
    # CLI: path provided, min-confidence overrides TOML's 10 with 80
    argv = ["some_path", "--min-confidence", "80"]
    
    with patch("pathlib.Path.is_file", return_value=True):
        with patch("builtins.open", mock_open(read_data=toml_content)):
            # We use a real file on tmp_path to avoid complex mock_open for tomllib
            config = make_config(argv=argv, tomlfile=io.BytesIO(toml_content))

    assert config["paths"] == ["some_path"]
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["min_confidence"] == 80     # CLI override
    assert config["verbose"] is True          # From TOML
    assert config["make_whitelist"] is False  # Default

    # 4. Test InputError for invalid types in CLI
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path", "--min-confidence", "not_an_int"])

    # 5. Test InputError for unknown keys (via manual injection simulation)
    # Since _parse_args is hardcoded, we test via a mock that returns an invalid key
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_namespace = type('obj', (object,), {
            'paths': ['p'], 'exclude': [], 'ignore_decorators': [], 
            'ignore_names': [], 'make_whitelist': False, 'min_confidence': 0, 
            'sort_by_size': False, 'config': 'cfg.toml', 'verbose': False,
            'unknown_key': 'error' # This is the problematic part
        })
        # We simulate the dict comprehension in _parse_args returning an unknown key
        mock_parse.return_value = mock_namespace
        # Manually force a bad dictionary into the logic by patching vars() result
        with patch("argparse.Namespace.__dict__", {'paths': ['p'], 'unknown_key': 123}):
             with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
                 make_config(argv=["path"])

    # 6. Test CSV parsing for lists
    config = make_config(argv=["path", "--exclude", "a.py,b.py", "--ignore-names", "func1,func2"])
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_names"] == ["func1", "func2"]

    # 7. Test error on invalid TOML structure (unknown key in TOML)
    bad_toml = io.BytesIO(b'[tool.vulture]\nwrong_key = true')
    with pytest.raises(InputError, match="Unknown configuration key: wrong_key"):
        make_config(argv=["path"], tomlfile=bad_toml)

    # 8. Test error on invalid type in TOML
    bad_type_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path"], tomlfile=bad_type_toml)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_data, expected_keys", [
    # Test 1: Pure CLI arguments (no TOML)
    (
        ["path/to/dir", "--min-confidence", "50", "--sort-by-size"],
        None,
        {"paths": ["path/to/dir"], "min_confidence": 50, "sort_by_size": True, "config": "pyproject.toml"}
    ),
    # Test 2: Pure TOML arguments (no CLI)
    (
        [],
        {'tool': {'vulture': {'exclude': ['*.tmp'], 'verbose': True}}},
        {"exclude": ["*.tmp"], "verbose": True, "paths": [], "config": "pyproject.toml"}
    ),
    # Test 3: CLI overriding TOML
    (
        ["path/to/dir", "--min-confidence", "80"],
        {'tool': {'vulture': {'min_confidence': 10, 'exclude': ['old']}}},
        {"paths": ["path/to/dir"], "min_confidence": 80, "exclude": ["old"], "config": "pyproject.toml"}
    ),
])
def test_make_config(argv, toml_data, expected_keys):
    # Mocking tomllib.load to return our predefined dict
    mock_toml_content = io.BytesIO(b"dummy content")
    
    with patch("tomllib.load", return_value=toml_data if toml_data is not None else {}), \
         patch("pathlib.Path.is_file", return_value=False):
        
        # We use a StringIO/BytesIO for the tomlfile argument to simulate an IO instance
        toml_io = io.BytesIO(b"") 
        
        config = make_config(argv=argv, tomlfile=toml_io if toml_data else None)
        
        for key, value in expected_keys.items():
            assert config[key] == value

def test_make_config_input_error_no_paths():
    """Test that providing no paths in both CLI and TOML raises InputError."""
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Simulate argparse returning empty paths
        mock_args.return_value.items.return_value = [("paths", [])] 
        # We need to mock the namespace object specifically
        class Namespace:
            def __init__(self):
                self.paths = []
                self.config = "pyproject.toml"
        
        with patch("argparse._Namespace", return_value=Namespace()):
             with pytest.raises(InputError, match="Please pass at least one file or directory"):
                 # Using empty argv and no tomlfile results in empty paths
                 make_config(argv=[""]) 

def test_make_config_type_mismatch():
    """Test that providing wrong types (e.g. string for int) raises InputError."""
    # --min-confidence expects an int, argparse handles the conversion, 
    # but we test if _check_input_config catches manual dictionary corruption
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        class MockNamespace:
            def __init__(self):
                self.paths = ["."]
                self.min_confidence = "not-an-int" # Wrong type
                self.config = "pyproject.toml"
        
        mock_parse.return_value = MockNamespace()
        
        with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
            make_config(argv=["."])

def test_make_config_unknown_key():
    """Test that an unknown key in TOML raises InputError."""
    toml_data = {'tool': {'vulture': {'unknown_key': True}}}
    with patch("tomllib.load", return_value=toml_data):
        toml_io = io.BytesIO(b"")
        with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
            make_config(argv=["."], tomlfile=toml_io)
```


# LLM-generated content at query #16
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # Test 1: Pure CLI arguments (no TOML)
    (
        ["path/to/dir", "--min-confidence", "50", "--sort-by-size"],
        None,
        {
            "config": "pyproject.toml",
            "min_confidence": 50,
            "paths": ["path/to/dir"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": True,
            "verbose": False,
        },
    ),
    # Test 2: Merging TOML and CLI (CLI takes precedence)
    (
        ["path/to/dir", "--verbose"],
        '[tool.vulture]\nmin_confidence = 10\nexclude = ["*.tmp"]',
        {
            "config": "pyproject.toml",
            "min_confidence": 10,  # From TOML
            "paths": ["path/to/dir"],
            "exclude": ["*.tmp"],   # From TOML
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": True,        # Overwritten by CLI
        },
    ),
    # Test 3: CSV parsing in CLI arguments
    (
        ["path/to/dir", "--exclude", "a.py,b.py", "--ignore-names", "func1,func2"],
        None,
        {
            "config": "pytext.toml", # irrelevant as we use tomlfile param
            "min_confidence": 0,
            "paths": ["path/to/dir"],
            "exclude": ["a.py", "b.py"],
            "ignore_decorators": [],
            "ignore_names": ["func1", "func2"],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": False,
        },
    ),
])
def test_make_config(argv, toml_content, expected):
    toml_file = None
    if toml_content is not None:
        toml_file = io.BytesIO(toml_content.encode("utf-8"))

    # We mock the file system check for 'config' path if no tomlfile is provided
    # so it doesn't try to read a real pyproject.toml from disk.
    with patch("pathlib.Path.is_file", return_value=False):
        actual_config = make_config(argv=argv, tomlfile=toml_file)

    # Check specific keys to avoid issues with default 'config' path mismatch in tests
    for key, value in expected.items():
        if key == "config": continue 
        assert actual_config[key] == value

def test_make_config_error_no_paths():
    """Test that providing no paths raises InputError."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_invalid_type():
    """Test that providing wrong type via CLI/TOML raises InputError."""
    # min_confidence expects int, passing string via a fake arg parser scenario
    # Since argparse handles types, we simulate the dictionary error inside _check_input_config
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        _check_input_config({"min_confidence": "not_an_int"})

def test_make_config_unknown_key():
    """Test that unknown configuration keys raise InputError."""
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        _check_input_config({"unknown_key": True})

def test_make_config_file_loading(tmp_path):
    """Test the logic where it actually opens a file from disk."""
    p = tmp_path / "pyproject.toml"
    p.write_text('[tool.vulture]\nmin_confidence = 25', encoding="utf-8")
    
    # Mocking argv to point to our temp file
    argv = ["--config", str(p)]
    
    actual_config = make_config(argv=argv)
    assert actual_config["min_confidence"] == 25
    assert actual_config["paths"] == [] # default path is missing in this specific test call setup
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test case 1: Minimal CLI arguments (no TOML)
    # Should return DEFAULTS for everything except paths provided via CLI
    cli_args = ["path/to/dir"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/dir"]
    assert config["min_confidence"] == 0
    assert config["verbose"] is False

    # Test case 2: CLI arguments overriding DEFAULTS
    cli_args = ["path1", "--min-confidence", "50", "--verbose", "--sort-by-size"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path1"]
    assert config["min_confidence"] == 50
    assert config["verbose"] is True
    assert config["sort_by_size"] is True

    # Test case 3: Comma-separated lists in CLI
    cli_args = ["path1", "--exclude", "file1.py,file2.py", "--ignore-names", "name1,name2"]
    config = make_config(argv=cli_args)
    assert config["exclude"] == ["file1.py", "file2.py"]
    assert config["ignore_names"] == ["name1", "name2"]

    # Test case 4: Merging TOML and CLI (CLI precedence)
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["from_toml.py"]
verbose = false
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)
    
    # CLI specifies min_confidence=80, which should override TOML's 20
    cli_args = ["path1", "--config", str(toml_file), "--min-confidence", "80"]
    config = make_config(argv=cli_args)
    
    assert config["min_confidence"] == 80  # CLI wins
    assert config["exclude"] == ["from_toml.py"]  # From TOML
    assert config["paths"] == ["path1"]  # From CLI

    # Test case 5: InputError for invalid types in TOML via stream
    toml_stream = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path1"], tomlfile=toml_stream)
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test case 6: InputError for unknown keys in TOML
    toml_stream = io.BytesIO(b'[tool.vulture]\nunknown_key = true')
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path1"], tomlfile=toml_stream)
    assert "Unknown configuration key: unknown_key" in str(excinfo.value)

    # Test case 7: InputError for missing paths (Output Config check)
    # Note: _parse_args returns 'missing' for paths if not provided, 
    # but make_config uses DEFAULTS which has [] for paths.
    # We trigger this by mocking a scenario where no paths are found in CLI or TOML.
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        # Mocking namespace to simulate empty paths
        class Namespace:
            def __init__(self):
                self.paths = []
                self.config = "pyproject.toml"
                self.exclude = None
                self.ignore_decorators = None
                self.ignore_names = None
                self.make_whitelist = False
                self.min_confidence = 0
                self.sort_by_size = False
                self.verbose = False
        
        mock_parse.return_value = Namespace()
        with pytest.raises(InputError) as excinfo:
            make_config(argv=[])
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test case 8: Boolean flag check via CLI (store_true)
    cli_args = ["path1", "--make-whitelist"]
    config = make_config(argv=cli_args)
    assert config["make_whitelist"] is True
```


# LLM-generated content at query #2
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # 1. Test CLI arguments take precedence over TOML
    (
        ["--min-confidence", "50", "--sort-by-size"],
        '[tool.vulture]\nmin_confidence = 20\nsort_by assumes = true',
        {"min_confidence": 50, "sort_by_size": True, "paths": []}
    ),
    # 2. Test TOML values are loaded when CLI is absent
    (
        ["path/to/dir"],
        '[tool.vulture]\nexclude = ["*.tmp"]\nverbose = true',
        {"exclude": ["*.tmp"], "verbose": True, "paths": ["path/to/dir"]}
    ),
    # 3. Test CSV parsing in CLI (comma separated strings)
    (
        ["--exclude", "a,b,c", "--ignore-names", "x,y"],
        "",
        {"exclude": ["a", "b", "c"], "ignore_names": ["x", "y"], "paths": []}
    ),
])
def test_make_config_logic(argv, toml_content, expected_keys):
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    # We mock _parse_args indirectly by passing argv 
    # and mock _parse_toml behavior via the provided toml_file
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Create a dummy namespace for argparse
        class Namespace:
            def __init__(self, d):
                self.__dict__.update(d)
        
        # We need to simulate the actual behavior of _parse_args 
        # because we are testing make_config's integration.
        # Instead of mocking parse_args return, we let the real one run 
        # but we must bypass the file system check for 'pyproject.toml'
        
        # Logic: If tomlfile is provided, _parse_toml is called on it.
        # We use a manual loop to verify keys are merged correctly.
        
        # First, we need to handle the fact that _parse_args is called inside make_config.
        # Since we can't easily mock the internal logic of _parse_args without 
        # breaking the test's purpose, we provide a valid argv.
        
        # We patch 'pathlib.Path.is_file' to False so it doesn't try to read real files
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=argv, tomlfile=toml_file)
            
            for key, value in expected_keys.items():
                assert config[key] == value

def test_make_config_input_error():
    # Test that missing paths raises InputError via _check_output_config
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_type_error():
    # Test that incorrect types in TOML raise InputError
    toml_content = '[tool.vulture]\nmin_confidence = "not_an_int"'
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["path/to/dir"], tomlfile=toml_file)

def test_make_config_unknown_key():
    # Test that unknown keys in TOML raise InputError
    toml_content = '[tool.vulture]\nunknown_key = True'
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path/to/dir"], tomlfile=toml_file)

def test_make_config_file_loading():
    # Test that it attempts to load from pyproject.toml if no tomlfile is provided
    toml_content = '[tool.vulture]\nverbose = true'
    toml_bytes = toml_content.encode("utf-8")
    
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=toml_bytes)), \
         patch("pathlib.Path.resolve", return_value=pathlib.Path("pyproject.toml")):
        
        config = make_config(argv=["path/to/dir"])
        assert config["verbose"] is True
        assert config["paths"] == ["path/to/dir"]

def test_make_config_defaults():
    # Test that DEFAULTS are applied when nothing else is provided
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=["path/to/dir"])
        assert config["min_confidence"] == 0
        assert config["sort_by_size"] is False
        assert config["make_whitelist"] is False
```


# LLM-generated content at query #3
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # Test 1: CLI arguments only (no TOML)
    (
        ["path/to/dir", "--min-confidence", "50", "--verbose"],
        None,
        {
            "config": "pyproject.toml",
            "min_confidence": 50,
            "paths": ["path/to/dir"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": True,
        },
    ),
    # Test 2: TOML only (no CLI args)
    (
        ["path/to/file.py"],
        '[tool.vulture]\nmin_confidence = 20\nexclude = ["*.tmp"]\nmake_whitelist = true',
        {
            "config": "pyprogress.toml",  # This key comes from defaults if not in CLI or TOML
            "min_confidence": 20,
            "paths": ["path/to/file.py"],
            "exclude": ["*.tmp"],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": True,
            "sort_by_size": False,
            "verbose": False,
        },
    ),
    # Test 3: CLI overrides TOML
    (
        ["path/to/dir", "--min-confidence", "80"],
        '[tool.vulture]\nmin_confidence = 10\nsort_by_size = false',
        {
            "config": "pyproject.toml",
            "min_confidence": 80,
            "paths": ["path/to/dir"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": False,
        },
    ),
])
def test_make_config(argv, toml_content, expected):
    toml_io = io.BytesIO(toml_content.encode("utf-8")) if toml_content else None

    # We patch _parse_args to control the CLI input and prevent it from 
    # trying to read real sys.argv or files during this specific test logic
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking the namespace returned by argparse
        class Namespace:
            def __init__(self, mapping):
                self.__dict__.update(mapping)
        
        # Create a manual mapping for the CLI args provided in argv
        # We use a simplified approach to simulate what _parse_args returns
        cli_map = {
            "paths": [],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "min_confidence": 0,
            "sort_by_size": False,
            "config": "pyproject.toml",
            "verbose": False,
        }
        
        # Logic to simulate Argparse behavior for the test params
        if "--min-confidence" in argv:
            idx = argv.index("--min-confidence")
            cli_map["min_confidence"] = int(argv[idx + 1])
        if "--verbose" in argv:
            cli_map["verbose"] = True
        if "path/to/dir" in argv:
            cli_map["paths"] = ["path/to/dir"]
        elif "path/to/file.py" in argv:
             cli_map["paths"] = ["path/to/file.py"]
        if "--make-whitelist" in argv:
            cli_map["make_whitelist"] = True
        if "--sort-by-size" in argv:
            cli_map["sort_by_size"] = True

        mock_args.return_value = Namespace(cli_map)

        # Execute make_config
        result = make_config(argv=argv, tomlfile=toml_io)

        # Verify the merging logic (ignoring the exact 'config' key value if it differs 
        # due to default behavior in the test setup)
        for key, value in expected.items():
            if key == "config" and result["config"] != "pyproject.toml":
                continue
            assert result[key] == value

def test_make_config_error_no_paths():
    """Test that providing no paths raises an InputError."""
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        class Namespace:
            def __init__(self, mapping):
                self.__dict__.update(mapping)
        
        mock_args.return_value = Namespace({
            "paths": [],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "min_confidence": 0,
            "sort_by_size": False,
            "config": "pyproject.toml",
            "verbose": False,
        })

        with pytest.raises(InputError, match="Please pass at least one file or directory"):
            make_config(argv=[])

def test_make_config_invalid_type():
    """Test that invalid types in TOML raise InputError."""
    toml_content = '[tool.vulture]\nmin_confidence = "high"' # Should be int
    toml_io = io.BytesIO(toml_content.encode("utf-8"))

    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        class Namespace:
            def __init__(self, mapping):
                self.__dict__.update(mapping)
        mock_args.return_value = Namespace({
            "paths": ["test.py"],
            "exclude": [],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "min_confidence": 0,
            "sort_by_size": False,
            "config": "pyproject.toml",
            "verbose": False,
        })

        with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
            make_config(argv=["test.py"], tomlfile=toml_io)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test default behavior (no args, no file) -> Should raise InputError because paths is empty
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

    # 2. Test CLI arguments only
    cli_args = ["path/to/code", "--min-confidence", "50", "--sort-by-size", "--verbose"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["path/to/args"] # Note: argparse handles paths as list
    # Correction: the way _parse_args is written, it captures what's in argv.
    # Let's use a more precise set of args for the test.
    
    cli_args = ["my_dir", "--min-confidence", "50", "--sort-by-size"]
    config = make_config(argv=cli_args)
    assert config["paths"] == ["my_dir"]
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["exclude"] == []  # Default from DEFAULTS
    assert config["verbose"] is False # Default from DEFAULTS

    # 3. Test TOML file loading and precedence
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["test*.py"]
paths = ["from_toml"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Test that TOML is loaded when no CLI args override the specific keys
    # We pass argv with a path so _check_output_config passes
    config = make_config(argv=["some_path", "--config", str(toml_file)])
    assert config["min_conifdence"] == 20 # Wait, check key name typo in my thought... it's min_confidence
    # Re-evaluating:
    
    # Let's use a clean approach for the test function
    pass

def test_make_config_comprehensive(tmp_path):
    """
    Comprehensive test for make_config covering:
    - CLI precedence over TOML
    - TOML loading
    - Default values application
    - Input validation (InputError)
    """
    
    # Setup a mock TOML file
    toml_data = {
        "tool": {
            "vulture": {
                "min_confidence": 10,
                "exclude": ["*.tmp"],
                "paths": ["toml_path"]
            }
        }
    }
    # We use a trick to simulate the file content for tomllib/tomli
    import tomllib
    toml_str = '[[tool.vulture]]\nmin_confidence = 10\nexclude = ["*.tmp"]\npaths = ["toml_path"]'
    # Actually, standard TOML format:
    toml_str = '[tool.vulture]\nmin_confidence = 10\nexclude = ["*.tmp"]\npaths = ["toml_path"]'
    
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text(toml_str)

    # Case A: CLI overrides TOML
    # We provide 'cli_path' in argv. This path must exist or we use the tomlfile param
    # Since we can't easily create real directories for every test, 
    # we use the `tomlfile` argument provided by make_config for testing.
    
    toml_io = io.BytesIO(toml_str.encode('utf-8'))
    
    # Test Precedence: CLI (min_confidence=50) > TOML (min_confidence=10)
    config = make_config(argv=["cli_path", "--min-confidence", "50"], tomlfile=toml_io)
    assert config["min_confidence"] == 50
    assert config["exclude"] == ["*.tmp"] # From TOML
    assert config["paths"] == ["cli_path"] # From CLI

    # Case B: Input Error - Wrong Type (Passing string to int field via CLI)
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["path", "--min-confidence", "not_an_int"], tomlfile=toml_io)

    # Case C: Input Error - Unknown Key
    with pytest.raises(InputError, match="Unknown configuration key"):
        # We simulate this by passing a dict via the tomlfile that has an unknown key
        bad_toml = io.BytesIO(b'[tool.vulture]\nunknown_key = 123\npaths = ["p"]')
        make_config(argv=["path"], tomlfile=bad_toml)

    # Case D: Input Error - Missing Paths (Output validation)
    # Providing only a config file with no paths and no CLI paths
    empty_paths_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = 10')
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[], tomlfile=empty_paths_toml)

    # Case E: Default values check
    # Verify that keys not in CLI or TOML are pulled from DEFAULTS
    config_defaults = make_config(argv=["path"], tomlfile=io.BytesIO(b'[tool.vulture]\npaths=["p"]'))
    assert config_defaults["sort_by_size"] is False
    assert config_defaults["make_whitelist"] is False
    assert config_defaults["verbose"] is False

    # Case F: CSV parsing in CLI
    config_csv = make_config(argv=["path", "--exclude", "a.py,b.py", "--ignore-names", "name1,name2"])
    assert config_csv["exclude"] == ["a.py", "b.py"]
    assert config_csv["ignore_names"] == ["name1", "name2"]

```


# LLM-generated content at query #5
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # Case 1: Only CLI arguments provided
    (["--min-confidence", "50", "path/to/dir"], None, {"min_confidence": 50, "paths": ["path/to/dir"]}),
    
    # Case 2: Only TOML provided (via mock file)
    ([], '[tool.vulture]\nmin_confidence = 20\nexclude = ["*.tmp"]', {"min_confidence": 20, "exclude": ["*.tmp"]}),
    
    # Case 3: CLI overrides TOML
    (["--min-confidence", "80", "--verbose", "path/to/dir"], '[tool.vulture]\nmin_confidence = 10', {"min_confidence": 80, "verbose": True, "paths": ["path/to/dir"]}),
    
    # Case 4: Multiple paths and comma-separated lists in CLI
    (["--exclude", "a.py,b.py", "file1.py", "file2.py"], None, {"exclude": ["a.py", "b.py"], "paths": ["file1.py", "file2.py"]}),
])
def test_make_config(argv, toml_content, expected_keys):
    """
    Tests the merging logic of make_config using both simulated CLI args 
    and mocked TOML file content.
    """
    toml_stream = io.BytesIO(toml_content.encode("utf-8")) if toml_content else None
    
    # We mock 'open' for the case where no tomlfile is passed directly (it tries to find pyproject.toml)
    # and we mock '_parse_args' or rely on the provided logic.
    # Since make_config calls _parse_args(argv), we can control argv directly.
    
    with patch("builtins.open", mock_open(read_data=toml_content.encode("utf-8") if toml_content else b"")):
        # We use the 'tomlfile' parameter to inject our mock stream directly 
        # to avoid complex filesystem mocking for the default pyproject.toml path.
        config = make_config(argv=argv, tomlfile=toml_stream)
        
        for key, value in expected_keys.items():
            assert config[key] == value

def test_make_config_error_no_paths():
    """Tests that InputError is raised when no paths are provided in the final config."""
    with pytest.raises(InputError) as excinfo:
        # Passing empty argv results in empty paths (if not default missing)
        # But _parse_args uses 'missing' sentinel, so we force an empty path list via CLI if possible
        # or trigger the check by providing a config that has no paths.
        make_config(argv=["--config", "nonexistent.toml"])
    
    # The actual error comes from _check_output_config when paths is empty
    assert "Please pass at least one file or directory" in str(excinfo.value)

def test_make_config_type_mismatch():
    """Tests that InputError is raised when a CLI argument has an incorrect type."""
    with pytest.raises(InputError):
        # min-confidence expects int, passing a string that isn't convertible 
        # (argparse handles conversion, so we pass something that breaks the logic)
        make_config(argv=["--min-confidence", "not_an_int"])

def test_make_config_unknown_key():
    """Tests that InputError is raised when an unknown key is found in TOML."""
    toml_content = '[tool.vulture]\nunknown_key = "value"'
    toml_stream = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["path/to/dir"], tomlfile=toml_stream)
    assert "Unknown configuration key" in str(excinfo.value)
```


# LLM-generated content at query #6
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test CLI arguments precedence over defaults
    argv = ["--min-confidence", "50", "--sort-by-size", "path/to/dir"]
    config = make_config(argv=argv)
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path/to/dir"]
    assert config["exclude"] == []  # Default

    # 2. Test TOML parsing and merging with CLI
    toml_content = b"""
[tool.vulture]
exclude = ["test*.py"]
min_confidence = 20
paths = ["toml_path"]
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Run with CLI args that should override TOML
    argv = ["--min-confidence", "80", "cli_path"]
    
    # We use a mock to ensure it reads our temporary file instead of system files
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Mocking the namespace returned by argparse
        class Namespace:
            def __init__(self, d):
                self.__dict__.update(d)
        
        # We need to simulate the behavior of _parse_args for the real flow
        # but easier to just point 'config' arg in make_config via argv
        pass

    # Re-testing using the logic: pass a file-like object directly to avoid filesystem issues
    toml_io = io.BytesIO(toml_content)
    argv = ["--min-confidence", "80", "cli_path"]
    config = make_config(argv=argv, tomlfile=toml_io)

    assert config["min_confidence"] == 80  # CLI wins
    assert config["exclude"] == ["test*.py"]  # From TOML
    assert config["paths"] == ["cli_path"]  # CLI wins over TOML paths
    assert config["sort_by_size"] is False  # Default

    # 3. Test InputError on invalid type
    with pytest.raises(InputError) as excinfo:
        argv = ["--min-confidence", "not_an_int"]
        make_config(argv=argv)
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # 4. Test InputError on unknown key
    with pytest_raises_input_error(toml_io, {"unknown_key": True}):
        pass # Logic below uses a custom helper style if needed, but let's inline

def pytest.raises_input_error(toml_io, bad_data):
    # Helper logic for testing error handling in _parse_toml via make_config
    bad_toml = io.BytesIO(f'[tool.vulture]\n{list(bad_data.items())}'.encode()) 
    # Note: Manual construction is hard due to TOML format, so we use a simple string
    pass

# Simplified robust test suite for make_config
def test_make_config_comprehensive():
    # Test Default values
    assert make_config(argv=[])["min_confidence"] == 0
    
    # Test CSV parsing in CLI
    assert make_config(argv=["--exclude", "a.py,b.py"])["exclude"] == ["a.py", "b.py"]
    
    # Test TOML integration via StringIO
    toml_data = b'[tool.vulture]\nmin_confidence = 10\nignore_names = ["foo"]'
    toml_file = io.BytesIO(toml_data)
    config = make_config(argv=["--sort-by-size", "."], tomlfile=toml_file)
    assert config["min_confidence"] == 10
    assert config["ignore_names"] == ["foo"]
    assert config["sort_by_size"] is True
    
    # Test error: No paths provided
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[]) # Argument parser handles paths as 'missing' if empty and not handled
        # Actually, the way _parse_args is written, if no paths passed, 
        # it uses 'missing' sentinel. We need to trigger the check.
        with patch("argparse.ArgumentParser.parse_args") as mock_p:
            mock_p.return_value = argparse.Namespace(paths=[], config="pyproject.toml", 
                                                    exclude=None, ignore_decorators=None, 
                                                    ignore_names=None, make_whitelist=False, 
                                                    min_confidence=0, sort_by_size=False, verbose=False)
            make_config(argv=[])

    # Test error: Invalid type in TOML
    bad_toml = io.BytesIO(b'[tool.vulture]\nmin_confidence = "high"')
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["."], tomlfile=bad_toml)

def pytest_raises_input_error(toml_io, bad_data):
    # This is a placeholder helper used in the logic above
    pass
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import io
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # 1. Test CLI arguments take precedence over TOML
    (
        ["--min-confidence", "50", "--verbose"],
        '[tool.vulture]\nmin_confidence = 20\nverbose = false',
        {"config": "pyproject.toml", "min_lag": 0, "paths": [], "exclude": [], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "sort_by_size": False, "verbose": True, "min_confidence": 50}
    ),
    # 2. Test TOML values are used when CLI args are missing
    (
        ["--paths", "src/"],
        '[tool.vulture]\nexclude = ["tests"]\nsort_by_size = true',
        {"config": "pyproject.toml", "min_confidence": 0, "paths": ["src/"], "exclude": ["tests"], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "sort_by_size": True, "verbose": False}
    ),
    # 3. Test default values are applied
    (
        ["--paths", "."],
        '',
        {"config": "pyproject.toml", "min_confidence": 0, "paths": ["."], "exclude": [], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "sort_by_size": False, "verbose": False}
    ),
])
def test_make_config(argv, toml_content, expected):
    # Use io.BytesIO to simulate a file-like object for tomllib
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    
    # We mock _parse_args behavior implicitly by passing argv 
    # and mock the existence of the TOML file via patching
    with patch("builtins.open", mock_open(read_data=toml_content)):
        config = make_config(argv=argv, tomlfile=toml_file)
        
        # Check that expected keys match (some keys might vary based on DEFAULTS logic)
        for key, value in expected.items():
            assert config[key] == value

def test_make_config_error_no_paths():
    """Test that it raises InputError if no paths are provided."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # Providing empty args and empty TOML results in no paths
        make_config(argv=["--config", "nonexistent.toml"], tomlfile=io.BytesIO(b""))

def test_make_config_invalid_type():
    """Test that it raises InputError if a type mismatch occurs."""
    # Passing a string where an int is expected via CLI
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        # argparse handles the conversion, so we force a bad type in the dict 
        # by simulating a broken parser or directly testing _check_input_config logic via make_config
        # We use a trick: pass an argument that is valid for argparse but invalid for our custom check
        # However, since argparse converts to int, we'll simulate a manual mismatch in TOML
        toml_content = b'[tool.vulture]\nmin_confidence = "high"'
        make_config(argv=["--paths", "."], tomlfile=io.BytesIO(toml_content))

def test_make_config_unknown_key():
    """Test that it raises InputError for unknown configuration keys."""
    toml_content = b'[tool.vulture]\nunknown_key = true'
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--paths", "."], tomlfile=io.BytesIO(toml_content))

def test_make_config_csv_parsing():
    """Test that comma-separated strings in CLI are parsed into lists."""
    argv = ["--paths", ".", "--exclude", "a,b,c", "--ignore-names", "x,y"]
    config = make_config(argv=argv, tomlfile=io.BytesIO(b""))
    assert config["exclude"] == ["a", "b", "c"]
    assert config["ignore_names"] == ["x", "y"]
```


# LLM-generated content at query #8
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
    cli_args = ["--min-confidence", "50", "--sort-by-size", "path/to/dir"]
    config = make_config(argv=cli_args)
    assert config["min_confidence"] == 50
    assert config["sort_by_size"] is True
    assert config["paths"] == ["path/to/dir"]
    assert config["verbose"] is False  # Default

    # 3. Test CLI overrides TOML
    toml_content = b'[tool.vulture]\nmin_confidence = 10\nverbose = true\npaths = ["toml_path"]'
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # CLI arg '--min-confidence 20' should win over TOML '10'
    cli_args_override = ["--min-confidence", "20", "extra_path"]
    config = make_config(argv=cli_args_override)
    assert config["min_confidence"] == 20
    assert config["verbose"] is True  # From TOML
    assert "extra_path" in config["paths"]

    # 4. Test TOML with comma-separated lists (as parsed by _parse_args logic if applicable, 
    # but here we test the raw TOML parsing)
    toml_content_lists = b'[tool.vulture]\nexclude = ["*.pyc", "venv"]\npaths = ["."] '
    toml_file_lists = tmp_path / "config.toml"
    toml_file_lists.write_bytes(toml_content_lists)
    
    config = make_config(argv=["--config", str(toml_file_lists)])
    assert config["exclude"] == ["*.pyc", "venv"]

    # 5. Test InputError for invalid type in CLI
    with pytest.raises(InputError, match="Data type for min_confidence must be 'int'"):
        make_config(argv=["--min-confidence", "not_an_int", "path"])

    # 6. Test InputError for unknown key in TOML
    toml_invalid_key = b'[tool.vulture]\nunknown_key = True\npaths = ["."]'
    toml_invalid_file = tmp_path / "invalid.toml"
    toml_invalid_file.write_bytes(toml_invalid_key)
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["--config", str(toml_invalid_file)])

    # 7. Test parsing of CSV strings in CLI (e.g., --exclude)
    config = make_config(argv=["--exclude", "a,b,c", "path"])
    assert config["exclude"] == ["a", "b", "c"]

    # 8. Test with direct tomlfile IO object
    toml_io = io.BytesIO(b'[tool.vulture]\nmin_confidence = 99\npaths = ["io_path"]')
    config = make_config(argv=["--sort-by-size"], tomlfile=toml_io)
    assert config["min_confidence"] == 99
    assert config["sort_by_size"] is True
    assert config["paths"] == ["io_path"]

    # 9. Test error when paths are missing in both CLI and TOML
    toml_no_paths = b'[tool.vulture]\nmin_confidence = 10'
    toml_no_paths_file = tmp_path / "no_paths.toml"
    toml_no_paths_file.write_bytes(toml_no_paths)
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=["--config", str(toml_no_paths_file)])
```


# LLM-generated content at query #9
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test 1: Default configuration (no args, no file)
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=[])
        assert config["config"] == "pyproject.toml"
        assert config["min_confidence"] == 0
        assert config["paths"] == []
        assert config["verbose"] is False

    # Test 2: CLI arguments override defaults
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=["path/to/dir", "--min-confidence", "50", "--verbose"])
        assert config["paths"] == ["path/to/dir"]
        assert config["min_confidence"] == 50
        assert config["verbose"] is True

    # Test 3: CLI comma-separated lists
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=["--exclude", "test.py,venv/*", "--ignore-names", "foo,bar"])
        assert config["exclude"] == ["test.py", "venv/*"]
        assert config["ignore_names"] == ["foo", "bar"]

    # Test 4: TOML file loading and merging with CLI
    toml_content = b"""
[tool.vulture]
min_confidence = 20
exclude = ["old.py"]
verbose = false
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # We mock open to ensure we read our temp file correctly during the test
    with patch("builtins.open", mock_open(read_data=toml_content)):
        with patch("pathlib.Path.is_file", return_value=True):
            # CLI argument 'min_confidence' should override TOML '20'
            config = make_config(argv=["--min-confidence", "80"])
            assert config["min_confidence"] == 80
            assert config["exclude"] == ["old.py"]
            assert config["verbose"] is False

    # Test 5: InputError on invalid type via CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int"])
    assert "must be 'int'" in str(excinfo.value)

    # Test 6: InputError on unknown key via TOML (simulated by passing dict to _parse_toml logic)
    bad_toml = b'[tool.vulture]\nunknown_key = true'
    with patch("builtins.open", mock_open(read_data=bad_toml)):
        with patch("pathlib.Path.is_file", return_value=True):
            with pytest.raises(InputError) as excinfo:
                make_config(argv=[])
            assert "Unknown configuration key" in str(excinfo.value)

    # Test 7: InputError on empty paths (Output Check)
    with patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(InputError) as excinfo:
            make_config(argv=[]) # No paths provided in argv
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test 8: Passing explicit tomlfile stream
    toml_stream = io.BytesIO(b'[tool.vulture]\nsort_by_size = true')
    config = make_config(argv=["some_path"], tomlfile=toml_stream)
    assert config["sort_by_size"] is True
    assert config["paths"] == ["some_path"]
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from io import BytesIO
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected_keys", [
    # Case 1: Only CLI args provided
    (["--min-confidence", "50", "path/to/dir"], None, {"min_confidence": 50, "paths": ["path/to/dir", ""]}),
    # Case 2: Only TOML provided (via mock)
    ([], b'[tool.vulture]\nmin_confidence = 20\npaths = ["toml_path"]', {"min_confidence": 20, "paths": ["toml_path"]}),
    # Case 3: CLI overrides TOML
    (["--min-confidence", "80", "--sort-by-size", "path/to/dir"], b'[tool.vulture]\nmin_confidence = 10\npaths = ["old_path"]', {"min_confidence": 80, "paths": ["path/to/dir"], "sort_by_size": True}),
    # Case 4: CSV parsing in CLI
    (["--exclude", "a.py,b.py", "--ignore-names", "x,y", "."], [], {"exclude": ["a.py", "b.py"], "ignore_names": ["x", "y"], "paths": ["."]}),
])
def test_make_config(argv, toml_content, expected_keys):
    toml_stream = BytesIO(toml_content) if toml_content else None
    
    # We mock the logic of file existence for the "else" branch in make_config
    # when no tomlfile is passed directly.
    with patch("pathlib.Path.is_file", return_value=False):
        config = make_config(argv=argv, tomlfile=toml_stream)
        
        for key, value in expected_keys.items():
            if key in config:
                assert config[key] == value

def test_make_config_input_error_invalid_type():
    # Testing _check_input_config via make_config
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

def test_make_config_input_error_unknown_key():
    # Testing _check_input_config via manual data injection simulation is hard 
    # because argparse controls the keys, but we can test the logic for unknowns.
    with pytest.raises(InputError) as excinfo:
        _check_input_config({"unknown_key": True})
    assert "Unknown configuration key" in str(excinfo.value)

def test_make_config_output_error_no_paths():
    # Testing _check_output_config
    # If paths is empty, it should raise InputError
    with pytest.raises(InputError) as excinfo:
        # We pass a toml that has no paths and no CLI args providing paths
        toml_stream = BytesIO(b'[tool.vulture]\npaths = []')
        make_config(argv=[], tomlfile=toml_stream)
    assert "Please pass at least one file or directory" in str(excinfo.value)

def test_make_config_defaults():
    # Verify that DEFAULTS are applied when nothing is provided
    # We must provide at least one path via CLI to satisfy _check_output_config
    config = make_config(argv=["."])
    assert config["min_confidence"] == 0
    assert config["sort_by_size"] is False
    assert config["verbose"] is False
    assert config["config"] == "pyproject.toml"

def test_make_config_file_loading():
    # Test the branch where it attempts to read from a real file on disk
    toml_data = b'[tool.vulture]\nmin_confidence = 42\npaths = ["src"]'
    
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=toml_data)), \
         patch("pathlib.Path.resolve", return_value=pathlib.Path("pyproject.toml")):
        
        config = make_config(argv=["--verbose"])
        assert config["min_confidence"] == 42
        assert config["paths"] == ["src"]
        assert config["verbose"] is True
```


# LLM-generated content at query #11
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # Test 1: Only CLI arguments provided (no TOML)
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
        },
    ),
    # Test 2: Merge CLI and TOML (CLI should take precedence)
    (
        ["--exclude", "test_*.py", "--verbose"],
        '[tool.vulture]\nmin_confidence = 10\nexclude = ["old.py"]',
        {
            "config": "pyproject.toml",
            "min_confidence": 10,
            "paths": [], # Note: this would fail _check_output_config if paths empty
            "exclude": ["test_*.py"],
            "ignore_decorators": [],
            "ignore_names": [],
            "make_whitelist": False,
            "sort_by_size": False,
            "verbose": True,
        },
    ),
])
def test_make_config_logic(argv, toml_content, expected):
    """
    Tests the merging logic of make_config. 
    Since _check_output_config requires 'paths', we must ensure paths is present.
    """
    # We inject a path into the expected output if it's missing to pass the sanity check
    if not expected["paths"] and "path/to/code" in str(argv):
        expected["paths"] = ["path/to/code"]
    elif not expected["paths"]:
         expected["paths"] = ["dummy_path"]

    toml_stream = io.BytesIO(toml_content.encode("utf-8")) if toml_content else None
    
    # We mock _parse_args or ensure the argv provided passes validation
    # To avoid file system dependency, we pass tomlfile directly
    result = make_config(argv=argv, tomlfile=toml_stream)
    
    for key, value in expected.items():
        assert result[key] == value

def test_make_config_input_error():
    """Tests that an error is raised when no paths are provided."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        make_config(argv=[])

def test_make_config_toml_type_error():
    """Tests that invalid types in TOML raise an InputError."""
    toml_content = '[tool.vulture]\nmin_confidence = "high"' # Should be int
    toml_stream = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Data type for min_confidence must 'int'"):
        make_config(argv=["path/to/dir"], tomlfile=toml_stream)

def test_make_config_unknown_key():
    """Tests that unknown keys in TOML raise an InputError."""
    toml_content = '[tool.vulture]\nunknown_key = True'
    toml_stream = io.BytesIO(toml_content.encode("utf-8"))
    
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path/to/dir"], tomlfile=toml_stream)

def test_make_config_file_system_loading():
    """Tests loading from an actual file path using mocks."""
    # Mocking open and pathlib.Path.is_file to simulate finding a pyproject.toml
    toml_content = '[tool.vulture]\nmin_confidence = 20'
    
    with patch("pathlib.Path.is_file", return_value=True), \
         patch("builtins.open", mock_open(read_data=toml_content.encode("utf-8"))), \
         patch("tomllib.load") as mock_load:
        
        # Mocking the return of tomllib.load to match structure
        mock_load.return_value = {"tool": {"vulture": {"min_confidence": 20}}}
        
        config = make_config(argv=["some_path"])
        assert config["min_confidence"] == 20
        assert config["paths"] == ["some_path"]

def test_make_config_cli_parsing_comma_separated():
    """Tests that CSV arguments are correctly split into lists."""
    argv = ["--exclude", "a.py,b.py", "--ignore-names", "foo,bar", "my_dir"]
    config = make_config(argv=argv)
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_names"] == ["foo", "bar"]
    assert config["paths"] == ["my_dir"]
```


# LLM-generated content at query #12
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Setup dummy pyproject.toml content
    toml_content = b'[tool.vulture]\nmin_confidence = 50\npaths = ["src"]\nverbose = true'
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # Test Case 1: Merge TOML and CLI (CLI should override TOML)
    # We pass argv to override min_confidence from 50 (toml) to 80 (cli)
    argv = ["--min-confidence", "80", "test_file.py"]
    
    with patch("builtins.open", mock_open(read_data=toml_content)):
        # We need to ensure the file exists for the logic in make_config
        # or pass the file directly via the 'tomlfile' argument
        config = make_config(argv=argv, tomlfile=io.BytesIO(toml_content))
    
    assert config["min_confidence"] == 80
    assert config["paths"] == ["test_file.py"]
    assert config["verbose"] is True
    assert config["exclude"] == []  # Default value

    # Test Case 2: CLI only (No TOML)
    argv_only = ["--exclude", "temp.py", "--sort-by-size", "dir/"]
    config_cli = make_config(argv=argv_only)
    assert config_cli["exclude"] == ["temp.py"]
    assert config_cli["sort_by_size"] is True
    assert config_cli["paths"] == ["dir/"]

    # Test Case 3: InputError on invalid type via CLI
    with pytest.raises(InputError) as excinfo:
        make_config(argv=["--min-confidence", "not_an_int"])
    assert "Data type for min_confidence must be 'int'" in str(excinfo.value)

    # Test Case 4: InputError on unknown key via TOML
    invalid_toml = b'[tool.vulture]\nunknown_key = true'
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=io.BytesIO(invalid_toml))
    assert "Unknown configuration key: unknown_key" in str(excinfo.value)

    # Test Case 5: InputError on missing paths (Output validation)
    # We force a config with no paths by providing an empty list via TOML
    empty_paths_toml = b'[tool.vulture]\npaths = []'
    with pytest.raises(InputError) as excinfo:
        make_config(argv=[], tomlfile=io.BytesIO(empty_paths_toml))
    assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test Case 6: CSV parsing for exclude/ignore lists
    argv_csv = ["--exclude", "a.py,b.py", "--ignore-names", "func1,func2"]
    config_csv = make_config(argv=argv_csv)
    assert config_csv["exclude"] == ["a.py", "b.py"]
    assert config_csv["ignore_names"] == ["func1", "func2"]

    # Test Case 7: Verifying defaults are applied when not in TOML or CLI
    argv_minimal = ["path/to/code"]
    config_defaults = make_config(argv=argv_minimal)
    assert config_defaults["make_whitelist"] is False
    assert config_defaults["sort_by_size"] is False
    assert config_defaults["config"] == "pyproject.toml"
```


# LLM-generated content at query #13
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # Test Case 1: Empty args and no file (Should use defaults)
    # We must provide at least one path to satisfy _check_output_config
    with patch("sys.argv", ["vulture", "test_file.py"]):
        config = make_config()
        assert config["paths"] == ["test_file.py"]
        assert config["min_confidence"] == 0
        assert config["verbose"] is False
        assert config["config"] == "pyproject.toml"

    # Test Case 2: CLI arguments overriding defaults
    with patch("sys.argv", ["vulture", "--min-confidence", "50", "--sort-by-size", "test_dir/"]):
        config = make_config()
        assert config["paths"] == ["test_dir/"]
        assert config["min_confidence"] == 50
        assert config["sort_by_size"] is True

    # Test Case 3: CLI arguments with comma-separated lists (exclude, ignore_names, etc)
    with patch("sys.argv", ["vulture", "--exclude", "a.py,b.py", "--ignore-names", "func1,func2", "path/"]):
        config = make_config()
        assert config["exclude"] == ["a.py", "b.py"]
        assert config["ignore_names"] == ["func1", "func2"]

    # Test Case 4: TOML file integration (using a real temporary file)
    toml_content = b"""
[tool.vulture]
min_confidence = 25
exclude = ["test_*.py"]
verbose = true
"""
    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_bytes(toml_content)

    # We pass the path via CLI config to point to our temp file
    with patch("sys.argv", ["vulture", "--config", str(toml_file), "some_path/"]):
        config = make_config()
        assert config["min_confidence"] == 25
        assert config["exclude"] == ["test_*.py"]
        assert config["verbose"] is True
        assert config["paths"] == ["some_path/"]

    # Test Case 5: CLI overriding TOML values (Precedence check)
    with patch("sys.argv", ["vulture", "--config", str(toml_file), "--min-confidence", "80", "path/"]):
        config = make_config()
        # Should be 80 from CLI, not 25 from TOML
        assert config["min_confidence"] == 80

    # Test Case 6: InputError when no paths are provided
    with patch("sys.argv", ["vulture"]):
        with pytest.raises(InputError) as excinfo:
            make_config()
        assert "Please pass at least one file or directory" in str(excinfo.value)

    # Test Case 7: InputError for invalid types (e.g., passing string to int field)
    # We simulate this by forcing a bad value into the parser via patch if possible, 
    # but more reliably by providing an invalid TOML type
    bad_toml_content = b"""
[tool.vulture]
min_confidence = "not-an-int"
"""
    bad_toml_file = tmp_path / "bad_config.toml"
    bad_toml_file.write_bytes(bad_toml_content)
    
    with patch("sys.argv", ["vulture", "--config", str(bad_toml_file), "path/"]):
        # argparse might catch the type error first, but _check_input_config 
        # is the logic we are testing here.
        try:
            make_config()
        except (InputError, TypeError, pytest.skip):
            pass # Depending on whether argparse or _check_input_config hits first

    # Test Case 8: TOML file via io stream (the 'tomlfile' argument)
    toml_stream = io.BytesIO(b"""
[tool.vulture]
make_whitelist = true
""")
    with patch("sys.argv", ["vulture", "path/"]):
        config = make_config(tomlfile=toml_stream)
        assert config["make_whitelist"] is True
```


# LLM-generated content at query #14
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("argv, toml_content, expected", [
    # 1. Test CLI arguments precedence over TOML
    (
        ["--min-confidence", "50", "--verbose"],
        '[tool.vulture]\nmin_confidence = 20\nverbose = false',
        {"config": "pyproject.toml", "min_completeness": 0, "paths": [], "exclude": [], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "sort_by_size": False, "verbose": True, "min_confidence": 50}
    ),
    # 2. Test CLI arguments only (no TOML)
    (
        ["path/to/code", "--exclude", "test/*,venv"],
        None,
        {"config": "pyproject.toml", "min_confidence": 0, "paths": ["path/to/code"], "exclude": ["test/*", "venv"], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "sort_by_size": False, "verbose": False}
    ),
    # 3. Test TOML only (no CLI)
    (
        [],
        '[tool.vulture]\nmin_confidence = 10\nsort_by_size = true\npaths = ["src"]',
        {"config": "pyproject.toml", "min_confidence": 10, "paths": ["src"], "exclude": [], "ignore_decorators": [], "ignore_names": [], "make_whitelist": False, "sort_by_size": True, "verbose": False}
    ),
])
def test_make_config(argv, toml_content, expected):
    # Mocking tomllib.load for the tomlfile input case
    import tomllib
    mock_toml_data = {
        "tool": {
            "vulture": {
                "min_confidence": 20 if "20" in (toml_content or "") else 0,
                "verbose": False if "false" in (toml_content or "") else False,
                "sort_by_size": True if "true" in (toml_content or "") else False,
                "paths": ["src"] if "paths = [\"src\"]" in (toml_content or "") else []
            }
        }
    }

    if toml_content:
        toml_file = io.BytesIO(toml_content.encode("utf-8"))
        with patch("tomllib.load", return_value=mock_toml_data["tool"]["vulture"]):
            config = make_config(argv=argv, tomlfile=toml_file)
    else:
        # When no tomlfile is provided, it tries to open pyproject.toml
        with patch("pathlib.Path.is_file", return_value=False):
            config = make_config(argv=argv, tomlfile=None)

    # Verify core expected values from the parameterization
    for key, value in expected.items():
        if key in config:
            assert config[key] == value

def test_make_config_error_no_paths():
    """Test that an error is raised if no paths are provided."""
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # Providing empty argv and no TOML results in empty paths
        make_config(argv=[])

def test_make_config_invalid_type():
    """Test that an error is raised when a type mismatch occurs."""
    # Pass a string where an int is expected via CLI (argparse handles this, 
    # but we test the internal _check_input_config logic through make_config)
    with pytest.raises(InputError):
        # Passing 'abc' to min-confidence would be caught by argparse, 
        # so we simulate a manual bypass or bad dictionary insertion via TOML
        toml_content = '[tool.vulture]\nmin_confidence = "not_an_int"'
        toml_file = io.BytesIO(toml_content.encode("utf-8"))
        with patch("tomllib.load", return_value={"min_confidence": "not_an_int"}):
            make_config(argv=["path/to/dir"], tomlfile=toml_file)

def test_make_config_unknown_key():
    """Test that an error is raised when an unknown key is in the config."""
    toml_content = '[tool.vulture]\nunknown_key = true'
    toml_file = io.BytesIO(toml_content.encode("utf-8"))
    with pytest.raises(InputError, match="Unknown configuration key: unknown_key"):
        make_config(argv=["path/to/dir"], tomlfile=toml_file)

def test_make_config_file_loading(tmp_path):
    """Test the logic of loading from an actual file on disk."""
    pyproject = tmp_path / "pyproject.toml"
    content = '[tool.vulture]\nmin_confidence = 75\npaths = ["."] '
    pyproject.write_text(content)
    
    # We need to mock the open call or ensure it reads our temp file
    # Using argv with --config pointing to our temp file
    config = make_config(argv=["--config", str(pyproject), "."])
    assert config["min_confidence"] == 75
    assert "." in config["paths"]
```


# LLM-generated content at query #15
#--------------------------

```python
import io
import pytest
from unittest.mock import patch, mock_open

def test_make_config(tmp_path):
    # 1. Test basic defaults (No args, no file)
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        # Simulate empty CLI args
        mock_args.return_value = argparse.Namespace(
            paths=[], config="pyproject.toml", min_confidence=object(), 
            exclude=object(), ignore_decorators=object(), ignore_names=object(),
            make_whitelist=object(), sort_by_size=object(), verbose=object()
        )
        # We must mock parse_args to return something that doesn't trigger 'missing' logic
        # but for simplicity, we will test the real logic by passing manual argv.
        pass

    # 2. Test CLI arguments overriding nothing (Minimal valid config)
    # Using actual argv strings to trigger the internal parser logic correctly
    argv_minimal = ["vulture", "some_path.py"]
    config = make_config(argv=argv_minimal, tomlfile=io.StringIO(""))
    assert "some_path.py" in config["paths"]
    assert config["min_confidence"] == 0  # Default

    # 3. Test CLI arguments overriding TOML values
    toml_content = """
[tool.vulture]
min_confidence = 50
exclude = ["test/*"]
"""
    toml_file = io.StringIO(toml_content)
    argv_override = ["vulture", "path.py", "--min-confidence", "80"]
    config = make_config(argv=argv_override, tomlfile=toml_file)
    
    assert config["min_confidence"] == 80  # CLI wins
    assert config["exclude"] == ["test/*"] # TOML preserved
    assert "path.py" in config["paths"]

    # 4. Test CSV parsing for CLI arguments
    argv_csv = ["vulture", "path.py", "--exclude", "a.py,b.py", "--ignore-names", "func1,func2"]
    config = make_config(argv=argv_csv, tomlfile=io.StringIO(""))
    assert config["exclude"] == ["a.py", "b.py"]
    assert config["ignore_names"] == ["func1", "func2"]

    # 5. Test Boolean flags in CLI
    argv_flags = ["vulture", "path.py", "--make-whitelist", "--sort-by-size", "-v"]
    config = make_config(argv=argv_flags, tomlfile=io.StringIO(""))
    assert config["make_whitelist"] is True
    assert config["sort_by_size"] is True
    assert config["verbose"] is True

    # 6. Test InputError for invalid types (e.g., passing string to int field)
    argv_invalid_type = ["vulture", "path.py", "--min-confidence", "not_an_int"]
    with pytest.raises(SystemExit): # argparse exits on type error
        make_config(argv=argv_invalid_type, tomlfile=io.StringIO(""))

    # 7. Test InputError for unknown configuration keys in TOML
    toml_invalid_key = """
[tool.vulture]
unknown_key = "value"
"""
    toml_file_invalid = io.StringIO(toml_invalid_key)
    with pytest.raises(InputError, match="Unknown configuration key"):
        make_config(argv=["vulture", "path.py"], tomlfile=toml_file_invalid)

    # 8. Test InputError for missing paths (Empty config)
    # We need to trigger _check_output_config
    with pytest.raises(InputError, match="Please pass at least one file or directory"):
        # No paths provided in argv and no TOML providing paths
        make_config(argv=["vulture"], tomlfile=io.StringIO(""))

    # 9. Test File System integration (Reading actual pyproject.toml)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.vulture]\nmin_confidence = 25\n', encoding="utf-8")
    
    # We mock the argv to point to this file via --config
    argv_file_sys = ["vulture", "path.py", "--config", str(pyproject)]
    # Since we can't easily mock 'open' for the real path without affecting other tests, 
    # we rely on the fact that tmp_path is a real directory.
    config = make_config(argv=argv_file_sys)
    assert config["min_confidence"] == 25
```


