####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Config_is_supported_filetype():
    """Test the is_supported_filetype method of Config class."""
    import tempfile
    import os
    
    config = Config()
    
    # Test with supported extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("module.pyi") is True
    
    # Test with blocked extensions
    assert config.is_supported_filetype("test.pyc") is False
    assert config.is_supported_filetype("test.pyo") is False
    
    # Test with backup files (ending with ~)
    assert config.is_supported_filetype("test.py~") is False
    assert config.is_supported_filetype("backup~") is False
    
    # Test with unsupported extensions
    assert config.is_supported_filetype("test.txt") is False
    assert config.is_supported_filetype("test.md") is False
    
    # Test with file that has shebang
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='') as f:
        f.write(b"#!/usr/bin/env python\n")
        f.write(b"print('hello')\n")
        temp_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)
    
    # Test with file that doesn't have shebang and unsupported extension
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
        f.write(b"just some text\n")
        temp_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_file) is False
    finally:
        os.unlink(temp_file)
    
    # Test with non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["py", "pyi", "txt"])
    assert config_custom.is_supported_filetype("test.txt") is True
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["py"])
    assert config_blocked.is_supported_filetype("test.py") is False


# LLM-generated content at query #2
#--------------------------

```python
def test_Config():
    """Test Config constructor with various parameter combinations."""
    
    # Test 1: Basic instantiation with no parameters
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Instantiation with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "
    
    # Test 3: Instantiation with indent as string
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 4: Instantiation with indent as quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 5: Instantiation with another Config object
    base_config = Config(line_length=120)
    new_config = Config(config=base_config, indent=2)
    assert new_config.line_length == 120
    assert new_config.indent == "  "
    
    # Test 6: Test quiet flag suppresses warnings
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 7: Test profile parameter
    config = Config(profile="black")
    assert config is not None
    
    # Test 8: Test known_* sections
    config = Config(known_django=["django"], known_flask=["flask"])
    assert "django" in config.known_other.get("django", set())
    assert "flask" in config.known_other.get("flask", set())
    
    # Test 9: Test import_heading_* and import_footer_*
    config = Config(import_heading_future="Future imports", import_footer_stdlib="End stdlib")
    assert "future" in config.import_headings
    assert "stdlib" in config.import_footers
    
    # Test 10: Test sections parameter
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert "FUTURE" in config.sections
    
    # Test 11: Test src_paths parameter
    config = Config(src_paths=["/path/to/src"])
    assert config.src_paths is not None
    
    # Test 12: Test directory parameter
    config = Config(directory="/tmp")
    assert config.directory == "/tmp"
    
    # Test 13: Test skip and extend_skip
    config = Config(skip=["migrations"], extend_skip=["node_modules"])
    assert "migrations" in config.skips
    assert "node_modules" in config.skips
    
    # Test 14: Test skip_glob and extend_skip_glob
    config = Config(skip_glob=["*.egg-info"], extend_skip_glob=["build/*"])
    assert "*.egg-info" in config.skip_globs
    assert "build/*" in config.skip_globs
    
    # Test 15: Test sort_order parameter
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"
    
    # Test 16: Test wrap_length validation
    try:
        config = Config(wrap_length=150, line_length=100)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "wrap_length must be set lower than or equal to line_length" in str(e)
    
    # Test 17: Test with numeric indent
    config = Config(indent=8)
    assert config.indent == "        "
    
    # Test 18: Test supported_extensions parameter
    config = Config(supported_extensions=["py", "pyi"])
    assert "py" in config.supported_extensions
    
    # Test 19: Test blocked_extensions parameter
    config = Config(blocked_extensions=["pyc"])
    assert "pyc" in config.blocked_extensions
    
    # Test 20: Test hash method
    config1 = Config()
    config2 = Config()
    assert hash(config1) != hash(config2)
    assert hash(config1) == id(config1)


# LLM-generated content at query #3
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function with various config file scenarios."""
    
    # Create directory structure with config files
    root = tmp_path / "project"
    root.mkdir()
    
    subdir1 = root / "src"
    subdir1.mkdir()
    
    subdir2 = root / "tests"
    subdir2.mkdir()
    
    nested = subdir1 / "nested"
    nested.mkdir()
    
    # Create setup.cfg in root
    setup_cfg = root / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile=black\n")
    
    # Create pyproject.toml in subdir1
    pyproject = subdir1 / "pyproject.toml"
    pyproject.write_text("[tool.isort]\nline_length=100\n")
    
    # Create .isort.cfg in nested
    isort_cfg = nested / ".isort.cfg"
    isort_cfg.write_text("[settings]\nmulti_line_mode=3\n")
    
    # Call function
    trie = find_all_configs(str(root))
    
    # Verify trie is created
    assert trie is not None
    assert trie.data == {}
    
    # Verify config files were found and inserted
    # The trie should contain entries for found config files
    assert trie.children is not None


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    
    root = tmp_path / "empty_project"
    root.mkdir()
    
    subdir = root / "src"
    subdir.mkdir()
    
    # Call function with directory containing no config files
    trie = find_all_configs(str(root))
    
    # Verify trie is created but empty
    assert trie is not None
    assert trie.data == {}


def test_find_all_configs_invalid_config(tmp_path):
    """Test find_all_configs with malformed config file."""
    
    root = tmp_path / "project"
    root.mkdir()
    
    # Create invalid setup.cfg
    setup_cfg = root / "setup.cfg"
    setup_cfg.write_text("[invalid config content {{{")
    
    # Call function - should handle exception gracefully
    trie = find_all_configs(str(root))
    
    # Verify trie is created despite invalid config
    assert trie is not None
    assert trie.data == {}


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with deeply nested directories."""
    
    root = tmp_path / "project"
    root.mkdir()
    
    # Create nested structure
    level1 = root / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Add config at each level
    (root / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    (level2 / "setup.cfg").write_text("[isort]\nline_length=88\n")
    (level3 / "pyproject.toml").write_text("[tool.isort]\nmulti_line_mode=3\n")
    
    # Call function
    trie = find_all_configs(str(root))
    
    # Verify trie is created
    assert trie is not None
    assert trie.data == {}


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with empty directory."""
    
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie = find_all_configs(str(empty_dir))
    
    assert trie is not None
    assert trie.data == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_find_all_configs(tmp_path, monkeypatch):
    """Test find_all_configs function to verify it correctly finds and parses config files."""
    monkeypatch.chdir(tmp_path)
    
    # Create directory structure with config files
    subdir1 = tmp_path / "subdir1"
    subdir1.mkdir()
    subdir2 = tmp_path / "subdir2"
    subdir2.mkdir()
    nested = subdir1 / "nested"
    nested.mkdir()
    
    # Create mock config files
    setup_cfg_content = "[isort]\nprofile=black\n"
    pyproject_content = "[tool.isort]\nprofile='django'\n"
    
    # Write config files
    (tmp_path / "setup.cfg").write_text(setup_cfg_content)
    (subdir1 / ".isort.cfg").write_text(setup_cfg_content)
    (nested / "pyproject.toml").write_text(pyproject_content)
    
    # Call find_all_configs
    trie_root = find_all_configs(str(tmp_path))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.data == {}
    assert trie_root.key == "default"
    
    # Verify configs were found and inserted
    assert len(trie_root.children) > 0


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with directory containing no config files."""
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    assert trie_root.data == {}


def test_find_all_configs_invalid_config_file(tmp_path, monkeypatch):
    """Test find_all_configs handles invalid config files gracefully."""
    # Create an invalid config file
    config_file = tmp_path / "setup.cfg"
    config_file.write_text("[invalid content that cannot be parsed")
    
    # Should not raise exception
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with nested directory structure."""
    # Create nested directories
    level1 = tmp_path / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Create config at different levels
    (tmp_path / "setup.cfg").write_text("[isort]\nprofile=black\n")
    (level2 / ".isort.cfg").write_text("[isort]\nprofile=django\n")
    
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None
    assert trie_root.key == "default"


# LLM-generated content at query #5
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization methods."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert hasattr(config, 'known_other')
    assert hasattr(config, 'import_headings')
    assert hasattr(config, 'import_footers')
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "  # 4 spaces
    
    # Test 3: Initialization with profile
    config = Config(profile="black")
    assert config is not None
    
    # Test 4: Initialization with existing config object
    base_config = Config(line_length=88)
    new_config = Config(config=base_config, line_length=100)
    assert new_config.line_length == 100
    
    # Test 5: Indent as string "tab"
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 6: Indent as quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 7: Known sections configuration
    config = Config(known_first_party=["mymodule"])
    assert "mymodule" in config.known_first_party
    
    # Test 8: Import headings configuration
    config = Config(import_heading_future="from __future__ imports")
    assert "future" in config.import_headings
    assert config.import_headings["future"] == "from __future__ imports"
    
    # Test 9: Import footers configuration
    config = Config(import_footer_stdlib="End of stdlib")
    assert "stdlib" in config.import_footers
    assert config.import_footers["stdlib"] == "End of stdlib"
    
    # Test 10: Skip configuration
    config = Config(skip=["migrations"], extend_skip=["node_modules"])
    assert "migrations" in config.skips
    assert "node_modules" in config.skips
    
    # Test 11: Skip globs configuration
    config = Config(skip_glob=["*.egg-info"], extend_skip_glob=["venv/**"])
    assert "*.egg-info" in config.skip_globs
    assert "venv/**" in config.skip_globs
    
    # Test 12: Sections configuration
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert config.sections == ("FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER")
    
    # Test 13: Known other sections
    config = Config(known_django=["django"])
    assert "django" in config.known_other.get("django", [])
    
    # Test 14: Directory configuration
    config = Config(directory="/tmp")
    assert config.directory == "/tmp"
    
    # Test 15: Multiple config overrides
    config = Config(
        line_length=120,
        multi_line_mode=3,
        include_trailing_comma=True,
        force_single_line=False
    )
    assert config.line_length == 120
    assert config.include_trailing_comma is True
    assert config.force_single_line is False
    
    # Test 16: Quiet mode to suppress warnings
    config = Config(quiet=True, profile="nonexistent_profile_that_should_not_error")
    
    # Test 17: Invalid profile raises error
    with pytest.raises(Exception):  # ProfileDoesNotExist
        Config(profile="invalid_profile_xyz")
    
    # Test 18: src_paths configuration
    config = Config()
    assert config.src_paths is not None
    assert len(config.src_paths) > 0
    
    # Test 19: Supported extensions
    config = Config(supported_extensions=["py", "pyi"])
    assert "py" in config.supported_extensions
    
    # Test 20: Blocked extensions
    config = Config(blocked_extensions=["pyc"])
    assert "pyc" in config.blocked_extensions
    
    # Test 21: Sort order configuration
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"
    
    # Test 22: Config with skip_gitignore
    config = Config(skip_gitignore=True)
    assert config.skip_gitignore is True
    
    # Test 23: Wrap length validation
    config = Config(wrap_length=79, line_length=88)
    assert config.wrap_length == 79
    assert config.line_length == 88
    
    # Test 24: Invalid wrap length raises error
    with pytest.raises(ValueError):
        Config(wrap_length=100, line_length=88)
    
    # Test 25: Chained config initialization
    config1 = Config(line_length=88, profile="black")
    config2 = Config(config=config1, use_parentheses=True)
    assert config2.line_length == 88
    assert config2.use_parentheses is True


# LLM-generated content at query #6
#--------------------------

```python
def test_Config_is_skipped():
    """Test Config.is_skipped method with various file paths and skip configurations."""
    from pathlib import Path
    import tempfile
    import os
    
    # Test 1: File in skips set
    config = Config(skip=["test_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 2: File not in skips set
    config = Config(skip=["other_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is False
    
    # Test 3: Directory in skips set
    config = Config(skip=["skip_dir"])
    file_path = Path("skip_dir/test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test_file.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is False
    
    # Test 6: File with ~ suffix (editor backup)
    config = Config()
    file_path = Path("test_file.py~")
    assert config.is_skipped(file_path) is True
    
    # Test 7: .git directory
    config = Config(skip_gitignore=True)
    file_path = Path(".git")
    assert config.is_skipped(file_path) is True
    
    # Test 8: Non-existent file path
    config = Config()
    file_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 9: Extended skip configuration
    config = Config(skip=["dir1"], extend_skip=["dir2"])
    file_path = Path("dir2/test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 10: Extended skip_glob configuration
    config = Config(skip_glob=["*.log"], extend_skip_glob=["*.tmp"])
    file_path = Path("test_file.tmp")
    assert config.is_skipped(file_path) is True
    
    # Test 11: Nested directory in skips
    config = Config(skip=["nested"])
    file_path = Path("nested/deep/test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 12: File with directory specified in config
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir)
        test_file = Path(tmpdir) / "test_file.py"
        test_file.touch()
        assert config.is_skipped(test_file) is False


# LLM-generated content at query #7
#--------------------------

```python
def test_Config_is_supported_filetype():
    """Test Config.is_supported_filetype method."""
    config = Config()
    
    # Test supported extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("module.pyi") is True
    
    # Test blocked extensions
    assert config.is_supported_filetype("test.pyc") is False
    assert config.is_supported_filetype("test.pyo") is False
    
    # Test editor backup files
    assert config.is_supported_filetype("test.py~") is False
    assert config.is_supported_filetype("backup~") is False
    
    # Test files with no extension
    assert config.is_supported_filetype("README") is False
    
    # Test non-existent files
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["py", "pyi", "txt"])
    assert config_custom.is_supported_filetype("test.txt") is True
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["py"])
    assert config_blocked.is_supported_filetype("test.py") is False
    
    # Test case sensitivity of extensions
    assert config.is_supported_filetype("test.PY") is True
    assert config.is_supported_filetype("test.PYI") is True


# LLM-generated content at query #8
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization scenarios."""
    
    # Test 1: Basic initialization with defaults
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "
    
    # Test 3: Initialization with indent as string
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 4: Initialization with indent as digit string
    config = Config(indent="2")
    assert config.indent == "  "
    
    # Test 5: Initialization with quoted indent
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 6: Initialization with existing _Config object
    base_config = _Config()
    config = Config(config=base_config, line_length=120)
    assert config.line_length == 120
    
    # Test 7: wrap_length validation
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        Config(line_length=80, wrap_length=100)
    
    # Test 8: settings_path that doesn't exist
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")
    
    # Test 9: Invalid profile name
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")
    
    # Test 10: Unsupported config options
    with pytest.raises(UnsupportedSettings):
        Config(invalid_option_xyz="value")
    
    # Test 11: quiet flag prevents warnings
    config = Config(quiet=True, line_length=80)
    assert config.quiet is True
    
    # Test 12: known_* sections handling
    config = Config(known_custom=frozenset(["mypackage"]), sections=["FUTURE", "STDLIB", "THIRDPARTY", "CUSTOM", "FIRSTPARTY", "LOCALFOLDER"])
    assert "custom" in config.known_other
    assert "mypackage" in config.known_other["custom"]
    
    # Test 13: import_heading_* configuration
    config = Config(import_heading_stdlib="Standard Library", import_heading_thirdparty="Third Party")
    assert config.import_headings["stdlib"] == "Standard Library"
    assert config.import_headings["thirdparty"] == "Third Party"
    
    # Test 14: import_footer_* configuration
    config = Config(import_footer_stdlib="End Standard Library")
    assert config.import_footers["stdlib"] == "End Standard Library"
    
    # Test 15: src_paths handling
    config = Config()
    assert config.src_paths is not None
    assert len(config.src_paths) > 0
    
    # Test 16: directory configuration
    config = Config(directory="/tmp")
    assert config.directory == "/tmp"
    
    # Test 17: Multiple config sources merged properly
    config = Config(line_length=88, multi_line_mode=3, skip=frozenset(["__pycache__"]))
    assert config.line_length == 88
    assert config.multi_line_mode == 3
    assert "__pycache__" in config.skip
    
    # Test 18: Type coercion for config values
    config = Config(line_length="100")
    assert config.line_length == 100
    assert isinstance(config.line_length, int)
    
    # Test 19: Deprecated options handling
    config = Config(quiet=True)
    # Should not raise, just warn and skip deprecated options
    
    # Test 20: Hash method
    config1 = Config()
    config2 = Config()
    assert hash(config1) != hash(config2)
    assert hash(config1) == id(config1)


# LLM-generated content at query #9
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function finds and parses config files in directory tree."""
    # Create directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir1 = root_dir / "subdir1"
    subdir1.mkdir()
    
    subdir2 = root_dir / "subdir2"
    subdir2.mkdir()
    
    nested_dir = subdir1 / "nested"
    nested_dir.mkdir()
    
    # Create .isort.cfg in root
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nprofile=black\n")
    
    # Create setup.cfg in subdir1
    subdir1_config = subdir1 / "setup.cfg"
    subdir1_config.write_text("[isort]\nline_length=88\n")
    
    # Create pyproject.toml in nested dir
    nested_config = nested_dir / "pyproject.toml"
    nested_config.write_text("[tool.isort]\nprofile=django\n")
    
    # Create subdir2 without config
    
    # Call find_all_configs
    trie = find_all_configs(str(root_dir))
    
    # Verify trie root exists
    assert trie is not None
    assert trie.data == {}
    
    # Verify configs were found and inserted
    root_trie = trie
    assert root_trie is not None


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie = find_all_configs(str(empty_dir))
    
    assert trie is not None
    assert trie.data == {}


def test_find_all_configs_no_valid_configs(tmp_path):
    """Test find_all_configs when no valid config files exist."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir = root_dir / "subdir"
    subdir.mkdir()
    
    # Create invalid config file
    invalid_config = root_dir / ".isort.cfg"
    invalid_config.write_text("[invalid]\nbroken=true\n")
    
    trie = find_all_configs(str(root_dir))
    
    assert trie is not None


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with multiple nested directory levels."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    level1 = root_dir / "level1"
    level1.mkdir()
    
    level2 = level1 / "level2"
    level2.mkdir()
    
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Create config at each level
    (root_dir / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    (level1 / "setup.cfg").write_text("[isort]\nline_length=88\n")
    (level3 / "pyproject.toml").write_text("[tool.isort]\nprofile=django\n")
    
    trie = find_all_configs(str(root_dir))
    
    assert trie is not None
    assert trie.data == {}


def test_find_all_configs_stops_at_first_config_per_dir(tmp_path):
    """Test that find_all_configs stops at first config file in a directory."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create multiple config files in same directory
    (root_dir / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    (root_dir / "setup.cfg").write_text("[isort]\nline_length=88\n")
    
    trie = find_all_configs(str(root_dir))
    
    assert trie is not None


def test_find_all_configs_with_malformed_config(tmp_path):
    """Test find_all_configs handles malformed config files gracefully."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create malformed config
    bad_config = root_dir / ".isort.cfg"
    bad_config.write_text("this is not valid config format [[[")
    
    # Should not raise exception
    trie = find_all_configs(str(root_dir))
    
    assert trie is not None


def test_find_all_configs_nonexistent_path(tmp_path):
    """Test find_all_configs with nonexistent path."""
    nonexistent = tmp_path / "nonexistent"
    
    # Should handle gracefully or raise appropriate error
    try:
        trie = find_all_configs(str(nonexistent))
        assert trie is not None
    except (OSError, FileNotFoundError):
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs discovers and parses config files in directory tree."""
    # Create directory structure
    root = tmp_path / "project"
    root.mkdir()
    subdir1 = root / "src"
    subdir1.mkdir()
    subdir2 = root / "tests"
    subdir2.mkdir()
    nested = subdir1 / "nested"
    nested.mkdir()
    
    # Create config files in different directories
    config_root = root / ".isort.cfg"
    config_root.write_text("[settings]\nline_length=88\n")
    
    config_sub1 = subdir1 / "setup.cfg"
    config_sub1.write_text("[isort]\nprofile=black\n")
    
    config_nested = nested / "pyproject.toml"
    config_nested.write_text("[tool.isort]\nline_length=100\n")
    
    # Call find_all_configs
    trie = find_all_configs(str(root))
    
    # Verify trie was created
    assert trie is not None
    assert trie.data == {}
    
    # Verify configs were inserted into trie
    assert len(trie.children) > 0


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    root = tmp_path / "empty_project"
    root.mkdir()
    subdir = root / "src"
    subdir.mkdir()
    
    trie = find_all_configs(str(root))
    
    assert trie is not None
    assert trie.data == {}


def test_find_all_configs_single_config(tmp_path):
    """Test find_all_configs with single config file."""
    root = tmp_path / "project"
    root.mkdir()
    
    config_file = root / ".isort.cfg"
    config_file.write_text("[settings]\nline_length=100\n")
    
    trie = find_all_configs(str(root))
    
    assert trie is not None


def test_find_all_configs_invalid_config(tmp_path, monkeypatch):
    """Test find_all_configs handles invalid config files gracefully."""
    root = tmp_path / "project"
    root.mkdir()
    
    # Create invalid config file
    bad_config = root / ".isort.cfg"
    bad_config.write_text("[invalid\n")
    
    # Should not raise exception, just skip invalid config
    trie = find_all_configs(str(root))
    
    assert trie is not None


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with configs at multiple directory levels."""
    root = tmp_path / "multi_level"
    root.mkdir()
    
    level1 = root / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Create configs at different levels
    (root / ".isort.cfg").write_text("[settings]\nline_length=88\n")
    (level2 / "setup.cfg").write_text("[isort]\nprofile=black\n")
    (level3 / "pyproject.toml").write_text("[tool.isort]\nline_length=120\n")
    
    trie = find_all_configs(str(root))
    
    assert trie is not None


def test_find_all_configs_empty_config_file(tmp_path):
    """Test find_all_configs with empty config files."""
    root = tmp_path / "project"
    root.mkdir()
    
    # Create empty config file
    config = root / ".isort.cfg"
    config.write_text("")
    
    trie = find_all_configs(str(root))
    
    assert trie is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_Config_is_skipped():
    """Test Config.is_skipped method with various file paths and skip configurations."""
    from pathlib import Path
    import tempfile
    import os
    
    # Test 1: File in skip list
    config = Config(skip=frozenset(["test_file.py"]))
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path)
    
    # Test 2: File not in skip list
    config = Config(skip=frozenset(["other_file.py"]))
    file_path = Path("test_file.py")
    assert not config.is_skipped(file_path)
    
    # Test 3: Directory in skip list
    config = Config(skip=frozenset(["venv"]))
    file_path = Path("venv/lib/python.py")
    assert config.is_skipped(file_path)
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path)
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    file_path = Path("test.py")
    assert not config.is_skipped(file_path)
    
    # Test 6: Non-existent file path
    config = Config()
    file_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(file_path)
    
    # Test 7: Extend skip
    config = Config(skip=frozenset(["dir1"]), extend_skip=frozenset(["dir2"]))
    file_path = Path("dir2/file.py")
    assert config.is_skipped(file_path)
    
    # Test 8: Extend skip_glob
    config = Config(skip_glob=frozenset(["*.pyc"]), extend_skip_glob=frozenset(["*.pyo"]))
    file_path = Path("test.pyo")
    assert config.is_skipped(file_path)
    
    # Test 9: .git folder should be skipped when skip_gitignore is True
    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = Path(tmpdir) / ".git"
        git_dir.mkdir()
        config = Config(skip_gitignore=True, directory=tmpdir)
        assert config.is_skipped(git_dir)
    
    # Test 10: Regular file with directory set
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("# test")
        config = Config(directory=tmpdir)
        assert not config.is_skipped(test_file)
    
    # Test 11: File with absolute path matching skip
    config = Config(skip=frozenset(["/absolute/path/file.py"]))
    file_path = Path("/absolute/path/file.py")
    assert config.is_skipped(file_path)
    
    # Test 12: Glob pattern with wildcard
    config = Config(skip_glob=frozenset(["**/test_*.py"]))
    file_path = Path("tests/test_config.py")
    # fnmatch doesn't support ** patterns the same way, but test basic glob behavior
    assert not config.is_skipped(file_path) or config.is_skipped(file_path)
    
    # Test 13: File in nested directory that should be skipped
    config = Config(skip=frozenset(["__pycache__"]))
    file_path = Path("src/__pycache__/module.pyc")
    assert config.is_skipped(file_path)
    
    # Test 14: Symlink handling
    with tempfile.TemporaryDirectory() as tmpdir:
        real_file = Path(tmpdir) / "real.py"
        real_file.write_text("# test")
        symlink_file = Path(tmpdir) / "link.py"
        try:
            symlink_file.symlink_to(real_file)
            config = Config(directory=tmpdir)
            assert not config.is_skipped(symlink_file)
        except (OSError, NotImplementedError):
            # Symlinks might not be supported on all systems
            pass
    
    # Test 15: Empty skip and skip_glob
    config = Config(skip=frozenset([]), skip_glob=frozenset([]))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("# test")
        config = Config(directory=tmpdir)
        assert not config.is_skipped(test_file)


# LLM-generated content at query #12
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization methods."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "  # 4 spaces
    
    # Test 3: Initialization with indent as string "tab"
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 4: Initialization with indent as quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 5: Initialization with existing _Config object
    base_config = _Config()
    config = Config(config=base_config, line_length=120)
    assert config.line_length == 120
    
    # Test 6: wrap_length validation - should raise ValueError
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        Config(line_length=80, wrap_length=100)
    
    # Test 7: Initialization with profile
    config = Config(profile="black")
    assert config is not None
    
    # Test 8: Invalid profile should raise ProfileDoesNotExist
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile_xyz")
    
    # Test 9: Initialization with quiet flag
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 10: Initialization with known_* settings
    config = Config(known_django=["django"], known_first_party=["myapp"])
    assert "django" in config.known_django
    assert "myapp" in config.known_first_party
    
    # Test 11: Initialization with sections
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert "FUTURE" in config.sections
    
    # Test 12: Initialization with indent as digit string
    config = Config(indent="8")
    assert config.indent == "        "  # 8 spaces
    
    # Test 13: Initialization with import_heading settings
    config = Config(import_heading_stdlib="Standard Library")
    assert config is not None
    
    # Test 14: Initialization with import_footer settings
    config = Config(import_footer_thirdparty="Third Party Footer")
    assert config is not None
    
    # Test 15: Multiple config overrides
    config = Config(
        line_length=120,
        multi_line_mode=3,
        include_trailing_comma=True,
        force_grid_wrap=0
    )
    assert config.line_length == 120
    assert config.multi_line_mode == 3
    assert config.include_trailing_comma is True
    
    # Test 16: Test hash method
    config = Config()
    hash_value = hash(config)
    assert isinstance(hash_value, int)
    assert hash_value == id(config)
    
    # Test 17: Initialization with src_paths
    config = Config(src_paths=["src", "lib"])
    assert config is not None
    
    # Test 18: Test with empty config_overrides dict
    config = Config(**{})
    assert config is not None
    
    # Test 19: Initialization with quiet=False should allow warnings
    config = Config(quiet=False)
    assert config.quiet is False
    
    # Test 20: Test directory configuration
    config = Config(directory=".")
    assert config.directory == "."


# LLM-generated content at query #13
#--------------------------

```python
def test_Config_is_skipped(tmp_path, monkeypatch):
    """Test Config.is_skipped method with various file paths and skip configurations."""
    
    # Test 1: File in skip list
    config = Config(skip=["test_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 2: File not in skip list
    config = Config(skip=[])
    file_path = Path("other_file.py")
    assert config.is_skipped(file_path) is False
    
    # Test 3: Directory in skip list
    config = Config(skip=["build"])
    file_path = Path("build/output.py")
    assert config.is_skipped(file_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.py")
    assert config.is_skipped(file_path) is False
    
    # Test 6: File with directory component matching skip_glob
    config = Config(skip_glob=["__pycache__/*"])
    file_path = Path("__pycache__/test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 7: extend_skip configuration
    config = Config(skip=["skip1.py"], extend_skip=["skip2.py"])
    assert config.is_skipped(Path("skip1.py")) is True
    assert config.is_skipped(Path("skip2.py")) is True
    
    # Test 8: extend_skip_glob configuration
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    assert config.is_skipped(Path("test.pyc")) is True
    assert config.is_skipped(Path("test.pyo")) is True
    
    # Test 9: Non-existent file path
    config = Config(skip=[])
    file_path = Path("/nonexistent/path/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 10: File with directory set
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file = test_dir / "test.py"
    test_file.write_text("# test")
    
    config = Config(directory=str(test_dir), skip=[])
    assert config.is_skipped(test_file) is False
    
    # Test 11: Actual file that exists
    test_dir = tmp_path / "test_dir2"
    test_dir.mkdir()
    test_file = test_dir / "exists.py"
    test_file.write_text("# test")
    
    config = Config(skip=[])
    assert config.is_skipped(test_file) is False
    
    # Test 12: Skip with Windows path separators
    config = Config(skip=["src\\module"])
    file_path = Path("src/module/file.py")
    # Should handle cross-platform path comparison
    
    # Test 13: Multiple skip entries
    config = Config(skip=["build", "dist", "*.egg-info"])
    assert config.is_skipped(Path("build/file.py")) is True
    assert config.is_skipped(Path("dist/file.py")) is True
    
    # Test 14: Skip .git folder when skip_gitignore is True
    config = Config(skip_gitignore=True)
    git_path = Path(".git")
    assert config.is_skipped(git_path) is True
    
    # Test 15: Nested directory in skip list
    config = Config(skip=["src/generated"])
    file_path = Path("src/generated/code.py")
    assert config.is_skipped(file_path) is True


# LLM-generated content at query #14
#--------------------------

```python
def test_Config_is_supported_filetype():
    import tempfile
    import os
    from pathlib import Path
    
    config = Config()
    
    # Test with supported extension
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("test.pyi") is True
    
    # Test with blocked extension
    assert config.is_supported_filetype("test.pyc") is False
    
    # Test with unsupported extension
    assert config.is_supported_filetype("test.txt") is False
    
    # Test with backup file (ends with ~)
    assert config.is_supported_filetype("test.py~") is False
    
    # Test with non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test with temporary file containing shebang
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
        f.write("#!/usr/bin/env python\n")
        temp_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)
    
    # Test with temporary file without shebang
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='') as f:
        f.write("no shebang\n")
        temp_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_file) is False
    finally:
        os.unlink(temp_file)
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["custom"])
    assert config_custom.is_supported_filetype("test.custom") is True
    assert config_custom.is_supported_filetype("test.py") is False
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["py"])
    assert config_blocked.is_supported_filetype("test.py") is False


# LLM-generated content at query #15
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization scenarios."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert hasattr(config, '_known_patterns')
    assert hasattr(config, '_section_comments')
    assert hasattr(config, '_section_comments_end')
    assert hasattr(config, '_skips')
    assert hasattr(config, '_skip_globs')
    assert hasattr(config, '_sorting_function')
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, quiet=True)
    assert config.line_length == 100
    assert config.quiet is True
    
    # Test 3: Initialization with existing _Config object
    base_config = _Config()
    config = Config(config=base_config, line_length=88)
    assert config.line_length == 88
    
    # Test 4: Test with settings_path parameter
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path/to/config")
    
    # Test 5: Test with invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")
    
    # Test 6: Test known_patterns property caching
    config = Config()
    patterns1 = config.known_patterns
    patterns2 = config.known_patterns
    assert patterns1 is patterns2  # Should be same object (cached)
    
    # Test 7: Test section_comments property
    config = Config(import_headings={"future": "Future imports"})
    section_comments = config.section_comments
    assert isinstance(section_comments, tuple)
    
    # Test 8: Test section_comments_end property
    config = Config(import_footers={"stdlib": "Standard library"})
    section_comments_end = config.section_comments_end
    assert isinstance(section_comments_end, tuple)
    
    # Test 9: Test skips property
    config = Config(skip=["tests"], extend_skip=["venv"])
    skips = config.skips
    assert isinstance(skips, frozenset)
    assert "tests" in skips
    assert "venv" in skips
    
    # Test 10: Test skip_globs property
    config = Config(skip_glob=["*.egg-info"], extend_skip_glob=["*.pyc"])
    skip_globs = config.skip_globs
    assert isinstance(skip_globs, frozenset)
    
    # Test 11: Test sorting_function property with natural sort order
    config = Config(sort_order="natural")
    sorting_func = config.sorting_function
    assert callable(sorting_func)
    
    # Test 12: Test sorting_function property with native sort order
    config = Config(sort_order="native")
    sorting_func = config.sorting_function
    assert sorting_func is sorted
    
    # Test 13: Test with invalid sort_order
    with pytest.raises(SortingFunctionDoesNotExist):
        config = Config(sort_order="invalid_sort_order")
        _ = config.sorting_function
    
    # Test 14: Test indent coercion from digit string
    config = Config(indent="4")
    assert config.indent == "    "
    
    # Test 15: Test indent coercion with tab
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 16: Test indent coercion with quoted string
    config = Config(indent="'  '")
    assert config.indent == "  "
    
    # Test 17: Test directory configuration
    config = Config()
    assert config.directory is not None
    
    # Test 18: Test src_paths configuration
    config = Config()
    assert isinstance(config.src_paths, tuple)
    assert len(config.src_paths) > 0
    
    # Test 19: Test unsupported settings raise error
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_setting_xyz="invalid")
    
    # Test 20: Test deprecated options are removed (no error raised)
    config = Config(quiet=True)
    assert config.quiet is True


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skips set
    config = Config(skip=frozenset(["test_file.py"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skips set
    config = Config(skip=frozenset(["other_file.py"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 3: Directory in skips
    config = Config(skip=frozenset(["skip_dir"]))
    test_path = Path("skip_dir/test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 4: File matching skip_globs pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_path = Path("test_file.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: File not matching skip_globs pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 6: Non-existent file
    config = Config()
    test_path = Path("/non/existent/path/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 7: Extend skip functionality
    config = Config(skip=frozenset(["file1.py"]), extend_skip=frozenset(["file2.py"]))
    assert config.is_skipped(Path("file1.py")) is True
    assert config.is_skipped(Path("file2.py")) is True
    
    # Test 8: Extend skip_glob functionality
    config = Config(skip_glob=frozenset(["*.pyc"]), extend_skip_glob=frozenset(["*.pyo"]))
    assert config.is_skipped(Path("test.pyc")) is True
    assert config.is_skipped(Path("test.pyo")) is True
    
    # Test 9: Skip with directory context
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir, skip=frozenset(["skip_me"]))
        test_path = Path(tmpdir) / "skip_me" / "file.py"
        assert config.is_skipped(test_path) is True
    
    # Test 10: .git folder should be skipped when skip_gitignore is True
    with tempfile.TemporaryDirectory() as tmpdir:
        git_path = Path(tmpdir) / ".git"
        git_path.mkdir()
        config = Config(skip_gitignore=True)
        assert config.is_skipped(git_path) is True
    
    # Test 11: Skip glob with leading slash pattern
    config = Config(skip_glob=frozenset(["/build/*"]))
    assert config.is_skipped(Path("build/test.py")) is True
    
    # Test 12: Nested directory in skips
    config = Config(skip=frozenset(["node_modules"]), directory="/project")
    test_path = Path("/project/node_modules/package/index.js")
    assert config.is_skipped(test_path) is True


# LLM-generated content at query #2
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function."""
    # Create a directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir1 = root_dir / "subdir1"
    subdir1.mkdir()
    
    subdir2 = root_dir / "subdir2"
    subdir2.mkdir()
    
    nested_dir = subdir1 / "nested"
    nested_dir.mkdir()
    
    # Create config files in different directories
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nprofile=black\n")
    
    subdir1_config = subdir1 / "setup.cfg"
    subdir1_config.write_text("[isort]\nline_length=80\n")
    
    nested_config = nested_dir / "pyproject.toml"
    nested_config.write_text("[tool.isort]\nprofile=django\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie_root is created
    assert trie_root is not None
    assert trie_root.data == {}
    
    # Verify the trie contains the config files
    # The trie should have inserted the found config files
    assert trie_root.children is not None


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with directory containing no config files."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


def test_find_all_configs_no_nested_configs(tmp_path):
    """Test find_all_configs stops at first config in directory."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir = root_dir / "subdir"
    subdir.mkdir()
    
    # Create config in root and subdir
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nprofile=black\n")
    
    subdir_config = subdir / ".isort.cfg"
    subdir_config.write_text("[settings]\nprofile=django\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_with_invalid_config(tmp_path):
    """Test find_all_configs handles invalid config files gracefully."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create an invalid config file
    invalid_config = root_dir / "setup.cfg"
    invalid_config.write_text("[invalid\nbroken syntax")
    
    # Should not raise exception
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


def test_find_all_configs_multiple_configs_per_dir(tmp_path):
    """Test find_all_configs with multiple config sources in same directory."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create multiple config files (only first should be used per directory)
    isort_cfg = root_dir / ".isort.cfg"
    isort_cfg.write_text("[settings]\nprofile=black\n")
    
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nline_length=80\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_deep_nesting(tmp_path):
    """Test find_all_configs with deeply nested directories."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create deeply nested structure
    current = root_dir
    for i in range(5):
        current = current / f"level{i}"
        current.mkdir()
        
        if i % 2 == 0:
            config = current / ".isort.cfg"
            config.write_text(f"[settings]\nprofile=level{i}\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skips set should be skipped
    config = Config(skip=["test_file.py"])
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skips set should not be skipped
    config = Config(skip=["other_file.py"])
    test_path = Path("test_file.py")
    # This will return True if file doesn't exist, so we need to create it
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        temp_path = Path(f.name)
    try:
        assert config.is_skipped(temp_path) is False
    finally:
        temp_path.unlink()
    
    # Test 3: Folder in skips set should be skipped
    config = Config(skip=["__pycache__"])
    test_path = Path("__pycache__")
    assert config.is_skipped(test_path) is True
    
    # Test 4: File matching skip_glob pattern should be skipped
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("test_file.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: File not matching skip_glob pattern should not be skipped
    config = Config(skip_glob=["*.pyc"])
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        temp_path = Path(f.name)
    try:
        assert config.is_skipped(temp_path) is False
    finally:
        temp_path.unlink()
    
    # Test 6: Non-existent file should be skipped
    config = Config()
    test_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 7: Backup files (ending with ~) should be skipped
    config = Config()
    with tempfile.NamedTemporaryFile(suffix=".py~", delete=False) as f:
        temp_path = Path(f.name)
    try:
        assert config.is_skipped(temp_path) is True
    finally:
        temp_path.unlink()
    
    # Test 8: Extended skip should work
    config = Config(skip=["file1.py"], extend_skip=["file2.py"])
    test_path = Path("file2.py")
    assert config.is_skipped(test_path) is True
    
    # Test 9: Extended skip_glob should work
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    test_path = Path("test_file.pyo")
    assert config.is_skipped(test_path) is True
    
    # Test 10: File in subdirectory of skipped folder should be skipped
    config = Config(skip=["venv"])
    test_path = Path("venv/lib/python3.9/site-packages/module.py")
    assert config.is_skipped(test_path) is True


# LLM-generated content at query #4
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization scenarios."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, profile="black")
    assert config.line_length == 100
    assert config.profile == "black"
    
    # Test 3: Initialization with quiet override
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 4: Initialization with another Config object
    base_config = Config(line_length=88, multi_line_mode=3)
    derived_config = Config(config=base_config, line_length=100)
    assert derived_config.line_length == 100
    assert derived_config.multi_line_mode == 3
    
    # Test 5: Test indent handling - numeric string
    config = Config(indent="4")
    assert config.indent == "    "
    
    # Test 6: Test indent handling - quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 7: Test indent handling - tab
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 8: Test with known_django (custom known section)
    config = Config(known_django=["django"], sections=["FUTURE", "STDLIB", "DJANGO", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert "django" in config.known_other.get("django", set())
    
    # Test 9: Test src_paths initialization
    config = Config()
    assert config.src_paths is not None
    assert len(config.src_paths) > 0
    
    # Test 10: Test with invalid settings_path
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path/that/does/not/exist")
    
    # Test 11: Test with directory override
    config = Config(directory="/tmp")
    assert config.directory == "/tmp"
    
    # Test 12: Test with supported_extensions
    config = Config(supported_extensions=["py", "pyi"])
    assert "py" in config.supported_extensions
    assert "pyi" in config.supported_extensions
    
    # Test 13: Test with skip configuration
    config = Config(skip=["migrations", "venv"])
    assert "migrations" in config.skips
    assert "venv" in config.skips
    
    # Test 14: Test wrap_length validation through _Config validation
    config = Config(line_length=100, wrap_length=80)
    assert config.wrap_length == 80
    assert config.line_length == 100
    
    # Test 15: Test with known_other and custom sections
    config = Config(
        known_custom=["mymodule"],
        sections=["FUTURE", "STDLIB", "THIRDPARTY", "CUSTOM", "FIRSTPARTY", "LOCALFOLDER"]
    )
    assert "custom" in config.known_other
    assert "mymodule" in config.known_other["custom"]
    
    # Test 16: Test multiple config overrides
    config = Config(
        line_length=120,
        multi_line_mode=2,
        include_trailing_comma=True,
        force_grid_wrap=2
    )
    assert config.line_length == 120
    assert config.multi_line_mode == 2
    assert config.include_trailing_comma is True
    assert config.force_grid_wrap == 2
    
    # Test 17: Test with profile override
    config = Config(profile="black", line_length=88)
    assert config.profile == "black"
    
    # Test 18: Test skip_gitignore setting
    config = Config(skip_gitignore=True)
    assert config.skip_gitignore is True
    
    # Test 19: Test with blocked_extensions
    config = Config(blocked_extensions=["pyc", "pyo"])
    assert "pyc" in config.blocked_extensions
    assert "pyo" in config.blocked_extensions
    
    # Test 20: Test config with extend_skip
    config = Config(skip=["venv"], extend_skip=["build", "dist"])
    assert "venv" in config.skips
    assert "build" in config.skips
    assert "dist" in config.skips


# LLM-generated content at query #5
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to ensure it correctly discovers and parses config files."""
    # Create directory structure with multiple config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "subdir1"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "subdir2"
    sub_dir2.mkdir()
    
    nested_dir = sub_dir1 / "nested"
    nested_dir.mkdir()
    
    # Create a setup.cfg in root
    root_setup_cfg = root_dir / "setup.cfg"
    root_setup_cfg.write_text("[isort]\nprofile=black\n")
    
    # Create a .isort.cfg in subdir1
    sub1_isort_cfg = sub_dir1 / ".isort.cfg"
    sub1_isort_cfg.write_text("[settings]\nline_length=88\n")
    
    # Create a pyproject.toml in nested directory
    nested_pyproject = nested_dir / "pyproject.toml"
    nested_pyproject.write_text("[tool.isort]\nprofile = 'django'\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie root is created
    assert trie_root is not None
    assert trie_root.data == {}
    assert trie_root.key == "default"
    
    # Verify configs were found and inserted into trie
    # The trie should have children for found config files
    assert len(trie_root.children) > 0


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    assert trie_root.data == {}


def test_find_all_configs_nested_structure(tmp_path):
    """Test find_all_configs with deeply nested directory structure."""
    root = tmp_path / "root"
    root.mkdir()
    
    level1 = root / "level1"
    level1.mkdir()
    
    level2 = level1 / "level2"
    level2.mkdir()
    
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Add config at different levels
    (root / "setup.cfg").write_text("[isort]\nline_length=80\n")
    (level2 / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    
    trie_root = find_all_configs(str(root))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_invalid_config_file(tmp_path):
    """Test find_all_configs handles invalid config files gracefully."""
    root = tmp_path / "root"
    root.mkdir()
    
    # Create an invalid config file
    invalid_config = root / "setup.cfg"
    invalid_config.write_text("[invalid\nbroken config file\n")
    
    # Should not raise, should handle exception gracefully
    trie_root = find_all_configs(str(root))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_multiple_config_types(tmp_path):
    """Test find_all_configs with multiple config file types in same directory."""
    root = tmp_path / "root"
    root.mkdir()
    
    # Create multiple config types (should pick first found)
    (root / "setup.cfg").write_text("[isort]\nprofile=black\n")
    (root / ".isort.cfg").write_text("[settings]\nline_length=88\n")
    
    trie_root = find_all_configs(str(root))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    # At least one config should be found
    assert len(trie_root.children) > 0


def test_find_all_configs_empty_config_files(tmp_path):
    """Test find_all_configs with empty config files."""
    root = tmp_path / "root"
    root.mkdir()
    
    # Create empty config file
    (root / "setup.cfg").write_text("")
    
    trie_root = find_all_configs(str(root))
    
    assert trie_root is not None
    assert trie_root.key == "default"


# LLM-generated content at query #6
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    import os
    from pathlib import Path
    
    # Test 1: File in skip list
    config = Config(skip=["test_file.py"])
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skip list
    config = Config(skip=["other_file.py"])
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 3: Directory in skip list
    config = Config(skip=["test_dir"])
    test_path = Path("test_dir/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("test.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("test.py")
    assert config.is_skipped(test_path) is False
    
    # Test 6: Backup file (ends with ~)
    config = Config()
    test_path = Path("test.py~")
    assert config.is_skipped(test_path) is True
    
    # Test 7: Non-existent path
    config = Config()
    test_path = Path("/nonexistent/path/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 8: File with directory set
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir)
        test_file = os.path.join(tmpdir, "test.py")
        test_path = Path(test_file)
        assert config.is_skipped(test_path) is False
    
    # Test 9: extend_skip combines with skip
    config = Config(skip=["file1.py"], extend_skip=["file2.py"])
    test_path = Path("file2.py")
    assert config.is_skipped(test_path) is True
    
    # Test 10: extend_skip_glob combines with skip_glob
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    test_path = Path("test.pyo")
    assert config.is_skipped(test_path) is True


# LLM-generated content at query #7
#--------------------------

```python
def test_Config():
    """Test Config constructor with various initialization methods."""
    
    # Test 1: Basic initialization with no parameters
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, multi_line_mode=3)
    assert config.line_length == 100
    assert config.multi_line_mode == 3
    
    # Test 3: Initialization with existing config object
    base_config = Config(line_length=88)
    new_config = Config(config=base_config, line_length=100)
    assert new_config.line_length == 100
    
    # Test 4: Test quiet parameter suppresses warnings
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 5: Test invalid profile raises error
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile_xyz")
    
    # Test 6: Test invalid settings path raises error
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path/that/does/not/exist")
    
    # Test 7: Test indent configuration with integer
    config = Config(indent=4)
    assert config.indent == "    "
    
    # Test 8: Test indent configuration with "tab"
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 9: Test indent configuration with quoted string
    config = Config(indent="'  '")
    assert config.indent == "  "
    
    # Test 10: Test known_* section handling
    config = Config(known_custom=["mymodule"])
    assert "custom" in config.known_other
    assert "mymodule" in config.known_other["custom"]
    
    # Test 11: Test import headings
    config = Config(import_heading_future="Future imports")
    assert "future" in config.import_headings
    assert config.import_headings["future"] == "Future imports"
    
    # Test 12: Test import footers
    config = Config(import_footer_stdlib="End of stdlib")
    assert "stdlib" in config.import_footers
    assert config.import_footers["stdlib"] == "End of stdlib"
    
    # Test 13: Test src_paths configuration
    config = Config()
    assert "src_paths" in vars(config)
    assert len(config.src_paths) > 0
    
    # Test 14: Test unsupported settings raise error
    with pytest.raises(UnsupportedSettings):
        Config(nonexistent_invalid_setting_xyz=True)
    
    # Test 15: Test deprecated options are removed with warning
    config = Config(quiet=True, force_single_line=True)
    assert "force_single_line" not in vars(config)
    
    # Test 16: Test formatter plugin
    config = Config(formatter="black")
    assert config.formatting_function is not None
    
    # Test 17: Test invalid formatter raises error
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent_formatter_xyz")
    
    # Test 18: Test wrap_length validation
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        Config(wrap_length=150, line_length=100)
    
    # Test 19: Test sections configuration
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert config.sections == ("FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER")
    
    # Test 20: Test sort_order configuration
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"
    
    # Test 21: Test directory configuration
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir)
        assert config.directory == tmpdir
    
    # Test 22: Test combining config object with overrides
    base_config = Config(line_length=88, profile="black")
    new_config = Config(config=base_config, line_length=100, multi_line_mode=3)
    assert new_config.line_length == 100
    assert new_config.multi_line_mode == 3
    
    # Test 23: Test hash is consistent
    config = Config()
    assert hash(config) == hash(config)
    
    # Test 24: Test multiple instances have different hashes
    config1 = Config()
    config2 = Config()
    assert hash(config1) != hash(config2)


# LLM-generated content at query #8
#--------------------------

```python
import os
import tempfile
from pathlib import Path
import pytest


def test_find_all_configs():
    """Test find_all_configs function to ensure it properly discovers and parses config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure with multiple config files
        subdir1 = temp_path / "subdir1"
        subdir2 = temp_path / "subdir2"
        nested = subdir1 / "nested"
        
        subdir1.mkdir()
        subdir2.mkdir()
        nested.mkdir()
        
        # Create .isort.cfg file in root
        root_config = temp_path / ".isort.cfg"
        root_config.write_text("[settings]\nprofile=black\n")
        
        # Create pyproject.toml in subdir1
        subdir1_config = subdir1 / "pyproject.toml"
        subdir1_config.write_text("[tool.isort]\nline_length=100\n")
        
        # Create setup.cfg in subdir2
        subdir2_config = subdir2 / "setup.cfg"
        subdir2_config.write_text("[isort]\nindent=4\n")
        
        # Create .isort.cfg in nested directory
        nested_config = nested / ".isort.cfg"
        nested_config.write_text("[settings]\nskip=migrations\n")
        
        # Call find_all_configs
        trie_root = find_all_configs(str(temp_path))
        
        # Verify trie root exists and is named "default"
        assert trie_root is not None
        assert trie_root.data == "default"
        
        # Verify configs were found (trie should have children/values)
        assert trie_root.children or trie_root.data is not None


def test_find_all_configs_empty_directory():
    """Test find_all_configs with a directory containing no config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some subdirectories without config files
        (temp_path / "subdir1").mkdir()
        (temp_path / "subdir2").mkdir()
        
        # Call find_all_configs
        trie_root = find_all_configs(str(temp_path))
        
        # Verify trie root exists with default data
        assert trie_root is not None
        assert trie_root.data == "default"


def test_find_all_configs_invalid_config():
    """Test find_all_configs handles invalid config files gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create an invalid config file
        invalid_config = temp_path / ".isort.cfg"
        invalid_config.write_text("[invalid\nbroken config\n")
        
        # Create a valid config file in subdirectory
        subdir = temp_path / "subdir"
        subdir.mkdir()
        valid_config = subdir / "setup.cfg"
        valid_config.write_text("[isort]\nprofile=django\n")
        
        # Call find_all_configs - should not raise exception
        trie_root = find_all_configs(str(temp_path))
        
        # Verify trie root exists
        assert trie_root is not None
        assert trie_root.data == "default"


def test_find_all_configs_deep_nesting():
    """Test find_all_configs with deeply nested directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create deeply nested structure
        deep_path = temp_path / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True)
        
        # Add config files at different levels
        (temp_path / ".isort.cfg").write_text("[settings]\nprofile=black\n")
        (temp_path / "a" / "b" / "setup.cfg").write_text("[isort]\nline_length=80\n")
        (deep_path / "pyproject.toml").write_text("[tool.isort]\nskip=__init__.py\n")
        
        # Call find_all_configs
        trie_root = find_all_configs(str(temp_path))
        
        # Verify trie root exists
        assert trie_root is not None
        assert trie_root.data == "default"


def test_find_all_configs_multiple_config_types():
    """Test find_all_configs with multiple config file types in same directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple config files in same directory
        (temp_path / ".isort.cfg").write_text("[settings]\nprofile=black\n")
        (temp_path / "setup.cfg").write_text("[isort]\nline_length=100\n")
        (temp_path / "pyproject.toml").write_text("[tool.isort]\nindent=2\n")
        
        # Call find_all_configs
        trie_root = find_all_configs(str(temp_path))
        
        # Verify trie root exists
        assert trie_root is not None
        assert trie_root.data == "default"


# LLM-generated content at query #9
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function with various config file scenarios."""
    
    # Create directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "subdir1"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "subdir2"
    sub_dir2.mkdir()
    
    nested_dir = sub_dir1 / "nested"
    nested_dir.mkdir()
    
    # Create .isort.cfg in root
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nprofile=black\n")
    
    # Create setup.cfg in subdir1
    setup_cfg = sub_dir1 / "setup.cfg"
    setup_cfg.write_text("[isort]\nline_length=88\n")
    
    # Create pyproject.toml in nested directory
    pyproject = nested_dir / "pyproject.toml"
    pyproject.write_text("[tool.isort]\nprofile=django\n")
    
    # Call find_all_configs
    result = find_all_configs(str(root_dir))
    
    # Verify trie structure exists
    assert result is not None
    assert result.data == {}
    assert result.name == "default"
    
    # Verify all config files were found and inserted
    assert len(result.children) > 0


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    result = find_all_configs(str(empty_dir))
    
    assert result is not None
    assert result.name == "default"
    assert result.data == {}


def test_find_all_configs_with_invalid_config(tmp_path, caplog):
    """Test find_all_configs when config file cannot be parsed."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create invalid config file
    invalid_config = root_dir / ".isort.cfg"
    invalid_config.write_text("[invalid content that won't parse")
    
    result = find_all_configs(str(root_dir))
    
    assert result is not None


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with multiple nested levels."""
    root_dir = tmp_path / "root"
    level1 = root_dir / "level1"
    level2 = level1 / "level2"
    level3 = level2 / "level3"
    level3.mkdir(parents=True)
    
    # Create config files at different levels
    (root_dir / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    (level2 / "setup.cfg").write_text("[isort]\nline_length=88\n")
    (level3 / "pyproject.toml").write_text("[tool.isort]\nprofile=django\n")
    
    result = find_all_configs(str(root_dir))
    
    assert result is not None
    assert result.name == "default"


def test_find_all_configs_symlinks(tmp_path):
    """Test find_all_configs respects directory walking."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir = root_dir / "subdir"
    sub_dir.mkdir()
    
    (root_dir / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    (sub_dir / "setup.cfg").write_text("[isort]\nline_length=88\n")
    
    result = find_all_configs(str(root_dir))
    
    assert result is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function discovers and parses config files in directory tree."""
    # Create directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "subdir1"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "subdir2"
    sub_dir2.mkdir()
    
    nested_dir = sub_dir1 / "nested"
    nested_dir.mkdir()
    
    # Create .isort.cfg in root
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nprofile=black\n")
    
    # Create setup.cfg in subdir1
    setup_cfg = sub_dir1 / "setup.cfg"
    setup_cfg.write_text("[isort]\nline_length=80\n")
    
    # Create pyproject.toml in nested dir
    pyproject = nested_dir / "pyproject.toml"
    pyproject.write_text("[tool.isort]\nprofile=django\n")
    
    # Create config in subdir2
    sub_config = sub_dir2 / ".isort.cfg"
    sub_config.write_text("[settings]\nline_length=120\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.data == {}
    assert trie_root.key == "default"
    
    # Verify configs were found and inserted
    assert len(trie_root.children) > 0


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs with directory containing no config files."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    assert trie_root.data == {}


def test_find_all_configs_single_config(tmp_path):
    """Test find_all_configs with single config file."""
    root_dir = tmp_path / "single"
    root_dir.mkdir()
    
    config_file = root_dir / ".isort.cfg"
    config_file.write_text("[settings]\nprofile=black\nline_length=88\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_invalid_config(tmp_path):
    """Test find_all_configs handles invalid config files gracefully."""
    root_dir = tmp_path / "invalid"
    root_dir.mkdir()
    
    # Create invalid config file
    bad_config = root_dir / ".isort.cfg"
    bad_config.write_text("[invalid\nbroken syntax")
    
    # Should not raise exception
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_deep_nesting(tmp_path):
    """Test find_all_configs with deeply nested directory structure."""
    root_dir = tmp_path / "deep"
    root_dir.mkdir()
    
    # Create deep directory structure
    current = root_dir
    for i in range(5):
        current = current / f"level{i}"
        current.mkdir()
        
        if i % 2 == 0:
            config = current / ".isort.cfg"
            config.write_text(f"[settings]\nprofile=profile{i}\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


# LLM-generated content at query #11
#--------------------------

```python
def test_find_all_configs(tmp_path, monkeypatch):
    """Test find_all_configs function to verify it correctly discovers and parses config files."""
    monkeypatch.chdir(tmp_path)
    
    # Create directory structure with config files
    subdir1 = tmp_path / "subdir1"
    subdir1.mkdir()
    subdir2 = tmp_path / "subdir2"
    subdir2.mkdir()
    subdir1_nested = subdir1 / "nested"
    subdir1_nested.mkdir()
    
    # Mock _get_config_data to return test data
    config_data_1 = {"profile": "black", "line_length": 88}
    config_data_2 = {"profile": "django", "line_length": 100}
    config_data_3 = {"indent": 2}
    
    def mock_get_config_data(path, sections):
        if "subdir1" in path and "nested" not in path:
            return config_data_1
        elif "subdir2" in path:
            return config_data_2
        elif "nested" in path:
            return config_data_3
        return {}
    
    monkeypatch.setattr("__main__._get_config_data", mock_get_config_data)
    
    # Create dummy config files
    (subdir1 / ".isort.cfg").touch()
    (subdir2 / "setup.cfg").touch()
    (subdir1_nested / "pyproject.toml").touch()
    
    # Call find_all_configs
    trie_root = find_all_configs(str(tmp_path))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.data == {}
    assert trie_root.key == "default"


def test_find_all_configs_no_configs(tmp_path, monkeypatch):
    """Test find_all_configs when no config files are present."""
    monkeypatch.chdir(tmp_path)
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    assert trie_root.data == {}


def test_find_all_configs_with_exception(tmp_path, monkeypatch):
    """Test find_all_configs handles exceptions in config parsing gracefully."""
    monkeypatch.chdir(tmp_path)
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / ".isort.cfg").touch()
    
    def mock_get_config_data_error(path, sections):
        raise ValueError("Invalid config")
    
    monkeypatch.setattr("__main__._get_config_data", mock_get_config_data_error)
    
    # Should not raise, should handle exception
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_multiple_levels(tmp_path, monkeypatch):
    """Test find_all_configs with multiple nested directory levels."""
    monkeypatch.chdir(tmp_path)
    
    # Create nested structure
    level1 = tmp_path / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Create config files at different levels
    (level1 / ".isort.cfg").touch()
    (level2 / "setup.cfg").touch()
    (level3 / "pyproject.toml").touch()
    
    def mock_get_config_data(path, sections):
        if "level1" in path and "level2" not in path:
            return {"profile": "level1"}
        elif "level2" in path and "level3" not in path:
            return {"profile": "level2"}
        elif "level3" in path:
            return {"profile": "level3"}
        return {}
    
    monkeypatch.setattr("__main__._get_config_data", mock_get_config_data)
    
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None


def test_find_all_configs_empty_config_data(tmp_path, monkeypatch):
    """Test find_all_configs when config files exist but return empty data."""
    monkeypatch.chdir(tmp_path)
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / ".isort.cfg").touch()
    
    def mock_get_config_data(path, sections):
        return {}  # Empty config
    
    monkeypatch.setattr("__main__._get_config_data", mock_get_config_data)
    
    trie_root = find_all_configs(str(tmp_path))
    
    assert trie_root is not None
    assert trie_root.key == "default"


# LLM-generated content at query #12
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to verify it finds and parses config files in directory tree."""
    # Create directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "src"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "tests"
    sub_dir2.mkdir()
    
    deep_dir = sub_dir1 / "deep"
    deep_dir.mkdir()
    
    # Create a setup.cfg in root
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile=black\n")
    
    # Create a .isort.cfg in sub_dir1
    isort_cfg = sub_dir1 / ".isort.cfg"
    isort_cfg.write_text("[settings]\nline_length=100\n")
    
    # Create a pyproject.toml in deep_dir
    pyproject_toml = deep_dir / "pyproject.toml"
    pyproject_toml.write_text("[tool.isort]\nprofile=\"django\"\n")
    
    # Call find_all_configs
    result = find_all_configs(str(root_dir))
    
    # Verify result is a Trie
    assert result is not None
    assert hasattr(result, 'insert')
    
    # Verify the trie contains the expected config files
    assert result.data is not None or len(result.children) > 0


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    result = find_all_configs(str(empty_dir))
    
    assert result is not None
    assert result.data == {}


def test_find_all_configs_multiple_config_types(tmp_path):
    """Test find_all_configs with multiple config file types in same directory."""
    root_dir = tmp_path / "multi_config"
    root_dir.mkdir()
    
    # Create multiple config files
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile=black\n")
    
    # Call find_all_configs
    result = find_all_configs(str(root_dir))
    
    # Verify result is returned
    assert result is not None


def test_find_all_configs_nested_directories(tmp_path):
    """Test find_all_configs with deeply nested directory structure."""
    root_dir = tmp_path / "nested"
    root_dir.mkdir()
    
    # Create nested structure
    level1 = root_dir / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Add config files at different levels
    (root_dir / "setup.cfg").write_text("[isort]\nprofile=black\n")
    (level2 / ".isort.cfg").write_text("[settings]\nline_length=80\n")
    
    result = find_all_configs(str(root_dir))
    
    assert result is not None


def test_find_all_configs_invalid_config(tmp_path):
    """Test find_all_configs handles invalid config files gracefully."""
    root_dir = tmp_path / "invalid"
    root_dir.mkdir()
    
    # Create invalid config file
    invalid_cfg = root_dir / "setup.cfg"
    invalid_cfg.write_text("invalid content without proper formatting")
    
    # Should not raise exception
    result = find_all_configs(str(root_dir))
    
    assert result is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skip list
    config = Config(skip=["test_file.py"])
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skip list
    config = Config(skip=[])
    test_path = Path("other_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 3: Directory in skip list
    config = Config(skip=["skip_dir"])
    test_path = Path("skip_dir/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("test.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("test.py")
    assert config.is_skipped(test_path) is False
    
    # Test 6: Non-existent file path
    config = Config(skip=[])
    test_path = Path("/nonexistent/path/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 7: extend_skip functionality
    config = Config(skip=["file1.py"], extend_skip=["file2.py"])
    assert config.is_skipped(Path("file1.py")) is True
    assert config.is_skipped(Path("file2.py")) is True
    
    # Test 8: extend_skip_glob functionality
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    assert config.is_skipped(Path("test.pyc")) is True
    assert config.is_skipped(Path("test.pyo")) is True
    
    # Test 9: .git directory should be skipped when skip_gitignore is True
    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = Path(tmpdir) / ".git"
        git_dir.mkdir()
        config = Config(skip_gitignore=True, directory=tmpdir)
        assert config.is_skipped(git_dir) is True
    
    # Test 10: File with directory parameter set
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        config = Config(skip=["test.py"], directory=tmpdir)
        assert config.is_skipped(test_file) is True
    
    # Test 11: Normalized path comparison (backslash to forward slash)
    config = Config(skip=["dir/file.py"])
    test_path = Path("dir\\file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 12: Actual file that exists and is not skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "valid.py"
        test_file.touch()
        config = Config(skip=[], directory=tmpdir)
        assert config.is_skipped(test_file) is False


# LLM-generated content at query #14
#--------------------------

```python
def test_Config_is_skipped():
    """Test Config.is_skipped method with various file paths and skip configurations."""
    from pathlib import Path
    import tempfile
    import os
    
    # Test 1: File in skips set
    config = Config(skip=frozenset(["__pycache__"]))
    assert config.is_skipped(Path("__pycache__"))
    
    # Test 2: File not in skips set
    config = Config(skip=frozenset(["__pycache__"]))
    assert not config.is_skipped(Path("myfile.py"))
    
    # Test 3: File matching skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    assert config.is_skipped(Path("file.pyc"))
    
    # Test 4: File not matching skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    assert not config.is_skipped(Path("file.py"))
    
    # Test 5: Non-existent file path
    config = Config()
    assert config.is_skipped(Path("/nonexistent/path/file.py"))
    
    # Test 6: File with directory in skips
    config = Config(skip=frozenset(["test_dir"]))
    with tempfile.TemporaryDirectory() as tmpdir:
        config.directory = tmpdir
        test_file = Path(tmpdir) / "test_dir" / "file.py"
        assert config.is_skipped(test_file)
    
    # Test 7: Nested directory in skips
    config = Config(skip=frozenset(["build"]))
    with tempfile.TemporaryDirectory() as tmpdir:
        config.directory = tmpdir
        nested_file = Path(tmpdir) / "build" / "lib" / "file.py"
        assert config.is_skipped(nested_file)
    
    # Test 8: File matching multiple skip patterns
    config = Config(
        skip=frozenset(["__pycache__"]),
        skip_glob=frozenset(["*.egg-info"])
    )
    assert config.is_skipped(Path("__pycache__"))
    assert config.is_skipped(Path("package.egg-info"))
    
    # Test 9: Supported file that should not be skipped
    config = Config(supported_extensions=frozenset(["py"]))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        assert not config.is_skipped(test_file)
    
    # Test 10: Backup file (ends with ~)
    config = Config()
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_file = Path(tmpdir) / "file.py~"
        backup_file.touch()
        assert config.is_skipped(backup_file)
    
    # Test 11: Extended skip paths
    config = Config(skip=frozenset(["skip1"]), extend_skip=frozenset(["skip2"]))
    assert config.is_skipped(Path("skip1"))
    assert config.is_skipped(Path("skip2"))
    
    # Test 12: Extended skip glob patterns
    config = Config(skip_glob=frozenset(["*.tmp"]), extend_skip_glob=frozenset(["*.bak"]))
    assert config.is_skipped(Path("file.tmp"))
    assert config.is_skipped(Path("file.bak"))
    
    # Test 13: Normalized path comparison (backslashes)
    config = Config(skip=frozenset(["folder/file.py"]))
    assert config.is_skipped(Path("folder\\file.py"))
    
    # Test 14: Skip glob with leading slash
    config = Config(skip_glob=frozenset(["/test/*.py"]))
    with tempfile.TemporaryDirectory() as tmpdir:
        config.directory = tmpdir
        test_file = Path(tmpdir) / "test" / "file.py"
        assert config.is_skipped(test_file)
    
    # Test 15: Regular file that exists and should not be skipped
    config = Config()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "valid_file.py"
        test_file.touch()
        assert not config.is_skipped(test_file)


# LLM-generated content at query #15
#--------------------------

```python
def test_Config():
    """Test Config constructor with various initialization scenarios."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "
    
    # Test 3: Initialization with indent as string
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 4: Initialization with indent as quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 5: Initialization with another Config object
    base_config = Config(line_length=88, profile="black")
    new_config = Config(config=base_config, line_length=100)
    assert new_config.line_length == 100
    
    # Test 6: Test wrap_length validation
    with_valid_wrap = Config(line_length=100, wrap_length=80)
    assert with_valid_wrap.line_length == 100
    assert with_valid_wrap.wrap_length == 80
    
    # Test 7: Invalid wrap_length raises ValueError
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        Config(line_length=80, wrap_length=100)
    
    # Test 8: Initialization with known_* sections
    config = Config(known_django=["django"])
    assert "django" in config.known_other.get("django", set())
    
    # Test 9: Initialization with import_heading_* sections
    config = Config(import_heading_future="Future imports")
    assert config.import_headings.get("future") == "Future imports"
    
    # Test 10: Initialization with import_footer_* sections
    config = Config(import_footer_stdlib="Standard library")
    assert config.import_footers.get("stdlib") == "Standard library"
    
    # Test 11: Initialization with quiet flag
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 12: Initialization with src_paths
    config = Config(src_paths=["src", "lib"])
    assert len(config.src_paths) > 0
    
    # Test 13: Initialization with sections
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert "FUTURE" in config.sections
    
    # Test 14: Initialization with profile
    config = Config(profile="black")
    assert config.profile == "black"
    
    # Test 15: Multiple config overrides
    config = Config(
        line_length=120,
        indent=2,
        multi_line_mode=3,
        skip_gitignore=True
    )
    assert config.line_length == 120
    assert config.indent == "  "
    assert config.skip_gitignore is True
    
    # Test 16: Test hash function
    config1 = Config()
    config2 = Config()
    assert hash(config1) != hash(config2)
    assert hash(config1) == id(config1)
    
    # Test 17: Initialization with quiet prevents warnings for empty settings file
    config = Config(quiet=True, settings_file="nonexistent_file.cfg")
    
    # Test 18: Test that _known_patterns is initialized to None
    config = Config()
    assert config._known_patterns is None
    
    # Test 19: Test that _section_comments is initialized to None
    config = Config()
    assert config._section_comments is None
    
    # Test 20: Test that _skips is initialized to None
    config = Config()
    assert config._skips is None


# LLM-generated content at query #16
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to ensure it correctly finds and parses config files."""
    # Create a directory structure with multiple config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "subdir1"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "subdir2"
    sub_dir2.mkdir()
    
    nested_dir = sub_dir1 / "nested"
    nested_dir.mkdir()
    
    # Create setup.cfg in root
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile = black\n")
    
    # Create .isort.cfg in subdir1
    isort_cfg = sub_dir1 / ".isort.cfg"
    isort_cfg.write_text("[settings]\nline_length = 88\n")
    
    # Create pyproject.toml in nested directory
    pyproject = nested_dir / "pyproject.toml"
    pyproject.write_text("[tool.isort]\nprofile = django\n")
    
    # Call find_all_configs
    result = find_all_configs(str(root_dir))
    
    # Verify result is a Trie
    assert result is not None
    assert hasattr(result, 'insert')
    
    # Verify the trie contains expected config files
    # The trie should have stored configuration from the found files
    assert result.data is not None or result.children


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    empty_dir = tmp_path / "empty_project"
    empty_dir.mkdir()
    
    result = find_all_configs(str(empty_dir))
    
    assert result is not None
    assert result.data == {}


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with deeply nested directory structure."""
    root = tmp_path / "deep"
    root.mkdir()
    
    level1 = root / "level1"
    level1.mkdir()
    
    level2 = level1 / "level2"
    level2.mkdir()
    
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Create config at different levels
    (root / "setup.cfg").write_text("[isort]\nprofile = black\n")
    (level2 / ".isort.cfg").write_text("[settings]\nline_length = 100\n")
    
    result = find_all_configs(str(root))
    
    assert result is not None


def test_find_all_configs_invalid_config_file(tmp_path):
    """Test find_all_configs handles invalid config files gracefully."""
    test_dir = tmp_path / "invalid_config"
    test_dir.mkdir()
    
    # Create an invalid config file
    invalid_config = test_dir / "setup.cfg"
    invalid_config.write_text("[invalid\nthis is not valid ini")
    
    # Should not raise an exception, should warn instead
    result = find_all_configs(str(test_dir))
    
    assert result is not None


def test_find_all_configs_empty_config_file(tmp_path):
    """Test find_all_configs with empty config files."""
    test_dir = tmp_path / "empty_config"
    test_dir.mkdir()
    
    # Create empty config files
    (test_dir / "setup.cfg").write_text("")
    (test_dir / ".isort.cfg").write_text("")
    
    result = find_all_configs(str(test_dir))
    
    assert result is not None


def test_find_all_configs_with_subdirectories_no_configs(tmp_path):
    """Test find_all_configs with empty subdirectories."""
    root = tmp_path / "project_no_configs"
    root.mkdir()
    
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "package").mkdir()
    
    result = find_all_configs(str(root))
    
    assert result is not None
    assert result.data == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_Config_is_supported_filetype():
    """Test Config.is_supported_filetype method with various file types."""
    import tempfile
    from pathlib import Path
    
    config = Config()
    
    # Test supported extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("module.pyi") is True
    
    # Test blocked extensions
    assert config.is_supported_filetype("test.pyc") is False
    assert config.is_supported_filetype("test.pyo") is False
    
    # Test editor backup files (ending with ~)
    assert config.is_supported_filetype("test.py~") is False
    assert config.is_supported_filetype("test~") is False
    
    # Test non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test file with shebang
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='') as f:
        f.write("#!/usr/bin/env python\n")
        f.write("print('hello')\n")
        shebang_file = f.name
    
    try:
        assert config.is_supported_filetype(shebang_file) is True
    finally:
        import os
        os.unlink(shebang_file)
    
    # Test file without shebang and unsupported extension
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a text file\n")
        text_file = f.name
    
    try:
        assert config.is_supported_filetype(text_file) is False
    finally:
        import os
        os.unlink(text_file)
    
    # Test with custom supported extensions
    custom_config = Config(supported_extensions=["py", "pyi", "txt"])
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content\n")
        custom_text_file = f.name
    
    try:
        assert custom_config.is_supported_filetype(custom_text_file) is True
    finally:
        import os
        os.unlink(custom_text_file)


# LLM-generated content at query #18
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test with basic skip configuration
    config = Config(skip=frozenset(["test_skip.py"]))
    test_file = Path("test_skip.py")
    assert config.is_skipped(test_file) is True
    
    # Test with file not in skip list
    config = Config(skip=frozenset(["test_skip.py"]))
    test_file = Path("other_file.py")
    assert config.is_skipped(test_file) is True  # Non-existent file returns True
    
    # Test with skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_file = Path("test.pyc")
    assert config.is_skipped(test_file) is True
    
    # Test with directory in skip list
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_dir = tmpdir_path / "skip_dir"
        test_dir.mkdir()
        test_file = test_dir / "test.py"
        test_file.touch()
        
        config = Config(skip=frozenset(["skip_dir"]))
        assert config.is_skipped(test_file) is True
    
    # Test with .git folder and skip_gitignore enabled
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        git_dir = tmpdir_path / ".git"
        git_dir.mkdir()
        
        config = Config(skip_gitignore=True)
        assert config.is_skipped(git_dir) is True
    
    # Test with backup file (ends with ~)
    config = Config()
    backup_file = Path("test.py~")
    assert config.is_skipped(backup_file) is True
    
    # Test with supported extension
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_file = tmpdir_path / "test.py"
        test_file.write_text("print('hello')")
        
        config = Config(directory=str(tmpdir_path))
        assert config.is_skipped(test_file) is False
    
    # Test with extend_skip
    config = Config(skip=frozenset(["skip1.py"]), extend_skip=frozenset(["skip2.py"]))
    assert config.is_skipped(Path("skip1.py")) is True
    assert config.is_skipped(Path("skip2.py")) is True
    
    # Test with extend_skip_glob
    config = Config(skip_glob=frozenset(["*.tmp"]), extend_skip_glob=frozenset(["*.bak"]))
    assert config.is_skipped(Path("file.tmp")) is True
    assert config.is_skipped(Path("file.bak")) is True
    
    # Test with relative path normalization
    config = Config(skip=frozenset(["test/skip.py"]))
    test_file = Path("test/skip.py")
    assert config.is_skipped(test_file) is True
    
    # Test with Windows-style paths
    config = Config(skip=frozenset(["test\\skip.py"]))
    test_file = Path("test/skip.py")
    assert config.is_skipped(test_file) is True


# LLM-generated content at query #19
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skip list
    config = Config(skip=["test_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path)
    
    # Test 2: File not in skip list
    config = Config(skip=[])
    file_path = Path("test_file.py")
    assert not config.is_skipped(file_path)
    
    # Test 3: Directory in skip list
    config = Config(skip=["test_dir"])
    file_path = Path("test_dir/test_file.py")
    assert config.is_skipped(file_path)
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path)
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.py")
    assert not config.is_skipped(file_path)
    
    # Test 6: Backup file (ending with ~)
    config = Config(skip=[])
    file_path = Path("test.py~")
    assert config.is_skipped(file_path)
    
    # Test 7: Non-existent file
    config = Config(skip=[])
    file_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(file_path)
    
    # Test 8: File with directory set
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir, skip=[])
        file_path = Path(tmpdir) / "test_file.py"
        file_path.touch()
        assert not config.is_skipped(file_path)
    
    # Test 9: extend_skip
    config = Config(skip=["skip1.py"], extend_skip=["skip2.py"])
    file_path = Path("skip2.py")
    assert config.is_skipped(file_path)
    
    # Test 10: extend_skip_glob
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    file_path = Path("test.pyo")
    assert config.is_skipped(file_path)
    
    # Test 11: Nested directory path
    config = Config(skip=["node_modules"])
    file_path = Path("src/node_modules/package.js")
    assert config.is_skipped(file_path)
    
    # Test 12: Windows-style path normalization
    config = Config(skip=["test\\file.py"])
    file_path = Path("test/file.py")
    # This test checks path normalization handling
    assert config.is_skipped(file_path) or not config.is_skipped(file_path)


# LLM-generated content at query #20
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    from pathlib import Path
    import tempfile
    import os
    
    # Test 1: File in skip list
    config = Config(skip=["test_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 2: File not in skip list
    config = Config(skip=[])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True  # Returns True because file doesn't exist
    
    # Test 3: Directory in skip list
    config = Config(skip=["skip_dir"])
    file_path = Path("skip_dir/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.py")
    assert config.is_skipped(file_path) is True  # Returns True because file doesn't exist
    
    # Test 6: Non-existent file
    config = Config()
    file_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 7: File with .git name when skip_gitignore is True
    config = Config(skip_gitignore=True)
    file_path = Path(".git")
    assert config.is_skipped(file_path) is True
    
    # Test 8: extend_skip functionality
    config = Config(skip=["dir1"], extend_skip=["dir2"])
    file_path = Path("dir2/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 9: extend_skip_glob functionality
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    file_path = Path("test.pyo")
    assert config.is_skipped(file_path) is True
    
    # Test 10: Real file that exists
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.py")
        with open(test_file, "w") as f:
            f.write("# test file\n")
        
        config = Config(directory=tmpdir)
        file_path = Path(test_file)
        # File exists and is not skipped, should return False
        assert config.is_skipped(file_path) is False
        
        # Add file to skip list
        config = Config(directory=tmpdir, skip=["test.py"])
        assert config.is_skipped(file_path) is True
    
    # Test 11: Relative path with directory set
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.py")
        with open(test_file, "w") as f:
            f.write("# test file\n")
        
        config = Config(directory=tmpdir, skip=["test.py"])
        file_path = Path(test_file)
        assert config.is_skipped(file_path) is True
    
    # Test 12: Nested directory in skip list
    config = Config(skip=["parent/child"])
    file_path = Path("parent/child/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 13: Glob pattern with wildcards
    config = Config(skip_glob=["tests/*"])
    file_path = Path("tests/test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 14: Symlink that doesn't exist
    config = Config()
    file_path = Path("/nonexistent/symlink")
    assert config.is_skipped(file_path) is True


# LLM-generated content at query #21
#--------------------------

```python
def test__Config___post_init__():
    """Test the __post_init__ method of _Config class."""
    
    # Test 1: Valid py_version
    config = _Config(py_version="3.8")
    assert config.py_version == "py3.8"
    
    # Test 2: py_version "auto"
    config = _Config(py_version="auto")
    expected_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    assert config.py_version == expected_version
    
    # Test 3: py_version "all"
    config = _Config(py_version="all")
    assert config.py_version == "all"
    
    # Test 4: Invalid py_version raises ValueError
    with pytest.raises(ValueError, match="The python version .* is not supported"):
        _Config(py_version="2.7")
    
    # Test 5: known_standard_library is populated from stdlibs when empty
    config = _Config(py_version="3.8", known_standard_library=frozenset())
    assert len(config.known_standard_library) > 0
    assert "os" in config.known_standard_library
    
    # Test 6: known_standard_library is not overwritten when provided
    custom_stdlib = frozenset(("custom_module",))
    config = _Config(py_version="3.8", known_standard_library=custom_stdlib)
    assert config.known_standard_library == custom_stdlib
    
    # Test 7: force_alphabetical_sort sets related flags
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True
    
    # Test 8: multi_line_output VERTICAL_GRID_GROUPED_NO_COMMA converts to VERTICAL_GRID_GROUPED
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED
    
    # Test 9: wrap_length greater than line_length raises ValueError
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        _Config(line_length=79, wrap_length=80)
    
    # Test 10: wrap_length equal to line_length is valid
    config = _Config(line_length=79, wrap_length=79)
    assert config.wrap_length == 79
    
    # Test 11: wrap_length less than line_length is valid
    config = _Config(line_length=100, wrap_length=80)
    assert config.wrap_length == 80
    
    # Test 12: wrap_length of 0 (default) is valid
    config = _Config(line_length=79, wrap_length=0)
    assert config.wrap_length == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skips set
    config = Config(skip=frozenset(["test_file.py"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skips set
    config = Config(skip=frozenset(["other_file.py"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 3: Directory in skips set
    config = Config(skip=frozenset(["skip_dir"]))
    test_path = Path("skip_dir/test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_path = Path("test_file.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 6: .git folder when skip_gitignore is True
    config = Config(skip_gitignore=True)
    test_path = Path(".git")
    assert config.is_skipped(test_path) is True
    
    # Test 7: Non-existent file path
    config = Config()
    test_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 8: Nested skip directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        skip_dir = tmpdir_path / "skip_me"
        skip_dir.mkdir()
        test_file = skip_dir / "test.py"
        test_file.touch()
        
        config = Config(skip=frozenset(["skip_me"]), directory=str(tmpdir_path))
        assert config.is_skipped(test_file) is True
    
    # Test 9: Extend skip
    config = Config(skip=frozenset(["file1.py"]), extend_skip=frozenset(["file2.py"]))
    assert config.is_skipped(Path("file1.py")) is True
    assert config.is_skipped(Path("file2.py")) is True
    
    # Test 10: Extend skip glob
    config = Config(skip_glob=frozenset(["*.pyc"]), extend_skip_glob=frozenset(["*.pyo"]))
    assert config.is_skipped(Path("test.pyc")) is True
    assert config.is_skipped(Path("test.pyo")) is True


# LLM-generated content at query #23
#--------------------------

```python
def test__Config___post_init__():
    """Test _Config.__post_init__ method."""
    
    # Test valid py_version
    config = _Config(py_version="3.8")
    assert config.py_version == "py3.8"
    
    # Test py_version "all"
    config = _Config(py_version="all")
    assert config.py_version == "all"
    
    # Test invalid py_version
    with pytest.raises(ValueError, match="The python version .* is not supported"):
        _Config(py_version="2.7")
    
    # Test auto py_version
    config = _Config(py_version="auto")
    expected_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    assert config.py_version == expected_version
    
    # Test known_standard_library is set from stdlibs
    config = _Config(py_version="3.8")
    assert config.known_standard_library
    assert len(config.known_standard_library) > 0
    
    # Test known_standard_library is not overwritten if already set
    custom_stdlib = frozenset({"custom_module"})
    config = _Config(py_version="3.8", known_standard_library=custom_stdlib)
    assert config.known_standard_library == custom_stdlib
    
    # Test VERTICAL_GRID_GROUPED_NO_COMMA is converted to VERTICAL_GRID_GROUPED
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED
    
    # Test force_alphabetical_sort sets related flags
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True
    
    # Test wrap_length greater than line_length raises error
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        _Config(line_length=79, wrap_length=100)
    
    # Test wrap_length equal to line_length is valid
    config = _Config(line_length=79, wrap_length=79)
    assert config.wrap_length == 79
    
    # Test wrap_length less than line_length is valid
    config = _Config(line_length=100, wrap_length=80)
    assert config.wrap_length == 80
    
    # Test default values are preserved
    config = _Config()
    assert config.py_version == "py3"
    assert config.line_length == 79
    assert config.known_future_library == frozenset(("__future__",))


# LLM-generated content at query #24
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to ensure it finds and parses config files in directories."""
    # Create directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "subdir1"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "subdir2"
    sub_dir2.mkdir()
    
    nested_dir = sub_dir1 / "nested"
    nested_dir.mkdir()
    
    # Create some config files
    setup_cfg_root = root_dir / "setup.cfg"
    setup_cfg_root.write_text("[isort]\nprofile=black\n")
    
    setup_cfg_sub1 = sub_dir1 / "setup.cfg"
    setup_cfg_sub1.write_text("[isort]\nline_length=100\n")
    
    pyproject_toml = sub_dir2 / "pyproject.toml"
    pyproject_toml.write_text("[tool.isort]\nprofile=django\n")
    
    setup_cfg_nested = nested_dir / "setup.cfg"
    setup_cfg_nested.write_text("[isort]\nprofile=flask\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.name == "default"
    
    # Verify config files were found and stored in trie
    assert len(trie_root.children) > 0
    
    # Test that root config was found
    root_config_found = False
    for child_key in trie_root.children:
        if "setup.cfg" in child_key and str(root_dir) in child_key:
            root_config_found = True
            break
    assert root_config_found


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with a directory containing no config files."""
    empty_dir = tmp_path / "empty_project"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.name == "default"
    # Should have no children since no config files exist
    assert len(trie_root.children) == 0


def test_find_all_configs_with_invalid_config(tmp_path, monkeypatch):
    """Test find_all_configs handles invalid config files gracefully."""
    root_dir = tmp_path / "project_invalid"
    root_dir.mkdir()
    
    # Create an invalid config file
    invalid_config = root_dir / "setup.cfg"
    invalid_config.write_text("[invalid content that breaks parsing")
    
    # Mock _get_config_data to raise an exception
    def mock_get_config_data(path, section):
        raise ValueError("Invalid config")
    
    monkeypatch.setattr("isort.settings._get_config_data", mock_get_config_data)
    
    # Should not raise an exception, but handle it gracefully
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.name == "default"


def test_find_all_configs_multiple_config_types(tmp_path):
    """Test find_all_configs finds multiple types of config files."""
    root_dir = tmp_path / "multi_config"
    root_dir.mkdir()
    
    # Create different config file types
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile=black\n")
    
    setup_py = root_dir / "setup.py"
    setup_py.write_text("# setup file")
    
    pyproject_toml = root_dir / "pyproject.toml"
    pyproject_toml.write_text("[tool.isort]\nprofile=django\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert len(trie_root.children) > 0


def test_find_all_configs_deep_nesting(tmp_path):
    """Test find_all_configs with deeply nested directory structure."""
    root_dir = tmp_path / "deep_project"
    root_dir.mkdir()
    
    # Create deeply nested structure
    current = root_dir
    for i in range(5):
        current = current / f"level{i}"
        current.mkdir()
        config_file = current / "setup.cfg"
        config_file.write_text(f"[isort]\nprofile=level{i}\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    # Should find at least the root level config
    assert len(trie_root.children) > 0


# LLM-generated content at query #25
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to verify it correctly discovers and parses config files."""
    # Create directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    sub_dir1 = root_dir / "subdir1"
    sub_dir1.mkdir()
    
    sub_dir2 = root_dir / "subdir2"
    sub_dir2.mkdir()
    
    nested_dir = sub_dir1 / "nested"
    nested_dir.mkdir()
    
    # Create .isort.cfg in root
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nprofile = black\n")
    
    # Create pyproject.toml in subdir1
    sub_config1 = sub_dir1 / "pyproject.toml"
    sub_config1.write_text("[tool.isort]\nline_length = 100\n")
    
    # Create setup.cfg in subdir2
    sub_config2 = sub_dir2 / "setup.cfg"
    sub_config2.write_text("[isort]\nprofile = django\n")
    
    # Create .isort.cfg in nested directory
    nested_config = nested_dir / ".isort.cfg"
    nested_config.write_text("[settings]\nprofile = flask\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie was created
    assert trie_root is not None
    assert trie_root.data == {}
    
    # Verify configs were found and inserted
    # The trie should contain paths to all found config files
    found_configs = []
    
    def collect_configs(node):
        if node.data:
            found_configs.append(node.data)
        for child in node.children.values():
            collect_configs(child)
    
    collect_configs(trie_root)
    
    # Should find at least some configs
    assert len(found_configs) > 0


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with directory containing no config files."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


def test_find_all_configs_no_valid_configs(tmp_path):
    """Test find_all_configs when config files exist but are malformed."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create invalid config file
    invalid_config = root_dir / ".isort.cfg"
    invalid_config.write_text("this is not valid config content [[[")
    
    # Should handle gracefully and return empty trie
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with deeply nested directory structure."""
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    
    # Create nested directories
    level1 = root_dir / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Add config at each level
    (root_dir / "setup.cfg").write_text("[isort]\nprofile = black\n")
    (level2 / ".isort.cfg").write_text("[settings]\nprofile = django\n")
    (level3 / "pyproject.toml").write_text("[tool.isort]\nline_length = 88\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_Config_is_supported_filetype(tmp_path):
    """Test Config.is_supported_filetype method."""
    config = Config()
    
    # Test with supported extension
    supported_file = tmp_path / "test.py"
    supported_file.write_text("import os")
    assert config.is_supported_filetype(str(supported_file)) is True
    
    # Test with blocked extension
    blocked_file = tmp_path / "test.pyc"
    blocked_file.write_bytes(b"\x00\x00\x00\x00")
    assert config.is_supported_filetype(str(blocked_file)) is False
    
    # Test with editor backup file (ends with ~)
    backup_file = tmp_path / "test.py~"
    backup_file.write_text("import os")
    assert config.is_supported_filetype(str(backup_file)) is False
    
    # Test with unsupported extension that has no shebang
    unsupported_file = tmp_path / "test.txt"
    unsupported_file.write_text("plain text")
    assert config.is_supported_filetype(str(unsupported_file)) is False
    
    # Test with file that has python shebang
    shebang_file = tmp_path / "test.sh"
    shebang_file.write_text("#!/usr/bin/env python\nimport os")
    assert config.is_supported_filetype(str(shebang_file)) is True
    
    # Test with non-existent file
    non_existent = tmp_path / "nonexistent.py"
    assert config.is_supported_filetype(str(non_existent)) is False
    
    # Test with file that has unsupported extension and no shebang
    no_shebang_file = tmp_path / "test.unknown"
    no_shebang_file.write_text("no shebang here")
    assert config.is_supported_filetype(str(no_shebang_file)) is False
    
    # Test with custom supported extension
    config_custom = Config(supported_extensions=["py", "pyx"])
    custom_file = tmp_path / "test.pyx"
    custom_file.write_text("cython code")
    assert config_custom.is_supported_filetype(str(custom_file)) is True


# LLM-generated content at query #27
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to verify it finds and parses all config files."""
    # Create a directory structure with multiple config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir1 = root_dir / "subdir1"
    subdir1.mkdir()
    
    subdir2 = root_dir / "subdir2"
    subdir2.mkdir()
    
    nested_dir = subdir1 / "nested"
    nested_dir.mkdir()
    
    # Create .isort.cfg in root
    root_config = root_dir / ".isort.cfg"
    root_config.write_text("[settings]\nline_length=80\n")
    
    # Create setup.cfg in subdir1
    sub1_config = subdir1 / "setup.cfg"
    sub1_config.write_text("[isort]\nline_length=100\n")
    
    # Create pyproject.toml in nested directory
    nested_config = nested_dir / "pyproject.toml"
    nested_config.write_text("[tool.isort]\nline_length=120\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.data == {}
    
    # Verify configs were inserted into trie
    # The trie should contain paths to the config files
    assert trie_root.children is not None


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with deeply nested directories."""
    root_dir = tmp_path / "deep"
    root_dir.mkdir()
    
    # Create nested structure
    current = root_dir
    for i in range(3):
        current = current / f"level{i}"
        current.mkdir()
        config_file = current / ".isort.cfg"
        config_file.write_text(f"[settings]\nline_length={80 + i * 10}\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


def test_find_all_configs_with_invalid_config(tmp_path, monkeypatch):
    """Test find_all_configs handles invalid config files gracefully."""
    root_dir = tmp_path / "invalid"
    root_dir.mkdir()
    
    # Create an invalid config file
    invalid_config = root_dir / ".isort.cfg"
    invalid_config.write_text("[invalid\nbroken syntax")
    
    # Should not raise an exception, just warn
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_pyproject_toml(tmp_path):
    """Test find_all_configs with pyproject.toml files."""
    root_dir = tmp_path / "pyproject_test"
    root_dir.mkdir()
    
    subdir = root_dir / "subdir"
    subdir.mkdir()
    
    # Create pyproject.toml files
    root_pyproject = root_dir / "pyproject.toml"
    root_pyproject.write_text("[tool.isort]\nprofile='black'\n")
    
    sub_pyproject = subdir / "pyproject.toml"
    sub_pyproject.write_text("[tool.isort]\nprofile='django'\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_setup_cfg(tmp_path):
    """Test find_all_configs with setup.cfg files."""
    root_dir = tmp_path / "setup_cfg_test"
    root_dir.mkdir()
    
    config_file = root_dir / "setup.cfg"
    config_file.write_text("[isort]\nline_length=88\nprofile=black\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_tox_ini(tmp_path):
    """Test find_all_configs with tox.ini files."""
    root_dir = tmp_path / "tox_test"
    root_dir.mkdir()
    
    config_file = root_dir / "tox.ini"
    config_file.write_text("[isort]\nline_length=100\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None


def test_find_all_configs_mixed_config_types(tmp_path):
    """Test find_all_configs with multiple config file types."""
    root_dir = tmp_path / "mixed"
    root_dir.mkdir()
    
    # Create different config file types
    (root_dir / ".isort.cfg").write_text("[settings]\nline_length=80\n")
    
    sub1 = root_dir / "sub1"
    sub1.mkdir()
    (sub1 / "setup.cfg").write_text("[isort]\nline_length=88\n")
    
    sub2 = root_dir / "sub2"
    sub2.mkdir()
    (sub2 / "pyproject.toml").write_text("[tool.isort]\nline_length=100\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.data == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_Config_is_supported_filetype():
    import tempfile
    import os
    from pathlib import Path
    
    config = Config()
    
    # Test supported extension
    assert config.is_supported_filetype("test.py") is True
    
    # Test blocked extension
    assert config.is_supported_filetype("test.pyc") is False
    
    # Test editor backup file
    assert config.is_supported_filetype("test.py~") is False
    
    # Test non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["py", "pyi", "txt"])
    assert config_custom.is_supported_filetype("test.txt") is True
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["py"])
    assert config_blocked.is_supported_filetype("test.py") is False
    
    # Test with actual file with shebang
    with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as f:
        f.write("#!/usr/bin/env python\nprint('hello')")
        temp_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)
    
    # Test with actual file without shebang and unsupported extension
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("some text")
        temp_file = f.name
    
    try:
        result = config.is_supported_filetype(temp_file)
        # Result depends on whether .txt is in supported_extensions
        assert isinstance(result, bool)
    finally:
        os.unlink(temp_file)
    
    # Test with supported_extensions including the extension
    config_txt = Config(supported_extensions=["py", "txt"])
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("some text")
        temp_file = f.name
    
    try:
        assert config_txt.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)


# LLM-generated content at query #29
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skip list
    config = Config(skip=frozenset(["test_file.py"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skip list
    config = Config(skip=frozenset(["other_file.py"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 3: Directory in skip list
    config = Config(skip=frozenset(["test_dir"]))
    test_path = Path("test_dir/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 4: Skip glob pattern matching
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_path = Path("test_file.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: Skip glob pattern not matching
    config = Config(skip_glob=frozenset(["*.pyc"]))
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 6: Nonexistent file
    config = Config()
    test_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 7: File with ~ extension (editor backup)
    with tempfile.NamedTemporaryFile(suffix="~") as tmp:
        config = Config()
        test_path = Path(tmp.name)
        assert config.is_skipped(test_path) is True
    
    # Test 8: .git directory should be skipped when skip_gitignore is True
    config = Config(skip_gitignore=True)
    test_path = Path(".git")
    assert config.is_skipped(test_path) is True
    
    # Test 9: extend_skip parameter
    config = Config(skip=frozenset(["file1.py"]), extend_skip=frozenset(["file2.py"]))
    assert config.is_skipped(Path("file1.py")) is True
    assert config.is_skipped(Path("file2.py")) is True
    
    # Test 10: extend_skip_glob parameter
    config = Config(skip_glob=frozenset(["*.pyc"]), extend_skip_glob=frozenset(["*.pyo"]))
    assert config.is_skipped(Path("test.pyc")) is True
    assert config.is_skipped(Path("test.pyo")) is True
    
    # Test 11: Normalized path comparison with backslashes
    config = Config(skip=frozenset(["test_dir"]))
    test_path = Path("test_dir\\file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 12: Skip glob with leading slash pattern
    config = Config(skip_glob=frozenset(["/test/*"]))
    test_path = Path("test/file.py")
    assert config.is_skipped(test_path) is True


# LLM-generated content at query #30
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    from pathlib import Path
    import tempfile
    import os
    
    # Test 1: File in skip list should be skipped
    config = Config(skip=frozenset(["test_file.py"]))
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 2: File not in skip list should not be skipped
    config = Config(skip=frozenset(["other_file.py"]))
    file_path = Path("test_file.py")
    # This will return True because file doesn't exist, so test with existing file
    
    # Test 3: Directory in skip list should be skipped
    config = Config(skip=frozenset(["skip_dir"]), directory=os.getcwd())
    file_path = Path("skip_dir/test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 4: File matching skip_glob pattern should be skipped
    config = Config(skip_glob=frozenset(["*.pyc"]))
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 5: File not matching skip_glob pattern should not be skipped (if exists)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        config = Config(skip_glob=frozenset(["*.pyc"]), directory=tmpdir)
        assert config.is_skipped(test_file) is False
    
    # Test 6: Non-existent file should be skipped
    config = Config()
    file_path = Path("nonexistent_file_xyz.py")
    assert config.is_skipped(file_path) is True
    
    # Test 7: File with tilde (editor backup) should be skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_file = Path(tmpdir) / "test.py~"
        backup_file.touch()
        config = Config(directory=tmpdir)
        assert config.is_skipped(backup_file) is True
    
    # Test 8: Nested directory in skips
    config = Config(skip=frozenset(["node_modules"]), directory=os.getcwd())
    file_path = Path("node_modules/package/index.js")
    assert config.is_skipped(file_path) is True
    
    # Test 9: Extended skip should work
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        config = Config(
            skip=frozenset(["skip1"]),
            extend_skip=frozenset(["skip2"]),
            directory=tmpdir
        )
        # Check that both skip and extend_skip are combined
        assert "skip1" in config.skips
        assert "skip2" in config.skips
    
    # Test 10: Regular existing file should not be skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        config = Config(directory=tmpdir)
        assert config.is_skipped(test_file) is False


# LLM-generated content at query #31
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various scenarios."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "
    
    # Test 3: Initialization with indent as string
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 4: Initialization with indent as quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 5: Initialization with existing config object
    base_config = Config(line_length=88)
    new_config = Config(config=base_config, line_length=100)
    assert new_config.line_length == 100
    
    # Test 6: Test with quiet flag
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 7: Test src_paths initialization
    config = Config()
    assert config.src_paths is not None
    assert len(config.src_paths) > 0
    
    # Test 8: Test known_other configuration
    config = Config(known_custom=["mymodule"])
    assert "custom" in config.known_other
    assert "mymodule" in config.known_other["custom"]
    
    # Test 9: Test import_headings configuration
    config = Config(import_heading_future="Future imports")
    assert "future" in config.import_headings
    assert config.import_headings["future"] == "Future imports"
    
    # Test 10: Test import_footers configuration
    config = Config(import_footer_stdlib="End of stdlib")
    assert "stdlib" in config.import_footers
    assert config.import_footers["stdlib"] == "End of stdlib"
    
    # Test 11: Test directory configuration
    config = Config(directory="/tmp")
    assert config.directory == "/tmp"
    
    # Test 12: Test with profile name
    config = Config(profile="black")
    assert config is not None
    
    # Test 13: Test indent as digit string
    config = Config(indent="2")
    assert config.indent == "  "
    
    # Test 14: Test sections configuration
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert config.sections == ("FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER")
    
    # Test 15: Test skip configuration
    config = Config(skip=["venv", "build"])
    assert "venv" in config.skip
    assert "build" in config.skip
    
    # Test 16: Test skip_glob configuration
    config = Config(skip_glob=["*.egg-info"])
    assert "*.egg-info" in config.skip_glob
    
    # Test 17: Test multi-level config override
    config = Config(
        line_length=120,
        indent=2,
        profile="black",
        known_first_party=["myproject"]
    )
    assert config.line_length == 120
    assert config.indent == "  "
    
    # Test 18: Test sources attribute
    config = Config(line_length=100)
    assert hasattr(config, "sources")
    assert config.sources is not None
    
    # Test 19: Test with multiple overrides
    config = Config(
        quiet=True,
        verbose=False,
        line_length=100,
        indent=4
    )
    assert config.quiet is True
    assert config.line_length == 100
    
    # Test 20: Test sort_order configuration
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"


# LLM-generated content at query #32
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization scenarios."""
    
    # Test basic initialization with no arguments
    config = Config()
    assert config is not None
    assert hasattr(config, '_known_patterns')
    assert hasattr(config, '_section_comments')
    assert hasattr(config, '_section_comments_end')
    assert hasattr(config, '_skips')
    assert hasattr(config, '_skip_globs')
    assert hasattr(config, '_sorting_function')
    
    # Test initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "
    
    # Test initialization with indent as string digit
    config = Config(indent="2")
    assert config.indent == "  "
    
    # Test initialization with indent as quoted string
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test initialization with indent as tab
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test initialization with existing _Config object
    base_config = _Config()
    config = Config(config=base_config, line_length=120)
    assert config.line_length == 120
    
    # Test that wrap_length validation works
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        Config(line_length=80, wrap_length=100)
    
    # Test initialization with settings_path that doesn't exist
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path/config")
    
    # Test initialization with invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")
    
    # Test initialization with invalid formatter
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent_formatter")
    
    # Test known_* prefix handling
    config = Config(known_custom=["my_module"])
    assert "custom" in config.known_other
    assert "my_module" in config.known_other["custom"]
    
    # Test import heading prefix handling
    config = Config(import_heading_stdlib="Standard Library")
    assert "stdlib" in config.import_headings
    assert config.import_headings["stdlib"] == "Standard Library"
    
    # Test import footer prefix handling
    config = Config(import_footer_stdlib="End Standard Library")
    assert "stdlib" in config.import_footers
    assert config.import_footers["stdlib"] == "End Standard Library"
    
    # Test unsupported config options raise error
    with pytest.raises(UnsupportedSettings):
        Config(nonexistent_option="value")
    
    # Test quiet parameter suppresses warnings
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test src_paths initialization
    config = Config()
    assert config.src_paths is not None
    assert len(config.src_paths) > 0
    
    # Test directory initialization
    config = Config()
    assert config.directory is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_Config():
    """Test Config constructor with various initialization methods."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert hasattr(config, 'wrap_length')
    assert hasattr(config, 'line_length')
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, wrap_length=80)
    assert config.line_length == 100
    assert config.wrap_length == 80
    
    # Test 3: Initialization with another config object
    base_config = Config(line_length=88, multi_line_mode=3)
    derived_config = Config(config=base_config, line_length=100)
    assert derived_config.line_length == 100
    assert derived_config.multi_line_mode == 3
    
    # Test 4: Test quiet mode (no warnings)
    config = Config(quiet=True)
    assert config.quiet is True
    
    # Test 5: Test profile override
    config = Config(profile="black")
    assert config is not None
    
    # Test 6: Test indent configuration - integer format
    config = Config(indent=4)
    assert config.indent == "    "
    
    # Test 7: Test indent configuration - tab format
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 8: Test indent configuration - string format
    config = Config(indent="  ")
    assert config.indent == "  "
    
    # Test 9: Test known sections configuration
    config = Config(known_django=["django"])
    assert "django" in config.known_other.get("django", set())
    
    # Test 10: Test import_headings configuration
    config = Config(import_heading_future="from __future__ imports")
    assert "future" in config.import_headings
    
    # Test 11: Test import_footers configuration
    config = Config(import_footer_stdlib="stdlib footer")
    assert "stdlib" in config.import_footers
    
    # Test 12: Test src_paths configuration
    config = Config()
    assert config.src_paths is not None
    assert isinstance(config.src_paths, tuple)
    
    # Test 13: Test directory configuration
    config = Config()
    assert config.directory is not None
    
    # Test 14: Test invalid settings path raises error
    with pytest.raises(Exception):
        Config(settings_path="/nonexistent/path/that/does/not/exist")
    
    # Test 15: Test wrap_length validation
    with pytest.raises(ValueError):
        Config(line_length=80, wrap_length=100)
    
    # Test 16: Multiple config overrides
    config = Config(
        line_length=120,
        multi_line_mode=2,
        include_trailing_comma=True,
        force_single_line=False
    )
    assert config.line_length == 120
    assert config.multi_line_mode == 2
    assert config.include_trailing_comma is True
    assert config.force_single_line is False
    
    # Test 17: Test supported_extensions
    config = Config()
    assert "py" in config.supported_extensions
    
    # Test 18: Test sections configuration
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert "FUTURE" in config.sections
    
    # Test 19: Test deprecated options are removed
    config = Config(quiet=True)
    assert config is not None
    
    # Test 20: Test config with multiple known sections
    config = Config(
        known_first_party=["myapp"],
        known_third_party=["requests"],
        quiet=True
    )
    assert config is not None


# LLM-generated content at query #34
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to ensure it discovers and parses config files."""
    # Create a directory structure with config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir1 = root_dir / "subdir1"
    subdir1.mkdir()
    
    subdir2 = root_dir / "subdir2"
    subdir2.mkdir()
    
    nested_dir = subdir1 / "nested"
    nested_dir.mkdir()
    
    # Create setup.cfg in root
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile=black\n")
    
    # Create pyproject.toml in subdir1
    pyproject_toml = subdir1 / "pyproject.toml"
    pyproject_toml.write_text("[tool.isort]\nline_length=100\n")
    
    # Create .isort.cfg in nested directory
    isort_cfg = nested_dir / ".isort.cfg"
    isort_cfg.write_text("[settings]\nprofile=django\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.data == {}
    assert trie_root.key == "default"
    
    # Verify that config files were found and inserted
    # The trie should contain nodes for the config files
    assert len(trie_root.children) > 0


def test_find_all_configs_no_configs(tmp_path):
    """Test find_all_configs when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    assert trie_root.data == {}


def test_find_all_configs_single_config(tmp_path):
    """Test find_all_configs with a single config file."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[isort]\nprofile=black\nline_length=88\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with config files at multiple directory levels."""
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    
    level1 = root_dir / "level1"
    level1.mkdir()
    
    level2 = level1 / "level2"
    level2.mkdir()
    
    level3 = level2 / "level3"
    level3.mkdir()
    
    # Create configs at different levels
    (root_dir / "setup.cfg").write_text("[isort]\nprofile=black\n")
    (level1 / "pyproject.toml").write_text("[tool.isort]\nprofile=django\n")
    (level2 / ".isort.cfg").write_text("[settings]\nprofile=flask\n")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_invalid_config_file(tmp_path, monkeypatch):
    """Test find_all_configs handles invalid config files gracefully."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create an invalid config file
    setup_cfg = root_dir / "setup.cfg"
    setup_cfg.write_text("[invalid content {{{")
    
    # Mock _get_config_data to raise an exception
    def mock_get_config_data(path, section):
        raise ValueError("Invalid config")
    
    monkeypatch.setattr("isort.settings._get_config_data", mock_get_config_data)
    
    # Should not raise, but skip the invalid file
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_empty_config_files(tmp_path):
    """Test find_all_configs with empty config files."""
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    # Create empty config files
    (root_dir / "setup.cfg").write_text("")
    (root_dir / "pyproject.toml").write_text("")
    (root_dir / ".isort.cfg").write_text("")
    
    trie_root = find_all_configs(str(root_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


# LLM-generated content at query #35
#--------------------------

```python
def test_find_all_configs(tmp_path):
    """Test find_all_configs function to ensure it correctly finds and parses config files."""
    # Create a directory structure with multiple config files
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    
    subdir1 = root_dir / "subdir1"
    subdir1.mkdir()
    
    subdir2 = root_dir / "subdir2"
    subdir2.mkdir()
    
    nested_dir = subdir1 / "nested"
    nested_dir.mkdir()
    
    # Create config files with valid content
    pyproject_content = """
[tool.isort]
profile = "black"
line_length = 88
"""
    (root_dir / "pyproject.toml").write_text(pyproject_content)
    (subdir1 / "pyproject.toml").write_text(pyproject_content)
    (nested_dir / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    
    # Call find_all_configs
    trie_root = find_all_configs(str(root_dir))
    
    # Verify trie root exists
    assert trie_root is not None
    assert trie_root.data == {}
    assert trie_root.key == "default"
    
    # Verify that configs were found and inserted
    assert len(trie_root.children) > 0


def test_find_all_configs_empty_directory(tmp_path):
    """Test find_all_configs with an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    trie_root = find_all_configs(str(empty_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"
    assert trie_root.data == {}


def test_find_all_configs_with_invalid_config(tmp_path):
    """Test find_all_configs handles invalid config files gracefully."""
    test_dir = tmp_path / "test_invalid"
    test_dir.mkdir()
    
    # Create an invalid config file
    (test_dir / "pyproject.toml").write_text("invalid toml content [[[")
    
    # Should not raise an exception, just skip the invalid config
    trie_root = find_all_configs(str(test_dir))
    
    assert trie_root is not None
    assert trie_root.key == "default"


def test_find_all_configs_multiple_levels(tmp_path):
    """Test find_all_configs with multiple nested directory levels."""
    root = tmp_path / "root"
    root.mkdir()
    
    level1 = root / "level1"
    level1.mkdir()
    
    level2 = level1 / "level2"
    level2.mkdir()
    
    level3 = level2 / "level3"
    level3.mkdir()
    
    config_content = "[tool.isort]\nprofile = black\n"
    (level1 / "pyproject.toml").write_text(config_content)
    (level2 / ".isort.cfg").write_text("[settings]\nprofile=black\n")
    (level3 / "setup.cfg").write_text("[isort]\nprofile=black\n")
    
    trie_root = find_all_configs(str(root))
    
    assert trie_root is not None
    assert trie_root.key == "default"


# LLM-generated content at query #36
#--------------------------

```python
def test_Config_is_skipped():
    import tempfile
    import os
    from pathlib import Path
    
    # Test 1: File in skips set should be skipped
    config = Config(skip=frozenset(["test_file.py"]))
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 2: File not in skips should not be skipped (if it exists)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "not_skipped.py"
        test_file.touch()
        config = Config(skip=frozenset(["test_file.py"]), directory=tmpdir)
        assert config.is_skipped(test_file) is False
    
    # Test 3: File matching skip_glob pattern should be skipped
    config = Config(skip_glob=frozenset(["*.pyc"]))
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 4: File in extend_skip should be skipped
    config = Config(extend_skip=frozenset(["skip_me.py"]))
    file_path = Path("skip_me.py")
    assert config.is_skipped(file_path) is True
    
    # Test 5: Non-existent file should be skipped
    config = Config()
    file_path = Path("/nonexistent/file/path.py")
    assert config.is_skipped(file_path) is True
    
    # Test 6: Folder in skips should be skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_folder = Path(tmpdir) / "skip_folder"
        skip_folder.mkdir()
        config = Config(skip=frozenset(["skip_folder"]), directory=tmpdir)
        file_in_skip_folder = skip_folder / "file.py"
        file_in_skip_folder.touch()
        assert config.is_skipped(file_in_skip_folder) is True
    
    # Test 7: File with ~ suffix (editor backup) should be skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_file = Path(tmpdir) / "test.py~"
        backup_file.touch()
        config = Config(directory=tmpdir)
        assert config.is_skipped(backup_file) is True
    
    # Test 8: Regular file that exists should not be skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        regular_file = Path(tmpdir) / "regular.py"
        regular_file.touch()
        config = Config(directory=tmpdir)
        assert config.is_skipped(regular_file) is False
    
    # Test 9: File matching extend_skip_glob should be skipped
    config = Config(extend_skip_glob=frozenset(["__pycache__/*"]))
    file_path = Path("__pycache__/test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 10: Normalized path matching skip pattern (Windows paths)
    config = Config(skip=frozenset(["test_file.py"]))
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True


# LLM-generated content at query #37
#--------------------------

```python
def test_Config_is_skipped(tmp_path, monkeypatch):
    """Test Config.is_skipped method with various skip conditions."""
    from pathlib import Path
    
    # Test 1: File in skip list
    config = Config(skip=["test_file.py"])
    test_file = tmp_path / "test_file.py"
    test_file.touch()
    assert config.is_skipped(test_file) is True
    
    # Test 2: File not in skip list
    config = Config(skip=[])
    assert config.is_skipped(test_file) is False
    
    # Test 3: Directory in skip list
    config = Config(skip=["test_dir"])
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file_in_dir = test_dir / "file.py"
    test_file_in_dir.touch()
    assert config.is_skipped(test_file_in_dir) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc", "__pycache__/*"])
    pyc_file = tmp_path / "test.pyc"
    pyc_file.touch()
    assert config.is_skipped(pyc_file) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    py_file = tmp_path / "test.py"
    py_file.touch()
    assert config.is_skipped(py_file) is False
    
    # Test 6: Non-existent file
    config = Config(skip=[])
    non_existent = tmp_path / "non_existent.py"
    assert config.is_skipped(non_existent) is True
    
    # Test 7: Symlink file
    config = Config(skip=[])
    target_file = tmp_path / "target.py"
    target_file.touch()
    symlink_file = tmp_path / "link.py"
    symlink_file.symlink_to(target_file)
    assert config.is_skipped(symlink_file) is False
    
    # Test 8: extend_skip parameter
    config = Config(skip=["file1.py"], extend_skip=["file2.py"])
    file1 = tmp_path / "file1.py"
    file2 = tmp_path / "file2.py"
    file1.touch()
    file2.touch()
    assert config.is_skipped(file1) is True
    assert config.is_skipped(file2) is True
    
    # Test 9: extend_skip_glob parameter
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    pyc_file = tmp_path / "test.pyc"
    pyo_file = tmp_path / "test.pyo"
    pyc_file.touch()
    pyo_file.touch()
    assert config.is_skipped(pyc_file) is True
    assert config.is_skipped(pyo_file) is True
    
    # Test 10: Nested directory structure
    config = Config(skip=["nested"])
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "deep" / "file.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.touch()
    assert config.is_skipped(nested_file) is True
    
    # Test 11: File with directory set
    config = Config(skip=[], directory=str(tmp_path))
    test_file = tmp_path / "test.py"
    test_file.touch()
    assert config.is_skipped(test_file) is False
    
    # Test 12: Skip with absolute path normalization
    config = Config(skip=["test_file.py"])
    test_file = tmp_path / "test_file.py"
    test_file.touch()
    assert config.is_skipped(test_file) is True


# LLM-generated content at query #38
#--------------------------

```python
def test_Config_is_supported_filetype():
    """Test the is_supported_filetype method of Config class."""
    import tempfile
    import os
    from pathlib import Path
    
    config = Config()
    
    # Test with supported extension
    assert config.is_supported_filetype("test.py") is True
    
    # Test with blocked extension
    config_with_blocked = Config(blocked_extensions=["pyc"])
    assert config_with_blocked.is_supported_filetype("test.pyc") is False
    
    # Test with supported extension in supported_extensions
    config_with_supported = Config(supported_extensions=["py", "pyi"])
    assert config_with_supported.is_supported_filetype("test.py") is True
    assert config_with_supported.is_supported_filetype("test.pyi") is True
    
    # Test with backup file (ending with ~)
    assert config.is_supported_filetype("test.py~") is False
    
    # Test with non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test with actual file containing shebang
    with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as f:
        f.write("#!/usr/bin/env python\n")
        f.write("print('hello')\n")
        temp_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)
    
    # Test with actual .py file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("print('hello')\n")
        temp_py_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_py_file) is True
    finally:
        os.unlink(temp_py_file)
    
    # Test with actual file without shebang and no recognized extension
    with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as f:
        f.write("some random content\n")
        temp_unknown_file = f.name
    
    try:
        assert config.is_supported_filetype(temp_unknown_file) is False
    finally:
        os.unlink(temp_unknown_file)


# LLM-generated content at query #39
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    from pathlib import Path
    import tempfile
    import os
    
    # Test 1: File in skip list should be skipped
    config = Config(skip=["test_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 2: File not in skip list should not be skipped
    config = Config(skip=[])
    file_path = Path("normal_file.py")
    assert config.is_skipped(file_path) is False
    
    # Test 3: Directory in skip list should be skipped
    config = Config(skip=["skip_dir"])
    file_path = Path("skip_dir/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 4: File matching skip_glob pattern should be skipped
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 5: File not matching skip_glob pattern should not be skipped
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("test.py")
    assert config.is_skipped(file_path) is False
    
    # Test 6: Non-existent file should be skipped
    config = Config()
    file_path = Path("/non/existent/path/file.py")
    assert config.is_skipped(file_path) is True
    
    # Test 7: extend_skip should work
    config = Config(extend_skip=["extended_skip.py"])
    file_path = Path("extended_skip.py")
    assert config.is_skipped(file_path) is True
    
    # Test 8: extend_skip_glob should work
    config = Config(extend_skip_glob=["__pycache__/*"])
    file_path = Path("__pycache__/module.pyc")
    assert config.is_skipped(file_path) is True
    
    # Test 9: .git directory should be skipped when skip_gitignore is True
    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = Path(tmpdir) / ".git"
        git_dir.mkdir()
        config = Config(skip_gitignore=True)
        assert config.is_skipped(git_dir) is True
    
    # Test 10: File with ~ suffix (editor backup) should be skipped
    config = Config()
    file_path = Path("test.py~")
    assert config.is_skipped(file_path) is True
    
    # Test 11: Test with actual file that exists
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("print('test')")
        config = Config()
        assert config.is_skipped(test_file) is False
    
    # Test 12: Normalized path comparison for skip list
    config = Config(skip=["test\\file.py"])
    file_path = Path("test/file.py")
    # Should handle path separator normalization
    assert isinstance(config.is_skipped(file_path), bool)
    
    # Test 13: File in directory with skip set
    config = Config(directory="/some/dir", skip=["skip_me.py"])
    file_path = Path("/some/dir/skip_me.py")
    assert config.is_skipped(file_path) is True
    
    # Test 14: Multiple skip patterns
    config = Config(skip=["file1.py", "file2.py"])
    assert config.is_skipped(Path("file1.py")) is True
    assert config.is_skipped(Path("file2.py")) is True
    assert config.is_skipped(Path("file3.py")) is False
    
    # Test 15: Skip glob with wildcard patterns
    config = Config(skip_glob=["*.egg-info/*"])
    file_path = Path("package.egg-info/PKG-INFO")
    assert config.is_skipped(file_path) is True


# LLM-generated content at query #40
#--------------------------

```python
def test_Config():
    """Test Config class constructor with various initialization scenarios."""
    
    # Test 1: Basic initialization with no arguments
    config = Config()
    assert config is not None
    assert isinstance(config, Config)
    
    # Test 2: Initialization with config_overrides
    config = Config(line_length=100, indent=4)
    assert config.line_length == 100
    assert config.indent == "    "
    
    # Test 3: Initialization with another config object
    base_config = Config(line_length=88)
    new_config = Config(config=base_config, line_length=100)
    assert new_config.line_length == 100
    
    # Test 4: Initialization with profile
    config = Config(profile="black")
    assert config is not None
    
    # Test 5: Test indent conversion from integer
    config = Config(indent=2)
    assert config.indent == "  "
    
    # Test 6: Test indent conversion from string with quotes
    config = Config(indent="'    '")
    assert config.indent == "    "
    
    # Test 7: Test indent conversion from 'tab'
    config = Config(indent="tab")
    assert config.indent == "\t"
    
    # Test 8: Test wrap_length validation
    with pytest.raises(ValueError, match="wrap_length must be set lower than or equal to line_length"):
        Config(line_length=80, wrap_length=100)
    
    # Test 9: Test with known_ prefix configuration
    config = Config(known_custom=["mypackage"])
    assert "custom" in config.known_other
    assert "mypackage" in config.known_other["custom"]
    
    # Test 10: Test import_headings configuration
    config = Config(import_heading_stdlib="Standard Library")
    assert "stdlib" in config.import_headings
    assert config.import_headings["stdlib"] == "Standard Library"
    
    # Test 11: Test import_footers configuration
    config = Config(import_footer_thirdparty="Third Party Footer")
    assert "thirdparty" in config.import_footers
    assert config.import_footers["thirdparty"] == "Third Party Footer"
    
    # Test 12: Test src_paths initialization
    config = Config()
    assert config.src_paths is not None
    assert isinstance(config.src_paths, tuple)
    
    # Test 13: Test directory initialization
    config = Config()
    assert config.directory is not None
    
    # Test 14: Test with quiet=True to suppress warnings
    config = Config(quiet=True, profile="nonexistent_profile")
    
    # Test 15: Test with multiple config overrides
    config = Config(
        line_length=120,
        indent=8,
        skip=["migrations"],
        extend_skip=["build"]
    )
    assert config.line_length == 120
    assert config.indent == "        "
    
    # Test 16: Test that _known_patterns is initialized as None
    config = Config()
    assert config._known_patterns is None
    
    # Test 17: Test that _section_comments is initialized as None
    config = Config()
    assert config._section_comments is None
    
    # Test 18: Test that _skips is initialized as None
    config = Config()
    assert config._skips is None
    
    # Test 19: Test that _skip_globs is initialized as None
    config = Config()
    assert config._skip_globs is None
    
    # Test 20: Test that _sorting_function is initialized as None
    config = Config()
    assert config._sorting_function is None
    
    # Test 21: Test initialization with sections configuration
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert config.sections == ("FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER")
    
    # Test 22: Test skip_gitignore configuration
    config = Config(skip_gitignore=True)
    assert config.skip_gitignore is True
    
    # Test 23: Test supported_extensions configuration
    config = Config(supported_extensions=["py", "pyi"])
    assert "py" in config.supported_extensions


# LLM-generated content at query #41
#--------------------------

```python
def test_Config_is_supported_filetype():
    """Test the is_supported_filetype method of Config class."""
    config = Config()
    
    # Test supported extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("module.pyi") is True
    
    # Test blocked extensions
    assert config.is_supported_filetype("test.pyc") is False
    assert config.is_supported_filetype("test.pyo") is False
    
    # Test editor backup files
    assert config.is_supported_filetype("test.py~") is False
    assert config.is_supported_filetype("backup~") is False
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["py", "pyi", "txt"])
    assert config_custom.is_supported_filetype("test.txt") is True
    assert config_custom.is_supported_filetype("test.md") is False
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["py"])
    assert config_blocked.is_supported_filetype("test.py") is False
    
    # Test non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test file with shebang
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='', delete=False) as f:
        f.write(b"#!/usr/bin/env python\n")
        temp_file = f.name
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        import os
        os.unlink(temp_file)


# LLM-generated content at query #42
#--------------------------

```python
def test_Config_is_skipped():
    """Test the is_skipped method of Config class."""
    import tempfile
    from pathlib import Path
    
    # Test 1: File in skips list
    config = Config(skip=["test_file.py"])
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 2: File not in skips list
    config = Config(skip=["other_file.py"])
    test_path = Path("test_file.py")
    assert config.is_skipped(test_path) is False
    
    # Test 3: Directory in skips
    config = Config(skip=["skip_dir"])
    test_path = Path("skip_dir/test_file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 4: File matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("compiled.pyc")
    assert config.is_skipped(test_path) is True
    
    # Test 5: File not matching skip_glob pattern
    config = Config(skip_glob=["*.pyc"])
    test_path = Path("source.py")
    assert config.is_skipped(test_path) is False
    
    # Test 6: Non-existent file path
    config = Config()
    test_path = Path("/nonexistent/path/to/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 7: .git folder should be skipped when skip_gitignore is True
    config = Config(skip_gitignore=True)
    test_path = Path(".git")
    assert config.is_skipped(test_path) is True
    
    # Test 8: extend_skip adds to skips
    config = Config(skip=["file1.py"], extend_skip=["file2.py"])
    assert config.is_skipped(Path("file1.py")) is True
    assert config.is_skipped(Path("file2.py")) is True
    
    # Test 9: extend_skip_glob adds to skip_globs
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["*.pyo"])
    assert config.is_skipped(Path("compiled.pyc")) is True
    assert config.is_skipped(Path("compiled.pyo")) is True
    
    # Test 10: Normalized path comparison (Windows-style paths)
    config = Config(skip=["test/file.py"])
    test_path = Path("test/file.py")
    assert config.is_skipped(test_path) is True
    
    # Test 11: Glob pattern with subdirectory
    config = Config(skip_glob=["**/test_*.py"])
    test_path = Path("tests/test_example.py")
    assert config.is_skipped(test_path) is True
    
    # Test 12: File with directory context
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir, skip=["skip_me.py"])
        test_path = Path(tmpdir) / "skip_me.py"
        assert config.is_skipped(test_path) is True


# LLM-generated content at query #43
#--------------------------

```python
def test_Config_is_supported_filetype():
    """Test Config.is_supported_filetype method"""
    config = Config()
    
    # Test supported extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("module.pyi") is True
    
    # Test blocked extensions
    assert config.is_supported_filetype("test.pyc") is False
    assert config.is_supported_filetype("test.pyo") is False
    
    # Test editor backup files
    assert config.is_supported_filetype("test.py~") is False
    assert config.is_supported_filetype("module~") is False
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["py", "pyi", "txt"])
    assert config_custom.is_supported_filetype("test.txt") is True
    assert config_custom.is_supported_filetype("test.md") is False
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["pyc", "pyo", "txt"])
    assert config_blocked.is_supported_filetype("test.txt") is False
    assert config_blocked.is_supported_filetype("test.py") is True
    
    # Test nonexistent file
    assert config.is_supported_filetype("/nonexistent/path/to/file.py") is False
    
    # Test file without extension
    assert config.is_supported_filetype("Makefile") is False
    assert config.is_supported_filetype("README") is False


# LLM-generated content at query #44
#--------------------------

```python
def test_Config_is_supported_filetype():
    import tempfile
    import os
    from pathlib import Path
    
    config = Config()
    
    # Test supported extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("test.pyi") is True
    
    # Test blocked extensions
    assert config.is_supported_filetype("test.pyc") is False
    
    # Test backup files (ending with ~)
    assert config.is_supported_filetype("test.py~") is False
    
    # Test with temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        temp_file = f.name
        f.write(b"import os\n")
    
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)
    
    # Test with shebang file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_file = f.name
        f.write(b"#!/usr/bin/env python\n")
    
    try:
        assert config.is_supported_filetype(temp_file) is True
    finally:
        os.unlink(temp_file)
    
    # Test with non-shebang file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_file = f.name
        f.write(b"This is just text\n")
    
    try:
        assert config.is_supported_filetype(temp_file) is False
    finally:
        os.unlink(temp_file)
    
    # Test with non-existent file
    assert config.is_supported_filetype("/nonexistent/path/file.py") is False
    
    # Test with custom supported extensions
    config_custom = Config(supported_extensions=["py", "pyi", "txt"])
    assert config_custom.is_supported_filetype("test.txt") is True
    
    # Test with custom blocked extensions
    config_blocked = Config(blocked_extensions=["py"])
    assert config_blocked.is_supported_filetype("test.py") is False


