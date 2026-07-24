####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Config_is_supported_filetype():
    config = Config()

    # Test supported file extensions
    assert config.is_supported_filetype("test.py") is True
    assert config.is_supported_filetype("test.pyw") is True

    # Test blocked file extensions
    assert config.is_supported_filetype("test.jpg") is False
    assert config.is_supported_filetype("test.png") is False

    # Test editor backup files
    assert config.is_supported_filetype("test.py~") is False

    # Test FIFO files
    with patch("os.stat") as mock_stat:
        mock_stat.return_value.st_mode = stat.S_IFIFO
        assert config.is_supported_filetype("test.py") is False

    # Test shebang detection
    with patch("builtins.open", mock_open(read_data=b"#!/usr/bin/env python3\n")):
        assert config.is_supported_filetype("test") is True

    # Test non-existent file
    with patch("builtins.open", side_effect=OSError):
        assert config.is_supported_filetype("nonexistent.py") is False


# LLM-generated content at query #2
#--------------------------

```python
def test_Config():
    # Test default initialization
    config = Config()
    assert config.line_length == 79
    assert config.wrap_length == 50
    assert config.indent == "    "
    assert config.profile == ""

    # Test initialization with config overrides
    config = Config(quiet=True, line_length=100, indent="\t")
    assert config.quiet is True
    assert config.line_length == 100
    assert config.indent == "\t"

    # Test initialization with invalid wrap_length
    with pytest.raises(ValueError):
        Config(wrap_length=100, line_length=50)

    # Test initialization with settings_file
    with pytest.raises(FileNotFoundError):
        Config(settings_file="nonexistent_file.py")

    # Test initialization with settings_path
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

    # Test initialization with profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")

    # Test initialization with config object
    base_config = _Config()
    config = Config(config=base_config)
    assert config.line_length == base_config.line_length
    assert config.wrap_length == base_config.wrap_length

    # Test initialization with unsupported config options
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

    # Test initialization with deprecated config options
    with pytest.warns(UserWarning):
        Config(quiet=False, deprecated_option="value")

    # Test initialization with known_other sections
    config = Config(known_other={"custom": {"custom_module"}})
    assert "custom" in config.known_other
    assert "custom_module" in config.known_other["custom"]

    # Test initialization with import_headings
    config = Config(import_heading_custom="Custom Heading")
    assert "custom" in config.import_headings
    assert config.import_headings["custom"] == "Custom Heading"

    # Test initialization with import_footers
    config = Config(import_footer_custom="Custom Footer")
    assert "custom" in config.import_footers
    assert config.import_footers["custom"] == "Custom Footer"

    # Test initialization with src_paths
    config = Config(src_paths=["src"])
    assert len(config.src_paths) == 1
    assert "src" in str(config.src_paths[0])

    # Test initialization with formatter
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent_formatter")

    # Test initialization with sort_order
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"
    assert config.sorting_function == sorting.naturally

    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="nonexistent_sort_order")


# LLM-generated content at query #3
#--------------------------

```python
def test_find_all_configs(tmp_path):
    # Test case 1: No config files in directory
    trie = find_all_configs(str(tmp_path))
    assert trie.data == {}
    assert trie.children == {}

    # Test case 2: Single config file in root directory
    config_file = tmp_path / "setup.cfg"
    config_file.write_text("[isort]\nprofile=black")
    trie = find_all_configs(str(tmp_path))
    assert len(trie.children) == 1
    assert "setup.cfg" in trie.children

    # Test case 3: Multiple config files in different directories
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    config_file1 = subdir / "pyproject.toml"
    config_file1.write_text('[tool.isort]\nprofile="black"')
    config_file2 = tmp_path / ".isort.cfg"
    config_file2.write_text("profile=black")
    trie = find_all_configs(str(tmp_path))
    assert len(trie.children) == 2
    assert "pyproject.toml" in trie.children
    assert ".isort.cfg" in trie.children

    # Test case 4: Invalid config file (should be skipped)
    invalid_config = tmp_path / "invalid.cfg"
    invalid_config.write_text("invalid content")
    trie = find_all_configs(str(tmp_path))
    assert "invalid.cfg" not in trie.children

    # Test case 5: Nested directories with config files
    nested_dir = subdir / "nested"
    nested_dir.mkdir()
    config_file3 = nested_dir / "setup.cfg"
    config_file3.write_text("[isort]\nprofile=black")
    trie = find_all_configs(str(tmp_path))
    assert len(trie.children) == 3
    assert "setup.cfg" in trie.children["subdir"].children


# LLM-generated content at query #4
#--------------------------

```python
def test_find_all_configs(tmp_path):
    # Create test config files
    config1 = tmp_path / "config1.py"
    config1.write_text("setting1 = 'value1'")
    config2 = tmp_path / "subdir" / "config2.py"
    config2.parent.mkdir()
    config2.write_text("setting2 = 'value2'")
    config3 = tmp_path / "subdir" / "nested" / "config3.py"
    config3.parent.mkdir(parents=True)
    config3.write_text("setting3 = 'value3'")

    # Create non-config file
    non_config = tmp_path / "not_config.txt"
    non_config.write_text("not a config")

    # Test finding all configs
    trie = find_all_configs(str(tmp_path))

    # Verify configs were found
    assert str(config1) in trie
    assert str(config2) in trie
    assert str(config3) in trie
    assert str(non_config) not in trie

    # Verify config data was parsed
    assert trie[str(config1)] == {"setting1": "value1"}
    assert trie[str(config2)] == {"setting2": "value2"}
    assert trie[str(config3)] == {"setting3": "value3"}

    # Test with empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_trie = find_all_configs(str(empty_dir))
    assert len(empty_trie) == 0

    # Test with directory containing no config files
    no_config_dir = tmp_path / "no_config"
    no_config_dir.mkdir()
    (no_config_dir / "file.txt").write_text("content")
    no_config_trie = find_all_configs(str(no_config_dir))
    assert len(no_config_trie) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_Config():
    # Test default initialization
    config = Config()
    assert config is not None

    # Test initialization with settings_file
    with pytest.raises(FileNotFoundError):
        Config(settings_file="nonexistent_file.py")

    # Test initialization with settings_path
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

    # Test initialization with config object
    base_config = _Config()
    config = Config(config=base_config)
    assert config is not None

    # Test initialization with config_overrides
    config = Config(quiet=True)
    assert config is not None

    # Test initialization with profile
    config = Config(profile="black")
    assert config is not None

    # Test initialization with invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="invalid_profile")

    # Test initialization with unsupported config options
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

    # Test initialization with deprecated options
    with pytest.warns(UserWarning):
        Config(skip="deprecated_option")

    # Test initialization with known_other sections
    config = Config(known_other={"custom_section": {"custom_module"}})
    assert config is not None

    # Test initialization with import_headings
    config = Config(known_custom_section={"custom_module"}, sections=("CUSTOM_SECTION",))
    assert config is not None

    # Test initialization with import_headings and import_footers
    config = Config(import_heading_custom_section="Custom Section", import_footer_custom_section="Custom Footer")
    assert config is not None

    # Test initialization with formatter
    config = Config(formatter="black")
    assert config is not None

    # Test initialization with invalid formatter
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="invalid_formatter")

    # Test initialization with sorting_function
    config = Config(sort_order="natural")
    assert config is not None

    # Test initialization with invalid sorting_function
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="invalid_sort_order")


# LLM-generated content at query #6
#--------------------------

```python
def test_Config_is_skipped():
    # Test case 1: File is in skip list
    config = Config(skip=["test.py"])
    assert config.is_skipped(Path("test.py")) is True

    # Test case 2: File is not in skip list
    config = Config(skip=["other.py"])
    assert config.is_skipped(Path("test.py")) is False

    # Test case 3: File matches skip_glob pattern
    config = Config(skip_glob=["*.txt"])
    assert config.is_skipped(Path("test.txt")) is True

    # Test case 4: File does not match skip_glob pattern
    config = Config(skip_glob=["*.txt"])
    assert config.is_skipped(Path("test.py")) is False

    # Test case 5: File is in a directory that is in skip list
    config = Config(skip=["skip_dir"])
    assert config.is_skipped(Path("skip_dir/test.py")) is True

    # Test case 6: File is not in a directory that is in skip list
    config = Config(skip=["skip_dir"])
    assert config.is_skipped(Path("other_dir/test.py")) is False

    # Test case 7: File is a directory
    config = Config()
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    assert config.is_skipped(test_dir) is False
    test_dir.rmdir()

    # Test case 8: File does not exist
    config = Config()
    assert config.is_skipped(Path("nonexistent.py")) is True

    # Test case 9: File is a symlink
    config = Config()
    test_file = Path("test_file.py")
    test_file.touch()
    symlink = Path("symlink.py")
    symlink.symlink_to(test_file)
    assert config.is_skipped(symlink) is False
    symlink.unlink()
    test_file.unlink()

    # Test case 10: File is skipped due to gitignore
    config = Config(skip_gitignore=True)
    git_dir = Path(".git")
    git_dir.mkdir(exist_ok=True)
    assert config.is_skipped(git_dir) is True
    git_dir.rmdir()

    # Test case 11: File is not skipped when skip_gitignore is False
    config = Config(skip_gitignore=False)
    assert config.is_skipped(Path("test.py")) is False


# LLM-generated content at query #7
#--------------------------

```python
def test_Config_is_supported_filetype():
    config = Config()

    # Test supported extension
    assert config.is_supported_filetype("test.py") is True

    # Test blocked extension
    config.blocked_extensions = ("py",)
    assert config.is_supported_filetype("test.py") is False

    # Test editor backup file
    assert config.is_supported_filetype("test.py~") is False

    # Test non-existent file
    assert config.is_supported_filetype("nonexistent.py") is False

    # Test file with shebang
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("#!/usr/bin/env python\n")
        f.flush()
        assert config.is_supported_filetype(f.name) is True
        os.unlink(f.name)

    # Test file without shebang
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')\n")
        f.flush()
        assert config.is_supported_filetype(f.name) is True
        os.unlink(f.name)

    # Test file with unsupported extension
    assert config.is_supported_filetype("test.txt") is False


# LLM-generated content at query #8
#--------------------------

```python
def test_find_all_configs(tmp_path):
    # Create test config files in different directories
    config_dir1 = tmp_path / "dir1"
    config_dir1.mkdir()
    config_file1 = config_dir1 / "setup.cfg"
    config_file1.write_text("[isort]\nprofile=black\n")

    config_dir2 = tmp_path / "dir2"
    config_dir2.mkdir()
    config_file2 = config_dir2 / ".isort.cfg"
    config_file2.write_text("[settings]\nline_length=120\n")

    # Create a directory without config files
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # Run find_all_configs
    trie_root = find_all_configs(str(tmp_path))

    # Verify the trie contains the expected config files
    assert trie_root.find(str(config_file1)) is not None
    assert trie_root.find(str(config_file2)) is not None

    # Verify the trie does not contain non-existent config files
    assert trie_root.find(str(empty_dir / "nonexistent.cfg")) is None

    # Verify the config data is correctly parsed
    config_data1 = trie_root.find(str(config_file1)).value
    assert config_data1.get("profile") == "black"

    config_data2 = trie_root.find(str(config_file2)).value
    assert config_data2.get("line_length") == 120

    # Verify the trie structure is correct
    assert len(trie_root.children) == 2  # dir1 and dir2
    assert str(config_dir1.name) in trie_root.children
    assert str(config_dir2.name) in trie_root.children


# LLM-generated content at query #9
#--------------------------

```python
def test_Config_is_supported_filetype():
    # Test with supported file extension
    config = Config()
    config.supported_extensions = {"py"}
    assert config.is_supported_filetype("test.py") is True

    # Test with blocked file extension
    config.blocked_extensions = {"txt"}
    assert config.is_supported_filetype("test.txt") is False

    # Test with editor backup file
    assert config.is_supported_filetype("test.py~") is False

    # Test with non-existent file
    assert config.is_supported_filetype("nonexistent.py") is False

    # Test with file that has shebang
    with open("test_script", "w") as f:
        f.write("#!/usr/bin/env python\n")
    assert config.is_supported_filetype("test_script") is True
    os.remove("test_script")

    # Test with file that doesn't have shebang
    with open("test_script_no_shebang", "w") as f:
        f.write("print('hello')\n")
    assert config.is_supported_filetype("test_script_no_shebang") is False
    os.remove("test_script_no_shebang")


# LLM-generated content at query #10
#--------------------------

```python
def test__Config___post_init__():
    # Test valid py_version
    config = _Config(py_version="38")
    assert config.py_version == "py38"

    # Test auto py_version
    config = _Config(py_version="auto")
    assert config.py_version == f"py{sys.version_info.major}{sys.version_info.minor}"

    # Test invalid py_version
    with pytest.raises(ValueError):
        _Config(py_version="invalid")

    # Test multi_line_output adjustment
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

    # Test force_alphabetical_sort implications
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True

    # Test wrap_length validation
    with pytest.raises(ValueError):
        _Config(wrap_length=100, line_length=79)

    # Test known_standard_library population
    config = _Config(py_version="38")
    assert config.known_standard_library == frozenset(getattr(stdlibs, "py38").stdlib)

    # Test hash
    config1 = _Config()
    config2 = _Config()
    assert hash(config1) == id(config1)
    assert hash(config2) == id(config2)


# LLM-generated content at query #11
#--------------------------

```python
def test_Config_is_supported_filetype():
    # Test with supported file extension
    config = Config()
    assert config.is_supported_filetype("test.py") is True

    # Test with blocked file extension
    config = Config(blocked_extensions=["txt"])
    assert config.is_supported_filetype("test.txt") is False

    # Test with editor backup file
    assert config.is_supported_filetype("test.py~") is False

    # Test with FIFO file
    with patch("os.stat") as mock_stat:
        mock_stat.return_value.st_mode = stat.S_IFIFO
        assert config.is_supported_filetype("test.py") is False

    # Test with file that has shebang
    with patch("builtins.open", mock_open(read_data=b"#!/usr/bin/env python\n")):
        assert config.is_supported_filetype("test.py") is True

    # Test with file that doesn't have shebang
    with patch("builtins.open", mock_open(read_data=b"print('hello')\n")):
        assert config.is_supported_filetype("test.py") is False

    # Test with file that raises OSError
    with patch("builtins.open", side_effect=OSError("File not found")):
        assert config.is_supported_filetype("test.py") is False


# LLM-generated content at query #12
#--------------------------

```python
def test_Config():
    # Test default initialization
    config = Config()
    assert config.wrap_length <= config.line_length
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

    # Test initialization with settings_file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nline_length=120\n")
        f.flush()
        config = Config(settings_file=f.name)
        assert config.line_length == 120
    os.unlink(f.name)

    # Test initialization with settings_path
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write('[tool.isort]\nline_length=120\n')
        config = Config(settings_path=tmpdir)
        assert config.line_length == 120

    # Test initialization with config object
    base_config = _Config(line_length=100)
    config = Config(config=base_config)
    assert config.line_length == 100

    # Test initialization with config_overrides
    config = Config(line_length=100)
    assert config.line_length == 100

    # Test wrap_length validation
    with pytest.raises(ValueError):
        Config(wrap_length=100, line_length=80)

    # Test profile loading
    config = Config(profile="black")
    assert config.line_length == 88

    # Test invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

    # Test invalid settings_path
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

    # Test deprecated options warning
    with pytest.warns(UserWarning):
        Config(force_grid_wrap=2)

    # Test unsupported config options
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

    # Test formatter plugin loading
    config = Config(formatter="black")
    assert config.formatting_function is not None

    # Test invalid formatter
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent")

    # Test sorting function
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

    # Test invalid sorting function
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="nonexistent")


# LLM-generated content at query #13
#--------------------------

```python
def test_Config():
    # Test default initialization
    config = Config()
    assert config.line_length == 79
    assert config.wrap_length == 5
    assert config.indent == "    "

    # Test initialization with config overrides
    config = Config(line_length=100, wrap_length=10, indent="\t")
    assert config.line_length == 100
    assert config.wrap_length == 10
    assert config.indent == "\t"

    # Test initialization with settings_file
    with patch("builtins.open", mock_open(read_data="[isort]\nline_length=120")):
        with patch("os.path.exists", return_value=True):
            config = Config(settings_file="test.ini")
            assert config.line_length == 120

    # Test initialization with settings_path
    with patch("os.path.exists", return_value=True):
        with patch("os.path.abspath", return_value="/test/path"):
            with patch("_find_config", return_value=("/test/path", {"line_length": 110})):
                config = Config(settings_path="/test/path")
                assert config.line_length == 110

    # Test initialization with config object
    base_config = _Config(line_length=90, wrap_length=5)
    config = Config(config=base_config, line_length=95)
    assert config.line_length == 95
    assert config.wrap_length == 5

    # Test profile handling
    with patch("profiles", {"black": {"line_length": 88}}):
        config = Config(profile="black")
        assert config.line_length == 88

    # Test invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

    # Test invalid settings path
    with patch("os.path.exists", return_value=False):
        with pytest.raises(InvalidSettingsPath):
            Config(settings_path="/invalid/path")

    # Test unsupported settings
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_setting="value")

    # Test deprecated settings warning
    with pytest.warns(UserWarning):
        Config(force_single_line=True)

    # Test wrap_length validation
    with pytest.raises(ValueError):
        Config(wrap_length=10, line_length=5)

    # Test indent parsing
    config = Config(indent="4")
    assert config.indent == "    "
    config = Config(indent="tab")
    assert config.indent == "\t"
    config = Config(indent="    ")
    assert config.indent == "    "

    # Test known_other handling
    config = Config(known_foo=["bar"])
    assert "known_other" in config.__dict__
    assert "foo" in config.known_other

    # Test import_headings and import_footers
    config = Config(import_heading_foo="bar", import_footer_baz="qux")
    assert "import_headings" in config.__dict__
    assert "import_footers" in config.__dict__
    assert config.import_headings["foo"] == "bar"
    assert config.import_footers["baz"] == "qux"

    # Test src_paths handling
    with patch("os.getcwd", return_value="/test"):
        config = Config(src_paths=["src", "tests"])
        assert len(config.src_paths) == 2
        assert all(isinstance(p, Path) for p in config.src_paths)

    # Test formatter plugin
    with patch("entry_points") as mock_entry_points:
        mock_plugin = MagicMock()
        mock_plugin.name = "test_formatter"
        mock_plugin.load.return_value = lambda x: x
        mock_entry_points.return_value = [mock_plugin]
        config = Config(formatter="test_formatter")
        assert config.formatting_function is not None

    # Test invalid formatter
    with patch("entry_points", return_value=[]):
        with pytest.raises(FormattingPluginDoesNotExist):
            Config(formatter="nonexistent")

    # Test sorting function
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally
    config = Config(sort_order="native")
    assert config.sorting_function == sorted

    # Test invalid sorting function
    with patch("entry_points", return_value=[]):
        with pytest.raises(SortingFunctionDoesNotExist):
            Config(sort_order="nonexistent")


# LLM-generated content at query #14
#--------------------------

```python
def test_Config_is_skipped():
    # Test case 1: File is in skip list
    config = Config(skip={"test.py"})
    assert config.is_skipped(Path("test.py"))

    # Test case 2: File is not in skip list
    config = Config(skip={"other.py"})
    assert not config.is_skipped(Path("test.py"))

    # Test case 3: File matches skip glob
    config = Config(skip_glob={"test_*.py"})
    assert config.is_skipped(Path("test_file.py"))

    # Test case 4: File does not match skip glob
    config = Config(skip_glob={"other_*.py"})
    assert not config.is_skipped(Path("test_file.py"))

    # Test case 5: File is in skip_gitignore and not in git ls-files
    config = Config(skip_gitignore=True)
    config.git_ls_files[Path("/test")] = {"/test/file1.py"}
    assert config.is_skipped(Path("/test/file2.py"))

    # Test case 6: File is in skip_gitignore and in git ls-files
    config = Config(skip_gitignore=True)
    config.git_ls_files[Path("/test")] = {"/test/file1.py"}
    assert not config.is_skipped(Path("/test/file1.py"))

    # Test case 7: File is a directory
    config = Config()
    assert config.is_skipped(Path("/test/directory"))

    # Test case 8: File is a symlink
    config = Config()
    with patch("os.path.islink", return_value=True):
        assert not config.is_skipped(Path("/test/symlink"))

    # Test case 9: File does not exist
    config = Config()
    with patch("os.path.exists", return_value=False):
        assert config.is_skipped(Path("/test/nonexistent.py"))

    # Test case 10: File is a FIFO
    config = Config()
    with patch("os.stat", return_value=Mock(st_mode=stat.S_IFIFO)):
        assert config.is_skipped(Path("/test/fifo"))

    # Test case 11: File is a backup file
    config = Config()
    assert config.is_skipped(Path("test.py~"))

    # Test case 12: File is not a supported filetype
    config = Config()
    with patch("builtins.open", side_effect=OSError):
        assert not config.is_skipped(Path("test.unsupported"))

    # Test case 13: File is a supported filetype
    config = Config()
    with patch("builtins.open", return_value=Mock(readline=Mock(return_value=b"#!/usr/bin/env python"))):
        assert config.is_skipped(Path("test.py")) == False


# LLM-generated content at query #15
#--------------------------

```python
def test_Config_is_supported_filetype():
    config = Config()

    # Test supported file extension
    assert config.is_supported_filetype("test.py") is True

    # Test blocked file extension
    config.blocked_extensions = ("py",)
    assert config.is_supported_filetype("test.py") is False

    # Test editor backup file
    assert config.is_supported_filetype("test.py~") is False

    # Test non-existent file
    assert config.is_supported_filetype("nonexistent.py") is False

    # Test file with shebang
    with open("test_shebang.py", "w") as f:
        f.write("#!/usr/bin/env python\n")
    assert config.is_supported_filetype("test_shebang.py") is True
    os.remove("test_shebang.py")

    # Test file without shebang
    with open("test_no_shebang.py", "w") as f:
        f.write("print('hello')\n")
    assert config.is_supported_filetype("test_no_shebang.py") is True
    os.remove("test_no_shebang.py")

    # Test non-Python file
    with open("test.txt", "w") as f:
        f.write("hello\n")
    assert config.is_supported_filetype("test.txt") is False
    os.remove("test.txt")


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Config_is_skipped():
    # Test basic skip functionality
    config = Config(skip=["test.py"])
    assert config.is_skipped(Path("test.py")) is True
    assert config.is_skipped(Path("other.py")) is False

    # Test skip with directory
    config = Config(skip=["tests/"])
    assert config.is_skipped(Path("tests/test.py")) is True
    assert config.is_skipped(Path("src/test.py")) is False

    # Test skip_glob functionality
    config = Config(skip_glob=["*.tmp"])
    assert config.is_skipped(Path("file.tmp")) is True
    assert config.is_skipped(Path("file.py")) is False

    # Test skip_gitignore functionality
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files") as mock_check:
        mock_check.return_value = Path("/git/root")
        config.git_ls_files[Path("/git/root")] = {"/git/root/file1.py", "/git/root/file2.py"}
        assert config.is_skipped(Path("/git/root/file1.py")) is False
        assert config.is_skipped(Path("/git/root/file3.py")) is True

    # Test non-existent file
    config = Config()
    assert config.is_skipped(Path("nonexistent.py")) is True

    # Test directory skip
    config = Config(skip=["test_dir"])
    assert config.is_skipped(Path("test_dir")) is True
    assert config.is_skipped(Path("test_dir/file.py")) is True

    # Test combined skip and skip_glob
    config = Config(skip=["skipme.py"], skip_glob=["*.tmp"])
    assert config.is_skipped(Path("skipme.py")) is True
    assert config.is_skipped(Path("file.tmp")) is True
    assert config.is_skipped(Path("other.py")) is False


# LLM-generated content at query #2
#--------------------------

```python
def test_find_all_configs():
    # Test case 1: Empty directory
    with tempfile.TemporaryDirectory() as tmpdir:
        trie = find_all_configs(tmpdir)
        assert trie == Trie("default", {})

    # Test case 2: Directory with no config files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a subdirectory
        os.makedirs(os.path.join(tmpdir, "subdir"))
        trie = find_all_configs(tmpdir)
        assert trie == Trie("default", {})

    # Test case 3: Directory with one config file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a config file
        config_file = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file, "w") as f:
            f.write("[settings]\nline_length=88\n")
        trie = find_all_configs(tmpdir)
        assert trie.children[".isort.cfg"].value == {"line_length": 88}

    # Test case 4: Directory with multiple config files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config files
        config_file1 = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file1, "w") as f:
            f.write("[settings]\nline_length=88\n")
        config_file2 = os.path.join(tmpdir, "setup.cfg")
        with open(config_file2, "w") as f:
            f.write("[isort]\nline_length=120\n")
        trie = find_all_configs(tmpdir)
        assert trie.children[".isort.cfg"].value == {"line_length": 88}
        assert trie.children["setup.cfg"].value == {"line_length": 120}

    # Test case 5: Nested directories with config files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested directories with config files
        config_file1 = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file1, "w") as f:
            f.write("[settings]\nline_length=88\n")
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        config_file2 = os.path.join(subdir, "setup.cfg")
        with open(config_file2, "w") as f:
            f.write("[isort]\nline_length=120\n")
        trie = find_all_configs(tmpdir)
        assert trie.children[".isort.cfg"].value == {"line_length": 88}
        assert trie.children["subdir"].children["setup.cfg"].value == {"line_length": 120}

    # Test case 6: Invalid config file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an invalid config file
        config_file = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file, "w") as f:
            f.write("invalid config content")
        trie = find_all_configs(tmpdir)
        assert trie == Trie("default", {})


# LLM-generated content at query #3
#--------------------------

```python
def test_Config_is_skipped():
    # Test case 1: File is in skip list
    config = Config(skip={"test_file.py"})
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True

    # Test case 2: File is not in skip list
    config = Config(skip={"other_file.py"})
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is False

    # Test case 3: File matches skip_glob pattern
    config = Config(skip_glob={"test_*"})
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is True

    # Test case 4: File does not match skip_glob pattern
    config = Config(skip_glob={"other_*"})
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) is False

    # Test case 5: File is a directory
    config = Config()
    file_path = Path("test_directory")
    file_path.mkdir(exist_ok=True)
    assert config.is_skipped(file_path) is True
    file_path.rmdir()

    # Test case 6: File is a symlink
    config = Config()
    file_path = Path("test_file.py")
    file_path.touch()
    symlink_path = Path("symlink_to_test")
    symlink_path.symlink_to(file_path)
    assert config.is_skipped(symlink_path) is False
    symlink_path.unlink()
    file_path.unlink()

    # Test case 7: File is not a file, directory, or symlink
    config = Config()
    file_path = Path("non_existent_file.py")
    assert config.is_skipped(file_path) is True

    # Test case 8: File is skipped due to gitignore
    config = Config(skip_gitignore=True)
    file_path = Path("test_file.py")
    file_path.touch()
    git_folder = Path("test_git_folder")
    git_folder.mkdir(exist_ok=True)
    subprocess.run(["git", "init"], cwd=git_folder, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=git_folder, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=git_folder, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=git_folder, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=git_folder, capture_output=True)
    config.git_ls_files[git_folder] = {str(git_folder / Path(f)) for f in ["test_file.py"]}
    assert config.is_skipped(file_path) is False
    file_path.unlink()
    git_folder.rmdir()


# LLM-generated content at query #4
#--------------------------

```python
def test_Config():
    # Test default initialization
    config = Config()
    assert config.wrap_length <= config.line_length
    assert config.source == "defaults"
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

    # Test initialization with config overrides
    config = Config(quiet=True, line_length=100, wrap_length=90)
    assert config.line_length == 100
    assert config.wrap_length == 90
    assert config.quiet is True

    # Test initialization with invalid wrap_length
    with pytest.raises(ValueError):
        Config(wrap_length=120, line_length=100)

    # Test initialization with settings_file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nline_length=88\n")
        f.flush()
        config = Config(settings_file=f.name)
        assert config.line_length == 88
        os.unlink(f.name)

    # Test initialization with invalid settings_path
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

    # Test initialization with profile
    config = Config(profile="black")
    assert "black" in str(config.source)

    # Test initialization with invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

    # Test initialization with config object
    base_config = _Config(line_length=100, wrap_length=90)
    config = Config(config=base_config, quiet=True)
    assert config.line_length == 100
    assert config.wrap_length == 90
    assert config.quiet is True

    # Test initialization with unsupported config option
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

    # Test initialization with deprecated config option
    with pytest.warns(UserWarning):
        Config(quiet=False, virtual_env="test")

    # Test initialization with known_other sections
    config = Config(known_other={"test": {"testmodule"}})
    assert "test" in config.known_other

    # Test initialization with import_headings
    config = Config(import_heading_test="Test Heading")
    assert "test" in config.import_headings

    # Test initialization with import_footers
    config = Config(import_footer_test="Test Footer")
    assert "test" in config.import_footers

    # Test initialization with formatter plugin
    config = Config(formatter="black")
    assert config.formatting_function is not None

    # Test initialization with invalid formatter plugin
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent")

    # Test initialization with sorting function
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

    # Test initialization with invalid sorting function
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="nonexistent")


# LLM-generated content at query #5
#--------------------------

```python
def test_find_all_configs():
    # Test case 1: No config files in the directory
    with tempfile.TemporaryDirectory() as tmpdir:
        trie = find_all_configs(tmpdir)
        assert trie.value == "default"
        assert trie.children == {}

    # Test case 2: Single config file in the root directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file, "w") as f:
            f.write("[settings]\nline_length=88\n")

        trie = find_all_configs(tmpdir)
        assert len(trie.children) == 1
        assert config_file in trie.children
        assert trie.children[config_file].value == {"line_length": 88}

    # Test case 3: Multiple config files in different subdirectories
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir1 = os.path.join(tmpdir, "subdir1")
        subdir2 = os.path.join(tmpdir, "subdir2")
        os.makedirs(subdir1)
        os.makedirs(subdir2)

        config_file1 = os.path.join(subdir1, "setup.cfg")
        with open(config_file1, "w") as f:
            f.write("[settings]\nline_length=100\n")

        config_file2 = os.path.join(subdir2, "pyproject.toml")
        with open(config_file2, "w") as f:
            f.write("[tool.isort]\nprofile=black\n")

        trie = find_all_configs(tmpdir)
        assert len(trie.children) == 2
        assert config_file1 in trie.children
        assert trie.children[config_file1].value == {"line_length": 100}
        assert config_file2 in trie.children
        assert trie.children[config_file2].value == {"profile": "black"}

    # Test case 4: Invalid config file (should be skipped)
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file, "w") as f:
            f.write("invalid config content")

        trie = find_all_configs(tmpdir)
        assert trie.value == "default"
        assert trie.children == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_find_all_configs(tmp_path):
    # Create a temporary directory structure with config files
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()

    # Create a valid config file
    valid_config = config_dir / "setup.cfg"
    valid_config.write_text("[isort]\nprofile = black\n")

    # Create an invalid config file
    invalid_config = config_dir / "pyproject.toml"
    invalid_config.write_text("invalid toml content")

    # Create a subdirectory with a config file
    subdir = config_dir / "subdir"
    subdir.mkdir()
    sub_config = subdir / ".isort.cfg"
    sub_config.write_text("[isort]\nline_length = 100\n")

    # Test finding all configs
    trie = find_all_configs(str(config_dir))

    # Verify the trie contains the valid config files
    assert str(valid_config) in trie
    assert str(sub_config) in trie

    # Verify the invalid config file is not in the trie
    assert str(invalid_config) not in trie

    # Verify the config data is correct
    assert trie[str(valid_config)] == {"profile": "black"}
    assert trie[str(sub_config)] == {"line_length": 100}

    # Test with a directory that has no config files
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    empty_trie = find_all_configs(str(empty_dir))
    assert len(empty_trie) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_Config():
    # Test default initialization
    config = Config()
    assert config.line_length == 79
    assert config.wrap_length == 5
    assert config.indent == "    "
    assert config.profile == ""

    # Test initialization with config overrides
    config = Config(line_length=120, wrap_length=10, indent="\t")
    assert config.line_length == 120
    assert config.wrap_length == 10
    assert config.indent == "\t"

    # Test initialization with settings_file
    with pytest.raises(FileNotFoundError):
        Config(settings_file="nonexistent_file.py")

    # Test initialization with settings_path
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

    # Test initialization with config object
    base_config = _Config(line_length=100, wrap_length=5)
    config = Config(config=base_config, line_length=120)
    assert config.line_length == 120
    assert config.wrap_length == 5

    # Test initialization with profile
    config = Config(profile="black")
    assert config.profile == "black"

    # Test initialization with invalid profile
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="invalid_profile")

    # Test initialization with invalid wrap_length
    with pytest.raises(ValueError):
        Config(wrap_length=10, line_length=5)

    # Test initialization with deprecated options
    with pytest.warns(UserWarning):
        Config(quiet=True, deprecated_option="value")

    # Test initialization with unsupported config options
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

    # Test initialization with known_other sections
    config = Config(known_other={"custom": {"custom_module"}})
    assert config.known_other == {"custom": frozenset({"custom_module"})}

    # Test initialization with import_headings
    config = Config(import_headings={"custom": "Custom Heading"})
    assert config.import_headings == {"custom": "Custom Heading"}

    # Test initialization with import_footers
    config = Config(import_footers={"custom": "Custom Footer"})
    assert config.import_footers == {"custom": "Custom Footer"}

    # Test initialization with src_paths
    config = Config(src_paths=["src"])
    assert len(config.src_paths) == 1
    assert "src" in str(config.src_paths[0])

    # Test initialization with formatter
    config = Config(formatter="black")
    assert config.formatting_function is not None

    # Test initialization with invalid formatter
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="invalid_formatter")

    # Test initialization with sort_order
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

    # Test initialization with invalid sort_order
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="invalid_sort_order")


# LLM-generated content at query #8
#--------------------------

```python
def test_Config_is_skipped():
    # Test basic file skipping
    config = Config(skip=["test.py"])
    assert config.is_skipped(Path("test.py"))
    assert not config.is_skipped(Path("other.py"))

    # Test directory skipping
    config = Config(skip=["tests"])
    assert config.is_skipped(Path("tests/test.py"))
    assert not config.is_skipped(Path("src/test.py"))

    # Test glob pattern skipping
    config = Config(skip_glob=["*.tmp"])
    assert config.is_skipped(Path("file.tmp"))
    assert not config.is_skipped(Path("file.py"))

    # Test gitignore skipping
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files") as mock_check:
        mock_check.return_value = Path("/repo")
        config.git_ls_files[Path("/repo")] = {"/repo/src/file.py"}
        assert not config.is_skipped(Path("/repo/src/file.py"))
        assert config.is_skipped(Path("/repo/ignored.py"))

    # Test non-existent file
    assert not config.is_skipped(Path("nonexistent.py"))

    # Test special files
    assert config.is_skipped(Path(".git"))
    assert config.is_skipped(Path("file~"))

    # Test with directory setting
    config = Config(directory="/project", skip=["skip_me"])
    assert config.is_skipped(Path("/project/skip_me"))
    assert not config.is_skipped(Path("/other/skip_me"))


# LLM-generated content at query #9
#--------------------------

```python
def test_Config_is_skipped():
    # Test basic skip functionality
    config = Config(skip={"test.py"})
    assert config.is_skipped(Path("test.py"))
    assert not config.is_skipped(Path("other.py"))

    # Test skip with directory
    config = Config(skip={"tests/"})
    assert config.is_skipped(Path("tests/file.py"))
    assert not config.is_skipped(Path("src/file.py"))

    # Test skip_glob functionality
    config = Config(skip_glob={"*.tmp"})
    assert config.is_skipped(Path("file.tmp"))
    assert not config.is_skipped(Path("file.py"))

    # Test skip_gitignore functionality
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files") as mock_check:
        mock_check.return_value = Path("/repo")
        config.git_ls_files[Path("/repo")] = {"/repo/file1.py", "/repo/file2.py"}
        assert not config.is_skipped(Path("/repo/file1.py"))
        assert config.is_skipped(Path("/repo/file3.py"))

    # Test non-existent file
    config = Config()
    assert config.is_skipped(Path("nonexistent.py"))

    # Test editor backup file
    config = Config()
    assert config.is_skipped(Path("file.py~"))

    # Test directory in skip
    config = Config(skip={"dir_to_skip"})
    assert config.is_skipped(Path("dir_to_skip"))


# LLM-generated content at query #10
#--------------------------

```python
def test_Config_is_skipped():
    # Test case 1: File is in skips list
    config = Config(skip={"test.py"})
    assert config.is_skipped(Path("test.py")) is True

    # Test case 2: File is not in skips list
    config = Config(skip={"other.py"})
    assert config.is_skipped(Path("test.py")) is False

    # Test case 3: File matches skip_glob pattern
    config = Config(skip_glob={"*.tmp"})
    assert config.is_skipped(Path("test.tmp")) is True

    # Test case 4: File does not match skip_glob pattern
    config = Config(skip_glob={"*.tmp"})
    assert config.is_skipped(Path("test.py")) is False

    # Test case 5: File is a directory
    config = Config()
    with tempfile.TemporaryDirectory() as tmpdir:
        assert config.is_skipped(Path(tmpdir)) is False

    # Test case 6: File is a symlink
    config = Config()
    with tempfile.NamedTemporaryFile() as tmpfile:
        with tempfile.NamedTemporaryFile() as tmplink:
            os.symlink(tmpfile.name, tmplink.name)
            assert config.is_skipped(Path(tmplink.name)) is False

    # Test case 7: File is skipped due to gitignore
    config = Config(skip_gitignore=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = Path(tmpdir) / ".git"
        git_dir.mkdir()
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        assert config.is_skipped(test_file) is True

    # Test case 8: File is not skipped when skip_gitignore is False
    config = Config(skip_gitignore=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        git_dir = Path(tmpdir) / ".git"
        git_dir.mkdir()
        test_file = Path(tmpdir) / "test.py"
        test_file.touch()
        assert config.is_skipped(test_file) is False

    # Test case 9: File is skipped due to parent directory in skips
    config = Config(skip={"dir_to_skip"})
    with tempfile.TemporaryDirectory() as tmpdir:
        skip_dir = Path(tmpdir) / "dir_to_skip"
        skip_dir.mkdir()
        test_file = skip_dir / "test.py"
        test_file.touch()
        assert config.is_skipped(test_file) is True

    # Test case 10: File is not skipped when parent directory is not in skips
    config = Config(skip={"other_dir"})
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_dir"
        test_dir.mkdir()
        test_file = test_dir / "test.py"
        test_file.touch()
        assert config.is_skipped(test_file) is False


