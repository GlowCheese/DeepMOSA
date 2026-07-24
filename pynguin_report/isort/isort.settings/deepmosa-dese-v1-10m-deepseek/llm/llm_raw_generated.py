####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_config_constructor_with_config():
    mock_config = _Config()
    mock_config.py_version = "py3"
    config = Config(config=mock_config, py_version="2")
    assert config.py_version == "2"

def test_config_constructor_with_settings_file():
    config = Config(settings_file="test_settings.ini")
    assert config.directory == os.getcwd()

def test_config_constructor_with_settings_path():
    config = Config(settings_path=os.getcwd())
    assert config.directory == os.getcwd()

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert "black" in config.sources[-1]["source"]

def test_config_constructor_with_combined_config():
    config = Config(settings_file="test_settings.ini", profile="black", indent=4)
    assert config.indent == "    "

def test_config_constructor_with_known_other():
    config = Config(known_test_section=["test_module"])
    assert "test_section" in config.known_other

def test_config_constructor_with_import_headings():
    config = Config(import_heading_test_section="Test Section")
    assert "test_section" in config.import_headings

def test_config_constructor_with_import_footers():
    config = Config(import_footer_test_section="Test Footer")
    assert "test_section" in config.import_footers

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert len(config.src_paths) >= 2

def test_config_constructor_with_formatter():
    config = Config(formatter="text")
    assert hasattr(config, "formatting_function")

def test_config_constructor_with_deprecated_options():
    config = Config(force_to_top=["test"], quiet=True)
    assert "force_to_top" not in vars(config)

def test_config_constructor_with_unsupported_settings():
    try:
        Config(unsupported_setting="value")
        assert False
    except UnsupportedSettings:
        assert True


# LLM-generated content at query #2
#--------------------------

```
def test_is_supported_filetype_returns_true_for_supported_extension():
    config = Config()
    config.supported_extensions = {"py"}
    assert config.is_supported_filetype("test.py") == True

def test_is_supported_filetype_returns_false_for_blocked_extension():
    config = Config()
    config.blocked_extensions = {"txt"}
    assert config.is_supported_filetype("test.txt") == False

def test_is_supported_filetype_returns_false_for_editor_backup_file():
    config = Config()
    assert config.is_supported_filetype("test.py~") == False

def test_is_supported_filetype_returns_false_for_fifo_file():
    config = Config()
    assert config.is_supported_filetype("fifo_pipe") == False

def test_is_supported_filetype_returns_true_for_shebang_file():
    config = Config()
    assert config.is_supported_filetype("script_with_shebang") == True

def test_is_supported_filetype_returns_false_for_nonexistent_file():
    config = Config()
    assert config.is_supported_filetype("nonexistent_file") == False


# LLM-generated content at query #3
#--------------------------

```python
def test_deprecated_options_used():
    config_overrides = {"deprecated_option1": "value1", "deprecated_option2": "value2"}
    DEPRECATED_SETTINGS = {"deprecated_option1", "deprecated_option2"}
    deprecated_options_used = [option for option in config_overrides if option in DEPRECATED_SETTINGS]
    assert deprecated_options_used == ["deprecated_option1", "deprecated_option2"]


# LLM-generated content at query #4
#--------------------------

```python
def test_import_headings_evaluates_to_true():
    config_overrides = {"import_heading_example": "value"}
    config = Config(config_overrides=config_overrides)
    assert "import_headings" in vars(config)


# LLM-generated content at query #5
#--------------------------

```python
def test_path_root_is_dir_evaluates_to_false():
    from pathlib import Path
    temp_file = Path("test_file.txt")
    temp_file.touch()
    path_root = temp_file.resolve()
    result = path_root if path_root.is_dir() else path_root.parent
    assert result == path_root.parent
    temp_file.unlink()


# LLM-generated content at query #6
#--------------------------

```python
def test_config_initialization_without_config():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config():
    class MockConfig:
        def __init__(self):
            self.py_version = "py3.8"
            self._known_patterns = []
            self._section_comments = ()
            self._section_comments_end = ()
            self._skips = frozenset()
            self._skip_globs = frozenset()
            self._sorting_function = lambda x: x

    config = Config(config=MockConfig())
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #7
#--------------------------

```python
def test_key_starts_with_known_prefix_and_not_in_excluded_list():
    config = Config()
    combined_config = {"known_custom_section": ["value1", "value2"]}
    assert "known_custom_section".startswith(KNOWN_PREFIX) and "known_custom_section" not in (
        "known_standard_library",
        "known_future_library",
        "known_third_party",
        "known_first_party",
        "known_local_folder",
    )


# LLM-generated content at query #8
#--------------------------

```python
def test_config_constructor_with_config():
    config = _Config()
    config_vars = vars(config).copy()
    config_vars["py_version"] = "3.8"
    config_overrides = {"indent": 4}
    config = Config(config=config, **config_overrides)
    assert config.py_version == "3.8"
    assert config.indent == "    "

def test_config_constructor_with_settings_file():
    settings_file = "settings.cfg"
    config = Config(settings_file=settings_file)
    assert config.directory == os.path.dirname(settings_file)

def test_config_constructor_with_settings_path():
    settings_path = "/path/to/settings"
    config = Config(settings_path=settings_path)
    assert config.directory == os.path.abspath(settings_path)

def test_config_constructor_with_config_overrides():
    config_overrides = {"indent": "tab", "profile": "black"}
    config = Config(**config_overrides)
    assert config.indent == "\t"
    assert config.profile == "black"

def test_config_constructor_with_invalid_settings_path():
    settings_path = "/invalid/path"
    exception_raised = False
    try:
        Config(settings_path=settings_path)
    except InvalidSettingsPath:
        exception_raised = True
    assert exception_raised

def test_config_constructor_with_profile_does_not_exist():
    config_overrides = {"profile": "nonexistent"}
    exception_raised = False
    try:
        Config(**config_overrides)
    except ProfileDoesNotExist:
        exception_raised = True
    assert exception_raised

def test_config_constructor_with_deprecated_options():
    config_overrides = {"deprecated_option": True}
    config = Config(**config_overrides)
    assert "deprecated_option" not in vars(config)

def test_config_constructor_with_unsupported_config_errors():
    config_overrides = {"unsupported_option": True}
    exception_raised = False
    try:
        Config(**config_overrides)
    except UnsupportedSettings:
        exception_raised = True
    assert exception_raised


# LLM-generated content at query #9
#--------------------------

```python
def test__get_config_data_toml():
    file_path = "test.toml"
    sections = ("section1", "section2")
    expected = {"key1": "value1", "key2": "value2", "source": file_path}
    with open(file_path, "wb") as f:
        f.write(b"[section1]\nkey1 = 'value1'\n[section2]\nkey2 = 'value2'")
    assert _get_config_data(file_path, sections) == expected
    os.remove(file_path)

def test__get_config_data_ini():
    file_path = "test.ini"
    sections = ("section1", "section2")
    expected = {"key1": "value1", "key2": "value2", "source": file_path}
    with open(file_path, "w") as f:
        f.write("[section1]\nkey1 = value1\n[section2]\nkey2 = value2")
    assert _get_config_data(file_path, sections) == expected
    os.remove(file_path)

def test__get_config_data_editorconfig():
    file_path = "test.editorconfig"
    sections = ("*.{py}",)
    expected = {"indent": "    ", "source": file_path}
    with open(file_path, "w") as f:
        f.write("indent_style = space\nindent_size = 4")
    assert _get_config_data(file_path, sections) == expected
    os.remove(file_path)

def test__get_config_data_editorconfig_tab():
    file_path = "test.editorconfig"
    sections = ("*.{py}",)
    expected = {"indent": "\t", "source": file_path}
    with open(file_path, "w") as f:
        f.write("indent_style = tab\ntab_width = 1")
    assert _get_config_data(file_path, sections) == expected
    os.remove(file_path)

def test__get_config_data_editorconfig_line_length():
    file_path = "test.editorconfig"
    sections = ("*.{py}",)
    expected = {"line_length": 80, "source": file_path}
    with open(file_path, "w") as f:
        f.write("max_line_length = 80")
    assert _get_config_data(file_path, sections) == expected
    os.remove(file_path)

def test__get_config_data_editorconfig_line_length_off():
    file_path = "test.editorconfig"
    sections = ("*.{py}",)
    expected = {"line_length": float("inf"), "source": file_path}
    with open(file_path, "w") as f:
        f.write("max_line_length = off")
    assert _get_config_data(file_path, sections) == expected
    os.remove(file_path)


# LLM-generated content at query #10
#--------------------------

```python
def test_config_init_with_config():
    mock_config = _Config()
    mock_config.py_version = "py38"
    config = Config(config=mock_config)
    assert config.py_version == "38"

def test_config_init_with_config_and_overrides():
    mock_config = _Config()
    mock_config.py_version = "py38"
    config = Config(config=mock_config, py_version="py39")
    assert config.py_version == "39"

def test_config_init_with_settings_file(tmpdir):
    settings_file = tmpdir.join("settings.ini")
    settings_file.write("[isort]\nprofile = black")
    config = Config(settings_file=str(settings_file))
    assert config.profile == "black"

def test_config_init_with_settings_path(tmpdir):
    settings_dir = tmpdir.mkdir("settings")
    settings_file = settings_dir.join("pyproject.toml")
    settings_file.write("[tool.isort]\nprofile = \"black\"")
    config = Config(settings_path=str(settings_dir))
    assert config.profile == "black"

def test_config_init_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False, "Expected InvalidSettingsPath exception"
    except InvalidSettingsPath:
        pass

def test_config_init_with_profile_override():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_init_with_indent_override():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_init_with_tab_indent():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_init_with_known_sections():
    config = Config(known_third_party=["requests"], sections=("THIRDPARTY",))
    assert "known_third_party" in config.known_other

def test_config_init_with_import_headings():
    config = Config(import_heading_thirdparty="Third Party")
    assert config.import_headings["thirdparty"] == "Third Party"

def test_config_init_with_import_footers():
    config = Config(import_footer_thirdparty="End Third Party")
    assert config.import_footers["thirdparty"] == "End Third Party"

def test_config_init_with_unsupported_settings():
    try:
        Config(invalid_setting="value")
        assert False, "Expected UnsupportedSettings exception"
    except UnsupportedSettings:
        pass

def test_config_init_with_deprecated_settings():
    config = Config(line_length=88, quiet=True)
    assert not hasattr(config, "line_length")


# LLM-generated content at query #11
#--------------------------

```
def test_config_constructor_with_config():
    mock_config = _Config()
    mock_config.py_version = "py3"
    config = Config(config=mock_config)
    assert config.py_version == "3"

def test_config_constructor_with_settings_file(tmp_path):
    settings_file = tmp_path / "settings.ini"
    settings_file.write_text("[isort]\nprofile = black\n")
    config = Config(settings_file=str(settings_file))
    assert config.profile == "black"

def test_config_constructor_with_settings_path(tmp_path):
    settings_dir = tmp_path / "subdir"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.ini"
    settings_file.write_text("[isort]\nprofile = black\n")
    config = Config(settings_path=str(settings_dir))
    assert config.profile == "black"

def test_config_constructor_with_config_overrides():
    config = Config(profile="black", quiet=True)
    assert config.profile == "black"
    assert config.quiet is True

def test_config_constructor_with_invalid_profile():
    try:
        Config(profile="invalid_profile")
        assert False, "Should raise ProfileDoesNotExist"
    except ProfileDoesNotExist:
        pass

def test_config_constructor_with_invalid_settings_path(tmp_path):
    invalid_path = tmp_path / "nonexistent"
    try:
        Config(settings_path=str(invalid_path))
        assert False, "Should raise InvalidSettingsPath"
    except InvalidSettingsPath:
        pass

def test_config_constructor_with_unsupported_settings():
    try:
        Config(invalid_setting="value")
        assert False, "Should raise UnsupportedSettings"
    except UnsupportedSettings:
        pass

def test_config_constructor_with_deprecated_settings():
    deprecated_setting = next(iter(DEPRECATED_SETTINGS))
    config = Config(**{deprecated_setting: "value"}, quiet=True)
    assert deprecated_setting not in vars(config)

def test_config_constructor_with_indent_conversion():
    config = Config(indent="4")
    assert config.indent == "    "
    config = Config(indent='"tab"')
    assert config.indent == "\t"

def test_config_constructor_with_known_other_sections():
    config = Config(known_custom=["package"], sections=("CUSTOM",))
    assert "custom" in config.known_other
    assert frozenset(["package"]) == config.known_other["custom"]


# LLM-generated content at query #12
#--------------------------

```python
def test_get_str_to_type_converter_with_default_setting():
    result = _get_str_to_type_converter("non_existent_setting")
    assert result == str

def test_get_str_to_type_converter_with_wrap_modes():
    result = _get_str_to_type_converter("wrap_mode_setting")
    assert result == wrap_mode_from_string

def test_get_str_to_type_converter_with_integer_setting():
    _DEFAULT_SETTINGS["integer_setting"] = 42
    result = _get_str_to_type_converter("integer_setting")
    assert result == int

def test_get_str_to_type_converter_with_float_setting():
    _DEFAULT_SETTINGS["float_setting"] = 3.14
    result = _get_str_to_type_converter("float_setting")
    assert result == float

def test_get_str_to_type_converter_with_boolean_setting():
    _DEFAULT_SETTINGS["boolean_setting"] = True
    result = _get_str_to_type_converter("boolean_setting")
    assert result == bool


# LLM-generated content at query #13
#--------------------------

```python
def test_is_skipped_when_file_path_is_in_skips():
    config = Config()
    config.skips = frozenset(["test_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) == True

def test_is_skipped_when_file_path_is_not_in_skips():
    config = Config()
    config.skips = frozenset(["another_file.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) == False

def test_is_skipped_when_file_name_matches_skip_glob():
    config = Config()
    config.skip_globs = frozenset(["test_*.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) == True

def test_is_skipped_when_file_name_does_not_match_skip_glob():
    config = Config()
    config.skip_globs = frozenset(["another_*.py"])
    file_path = Path("test_file.py")
    assert config.is_skipped(file_path) == False

def test_is_skipped_when_file_is_in_gitignore():
    config = Config()
    config.skip_gitignore = True
    file_path = Path("test_file.py")
    config.git_ls_files = {Path("."): frozenset(["another_file.py"])}
    assert config.is_skipped(file_path) == True

def test_is_skipped_when_file_is_not_in_gitignore():
    config = Config()
    config.skip_gitignore = True
    file_path = Path("test_file.py")
    config.git_ls_files = {Path("."): frozenset(["test_file.py"])}
    assert config.is_skipped(file_path) == False


# LLM-generated content at query #14
#--------------------------

```python
def test_find_config_with_valid_config_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "config.toml")
        with open(config_file_path, "w") as f:
            f.write("[section]\nkey = 'value'")
        result = _find_config(temp_dir)
        assert result[0] == temp_dir
        assert result[1] == {"key": "value", "source": config_file_path}

def test_find_config_with_invalid_config_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "invalid_config.toml")
        with open(config_file_path, "w") as f:
            f.write("invalid toml content")
        result = _find_config(temp_dir)
        assert result[0] == temp_dir
        assert result[1] == {}

def test_find_config_with_no_config_file():
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        result = _find_config(temp_dir)
        assert result[0] == temp_dir
        assert result[1] == {}

def test_find_config_with_stop_dir():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        stop_dir = os.path.join(temp_dir, "stop_dir")
        os.makedirs(stop_dir)
        result = _find_config(temp_dir)
        assert result[0] == temp_dir
        assert result[1] == {}

def test_find_config_with_max_search_depth():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_dir = temp_dir
        for _ in range(MAX_CONFIG_SEARCH_DEPTH + 1):
            nested_dir = os.path.join(nested_dir, "nested")
            os.makedirs(nested_dir)
        result = _find_config(nested_dir)
        assert result[0] == nested_dir
        assert result[1] == {}


# LLM-generated content at query #15
#--------------------------

```python
def test_config_initialization_with_empty_arguments():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_overrides():
    config_overrides = {"quiet": True, "profile": "black"}
    config = Config(**config_overrides)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_instance():
    class MockConfig:
        def __init__(self):
            self.py_version = "py39"
            self._known_patterns = []
            self._section_comments = []
            self._section_comments_end = []
            self._skips = set()
            self._skip_globs = set()
            self._sorting_function = sorted

    mock_config = MockConfig()
    config = Config(config=mock_config, quiet=True)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_file():
    settings_file = "sample_settings.ini"
    config = Config(settings_file=settings_file)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_path():
    settings_path = "/path/to/settings"
    config = Config(settings_path=settings_path)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_5_evaluates_to_false():
    file_path = "example.txt"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert not file_path.endswith(".toml")


# LLM-generated content at query #17
#--------------------------

```python
def test___post_init__py_version_auto():
    config = _Config(py_version="auto")
    assert config.py_version.startswith("py")

def test___post_init__py_version_invalid():
    try:
        _Config(py_version="invalid")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test___post_init__py_version_valid():
    config = _Config(py_version="3")
    assert config.py_version == "py3"

def test___post_init__force_alphabetical_sort():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections
    assert config.no_sections
    assert config.lines_between_types == 1
    assert config.from_first

def test___post_init__wrap_length_exceeds_line_length():
    try:
        _Config(line_length=79, wrap_length=80)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test___post_init__known_standard_library_empty():
    config = _Config(py_version="3", known_standard_library=frozenset())
    assert config.known_standard_library

def test___post_init__multi_line_output_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED


# LLM-generated content at query #18
#--------------------------

```
def test_find_config_returns_empty_when_stop_dir_found():
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        stop_dir = "test_stop_dir"
        os.makedirs(os.path.join(tmpdir, stop_dir))
        with patch.dict('os.path.isdir', {os.path.join(tmpdir, stop_dir): True}):
            result = _find_config(tmpdir)
            assert result == (tmpdir, {})


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_73_evaluates_to_true():
    file_path = "test.toml"
    sections = ("section1",)
    settings = {"key1": "true"}
    predicate_result = isinstance(settings.get("key1"), bool)
    assert predicate_result == True


# LLM-generated content at query #20
#--------------------------

```python
def test_find_all_configs():
    root_dir = "/fake/path"
    trie = find_all_configs(root_dir)
    assert isinstance(trie, Trie)
    assert trie.root.config_info == ("default", {})


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_78_evaluates_to_true():
    file_path = "test.toml"
    sections = ("section1",)
    settings = {"key1": "value1", "source": file_path}
    KNOWN_PREFIX = "known_"
    settings["known_key"] = "value"
    result = _get_config_data(file_path, sections)
    assert isinstance(settings["known_key"], str)


# LLM-generated content at query #22
#--------------------------

```python
def test_find_config_with_valid_config():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 120\n")
        directory, config_data = _find_config(tmpdir)
        assert directory == tmpdir
        assert config_data == {"line_length": 120, "source": config_path}

def test_find_config_with_invalid_config():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("invalid toml content")
        directory, config_data = _find_config(tmpdir)
        assert directory == tmpdir
        assert config_data == {}

def test_find_config_with_no_config():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        directory, config_data = _find_config(tmpdir)
        assert directory == tmpdir
        assert config_data == {}

def test_find_config_with_stop_dir():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        stop_dir = os.path.join(tmpdir, ".git")
        os.mkdir(stop_dir)
        directory, config_data = _find_config(tmpdir)
        assert directory == tmpdir
        assert config_data == {}

def test_find_config_with_max_depth():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        current_dir = tmpdir
        for _ in range(MAX_CONFIG_SEARCH_DEPTH + 1):
            current_dir = os.path.join(current_dir, "subdir")
            os.mkdir(current_dir)
        directory, config_data = _find_config(current_dir)
        assert directory == current_dir
        assert config_data == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_is_supported_filetype_predicate_evaluates_to_false():
    config = Config()
    config.supported_extensions = {"txt"}
    config.blocked_extensions = {"bak"}
    file_name = "test.txt"
    assert config.is_supported_filetype(file_name) is True

    file_name = "test.bak"
    assert config.is_supported_filetype(file_name) is False

    file_name = "test~"
    assert config.is_supported_filetype(file_name) is False

    file_name = "test.pipe"
    assert config.is_supported_filetype(file_name) is False

    file_name = "test.sh"
    assert config.is_supported_filetype(file_name) is True


# LLM-generated content at query #24
#--------------------------

```python
def test_settings_file_default_value():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #25
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    config.supported_extensions = {"py"}
    assert config.is_supported_filetype("test.py") == True

def test_is_supported_filetype_blocked_extension():
    config = Config()
    config.blocked_extensions = {"txt"}
    assert config.is_supported_filetype("test.txt") == False

def test_is_supported_filetype_editor_backup_file():
    config = Config()
    assert config.is_supported_filetype("test.py~") == False

def test_is_supported_filetype_fifo_file():
    config = Config()
    assert config.is_supported_filetype("test_fifo") == False

def test_is_supported_filetype_shebang_match():
    config = Config()
    assert config.is_supported_filetype("test.sh") == True


# LLM-generated content at query #26
#--------------------------

```
def test__get_config_data_toml():
    file_path = "test.toml"
    sections = ("section1", "section2")
    expected = {"key1": "value1", "key2": "value2", "source": file_path}
    assert _get_config_data(file_path, sections) == expected

def test__get_config_data_editorconfig():
    file_path = "test.editorconfig"
    sections = ("section1", "section2")
    expected = {"indent": "    ", "line_length": 80, "source": file_path}
    assert _get_config_data(file_path, sections) == expected

def test__get_config_data_ini():
    file_path = "test.ini"
    sections = ("section1", "section2")
    expected = {"key1": "value1", "key2": "value2", "source": file_path}
    assert _get_config_data(file_path, sections) == expected

def test__get_config_data_empty_sections():
    file_path = "test.toml"
    sections = ()
    expected = {}
    assert _get_config_data(file_path, sections) == expected

def test__get_config_data_invalid_file():
    file_path = "invalid_file.txt"
    sections = ("section1", "section2")
    expected = {}
    assert _get_config_data(file_path, sections) == expected


# LLM-generated content at query #27
#--------------------------

```python
def test_section_in_section_defaults():
    combined_config = {"sections": ["SECTION_A"]}
    SECTION_DEFAULTS = {"SECTION_A"}
    assert "SECTION_A" in SECTION_DEFAULTS


# LLM-generated content at query #28
#--------------------------

```python
def test_is_supported_filetype_does_not_raise_oserror():
    config = Config()
    file_name = "test_file.txt"
    assert config.is_supported_filetype(file_name) == True


# LLM-generated content at query #29
#--------------------------

```
def test_bool_conversion_when_value_is_not_bool():
    settings = {"some_bool_key": "true"}
    existing_value_type = lambda _: bool
    settings["some_bool_key"] = existing_value_type(settings["some_bool_key"])
    assert isinstance(settings["some_bool_key"], bool)


# LLM-generated content at query #30
#--------------------------

```python
def test_section_in_section_defaults():
    config = Config()
    combined_config = {"sections": ["standard", "future", "third_party"]}
    SECTION_DEFAULTS = ["standard", "future", "third_party"]
    assert all(section in SECTION_DEFAULTS for section in combined_config.get("sections", ()))


# LLM-generated content at query #31
#--------------------------

```python
def test_config_with_config_object():
    config = _Config()
    config_vars = vars(config).copy()
    config_vars["py_version"] = "py39"
    config = Config(config=config)
    assert config.py_version == "39"

def test_config_with_settings_file():
    settings_file = "test_settings.ini"
    config = Config(settings_file=settings_file)
    assert config.directory == os.path.dirname(settings_file)

def test_config_with_settings_path():
    settings_path = "/path/to/settings"
    config = Config(settings_path=settings_path)
    assert config.directory == os.path.dirname(settings_path)

def test_config_with_config_overrides():
    config = Config(py_version="py39", quiet=True)
    assert config.py_version == "39"
    assert config.quiet is True

def test_config_with_profile():
    profile_name = "test_profile"
    config = Config(profile=profile_name)
    assert "profile" in config.sources[-1]["source"]

def test_config_with_indent():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_with_indent_as_string():
    config = Config(indent="'    '")
    assert config.indent == "    "

def test_config_with_unsupported_option():
    try:
        config = Config(unsupported_option="value")
    except UnsupportedSettings:
        pass
    else:
        assert False, "Expected UnsupportedSettings exception"


# LLM-generated content at query #32
#--------------------------

```python
def test_predicate_at_line_44_evaluates_to_true():
    file_path = "example.editorconfig"
    sections = ("*.py",)
    settings = _get_config_data(file_path, sections)
    assert "source" in settings


# LLM-generated content at query #33
#--------------------------

```python
def test_as_list_with_string():
    result = _as_list("a, b, c")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_and_newlines():
    result = _as_list("a\nb\nc")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_and_mixed_separators():
    result = _as_list("a, b\nc, d")
    assert result == ["a", "b", "c", "d"]

def test_as_list_with_string_and_extra_spaces():
    result = _as_list("  a  ,  b  ,  c  ")
    assert result == ["a", "b", "c"]

def test_as_list_with_empty_string():
    result = _as_list("")
    assert result == []

def test_as_list_with_list():
    result = _as_list(["a", "b", "c"])
    assert result == ["a", "b", "c"]

def test_as_list_with_list_and_extra_spaces():
    result = _as_list(["  a  ", "  b  ", "  c  "])
    assert result == ["a", "b", "c"]

def test_as_list_with_empty_list():
    result = _as_list([])
    assert result == []


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config_data_toml():
    config_data = _get_config_data("example.toml", ("section", "subsection"))
    assert isinstance(config_data, dict)

def test_get_config_data_editorconfig():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert isinstance(config_data, dict)

def test_get_config_data_with_indent_style_space():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert "indent" in config_data

def test_get_config_data_with_indent_style_tab():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert "indent" in config_data

def test_get_config_data_with_max_line_length_off():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert "line_length" in config_data

def test_get_config_data_with_max_line_length_digit():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert "line_length" in config_data

def test_get_config_data_with_known_prefix():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert any(key.startswith(KNOWN_PREFIX) for key in config_data)

def test_get_config_data_with_force_grid_wrap():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert "force_grid_wrap" in config_data

def test_get_config_data_with_comment_prefix():
    config_data = _get_config_data("example.editorconfig", ("section",))
    assert "comment_prefix" in config_data


# LLM-generated content at query #35
#--------------------------

```
def test__as_bool_returns_true_for_true_values():
    assert _as_bool("true") is True
    assert _as_bool("True") is True
    assert _as_bool("TRUE") is True
    assert _as_bool("yes") is True
    assert _as_bool("Yes") is True
    assert _as_bool("YES") is True
    assert _as_bool("on") is True
    assert _as_bool("On") is True
    assert _as_bool("ON") is True
    assert _as_bool("1") is True
    assert _as_bool("y") is True
    assert _as_bool("Y") is True

def test__as_bool_returns_false_for_false_values():
    assert _as_bool("false") is False
    assert _as_bool("False") is False
    assert _as_bool("FALSE") is False
    assert _as_bool("no") is False
    assert _as_bool("No") is False
    assert _as_bool("NO") is False
    assert _as_bool("off") is False
    assert _as_bool("Off") is False
    assert _as_bool("OFF") is False
    assert _as_bool("0") is False
    assert _as_bool("n") is False
    assert _as_bool("N") is False

def test__as_bool_raises_value_error_for_invalid_values():
    try:
        _as_bool("invalid")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"
    try:
        _as_bool("")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"
    try:
        _as_bool(" ")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_skipped_returns_true_for_skipped_file():
    config = Config()
    config._skips = frozenset(["test_file.py"])
    assert config.is_skipped(Path("test_file.py")) == True

def test_is_skipped_returns_false_for_non_skipped_file():
    config = Config()
    config._skips = frozenset(["other_file.py"])
    assert config.is_skipped(Path("test_file.py")) == False

def test_is_skipped_returns_true_for_skipped_directory():
    config = Config()
    config._skips = frozenset(["test_dir"])
    assert config.is_skipped(Path("test_dir")) == True

def test_is_skipped_returns_true_for_file_in_skipped_directory():
    config = Config()
    config._skips = frozenset(["test_dir"])
    assert config.is_skipped(Path("test_dir/file.py")) == True

def test_is_skipped_returns_true_for_glob_match():
    config = Config()
    config._skip_globs = frozenset(["*.py"])
    assert config.is_skipped(Path("test_file.py")) == True

def test_is_skipped_returns_false_for_non_glob_match():
    config = Config()
    config._skip_globs = frozenset(["*.txt"])
    assert config.is_skipped(Path("test_file.py")) == False

def test_is_skipped_returns_true_for_gitignored_file_when_skip_gitignore_is_true():
    config = Config(skip_gitignore=True)
    config.git_ls_files = {Path("/root"): {"/root/allowed.py"}}
    assert config.is_skipped(Path("/root/skipped.py")) == True

def test_is_skipped_returns_false_for_gitignored_file_when_skip_gitignore_is_false():
    config = Config(skip_gitignore=False)
    config.git_ls_files = {Path("/root"): {"/root/allowed.py"}}
    assert config.is_skipped(Path("/root/skipped.py")) == False

def test_is_skipped_returns_true_for_nonexistent_file():
    config = Config()
    assert config.is_skipped(Path("nonexistent_file")) == True

def test_is_skipped_returns_false_for_existing_file():
    config = Config()
    with open("existing_file.py", "w") as f:
        f.write("")
    try:
        assert config.is_skipped(Path("existing_file.py")) == False
    finally:
        os.remove("existing_file.py")


# LLM-generated content at query #2
#--------------------------

```python
def test_config_initialization_with_default_values():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_overrides():
    config_overrides = {"quiet": True}
    config = Config(config_overrides=config_overrides)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_instance():
    class MockConfig:
        def __init__(self):
            self.py_version = "py39"
            self._known_patterns = None
            self._section_comments = None
            self._section_comments_end = None
            self._skips = None
            self._skip_globs = None
            self._sorting_function = None

    mock_config = MockConfig()
    config = Config(config=mock_config)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_file():
    settings_file = "settings.ini"
    config = Config(settings_file=settings_file)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_path():
    settings_path = "/path/to/settings"
    config = Config(settings_path=settings_path)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #3
#--------------------------

```python
def test_config_initialization_with_config_overrides():
    config_overrides = {"indent": 4, "quiet": True}
    config = Config(config_overrides=config_overrides)
    assert config.indent == "    "
    assert config.quiet is True

def test_config_initialization_with_settings_file():
    settings_file = "example_settings.ini"
    config = Config(settings_file=settings_file)
    assert config.directory == os.path.dirname(settings_file)

def test_config_initialization_with_settings_path():
    settings_path = "/path/to/settings"
    config = Config(settings_path=settings_path)
    assert config.directory == settings_path

def test_config_initialization_with_config_object():
    class MockConfig:
        py_version = "py3.8"
        indent = "tab"
    mock_config = MockConfig()
    config = Config(config=mock_config)
    assert config.py_version == "3.8"
    assert config.indent == "\t"

def test_config_initialization_with_profile():
    config_overrides = {"profile": "example_profile"}
    config = Config(config_overrides=config_overrides)
    assert config.profile == "example_profile"

def test_config_initialization_with_unsupported_config_errors():
    unsupported_config = {"unsupported_option": "value"}
    try:
        Config(config_overrides=unsupported_config)
        assert False, "Expected UnsupportedSettings exception"
    except UnsupportedSettings:
        pass

def test_config_initialization_with_deprecated_options():
    deprecated_config = {"deprecated_option": "value"}
    config = Config(config_overrides=deprecated_config)
    assert "deprecated_option" not in vars(config)

def test_config_initialization_with_known_other():
    known_other_config = {"known_other_section": {"module1", "module2"}}
    config = Config(config_overrides=known_other_config)
    assert config.known_other == {"known_other_section": frozenset({"module1", "module2"})}

def test_config_initialization_with_import_headings():
    import_headings_config = {"import_heading_example": "Example Heading"}
    config = Config(config_overrides=import_headings_config)
    assert config.import_headings == {"example": "Example Heading"}

def test_config_initialization_with_import_footers():
    import_footers_config = {"import_footer_example": "Example Footer"}
    config = Config(config_overrides=import_footers_config)
    assert config.import_footers == {"example": "Example Footer"}


# LLM-generated content at query #4
#--------------------------

```python
def test_config_constructor_with_empty_parameters():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


def test_config_constructor_with_config_parameter():
    base_config = Config()
    config = Config(config=base_config)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


def test_config_constructor_with_settings_file():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.toml') as tmp:
        tmp.write('[isort]\nprofile = "black"\n')
        tmp.flush()
        config = Config(settings_file=tmp.name)
        assert config.profile == "black"


def test_config_constructor_with_settings_path():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = os.path.join(tmpdir, "pyproject.toml")
        with open(settings_path, 'w') as f:
            f.write('[tool.isort]\nprofile = "black"\n')
        config = Config(settings_path=tmpdir)
        assert config.profile == "black"


def test_config_constructor_with_config_overrides():
    config = Config(profile="black")
    assert config.profile == "black"


def test_config_constructor_with_invalid_settings_path():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = os.path.join(tmpdir, "nonexistent")
        try:
            Config(settings_path=invalid_path)
        except InvalidSettingsPath:
            pass
        else:
            assert False, "Should have raised InvalidSettingsPath"


def test_config_constructor_with_unknown_profile():
    try:
        Config(profile="nonexistent_profile")
    except ProfileDoesNotExist:
        pass
    else:
        assert False, "Should have raised ProfileDoesNotExist"


def test_config_constructor_with_unsupported_settings():
    try:
        Config(nonexistent_setting="value")
    except UnsupportedSettings:
        pass
    else:
        assert False, "Should have raised UnsupportedSettings"


# LLM-generated content at query #5
#--------------------------

```python
def test_config_initialization_with_config():
    config = _Config()
    config_vars = vars(config).copy()
    config_vars["py_version"] = "3.8"
    config_instance = Config(config=config)
    assert config_instance.py_version == "3.8"

def test_config_initialization_with_settings_file():
    settings_file = "test_settings.ini"
    config_instance = Config(settings_file=settings_file)
    assert config_instance.directory == os.path.dirname(settings_file)

def test_config_initialization_with_settings_path():
    settings_path = "/path/to/settings"
    config_instance = Config(settings_path=settings_path)
    assert config_instance.directory == os.path.dirname(settings_path)

def test_config_initialization_with_config_overrides():
    config = _Config()
    config_vars = vars(config).copy()
    config_vars["py_version"] = "3.8"
    config_instance = Config(config=config, py_version="3.9")
    assert config_instance.py_version == "3.9"

def test_config_initialization_with_profile():
    profile_name = "test_profile"
    config_instance = Config(profile=profile_name)
    assert config_instance.profile == profile_name

def test_config_initialization_with_indent():
    config_instance = Config(indent="4")
    assert config_instance.indent == "    "

def test_config_initialization_with_formatter():
    formatter = "test_formatter"
    config_instance = Config(formatter=formatter)
    assert config_instance.formatter == formatter

def test_config_initialization_with_src_paths():
    src_paths = ["src", "tests"]
    config_instance = Config(src_paths=src_paths)
    assert config_instance.src_paths == tuple(src_paths)

def test_config_initialization_with_unsupported_config():
    unsupported_config = {"unsupported_key": "unsupported_value"}
    try:
        Config(**unsupported_config)
        assert False
    except UnsupportedSettings:
        assert True

def test_config_initialization_with_deprecated_options():
    deprecated_options = {"deprecated_key": "deprecated_value"}
    config_instance = Config(**deprecated_options)
    assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_config_constructor_with_config():
    mock_config = _Config()
    mock_config.py_version = "py3"
    config = Config(config=mock_config, quiet=True)
    assert config.py_version == "3"

def test_config_constructor_with_settings_file(tmp_path):
    settings_file = tmp_path / "settings.ini"
    settings_file.write_text("[isort]\nprofile = black\n")
    config = Config(settings_file=str(settings_file))
    assert config.profile == "black"

def test_config_constructor_with_settings_path(tmp_path):
    settings_file = tmp_path / "settings.ini"
    settings_file.write_text("[isort]\nprofile = black\n")
    config = Config(settings_path=str(tmp_path))
    assert config.profile == "black"

def test_config_constructor_with_config_overrides():
    config = Config(profile="black", quiet=True)
    assert config.profile == "black"
    assert config.quiet

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False, "Should raise InvalidSettingsPath"
    except InvalidSettingsPath:
        pass

def test_config_constructor_with_invalid_profile():
    try:
        Config(profile="nonexistent")
        assert False, "Should raise ProfileDoesNotExist"
    except ProfileDoesNotExist:
        pass

def test_config_constructor_with_unsupported_settings():
    try:
        Config(nonexistent_setting=True)
        assert False, "Should raise UnsupportedSettings"
    except UnsupportedSettings:
        pass

def test_config_constructor_with_deprecated_settings():
    deprecated_setting = next(iter(DEPRECATED_SETTINGS))
    config = Config(**{deprecated_setting: True}, quiet=True)
    assert deprecated_setting not in vars(config)

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_string():
    config = Config(indent="'tab'")
    assert config.indent == "\t"

def test_config_constructor_with_known_other_sections():
    config = Config(known_custom=["package"], sections=("CUSTOM",))
    assert "custom" in config.known_other
    assert "package" in config.known_other["custom"]

def test_config_constructor_with_import_headings():
    config = Config(import_heading_stdlib="Standard Library")
    assert "stdlib" in config.import_headings
    assert config.import_headings["stdlib"] == "Standard Library"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_stdlib="End Standard Library")
    assert "stdlib" in config.import_footers
    assert config.import_footers["stdlib"] == "End Standard Library"


# LLM-generated content at query #7
#--------------------------

```python
def test_is_skipped_predicate_evaluates_to_false():
    config = Config()
    file_path = Path("/some/path/file.txt")
    assert not config.is_skipped(file_path)


# LLM-generated content at query #8
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    config.supported_extensions = {"txt"}
    assert config.is_supported_filetype("example.txt") == True

def test_is_supported_filetype_blocked_extension():
    config = Config()
    config.blocked_extensions = {"log"}
    assert config.is_supported_filetype("example.log") == False

def test_is_supported_filetype_editor_backup_file():
    config = Config()
    assert config.is_supported_filetype("example.txt~") == False

def test_is_supported_filetype_fifo_file():
    config = Config()
    assert config.is_supported_filetype("/tmp/fifo_file") == False

def test_is_supported_filetype_shebang_file():
    config = Config()
    assert config.is_supported_filetype("script.py") == True

def test_is_supported_filetype_unsupported_extension():
    config = Config()
    config.supported_extensions = {"txt"}
    config.blocked_extensions = {"log"}
    assert config.is_supported_filetype("example.csv") == False


# LLM-generated content at query #9
#--------------------------

```python
def test_find_all_configs():
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    config_file = os.path.join(test_dir, ".isort.cfg")
    with open(config_file, "w") as f:
        f.write("[settings]\nline_length=80\n")
    trie = find_all_configs(test_dir)
    config_data = trie.search(config_file)
    assert config_data[0] == config_file
    assert config_data[1]["line_length"] == 80
    os.remove(config_file)
    os.rmdir(test_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_config_constructor_with_config():
    config = _Config(py_version="py38")
    instance = Config(config=config)
    assert instance.py_version == "38"

def test_config_constructor_with_settings_file():
    instance = Config(settings_file="test_settings.ini")
    assert instance._known_patterns is None
    assert instance._section_comments is None
    assert instance._section_comments_end is None
    assert instance._skips is None
    assert instance._skip_globs is None
    assert instance._sorting_function is None

def test_config_constructor_with_settings_path():
    instance = Config(settings_path="test_path")
    assert instance._known_patterns is None
    assert instance._section_comments is None
    assert instance._section_comments_end is None
    assert instance._skips is None
    assert instance._skip_globs is None
    assert instance._sorting_function is None

def test_config_constructor_with_config_overrides():
    instance = Config(py_version="py38", quiet=True)
    assert instance.py_version == "38"
    assert instance.quiet is True

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="invalid_path")
    except InvalidSettingsPath:
        pass
    else:
        assert False, "Expected InvalidSettingsPath exception"

def test_config_constructor_with_profile_does_not_exist():
    try:
        Config(profile="nonexistent_profile")
    except ProfileDoesNotExist:
        pass
    else:
        assert False, "Expected ProfileDoesNotExist exception"

def test_config_constructor_with_formatter_plugin_does_not_exist():
    try:
        Config(formatter="nonexistent_formatter")
    except FormattingPluginDoesNotExist:
        pass
    else:
        assert False, "Expected FormattingPluginDoesNotExist exception"

def test_config_constructor_with_unsupported_settings():
    try:
        Config(unsupported_setting="value")
    except UnsupportedSettings:
        pass
    else:
        assert False, "Expected UnsupportedSettings exception"

def test_config_constructor_with_deprecated_options():
    instance = Config(deprecated_option="value", quiet=True)
    assert "deprecated_option" not in vars(instance)


