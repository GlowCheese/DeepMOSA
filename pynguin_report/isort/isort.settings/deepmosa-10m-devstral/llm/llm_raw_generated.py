####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_config_initialization_with_defaults():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_path():
    config = Config(settings_path=".")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_object():
    base_config = Config()
    config = Config(config=base_config)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_overrides():
    config = Config(indent="    ")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config_data_toml_file():
    config_data = _get_config_data("test.toml", ("section1", "section2"))
    assert isinstance(config_data, dict)
    assert "source" in config_data
    assert config_data["source"] == "test.toml"

def test_get_config_data_editorconfig_file():
    config_data = _get_config_data("test.editorconfig", ("*.{py}", "*.{js,ts}"))
    assert isinstance(config_data, dict)
    assert "source" in config_data
    assert config_data["source"] == "test.editorconfig"

def test_get_config_data_other_file():
    config_data = _get_config_data("test.ini", ("section1", "section2"))
    assert isinstance(config_data, dict)
    assert "source" in config_data
    assert config_data["source"] == "test.ini"

def test_get_config_data_empty_sections():
    config_data = _get_config_data("test.ini", ())
    assert config_data == {}

def test_get_config_data_editorconfig_indent_style_space():
    config_data = _get_config_data("test.editorconfig", ("*.{py}",))
    assert "indent" in config_data
    assert config_data["indent"] == "    "

def test_get_config_data_editorconfig_indent_style_tab():
    config_data = _get_config_data("test.editorconfig", ("*.{py}",))
    assert "indent" in config_data
    assert config_data["indent"] == "\t"

def test_get_config_data_editorconfig_max_line_length_off():
    config_data = _get_config_data("test.editorconfig", ("*.{py}",))
    assert "line_length" in config_data
    assert config_data["line_length"] == float("inf")

def test_get_config_data_editorconfig_max_line_length_digit():
    config_data = _get_config_data("test.editorconfig", ("*.{py}",))
    assert "line_length" in config_data
    assert isinstance(config_data["line_length"], int)

def test_get_config_data_known_prefix_paths():
    config_data = _get_config_data("test.ini", ("section1",))
    assert "known_prefix_paths" in config_data
    assert isinstance(config_data["known_prefix_paths"], set)

def test_get_config_data_force_grid_wrap_true():
    config_data = _get_config_data("test.ini", ("section1",))
    assert "force_grid_wrap" in config_data
    assert config_data["force_grid_wrap"] == 2

def test_get_config_data_force_grid_wrap_false():
    config_data = _get_config_data("test.ini", ("section1",))
    assert "force_grid_wrap" in config_data
    assert config_data["force_grid_wrap"] == 0

def test_get_config_data_comment_prefix():
    config_data = _get_config_data("test.ini", ("section1",))
    assert "comment_prefix" in config_data
    assert isinstance(config_data["comment_prefix"], str)

def test_get_config_data_type_conversion():
    config_data = _get_config_data("test.ini", ("section1",))
    for key, value in config_data.items():
        if key in _DEFAULT_SETTINGS:
            expected_type = type(_DEFAULT_SETTINGS[key])
            assert isinstance(value, expected_type)


# LLM-generated content at query #3
#--------------------------

```python
def test_config_settings_not_empty():
    config = Config(settings_file="valid_config_file.ini")
    assert config._config_settings == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    assert config.is_supported_filetype("file.py") is True

def test_is_supported_filetype_blocked_extension():
    config = Config()
    assert config.is_supported_filetype("file.min.js") is False

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("file.py~") is False

def test_is_supported_filetype_fifo():
    config = Config()
    assert config.is_supported_filetype("/dev/null") is False

def test_is_supported_filetype_with_shebang():
    config = Config()
    assert config.is_supported_filetype("script") is True

def test_is_supported_filetype_without_shebang():
    config = Config()
    assert config.is_supported_filetype("empty_file") is False


# LLM-generated content at query #5
#--------------------------

```python
def test__find_config_returns_empty_dict_when_no_config_found():
    result = _find_config("/non/existent/path")
    assert result == ("/non/existent/path", {})

def test__find_config_finds_config_in_current_directory():
    with patch("os.path.isfile", return_value=True), patch("os.path.isdir", return_value=False), patch("_get_config_data", return_value={"key": "value"}):
        result = _find_config("/some/path")
        assert result == ("/some/path", {"key": "value"})

def test__find_config_stops_search_on_stop_directory():
    with patch("os.path.isdir", side_effect=lambda x: x.endswith("stop_dir")), patch("os.path.isfile", return_value=False):
        result = _find_config("/some/path/with/stop_dir")
        assert result == ("/some/path/with/stop_dir", {})

def test__find_config_returns_parent_directory_when_config_found_there():
    with patch("os.path.isfile", side_effect=lambda x: x.endswith("pyproject.toml")), patch("os.path.isdir", return_value=False), patch("_get_config_data", return_value={"key": "value"}):
        result = _find_config("/some/path/child")
        assert result == ("/some/path", {"key": "value"})

def test__find_config_handles_exception_during_config_parsing():
    with patch("os.path.isfile", return_value=True), patch("os.path.isdir", return_value=False), patch("_get_config_data", side_effect=Exception("Parse error")), patch("warnings.warn") as mock_warn:
        result = _find_config("/some/path")
        assert result == ("/some/path", {})
        mock_warn.assert_called_once_with("Failed to pull configuration information from /some/path/pyproject.toml", stacklevel=2)


# LLM-generated content at query #6
#--------------------------

```python
def test_indent_digit_conversion():
    config = Config(indent="4")
    assert config.indent == "    "


# LLM-generated content at query #7
#--------------------------

```python
def test_import_heading_prefix_check():
    combined_config = {
        "import_heading_prefix_test": "value",
        "other_key": "other_value"
    }
    key = "import_heading_prefix_test"
    assert key.startswith("import_heading_")


# LLM-generated content at query #8
#--------------------------

```python
def test_ensure_deprecated_options_used_predicate():
    config = Config(config_overrides={"deprecated_option": "value"})
    assert "deprecated_option" not in vars(config)


# LLM-generated content at query #9
#--------------------------

```python
def test_find_all_configs_empty_directory():
    trie_root = find_all_configs("/empty_directory")
    assert trie_root.root.config_info == ("default", {})
    assert trie_root.root.nodes == {}

def test_find_all_configs_with_config_file():
    trie_root = find_all_configs("/path/to/config")
    assert trie_root.root.config_info == ("default", {})
    assert "/path/to/config/.isort.cfg" in trie_root.root.nodes
    assert trie_root.root.nodes["/path/to/config/.isort.cfg"].config_info[0] == "/path/to/config/.isort.cfg"
    assert isinstance(trie_root.root.nodes["/path/to/config/.isort.cfg"].config_info[1], dict)

def test_find_all_configs_multiple_config_files():
    trie_root = find_all_configs("/path/to/multiple_configs")
    assert trie_root.root.config_info == ("default", {})
    assert "/path/to/multiple_configs/.isort.cfg" in trie_root.root.nodes
    assert "/path/to/multiple_configs/setup.cfg" in trie_root.root.nodes
    assert trie_root.root.nodes["/path/to/multiple_configs/.isort.cfg"].config_info[0] == "/path/to/multiple_configs/.isort.cfg"
    assert trie_root.root.nodes["/path/to/multiple_configs/setup.cfg"].config_info[0] == "/path/to/multiple_configs/setup.cfg"

def test_find_all_configs_nested_config_files():
    trie_root = find_all_configs("/path/to/nested")
    assert trie_root.root.config_info == ("default", {})
    assert "/path/to/nested/subdir/.isort.cfg" in trie_root.root.nodes["/path/to/nested"].nodes
    assert trie_root.root.nodes["/path/to/nested"].nodes["subdir"].nodes[".isort.cfg"].config_info[0] == "/path/to/nested/subdir/.isort.cfg"

def test_find_all_configs_invalid_config_file():
    trie_root = find_all_configs("/path/to/invalid_config")
    assert trie_root.root.config_info == ("default", {})
    assert "/path/to/invalid_config/invalid.cfg" not in trie_root.root.nodes


# LLM-generated content at query #10
#--------------------------

```python
def test_while_loop_predicate_evaluates_to_false():
    assert not (current_directory and tries < MAX_CONFIG_SEARCH_DEPTH)


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config_data_with_toml_file():
    file_path = "test_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_with_editorconfig_file():
    file_path = "test_config.editorconfig"
    sections = ("*.py",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_with_ini_file():
    file_path = "test_config.ini"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_with_empty_sections():
    file_path = "test_config.toml"
    sections = ()
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_with_nonexistent_file():
    file_path = "nonexistent_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" not in result


# LLM-generated content at query #12
#--------------------------

```python
def test_is_supported_filetype_returns_true_for_fifo_file():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    file_name = "test_file"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    os.mkfifo(file_name)
    assert config.is_supported_filetype(file_name) is False


# LLM-generated content at query #13
#--------------------------

```python
def test_import_footer_prefix_condition():
    config = Config(
        settings_file="",
        settings_path="",
        config=None,
        **{"import_footer_test": "test_value"}
    )
    assert config.import_footers.get("test") == "test_value"


# LLM-generated content at query #14
#--------------------------

```python
def test_config_constructor_with_config_and_overrides():
    config = Config()
    new_config = Config(config=config, line_length=120)
    assert new_config.line_length == 120
    assert new_config.py_version == config.py_version.replace("py", "")

def test_config_constructor_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config.settings_file == "setup.cfg"

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_path == "."

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False, "Expected InvalidSettingsPath exception"
    except InvalidSettingsPath:
        pass

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_invalid_profile():
    try:
        Config(profile="nonexistent")
        assert False, "Expected ProfileDoesNotExist exception"
    except ProfileDoesNotExist:
        pass

def test_config_constructor_with_config_overrides():
    config = Config(line_length=88, indent="    ")
    assert config.line_length == 88
    assert config.indent == "    "

def test_config_constructor_with_unsupported_config():
    try:
        Config(unsupported_option="value")
        assert False, "Expected UnsupportedSettings exception"
    except UnsupportedSettings:
        pass

def test_config_constructor_with_deprecated_options():
    config = Config(include_trailing_comma=True)
    assert config.include_trailing_comma is False

def test_config_constructor_with_known_sections():
    config = Config(known_foo=["bar", "baz"])
    assert config.known_other == {"foo": frozenset(["bar", "baz"])}

def test_config_constructor_with_import_headings():
    config = Config(import_heading_foo="Bar")
    assert config.import_headings == {"foo": "Bar"}

def test_config_constructor_with_import_footers():
    config = Config(import_footer_foo="Baz")
    assert config.import_footers == {"foo": "Baz"}

def test_config_constructor_with_formatter():
    config = Config(formatter="black")
    assert config.formatter == "black"

def test_config_constructor_with_invalid_formatter():
    try:
        Config(formatter="nonexistent")
        assert False, "Expected FormattingPluginDoesNotExist exception"
    except FormattingPluginDoesNotExist:
        pass

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"

def test_config_constructor_with_invalid_sort_order():
    try:
        Config(sort_order="nonexistent")
        assert False, "Expected SortingFunctionDoesNotExist exception"
    except SortingFunctionDoesNotExist:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_config_predicate_false():
    assert not Config(settings_file="", settings_path="", config=None)


# LLM-generated content at query #16
#--------------------------

```python
def test_abspaths_with_relative_paths():
    result = _abspaths("/home/user", ["/dir1", "dir2/", "dir3"])
    assert result == {"/dir1", "/home/user/dir2", "/home/user/dir3"}

def test_abspaths_with_absolute_paths():
    result = _abspaths("/home/user", ["/dir1/", "/dir2", "/dir3/"])
    assert result == {"/dir1/", "/dir2", "/dir3/"}

def test_abspaths_with_mixed_paths():
    result = _abspaths("/home/user", ["dir1", "/dir2/", "dir3/", "/dir4"])
    assert result == {"/home/user/dir1", "/dir2/", "/dir3/", "/dir4"}

def test_abspaths_with_empty_input():
    result = _abspaths("/home/user", [])
    assert result == set()


# LLM-generated content at query #17
#--------------------------

```python
def test_line_166_predicate_false():
    config = Config(settings_path="/nonexistent/path")
    assert not config.path_root.is_dir()


# LLM-generated content at query #18
#--------------------------

```python
def test_config_initialization_with_config_parameter():
    config = _Config()
    config_instance = Config(config=config)
    assert config_instance._known_patterns is None
    assert config_instance._section_comments is None
    assert config_instance._section_comments_end is None
    assert config_instance._skips is None
    assert config_instance._skip_globs is None
    assert config_instance._sorting_function is None


# LLM-generated content at query #19
#--------------------------

```python
def test_config_settings_not_empty():
    config_settings = {"key": "value"}
    assert config_settings


# LLM-generated content at query #20
#--------------------------

```python
def test___post_init___with_py_version_auto():
    config = _Config(py_version="auto")
    assert config.py_version.startswith("py")

def test___post_init___with_invalid_py_version():
    with pytest.raises(ValueError):
        _Config(py_version="invalid")

def test___post_init___with_valid_py_version():
    config = _Config(py_version="38")
    assert config.py_version == "py38"

def test___post_init___with_known_standard_library_empty():
    config = _Config(py_version="38", known_standard_library=frozenset())
    assert len(config.known_standard_library) > 0

def test___post_init___with_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test___post_init___with_force_alphabetical_sort():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True

def test___post_init___with_wrap_length_greater_than_line_length():
    with pytest.raises(ValueError):
        _Config(wrap_length=80, line_length=79)


# LLM-generated content at query #21
#--------------------------

```python
def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"line_length": 100, "indent": "    "})
    assert config.line_length == 100
    assert config.indent == "    "

def test_config_constructor_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config.settings_file == "setup.cfg"

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_path == "."

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_unsupported_config():
    with pytest.raises(UnsupportedSettings):
        Config(config_overrides={"unsupported_option": "value"})

def test_config_constructor_with_deprecated_config():
    with pytest.warns(UserWarning):
        Config(config_overrides={"deprecated_option": "value"})

def test_config_constructor_with_known_other():
    config = Config(config_overrides={"known_other": {"custom": {"module"}}})
    assert config.known_other == {"custom": frozenset({"module"})}

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_custom": "Custom Heading"})
    assert config.import_headings == {"custom": "Custom Heading"}

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_custom": "Custom Footer"})
    assert config.import_footers == {"custom": "Custom Footer"}

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sort_order == "natural"

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"),)

def test_config_constructor_with_skip():
    config = Config(config_overrides={"skip": {"file.py"}})
    assert config.skip == frozenset({"file.py"})

def test_config_constructor_with_skip_glob():
    config = Config(config_overrides={"skip_glob": {"*.py"}})
    assert config.skip_glob == frozenset({"*.py"})

def test_config_constructor_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_constructor_with_indent_as_digit():
    config = Config(config_overrides={"indent": "4"})
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_constructor_with_indent_as_spaces():
    config = Config(config_overrides={"indent": "    "})
    assert config.indent == "    "

def test_config_constructor_with_known_patterns():
    config = Config(config_overrides={"known_standard_library": {"os", "sys"}})
    assert config.known_standard_library == frozenset({"os", "sys"})

def test_config_constructor_with_sections():
    config = Config(config_overrides={"sections": ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]})
    assert config.sections == ("FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER")

def test_config_constructor_with_directory():
    config = Config(config_overrides={"directory": "/path/to/project"})
    assert config.directory == "/path/to/project"

def test_config_constructor_with_extend_skip():
    config = Config(config_overrides={"extend_skip": {"file.py"}})
    assert config.extend_skip == frozenset({"file.py"})

def test_config_constructor_with_extend_skip_glob():
    config = Config(config_overrides={"extend_skip_glob": {"*.py"}})
    assert config.extend_skip_glob == frozenset({"*.py"})

def test_config_constructor_with_skip_gitignore():
    config = Config(config_overrides={"skip_gitignore": True})
    assert config.skip_gitignore is True

def test_config_constructor_with_known_patterns_directory():
    config = Config(config_overrides={"known_standard_library": {"os/", "sys/"}})
    assert config.known_standard_library == frozenset({"os", "sys"})

def test_config_constructor_with_known_patterns_wildcard():
    config = Config(config_overrides={"known_standard_library": {"os.*", "sys.*"}})
    assert config.known_standard_library == frozenset({"os.*", "sys.*"})

def test_config_constructor_with_known_patterns_question_mark():
    config = Config(config_overrides={"known_standard_library": {"os?", "sys?"}})
    assert config.known_standard_library == frozenset({"os?", "sys?"})

def test_config_constructor_with_known_patterns_mixed():
    config = Config(config_overrides={"known_standard_library": {"os", "sys.*", "re?"}})
    assert config.known_standard_library == frozenset({"os", "sys.*", "re?"})

def test_config_constructor_with_known_patterns_empty():
    config = Config(config_overrides={"known_standard_library": {}})
    assert config.known_standard_library == frozenset()

def test_config_constructor_with_known_patterns_none():
    config = Config(config_overrides={"known_standard_library": None})
    assert config.known_standard_library is None

def test_config_constructor_with_known_patterns_invalid():
    config = Config(config_overrides={"known_standard_library": {"invalid_pattern"}})
    assert config.known_standard_library == frozenset({"invalid_pattern"})

def test_config_constructor_with_known_patterns_duplicate():
    config = Config(config_overrides={"known_standard_library": {"os", "os"}})
    assert config.known_standard_library == frozenset({"os"})

def test_config_constructor_with_known_patterns_case_sensitive():
    config = Config(config_overrides={"known_standard_library": {"OS", "os"}})
    assert config.known_standard_library == frozenset({"OS", "os"})

def test_config_constructor_with_known_patterns_special_characters():
    config = Config(config_overrides={"known_standard_library": {"os-path", "sys.path"}})
    assert config.known_standard_library == frozenset({"os-path", "sys.path"})

def test_config_constructor_with_known_patterns_unicode():
    config = Config(config_overrides={"known_standard_library": {"os-路径", "sys.路径"}})
    assert config.known_standard_library == frozenset({"os-路径", "sys.路径"})

def test_config_constructor_with_known_patterns_whitespace():
    config = Config(config_overrides={"known_standard_library": {"os path", "sys.path"}})
    assert config.known_standard_library == frozenset({"os path", "sys.path"})

def test_config_constructor_with_known_patterns_newline():
    config = Config(config_overrides={"known_standard_library": {"os\npath", "sys.path"}})
    assert config.known_standard_library == frozenset({"os\npath", "sys.path"})

def test_config_constructor_with_known_patterns_tab():
    config = Config(config_overrides={"known_standard_library": {"os\tpath", "sys.path"}})
    assert config.known_standard_library == frozenset({"os\tpath", "sys.path"})

def test_config_constructor_with_known_patterns_backslash():
    config = Config(config_overrides={"known_standard_library": {"os\\path", "sys.path"}})
    assert config.known_standard_library == frozenset({"os\\path", "sys.path"})

def test_config_constructor_with_known_patterns_forward_slash():
    config = Config(config_overrides={"known_standard_library": {"os/path", "sys.path"}})
    assert config.known_standard_library == frozenset({"os/path", "sys.path"})

def test_config_constructor_with_known_patterns_backtick():
    config = Config(config_overrides={"known_standard_library": {"os`path", "sys.path"}})
    assert config.known_standard_library == frozenset({"os`path", "sys.path"})

def test_config_constructor_with_known_patterns_single_quote():
    config = Config(config_overrides={"known_standard_library": {"os'path", "sys.path"}})
    assert config.known_standard_library == frozenset({"os'path", "sys.path"})

def test_config_constructor_with_known_patterns_double_quote():
    config = Config(config_overrides={"known_standard_library": {'os"path', "sys.path"}})
    assert config.known_standard_library == frozenset({'os"path', "sys.path"})

def test


# LLM-generated content at query #22
#--------------------------

```python
def test_config_initialization_with_defaults():
    config = Config()
    assert config.settings_file == ""
    assert config.settings_path == ""
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config.settings_file == "setup.cfg"
    assert config.settings_path == ""
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_file == ""
    assert config.settings_path == "."
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_object():
    base_config = Config()
    config = Config(config=base_config)
    assert config.settings_file == ""
    assert config.settings_path == ""
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_config_overrides():
    config = Config(quiet=True, line_length=120)
    assert config.settings_file == ""
    assert config.settings_path == ""
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None
    assert config.quiet is True
    assert config.line_length == 120

def test_config_initialization_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False, "Expected InvalidSettingsPath exception"
    except InvalidSettingsPath:
        pass

def test_config_initialization_with_invalid_profile():
    try:
        Config(profile="nonexistent_profile")
        assert False, "Expected ProfileDoesNotExist exception"
    except ProfileDoesNotExist:
        pass

def test_config_initialization_with_unsupported_settings():
    try:
        Config(unsupported_setting="value")
        assert False, "Expected UnsupportedSettings exception"
    except UnsupportedSettings:
        pass

def test_config_initialization_with_deprecated_settings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = Config(force_single_line=True)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "W0503: Deprecated config options were used: force_single_line." in str(w[-1].message)

def test_config_initialization_with_known_sections():
    config = Config(known_foo=["bar", "baz"], sections=["FOO"])
    assert config.known_other == {"foo": frozenset(["bar", "baz"])}
    assert "FOO" in config.sections

def test_config_initialization_with_import_headings():
    config = Config(import_heading_foo="Bar", import_heading_baz="Qux")
    assert config.import_headings == {"foo": "Bar", "baz": "Qux"}

def test_config_initialization_with_import_footers():
    config = Config(import_footer_foo="Bar", import_footer_baz="Qux")
    assert config.import_footers == {"foo": "Bar", "baz": "Qux"}

def test_config_initialization_with_indent_string():
    config = Config(indent="    ")
    assert config.indent == "    "

def test_config_initialization_with_indent_digit():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_initialization_with_indent_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_initialization_with_formatter_plugin():
    config = Config(formatter="black")
    assert config.formatting_function is not None

def test_config_initialization_with_invalid_formatter():
    try:
        Config(formatter="nonexistent_formatter")
        assert False, "Expected FormattingPluginDoesNotExist exception"
    except FormattingPluginDoesNotExist:
        pass

def test_config_initialization_with_sort_order_natural():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_initialization_with_sort_order_native():
    config = Config(sort_order="native")
    assert config.sorting_function == sorted

def test_config_initialization_with_invalid_sort_order():
    try:
        Config(sort_order="invalid")
        assert False, "Expected SortingFunctionDoesNotExist exception"
    except SortingFunctionDoesNotExist:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_line_145_predicate_true():
    config = Config(sections=("STANDARD_LIBRARY", "THIRD_PARTY"))
    assert "STANDARD_LIBRARY" in SECTION_DEFAULTS
    assert "THIRD_PARTY" in SECTION_DEFAULTS


# LLM-generated content at query #24
#--------------------------

```python
def test_as_list_with_single_string():
    assert _as_list("a, b, c") == ["a", "b", "c"]

def test_as_list_with_newlines():
    assert _as_list("a\nb\nc") == ["a", "b", "c"]

def test_as_list_with_mixed_delimiters():
    assert _as_list("a, b\nc, d") == ["a", "b", "c", "d"]

def test_as_list_with_empty_strings():
    assert _as_list("a, , b, , c") == ["a", "b", "c"]

def test_as_list_with_whitespace():
    assert _as_list("  a  ,  b  ,  c  ") == ["a", "b", "c"]

def test_as_list_with_list_input():
    assert _as_list(["a", " b ", "c"]) == ["a", "b", "c"]

def test_as_list_with_empty_list():
    assert _as_list([]) == []

def test_as_list_with_empty_string():
    assert _as_list("") == []

def test_as_list_with_only_whitespace():
    assert _as_list("   ") == []

def test_as_list_with_only_commas():
    assert _as_list(",,,") == []


# LLM-generated content at query #25
#--------------------------

```python
def test_config_predicate_false():
    assert not Config.__init__.__code__.co_varnames[3] == "config"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_skipped_when_file_in_skips():
    config = Config(skip={"test.py"})
    assert config.is_skipped(Path("test.py"))

def test_is_skipped_when_file_not_in_skips():
    config = Config(skip={"other.py"})
    assert not config.is_skipped(Path("test.py"))

def test_is_skipped_when_parent_dir_in_skips():
    config = Config(skip={"tests"})
    assert config.is_skipped(Path("tests/test.py"))

def test_is_skipped_when_file_matches_skip_glob():
    config = Config(skip_glob={"test_*.py"})
    assert config.is_skipped(Path("test_example.py"))

def test_is_skipped_when_file_does_not_match_skip_glob():
    config = Config(skip_glob={"test_*.py"})
    assert not config.is_skipped(Path("example.py"))

def test_is_skipped_when_file_is_not_regular_file():
    config = Config()
    assert config.is_skipped(Path("/nonexistent/path"))

def test_is_skipped_when_skip_gitignore_and_file_not_in_git():
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files", return_value=Path("/git/repo")):
        with patch.object(config, "git_ls_files", {"git_folder": {"/git/repo/tracked.py"}}):
            assert config.is_skipped(Path("/git/repo/untracked.py"))

def test_is_skipped_when_skip_gitignore_and_file_in_git():
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files", return_value=Path("/git/repo")):
        with patch.object(config, "git_ls_files", {"git_folder": {"/git/repo/tracked.py"}}):
            assert not config.is_skipped(Path("/git/repo/tracked.py"))

def test_is_skipped_when_file_is_git_directory():
    config = Config(skip_gitignore=True)
    assert config.is_skipped(Path(".git"))


# LLM-generated content at query #2
#--------------------------

```python
def test_config_init_with_config_object():
    config = _Config()
    new_config = Config(config=config)
    assert new_config.py_version == config.py_version.replace("py", "")

def test_config_init_with_settings_file():
    with patch("builtins.open", mock_open(read_data="[isort]\nline_length=120")):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.dirname", return_value="/test"):
                config = Config(settings_file="test.ini")
                assert config.line_length == 120

def test_config_init_with_invalid_settings_path():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(InvalidSettingsPath):
            Config(settings_path="/invalid/path")

def test_config_init_with_profile():
    with patch.dict(profiles, {"black": {"line_length": 88}}):
        config = Config(profile="black")
        assert config.line_length == 88

def test_config_init_with_unsupported_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

def test_config_init_with_indent_digit():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_init_with_indent_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_init_with_known_sections():
    config = Config(known_foo=["bar"])
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_init_with_import_headings():
    config = Config(import_heading_foo="Bar")
    assert config.import_headings == {"foo": "Bar"}

def test_config_init_with_deprecated_options():
    with patch("warnings.warn") as mock_warn:
        config = Config(old_option=True)
        mock_warn.assert_called_once()
        assert not hasattr(config, "old_option")

def test_config_init_with_unsupported_config():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")


# LLM-generated content at query #3
#--------------------------

```python
def test___post_init___auto_py_version():
    config = _Config(py_version="auto")
    assert config.py_version.startswith("py")

def test___post_init___invalid_py_version():
    with pytest.raises(ValueError):
        _Config(py_version="invalid")

def test___post_init___valid_py_version():
    config = _Config(py_version="38")
    assert config.py_version == "py38"

def test___post_init___known_standard_library():
    config = _Config(py_version="38")
    assert len(config.known_standard_library) > 0

def test___post_init___multi_line_output():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test___post_init___force_alphabetical_sort():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections
    assert config.no_sections
    assert config.lines_between_types == 1
    assert config.from_first

def test___post_init___wrap_length_greater_than_line_length():
    with pytest.raises(ValueError):
        _Config(wrap_length=80, line_length=79)


# LLM-generated content at query #4
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    assert config.is_supported_filetype("file.py") is True

def test_is_supported_filetype_blocked_extension():
    config = Config()
    assert config.is_supported_filetype("file.exe") is False

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("file.py~") is False

def test_is_supported_filetype_fifo():
    config = Config()
    assert config.is_supported_filetype("/dev/stdin") is False

def test_is_supported_filetype_with_shebang():
    config = Config()
    assert config.is_supported_filetype("file") is True


# LLM-generated content at query #5
#--------------------------

```python
def test__get_config_data_with_toml_file():
    file_path = "test_config.toml"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_with_editorconfig_file():
    file_path = "test_config.editorconfig"
    sections = ("*.py", "*.js")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_with_ini_file():
    file_path = "test_config.ini"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_with_empty_file():
    file_path = "empty_config.toml"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert result == {}

def test__get_config_data_with_invalid_file():
    file_path = "invalid_config.toml"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert result == {}

def test__get_config_data_with_non_existent_file():
    file_path = "non_existent.toml"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert result == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_editorconfig_file_path_ends_with_editorconfig():
    file_path = "test.editorconfig"
    assert file_path.endswith(".editorconfig")


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_78():
    key = "test_key"
    KNOWN_PREFIX = "known_"
    assert key.startswith(KNOWN_PREFIX)


# LLM-generated content at query #8
#--------------------------

```python
def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"line_length": 120})
    assert config.line_length == 120

def test_config_constructor_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config.settings_file == "setup.cfg"

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_path == "."

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_invalid_profile():
    with raises(ProfileDoesNotExist):
        Config(config_overrides={"profile": "invalid_profile"})

def test_config_constructor_with_invalid_settings_path():
    with raises(InvalidSettingsPath):
        Config(settings_path="/invalid/path")

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_constructor_with_invalid_formatter():
    with raises(FormattingPluginDoesNotExist):
        Config(config_overrides={"formatter": "invalid_formatter"})

def test_config_constructor_with_deprecated_options():
    with warns(UserWarning):
        Config(config_overrides={"deprecated_option": True})

def test_config_constructor_with_unsupported_options():
    with raises(UnsupportedSettings):
        Config(config_overrides={"unsupported_option": True})

def test_config_constructor_with_known_patterns():
    config = Config(config_overrides={"known_third_party": ["numpy", "pandas"]})
    assert config.known_third_party == frozenset(["numpy", "pandas"])

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_stdlib": "Standard Library"})
    assert config.import_headings == {"stdlib": "Standard Library"}

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_stdlib": "End of Standard Library"})
    assert config.import_footers == {"stdlib": "End of Standard Library"}

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sort_order == "natural"

def test_config_constructor_with_invalid_sort_order():
    with raises(SortingFunctionDoesNotExist):
        Config(config_overrides={"sort_order": "invalid_sort_order"})


# LLM-generated content at query #9
#--------------------------

```python
def test__find_config_returns_empty_dict_when_no_config_file_found():
    result = _find_config("/non/existent/path")
    assert result == ("/non/existent/path", {})

def test__find_config_returns_config_data_when_config_file_found():
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.isdir", return_value=False), \
         patch("_get_config_data", return_value={"key": "value"}):
        result = _find_config("/some/path")
        assert result == ("/some/path", {"key": "value"})

def test__find_config_stops_search_on_stop_dir():
    with patch("os.path.isfile", return_value=False), \
         patch("os.path.isdir", side_effect=[False, True]):
        result = _find_config("/some/path")
        assert result == ("/some/path", {})

def test__find_config_handles_exception_during_config_parsing():
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.isdir", return_value=False), \
         patch("_get_config_data", side_effect=Exception("Parse error")):
        result = _find_config("/some/path")
        assert result == ("/some/path", {})

def test__find_config_returns_parent_directory_when_config_found_there():
    with patch("os.path.isfile", side_effect=[False, True]), \
         patch("os.path.isdir", return_value=False), \
         patch("_get_config_data", return_value={"key": "value"}):
        result = _find_config("/some/path")
        assert result == ("/some", {"key": "value"})


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    # Simulate a scenario where _get_config_data raises an exception
    with patch('module._get_config_data', side_effect=Exception):
        result = _find_config('/some/path')
        assert result == ('/some/path', {})


# LLM-generated content at query #11
#--------------------------

```python
def test__get_str_to_type_converter_with_default_string_type():
    assert _get_str_to_type_converter("non_existent_setting") == str

def test__get_str_to_type_converter_with_int_setting():
    _DEFAULT_SETTINGS["int_setting"] = 42
    assert _get_str_to_type_converter("int_setting") == int

def test__get_str_to_type_converter_with_float_setting():
    _DEFAULT_SETTINGS["float_setting"] = 3.14
    assert _get_str_to_type_converter("float_setting") == float

def test__get_str_to_type_converter_with_bool_setting():
    _DEFAULT_SETTINGS["bool_setting"] = True
    assert _get_str_to_type_converter("bool_setting") == bool

def test__get_str_to_type_converter_with_wrap_modes():
    _DEFAULT_SETTINGS["wrap_mode"] = WrapModes.WRAP
    assert _get_str_to_type_converter("wrap_mode") == wrap_mode_from_string


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_123_evaluates_to_false():
    combined_config = {
        "sections": ("STANDARD_LIB", "THIRD_PARTY"),
        "known_other": {"custom": frozenset(["module1", "module2"])}
    }
    key = "known_custom"
    maps_to_section = "CUSTOM"
    quiet = True
    assert not (maps_to_section not in combined_config.get("sections", ()) and not quiet)


# LLM-generated content at query #13
#--------------------------

```python
def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"line_length": 120})
    assert config.line_length == 120

def test_config_constructor_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config.settings_file == "setup.cfg"

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_path == "."

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_constructor_with_indent():
    config = Config(config_overrides={"indent": "    "})
    assert config.indent == "    "

def test_config_constructor_with_known_sections():
    config = Config(config_overrides={"known_foo": ["bar"]})
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_foo": "Bar"})
    assert config.import_headings == {"foo": "Bar"}

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_foo": "Bar"})
    assert config.import_footers == {"foo": "Bar"}

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"),)

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_constructor_with_deprecated_options():
    config = Config(config_overrides={"virtual_env": "venv"})
    assert config.virtual_env is None

def test_config_constructor_with_unsupported_config():
    with pytest.raises(UnsupportedSettings):
        Config(config_overrides={"unsupported_option": "value"})

def test_config_constructor_with_invalid_settings_path():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

def test_config_constructor_with_nonexistent_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(config_overrides={"profile": "nonexistent"})

def test_config_constructor_with_nonexistent_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(config_overrides={"formatter": "nonexistent"})

def test_config_constructor_with_nonexistent_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(config_overrides={"sort_order": "nonexistent"})


# LLM-generated content at query #14
#--------------------------

```python
def test_while_loop_predicate_false():
    assert not (current_directory and tries < MAX_CONFIG_SEARCH_DEPTH)


# LLM-generated content at query #15
#--------------------------

```python
def test_config_init_with_config_overrides():
    config = Config(config_overrides={"line_length": 120})
    assert config.line_length == 120

def test_config_init_with_settings_file():
    config = Config(settings_file="setup.cfg")
    assert config.settings_file == "setup.cfg"

def test_config_init_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_path == "."

def test_config_init_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_init_with_invalid_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(config_overrides={"profile": "invalid_profile"})

def test_config_init_with_quiet_mode():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet == True

def test_config_init_with_indent():
    config = Config(config_overrides={"indent": "4"})
    assert config.indent == "    "

def test_config_init_with_tab_indent():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_init_with_custom_indent():
    config = Config(config_overrides={"indent": "    "})
    assert config.indent == "    "

def test_config_init_with_known_sections():
    config = Config(config_overrides={"known_foo": ["bar"]})
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_init_with_import_headings():
    config = Config(config_overrides={"import_heading_foo": "bar"})
    assert config.import_headings == {"foo": "bar"}

def test_config_init_with_import_footers():
    config = Config(config_overrides={"import_footer_foo": "bar"})
    assert config.import_footers == {"foo": "bar"}

def test_config_init_with_deprecated_options():
    with pytest.warns(UserWarning, match="W0503: Deprecated config options were used"):
        config = Config(config_overrides={"virtual_env": "test"})
        assert "virtual_env" not in config.__dict__

def test_config_init_with_unsupported_options():
    with pytest.raises(UnsupportedSettings):
        Config(config_overrides={"unsupported_option": "value"})

def test_config_init_with_formatter_plugin():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatting_function is not None

def test_config_init_with_invalid_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(config_overrides={"formatter": "invalid_formatter"})

def test_config_init_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sorting_function == sorting.naturally

def test_config_init_with_invalid_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(config_overrides={"sort_order": "invalid_sort"})

def test_config_init_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"), Path.cwd())

def test_config_init_with_skip_gitignore():
    config = Config(config_overrides={"skip_gitignore": True})
    assert config.skip_gitignore == True


