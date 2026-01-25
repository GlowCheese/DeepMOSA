####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_config_constructor_defaults():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_settings_file():
    config = Config(settings_file="test.py")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_config_object():
    base_config = Config()
    config = Config(config=base_config)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_config_overrides():
    config = Config(quiet=True)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #2
#--------------------------

```python
def test_known_prefix_condition():
    key = "known_custom_section"
    assert key.startswith(KNOWN_PREFIX) and key not in (
        "known_standard_library",
        "known_future_library",
        "known_third_party",
        "known_first_party",
        "known_local_folder",
    )


# LLM-generated content at query #3
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    assert config.is_supported_filetype("test.py") is True

def test_is_supported_filetype_blocked_extension():
    config = Config()
    assert config.is_supported_filetype("test.exe") is False

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("test.py~") is False

def test_is_supported_filetype_nonexistent_file():
    config = Config()
    assert config.is_supported_filetype("nonexistent_file.py") is False

def test_is_supported_filetype_fifo_file():
    config = Config()
    assert config.is_supported_filetype("/dev/zero") is False

def test_is_supported_filetype_with_shebang():
    config = Config()
    assert config.is_supported_filetype("test_script") is True


# LLM-generated content at query #4
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    assert config.is_supported_filetype("test.py") is True

def test_is_supported_filetype_blocked_extension():
    config = Config(blocked_extensions=["txt"])
    assert config.is_supported_filetype("test.txt") is False

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("test.py~") is False

def test_is_supported_filetype_fifo():
    config = Config()
    assert config.is_supported_filetype("/dev/null") is False

def test_is_supported_filetype_shebang():
    config = Config()
    assert config.is_supported_filetype("test.sh") is True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_config_init_with_config_overrides():
    config = Config(config_overrides={"line_length": 100})
    assert config.line_length == 100

def test_config_init_with_settings_file():
    config = Config(settings_file="pyproject.toml")
    assert config.settings_file == "pyproject.toml"

def test_config_init_with_settings_path():
    config = Config(settings_path=".")
    assert config.settings_path == "."

def test_config_init_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_init_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_init_with_indent():
    config = Config(config_overrides={"indent": "4"})
    assert config.indent == "    "

def test_config_init_with_tab_indent():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_init_with_known_sections():
    config = Config(config_overrides={"known_foo": ["bar"]})
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_init_with_import_headings():
    config = Config(config_overrides={"import_heading_foo": "Bar"})
    assert config.import_headings == {"foo": "Bar"}

def test_config_init_with_import_footers():
    config = Config(config_overrides={"import_footer_foo": "Bar"})
    assert config.import_footers == {"foo": "Bar"}

def test_config_init_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"), Path.cwd())

def test_config_init_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatting_function is not None

def test_config_init_with_unsupported_config():
    with raises(UnsupportedSettings):
        Config(config_overrides={"unsupported_option": "value"})

def test_config_init_with_deprecated_config():
    with warns():
        Config(config_overrides={"deprecated_option": "value"})

def test_config_init_with_skips():
    config = Config(config_overrides={"skip": ["foo"], "extend_skip": ["bar"]})
    assert config.skips == frozenset(["foo", "bar"])

def test_config_init_with_skip_globs():
    config = Config(config_overrides={"skip_glob": ["foo"], "extend_skip_glob": ["bar"]})
    assert config.skip_globs == frozenset(["foo", "bar"])

def test_config_init_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sorting_function == sorting.naturally

def test_config_init_with_custom_sort_order():
    config = Config(config_overrides={"sort_order": "custom"})
    assert config.sorting_function is not None


# LLM-generated content at query #2
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    assert config.is_supported_filetype("file.py") is True

def test_is_supported_filetype_blocked_extension():
    config = Config(blocked_extensions=["txt"])
    assert config.is_supported_filetype("file.txt") is False

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("file.py~") is False

def test_is_supported_filetype_fifo():
    config = Config()
    assert config.is_supported_filetype("/dev/stdin") is False

def test_is_supported_filetype_shebang():
    config = Config()
    assert config.is_supported_filetype("script") is True


# LLM-generated content at query #3
#--------------------------

```python
def test__get_config_data_toml_file():
    file_path = "test.toml"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_editorconfig_file():
    file_path = "test.editorconfig"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_other_config_file():
    file_path = "test.ini"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_empty_file():
    file_path = "empty.ini"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" not in result

def test__get_config_data_editorconfig_indent_style_space():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "indent" in result
    assert result["indent"] == "    "

def test__get_config_data_editorconfig_indent_style_tab():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "indent" in result
    assert result["indent"] == "\t"

def test__get_config_data_editorconfig_max_line_length_off():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "line_length" in result
    assert result["line_length"] == float("inf")

def test__get_config_data_editorconfig_max_line_length_digit():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "line_length" in result
    assert isinstance(result["line_length"], int)

def test__get_config_data_editorconfig_filter_keys():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key in result:
        assert key in _DEFAULT_SETTINGS or key.startswith(KNOWN_PREFIX)

def test__get_config_data_tuple_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if _get_str_to_type_converter(key) is tuple:
            assert isinstance(value, tuple)

def test__get_config_data_frozenset_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if _get_str_to_type_converter(key) is frozenset:
            assert isinstance(value, frozenset)

def test__get_config_data_bool_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if _get_str_to_type_converter(key) is bool:
            assert isinstance(value, bool)

def test__get_config_data_known_prefix_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if key.startswith(KNOWN_PREFIX):
            assert isinstance(value, set)

def test__get_config_data_force_grid_wrap_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    if "force_grid_wrap" in result:
        assert isinstance(result["force_grid_wrap"], int)

def test__get_config_data_comment_prefix_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    if "comment_prefix" in result:
        assert isinstance(result["comment_prefix"], str)


# LLM-generated content at query #4
#--------------------------

```python
def test_profile_not_in_profiles():
    config_overrides = {"profile": "non_existent_profile"}
    assert config_overrides.get("profile", "") not in profiles


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_78_evaluates_to_true():
    key = "some_key"
    KNOWN_PREFIX = "known_"
    assert key.startswith(KNOWN_PREFIX)


# LLM-generated content at query #6
#--------------------------

```python
def test_maps_to_section_in_known_section_mapping():
    key = "known_custom_section"
    value = ["custom_module"]
    combined_config = {key: value}
    KNOWN_SECTION_MAPPING = {"CUSTOM_SECTION": "CUSTOM"}
    import_heading = key[len("known_"):].lower()
    maps_to_section = import_heading.upper()
    assert maps_to_section in KNOWN_SECTION_MAPPING


# LLM-generated content at query #7
#--------------------------

```python
def test__find_config_found():
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.isdir", return_value=False), \
         patch("_get_config_data", return_value={"key": "value"}) as mock_get_config:
        result = _find_config("/some/path")
        assert result == ("/some/path", {"key": "value"})
        mock_get_config.assert_called_once()

def test__find_config_not_found():
    with patch("os.path.isfile", return_value=False), \
         patch("os.path.isdir", return_value=False):
        result = _find_config("/some/path")
        assert result == ("/some/path", {})

def test__find_config_stop_dir():
    with patch("os.path.isfile", return_value=False), \
         patch("os.path.isdir", side_effect=[False, True]):
        result = _find_config("/some/path")
        assert result == ("/some/path", {})

def test__find_config_exception():
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.isdir", return_value=False), \
         patch("_get_config_data", side_effect=Exception("test error")) as mock_get_config, \
         patch("warnings.warn") as mock_warn:
        result = _find_config("/some/path")
        assert result == ("/some/path", {})
        mock_get_config.assert_called_once()
        mock_warn.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_line_43_predicate_true():
    config = Config(settings_file="nonexistent_file.cfg", quiet=False)
    assert not config._known_patterns
    assert not config._section_comments
    assert not config._section_comments_end
    assert not config._skips
    assert not config._skip_globs
    assert not config._sorting_function


# LLM-generated content at query #9
#--------------------------

```python
def test_config_initialization_with_defaults():
    config = Config()
    assert config.py_version == "310"
    assert config.line_length == 79
    assert config.indent == "    "
    assert config.force_single_line is False
    assert config.force_grid_wrap == 0
    assert config.use_parentheses is True
    assert config.ensure_newline_before_comments is True
    assert config.lines_after_imports == -1
    assert config.lines_between_types == 0
    assert config.lines_between_sections == 1
    assert config.multi_line_output == 3
    assert config.include_trailing_comma is True
    assert config.force_sort_within_sections is False
    assert config.force_alphabetical_sort_within_sections is False
    assert config.force_to_top == ()
    assert config.order_by_type is True
    assert config.atomic is True
    assert config.lines_between_classes == 2
    assert config.lines_between_functions == 2
    assert config.combine_as_imports is True
    assert config.combine_star is True
    assert config.force_alphabetical_sort is False
    assert config.force_straight_quotes is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.force_alphabetical_sort_by_case is False
    assert config.


# LLM-generated content at query #10
#--------------------------

```python
def test_find_config_finds_pyproject_toml():
    assert _find_config(os.path.dirname(__file__))[0] == os.path.dirname(__file__)

def test_find_config_finds_setup_cfg():
    assert _find_config(os.path.dirname(__file__))[0] == os.path.dirname(__file__)

def test_find_config_finds_tox_ini():
    assert _find_config(os.path.dirname(__file__))[0] == os.path.dirname(__file__)

def test_find_config_finds_setup_cfg_in_parent_dir():
    assert _find_config(os.path.dirname(__file__))[0] == os.path.dirname(os.path.dirname(__file__))

def test_find_config_returns_empty_dict_when_no_config_found():
    assert _find_config("/nonexistent/path")[1] == {}

def test_find_config_stops_at_git_dir():
    assert _find_config(os.path.join(os.path.dirname(__file__), ".git"))[0] == os.path.dirname(__file__)

def test_find_config_stops_at_hg_dir():
    assert _find_config(os.path.join(os.path.dirname(__file__), ".hg"))[0] == os.path.dirname(__file__)

def test_find_config_stops_at_svn_dir():
    assert _find_config(os.path.join(os.path.dirname(__file__), ".svn"))[0] == os.path.dirname(__file__)

def test_find_config_stops_at_root_dir():
    assert _find_config("/")[0] == "/"

def test_find_config_returns_correct_config_data():
    config_path, config_data = _find_config(os.path.dirname(__file__))
    assert config_data["source"] == os.path.join(config_path, "pyproject.toml")

def test_find_config_ignores_invalid_config_files():
    assert _find_config(os.path.join(os.path.dirname(__file__), "invalid_config.toml"))[1] == {}


# LLM-generated content at query #11
#--------------------------

```python
def test__as_list_with_single_string():
    assert _as_list("a,b,c") == ["a", "b", "c"]

def test__as_list_with_single_string_with_newlines():
    assert _as_list("a,b\nc,d") == ["a", "b", "c", "d"]

def test__as_list_with_single_string_with_whitespace():
    assert _as_list(" a , b , c ") == ["a", "b", "c"]

def test__as_list_with_single_string_empty_items():
    assert _as_list("a,,b,c") == ["a", "b", "c"]

def test__as_list_with_list_input():
    assert _as_list([" a ", " b ", " c "]) == ["a", "b", "c"]

def test__as_list_with_empty_string():
    assert _as_list("") == []

def test__as_list_with_only_whitespace():
    assert _as_list("   ") == []

def test__as_list_with_only_commas():
    assert _as_list(",,,") == []


# LLM-generated content at query #12
#--------------------------

```python
def test_config_constructor_default():
    config = Config()
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nline_length=120\n")
        f.flush()
        config = Config(settings_file=f.name)
    assert config.line_length == 120
    os.unlink(f.name)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.isort]\nline_length=120\n")
        config = Config(settings_path=tmpdir)
    assert config.line_length == 120

def test_config_constructor_with_config_overrides():
    config = Config(quiet=True, line_length=120)
    assert config.quiet is True
    assert config.line_length == 120

def test_config_constructor_with_invalid_settings_path():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.line_length == 88
    assert config.multi_line_output == 3

def test_config_constructor_with_invalid_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

def test_config_constructor_with_known_sections():
    config = Config(known_django=["django", "rest_framework"])
    assert "django" in config.known_django
    assert "rest_framework" in config.known_django

def test_config_constructor_with_import_headings():
    config = Config(import_heading_django="Django imports")
    assert config.import_headings == {"django": "Django imports"}

def test_config_constructor_with_import_footers():
    config = Config(import_footer_django="End of Django imports")
    assert config.import_footers == {"django": "End of Django imports"}

def test_config_constructor_with_deprecated_options():
    with pytest.warns(UserWarning, match="W0503: Deprecated config options were used"):
        config = Config(force_single_line=True)
    assert not hasattr(config, "force_single_line")

def test_config_constructor_with_unsupported_options():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

def test_config_constructor_with_formatter():
    config = Config(formatter="black")
    assert config.formatting_function is not None

def test_config_constructor_with_invalid_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent")

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_invalid_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="nonexistent")

def test_config_constructor_with_config_object():
    base_config = Config(line_length=120)
    config = Config(config=base_config, line_length=100)
    assert config.line_length == 100


# LLM-generated content at query #13
#--------------------------

```python
def test_post_init_default_py_version():
    config = _Config()
    assert config.py_version == "py3"

def test_post_init_auto_py_version():
    config = _Config(py_version="auto")
    assert config.py_version == f"py{sys.version_info.major}{sys.version_info.minor}"

def test_post_init_invalid_py_version():
    with pytest.raises(ValueError):
        _Config(py_version="invalid")

def test_post_init_known_standard_library():
    config = _Config()
    assert config.known_standard_library == frozenset(getattr(stdlibs, config.py_version).stdlib)

def test_post_init_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test_post_init_force_alphabetical_sort():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True

def test_post_init_wrap_length_greater_than_line_length():
    with pytest.raises(ValueError):
        _Config(wrap_length=80, line_length=79)


# LLM-generated content at query #14
#--------------------------

```python
def test_is_supported_filetype_oserror():
    config = Config()
    assert not config.is_supported_filetype("nonexistent_file.py")


# LLM-generated content at query #15
#--------------------------

```python
def test_indent_is_digit():
    config = Config(indent="4")
    assert config.indent == "    "


# LLM-generated content at query #16
#--------------------------

```python
def test_is_skipped_returns_true_for_skipped_file():
    config = Config(skip={".gitignore"})
    assert config.is_skipped(Path(".gitignore"))

def test_is_skipped_returns_false_for_non_skipped_file():
    config = Config()
    assert not config.is_skipped(Path("test.py"))

def test_is_skipped_returns_true_for_file_matching_skip_glob():
    config = Config(skip_glob={"*.tmp"})
    assert config.is_skipped(Path("temp.tmp"))

def test_is_skipped_returns_false_for_file_not_matching_skip_glob():
    config = Config(skip_glob={"*.tmp"})
    assert not config.is_skipped(Path("test.py"))

def test_is_skipped_returns_true_for_file_in_skipped_directory():
    config = Config(skip={"__pycache__"})
    assert config.is_skipped(Path("__pycache__/test.py"))

def test_is_skipped_returns_false_for_file_not_in_skipped_directory():
    config = Config(skip={"__pycache__"})
    assert not config.is_skipped(Path("src/test.py"))

def test_is_skipped_returns_true_for_non_existent_file():
    config = Config()
    assert config.is_skipped(Path("non_existent_file.py"))

def test_is_skipped_returns_true_for_editor_backup_file():
    config = Config()
    assert config.is_skipped(Path("test.py~"))

def test_is_skipped_returns_false_for_non_editor_backup_file():
    config = Config()
    assert not config.is_skipped(Path("test.py"))

def test_is_skipped_returns_true_for_git_ignored_file():
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files") as mock_check:
        mock_check.return_value = Path("/test")
        config.git_ls_files[Path("/test")] = {"/test/file.py"}
        assert config.is_skipped(Path("/test/ignored.py"))

def test_is_skipped_returns_false_for_git_tracked_file():
    config = Config(skip_gitignore=True)
    with patch.object(config, "_check_folder_git_ls_files") as mock_check:
        mock_check.return_value = Path("/test")
        config.git_ls_files[Path("/test")] = {"/test/file.py"}
        assert not config.is_skipped(Path("/test/file.py"))


# LLM-generated content at query #17
#--------------------------

```python
def test_is_supported_filetype_oserror_in_stat():
    config = Config()
    assert config.is_supported_filetype("nonexistent_file.py") is False


# LLM-generated content at query #18
#--------------------------

```python
def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"line_length": 100})
    assert config.line_length == 100

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
    config = Config(config_overrides={"indent": "4"})
    assert config.indent == "    "

def test_config_constructor_with_indent_tab():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_constructor_with_known_sections():
    config = Config(config_overrides={"known_foo": ["bar"]})
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_foo": "bar"})
    assert config.import_headings == {"foo": "bar"}

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_foo": "bar"})
    assert config.import_footers == {"foo": "bar"}

def test_config_constructor_with_deprecated_options():
    config = Config(config_overrides={"virtual_env": "test"})
    assert "virtual_env" not in config.__dict__

def test_config_constructor_with_unsupported_config():
    try:
        config = Config(config_overrides={"unsupported_option": "test"})
    except UnsupportedSettings as e:
        assert "unsupported_option" in str(e)

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatting_function is not None

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_config_object():
    base_config = Config()
    config = Config(config=base_config, config_overrides={"line_length": 100})
    assert config.line_length == 100

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert len(config.src_paths) == 1
    assert "src" in str(config.src_paths[0])

def test_config_constructor_with_directory():
    config = Config(config_overrides={"directory": "."})
    assert config.directory == "."

def test_config_constructor_with_sections():
    config = Config(config_overrides={"sections": ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]})
    assert config.sections == ("FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER")

def test_config_constructor_with_skip():
    config = Config(config_overrides={"skip": ["test.py"]})
    assert "test.py" in config.skips

def test_config_constructor_with_skip_glob():
    config = Config(config_overrides={"skip_glob": ["*.py"]})
    assert "*.py" in config.skip_globs


# LLM-generated content at query #19
#--------------------------

```python
def test_get_str_to_type_converter_with_default_string():
    assert _get_str_to_type_converter("nonexistent_setting") == str

def test_get_str_to_type_converter_with_int_setting():
    _DEFAULT_SETTINGS["int_setting"] = 42
    assert _get_str_to_type_converter("int_setting") == int

def test_get_str_to_type_converter_with_float_setting():
    _DEFAULT_SETTINGS["float_setting"] = 3.14
    assert _get_str_to_type_converter("float_setting") == float

def test_get_str_to_type_converter_with_bool_setting():
    _DEFAULT_SETTINGS["bool_setting"] = True
    assert _get_str_to_type_converter("bool_setting") == bool

def test_get_str_to_type_converter_with_wrap_modes():
    _DEFAULT_SETTINGS["wrap_mode"] = WrapModes.CLAMP
    assert _get_str_to_type_converter("wrap_mode") == wrap_mode_from_string


# LLM-generated content at query #20
#--------------------------

```python
def test_wrap_modes_type_converter():
    assert _get_str_to_type_converter("wrap_mode") == wrap_mode_from_string


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_123_evaluates_to_false():
    assert not (maps_to_section not in combined_config.get("sections", ()) and not quiet)


# LLM-generated content at query #22
#--------------------------

```python
def test__get_config_data_toml_file():
    file_path = "test.toml"
    sections = ("section1", "section2")
    with open(file_path, "w") as f:
        f.write("[section1]\nkey1 = 'value1'\nkey2 = 123\n\n[section2]\nkey3 = true\nkey4 = [1, 2, 3]")
    result = _get_config_data(file_path, sections)
    assert result == {"key1": "value1", "key2": 123, "key3": True, "key4": [1, 2, 3], "source": file_path}
    os.remove(file_path)

def test__get_config_data_editorconfig_file():
    file_path = "test.editorconfig"
    sections = ("*.py",)
    with open(file_path, "w") as f:
        f.write("root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88")
    result = _get_config_data(file_path, sections)
    assert result == {"indent": "    ", "line_length": 88, "source": file_path}
    os.remove(file_path)

def test__get_config_data_ini_file():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\nkey1 = value1\nkey2 = 123")
    result = _get_config_data(file_path, sections)
    assert result == {"key1": "value1", "key2": 123, "source": file_path}
    os.remove(file_path)

def test__get_config_data_empty_file():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("")
    result = _get_config_data(file_path, sections)
    assert result == {}
    os.remove(file_path)

def test__get_config_data_missing_sections():
    file_path = "test.ini"
    sections = ("section1", "section2")
    with open(file_path, "w") as f:
        f.write("[section3]\nkey1 = value1")
    result = _get_config_data(file_path, sections)
    assert result == {}
    os.remove(file_path)

def test__get_config_data_force_grid_wrap():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\nforce_grid_wrap = false")
    result = _get_config_data(file_path, sections)
    assert result == {"force_grid_wrap": 0, "source": file_path}
    os.remove(file_path)

def test__get_config_data_comment_prefix():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\ncomment_prefix = '#'")
    result = _get_config_data(file_path, sections)
    assert result == {"comment_prefix": "#", "source": file_path}
    os.remove(file_path)

def test__get_config_data_abspaths():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\nknown_prefix.paths = path1, path2")
    result = _get_config_data(file_path, sections)
    assert result == {"known_prefix.paths": {os.path.join(os.path.dirname(file_path), "path1"), os.path.join(os.path.dirname(file_path), "path2")}, "source": file_path}
    os.remove(file_path)


# LLM-generated content at query #23
#--------------------------

```python
def test_config_initialization_with_config_parameter():
    config_instance = Config(config=_Config())
    assert config_instance._known_patterns is None
    assert config_instance._section_comments is None
    assert config_instance._section_comments_end is None
    assert config_instance._skips is None
    assert config_instance._skip_globs is None
    assert config_instance._sorting_function is None


# LLM-generated content at query #24
#--------------------------

```python
def test_vertical_grid_grouped_no_comma_conversion():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED


# LLM-generated content at query #25
#--------------------------

```python
def test_deprecated_options_used():
    config = Config(**{"deprecated_option": "value"})
    assert "deprecated_option" not in config.__dict__


# LLM-generated content at query #26
#--------------------------

```python
def test_wrap_modes_type_converter():
    assert _get_str_to_type_converter("wrap_mode") == wrap_mode_from_string


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_skipped_returns_true_for_exact_skip_path():
    config = Config(skip={"test_file.py"})
    assert config.is_skipped(Path("test_file.py")) is True

def test_is_skipped_returns_true_for_parent_directory_in_skip():
    config = Config(skip={"test_dir"})
    assert config.is_skipped(Path("test_dir/subfile.py")) is True

def test_is_skipped_returns_true_for_skip_glob_match():
    config = Config(skip_glob={"*.pyc"})
    assert config.is_skipped(Path("test.pyc")) is True

def test_is_skipped_returns_true_for_skip_glob_match_with_path():
    config = Config(skip_glob={"test_*"})
    assert config.is_skipped(Path("test_file.py")) is True

def test_is_skipped_returns_false_for_non_matching_file():
    config = Config(skip={"other_file.py"})
    assert config.is_skipped(Path("test_file.py")) is False

def test_is_skipped_returns_true_for_non_existent_file():
    config = Config()
    assert config.is_skipped(Path("non_existent_file.py")) is True

def test_is_skipped_returns_true_for_gitignore_when_skip_gitignore_enabled():
    config = Config(skip_gitignore=True)
    config.git_ls_files = {Path("/test"): {"/test/committed_file.py"}}
    assert config.is_skipped(Path("/test/ignored_file.py")) is True

def test_is_skipped_returns_false_for_git_tracked_file_when_skip_gitignore_enabled():
    config = Config(skip_gitignore=True)
    config.git_ls_files = {Path("/test"): {"/test/committed_file.py"}}
    assert config.is_skipped(Path("/test/committed_file.py")) is False

def test_is_skipped_returns_true_for_git_directory():
    config = Config(skip_gitignore=True)
    assert config.is_skipped(Path(".git")) is True


# LLM-generated content at query #2
#--------------------------

```python
def test_config_init_with_defaults():
    config = Config()
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_init_with_settings_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nprofile=black\n")
        settings_file = f.name

    config = Config(settings_file=settings_file)
    assert config.profile == "black"
    os.unlink(settings_file)

def test_config_init_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = os.path.join(tmpdir, "test")
        os.makedirs(settings_path)
        config = Config(settings_path=settings_path)
        assert config.directory == settings_path

def test_config_init_with_config_overrides():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_init_with_config_object():
    base_config = Config(indent="4")
    config = Config(config=base_config, line_length=100)
    assert config.indent == "    "
    assert config.line_length == 100

def test_config_init_with_invalid_settings_path():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

def test_config_init_with_invalid_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")

def test_config_init_with_unsupported_settings():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_setting="value")

def test_config_init_with_deprecated_settings():
    with pytest.warns(UserWarning):
        Config(force_single_line=True)

def test_config_init_with_known_sections():
    config = Config(known_foo=["bar"])
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_init_with_import_headings():
    config = Config(import_heading_foo="Bar")
    assert config.import_headings == {"foo": "Bar"}

def test_config_init_with_import_footers():
    config = Config(import_footer_foo="Bar")
    assert config.import_footers == {"foo": "Bar"}

def test_config_init_with_custom_sorting_function():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_init_with_invalid_sorting_function():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="invalid")


# LLM-generated content at query #3
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
    config = Config(settings_file="test_settings.cfg")
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
    config = Config(quiet=True)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #4
#--------------------------

```python
def test_find_config_returns_empty_dict_when_no_config_file_found():
    result = _find_config("/non/existent/path")
    assert result == ("/non/existent/path", {})

def test_find_config_returns_correct_config_when_file_exists():
    # Assuming a test config file exists at the path
    result = _find_config(os.path.dirname(__file__))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], dict)

def test_find_config_stops_search_on_stop_dir():
    # Assuming a stop directory exists in the path
    result = _find_config(os.path.join(os.path.dirname(__file__), "stop_dir"))
    assert result == (os.path.join(os.path.dirname(__file__), "stop_dir"), {})

def test_find_config_returns_correct_directory_and_config():
    # Assuming a config file exists in the parent directory
    result = _find_config(os.path.dirname(__file__))
    assert result[0] == os.path.dirname(__file__)
    assert isinstance(result[1], dict)


# LLM-generated content at query #5
#--------------------------

```python
def test_import_heading_prefix_detection():
    combined_config = {
        "import_heading_prefix_test": "test_value",
        "some_other_key": "other_value"
    }
    key = "import_heading_prefix_test"
    value = "test_value"
    assert key.startswith("import_heading_prefix")
    assert import_headings[key[len("import_heading_prefix"):].lower()] == str(value)


# LLM-generated content at query #6
#--------------------------

```python
def test_empty_string():
    assert _as_list("") == []

def test_single_item():
    assert _as_list("item") == ["item"]

def test_multiple_items_comma_separated():
    assert _as_list("item1, item2, item3") == ["item1", "item2", "item3"]

def test_multiple_items_newline_separated():
    assert _as_list("item1\nitem2\nitem3") == ["item1", "item2", "item3"]

def test_mixed_separators():
    assert _as_list("item1, item2\nitem3") == ["item1", "item2", "item3"]

def test_whitespace_handling():
    assert _as_list("  item1  ,  item2  ") == ["item1", "item2"]

def test_empty_items_filtered():
    assert _as_list("item1, , item2") == ["item1", "item2"]

def test_list_input():
    assert _as_list(["item1", " item2 ", "item3"]) == ["item1", "item2", "item3"]

def test_list_with_empty_items():
    assert _as_list(["item1", "", "item2"]) == ["item1", "item2"]


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_false():
    path = ""
    tries = MAX_CONFIG_SEARCH_DEPTH
    assert not (path and tries < MAX_CONFIG_SEARCH_DEPTH)


# LLM-generated content at query #8
#--------------------------

```python
def test_path_root_is_dir():
    config = Config(settings_path="/path/to/existing/directory")
    assert not (Path("/path/to/existing/directory").resolve().is_dir() is False)


# LLM-generated content at query #9
#--------------------------

```python
def test_config_settings_empty_and_quiet_false_shows_warning():
    config = Config(settings_file="test.cfg", quiet=False)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #10
#--------------------------

```python
def test_profile_name_exists_in_profiles():
    config = Config(profile="black")
    assert config.profile == "black"


# LLM-generated content at query #11
#--------------------------

```python
def test_is_skipped_predicate_false():
    config = Config()
    file_path = Path("/some/path")
    assert not (config.directory and Path(config.directory) in file_path.resolve().parents)


# LLM-generated content at query #12
#--------------------------

```python
def test_abspaths_with_absolute_paths():
    cwd = "/home/user"
    values = ["/absolute/path1/", "/absolute/path2"]
    result = _abspaths(cwd, values)
    assert result == {"/absolute/path1/", "/absolute/path2"}

def test_abspaths_with_relative_paths():
    cwd = "/home/user"
    values = ["relative/path1/", "relative/path2"]
    result = _abspaths(cwd, values)
    assert result == {"/home/user/relative/path1/", "relative/path2"}

def test_abspaths_with_mixed_paths():
    cwd = "/home/user"
    values = ["/absolute/path1/", "relative/path2/"]
    result = _abspaths(cwd, values)
    assert result == {"/absolute/path1/", "/home/user/relative/path2/"}

def test_abspaths_with_empty_values():
    cwd = "/home/user"
    values = []
    result = _abspaths(cwd, values)
    assert result == set()

def test_abspaths_with_duplicate_paths():
    cwd = "/home/user"
    values = ["/absolute/path1/", "/absolute/path1/"]
    result = _abspaths(cwd, values)
    assert result == {"/absolute/path1/"}


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    assert not (False or True)


# LLM-generated content at query #14
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

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("test.py~") == False

def test_is_supported_filetype_fifo():
    config = Config()
    import os
    import stat
    os.makedirs("/tmp", exist_ok=True)
    fifo_path = "/tmp/test_fifo"
    os.mkfifo(fifo_path)
    assert config.is_supported_filetype(fifo_path) == False
    os.remove(fifo_path)

def test_is_supported_filetype_nonexistent_file():
    config = Config()
    assert config.is_supported_filetype("nonexistent_file.py") == False

def test_is_supported_filetype_shebang():
    config = Config()
    with open("test_shebang.py", "w") as f:
        f.write("#!/usr/bin/env python\nprint('hello')")
    assert config.is_supported_filetype("test_shebang.py") == True
    os.remove("test_shebang.py")

def test_is_supported_filetype_no_shebang():
    config = Config()
    with open("test_no_shebang.py", "w") as f:
        f.write("print('hello')")
    assert config.is_supported_filetype("test_no_shebang.py") == False
    os.remove("test_no_shebang.py")


# LLM-generated content at query #15
#--------------------------

```python
def test_import_footers_predicate():
    config_overrides = {"import_footer_test": "footer_value"}
    combined_config = {"import_footer_test": "footer_value"}
    import_footers = {}
    for key, value in tuple(combined_config.items()):
        if key.startswith("import_footer"):
            import_footers[key[len("import_footer") :].lower()] = str(value)
    assert "test" in import_footers


# LLM-generated content at query #16
#--------------------------

```python
def test_config_initialization_with_defaults():
    config = Config()
    assert config.settings_file == ""
    assert config.settings_path == ""
    assert config.config is None
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

def test_config_initialization_with_settings_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nline_length=120\n")
        settings_file = f.name

    config = Config(settings_file=settings_file)
    assert config.settings_file == settings_file
    assert config.line_length == 120

    os.unlink(settings_file)

def test_config_initialization_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = os.path.join(tmpdir, "test")
        os.makedirs(settings_path)

        config = Config(settings_path=settings_path)
        assert config.settings_path == os.path.abspath(settings_path)

def test_config_initialization_with_config_overrides():
    config = Config(quiet=True, line_length=88)
    assert config.quiet is True
    assert config.line_length == 88

def test_config_initialization_with_existing_config():
    existing_config = Config(quiet=True, line_length=88)
    new_config = Config(config=existing_config, line_length=120)
    assert new_config.quiet is True
    assert new_config.line_length == 120

def test_config_initialization_with_profile():
    config = Config(profile="black")
    assert config.profile == "black"
    assert config.source == "black profile"

def test_config_initialization_with_invalid_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="invalid_profile")

def test_config_initialization_with_indent():
    config = Config(indent="4")
    assert config.indent == "    "

    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_initialization_with_known_sections():
    config = Config(known_foo="bar")
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_initialization_with_import_headings():
    config = Config(import_heading_foo="bar")
    assert config.import_headings == {"foo": "bar"}

def test_config_initialization_with_import_footers():
    config = Config(import_footer_foo="bar")
    assert config.import_footers == {"foo": "bar"}

def test_config_initialization_with_unsupported_config():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

def test_config_initialization_with_deprecated_config():
    with pytest.warns(UserWarning):
        Config(force_single_line=True)

def test_config_initialization_with_formatter():
    config = Config(formatter="black")
    assert config.formatter == "black"
    assert callable(config.formatting_function)

def test_config_initialization_with_invalid_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="invalid_formatter")

def test_config_initialization_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

    config = Config(sort_order="native")
    assert config.sorting_function == sorted

def test_config_initialization_with_invalid_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="invalid_sort_order")


# LLM-generated content at query #17
#--------------------------

```python
def test_config_default_initialization():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_with_settings_file():
    config = Config(settings_file="test.py")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_with_settings_path():
    config = Config(settings_path="test")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_with_config_overrides():
    config = Config(indent="4")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_with_existing_config():
    existing_config = Config()
    config = Config(config=existing_config)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_with_profile():
    config = Config(profile="black")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_with_unsupported_config():
    try:
        config = Config(unsupported_option="value")
    except UnsupportedSettings:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_find_all_configs_empty_path():
    result = find_all_configs("")
    assert isinstance(result, Trie)
    assert result.root.config_info == ("default", {})

def test_find_all_configs_no_config_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_all_configs(tmpdir)
        assert isinstance(result, Trie)
        assert result.root.config_info == ("default", {})

def test_find_all_configs_with_valid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file, "w") as f:
            f.write("[settings]\nline_length=88\n")

        result = find_all_configs(tmpdir)
        assert isinstance(result, Trie)
        assert result.root.nodes[tmpdir.split(os.sep)[-1]].config_info[1]["line_length"] == 88

def test_find_all_configs_with_invalid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file, "w") as f:
            f.write("invalid config content")

        result = find_all_configs(tmpdir)
        assert isinstance(result, Trie)
        assert result.root.config_info == ("default", {})

def test_find_all_configs_nested_configs():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)

        config_file1 = os.path.join(tmpdir, ".isort.cfg")
        with open(config_file1, "w") as f:
            f.write("[settings]\nline_length=88\n")

        config_file2 = os.path.join(subdir, "pyproject.toml")
        with open(config_file2, "w") as f:
            f.write("[tool.isort]\nline_length=120\n")

        result = find_all_configs(tmpdir)
        assert isinstance(result, Trie)
        assert result.root.nodes[tmpdir.split(os.sep)[-1]].config_info[1]["line_length"] == 88
        assert result.root.nodes[tmpdir.split(os.sep)[-1]].nodes["subdir"].config_info[1]["line_length"] == 120


# LLM-generated content at query #19
#--------------------------

```python
def test_config_constructor_with_no_arguments():
    config = Config()
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nline_length = 100\n")
        f.flush()
        config = Config(settings_file=f.name)
    assert config.line_length == 100
    os.unlink(f.name)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.isort]\nline_length = 120\n")
        config = Config(settings_path=tmpdir)
    assert config.line_length == 120

def test_config_constructor_with_config_object():
    base_config = Config(line_length=80)
    new_config = Config(config=base_config, line_length=100)
    assert new_config.line_length == 100

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.line_length == 88
    assert config.multi_line_output == 3

def test_config_constructor_with_invalid_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

def test_config_constructor_with_indent_as_int():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_sections():
    config = Config(known_foo=["bar"])
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_constructor_with_import_headings():
    config = Config(import_heading_foo="Bar Imports")
    assert config.import_headings == {"foo": "Bar Imports"}

def test_config_constructor_with_import_footers():
    config = Config(import_footer_foo="End of Bar Imports")
    assert config.import_footers == {"foo": "End of Bar Imports"}

def test_config_constructor_with_deprecated_options():
    with pytest.warns(UserWarning, match="W0503: Deprecated config options were used"):
        config = Config(force_single_line=True)

def test_config_constructor_with_unsupported_options():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="black")
    assert config.formatting_function is not None

def test_config_constructor_with_invalid_formatter_plugin():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent")

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_invalid_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="invalid")


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config_data_toml_file():
    file_path = "test.toml"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_editorconfig_file():
    file_path = "test.editorconfig"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_other_config_file():
    file_path = "test.ini"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test_get_config_data_empty_file():
    file_path = "empty.ini"
    sections = ("section1", "section2")
    result = _get_config_data(file_path, sections)
    assert result == {}

def test_get_config_data_editorconfig_indent_style_space():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "indent" in result
    assert result["indent"] == "    "

def test_get_config_data_editorconfig_indent_style_tab():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "indent" in result
    assert result["indent"] == "\t"

def test_get_config_data_editorconfig_max_line_length_off():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "line_length" in result
    assert result["line_length"] == float("inf")

def test_get_config_data_editorconfig_max_line_length_digit():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "line_length" in result
    assert result["line_length"] == 80

def test_get_config_data_editorconfig_filtered_keys():
    file_path = "test.editorconfig"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key in result:
        assert key in _DEFAULT_SETTINGS or key.startswith(KNOWN_PREFIX)

def test_get_config_data_tuple_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if _get_str_to_type_converter(key) is tuple:
            assert isinstance(value, tuple)

def test_get_config_data_frozenset_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if _get_str_to_type_converter(key) is frozenset:
            assert isinstance(value, frozenset)

def test_get_config_data_bool_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if _get_str_to_type_converter(key) is bool:
            assert isinstance(value, bool)

def test_get_config_data_known_prefix_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    for key, value in result.items():
        if key.startswith(KNOWN_PREFIX):
            assert isinstance(value, set)

def test_get_config_data_force_grid_wrap_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    if "force_grid_wrap" in result:
        assert isinstance(result["force_grid_wrap"], int)

def test_get_config_data_comment_prefix_conversion():
    file_path = "test.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    if "comment_prefix" in result:
        assert isinstance(result["comment_prefix"], str)


# LLM-generated content at query #21
#--------------------------

```python
def test_config_parameter_is_none():
    config_instance = Config()
    assert config_instance is not None


# LLM-generated content at query #22
#--------------------------

```python
def test_is_supported_filetype_with_supported_extension():
    config = Config()
    assert config.is_supported_filetype("test.py") is True

def test_is_supported_filetype_with_blocked_extension():
    config = Config(blocked_extensions=["txt"])
    assert config.is_supported_filetype("test.txt") is False

def test_is_supported_filetype_with_editor_backup_file():
    config = Config()
    assert config.is_supported_filetype("test.py~") is False

def test_is_supported_filetype_with_fifo_file():
    import os
    import stat
    config = Config()
    fifo_path = "test_fifo"
    os.mkfifo(fifo_path)
    assert config.is_supported_filetype(fifo_path) is False
    os.unlink(fifo_path)

def test_is_supported_filetype_with_nonexistent_file():
    config = Config()
    assert config.is_supported_filetype("nonexistent_file.py") is False

def test_is_supported_filetype_with_shebang():
    import os
    config = Config()
    test_file = "test_shebang"
    with open(test_file, "w") as f:
        f.write("#!/usr/bin/env python\n")
    assert config.is_supported_filetype(test_file) is True
    os.unlink(test_file)

def test_is_supported_filetype_without_shebang():
    import os
    config = Config()
    test_file = "test_no_shebang"
    with open(test_file, "w") as f:
        f.write("print('hello')\n")
    assert config.is_supported_filetype(test_file) is False
    os.unlink(test_file)


# LLM-generated content at query #23
#--------------------------

```python
def test___post_init___default_py_version():
    config = _Config()
    assert config.py_version == "py3"

def test___post_init___auto_py_version():
    config = _Config(py_version="auto")
    assert config.py_version == f"py{sys.version_info.major}{sys.version_info.minor}"

def test___post_init___invalid_py_version():
    with pytest.raises(ValueError):
        _Config(py_version="invalid")

def test___post_init___py_version_all():
    config = _Config(py_version="all")
    assert config.py_version == "all"

def test___post_init___known_standard_library_populated():
    config = _Config()
    assert len(config.known_standard_library) > 0

def test___post_init___vertical_grid_grouped_no_comma_converted():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test___post_init___force_alphabetical_sort_sets_related_flags():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True

def test___post_init___wrap_length_exceeds_line_length():
    with pytest.raises(ValueError):
        _Config(wrap_length=80, line_length=79)


# LLM-generated content at query #24
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
    sections = ("*.py",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_with_ini_file():
    file_path = "test_config.ini"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_with_empty_sections():
    file_path = "test_config.toml"
    sections = ()
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" in result
    assert result["source"] == file_path

def test__get_config_data_with_nonexistent_file():
    file_path = "nonexistent_file.toml"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert isinstance(result, dict)
    assert "source" not in result


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_false():
    config = Config(settings_file="nonexistent_file.cfg", quiet=True)
    assert config.quiet == True


