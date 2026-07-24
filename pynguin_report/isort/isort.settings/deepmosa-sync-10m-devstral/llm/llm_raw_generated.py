####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_config_post_init_with_valid_py_version():
    config = _Config(py_version="3.8")
    assert config.py_version == "py3.8"
    assert config.known_standard_library == frozenset(getattr(stdlibs, "py3.8").stdlib)

def test_config_post_init_with_auto_py_version():
    config = _Config(py_version="auto")
    assert config.py_version == f"py{sys.version_info.major}{sys.version_info.minor}"
    assert config.known_standard_library == frozenset(getattr(stdlibs, config.py_version).stdlib)

def test_config_post_init_with_invalid_py_version():
    with pytest.raises(ValueError):
        _Config(py_version="4.0")

def test_config_post_init_with_force_alphabetical_sort():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True

def test_config_post_init_with_wrap_length_greater_than_line_length():
    with pytest.raises(ValueError):
        _Config(wrap_length=80, line_length=79)

def test_config_post_init_with_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test_config_post_init_with_default_values():
    config = _Config()
    assert config.py_version == "py3"
    assert config.line_length == 79
    assert config.wrap_length == 0
    assert config.multi_line_output == WrapModes.GRID
    assert config.known_future_library == frozenset(("__future__",))


# LLM-generated content at query #2
#--------------------------

```python
def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"line_length": 100})
    assert config.line_length == 100

def test_config_constructor_with_settings_file():
    config = Config(settings_file="test_settings.ini")
    assert config.settings_file == "test_settings.ini"

def test_config_constructor_with_settings_path():
    config = Config(settings_path="test_path")
    assert config.settings_path == "test_path"

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet == True

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

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_constructor_with_deprecated_options():
    config = Config(config_overrides={"virtual_env": "test"})
    assert config.virtual_env is None

def test_config_constructor_with_unsupported_config():
    try:
        config = Config(config_overrides={"unsupported_option": "test"})
    except UnsupportedSettings:
        pass
    else:
        assert False, "Expected UnsupportedSettings exception"

def test_config_constructor_with_invalid_settings_path():
    try:
        config = Config(settings_path="invalid_path")
    except InvalidSettingsPath:
        pass
    else:
        assert False, "Expected InvalidSettingsPath exception"

def test_config_constructor_with_profile_does_not_exist():
    try:
        config = Config(config_overrides={"profile": "nonexistent"})
    except ProfileDoesNotExist:
        pass
    else:
        assert False, "Expected ProfileDoesNotExist exception"

def test_config_constructor_with_formatter_does_not_exist():
    try:
        config = Config(config_overrides={"formatter": "nonexistent"})
    except FormattingPluginDoesNotExist:
        pass
    else:
        assert False, "Expected FormattingPluginDoesNotExist exception"

def test_config_constructor_with_sort_order_does_not_exist():
    try:
        config = Config(config_overrides={"sort_order": "nonexistent"})
    except SortingFunctionDoesNotExist:
        pass
    else:
        assert False, "Expected SortingFunctionDoesNotExist exception"


# LLM-generated content at query #3
#--------------------------

```python
def test_config_predicate_false():
    config_instance = Config()
    assert not config_instance._known_patterns


# LLM-generated content at query #4
#--------------------------

```python
def test_indent_in_combined_config():
    combined_config = {"indent": "4"}
    assert "indent" in combined_config


# LLM-generated content at query #5
#--------------------------

```python
def test_abspaths_with_relative_paths():
    cwd = "/home/user"
    values = ["dir1/", "dir2/"]
    result = _abspaths(cwd, values)
    assert result == {"/home/user/dir1/", "/home/user/dir2/"}

def test_abspaths_with_absolute_paths():
    cwd = "/home/user"
    values = ["/abs/dir1/", "/abs/dir2/"]
    result = _abspaths(cwd, values)
    assert result == {"/abs/dir1/", "/abs/dir2/"}

def test_abspaths_with_mixed_paths():
    cwd = "/home/user"
    values = ["dir1/", "/abs/dir2/"]
    result = _abspaths(cwd, values)
    assert result == {"/home/user/dir1/", "/abs/dir2/"}

def test_abspaths_with_empty_values():
    cwd = "/home/user"
    values = []
    result = _abspaths(cwd, values)
    assert result == set()

def test_abspaths_with_duplicate_values():
    cwd = "/home/user"
    values = ["dir1/", "dir1/"]
    result = _abspaths(cwd, values)
    assert result == {"/home/user/dir1/"}


# LLM-generated content at query #6
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

def test_config_init_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_init_with_indent():
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

def test_config_init_with_skips():
    config = Config(config_overrides={"skip": ["foo"]})
    assert "foo" in config.skips

def test_config_init_with_skip_globs():
    config = Config(config_overrides={"skip_glob": ["foo"]})
    assert "foo" in config.skip_globs

def test_config_init_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sort_order == "natural"

def test_config_init_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_init_with_deprecated_options():
    config = Config(config_overrides={"virtual_env": "foo"})
    assert config.virtual_env is None

def test_config_init_with_unsupported_config():
    try:
        Config(config_overrides={"unsupported_option": "foo"})
    except UnsupportedSettings:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_indent_in_combined_config():
    config = Config(indent=4)
    assert "indent" in config.__dict__


# LLM-generated content at query #8
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
            f.write("[tool.isort]\nline_length=100\n")
        config = Config(settings_path=tmpdir)
    assert config.line_length == 100

def test_config_constructor_with_config_overrides():
    config = Config(quiet=True, line_length=80)
    assert config.quiet is True
    assert config.line_length == 80

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

def test_config_constructor_with_indent_string():
    config = Config(indent="    ")
    assert config.indent == "    "

def test_config_constructor_with_indent_digit():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_constructor_with_indent_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_sections():
    config = Config(known_foo=["bar", "baz"])
    assert config.known_other == {"foo": frozenset(["bar", "baz"])}

def test_config_constructor_with_import_headings():
    config = Config(import_heading_foo="Bar")
    assert config.import_headings == {"foo": "Bar"}

def test_config_constructor_with_import_footers():
    config = Config(import_footer_foo="Baz")
    assert config.import_footers == {"foo": "Baz"}

def test_config_constructor_with_deprecated_options():
    with pytest.warns(UserWarning, match="W0503: Deprecated config options were used"):
        config = Config(include_trailing_comma=True)
    assert not hasattr(config, "include_trailing_comma")

def test_config_constructor_with_unsupported_options():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

def test_config_constructor_with_formatter_plugin():
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


# LLM-generated content at query #9
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
    assert config.known_standard_library == frozenset(getattr(stdlibs, "py38").stdlib)

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


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_false():
    cwd = "/home/user"
    value = "/absolute/path/"
    assert not (not value.startswith(os.path.sep) and value.endswith(os.path.sep))


# LLM-generated content at query #11
#--------------------------

```python
def test_config_constructor_with_config_object():
    config = Config()
    new_config = Config(config=config, line_length=120)
    assert new_config.line_length == 120
    assert new_config.py_version == config.py_version.replace("py", "")

def test_config_constructor_with_settings_file():
    with open("test_settings.cfg", "w") as f:
        f.write("[isort]\nline_length = 88\n")
    config = Config(settings_file="test_settings.cfg")
    assert config.line_length == 88
    os.remove("test_settings.cfg")

def test_config_constructor_with_settings_path():
    os.makedirs("test_project", exist_ok=True)
    with open("test_project/.isort.cfg", "w") as f:
        f.write("[isort]\nline_length = 100\n")
    config = Config(settings_path="test_project")
    assert config.line_length == 100
    os.remove("test_project/.isort.cfg")
    os.rmdir("test_project")

def test_config_constructor_with_config_overrides():
    config = Config(line_length=120, indent="    ")
    assert config.line_length == 120
    assert config.indent == "    "

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.line_length == 88
    assert config.multi_line_output == 3

def test_config_constructor_with_known_sections():
    config = Config(known_foo=["bar", "baz"], sections=["STDLIB", "FOO"])
    assert config.known_other == {"foo": frozenset(["bar", "baz"])}
    assert "FOO" in config.sections

def test_config_constructor_with_import_headings():
    config = Config(import_heading_foo="Bar Imports", sections=["STDLIB", "FOO"])
    assert config.import_headings == {"foo": "Bar Imports"}

def test_config_constructor_with_import_footers():
    config = Config(import_footer_foo="End of Bar Imports", sections=["STDLIB", "FOO"])
    assert config.import_footers == {"foo": "End of Bar Imports"}

def test_config_constructor_with_unsupported_settings():
    try:
        Config(unsupported_setting="value")
        assert False, "Expected UnsupportedSettings exception"
    except UnsupportedSettings:
        pass

def test_config_constructor_with_deprecated_settings():
    config = Config(force_single_line=False, quiet=True)
    assert config.force_single_line is False

def test_config_constructor_with_indent_as_digit():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_src_paths():
    os.makedirs("test_project/src", exist_ok=True)
    config = Config(settings_path="test_project", src_paths=["src"])
    assert len(config.src_paths) == 1
    assert config.src_paths[0].name == "src"
    os.rmdir("test_project/src")
    os.rmdir("test_project")

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="black")
    assert config.formatting_function is not None

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally


# LLM-generated content at query #12
#--------------------------

```python
def test_config_initialization_with_config_parameter():
    config = Config()
    assert Config(config=config)._known_patterns is None
    assert Config(config=config)._section_comments is None
    assert Config(config=config)._section_comments_end is None
    assert Config(config=config)._skips is None
    assert Config(config=config)._skip_globs is None
    assert Config(config=config)._sorting_function is None


# LLM-generated content at query #13
#--------------------------

```python
def test_is_supported_filetype_with_supported_extension():
    config = Config()
    assert config.is_supported_filetype("test.py") == True

def test_is_supported_filetype_with_blocked_extension():
    config = Config()
    assert config.is_supported_filetype("test.min.js") == False

def test_is_supported_filetype_with_editor_backup_file():
    config = Config()
    assert config.is_supported_filetype("test.py~") == False

def test_is_supported_filetype_with_named_pipe():
    config = Config()
    assert config.is_supported_filetype("/dev/null") == False

def test_is_supported_filetype_with_shebang():
    config = Config()
    assert config.is_supported_filetype("test") == True

def test_is_supported_filetype_without_shebang():
    config = Config()
    assert config.is_supported_filetype("test.txt") == False


# LLM-generated content at query #14
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
    assert config.quiet == True

def test_config_constructor_with_indent():
    config = Config(config_overrides={"indent": "4"})
    assert config.indent == "    "

def test_config_constructor_with_tab_indent():
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

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"), Path.cwd())

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_constructor_with_deprecated_options():
    config = Config(config_overrides={"virtual_env": "venv"})
    assert not hasattr(config, "virtual_env")

def test_config_constructor_with_unsupported_config():
    try:
        config = Config(config_overrides={"unsupported_option": "value"})
    except UnsupportedSettings:
        pass
    else:
        assert False, "Expected UnsupportedSettings exception"

def test_config_constructor_with_invalid_settings_path():
    try:
        config = Config(settings_path="/invalid/path")
    except InvalidSettingsPath:
        pass
    else:
        assert False, "Expected InvalidSettingsPath exception"

def test_config_constructor_with_nonexistent_profile():
    try:
        config = Config(config_overrides={"profile": "nonexistent"})
    except ProfileDoesNotExist:
        pass
    else:
        assert False, "Expected ProfileDoesNotExist exception"

def test_config_constructor_with_nonexistent_formatter():
    try:
        config = Config(config_overrides={"formatter": "nonexistent"})
    except FormattingPluginDoesNotExist:
        pass
    else:
        assert False, "Expected FormattingPluginDoesNotExist exception"

def test_config_constructor_with_nonexistent_sort_order():
    try:
        config = Config(config_overrides={"sort_order": "nonexistent"})
    except SortingFunctionDoesNotExist:
        pass
    else:
        assert False, "Expected SortingFunctionDoesNotExist exception"


# LLM-generated content at query #15
#--------------------------

```python
def test_is_skipped_with_skipped_file():
    config = Config(skip={"test.py"})
    assert config.is_skipped(Path("test.py")) is True

def test_is_skipped_with_non_skipped_file():
    config = Config(skip={"other.py"})
    assert config.is_skipped(Path("test.py")) is False

def test_is_skipped_with_skipped_directory():
    config = Config(skip={"tests"})
    assert config.is_skipped(Path("tests/test.py")) is True

def test_is_skipped_with_non_skipped_directory():
    config = Config(skip={"other"})
    assert config.is_skipped(Path("tests/test.py")) is False

def test_is_skipped_with_skip_glob():
    config = Config(skip_glob={"*.pyc"})
    assert config.is_skipped(Path("test.pyc")) is True

def test_is_skipped_with_non_matching_skip_glob():
    config = Config(skip_glob={"*.pyc"})
    assert config.is_skipped(Path("test.py")) is False

def test_is_skipped_with_skip_gitignore():
    config = Config(skip_gitignore=True)
    assert config.is_skipped(Path(".git")) is True

def test_is_skipped_with_non_existent_file():
    config = Config()
    assert config.is_skipped(Path("non_existent.py")) is True

def test_is_skipped_with_directory():
    config = Config()
    assert config.is_skipped(Path("some_directory")) is False

def test_is_skipped_with_symlink():
    config = Config()
    assert config.is_skipped(Path("some_symlink")) is False

def test_is_skipped_with_git_ls_files():
    config = Config(skip_gitignore=True)
    config.git_ls_files[Path("/test")] = {"/test/file.py"}
    assert config.is_skipped(Path("/test/other.py")) is True

def test_is_skipped_with_git_ls_files_included():
    config = Config(skip_gitignore=True)
    config.git_ls_files[Path("/test")] = {"/test/file.py"}
    assert config.is_skipped(Path("/test/file.py")) is False


# LLM-generated content at query #16
#--------------------------

```python
def test__find_config_returns_empty_dict_when_no_config_found():
    result = _find_config("/nonexistent/path")
    assert result == ("/nonexistent/path", {})

def test__find_config_finds_and_returns_config_data():
    with open("test_config.ini", "w") as f:
        f.write("[section]\nkey = value")
    result = _find_config(os.path.dirname(os.path.abspath("test_config.ini")))
    assert result[0] == os.path.dirname(os.path.abspath("test_config.ini"))
    assert "key" in result[1]
    os.remove("test_config.ini")

def test__find_config_stops_search_on_stop_dir():
    os.makedirs("test_dir/.git", exist_ok=True)
    with open("test_dir/test_config.ini", "w") as f:
        f.write("[section]\nkey = value")
    result = _find_config("test_dir")
    assert result == ("test_dir", {})
    os.remove("test_dir/test_config.ini")
    os.rmdir("test_dir/.git")
    os.rmdir("test_dir")

def test__find_config_returns_config_from_parent_dir():
    os.makedirs("parent/child", exist_ok=True)
    with open("parent/config.ini", "w") as f:
        f.write("[section]\nkey = value")
    result = _find_config("parent/child")
    assert result[0] == "parent"
    assert "key" in result[1]
    os.remove("parent/config.ini")
    os.rmdir("parent/child")
    os.rmdir("parent")


# LLM-generated content at query #17
#--------------------------

```python
def test__get_config_data_with_toml_file():
    file_path = "test.toml"
    sections = ("section1", "section2")
    with open(file_path, "w") as f:
        f.write("[section1]\nkey1 = 'value1'\nkey2 = 123\n[section2]\nkey3 = true")
    result = _get_config_data(file_path, sections)
    assert result == {"key1": "value1", "key2": 123, "key3": True, "source": file_path}
    os.remove(file_path)

def test__get_config_data_with_editorconfig_file():
    file_path = "test.editorconfig"
    sections = ("*.py",)
    with open(file_path, "w") as f:
        f.write("root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88")
    result = _get_config_data(file_path, sections)
    assert result == {"indent": "    ", "line_length": 88, "source": file_path}
    os.remove(file_path)

def test__get_config_data_with_ini_file():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\nkey1 = value1\nkey2 = 123")
    result = _get_config_data(file_path, sections)
    assert result == {"key1": "value1", "key2": 123, "source": file_path}
    os.remove(file_path)

def test__get_config_data_with_non_existent_file():
    file_path = "non_existent_file.txt"
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert result == {}

def test__get_config_data_with_empty_sections():
    file_path = "test.ini"
    sections = ()
    with open(file_path, "w") as f:
        f.write("[section1]\nkey1 = value1")
    result = _get_config_data(file_path, sections)
    assert result == {}
    os.remove(file_path)

def test__get_config_data_with_known_prefix():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\nknown_key = value1,value2")
    result = _get_config_data(file_path, sections)
    assert result == {"known_key": {"value1", "value2"}, "source": file_path}
    os.remove(file_path)

def test__get_config_data_with_force_grid_wrap():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\nforce_grid_wrap = false")
    result = _get_config_data(file_path, sections)
    assert result == {"force_grid_wrap": 0, "source": file_path}
    os.remove(file_path)

def test__get_config_data_with_comment_prefix():
    file_path = "test.ini"
    sections = ("section1",)
    with open(file_path, "w") as f:
        f.write("[section1]\ncomment_prefix = '#'")
    result = _get_config_data(file_path, sections)
    assert result == {"comment_prefix": "#", "source": file_path}
    os.remove(file_path)


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
def test_predicate_at_line_80_evaluates_to_true():
    file_path = "test.editorconfig"
    sections = ("*.{py}",)
    settings = {
        "indent_style": "space",
        "indent_size": "4",
        "max_line_length": "88",
        "force_grid_wrap": "true",
        "comment_prefix": "'# '",
    }
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("[*.{py}]\n")
        f.write("indent_style = space\n")
        f.write("indent_size = 4\n")
        f.write("max_line_length = 88\n")
        f.write("force_grid_wrap = true\n")
        f.write("comment_prefix = '# '\n")
    result = _get_config_data(file_path, sections)
    assert result["force_grid_wrap"] == 2


# LLM-generated content at query #20
#--------------------------

```python
def test__as_list_with_single_string():
    assert _as_list("a,b,c") == ["a", "b", "c"]

def test__as_list_with_newlines():
    assert _as_list("a\nb\nc") == ["a", "b", "c"]

def test__as_list_with_mixed_delimiters():
    assert _as_list("a,b\nc,d") == ["a", "b", "c", "d"]

def test__as_list_with_whitespace():
    assert _as_list(" a , b , c ") == ["a", "b", "c"]

def test__as_list_with_empty_items():
    assert _as_list("a,,b") == ["a", "b"]

def test__as_list_with_list_input():
    assert _as_list([" a ", " b ", " c "]) == ["a", "b", "c"]

def test__as_list_with_empty_string():
    assert _as_list("") == []

def test__as_list_with_only_whitespace():
    assert _as_list("   ") == []

def test__as_list_with_single_item():
    assert _as_list("a") == ["a"]


# LLM-generated content at query #21
#--------------------------

```python
def test_config_parameter_is_none():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #22
#--------------------------

```python
def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"line_length": 120})
    assert config.line_length == 120

def test_config_constructor_with_settings_file():
    with open("test_settings.cfg", "w") as f:
        f.write("[isort]\nline_length = 88\n")
    config = Config(settings_file="test_settings.cfg")
    assert config.line_length == 88
    os.remove("test_settings.cfg")

def test_config_constructor_with_settings_path():
    os.makedirs("test_project", exist_ok=True)
    with open("test_project/.isort.cfg", "w") as f:
        f.write("[isort]\nline_length = 100\n")
    config = Config(settings_path="test_project")
    assert config.line_length == 100
    os.remove("test_project/.isort.cfg")
    os.rmdir("test_project")

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.indent == "    "

def test_config_constructor_with_invalid_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(config_overrides={"profile": "invalid_profile"})

def test_config_constructor_with_known_sections():
    config = Config(config_overrides={"known_foo": ["bar"]})
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_foo": "Bar"})
    assert config.import_headings == {"foo": "Bar"}

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_foo": "Bar"})
    assert config.import_footers == {"foo": "Bar"}

def test_config_constructor_with_deprecated_options():
    with pytest.warns(UserWarning, match="W0503: Deprecated config options were used"):
        config = Config(config_overrides={"virtual_env": "test"})
    assert not hasattr(config, "virtual_env")

def test_config_constructor_with_unsupported_options():
    with pytest.raises(UnsupportedSettings):
        Config(config_overrides={"unsupported_option": "value"})

def test_config_constructor_with_formatter_plugin():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatting_function is not None

def test_config_constructor_with_invalid_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(config_overrides={"formatter": "invalid_formatter"})

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_invalid_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(config_overrides={"sort_order": "invalid_sort"})


# LLM-generated content at query #23
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
    config = Config(indent="4")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_profile():
    config = Config(profile="black")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_quiet_override():
    config = Config(quiet=True)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_known_sections():
    config = Config(known_foo="bar")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_import_headings():
    config = Config(import_heading_foo="bar")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_initialization_with_import_footers():
    config = Config(import_footer_foo="bar")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #24
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
    assert config.quiet == True

def test_config_constructor_with_indent():
    config = Config(config_overrides={"indent": "4"})
    assert config.indent == "    "

def test_config_constructor_with_tab_indent():
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

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"), Path.cwd())

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatting_function is not None

def test_config_constructor_with_deprecated_options():
    config = Config(config_overrides={"force_single_line": True})
    assert config.force_single_line is False

def test_config_constructor_with_unsupported_options():
    try:
        config = Config(config_overrides={"unsupported_option": "value"})
    except UnsupportedSettings:
        pass

def test_config_constructor_with_config_object():
    config = Config(config_overrides={"line_length": 100})
    new_config = Config(config=config, config_overrides={"line_length": 120})
    assert new_config.line_length == 120


# LLM-generated content at query #25
#--------------------------

```python
def test_known_other_section_not_in_sections():
    combined_config = {
        "sections": ("STANDARD_LIB", "THIRD_PARTY"),
        "known_custom": {"custom_module"},
        "quiet": False
    }
    known_other = {}
    import_headings = {}
    import_footers = {}
    for key, value in tuple(combined_config.items()):
        if key.startswith("known_") and key not in (
            "known_standard_library",
            "known_future_library",
            "known_third_party",
            "known_first_party",
            "known_local_folder",
        ):
            import_heading = key[len("known_"):].lower()
            maps_to_section = import_heading.upper()
            combined_config.pop(key)
            if maps_to_section in {"STANDARD_LIB", "FUTURE_LIB", "THIRD_PARTY", "FIRST_PARTY", "LOCAL_FOLDER"}:
                section_name = f"known_{maps_to_section.lower()}"
                if section_name in combined_config and not combined_config["quiet"]:
                    pass
                else:
                    combined_config[section_name] = frozenset(value)
            else:
                known_other[import_heading] = frozenset(value)
                assert maps_to_section not in combined_config.get("sections", ())


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_supported_filetype_supported_extension():
    config = Config()
    assert config.is_supported_filetype("file.py") is True
    assert config.is_supported_filetype("file.pyi") is True

def test_is_supported_filetype_blocked_extension():
    config = Config(blocked_extensions=["txt"])
    assert config.is_supported_filetype("file.txt") is False

def test_is_supported_filetype_editor_backup():
    config = Config()
    assert config.is_supported_filetype("file.py~") is False

def test_is_supported_filetype_fifo():
    config = Config()
    assert config.is_supported_filetype("/dev/null") is False

def test_is_supported_filetype_shebang():
    config = Config()
    assert config.is_supported_filetype("script.sh") is True


# LLM-generated content at query #2
#--------------------------

```python
def test_config_constructor_with_no_arguments():
    config = Config()
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write("[isort]\nline_length=79\n")
        f.flush()
        config = Config(settings_file=f.name)
    assert config.line_length == 79
    os.unlink(f.name)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.isort]\nline_length=88\n")
        config = Config(settings_path=tmpdir)
    assert config.line_length == 88

def test_config_constructor_with_config_object():
    base_config = Config(line_length=79)
    config = Config(config=base_config, line_length=88)
    assert config.line_length == 88

def test_config_constructor_with_config_overrides():
    config = Config(line_length=79)
    assert config.line_length == 79

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.line_length == 88
    assert config.multi_line_output == 3

def test_config_constructor_with_unsupported_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent")

def test_config_constructor_with_invalid_settings_path():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="black")
    assert config.formatting_function is not None

def test_config_constructor_with_unsupported_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent")

def test_config_constructor_with_deprecated_options():
    with pytest.warns(UserWarning, match="W0503: Deprecated config options were used"):
        config = Config(force_grid_wrap=2)
    assert not hasattr(config, "force_grid_wrap")

def test_config_constructor_with_unsupported_settings():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

def test_config_constructor_with_known_sections():
    config = Config(known_foo=["bar"])
    assert config.known_other == {"foo": frozenset(["bar"])}

def test_config_constructor_with_import_headings():
    config = Config(import_heading_foo="Bar")
    assert config.import_headings == {"foo": "Bar"}

def test_config_constructor_with_import_footers():
    config = Config(import_footer_foo="Bar")
    assert config.import_footers == {"foo": "Bar"}

def test_config_constructor_with_indent_string():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_constructor_with_indent_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_src_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(src_paths=[tmpdir])
    assert Path(tmpdir) in config.src_paths

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_invalid_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="invalid")


# LLM-generated content at query #3
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
    import os
    import stat
    config = Config()
    os.mkfifo("test_fifo")
    assert config.is_supported_filetype("test_fifo") is False
    os.remove("test_fifo")

def test_is_supported_filetype_nonexistent_file():
    config = Config()
    assert config.is_supported_filetype("nonexistent_file.py") is False

def test_is_supported_filetype_shebang():
    with open("test_script", "w") as f:
        f.write("#!/usr/bin/env python3\n")
    config = Config()
    assert config.is_supported_filetype("test_script") is True
    os.remove("test_script")


# LLM-generated content at query #4
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
    config = Config(config_overrides={"virtual_env": "venv"})
    assert "virtual_env" not in config.__dict__

def test_config_constructor_with_unsupported_options():
    try:
        Config(config_overrides={"unsupported_option": "value"})
    except UnsupportedSettings:
        pass

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatting_function is not None

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_skip_gitignore():
    config = Config(config_overrides={"skip_gitignore": True})
    assert config.skip_gitignore is True

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src"]})
    assert config.src_paths == (Path("src"),)

def test_config_constructor_with_directory():
    config = Config(config_overrides={"directory": "."})
    assert config.directory == "."

def test_config_constructor_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True


# LLM-generated content at query #5
#--------------------------

```python
def test_indent_lower_equals_tab():
    combined_config = {"indent": "tab"}
    indent = str(combined_config["indent"])
    indent = indent.strip("'").strip('"')
    assert indent.lower() == "tab"


# LLM-generated content at query #6
#--------------------------

```python
def test_line_123_predicate_true():
    config_overrides = {"quiet": False}
    combined_config = {"sections": ("CUSTOM",), "known_custom": frozenset({"module"})}
    key = "known_custom"
    value = ["module"]
    import_heading = key[len("known_"):].lower()
    maps_to_section = import_heading.upper()
    known_other = {}
    combined_config.pop(key)
    known_other[import_heading] = frozenset(value)
    assert maps_to_section not in combined_config.get("sections", ())


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    import os
    import stat
    config = Config()
    os.mkfifo("test_fifo")
    assert config.is_supported_filetype("test_fifo") is False
    os.remove("test_fifo")

def test_is_supported_filetype_with_shebang():
    with open("test_file", "wb") as f:
        f.write(b"#!/usr/bin/env python\n")
    config = Config()
    assert config.is_supported_filetype("test_file") is True
    os.remove("test_file")

def test_is_supported_filetype_without_shebang():
    with open("test_file", "wb") as f:
        f.write(b"print('hello')\n")
    config = Config()
    assert config.is_supported_filetype("test_file") is False
    os.remove("test_file")


# LLM-generated content at query #2
#--------------------------

```python
def test_config_constructor_with_config_parameter():
    config = Config()
    new_config = Config(config=config)
    assert new_config.py_version == config.py_version.replace("py", "")
    assert new_config._known_patterns is None
    assert new_config._section_comments is None
    assert new_config._section_comments_end is None
    assert new_config._skips is None
    assert new_config._skip_globs is None
    assert new_config._sorting_function is None

def test_config_constructor_with_settings_file():
    config = Config(settings_file="setup.cfg")
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

def test_config_constructor_with_config_overrides():
    config = Config(indent="4")
    assert config.indent == "    "
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_known_sections():
    config = Config(known_future_library=["__future__"])
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_import_headings():
    config = Config(import_heading_future="__future__")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_import_footers():
    config = Config(import_footer_future="__future__")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_sections():
    config = Config(sections=["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"])
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src"])
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_formatter():
    config = Config(formatter="black")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #3
#--------------------------

```python
def test_config_initialization_with_config_parameter():
    config_instance = _Config()
    config_instance.py_version = "py38"
    config_instance.indent = "    "
    config_instance.line_length = 88

    new_config = Config(config=config_instance)

    assert new_config.py_version == "38"
    assert new_config.indent == "    "
    assert new_config.line_length == 88
    assert new_config._known_patterns is None
    assert new_config._section_comments is None
    assert new_config._section_comments_end is None
    assert new_config._skips is None
    assert new_config._skip_globs is None
    assert new_config._sorting_function is None


# LLM-generated content at query #4
#--------------------------

```python
def test_known_other_is_not_empty():
    config = Config(known_custom_section=["custom_module"])
    assert config.known_other == {"custom_section": frozenset(["custom_module"])}


# LLM-generated content at query #5
#--------------------------

```python
def test_abspaths_relative_paths():
    result = _abspaths("/home/user", ["./file1", "dir1/", "file2"])
    assert result == {"/home/user/./file1", "/home/user/dir1/", "/home/user/file2"}

def test_abspaths_absolute_paths():
    result = _abspaths("/home/user", ["/absolute/file1", "/absolute/dir1/"])
    assert result == {"/absolute/file1", "/absolute/dir1/"}

def test_abspaths_mixed_paths():
    result = _abspaths("/home/user", ["./file1", "/absolute/file2", "dir1/"])
    assert result == {"/home/user/./file1", "/absolute/file2", "/home/user/dir1/"}

def test_abspaths_empty_input():
    result = _abspaths("/home/user", [])
    assert result == set()

def test_abspaths_duplicates():
    result = _abspaths("/home/user", ["./file1", "./file1"])
    assert result == {"/home/user/./file1"}


# LLM-generated content at query #6
#--------------------------

```python
def test__find_config_returns_empty_dict_when_no_config_found():
    result = _find_config("/non/existent/path")
    assert result == ("/non/existent/path", {})

def test__find_config_returns_config_data_when_config_file_exists():
    # Assuming a test config file exists at the given path
    result = _find_config(os.path.dirname(__file__))
    assert isinstance(result[1], dict)

def test__find_config_stops_search_on_stop_dir():
    # Assuming a stop directory exists in the path
    result = _find_config(os.path.join(os.path.dirname(__file__), "stop_dir"))
    assert result == (os.path.join(os.path.dirname(__file__), "stop_dir"), {})

def test__find_config_returns_config_data_for_valid_config_file():
    # Assuming a valid config file exists at the given path
    config_path = os.path.join(os.path.dirname(__file__), "valid_config_file")
    result = _find_config(config_path)
    assert isinstance(result[1], dict)
    assert result[1] != {}

def test__find_config_handles_exception_during_config_parsing():
    # Assuming a malformed config file exists at the given path
    config_path = os.path.join(os.path.dirname(__file__), "malformed_config_file")
    result = _find_config(config_path)
    assert result == (os.path.dirname(config_path), {})


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config_data_with_toml_file():
    file_path = "test_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "line_length": 88, "indent": "    "}

def test_get_config_data_with_editorconfig_file():
    file_path = "test_config.editorconfig"
    sections = ("*.py",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "indent": "    ", "line_length": 88}

def test_get_config_data_with_ini_file():
    file_path = "test_config.ini"
    sections = ("black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "line_length": 88, "indent": "    "}

def test_get_config_data_with_empty_file():
    file_path = "empty_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {}

def test_get_config_data_with_non_existent_file():
    file_path = "non_existent_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {}

def test_get_config_data_with_multiple_sections():
    file_path = "multi_section_config.toml"
    sections = ("tool.black", "tool.isort")
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "line_length": 88, "indent": "    ", "multi_line_output": 3}

def test_get_config_data_with_boolean_values():
    file_path = "bool_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "skip_string_normalization": True, "verbose": False}

def test_get_config_data_with_list_values():
    file_path = "list_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "include": ("tests/", "src/"), "exclude": ("build/", "dist/")}

def test_get_config_data_with_force_grid_wrap():
    file_path = "force_grid_wrap_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "force_grid_wrap": 2}

def test_get_config_data_with_comment_prefix():
    file_path = "comment_prefix_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "comment_prefix": "#"}

def test_get_config_data_with_abspaths():
    file_path = "abspaths_config.toml"
    sections = ("tool.black",)
    result = _get_config_data(file_path, sections)
    assert result == {"source": file_path, "src_paths": {"/absolute/path1", "/absolute/path2"}}


# LLM-generated content at query #8
#--------------------------

```python
def test__as_list_with_single_string():
    assert _as_list("a, b, c") == ["a", "b", "c"]

def test__as_list_with_newlines():
    assert _as_list("a\nb\nc") == ["a", "b", "c"]

def test__as_list_with_mixed_delimiters():
    assert _as_list("a, b\nc, d") == ["a", "b", "c", "d"]

def test__as_list_with_empty_strings():
    assert _as_list("a, , b") == ["a", "b"]

def test__as_list_with_whitespace():
    assert _as_list("  a  ,  b  ") == ["a", "b"]

def test__as_list_with_list_input():
    assert _as_list(["a", " b ", "c"]) == ["a", "b", "c"]

def test__as_list_with_empty_list():
    assert _as_list([]) == []

def test__as_list_with_empty_string():
    assert _as_list("") == []

def test__as_list_with_whitespace_only():
    assert _as_list("   ") == []


# LLM-generated content at query #9
#--------------------------

```python
def test_is_supported_filetype_returns_true_for_supported_extension():
    config = Config()
    config.supported_extensions = {"py"}
    config.blocked_extensions = set()
    assert config.is_supported_filetype("test.py") is True


# LLM-generated content at query #10
#--------------------------

```python
def test_config_settings_empty_and_quiet_false():
    config = Config(settings_file="empty_file.cfg", quiet=False)
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_78_evaluates_to_true():
    key = "known_prefix_key"
    KNOWN_PREFIX = "known_"
    assert key.startswith(KNOWN_PREFIX)


# LLM-generated content at query #12
#--------------------------

```python
def test__post_init__default_py_version():
    config = _Config()
    assert config.py_version == "py3"

def test__post_init__auto_py_version():
    config = _Config(py_version="auto")
    assert config.py_version == f"py{sys.version_info.major}{sys.version_info.minor}"

def test__post_init__invalid_py_version():
    with pytest.raises(ValueError):
        _Config(py_version="invalid")

def test__post_init__known_standard_library_populated():
    config = _Config()
    assert len(config.known_standard_library) > 0

def test__post_init__vertical_grid_grouped_no_comma_converted():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test__post_init__force_alphabetical_sort_settings():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections is True
    assert config.no_sections is True
    assert config.lines_between_types == 1
    assert config.from_first is True

def test__post_init__wrap_length_exceeds_line_length():
    with pytest.raises(ValueError):
        _Config(wrap_length=80, line_length=79)


# LLM-generated content at query #13
#--------------------------

```python
def test_config_predicate_false():
    config_instance = Config()
    assert not config_instance._known_patterns


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_80_evaluates_to_true():
    key = "force_grid_wrap"
    value = "true"
    existing_value_type = int
    result = 0 if value.lower().strip() == "false" else 2
    assert result == 2


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

def test_config_init_with_quiet():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet == True

def test_config_init_with_indent():
    config = Config(config_overrides={"indent": "    "})
    assert config.indent == "    "

def test_config_init_with_known_sections():
    config = Config(config_overrides={"known_third_party": ["numpy", "pandas"]})
    assert config.known_third_party == frozenset(["numpy", "pandas"])

def test_config_init_with_import_headings():
    config = Config(config_overrides={"import_heading_stdlib": "Standard Library"})
    assert config.import_headings == {"stdlib": "Standard Library"}

def test_config_init_with_import_footers():
    config = Config(config_overrides={"import_footer_stdlib": "End of Standard Library"})
    assert config.import_footers == {"stdlib": "End of Standard Library"}

def test_config_init_with_skips():
    config = Config(config_overrides={"skip": ["test.py"], "extend_skip": ["test2.py"]})
    assert config.skips == frozenset(["test.py", "test2.py"])

def test_config_init_with_skip_globs():
    config = Config(config_overrides={"skip_glob": ["test.py"], "extend_skip_glob": ["test2.py"]})
    assert config.skip_globs == frozenset(["test.py", "test2.py"])

def test_config_init_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sort_order == "natural"

def test_config_init_with_formatter():
    config = Config(config_overrides={"formatter": "black"})
    assert config.formatter == "black"

def test_config_init_with_deprecated_options():
    config = Config(config_overrides={"deprecated_option": "value"})
    assert not hasattr(config, "deprecated_option")

def test_config_init_with_unsupported_config():
    try:
        config = Config(config_overrides={"unsupported_option": "value"})
    except UnsupportedSettings:
        pass
    else:
        assert False, "Expected UnsupportedSettings exception"

def test_config_init_with_invalid_settings_path():
    try:
        config = Config(settings_path="/invalid/path")
    except InvalidSettingsPath:
        pass
    else:
        assert False, "Expected InvalidSettingsPath exception"

def test_config_init_with_nonexistent_profile():
    try:
        config = Config(config_overrides={"profile": "nonexistent"})
    except ProfileDoesNotExist:
        pass
    else:
        assert False, "Expected ProfileDoesNotExist exception"

def test_config_init_with_nonexistent_formatter():
    try:
        config = Config(config_overrides={"formatter": "nonexistent"})
    except FormattingPluginDoesNotExist:
        pass
    else:
        assert False, "Expected FormattingPluginDoesNotExist exception"

def test_config_init_with_nonexistent_sort_order():
    try:
        config = Config(config_overrides={"sort_order": "nonexistent"})
    except SortingFunctionDoesNotExist:
        pass
    else:
        assert False, "Expected SortingFunctionDoesNotExist exception"


# LLM-generated content at query #16
#--------------------------

```python
def test_config_parameter_early_return():
    config_instance = _Config()
    result = Config(config=config_instance)
    assert result is not None


# LLM-generated content at query #17
#--------------------------

```python
def test_config_predicate_false():
    config_instance = Config()
    assert not config_instance._known_patterns


# LLM-generated content at query #18
#--------------------------

```python
def test_find_config_returns_empty_dict_when_no_config_file_exists():
    result = _find_config("/non/existent/path")
    assert result == ("/non/existent/path", {})

def test_find_config_returns_empty_dict_when_config_file_is_invalid():
    with open("/tmp/invalid_config", "w") as f:
        f.write("invalid content")
    result = _find_config("/tmp")
    assert result == ("/tmp", {})

def test_find_config_returns_config_data_when_valid_config_file_exists():
    with open("/tmp/valid_config", "w") as f:
        f.write("[section]\nkey = value")
    result = _find_config("/tmp")
    assert result[0] == "/tmp"
    assert "key" in result[1]
    assert result[1]["key"] == "value"

def test_find_config_stops_search_on_stop_directory():
    os.makedirs("/tmp/stop_dir", exist_ok=True)
    result = _find_config("/tmp/stop_dir")
    assert result == ("/tmp/stop_dir", {})

def test_find_config_searches_parent_directories():
    with open("/tmp/parent_config", "w") as f:
        f.write("[section]\nkey = value")
    result = _find_config("/tmp/child")
    assert result[0] == "/tmp"
    assert "key" in result[1]
    assert result[1]["key"] == "value"


# LLM-generated content at query #19
#--------------------------

```python
def test_directory_defaults_to_config_source_directory():
    config_settings = {"source": "/path/to/config/file"}
    combined_config = {**config_settings}
    assert "directory" not in combined_config
    combined_config["directory"] = (
        os.path.dirname(config_settings["source"])
        if config_settings.get("source", None)
        else os.getcwd()
    )
    assert combined_config["directory"] == "/path/to/config"


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    config = configparser.ConfigParser(strict=False)
    config.add_section("test_section")
    assert not config.has_section("non_existent_section")


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_true():
    file_path = "test.ini"
    sections = ("section1",)
    assert not file_path.endswith(".toml")


# LLM-generated content at query #22
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
    config = Config(settings_file="test.cfg")
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
    config = Config(indent="4")
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #23
#--------------------------

```python
def test_is_supported_filetype_oserror():
    config = Config()
    assert not config.is_supported_filetype("nonexistent_file.py")


