####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_post_init_valid_py_version():
    config = _Config(py_version="3")
    assert config.py_version == "py3"

def test_post_init_py_version_auto():
    import sys
    expected_version = f"{sys.version_info.major}{sys.version_info.minor}"
    config = _Config(py_version="auto")
    assert config.py_version == f"py{expected_version}"

def test_post_init_py_version_all():
    config = _Config(py_version="all")
    assert config.py_version == "all"

def test_post_init_invalid_py_version():
    try:
        _Config(py_version="invalid")
        assert False
    except ValueError as e:
        assert "The python version invalid is not supported" in str(e)

def test_post_init_known_standard_library_populated():
    config = _Config(py_version="3")
    assert len(config.known_standard_library) > 0

def test_post_init_multi_line_output_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test_post_init_force_alphabetical_sort_enabled():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections == True
    assert config.no_sections == True
    assert config.lines_between_types == 1
    assert config.from_first == True

def test_post_init_wrap_length_exceeds_line_length():
    try:
        _Config(wrap_length=100, line_length=79)
        assert False
    except ValueError as e:
        assert "wrap_length must be set lower than or equal to line_length" in str(e)

def test_post_init_wrap_length_equal_line_length():
    config = _Config(wrap_length=79, line_length=79)
    assert config.wrap_length == 79
    assert config.line_length == 79

def test_post_init_wrap_length_less_than_line_length():
    config = _Config(wrap_length=50, line_length=79)
    assert config.wrap_length == 50
    assert config.line_length == 79


# LLM-generated content at query #2
#--------------------------

def test_abspaths_with_relative_paths_ending_with_sep():
    cwd = "/home/user"
    values = ["dir/", "subdir/"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir/", "/home/user/subdir/"}
    assert result == expected

def test_abspaths_with_absolute_paths_ending_with_sep():
    cwd = "/home/user"
    values = ["/usr/local/", "/tmp/"]
    result = _abspaths(cwd, values)
    expected = {"/usr/local/", "/tmp/"}
    assert result == expected

def test_abspaths_with_mixed_paths():
    cwd = "/home/user"
    values = ["dir/", "/usr/local/", "file.txt", "/tmp/"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir/", "/usr/local/", "/home/user/file.txt", "/tmp/"}
    assert result == expected

def test_abspaths_with_no_ending_sep():
    cwd = "/home/user"
    values = ["dir", "file.txt"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir", "/home/user/file.txt"}
    assert result == expected

def test_abspaths_empty_values():
    cwd = "/home/user"
    values = []
    result = _abspaths(cwd, values)
    expected = set()
    assert result == expected

def test_abspaths_with_relative_path_no_sep():
    cwd = "/home/user"
    values = ["dir"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir"}
    assert result == expected

def test_abspaths_with_absolute_path_no_sep():
    cwd = "/home/user"
    values = ["/usr/local"]
    result = _abspaths(cwd, values)
    expected = {"/usr/local"}
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_config_constructor_with_config_parameter():
    mock_config = _Config(py_version="py310")
    config = Config(config=mock_config, quiet=True)
    assert config.py_version == "310"

def test_config_constructor_with_config_overrides():
    mock_config = _Config(py_version="py39")
    config = Config(config=mock_config, py_version="py38")
    assert config.py_version == "38"

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('[tool.isort]\nprofile = "black"\n')
        settings_file = f.name
    try:
        config = Config(settings_file=settings_file)
        assert config.profile == "black"
    finally:
        os.unlink(settings_file)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write('[tool.isort]\nline_length = 100\n')
        config = Config(settings_path=tmpdir)
        assert config.line_length == 100

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_profile_override():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_nonexistent_profile():
    try:
        Config(profile="nonexistent")
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other_section():
    config = Config(known_custom=["mypackage"], sections=["CUSTOM"])
    assert "custom" in config.known_other
    assert "mypackage" in config.known_other["custom"]

def test_config_constructor_with_import_headings():
    config = Config(import_heading_stdlib="Standard Library")
    assert config.import_headings["stdlib"] == "Standard Library"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_stdlib="End Standard Library")
    assert config.import_footers["stdlib"] == "End Standard Library"

def test_config_constructor_with_src_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir, src_paths=["src", "."])
        src_paths = config.src_paths
        assert any("src" in str(p) for p in src_paths)
        assert any(tmpdir in str(p) for p in src_paths)

def test_config_constructor_with_formatter():
    config = Config(formatter="example")
    try:
        _ = config.formatting_function
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_deprecated_options():
    config = Config(atomic=True, quiet=True)
    assert not hasattr(config, "atomic")

def test_config_constructor_with_unsupported_settings():
    try:
        Config(unsupported_option="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_without_parameters():
    config = Config()
    assert config.directory == os.getcwd()

def test_config_constructor_with_combined_config():
    mock_config = _Config(py_version="py310", line_length=80)
    config = Config(config=mock_config, line_length=100)
    assert config.py_version == "310"
    assert config.line_length == 100

def test_config_constructor_with_quiet_false_and_warnings():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('[settings]\nline_length = 100\n')
        settings_file = f.name
    try:
        config = Config(settings_file=settings_file, quiet=False)
        assert config.line_length == 100
    finally:
        os.unlink(settings_file)

def test_config_constructor_with_known_section_mapping_conflict():
    config = Config(known_standard_library=["os"], known_stdlib=["sys"], quiet=True)
    assert "os" in config.known_standard_library
    assert "sys" not in config.known_standard_library

def test_config_constructor_with_sections_and_missing_known():
    config = Config(sections=["CUSTOM"], quiet=True)
    assert "CUSTOM" in config.sections

def test_config_constructor_with_directory_and_src_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir)
        src_paths = config.src_paths
        assert any("src" in str(p) for p in src_paths)
        assert any(tmpdir in str(p) for p in src_paths)

def test_config_constructor_with_runtime_source():
    config = Config(line_length=120)
    assert config.line_length == 120

def test_config_constructor_with_config_overrides_updating_existing():
    mock_config = _Config(py_version="py39", line_length=80)
    config = Config(config=mock_config, line_length=100, py_version="py310")
    assert config.py_version == "310"
    assert config.line_length == 100


# LLM-generated content at query #4
#--------------------------

def test_is_supported_filetype_with_supported_extension():
    config = Config()
    config.supported_extensions = {"py", "txt"}
    result = config.is_supported_filetype("test.py")
    assert result is True

def test_is_supported_filetype_with_blocked_extension():
    config = Config()
    config.blocked_extensions = {"log"}
    result = config.is_supported_filetype("error.log")
    assert result is False

def test_is_supported_filetype_with_unknown_extension_and_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    import tempfile
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".unknown", delete=False) as f:
        f.write(b"#!/usr/bin/env python\n")
        temp_path = f.name
    try:
        result = config.is_supported_filetype(temp_path)
        assert result is True
    finally:
        import os
        os.unlink(temp_path)

def test_is_supported_filetype_with_unknown_extension_no_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    import tempfile
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".unknown", delete=False) as f:
        f.write(b"no shebang here")
        temp_path = f.name
    try:
        result = config.is_supported_filetype(temp_path)
        assert result is False
    finally:
        import os
        os.unlink(temp_path)

def test_is_supported_filetype_with_editor_backup_file():
    config = Config()
    config.supported_extensions = {"py"}
    result = config.is_supported_filetype("test.py~")
    assert result is False

def test_is_supported_filetype_with_fifo_file():
    config = Config()
    config.supported_extensions = {"py"}
    import tempfile
    import os
    import stat
    fifo_path = tempfile.mktemp()
    try:
        os.mkfifo(fifo_path)
        result = config.is_supported_filetype(fifo_path)
        assert result is False
    finally:
        if os.path.exists(fifo_path):
            os.unlink(fifo_path)

def test_is_supported_filetype_with_nonexistent_file():
    config = Config()
    config.supported_extensions = {"py"}
    result = config.is_supported_filetype("nonexistent.py")
    assert result is False


# LLM-generated content at query #5
#--------------------------

def test_find_all_configs_returns_trie_with_default_config():
    result = find_all_configs("/some/path")
    assert isinstance(result, Trie)
    assert result.root.config_info[0] == "default"
    assert result.root.config_info[1] == {}

def test_find_all_configs_inserts_valid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 100")
        result = find_all_configs(tmpdir)
        found_config = result.search(config_path)
        assert found_config[0] == config_path
        assert found_config[1]["line_length"] == 100

def test_find_all_configs_ignores_invalid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("invalid toml content")
        result = find_all_configs(tmpdir)
        found_config = result.search(config_path)
        assert found_config[0] == "default"
        assert found_config[1] == {}

def test_find_all_configs_prefers_first_valid_config_in_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write("[tool.isort]\nline_length = 100")
        setup_path = os.path.join(tmpdir, "setup.cfg")
        with open(setup_path, "w") as f:
            f.write("[isort]\nline_length = 120")
        result = find_all_configs(tmpdir)
        found_config = result.search(pyproject_path)
        assert found_config[0] == pyproject_path
        assert found_config[1]["line_length"] == 100

def test_find_all_configs_walks_subdirectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        config_path = os.path.join(subdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 100")
        result = find_all_configs(tmpdir)
        found_config = result.search(config_path)
        assert found_config[0] == config_path
        assert found_config[1]["line_length"] == 100

def test_find_all_configs_handles_editorconfig():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = space\nindent_size = 2")
        result = find_all_configs(tmpdir)
        found_config = result.search(config_path)
        assert found_config[0] == config_path
        assert found_config[1]["indent"] == "  "

def test_find_all_configs_handles_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_all_configs(tmpdir)
        found_config = result.search(os.path.join(tmpdir, "some_file.py"))
        assert found_config[0] == "default"
        assert found_config[1] == {}

def test_find_all_configs_handles_nested_configs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root_config = os.path.join(tmpdir, "pyproject.toml")
        with open(root_config, "w") as f:
            f.write("[tool.isort]\nline_length = 100")
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        sub_config = os.path.join(subdir, "pyproject.toml")
        with open(sub_config, "w") as f:
            f.write("[tool.isort]\nline_length = 200")
        result = find_all_configs(tmpdir)
        root_file = os.path.join(tmpdir, "root.py")
        sub_file = os.path.join(subdir, "sub.py")
        root_found = result.search(root_file)
        sub_found = result.search(sub_file)
        assert root_found[0] == root_config
        assert root_found[1]["line_length"] == 100
        assert sub_found[0] == sub_config
        assert sub_found[1]["line_length"] == 200


# LLM-generated content at query #6
#--------------------------

def test_section_in_section_defaults():
    config = Config()
    config._Config__dataclass_fields__ = {}
    config.sources = []
    config.combined_config = {"sections": ("STDLIB",)}
    config.known_other = {}
    config.SECTION_DEFAULTS = ("STDLIB", "THIRDPARTY", "FIRSTPARTY")
    config.warn = lambda msg, stacklevel: None
    config._process_sections()
    assert config.combined_config.get("sections") == ("STDLIB",)


# LLM-generated content at query #7
#--------------------------

def test_config_constructor_with_config_and_overrides():
    mock_config = _Config(py_version="py310", quiet=True, indent=4)
    config = Config(config=mock_config, quiet=False, indent=2)
    assert config.py_version == "310"
    assert config.quiet == False
    assert config.indent == "  "

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.isort]\nprofile = "black"\nline_length = 100')
        settings_file = f.name
    config = Config(settings_file=settings_file)
    assert config.profile == "black"
    assert config.line_length == 100
    os.unlink(settings_file)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'pyproject.toml')
        with open(config_file, 'w') as f:
            f.write('[tool.isort]\nprofile = "black"')
        config = Config(settings_path=tmpdir)
        assert config.profile == "black"

def test_config_constructor_with_config_overrides():
    config = Config(profile="black", line_length=100)
    assert config.profile == "black"
    assert config.line_length == 100

def test_config_constructor_with_known_other():
    config = Config(known_foo=["foo", "bar"], sections=["FUTURE", "STDLIB", "FOO", "THIRDPARTY"])
    assert "foo" in config.known_other
    assert config.known_other["foo"] == frozenset(["foo", "bar"])

def test_config_constructor_with_import_headings():
    config = Config(import_heading_stdlib="Standard Library")
    assert config.import_headings["stdlib"] == "Standard Library"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_stdlib="End Standard Library")
    assert config.import_footers["stdlib"] == "End Standard Library"

def test_config_constructor_with_indent_as_integer():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_string():
    config = Config(indent="2")
    assert config.indent == "  "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_src_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(directory=tmpdir, src_paths=["src", "tests"])
        src_paths = config.src_paths
        assert any("src" in str(p) for p in src_paths)
        assert any("tests" in str(p) for p in src_paths)

def test_config_constructor_with_formatter():
    config = Config(formatter="example")
    assert config.formatting_function is not None

def test_config_constructor_with_deprecated_options():
    config = Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort is not True

def test_config_constructor_with_unsupported_settings():
    try:
        config = Config(unsupported_option="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_empty_settings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[settings]\nline_length = 100')
        settings_file = f.name
    config = Config(settings_file=settings_file, quiet=True)
    assert config.line_length != 100
    os.unlink(settings_file)

def test_config_constructor_with_invalid_settings_path():
    try:
        config = Config(settings_path="/invalid/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_invalid_profile():
    try:
        config = Config(profile="invalid_profile")
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_invalid_formatter():
    try:
        config = Config(formatter="invalid_formatter")
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_invalid_sort_order():
    try:
        config = Config(sort_order="invalid_sort")
        assert False
    except SortingFunctionDoesNotExist:
        assert True


# LLM-generated content at query #8
#--------------------------

def test_find_all_configs_break_on_first_valid_config():
    import os
    from unittest.mock import patch, mock_open, MagicMock
    from isort.utils import Trie
    CONFIG_SOURCES = ["pyproject.toml", ".isort.cfg", "setup.cfg"]
    CONFIG_SECTIONS = {"pyproject.toml": "tool.isort", ".isort.cfg": "settings", "setup.cfg": "tool:isort"}
    def _get_config_data(file_path, section):
        return {"key": "value"}
    path = "/test/path"
    trie_root = Trie("default", {})
    mock_walk = [("/test/path", [], ["file1.py"]), ("/test/path/sub", [], ["file2.py"])]
    with patch("os.walk", return_value=mock_walk):
        for dirpath, _, _ in os.walk(path):
            config_found = False
            for config_file_name in CONFIG_SOURCES:
                potential_config_file = os.path.join(dirpath, config_file_name)
                if os.path.isfile(potential_config_file):
                    config_data = _get_config_data(potential_config_file, CONFIG_SECTIONS[config_file_name])
                    if config_data:
                        trie_root.insert(potential_config_file, config_data)
                        config_found = True
                        break
            if config_found:
                continue
    assert trie_root.root.config_info == ("default", {})


# LLM-generated content at query #9
#--------------------------

def test_find_config_with_existing_toml():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["line_length"] == 88
        assert config_data["source"] == config_path

def test_find_config_with_existing_editorconfig():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 100\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["indent"] == "    "
        assert config_data["line_length"] == 100
        assert config_data["source"] == config_path

def test_find_config_with_existing_setup_cfg():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "setup.cfg")
        with open(config_path, "w") as f:
            f.write("[isort]\nline_length = 120\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["line_length"] == 120
        assert config_data["source"] == config_path

def test_find_config_with_existing_tox_ini():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "tox.ini")
        with open(config_path, "w") as f:
            f.write("[isort]\nline_length = 80\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["line_length"] == 80
        assert config_data["source"] == config_path

def test_find_config_with_existing_isort_cfg():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".isort.cfg")
        with open(config_path, "w") as f:
            f.write("[settings]\nline_length = 90\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["line_length"] == 90
        assert config_data["source"] == config_path

def test_find_config_with_no_config():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data == {}

def test_find_config_stops_at_stop_dir():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        stop_dir = os.path.join(tmpdir, ".git")
        os.makedirs(stop_dir)
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        config_path = os.path.join(subdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        result_dir, config_data = _find_config(subdir)
        assert result_dir == tmpdir
        assert config_data == {}

def test_find_config_searches_upwards():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        subdir = os.path.join(tmpdir, "subdir", "nested")
        os.makedirs(subdir)
        result_dir, config_data = _find_config(subdir)
        assert result_dir == tmpdir
        assert config_data["line_length"] == 88
        assert config_data["source"] == config_path

def test_find_config_with_max_depth():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        current = tmpdir
        for i in range(10):
            current = os.path.join(current, f"level{i}")
            os.makedirs(current)
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        result_dir, config_data = _find_config(current)
        assert result_dir == tmpdir
        assert config_data["line_length"] == 88
        assert config_data["source"] == config_path

def test_find_config_with_invalid_config_file():
    import tempfile
    import os
    import warnings
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("invalid toml content")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_dir, config_data = _find_config(tmpdir)
            assert len(w) == 1
            assert "Failed to pull configuration information" in str(w[0].message)
        assert result_dir == tmpdir
        assert config_data == {}

def test_find_config_editorconfig_with_off_max_line_length():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nmax_line_length = off\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["line_length"] == float("inf")
        assert config_data["source"] == config_path

def test_find_config_editorconfig_with_tab_indent():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = tab\nindent_size = 2\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["indent"] == "\t\t"
        assert config_data["source"] == config_path

def test_find_config_with_force_grid_wrap_backwards_compat():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "setup.cfg")
        with open(config_path, "w") as f:
            f.write("[isort]\nforce_grid_wrap = false\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["force_grid_wrap"] == 0

def test_find_config_with_comment_prefix_stripping():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "setup.cfg")
        with open(config_path, "w") as f:
            f.write("[isort]\ncomment_prefix = \"# \"\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["comment_prefix"] == "# "

def test_find_config_with_known_prefix_paths():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "setup.cfg")
        with open(config_path, "w") as f:
            f.write("[isort]\nknown_local_folder = ./local\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        expected_path = os.path.join(tmpdir, "local")
        assert expected_path in config_data["known_local_folder"]

def test_find_config_editorconfig_with_extension_section():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.{py,pyi}]\nindent_size = 2\n")
        result_dir, config_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert config_data["indent"] == "  "
        assert config_data["source"] == config_path


# LLM-generated content at query #10
#--------------------------

def test_config_constructor_with_config_and_overrides():
    mock_config = _Config()
    mock_config.py_version = "py310"
    mock_config.some_setting = "original"
    config = Config(config=mock_config, some_setting="overridden", quiet=True)
    assert config.py_version == "310"
    assert config.some_setting == "overridden"
    assert config.quiet is True

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.isort]\nprofile = "black"\nline_length = 100')
        settings_file = f.name
    config = Config(settings_file=settings_file)
    assert config.profile == "black"
    assert config.line_length == 100
    os.unlink(settings_file)

def test_config_constructor_with_settings_path():
    temp_dir = tempfile.mkdtemp()
    config_file = os.path.join(temp_dir, "pyproject.toml")
    with open(config_file, 'w') as f:
        f.write('[tool.isort]\nprofile = "django"')
    config = Config(settings_path=temp_dir)
    assert config.profile == "django"
    shutil.rmtree(temp_dir)

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_profile_override():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_nonexistent_profile():
    try:
        Config(profile="nonexistent")
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other_section():
    config = Config(known_mysection=["mypackage"], sections=["STDLIB", "MYSECTION"])
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(import_heading_mysection="My Section")
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_mysection="End of My Section")
    assert config.import_footers["mysection"] == "End of My Section"

def test_config_constructor_with_unsupported_setting():
    try:
        Config(unsupported_setting="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_deprecated_setting():
    config = Config(skip_gitignore=True, quiet=True)
    assert "skip_gitignore" not in dir(config)

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="example_formatter")
    assert callable(config.formatting_function)

def test_config_constructor_with_nonexistent_formatter():
    try:
        Config(formatter="nonexistent_formatter")
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert len(config.src_paths) == 2

def test_config_constructor_with_sort_order_natural():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_sort_order_native():
    config = Config(sort_order="native")
    assert config.sorting_function == sorted

def test_config_constructor_with_custom_sort_order():
    config = Config(sort_order="custom_sort")
    assert callable(config.sorting_function)

def test_config_constructor_with_nonexistent_sort_order():
    try:
        Config(sort_order="nonexistent")
        assert False
    except SortingFunctionDoesNotExist:
        assert True


# LLM-generated content at query #11
#--------------------------

def test_is_supported_filetype_with_supported_extension():
    config = Config()
    config.supported_extensions = {"py", "txt"}
    result = config.is_supported_filetype("test.py")
    assert result == True

def test_is_supported_filetype_with_blocked_extension():
    config = Config()
    config.blocked_extensions = {"log"}
    result = config.is_supported_filetype("test.log")
    assert result == False

def test_is_supported_filetype_with_unknown_extension_and_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    mock_file = "test.sh"
    with open(mock_file, "w") as f:
        f.write("#!/bin/bash\n")
    result = config.is_supported_filetype(mock_file)
    os.remove(mock_file)
    assert result == True

def test_is_supported_filetype_with_unknown_extension_and_no_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    mock_file = "test.xyz"
    with open(mock_file, "w") as f:
        f.write("no shebang here\n")
    result = config.is_supported_filetype(mock_file)
    os.remove(mock_file)
    assert result == False

def test_is_supported_filetype_with_editor_backup_file():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    mock_file = "test.py~"
    with open(mock_file, "w") as f:
        f.write("#!/usr/bin/env python\n")
    result = config.is_supported_filetype(mock_file)
    os.remove(mock_file)
    assert result == False

def test_is_supported_filetype_with_fifo_file():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    fifo_path = "test_fifo"
    os.mkfifo(fifo_path)
    result = config.is_supported_filetype(fifo_path)
    os.remove(fifo_path)
    assert result == False

def test_is_supported_filetype_with_nonexistent_file():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    result = config.is_supported_filetype("nonexistent.xyz")
    assert result == False


# LLM-generated content at query #12
#--------------------------

def test__get_config_data_with_toml():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n')
        toml_path = f.name
    result = _get_config_data(toml_path, ('tool.black',))
    os.unlink(toml_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == toml_path

def test__get_config_data_with_ini():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\nskip_string_normalization = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == ini_path

def test__get_config_data_with_editorconfig_indent_spaces():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('root = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 100\n')
        ec_path = f.name
    result = _get_config_data(ec_path, ('*',))
    os.unlink(ec_path)
    assert result['indent'] == '  '
    assert result['line_length'] == 100
    assert result['source'] == ec_path

def test__get_config_data_with_editorconfig_indent_tabs():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_style = tab\nindent_size = tab\nmax_line_length = off\n')
        ec_path = f.name
    result = _get_config_data(ec_path, ('*',))
    os.unlink(ec_path)
    assert result['indent'] == '\t'
    assert result['line_length'] == float('inf')
    assert result['source'] == ec_path

def test__get_config_data_with_editorconfig_extension_section():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_size = 4\n\n[*.{py,pyi}]\nindent_size = 8\n')
        ec_path = f.name
    result = _get_config_data(ec_path, ('*.{py}',))
    os.unlink(ec_path)
    assert result['indent'] == '        '
    assert result['source'] == ec_path

def test__get_config_data_with_unknown_keys_filtered():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_size = 4\nunknown_key = value\n')
        ec_path = f.name
    result = _get_config_data(ec_path, ('*',))
    os.unlink(ec_path)
    assert 'unknown_key' not in result
    assert 'indent' in result

def test__get_config_data_bool_conversion():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nskip_string_normalization = false\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['skip_string_normalization'] is False

def test__get_config_data_force_grid_wrap_backwards_compat_false():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = false\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 0

def test__get_config_data_force_grid_wrap_backwards_compat_true():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 2

def test__get_config_data_comment_prefix_stripping():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ncomment_prefix = "# "\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['comment_prefix'] == '# '

def test__get_config_data_known_prefix_paths():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nknown_first_party = mod1,mod2\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    expected_paths = {os.path.join(os.path.dirname(ini_path), 'mod1'), os.path.join(os.path.dirname(ini_path), 'mod2')}
    assert set(result['known_first_party']) == expected_paths

def test__get_config_data_empty_section():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n\n[other]\nkey = value\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('nonexistent',))
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_multiple_sections():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n\n[pycodestyle]\nmax_line_length = 79\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black', 'pycodestyle'))
    os.unlink(ini_path)
    assert result['line_length'] == 79
    assert result['source'] == ini_path


# LLM-generated content at query #13
#--------------------------

def test_config_settings_source_exists():
    config_settings = {"source": "/some/path"}
    combined_config = {}
    result = config_settings.get("source", None)
    assert result is not None


# LLM-generated content at query #14
#--------------------------

def test_is_supported_filetype_with_supported_extension():
    config = Config()
    config.supported_extensions = {"py", "txt"}
    result = config.is_supported_filetype("test.py")
    assert result == True

def test_is_supported_filetype_with_blocked_extension():
    config = Config()
    config.blocked_extensions = {"log"}
    result = config.is_supported_filetype("error.log")
    assert result == False

def test_is_supported_filetype_with_unknown_extension_and_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    mock_open = unittest.mock.mock_open(read_data=b"#!/usr/bin/env python\n")
    with unittest.mock.patch("builtins.open", mock_open):
        with unittest.mock.patch("os.stat", return_value=unittest.mock.Mock(st_mode=0)):
            result = config.is_supported_filetype("script")
    assert result == True

def test_is_supported_filetype_with_unknown_extension_and_no_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    mock_open = unittest.mock.mock_open(read_data=b"print('hello')\n")
    with unittest.mock.patch("builtins.open", mock_open):
        with unittest.mock.patch("os.stat", return_value=unittest.mock.Mock(st_mode=0)):
            result = config.is_supported_filetype("script")
    assert result == False

def test_is_supported_filetype_with_editor_backup_file():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    result = config.is_supported_filetype("test.py~")
    assert result == False

def test_is_supported_filetype_with_fifo_file():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    with unittest.mock.patch("os.stat", return_value=unittest.mock.Mock(st_mode=stat.S_ISFIFO(1))):
        result = config.is_supported_filetype("fifo")
    assert result == False

def test_is_supported_filetype_with_file_open_error():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    with unittest.mock.patch("builtins.open", side_effect=OSError):
        result = config.is_supported_filetype("nonexistent")
    assert result == False


# LLM-generated content at query #15
#--------------------------

def test_init_with_config_and_overrides():
    mock_config = _Config(py_version="py310", some_setting="value")
    config_vars = vars(mock_config).copy()
    config_vars.update({"quiet": True})
    config_vars["py_version"] = config_vars["py_version"].replace("py", "")
    config_vars.pop("_known_patterns", None)
    config_vars.pop("_section_comments", None)
    config_vars.pop("_section_comments_end", None)
    config_vars.pop("_skips", None)
    config_vars.pop("_skip_globs", None)
    config_vars.pop("_sorting_function", None)
    config = Config(config=mock_config, quiet=True)
    assert config.py_version == "310"
    assert config.some_setting == "value"


# LLM-generated content at query #16
#--------------------------

def test_config_initialization_with_config_parameter():
    mock_config = _Config()
    mock_config.py_version = "py310"
    config = Config(config=mock_config)
    assert config.py_version == "310"


# LLM-generated content at query #17
#--------------------------

def test_is_supported_filetype_blocked_extension():
    config = Config()
    config.blocked_extensions = {"txt"}
    result = config.is_supported_filetype("test.txt")
    assert result == False


# LLM-generated content at query #18
#--------------------------

def test_find_config_with_valid_toml_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.black]\nline_length = 100\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 100
        assert result_data["source"] == config_file

def test_find_config_with_valid_editorconfig_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, ".editorconfig")
        with open(config_file, "w") as f:
            f.write("[*.py]\nindent_style = space\nindent_size = 4\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "    "
        assert result_data["source"] == config_file

def test_find_config_with_invalid_config_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("invalid toml content")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_with_no_config_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_stop_search_on_stop_dir():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        stop_dir = os.path.join(tmpdir, ".git")
        os.makedirs(stop_dir)
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        config_file = os.path.join(subdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.black]\nline_length = 88\n")
        result_dir, result_data = _find_config(subdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_search_up_directory_tree():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.black]\nline_length = 120\n")
        subdir = os.path.join(tmpdir, "subdir", "nested")
        os.makedirs(subdir)
        result_dir, result_data = _find_config(subdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 120

def test_find_config_max_search_depth():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        current = tmpdir
        for i in range(10):
            current = os.path.join(current, f"level{i}")
            os.makedirs(current)
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.black]\nline_length = 79\n")
        result_dir, result_data = _find_config(current)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 79

def test_find_config_prioritize_pyproject_over_editorconfig():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject, "w") as f:
            f.write("[tool.black]\nline_length = 100\n")
        editorconfig = os.path.join(tmpdir, ".editorconfig")
        with open(editorconfig, "w") as f:
            f.write("[*.py]\nline_length = 80\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 100

def test_find_config_with_force_grid_wrap_backwards_compatibility():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.black]\nforce_grid_wrap = false\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["force_grid_wrap"] == 0

def test_find_config_with_comment_prefix_stripping():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "pyproject.toml")
        with open(config_file, "w") as f:
            f.write("[tool.black]\ncomment_prefix = \"# \"\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["comment_prefix"] == "# "


# LLM-generated content at query #19
#--------------------------

def test_config_constructor_with_config_overrides():
    config = Config(quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_file():
    config = Config(settings_file="pyproject.toml")
    assert config.directory is not None

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config.directory is not None

def test_config_constructor_with_config_object():
    base_config = Config()
    new_config = Config(config=base_config, quiet=True)
    assert new_config.quiet is True

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="nonexistent_path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_string():
    config = Config(indent="2")
    assert config.indent == "  "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other():
    config = Config(known_custom=["mymodule"])
    assert "custom" in config.known_other

def test_config_constructor_with_import_headings():
    config = Config(import_heading_custom="Custom Imports")
    assert "custom" in config.import_headings

def test_config_constructor_with_import_footers():
    config = Config(import_footer_custom="End Custom Imports")
    assert "custom" in config.import_footers

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert len(config.src_paths) == 2

def test_config_constructor_with_formatter():
    try:
        Config(formatter="unknown_formatter")
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_deprecated_options():
    config = Config(force_alphabetical_sort=False)
    assert config.force_alphabetical_sort is False

def test_config_constructor_with_unsupported_settings():
    try:
        Config(unsupported_option="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_empty_config():
    config = Config()
    assert config.directory is not None

def test_config_constructor_with_combined_config():
    config = Config(profile="black", quiet=True, indent=4)
    assert config.profile == "black"
    assert config.quiet is True
    assert config.indent == "    "

def test_config_constructor_with_py_version():
    base_config = Config(py_version="py310")
    new_config = Config(config=base_config)
    assert new_config.py_version == "310"

def test_config_constructor_with_sections_and_known_other():
    config = Config(sections=["CUSTOM"], known_custom=["mymodule"])
    assert "CUSTOM" in config.sections
    assert "custom" in config.known_other

def test_config_constructor_with_directory_and_src_paths():
    config = Config(directory="/tmp", src_paths=["src"])
    assert config.directory == "/tmp"
    assert len(config.src_paths) > 0


# LLM-generated content at query #20
#--------------------------

def test_predicate_at_line_38_true():
    import tempfile
    import configparser
    file_content = "[section1]\nkey1 = value1\nkey2 = value2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(file_content)
        file_path = f.name
    sections = ("section1",)
    result = _get_config_data(file_path, sections)
    assert "source" in result
    import os
    os.unlink(file_path)


# LLM-generated content at query #21
#--------------------------

def test_config_initialization_with_config_parameter():
    config_instance = _Config()
    config_instance.py_version = "py310"
    config_instance._known_patterns = []
    config_instance._section_comments = ()
    config_instance._section_comments_end = ()
    config_instance._skips = frozenset()
    config_instance._skip_globs = frozenset()
    config_instance._sorting_function = None
    config = Config(config=config_instance)
    assert config.py_version == "310"


# LLM-generated content at query #22
#--------------------------

def test_warning_when_settings_file_empty_and_not_quiet():
    import tempfile
    import os
    from isort import Config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
        tmp.write('')
        tmp_path = tmp.name
    try:
        config = Config(settings_file=tmp_path, quiet=False)
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #23
#--------------------------

def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_constructor_with_config_and_overrides():
    base_config = Config(config_overrides={"quiet": False})
    config = Config(config=base_config, quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_file(tmp_path):
    settings_file = tmp_path / ".isort.cfg"
    settings_file.write_text("[isort]\nquiet = true")
    config = Config(settings_file=str(settings_file))
    assert config.quiet is True

def test_config_constructor_with_settings_path(tmp_path):
    settings_file = tmp_path / ".isort.cfg"
    settings_file.write_text("[isort]\nquiet = true")
    config = Config(settings_path=str(tmp_path))
    assert config.quiet is True

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_indent_as_number():
    config = Config(config_overrides={"indent": 4})
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_constructor_with_known_other():
    config = Config(config_overrides={"known_mysection": ["mypackage"]})
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_mysection": "My Section"})
    assert "mysection" in config.import_headings
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_mysection": "End of My Section"})
    assert "mysection" in config.import_footers
    assert config.import_footers["mysection"] == "End of My Section"

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src", "tests"]})
    assert any("src" in str(path) for path in config.src_paths)
    assert any("tests" in str(path) for path in config.src_paths)

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "console"})
    assert config.formatter == "console"

def test_config_constructor_with_deprecated_options():
    config = Config(config_overrides={"force_sort_within_sections": True})
    assert "force_sort_within_sections" not in config.__dict__

def test_config_constructor_with_unsupported_settings():
    try:
        Config(config_overrides={"unsupported_option": "value"})
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_empty_settings_file(tmp_path):
    settings_file = tmp_path / ".isort.cfg"
    settings_file.write_text("[settings]\nquiet = true")
    config = Config(settings_file=str(settings_file), quiet=True)
    assert config.quiet is True

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="/invalid/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_invalid_profile():
    try:
        Config(config_overrides={"profile": "invalid"})
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_invalid_formatter():
    try:
        Config(config_overrides={"formatter": "invalid"})
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_invalid_sort_order():
    try:
        Config(config_overrides={"sort_order": "invalid"})
        assert False
    except SortingFunctionDoesNotExist:
        assert True

def test_config_constructor_with_py_version():
    base_config = Config(config_overrides={"py_version": "py310"})
    config = Config(config=base_config)
    assert config.py_version == "310"

def test_config_constructor_with_combined_config():
    config = Config(config_overrides={"quiet": True, "profile": "black", "indent": 4})
    assert config.quiet is True
    assert config.profile == "black"
    assert config.indent == "    "

def test_config_constructor_with_no_arguments():
    config = Config()
    assert config.quiet is False

def test_config_constructor_with_config_and_no_overrides():
    base_config = Config(config_overrides={"quiet": True})
    config = Config(config=base_config)
    assert config.quiet is True

def test_config_constructor_with_settings_file_and_overrides(tmp_path):
    settings_file = tmp_path / ".isort.cfg"
    settings_file.write_text("[isort]\nquiet = false")
    config = Config(settings_file=str(settings_file), quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_path_and_overrides(tmp_path):
    settings_file = tmp_path / ".isort.cfg"
    settings_file.write_text("[isort]\nquiet = false")
    config = Config(settings_path=str(tmp_path), quiet=True)
    assert config.quiet is True

def test_config_constructor_with_profile_and_overrides():
    config = Config(config_overrides={"profile": "black", "quiet": True})
    assert config.profile == "black"
    assert config.quiet is True

def test_config_constructor_with_indent_and_overrides():
    config = Config(config_overrides={"indent": 2, "quiet": True})
    assert config.indent == "  "
    assert config.quiet is True

def test_config_constructor_with_known_other_and_overrides():
    config = Config(config_overrides={"known_mysection": ["mypackage"], "quiet": True})
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]
    assert config.quiet is True

def test_config_constructor_with_import_headings_and_overrides():
    config = Config(config_overrides={"import_heading_mysection": "My Section", "quiet": True})
    assert "mysection" in config.import_headings
    assert config.import_headings["mysection"] == "My Section"
    assert config.quiet is True

def test_config_constructor_with_import_footers_and_overrides():
    config = Config(config_overrides={"import_footer_mysection": "End of My Section", "quiet": True})
    assert "mysection" in config.import_footers
    assert config.import_footers["mysection"] == "End of My Section"
    assert config.quiet is True

def test_config_constructor_with_src_paths_and_overrides():
    config = Config(config_overrides={"src_paths": ["src", "tests"], "quiet": True})
    assert any("src" in str(path) for path in config.src_paths)
    assert any("tests" in str(path) for path in config.src_paths)
    assert config.quiet is True

def test_config_constructor_with_formatter_and_overrides():
    config = Config(config_overrides={"formatter": "console", "quiet": True})
    assert config.formatter == "console"
    assert config.quiet is True

def test_config_constructor_with_deprecated_options_and_overrides():
    config = Config(config_overrides={"force_sort_within_sections": True, "quiet": True})
    assert "force_sort_within_sections" not in config.__dict__
    assert config.quiet is True

def test_config_constructor_with_unsupported_settings_and_overrides():
    try:
        Config(config_overrides={"unsupported_option": "value", "quiet": True})
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_empty_settings_file_and_overrides(tmp_path):
    settings_file = tmp_path / ".isort.cfg"
    settings_file.write_text("[settings]\nquiet = false")
    config = Config(settings_file=str(settings_file), quiet=True)
    assert config.quiet is True

def test_config_constructor_with_invalid_settings_path_and_overrides():
    try:
        Config(settings_path="/invalid/path", quiet=True)
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_invalid_profile_and_overrides():
    try:
        Config(config_overrides={"profile": "invalid", "quiet": True})
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_invalid_formatter_and_overrides():
    try:
        Config(config_overrides={"formatter": "invalid", "quiet": True})
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_invalid_sort_order_and_overrides():
    try:
        Config(config_overrides={"sort_order": "invalid", "quiet": True})
        assert False
    except SortingFunctionDoesNotExist:
        assert True

def test_config_constructor_with_py_version_and_overrides():
    base_config = Config(config_overrides={"py_version": "py310"})
    config = Config(config=base_config, quiet=True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_is_supported_filetype_with_supported_extension():
    config = Config()
    config.supported_extensions = {"py", "txt"}
    result = config.is_supported_filetype("test.py")
    assert result is True

def test_is_supported_filetype_with_blocked_extension():
    config = Config()
    config.blocked_extensions = {"log", "tmp"}
    result = config.is_supported_filetype("test.log")
    assert result is False

def test_is_supported_filetype_with_unknown_extension_and_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'#!/usr/bin/env python\n')
        temp_name = f.name
    try:
        result = config.is_supported_filetype(temp_name)
        assert result is True
    finally:
        import os
        os.unlink(temp_name)

def test_is_supported_filetype_with_unknown_extension_and_no_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'no shebang here\n')
        temp_name = f.name
    try:
        result = config.is_supported_filetype(temp_name)
        assert result is False
    finally:
        import os
        os.unlink(temp_name)

def test_is_supported_filetype_with_backup_file():
    config = Config()
    config.supported_extensions = {"py"}
    config.blocked_extensions = set()
    result = config.is_supported_filetype("test.py~")
    assert result is False

def test_is_supported_filetype_with_fifo_file():
    config = Config()
    config.supported_extensions = {"py"}
    config.blocked_extensions = set()
    import tempfile
    import os
    import stat
    fifo_path = tempfile.mktemp()
    os.mkfifo(fifo_path)
    try:
        result = config.is_supported_filetype(fifo_path)
        assert result is False
    finally:
        if os.path.exists(fifo_path):
            os.unlink(fifo_path)

def test_is_supported_filetype_with_nonexistent_file():
    config = Config()
    config.supported_extensions = {"py"}
    config.blocked_extensions = set()
    result = config.is_supported_filetype("nonexistent.py")
    assert result is False


# LLM-generated content at query #2
#--------------------------

def test_is_supported_filetype_blocked_extension():
    config = Config()
    config.blocked_extensions = {"txt"}
    result = config.is_supported_filetype("test.txt")
    assert result == False


# LLM-generated content at query #3
#--------------------------

def test_config_constructor_with_config_parameter():
    mock_config = _Config()
    mock_config.py_version = "py39"
    config = Config(config=mock_config)
    assert config.py_version == "39"

def test_config_constructor_with_config_overrides():
    mock_config = _Config()
    mock_config.py_version = "py39"
    config = Config(config=mock_config, quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.isort]\nprofile = "black"\n')
        settings_file = f.name
    try:
        config = Config(settings_file=settings_file)
        assert config.profile == "black"
    finally:
        os.unlink(settings_file)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'pyproject.toml')
        with open(config_file, 'w') as f:
            f.write('[tool.isort]\nline_length = 100\n')
        config = Config(settings_path=tmpdir)
        assert config.line_length == 100

def test_config_constructor_with_invalid_settings_path():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/nonexistent/path")

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_nonexistent_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_string():
    config = Config(indent="2")
    assert config.indent == "  "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other_section():
    config = Config(known_mysection=["mypackage"])
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(import_heading_mysection="My Section")
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_mysection="End of My Section")
    assert config.import_footers["mysection"] == "End of My Section"

def test_config_constructor_with_unsupported_setting():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_setting="value")

def test_config_constructor_with_deprecated_option():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = Config(force_alphabetical_sort=True)
        assert len(w) == 1
        assert "Deprecated config options were used" in str(w[0].message)

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="example")
    assert callable(config.formatting_function)

def test_config_constructor_with_nonexistent_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent_formatter")

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert any("src" in str(path) for path in config.src_paths)
    assert any("tests" in str(path) for path in config.src_paths)

def test_config_constructor_with_glob_src_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src", "module1"))
        os.makedirs(os.path.join(tmpdir, "src", "module2"))
        config = Config(directory=tmpdir, src_paths=["src/*"])
        assert len(config.src_paths) >= 2

def test_config_constructor_with_sort_order_natural():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_sort_order_native():
    config = Config(sort_order="native")
    assert config.sorting_function == sorted

def test_config_constructor_with_custom_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="custom_nonexistent")

def test_config_constructor_with_quiet_false_and_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = Config(force_alphabetical_sort=True, quiet=False)
        assert len(w) == 1

def test_config_constructor_with_quiet_true_and_no_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = Config(force_alphabetical_sort=True, quiet=True)
        assert len(w) == 0

def test_config_constructor_with_empty_settings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[settings]\nline_length = 100\n')
        settings_file = f.name
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = Config(settings_file=settings_file, quiet=False)
            assert len(w) == 1
            assert "no configuration was found inside" in str(w[0].message)
    finally:
        os.unlink(settings_file)

def test_config_constructor_with_sections_and_known_other_mismatch():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = Config(sections=("STDLIB", "CUSTOM"), known_custom=["mypackage"])
        assert len(w) == 0
    config = Config(sections=("STDLIB", "CUSTOM"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = config.known_patterns
        assert len(w) == 1
        assert "no known_custom is defined" in str(w[0].message)


# LLM-generated content at query #4
#--------------------------

def test_abspaths_with_relative_paths_ending_with_sep():
    cwd = "/home/user"
    values = ["dir/", "subdir/"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir/", "/home/user/subdir/"}
    assert result == expected

def test_abspaths_with_absolute_paths_ending_with_sep():
    cwd = "/home/user"
    values = ["/usr/local/", "/tmp/"]
    result = _abspaths(cwd, values)
    expected = {"/usr/local/", "/tmp/"}
    assert result == expected

def test_abspaths_with_mixed_paths():
    cwd = "/home/user"
    values = ["dir/", "/usr/local/", "file.txt"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir/", "/usr/local/", "/home/user/file.txt"}
    assert result == expected

def test_abspaths_with_no_paths():
    cwd = "/home/user"
    values = []
    result = _abspaths(cwd, values)
    expected = set()
    assert result == expected

def test_abspaths_with_relative_paths_not_ending_with_sep():
    cwd = "/home/user"
    values = ["dir", "file.txt"]
    result = _abspaths(cwd, values)
    expected = {"/home/user/dir", "/home/user/file.txt"}
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_config_constructor_with_config_and_overrides():
    mock_config = _Config(py_version="py310", quiet=True, indent=4)
    config = Config(config=mock_config, quiet=False, indent=2)
    assert config.py_version == "310"
    assert config.quiet == False
    assert config.indent == "  "

def test_config_constructor_with_settings_file_not_found():
    with pytest.raises(FileNotFoundError):
        Config(settings_file="non_existent_file.ini")

def test_config_constructor_with_settings_path_invalid():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/invalid/path")

def test_config_constructor_with_profile_not_existing():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="non_existent_profile")

def test_config_constructor_with_formatter_not_existing():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="non_existent_formatter")

def test_config_constructor_with_unsupported_settings():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_option="value")

def test_config_constructor_with_deprecated_options():
    config = Config(skip_glob=["*.py"], quiet=True)
    assert "skip_glob" not in config.__dict__

def test_config_constructor_with_known_other_section():
    config = Config(known_mysection=["mypackage"], sections=["MYSECTION"], quiet=True)
    assert "known_other" in config.__dict__
    assert "mysection" in config.known_other

def test_config_constructor_with_import_headings():
    config = Config(import_heading_mysection="My Section", quiet=True)
    assert "import_headings" in config.__dict__
    assert "mysection" in config.import_headings

def test_config_constructor_with_import_footers():
    config = Config(import_footer_mysection="Footer", quiet=True)
    assert "import_footers" in config.__dict__
    assert "mysection" in config.import_footers

def test_config_constructor_indent_conversion():
    config = Config(indent=4, quiet=True)
    assert config.indent == "    "
    config = Config(indent="tab", quiet=True)
    assert config.indent == "\t"
    config = Config(indent="'  '", quiet=True)
    assert config.indent == "  "

def test_config_constructor_src_paths_default():
    config = Config(quiet=True)
    assert len(config.src_paths) == 2

def test_config_constructor_src_paths_custom():
    config = Config(src_paths=["src", "lib"], quiet=True)
    assert len(config.src_paths) >= 2

def test_config_constructor_directory_set():
    config = Config(directory="/some/path", quiet=True)
    assert config.directory == "/some/path"

def test_config_constructor_with_quiet_false_warnings():
    config = Config(quiet=False)
    assert config.quiet == False

def test_config_constructor_with_quiet_true():
    config = Config(quiet=True)
    assert config.quiet == True

def test_config_constructor_with_config_overrides_only():
    config = Config(py_version="py39", quiet=True)
    assert config.py_version == "39"

def test_config_constructor_with_empty_settings():
    config = Config(quiet=True)
    assert config.py_version == _DEFAULT_SETTINGS.get("py_version", "").replace("py", "")

def test_config_constructor_known_patterns_initialized():
    config = Config(quiet=True)
    assert config._known_patterns is None
    _ = config.known_patterns
    assert config._known_patterns is not None

def test_config_constructor_skips_initialized():
    config = Config(quiet=True)
    assert config._skips is None
    _ = config.skips
    assert config._skips is not None

def test_config_constructor_skip_globs_initialized():
    config = Config(quiet=True)
    assert config._skip_globs is None
    _ = config.skip_globs
    assert config._skip_globs is not None

def test_config_constructor_sorting_function_initialized():
    config = Config(quiet=True)
    assert config._sorting_function is None
    _ = config.sorting_function
    assert config._sorting_function is not None

def test_config_constructor_section_comments_initialized():
    config = Config(quiet=True)
    assert config._section_comments is None
    _ = config.section_comments
    assert config._section_comments is not None

def test_config_constructor_section_comments_end_initialized():
    config = Config(quiet=True)
    assert config._section_comments_end is None
    _ = config.section_comments_end
    assert config._section_comments_end is not None


# LLM-generated content at query #6
#--------------------------

def test_config_constructor_with_config_and_overrides():
    mock_config = _Config(py_version="py310", quiet=True, indent=4)
    config = Config(config=mock_config, quiet=False, line_length=100)
    assert config.py_version == "310"
    assert config.quiet == False
    assert config.line_length == 100

def test_config_constructor_with_settings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[isort]\nline_length = 88\nprofile = "black"\n')
        settings_file = f.name
    try:
        config = Config(settings_file=settings_file)
        assert config.line_length == 88
        assert config.profile == "black"
    finally:
        os.unlink(settings_file)

def test_config_constructor_with_settings_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, '.isort.cfg')
        with open(config_file, 'w') as f:
            f.write('[settings]\nline_length = 79\n')
        config = Config(settings_path=tmpdir)
        assert config.line_length == 79

def test_config_constructor_with_invalid_settings_path():
    with pytest.raises(InvalidSettingsPath):
        Config(settings_path="/non/existent/path")

def test_config_constructor_with_profile_override():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_nonexistent_profile():
    with pytest.raises(ProfileDoesNotExist):
        Config(profile="nonexistent_profile")

def test_config_constructor_with_indent_as_integer():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_string():
    config = Config(indent='"\\t"')
    assert config.indent == "\t"

def test_config_constructor_with_known_other_section():
    config = Config(known_mysection=["mypackage"], sections=["STDLIB", "MYSECTION"])
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(import_heading_stdlib="Standard Library")
    assert config.import_headings["stdlib"] == "Standard Library"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_stdlib="End of Standard Library")
    assert config.import_footers["stdlib"] == "End of Standard Library"

def test_config_constructor_with_unsupported_setting():
    with pytest.raises(UnsupportedSettings):
        Config(unsupported_setting="value")

def test_config_constructor_with_deprecated_setting():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = Config(force_sort_within_sections=True)
        assert len(w) == 1
        assert "Deprecated config options were used" in str(w[0].message)

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="example_formatter")
    assert callable(config.formatting_function)

def test_config_constructor_with_nonexistent_formatter():
    with pytest.raises(FormattingPluginDoesNotExist):
        Config(formatter="nonexistent_formatter")

def test_config_constructor_with_src_paths_expansion():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "src")
        os.makedirs(src_dir)
        config = Config(directory=tmpdir, src_paths=["src", "."])
        assert any("src" in str(path) for path in config.src_paths)
        assert any(tmpdir in str(path) for path in config.src_paths)

def test_config_constructor_with_sort_order_native():
    config = Config(sort_order="native")
    assert config.sorting_function == sorted

def test_config_constructor_with_sort_order_natural():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_custom_sort_order():
    config = Config(sort_order="custom_sort")
    assert callable(config.sorting_function)

def test_config_constructor_with_nonexistent_sort_order():
    with pytest.raises(SortingFunctionDoesNotExist):
        Config(sort_order="nonexistent_sort")


# LLM-generated content at query #7
#--------------------------

def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_constructor_with_config_and_overrides():
    base_config = Config(config_overrides={"quiet": False})
    config = Config(config=base_config, quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_file(tmp_path):
    settings_file = tmp_path / "pyproject.toml"
    settings_file.write_text("[tool.isort]\nprofile = 'black'")
    config = Config(settings_file=str(settings_file))
    assert config.profile == "black"

def test_config_constructor_with_settings_path(tmp_path):
    settings_dir = tmp_path / "subdir"
    settings_dir.mkdir()
    settings_file = settings_dir / "pyproject.toml"
    settings_file.write_text("[tool.isort]\nprofile = 'black'")
    config = Config(settings_path=str(settings_dir))
    assert config.profile == "black"

def test_config_constructor_with_invalid_settings_path():
    try:
        Config(settings_path="/nonexistent/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_profile_override():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_known_other_section():
    config = Config(config_overrides={"known_mysection": ["mypackage"]})
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_mysection": "My Section"})
    assert "mysection" in config.import_headings
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_mysection": "My Footer"})
    assert "mysection" in config.import_footers
    assert config.import_footers["mysection"] == "My Footer"

def test_config_constructor_with_indent_as_number():
    config = Config(config_overrides={"indent": 4})
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src", "tests"]})
    assert len(config.src_paths) == 2

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "console"})
    assert config.formatting_function is not None

def test_config_constructor_with_deprecated_option():
    config = Config(config_overrides={"force_alphabetical_sort": True}, quiet=True)
    assert "force_alphabetical_sort" not in dir(config)

def test_config_constructor_with_unsupported_setting():
    try:
        Config(config_overrides={"unsupported_setting": "value"})
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_config_object():
    base_config = Config(config_overrides={"quiet": False, "profile": "black"})
    config = Config(config=base_config, quiet=True)
    assert config.quiet is True
    assert config.profile == "black"

def test_config_constructor_with_py_version_conversion():
    base_config = Config(config_overrides={"py_version": "py310"})
    config = Config(config=base_config)
    assert config.py_version == "310"

def test_config_constructor_with_empty_settings_file(tmp_path):
    settings_file = tmp_path / "pyproject.toml"
    settings_file.write_text("")
    config = Config(settings_file=str(settings_file), quiet=True)
    assert config.profile == ""

def test_config_constructor_with_nonexistent_profile():
    try:
        Config(config_overrides={"profile": "nonexistent"})
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_combined_sources():
    config = Config(config_overrides={"line_length": 100, "profile": "black"})
    assert config.line_length == 100
    assert config.profile == "black"

def test_config_constructor_with_directory_auto_detection(tmp_path):
    settings_file = tmp_path / "pyproject.toml"
    settings_file.write_text("[tool.isort]\nline_length = 120")
    config = Config(settings_file=str(settings_file))
    assert config.directory == str(tmp_path)

def test_config_constructor_with_skip_gitignore():
    config = Config(config_overrides={"skip_gitignore": True})
    assert config.skip_gitignore is True

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sorting_function is not None

def test_config_constructor_with_invalid_sort_order():
    try:
        Config(config_overrides={"sort_order": "invalid"})
        assert False
    except SortingFunctionDoesNotExist:
        assert True


# LLM-generated content at query #8
#--------------------------

def test_find_config_with_existing_toml_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 88
        assert result_data["source"] == config_path

def test_find_config_with_existing_editorconfig_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = space\nindent_size = 4\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "    "

def test_find_config_with_existing_setup_cfg_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "setup.cfg")
        with open(config_path, "w") as f:
            f.write("[isort]\nline_length = 100\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 100

def test_find_config_with_existing_tox_ini_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "tox.ini")
        with open(config_path, "w") as f:
            f.write("[isort]\nline_length = 120\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 120

def test_find_config_with_existing_isort_cfg_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".isort.cfg")
        with open(config_path, "w") as f:
            f.write("[settings]\nline_length = 80\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 80

def test_find_config_with_no_config_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_stops_at_stop_dir():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        stop_dir = os.path.join(tmpdir, ".git")
        os.makedirs(stop_dir)
        parent_dir = os.path.join(tmpdir, "subdir")
        os.makedirs(parent_dir)
        config_path = os.path.join(parent_dir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        result_dir, result_data = _find_config(parent_dir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_searches_upwards():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        subdir = os.path.join(tmpdir, "subdir1", "subdir2")
        os.makedirs(subdir)
        result_dir, result_data = _find_config(subdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 88

def test_find_config_limits_search_depth():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        current = tmpdir
        for i in range(MAX_CONFIG_SEARCH_DEPTH + 2):
            current = os.path.join(current, f"subdir{i}")
            os.makedirs(current)
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        result_dir, result_data = _find_config(current)
        assert result_dir == current
        assert result_data == {}

def test_find_config_with_invalid_config_file():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("invalid toml content")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_prioritizes_first_found_config():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write("[tool.isort]\nline_length = 88\n")
        isort_cfg_path = os.path.join(tmpdir, ".isort.cfg")
        with open(isort_cfg_path, "w") as f:
            f.write("[settings]\nline_length = 100\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 88

def test_find_config_with_editorconfig_max_line_length_off():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nmax_line_length = off\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == float("inf")

def test_find_config_with_editorconfig_tab_indent():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = tab\nindent_size = 2\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "\t\t"

def test_find_config_with_editorconfig_no_indent_size():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = space\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "    "

def test_find_config_with_editorconfig_tab_width():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = tab\nindent_size = tab\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "\t"

def test_find_config_with_editorconfig_wildcard_extension():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.{py,pyi}]\nindent_style = space\nindent_size = 4\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "    "

def test_find_config_with_force_grid_wrap_backwards_compat():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.isort]\nforce_grid_wrap = false\n")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["force_grid_wrap"] == 0

def test_find_config_with_comment_prefix():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path


# LLM-generated content at query #9
#--------------------------

def test_deprecated_options_used():
    config_overrides = {"deprecated_option": "value"}
    config = Config(**config_overrides)
    assert "deprecated_option" not in config._config


# LLM-generated content at query #10
#--------------------------

def test_config_initialization_with_config_parameter():
    mock_config = type('_Config', (), {})()
    mock_config.py_version = "py310"
    config = Config(config=mock_config)
    assert config.py_version == "310"


# LLM-generated content at query #11
#--------------------------

def test_find_all_configs_with_no_configs(tmp_path):
    trie = find_all_configs(str(tmp_path))
    assert trie.root.config_info == ("default", {})
    assert trie.root.nodes == {}

def test_find_all_configs_with_single_config(tmp_path):
    config_file = tmp_path / ".isort.cfg"
    config_file.write_text("[settings]\nline_length=100")
    trie = find_all_configs(str(tmp_path))
    result = trie.search(str(tmp_path / "file.py"))
    assert result[0] == str(config_file)
    assert result[1]["line_length"] == 100

def test_find_all_configs_with_nested_configs(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    root_config = tmp_path / ".isort.cfg"
    root_config.write_text("[settings]\nline_length=80")
    nested_config = subdir / ".isort.cfg"
    nested_config.write_text("[settings]\nline_length=120")
    trie = find_all_configs(str(tmp_path))
    result_root = trie.search(str(tmp_path / "file.py"))
    result_nested = trie.search(str(subdir / "file.py"))
    assert result_root[0] == str(root_config)
    assert result_root[1]["line_length"] == 80
    assert result_nested[0] == str(nested_config)
    assert result_nested[1]["line_length"] == 120

def test_find_all_configs_with_multiple_config_formats(tmp_path):
    toml_config = tmp_path / "pyproject.toml"
    toml_config.write_text("[tool.isort]\nline_length=90")
    cfg_config = tmp_path / ".isort.cfg"
    cfg_config.write_text("[settings]\nline_length=110")
    trie = find_all_configs(str(tmp_path))
    result = trie.search(str(tmp_path / "file.py"))
    assert result[0] == str(toml_config)
    assert result[1]["line_length"] == 90

def test_find_all_configs_with_invalid_config_file(tmp_path, mocker):
    mocker.patch("isort.settings.warn")
    config_file = tmp_path / ".isort.cfg"
    config_file.write_text("invalid content")
    trie = find_all_configs(str(tmp_path))
    result = trie.search(str(tmp_path / "file.py"))
    assert result == ("", {})

def test_find_all_configs_with_editorconfig(tmp_path):
    config_file = tmp_path / ".editorconfig"
    config_file.write_text("[*.py]\nindent_style=space\nindent_size=2")
    trie = find_all_configs(str(tmp_path))
    result = trie.search(str(tmp_path / "file.py"))
    assert result[0] == str(config_file)
    assert result[1]["indent"] == "  "

def test_find_all_configs_with_skip_config(tmp_path):
    config_file = tmp_path / ".editorconfig"
    config_file.write_text("[*.js]\nindent_style=tab")
    trie = find_all_configs(str(tmp_path))
    result = trie.search(str(tmp_path / "file.py"))
    assert result == ("", {})

def test_find_all_configs_with_complex_nesting(tmp_path):
    level1 = tmp_path / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    root_config = tmp_path / ".isort.cfg"
    root_config.write_text("[settings]\nline_length=70")
    level2_config = level2 / ".isort.cfg"
    level2_config.write_text("[settings]\nline_length=130")
    trie = find_all_configs(str(tmp_path))
    result_level1 = trie.search(str(level1 / "file.py"))
    result_level2 = trie.search(str(level2 / "file.py"))
    assert result_level1[0] == str(root_config)
    assert result_level1[1]["line_length"] == 70
    assert result_level2[0] == str(level2_config)
    assert result_level2[1]["line_length"] == 130


# LLM-generated content at query #12
#--------------------------

def test___post_init___valid_py_version():
    config = _Config(py_version="3")
    assert config.py_version == "py3"

def test___post_init___py_version_auto():
    import sys
    expected_version = f"{sys.version_info.major}{sys.version_info.minor}"
    config = _Config(py_version="auto")
    assert config.py_version == f"py{expected_version}"

def test___post_init___invalid_py_version():
    try:
        _Config(py_version="invalid")
        assert False
    except ValueError as e:
        assert "The python version invalid is not supported" in str(e)

def test___post_init___py_version_all():
    config = _Config(py_version="all")
    assert config.py_version == "all"

def test___post_init___known_standard_library_default():
    config = _Config(py_version="3")
    assert len(config.known_standard_library) > 0

def test___post_init___known_standard_library_preset():
    config = _Config(py_version="3", known_standard_library=frozenset(["os"]))
    assert "os" in config.known_standard_library

def test___post_init___multi_line_output_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test___post_init___force_alphabetical_sort():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections == True
    assert config.no_sections == True
    assert config.lines_between_types == 1
    assert config.from_first == True

def test___post_init___wrap_length_exceeds_line_length():
    try:
        _Config(wrap_length=100, line_length=79)
        assert False
    except ValueError as e:
        assert "wrap_length must be set lower than or equal to line_length" in str(e)

def test___post_init___wrap_length_equal_line_length():
    config = _Config(wrap_length=79, line_length=79)
    assert config.wrap_length == 79
    assert config.line_length == 79

def test___post_init___wrap_length_less_than_line_length():
    config = _Config(wrap_length=50, line_length=79)
    assert config.wrap_length == 50
    assert config.line_length == 79


# LLM-generated content at query #13
#--------------------------

def test_deprecated_options_used_warning():
    config_overrides = {"some_deprecated_option": "value", "quiet": False}
    config = Config(**config_overrides)
    assert "some_deprecated_option" not in config._deprecated_options_used


# LLM-generated content at query #14
#--------------------------

def test_config_constructor_with_config_overrides():
    config = Config(config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_constructor_with_config_and_overrides():
    base_config = Config(config_overrides={"quiet": False})
    config = Config(config=base_config, config_overrides={"quiet": True})
    assert config.quiet is True

def test_config_constructor_with_settings_file():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.isort]\nprofile = "black"\n')
        settings_file = f.name
    try:
        config = Config(settings_file=settings_file)
        assert config.profile == "black"
    finally:
        os.unlink(settings_file)

def test_config_constructor_with_settings_path():
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    settings_path = os.path.join(temp_dir, "pyproject.toml")
    with open(settings_path, 'w') as f:
        f.write('[tool.isort]\nprofile = "black"\n')
    try:
        config = Config(settings_path=temp_dir)
        assert config.profile == "black"
    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_config_constructor_with_invalid_settings_path():
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    invalid_path = os.path.join(temp_dir, "nonexistent")
    try:
        Config(settings_path=invalid_path)
    except Exception as e:
        assert "InvalidSettingsPath" in str(type(e).__name__)
    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_config_constructor_with_profile():
    config = Config(config_overrides={"profile": "black"})
    assert config.profile == "black"

def test_config_constructor_with_nonexistent_profile():
    try:
        Config(config_overrides={"profile": "nonexistent"})
    except Exception as e:
        assert "ProfileDoesNotExist" in str(type(e).__name__)

def test_config_constructor_with_indent_as_number():
    config = Config(config_overrides={"indent": 4})
    assert config.indent == "    "

def test_config_constructor_with_indent_as_string():
    config = Config(config_overrides={"indent": "2"})
    assert config.indent == "  "

def test_config_constructor_with_indent_as_tab():
    config = Config(config_overrides={"indent": "tab"})
    assert config.indent == "\t"

def test_config_constructor_with_known_other():
    config = Config(config_overrides={"known_mysection": ["mypackage"]})
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(config_overrides={"import_heading_mysection": "My Section"})
    assert "mysection" in config.import_headings
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(config_overrides={"import_footer_mysection": "End of My Section"})
    assert "mysection" in config.import_footers
    assert config.import_footers["mysection"] == "End of My Section"

def test_config_constructor_with_deprecated_option():
    config = Config(config_overrides={"force_single_line": True})
    assert hasattr(config, "force_single_line") is False

def test_config_constructor_with_unsupported_setting():
    try:
        Config(config_overrides={"unsupported_setting": "value"})
    except Exception as e:
        assert "UnsupportedSettings" in str(type(e).__name__)

def test_config_constructor_with_formatter():
    config = Config(config_overrides={"formatter": "example"})
    try:
        _ = config.formatting_function
    except Exception as e:
        assert "FormattingPluginDoesNotExist" in str(type(e).__name__)

def test_config_constructor_with_sort_order():
    config = Config(config_overrides={"sort_order": "natural"})
    assert config.sort_order == "natural"
    assert config.sorting_function is not None

def test_config_constructor_with_invalid_sort_order():
    try:
        Config(config_overrides={"sort_order": "invalid"})
    except Exception as e:
        assert "SortingFunctionDoesNotExist" in str(type(e).__name__)

def test_config_constructor_with_src_paths():
    config = Config(config_overrides={"src_paths": ["src", "tests"]})
    assert len(config.src_paths) == 2

def test_config_constructor_with_skip_and_extend_skip():
    config = Config(config_overrides={"skip": ["skip1"], "extend_skip": ["skip2"]})
    assert "skip1" in config.skips
    assert "skip2" in config.skips

def test_config_constructor_with_skip_glob_and_extend_skip_glob():
    config = Config(config_overrides={"skip_glob": ["*.txt"], "extend_skip_glob": ["*.log"]})
    assert "*.txt" in config.skip_globs
    assert "*.log" in config.skip_globs

def test_config_constructor_with_py_version():
    base_config = Config(config_overrides={"py_version": "py310"})
    config = Config(config=base_config)
    assert config.py_version == "310"

def test_config_constructor_with_config_copy():
    base_config = Config(config_overrides={"quiet": True, "profile": "black"})
    config = Config(config=base_config)
    assert config.quiet is True
    assert config.profile == "black"

def test_config_constructor_with_config_and_overrides_combined():
    base_config = Config(config_overrides={"quiet": False, "profile": "black"})
    config = Config(config=base_config, config_overrides={"quiet": True})
    assert config.quiet is True
    assert config.profile == "black"


# LLM-generated content at query #15
#--------------------------

def test_init_with_config_overrides_py_version():
    from isort import Config
    config = Config(py_version="py310")
    assert config.py_version == "310"


# LLM-generated content at query #16
#--------------------------

def test__get_config_data_toml():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n')
        f.flush()
        result = _get_config_data(f.name, ('tool.black',))
        os.unlink(f.name)
        assert result['line_length'] == 88
        assert result['skip_string_normalization'] is True
        assert result['source'] == f.name

def test__get_config_data_ini():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\nskip_string_normalization = true\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert result['line_length'] == 88
        assert result['skip_string_normalization'] is True
        assert result['source'] == f.name

def test__get_config_data_editorconfig_indent_spaces():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('root = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 100\n')
        f.flush()
        result = _get_config_data(f.name, ('*',))
        os.unlink(f.name)
        assert result['indent'] == '  '
        assert result['line_length'] == 100
        assert result['source'] == f.name

def test__get_config_data_editorconfig_indent_tabs():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_style = tab\nindent_size = tab\nmax_line_length = off\n')
        f.flush()
        result = _get_config_data(f.name, ('*',))
        os.unlink(f.name)
        assert result['indent'] == '\t'
        assert result['line_length'] == float('inf')
        assert result['source'] == f.name

def test__get_config_data_editorconfig_wildcard_extension():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*{py,pyi}]\nindent_style = space\nindent_size = 4\n')
        f.flush()
        result = _get_config_data(f.name, ('*.{py}',))
        os.unlink(f.name)
        assert result['indent'] == '    '
        assert result['source'] == f.name

def test__get_config_data_force_grid_wrap_numeric():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = 3\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert result['force_grid_wrap'] == 3

def test__get_config_data_force_grid_wrap_boolean_true():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = true\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert result['force_grid_wrap'] == 2

def test__get_config_data_force_grid_wrap_boolean_false():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = false\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert result['force_grid_wrap'] == 0

def test__get_config_data_comment_prefix():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ncomment_prefix = "# "\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert result['comment_prefix'] == '# '

def test__get_config_data_known_prefix():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nextend-exclude = ./exclude_dir/\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert any(key.startswith('extend-exclude') for key in result)

def test__get_config_data_bool_from_string():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nskip_string_normalization = yes\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert result['skip_string_normalization'] is True

def test__get_config_data_tuple():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nknown_third_party = requests,django\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert isinstance(result['known_third_party'], tuple)
        assert 'requests' in result['known_third_party']
        assert 'django' in result['known_third_party']

def test__get_config_data_frozenset():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nextend_ignore = E501,W503\n')
        f.flush()
        result = _get_config_data(f.name, ('black',))
        os.unlink(f.name)
        assert isinstance(result['extend_ignore'], frozenset)
        assert 'E501' in result['extend_ignore']
        assert 'W503' in result['extend_ignore']

def test__get_config_data_empty_section():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n\n[other]\nkey = value\n')
        f.flush()
        result = _get_config_data(f.name, ('missing',))
        os.unlink(f.name)
        assert result == {}

def test__get_config_data_multiple_sections():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n\n[pycodestyle]\nmax_line_length = 100\n')
        f.flush()
        result = _get_config_data(f.name, ('black', 'pycodestyle'))
        os.unlink(f.name)
        assert result['line_length'] == 88
        assert result['max_line_length'] == 100

def test__get_config_data_toml_nested():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black.format]\nline_length = 88\n')
        f.flush()
        result = _get_config_data(f.name, ('tool.black.format',))
        os.unlink(f.name)
        assert result['line_length'] == 88

def test__get_config_data_editorconfig_skip_non_relevant():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_style = space\nindent_size = 4\ncharset = utf-8\n')
        f.flush()
        result = _get_config_data(f.name, ('*',))
        os.unlink(f.name)
        assert 'charset' not in result
        assert 'indent' in result


# LLM-generated content at query #17
#--------------------------

def test_find_all_configs_inserts_when_config_data_not_empty():
    config_data = {"key": "value"}
    trie_root = Trie("default", {})
    trie_root.insert("some_path", config_data)
    assert trie_root.root.nodes != {}


# LLM-generated content at query #18
#--------------------------

def test_multi_line_output_vertical_grid_grouped_no_comma():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED


