####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_config_constructor_with_config_argument():
    base_config = Config()
    new_config = Config(config=base_config)
    assert new_config.directory == base_config.directory

def test_config_constructor_with_combined_config():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other_section():
    config = Config(known_mysection=["mypackage"])
    assert "mysection" in config.known_other

def test_config_constructor_with_import_headings():
    config = Config(import_heading_mysection="My Section")
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_mysection="End of My Section")
    assert config.import_footers["mysection"] == "End of My Section"

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert len(config.src_paths) == 2

def test_config_constructor_with_formatter():
    config = Config(formatter="color_output")
    assert config.formatting_function is not None

def test_config_constructor_with_deprecated_options():
    config = Config(force_alphabetical_sort=False)
    assert config.force_alphabetical_sort is False

def test_config_constructor_with_unsupported_settings():
    try:
        config = Config(unsupported_option="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_py_version_override():
    base_config = Config(py_version="py310")
    new_config = Config(config=base_config)
    assert new_config.py_version == "310"

def test_config_constructor_with_skip_gitignore():
    config = Config(skip_gitignore=True)
    assert config.skip_gitignore is True

def test_config_constructor_with_sort_order():
    config = Config(sort_order="natural")
    assert config.sort_order == "natural"

def test_config_constructor_with_empty_settings_file():
    config = Config(settings_file="nonexistent.toml")
    assert config.directory is not None

def test_config_constructor_with_invalid_settings_path():
    try:
        config = Config(settings_path="/nonexistent/path")
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


# LLM-generated content at query #2
#--------------------------

def test_find_config_with_valid_toml_file():
    test_dir = "/tmp/test_config"
    os.makedirs(test_dir, exist_ok=True)
    config_content = b"[tool.isort]\nline_length = 100\n"
    config_file = os.path.join(test_dir, "pyproject.toml")
    with open(config_file, "wb") as f:
        f.write(config_content)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data["line_length"] == 100
    assert config_data["source"] == config_file
    os.remove(config_file)
    os.rmdir(test_dir)

def test_find_config_with_valid_editorconfig_file():
    test_dir = "/tmp/test_config"
    os.makedirs(test_dir, exist_ok=True)
    config_content = "[*.py]\nindent_style = space\nindent_size = 2\n"
    config_file = os.path.join(test_dir, ".editorconfig")
    with open(config_file, "w") as f:
        f.write(config_content)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data["indent"] == "  "
    assert config_data["source"] == config_file
    os.remove(config_file)
    os.rmdir(test_dir)

def test_find_config_with_valid_ini_file():
    test_dir = "/tmp/test_config"
    os.makedirs(test_dir, exist_ok=True)
    config_content = "[isort]\nline_length = 120\n"
    config_file = os.path.join(test_dir, ".isort.cfg")
    with open(config_file, "w") as f:
        f.write(config_content)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data["line_length"] == 120
    assert config_data["source"] == config_file
    os.remove(config_file)
    os.rmdir(test_dir)

def test_find_config_with_no_config_file():
    test_dir = "/tmp/test_no_config"
    os.makedirs(test_dir, exist_ok=True)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data == {}
    os.rmdir(test_dir)

def test_find_config_with_stop_directory():
    test_dir = "/tmp/test_stop"
    os.makedirs(test_dir, exist_ok=True)
    stop_dir = os.path.join(test_dir, ".git")
    os.makedirs(stop_dir, exist_ok=True)
    config_file = os.path.join(test_dir, ".isort.cfg")
    with open(config_file, "w") as f:
        f.write("[isort]\nline_length = 80\n")
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data == {}
    os.remove(config_file)
    os.rmdir(stop_dir)
    os.rmdir(test_dir)

def test_find_config_with_max_search_depth():
    test_dir = "/tmp/test_depth"
    os.makedirs(test_dir, exist_ok=True)
    sub_dir = os.path.join(test_dir, "sub1", "sub2", "sub3", "sub4", "sub5", "sub6")
    os.makedirs(sub_dir, exist_ok=True)
    config_file = os.path.join(test_dir, ".isort.cfg")
    with open(config_file, "w") as f:
        f.write("[isort]\nline_length = 90\n")
    result_dir, config_data = _find_config(sub_dir)
    assert result_dir == test_dir
    assert config_data["line_length"] == 90
    os.remove(config_file)
    for root, dirs, files in os.walk(test_dir, topdown=False):
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(test_dir)

def test_find_config_with_invalid_config_file():
    test_dir = "/tmp/test_invalid"
    os.makedirs(test_dir, exist_ok=True)
    config_file = os.path.join(test_dir, "pyproject.toml")
    with open(config_file, "w") as f:
        f.write("invalid toml content")
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data == {}
    os.remove(config_file)
    os.rmdir(test_dir)

def test_find_config_with_extension_specific_editorconfig():
    test_dir = "/tmp/test_ext"
    os.makedirs(test_dir, exist_ok=True)
    config_content = "[*.{py}]\nindent_style = tab\nindent_size = 4\n"
    config_file = os.path.join(test_dir, ".editorconfig")
    with open(config_file, "w") as f:
        f.write(config_content)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data["indent"] == "\t\t\t\t"
    os.remove(config_file)
    os.rmdir(test_dir)

def test_find_config_with_multiple_extension_editorconfig():
    test_dir = "/tmp/test_multi_ext"
    os.makedirs(test_dir, exist_ok=True)
    config_content = "[*.{py,js}]\nmax_line_length = 120\n"
    config_file = os.path.join(test_dir, ".editorconfig")
    with open(config_file, "w") as f:
        f.write(config_content)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data["line_length"] == 120
    os.remove(config_file)
    os.rmdir(test_dir)

def test_find_config_with_off_max_line_length():
    test_dir = "/tmp/test_off"
    os.makedirs(test_dir, exist_ok=True)
    config_content = "[*.py]\nmax_line_length = off\n"
    config_file = os.path.join(test_dir, ".editorconfig")
    with open(config_file, "w") as f:
        f.write(config_content)
    result_dir, config_data = _find_config(test_dir)
    assert result_dir == test_dir
    assert config_data["line_length"] == float("inf")
    os.remove(config_file)
    os.rmdir(test_dir)


# LLM-generated content at query #3
#--------------------------

def test_config_initialization_without_settings_file_or_path():
    config = Config()
    assert config._known_patterns is None
    assert config._section_comments is None
    assert config._section_comments_end is None
    assert config._skips is None
    assert config._skip_globs is None
    assert config._sorting_function is None


# LLM-generated content at query #4
#--------------------------

def test__get_config_data_with_toml_file():
    import os
    import tempfile
    toml_content = """
[tool.black]
line_length = 88
skip_string_normalization = true
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = f.name
    try:
        result = _get_config_data(toml_path, ('tool.black',))
        assert result['line_length'] == 88
        assert result['skip_string_normalization'] is True
        assert result['source'] == toml_path
    finally:
        os.unlink(toml_path)

def test__get_config_data_with_ini_file():
    import os
    import tempfile
    ini_content = """
[black]
line_length = 100
skip_string_normalization = false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['line_length'] == 100
        assert result['skip_string_normalization'] is False
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_editorconfig_file():
    import os
    import tempfile
    editorconfig_content = """
root = true

[*]
indent_style = space
indent_size = 2
max_line_length = 80
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(editorconfig_content)
        editorconfig_path = f.name
    try:
        result = _get_config_data(editorconfig_path, ('*',))
        assert result['indent'] == '  '
        assert result['line_length'] == 80
        assert result['source'] == editorconfig_path
    finally:
        os.unlink(editorconfig_path)

def test__get_config_data_with_editorconfig_file_tab_indent():
    import os
    import tempfile
    editorconfig_content = """
[*]
indent_style = tab
tab_width = 4
max_line_length = off
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(editorconfig_content)
        editorconfig_path = f.name
    try:
        result = _get_config_data(editorconfig_path, ('*',))
        assert result['indent'] == '\t\t\t\t'
        assert result['line_length'] == float('inf')
        assert result['source'] == editorconfig_path
    finally:
        os.unlink(editorconfig_path)

def test__get_config_data_with_editorconfig_file_extension_section():
    import os
    import tempfile
    editorconfig_content = """
[*.{py,pyi}]
indent_size = 4
line_length = 79
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(editorconfig_content)
        editorconfig_path = f.name
    try:
        result = _get_config_data(editorconfig_path, ('*.{py}',))
        assert result['indent'] == '    '
        assert result['line_length'] == 79
        assert result['source'] == editorconfig_path
    finally:
        os.unlink(editorconfig_path)

def test__get_config_data_with_force_grid_wrap_string():
    import os
    import tempfile
    ini_content = """
[black]
force_grid_wrap = 3
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['force_grid_wrap'] == 3
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_force_grid_wrap_boolean_backwards_compat():
    import os
    import tempfile
    ini_content = """
[black]
force_grid_wrap = false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['force_grid_wrap'] == 0
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_comment_prefix():
    import os
    import tempfile
    ini_content = """
[black]
comment_prefix = "# "
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['comment_prefix'] == '# '
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_known_prefix():
    import os
    import tempfile
    ini_content = """
[black]
known_third_party = requests,django
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert isinstance(result['known_third_party'], frozenset)
        assert 'requests' in result['known_third_party']
        assert 'django' in result['known_third_party']
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_tuple_type():
    import os
    import tempfile
    ini_content = """
[black]
pyproject_include = a,b,c
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert isinstance(result['pyproject_include'], tuple)
        assert result['pyproject_include'] == ('a', 'b', 'c')
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_empty_settings():
    import os
    import tempfile
    ini_content = """
[other_section]
key = value
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result == {}
    finally:
        os.unlink(ini_path)


# LLM-generated content at query #5
#--------------------------

def test_as_list_with_list_input():
    result = _as_list(["a", " b ", "c"])
    assert result == ["a", "b", "c"]

def test_as_list_with_string_single_item():
    result = _as_list("hello")
    assert result == ["hello"]

def test_as_list_with_string_multiple_comma():
    result = _as_list("a,b,c")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_multiple_comma_spaces():
    result = _as_list(" a , b , c ")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_newline_separator():
    result = _as_list("a\nb\nc")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_mixed_separators():
    result = _as_list("a,b\nc")
    assert result == ["a", "b", "c"]

def test_as_list_with_empty_string():
    result = _as_list("")
    assert result == []

def test_as_list_with_only_commas_and_spaces():
    result = _as_list(" , , ")
    assert result == []

def test_as_list_with_empty_list():
    result = _as_list([])
    assert result == []

def test_as_list_with_list_containing_empty_strings():
    result = _as_list(["", "a", " ", "b"])
    assert result == ["a", "b"]


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_27_true_for_editorconfig_with_wildcard_extension():
    import os
    import tempfile
    test_content = "[*.{py,js}]\nindent_style = space\nindent_size = 4\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(test_content)
        f.flush()
        file_path = f.name
    try:
        sections = ("*.{py}",)
        result = _get_config_data(file_path, sections)
        assert "indent_style" not in result
        assert result.get("indent") == "    "
    finally:
        os.unlink(file_path)


# LLM-generated content at query #7
#--------------------------

def test_config_constructor_with_config_parameter():
    mock_config = _Config()
    mock_config.py_version = "py310"
    config = Config(config=mock_config)
    assert config.py_version == "310"

def test_config_constructor_with_config_overrides():
    mock_config = _Config()
    mock_config.py_version = "py310"
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
    try:
        config = Config(settings_path="/nonexistent/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_profile_override():
    config = Config(profile="black")
    assert config.profile == "black"

def test_config_constructor_with_nonexistent_profile():
    try:
        config = Config(profile="nonexistent")
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
    config = Config(known_mysection=["mypackage"])
    assert "mysection" in config.known_other
    assert "mypackage" in config.known_other["mysection"]

def test_config_constructor_with_import_headings():
    config = Config(import_heading_mysection="My Section")
    assert config.import_headings["mysection"] == "My Section"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_mysection="Footer")
    assert config.import_footers["mysection"] == "Footer"

def test_config_constructor_with_unsupported_setting():
    try:
        config = Config(unsupported_setting="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_deprecated_option():
    config = Config(force_sort_within_sections=True, quiet=True)
    assert "force_sort_within_sections" not in dir(config)

def test_config_constructor_with_formatter_plugin():
    config = Config(formatter="example")
    assert config.formatting_function is not None

def test_config_constructor_with_nonexistent_formatter():
    try:
        config = Config(formatter="nonexistent")
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
    try:
        config = Config(sort_order="custom")
        assert False
    except SortingFunctionDoesNotExist:
        assert True

def test_config_constructor_with_skip_gitignore():
    config = Config(skip_gitignore=True)
    assert config.skip_gitignore is True

def test_config_constructor_with_skips():
    config = Config(skip=["venv"], extend_skip=["tests"])
    assert "venv" in config.skips
    assert "tests" in config.skips

def test_config_constructor_with_skip_globs():
    config = Config(skip_glob=["*.pyc"], extend_skip_glob=["__pycache__"])
    assert "*.pyc" in config.skip_globs
    assert "__pycache__" in config.skip_globs


# LLM-generated content at query #8
#--------------------------

def test_profile_name_not_in_profiles_and_plugin_exists():
    from importlib.metadata import entry_points
    mock_profiles = {}
    mock_plugin_name = "test_profile"
    mock_plugin = type("MockPlugin", (), {"load": lambda: {}})
    mock_entry_points = lambda group: [type("MockEntryPoint", (), {"name": mock_plugin_name, "load": mock_plugin.load})]
    original_entry_points = entry_points
    entry_points = mock_entry_points
    config = Config(profile=mock_plugin_name)
    entry_points = original_entry_points


# LLM-generated content at query #9
#--------------------------

def test_directory_not_in_combined_config_without_config_settings_source():
    config = Config(settings_file="", settings_path="", config=None)
    assert "directory" not in config._combined_config


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_14_evaluates_to_true():
    from unittest.mock import mock_open, patch
    file_path = "test.editorconfig"
    sections = ("*.py",)
    mock_file_content = "[*.py]\nindent_style = space\nindent_size = 4\n"
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        result = _get_config_data(file_path, sections)
    assert "indent" in result
    assert result["indent"] == "    "


# LLM-generated content at query #11
#--------------------------

def test__get_config_data_toml():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n')
        toml_path = f.name
    result = _get_config_data(toml_path, ('tool.black',))
    os.unlink(toml_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == toml_path

def test__get_config_data_ini():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\nskip_string_normalization = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == ini_path

def test__get_config_data_editorconfig_indent_spaces():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('root = true\n\n[*]\nindent_style = space\nindent_size = 2\nmax_line_length = 100\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '  '
    assert result['line_length'] == 100
    assert result['source'] == editorconfig_path

def test__get_config_data_editorconfig_indent_tabs():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_style = tab\ntab_width = 4\nmax_line_length = off\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '\t\t\t\t'
    assert result['line_length'] == float('inf')
    assert result['source'] == editorconfig_path

def test__get_config_data_editorconfig_wildcard_extension():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*{py,pyi}]\nline_length = 120\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.{py}',))
    os.unlink(editorconfig_path)
    assert result['line_length'] == 120
    assert result['source'] == editorconfig_path

def test__get_config_data_force_grid_wrap_numeric():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = 3\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 3

def test__get_config_data_force_grid_wrap_boolean_true():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 2

def test__get_config_data_force_grid_wrap_boolean_false():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = false\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 0

def test__get_config_data_comment_prefix_stripping():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ncomment_prefix = "# "\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['comment_prefix'] == '# '

def test__get_config_data_known_prefix_abspaths():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nextend-exclude = foo/, bar/\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    expected_paths = {os.path.join(os.path.dirname(ini_path), 'foo/'), os.path.join(os.path.dirname(ini_path), 'bar/')}
    assert set(result['extend-exclude']) == expected_paths

def test__get_config_data_bool_conversion():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nskip_magic_trailing_comma = yes\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['skip_magic_trailing_comma'] is True

def test__get_config_data_tuple_conversion():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nknown_third_party = requests, pytest\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['known_third_party'] == ('requests', 'pytest')

def test__get_config_data_frozenset_conversion():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\npyink_extensions = foo, bar\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['pyink_extensions'] == frozenset({'foo', 'bar'})

def test__get_config_data_empty_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_multiple_sections():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n\n[pyink]\nline_length = 120\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black', 'pyink'))
    os.unlink(ini_path)
    assert result['line_length'] == 120

def test__get_config_data_nested_toml():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black.format]\nline_length = 88\n')
        toml_path = f.name
    result = _get_config_data(toml_path, ('tool.black.format',))
    os.unlink(toml_path)
    assert result['line_length'] == 88
    assert result['source'] == toml_path


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_is_skipped_skips_exact_path_match():
    config = Config(skips=["my_file.py"])
    file_path = Path("my_file.py")
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_skips_exact_normalized_path():
    config = Config(skips=["my_file.py"])
    file_path = Path("some_folder") / ".." / "my_file.py"
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_skips_parent_folder():
    config = Config(skips=["my_folder"])
    file_path = Path("my_folder") / "nested" / "file.py"
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_skips_by_glob():
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("module.pyc")
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_skips_by_glob_with_leading_slash():
    config = Config(skip_glob=["*.pyc"])
    file_path = Path("subdir") / "module.pyc"
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_does_not_skip_when_no_match():
    config = Config(skips=["other.py"], skip_glob=["*.pyc"])
    file_path = Path("my_file.py")
    result = config.is_skipped(file_path)
    assert result is False

def test_is_skipped_skips_non_existent_path():
    config = Config()
    file_path = Path("non_existent_file.py")
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_skips_gitignored_file_when_skip_gitignore_true():
    config = Config(skip_gitignore=True)
    file_path = Path("ignored.py")
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_does_not_skip_gitignored_file_when_skip_gitignore_false():
    config = Config(skip_gitignore=False)
    file_path = Path("ignored.py")
    result = config.is_skipped(file_path)
    assert result is False

def test_is_skipped_skips_dot_git_folder():
    config = Config(skip_gitignore=True)
    file_path = Path(".git")
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_uses_extend_skip():
    config = Config(skip={"skip1"}, extend_skip={"skip2"})
    file_path = Path("skip2")
    result = config.is_skipped(file_path)
    assert result is True

def test_is_skipped_uses_extend_skip_glob():
    config = Config(skip_glob={"*.tmp"}, extend_skip_glob={"*.log"})
    file_path = Path("file.log")
    result = config.is_skipped(file_path)
    assert result is True


# LLM-generated content at query #2
#--------------------------

def test_config_constructor_with_config_overrides():
    config = Config(quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_file():
    config = Config(settings_file="test.ini")
    assert config.directory == os.getcwd()

def test_config_constructor_with_settings_path():
    config = Config(settings_path=os.getcwd())
    assert config.directory == os.getcwd()

def test_config_constructor_with_config_argument():
    base_config = Config()
    config = Config(config=base_config)
    assert config.py_version == base_config.py_version

def test_config_constructor_with_profile():
    config = Config(profile="black")
    assert "black" in str(config.sources)

def test_config_constructor_with_indent_as_number():
    config = Config(indent=4)
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other():
    config = Config(known_custom=["mypackage"])
    assert "custom" in config.known_other

def test_config_constructor_with_import_headings():
    config = Config(import_heading_stdlib="Standard Library")
    assert config.import_headings["stdlib"] == "Standard Library"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_stdlib="End Standard Library")
    assert config.import_footers["stdlib"] == "End Standard Library"

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert any("src" in str(path) for path in config.src_paths)

def test_config_constructor_with_formatter():
    config = Config(formatter="color_output")
    assert config.formatting_function is not None

def test_config_constructor_with_deprecated_options():
    config = Config(force_alphabetical_sort=True)
    assert "force_alphabetical_sort" not in config.__dict__

def test_config_constructor_with_unsupported_settings():
    try:
        config = Config(unsupported_option="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_combined_config():
    config = Config(known_third_party=["requests"], known_first_party=["mylib"])
    assert "requests" in config.known_third_party
    assert "mylib" in config.known_first_party

def test_config_constructor_with_sections_and_known_other_mismatch():
    config = Config(sections=["CUSTOM"], known_custom=[])
    assert "CUSTOM" in config.sections

def test_config_constructor_with_directory_override():
    config = Config(directory="/tmp")
    assert config.directory == "/tmp"

def test_config_constructor_with_quiet_false_and_warnings():
    config = Config(quiet=False, known_custom=["test"])
    assert config.quiet is False

def test_config_constructor_with_empty_settings_file():
    config = Config(settings_file="empty.ini")
    assert config.directory == os.getcwd()

def test_config_constructor_with_invalid_settings_path():
    try:
        config = Config(settings_path="/nonexistent/path")
        assert False
    except InvalidSettingsPath:
        assert True

def test_config_constructor_with_profile_does_not_exist():
    try:
        config = Config(profile="nonexistent")
        assert False
    except ProfileDoesNotExist:
        assert True

def test_config_constructor_with_formatting_plugin_does_not_exist():
    try:
        config = Config(formatter="nonexistent")
        assert False
    except FormattingPluginDoesNotExist:
        assert True

def test_config_constructor_with_sorting_function_does_not_exist():
    try:
        config = Config(sort_order="nonexistent")
        assert False
    except SortingFunctionDoesNotExist:
        assert True


# LLM-generated content at query #3
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
    mock_open = unittest.mock.mock_open(read_data=b"#!/usr/bin/env python\n")
    with unittest.mock.patch("builtins.open", mock_open):
        result = config.is_supported_filetype("script")
    assert result is True

def test_is_supported_filetype_with_unknown_extension_and_no_shebang():
    config = Config()
    config.supported_extensions = set()
    config.blocked_extensions = set()
    mock_open = unittest.mock.mock_open(read_data=b"print('Hello')")
    with unittest.mock.patch("builtins.open", mock_open):
        result = config.is_supported_filetype("script")
    assert result is False

def test_is_supported_filetype_with_editor_backup_file():
    config = Config()
    result = config.is_supported_filetype("test.py~")
    assert result is False

def test_is_supported_filetype_with_fifo_file():
    config = Config()
    mock_stat = unittest.mock.Mock()
    mock_stat.st_mode = stat.S_IFIFO
    with unittest.mock.patch("os.stat", return_value=mock_stat):
        result = config.is_supported_filetype("fifo")
    assert result is False

def test_is_supported_filetype_with_file_open_error():
    config = Config()
    with unittest.mock.patch("builtins.open", side_effect=OSError):
        result = config.is_supported_filetype("missing.txt")
    assert result is False


# LLM-generated content at query #4
#--------------------------

def test_config_constructor_with_config_overrides():
    config = Config(quiet=True)
    assert config.quiet is True

def test_config_constructor_with_settings_file():
    config = Config(settings_file="test.ini")
    assert config.directory == os.getcwd()

def test_config_constructor_with_settings_path():
    config = Config(settings_path=".")
    assert config.directory == os.getcwd()

def test_config_constructor_with_config_object():
    base_config = Config()
    new_config = Config(config=base_config, quiet=True)
    assert new_config.quiet is True

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
    config = Config(known_custom=["mypackage"])
    assert "custom" in config.known_other

def test_config_constructor_with_import_headings():
    config = Config(import_heading_custom="Custom Imports")
    assert config.import_headings["custom"] == "Custom Imports"

def test_config_constructor_with_import_footers():
    config = Config(import_footer_custom="End Custom Imports")
    assert config.import_footers["custom"] == "End Custom Imports"

def test_config_constructor_with_src_paths():
    config = Config(src_paths=["src", "tests"])
    assert len(config.src_paths) == 2

def test_config_constructor_with_formatter():
    config = Config(formatter="color")
    assert callable(config.formatting_function)

def test_config_constructor_with_deprecated_options():
    config = Config(force_alphabetical_sort=False)
    assert config.force_alphabetical_sort is False

def test_config_constructor_with_unsupported_settings():
    try:
        config = Config(unsupported_option="value")
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_with_empty_settings_file():
    config = Config(settings_file="empty.ini")
    assert config.directory == os.getcwd()

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

def test_config_constructor_with_known_section_mapping():
    config = Config(known_standard_library=["os"])
    assert "os" in config.known_standard_library

def test_config_constructor_with_sections_and_known_other_mismatch():
    config = Config(sections=("CUSTOM",), known_custom=["mypackage"])
    assert "custom" in config.known_other

def test_config_constructor_with_combined_config_sources():
    config = Config(settings_file="test.ini", quiet=True, profile="black")
    assert config.quiet is True
    assert config.profile == "black"

def test_config_constructor_with_py_version_override():
    base_config = Config(py_version="py310")
    new_config = Config(config=base_config)
    assert new_config.py_version == "310"


# LLM-generated content at query #5
#--------------------------

def test_profile_name_not_in_profiles_and_plugin_entry_points_exist():
    from unittest.mock import MagicMock, patch
    mock_plugin = MagicMock()
    mock_plugin.name = "test_profile"
    mock_plugin.load.return_value = {"some": "config"}
    with patch("isort.Config.profiles", {}):
        with patch("importlib.metadata.entry_points", return_value=[mock_plugin]):
            config = Config(profile="test_profile")
            assert "test_profile" in profiles
            assert profiles["test_profile"] == {"some": "config"}


# LLM-generated content at query #6
#--------------------------

def test___post_init___valid_py_version_auto():
    import sys
    original_version_info = sys.version_info
    sys.version_info = type('version_info', (), {'major': 3, 'minor': 8})()
    config = _Config(py_version="auto")
    assert config.py_version == "py38"
    sys.version_info = original_version_info

def test___post_init___valid_py_version_all():
    config = _Config(py_version="all")
    assert config.py_version == "all"

def test___post_init___valid_py_version_specific():
    config = _Config(py_version="310")
    assert config.py_version == "py310"

def test___post_init___invalid_py_version_raises_value_error():
    try:
        _Config(py_version="99")
    except ValueError as e:
        assert "The python version 99 is not supported" in str(e)

def test___post_init___known_standard_library_populated_when_empty():
    config = _Config(py_version="py38")
    assert len(config.known_standard_library) > 0

def test___post_init___multi_line_output_vertical_grid_grouped_no_comma_converted():
    config = _Config(multi_line_output=WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA)
    assert config.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED

def test___post_init___force_alphabetical_sort_sets_related_attributes():
    config = _Config(force_alphabetical_sort=True)
    assert config.force_alphabetical_sort_within_sections == True
    assert config.no_sections == True
    assert config.lines_between_types == 1
    assert config.from_first == True

def test___post_init___wrap_length_greater_than_line_length_raises_value_error():
    try:
        _Config(wrap_length=100, line_length=79)
    except ValueError as e:
        assert "wrap_length must be set lower than or equal to line_length" in str(e)

def test___post_init___wrap_length_equal_to_line_length_no_error():
    config = _Config(wrap_length=79, line_length=79)
    assert config.wrap_length == 79
    assert config.line_length == 79

def test___post_init___wrap_length_less_than_line_length_no_error():
    config = _Config(wrap_length=50, line_length=79)
    assert config.wrap_length == 50
    assert config.line_length == 79


# LLM-generated content at query #7
#--------------------------

def test_is_supported_filetype_os_error_occurs():
    config = Config()
    config.supported_extensions = frozenset()
    config.blocked_extensions = frozenset()
    result = config.is_supported_filetype("test_file.txt")
    assert result == False


# LLM-generated content at query #8
#--------------------------

def test_find_config_finds_toml_file():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.black]\nline_length = 100")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 100
        assert result_data["source"] == config_path

def test_find_config_finds_editorconfig_file():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, ".editorconfig")
        with open(config_path, "w") as f:
            f.write("[*.py]\nindent_style = space\nindent_size = 2")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["indent"] == "  "
        assert result_data["source"] == config_path

def test_find_config_finds_cfg_file():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "setup.cfg")
        with open(config_path, "w") as f:
            f.write("[tool:black]\nline_length = 88")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 88
        assert result_data["source"] == config_path

def test_find_config_finds_ini_file():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.black]\nskip_string_normalization = true")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data["skip_string_normalization"] is True
        assert result_data["source"] == config_path

def test_find_config_stops_at_stop_dir():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        stop_dir = os.path.join(subdir, ".git")
        os.makedirs(stop_dir)
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.black]\nline_length = 100")
        result_dir, result_data = _find_config(subdir)
        assert result_dir == subdir
        assert result_data == {}

def test_find_config_searches_upwards():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "a", "b", "c")
        os.makedirs(subdir)
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("[tool.black]\nline_length = 120")
        result_dir, result_data = _find_config(subdir)
        assert result_dir == tmpdir
        assert result_data["line_length"] == 120
        assert result_data["source"] == config_path

def test_find_config_limits_search_depth():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        current = tmpdir
        for i in range(10):
            current = os.path.join(current, str(i))
            os.makedirs(current)
        result_dir, result_data = _find_config(current)
        assert result_dir == current
        assert result_data == {}

def test_find_config_handles_invalid_config_gracefully():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "pyproject.toml")
        with open(config_path, "w") as f:
            f.write("invalid toml content !@#$%")
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_returns_empty_dict_when_no_config():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result_dir, result_data = _find_config(tmpdir)
        assert result_dir == tmpdir
        assert result_data == {}

def test_find_config_prefers_closer_config():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        root_config = os.path.join(tmpdir, "pyproject.toml")
        with open(root_config, "w") as f:
            f.write("[tool.black]\nline_length = 100")
        sub_config = os.path.join(subdir, "pyproject.toml")
        with open(sub_config, "w") as f:
            f.write("[tool.black]\nline_length = 80")
        result_dir, result_data = _find_config(subdir)
        assert result_dir == subdir
        assert result_data["line_length"] == 80
        assert result_data["source"] == sub_config


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_3_evaluates_to_false():
    config = _Config()
    test_config = Config(config=config)
    assert test_config._known_patterns is None
    assert test_config._section_comments is None
    assert test_config._section_comments_end is None
    assert test_config._skips is None
    assert test_config._skip_globs is None
    assert test_config._sorting_function is None


# LLM-generated content at query #10
#--------------------------

def test_config_constructor_with_config_parameter():
    config_vars = {"quiet": True, "py_version": "py310"}
    config = _Config(**config_vars)
    new_config = Config(config=config)
    assert new_config.quiet == True
    assert new_config.py_version == "310"

def test_config_constructor_with_config_overrides():
    config_vars = {"quiet": False, "py_version": "py39"}
    config = _Config(**config_vars)
    new_config = Config(config=config, quiet=True)
    assert new_config.quiet == True
    assert new_config.py_version == "39"

def test_config_constructor_with_settings_file():
    settings_file = "test.toml"
    config = Config(settings_file=settings_file)
    assert config.directory == os.getcwd()

def test_config_constructor_with_settings_path():
    settings_path = os.getcwd()
    config = Config(settings_path=settings_path)
    assert config.directory == os.getcwd()

def test_config_constructor_with_profile():
    profile_name = "black"
    config = Config(profile=profile_name)
    assert config.profile == profile_name

def test_config_constructor_with_indent_as_digit():
    config = Config(indent="4")
    assert config.indent == "    "

def test_config_constructor_with_indent_as_tab():
    config = Config(indent="tab")
    assert config.indent == "\t"

def test_config_constructor_with_known_other():
    known_other_key = "custom_section"
    known_other_value = ["mypackage"]
    config = Config(**{f"known_{known_other_key}": known_other_value})
    assert config.known_other[known_other_key] == frozenset(known_other_value)

def test_config_constructor_with_import_headings():
    heading_key = "custom_section"
    heading_value = "Custom Section"
    config = Config(**{f"import_heading_{heading_key}": heading_value})
    assert config.import_headings[heading_key] == heading_value

def test_config_constructor_with_import_footers():
    footer_key = "custom_section"
    footer_value = "End of Custom Section"
    config = Config(**{f"import_footer_{footer_key}": footer_value})
    assert config.import_footers[footer_key] == footer_value

def test_config_constructor_with_src_paths():
    src_paths = ["src", "tests"]
    config = Config(src_paths=src_paths)
    assert len(config.src_paths) == 2

def test_config_constructor_with_formatter():
    formatter_name = "custom_formatter"
    config = Config(formatter=formatter_name)
    assert config.formatter == formatter_name

def test_config_constructor_with_deprecated_options():
    deprecated_option = "force_single_line"
    config = Config(**{deprecated_option: True})
    assert deprecated_option not in config.__dict__

def test_config_constructor_with_unsupported_settings():
    unsupported_option = "unsupported_option"
    unsupported_value = "some_value"
    try:
        Config(**{unsupported_option: unsupported_value})
        assert False
    except UnsupportedSettings:
        assert True

def test_config_constructor_without_parameters():
    config = Config()
    assert config.directory == os.getcwd()
    assert config.src_paths == (Path(os.getcwd()) / "src", Path(os.getcwd()))

def test_config_constructor_with_quiet_override():
    config = Config(quiet=True)
    assert config.quiet == True

def test_config_constructor_with_py_version_override():
    config = Config(py_version="py38")
    assert config.py_version == "38"

def test_config_constructor_with_directory_override():
    directory = "/tmp"
    config = Config(directory=directory)
    assert config.directory == directory

def test_config_constructor_with_skip_and_extend_skip():
    skip = {"skip1"}
    extend_skip = {"skip2"}
    config = Config(skip=skip, extend_skip=extend_skip)
    assert config.skips == frozenset({"skip1", "skip2"})

def test_config_constructor_with_skip_glob_and_extend_skip_glob():
    skip_glob = {"*.pyc"}
    extend_skip_glob = {"*.pyo"}
    config = Config(skip_glob=skip_glob, extend_skip_glob=extend_skip_glob)
    assert config.skip_globs == frozenset({"*.pyc", "*.pyo"})

def test_config_constructor_with_sort_order_natural():
    config = Config(sort_order="natural")
    assert config.sorting_function == sorting.naturally

def test_config_constructor_with_sort_order_native():
    config = Config(sort_order="native")
    assert config.sorting_function == sorted

def test_config_constructor_with_custom_sort_order():
    sort_order = "custom"
    try:
        Config(sort_order=sort_order)
        assert False
    except SortingFunctionDoesNotExist:
        assert True


# LLM-generated content at query #11
#--------------------------

def test_formatter_plugin_found():
    mock_plugin = type('MockPlugin', (), {'name': 'test_formatter', 'load': lambda: 'formatting_function'})()
    mock_entry_points = lambda group: [mock_plugin] if group == 'isort.formatters' else []
    combined_config = {'formatter': 'test_formatter'}
    for plugin in mock_entry_points('isort.formatters'):
        if plugin.name == combined_config['formatter']:
            combined_config['formatting_function'] = plugin.load()
            break
    else:
        raise Exception('Plugin not found')
    assert combined_config['formatting_function'] == 'formatting_function'


# LLM-generated content at query #12
#--------------------------

def test__get_config_data_toml():
    import os
    import tempfile
    toml_content = """
[tool.black]
line_length = 88
target_version = ["py37", "py38"]
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        toml_path = f.name
    sections = ("tool.black",)
    result = _get_config_data(toml_path, sections)
    os.unlink(toml_path)
    assert result["line_length"] == 88
    assert result["target_version"] == ("py37", "py38")
    assert result["source"] == toml_path

def test__get_config_data_ini():
    import os
    import tempfile
    ini_content = """
[*.py]
indent = "    "
line_length = 100
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["indent"] == "    "
    assert result["line_length"] == 100
    assert result["source"] == ini_path

def test__get_config_data_editorconfig_indent_spaces():
    import os
    import tempfile
    editorconfig_content = """
root = true

[*]
indent_style = space
indent_size = 2
max_line_length = 80
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(editorconfig_content)
        editorconfig_path = f.name
    sections = ("*",)
    result = _get_config_data(editorconfig_path, sections)
    os.unlink(editorconfig_path)
    assert result["indent"] == "  "
    assert result["line_length"] == 80
    assert result["source"] == editorconfig_path

def test__get_config_data_editorconfig_indent_tabs():
    import os
    import tempfile
    editorconfig_content = """
root = true

[*]
indent_style = tab
tab_width = 4
max_line_length = off
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(editorconfig_content)
        editorconfig_path = f.name
    sections = ("*",)
    result = _get_config_data(editorconfig_path, sections)
    os.unlink(editorconfig_path)
    assert result["indent"] == "\t\t\t\t"
    assert result["line_length"] == float('inf')
    assert result["source"] == editorconfig_path

def test__get_config_data_editorconfig_extension_section():
    import os
    import tempfile
    editorconfig_content = """
root = true

[*.{py,pyi}]
indent_style = space
indent_size = 4
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write(editorconfig_content)
        editorconfig_path = f.name
    sections = ("*.{py}",)
    result = _get_config_data(editorconfig_path, sections)
    os.unlink(editorconfig_path)
    assert result["indent"] == "    "
    assert result["source"] == editorconfig_path

def test__get_config_data_force_grid_wrap_numeric():
    import os
    import tempfile
    ini_content = """
[*.py]
force_grid_wrap = 3
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["force_grid_wrap"] == 3

def test__get_config_data_force_grid_wrap_boolean_false():
    import os
    import tempfile
    ini_content = """
[*.py]
force_grid_wrap = false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["force_grid_wrap"] == 0

def test__get_config_data_force_grid_wrap_boolean_true():
    import os
    import tempfile
    ini_content = """
[*.py]
force_grid_wrap = true
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["force_grid_wrap"] == 2

def test__get_config_data_comment_prefix():
    import os
    import tempfile
    ini_content = """
[*.py]
comment_prefix = "# "
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["comment_prefix"] == "# "

def test__get_config_data_known_prefix():
    import os
    import tempfile
    ini_content = """
[*.py]
known_third_party = requests,django
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert isinstance(result["known_third_party"], frozenset)
    assert "requests" in result["known_third_party"]
    assert "django" in result["known_third_party"]

def test__get_config_data_bool_conversion():
    import os
    import tempfile
    ini_content = """
[*.py]
skip_gitignore = true
multi_line_output = 3
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["skip_gitignore"] is True
    assert result["multi_line_output"] == 3

def test__get_config_data_empty_section():
    import os
    import tempfile
    ini_content = """
[*.py]
line_length = 120
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.c",)
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_multiple_sections():
    import os
    import tempfile
    ini_content = """
[*.py]
line_length = 100

[*.js]
line_length = 80
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
    sections = ("*.py", "*.js")
    result = _get_config_data(ini_path, sections)
    os.unlink(ini_path)
    assert result["line_length"] == 80


# LLM-generated content at query #13
#--------------------------

def test__get_config_data_toml():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n')
        toml_path = f.name
    result = _get_config_data(toml_path, ('tool.black',))
    os.unlink(toml_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == toml_path

def test__get_config_data_ini():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\nskip_string_normalization = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == ini_path

def test__get_config_data_editorconfig():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 88\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.py',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '    '
    assert result['line_length'] == 88
    assert result['source'] == editorconfig_path

def test__get_config_data_editorconfig_tab():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*.py]\nindent_style = tab\nindent_size = 2\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.py',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '\t\t'

def test__get_config_data_editorconfig_line_length_off():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*.py]\nmax_line_length = off\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.py',))
    os.unlink(editorconfig_path)
    assert result['line_length'] == float('inf')

def test__get_config_data_editorconfig_wildcard_extension():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*.{py,pyi}]\nindent_size = 4\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.{py}',))
    os.unlink(editorconfig_path)
    assert result['indent_size'] == 4

def test__get_config_data_force_grid_wrap_backwards_compat_false():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = false\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 0

def test__get_config_data_force_grid_wrap_backwards_compat_true():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 2

def test__get_config_data_comment_prefix():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ncomment_prefix = "# "\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['comment_prefix'] == '# '

def test__get_config_data_known_prefix():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nextend-exclude = foo, bar\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert isinstance(result['extend-exclude'], set)
    assert os.path.join(os.path.dirname(ini_path), 'foo') in result['extend-exclude']
    assert os.path.join(os.path.dirname(ini_path), 'bar') in result['extend-exclude']

def test__get_config_data_bool_conversion():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nskip_magic_trailing_comma = yes\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['skip_magic_trailing_comma'] is True

def test__get_config_data_tuple_conversion():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\npyproject_include = a, b, c\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['pyproject_include'] == ('a', 'b', 'c')

def test__get_config_data_frozenset_conversion():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nrequired_version = 1.0, 2.0\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert isinstance(result['required_version'], frozenset)
    assert '1.0' in result['required_version']
    assert '2.0' in result['required_version']

def test__get_config_data_empty_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_nonexistent_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[other]\nline_length = 100\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_multiple_sections():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n[pyproject]\nline_length = 100\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black', 'pyproject'))
    os.unlink(ini_path)
    assert result['line_length'] == 100


# LLM-generated content at query #14
#--------------------------

def test_comment_prefix_strips_quotes():
    settings = {"comment_prefix": "'test'"}
    key = "comment_prefix"
    value = settings[key]
    result = str(value).strip("'").strip('"')
    assert result == "test"


# LLM-generated content at query #15
#--------------------------

def test_maps_to_section_in_known_section_mapping():
    config = Config(config_overrides={"known_foo": ["bar"]})
    assert "known_foo" not in config._known_patterns


# LLM-generated content at query #16
#--------------------------

def test_as_list_with_list_input():
    result = _as_list(["a", "b", "c"])
    assert result == ["a", "b", "c"]

def test_as_list_with_string_single_item():
    result = _as_list("hello")
    assert result == ["hello"]

def test_as_list_with_string_multiple_comma_separated():
    result = _as_list("a,b,c")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_multiple_newline_separated():
    result = _as_list("a\nb\nc")
    assert result == ["a", "b", "c"]

def test_as_list_with_string_mixed_separators():
    result = _as_list("a,b\nc")
    assert result == ["a", "b", "c"]

def test_as_list_with_whitespace_around_items():
    result = _as_list("  a , b ,  c  ")
    assert result == ["a", "b", "c"]

def test_as_list_with_empty_string():
    result = _as_list("")
    assert result == []

def test_as_list_with_only_commas_and_newlines():
    result = _as_list(",\n,,")
    assert result == []

def test_as_list_with_list_containing_whitespace():
    result = _as_list(["  x  ", " y ", "z"])
    assert result == ["x", "y", "z"]


# LLM-generated content at query #17
#--------------------------

def test_formatter_plugin_found():
    mock_plugin = type('MockPlugin', (), {'name': 'custom_formatter'})()
    mock_entry_points = type('MockEntryPoints', (), {'group': lambda self, group: [mock_plugin] if group == 'isort.formatters' else []})()
    config = Config(formatter='custom_formatter')
    assert config.formatting_function is not None


# LLM-generated content at query #18
#--------------------------

def test_warning_when_settings_file_empty_and_not_quiet():
    import os
    import tempfile

    from isort import Config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp:
        tmp.write('')
        tmp_path = tmp.name
    try:
        config = Config(settings_file=tmp_path, quiet=False)
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #19
#--------------------------

def test__get_config_data_with_toml_file():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.black]\nline_length = 88\nskip_string_normalization = true\n')
        toml_path = f.name
    try:
        result = _get_config_data(toml_path, ('tool.black',))
        assert result['line_length'] == 88
        assert result['skip_string_normalization'] is True
        assert result['source'] == toml_path
    finally:
        os.unlink(toml_path)

def test__get_config_data_with_ini_file():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 100\nskip_string_normalization = false\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['line_length'] == 100
        assert result['skip_string_normalization'] is False
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_editorconfig_file():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('indent_style = space\nindent_size = 2\nmax_line_length = 80\n')
        editorconfig_path = f.name
    try:
        result = _get_config_data(editorconfig_path, ('*',))
        assert result['indent'] == '  '
        assert result['line_length'] == 80
        assert result['source'] == editorconfig_path
    finally:
        os.unlink(editorconfig_path)

def test__get_config_data_with_editorconfig_file_tab_indent():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('indent_style = tab\ntab_width = 4\nmax_line_length = off\n')
        editorconfig_path = f.name
    try:
        result = _get_config_data(editorconfig_path, ('*',))
        assert result['indent'] == '\t\t\t\t'
        assert result['line_length'] == float('inf')
        assert result['source'] == editorconfig_path
    finally:
        os.unlink(editorconfig_path)

def test__get_config_data_with_editorconfig_file_extension_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*]\nindent_style = space\nindent_size = 4\n\n[*.{py,pyi}]\nmax_line_length = 120\n')
        editorconfig_path = f.name
    try:
        result = _get_config_data(editorconfig_path, ('*.{py}',))
        assert result['indent'] == '    '
        assert result['line_length'] == 120
        assert result['source'] == editorconfig_path
    finally:
        os.unlink(editorconfig_path)

def test__get_config_data_with_unknown_setting():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 100\nunknown_setting = value\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['line_length'] == 100
        assert 'unknown_setting' not in result
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_force_grid_wrap_string():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = false\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['force_grid_wrap'] == 0
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_comment_prefix():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ncomment_prefix = "# "\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result['comment_prefix'] == '# '
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_frozenset_type():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nknown_third_party = requests, pytest\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert isinstance(result['known_third_party'], frozenset)
        assert 'requests' in result['known_third_party']
        assert 'pytest' in result['known_third_party']
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_tuple_type():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ninclude = *.py, *.pyi\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert isinstance(result['include'], tuple)
        assert '*.py' in result['include']
        assert '*.pyi' in result['include']
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_known_prefix():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nextra_standard_library = lib1, lib2\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert isinstance(result['extra_standard_library'], set)
        assert any('lib1' in path for path in result['extra_standard_library'])
        assert any('lib2' in path for path in result['extra_standard_library'])
        assert result['source'] == ini_path
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_empty_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result == {}
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_nonexistent_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[other]\nline_length = 100\n')
        ini_path = f.name
    try:
        result = _get_config_data(ini_path, ('black',))
        assert result == {}
    finally:
        os.unlink(ini_path)

def test__get_config_data_with_multiple_sections():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.black]\nline_length = 88\n[tool.isort]\nprofile = black\n')
        toml_path = f.name
    try:
        result = _get_config_data(toml_path, ('tool.black', 'tool.isort'))
        assert result['line_length'] == 88
        assert result['profile'] == 'black'
        assert result['source'] == toml_path
    finally:
        os.unlink(toml_path)

def test__get_config_data_with_nested_toml_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[tool.black.format]\nline_length = 88\n')
        toml_path = f.name
    try:
        result = _get_config_data(toml_path, ('tool.black.format',))
        assert result['line_length'] == 88
        assert result['source'] == toml_path
    finally:
        os.unlink(toml_path)


# LLM-generated content at query #20
#--------------------------

def test__get_config_data_with_toml():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.black]\nline_length = 88\nskip_string_normalization = true\n')
        toml_path = f.name
    result = _get_config_data(toml_path, ('tool.black',))
    os.unlink(toml_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == toml_path

def test__get_config_data_with_ini():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\nskip_string_normalization = true\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['line_length'] == 88
    assert result['skip_string_normalization'] is True
    assert result['source'] == ini_path

def test__get_config_data_with_editorconfig():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('root = true\n\n[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 79\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.py',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '    '
    assert result['line_length'] == 79
    assert result['source'] == editorconfig_path

def test__get_config_data_with_editorconfig_tab_indent():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*.py]\nindent_style = tab\nindent_size = 2\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.py',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '\t\t'

def test__get_config_data_with_editorconfig_off_line_length():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*.py]\nmax_line_length = off\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.py',))
    os.unlink(editorconfig_path)
    assert result['line_length'] == float('inf')

def test__get_config_data_with_editorconfig_extension_wildcard():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.editorconfig', delete=False) as f:
        f.write('[*.{py,pyi}]\nindent_size = 2\n')
        editorconfig_path = f.name
    result = _get_config_data(editorconfig_path, ('*.{py}',))
    os.unlink(editorconfig_path)
    assert result['indent'] == '  '

def test__get_config_data_with_force_grid_wrap_string():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nforce_grid_wrap = false\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['force_grid_wrap'] == 0

def test__get_config_data_with_comment_prefix():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ncomment_prefix = "# "\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result['comment_prefix'] == '# '

def test__get_config_data_with_known_prefix():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nknown_third_party = requests,django\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert isinstance(result['known_third_party'], frozenset)
    assert 'requests' in result['known_third_party']
    assert 'django' in result['known_third_party']

def test__get_config_data_with_tuple_type():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\ninclude = "*.py,*.pyi"\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert isinstance(result['include'], tuple)
    assert '*.py' in result['include']
    assert '*.pyi' in result['include']

def test__get_config_data_with_empty_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_with_nonexistent_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[other]\nline_length = 100\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black',))
    os.unlink(ini_path)
    assert result == {}

def test__get_config_data_with_multiple_sections():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('[black]\nline_length = 88\n[pycodestyle]\nmax_line_length = 79\n')
        ini_path = f.name
    result = _get_config_data(ini_path, ('black', 'pycodestyle'))
    os.unlink(ini_path)
    assert result['line_length'] == 79
    assert result['source'] == ini_path

def test__get_config_data_with_toml_nested_section():
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        f.write(b'[tool.isort]\nprofile = "black"\n')
        toml_path = f.name
    result = _get_config_data(toml_path, ('tool.isort',))
    os.unlink(toml_path)
    assert result['profile'] == 'black'
    assert result['source'] == toml_path


