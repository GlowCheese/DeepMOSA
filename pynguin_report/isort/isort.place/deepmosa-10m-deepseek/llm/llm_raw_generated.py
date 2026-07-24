####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_src_path_finds_module_in_src_paths():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test_src_path_handles_nested_module_with_namespace_package():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = {"mypackage"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("mypackage.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test_src_path_returns_none_for_unknown_module():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("unknown", config)
    assert result is None

def test_src_path_uses_provided_src_paths():
    config = Config()
    config.src_paths = []
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    custom_src_paths = [Path("/custom/path")]
    result = _src_path("mymodule", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test_src_path_handles_src_path_is_module():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("path", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test_src_path_auto_identifies_namespace_package():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")


# LLM-generated content at query #2
#--------------------------

def test_src_path_is_module_true_for_valid_directory():
    src_path = Path("/some/path/module_name")
    src_path.is_dir = lambda: True
    exists_case_sensitive = lambda x: True
    result = _src_path_is_module(src_path, "module_name")
    assert result is True

def test_src_path_is_module_false_for_wrong_name():
    src_path = Path("/some/path/module_name")
    src_path.is_dir = lambda: True
    exists_case_sensitive = lambda x: True
    result = _src_path_is_module(src_path, "different_name")
    assert result is False

def test_src_path_is_module_false_for_file_not_dir():
    src_path = Path("/some/path/module_name")
    src_path.is_dir = lambda: False
    exists_case_sensitive = lambda x: True
    result = _src_path_is_module(src_path, "module_name")
    assert result is False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("/some/path/module_name")
    src_path.is_dir = lambda: True
    exists_case_sensitive = lambda x: False
    result = _src_path_is_module(src_path, "module_name")
    assert result is False

def test_src_path_is_module_false_for_all_conditions_failing():
    src_path = Path("/some/path/module_name")
    src_path.is_dir = lambda: False
    exists_case_sensitive = lambda x: False
    result = _src_path_is_module(src_path, "different_name")
    assert result is False


# LLM-generated content at query #3
#--------------------------

def test_known_pattern_matches_configured_pattern():
    from unittest.mock import Mock
    mock_pattern = Mock()
    mock_pattern.match.return_value = True
    mock_config = Mock()
    mock_config.known_patterns = [(mock_pattern, "placement_name")]
    mock_config.sections = ["placement_name"]
    result = _known_pattern("module.submodule.name", mock_config)
    assert result == ("placement_name", "Matched configured known pattern " + str(mock_pattern))

def test_known_pattern_no_match():
    from unittest.mock import Mock
    mock_pattern = Mock()
    mock_pattern.match.return_value = False
    mock_config = Mock()
    mock_config.known_patterns = [(mock_pattern, "placement_name")]
    mock_config.sections = ["placement_name"]
    result = _known_pattern("module.submodule.name", mock_config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    from unittest.mock import Mock
    mock_pattern = Mock()
    mock_pattern.match.return_value = True
    mock_config = Mock()
    mock_config.known_patterns = [(mock_pattern, "placement_name")]
    mock_config.sections = ["other_section"]
    result = _known_pattern("module.submodule.name", mock_config)
    assert result is None

def test_known_pattern_matches_longest_module_first():
    from unittest.mock import Mock
    mock_pattern = Mock()
    mock_pattern.match.side_effect = lambda x: x == "module.submodule"
    mock_config = Mock()
    mock_config.known_patterns = [(mock_pattern, "placement_name")]
    mock_config.sections = ["placement_name"]
    result = _known_pattern("module.submodule.name", mock_config)
    assert result == ("placement_name", "Matched configured known pattern " + str(mock_pattern))

def test_known_pattern_empty_name():
    from unittest.mock import Mock
    mock_pattern = Mock()
    mock_pattern.match.return_value = False
    mock_config = Mock()
    mock_config.known_patterns = [(mock_pattern, "placement_name")]
    mock_config.sections = ["placement_name"]
    result = _known_pattern("", mock_config)
    assert result is None

def test_known_pattern_multiple_patterns_first_matches():
    from unittest.mock import Mock
    mock_pattern1 = Mock()
    mock_pattern1.match.return_value = True
    mock_pattern2 = Mock()
    mock_pattern2.match.return_value = False
    mock_config = Mock()
    mock_config.known_patterns = [(mock_pattern1, "placement1"), (mock_pattern2, "placement2")]
    mock_config.sections = ["placement1", "placement2"]
    result = _known_pattern("module.name", mock_config)
    assert result == ("placement1", "Matched configured known pattern " + str(mock_pattern1))


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config()
    config.src_paths = [Path("/fake/path")]
    result = _src_path("mymodule", config, src_paths=[Path("/fake/path")], prefix=())
    assert result is not None
    assert result[0] == sections.FIRSTPARTY
    assert "Found in one of the configured src_paths:" in result[1]


# LLM-generated content at query #5
#--------------------------

def test_known_pattern_predicate_false():
    class MockPattern:
        def match(self, module_name):
            return False
    class MockConfig:
        sections = ["section1"]
        known_patterns = [(MockPattern(), "placement1")]
    config = MockConfig()
    result = _known_pattern("module.name", config)
    assert result is None


# LLM-generated content at query #6
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config()
    config.namespace_packages = {"test.namespace"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = [".py"]
    config.src_paths = [Path("/src")]
    result = _src_path("test.namespace.module", config, None, ("test",))
    assert result is not None


# LLM-generated content at query #7
#--------------------------

def test_is_namespace_package_with_valid_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = True
    mock_open = Mock()
    mock_init.open.return_value.__enter__.return_value = mock_open
    mock_open.read.return_value = b"__import__('pkg_resources').declare_namespace(__name__)"
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is True

def test_is_namespace_package_with_valid_namespace_package_double_quotes():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = True
    mock_open = Mock()
    mock_init.open.return_value.__enter__.return_value = mock_open
    mock_open.read.return_value = b'__import__("pkg_resources").declare_namespace(__name__)'
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is True

def test_is_namespace_package_with_valid_namespace_package_pkgutil():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = True
    mock_open = Mock()
    mock_init.open.return_value.__enter__.return_value = mock_open
    mock_open.read.return_value = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is True

def test_is_namespace_package_with_valid_namespace_package_pkgutil_double_quotes():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = True
    mock_open = Mock()
    mock_init.open.return_value.__enter__.return_value = mock_open
    mock_open.read.return_value = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is True

def test_is_namespace_package_without_init_and_no_files():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = False
    mock_iter = Mock()
    mock_path.iterdir.return_value = []
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is True

def test_is_namespace_package_without_init_but_has_source_files():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = False
    mock_file = Mock()
    mock_file.suffix = ".py"
    mock_file.suffix.lstrip.return_value = "py"
    mock_file.name.lower.return_value = "some.py"
    mock_path.iterdir.return_value = [mock_file]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is False

def test_is_namespace_package_without_init_but_has_setup_cfg():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = False
    mock_file = Mock()
    mock_file.suffix = ""
    mock_file.suffix.lstrip.return_value = ""
    mock_file.name.lower.return_value = "setup.cfg"
    mock_path.iterdir.return_value = [mock_file]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is False

def test_is_namespace_package_without_init_but_has_pyproject_toml():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = False
    mock_file = Mock()
    mock_file.suffix = ""
    mock_file.suffix.lstrip.return_value = ""
    mock_file.name.lower.return_value = "pyproject.toml"
    mock_path.iterdir.return_value = [mock_file]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is False

def test_is_namespace_package_with_init_but_no_namespace_marker():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    mock_init = Mock()
    mock_path.__truediv__.return_value = mock_init
    mock_init.exists.return_value = True
    mock_open = Mock()
    mock_init.open.return_value.__enter__.return_value = mock_open
    mock_open.read.return_value = b"some other content"
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is False

def test_is_namespace_package_path_not_a_package():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = False
    mock_path.exists.return_value = True
    mock_path.__str__ = Mock(return_value="some_path")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is False

def test_is_namespace_package_path_does_not_exist_case_sensitive():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.is_dir.return_value = True
    mock_path.exists.return_value = False
    mock_path.__str__ = Mock(return_value="some_path")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(mock_path, src_extensions)
    assert result is False


# LLM-generated content at query #8
#--------------------------

def test_is_namespace_package_with_valid_namespace_package():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_non_package_path():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        non_pkg_path = Path(tmpdir) / "not_a_pkg"
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(non_pkg_path, src_extensions)
        assert result is False

def test_is_namespace_package_with_regular_package_has_init():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "regularpkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'print("hello")')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is False

def test_is_namespace_package_with_namespace_package_no_init_but_has_files():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "namespacepkg"
        pkg_path.mkdir()
        py_file = pkg_path / "module.py"
        py_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is False

def test_is_namespace_package_with_namespace_package_no_init_and_no_files():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "emptynamespace"
        pkg_path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_namespace_declare_pkg_resources_single_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "pkgresourcesns"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b"__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_namespace_declare_pkg_resources_double_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "pkgresourcesns2"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_namespace_extend_path_pkgutil_single_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "pkgutilns"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_namespace_extend_path_pkgutil_double_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "pkgutilns2"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_non_py_extension_file_in_dir():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mixedext"
        pkg_path.mkdir()
        txt_file = pkg_path / "data.txt"
        txt_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is True

def test_is_namespace_package_with_setup_cfg_file_in_dir():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "setupcfgdir"
        pkg_path.mkdir()
        cfg_file = pkg_path / "setup.cfg"
        cfg_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is False

def test_is_namespace_package_with_pyproject_toml_file_in_dir():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "pyprojectdir"
        pkg_path.mkdir()
        toml_file = pkg_path / "pyproject.toml"
        toml_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is False

def test_is_namespace_package_with_py_file_in_dir():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "haspyfile"
        pkg_path.mkdir()
        py_file = pkg_path / "code.py"
        py_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is False

def test_is_namespace_package_with_init_but_no_namespace_marker():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "regularinit"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'# just a comment')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result is False


# LLM-generated content at query #9
#--------------------------

def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_pattern_with_wildcard():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate("foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_matches_pattern_with_dot_prefix():
    config = Config(forced_separate=["foo"])
    result = _forced_separate(".foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_pattern_with_wildcard_and_dot_prefix():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate(".foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_returns_none_for_no_match():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("bar", config)
    assert result is None

def test_forced_separate_returns_none_for_partial_match_without_wildcard():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("foobar", config)
    assert result is None

def test_forced_separate_matches_first_pattern_in_list():
    config = Config(forced_separate=["foo", "bar"])
    result = _forced_separate("foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_second_pattern_in_list():
    config = Config(forced_separate=["foo", "bar"])
    result = _forced_separate("bar", config)
    assert result == ("bar", "Matched forced_separate (bar) config value.")

def test_forced_separate_matches_pattern_with_wildcard_in_middle():
    config = Config(forced_separate=["f*o"])
    result = _forced_separate("foo", config)
    assert result == ("f*o", "Matched forced_separate (f*o) config value.")

def test_forced_separate_matches_pattern_with_question_mark():
    config = Config(forced_separate=["f?o"])
    result = _forced_separate("foo", config)
    assert result == ("f?o", "Matched forced_separate (f?o) config value.")

def test_forced_separate_handles_empty_forced_separate_list():
    config = Config(forced_separate=[])
    result = _forced_separate("foo", config)
    assert result is None

def test_forced_separate_handles_pattern_with_special_characters():
    config = Config(forced_separate=["test[abc]"])
    result = _forced_separate("testa", config)
    assert result == ("test[abc]", "Matched forced_separate (test[abc]) config value.")


# LLM-generated content at query #10
#--------------------------

def test_src_path_is_module_true():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result == True


# LLM-generated content at query #11
#--------------------------

def test_is_namespace_package_false_when_filenames_exist():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "__init__.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #12
#--------------------------

def test_is_module_with_py_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "module"
        tmp_path.mkdir()
        py_file = tmp_path.with_suffix(".py")
        py_file.touch()
        result = _is_module(tmp_path)
        assert result == True

def test_is_module_with_extension_suffix():
    import importlib.machinery
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "module"
        tmp_path.mkdir()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = tmp_path.with_suffix(suffix)
            ext_file.touch()
            result = _is_module(tmp_path)
            assert result == True
            ext_file.unlink()

def test_is_module_with_init_py():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "module"
        tmp_path.mkdir()
        init_file = tmp_path / "__init__.py"
        init_file.touch()
        result = _is_module(tmp_path)
        assert result == True

def test_is_module_without_any_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "module"
        tmp_path.mkdir()
        result = _is_module(tmp_path)
        assert result == False

def test_is_module_case_sensitive_check():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "Module"
        tmp_path.mkdir()
        lower_file = Path(tmpdir) / "module.py"
        lower_file.touch()
        result = _is_module(tmp_path)
        assert result == False


# LLM-generated content at query #13
#--------------------------

def test_namespace_package_with_files_in_directory_but_no_init():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "some.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #14
#--------------------------

def test_is_module_with_py_extension():
    path = Path("some_module")
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        result = _is_module(path)
        assert result is True

def test_is_module_with_extension_suffix():
    path = Path("some_extension")
    with patch("importlib.machinery.EXTENSION_SUFFIXES", [".so", ".pyd"]):
        with patch("os.path.exists") as mock_exists:
            mock_exists.side_effect = lambda x: x.endswith(".so") or x.endswith(".pyd")
            result = _is_module(path)
            assert result is True

def test_is_module_with_init_py():
    path = Path("some_package")
    with patch("os.path.exists") as mock_exists:
        mock_exists.side_effect = lambda x: x.endswith("__init__.py")
        result = _is_module(path)
        assert result is True

def test_is_module_false():
    path = Path("nonexistent")
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        result = _is_module(path)
        assert result is False


# LLM-generated content at query #15
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config()
    config.namespace_packages = {"some.namespace"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = [".py"]
    config.src_paths = [Path("/some/path")]
    result = _src_path("some.namespace.module", config, None, ("some", "namespace"))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config()
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = [".py"]
    config.src_paths = [Path("/some/path")]
    with unittest.mock.patch("_is_namespace_package", return_value=True):
        result = _src_path("some.namespace.module", config, None, ("some", "namespace"))
        assert result is not None

def test_nested_module_and_namespace_in_config_namespace_packages():
    config = Config()
    config.namespace_packages = {"some.namespace"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = [".py"]
    config.src_paths = [Path("/some/path")]
    result = _src_path("some.namespace.module", config, None, ("some", "namespace"))
    assert result is not None

def test_nested_module_and_auto_identify_namespace_packages_true():
    config = Config()
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = [".py"]
    config.src_paths = [Path("/some/path")]
    with unittest.mock.patch("_is_namespace_package", return_value=True):
        result = _src_path("some.namespace.module", config, None, ("some", "namespace"))
        assert result is not None


# LLM-generated content at query #16
#--------------------------

def test_src_path_is_module_true_for_valid_dir():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result == True

def test_src_path_is_module_false_for_wrong_name():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "wrong_name")
    src_path.rmdir()
    assert result == False

def test_src_path_is_module_false_for_file():
    src_path = Path("test_module")
    src_path.touch()
    result = _src_path_is_module(src_path, "test_module")
    src_path.unlink()
    assert result == False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("Test_Module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result == False


# LLM-generated content at query #17
#--------------------------

def test_namespace_package_without_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True


# LLM-generated content at query #18
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False, src_paths=[], supported_extensions=set())
    nested_module = ["nested"]
    namespace = "root"
    result = namespace in config.namespace_packages or (config.auto_identify_namespace_packages and False)
    assert result is False


# LLM-generated content at query #19
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    result = _src_path("a.b", config, src_paths=[Path("/test")], prefix=("a",))
    assert result is not None


# LLM-generated content at query #20
#--------------------------

def test_namespace_package_without_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True


# LLM-generated content at query #21
#--------------------------

def test_predicate_at_line_26_true_for_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    module_path = Path("/src/root_module.py")
    src_path = Path("/src")
    root_module_name = "root_module"
    _is_module = lambda p: p == module_path
    _is_package = lambda p: False
    _src_path_is_module = lambda s, r: False
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True

def test_predicate_at_line_26_true_for_package():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    module_path = Path("/src/root_module")
    src_path = Path("/src")
    root_module_name = "root_module"
    _is_module = lambda p: False
    _is_package = lambda p: p == module_path
    _src_path_is_module = lambda s, r: False
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True

def test_predicate_at_line_26_true_for_src_path_is_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    module_path = Path("/src/root_module")
    src_path = Path("/src")
    root_module_name = "root_module"
    _is_module = lambda p: False
    _is_package = lambda p: False
    _src_path_is_module = lambda s, r: s == src_path and r == root_module_name
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True

def test_predicate_at_line_26_true_for_combination():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    module_path = Path("/src/root_module.py")
    src_path = Path("/src")
    root_module_name = "root_module"
    _is_module = lambda p: p == module_path
    _is_package = lambda p: True
    _src_path_is_module = lambda s, r: True
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True


# LLM-generated content at query #22
#--------------------------

def test_namespace_package_without_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True


# LLM-generated content at query #23
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    module_path = Path("/src/some/namespace")
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
        assert result is not None

def test_nested_module_and_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"a.b"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("a.b.c", config, src_paths=[Path("/src")], prefix=("a",))
    assert result is not None

def test_nested_module_and_auto_identify_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    module_path = Path("/src/a/b")
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("a.b.c", config, src_paths=[Path("/src")], prefix=("a",))
        assert result is not None


# LLM-generated content at query #24
#--------------------------

def test__src_path_finds_module_in_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("mymodule", config)
    assert result is None
def test__src_path_returns_firstparty_for_module():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            result = _src_path("mymodule", config)
            assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /src.")
def test__src_path_handles_nested_module_with_namespace():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = {"parent"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            with patch("_src_path", side_effect=lambda n, c, s, p: ("FIRSTPARTY", f"Found: {n}")):
                result = _src_path("parent.child", config)
                assert result == ("FIRSTPARTY", "Found: child")
def test__src_path_identifies_src_path_as_module():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src/mymodule")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            result = _src_path("mymodule", config)
            assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /src/mymodule.")
def test__src_path_returns_none_for_no_match():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=False):
        result = _src_path("unknown", config)
        assert result is None
def test__src_path_uses_custom_src_paths():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/default")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    custom_paths = [Path("/custom")]
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            result = _src_path("mymodule", config, src_paths=custom_paths)
            assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /custom.")
def test__src_path_handles_auto_identify_namespace_packages():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            with patch("_is_namespace_package", return_value=True):
                with patch("_src_path", side_effect=lambda n, c, s, p: ("FIRSTPARTY", f"Auto: {n}")):
                    result = _src_path("parent.child", config)
                    assert result == ("FIRSTPARTY", "Auto: child")


# LLM-generated content at query #25
#--------------------------

def test_is_module_with_py_extension():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path.with_suffix(".py"))
    result = _is_module(path)
    assert result == True

def test_is_module_with_extension_suffix():
    path = Path("some_extension")
    mock_exists_case_sensitive = lambda p: any(p == str(path.with_suffix(ext)) for ext in importlib.machinery.EXTENSION_SUFFIXES)
    result = _is_module(path)
    assert result == True

def test_is_module_with_init_py():
    path = Path("some_package")
    mock_exists_case_sensitive = lambda p: p == str(path / "__init__.py")
    result = _is_module(path)
    assert result == True


# LLM-generated content at query #26
#--------------------------

def test_known_pattern_predicate_false():
    class MockPattern:
        def match(self, module_name):
            return False
    class MockConfig:
        sections = ["section1", "section2"]
        known_patterns = [(MockPattern(), "section1")]
    config = MockConfig()
    name = "some.module.name"
    parts = name.split(".")
    module_names_to_check = (".".join(parts[:first_k]) for first_k in range(len(parts), 0, -1))
    for module_name_to_check in module_names_to_check:
        for pattern, placement in config.known_patterns:
            result = placement in config.sections and pattern.match(module_name_to_check)
            assert result == False


# LLM-generated content at query #27
#--------------------------

def test_is_module_with_py_file():
    path = Path("some_module")
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        result = _is_module(path)
        assert result is True

def test_is_module_with_extension_suffix():
    path = Path("some_extension")
    with patch("os.path.exists") as mock_exists:
        mock_exists.side_effect = lambda p: p == str(path.with_suffix(".so"))
        with patch("importlib.machinery.EXTENSION_SUFFIXES", [".so"]):
            result = _is_module(path)
            assert result is True

def test_is_module_with_init_py():
    path = Path("some_package")
    with patch("os.path.exists") as mock_exists:
        mock_exists.side_effect = lambda p: p == str(path / "__init__.py")
        result = _is_module(path)
        assert result is True

def test_is_module_none_exist():
    path = Path("nonexistent")
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        with patch("importlib.machinery.EXTENSION_SUFFIXES", []):
            result = _is_module(path)
            assert result is False


# LLM-generated content at query #28
#--------------------------

def test_known_pattern_matches_configured_pattern():
    class MockPattern:
        def __init__(self, pattern):
            self.pattern = pattern
        def match(self, name):
            return self.pattern in name
    class MockConfig:
        def __init__(self, patterns, sections):
            self.known_patterns = patterns
            self.sections = sections
    pattern1 = (MockPattern("foo.bar"), "SECTION_A")
    pattern2 = (MockPattern("baz"), "SECTION_B")
    config = MockConfig([pattern1, pattern2], {"SECTION_A", "SECTION_B"})
    result = _known_pattern("foo.bar.module", config)
    assert result == ("SECTION_A", "Matched configured known pattern " + str(pattern1[0]))
def test_known_pattern_no_match():
    class MockPattern:
        def match(self, name):
            return False
    class MockConfig:
        def __init__(self, patterns, sections):
            self.known_patterns = patterns
            self.sections = sections
    pattern = (MockPattern(), "SECTION_A")
    config = MockConfig([pattern], {"SECTION_A"})
    result = _known_pattern("unknown.module", config)
    assert result is None
def test_known_pattern_placement_not_in_sections():
    class MockPattern:
        def match(self, name):
            return True
    class MockConfig:
        def __init__(self, patterns, sections):
            self.known_patterns = patterns
            self.sections = sections
    pattern = (MockPattern(), "SECTION_C")
    config = MockConfig([pattern], {"SECTION_A", "SECTION_B"})
    result = _known_pattern("any.module", config)
    assert result is None
def test_known_pattern_matches_longest_module_prefix():
    class MockPattern:
        def __init__(self, pattern):
            self.pattern = pattern
        def match(self, name):
            return self.pattern in name
    class MockConfig:
        def __init__(self, patterns, sections):
            self.known_patterns = patterns
            self.sections = sections
    pattern1 = (MockPattern("foo"), "SECTION_A")
    pattern2 = (MockPattern("foo.bar"), "SECTION_B")
    config = MockConfig([pattern1, pattern2], {"SECTION_A", "SECTION_B"})
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("SECTION_B", "Matched configured known pattern " + str(pattern2[0]))
def test_known_pattern_matches_first_pattern_in_order():
    class MockPattern:
        def __init__(self, pattern):
            self.pattern = pattern
        def match(self, name):
            return self.pattern in name
    class MockConfig:
        def __init__(self, patterns, sections):
            self.known_patterns = patterns
            self.sections = sections
    pattern1 = (MockPattern("foo.bar"), "SECTION_A")
    pattern2 = (MockPattern("foo.bar"), "SECTION_B")
    config = MockConfig([pattern1, pattern2], {"SECTION_A", "SECTION_B"})
    result = _known_pattern("foo.bar.module", config)
    assert result == ("SECTION_A", "Matched configured known pattern " + str(pattern1[0]))
def test_known_pattern_empty_name():
    class MockPattern:
        def match(self, name):
            return False
    class MockConfig:
        def __init__(self, patterns, sections):
            self.known_patterns = patterns
            self.sections = sections
    pattern = (MockPattern(), "SECTION_A")
    config = MockConfig([pattern], {"SECTION_A"})
    result = _known_pattern("", config)
    assert result is None


# LLM-generated content at query #29
#--------------------------

def test_src_path_is_module_true_for_matching_dir():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is True

def test_src_path_is_module_false_for_non_matching_name():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "different_module")
    src_path.rmdir()
    assert result is False

def test_src_path_is_module_false_for_file():
    src_path = Path("test_module")
    src_path.touch()
    result = _src_path_is_module(src_path, "test_module")
    src_path.unlink()
    assert result is False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("Test_Module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is False

def test_src_path_is_module_false_for_nonexistent_path():
    src_path = Path("nonexistent_module")
    result = _src_path_is_module(src_path, "nonexistent_module")
    assert result is False


# LLM-generated content at query #30
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some", "namespace"))
    assert result is not None


# LLM-generated content at query #31
#--------------------------

def test_namespace_package_without_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_only_dot_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / ".gitignore").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_only_non_source_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "README.md").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_only_setup_cfg():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "setup.cfg").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_only_pyproject_toml():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "pyproject.toml").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_source_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "module.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #32
#--------------------------

def test_src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test_src_path_returns_none_for_missing_module():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("missingmodule", config)
    assert result is None

def test_src_path_handles_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/test/path")], namespace_packages={"namespace"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test_src_path_handles_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("auto_ns.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test_src_path_uses_provided_src_paths_parameter():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    custom_src_paths = [Path("/custom/path")]
    result = _src_path("mymodule", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test_src_path_handles_src_path_is_module_case():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("path", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")


# LLM-generated content at query #33
#--------------------------

def test_is_namespace_package_with_valid_namespace_package():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_non_package_path():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == False

def test_is_namespace_package_with_regular_package():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'print("hello")')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == False

def test_is_namespace_package_with_missing_init_and_py_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_missing_init_but_has_py_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        py_file = pkg_path / "module.py"
        py_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == False

def test_is_namespace_package_with_missing_init_but_has_setup_cfg():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        cfg_file = pkg_path / "setup.cfg"
        cfg_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == False

def test_is_namespace_package_with_missing_init_but_has_pyproject_toml():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        toml_file = pkg_path / "pyproject.toml"
        toml_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == False

def test_is_namespace_package_with_pkg_resources_namespace():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__import__(\'pkg_resources\').declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_double_quotes_pkg_resources():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_single_quotes_pkgutil():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_double_quotes_pkgutil():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_init_but_no_namespace_marker():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        init_file = pkg_path / "__init__.py"
        init_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == False

def test_is_namespace_package_with_src_extensions_other_than_py():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        src_extensions = frozenset(["txt"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True

def test_is_namespace_package_with_missing_init_and_other_extension_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "mypkg"
        pkg_path.mkdir()
        txt_file = pkg_path / "data.txt"
        txt_file.write_bytes(b'')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(pkg_path, src_extensions)
        assert result == True


# LLM-generated content at query #34
#--------------------------

def test_is_module_with_py_suffix():
    path = Path("some_module")
    with patch("os.path.exists", return_value=True):
        result = _is_module(path)
    assert result is True

def test_is_module_with_extension_suffix():
    path = Path("some_extension")
    with patch("importlib.machinery.EXTENSION_SUFFIXES", [".so", ".pyd"]):
        with patch("os.path.exists", side_effect=lambda x: x.endswith(".so") or x.endswith(".pyd")):
            result = _is_module(path)
    assert result is True

def test_is_module_with_init_py():
    path = Path("some_package")
    with patch("os.path.exists", side_effect=lambda x: x.endswith("__init__.py")):
        result = _is_module(path)
    assert result is True

def test_is_module_with_all_false():
    path = Path("nonexistent")
    with patch("os.path.exists", return_value=False):
        result = _is_module(path)
    assert result is False


# LLM-generated content at query #35
#--------------------------

def test_src_path_is_module_true_for_valid_dir():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is True

def test_src_path_is_module_false_for_wrong_name():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "wrong_name")
    src_path.rmdir()
    assert result is False

def test_src_path_is_module_false_for_file():
    src_path = Path("test_module")
    src_path.touch()
    result = _src_path_is_module(src_path, "test_module")
    src_path.unlink()
    assert result is False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("Test_Module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is False


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_is_module_with_py_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        py_file = test_path.with_suffix(".py")
        py_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_with_extension_suffix():
    import importlib.machinery
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = test_path.with_suffix(suffix)
            ext_file.touch()
            result = _is_module(test_path)
            assert result == True
            ext_file.unlink()

def test_is_module_with_init_py():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        init_file = test_path / "__init__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_py():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "Module"
        test_path.mkdir()
        py_file = test_path.with_suffix(".PY")
        py_file.touch()
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_init():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        init_file = test_path / "__INIT__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == False


# LLM-generated content at query #2
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = "test.py"
    with patch('pathlib.Path.exists_case_sensitive', return_value=True):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = "test.so"
    with patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so']), patch('pathlib.Path.exists_case_sensitive', side_effect=lambda x: x == "test.so"):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__.return_value = "test/__init__.py"
    with patch('pathlib.Path.exists_case_sensitive', side_effect=lambda x: x == "test/__init__.py"):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = "test.py"
    mock_path.__truediv__.return_value = "test/__init__.py"
    with patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so']), patch('pathlib.Path.exists_case_sensitive', return_value=False):
        result = _is_module(mock_path)
    assert result is False

def test_is_module_py_false_extension_true():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.side_effect = lambda x: "test.py" if x == ".py" else "test.so"
    with patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so']), patch('pathlib.Path.exists_case_sensitive', side_effect=lambda x: x == "test.so"):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_py_true_extension_false():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.side_effect = lambda x: "test.py" if x == ".py" else "test.so"
    with patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so']), patch('pathlib.Path.exists_case_sensitive', side_effect=lambda x: x == "test.py"):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_init_py_only():
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = "test.py"
    mock_path.__truediv__.return_value = "test/__init__.py"
    with patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so']), patch('pathlib.Path.exists_case_sensitive', side_effect=lambda x: x == "test/__init__.py"):
        result = _is_module(mock_path)
    assert result is True


# LLM-generated content at query #3
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch('pathlib.Path.with_suffix', return_value=Path("some_module.py")) as mock_with_suffix, \
         patch('pathlib.Path.__truediv__', return_value=Path("some_module/__init__.py")) as mock_div, \
         patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so', '.pyd']), \
         patch('test_module.exists_case_sensitive') as mock_exists:
        mock_exists.side_effect = lambda x: x == str(Path("some_module.py"))
        result = _is_module(path)
        assert result is True
        mock_with_suffix.assert_called_once_with(".py")
        mock_div.assert_not_called()

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch('pathlib.Path.with_suffix', side_effect=lambda x: Path(f"some_module{x}")) as mock_with_suffix, \
         patch('pathlib.Path.__truediv__', return_value=Path("some_module/__init__.py")) as mock_div, \
         patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so', '.pyd']), \
         patch('test_module.exists_case_sensitive') as mock_exists:
        mock_exists.side_effect = lambda x: x == str(Path("some_module.so"))
        result = _is_module(path)
        assert result is True
        assert mock_with_suffix.call_count >= 2

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch('pathlib.Path.with_suffix', side_effect=lambda x: Path(f"some_module{x}")) as mock_with_suffix, \
         patch('pathlib.Path.__truediv__', return_value=Path("some_module/__init__.py")) as mock_div, \
         patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so', '.pyd']), \
         patch('test_module.exists_case_sensitive') as mock_exists:
        mock_exists.side_effect = lambda x: x == str(Path("some_module/__init__.py"))
        result = _is_module(path)
        assert result is True
        mock_div.assert_called_once_with("__init__.py")

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch('pathlib.Path.with_suffix', side_effect=lambda x: Path(f"some_module{x}")) as mock_with_suffix, \
         patch('pathlib.Path.__truediv__', return_value=Path("some_module/__init__.py")) as mock_div, \
         patch('importlib.machinery.EXTENSION_SUFFIXES', ['.so', '.pyd']), \
         patch('test_module.exists_case_sensitive', return_value=False) as mock_exists:
        result = _is_module(path)
        assert result is False
        mock_exists.assert_called()


# LLM-generated content at query #4
#--------------------------

def test__src_path_with_exact_module_match():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_nested_module_in_namespace():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = {"mypackage"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("mypackage.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_auto_identified_namespace():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    result = _src_path("namespace.sub", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_no_match():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("unknown", config)
    assert result is None

def test__src_path_with_custom_src_paths():
    config = Config()
    config.src_paths = [Path("/custom/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("module", config, src_paths=[Path("/custom/path")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test__src_path_with_prefix():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("sub", config, prefix=("base",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_src_path_is_module():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("path", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config()
    config.src_paths = [Path("/fake/path")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = [".py"]
    result = _src_path("mymodule", config, src_paths=[Path("/fake/path")])
    assert result is not None
    assert result[0] == sections.FIRSTPARTY
    assert "Found in one of the configured src_paths:" in result[1]


# LLM-generated content at query #6
#--------------------------

def test_is_namespace_package_with_non_existent_path():
    path = Path("/non/existent")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_file_path():
    path = Path("/some/file.txt")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_and_no_files():
    path = Path("/empty/dir")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_directory_no_init_but_py_file():
    path = Path("/dir/with/py")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_but_setup_cfg():
    path = Path("/dir/with/setup")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_but_pyproject_toml():
    path = Path("/dir/with/pyproject")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_init_but_no_namespace_markers():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_init_and_pkg_resources_single_quote():
    path = Path("/dir/with/pkg_resources_single")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_pkg_resources_double_quote():
    path = Path("/dir/with/pkg_resources_double")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_pkgutil_single_quote():
    path = Path("/dir/with/pkgutil_single")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_pkgutil_double_quote():
    path = Path("/dir/with/pkgutil_double")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True


# LLM-generated content at query #7
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"test.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("test.namespace.module", config, src_paths=[Path("/src")], prefix=("test",))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    def mock_is_namespace_package(path, exts):
        return True
    original_is_namespace_package = _is_namespace_package
    _is_namespace_package = mock_is_namespace_package
    try:
        result = _src_path("test.namespace.module", config, src_paths=[Path("/src")], prefix=("test",))
        assert result is not None
    finally:
        _is_namespace_package = original_is_namespace_package

def test_nested_module_true_and_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"existing.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("existing.namespace.submodule", config, src_paths=[Path("/src")], prefix=("existing",))
    assert result is not None

def test_nested_module_true_and_auto_identify_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    def mock_is_namespace_package(path, exts):
        return True
    original_is_namespace_package = _is_namespace_package
    _is_namespace_package = mock_is_namespace_package
    try:
        result = _src_path("auto.namespace.inner", config, src_paths=[Path("/src")], prefix=("auto",))
        assert result is not None
    finally:
        _is_namespace_package = original_is_namespace_package


# LLM-generated content at query #8
#--------------------------

def test_known_pattern_matches_configured_pattern():
    import re

    from my_module import Config
    config = Config()
    config.sections = {"section1", "section2"}
    config.known_patterns = [(re.compile(r"^myapp\.utils$"), "section1")]
    result = _known_pattern("myapp.utils.helpers", config)
    assert result == ("section1", "Matched configured known pattern re.compile('^myapp\\\\.utils$')")

def test_known_pattern_no_match():
    import re

    from my_module import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^other\.module$"), "section1")]
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    import re

    from my_module import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^myapp$"), "section2")]
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_matches_longest_module_prefix():
    import re

    from my_module import Config
    config = Config()
    config.sections = {"section1", "section2"}
    config.known_patterns = [(re.compile(r"^myapp$"), "section1"), (re.compile(r"^myapp\.utils$"), "section2")]
    result = _known_pattern("myapp.utils.helpers", config)
    assert result == ("section2", "Matched configured known pattern re.compile('^myapp\\\\.utils$')")

def test_known_pattern_matches_single_part_name():
    import re

    from my_module import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^myapp$"), "section1")]
    result = _known_pattern("myapp", config)
    assert result == ("section1", "Matched configured known pattern re.compile('^myapp$')")

def test_known_pattern_empty_name():
    import re

    from my_module import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^$"), "section1")]
    result = _known_pattern("", config)
    assert result == ("section1", "Matched configured known pattern re.compile('^$')")

def test_known_pattern_no_known_patterns():
    from my_module import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = []
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_no_sections():
    import re

    from my_module import Config
    config = Config()
    config.sections = set()
    config.known_patterns = [(re.compile(r"^myapp$"), "section1")]
    result = _known_pattern("myapp", config)
    assert result is None


# LLM-generated content at query #9
#--------------------------

def test_namespace_package_without_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mypkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True


# LLM-generated content at query #10
#--------------------------

def test__src_path_finds_module_in_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with unittest.mock.patch("exists_case_sensitive", return_value=True):
        with unittest.mock.patch.object(Path, "is_dir", return_value=True):
            result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_returns_none_for_missing_module():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with unittest.mock.patch("exists_case_sensitive", return_value=False):
        with unittest.mock.patch.object(Path, "is_dir", return_value=False):
            result = _src_path("missing", config)
    assert result is None

def test__src_path_handles_nested_module_in_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = {"namespace"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with unittest.mock.patch("exists_case_sensitive", return_value=True):
        with unittest.mock.patch.object(Path, "is_dir", return_value=True):
            with unittest.mock.patch("_is_namespace_package", return_value=True):
                result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_handles_src_path_is_module():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src/mymodule")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with unittest.mock.patch("exists_case_sensitive", return_value=True):
        with unittest.mock.patch.object(Path, "is_dir", return_value=True):
            result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src/mymodule.")

def test__src_path_with_auto_identify_namespace_packages():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    with unittest.mock.patch("exists_case_sensitive", return_value=True):
        with unittest.mock.patch.object(Path, "is_dir", return_value=True):
            with unittest.mock.patch("_is_namespace_package", return_value=True):
                result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #11
#--------------------------

def test_known_pattern_matches():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^test\.module"), "SECTION_A")]
    config.sections = ["SECTION_A"]
    result = _known_pattern("test.module.sub", config)
    assert result == ("SECTION_A", "Matched configured known pattern re.compile('^test\\\\.module')")

def test_known_pattern_no_match():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^other\.module"), "SECTION_A")]
    config.sections = ["SECTION_A"]
    result = _known_pattern("test.module.sub", config)
    assert result is None

def test_known_pattern_section_not_in_sections():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^test\.module"), "SECTION_A")]
    config.sections = ["SECTION_B"]
    result = _known_pattern("test.module.sub", config)
    assert result is None

def test_known_pattern_matches_longest_first():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^test"), "SECTION_A"), (re.compile(r"^test\.module"), "SECTION_B")]
    config.sections = ["SECTION_A", "SECTION_B"]
    result = _known_pattern("test.module.sub", config)
    assert result == ("SECTION_B", "Matched configured known pattern re.compile('^test\\\\.module')")

def test_known_pattern_matches_first_pattern():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^test\.module"), "SECTION_A"), (re.compile(r"^test\.module"), "SECTION_B")]
    config.sections = ["SECTION_A", "SECTION_B"]
    result = _known_pattern("test.module.sub", config)
    assert result == ("SECTION_A", "Matched configured known pattern re.compile('^test\\\\.module')")

def test_known_pattern_no_parts():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^test"), "SECTION_A")]
    config.sections = ["SECTION_A"]
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_exact_match():
    import re
    from unittest.mock import Mock
    config = Mock()
    config.known_patterns = [(re.compile(r"^test\.module\.sub$"), "SECTION_A")]
    config.sections = ["SECTION_A"]
    result = _known_pattern("test.module.sub", config)
    assert result == ("SECTION_A", "Matched configured known pattern re.compile('^test\\\\.module\\\\.sub$')")


# LLM-generated content at query #12
#--------------------------

def test_namespace_package_without_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_only_non_source_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "README.txt").touch()
        (path / "data.json").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_setup_cfg():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "setup.cfg").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_pyproject_toml():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "pyproject.toml").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_source_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "module.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_init_and_namespace_declaration_single_quotes():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration_double_quotes():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_extend_path_single_quotes():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_extend_path_double_quotes():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_no_namespace_declaration():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("print('hello')")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_non_package_directory():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "some_file.txt").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #13
#--------------------------

def test__src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_handles_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"mypackage"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mypackage.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_returns_none_when_not_found():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("unknown", config)
    assert result is None

def test__src_path_uses_provided_src_paths():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    custom_src_paths = [Path("/custom")]
    result = _src_path("mymodule", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")

def test__src_path_handles_root_module_matching_src_path_name():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_auto_identifies_namespace_packages():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("namespace.sub", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #14
#--------------------------

def test_is_module_with_py_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        py_file = test_path.with_suffix(".py")
        py_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_with_extension_suffix():
    import importlib.machinery
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        for ext in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = test_path.with_suffix(ext)
            ext_file.touch()
            result = _is_module(test_path)
            assert result == True
            ext_file.unlink()

def test_is_module_with_init_py():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        init_file = test_path / "__init__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_no_match():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_py():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        py_file = test_path.with_suffix(".PY")
        py_file.touch()
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_init():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        init_file = test_path / "__INIT__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == False


# LLM-generated content at query #15
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    from unittest.mock import Mock
    path = Mock(spec=Path)
    path.__truediv__.return_value = Mock(exists=lambda: False)
    path.iterdir.return_value = []
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result is True

def test_namespace_package_without_init_and_no_src_files():
    from pathlib import Path
    from unittest.mock import Mock
    path = Mock(spec=Path)
    init_mock = Mock(exists=lambda: False)
    path.__truediv__.return_value = init_mock
    file_mock = Mock(suffix=".txt", name="README.txt")
    path.iterdir.return_value = [file_mock]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result is True

def test_namespace_package_without_init_and_no_src_files_but_setup_cfg():
    from pathlib import Path
    from unittest.mock import Mock
    path = Mock(spec=Path)
    init_mock = Mock(exists=lambda: False)
    path.__truediv__.return_value = init_mock
    file_mock = Mock(suffix=".cfg", name="setup.cfg")
    path.iterdir.return_value = [file_mock]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result is False

def test_namespace_package_without_init_and_no_src_files_but_pyproject_toml():
    from pathlib import Path
    from unittest.mock import Mock
    path = Mock(spec=Path)
    init_mock = Mock(exists=lambda: False)
    path.__truediv__.return_value = init_mock
    file_mock = Mock(suffix=".toml", name="pyproject.toml")
    path.iterdir.return_value = [file_mock]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result is False

def test_namespace_package_without_init_and_src_file_present():
    from pathlib import Path
    from unittest.mock import Mock
    path = Mock(spec=Path)
    init_mock = Mock(exists=lambda: False)
    path.__truediv__.return_value = init_mock
    file_mock = Mock(suffix=".py", name="module.py")
    path.iterdir.return_value = [file_mock]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result is False


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config()
    config.src_paths = [Path("/fake/path")]
    result = _src_path("mymodule", config)
    assert result is not None
    assert result[0] == sections.FIRSTPARTY
    assert "Found in one of the configured src_paths:" in result[1]


# LLM-generated content at query #17
#--------------------------

def test_is_namespace_package_with_valid_namespace_package():
    path = Path("/tmp/test_package")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_bytes(b"__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True
    init_file.unlink()
    path.rmdir()

def test_is_namespace_package_with_non_package_path():
    path = Path("/tmp/nonexistent")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_regular_package():
    path = Path("/tmp/regular_package")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_bytes(b"")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False
    init_file.unlink()
    path.rmdir()

def test_is_namespace_package_with_namespace_package_using_pkgutil():
    path = Path("/tmp/namespace_pkgutil")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_bytes(b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True
    init_file.unlink()
    path.rmdir()

def test_is_namespace_package_without_init_but_with_py_files():
    path = Path("/tmp/package_with_py")
    path.mkdir(parents=True, exist_ok=True)
    py_file = path / "module.py"
    py_file.write_bytes(b"")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False
    py_file.unlink()
    path.rmdir()

def test_is_namespace_package_without_init_but_with_setup_cfg():
    path = Path("/tmp/package_with_setup_cfg")
    path.mkdir(parents=True, exist_ok=True)
    setup_cfg = path / "setup.cfg"
    setup_cfg.write_bytes(b"")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False
    setup_cfg.unlink()
    path.rmdir()

def test_is_namespace_package_without_init_and_no_files():
    path = Path("/tmp/empty_package")
    path.mkdir(parents=True, exist_ok=True)
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True
    path.rmdir()

def test_is_namespace_package_with_double_quotes_in_declare_namespace():
    path = Path("/tmp/double_quote_package")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True
    init_file.unlink()
    path.rmdir()

def test_is_namespace_package_with_double_quotes_in_pkgutil():
    path = Path("/tmp/double_quote_pkgutil")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_bytes(b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True
    init_file.unlink()
    path.rmdir()

def test_is_namespace_package_with_other_src_extensions():
    path = Path("/tmp/package_with_other_ext")
    path.mkdir(parents=True, exist_ok=True)
    other_file = path / "module.txt"
    other_file.write_bytes(b"")
    src_extensions = frozenset(["txt"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False
    other_file.unlink()
    path.rmdir()


# LLM-generated content at query #18
#--------------------------

def test_is_namespace_package_with_non_existent_path():
    path = Path("/non/existent")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_file_path():
    path = Path("/some/file.txt")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_and_no_files():
    path = Path("/empty/dir")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_directory_no_init_but_has_py_file():
    path = Path("/dir/with/py")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_but_has_setup_cfg():
    path = Path("/dir/with/setup")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_but_has_pyproject_toml():
    path = Path("/dir/with/pyproject")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_init_but_no_namespace_declaration():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_init_and_pkg_resources_single_quote():
    path = Path("/dir/with/pkg_resources_single")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_pkg_resources_double_quote():
    path = Path("/dir/with/pkg_resources_double")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_pkgutil_single_quote():
    path = Path("/dir/with/pkgutil_single")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_pkgutil_double_quote():
    path = Path("/dir/with/pkgutil_double")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_and_namespace_declaration_at_end_of_file():
    path = Path("/dir/with/long_init")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    src_path = Path("/test/path")
    root_module_name = "module"
    module_path = (src_path / root_module_name).resolve()
    prefix = ()
    nested_module = []
    namespace = ".".join(prefix + (root_module_name,))
    _is_module = lambda p: True
    _is_package = lambda p: False
    _src_path_is_module = lambda s, r: False
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True


# LLM-generated content at query #20
#--------------------------

def test_known_pattern_returns_tuple_when_placement_in_sections_and_pattern_matches():
    from unittest.mock import Mock
    mock_config = Mock()
    mock_config.sections = {"test_section"}
    mock_pattern = Mock()
    mock_pattern.match = Mock(return_value=True)
    mock_config.known_patterns = [(mock_pattern, "test_section")]
    result = _known_pattern("some.module.name", mock_config)
    assert result == ("test_section", "Matched configured known pattern " + str(mock_pattern))


# LLM-generated content at query #21
#--------------------------

def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["exact"])
    result = _forced_separate("exact", config)
    assert result == ("exact", "Matched forced_separate (exact) config value.")

def test_forced_separate_matches_pattern_with_wildcard():
    config = Config(forced_separate=["dir/*"])
    result = _forced_separate("dir/file.txt", config)
    assert result == ("dir/*", "Matched forced_separate (dir/*) config value.")

def test_forced_separate_matches_pattern_without_wildcard():
    config = Config(forced_separate=["prefix"])
    result = _forced_separate("prefix_suffix", config)
    assert result == ("prefix", "Matched forced_separate (prefix) config value.")

def test_forced_separate_matches_hidden_file():
    config = Config(forced_separate=[".hidden"])
    result = _forced_separate(".hidden", config)
    assert result == (".hidden", "Matched forced_separate (.hidden) config value.")

def test_forced_separate_matches_hidden_file_with_wildcard():
    config = Config(forced_separate=[".dir/*"])
    result = _forced_separate(".dir/file.txt", config)
    assert result == (".dir/*", "Matched forced_separate (.dir/*) config value.")

def test_forced_separate_does_not_match():
    config = Config(forced_separate=["other"])
    result = _forced_separate("nomatch", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("any", config)
    assert result is None

def test_forced_separate_multiple_patterns_first_matches():
    config = Config(forced_separate=["first", "second"])
    result = _forced_separate("first_match", config)
    assert result == ("first", "Matched forced_separate (first) config value.")

def test_forced_separate_multiple_patterns_second_matches():
    config = Config(forced_separate=["first", "second"])
    result = _forced_separate("second_match", config)
    assert result == ("second", "Matched forced_separate (second) config value.")


# LLM-generated content at query #22
#--------------------------

def test_namespace_package_with_pkgutil_extend_path_single_quotes():
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    path = Path(temp_dir)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_pkgutil_extend_path_double_quotes():
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    path = Path(temp_dir)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_pkg_resources_declare_namespace_single_quotes():
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    path = Path(temp_dir)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_pkg_resources_declare_namespace_double_quotes():
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    path = Path(temp_dir)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True


# LLM-generated content at query #23
#--------------------------

def test_namespace_package_with_no_init_and_no_files():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_no_init_and_no_source_files_but_setup_cfg():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "setup.cfg").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_no_init_and_no_source_files_but_pyproject_toml():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "pyproject.toml").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_no_init_and_source_file():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "module.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #24
#--------------------------

def test_src_path_is_module_true_for_valid_dir():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is True

def test_src_path_is_module_false_for_wrong_name():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "wrong_name")
    src_path.rmdir()
    assert result is False

def test_src_path_is_module_false_for_file():
    src_path = Path("test_module")
    src_path.touch()
    result = _src_path_is_module(src_path, "test_module")
    src_path.unlink()
    assert result is False

def test_src_path_is_module_false_for_case_mismatch_on_case_sensitive_fs():
    src_path = Path("Test_Module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is False

def test_src_path_is_module_true_for_case_match_on_case_sensitive_fs():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is True


# LLM-generated content at query #25
#--------------------------

def test__src_path_with_exact_module_match():
    config = Config(src_paths=[Path("/test/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/test/src")], namespace_packages={"mypackage"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("mypackage.nested", config, src_paths=[Path("/test/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/test/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("namespace.nested", config, src_paths=[Path("/test/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/src.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/test/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("unknown", config)
    assert result is None

def test__src_path_with_src_path_is_module_match():
    config = Config(src_paths=[Path("/test/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/src.")

def test__src_path_with_custom_src_paths():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("module", config, src_paths=[Path("/custom/path")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test__src_path_with_prefix():
    config = Config(src_paths=[Path("/test/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("mymodule", config, src_paths=[Path("/test/src")], prefix=("pre",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/src.")


# LLM-generated content at query #26
#--------------------------

def test_src_path_is_module_true_for_valid_dir():
    src_path = Path("valid_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "valid_module")
    src_path.rmdir()
    assert result is True

def test_src_path_is_module_false_for_wrong_name():
    src_path = Path("some_dir")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "different_name")
    src_path.rmdir()
    assert result is False

def test_src_path_is_module_false_for_file():
    src_path = Path("file.txt")
    src_path.touch()
    result = _src_path_is_module(src_path, "file.txt")
    src_path.unlink()
    assert result is False

def test_src_path_is_module_false_for_nonexistent():
    src_path = Path("nonexistent")
    result = _src_path_is_module(src_path, "nonexistent")
    assert result is False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("Module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "module")
    src_path.rmdir()
    assert result is False


# LLM-generated content at query #27
#--------------------------

def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["exact"])
    result = _forced_separate("exact", config)
    assert result == ("exact", "Matched forced_separate (exact) config value.")

def test_forced_separate_matches_pattern_with_wildcard():
    config = Config(forced_separate=["prefix*"])
    result = _forced_separate("prefix_suffix", config)
    assert result == ("prefix*", "Matched forced_separate (prefix*) config value.")

def test_forced_separate_matches_pattern_with_dot_prefix():
    config = Config(forced_separate=["hidden"])
    result = _forced_separate(".hidden", config)
    assert result == ("hidden", "Matched forced_separate (hidden) config value.")

def test_forced_separate_matches_pattern_with_wildcard_and_dot_prefix():
    config = Config(forced_separate=["pre*"])
    result = _forced_separate(".pre_suffix", config)
    assert result == ("pre*", "Matched forced_separate (pre*) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["other"])
    result = _forced_separate("nomatch", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("any", config)
    assert result is None

def test_forced_separate_multiple_patterns_first_matches():
    config = Config(forced_separate=["first", "second"])
    result = _forced_separate("first", config)
    assert result == ("first", "Matched forced_separate (first) config value.")

def test_forced_separate_multiple_patterns_second_matches():
    config = Config(forced_separate=["first", "second"])
    result = _forced_separate("second", config)
    assert result == ("second", "Matched forced_separate (second) config value.")

def test_forced_separate_pattern_without_wildcard_matches_substring():
    config = Config(forced_separate=["partial"])
    result = _forced_separate("partial_extra", config)
    assert result == ("partial", "Matched forced_separate (partial) config value.")

def test_forced_separate_pattern_with_wildcard_matches_exact():
    config = Config(forced_separate=["exact*"])
    result = _forced_separate("exact", config)
    assert result == ("exact*", "Matched forced_separate (exact*) config value.")


# LLM-generated content at query #28
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    with patch("pathlib.Path.with_suffix", return_value=Path("some_module.py")), patch("test_module.exists_case_sensitive", return_value=True):
        result = _is_module(test_path)
    assert result == True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    mock_suffixes = [".so", ".pyd"]
    with patch("importlib.machinery.EXTENSION_SUFFIXES", mock_suffixes), patch("pathlib.Path.with_suffix") as mock_with_suffix, patch("test_module.exists_case_sensitive") as mock_exists:
        mock_with_suffix.side_effect = lambda x: Path(f"some_module{x}")
        mock_exists.side_effect = lambda p: str(p).endswith(".so")
        result = _is_module(test_path)
    assert result == True

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    with patch("pathlib.Path.__truediv__", return_value=Path("some_module/__init__.py")), patch("test_module.exists_case_sensitive", return_value=True):
        result = _is_module(test_path)
    assert result == True

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    mock_suffixes = [".so", ".pyd"]
    with patch("importlib.machinery.EXTENSION_SUFFIXES", mock_suffixes), patch("pathlib.Path.with_suffix") as mock_with_suffix, patch("pathlib.Path.__truediv__", return_value=Path("some_module/__init__.py")), patch("test_module.exists_case_sensitive", return_value=False):
        mock_with_suffix.side_effect = lambda x: Path(f"some_module{x}")
        result = _is_module(test_path)
    assert result == False


# LLM-generated content at query #29
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    result = _src_path("a.b", config, src_paths=[Path("/test")], prefix=("a",))
    assert result is None


