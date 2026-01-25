####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_known_pattern_matches():
    class MockPattern:
        def __init__(self, pattern):
            self.pattern = pattern
        def match(self, name):
            return self.pattern in name
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1", "section2"], [(MockPattern("test"), "section1"), (MockPattern("foo.bar"), "section2")])
    result = _known_pattern("test.module", config)
    assert result == ("section1", "Matched configured known pattern <__main__.test_known_pattern_matches.<locals>.MockPattern object>")

def test_known_pattern_matches_deeper():
    class MockPattern:
        def __init__(self, pattern):
            self.pattern = pattern
        def match(self, name):
            return self.pattern == name
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1"], [(MockPattern("a.b"), "section1")])
    result = _known_pattern("a.b.c", config)
    assert result == ("section1", "Matched configured known pattern <__main__.test_known_pattern_matches_deeper.<locals>.MockPattern object>")

def test_known_pattern_no_match():
    class MockPattern:
        def match(self, name):
            return False
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1"], [(MockPattern(), "section1")])
    result = _known_pattern("unknown.module", config)
    assert result is None

def test_known_pattern_section_not_in_config():
    class MockPattern:
        def match(self, name):
            return True
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1"], [(MockPattern(), "section2")])
    result = _known_pattern("any.module", config)
    assert result is None

def test_known_pattern_empty_name():
    class MockPattern:
        def match(self, name):
            return False
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1"], [(MockPattern(), "section1")])
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_multiple_patterns_first_wins():
    class MockPattern:
        def __init__(self, matches):
            self.matches = matches
        def match(self, name):
            return self.matches
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1", "section2"], [(MockPattern(True), "section1"), (MockPattern(True), "section2")])
    result = _known_pattern("module", config)
    assert result == ("section1", "Matched configured known pattern <__main__.test_known_pattern_multiple_patterns_first_wins.<locals>.MockPattern object>")

def test_known_pattern_check_order_longest_first():
    class MockPattern:
        def __init__(self, target):
            self.target = target
        def match(self, name):
            return name == self.target
    class MockConfig:
        def __init__(self, sections, known_patterns):
            self.sections = sections
            self.known_patterns = known_patterns
    config = MockConfig(["section1", "section2"], [(MockPattern("a"), "section1"), (MockPattern("a.b"), "section2")])
    result = _known_pattern("a.b.c", config)
    assert result == ("section2", "Matched configured known pattern <__main__.test_known_pattern_check_order_longest_first.<locals>.MockPattern object>")


# LLM-generated content at query #2
#--------------------------

def test__src_path_with_exact_module_match_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_package_match_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mypackage", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"mynamespace"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mynamespace.nested", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_nested_module_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("mynamespace.nested", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_no_match_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("unknown", config, src_paths=[Path("/src")])
    assert result is None

def test__src_path_with_src_path_is_module_match():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("src", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_custom_src_paths_argument():
    config = Config(src_paths=[Path("/other")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config, src_paths=[Path("/custom")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")

def test__src_path_with_prefix_argument():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("submodule", config, src_paths=[Path("/src")], prefix=("mypackage",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #3
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "module"
        test_path.mkdir()
        py_file = test_path.with_suffix(".py")
        py_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    import tempfile
    import importlib.machinery
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "module"
        test_path.mkdir()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = test_path.with_suffix(suffix)
            ext_file.touch()
            result = _is_module(test_path)
            assert result == True
            ext_file.unlink()

def test_is_module_with_init_py():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "module"
        test_path.mkdir()
        init_file = test_path / "__init__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_without_any_module_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "not_a_module"
        test_path.mkdir()
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_check():
    from pathlib import Path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "Module"
        test_path.mkdir()
        lower_file = test_path.with_suffix(".py")
        lower_file.touch()
        upper_file = Path(str(test_path).upper())
        result = _is_module(upper_file)
        assert result == False


# LLM-generated content at query #4
#--------------------------

def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["src"])
    result = _forced_separate("src", config)
    assert result == ("src", "Matched forced_separate (src) config value.")

def test_forced_separate_matches_pattern_with_trailing_content():
    config = Config(forced_separate=["src"])
    result = _forced_separate("src/main.py", config)
    assert result == ("src", "Matched forced_separate (src) config value.")

def test_forced_separate_matches_pattern_with_dot_prefix():
    config = Config(forced_separate=["src"])
    result = _forced_separate(".src", config)
    assert result == ("src", "Matched forced_separate (src) config value.")

def test_forced_separate_matches_pattern_with_dot_prefix_and_trailing_content():
    config = Config(forced_separate=["src"])
    result = _forced_separate(".src/main.py", config)
    assert result == ("src", "Matched forced_separate (src) config value.")

def test_forced_separate_matches_wildcard_pattern():
    config = Config(forced_separate=["src/*"])
    result = _forced_separate("src/main.py", config)
    assert result == ("src/*", "Matched forced_separate (src/*) config value.")

def test_forced_separate_matches_wildcard_pattern_with_dot_prefix():
    config = Config(forced_separate=["src/*"])
    result = _forced_separate(".src/main.py", config)
    assert result == ("src/*", "Matched forced_separate (src/*) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["src"])
    result = _forced_separate("tests", config)
    assert result is None

def test_forced_separate_no_match_for_partial():
    config = Config(forced_separate=["src"])
    result = _forced_separate("source", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("src", config)
    assert result is None

def test_forced_separate_multiple_patterns_first_matches():
    config = Config(forced_separate=["docs", "src"])
    result = _forced_separate("src", config)
    assert result == ("docs", "Matched forced_separate (docs) config value.")

def test_forced_separate_multiple_patterns_second_matches():
    config = Config(forced_separate=["docs", "src"])
    result = _forced_separate("src/main.py", config)
    assert result == ("docs", "Matched forced_separate (docs) config value.")


# LLM-generated content at query #5
#--------------------------

def test_is_namespace_package_with_valid_namespace_package_without_init():
    from pathlib import Path
    from unittest.mock import Mock, patch
    temp_dir = Path("/fake/path")
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value=str(temp_dir))
    mock_path.is_dir = Mock(return_value=True)
    mock_path.exists = Mock(return_value=False)
    mock_path.iterdir = Mock(return_value=[])
    mock_path.__truediv__ = Mock(return_value=mock_path)
    with patch('pathlib.Path.exists', return_value=False):
        result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is True

def test_is_namespace_package_with_non_package_path():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=False)
    result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is False

def test_is_namespace_package_with_init_containing_pkg_resources_declare_namespace_single_quotes():
    from pathlib import Path
    from unittest.mock import Mock, mock_open
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=True)
    mock_open_file = mock_open(read_data=b"__import__('pkg_resources').declare_namespace(__name__)")
    with patch('builtins.open', mock_open_file):
        result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is True

def test_is_namespace_package_with_init_containing_pkg_resources_declare_namespace_double_quotes():
    from pathlib import Path
    from unittest.mock import Mock, mock_open
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=True)
    mock_open_file = mock_open(read_data=b'__import__("pkg_resources").declare_namespace(__name__)')
    with patch('builtins.open', mock_open_file):
        result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is True

def test_is_namespace_package_with_init_containing_pkgutil_extend_path_single_quotes():
    from pathlib import Path
    from unittest.mock import Mock, mock_open
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=True)
    mock_open_file = mock_open(read_data=b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    with patch('builtins.open', mock_open_file):
        result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is True

def test_is_namespace_package_with_init_containing_pkgutil_extend_path_double_quotes():
    from pathlib import Path
    from unittest.mock import Mock, mock_open
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=True)
    mock_open_file = mock_open(read_data=b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    with patch('builtins.open', mock_open_file):
        result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is True

def test_is_namespace_package_with_init_missing_namespace_markers():
    from pathlib import Path
    from unittest.mock import Mock, mock_open
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=True)
    mock_open_file = mock_open(read_data=b"print('hello')")
    with patch('builtins.open', mock_open_file):
        result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is False

def test_is_namespace_package_without_init_but_with_py_files():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=False)
    mock_file = Mock()
    mock_file.suffix = ".py"
    mock_file.name = "module.py"
    mock_path.iterdir = Mock(return_value=[mock_file])
    result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is False

def test_is_namespace_package_without_init_but_with_setup_cfg():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=False)
    mock_file = Mock()
    mock_file.suffix = ".cfg"
    mock_file.name = "setup.cfg"
    mock_path.iterdir = Mock(return_value=[mock_file])
    result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is False

def test_is_namespace_package_without_init_but_with_pyproject_toml():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=False)
    mock_file = Mock()
    mock_file.suffix = ".toml"
    mock_file.name = "pyproject.toml"
    mock_path.iterdir = Mock(return_value=[mock_file])
    result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is False

def test_is_namespace_package_without_init_and_no_files():
    from pathlib import Path
    from unittest.mock import Mock
    mock_path = Mock(spec=Path)
    mock_path.__str__ = Mock(return_value="/fake/path")
    mock_path.is_dir = Mock(return_value=True)
    mock_init = Mock()
    mock_path.__truediv__ = Mock(return_value=mock_init)
    mock_init.exists = Mock(return_value=False)
    mock_path.iterdir = Mock(return_value=[])
    result = _is_namespace_package(mock_path, frozenset(["py"]))
    assert result is True


# LLM-generated content at query #6
#--------------------------

def test_src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config)
    expected = (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
    assert result == expected

def test_src_path_returns_none_for_missing_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("missing", config)
    assert result is None

def test_src_path_handles_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("parent.child", config)
    expected = (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
    assert result == expected

def test_src_path_handles_nested_module_with_auto_identify():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("parent.child", config)
    expected = (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
    assert result == expected

def test_src_path_uses_custom_src_paths():
    config = Config(src_paths=[Path("/custom")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config, src_paths=[Path("/custom")])
    expected = (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")
    assert result == expected

def test_src_path_handles_root_module_matching_src_path_name():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("src", config)
    expected = (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_16_evaluates_to_true():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/some/path")]
    src_path = Path("/some/path/root_module_name")
    module_path = Mock()
    module_path.is_dir.return_value = False
    src_path.name = "root_module_name"
    prefix = ()
    result = not prefix and not module_path.is_dir() and src_path.name == root_module_name
    assert result == True


# LLM-generated content at query #8
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
        assert result is not None

def test_nested_module_true_and_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
    assert result is not None

def test_nested_module_true_and_auto_identify_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
        assert result is not None


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_16_true():
    import sys
    from pathlib import Path
    from unittest.mock import Mock
    sys.modules.pop('_test_module', None)
    class MockConfig:
        src_paths = []
        namespace_packages = set()
        auto_identify_namespace_packages = False
        supported_extensions = ()
    config = MockConfig()
    src_path = Path('/tmp/test_root')
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / '__init__.py').touch()
    config.src_paths = [src_path]
    name = 'test_root'
    result = _src_path(name, config, src_paths=None, prefix=())
    assert result is not None
    assert sections.FIRSTPARTY in result
    import shutil
    shutil.rmtree(src_path, ignore_errors=True)


# LLM-generated content at query #10
#--------------------------

def test_is_module_with_py_file():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == "some_module.py"
    global exists_case_sensitive
    original_exists_case_sensitive = exists_case_sensitive
    exists_case_sensitive = mock_exists_case_sensitive
    result = _is_module(path)
    exists_case_sensitive = original_exists_case_sensitive
    assert result == True

def test_is_module_with_extension_suffix():
    path = Path("some_extension")
    mock_exists_case_sensitive = lambda p: p == "some_extension.pyd"
    global exists_case_sensitive
    original_exists_case_sensitive = exists_case_sensitive
    exists_case_sensitive = mock_exists_case_sensitive
    importlib.machinery.EXTENSION_SUFFIXES = [".pyd"]
    result = _is_module(path)
    exists_case_sensitive = original_exists_case_sensitive
    assert result == True

def test_is_module_with_init_py():
    path = Path("some_package")
    mock_exists_case_sensitive = lambda p: p == "some_package/__init__.py"
    global exists_case_sensitive
    original_exists_case_sensitive = exists_case_sensitive
    exists_case_sensitive = mock_exists_case_sensitive
    result = _is_module(path)
    exists_case_sensitive = original_exists_case_sensitive
    assert result == True


# LLM-generated content at query #11
#--------------------------

def test_src_path_is_module_true_for_valid_module_dir():
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

def test_src_path_is_module_false_for_nonexistent_path():
    src_path = Path("nonexistent_module")
    result = _src_path_is_module(src_path, "nonexistent_module")
    assert result == False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("TestModule")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "testmodule")
    src_path.rmdir()
    assert result == False


# LLM-generated content at query #12
#--------------------------

def test_known_pattern_predicate_false():
    class MockPattern:
        def match(self, module_name):
            return False
    class MockConfig:
        sections = ["section1", "section2"]
        known_patterns = [(MockPattern(), "placement1")]
    config = MockConfig()
    name = "module.submodule"
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #13
#--------------------------

def test_is_module_with_py_extension():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path.with_suffix(".py"))
    original_exists_case_sensitive = __import__('builtins').__dict__.get('exists_case_sensitive')
    __import__('builtins').__dict__['exists_case_sensitive'] = mock_exists_case_sensitive
    try:
        result = _is_module(path)
        assert result == True
    finally:
        if original_exists_case_sensitive is not None:
            __import__('builtins').__dict__['exists_case_sensitive'] = original_exists_case_sensitive
        else:
            del __import__('builtins').__dict__['exists_case_sensitive']

def test_is_module_with_extension_suffix():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path.with_suffix(importlib.machinery.EXTENSION_SUFFIXES[0]))
    original_exists_case_sensitive = __import__('builtins').__dict__.get('exists_case_sensitive')
    __import__('builtins').__dict__['exists_case_sensitive'] = mock_exists_case_sensitive
    try:
        result = _is_module(path)
        assert result == True
    finally:
        if original_exists_case_sensitive is not None:
            __import__('builtins').__dict__['exists_case_sensitive'] = original_exists_case_sensitive
        else:
            del __import__('builtins').__dict__['exists_case_sensitive']

def test_is_module_with_init_py():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path / "__init__.py")
    original_exists_case_sensitive = __import__('builtins').__dict__.get('exists_case_sensitive')
    __import__('builtins').__dict__['exists_case_sensitive'] = mock_exists_case_sensitive
    try:
        result = _is_module(path)
        assert result == True
    finally:
        if original_exists_case_sensitive is not None:
            __import__('builtins').__dict__['exists_case_sensitive'] = original_exists_case_sensitive
        else:
            del __import__('builtins').__dict__['exists_case_sensitive']


# LLM-generated content at query #14
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_source_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "README.txt").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_source_files_but_setup_cfg():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "setup.cfg").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_no_source_files_but_pyproject_toml():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "pyproject.toml").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_source_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "module.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_init_and_namespace_declaration_pkg_resources_single_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration_pkg_resources_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration_pkgutil_single_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration_pkgutil_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_no_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_init_and_namespace_declaration_after_4096_bytes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        content = " " * 4096 + "__import__('pkg_resources').declare_namespace(__name__)"
        init_file.write_text(content)
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_not_a_package():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #15
#--------------------------

def test__src_path_with_exact_module_match():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"mypackage"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mypackage.nested", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("namespace.nested", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_no_match():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("unknown", config)
    assert result is None

def test__src_path_with_custom_src_paths():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("module", config, src_paths=[Path("/custom")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")

def test__src_path_with_prefix():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("sub.module", config, src_paths=[Path("/src")], prefix=("base",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #16
#--------------------------

def test_src_path_finds_module_in_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with Mock() as mock_exists:
        mock_exists.return_value = True
        import sys
        sys.modules['exists_case_sensitive'] = mock_exists
        result = _src_path("mymodule", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
def test_src_path_returns_none_for_missing_module():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with Mock() as mock_exists:
        mock_exists.return_value = False
        import sys
        sys.modules['exists_case_sensitive'] = mock_exists
        result = _src_path("missingmodule", config)
        assert result is None
def test_src_path_handles_nested_module_in_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = {"mypackage"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with Mock() as mock_exists:
        mock_exists.return_value = True
        import sys
        sys.modules['exists_case_sensitive'] = mock_exists
        result = _src_path("mypackage.nested", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
def test_src_path_handles_auto_identified_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    with Mock() as mock_exists:
        mock_exists.return_value = True
        import sys
        sys.modules['exists_case_sensitive'] = mock_exists
        mock_is_namespace = Mock(return_value=True)
        sys.modules['_is_namespace_package'] = mock_is_namespace
        result = _src_path("namespace.nested", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
def test_src_path_with_custom_src_paths_argument():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/default")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    custom_src_paths = [Path("/custom")]
    with Mock() as mock_exists:
        mock_exists.return_value = True
        import sys
        sys.modules['exists_case_sensitive'] = mock_exists
        result = _src_path("mymodule", config, src_paths=custom_src_paths)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")
def test_src_path_src_path_is_module_condition():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with Mock() as mock_exists:
        mock_exists.return_value = True
        import sys
        sys.modules['exists_case_sensitive'] = mock_exists
        mock_src_path_is_module = Mock(return_value=True)
        sys.modules['_src_path_is_module'] = mock_src_path_is_module
        result = _src_path("src", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    name = "module"
    src_paths = [Path("/test/path")]
    prefix = ()
    root_module_name, *nested_module = name.split(".", 1)
    new_prefix = (*prefix, root_module_name)
    namespace = ".".join(new_prefix)
    src_path = src_paths[0]
    module_path = (src_path / root_module_name).resolve()
    result = (_is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name))
    assert result == True


# LLM-generated content at query #18
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False, src_paths=[], supported_extensions=[])
    result = _src_path("a.b", config, src_paths=[], prefix=())
    assert result is None


# LLM-generated content at query #19
#--------------------------

def test_is_namespace_package_returns_false_for_non_package():
    path = Path("/non/existent")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_returns_false_for_directory_without_init_and_python_files():
    path = Path("/empty/dir")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_returns_false_for_directory_without_init_but_has_py_file():
    path = Path("/dir/with/py")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_returns_false_for_directory_without_init_but_has_setup_cfg():
    path = Path("/dir/with/config")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_returns_false_for_directory_without_init_but_has_pyproject_toml():
    path = Path("/dir/with/pyproject")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_returns_false_for_init_without_namespace_declaration():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_returns_true_for_init_with_pkg_resources_single_quote():
    path = Path("/dir/with/namespace1")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_returns_true_for_init_with_pkg_resources_double_quote():
    path = Path("/dir/with/namespace2")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_returns_true_for_init_with_pkgutil_single_quote():
    path = Path("/dir/with/namespace3")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_returns_true_for_init_with_pkgutil_double_quote():
    path = Path("/dir/with/namespace4")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_returns_false_for_directory_without_init_but_has_other_src_extension():
    path = Path("/dir/with/other_ext")
    src_extensions = frozenset(["py", "pyx"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False


# LLM-generated content at query #20
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config(src_paths=[Path("/fake/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    result = _src_path("mymodule", config, src_paths=[Path("/fake/src")], prefix=())
    assert result is not None
    assert result[0] == sections.FIRSTPARTY
    assert "Found in one of the configured src_paths:" in result[1]


# LLM-generated content at query #21
#--------------------------

def test_src_path_found_module_in_src_paths():
    config = Config()
    config.src_paths = [Path("/src")]
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_found_package_in_src_paths():
    config = Config()
    config.src_paths = [Path("/src")]
    result = _src_path("mypackage", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_found_src_path_is_module():
    config = Config()
    config.src_paths = [Path("/src")]
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_namespace_package_with_nested_module():
    config = Config()
    config.src_paths = [Path("/src")]
    config.namespace_packages = ["namespace"]
    result = _src_path("namespace.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_auto_identify_namespace_packages():
    config = Config()
    config.src_paths = [Path("/src")]
    config.auto_identify_namespace_packages = True
    result = _src_path("namespace.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_not_found():
    config = Config()
    config.src_paths = [Path("/src")]
    result = _src_path("unknown", config)
    assert result is None

def test_src_path_with_custom_src_paths():
    config = Config()
    custom_src_paths = [Path("/custom")]
    result = _src_path("mymodule", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")

def test_src_path_with_prefix():
    config = Config()
    config.src_paths = [Path("/src")]
    result = _src_path("submodule", config, prefix=("parent",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_namespace_package_not_in_config():
    config = Config()
    config.src_paths = [Path("/src")]
    config.namespace_packages = []
    result = _src_path("namespace.submodule", config)
    assert result is None

def test_src_path_auto_identify_namespace_packages_disabled():
    config = Config()
    config.src_paths = [Path("/src")]
    config.auto_identify_namespace_packages = False
    result = _src_path("namespace.submodule", config)
    assert result is None


# LLM-generated content at query #22
#--------------------------

def test_src_path_finds_module_in_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("mymodule", config)
    assert result is None

def test_src_path_returns_firstparty_on_module_match():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("_is_module", return_value=True):
        result = _src_path("mymodule", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_handles_nested_module_with_namespace():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = {"mypackage"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("_is_namespace_package", return_value=False):
        with patch("_is_module", return_value=True):
            result = _src_path("mypackage.submodule", config)
            assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_auto_identifies_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    with patch("_is_namespace_package", return_value=True):
        with patch("_is_module", return_value=True):
            result = _src_path("mypackage.submodule", config)
            assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_src_path_is_module_match():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("_src_path_is_module", return_value=True):
        result = _src_path("src", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_with_custom_src_paths():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/custom")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    custom_src_paths = [Path("/custom")]
    with patch("_is_module", return_value=True):
        result = _src_path("mymodule", config, src_paths=custom_src_paths)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")

def test_src_path_with_prefix():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("_is_module", return_value=True):
        result = _src_path("submodule", config, prefix=("mypackage",))
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_no_match_returns_none():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("_is_module", return_value=False):
        with patch("_is_package", return_value=False):
            with patch("_src_path_is_module", return_value=False):
                result = _src_path("unknown", config)
                assert result is None


# LLM-generated content at query #23
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("pathlib.Path.with_suffix", return_value=Path("some_module.py")), patch("test_module.exists_case_sensitive", return_value=True):
        result = test_module._is_module(path)
        assert result == True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("pathlib.Path.with_suffix", side_effect=[Path("some_module.py"), Path("some_module.so")]), patch("test_module.exists_case_sensitive", side_effect=[False, True]):
        result = test_module._is_module(path)
        assert result == True

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("pathlib.Path.with_suffix", return_value=Path("some_module.py")), patch("test_module.exists_case_sensitive", side_effect=[False, False, True]):
        result = test_module._is_module(path)
        assert result == True

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("pathlib.Path.with_suffix", return_value=Path("some_module.py")), patch("test_module.exists_case_sensitive", return_value=False):
        result = test_module._is_module(path)
        assert result == False


# LLM-generated content at query #24
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    result = _src_path("a.b", config, src_paths=[Path("/src")], prefix=("a",))
    assert result is None


# LLM-generated content at query #25
#--------------------------

def test_known_pattern_matches_configured_pattern():
    import re
    from mymodule import Config
    config = Config()
    config.sections = {"section1", "section2"}
    config.known_patterns = [(re.compile(r"^myapp\.utils$"), "section1")]
    result = _known_pattern("myapp.utils.helpers", config)
    assert result == ("section1", "Matched configured known pattern re.compile('^myapp\\\\.utils$')")

def test_known_pattern_no_match():
    import re
    from mymodule import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^other\.module$"), "section1")]
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    import re
    from mymodule import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^myapp$"), "section2")]
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_matches_longest_module_prefix():
    import re
    from mymodule import Config
    config = Config()
    config.sections = {"section1", "section2"}
    config.known_patterns = [(re.compile(r"^myapp$"), "section1"), (re.compile(r"^myapp\.utils$"), "section2")]
    result = _known_pattern("myapp.utils.helpers", config)
    assert result == ("section2", "Matched configured known pattern re.compile('^myapp\\\\.utils$')")

def test_known_pattern_empty_name():
    import re
    from mymodule import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [(re.compile(r"^$"), "section1")]
    result = _known_pattern("", config)
    assert result == ("section1", "Matched configured known pattern re.compile('^$')")

def test_known_pattern_no_known_patterns():
    from mymodule import Config
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = []
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_no_sections():
    import re
    from mymodule import Config
    config = Config()
    config.sections = set()
    config.known_patterns = [(re.compile(r"^myapp$"), "section1")]
    result = _known_pattern("myapp", config)
    assert result is None


# LLM-generated content at query #26
#--------------------------

def test_src_path_is_module_true():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result is True


# LLM-generated content at query #27
#--------------------------

def test_known_pattern_matches_configured_pattern():
    import re
    from collections import namedtuple
    Config = namedtuple('Config', ['known_patterns', 'sections'])
    pattern = re.compile(r"^myapp\.utils$")
    config = Config(known_patterns=[(pattern, "THIRD_PARTY")], sections=["THIRD_PARTY"])
    result = _known_pattern("myapp.utils.logging", config)
    assert result == ("THIRD_PARTY", "Matched configured known pattern " + str(pattern))

def test_known_pattern_no_match():
    import re
    from collections import namedtuple
    Config = namedtuple('Config', ['known_patterns', 'sections'])
    pattern = re.compile(r"^other\.module$")
    config = Config(known_patterns=[(pattern, "THIRD_PARTY")], sections=["THIRD_PARTY"])
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_section_not_in_sections():
    import re
    from collections import namedtuple
    Config = namedtuple('Config', ['known_patterns', 'sections'])
    pattern = re.compile(r"^myapp$")
    config = Config(known_patterns=[(pattern, "THIRD_PARTY")], sections=["FIRST_PARTY"])
    result = _known_pattern("myapp.utils", config)
    assert result is None

def test_known_pattern_matches_longest_module():
    import re
    from collections import namedtuple
    Config = namedtuple('Config', ['known_patterns', 'sections'])
    pattern1 = re.compile(r"^myapp$")
    pattern2 = re.compile(r"^myapp\.utils$")
    config = Config(known_patterns=[(pattern1, "FIRST_PARTY"), (pattern2, "THIRD_PARTY")], sections=["FIRST_PARTY", "THIRD_PARTY"])
    result = _known_pattern("myapp.utils.logging", config)
    assert result == ("THIRD_PARTY", "Matched configured known pattern " + str(pattern2))

def test_known_pattern_empty_name():
    import re
    from collections import namedtuple
    Config = namedtuple('Config', ['known_patterns', 'sections'])
    pattern = re.compile(r"^$")
    config = Config(known_patterns=[(pattern, "FIRST_PARTY")], sections=["FIRST_PARTY"])
    result = _known_pattern("", config)
    assert result == ("FIRST_PARTY", "Matched configured known pattern " + str(pattern))

def test_known_pattern_multiple_patterns_first_matches():
    import re
    from collections import namedtuple
    Config = namedtuple('Config', ['known_patterns', 'sections'])
    pattern1 = re.compile(r"^myapp\.utils$")
    pattern2 = re.compile(r"^myapp\.utils\.logging$")
    config = Config(known_patterns=[(pattern1, "FIRST_PARTY"), (pattern2, "THIRD_PARTY")], sections=["FIRST_PARTY", "THIRD_PARTY"])
    result = _known_pattern("myapp.utils.logging", config)
    assert result == ("FIRST_PARTY", "Matched configured known pattern " + str(pattern1))


# LLM-generated content at query #28
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"test.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("test.namespace.module", config, src_paths=[Path("/src")], prefix=("test",))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    module_path = Path("/src/test/namespace")
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("test.namespace.module", config, src_paths=[Path("/src")], prefix=("test",))
        assert result is not None

def test_nested_module_true_and_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"a.b"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("a.b.c", config, src_paths=[Path("/src")], prefix=("a",))
    assert result is not None

def test_nested_module_true_and_auto_identify_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("x.y.z", config, src_paths=[Path("/src")], prefix=("x",))
        assert result is not None


# LLM-generated content at query #29
#--------------------------

def test_is_module_with_py_extension():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path.with_suffix(".py"))
    original_exists_case_sensitive = __import__('__main__').exists_case_sensitive
    __import__('__main__').exists_case_sensitive = mock_exists_case_sensitive
    result = __import__('__main__')._is_module(path)
    __import__('__main__').exists_case_sensitive = original_exists_case_sensitive
    assert result == True

def test_is_module_with_extension_suffix():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path.with_suffix(".so"))
    original_exists_case_sensitive = __import__('__main__').exists_case_sensitive
    original_EXTENSION_SUFFIXES = __import__('importlib').machinery.EXTENSION_SUFFIXES
    __import__('importlib').machinery.EXTENSION_SUFFIXES = [".so"]
    __import__('__main__').exists_case_sensitive = mock_exists_case_sensitive
    result = __import__('__main__')._is_module(path)
    __import__('__main__').exists_case_sensitive = original_exists_case_sensitive
    __import__('importlib').machinery.EXTENSION_SUFFIXES = original_EXTENSION_SUFFIXES
    assert result == True

def test_is_module_with_init_py():
    path = Path("some_module")
    mock_exists_case_sensitive = lambda p: p == str(path / "__init__.py")
    original_exists_case_sensitive = __import__('__main__').exists_case_sensitive
    __import__('__main__').exists_case_sensitive = mock_exists_case_sensitive
    result = __import__('__main__')._is_module(path)
    __import__('__main__').exists_case_sensitive = original_exists_case_sensitive
    assert result == True


# LLM-generated content at query #30
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_src_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "README.txt").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b"__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_extend_path():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_extend_path_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True


# LLM-generated content at query #31
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        result = _is_module(test_path)
    assert result == True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import patch
    import importlib.machinery
    test_path = Path("some_module")
    with patch('os.path.exists') as mock_exists:
        mock_exists.side_effect = lambda x: x == str(test_path.with_suffix(importlib.machinery.EXTENSION_SUFFIXES[0]))
        result = _is_module(test_path)
    assert result == True

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    with patch('os.path.exists') as mock_exists:
        mock_exists.side_effect = lambda x: x == str(test_path / "__init__.py")
        result = _is_module(test_path)
    assert result == True

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        result = _is_module(test_path)
    assert result == False

def test_is_module_case_sensitive_check():
    from pathlib import Path
    from unittest.mock import patch
    test_path = Path("some_module")
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        result = _is_module(test_path)
    assert result == False


# LLM-generated content at query #32
#--------------------------

def test_namespace_in_config_namespace_packages_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False, src_paths=[Path("/test")], supported_extensions=[".py"])
    result = _src_path("module", config, src_paths=[Path("/test")], prefix=("existing",))
    assert result is None


# LLM-generated content at query #33
#--------------------------

def test_src_path_is_module_true():
    src_path = Path("test_module")
    src_path.mkdir()
    result = _src_path_is_module(src_path, "test_module")
    src_path.rmdir()
    assert result == True


# LLM-generated content at query #34
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    result = _src_path("a.b", config, src_paths=[Path("/src")], prefix=("a",))
    assert result is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_returns_none_for_missing_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("missing", config)
    assert result is None

def test_src_path_handles_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"mypackage"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("mypackage.nested", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_handles_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset())
    result = _src_path("namespace.nested", config, src_paths=[Path("/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_handles_root_module_matching_src_path_name():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_uses_provided_src_paths_parameter():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("mymodule", config, src_paths=[Path("/custom")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")


# LLM-generated content at query #2
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

def test__src_path_returns_firstparty_on_module_match():
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

def test__src_path_handles_nested_module_with_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = {"mypackage"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            with patch("_is_namespace_package", return_value=False):
                result = _src_path("mypackage.nested", config)
                assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /src.")

def test__src_path_handles_src_path_is_module_case():
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

def test__src_path_with_custom_src_paths_argument():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/default")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    custom_src_paths = [Path("/custom")]
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            result = _src_path("mymodule", config, src_paths=custom_src_paths)
            assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /custom.")

def test__src_path_with_prefix_argument():
    from pathlib import Path
    from unittest.mock import Mock, patch
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    with patch("exists_case_sensitive", return_value=True):
        with patch.object(Path, "is_dir", return_value=True):
            result = _src_path("nested", config, prefix=("mypackage",))
            assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /src.")

def test__src_path_returns_none_when_no_match():
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
                result = _src_path("mypackage.nested", config)
                assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = Path("test.py")
    with patch("pathlib.Path.exists_case_sensitive", return_value=True):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import importlib.machinery
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = Path("test.so")
    with patch("importlib.machinery.EXTENSION_SUFFIXES", [".so"]), patch("pathlib.Path.exists_case_sensitive", side_effect=lambda x: str(x).endswith(".so")):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    mock_path = MagicMock(spec=Path)
    mock_path.__truediv__.return_value = Path("test/__init__.py")
    with patch("pathlib.Path.exists_case_sensitive", return_value=True):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import importlib.machinery
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = Path("test.txt")
    mock_path.__truediv__.return_value = Path("test/__init__.txt")
    with patch("importlib.machinery.EXTENSION_SUFFIXES", []), patch("pathlib.Path.exists_case_sensitive", return_value=False):
        result = _is_module(mock_path)
    assert result is False

def test_is_module_first_condition_true_short_circuit():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = Path("test.py")
    with patch("pathlib.Path.exists_case_sensitive", return_value=True):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_second_condition_true():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import importlib.machinery
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.side_effect = lambda x: Path("test.so") if x == ".so" else Path("test.py")
    with patch("importlib.machinery.EXTENSION_SUFFIXES", [".so"]), patch("pathlib.Path.exists_case_sensitive", side_effect=lambda x: str(x).endswith(".so")):
        result = _is_module(mock_path)
    assert result is True

def test_is_module_third_condition_true():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    mock_path = MagicMock(spec=Path)
    mock_path.with_suffix.return_value = Path("test.py")
    mock_path.__truediv__.return_value = Path("test/__init__.py")
    with patch("pathlib.Path.exists_case_sensitive", side_effect=lambda x: str(x).endswith("__init__.py")):
        result = _is_module(mock_path)
    assert result is True


# LLM-generated content at query #5
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "module.py"
        test_file.touch()
        result = _is_module(tmp_path / "module")
        assert result is True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    import tempfile
    import importlib.machinery
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            test_file = tmp_path / f"module{suffix}"
            test_file.touch()
            result = _is_module(tmp_path / "module")
            assert result is True
            test_file.unlink()

def test_is_module_with_init_py():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        init_file = tmp_path / "__init__.py"
        init_file.touch()
        result = _is_module(tmp_path)
        assert result is True

def test_is_module_no_match():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "module.txt"
        test_file.touch()
        result = _is_module(tmp_path / "module")
        assert result is False

def test_is_module_case_sensitive():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "Module.py"
        test_file.touch()
        result = _is_module(tmp_path / "module")
        assert result is False


# LLM-generated content at query #6
#--------------------------

def test_known_pattern_predicate_false():
    class MockPattern:
        def match(self, module_name):
            return False
    class MockConfig:
        sections = ["section1"]
        known_patterns = [(MockPattern(), "placement1")]
    config = MockConfig()
    result = _known_pattern("test.module.name", config)
    assert result is None


# LLM-generated content at query #7
#--------------------------

def test_is_namespace_package_with_nonexistent_path():
    path = Path("/nonexistent")
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

def test_is_namespace_package_with_init_containing_pkg_resources_single_quote():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    init_content = b"__import__('pkg_resources').declare_namespace(__name__)"
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_containing_pkg_resources_double_quote():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    init_content = b'__import__("pkg_resources").declare_namespace(__name__)'
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_containing_pkgutil_single_quote():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    init_content = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_containing_pkgutil_double_quote():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    init_content = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_not_containing_namespace_markers():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    init_content = b"print('hello')"
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_init_containing_marker_beyond_4096_bytes():
    path = Path("/dir/with/init")
    src_extensions = frozenset(["py"])
    init_content = b"a" * 4096 + b"__import__('pkg_resources').declare_namespace(__name__)"
    result = _is_namespace_package(path, src_extensions)
    assert result == False


# LLM-generated content at query #8
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"test.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("test.namespace.module", config, src_paths=[Path("/src")], prefix=("test",))
    assert result is not None


# LLM-generated content at query #9
#--------------------------

def test_src_path_is_module_true_for_matching_dir():
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


# LLM-generated content at query #10
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("pathlib.Path.with_suffix", return_value=Path("some_module.py")), patch("test_module.exists_case_sensitive", return_value=True):
        result = _is_module(path)
        assert result == True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import importlib.machinery
    path = Path("some_module")
    mock_suffix = ".so"
    with patch("importlib.machinery.EXTENSION_SUFFIXES", [mock_suffix]), patch("pathlib.Path.with_suffix", return_value=Path(f"some_module{mock_suffix}")), patch("test_module.exists_case_sensitive", return_value=True):
        result = _is_module(path)
        assert result == True

def test_is_module_with_init_py():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("pathlib.Path.__truediv__", return_value=Path("some_module/__init__.py")), patch("test_module.exists_case_sensitive", return_value=True):
        result = _is_module(path)
        assert result == True

def test_is_module_no_match():
    from pathlib import Path
    from unittest.mock import patch
    path = Path("some_module")
    with patch("test_module.exists_case_sensitive", return_value=False):
        result = _is_module(path)
        assert result == False


# LLM-generated content at query #11
#--------------------------

def test_known_pattern_predicate_false():
    class MockPattern:
        def match(self, module_name):
            return False
    class MockConfig:
        sections = ["section1"]
        known_patterns = [(MockPattern(), "placement1")]
    config = MockConfig()
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_16_true():
    class MockConfig:
        src_paths = []
    config = MockConfig()
    src_path = type('Path', (), {'name': 'root_module', 'resolve': lambda self: self})()
    module_path = type('Path', (), {'is_dir': lambda: False})()
    prefix = ()
    root_module_name = 'root_module'
    result = not prefix and not module_path.is_dir() and src_path.name == root_module_name
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_predicate_at_line_26_true():
    result = (
        _is_module(module_path)
        or _is_package(module_path)
        or _src_path_is_module(src_path, root_module_name)
    )
    assert result == True


# LLM-generated content at query #14
#--------------------------

def test_predicate_at_line_26_true():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    src_path = Path("/test/path")
    root_module_name = "module"
    module_path = (src_path / root_module_name).resolve()
    prefix = ()
    namespace = ".".join((*prefix, root_module_name))
    nested_module = []
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True


# LLM-generated content at query #15
#--------------------------

def test_src_path_finds_module_in_src_paths():
    config = Config()
    config.src_paths = [Path("/tmp/test_src")]
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp/test_src.")

def test_src_path_returns_none_for_missing_module():
    config = Config()
    config.src_paths = [Path("/tmp/test_src")]
    result = _src_path("missingmodule", config)
    assert result is None

def test_src_path_handles_nested_module_in_namespace_package():
    config = Config()
    config.src_paths = [Path("/tmp/test_src")]
    config.namespace_packages = {"namespace"}
    result = _src_path("namespace.submodule", config, prefix=("namespace",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp/test_src.")

def test_src_path_handles_auto_identify_namespace_packages():
    config = Config()
    config.src_paths = [Path("/tmp/test_src")]
    config.auto_identify_namespace_packages = True
    result = _src_path("namespace.submodule", config, prefix=("namespace",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp/test_src.")

def test_src_path_uses_provided_src_paths():
    config = Config()
    custom_src_paths = [Path("/custom/path")]
    result = _src_path("mymodule", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test_src_path_handles_root_module_matching_src_path_name():
    config = Config()
    config.src_paths = [Path("/tmp/test_src")]
    result = _src_path("test_src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp/test_src.")


# LLM-generated content at query #16
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[], supported_extensions=[])
    result = _src_path("some.namespace.module", config, src_paths=[], prefix=("some",))
    assert result is not None


# LLM-generated content at query #17
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_source_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "README.txt").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkg_resources_declare_namespace_single_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkg_resources_declare_namespace_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkgutil_extend_path_single_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkgutil_extend_path_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True


# LLM-generated content at query #18
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    result = _src_path("a.b", config, src_paths=[Path("/test")], prefix=("a",))
    assert result is not None


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_26_evaluates_to_true():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    src_path = Path("/src")
    root_module_name = "mymodule"
    module_path = (src_path / root_module_name).resolve()
    result = _is_module(module_path) or _is_package(module_path) or _src_path_is_module(src_path, root_module_name)
    assert result == True


# LLM-generated content at query #20
#--------------------------

def test_src_path_predicate_at_line_26_true():
    config = Config(src_paths=[Path("/test/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=[".py"])
    result = _src_path("module", config, src_paths=[Path("/test/path")], prefix=())
    assert result is not None
    assert result[0] == sections.FIRSTPARTY
    assert "Found in one of the configured src_paths:" in result[1]


# LLM-generated content at query #21
#--------------------------

def test_is_module_with_py_file():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        py_file = test_path.with_suffix(".py")
        py_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_with_extension_suffix():
    import tempfile
    import os
    from pathlib import Path
    import importlib.machinery
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = test_path.with_suffix(suffix)
            ext_file.touch()
            result = _is_module(test_path)
            assert result == True
            ext_file.unlink()

def test_is_module_with_init_py():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        test_path.mkdir()
        init_file = test_path / "__init__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_without_any():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "module"
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_py():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "Module"
        py_file_lower = test_path.with_suffix(".py")
        py_file_lower.touch()
        result = _is_module(Path(tmpdir) / "module")
        assert result == False

def test_is_module_case_sensitive_extension():
    import tempfile
    import os
    from pathlib import Path
    import importlib.machinery
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "Module"
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        ext_file = test_path.with_suffix(suffix)
        ext_file.touch()
        result = _is_module(Path(tmpdir) / "module")
        assert result == False

def test_is_module_case_sensitive_init():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "Module"
        test_path.mkdir()
        init_file = test_path / "__init__.py"
        init_file.touch()
        result = _is_module(Path(tmpdir) / "module")
        assert result == False


# LLM-generated content at query #22
#--------------------------

def test_is_module_with_py_file():
    from pathlib import Path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "module"
        py_file = test_path.with_suffix(".py")
        py_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_with_extension_suffix():
    from pathlib import Path
    import tempfile
    import importlib.machinery
    import sys
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "module"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = test_path.with_suffix(suffix)
            ext_file.touch()
            result = _is_module(test_path)
            assert result == True
            ext_file.unlink()

def test_is_module_with_init_py():
    from pathlib import Path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "package"
        test_path.mkdir()
        init_file = test_path / "__init__.py"
        init_file.touch()
        result = _is_module(test_path)
        assert result == True

def test_is_module_no_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "nonexistent"
        result = _is_module(test_path)
        assert result == False

def test_is_module_case_sensitive_mismatch():
    from pathlib import Path
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "Module"
        lower_file = tmp_path / "module.py"
        lower_file.touch()
        result = _is_module(test_path)
        assert result == False


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_26_true():
    class MockConfig:
        src_paths = []
        namespace_packages = []
        auto_identify_namespace_packages = False
        supported_extensions = []

    config = MockConfig()
    src_path = type('Path', (), {'resolve': lambda self: self, 'is_dir': lambda: False, 'name': 'mymodule'})()
    root_module_name = 'mymodule'
    result = _src_path_is_module(src_path, root_module_name)
    assert result is True


# LLM-generated content at query #24
#--------------------------

def test_src_path_finds_module_in_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    mock_module_path = Path("/src/module.py")
    mock_module_path.touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
def test_src_path_finds_package_in_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    mock_package_path = Path("/src/package")
    mock_package_path.mkdir()
    (mock_package_path / "__init__.py").touch()
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
def test_src_path_finds_root_module_as_src_path():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src/module")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    mock_src_path = Path("/src/module")
    mock_src_path.mkdir(parents=True)
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src/module.")
def test_src_path_handles_nested_module_in_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = {"namespace"}
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    mock_namespace_path = Path("/src/namespace")
    mock_namespace_path.mkdir()
    mock_nested_path = mock_namespace_path / "nested"
    mock_nested_path.mkdir()
    (mock_nested_path / "__init__.py").touch()
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")
def test_src_path_returns_none_when_module_not_found():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    result = _src_path("nonexistent", config)
    assert result is None
def test_src_path_uses_provided_src_paths():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = False
    config.supported_extensions = frozenset(["py"])
    custom_src_paths = [Path("/custom")]
    mock_module_path = Path("/custom/module.py")
    mock_module_path.touch()
    result = _src_path("module", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")
def test_src_path_handles_auto_identified_namespace_package():
    from pathlib import Path
    from unittest.mock import Mock
    config = Mock()
    config.src_paths = [Path("/src")]
    config.namespace_packages = set()
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    mock_namespace_path = Path("/src/namespace")
    mock_namespace_path.mkdir()
    init_file = mock_namespace_path / "__init__.py"
    init_file.write_bytes(b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    mock_nested_path = mock_namespace_path / "nested"
    mock_nested_path.mkdir()
    (mock_nested_path / "__init__.py").touch()
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")


# LLM-generated content at query #25
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[], supported_extensions=set())
    src_paths = [Path("/some/path")]
    result = _src_path("some.namespace.module", config, src_paths, ("some", "namespace"))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[], supported_extensions={".py"})
    src_paths = [Path("/some/path")]
    with unittest.mock.patch('_is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths, ("some", "namespace"))
        assert result is not None

def test_nested_module_true_and_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[], supported_extensions=set())
    src_paths = [Path("/some/path")]
    result = _src_path("some.namespace.module", config, src_paths, ("some", "namespace"))
    assert result is not None

def test_nested_module_true_and_auto_identify_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[], supported_extensions={".py"})
    src_paths = [Path("/some/path")]
    with unittest.mock.patch('_is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths, ("some", "namespace"))
        assert result is not None


# LLM-generated content at query #26
#--------------------------

def test_src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mymodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_returns_none_for_missing_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("missing", config)
    assert result is None

def test_src_path_handles_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"mypackage"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("mypackage.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_auto_identifies_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_uses_provided_src_paths():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    custom_src_paths = [Path("/custom")]
    result = _src_path("mymodule", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom.")

def test_src_path_handles_root_module_matching_src_path_name():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_with_prefix():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    result = _src_path("nested", config, src_paths=[Path("/src/mypackage")], prefix=("mypackage",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src/mypackage.")


# LLM-generated content at query #27
#--------------------------

def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_pattern_with_wildcard():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate("foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_matches_dot_prefix():
    config = Config(forced_separate=["foo"])
    result = _forced_separate(".foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_dot_prefix_with_wildcard():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate(".foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("bar", config)
    assert result is None

def test_forced_separate_no_match_with_wildcard():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate("bar", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("foo", config)
    assert result is None

def test_forced_separate_multiple_patterns_first_matches():
    config = Config(forced_separate=["foo", "bar"])
    result = _forced_separate("foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_multiple_patterns_second_matches():
    config = Config(forced_separate=["foo", "bar"])
    result = _forced_separate("bar", config)
    assert result == ("bar", "Matched forced_separate (bar) config value.")

def test_forced_separate_pattern_without_wildcard_matches_substring():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("foobar", config)
    assert result is None

def test_forced_separate_pattern_with_wildcard_matches_exact():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate("foo", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")


# LLM-generated content at query #28
#--------------------------

def test__src_path_with_exact_module_match():
    config = Config()
    config.src_paths = [Path("/test/path")]
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.namespace_packages = {"namespace"}
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_auto_identify_namespace_packages_enabled():
    config = Config()
    config.src_paths = [Path("/test/path")]
    config.auto_identify_namespace_packages = True
    config.supported_extensions = frozenset(["py"])
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_no_match():
    config = Config()
    config.src_paths = [Path("/test/path")]
    result = _src_path("unknown", config)
    assert result is None

def test__src_path_with_custom_src_paths():
    config = Config()
    custom_src_paths = [Path("/custom/path")]
    result = _src_path("module", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test__src_path_with_prefix():
    config = Config()
    config.src_paths = [Path("/test/path")]
    prefix = ("pre", "fix")
    result = _src_path("module", config, prefix=prefix)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")

def test__src_path_with_src_path_is_module():
    config = Config()
    config.src_paths = [Path("/test/path")]
    result = _src_path("path", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test/path.")


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = []
    config.sections = []
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #31
#--------------------------

def test_known_pattern_predicate_false():
    class MockPattern:
        def match(self, module_name):
            return False
    class MockConfig:
        sections = ["section1"]
        known_patterns = [(MockPattern(), "placement1")]
    config = MockConfig()
    result = _known_pattern("test.module.name", config)
    assert result is None


# LLM-generated content at query #32
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    from unittest.mock import Mock, patch
    path = Mock(spec=Path)
    path.__truediv__.return_value = Mock(exists=lambda: False)
    path.iterdir.return_value = []
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_without_init_and_no_source_files():
    from pathlib import Path
    from unittest.mock import Mock, patch
    path = Mock(spec=Path)
    init_file = Mock(exists=lambda: False)
    path.__truediv__.return_value = init_file
    file1 = Mock(suffix=".txt", name="file.txt")
    file1.suffix.lstrip.return_value = "txt"
    file1.name.lower.return_value = "file.txt"
    path.iterdir.return_value = [file1]
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_init_and_namespace_declaration():
    from pathlib import Path
    from unittest.mock import Mock, patch
    path = Mock(spec=Path)
    init_file = Mock(exists=lambda: True)
    path.__truediv__.return_value = init_file
    mock_open = Mock()
    mock_open.__enter__.return_value.read.return_value = b"__import__('pkg_resources').declare_namespace(__name__)"
    init_file.open.return_value = mock_open
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_init_and_namespace_declaration_double_quotes():
    from pathlib import Path
    from unittest.mock import Mock, patch
    path = Mock(spec=Path)
    init_file = Mock(exists=lambda: True)
    path.__truediv__.return_value = init_file
    mock_open = Mock()
    mock_open.__enter__.return_value.read.return_value = b'__import__("pkg_resources").declare_namespace(__name__)'
    init_file.open.return_value = mock_open
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_init_and_pkgutil_extend_path():
    from pathlib import Path
    from unittest.mock import Mock, patch
    path = Mock(spec=Path)
    init_file = Mock(exists=lambda: True)
    path.__truediv__.return_value = init_file
    mock_open = Mock()
    mock_open.__enter__.return_value.read.return_value = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    init_file.open.return_value = mock_open
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_namespace_package_with_init_and_pkgutil_extend_path_double_quotes():
    from pathlib import Path
    from unittest.mock import Mock, patch
    path = Mock(spec=Path)
    init_file = Mock(exists=lambda: True)
    path.__truediv__.return_value = init_file
    mock_open = Mock()
    mock_open.__enter__.return_value.read.return_value = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    init_file.open.return_value = mock_open
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True


# LLM-generated content at query #33
#--------------------------

def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_pattern_with_wildcard():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate("foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_matches_pattern_without_wildcard_appended():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("foobar", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_dot_prefixed_name():
    config = Config(forced_separate=["foo"])
    result = _forced_separate(".foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_matches_dot_prefixed_name_with_wildcard():
    config = Config(forced_separate=["foo*"])
    result = _forced_separate(".foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["foo"])
    result = _forced_separate("bar", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("foo", config)
    assert result is None

def test_forced_separate_multiple_patterns_first_matches():
    config = Config(forced_separate=["bar", "foo", "baz"])
    result = _forced_separate("foo", config)
    assert result == ("foo", "Matched forced_separate (foo) config value.")

def test_forced_separate_multiple_patterns_second_matches():
    config = Config(forced_separate=["bar", "foo*", "baz"])
    result = _forced_separate("foobar", config)
    assert result == ("foo*", "Matched forced_separate (foo*) config value.")

def test_forced_separate_pattern_matches_subdirectory():
    config = Config(forced_separate=["foo/bar"])
    result = _forced_separate("foo/bar", config)
    assert result == ("foo/bar", "Matched forced_separate (foo/bar) config value.")

def test_forced_separate_pattern_matches_subdirectory_with_wildcard():
    config = Config(forced_separate=["foo/*"])
    result = _forced_separate("foo/bar", config)
    assert result == ("foo/*", "Matched forced_separate (foo/*) config value.")


# LLM-generated content at query #34
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_src_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "README.txt").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_matching_extensions():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "file.txt").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_setup_cfg():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "setup.cfg").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_pyproject_toml():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "pyproject.toml").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_py_file():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "module.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_init_and_namespace_declaration_single_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_namespace_declaration_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_extend_path_single_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_extend_path_double_quotes():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_and_no_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_not_a_package():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_file.py"
        path.touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


# LLM-generated content at query #35
#--------------------------

def test_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
    assert result is not None

def test_auto_identify_namespace_packages_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
        assert result is not None

def test_nested_module_true_and_namespace_in_config_namespace_packages():
    config = Config(namespace_packages={"some.namespace"}, auto_identify_namespace_packages=False, src_paths=[Path("/src")], supported_extensions=[".py"])
    result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
    assert result is not None

def test_nested_module_true_and_auto_identify_true_and_is_namespace_package():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=True, src_paths=[Path("/src")], supported_extensions=[".py"])
    with unittest.mock.patch('module._is_namespace_package', return_value=True):
        result = _src_path("some.namespace.module", config, src_paths=[Path("/src")], prefix=("some",))
        assert result is not None


# LLM-generated content at query #36
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    nested_module = ["nested"]
    namespace = "prefix.root"
    result = namespace in config.namespace_packages or (config.auto_identify_namespace_packages and _is_namespace_package(module_path, config.supported_extensions))
    assert result is False


# LLM-generated content at query #37
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

def test_is_namespace_package_with_directory_no_init_but_py_files():
    path = Path("/dir/with/py")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_but_setup_cfg():
    path = Path("/dir/with/cfg")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_directory_no_init_but_pyproject_toml():
    path = Path("/dir/with/toml")
    src_extensions = frozenset(["py"])
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_init_containing_pkg_resources_single_quotes():
    path = Path("/dir/with/init1")
    src_extensions = frozenset(["py"])
    init_content = b"__import__('pkg_resources').declare_namespace(__name__)"
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_containing_pkg_resources_double_quotes():
    path = Path("/dir/with/init2")
    src_extensions = frozenset(["py"])
    init_content = b'__import__("pkg_resources").declare_namespace(__name__)'
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_containing_pkgutil_single_quotes():
    path = Path("/dir/with/init3")
    src_extensions = frozenset(["py"])
    init_content = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_containing_pkgutil_double_quotes():
    path = Path("/dir/with/init4")
    src_extensions = frozenset(["py"])
    init_content = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    result = _is_namespace_package(path, src_extensions)
    assert result == True

def test_is_namespace_package_with_init_without_namespace_markers():
    path = Path("/dir/with/init5")
    src_extensions = frozenset(["py"])
    init_content = b"print('hello')"
    result = _is_namespace_package(path, src_extensions)
    assert result == False

def test_is_namespace_package_with_src_extensions_including_txt():
    path = Path("/dir/with/txt")
    src_extensions = frozenset(["py", "txt"])
    result = _is_namespace_package(path, src_extensions)
    assert result == True


# LLM-generated content at query #38
#--------------------------

def test_is_module_with_py_extension():
    path = Path("some_module")
    mock_exists = lambda p: p == str(path.with_suffix(".py"))
    global exists_case_sensitive
    original_exists = exists_case_sensitive
    exists_case_sensitive = mock_exists
    result = _is_module(path)
    exists_case_sensitive = original_exists
    assert result == True

def test_is_module_with_extension_suffix():
    path = Path("some_module")
    mock_exists = lambda p: p == str(path.with_suffix(importlib.machinery.EXTENSION_SUFFIXES[0]))
    global exists_case_sensitive
    original_exists = exists_case_sensitive
    exists_case_sensitive = mock_exists
    result = _is_module(path)
    exists_case_sensitive = original_exists
    assert result == True

def test_is_module_with_init_py():
    path = Path("some_module")
    mock_exists = lambda p: p == str(path / "__init__.py")
    global exists_case_sensitive
    original_exists = exists_case_sensitive
    exists_case_sensitive = mock_exists
    result = _is_module(path)
    exists_case_sensitive = original_exists
    assert result == True


# LLM-generated content at query #39
#--------------------------

def test_namespace_not_in_config_and_auto_identify_false():
    config = Config(namespace_packages=set(), auto_identify_namespace_packages=False)
    result = _src_path("a.b", config, src_paths=[Path("/tmp")], prefix=("a",))
    assert result is not None


# LLM-generated content at query #40
#--------------------------

def test_src_path_is_module_true_for_matching_dir():
    src_path = Path("/some/path/mymodule")
    src_path.is_dir = lambda: True
    exists_case_sensitive = lambda x: True
    result = _src_path_is_module(src_path, "mymodule")
    assert result is True

def test_src_path_is_module_false_for_wrong_name():
    src_path = Path("/some/path/mymodule")
    src_path.is_dir = lambda: True
    exists_case_sensitive = lambda x: True
    result = _src_path_is_module(src_path, "othermodule")
    assert result is False

def test_src_path_is_module_false_for_file():
    src_path = Path("/some/path/mymodule")
    src_path.is_dir = lambda: False
    exists_case_sensitive = lambda x: True
    result = _src_path_is_module(src_path, "mymodule")
    assert result is False

def test_src_path_is_module_false_for_case_mismatch():
    src_path = Path("/some/path/mymodule")
    src_path.is_dir = lambda: True
    exists_case_sensitive = lambda x: False
    result = _src_path_is_module(src_path, "mymodule")
    assert result is False

def test_src_path_is_module_false_for_all_conditions_failing():
    src_path = Path("/some/path/mymodule")
    src_path.is_dir = lambda: False
    exists_case_sensitive = lambda x: False
    result = _src_path_is_module(src_path, "othermodule")
    assert result is False


# LLM-generated content at query #41
#--------------------------

def test_namespace_package_without_init_and_no_files():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_without_init_and_no_source_files_but_has_setup_cfg():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "setup.cfg").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_no_source_files_but_has_pyproject_toml():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "pyproject.toml").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_without_init_and_has_source_file():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        (path / "module.py").touch()
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_namespace_package_with_init_containing_pkg_resources_declare_namespace_single_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b"__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkg_resources_declare_namespace_double_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkgutil_extend_path_single_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_containing_pkgutil_extend_path_double_quotes():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == True

def test_namespace_package_with_init_without_namespace_declaration():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_bytes(b"")
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False

def test_not_a_package():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pkg"
        src_extensions = frozenset(["py"])
        result = _is_namespace_package(path, src_extensions)
        assert result == False


