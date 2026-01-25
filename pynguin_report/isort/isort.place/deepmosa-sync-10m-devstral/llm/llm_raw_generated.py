####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_known_pattern_matches_first_segment():
    config = Config(known_patterns=[(re.compile("foo"), "placement1")], sections=["placement1"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement1", "Matched configured known pattern foo")

def test_known_pattern_matches_middle_segment():
    config = Config(known_patterns=[(re.compile("bar"), "placement2")], sections=["placement2"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement2", "Matched configured known pattern bar")

def test_known_pattern_matches_last_segment():
    config = Config(known_patterns=[(re.compile("baz"), "placement3")], sections=["placement3"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement3", "Matched configured known pattern baz")

def test_known_pattern_no_match():
    config = Config(known_patterns=[(re.compile("qux"), "placement4")], sections=["placement4"])
    result = _known_pattern("foo.bar.baz", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile("foo"), "placement5")], sections=["placement1"])
    result = _known_pattern("foo.bar.baz", config)
    assert result is None

def test_known_pattern_empty_name():
    config = Config(known_patterns=[(re.compile("foo"), "placement1")], sections=["placement1"])
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_single_segment_match():
    config = Config(known_patterns=[(re.compile("single"), "placement1")], sections=["placement1"])
    result = _known_pattern("single", config)
    assert result == ("placement1", "Matched configured known pattern single")

def test_known_pattern_single_segment_no_match():
    config = Config(known_patterns=[(re.compile("other"), "placement1")], sections=["placement1"])
    result = _known_pattern("single", config)
    assert result is None


# LLM-generated content at query #2
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = [("test.*", "section1")]
    config.sections = ["section2"]
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #3
#--------------------------

```python
def test_src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test_src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    result = _src_path("module", config)
    assert result is None

def test_src_path_with_non_existing_module():
    config = Config(src_paths=[Path("/tmp")])
    result = _src_path("non_existing_module", config)
    assert result is None

def test_src_path_with_existing_module_in_src_paths():
    config = Config(src_paths=[Path("/tmp")])
    (Path("/tmp") / "existing_module.py").touch()
    result = _src_path("existing_module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test_src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/tmp")], namespace_packages={"parent"})
    (Path("/tmp") / "parent").mkdir()
    (Path("/tmp") / "parent" / "child.py").touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test_src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/tmp")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    (Path("/tmp") / "parent").mkdir()
    (Path("/tmp") / "parent" / "child.py").touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test_src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/tmp")])
    (Path("/tmp") / "module").mkdir()
    (Path("/tmp") / "module" / "__init__.py").touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test_src_path_with_prefix():
    config = Config(src_paths=[Path("/tmp")])
    (Path("/tmp") / "parent").mkdir()
    (Path("/tmp") / "parent" / "child").mkdir()
    (Path("/tmp") / "parent" / "child" / "__init__.py").touch()
    result = _src_path("parent.child", config, prefix=("parent",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test_src_path_with_non_dir_module_in_root():
    config = Config(src_paths=[Path("/tmp")])
    (Path("/tmp") / "module.py").touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")


# LLM-generated content at query #4
#--------------------------

```python
def test_forced_separate_matching_pattern():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("test_file", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matching_hidden_pattern():
    config = Config(forced_separate=["test*"])
    result = _forced_separate(".test_file", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_non_matching_pattern():
    config = Config(forced_separate=["other*"])
    result = _forced_separate("test_file", config)
    assert result is None

def test_forced_separate_without_asterisk():
    config = Config(forced_separate=["test"])
    result = _forced_separate("test_file", config)
    assert result == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("test_file", config)
    assert result is None

def test_forced_separate_multiple_patterns():
    config = Config(forced_separate=["test*", "other*"])
    result = _forced_separate("other_file", config)
    assert result == ("other*", "Matched forced_separate (other*) config value.")


# LLM-generated content at query #5
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/non/existent/path")])
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("your_module.exists_case_sensitive", return_value=True):
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("your_module.exists_case_sensitive", return_value=True), \
         patch("your_module._is_namespace_package", return_value=True):
        result = _src_path("parent.child", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_module_in_root_src_path():
    config = Config(src_paths=[Path("/path/to/module")])
    with patch("pathlib.Path.is_dir", return_value=False), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("your_module.exists_case_sensitive", return_value=True):
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/module.")

def test__src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("your_module.exists_case_sensitive", return_value=True), \
         patch("your_module._is_namespace_package", return_value=True):
        result = _src_path("parent.child", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #6
#--------------------------

```python
def test__src_path_with_nested_module_and_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"test"}, supported_extensions=frozenset({"py"}))
    assert _src_path("test.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_nested_module_and_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("test.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/src")])
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/src")])
    assert _src_path("package", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/src/module")])
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src/module.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/src")])
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #7
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    path = Path("not_a_package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    path = Path("package_with_init")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration_single_quotes():
    path = Path("namespace_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration_double_quotes():
    path = Path("namespace_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration_single_quotes():
    path = Path("namespace_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration_double_quotes():
    path = Path("namespace_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_source_files_exist():
    path = Path("package_with_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_setup_cfg_exists():
    path = Path("package_with_setup")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_pyproject_toml_exists():
    path = Path("package_with_pyproject")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files():
    path = Path("empty_package")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_16():
    name = "module"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src/module")]
    prefix = ()
    module_path = (Path("src/module")).resolve()
    assert not prefix and not module_path.is_dir() and Path("src").name == "module"


# LLM-generated content at query #9
#--------------------------

```python
def test_forced_separate_matches_with_and_without_leading_dot():
    config = Config(forced_separate=["test"])
    assert _forced_separate("testfile", config) == ("test", "Matched forced_separate (test) config value.")
    assert _forced_separate(".testfile", config) == ("test", "Matched forced_separate (test) config value.")


# LLM-generated content at query #10
#--------------------------

```python
def test_src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == True

def test_src_path_is_module_returns_false_for_non_directory():
    src_path = Path("not_a_directory")
    src_path.is_dir = lambda: False
    assert _src_path_is_module(src_path, "not_a_directory") == False

def test_src_path_is_module_returns_false_for_name_mismatch():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name") == False

def test_src_path_is_module_returns_false_for_case_sensitive_mismatch():
    src_path = Path("MODULE_NAME")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name") == False


# LLM-generated content at query #11
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(str(path.with_suffix(ext_suffix)))
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("package")
    assert _is_module(path) == exists_case_sensitive("package/__init__.py")

def test_is_module_non_existent():
    path = Path("nonexistent")
    assert _is_module(path) == False


# LLM-generated content at query #12
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = [("pattern", "placement")]
    config.sections = []
    name = "test.module"
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test.namespace"},
        auto_identify_namespace_packages=False,
        src_paths=[Path("/test/path")],
        supported_extensions=[".py"]
    )
    name = "test.namespace.module"
    src_paths = [Path("/test/path")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #14
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("", config) is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    src_path = Path("/some/path/module.py")
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    src_path = Path("/some/path/package")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "__init__.py").touch()
    assert _src_path("package", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/some/path")], namespace_packages={"parent"})
    src_path = Path("/some/path/parent")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "child.py").touch()
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path/parent.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/some/path")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    src_path = Path("/some/path/parent")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "child.py").touch()
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path/parent.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/some/path/module")])
    src_path = Path("/some/path/module")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "__init__.py").touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path/module.")


# LLM-generated content at query #15
#--------------------------

```python
def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/test")], namespace_packages={"test"}, supported_extensions={"py"})
    assert _src_path("test.module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /test.")


# LLM-generated content at query #16
#--------------------------

```python
def test__src_path_with_exact_match_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #17
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("test_module")
    (path / "test_module.py").touch()
    assert _is_module(path) is True


# LLM-generated content at query #18
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    assert _is_module(Path("example.py")) is True

def test_is_module_returns_true_for_extension_suffix():
    assert _is_module(Path("example.so")) is True

def test_is_module_returns_true_for_init_file():
    assert _is_module(Path("example/__init__.py")) is True


# LLM-generated content at query #19
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(f"module{ext_suffix}")
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("package")
    assert _is_module(path) == exists_case_sensitive("package/__init__.py")

def test_is_module_without_any_file():
    path = Path("nonexistent")
    assert _is_module(path) == False


# LLM-generated content at query #20
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = [("pattern1", "placement1"), ("pattern2", "placement2")]
    config.sections = ["section1", "section2"]
    result = _known_pattern("test.module.name", config)
    assert result is None


# LLM-generated content at query #21
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #22
#--------------------------

```python
def test__src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_finds_nested_module():
    config = Config(src_paths=[Path("/project/src")])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_finds_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"})
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_returns_none_when_not_found():
    config = Config(src_paths=[Path("/project/src")])
    result = _src_path("nonexistent", config)
    assert result is None

def test__src_path_finds_module_in_custom_src_paths():
    config = Config(src_paths=[Path("/custom/src")])
    result = _src_path("module", config, src_paths=[Path("/custom/src")])
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/src.")

def test__src_path_finds_module_with_prefix():
    config = Config(src_paths=[Path("/project/src")])
    result = _src_path("module", config, prefix=("parent",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")


# LLM-generated content at query #23
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_namespace_package_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True)
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_module_not_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #24
#--------------------------

```python
def test__is_namespace_package_returns_true_when_package_has_no_init_file_and_no_source_files():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)
        path.mkdir()
        assert _is_namespace_package(path, frozenset([".py"])) is True


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "test_module"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=[],
        auto_identify_namespace_packages=False
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is None


# LLM-generated content at query #26
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(f"module{ext_suffix}")
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module/__init__.py")

def test_is_module_no_match():
    path = Path("not_a_module")
    assert _is_module(path) == False


# LLM-generated content at query #27
#--------------------------

```python
def test__src_path_simple_module():
    config = Config(src_paths=[Path("/project")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_nested_module():
    config = Config(src_paths=[Path("/project")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_namespace_package():
    config = Config(src_paths=[Path("/project")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/parent.")

def test__src_path_auto_identify_namespace():
    config = Config(src_paths=[Path("/project")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/parent.")

def test__src_path_not_found():
    config = Config(src_paths=[Path("/project")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test_namespace"},
        auto_identify_namespace_packages=False,
        src_paths=[Path("/test/src")],
        supported_extensions=[".py"]
    )
    name = "test_namespace.submodule"
    src_paths = [Path("/test/src")]
    prefix = ()
    new_prefix = ("test_namespace",)
    namespace = "test_namespace"
    module_path = (Path("/test/src") / "test_namespace").resolve()

    assert namespace in config.namespace_packages or (
        config.auto_identify_namespace_packages
        and _is_namespace_package(module_path, config.supported_extensions)
    )


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages=["module"],
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #30
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/tmp")])
    result = _src_path("", config)
    assert result is None

def test__src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/tmp")])
    result = _src_path("non_existent_module", config)
    assert result is None

def test__src_path_with_existing_module():
    config = Config(src_paths=[Path("/tmp")])
    module_path = Path("/tmp/existing_module.py")
    module_path.touch()
    result = _src_path("existing_module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/tmp")])
    module_path = Path("/tmp/parent/child.py")
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/tmp")], namespace_packages=["parent"])
    module_path = Path("/tmp/parent/child.py")
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/tmp")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    module_path = Path("/tmp/parent")
    module_path.mkdir(parents=True, exist_ok=True)
    (module_path / "child.py").touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/tmp/module")])
    module_path = Path("/tmp/module")
    module_path.mkdir(parents=True, exist_ok=True)
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp/module.")


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_26():
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False, supported_extensions=[".py"])
    name = "module.submodule"
    src_paths = [Path("src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module.py")
    assert _is_module(path) == True

def test_is_module_with_extension_suffix():
    path = Path("module.so")
    assert _is_module(path) == True

def test_is_module_with_init_file():
    path = Path("package")
    assert _is_module(path) == True

def test_is_module_without_any_file():
    path = Path("nonexistent")
    assert _is_module(path) == False


# LLM-generated content at query #33
#--------------------------

```python
def test__is_namespace_package_returns_true_when_package_has_no_init_file_and_no_source_files():
    from pathlib import Path

    path = Path("test_package")
    path.mkdir()
    src_extensions = frozenset([".py"])

    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #34
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    path = Path("not_a_package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #35
#--------------------------

```python
def test__src_path_is_module_returns_true():
    assert _src_path_is_module(Path("module"), "module") is True


# LLM-generated content at query #36
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None, namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("", config)
    assert result is None

def test__src_path_with_nested_module_and_namespace_package():
    config = Config(src_paths=[Path("/path")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.resolve", return_value=Path("/path/parent")):
                with patch("_src_path", return_value=("FIRSTPARTY", "Found in one of the configured src_paths: /path.")):
                    result = _src_path("parent.child", config)
                    assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /path.")

def test__src_path_with_nested_module_and_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/path")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({".py"}))
    with patch("builtins.exists_case_sensitive", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.resolve", return_value=Path("/path/parent")):
                with patch("_is_namespace_package", return_value=True):
                    with patch("_src_path", return_value=("FIRSTPARTY", "Found in one of the configured src_paths: /path.")):
                        result = _src_path("parent.child", config)
                        assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /path.")

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.resolve", return_value=Path("/path/module")):
                with patch("_is_module", return_value=True):
                    result = _src_path("module", config)
                    assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /path.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.resolve", return_value=Path("/path/package")):
                with patch("_is_package", return_value=True):
                    result = _src_path("package", config)
                    assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /path.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/module")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.resolve", return_value=Path("/path/module")):
                with patch("_src_path_is_module", return_value=True):
                    result = _src_path("module", config)
                    assert result == ("FIRSTPARTY", "Found in one of the configured src_paths: /path/module.")


# LLM-generated content at query #37
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    path = Path("non_existent_path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)
    init_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_init_file_exists_with_pkg_resources_declaration():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)
    init_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_init_file_exists_with_pkgutil_declaration():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)
    init_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_no_init_file_but_has_source_files():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    source_file = path / "module.py"
    source_file.write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)
    source_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_no_init_file_but_has_setup_cfg():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    setup_cfg = path / "setup.cfg"
    setup_cfg.write_text("[metadata]\nname = test_package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)
    setup_cfg.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_no_init_file_but_has_pyproject_toml():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    pyproject_toml = path / "pyproject.toml"
    pyproject_toml.write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)
    pyproject_toml.unlink()
    path.rmdir()

def test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files():
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)
    path.rmdir()


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages={"module"},
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #39
#--------------------------

```python
def test_known_pattern_predicate_false():
    name = "test.module"
    config = Config()
    config.known_patterns = [("test.*", "placement")]
    config.sections = ["other_section"]
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #40
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("src", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    config = Config(
        namespace_packages=(),
        auto_identify_namespace_packages=False,
        src_paths=(Path("src"),),
        supported_extensions=(".py",)
    )
    name = "module.submodule"
    src_paths = (Path("src"),)
    prefix = ()
    namespace = "module"
    module_path = (src_paths[0] / "module").resolve()
    assert not (
        namespace in config.namespace_packages
        or (
            config.auto_identify_namespace_packages
            and _is_namespace_package(module_path, config.supported_extensions)
        )
    )


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "module.submodule"
    config = Config(
        namespace_packages=(),
        auto_identify_namespace_packages=False,
        src_paths=(Path("/path/to/src"),),
        supported_extensions=(".py",)
    )
    src_paths = (Path("/path/to/src"),)
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is None


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["module"],
        auto_identify_namespace_packages=False
    )
    result = _src_path(name, config)
    assert result is not None


# LLM-generated content at query #44
#--------------------------

```python
def test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name") is True

def test_src_path_is_not_module_when_name_does_not_match():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name") is False

def test_src_path_is_not_module_when_not_directory():
    src_path = Path("module_name")
    src_path.is_dir = lambda: False
    assert _src_path_is_module(src_path, "module_name") is False

def test_src_path_is_not_module_when_does_not_exist_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name") is False


# LLM-generated content at query #45
#--------------------------

```python
def test_src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module")

def test_src_path_is_module_returns_false_for_non_matching_name():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "valid_module")

def test_src_path_is_module_returns_false_for_non_directory():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: False
    assert not _src_path_is_module(src_path, "valid_module")

def test_src_path_is_module_returns_false_for_case_sensitive_mismatch():
    src_path = Path("Valid_Module")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "valid_module")


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test.namespace"},
        auto_identify_namespace_packages=False,
        src_paths=[Path("/test/path")],
        supported_extensions=[".py"]
    )
    name = "test.namespace.module"
    src_paths = [Path("/test/path")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #47
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    assert _is_module(path) is True


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test.namespace"},
        auto_identify_namespace_packages=False,
        supported_extensions={".py"},
        src_paths=[Path("/test/path")]
    )
    name = "test.namespace.module"
    src_paths = [Path("/test/path")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #49
#--------------------------

```python
def test__is_namespace_package_returns_true_for_valid_namespace_package():
    path = Path("valid_namespace_package")
    path.mkdir()
    (path / "__init__.py").write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #50
#--------------------------

```python
def test__is_namespace_package_with_namespace_declaration():
    path = Path("test_namespace")
    path.mkdir()
    init_file = path / "__init__.py"
    init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
    assert _is_namespace_package(path, frozenset({"py"})) is True


# LLM-generated content at query #51
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_init_file():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("# Regular package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_pkg_resources_declaration():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_pkgutil_declaration():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_py_files():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("# Some code")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_setup_cfg():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname=package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_pyproject_toml():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = []")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_empty_package():
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #52
#--------------------------

```python
def test__is_namespace_package_with_empty_directory():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_non_package():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "non_package"
        path.mkdir()
        (path / "file.txt").write_text("content")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "__init__.py").write_text("content")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_namespace_declaration():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "namespace_package"
        path.mkdir()
        (path / "__init__.py").write_text(
            "__import__('pkg_resources').declare_namespace(__name__)"
        )
        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pkgutil_declaration():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "namespace_package"
        path.mkdir()
        (path / "__init__.py").write_text(
            "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
        )
        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_py_files():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "module.py").write_text("content")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_setup_cfg():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "setup.cfg").write_text("content")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pyproject_toml():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "pyproject.toml").write_text("content")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #53
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_package():
    from pathlib import Path

    path = Path("/non/package/path")
    src_extensions = frozenset([".py"])

    assert _is_namespace_package(path, src_extensions) is False


# LLM-generated content at query #54
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_package():
    assert _is_namespace_package(Path("not_a_package"), frozenset([".py"])) is False


# LLM-generated content at query #55
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_package():
    assert _is_namespace_package(Path("/non/package"), frozenset([".py"])) is False


# LLM-generated content at query #56
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "module.submodule"
    config = Config(
        namespace_packages=(),
        auto_identify_namespace_packages=False,
        src_paths=(Path("/some/path"),),
        supported_extensions=(".py",)
    )
    src_paths = (Path("/some/path"),)
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is None


# LLM-generated content at query #57
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    (path / "module.py").touch()
    assert _is_module(path) is True


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #59
#--------------------------

```python
def test__src_path_is_module_returns_true_when_conditions_are_met():
    src_path = Path("example_module")
    src_path.is_dir = lambda: True
    module_name = "example_module"
    with patch("pathlib.Path.exists", return_value=True):
        assert _src_path_is_module(src_path, module_name) is True


# LLM-generated content at query #60
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages={"module"},
        auto_identify_namespace_packages=False,
        src_paths=[Path("/path/to/src")],
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #61
#--------------------------

```python
def test__is_namespace_package_returns_false_for_non_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_init():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("# Regular __init__.py")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_py_files():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("# Some module")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_setup_cfg():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_pyproject_toml():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_namespace_package_with_pkg_resources():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_namespace_package_with_pkgutil():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_empty_package():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    assert _is_namespace_package(path, frozenset([".py"]))


# LLM-generated content at query #62
#--------------------------

```python
def test_is_module_with_py_file():
    assert _is_module(Path("module")) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES:
        assert _is_module(Path("module")) == exists_case_sensitive(f"module{ext_suffix}")

def test_is_module_with_init_py():
    assert _is_module(Path("package")) == exists_case_sensitive("package/__init__.py")

def test_is_module_non_existent():
    assert _is_module(Path("nonexistent")) == False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test_src_path_with_empty_name():
    config = Config(src_paths=[Path("/some/path")])
    result = _src_path("", config)
    assert result is None

def test_src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/some/path")])
    result = _src_path("non_existent_module", config)
    assert result is None

def test_src_path_with_existing_module_file():
    config = Config(src_paths=[Path("/some/path")])
    module_path = Path("/some/path/module.py")
    module_path.touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")
    module_path.unlink()

def test_src_path_with_existing_package():
    config = Config(src_paths=[Path("/some/path")])
    package_path = Path("/some/path/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")
    (package_path / "__init__.py").unlink()
    package_path.rmdir()

def test_src_path_with_nested_module():
    config = Config(src_paths=[Path("/some/path")])
    package_path = Path("/some/path/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    (package_path / "nested.py").touch()
    result = _src_path("package.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")
    (package_path / "nested.py").unlink()
    (package_path / "__init__.py").unlink()
    package_path.rmdir()

def test_src_path_with_namespace_package():
    config = Config(src_paths=[Path("/some/path")], namespace_packages={"package"})
    package_path = Path("/some/path/package")
    package_path.mkdir()
    result = _src_path("package.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")
    package_path.rmdir()

def test_src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/some/path")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    package_path = Path("/some/path/package")
    package_path.mkdir()
    result = _src_path("package.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")
    package_path.rmdir()

def test_src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/some/path/module")])
    module_path = Path("/some/path/module")
    module_path.mkdir()
    (module_path / "__init__.py").touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path/module.")
    (module_path / "__init__.py").unlink()
    module_path.rmdir()


# LLM-generated content at query #2
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(f"module{ext_suffix}")
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module/__init__.py")

def test_is_module_non_existent():
    path = Path("non_existent_module")
    assert _is_module(path) == False


# LLM-generated content at query #3
#--------------------------

```python
def test_forced_separate_matching_pattern():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("testfile.txt", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matching_hidden_pattern():
    config = Config(forced_separate=["test*"])
    result = _forced_separate(".testfile.txt", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_non_matching_pattern():
    config = Config(forced_separate=["other*"])
    result = _forced_separate("testfile.txt", config)
    assert result is None

def test_forced_separate_exact_match_without_wildcard():
    config = Config(forced_separate=["test"])
    result = _forced_separate("testfile.txt", config)
    assert result == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_empty_forced_separate_list():
    config = Config(forced_separate=[])
    result = _forced_separate("testfile.txt", config)
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_forced_separate_matching_pattern():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("testfile.txt", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matching_pattern_with_dot_prefix():
    config = Config(forced_separate=["test*"])
    result = _forced_separate(".testfile.txt", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_non_matching_pattern():
    config = Config(forced_separate=["other*"])
    result = _forced_separate("testfile.txt", config)
    assert result is None

def test_forced_separate_pattern_without_wildcard():
    config = Config(forced_separate=["test"])
    result = _forced_separate("testfile.txt", config)
    assert result == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_multiple_patterns():
    config = Config(forced_separate=["test*", "other*"])
    result = _forced_separate("otherfile.txt", config)
    assert result == ("other*", "Matched forced_separate (other*) config value.")

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("testfile.txt", config)
    assert result is None


# LLM-generated content at query #5
#--------------------------

```python
def test_src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_finds_nested_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_returns_none_for_non_existent_module():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("nonexistent", config) is None

def test_src_path_handles_namespace_packages():
    config = Config(src_paths=[Path("/src")], namespace_packages={"parent"}, supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_handles_auto_identified_namespace_packages():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=True)
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_with_custom_src_paths():
    config = Config(src_paths=[Path("/custom/src")], namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/src.")

def test_src_path_with_prefix():
    config = Config(src_paths=[Path("/src")], namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("module", config, prefix=("parent",)) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test_src_path_with_none_src_paths():
    config = Config(src_paths=None, namespace_packages=set(), supported_extensions=frozenset([".py"]), auto_identify_namespace_packages=False)
    assert _src_path("module", config) is None


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_16_evaluates_to_true():
    name = "module"
    config = Config(src_paths=[Path("src")])
    src_paths = [Path("src/module")]
    prefix = ()
    root_module_name, *nested_module = name.split(".", 1)
    new_prefix = (*prefix, root_module_name)
    module_path = (src_paths[0] / root_module_name).resolve()
    assert not prefix and not module_path.is_dir() and src_paths[0].name == root_module_name


# LLM-generated content at query #7
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_exists_without_namespace_pattern():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("print('hello')")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_exists_with_pkg_resources_pattern():
    from pathlib import Path
    path = Path("/tmp/test_namespace_pkg")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_exists_with_pkgutil_pattern():
    from pathlib import Path
    path = Path("/tmp/test_namespace_pkgutil")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_no_init_but_has_py_files():
    from pathlib import Path
    path = Path("/tmp/test_with_py_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("x = 1")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_no_init_but_has_setup_cfg():
    from pathlib import Path
    path = Path("/tmp/test_with_setup_cfg")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_no_init_and_no_relevant_files():
    from pathlib import Path
    path = Path("/tmp/test_empty_namespace")
    path.mkdir(exist_ok=True)
    (path / "README.md").write_text("# Test")
    assert _is_namespace_package(path, frozenset([".py"]))


# LLM-generated content at query #8
#--------------------------

```python
def test__src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.mkdir(exist_ok=True)
    assert _src_path_is_module(src_path, "valid_module") is True

def test__src_path_is_module_returns_false_for_non_matching_name():
    src_path = Path("valid_module")
    src_path.mkdir(exist_ok=True)
    assert _src_path_is_module(src_path, "invalid_module") is False

def test__src_path_is_module_returns_false_for_file():
    src_path = Path("file_module.py")
    src_path.touch()
    assert _src_path_is_module(src_path, "file_module") is False

def test__src_path_is_module_returns_false_for_nonexistent_path():
    src_path = Path("nonexistent_module")
    assert _src_path_is_module(src_path, "nonexistent_module") is False

def test__src_path_is_module_returns_false_for_case_sensitive_mismatch():
    src_path = Path("CaseSensitiveModule")
    src_path.mkdir(exist_ok=True)
    assert _src_path_is_module(src_path, "casesensitivemodule") is False


# LLM-generated content at query #9
#--------------------------

```python
def test__src_path_is_module_returns_true_when_conditions_met():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is True


# LLM-generated content at query #10
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #11
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(f"module{ext_suffix}")
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module/__init__.py")

def test_is_module_non_existent():
    path = Path("non_existent_module")
    assert _is_module(path) == False


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_16():
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False, supported_extensions=[".py"])
    src_path = Path("src")
    name = "module"
    prefix = ()
    module_path = (src_path / name).resolve()
    assert not prefix and not module_path.is_dir() and src_path.name == name


# LLM-generated content at query #13
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "__init__.py").write_text("# regular package")
        assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_file_has_pkg_resources_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
        assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_file_has_pkgutil_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_py_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "module.py").write_text("# some code")
        assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_setup_cfg():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "setup.cfg").write_text("[metadata]\nname = test")
        assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_pyproject_toml():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
        assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_package_is_empty():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "package"
        path.mkdir()
        assert _is_namespace_package(path, frozenset([".py"]))


# LLM-generated content at query #14
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("test_module")
    (path / "test_module.py").touch()
    assert _is_module(path) is True


# LLM-generated content at query #15
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src/parent.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src/parent.")

def test__src_path_with_module_not_found():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("nonexistent", config)
    assert result is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("", config)
    assert result is None

def test__src_path_with_root_module_name_matching_src_path():
    config = Config(src_paths=[Path("/path/to/module")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/module.")


# LLM-generated content at query #16
#--------------------------

```python
def test_known_pattern_matches_first_pattern():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_matches_second_pattern():
    config = Config(known_patterns=[(re.compile("foo.*"), "placement1"), (re.compile("bar.*"), "placement2")], sections=["placement1", "placement2"])
    result = _known_pattern("bar.module", config)
    assert result == ("placement2", "Matched configured known pattern bar.*")

def test_known_pattern_no_match():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("other.module", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement2"])
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_matches_submodule():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("test.sub.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_empty_name():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("", config)
    assert result is None


# LLM-generated content at query #17
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_module_not_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #18
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_not_found():
    config = Config(src_paths=[Path("/fake/path")])
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_found_in_src_path():
    config = Config(src_paths=[Path("/real/path")])
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.with_suffix") as mock_with_suffix:
        mock_with_suffix.return_value.exists.return_value = True
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/real/path")], namespace_packages={"parent"})
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("_is_namespace_package", return_value=True), \
         patch("_src_path", return_value=("FIRSTPARTY", "Found in nested namespace.")) as mock_src_path:
        result = _src_path("parent.child", config)
        mock_src_path.assert_called_once_with("child", config, (Path("/real/path/parent"),), ("parent",))
        assert result == ("FIRSTPARTY", "Found in nested namespace.")

def test__src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/real/path")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("_is_namespace_package", return_value=True), \
         patch("_src_path", return_value=("FIRSTPARTY", "Found in auto-identified namespace.")) as mock_src_path:
        result = _src_path("parent.child", config)
        mock_src_path.assert_called_once_with("child", config, (Path("/real/path/parent"),), ("parent",))
        assert result == ("FIRSTPARTY", "Found in auto-identified namespace.")

def test__src_path_with_module_in_root_src_path():
    config = Config(src_paths=[Path("/root/module")])
    with patch("pathlib.Path.resolve", return_value=Path("/root/module")), \
         patch("pathlib.Path.is_dir", return_value=False), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("_is_module", return_value=True):
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /root/module.")


# LLM-generated content at query #19
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    assert _is_module(path) is True


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test_namespace"},
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"],
    )
    src_paths = [Path("/path/to/src")]
    name = "test_namespace.submodule"
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=True, supported_extensions=[".py"])
    src_paths = [Path("src")]
    prefix = ()
    module_path = (Path("src") / "module").resolve()
    assert (
        _is_module(module_path)
        or _is_package(module_path)
        or _src_path_is_module(Path("src"), "module")
    )


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_16():
    name = "module"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #24
#--------------------------

```python
def test__is_namespace_package_returns_true():
    path = Path("valid_namespace_package")
    src_extensions = frozenset([".py", ".pyx", ".pxd"])

    # Mocking the necessary conditions
    path.is_dir.return_value = True
    (path / "__init__.py").exists.return_value = True

    with patch("builtins.open", mock_open(read_data=b'__import__("pkg_resources").declare_namespace(__name__)')):
        assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #25
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    src_path = Path("/path/to/src")
    module_path = src_path / "module"
    module_path.mkdir()
    (module_path / "__init__.py").write_text("")
    config = Config(src_paths=[src_path])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_nested_module_in_src_paths():
    src_path = Path("/path/to/src")
    module_path = src_path / "parent" / "child"
    module_path.mkdir(parents=True)
    (module_path / "__init__.py").write_text("")
    config = Config(src_paths=[src_path])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_namespace_package():
    src_path = Path("/path/to/src")
    namespace_path = src_path / "namespace"
    namespace_path.mkdir()
    config = Config(src_paths=[src_path], namespace_packages=["namespace"])
    result = _src_path("namespace.module", config)
    assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_auto_identified_namespace_package():
    src_path = Path("/path/to/src")
    namespace_path = src_path / "namespace"
    namespace_path.mkdir()
    config = Config(src_paths=[src_path], auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    result = _src_path("namespace.module", config)
    assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_src_path_is_module():
    src_path = Path("/path/to/module")
    src_path.mkdir()
    (src_path / "__init__.py").write_text("")
    config = Config(src_paths=[src_path])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = None
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=[],
        auto_identify_namespace_packages=True,
        supported_extensions=[".py"]
    )
    name = "module.submodule"
    src_paths = [Path("/path/to/src")]
    prefix = ()
    module_path = Path("/path/to/src/module").resolve()
    assert _is_namespace_package(module_path, config.supported_extensions) is True
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #28
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None, namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch.object(Path, "is_dir", return_value=False), \
         patch.object(Path, "with_suffix", return_value=Path("/path/to/src/module.py")):
        result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch.object(Path, "is_dir", return_value=True):
        result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch.object(Path, "is_dir", return_value=True):
        result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"namespace"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch.object(Path, "is_dir", return_value=True), \
         patch("_is_namespace_package", return_value=True):
        result = _src_path("namespace.module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch.object(Path, "is_dir", return_value=True), \
         patch("_is_namespace_package", return_value=True):
        result = _src_path("namespace.module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_not_in_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch.object(Path, "is_dir", return_value=True), \
         patch("_is_namespace_package", return_value=False):
        result = _src_path("namespace.module", config)
    assert result is None


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["module"],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "module.submodule"
    config = Config(
        namespace_packages=[],
        auto_identify_namespace_packages=False,
        src_paths=[Path("/path/to/src")],
        supported_extensions=[".py"]
    )
    src_paths = None
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is None or result[0] != sections.FIRSTPARTY


# LLM-generated content at query #32
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_module_in_src_paths():
    src_path = Path("/path/to/src")
    module_path = src_path / "module"
    module_path.mkdir()
    (module_path / "__init__.py").touch()
    config = Config(src_paths=[src_path])
    assert _src_path("module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_nested_module_in_src_paths():
    src_path = Path("/path/to/src")
    module_path = src_path / "parent" / "child"
    module_path.mkdir(parents=True)
    (module_path / "__init__.py").touch()
    config = Config(src_paths=[src_path])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_namespace_package():
    src_path = Path("/path/to/src")
    module_path = src_path / "namespace"
    module_path.mkdir()
    (module_path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    config = Config(src_paths=[src_path], namespace_packages=["namespace"])
    assert _src_path("namespace", config) is None

def test__src_path_with_auto_identify_namespace_package():
    src_path = Path("/path/to/src")
    module_path = src_path / "namespace"
    module_path.mkdir()
    (module_path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
    assert _src_path("namespace", config) is None

def test__src_path_with_module_not_in_src_paths():
    src_path = Path("/path/to/src")
    config = Config(src_paths=[src_path])
    assert _src_path("nonexistent", config) is None

def test__src_path_with_root_module_in_src_paths():
    src_path = Path("/path/to/src")
    (src_path / "__init__.py").touch()
    config = Config(src_paths=[src_path])
    assert _src_path(src_path.name, config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


# LLM-generated content at query #33
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_init_file():
    from pathlib import Path
    path = Path("/package/with/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_init_file_with_declare_namespace():
    from pathlib import Path
    path = Path("/package/with/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_init_file_with_extend_path():
    from pathlib import Path
    path = Path("/package/with/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_no_init_file_and_no_source_files():
    from pathlib import Path
    path = Path("/package/without/init")
    path.mkdir(parents=True, exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_no_init_file_but_with_source_files():
    from pathlib import Path
    path = Path("/package/with/source")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_no_init_file_but_with_setup_cfg():
    from pathlib import Path
    path = Path("/package/with/setup")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_no_init_file_but_with_pyproject_toml():
    from pathlib import Path
    path = Path("/package/with/pyproject")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages=["test_namespace"],
        auto_identify_namespace_packages=False,
        src_paths=[Path("/test/src")],
        supported_extensions=[".py"]
    )
    name = "test_namespace.submodule"
    result = _src_path(name, config)
    assert result is not None


# LLM-generated content at query #35
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module.py")

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(str(path.with_suffix(ext_suffix)))
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive("module/__init__.py")

def test_is_module_no_match():
    path = Path("not_a_module")
    assert _is_module(path) == False


# LLM-generated content at query #36
#--------------------------

```python
def test__is_namespace_package_returns_true_when_package_and_no_init_file_and_no_source_files():
    from pathlib import Path
    path = Path("some_namespace_package")
    path.mkdir()
    src_extensions = frozenset([".py", ".pyx", ".pxd", ".pxi"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #37
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path)


# LLM-generated content at query #38
#--------------------------

```python
def test__src_path_is_module_returns_true_when_conditions_are_met():
    src_path = Path("example_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "example_module") is True


# LLM-generated content at query #39
#--------------------------

```python
def test__src_path__returns_none_when_no_src_paths_match():
    assert _src_path("nonexistent", Config(src_paths=[])) is None

def test__src_path__returns_firstparty_when_module_found_in_src_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "module.py"
        module_path.touch()
        config = Config(src_paths=[Path(tmpdir)])
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path__returns_firstparty_when_package_found_in_src_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "package"
        package_path.mkdir()
        (package_path / "__init__.py").touch()
        config = Config(src_paths=[Path(tmpdir)])
        result = _src_path("package", config)
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path__returns_firstparty_when_nested_module_in_namespace_package():
    with tempfile.TemporaryDirectory() as tmpdir:
        namespace_path = Path(tmpdir) / "namespace"
        namespace_path.mkdir()
        (namespace_path / "__init__.py").write_text("__import__('pkgutil').extend_path(__path__, __name__)")
        module_path = namespace_path / "module.py"
        module_path.touch()
        config = Config(src_paths=[Path(tmpdir)], namespace_packages={"namespace"})
        result = _src_path("namespace.module", config)
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path__returns_firstparty_when_src_path_is_module():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "module"
        src_path.mkdir()
        (src_path / "__init__.py").touch()
        config = Config(src_paths=[src_path])
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_16_evaluates_to_true():
    name = "module"
    config = Config(src_paths=[Path("module")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("module")]
    prefix = ()
    assert _src_path(name, config, src_paths, prefix) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_paths[0]}.")


# LLM-generated content at query #41
#--------------------------

```python
def test__src_path_found_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_not_found():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("nonexistent", config) is None

def test__src_path_nested_module():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #42
#--------------------------

```python
def test__is_namespace_package_returns_true_for_valid_namespace_package():
    path = Path("valid_namespace_pkg")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages={"module"},
        src_paths=[Path("src")],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #44
#--------------------------

```python
def test_known_pattern_matches_first_segment():
    config = Config(known_patterns=[(re.compile("foo"), "placement1")], sections=["placement1"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement1", "Matched configured known pattern foo")

def test_known_pattern_matches_middle_segment():
    config = Config(known_patterns=[(re.compile("bar"), "placement2")], sections=["placement2"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement2", "Matched configured known pattern bar")

def test_known_pattern_matches_last_segment():
    config = Config(known_patterns=[(re.compile("baz"), "placement3")], sections=["placement3"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement3", "Matched configured known pattern baz")

def test_known_pattern_no_match():
    config = Config(known_patterns=[(re.compile("qux"), "placement4")], sections=["placement4"])
    result = _known_pattern("foo.bar.baz", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile("foo"), "placement5")], sections=["placement6"])
    result = _known_pattern("foo.bar.baz", config)
    assert result is None

def test_known_pattern_multiple_patterns_first_match_wins():
    config = Config(known_patterns=[(re.compile("foo"), "placement1"), (re.compile("bar"), "placement2")], sections=["placement1", "placement2"])
    result = _known_pattern("foo.bar.baz", config)
    assert result == ("placement1", "Matched configured known pattern foo")

def test_known_pattern_empty_name():
    config = Config(known_patterns=[(re.compile("foo"), "placement1")], sections=["placement1"])
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_single_segment_match():
    config = Config(known_patterns=[(re.compile("foo"), "placement1")], sections=["placement1"])
    result = _known_pattern("foo", config)
    assert result == ("placement1", "Matched configured known pattern foo")

def test_known_pattern_single_segment_no_match():
    config = Config(known_patterns=[(re.compile("bar"), "placement1")], sections=["placement1"])
    result = _known_pattern("foo", config)
    assert result is None


# LLM-generated content at query #45
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_module_not_found():
    config = Config(src_paths=[Path("/tmp")])
    assert _src_path("nonexistent_module", config) is None

def test__src_path_with_module_in_src_path():
    config = Config(src_paths=[Path("/tmp")])
    module_path = Path("/tmp/module.py")
    module_path.touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_package_in_src_path():
    config = Config(src_paths=[Path("/tmp")])
    package_path = Path("/tmp/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    assert _src_path("package", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/tmp")])
    package_path = Path("/tmp/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    (package_path / "nested.py").touch()
    assert _src_path("package.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/tmp")], namespace_packages={"package"})
    package_path = Path("/tmp/package")
    package_path.mkdir()
    assert _src_path("package.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/tmp")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    package_path = Path("/tmp/package")
    package_path.mkdir()
    assert _src_path("package.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/tmp/module")])
    module_path = Path("/tmp/module.py")
    module_path.touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /tmp/module.")


# LLM-generated content at query #46
#--------------------------

```python
def test__is_namespace_package_returns_true_for_valid_namespace():
    path = Path("valid_namespace")
    path.mkdir()
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #47
#--------------------------

```python
def test_src_path_is_module_when_path_matches_module_name():
    src_path = Path("module_name")
    src_path.mkdir(exist_ok=True)
    assert _src_path_is_module(src_path, "module_name") is True
    src_path.rmdir()

def test_src_path_is_not_module_when_path_does_not_match_module_name():
    src_path = Path("different_name")
    src_path.mkdir(exist_ok=True)
    assert _src_path_is_module(src_path, "module_name") is False
    src_path.rmdir()

def test_src_path_is_not_module_when_path_is_not_directory():
    src_path = Path("module_name")
    src_path.touch()
    assert _src_path_is_module(src_path, "module_name") is False
    src_path.unlink()

def test_src_path_is_not_module_when_path_does_not_exist_case_sensitively():
    src_path = Path("module_name")
    src_path.mkdir(exist_ok=True)
    assert _src_path_is_module(src_path, "MODULE_NAME") is False
    src_path.rmdir()


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_evaluates_to_false():
    path = Path("some_path")
    src_extensions = frozenset([".py"])
    filenames = [path / "file.py"]
    assert not filenames


# LLM-generated content at query #49
#--------------------------

```python
def test__is_namespace_package_returns_true_for_namespace_package_without_init():
    from pathlib import Path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "namespace_pkg"
        path.mkdir()

        # Create a Python file to ensure it's a package
        (path / "module.py").write_text("# empty")

        assert _is_namespace_package(path, frozenset([".py"])) is True


# LLM-generated content at query #50
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    from pathlib import Path
    path = Path("non_existent_path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    path = Path("test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_file_exists_with_pkg_resources_declaration():
    from pathlib import Path
    path = Path("test_namespace_pkg_resources")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_file_exists_with_pkgutil_declaration():
    from pathlib import Path
    path = Path("test_namespace_pkgutil")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_no_init_file_but_has_py_files():
    from pathlib import Path
    path = Path("test_namespace_with_py_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# Python module")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_no_init_file_but_has_setup_cfg():
    from pathlib import Path
    path = Path("test_namespace_with_setup_cfg")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_no_init_file_and_no_py_files():
    from pathlib import Path
    path = Path("test_namespace_empty")
    path.mkdir(exist_ok=True)
    assert _is_namespace_package(path, frozenset([".py"]))


# LLM-generated content at query #51
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_but_no_init_file():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_but_no_namespace_declaration():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_with_pkg_resources_declaration():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_with_pkgutil_declaration():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_with_mixed_quotes_declaration():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_with_single_quotes_declaration():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_with_pkgutil_single_quotes_declaration():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_but_has_source_files():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    (path / "module.py").write_text("# Some code")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_but_has_setup_cfg():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    (path / "setup.cfg").write_text("[metadata]")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_and_init_file_but_has_pyproject_toml():
    from pathlib import Path
    path = Path("/existing/package/path")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    (path / "pyproject.toml").write_text("[build-system]")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    path = Path("test_package")
    path.mkdir()
    (path / "module.py").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_false_when_filenames_exist():
    path = Path("test_path")
    src_extensions = frozenset([".py"])
    (path / "test.py").touch()
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #54
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert _is_namespace_package(path, frozenset({".py"})) is False

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("print('hello')")
    assert _is_namespace_package(path, frozenset({".py"})) is False

def test__is_namespace_package_returns_true_when_init_file_contains_pkg_resources_declaration():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
    assert _is_namespace_package(path, frozenset({".py"})) is True

def test__is_namespace_package_returns_true_when_init_file_contains_pkgutil_declaration():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    assert _is_namespace_package(path, frozenset({".py"})) is True

def test__is_namespace_package_returns_false_when_package_contains_py_files():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("print('hello')")
    assert _is_namespace_package(path, frozenset({".py"})) is False

def test__is_namespace_package_returns_false_when_package_contains_setup_cfg():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname=test")
    assert _is_namespace_package(path, frozenset({".py"})) is False

def test__is_namespace_package_returns_false_when_package_contains_pyproject_toml():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    assert _is_namespace_package(path, frozenset({".py"})) is False

def test__is_namespace_package_returns_true_when_package_is_empty():
    from pathlib import Path
    path = Path("/existing/package")
    path.mkdir(parents=True, exist_ok=True)
    assert _is_namespace_package(path, frozenset({".py"})) is True


# LLM-generated content at query #55
#--------------------------

```python
def test_is_namespace_package_without_init_file():
    path = Path("test_namespace")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# test module")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is False


# LLM-generated content at query #56
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("test_package")
    src_extensions = frozenset([".py"])
    init_file = path / "__init__.py"
    init_file.write_bytes(b"__import__('pkg_resources').declare_namespace(__name__)")
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #57
#--------------------------

```python
def test__is_namespace_package_returns_true_when_init_file_does_not_exist_and_no_source_files():
    path = Path("some_namespace_package")
    path.mkdir()
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test_namespace"},
        auto_identify_namespace_packages=False,
        src_paths=[Path("/test/src")],
        supported_extensions=[".py"]
    )
    name = "test_namespace.submodule"
    src_paths = [Path("/test/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_at_line_16():
    name = "module"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #60
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    path = Path("/valid/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("# Regular package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_contains_pkg_resources_declare_namespace():
    from pathlib import Path
    path = Path("/valid/namespace/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_contains_pkgutil_extend_path():
    from pathlib import Path
    path = Path("/valid/namespace/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_package_contains_py_files():
    from pathlib import Path
    path = Path("/valid/package/with/files")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("# Some code")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_package_contains_setup_cfg():
    from pathlib import Path
    path = Path("/valid/package/with/setup")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname=test")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_package_contains_pyproject_toml():
    from pathlib import Path
    path = Path("/valid/package/with/pyproject")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = []")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #61
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = [("test.*", "section1")]
    config.sections = ["section2"]
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=True, supported_extensions=[".py"])
    src_paths = [Path("src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #63
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.sections = {"section1"}
    config.known_patterns = [("pattern1", "section2")]
    name = "test.module"
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #64
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    path.with_suffix(".py").touch()
    assert _is_module(path) is True


# LLM-generated content at query #65
#--------------------------

```python
def test__is_namespace_package_returns_false_for_non_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_init():
    from pathlib import Path
    path = Path("/existing/package/with/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("__path__ = []")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_non_namespace_init():
    from pathlib import Path
    path = Path("/existing/package/with/non_namespace/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("# regular init file")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_package_with_pkg_resources_namespace():
    from pathlib import Path
    path = Path("/existing/package/with/pkg_resources/namespace/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_package_with_pkgutil_namespace():
    from pathlib import Path
    path = Path("/existing/package/with/pkgutil/namespace/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_py_files():
    from pathlib import Path
    path = Path("/existing/package/with/py/files")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("# some code")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_setup_cfg():
    from pathlib import Path
    path = Path("/existing/package/with/setup.cfg")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_for_package_with_pyproject_toml():
    from pathlib import Path
    path = Path("/existing/package/with/pyproject.toml")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_empty_package():
    from pathlib import Path
    path = Path("/existing/empty/package")
    path.mkdir(parents=True, exist_ok=True)
    assert _is_namespace_package(path, frozenset([".py"]))


