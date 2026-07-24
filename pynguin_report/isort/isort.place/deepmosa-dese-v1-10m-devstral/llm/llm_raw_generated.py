####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/some/path")])
    result = _src_path("", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    module_path = Path("/some/path/module.py")
    module_path.touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    package_path = Path("/some/path/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/some/path")])
    module_path = Path("/some/path/parent")
    module_path.mkdir()
    (module_path / "__init__.py").touch()
    nested_module_path = module_path / "child.py"
    nested_module_path.touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/some/path")], namespace_packages={"parent"})
    namespace_path = Path("/some/path/parent")
    namespace_path.mkdir()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/some/path")], auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    namespace_path = Path("/some/path/parent")
    namespace_path.mkdir()
    (namespace_path / "child.py").touch()
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/some/path/module")])
    module_path = Path("/some/path/module")
    module_path.mkdir()
    (module_path / "__init__.py").touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path/module.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/some/path")])
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #2
#--------------------------

```python
def test_known_pattern_matches_first_pattern():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_matches_second_pattern():
    config = Config(known_patterns=[(re.compile("foo.*"), "placement1"), (re.compile("bar.*"), "placement2")], sections=["placement1", "placement2"])
    result = _known_pattern("bar.baz", config)
    assert result == ("placement2", "Matched configured known pattern bar.*")

def test_known_pattern_no_match():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("other.module", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile("test.*"), "invalid_placement")], sections=["placement1"])
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_matches_submodule():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("parent.test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_empty_name():
    config = Config(known_patterns=[(re.compile("test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("", config)
    assert result is None


# LLM-generated content at query #3
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

def test_is_module_non_existent():
    path = Path("non_existent_module")
    assert _is_module(path) == False


# LLM-generated content at query #4
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/non/existent/path")])
    assert _src_path("module", config) is None

def test__src_path_with_existing_module_file():
    config = Config(src_paths=[Path("/existing/path")])
    module_path = Path("/existing/path/module.py")
    module_path.touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_existing_package():
    config = Config(src_paths=[Path("/existing/path")])
    package_path = Path("/existing/path/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    assert _src_path("package", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/existing/path")])
    package_path = Path("/existing/path/package")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    (package_path / "nested.py").touch()
    assert _src_path("package.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/existing/path")], namespace_packages={"package"})
    package_path = Path("/existing/path/package")
    package_path.mkdir()
    assert _src_path("package.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/existing/path")], auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    package_path = Path("/existing/path/package")
    package_path.mkdir()
    (package_path / "module.py").touch()
    assert _src_path("package.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/existing/path/module")])
    module_path = Path("/existing/path/module")
    module_path.mkdir()
    (module_path / "__init__.py").touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path/module.")

def test__src_path_with_non_matching_module_name():
    config = Config(src_paths=[Path("/existing/path")])
    module_path = Path("/existing/path/other_module.py")
    module_path.touch()
    assert _src_path("module", config) is None


# LLM-generated content at query #5
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("test_module")
    assert _is_module(path) == True


# LLM-generated content at query #6
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False, supported_extensions=[".py"])
    src_paths = [Path("/path/to/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_evaluates_to_false():
    name = "unknown.module"
    config = Config(known_patterns=[(re.compile(r"known.*"), "placement")], sections=[])
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #9
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #10
#--------------------------

```python
def test_src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == True

def test_src_path_is_module_returns_false_for_non_matching_name():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == False

def test_src_path_is_module_returns_false_for_non_directory():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: False
    assert _src_path_is_module(src_path, "valid_module") == False

def test_src_path_is_module_returns_false_for_case_sensitive_mismatch():
    src_path = Path("Valid_Module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == False


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
    path = Path("package")
    assert _is_module(path) == exists_case_sensitive("package/__init__.py")

def test_is_module_with_no_matching_files():
    path = Path("nonexistent")
    assert _is_module(path) == False


# LLM-generated content at query #12
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

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"})
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True)
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #13
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_containing_init():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_missing_init():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_containing_py_file():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_containing_setup_cfg():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_containing_pyproject_toml():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_namespace_package_pkg_resources():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_namespace_package_pkgutil():
    path = Path("/existing/package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_26():
    name = "module.submodule"
    config = Config(src_paths=[Path("src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    result = _src_path(name, config)
    assert result is not None


# LLM-generated content at query #15
#--------------------------

```python
def test__src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == True

def test__src_path_is_module_returns_false_for_non_matching_name():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == False

def test__src_path_is_module_returns_false_for_non_directory():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: False
    assert _src_path_is_module(src_path, "valid_module") == False

def test__src_path_is_module_returns_false_for_case_insensitive_match():
    src_path = Path("Valid_Module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == False


# LLM-generated content at query #16
#--------------------------

```python
def test_forced_separate_matching_pattern():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("testfile", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matching_pattern_with_dot_prefix():
    config = Config(forced_separate=["test*"])
    result = _forced_separate(".testfile", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_non_matching_pattern():
    config = Config(forced_separate=["other*"])
    result = _forced_separate("testfile", config)
    assert result is None

def test_forced_separate_pattern_without_wildcard():
    config = Config(forced_separate=["test"])
    result = _forced_separate("testfile", config)
    assert result == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("testfile", config)
    assert result is None


# LLM-generated content at query #17
#--------------------------

```python
def test__src_path_with_empty_name():
    assert _src_path("", Config(src_paths=[])) is None

def test__src_path_with_nonexistent_module():
    assert _src_path("nonexistent_module", Config(src_paths=[Path("/tmp")])) is None

def test__src_path_with_module_in_src_paths():
    src_path = Path("/tmp/project")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "module.py").touch()
    assert _src_path("module", Config(src_paths=[src_path])) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_nested_module():
    src_path = Path("/tmp/project")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "parent").mkdir()
    (src_path / "parent" / "child.py").touch()
    assert _src_path("parent.child", Config(src_paths=[src_path])) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_namespace_package():
    src_path = Path("/tmp/project")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "namespace").mkdir()
    (src_path / "namespace" / "__init__.py").write_text('__import__("pkgutil").extend_path(__path__, __name__)')
    (src_path / "namespace" / "module.py").touch()
    config = Config(src_paths=[src_path], auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("namespace.module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_explicit_namespace_package():
    src_path = Path("/tmp/project")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "namespace").mkdir()
    (src_path / "namespace" / "module.py").touch()
    config = Config(src_paths=[src_path], namespace_packages={"namespace"})
    assert _src_path("namespace.module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_root_module_in_src_path():
    src_path = Path("/tmp/module")
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "__init__.py").touch()
    assert _src_path("module", Config(src_paths=[src_path])) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


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

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"})
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_module_not_found():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #19
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = [("test.*", "section1")]
    config.sections = ["section2"]
    name = "test.module"
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test__src_path_is_module_returns_true_when_conditions_met():
    src_path = Path("module_name")
    src_path.is_dir.return_value = True
    exists_case_sensitive.return_value = True
    assert _src_path_is_module(src_path, "module_name") is True


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages=["module"],
        auto_identify_namespace_packages=False,
        src_paths=[Path("/path/to/src")],
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #22
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
    path = Path("nonexistent_module")
    assert _is_module(path) == False


# LLM-generated content at query #23
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #24
#--------------------------

```python
def test__src_path_with_exact_match_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == _src_path("child", config, (Path("/project/src/parent"),), ("parent",))

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("namespace.child", config) == _src_path("child", config, (Path("/project/src/namespace"),), ("namespace",))

def test__src_path_with_module_in_root_src_path():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("src", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #25
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_package():
    assert _is_namespace_package(Path("not_a_package"), frozenset([".py"])) is False


# LLM-generated content at query #26
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True

def test_is_module_returns_true_for_extension_suffix_file():
    path = Path("module.so")
    assert _is_module(path) is True

def test_is_module_returns_true_for_init_file():
    path = Path("package/__init__.py")
    assert _is_module(path) is True


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=[], auto_identify_namespace_packages=True, supported_extensions=[".py"])
    src_paths = [Path("/path/to/src")]
    prefix = ()
    module_path = (Path("/path/to/src") / "module").resolve()
    assert _is_namespace_package(module_path, config.supported_extensions) == True
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #28
#--------------------------

```python
def test_src_path_is_module_returns_true():
    src_path = Path("module_name")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "module_name") is True


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test"},
        auto_identify_namespace_packages=False,
        src_paths=[Path("/path/to/src")],
        supported_extensions=[".py"]
    )
    name = "test.submodule"
    src_paths = [Path("/path/to/src")]
    prefix = ()
    namespace = "test"
    module_path = (Path("/path/to/src") / "test").resolve()

    assert namespace in config.namespace_packages or (
        config.auto_identify_namespace_packages
        and _is_namespace_package(module_path, config.supported_extensions)
    )


# LLM-generated content at query #30
#--------------------------

```python
def test__src_path_is_module_returns_true_when_conditions_met():
    src_path = Path("test_module")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "test_module") is True


# LLM-generated content at query #31
#--------------------------

```python
def test__src_path_with_nested_module_in_namespace_package():
    config = Config(
        src_paths=[Path("/project")],
        namespace_packages={"project"},
        supported_extensions=frozenset({"py"}),
        auto_identify_namespace_packages=False,
    )
    result = _src_path("project.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_with_nested_module_in_auto_identified_namespace_package():
    config = Config(
        src_paths=[Path("/project")],
        namespace_packages=set(),
        supported_extensions=frozenset({"py"}),
        auto_identify_namespace_packages=True,
    )
    result = _src_path("project.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_with_module_in_src_paths():
    config = Config(
        src_paths=[Path("/project")],
        namespace_packages=set(),
        supported_extensions=frozenset({"py"}),
        auto_identify_namespace_packages=False,
    )
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_with_package_in_src_paths():
    config = Config(
        src_paths=[Path("/project")],
        namespace_packages=set(),
        supported_extensions=frozenset({"py"}),
        auto_identify_namespace_packages=False,
    )
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_with_src_path_is_module():
    config = Config(
        src_paths=[Path("/project")],
        namespace_packages=set(),
        supported_extensions=frozenset({"py"}),
        auto_identify_namespace_packages=False,
    )
    result = _src_path("project", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project.")

def test__src_path_with_no_match():
    config = Config(
        src_paths=[Path("/project")],
        namespace_packages=set(),
        supported_extensions=frozenset({"py"}),
        auto_identify_namespace_packages=False,
    )
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #32
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    from pathlib import Path
    assert not _is_namespace_package(Path("/non/existent/path"), frozenset({".py"}))

def test__is_namespace_package_with_empty_directory():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_with_init_file():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "__init__.py").write_text("")
        assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_with_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "__init__.py").write_text(
            "__import__('pkg_resources').declare_namespace(__name__)"
        )
        assert _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_with_pkgutil_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "__init__.py").write_text(
            "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
        )
        assert _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_with_py_file():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "module.py").write_text("")
        assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_with_setup_cfg():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "setup.cfg").write_text("")
        assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_with_pyproject_toml():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "pyproject.toml").write_text("")
        assert not _is_namespace_package(path, frozenset({".py"}))


# LLM-generated content at query #33
#--------------------------

```python
def test__is_namespace_package_returns_true_when_predicate_at_line_4_is_true():
    path = Path("some_path")
    src_extensions = frozenset([".py"])
    assert _is_package(path) is True


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #35
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/non/existent/path")])
    assert _src_path("module", config) is None

def test__src_path_with_existing_module_file():
    config = Config(src_paths=[Path("/existing/path")])
    module_path = Path("/existing/path/module.py")
    module_path.touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_existing_package():
    config = Config(src_paths=[Path("/existing/path")])
    package_path = Path("/existing/path/module")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/existing/path")])
    package_path = Path("/existing/path/module")
    package_path.mkdir()
    (package_path / "__init__.py").touch()
    nested_module_path = package_path / "nested"
    nested_module_path.mkdir()
    (nested_module_path / "__init__.py").touch()
    assert _src_path("module.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/existing/path")], namespace_packages={"module"})
    package_path = Path("/existing/path/module")
    package_path.mkdir()
    assert _src_path("module.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/existing/path")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    package_path = Path("/existing/path/module")
    package_path.mkdir()
    (package_path / "nested.py").touch()
    assert _src_path("module.nested", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path.")

def test__src_path_with_src_path_as_module():
    config = Config(src_paths=[Path("/existing/path/module")])
    module_path = Path("/existing/path/module")
    module_path.mkdir()
    (module_path / "__init__.py").touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existing/path/module.")


# LLM-generated content at query #36
#--------------------------

```python
def test__src_path_with_empty_name():
    result = _src_path("", config)
    assert result is None

def test__src_path_with_non_existent_module():
    result = _src_path("non_existent_module", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    result = _src_path("existing_module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: <src_path>.")

def test__src_path_with_nested_module_in_namespace_package():
    result = _src_path("namespace_package.nested_module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: <src_path>.")

def test__src_path_with_module_in_prefix():
    result = _src_path("module", config, prefix=("prefix",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: <src_path>.")

def test__src_path_with_custom_src_paths():
    custom_src_paths = [Path("/custom/path")]
    result = _src_path("module", config, src_paths=custom_src_paths)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")

def test__src_path_with_namespace_package_configured():
    config.namespace_packages = ["configured_namespace"]
    result = _src_path("configured_namespace.module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: <src_path>.")

def test__src_path_with_auto_identified_namespace_package():
    config.auto_identify_namespace_packages = True
    result = _src_path("auto_identified_namespace.module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: <src_path>.")


# LLM-generated content at query #37
#--------------------------

```python
def test_forced_separate_match_with_glob():
    config = Config(forced_separate=["test_*"])
    result = _forced_separate("test_file.py", config)
    assert result == ("test_*", "Matched forced_separate (test_*) config value.")

def test_forced_separate_match_without_glob():
    config = Config(forced_separate=["test"])
    result = _forced_separate("test_file.py", config)
    assert result == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_match_hidden_file():
    config = Config(forced_separate=[".hidden"])
    result = _forced_separate(".hidden_file", config)
    assert result == (".hidden", "Matched forced_separate (.hidden) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["other_*"])
    result = _forced_separate("test_file.py", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("test_file.py", config)
    assert result is None


# LLM-generated content at query #38
#--------------------------

```python
def test__is_namespace_package_with_valid_init_file():
    path = Path("valid_namespace_pkg")
    path.mkdir()
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')

    assert _is_namespace_package(path, frozenset({".py"})) is True


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_true():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py", ".pyx", ".pxd"])
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #40
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


# LLM-generated content at query #41
#--------------------------

```python
def test__src_path_with_empty_name():
    result = _src_path("", Config(src_paths=[]))
    assert result is None

def test__src_path_with_non_existent_module():
    result = _src_path("nonexistent", Config(src_paths=[Path("/tmp")]))
    assert result is None

def test__src_path_with_existing_module_in_src_paths():
    with TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "mymodule"
        module_path.mkdir()
        (module_path / "__init__.py").write_text("")
        result = _src_path("mymodule", Config(src_paths=[Path(tmpdir)]))
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path_with_nested_module():
    with TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "parent" / "child"
        module_path.mkdir(parents=True)
        (module_path / "__init__.py").write_text("")
        result = _src_path("parent.child", Config(src_paths=[Path(tmpdir)]))
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path_with_namespace_package():
    with TemporaryDirectory() as tmpdir:
        ns_path = Path(tmpdir) / "namespace"
        ns_path.mkdir()
        (ns_path / "module.py").write_text("")
        config = Config(src_paths=[Path(tmpdir)], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
        result = _src_path("namespace.module", config)
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path_with_explicit_namespace_package():
    with TemporaryDirectory() as tmpdir:
        ns_path = Path(tmpdir) / "namespace"
        ns_path.mkdir()
        (ns_path / "module.py").write_text("")
        config = Config(src_paths=[Path(tmpdir)], namespace_packages=["namespace"])
        result = _src_path("namespace.module", config)
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {Path(tmpdir)}.")

def test__src_path_with_src_path_as_module():
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "mymodule"
        src_path.mkdir()
        (src_path / "__init__.py").write_text("")
        result = _src_path("mymodule", Config(src_paths=[src_path]))
        assert result == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


# LLM-generated content at query #42
#--------------------------

```python
def test_known_pattern_predicate_false():
    name = "test.module"
    config = Config()
    config.known_patterns = [("pattern", "placement")]
    config.sections = []
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #43
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_empty_directory():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file_but_no_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        init_file = path / "__init__.py"
        init_file.write_text("# Regular __init__.py")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pkg_resources_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        init_file = path / "__init__.py"
        init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pkgutil_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        init_file = path / "__init__.py"
        init_file.write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_py_file_present():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        py_file = path / "module.py"
        py_file.write_text("# Some Python code")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_setup_cfg_present():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        setup_cfg = path / "setup.cfg"
        setup_cfg.write_text("[metadata]\nname = test")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pyproject_toml_present():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        pyproject_toml = path / "pyproject.toml"
        pyproject_toml.write_text("[build-system]\nrequires = ['setuptools']")
        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_no_files_and_no_init():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_true():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py"])
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #45
#--------------------------

```python
def test_src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == True

def test_src_path_is_module_returns_false_for_non_matching_name():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == False

def test_src_path_is_module_returns_false_for_non_directory():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: False
    assert _src_path_is_module(src_path, "valid_module") == False

def test_src_path_is_module_returns_false_for_case_sensitive_mismatch():
    src_path = Path("Valid_Module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive(str(path.with_suffix(".py")))

def test_is_module_with_extension_suffix():
    path = Path("module")
    assert _is_module(path) == any(
        exists_case_sensitive(str(path.with_suffix(ext_suffix)))
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("module")
    assert _is_module(path) == exists_case_sensitive(str(path / "__init__.py"))

def test_is_module_without_any_file():
    path = Path("module")
    assert _is_module(path) == False


# LLM-generated content at query #2
#--------------------------

```python
def test__src_path_with_empty_name():
    assert _src_path("", Config(src_paths=[Path(".")])) is None

def test__src_path_with_nonexistent_module():
    assert _src_path("nonexistent_module", Config(src_paths=[Path(".")])) is None

def test__src_path_with_root_module_in_src_paths():
    assert _src_path("root_module", Config(src_paths=[Path("root_module")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: root_module.")

def test__src_path_with_nested_module_in_src_paths():
    assert _src_path("root_module.nested_module", Config(src_paths=[Path("root_module")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: root_module.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("namespace_package")], namespace_packages=["namespace_package"])
    assert _src_path("namespace_package.nested_module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: namespace_package.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("auto_namespace_package")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    assert _src_path("auto_namespace_package.nested_module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: auto_namespace_package.")

def test__src_path_with_module_in_src_paths():
    assert _src_path("module_name", Config(src_paths=[Path("module_name.py")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: module_name.py.")

def test__src_path_with_package_in_src_paths():
    assert _src_path("package_name", Config(src_paths=[Path("package_name")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: package_name.")

def test__src_path_with_src_path_is_module():
    assert _src_path("module_name", Config(src_paths=[Path("module_name")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: module_name.")


# LLM-generated content at query #3
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    assert _is_module(path) is True


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    name = "test_module"
    config = Config(src_paths=[Path("/some/path")])
    src_paths = [Path("/some/path")]
    prefix = ("",)
    result = _src_path(name, config, src_paths, prefix)
    assert result is None


# LLM-generated content at query #5
#--------------------------

```python
def test_known_pattern_matches_first_pattern():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_matches_deepest_module():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1"), (re.compile(r"test.module.*"), "placement2")],
        sections={"placement1": {}, "placement2": {}}
    )
    result = _known_pattern("test.module.sub", config)
    assert result == ("placement2", "Matched configured known pattern test.module.*")

def test_known_pattern_no_match():
    config = Config(
        known_patterns=[(re.compile(r"other.*"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement2": {}}
    )
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_empty_name():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("", config)
    assert result is None


# LLM-generated content at query #6
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_empty_directory():
    path = Path("/tmp/empty_dir")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_non_namespace_init():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_py_files():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_setup_cfg():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname=test")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pyproject_toml():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pkg_resources_declare():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pkgutil_extend_path():
    path = Path("/tmp/pkg")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #7
#--------------------------

```python
def test_is_namespace_package_with_valid_files():
    path = Path("test_package")
    src_extensions = frozenset([".py", ".pyx", ".pxd"])
    (path / "module.py").write_text("")
    (path / "setup.cfg").write_text("")
    assert _is_namespace_package(path, src_extensions) == True


# LLM-generated content at query #8
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("print('hello')")

        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')

        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')

        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_has_source_files():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        (path / "module.py").write_text("print('hello')")

        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_has_setup_cfg():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        (path / "setup.cfg").write_text("[metadata]\nname = test")

        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_has_pyproject_toml():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")

        src_extensions = frozenset([".py"])
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()

        src_extensions = frozenset([".py"])
        assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #9
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("/some/path/module")
    assert _is_module(path) == exists_case_sensitive("/some/path/module.py")

def test_is_module_with_extension_suffix():
    path = Path("/some/path/module")
    assert _is_module(path) == any(
        exists_case_sensitive(f"/some/path/module{ext_suffix}")
        for ext_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

def test_is_module_with_init_py():
    path = Path("/some/path/module")
    assert _is_module(path) == exists_case_sensitive("/some/path/module/__init__.py")

def test_is_module_no_match():
    path = Path("/some/path/not_a_module")
    assert _is_module(path) == False


# LLM-generated content at query #10
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("test_module")
    (path / "test_module.py").write_text("")
    assert _is_module(path)

def test_is_module_with_extension_suffix():
    path = Path("test_module")
    (path / "test_module.so").write_text("")
    assert _is_module(path)

def test_is_module_with_init_file():
    path = Path("test_module")
    (path / "__init__.py").write_text("")
    assert _is_module(path)

def test_is_not_module():
    path = Path("test_module")
    assert not _is_module(path)


# LLM-generated content at query #11
#--------------------------

```python
def test__is_namespace_package_returns_true_when_predicate_at_line_4_is_true():
    assert _is_namespace_package(Path("valid_namespace_package"), frozenset([".py"])) is True


# LLM-generated content at query #12
#--------------------------

```python
def test_forced_separate_no_match():
    config = Config(forced_separate=[])
    assert _forced_separate("test.txt", config) is None

def test_forced_separate_exact_match():
    config = Config(forced_separate=["test"])
    assert _forced_separate("test.txt", config) == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_glob_match():
    config = Config(forced_separate=["test*"])
    assert _forced_separate("test123.txt", config) == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_hidden_file_match():
    config = Config(forced_separate=[".hidden"])
    assert _forced_separate(".hiddenfile", config) == (".hidden", "Matched forced_separate (.hidden) config value.")

def test_forced_separate_multiple_patterns():
    config = Config(forced_separate=["test", "data"])
    assert _forced_separate("data.csv", config) == ("data", "Matched forced_separate (data) config value.")


# LLM-generated content at query #13
#--------------------------

```python
def test__is_namespace_package_predicate_true():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #14
#--------------------------

```python
def test__is_namespace_package_without_init_file():
    path = Path("/some/path")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #15
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_false_when_has_init_without_namespace_declaration():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_true_when_has_init_with_pkg_resources_declaration():
    from pathlib import Path
    path = Path("/tmp/test_namespace_pkg_resources")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    assert _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_true_when_has_init_with_pkgutil_declaration():
    from pathlib import Path
    path = Path("/tmp/test_namespace_pkgutil")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    assert _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_false_when_has_py_files():
    from pathlib import Path
    path = Path("/tmp/test_package_with_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# Some code")
    assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_false_when_has_setup_cfg():
    from pathlib import Path
    path = Path("/tmp/test_package_with_setup")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_false_when_has_pyproject_toml():
    from pathlib import Path
    path = Path("/tmp/test_package_with_pyproject")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = []")
    assert not _is_namespace_package(path, frozenset({".py"}))

def test__is_namespace_package_returns_true_when_no_init_and_no_files():
    from pathlib import Path
    path = Path("/tmp/test_namespace_empty")
    path.mkdir(exist_ok=True)
    assert _is_namespace_package(path, frozenset({".py"}))


# LLM-generated content at query #16
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

    assert result is not None


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_true():
    path = Path("some_non_existent_path")
    src_extensions = frozenset([".py"])
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_true():
    path = Path("test_namespace_package")
    path.mkdir()
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_true():
    path = Path("test_package")
    path.mkdir()
    (path / "module.py").write_text("print('test')")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is False


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "module.submodule"
    config = Config(
        namespace_packages=[],
        auto_identify_namespace_packages=False,
        src_paths=[Path("/some/path")],
        supported_extensions=[".py"]
    )
    src_paths = [Path("/some/path")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is None


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages={"module"},
        src_paths=[Path("src")],
        auto_identify_namespace_packages=False
    )
    result = _src_path(name, config)
    assert result is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_is_namespace_package_without_init_file():
    path = Path("some_package")
    path.mkdir()
    (path / "module.py").write_text("")
    assert _is_namespace_package(path, frozenset(["py"])) is True


# LLM-generated content at query #25
#--------------------------

```python
def test__is_namespace_package_returns_true_for_valid_namespace_package():
    path = Path("valid_namespace_package")
    path.mkdir()
    (path / "module.py").write_text("print('hello')")
    assert _is_namespace_package(path, frozenset([".py"])) is True


# LLM-generated content at query #26
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

def test_is_module_non_existent():
    path = Path("non_existent_module")
    assert _is_module(path) == False


# LLM-generated content at query #27
#--------------------------

```python
def test_is_namespace_package_with_files():
    path = Path("test_package")
    path.mkdir()
    (path / "module.py").write_text("# test")
    assert _is_namespace_package(path, frozenset({".py"})) == False


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages=[],
        auto_identify_namespace_packages=True,
        src_paths=[Path("/path/to/src")],
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    module_path = (Path("/path/to/src/module").resolve())
    assert _is_namespace_package(module_path, config.supported_extensions) == True
    result = _src_path(name, config, src_paths, prefix)
    assert result == ("Found in one of the configured src_paths: /path/to/src.",)


# LLM-generated content at query #29
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module.py")
    assert _is_module(path) is True

def test_is_module_with_extension_suffix():
    path = Path("module.so")
    assert _is_module(path) is True

def test_is_module_with_init_file():
    path = Path("package")
    assert _is_module(path) is True

def test_is_module_without_any_file():
    path = Path("nonexistent")
    assert _is_module(path) is False


# LLM-generated content at query #30
#--------------------------

```python
def test_forced_separate_predicate_evaluates_to_true():
    config = Config()
    config.forced_separate = ["test"]
    name = "test_file"
    result = _forced_separate(name, config)
    assert result is not None


# LLM-generated content at query #31
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


# LLM-generated content at query #32
#--------------------------

```python
def test_known_pattern_predicate_false():
    name = "unknown.module"
    config = Config(known_patterns=[], sections=[])
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #33
#--------------------------

```python
def test_src_path_is_module_with_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module")

def test_src_path_is_module_with_invalid_module_name():
    src_path = Path("invalid_module")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "different_name")

def test_src_path_is_module_with_non_directory_path():
    src_path = Path("not_a_directory")
    src_path.is_dir = lambda: False
    assert not _src_path_is_module(src_path, "not_a_directory")

def test_src_path_is_module_with_case_sensitive_mismatch():
    src_path = Path("CaseSensitive")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "casesensitive")


# LLM-generated content at query #34
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config(known_patterns=[], sections=[])
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #35
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

def test_is_module_non_existent():
    path = Path("nonexistent")
    assert _is_module(path) == False


# LLM-generated content at query #36
#--------------------------

```python
def test_known_pattern_matches_first_pattern():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_matches_second_pattern():
    config = Config(
        known_patterns=[
            (re.compile(r"test.*"), "placement1"),
            (re.compile(r"module.*"), "placement2")
        ],
        sections={"placement1": {}, "placement2": {}}
    )
    result = _known_pattern("module.test", config)
    assert result == ("placement2", "Matched configured known pattern module.*")

def test_known_pattern_no_match():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("other.module", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement2": {}}
    )
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_empty_name():
    config = Config(
        known_patterns=[(re.compile(r"test.*"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_single_part_name():
    config = Config(
        known_patterns=[(re.compile(r"test"), "placement1")],
        sections={"placement1": {}}
    )
    result = _known_pattern("test", config)
    assert result == ("placement1", "Matched configured known pattern test")


# LLM-generated content at query #37
#--------------------------

```python
def test__is_namespace_package_without_init_file():
    path = Path("test_namespace")
    path.mkdir()
    (path / "module1.py").write_text("")
    src_extensions = frozenset({".py"})
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #38
#--------------------------

```python
def test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive():
    src_path = Path("/some/path/module_name")
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) == True

def test_src_path_is_not_module_when_name_does_not_match():
    src_path = Path("/some/path/different_name")
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) == False

def test_src_path_is_not_module_when_not_directory():
    src_path = Path("/some/path/module_name.py")
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) == False

def test_src_path_is_not_module_when_does_not_exist_case_sensitive():
    src_path = Path("/some/path/module_name")
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) == False


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_16_evaluates_to_true():
    name = "module"
    config = Config(src_paths=[Path("src")])
    src_paths = [Path("src/module")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #40
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #41
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
    src_paths = None
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #42
#--------------------------

```python
def test__is_namespace_package_returns_true_when_predicate_at_line_4_is_true():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #44
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

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"})
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

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #45
#--------------------------

```python
def test_forced_separate_predicate_evaluates_to_true():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("test_file", config)
    assert result is not None


# LLM-generated content at query #46
#--------------------------

```python
def test__src_path_is_module_returns_true_when_conditions_are_met():
    src_path = Path("module_name")
    src_path.is_dir.return_value = True
    exists_case_sensitive.return_value = True
    assert _src_path_is_module(src_path, "module_name") is True


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #48
#--------------------------

```python
def test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is True

def test_src_path_is_not_module_when_name_does_not_match():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is False

def test_src_path_is_not_module_when_not_directory():
    src_path = Path("module_name")
    src_path.is_dir = lambda: False
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is False

def test_src_path_is_not_module_when_case_sensitive_exists_fails():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is False


# LLM-generated content at query #49
#--------------------------

```python
from pathlib import Path

def test_src_path_is_module_when_module_name_matches_and_path_is_directory_and_exists_case_sensitive():
    src_path = Path("/path/to/module")
    module_name = "module"
    assert _src_path_is_module(src_path, module_name) is True

def test_src_path_is_not_module_when_module_name_does_not_match():
    src_path = Path("/path/to/module")
    module_name = "other_module"
    assert _src_path_is_module(src_path, module_name) is False

def test_src_path_is_not_module_when_path_is_not_directory():
    src_path = Path("/path/to/file.py")
    module_name = "file.py"
    assert _src_path_is_module(src_path, module_name) is False

def test_src_path_is_not_module_when_path_does_not_exist_case_sensitive():
    src_path = Path("/path/to/nonexistent")
    module_name = "nonexistent"
    assert _src_path_is_module(src_path, module_name) is False


# LLM-generated content at query #50
#--------------------------

```python
def test_known_pattern_predicate_false():
    name = "test.module"
    config = Config()
    config.known_patterns = [("pattern1", "placement1"), ("pattern2", "placement2")]
    config.sections = ["section1", "section2"]
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("example_namespace_package")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset({".py"})
    assert _is_namespace_package(path, src_extensions) == True


# LLM-generated content at query #52
#--------------------------

```python
def test__src_path_is_module_returns_true():
    src_path = Path("test_module")
    src_path.mkdir()
    module_name = "test_module"
    assert _src_path_is_module(src_path, module_name) is True


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "module.submodule"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is None


# LLM-generated content at query #54
#--------------------------

```python
def test_forced_separate_predicate_evaluates_to_true():
    config = Config(forced_separate=["test"])
    name = "test"
    result = _forced_separate(name, config)
    assert result is not None


# LLM-generated content at query #55
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("", config) is None

def test__src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("non_existent_module", config) is None

def test__src_path_with_existing_module_file():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("existing_module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_existing_package():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("existing_package", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/some/path")], namespace_packages={"namespace_package"})
    assert _src_path("namespace_package.nested_module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/some/path")], auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    assert _src_path("auto_namespace.nested_module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("module_name", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")


# LLM-generated content at query #56
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
    src_paths = [Path("/test/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config()
    config.namespace_packages = []
    config.auto_identify_namespace_packages = False
    name = "module.submodule"
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is None


# LLM-generated content at query #58
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
    src_paths = [Path("/path/to/src")]
    prefix = ()
    root_module_name, *nested_module = name.split(".", 1)
    new_prefix = (*prefix, root_module_name)
    namespace = ".".join(new_prefix)
    module_path = (src_paths[0] / root_module_name).resolve()
    assert not (namespace in config.namespace_packages or (config.auto_identify_namespace_packages and _is_namespace_package(module_path, config.supported_extensions)))


# LLM-generated content at query #59
#--------------------------

```python
def test_known_pattern_exact_match():
    config = Config()
    config.known_patterns = [("test.*", "placement1")]
    config.sections = ["placement1"]
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_partial_match():
    config = Config()
    config.known_patterns = [("test.*", "placement1")]
    config.sections = ["placement1"]
    result = _known_pattern("test.module.submodule", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_no_match():
    config = Config()
    config.known_patterns = [("other.*", "placement1")]
    config.sections = ["placement1"]
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_multiple_patterns():
    config = Config()
    config.known_patterns = [("test.*", "placement1"), ("other.*", "placement2")]
    config.sections = ["placement1", "placement2"]
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_placement_not_in_sections():
    config = Config()
    config.known_patterns = [("test.*", "placement1")]
    config.sections = ["placement2"]
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_empty_name():
    config = Config()
    config.known_patterns = [("test.*", "placement1")]
    config.sections = ["placement1"]
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_single_part_name():
    config = Config()
    config.known_patterns = [("test", "placement1")]
    config.sections = ["placement1"]
    result = _known_pattern("test", config)
    assert result == ("placement1", "Matched configured known pattern test")

def test_known_pattern_no_patterns():
    config = Config()
    config.known_patterns = []
    config.sections = ["placement1"]
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_no_sections():
    config = Config()
    config.known_patterns = [("test.*", "placement1")]
    config.sections = []
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #60
#--------------------------

```python
def test__is_namespace_package_returns_true_when_init_file_exists_with_namespace_declaration():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py"])
    init_file = path / "__init__.py"
    init_file.write_text("__import__('pkg_resources').declare_namespace(__name__)")
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #61
#--------------------------

```python
def test_forced_separate_predicate_evaluates_to_true():
    config = Config(forced_separate=["test*"])
    name = "test_file"
    result = _forced_separate(name, config)
    assert result is not None
    assert result[0] == "test*"
    assert result[1] == "Matched forced_separate (test*) config value."


# LLM-generated content at query #62
#--------------------------

```python
def test_src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "valid_module") == True

def test_src_path_is_module_returns_false_for_file():
    src_path = Path("file.py")
    src_path.is_dir = lambda: False
    assert _src_path_is_module(src_path, "file.py") == False

def test_src_path_is_module_returns_false_for_different_name():
    src_path = Path("module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "different_name") == False

def test_src_path_is_module_returns_false_for_case_insensitive_match():
    src_path = Path("Module")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module") == False


# LLM-generated content at query #63
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #64
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
    src_paths = None
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #65
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("# Regular package")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_file_has_pkg_resources_declaration():
    from pathlib import Path
    path = Path("/tmp/test_namespace_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_file_has_pkgutil_declaration():
    from pathlib import Path
    path = Path("/tmp/test_namespace_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_source_files():
    from pathlib import Path
    path = Path("/tmp/test_package_with_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# Some code")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_setup_cfg():
    from pathlib import Path
    path = Path("/tmp/test_package_with_setup")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname=test")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_pyproject_toml():
    from pathlib import Path
    path = Path("/tmp/test_package_with_pyproject")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = []")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_for_empty_directory():
    from pathlib import Path
    path = Path("/tmp/empty_namespace_package")
    path.mkdir(exist_ok=True)
    assert _is_namespace_package(path, frozenset([".py"]))


# LLM-generated content at query #66
#--------------------------

```python
def test_init_file_exists():
    init_file = Path("/some/path/__init__.py")
    assert init_file.exists() is True


# LLM-generated content at query #67
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages=("module",),
        auto_identify_namespace_packages=False,
        supported_extensions=(".py",),
        src_paths=()
    )
    src_paths = (Path("module"),)
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #68
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    assert _is_module(path) is True


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_true():
    path = Path("some_path")
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #70
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=[],
        auto_identify_namespace_packages=True,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    module_path = (Path("/path/to/src") / "module").resolve()
    assert _is_namespace_package(module_path, [".py"]) == True
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #71
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    path.touch()
    assert _is_module(path)


# LLM-generated content at query #72
#--------------------------

```python
def test_known_pattern_matches_first_pattern():
    config = Config(known_patterns=[(re.compile(r"test.*"), "section1")], sections=["section1"])
    result = _known_pattern("test.module", config)
    assert result == ("section1", "Matched configured known pattern test.*")

def test_known_pattern_matches_second_pattern():
    config = Config(known_patterns=[(re.compile(r"test.*"), "section1"), (re.compile(r"module.*"), "section2")], sections=["section1", "section2"])
    result = _known_pattern("test.module", config)
    assert result == ("section1", "Matched configured known pattern test.*")

def test_known_pattern_no_match():
    config = Config(known_patterns=[(re.compile(r"other.*"), "section1")], sections=["section1"])
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile(r"test.*"), "section2")], sections=["section1"])
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_empty_name():
    config = Config(known_patterns=[(re.compile(r"test.*"), "section1")], sections=["section1"])
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_single_part_name():
    config = Config(known_patterns=[(re.compile(r"test"), "section1")], sections=["section1"])
    result = _known_pattern("test", config)
    assert result == ("section1", "Matched configured known pattern test")

def test_known_pattern_multi_part_name():
    config = Config(known_patterns=[(re.compile(r"test.*module"), "section1")], sections=["section1"])
    result = _known_pattern("test.middle.module", config)
    assert result == ("section1", "Matched configured known pattern test.*module")


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=[],
        auto_identify_namespace_packages=True,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()

    # Setup mock filesystem
    module_path = Path("/path/to/src/module").resolve()
    module_path.mkdir(parents=True, exist_ok=True)
    (module_path / "__init__.py").touch()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #74
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("src")],
        namespace_packages=["module"],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #75
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
    namespace = "module"
    module_path = (Path("/path/to/src") / "module").resolve()
    nested_module = ["submodule"]
    result = namespace in config.namespace_packages or (
        config.auto_identify_namespace_packages
        and _is_namespace_package(module_path, config.supported_extensions)
    )
    assert not result


