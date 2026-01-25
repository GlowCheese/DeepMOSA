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

def test__src_path_with_existing_module():
    config = Config(src_paths=[Path("src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: src.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("src")])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("src")], namespace_packages=["parent"])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("src")], auto_identify_namespace_packages=True)
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: src.")

def test__src_path_with_non_existing_module():
    config = Config(src_paths=[Path("src")])
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #2
#--------------------------

```python
def test_namespace_not_in_config_and_not_auto_identified():
    config = Config(
        src_paths=[Path("/some/path")],
        namespace_packages=[],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    name = "module.submodule"
    src_paths = [Path("/some/path")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is None


# LLM-generated content at query #3
#--------------------------

```python
def test__known_pattern_matches_first_pattern():
    config = Config(
        sections={"section1", "section2"},
        known_patterns=[(re.compile(r"^test\.module$"), "section1")]
    )
    result = _known_pattern("test.module.submodule", config)
    assert result == ("section1", "Matched configured known pattern test\.module$")

def test__known_pattern_matches_second_pattern():
    config = Config(
        sections={"section1", "section2"},
        known_patterns=[
            (re.compile(r"^test\.module$"), "section1"),
            (re.compile(r"^test\.module\.submodule$"), "section2")
        ]
    )
    result = _known_pattern("test.module.submodule", config)
    assert result == ("section2", "Matched configured known pattern test\.module\.submodule$")

def test__known_pattern_no_match():
    config = Config(
        sections={"section1", "section2"},
        known_patterns=[(re.compile(r"^other\.module$"), "section1")]
    )
    result = _known_pattern("test.module", config)
    assert result is None

def test__known_pattern_placement_not_in_sections():
    config = Config(
        sections={"section1"},
        known_patterns=[(re.compile(r"^test\.module$"), "section2")]
    )
    result = _known_pattern("test.module", config)
    assert result is None

def test__known_pattern_empty_name():
    config = Config(
        sections={"section1"},
        known_patterns=[(re.compile(r"^test$"), "section1")]
    )
    result = _known_pattern("", config)
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config(
        namespace_packages={"test_namespace"},
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"],
        src_paths=[Path("/test/src")]
    )
    name = "test_namespace.submodule"
    src_paths = [Path("/test/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages=("module",),
        auto_identify_namespace_packages=False,
        src_paths=[Path("src")],
        supported_extensions=[".py"]
    )
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_forced_separate_matches_exact_pattern():
    config = Config(forced_separate=["test"])
    result = _forced_separate("test", config)
    assert result == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_matches_pattern_with_wildcard():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("test123", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matches_hidden_file():
    config = Config(forced_separate=[".hidden"])
    result = _forced_separate(".hidden", config)
    assert result == (".hidden", "Matched forced_separate (.hidden) config value.")

def test_forced_separate_matches_hidden_file_with_wildcard():
    config = Config(forced_separate=["hidden"])
    result = _forced_separate(".hidden123", config)
    assert result == ("hidden", "Matched forced_separate (hidden) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["test"])
    result = _forced_separate("other", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("test", config)
    assert result is None


# LLM-generated content at query #8
#--------------------------

```python
def test_src_path_is_module_when_name_matches_and_is_dir_and_exists_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name")

def test_src_path_is_not_module_when_name_does_not_match():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "module_name")

def test_src_path_is_not_module_when_not_a_directory():
    src_path = Path("module_name")
    src_path.is_dir = lambda: False
    assert not _src_path_is_module(src_path, "module_name")

def test_src_path_is_not_module_when_does_not_exist_case_sensitively():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "module_name")


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_16_evaluates_to_true():
    name = "module"
    config = Config(src_paths=[Path("src")])
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_forced_separate_predicate_false():
    config = Config()
    config.forced_separate = ["test"]
    assert _forced_separate("other", config) is None


# LLM-generated content at query #11
#--------------------------

```python
def test_known_pattern_predicate_false():
    config = Config()
    config.known_patterns = [("pattern1", "placement1")]
    config.sections = []
    name = "test.module"
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #12
#--------------------------

```python
def test__is_namespace_package_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file_double_quotes():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file_pkgutil():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file_pkgutil_double_quotes():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file_no_namespace():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("# Regular package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_without_init_file():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_without_init_file_with_py_files():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# Some code")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_without_init_file_with_setup_cfg():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_without_init_file_with_pyproject_toml():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #13
#--------------------------

```python
def test__src_path_is_module_returns_true():
    src_path = Path("test_module")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "test_module") is True


# LLM-generated content at query #14
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

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("src", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #15
#--------------------------

```python
def test_forced_separate_not_ends_with_asterisk():
    config = Config()
    config.forced_separate = ["test"]
    name = "test_file"
    result = _forced_separate(name, config)
    assert result is not None
    assert result[0] == "test"


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_true():
    path = Path("some_path")
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("test_namespace")
    src_extensions = frozenset([".py"])
    (path / "module.py").write_text("")
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #19
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_path_is_a_package_with_init_file():
    from pathlib import Path
    path = Path("/existing/package/with/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_path_is_a_package_without_init_file_and_no_source_files():
    from pathlib import Path
    path = Path("/existing/package/without/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_path_is_a_package_without_init_file_but_has_source_files():
    from pathlib import Path
    path = Path("/existing/package/with/source/files")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_path_is_a_package_without_init_file_but_has_setup_cfg():
    from pathlib import Path
    path = Path("/existing/package/with/setup.cfg")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = test")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_path_is_a_package_without_init_file_but_has_pyproject_toml():
    from pathlib import Path
    path = Path("/existing/package/with/pyproject.toml")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_path_is_a_package_with_init_file_containing_pkg_resources_declare_namespace():
    from pathlib import Path
    path = Path("/existing/package/with/namespace/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_path_is_a_package_with_init_file_containing_pkgutil_extend_path():
    from pathlib import Path
    path = Path("/existing/package/with/namespace/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_path_is_a_package_with_init_file_not_containing_namespace_declaration():
    from pathlib import Path
    path = Path("/existing/package/without/namespace/__init__.py")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("print('hello')")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("some_path")
    init_file = path / "__init__.py"
    assert not init_file.exists()


# LLM-generated content at query #21
#--------------------------

```python
def test_init_file_exists_with_namespace_declaration():
    path = Path("some_namespace_package")
    init_file = path / "__init__.py"
    init_file.write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')
    assert _is_namespace_package(path, frozenset([".py"])) is True


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #24
#--------------------------

```python
def test__is_namespace_package_with_non_package_path():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_init_file():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("__path__ = __import__('pkgutil').extend_path(__path__, __name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_init_file_no_namespace_declaration():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("# Regular package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_no_init_file_but_source_files():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "module.py").write_text("# Module content")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_no_init_file_no_source_files():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_setup_cfg():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]\nname = package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_having_pyproject_toml():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_init_file_using_double_quotes():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_package_init_file_using_pkg_resources():
    from pathlib import Path
    path = Path("/path/to/package")
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #25
#--------------------------

```python
def test_known_pattern_returns_none_when_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile("test"), "placement")], sections=[])
    assert _known_pattern("test.module", config) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_init_file_exists():
    path = Path("some_path")
    src_extensions = frozenset([".py"])
    init_file = path / "__init__.py"
    init_file.touch()
    assert _is_namespace_package(path, src_extensions) == True


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_evaluates_to_false():
    name = "test.module"
    config = Config(known_patterns=[], sections=[])
    result = _known_pattern(name, config)
    assert result is None


# LLM-generated content at query #28
#--------------------------

```python
def test_src_path_returns_none_for_non_existent_module():
    config = Config(src_paths=[Path("/non/existent/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({".py"}))
    assert _src_path("non_existent_module", config) is None

def test_src_path_returns_firstparty_for_existing_module():
    config = Config(src_paths=[Path(".")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({".py"}))
    assert _src_path("existing_module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: .")

def test_src_path_returns_firstparty_for_nested_module():
    config = Config(src_paths=[Path(".")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({".py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: .")

def test_src_path_returns_firstparty_for_namespace_package():
    config = Config(src_paths=[Path(".")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({".py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: .")

def test_src_path_returns_firstparty_for_auto_identified_namespace_package():
    config = Config(src_paths=[Path(".")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({".py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: .")


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test__src_path_with_empty_name():
    assert _src_path("", Config(src_paths=[])) is None

def test__src_path_with_non_existent_module():
    assert _src_path("non_existent_module", Config(src_paths=[Path("/tmp")])) is None

def test__src_path_with_module_in_src_paths():
    assert _src_path("existing_module", Config(src_paths=[Path("/path/to/src")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module():
    assert _src_path("parent.child", Config(src_paths=[Path("/path/to/src")])) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"})
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True, supported_extensions=frozenset({".py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_module_in_root_src_path():
    config = Config(src_paths=[Path("/path/to/src")])
    assert _src_path("src", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("test_namespace_package")
    src_extensions = frozenset([".py"])
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# dummy content")
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #32
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #33
#--------------------------

```python
def test__src_path_is_module_returns_true_for_valid_module():
    src_path = Path("valid_module")
    src_path.is_dir = lambda: True
    module_name = "valid_module"
    assert _src_path_is_module(src_path, module_name)


# LLM-generated content at query #34
#--------------------------

```python
def test_init_file_does_not_exist():
    path = Path("nonexistent_path")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is False


# LLM-generated content at query #35
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #36
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None, namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("module", config) is None

def test__src_path_with_module_in_src_paths():
    src_path = Path("/path/to/src")
    module_path = src_path / "module"
    module_path.mkdir()
    config = Config(src_paths=[src_path], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_nested_module_in_src_paths():
    src_path = Path("/path/to/src")
    module_path = src_path / "parent" / "child"
    module_path.mkdir(parents=True)
    config = Config(src_paths=[src_path], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_namespace_package():
    src_path = Path("/path/to/src")
    namespace_path = src_path / "namespace"
    namespace_path.mkdir()
    config = Config(src_paths=[src_path], namespace_packages={"namespace"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("namespace.module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_auto_identified_namespace_package():
    src_path = Path("/path/to/src")
    namespace_path = src_path / "namespace"
    namespace_path.mkdir()
    (namespace_path / "module.py").write_text("content")
    config = Config(src_paths=[src_path], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({".py"}))
    assert _src_path("namespace.module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_src_path_as_module():
    src_path = Path("/path/to/module")
    src_path.mkdir()
    config = Config(src_paths=[src_path], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


# LLM-generated content at query #37
#--------------------------

```python
def test__is_namespace_package_not_a_package():
    from pathlib import Path
    path = Path("/nonexistent")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_init_file():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("__import__('pkg_resources').declare_namespace(__name__)")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_without_init_file():
    from pathlib import Path
    path = Path("/tmp/test_namespace")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_other_files():
    from pathlib import Path
    path = Path("/tmp/test_with_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# test")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_setup_cfg():
    from pathlib import Path
    path = Path("/tmp/test_with_setup")
    path.mkdir(exist_ok=True)
    (path / "setup.cfg").write_text("[metadata]")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_with_pyproject_toml():
    from pathlib import Path
    path = Path("/tmp/test_with_pyproject")
    path.mkdir(exist_ok=True)
    (path / "pyproject.toml").write_text("[build-system]")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_init_without_namespace_declaration():
    from pathlib import Path
    path = Path("/tmp/test_regular_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("# regular package")
    src_extensions = frozenset([".py"])
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_init_with_pkgutil():
    from pathlib import Path
    path = Path("/tmp/test_pkgutil_namespace")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    name = "module"
    config = Config(
        namespace_packages=[],
        auto_identify_namespace_packages=False,
        src_paths=[Path("src")],
        supported_extensions=[".py"]
    )
    src_paths = [Path("src/module")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is None


# LLM-generated content at query #39
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    (path / "module.py").write_text("")
    assert _is_module(path)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_true():
    path = Path("test_package")
    src_extensions = frozenset([".py"])
    filenames = [path / "module.py"]
    assert filenames


# LLM-generated content at query #41
#--------------------------

```python
def test_src_path_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_nested_module():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_not_found():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("nonexistent", config) is None


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["module"],
        auto_identify_namespace_packages=False
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #43
#--------------------------

```python
def test__src_path_with_empty_name():
    assert _src_path("", Config(src_paths=[Path("/some/path")])) is None

def test__src_path_with_none_src_paths():
    assert _src_path("module", Config(src_paths=None)) is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/some/path")], namespace_packages={"parent"})
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/some/path")], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_module_not_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("nonexistent", config) is None

def test__src_path_with_prefix_and_module_in_src_paths():
    config = Config(src_paths=[Path("/some/path")])
    assert _src_path("module", config, prefix=("parent",)) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_custom_src_paths():
    config = Config(src_paths=[Path("/custom/path")])
    assert _src_path("module", config, src_paths=[Path("/another/path")]) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /another/path.")


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_true():
    path = Path("test_package")
    src_extensions = frozenset([".py"])
    (path / "module.py").write_text("# test module")
    assert _is_namespace_package(path, src_extensions) is False


# LLM-generated content at query #45
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    assert _is_module(Path("module.py"))


# LLM-generated content at query #46
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
    src_paths = [Path("src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #47
#--------------------------

```python
def test__src_path_with_simple_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_with_module_not_found():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("nonexistent", config) is None

def test__src_path_with_root_module_in_src_path():
    config = Config(src_paths=[Path("/project/src/module")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset([".py"]))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src/module.")


# LLM-generated content at query #48
#--------------------------

```python
def test__is_namespace_package_returns_true_for_namespace_package():
    path = Path("namespace_package")
    path.mkdir()
    (path / "__init__.py").write_bytes(b'__import__("pkg_resources").declare_namespace(__name__)')

    assert _is_namespace_package(path, frozenset([".py"])) is True


# LLM-generated content at query #49
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    assert _is_module(path) is True


# LLM-generated content at query #50
#--------------------------

```python
def test__is_namespace_package_returns_true_when_package_has_no_init_file_and_no_source_files():
    path = Path("/fake/namespace/package")
    src_extensions = frozenset([".py", ".pyx", ".pxd", ".pyi"])

    # Mocking the Path behavior
    path.is_dir.return_value = True
    path.iterdir.return_value = []

    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py"])
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #52
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
    namespace = "module"
    module_path = (Path("/path/to/src") / "module").resolve()
    assert not (namespace in config.namespace_packages or (config.auto_identify_namespace_packages and _is_namespace_package(module_path, config.supported_extensions)))


# LLM-generated content at query #53
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


# LLM-generated content at query #54
#--------------------------

```python
def test__src_path_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_namespace_package_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset(["py"]))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test__src_path_not_found():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    assert _src_path("nonexistent", config) is None

def test__src_path_root_module_in_src_path():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset(["py"]))
    assert _src_path("src", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")


# LLM-generated content at query #55
#--------------------------

```python
def test__is_namespace_package_returns_true_when_init_file_exists_and_contains_declare_namespace():
    path = Path("some_path")
    src_extensions = frozenset([".py"])
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')

    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #56
#--------------------------

```python
def test__src_path_is_module_returns_true():
    src_path = Path("module_name")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "module_name")


# LLM-generated content at query #57
#--------------------------

```python
def test_filenames_list_not_empty():
    path = Path("some_path")
    src_extensions = frozenset([".py"])
    filenames = [
        filepath
        for filepath in path.iterdir()
        if filepath.suffix.lstrip(".") in src_extensions
        or filepath.name.lower() in ("setup.cfg", "pyproject.toml")
    ]
    assert filenames


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    # Setup
    name = "module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=[],
        auto_identify_namespace_packages=True,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()

    # Mock _is_namespace_package to return True
    original_is_namespace_package = _is_namespace_package
    _is_namespace_package = lambda *args: True

    # Execute
    result = _src_path(name, config, src_paths, prefix)

    # Restore
    _is_namespace_package = original_is_namespace_package

    # Assert
    assert result is not None


# LLM-generated content at query #59
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


# LLM-generated content at query #60
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("test_module.py")
    assert _is_module(path) is True


# LLM-generated content at query #61
#--------------------------

```python
def test__is_namespace_package_without_init_file():
    path = Path("test_package")
    path.mkdir()
    (path / "module.py").write_text("")
    assert _is_namespace_package(path, frozenset(["py"])) is True


# LLM-generated content at query #62
#--------------------------

```python
def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"package"})
    result = _src_path("package.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_nested_module_in_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/src")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    result = _src_path("package.submodule", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_module_in_src_path():
    config = Config(src_paths=[Path("/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_package_in_src_path():
    config = Config(src_paths=[Path("/src")])
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/src/module")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src/module.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/src")])
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #63
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

def test_is_module_nonexistent():
    path = Path("nonexistent")
    assert _is_module(path) == False


# LLM-generated content at query #64
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset({".py"})
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text("print('hello')")
    src_extensions = frozenset({".py"})
    assert not _is_namespace_package(path, src_extensions)
    init_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_true_when_init_file_contains_pkg_resources_declaration():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
    src_extensions = frozenset({".py"})
    assert _is_namespace_package(path, src_extensions)
    init_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_true_when_init_file_contains_pkgutil_declaration():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    init_file = path / "__init__.py"
    init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    src_extensions = frozenset({".py"})
    assert _is_namespace_package(path, src_extensions)
    init_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_package_contains_py_files():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    py_file = path / "module.py"
    py_file.write_text("print('hello')")
    src_extensions = frozenset({".py"})
    assert not _is_namespace_package(path, src_extensions)
    py_file.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_package_contains_setup_cfg():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    setup_cfg = path / "setup.cfg"
    setup_cfg.write_text("[metadata]\nname=test")
    src_extensions = frozenset({".py"})
    assert not _is_namespace_package(path, src_extensions)
    setup_cfg.unlink()
    path.rmdir()

def test__is_namespace_package_returns_false_when_package_contains_pyproject_toml():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    pyproject_toml = path / "pyproject.toml"
    pyproject_toml.write_text("[build-system]\nrequires = ['setuptools']")
    src_extensions = frozenset({".py"})
    assert not _is_namespace_package(path, src_extensions)
    pyproject_toml.unlink()
    path.rmdir()

def test__is_namespace_package_returns_true_when_package_is_empty():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    src_extensions = frozenset({".py"})
    assert _is_namespace_package(path, src_extensions)
    path.rmdir()


# LLM-generated content at query #65
#--------------------------

```python
def test_init_file_exists():
    init_file = path / "__init__.py"
    assert init_file.exists()


# LLM-generated content at query #66
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


# LLM-generated content at query #67
#--------------------------

```python
def test_predicate_evaluates_to_true():
    path = Path("some_namespace_package")
    src_extensions = frozenset([".py"])

    # Mock the behavior to make the predicate at line 5 evaluate to True
    path.is_dir.return_value = True
    (path / "__init__.py").exists.return_value = False
    path.iterdir.return_value = []

    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #68
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_init_exists_without_namespace_declaration():
    from pathlib import Path
    path = Path("/tmp/test_package")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text("# regular package")
    assert not _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_contains_pkg_resources_declaration():
    from pathlib import Path
    path = Path("/tmp/test_namespace_pkg")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text('__import__("pkg_resources").declare_namespace(__name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_true_when_init_contains_pkgutil_declaration():
    from pathlib import Path
    path = Path("/tmp/test_namespace_pkgutil")
    path.mkdir(exist_ok=True)
    (path / "__init__.py").write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
    assert _is_namespace_package(path, frozenset([".py"]))

def test__is_namespace_package_returns_false_when_package_contains_source_files():
    from pathlib import Path
    path = Path("/tmp/test_package_with_files")
    path.mkdir(exist_ok=True)
    (path / "module.py").write_text("# some code")
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


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_true():
    path = Path("test_namespace_package")
    path.mkdir()
    (path / "module.py").write_text("# test module")
    src_extensions = frozenset([".py"])

    assert _is_namespace_package(path, src_extensions) is False


# LLM-generated content at query #70
#--------------------------

```python
def test_is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #71
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    (path / "module.py").touch()
    assert _is_module(path) is True


# LLM-generated content at query #72
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages=[],
        auto_identify_namespace_packages=True,
        supported_extensions=[".py"],
        src_paths=[Path("/path/to/src")]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()
    module_path = Path("/path/to/src/module").resolve()
    assert _is_namespace_package(module_path, config.supported_extensions) is True
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_26():
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    name = "module.submodule"
    src_paths = [Path("src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_module_with_py_file():
    assert _is_module(Path("module")) == True

def test_is_module_with_extension_suffix():
    assert _is_module(Path("module")) == True

def test_is_module_with_init_py():
    assert _is_module(Path("module")) == True

def test_is_module_without_any_file():
    assert _is_module(Path("not_module")) == False


# LLM-generated content at query #2
#--------------------------

```python
def test_known_pattern_matches_first_pattern():
    config = Config(known_patterns=[(re.compile(r"test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_matches_second_pattern():
    config = Config(known_patterns=[(re.compile(r"test.*"), "placement1"), (re.compile(r"module.*"), "placement2")], sections=["placement1", "placement2"])
    result = _known_pattern("module.submodule", config)
    assert result == ("placement2", "Matched configured known pattern module.*")

def test_known_pattern_no_match():
    config = Config(known_patterns=[(re.compile(r"test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("other.module", config)
    assert result is None

def test_known_pattern_placement_not_in_sections():
    config = Config(known_patterns=[(re.compile(r"test.*"), "placement1")], sections=["placement2"])
    result = _known_pattern("test.module", config)
    assert result is None

def test_known_pattern_matches_submodule():
    config = Config(known_patterns=[(re.compile(r"test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("parent.test.module", config)
    assert result == ("placement1", "Matched configured known pattern test.*")

def test_known_pattern_empty_name():
    config = Config(known_patterns=[(re.compile(r"test.*"), "placement1")], sections=["placement1"])
    result = _known_pattern("", config)
    assert result is None

def test_known_pattern_empty_config():
    config = Config(known_patterns=[], sections=[])
    result = _known_pattern("test.module", config)
    assert result is None


# LLM-generated content at query #3
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
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset())
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_7():
    name = "test_module"
    config = Config(src_paths=[Path("test_path")])
    src_paths = None
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert src_paths is None


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

def test__src_path_with_module_in_src_path():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["parent"])
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_auto_identify_namespace_packages():
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True)
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_module_not_in_src_path():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("nonexistent", config)
    assert result is None

def test__src_path_with_prefix():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("module.submodule", config, prefix=("module",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/src")])
    result = _src_path("src", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #6
#--------------------------

```python
def test_forced_separate_matching_pattern_with_star():
    config = Config(forced_separate=["test*"])
    result = _forced_separate("testfile", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matching_pattern_without_star():
    config = Config(forced_separate=["test"])
    result = _forced_separate("testfile", config)
    assert result == ("test*", "Matched forced_separate (test*) config value.")

def test_forced_separate_matching_hidden_file():
    config = Config(forced_separate=[".hidden"])
    result = _forced_separate(".hiddenfile", config)
    assert result == (".hidden*", "Matched forced_separate (.hidden*) config value.")

def test_forced_separate_no_match():
    config = Config(forced_separate=["other*"])
    result = _forced_separate("testfile", config)
    assert result is None

def test_forced_separate_empty_config():
    config = Config(forced_separate=[])
    result = _forced_separate("testfile", config)
    assert result is None


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test__src_path_found_in_src_paths():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_nested_module():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_not_found():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    result = _src_path("nonexistent", config)
    assert result is None

def test__src_path_with_prefix():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    result = _src_path("module", config, prefix=("parent",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"namespace"}, auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    result = _src_path("namespace.module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")


# LLM-generated content at query #10
#--------------------------

```python
def test_src_paths_is_not_none():
    src_paths = [Path("/some/path")]
    result = _src_path("module", Config(src_paths=src_paths), src_paths=src_paths)
    assert result is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_forced_separate_matches_without_dot():
    name = "test_file"
    config = Config(forced_separate=["test*"])
    assert _forced_separate(name, config) == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_matches_with_dot():
    name = ".test_file"
    config = Config(forced_separate=["test*"])
    assert _forced_separate(name, config) == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_matches_without_glob():
    name = "test_file"
    config = Config(forced_separate=["test"])
    assert _forced_separate(name, config) == ("test", "Matched forced_separate (test) config value.")

def test_forced_separate_matches_with_dot_and_without_glob():
    name = ".test_file"
    config = Config(forced_separate=["test"])
    assert _forced_separate(name, config) == ("test", "Matched forced_separate (test) config value.")


# LLM-generated content at query #12
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    path.with_suffix(".py").touch()
    assert _is_module(path)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    config = Config()
    config.namespace_packages = {"test_namespace"}
    config.auto_identify_namespace_packages = False
    name = "test_namespace.submodule"
    src_paths = [Path("/path/to/src")]

    result = _src_path(name, config, src_paths)

    assert result is not None


# LLM-generated content at query #14
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
    result = _src_path(name, config, src_paths, prefix)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    (path / "__init__.py").touch()
    assert _is_module(path) is True


# LLM-generated content at query #16
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
    result = _src_path(name, config)
    assert result is not None


# LLM-generated content at query #17
#--------------------------

```python
def test__src_path_with_no_nested_module_and_valid_module_path():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_no_nested_module_and_invalid_module_path():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("nonexistent", config) is None

def test__src_path_with_nested_module_and_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_nested_module_and_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/path/to/module")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset({"py"}))
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /path/to/module.")


# LLM-generated content at query #18
#--------------------------

```python
def test__src_path_returns_none_for_non_existent_module():
    config = Config(src_paths=[Path("/fake/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("non_existent_module", config)
    assert result is None

def test__src_path_returns_firstparty_for_module_in_src_path():
    config = Config(src_paths=[Path("/real/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("existing_module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test__src_path_returns_firstparty_for_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/real/path")], namespace_packages={"parent"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test__src_path_returns_firstparty_for_module_in_nested_src_path():
    config = Config(src_paths=[Path("/real/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("parent.child", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test__src_path_returns_firstparty_for_module_with_prefix():
    config = Config(src_paths=[Path("/real/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("module", config, prefix=("parent",))
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test__src_path_returns_firstparty_for_namespace_package_with_auto_identify():
    config = Config(src_paths=[Path("/real/path")], namespace_packages=set(), auto_identify_namespace_packages=True, supported_extensions=frozenset())
    result = _src_path("namespace_pkg", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")


# LLM-generated content at query #19
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None, namespace_packages=(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("module", config)
    assert result is None

def test__src_path_with_empty_name():
    config = Config(src_paths=[Path("/some/path")], namespace_packages=(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("", config)
    assert result is None

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/some/path")], namespace_packages=(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    (Path("/some/path/module.py").resolve()).touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/some/path")], namespace_packages=(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    (Path("/some/path/package").resolve()).mkdir()
    (Path("/some/path/package/__init__.py").resolve()).touch()
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/some/path")], namespace_packages=("namespace",), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    (Path("/some/path/namespace").resolve()).mkdir()
    (Path("/some/path/namespace/nested.py").resolve()).touch()
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_auto_identified_namespace_package():
    config = Config(src_paths=[Path("/some/path")], namespace_packages=(), auto_identify_namespace_packages=True, supported_extensions=frozenset({"py"}))
    (Path("/some/path/namespace").resolve()).mkdir()
    (Path("/some/path/namespace/nested.py").resolve()).touch()
    result = _src_path("namespace.nested", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_non_existent_module():
    config = Config(src_paths=[Path("/some/path")], namespace_packages=(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    result = _src_path("nonexistent", config)
    assert result is None

def test__src_path_with_src_path_as_module():
    config = Config(src_paths=[Path("/some/path/module")], namespace_packages=(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    (Path("/some/path/module").resolve()).mkdir()
    (Path("/some/path/module/__init__.py").resolve()).touch()
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path/module.")


# LLM-generated content at query #20
#--------------------------

```python
def test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    assert _src_path_is_module(src_path, "module_name")

def test_src_path_is_not_module_when_name_does_not_match():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "module_name")

def test_src_path_is_not_module_when_not_directory():
    src_path = Path("module_name")
    src_path.is_dir = lambda: False
    assert not _src_path_is_module(src_path, "module_name")

def test_src_path_is_not_module_when_does_not_exist_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    assert not _src_path_is_module(src_path, "module_name")


# LLM-generated content at query #21
#--------------------------

```python
def test__is_namespace_package_returns_false_when_path_is_not_a_package():
    from pathlib import Path
    path = Path("/non/existent/path")
    src_extensions = frozenset({".py"})
    assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text("print('hello')")
        src_extensions = frozenset({".py"})
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__import__("pkg_resources").declare_namespace(__name__)')
        src_extensions = frozenset({".py"})
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        init_file = path / "__init__.py"
        init_file.write_text('__path__ = __import__("pkgutil").extend_path(__path__, __name__)')
        src_extensions = frozenset({".py"})
        assert _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_has_source_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        (path / "module.py").write_text("print('hello')")
        src_extensions = frozenset({".py"})
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_has_setup_cfg():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        (path / "setup.cfg").write_text("[metadata]\nname = test")
        src_extensions = frozenset({".py"})
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_false_when_no_init_file_and_has_pyproject_toml():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        (path / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']")
        src_extensions = frozenset({".py"})
        assert not _is_namespace_package(path, src_extensions)

def test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_package"
        path.mkdir()
        src_extensions = frozenset({".py"})
        assert _is_namespace_package(path, src_extensions)


# LLM-generated content at query #22
#--------------------------

```python
def test__src_path_with_none_src_paths():
    config = Config(src_paths=None)
    assert _src_path("module", config) is None

def test__src_path_with_empty_src_paths():
    config = Config(src_paths=[])
    assert _src_path("module", config) is None

def test__src_path_with_module_in_src_path():
    src_path = Path("/path/to/src")
    module_path = src_path / "module"
    module_path.mkdir()
    config = Config(src_paths=[src_path])
    assert _src_path("module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_nested_module_in_src_path():
    src_path = Path("/path/to/src")
    module_path = src_path / "parent" / "child"
    module_path.mkdir(parents=True)
    config = Config(src_paths=[src_path])
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_namespace_package():
    src_path = Path("/path/to/src")
    namespace_path = src_path / "namespace"
    namespace_path.mkdir()
    config = Config(src_paths=[src_path], namespace_packages=["namespace"])
    assert _src_path("namespace.module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_auto_identified_namespace_package():
    src_path = Path("/path/to/src")
    namespace_path = src_path / "namespace"
    namespace_path.mkdir()
    (namespace_path / "module.py").write_text("content")
    config = Config(src_paths=[src_path], auto_identify_namespace_packages=True, supported_extensions=frozenset([".py"]))
    assert _src_path("namespace.module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")

def test__src_path_with_module_not_found():
    src_path = Path("/path/to/src")
    config = Config(src_paths=[src_path])
    assert _src_path("nonexistent", config) is None

def test__src_path_with_src_path_as_module():
    src_path = Path("/path/to/module")
    src_path.mkdir()
    config = Config(src_paths=[src_path])
    assert _src_path("module", config) == (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_true():
    name = "module.submodule"
    config = Config(
        namespace_packages={"module"},
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_16_evaluates_to_true():
    name = "module"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src")]
    prefix = ()
    root_module_name, *nested_module = name.split(".", 1)
    new_prefix = (*prefix, root_module_name)
    namespace = ".".join(new_prefix)
    src_path = src_paths[0]
    module_path = (src_path / root_module_name).resolve()
    assert not prefix and not module_path.is_dir() and src_path.name == root_module_name


# LLM-generated content at query #25
#--------------------------

```python
def test_src_path_returns_none_when_module_not_found():
    result = _src_path("nonexistent_module", Config(src_paths=[Path("/fake/path")]))
    assert result is None

def test_src_path_returns_firstparty_when_module_found_in_src_paths():
    with patch("builtins.exists_case_sensitive", return_value=True):
        result = _src_path("existing_module", Config(src_paths=[Path("/real/path")]))
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test_src_path_handles_nested_modules_with_namespace_packages():
    config = Config(
        src_paths=[Path("/real/path")],
        namespace_packages={"parent"},
        supported_extensions=frozenset([".py"])
    )
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.iterdir", return_value=[]):
        result = _src_path("parent.child", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test_src_path_handles_auto_identified_namespace_packages():
    config = Config(
        src_paths=[Path("/real/path")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"])
    )
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.iterdir", return_value=[]):
        result = _src_path("auto_ns.child", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path.")

def test_src_path_returns_firstparty_when_src_path_is_module():
    config = Config(src_paths=[Path("/real/path/module")])
    with patch("builtins.exists_case_sensitive", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True):
        result = _src_path("module", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /real/path/module.")


# LLM-generated content at query #26
#--------------------------

```python
def test__src_path_with_empty_name():
    result = _src_path("", Config(src_paths=[Path("/some/path")]))
    assert result is None

def test__src_path_with_nonexistent_module():
    result = _src_path("nonexistent", Config(src_paths=[Path("/some/path")]))
    assert result is None

def test__src_path_with_module_in_src_paths():
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/some/path/module")), \
         patch("_is_module", return_value=True):
        result = _src_path("module", Config(src_paths=[Path("/some/path")]))
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_package_in_src_paths():
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/some/path/package")), \
         patch("_is_package", return_value=True):
        result = _src_path("package", Config(src_paths=[Path("/some/path")]))
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_nested_module_in_namespace_package():
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/some/path/namespace")), \
         patch("_is_namespace_package", return_value=True):
        config = Config(src_paths=[Path("/some/path")], namespace_packages=["namespace"])
        result = _src_path("namespace.nested", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_auto_identified_namespace_package():
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/some/path/namespace")), \
         patch("_is_namespace_package", return_value=True):
        config = Config(src_paths=[Path("/some/path")], auto_identify_namespace_packages=True)
        result = _src_path("namespace.nested", config)
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_src_path_is_module():
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/some/path")), \
         patch("_src_path_is_module", return_value=True):
        result = _src_path("module", Config(src_paths=[Path("/some/path")]))
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /some/path.")

def test__src_path_with_custom_src_paths():
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.resolve", return_value=Path("/custom/path/module")), \
         patch("_is_module", return_value=True):
        result = _src_path("module", Config(src_paths=[Path("/custom/path")]))
        assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /custom/path.")


# LLM-generated content at query #27
#--------------------------

```python
def test__is_namespace_package_returns_true_when_package_and_init_file_has_declare_namespace():
    path = Path("some_path")
    src_extensions = frozenset([".py"])

    # Mocking the behavior of _is_package to return True
    original_is_package = _is_package
    _is_package = lambda p: True

    # Mocking the behavior of path.iterdir to return an empty list
    original_iterdir = path.iterdir
    path.iterdir = lambda: []

    # Mocking the behavior of init_file.exists to return True
    init_file = path / "__init__.py"
    original_exists = init_file.exists
    init_file.exists = lambda: True

    # Mocking the behavior of init_file.open to return a file-like object with the required content
    original_open = init_file.open
    init_file.open = lambda mode: io.BytesIO(b'__import__("pkg_resources").declare_namespace(__name__)')

    try:
        assert _is_namespace_package(path, src_extensions) is True
    finally:
        _is_package = original_is_package
        path.iterdir = original_iterdir
        init_file.exists = original_exists
        init_file.open = original_open


# LLM-generated content at query #28
#--------------------------

```python
def test__is_namespace_package_predicate_true():
    assert _is_namespace_package(Path("some_path"), frozenset({"py"})) is True


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "module.submodule"
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["module"], auto_identify_namespace_packages=False)
    src_paths = [Path("/path/to/src")]
    prefix = ()
    result = _src_path(name, config, src_paths, prefix)
    assert result is not None


# LLM-generated content at query #30
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #31
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

def test__src_path_with_existent_module():
    config = Config(src_paths=[Path("/existent/path")])
    (Path("/existent/path/module.py")).touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existent/path.")

def test__src_path_with_nested_module():
    config = Config(src_paths=[Path("/existent/path")])
    (Path("/existent/path/parent")).mkdir()
    (Path("/existent/path/parent/__init__.py")).touch()
    (Path("/existent/path/parent/child.py")).touch()
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existent/path.")

def test__src_path_with_namespace_package():
    config = Config(src_paths=[Path("/existent/path")], namespace_packages={"parent"})
    (Path("/existent/path/parent")).mkdir()
    (Path("/existent/path/parent/__init__.py")).write_text('__import__("pkg_resources").declare_namespace(__name__)')
    (Path("/existent/path/parent/child.py")).touch()
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existent/path.")

def test__src_path_with_auto_identify_namespace_package():
    config = Config(src_paths=[Path("/existent/path")], auto_identify_namespace_packages=True, supported_extensions={"py"})
    (Path("/existent/path/parent")).mkdir()
    (Path("/existent/path/parent/__init__.py")).write_text('__import__("pkg_resources").declare_namespace(__name__)')
    (Path("/existent/path/parent/child.py")).touch()
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existent/path.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/existent/path/module")])
    (Path("/existent/path/module/__init__.py")).touch()
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /existent/path/module.")


# LLM-generated content at query #32
#--------------------------

```python
def test_src_path_finds_module_in_src_paths():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), supported_extensions=frozenset(["py"]), auto_identify_namespace_packages=False)
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_finds_nested_module():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), supported_extensions=frozenset(["py"]), auto_identify_namespace_packages=False)
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_returns_none_when_module_not_found():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), supported_extensions=frozenset(["py"]), auto_identify_namespace_packages=False)
    assert _src_path("nonexistent", config) is None

def test_src_path_handles_namespace_packages():
    config = Config(src_paths=[Path("/project/src")], namespace_packages={"parent"}, supported_extensions=frozenset(["py"]), auto_identify_namespace_packages=False)
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_handles_auto_identified_namespace_packages():
    config = Config(src_paths=[Path("/project/src")], namespace_packages=set(), supported_extensions=frozenset(["py"]), auto_identify_namespace_packages=True)
    assert _src_path("parent.child", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src.")

def test_src_path_handles_src_path_is_module():
    config = Config(src_paths=[Path("/project/src/module")], namespace_packages=set(), supported_extensions=frozenset(["py"]), auto_identify_namespace_packages=False)
    assert _src_path("module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /project/src/module.")


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test__src_path_is_module_returns_true():
    assert _src_path_is_module(Path("valid_module"), "valid_module") is True


# LLM-generated content at query #35
#--------------------------

```python
def test__src_path_is_module_returns_true():
    src_path = Path("module_name")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "module_name") is True


# LLM-generated content at query #36
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path)


# LLM-generated content at query #37
#--------------------------

```python
def test_src_paths_is_not_none():
    src_paths = [Path("some_path")]
    config = Config()
    result = _src_path("module", config, src_paths)
    assert result is None


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_16_evaluates_to_true():
    name = "module"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src")]
    prefix = ()
    root_module_name, *nested_module = name.split(".", 1)
    new_prefix = (*prefix, root_module_name)
    namespace = ".".join(new_prefix)
    src_path = Path("src")
    module_path = (src_path / root_module_name).resolve()
    assert not prefix and not module_path.is_dir() and src_path.name == root_module_name


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_16():
    name = "module"
    config = Config(src_paths=[Path("src")], namespace_packages=[], auto_identify_namespace_packages=False)
    src_paths = [Path("src/module")]
    prefix = ()
    module_path = (src_paths[0] / "module").resolve()
    assert not prefix and not module_path.is_dir() and src_paths[0].name == "module"


# LLM-generated content at query #40
#--------------------------

```python
def test__src_path_returns_none_when_no_matching_path():
    config = Config(src_paths=[Path("/fake/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    assert _src_path("nonexistent.module", config) is None

def test__src_path_returns_firstparty_when_module_found_in_src_paths():
    config = Config(src_paths=[Path("/fake/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("pathlib.Path.is_dir", return_value=True), patch("pathlib.Path.exists", return_value=True):
        assert _src_path("existing.module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /fake/path.")

def test__src_path_handles_nested_modules_with_namespace_packages():
    config = Config(src_paths=[Path("/fake/path")], namespace_packages={"existing"}, auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("pathlib.Path.is_dir", return_value=True), patch("pathlib.Path.exists", return_value=True), patch("_is_namespace_package", return_value=True):
        assert _src_path("existing.nested.module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /fake/path.")

def test__src_path_handles_src_path_is_module_case():
    config = Config(src_paths=[Path("/fake/path")], namespace_packages=set(), auto_identify_namespace_packages=False, supported_extensions=frozenset())
    with patch("pathlib.Path.is_dir", return_value=True), patch("pathlib.Path.exists", return_value=True), patch("_src_path_is_module", return_value=True):
        assert _src_path("existing.module", config) == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /fake/path.")


# LLM-generated content at query #41
#--------------------------

```python
def test__src_path_with_nested_module_in_namespace_package():
    config = Config(src_paths=[Path("/src")], namespace_packages={"test"}, supported_extensions={"py"})
    result = _src_path("test.module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_module_in_src_paths():
    config = Config(src_paths=[Path("/src")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_package_in_src_paths():
    config = Config(src_paths=[Path("/src")])
    result = _src_path("package", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src.")

def test__src_path_with_src_path_is_module():
    config = Config(src_paths=[Path("/src/module")])
    result = _src_path("module", config)
    assert result == (sections.FIRSTPARTY, "Found in one of the configured src_paths: /src/module.")

def test__src_path_with_no_match():
    config = Config(src_paths=[Path("/src")])
    result = _src_path("nonexistent", config)
    assert result is None


# LLM-generated content at query #42
#--------------------------

```python
def test__is_namespace_package_returns_true_when_package_without_init_file():
    path = Path("some_package")
    src_extensions = frozenset([".py", ".pyx"])
    _is_package = lambda p: True
    assert _is_namespace_package(path, src_extensions) is True


# LLM-generated content at query #43
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    path = Path("module.py")
    assert _is_module(path) is True


# LLM-generated content at query #44
#--------------------------

```python
def test__is_module_returns_true_for_py_file():
    assert _is_module(Path("test_module.py")) is True


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_true():
    name = "test_module.submodule"
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["test_module"],
        auto_identify_namespace_packages=False,
        supported_extensions=[".py"]
    )
    src_paths = [Path("/path/to/src/test_module")]
    prefix = ()

    result = _src_path(name, config, src_paths, prefix)

    assert result is not None
    assert result[0] == sections.FIRSTPARTY
    assert "Found in one of the configured src_paths:" in result[1]


# LLM-generated content at query #46
#--------------------------

```python
def test_is_module_with_py_file():
    path = Path("module")
    path.with_suffix(".py").write_text("")
    assert _is_module(path)


# LLM-generated content at query #47
#--------------------------

```python
def test__is_namespace_package_returns_false_when_not_package():
    from pathlib import Path

    path = Path("/some/path")
    src_extensions = frozenset([".py"])

    assert not _is_namespace_package(path, src_extensions)


# LLM-generated content at query #48
#--------------------------

```python
def test_src_path_is_module_returns_true_when_conditions_met():
    src_path = Path("test_module")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "test_module") is True
    src_path.rmdir()

def test_src_path_is_module_returns_false_when_name_mismatch():
    src_path = Path("test_module")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "other_module") is False
    src_path.rmdir()

def test_src_path_is_module_returns_false_when_not_directory():
    src_path = Path("test_module")
    src_path.touch()
    assert _src_path_is_module(src_path, "test_module") is False
    src_path.unlink()

def test_src_path_is_module_returns_false_when_case_sensitive_mismatch():
    src_path = Path("Test_Module")
    src_path.mkdir()
    assert _src_path_is_module(src_path, "test_module") is False
    src_path.rmdir()


# LLM-generated content at query #49
#--------------------------

```python
def test_src_path_is_module_when_module_name_matches_path_name_and_is_directory_and_exists_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is True

def test_src_path_is_not_module_when_module_name_does_not_match_path_name():
    src_path = Path("different_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is False

def test_src_path_is_not_module_when_path_is_not_directory():
    src_path = Path("module_name")
    src_path.is_dir = lambda: False
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is False

def test_src_path_is_not_module_when_path_does_not_exist_case_sensitive():
    src_path = Path("module_name")
    src_path.is_dir = lambda: True
    module_name = "module_name"
    assert _src_path_is_module(src_path, module_name) is False


