####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils", config) == "FIRSTPARTY"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #2
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #3
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"])
    )
    assert module("project.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #4
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module("..parent_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"
    assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
    )
    assert module("auto_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #5
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.package") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("my_module", config) == "FIRSTPARTY"
    assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["my_namespace"],
    )
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #6
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module.exists_case_sensitive", return_value=True):
        with patch("isort.place.module.Path.is_dir", return_value=True):
            assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=False,
    )
    with patch("isort.place.module.exists_case_sensitive", return_value=True):
        with patch("isort.place.module.Path.is_dir", return_value=True):
            assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module.exists_case_sensitive", return_value=False):
        assert module("nonexistent", config) == DEFAULT_CONFIG.default_section


# LLM-generated content at query #7
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #8
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #9
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == config.default_section
    assert module("sys") == config.default_section
    assert module("django") == config.default_section
    assert module(".") == LOCAL
    assert module(".local") == LOCAL
    assert module("myproject") == sections.FIRSTPARTY
    assert module("myproject.utils") == sections.FIRSTPARTY
    assert module("thirdparty") == config.default_section
    assert module("thirdparty.utils") == config.default_section


# LLM-generated content at query #10
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #11
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_first_party=["my_project"])
    assert module("my_project.utils", config) == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.subpackage"]
    )
    assert module("project.subpackage.module", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #12
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place_module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place_module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #13
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module("..parent_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_module").mkdir()
        (src_path / "my_module" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_module", config) == "FIRSTPARTY"
        assert module("my_module.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #14
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #15
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == sections.STDLIB
    assert module("sys") == sections.STDLIB
    assert module("django") == sections.THIRDPARTY
    assert module("numpy") == sections.THIRDPARTY
    assert module("my_project") == sections.FIRSTPARTY
    assert module(".local_module") == LOCAL
    assert module("unknown_module") == config.default_section


# LLM-generated content at query #16
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["company"])
    assert module("company.module") == "FIRSTPARTY"
    assert module("company.sub.module") == "FIRSTPARTY"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #17
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os", config) == "STDLIB"
    assert module("django", config) == "THIRDPARTY"
    assert module("my_project", config) == "FIRSTPARTY"
    assert module(".local_module", config) == "LOCALFOLDER"
    assert module("custom_pattern", config) == "DEFAULT"


# LLM-generated content at query #18
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #19
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #20
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["mynamespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("mynamespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #21
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.sub", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"
    with patch("isort.place.module._is_package", return_value=True):
        assert module("my_package", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_pkg"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace_pkg.sub", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #22
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.sub"]
    )
    assert module("project.sub.module", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True
    )
    assert module("project.sub.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #23
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project.sub"])
    assert module("project.sub.module", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #24
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".another_local") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django", "pytest"])
    assert module("django", config) == "django"
    assert module("pytest", config) == "pytest"
    assert module("django.contrib", config) == "django"
    assert module("pytest.cov", config) == "pytest"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project") == "FIRSTPARTY"
        assert module("my_project.module") == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace") == "FIRSTPARTY"
        assert module("namespace.submodule") == "FIRSTPARTY"

    # Test auto identify namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "auto_namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("auto_namespace") == "FIRSTPARTY"
        assert module("auto_namespace.submodule") == "FIRSTPARTY"


# LLM-generated content at query #25
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils", config) == "FIRSTPARTY"
    assert module("mycompany.core.models", config) == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/home/user/projects/myproject")])
    assert module("myproject", config) == "FIRSTPARTY"
    assert module("myproject.utils", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/home/user/projects/namespace_pkg")],
        namespace_packages=["namespace_pkg"]
    )
    assert module("namespace_pkg.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("some_unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #26
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.module") == "FIRSTPARTY"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_pkg"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=False):
            with patch("builtins.open", mock_open(read_data=b"")):
                assert module("namespace_pkg.module", config) == "FIRSTPARTY"


# LLM-generated content at query #27
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "mymodule"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == "FIRSTPARTY"
        assert module("mymodule.submodule", config) == "FIRSTPARTY"

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #28
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.core", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test nested namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #29
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_utils.helper", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.utils", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.subpackage", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #30
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib.auth", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.sub_module", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #31
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #32
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils") == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module") == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace_pkg"])
    assert module("namespace_pkg.submodule") == "FIRSTPARTY"

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #33
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("project_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"")):
                assert module("project.namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #34
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_module.py").write_text("# test")
        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").write_text("# test")
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #35
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #36
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #37
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/src/module")):
            assert module("module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/src/namespace")):
            with patch("_is_namespace_package", return_value=True):
                assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #38
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.nested"]
    )
    assert module("project.nested.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #39
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #40
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "mymodule"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #41
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.sub"],
        auto_identify_namespace_packages=True
    )
    assert module("project.sub.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #42
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.sub"],
        auto_identify_namespace_packages=True
    )
    assert module("project.sub.module", config) == "FIRSTPARTY"

    # Test non-existent module falls back to default
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #43
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils") == "FIRSTPARTY"
    assert module("mycompany.core") == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/namespace")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.sub", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #44
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
    )
    assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
    )
    assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #45
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #46
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace", config) == sections.FIRSTPARTY
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #47
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module("..parent_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").touch()

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #48
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("myproject", config) == sections.FIRSTPARTY


# LLM-generated content at query #49
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #50
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #51
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #52
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django", "flask"])
    assert module("django", config) == "django"
    assert module("flask", config) == "flask"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"
    assert module("django.core", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #53
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #54
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_pkg"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace_pkg.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #55
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #56
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("myproject.module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #57
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #58
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").write_text("# test")
        config = Config(src_paths=[src_path])
        assert module("module", config) == "FIRSTPARTY"
        assert module("module.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace_package"
        src_path.mkdir()
        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace_package"],
        )
        assert module("namespace_package", config) == "FIRSTPARTY"
        assert module("namespace_package.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #59
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #60
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["company"])
    assert module("company.module", config) == "FIRSTPARTY"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.sub"],
    )
    assert module("project.sub.module", config) == "FIRSTPARTY"

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True,
    )
    assert module("project.sub.module", config) == "FIRSTPARTY"


# LLM-generated content at query #61
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils") == "FIRSTPARTY"
    assert module("mycompany.core.models") == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.utils", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.sub", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #62
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("namespace.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #63
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".another_local") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #64
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #65
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #66
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY


# LLM-generated content at query #67
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_module").mkdir()
        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #68
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test_"), "TESTS")])
    assert module("test_utils", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY
        assert module("my_project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        namespace_path = Path(tmpdir) / "namespace"
        namespace_path.mkdir()
        (namespace_path / "submodule.py").touch()

        config = Config(src_paths=[namespace_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #69
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^test_"), "TEST")])
    assert module("test_module", config) == "TEST"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=False):
            with patch("builtins.open", mock_open(read_data=b"")):
                assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #70
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")
        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"
        assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace package
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").write_text("")
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"


# LLM-generated content at query #71
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["numpy", "pandas"])
    assert module("numpy", config) == "numpy"
    assert module("pandas.core", config) == "pandas"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").write_text("# test")

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY


# LLM-generated content at query #72
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module("..parent_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/src/module")):
            assert module("module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/src/namespace")):
            with patch("_is_namespace_package", return_value=True):
                assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #73
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #74
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #75
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("sys") == sections.STANDARD_LIBRARY

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["django", "flask"])
    assert module("django", config) == "django"
    assert module("flask", config) == "flask"
    assert module("django.contrib", config) == "django"
    assert module("flask.ext", config) == "flask"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY
            assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("_is_namespace_package", return_value=True):
                assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #76
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile(r"^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").write_text("")

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"


# LLM-generated content at query #77
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test default section for unknown modules
    assert module("unknown_module") == "THIRDPARTY"
    assert module("unknown.package") == "THIRDPARTY"


# LLM-generated content at query #78
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local") == "LOCALFOLDER"
    assert module(".module") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src paths
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"
        assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "auto_namespace").mkdir()
        (src_path / "auto_namespace" / "module.py").touch()

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("auto_namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #79
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "my_module"
        module_path.mkdir()
        (module_path / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY
        assert module("my_module.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").write_text("")

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace", config) == sections.FIRSTPARTY
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "auto_namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").write_text("")

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("auto_namespace", config) == sections.FIRSTPARTY
        assert module("auto_namespace.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #80
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["numpy", "pandas"])
    assert module("numpy", config) == "numpy"
    assert module("pandas", config) == "pandas"
    assert module("numpy.core", config) == "numpy"
    assert module("pandas.io", config) == "pandas"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src paths
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY
        assert module("my_project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace_pkg"])
        assert module("namespace_pkg", config) == sections.FIRSTPARTY
        assert module("namespace_pkg.submodule", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "auto_ns_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("auto_ns_pkg", config) == sections.FIRSTPARTY
        assert module("auto_ns_pkg.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #81
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #82
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #83
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #84
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #85
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test module with multiple parts
    assert module("os.path") == DEFAULT_CONFIG.default_section

    # Test non-existent module
    assert module("nonexistent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #86
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.sub", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY
    with patch("isort.place.module._is_package", return_value=True):
        assert module("my_package", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #87
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("sys") == sections.STANDARD_LIBRARY

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), sections.FIRSTPARTY)])
    assert module("django.contrib", config) == sections.FIRSTPARTY

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True)
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        with patch("isort.place.module._is_namespace_package", return_value=True):
            assert module("auto_namespace.sub_module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section=sections.THIRDPARTY)
    assert module("unknown_module", config) == sections.THIRDPARTY


# LLM-generated content at query #88
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY
    assert module("src_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_package"]
    )
    assert module("namespace_package.submodule", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True
    )
    assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #89
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #90
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STDLIB
    assert module("sys") == sections.STDLIB

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.subpackage"]
    )
    assert module("project.subpackage.module", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("non_existent_module") == sections.THIRDPARTY


# LLM-generated content at query #91
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #92
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "mymodule").mkdir()
        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #93
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "my_module"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY
        assert module("my_module.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").write_text("print('hello')")

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #94
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_first_party=["company"])
    assert module("company.module", config) == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #95
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #96
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    import re
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto_identify_namespace_packages
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True)
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #97
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == "FIRSTPARTY"
        assert module("my_project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test non-existent module falls back to default
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #98
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["django", "flask"])
    assert module("django", config) == "django"
    assert module("flask", config) == "flask"
    assert module("django_ext", config) == "django"
    assert module("flask_app", config) == "flask"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #99
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils") == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project.sub"])
    assert module("project.sub.module", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"

    # Test with custom config
    config = Config(default_section="CUSTOM")
    assert module("unknown", config) == "CUSTOM"


# LLM-generated content at query #100
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test nested namespace package
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.nested", config) == "FIRSTPARTY"

    # Test case sensitivity
    config = Config()
    with patch("isort.utils.exists_case_sensitive", return_value=False):
        assert module("CaseSensitive", config) == "THIRDPARTY"


# LLM-generated content at query #101
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #102
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("myproject.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #103
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #104
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["mynamespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("mynamespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #105
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another.local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_utils", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY
        assert module("my_project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #106
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["pytest"])
    assert module("pytest", config) == "pytest"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.sub"],
        auto_identify_namespace_packages=True
    )
    assert module("project.sub.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #107
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.models", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #108
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"
    with patch("isort.place.module._is_package", return_value=True):
        assert module("my_package", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #109
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project", config) == "FIRSTPARTY"
        assert module("my_project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        namespace_path = Path(tmpdir) / "namespace"
        namespace_path.mkdir()
        config = Config(
            src_paths=[namespace_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #110
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known_patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #111
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"
    assert module("test_another", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".another.local") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"
    assert module("django.core", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("myproject.module", config) == "FIRSTPARTY"
        assert module("myproject", config) == "FIRSTPARTY"

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #112
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY
    assert module("src.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True
    )
    assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #113
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #114
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"
        assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").touch()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #115
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local_module.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("project_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("project.namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #116
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test"])
    assert module("test_module", config) == "test"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.module", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True
    )
    assert module("auto_namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #117
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/src/module")):
            assert module("module", config) == sections.FIRSTPARTY
            assert module("module.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/src/namespace")):
            with patch("pathlib.Path.iterdir", return_value=[]):
                assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #118
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local module
    assert module(".local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "THIRDPARTY")])
    assert module("django.contrib", config) == "THIRDPARTY"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent") == "STDLIB"


# LLM-generated content at query #119
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test.utils", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("myproject", config) == "FIRSTPARTY"
        assert module("myproject.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "subnamespace").mkdir()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace", config) == "FIRSTPARTY"
        assert module("namespace.subnamespace", config) == "FIRSTPARTY"

    # Test custom default section
    config = Config(default_section="CUSTOM")
    assert module("unknown_module", config) == "CUSTOM"


# LLM-generated content at query #120
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"])
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("auto_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #2
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"]),
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("some_module", config) == "THIRDPARTY"


# LLM-generated content at query #3
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #4
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_utils", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("isort.utils._is_namespace_package", return_value=True):
                assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("some_external_library", config) == "THIRDPARTY"


# LLM-generated content at query #5
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY
    with patch("isort.place.module._is_package", return_value=True):
        assert module("my_package", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #6
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("src_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #7
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".another_local") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"
    assert module("test_another", config) == "test*"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.core", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #8
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("myproject", config) == sections.FIRSTPARTY

    # Test namespace package
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #9
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == "STDLIB"
    assert module("django") == "THIRDPARTY"
    assert module("my_project") == "FIRSTPARTY"
    assert module(".local_module") == "LOCALFOLDER"
    assert module("unknown_module") == config.default_section


# LLM-generated content at query #10
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #11
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another.local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #12
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("project", config) == "FIRSTPARTY"
        assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("project", config) == "FIRSTPARTY"
        assert module("project.submodule", config) == "FIRSTPARTY"

    # Test third party
    assert module("requests") == "THIRDPARTY"
    assert module("flask") == "THIRDPARTY"


# LLM-generated content at query #13
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #14
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #15
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test.utils", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == "FIRSTPARTY"
        assert module("my_project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace_pkg"],
        )
        assert module("namespace_pkg.submodule", config) == "FIRSTPARTY"

    # Test auto-identify namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "auto_ns_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            auto_identify_namespace_packages=True,
        )
        assert module("auto_ns_pkg.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #16
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"
    assert module("src_module.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #17
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == "STDLIB"
    assert module("django") == "THIRDPARTY"
    assert module("my_project") == "FIRSTPARTY"
    assert module(".local_module") == "LOCALFOLDER"
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #18
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["numpy"])
    assert module("numpy", config) == "numpy"
    assert module("numpy.core", config) == "numpy"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project.sub", config) == "FIRSTPARTY"

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True,
    )
    assert module("project.sub", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #19
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #20
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #21
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.core", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_pkg"]
    )
    assert module("namespace_pkg.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #22
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #23
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace package
    config = Config(src_paths=[Path("/path/to/src")], auto_identify_namespace_packages=True)
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        with patch("isort.place.module._is_namespace_package", return_value=True):
            assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #24
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os", config) == sections.STANDARD_LIBRARY
    assert module("django", config) == sections.THIRDPARTY
    assert module(".local_module", config) == LOCAL
    assert module("my_project", config) == sections.FIRSTPARTY
    assert module("numpy", config) == sections.THIRDPARTY
    assert module("pytest", config) == sections.THIRDPARTY


# LLM-generated content at query #25
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #26
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #27
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").write_text("# test")
        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #28
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[("pytest", "THIRDPARTY")])
    assert module("pytest", config) == "THIRDPARTY"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.sub"]
    )
    assert module("project.sub.module", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #29
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY
    assert module("src.sub_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True
    )
    assert module("namespace.sub_module", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #30
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").touch()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #31
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #32
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src path
    config = Config(src_paths=[Path("/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/src")], namespace_packages=["namespace"])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #33
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"
    assert module("requests") == "THIRDPARTY"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/project")):
            assert module("project", config) == "FIRSTPARTY"
            assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project"],
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/project")):
            assert module("project.submodule", config) == "FIRSTPARTY"

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("pathlib.Path.resolve", return_value=Path("/path/to/project")):
            with patch("_is_namespace_package", return_value=True):
                assert module("project.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #34
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #35
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module("..parent_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY
        assert module("my_project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "auto_namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            auto_identify_namespace_packages=True,
        )
        assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #36
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #37
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #38
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #39
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section
    assert module("sys") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"
    assert module(".test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another.local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"
    assert module("test_example.module", config) == "TEST"

    # Test src paths
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "mymodule").mkdir()
        (src_path / "mymodule" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == sections.FIRSTPARTY
        assert module("mymodule.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").write_text("")

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test auto-identify namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "auto_ns").mkdir()
        (src_path / "auto_ns" / "module.py").write_text("")

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("auto_ns.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #40
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".another_local") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django", "flask"])
    assert module("django", config) == "django"
    assert module("flask", config) == "flask"
    assert module("django.contrib", config) == "django"
    assert module("flask.ext", config) == "flask"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another", config) == "TESTS"
    assert module("tests.utils", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.utils", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.subpackage", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset(["py"])
    )
    assert module("project.subpackage", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("requests", config) == "THIRDPARTY"


# LLM-generated content at query #41
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY
        assert module("my_project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module falls back to default
    assert module("nonexistent_module_xyz") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #42
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY
    with patch("isort.place.module._is_package", return_value=True):
        assert module("my_package", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #43
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.sub", config) == "FIRSTPARTY"


# LLM-generated content at query #44
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["pytest"])
    assert module("pytest", config) == "pytest"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("pathlib.Path.iterdir", return_value=[]):
                assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #45
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #46
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #47
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("myproject", config) == "FIRSTPARTY"
        assert module("myproject.module", config) == "FIRSTPARTY"

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #48
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"")):
                assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #49
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project"]
    )
    assert module("project.subpackage", config) == "FIRSTPARTY"

    # Test third party
    assert module("requests") == "THIRDPARTY"
    assert module("flask") == "THIRDPARTY"


# LLM-generated content at query #50
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local module
    assert module(".local") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django.*"), "THIRDPARTY")])
    assert module("django.contrib", config) == "THIRDPARTY"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.sub"]
    )
    assert module("project.sub.module", config) == "FIRSTPARTY"

    # Test auto identify namespace package
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True
    )
    assert module("project.sub.module", config) == "FIRSTPARTY"

    # Test module in src path
    config = Config(src_paths=[Path("/path/to/module")])
    assert module("module", config) == "FIRSTPARTY"


# LLM-generated content at query #51
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test.utils", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.module", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("external_library", config) == "THIRDPARTY"


# LLM-generated content at query #52
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").write_text("# test")
        config = Config(src_paths=[src_path])
        assert module("myproject", config) == sections.FIRSTPARTY


# LLM-generated content at query #53
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #54
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #55
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #56
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("mymodule", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["mynamespace"],
        auto_identify_namespace_packages=False
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("mynamespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #57
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #58
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #59
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another.local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").touch()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #60
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "my_module"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test non-existent module falls back to default
    assert module("nonexistent_module_12345") == "THIRDPARTY"


# LLM-generated content at query #61
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #62
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project_module", config) == "FIRSTPARTY"


# LLM-generated content at query #63
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "mymodule"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == "FIRSTPARTY"
        assert module("mymodule.sub", config) == "FIRSTPARTY"

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #64
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src_paths
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_package"]
    )
    assert module("namespace_package.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #65
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["numpy"])
    assert module("numpy", config) == "numpy"

    # Test local module
    assert module(".local") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #66
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test local module
    assert module(".local") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"


# LLM-generated content at query #67
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local_module.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_module.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "src"
        src_path.mkdir()
        (src_path / "namespace_package").mkdir()
        (src_path / "namespace_package" / "module.py").write_text("")

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace_package"],
        )
        assert module("namespace_package.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #68
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_utils", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.utils", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.sub", config) == "FIRSTPARTY"


# LLM-generated content at query #69
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_utils", config) == "TESTS"
    assert module("test_utils.helper", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #70
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #71
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["numpy", "pandas"])
    assert module("numpy", config) == "numpy"
    assert module("pandas.core", config) == "pandas"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        namespace_path = Path(tmpdir) / "namespace"
        namespace_path.mkdir()
        (namespace_path / "submodule.py").touch()
        config = Config(
            src_paths=[Path(tmpdir)],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #72
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True), patch(
        "pathlib.Path.resolve", return_value=Path("/path/to/src/module")
    ), patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("pathlib.Path.is_dir", return_value=True), patch(
        "pathlib.Path.resolve", return_value=Path("/path/to/src/namespace")
    ), patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test auto identify namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
        support_extensions=frozenset(["py"])
    )
    with patch("pathlib.Path.is_dir", return_value=True), patch(
        "pathlib.Path.resolve", return_value=Path("/path/to/src/namespace")
    ), patch("isort.utils.exists_case_sensitive", return_value=False), patch(
        "pathlib.Path.iterdir", return_value=[]
    ):
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #73
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src paths
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        namespace_path = Path(tmpdir) / "namespace"
        namespace_path.mkdir()
        (namespace_path / "submodule.py").touch()

        config = Config(
            src_paths=[Path(tmpdir)],
            namespace_packages=["namespace"],
        )
        assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #74
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django", "flask"])
    assert module("django", config) == "django"
    assert module("flask", config) == "flask"
    assert module("django.contrib", config) == "django"
    assert module("flask.ext", config) == "flask"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src paths
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"
        assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace_package").mkdir()
        (src_path / "namespace_package" / "module.py").write_text("")

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace_package"],
        )
        assert module("namespace_package", config) == "FIRSTPARTY"
        assert module("namespace_package.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #75
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("django") == sections.THIRDPARTY
    assert module(".local_module") == LOCAL
    assert module("my_project") == sections.FIRSTPARTY
    assert module("unknown_module") == config.default_section


# LLM-generated content at query #76
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local_module.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_module.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace package
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace_package").mkdir()

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("namespace_package", config) == sections.FIRSTPARTY
        assert module("namespace_package.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #77
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.core", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_namespace.sub_module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #78
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #79
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #80
#--------------------------

```python
def test_module():
    # Test default section
    config = Config()
    assert module("os") == config.default_section

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module") == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("django.*"), "DJANGO")])
    assert module("django.contrib") == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module") == sections.FIRSTPARTY


# LLM-generated content at query #81
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == "FIRSTPARTY"


# LLM-generated content at query #82
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #83
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.sub"],
        auto_identify_namespace_packages=True
    )
    assert module("project.sub.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #84
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["test_namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("test_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #85
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"
        assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.package", config) == "FIRSTPARTY"


# LLM-generated content at query #86
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["tests"])
    assert module("tests.test_module", config) == "tests"
    assert module("test_module", config) == "tests"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^django"), "DJANGO")])
    assert module("django.conf", config) == "DJANGO"
    assert module("django.apps", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == sections.FIRSTPARTY
        assert module("my_project", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace_pkg"],
        )
        assert module("namespace_pkg.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #87
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #88
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "mymodule"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #89
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module("..parent_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"
    assert module("src_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_package"],
    )
    assert module("namespace_package.submodule", config) == "FIRSTPARTY"

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
    )
    assert module("auto_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #90
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #91
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"],
    )
    assert module("project.namespace", config) == sections.FIRSTPARTY
    assert module("project.namespace.module", config) == sections.FIRSTPARTY

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True,
    )
    assert module("project.namespace", config) == sections.FIRSTPARTY
    assert module("project.namespace.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #92
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.submodule", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "submodule.py").write_text("")

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #93
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local") == "LOCALFOLDER"
    assert module(".local.module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"
    with patch("isort.place.module._is_package", return_value=True):
        assert module("my_package", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #94
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "mypackage").mkdir()
        (src_path / "mypackage" / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("mypackage", config) == sections.FIRSTPARTY


# LLM-generated content at query #95
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("project_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #96
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_package"],
    )
    assert module("namespace_package.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #97
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        module_path = src_path / "my_module"
        module_path.mkdir()
        (module_path / "__init__.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_module", config) == sections.FIRSTPARTY
        assert module("my_module.sub", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        (namespace_path / "module.py").write_text("test")

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #98
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module("..parent_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #99
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"
    assert module("django") == "THIRDPARTY"
    assert module("my_project") == "FIRSTPARTY"
    assert module(".local_module") == "LOCALFOLDER"
    assert module("pytest") == "THIRDPARTY"
    assert module("__future__") == "FUTURE"
    assert module("typing") == "TYPING"


# LLM-generated content at query #100
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("sys") == sections.STANDARD_LIBRARY

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("project_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test auto_identify_namespace_packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #101
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == sections.STDLIB
    assert module("django") == sections.THIRDPARTY
    assert module("my_project") == sections.FIRSTPARTY
    assert module(".local_module") == LOCAL
    config.known_patterns = [(re.compile("^test_.*"), "TESTS")]
    assert module("test_example", config) == "TESTS"
    config.forced_separate = ["custom_module"]
    assert module("custom_module", config) == "custom_module"


# LLM-generated content at query #102
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #103
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["custom"])
    assert module("custom_module", config) == "custom"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #104
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #105
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_paths
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"
    assert module("my_module.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #106
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".subpackage.module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils", config) == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.subpackage"]
    )
    assert module("project.subpackage.module", config) == "FIRSTPARTY"

    # Test default section for unknown modules
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #107
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"])
    )
    assert module("project.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #108
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".another.local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another.module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True,
    )
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #109
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #110
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test nested namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["parent"],
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True), \
         patch("isort.utils.exists_case_sensitive", return_value=True), \
         patch("isort.utils._is_namespace_package", return_value=True):
        assert module("parent.child", config) == sections.FIRSTPARTY


# LLM-generated content at query #111
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^test_.*"), "TEST")])
    assert module("test_example", config) == "TEST"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #112
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["numpy"])
    assert module("numpy", config) == "numpy"
    assert module("numpy.core", config) == "numpy"

    # Test local module
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module") as mock_is_module:
        mock_is_module.return_value = True
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False,
    )
    with patch("isort.place.module._is_namespace_package") as mock_is_namespace:
        mock_is_namespace.return_value = True
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #113
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.package") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #114
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"
    assert module("django.contrib", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.submodule", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "submodule.py").write_text("")

        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #115
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"])
    )
    assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY


