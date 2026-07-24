####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[("test*", "THIRDPARTY")])
    assert module("test_library", config) == "THIRDPARTY"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["project"])
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test auto_identify_namespace_packages
    config = Config(src_paths=[Path("/project/src")], auto_identify_namespace_packages=True)
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #2
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
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_module.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"
    assert module("src_module.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
    )
    assert module("namespace.module", config) == "FIRSTPARTY"


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test local module
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["django", "flask"])
    assert module("django", config) == "django"
    assert module("flask", config) == "flask"
    assert module("django_ext", config) == "django"
    assert module("flask_app", config) == "flask"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django.*"), "DJANGO")])
    assert module("django", config) == "DJANGO"
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("#")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == sections.FIRSTPARTY
        assert module("my_package.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").write_text("#")

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace"],
        )
        assert module("namespace.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #5
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == DEFAULT_CONFIG.default_section

    # Test local module
    assert module(".local_module") == LOCAL

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "mymodule").mkdir()
        config = Config(src_paths=[src_path])
        assert module("mymodule", config) == sections.FIRSTPARTY

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "namespace"
        namespace_path.mkdir()
        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        namespace_path = src_path / "auto_namespace"
        namespace_path.mkdir()
        config = Config(src_paths=[src_path], auto_identify_namespace_packages=True)
        assert module("auto_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #6
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
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #7
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
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test auto_identify_namespace_packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True,
    )
    assert module("project.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()
        config = Config(src_paths=[src_path])
        assert module("my_project.module", config) == "FIRSTPARTY"


# LLM-generated content at query #10
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
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test third party
    assert module("requests") == "THIRDPARTY"
    assert module("flask") == "THIRDPARTY"


# LLM-generated content at query #11
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local module
    assert module(".local_module") == "LOCALFOLDER"

    # Test known pattern
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
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
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.sub", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.sub", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.sub", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #13
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os", config) == "STDLIB"
    assert module("django", config) == "THIRDPARTY"
    assert module(".local_module", config) == "LOCALFOLDER"
    assert module("my_project", config) == "FIRSTPARTY"
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #14
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
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test.submodule", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=False
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #15
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os", config) == "STDLIB"
    assert module("django", config) == "THIRDPARTY"
    assert module(".local_module", config) == "LOCALFOLDER"
    assert module("my_project", config) == "FIRSTPARTY"
    assert module("unknown_module", config) == config.default_section


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
    assert module(".local_module") == LOCAL
    assert module(".another_local") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_another", config) == "TESTS"

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
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #17
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
        auto_identify_namespace_packages=True
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"")):
                assert module("my_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #18
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
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_utils", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("project_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("some_third_party", config) == "THIRDPARTY"


# LLM-generated content at query #19
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
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("non_existent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"
    assert module(".test_module", config) == "test*"

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module("..parent_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"
    assert module("django.shortcuts", config) == "DJANGO"

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
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

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
        src_path = Path(tmpdir) / "namespace_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace_pkg"])
        assert module("namespace_pkg", config) == "FIRSTPARTY"
        assert module("namespace_pkg.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #23
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

    # Test namespace package
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.submodule", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("non_existent_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #24
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

    # Test src path
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
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #25
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
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test nested namespace package
    config = Config(
        src_paths=[Path("/src")],
        namespace_packages=["parent"],
        auto_identify_namespace_packages=True
    )
    assert module("parent.child", config) == "FIRSTPARTY"


# LLM-generated content at query #26
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
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #27
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
    assert module(".local_module") == LOCAL
    assert module("..parent_module") == LOCAL

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
    assert module("non_existent_module") == "THIRDPARTY"


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
    assert module(".local_module") == LOCAL
    assert module(".local.submodule") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_utils", config) == "TESTS"
    assert module("test_utils.helper", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto-identify namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #29
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
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #30
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("sys") == sections.STANDARD_LIBRARY

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

    # Test non-existent module
    assert module("nonexistent_module") == sections.THIRDPARTY


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
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "my_package").mkdir()
        (src_path / "my_package" / "__init__.py").write_text("")

        config = Config(src_paths=[src_path])
        assert module("my_package", config) == "FIRSTPARTY"
        assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src"
        src_path.mkdir()
        (src_path / "namespace").mkdir()
        (src_path / "namespace" / "module.py").write_text("")

        config = Config(src_paths=[src_path], namespace_packages=["namespace"])
        assert module("namespace.module", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


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


# LLM-generated content at query #33
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

    # Test src_paths
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test auto_identify_namespace_packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #34
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["test*"])
    assert module("test_module", config) == "test*"

    # Test local folder
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

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #35
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
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("my_namespace.sub_module", config) == sections.FIRSTPARTY


# LLM-generated content at query #36
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
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"
    assert module("my_module.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace_pkg"]
    )
    assert module("namespace_pkg", config) == "FIRSTPARTY"
    assert module("namespace_pkg.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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

    # Test local folder
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    assert module("namespace.module", config) == "FIRSTPARTY"

    # Test auto_identify_namespace_packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
        supported_extensions=frozenset([".py"])
    )
    assert module("auto_namespace.module", config) == "FIRSTPARTY"

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("some_third_party", config) == "THIRDPARTY"


# LLM-generated content at query #2
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["numpy"])
    assert module("numpy", config) == "numpy"

    # Test local folder
    assert module(".local") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("mymodule", config) == "FIRSTPARTY"


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
    assert module(".local_module") == LOCAL
    assert module(".local.module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.module", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/project")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.namespace"],
    )
    assert module("project.namespace.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #4
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
    config = Config(known_patterns=[(re.compile(r"^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.module", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY
    assert module("project.module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.namespace"]
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto_identify_namespace_packages
    config = Config(
        src_paths=[Path("/project/src")],
        auto_identify_namespace_packages=True
    )
    assert module("project.namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #5
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("sys") == sections.STANDARD_LIBRARY

    # Test forced separate
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

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("project_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project.nested"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("project.nested.deep", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #6
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
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").touch()

        config = Config(src_paths=[src_path])
        assert module("my_project", config) == sections.FIRSTPARTY
        assert module("my_project.submodule", config) == sections.FIRSTPARTY

    # Test namespace packages
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "namespace_pkg"
        src_path.mkdir()
        (src_path / "subpkg").mkdir()

        config = Config(
            src_paths=[src_path],
            namespace_packages=["namespace_pkg"],
        )
        assert module("namespace_pkg", config) == sections.FIRSTPARTY
        assert module("namespace_pkg.subpkg", config) == sections.FIRSTPARTY

    # Test default section fallback
    assert module("unknown_module") == DEFAULT_CONFIG.default_section


# LLM-generated content at query #7
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
    assert module("test_another", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"
    assert module("my_package.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #8
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"
    assert module("django") == "THIRDPARTY"
    assert module("requests") == "THIRDPARTY"

    # Test forced separate
    config = Config(forced_separate=["custom"])
    assert module("custom_module", config) == "custom"
    assert module("custom_sub.module", config) == "custom"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_sub.module", config) == "TESTS"

    # Test src paths
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"
    assert module("src.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True
    )
    assert module("auto_namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #9
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["my_namespace"],
        auto_identify_namespace_packages=True
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"


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
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.sub_module", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("some_third_party_module", config) == "THIRDPARTY"


# LLM-generated content at query #11
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
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section with custom config
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #12
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
    assert module(".local_module") == LOCAL
    assert module(".sub.local_module") == LOCAL

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
        namespace_packages=["namespace_package"],
    )
    assert module("namespace_package", config) == sections.FIRSTPARTY
    assert module("namespace_package.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #13
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
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "myproject"
        src_path.mkdir()
        (src_path / "module.py").write_text("# test")

        config = Config(src_paths=[src_path])
        assert module("myproject", config) == sections.FIRSTPARTY


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
    config = Config(src_paths=[Path("/project/src")])
    with patch("builtins.exists_case_sensitive", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(src_paths=[Path("/project/src")], namespace_packages=["my_namespace"])
    with patch("builtins.exists_case_sensitive", return_value=True):
        assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test non-existent module falls back to default
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #15
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
        src_path = Path(tmpdir) / "namespace_pkg"
        src_path.mkdir()
        (src_path / "submodule.py").touch()

        config = Config(src_paths=[src_path], namespace_packages=["namespace_pkg"])
        assert module("namespace_pkg.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #16
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
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project.nested"],
        auto_identify_namespace_packages=True
    )
    assert module("project.nested", config) == sections.FIRSTPARTY

    # Test non-existent module
    assert module("nonexistent_module") == "THIRDPARTY"


# LLM-generated content at query #17
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

    # Test src path
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


# LLM-generated content at query #18
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
    assert module("test_utils", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"]
    )
    assert module("namespace.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #19
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
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("src_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["namespace"],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"")):
                assert module("namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #20
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
    assert module("test_example.sub", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.sub_module", config) == "FIRSTPARTY"

    # Test auto identify namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("auto_namespace.sub_module", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


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
    assert module(".local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile(r"^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "my_project"
        src_path.mkdir()
        (src_path / "module.py").write_text("# test")
        config = Config(src_paths=[src_path])
        assert module("my_project", config) == "FIRSTPARTY"


# LLM-generated content at query #22
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

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
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test auto_identify_namespace_packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        auto_identify_namespace_packages=True,
    )
    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("isort.utils.exists_case_sensitive", return_value=True):
            with patch("isort.utils._is_namespace_package", return_value=True):
                assert module("my_namespace.submodule", config) == sections.FIRSTPARTY

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("unknown_module", config) == "THIRDPARTY"


# LLM-generated content at query #23
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

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^django"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("mymodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #24
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
        assert module("src_module", config) == sections.FIRSTPARTY

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["namespace"])
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("namespace.module", config) == sections.FIRSTPARTY


# LLM-generated content at query #25
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == sections.STANDARD_LIBRARY
    assert module("sys") == sections.STANDARD_LIBRARY

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


# LLM-generated content at query #26
#--------------------------

```python
def test_module():
    config = Config()
    assert module("os") == sections.STDLIB
    assert module("django") == sections.THIRDPARTY
    assert module(".") == LOCAL
    assert module("local_module") == sections.FIRSTPARTY
    assert module("custom_module") == config.default_section


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
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_first_party=["mycompany"])
    assert module("mycompany.utils") == "FIRSTPARTY"
    assert module("mycompany.core") == "FIRSTPARTY"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    assert module("project", config) == "FIRSTPARTY"
    assert module("project.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["project"],
    )
    assert module("project.subpackage", config) == "FIRSTPARTY"

    # Test default section fallback
    assert module("unknown_module") == "THIRDPARTY"


# LLM-generated content at query #28
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
    config = Config(known_patterns=[(re.compile(r"^test.*"), "TESTS")])
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
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".sub.local_module") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_module", config) == "TESTS"
    assert module("test_package.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/project")])
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("project_module", config) == "FIRSTPARTY"
        assert module("project_module.submodule", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/project")],
        namespace_packages=["project_module"],
    )
    with patch("isort.utils.exists_case_sensitive", return_value=True):
        assert module("project_module.submodule", config) == "FIRSTPARTY"


# LLM-generated content at query #30
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
    config = Config(known_patterns=[(re.compile("django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"]
    )
    assert module("my_namespace.sub_module", config) == "FIRSTPARTY"


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
    assert module("django.apps", config) == "django"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"
    assert module(".local.submodule") == "LOCALFOLDER"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"
    assert module("test_example.submodule", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/project/src")])
    with patch("isort.place.module.exists_case_sensitive", return_value=True):
        with patch("isort.place.module.Path.is_dir", return_value=True):
            assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace packages
    config = Config(
        src_paths=[Path("/project/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module.exists_case_sensitive", return_value=True):
        with patch("isort.place.module.Path.is_dir", return_value=True):
            assert module("my_namespace.submodule", config) == "FIRSTPARTY"

    # Test default section fallback
    config = Config(default_section="THIRDPARTY")
    assert module("external_library", config) == "THIRDPARTY"


# LLM-generated content at query #32
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"
    assert module("sys") == "STDLIB"

    # Test forced_separate
    config = Config(forced_separate=["custom"])
    assert module("custom_module", config) == "custom"

    # Test local folder
    assert module(".local_module") == LOCAL

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_.*"), "TESTS")])
    assert module("test_example", config) == "TESTS"

    # Test src_path
    config = Config(src_paths=[Path("/path/to/src")])
    with patch("isort.place.module._is_module", return_value=True):
        assert module("my_module", config) == sections.FIRSTPARTY

    # Test namespace packages
    config = Config(
        src_paths=[Path("/path/to/src")],
        namespace_packages=["my_namespace"],
    )
    with patch("isort.place.module._is_namespace_package", return_value=True):
        assert module("my_namespace.submodule", config) == sections.FIRSTPARTY


# LLM-generated content at query #33
#--------------------------

```python
def test_module():
    # Test default section
    assert module("os") == "STDLIB"

    # Test local folder
    assert module(".local_module") == "LOCALFOLDER"

    # Test forced separate
    config = Config(forced_separate=["django"])
    assert module("django", config) == "django"

    # Test known patterns
    config = Config(known_patterns=[(re.compile("^test_"), "TESTS")])
    assert module("test_module", config) == "TESTS"

    # Test src path
    config = Config(src_paths=[Path("/path/to/src")])
    assert module("my_module", config) == "FIRSTPARTY"

    # Test namespace package
    config = Config(src_paths=[Path("/path/to/src")], namespace_packages=["my_namespace"])
    assert module("my_namespace.sub_module", config) == "FIRSTPARTY"

    # Test non-existent module
    assert module("non_existent_module") == "THIRDPARTY"


# LLM-generated content at query #34
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
    config = Config(known_patterns=[(re.compile("django.*"), "DJANGO")])
    assert module("django.contrib", config) == "DJANGO"

    # Test src_path
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "mypackage"
        src_path.mkdir()
        (src_path / "__init__.py").touch()
        config = Config(src_paths=[Path(tmpdir)])
        assert module("mypackage", config) == sections.FIRSTPARTY


# LLM-generated content at query #35
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


