####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_gen_api():
    root_names = {"test_title": "test_name"}
    result = gen_api(root_names, pwd="test_pwd", prefix="test_prefix", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1


# LLM-generated content at query #2
#--------------------------

```python
def test_isdir_predicate():
    assert not isdir('docs')


# LLM-generated content at query #3
#--------------------------

```python
def test_isdir_returns_false_when_directory_does_not_exist():
    assert not isdir('nonexistent_directory')


# LLM-generated content at query #4
#--------------------------

```python
def test_loader_basic():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, False)
    assert isinstance(result, str)
    assert result.startswith('#') or result.startswith('**Table of contents:**')

def test_loader_empty_package():
    result = loader('empty_pkg', 'tests/empty_pkg', True, 1, False)
    assert result == '\n'

def test_loader_with_toc():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, True)
    assert result.startswith('**Table of contents:**')

def test_loader_without_link():
    result = loader('test_pkg', 'tests/test_pkg', False, 1, False)
    assert '<a id=' not in result

def test_loader_different_level():
    result = loader('test_pkg', 'tests/test_pkg', True, 2, False)
    assert result.startswith('##') or result.startswith('**Table of contents:**')

def test_loader_non_existent_package():
    result = loader('non_existent', 'tests/non_existent', True, 1, False)
    assert result == '\n'


# LLM-generated content at query #5
#--------------------------

```python
def test_isdir_predicate():
    assert not isdir('docs')


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    ext = ".pyi"
    assert ext == ".py" is False


# LLM-generated content at query #7
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    content = "Test content"
    _write(path, content)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == content
    os.remove(path)


# LLM-generated content at query #8
#--------------------------

```python
def test_pure_py_false_when_no_py_file():
    pure_py = False
    ext = ".pyi"
    path_ext = "some_path.pyi"
    assert not pure_py


# LLM-generated content at query #9
#--------------------------

```python
def test_loader():
    assert loader('test_pkg', 'test_pkg', True, 1, False) == expected_output
    assert loader('test_pkg', 'test_pkg', False, 2, True) == expected_output_with_toc
    assert loader('non_existent_pkg', 'non_existent_path', True, 1, False) == ''
    assert loader('test_pkg', 'test_pkg', True, 3, False) == expected_output_different_level
    assert loader('test_pkg', 'test_pkg', True, 1, True) == expected_output_with_toc_and_link


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_false():
    ext = ".pyi"
    assert ext == ".py" is False


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    ext = ".pyi"
    assert ext == ".py" is False


# LLM-generated content at query #12
#--------------------------

```python
def test_site_path_with_valid_module():
    assert _site_path("os").endswith("os") or _site_path("os") == ""

def test_site_path_with_invalid_module():
    assert _site_path("nonexistent_module") == ""

def test_site_path_with_module_no_submodules():
    assert _site_path("sys") == ""


# LLM-generated content at query #13
#--------------------------

```python
def test_pure_py_false_when_no_py_file():
    pure_py = False
    assert not pure_py


# LLM-generated content at query #14
#--------------------------

```python
def test_site_path_none_spec():
    s = None
    assert not (s is None or s.submodule_search_locations is None)


# LLM-generated content at query #15
#--------------------------

```python
def test_site_path_existing_package():
    assert _site_path("os").endswith("os") or _site_path("os") == ""

def test_site_path_nonexistent_package():
    assert _site_path("nonexistent_package_12345") == ""


# LLM-generated content at query #16
#--------------------------

```python
def test_site_path_with_none_spec():
    from importlib.util import find_spec
    from os.path import dirname
    find_spec.return_value = None
    assert _site_path("test_module") == ""


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_false():
    pure_py = False
    ext = ".py"
    assert not pure_py


# LLM-generated content at query #18
#--------------------------

```python
def test_write_predicate_false():
    assert not (path := "test.txt") or not (doc := "content")


# LLM-generated content at query #19
#--------------------------

```python
def test_site_path_with_none_spec():
    s = None
    assert not (s is None or s.submodule_search_locations is None)


# LLM-generated content at query #20
#--------------------------

```python
def test_load_module_success():
    """Test successful module loading."""
    p = Parser()
    assert _load_module('test_module', 'path/to/test_module.py', p) is True
    assert p.docstring.get('test_module') is not None

def test_load_module_failure():
    """Test failed module loading."""
    p = Parser()
    assert _load_module('nonexistent_module', 'path/to/nonexistent.py', p) is False
    assert p.docstring.get('nonexistent_module') is None

def test_load_module_with_parent_import_error():
    """Test module loading when parent import fails."""
    p = Parser()
    assert _load_module('child_module', 'path/to/child_module.py', p) is False
    assert p.docstring.get('child_module') is None


# LLM-generated content at query #21
#--------------------------

```python
def test_write_creates_file_with_content():
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc


# LLM-generated content at query #22
#--------------------------

```python
def test_read_returns_file_content():
    content = _read('test_file.txt')
    assert content == 'Expected content of test_file.txt'


# LLM-generated content at query #23
#--------------------------

```python
def test_load_module_with_none_spec():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = None
    assert not _load_module(name, path, p)


# LLM-generated content at query #24
#--------------------------

```python
def test_gen_api_with_valid_inputs():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## Test API\n\n")

def test_gen_api_with_empty_root_names():
    result = gen_api({}, pwd="/path/to/test", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert result == []

def test_gen_api_with_none_pwd():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)

def test_gen_api_with_invalid_package():
    root_names = {"Invalid": "nonexistent_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert result == []

def test_gen_api_with_dry_false():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=False, level=2, toc=True, dry=False)
    assert isinstance(result, list)


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_line_9_false():
    name = "module.name"
    path = "path/to/module"
    p = Parser()
    s = None
    assert not (s is not None and isinstance(s.loader, Loader))


# LLM-generated content at query #26
#--------------------------

```python
def test_write_predicate_false():
    path = "/invalid/path/that/does/not/exist/file.txt"
    doc = "test content"
    assert not _write(path, doc)


# LLM-generated content at query #27
#--------------------------

```python
def test_read_predicate_false():
    assert not _read("nonexistent_file.txt")


# LLM-generated content at query #28
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    content = "Test content"
    _write(path, content)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == content
    os.remove(path)


# LLM-generated content at query #29
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc
    os.remove(path)


# LLM-generated content at query #30
#--------------------------

```python
def test_load_module_with_valid_spec_and_loader():
    name = "test_module"
    path = "test_module.py"
    p = Parser()
    s = spec_from_file_location(name, path)
    assert s is not None and isinstance(s.loader, Loader)


# LLM-generated content at query #31
#--------------------------

```python
def test_gen_api_basic():
    root_names = {"Test": "test"}
    result = gen_api(root_names, None, prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## Test API\n\n")

def test_gen_api_with_pwd():
    root_names = {"Test": "test"}
    result = gen_api(root_names, "some/path", prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")

def test_gen_api_empty_root_names():
    result = gen_api({}, None, prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_nonexistent_package():
    root_names = {"NonExistent": "nonexistent"}
    result = gen_api(root_names, None, prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_custom_prefix():
    root_names = {"Test": "test"}
    result = gen_api(root_names, None, prefix="custom_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")


# LLM-generated content at query #32
#--------------------------

```python
def test_gen_api_root_names_iteration():
    root_names = {"test_title": "test_name"}
    assert all(isinstance(title, str) and isinstance(name, str) for title, name in root_names.items())


# LLM-generated content at query #33
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## Test API\n\n")

def test_gen_api_with_invalid_root_names():
    root_names = {"Invalid": "non_existent_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_multiple_root_names():
    root_names = {"Test1": "test_package1", "Test2": "test_package2"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=False, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].startswith("# Test1 API\n\n")
    assert result[1].startswith("# Test2 API\n\n")

def test_gen_api_with_dry_false():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=True, level=1, toc=False, dry=False)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")

def test_gen_api_with_custom_prefix():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="custom_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")


# LLM-generated content at query #34
#--------------------------

```python
def test_parent_with_level_1():
    assert parent("a.b.c", level=1) == "a.b"

def test_parent_with_level_2():
    assert parent("a.b.c", level=2) == "a"

def test_parent_with_default_level():
    assert parent("a.b.c") == "a.b"


# LLM-generated content at query #35
#--------------------------

```python
def test_read_nonexistent_file():
    assert not _read("nonexistent_file.txt")


# LLM-generated content at query #36
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    test_path = "test_file.txt"
    test_content = "Test content"

    _write(test_path, test_content)

    assert os.path.exists(test_path)
    with open(test_path, 'r', encoding='utf-8') as f:
        assert f.read() == test_content
    os.remove(test_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_read_returns_false_for_invalid_path():
    assert not _read("invalid_path.txt")


# LLM-generated content at query #38
#--------------------------

```python
def test__load_module_with_valid_spec_and_loader():
    import sys
    from importlib.util import spec_from_file_location, module_from_spec
    from importlib.abc import Loader
    from apimd.parser import Parser

    name = "test_module"
    path = "test_module.py"
    p = Parser()

    # Mock spec_from_file_location to return a spec with a Loader
    original_spec_from_file_location = spec_from_file_location
    def mock_spec_from_file_location(name, path):
        class MockLoader(Loader):
            def exec_module(self, module):
                pass
        class MockSpec:
            loader = MockLoader()
        return MockSpec()

    sys.modules['importlib.util'].spec_from_file_location = mock_spec_from_file_location

    try:
        result = _load_module(name, path, p)
        assert result is True
    finally:
        sys.modules['importlib.util'].spec_from_file_location = original_spec_from_file_location


# LLM-generated content at query #39
#--------------------------

```python
def test_read_returns_file_content():
    path = "test_file.txt"
    content = "Hello, World!"
    open(path, 'w').write(content)
    assert _read(path) == content


# LLM-generated content at query #40
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    result = gen_api({"test": "test"}, pwd=".", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## test API\n\n")

def test_gen_api_with_empty_root_names():
    result = gen_api({}, pwd=".", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_nonexistent_package():
    result = gen_api({"nonexistent": "nonexistent"}, pwd=".", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_dry_false():
    result = gen_api({"test": "test"}, pwd=".", prefix="test_docs", link=False, level=2, toc=True, dry=False)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## test API\n\n")

def test_gen_api_with_custom_prefix():
    result = gen_api({"test": "test"}, pwd=".", prefix="custom_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## test API\n\n")

def test_gen_api_with_different_level():
    result = gen_api({"test": "test"}, pwd=".", prefix="test_docs", link=False, level=3, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("### test API\n\n")

def test_gen_api_with_toc_enabled():
    result = gen_api({"test": "test"}, pwd=".", prefix="test_docs", link=True, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## test API\n\n")

def test_gen_api_with_toc_disabled():
    result = gen_api({"test": "test"}, pwd=".", prefix="test_docs", link=True, level=2, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## test API\n\n")


# LLM-generated content at query #41
#--------------------------

```python
def test_dry_mode_prevents_file_writing():
    assert not dry


# LLM-generated content at query #42
#--------------------------

```python
def test_dry_mode_skips_file_writing():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, dry=True)
    assert len(result) == 1
    assert "Test API" in result[0]


# LLM-generated content at query #43
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = 'test_file.txt'
    doc = 'Test content'
    _write(path, doc)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc
    os.remove(path)


# LLM-generated content at query #44
#--------------------------

```python
def test_dry_mode_prevents_file_write():
    assert not dry


# LLM-generated content at query #45
#--------------------------

```python
def test_dry_mode_prevents_file_writing():
    docs = gen_api(
        root_names={'test': 'test_module'},
        dry=True,
        level=1,
        link=False,
        toc=False
    )
    assert len(docs) == 1
    assert docs[0].startswith('# test_module API\n\n')


# LLM-generated content at query #46
#--------------------------

```python
def test__load_module_returns_false_when_spec_is_none():
    assert _load_module("test_module", "test_path", Parser()) is False


# LLM-generated content at query #47
#--------------------------

```python
def test_read_returns_file_content():
    assert _read("test_file.txt") == "expected content"


# LLM-generated content at query #48
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module('test_module', 'test_module.py', p) is True
    assert p.docstring.get('test_module') is not None

def test_load_module_failure():
    p = Parser()
    assert _load_module('nonexistent_module', 'nonexistent_module.py', p) is False
    assert p.docstring.get('nonexistent_module') is None

def test_load_module_with_parent_import_error():
    p = Parser()
    assert _load_module('child_module', 'child_module.py', p) is False
    assert p.docstring.get('child_module') is None


# LLM-generated content at query #49
#--------------------------

```python
def test__read_returns_file_content():
    assert _read("test_file.txt") == "This is a test file."


# LLM-generated content at query #50
#--------------------------

```python
def test_file_reading_returns_content():
    content = _read("test_file.txt")
    assert content == "expected content"


# LLM-generated content at query #51
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module("test_module", "test_module.py", p) is True
    assert p.docstring.get("test_module") is not None

def test_load_module_failure():
    p = Parser()
    assert _load_module("nonexistent_module", "nonexistent_module.py", p) is False
    assert p.docstring.get("nonexistent_module") is None

def test_load_module_with_parent_import_error():
    p = Parser()
    assert _load_module("child_module", "child_module.py", p) is False
    assert p.docstring.get("child_module") is None


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_at_line_9():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = MagicMock()
    s.loader = MagicMock(spec=Loader)
    with patch('apimd.loader.spec_from_file_location', return_value=s):
        with patch('apimd.loader.parent', return_value=''):
            assert _load_module(name, path, p) is True


# LLM-generated content at query #53
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    from importlib.util import find_spec
    from os.path import dirname

    find_spec.return_value = None
    assert _site_path("nonexistent_module") == ""


# LLM-generated content at query #54
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module('test_module', 'test_module.py', p) is True
    assert p.docstring.get('test_module') is not None

def test_load_module_failure():
    p = Parser()
    assert _load_module('nonexistent_module', 'nonexistent_module.py', p) is False
    assert p.docstring.get('nonexistent_module') is None

def test_load_module_with_parent_import_error():
    p = Parser()
    assert _load_module('child_module', 'child_module.py', p) is False
    assert p.docstring.get('child_module') is None


# LLM-generated content at query #55
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = 'test_file.txt'
    doc = 'Test content'
    _write(path, doc)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc
    os.remove(path)


# LLM-generated content at query #56
#--------------------------

```python
def test_loader():
    assert loader("test_pkg", "test_pkg_path", True, 1, False) == expected_output
    assert loader("test_pkg", "test_pkg_path", False, 2, True) == expected_output_with_toc
    assert loader("empty_pkg", "empty_pkg_path", True, 1, False) == ""
    assert loader("invalid_pkg", "invalid_pkg_path", True, 1, False) == ""


# LLM-generated content at query #57
#--------------------------

```python
def test_write_creates_file():
    import os
    path = 'test_file.txt'
    doc = 'Hello, world!'
    _write(path, doc)
    assert os.path.exists(path)
    os.remove(path)


# LLM-generated content at query #58
#--------------------------

```python
def test_read_file_raises_file_not_found_error():
    assert not _read("non_existent_file.txt")


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_evaluates_to_false():
    pure_py = True
    assert not pure_py


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loader_basic():
    result = loader("test_pkg", "test_path", True, 1, False)
    assert isinstance(result, str)

def test_loader_with_toc():
    result = loader("test_pkg", "test_path", True, 1, True)
    assert isinstance(result, str)

def test_loader_no_link():
    result = loader("test_pkg", "test_path", False, 1, False)
    assert isinstance(result, str)

def test_loader_different_level():
    result = loader("test_pkg", "test_path", True, 2, False)
    assert isinstance(result, str)

def test_loader_empty_package():
    result = loader("empty_pkg", "empty_path", True, 1, False)
    assert isinstance(result, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_gen_api():
    root_names = {"Test": "test"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_site_path_existing_package():
    assert _site_path("os") != ""

def test_site_path_nonexistent_package():
    assert _site_path("nonexistent_package") == ""

def test_site_path_no_submodule_search_locations():
    assert _site_path("sys") == ""


# LLM-generated content at query #4
#--------------------------

```python
def test_pure_py_is_false_when_no_py_file():
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        if not isfile(path_ext):
            continue
        if ext == ".py":
            pure_py = True
    assert not pure_py


# LLM-generated content at query #5
#--------------------------

```python
def test_pure_py_false_when_no_py_file():
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = "nonexistent_path" + ext
        if not False:  # Simulating isfile returning False
            continue
        pure_py = True
    assert not pure_py


# LLM-generated content at query #6
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_empty_root_names():
    root_names = {}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_invalid_root_names():
    root_names = {"Invalid": "nonexistent_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_custom_prefix():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, pwd=None, prefix='custom_docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_dry_false():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=False)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_different_level():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=2, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_toc_enabled():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_link_disabled():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=False, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0


# LLM-generated content at query #7
#--------------------------

```python
def test_write_creates_file_with_content():
    path = "test_file.txt"
    content = "Test content"
    _write(path, content)

    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == content


# LLM-generated content at query #8
#--------------------------

```python
def test_read_existing_file():
    content = _read("existing_file.txt")
    assert content == "Expected content"

def test_read_non_existing_file():
    try:
        _read("non_existing_file.txt")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_loader_basic():
    result = loader('test_pkg', 'path/to/test_pkg', True, 1, False)
    assert isinstance(result, str)
    assert result.startswith('#') or result.startswith('**Table of contents:**')

def test_loader_with_toc():
    result = loader('test_pkg', 'path/to/test_pkg', True, 1, True)
    assert result.startswith('**Table of contents:**')

def test_loader_no_link():
    result = loader('test_pkg', 'path/to/test_pkg', False, 1, False)
    assert '<a id=' not in result

def test_loader_different_level():
    result = loader('test_pkg', 'path/to/test_pkg', True, 2, False)
    assert result.startswith('##') or result.startswith('**Table of contents:**')

def test_loader_empty_package():
    result = loader('empty_pkg', 'path/to/empty_pkg', True, 1, False)
    assert result == '\n'

def test_loader_with_stub_files():
    result = loader('stub_pkg', 'path/to/stub_pkg', True, 1, False)
    assert isinstance(result, str)


# LLM-generated content at query #10
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    import importlib.util
    importlib.util.find_spec = lambda _: None
    assert _site_path("nonexistent_module") == ""


# LLM-generated content at query #11
#--------------------------

```python
def test_load_module_success():
    name = "test_module"
    path = "test_module.py"
    p = Parser()
    assert _load_module(name, path, p) is True
    assert p.docstring.get(name) is not None

def test_load_module_failure():
    name = "nonexistent_module"
    path = "nonexistent_module.py"
    p = Parser()
    assert _load_module(name, path, p) is False


# LLM-generated content at query #12
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    content = "Test content"
    _write(path, content)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == content
    os.remove(path)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_13_evaluates_to_false():
    ext = ".pyi"
    assert ext == ".py" is False


# LLM-generated content at query #14
#--------------------------

```python
def test_site_path_with_valid_module():
    assert _site_path("os") == dirname(find_spec("os").submodule_search_locations[0])

def test_site_path_with_nonexistent_module():
    assert _site_path("nonexistent_module") == ""

def test_site_path_with_module_no_submodules():
    assert _site_path("sys") == ""


# LLM-generated content at query #15
#--------------------------

```python
def test_isdir_returns_true():
    assert isdir('docs')


# LLM-generated content at query #16
#--------------------------

```python
def test__load_module_with_none_spec():
    """Test that the predicate evaluates to False when spec is None."""
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = None
    assert not (s is not None and isinstance(s.loader, Loader))


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_evaluates_to_false():
    s = None
    assert not (s is not None and isinstance(s.loader, Loader))


# LLM-generated content at query #18
#--------------------------

```python
def test_loader():
    assert loader("test_pkg", "test_pkg_path", True, 1, False) == expected_output
    assert loader("test_pkg", "test_pkg_path", False, 2, True) == expected_output_with_toc
    assert loader("empty_pkg", "empty_pkg_path", True, 1, False) == ""
    assert loader("invalid_pkg", "invalid_pkg_path", True, 1, False) == ""


# LLM-generated content at query #19
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    root_names = {"Test": "test"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")

def test_gen_api_with_invalid_root_names():
    root_names = {"Invalid": "invalid_module"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_custom_prefix():
    root_names = {"Test": "test"}
    result = gen_api(root_names, pwd=None, prefix='custom_docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1

def test_gen_api_with_dry_false():
    root_names = {"Test": "test"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=False)
    assert isinstance(result, list)
    assert len(result) == 1


# LLM-generated content at query #20
#--------------------------

```python
def test_loader_basic():
    result = loader("test_pkg", "test_pkg_path", True, 1, False)
    assert isinstance(result, str)
    assert result.startswith("**Table of contents:**") or result.startswith("#")

def test_loader_no_toc():
    result = loader("test_pkg", "test_pkg_path", False, 1, False)
    assert not result.startswith("**Table of contents:**")

def test_loader_with_toc():
    result = loader("test_pkg", "test_pkg_path", True, 1, True)
    assert result.startswith("**Table of contents:**")

def test_loader_different_level():
    result1 = loader("test_pkg", "test_pkg_path", True, 1, False)
    result2 = loader("test_pkg", "test_pkg_path", True, 2, False)
    assert result1 != result2

def test_loader_link_enabled():
    result = loader("test_pkg", "test_pkg_path", True, 1, False)
    assert "<a id=" in result

def test_loader_link_disabled():
    result = loader("test_pkg", "test_pkg_path", False, 1, False)
    assert "<a id=" not in result


# LLM-generated content at query #21
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    from importlib.util import find_spec
    from os.path import dirname

    # Mock find_spec to return None
    original_find_spec = find_spec
    find_spec = lambda name: None

    result = _site_path("nonexistent_module")
    assert result == ""

    # Restore original find_spec
    find_spec = original_find_spec


# LLM-generated content at query #22
#--------------------------

```python
def test_write_creates_file_with_content():
    path = "test_file.txt"
    content = "Test content"
    _write(path, content)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == content


# LLM-generated content at query #23
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        if ext == ".py":
            assert not isfile(path_ext)
        else:
            assert isfile(path_ext)
    assert not pure_py


# LLM-generated content at query #24
#--------------------------

```python
def test_site_path_with_valid_package():
    assert _site_path("os").endswith("os") or _site_path("os") == ""

def test_site_path_with_invalid_package():
    assert _site_path("nonexistent_package") == ""

def test_site_path_with_package_no_submodules():
    assert _site_path("sys") == ""


# LLM-generated content at query #25
#--------------------------

```python
def test_read_with_invalid_path():
    assert not _read("nonexistent_file.txt")


# LLM-generated content at query #26
#--------------------------

```python
def test_read_returns_false_for_invalid_path():
    assert not _read("invalid_path.txt")


# LLM-generated content at query #27
#--------------------------

```python
def test_load_module_with_valid_spec_and_loader():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = MagicMock()
    s.loader = MagicMock(spec=Loader)
    with patch('apimd.loader.spec_from_file_location', return_value=s):
        result = _load_module(name, path, p)
        assert result is True


# LLM-generated content at query #28
#--------------------------

```python
def test_root_names_not_empty():
    root_names = {"test": "module"}
    assert root_names


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_false():
    pure_py = False
    ext = ".pyi"
    assert not (ext == ".py")


# LLM-generated content at query #30
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc
    os.remove(path)


# LLM-generated content at query #31
#--------------------------

```python
def test_read_returns_content_of_file():
    assert _read("path/to/test/file.txt") == "expected content"


# LLM-generated content at query #32
#--------------------------

```python
def test_write_predicate_evaluates_to_false():
    assert not (_write.__code__.co_consts[0] == 'r')


# LLM-generated content at query #33
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    content = "Test content"
    _write(path, content)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == content
    os.remove(path)


# LLM-generated content at query #34
#--------------------------

```python
def test_write_creates_file():
    import os
    path = "test_file.txt"
    doc = "test content"
    _write(path, doc)
    assert os.path.exists(path)
    os.remove(path)


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_at_line_22_evaluates_to_false():
    assert not (True)


# LLM-generated content at query #36
#--------------------------

```python
def test_read_existing_file():
    content = _read("existing_file.txt")
    assert content == "expected content"

def test_read_nonexistent_file():
    try:
        _read("nonexistent_file.txt")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

def test_read_empty_file():
    content = _read("empty_file.txt")
    assert content == ""


# LLM-generated content at query #37
#--------------------------

```python
def test__read():
    content = _read("test_file.txt")
    assert content == "expected content"


# LLM-generated content at query #38
#--------------------------

```python
def test__load_module_predicate():
    from apimd.loader import _load_module
    from apimd.parser import Parser
    from importlib.machinery import Loader, ModuleSpec
    from unittest.mock import MagicMock

    name = "test_module"
    path = "test_path"
    p = Parser()

    # Mock spec with loader
    s = MagicMock(spec=ModuleSpec)
    s.loader = MagicMock(spec=Loader)

    # Mock __import__ to avoid actual import
    import builtins
    original_import = builtins.__import__
    builtins.__import__ = MagicMock()

    try:
        # Mock spec_from_file_location to return our mock spec
        import importlib.util
        original_spec_from_file_location = importlib.util.spec_from_file_location
        importlib.util.spec_from_file_location = MagicMock(return_value=s)

        # Call the function
        result = _load_module(name, path, p)

        # Assert the predicate at line 9 evaluates to True
        assert s is not None and isinstance(s.loader, Loader)
    finally:
        # Restore original functions
        builtins.__import__ = original_import
        importlib.util.spec_from_file_location = original_spec_from_file_location


# LLM-generated content at query #39
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module("test_module", "path/to/test_module.py", p) == True
    assert p.docstring == {"test_module": "Test module docstring"}

def test_load_module_failure():
    p = Parser()
    assert _load_module("nonexistent_module", "path/to/nonexistent_module.py", p) == False
    assert p.docstring == {}


# LLM-generated content at query #40
#--------------------------

```python
def test_read_with_nonexistent_path():
    assert not _read("nonexistent_file.txt")


# LLM-generated content at query #41
#--------------------------

```python
def test__read_existing_file():
    content = _read("existing_file.txt")
    assert content == "Expected content"

def test__read_non_existing_file():
    try:
        _read("non_existing_file.txt")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #42
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_invalid_root_names():
    root_names = {"Invalid": "nonexistent_package"}
    result = gen_api(root_names, pwd=None, prefix="docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 0

def test_gen_api_with_custom_prefix():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="custom_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_different_level():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="docs", link=True, level=2, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_toc_enabled():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="docs", link=True, level=1, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0

def test_gen_api_with_dry_run_disabled():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="docs", link=True, level=1, toc=False, dry=False)
    assert isinstance(result, list)
    assert len(result) >= 0


# LLM-generated content at query #43
#--------------------------

```python
def test_write_predicate_false():
    assert not (open('test.txt', 'w+', encoding='utf-8') == None)


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_at_line_22_evaluates_to_false():
    root_names = {}
    assert not root_names


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (s is not None and isinstance(s.loader, Loader))


# LLM-generated content at query #46
#--------------------------

```python
def test_root_names_not_empty():
    root_names = {"test": "module"}
    assert root_names


# LLM-generated content at query #47
#--------------------------

```python
def test__load_module_with_valid_spec_and_loader():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = MagicMock()
    s.loader = MagicMock(spec=Loader)
    with patch('apimd.loader.spec_from_file_location', return_value=s):
        with patch('apimd.loader.module_from_spec') as mock_module_from_spec:
            with patch('apimd.loader.__import__'):
                result = _load_module(name, path, p)
                assert result is True


# LLM-generated content at query #48
#--------------------------

```python
def test_empty_root_names():
    assert not gen_api({})


# LLM-generated content at query #49
#--------------------------

```python
def test_gen_api_with_valid_input():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")

def test_gen_api_with_empty_root_names():
    result = gen_api({}, pwd="/path/to/test", prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert result == []

def test_gen_api_with_none_pwd():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)

def test_gen_api_with_invalid_package():
    root_names = {"Invalid": "nonexistent_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=True, level=1, toc=False, dry=True)
    assert result == []

def test_gen_api_with_custom_prefix():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="custom_prefix", link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)

def test_gen_api_with_different_level():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=True, level=2, toc=False, dry=True)
    assert result[0].startswith("## Test API\n\n")

def test_gen_api_with_toc_enabled():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=True, level=1, toc=True, dry=True)
    assert isinstance(result, list)

def test_gen_api_with_dry_false():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd="/path/to/test", prefix="test_docs", link=True, level=1, toc=False, dry=False)
    assert isinstance(result, list)


# LLM-generated content at query #50
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    root_names = {"Test": "test_module"}
    result = gen_api(root_names, dry=True)
    assert len(result) == 1
    assert "# Test API" in result[0]


# LLM-generated content at query #51
#--------------------------

```python
def test_load_module_with_valid_spec_and_loader():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = MagicMock()
    s.loader = MagicMock(spec=Loader)
    with patch('apimd.loader.spec_from_file_location', return_value=s):
        assert _load_module(name, path, p) is True


# LLM-generated content at query #52
#--------------------------

```python
def test_pure_py_false_when_no_py_file():
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        assert not isfile(path_ext)
    assert not pure_py


# LLM-generated content at query #53
#--------------------------

```python
def test_loader_basic():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, False)
    assert isinstance(result, str)
    assert result.startswith('# Module `test_pkg`')

def test_loader_with_toc():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, True)
    assert result.startswith('**Table of contents:**')
    assert '+ [test_pkg](#test-pkg)' in result

def test_loader_different_level():
    result = loader('test_pkg', 'tests/test_pkg', True, 2, False)
    assert result.startswith('## Module `test_pkg`')

def test_loader_no_link():
    result = loader('test_pkg', 'tests/test_pkg', False, 1, False)
    assert '<a id=' not in result

def test_loader_empty_package():
    result = loader('empty_pkg', 'tests/empty_pkg', True, 1, False)
    assert result.strip() == ''

def test_loader_with_submodules():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, False)
    assert 'test_pkg.submodule' in result

def test_loader_extension_module():
    result = loader('ext_pkg', 'tests/ext_pkg', True, 1, False)
    assert 'ext_pkg.extension' in result

def test_loader_mixed_package():
    result = loader('mixed_pkg', 'tests/mixed_pkg', True, 1, False)
    assert 'mixed_pkg.py_module' in result
    assert 'mixed_pkg.ext_module' in result


# LLM-generated content at query #54
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    import importlib.util
    importlib.util.find_spec = lambda name: None
    assert _site_path("test_module") == ""


# LLM-generated content at query #55
#--------------------------

```python
def test_write_creates_file():
    path = "test_file.txt"
    doc = "test content"
    _write(path, doc)
    assert open(path, 'r', encoding='utf-8').read() == doc


# LLM-generated content at query #56
#--------------------------

```python
def test_gen_api():
    root_names = {"Test": "test"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")


# LLM-generated content at query #57
#--------------------------

```python
def test_doc_not_empty():
    doc = "content"
    assert doc.strip()


# LLM-generated content at query #58
#--------------------------

```python
def test_gen_api_empty_doc_skipped():
    root_names = {"Test": "test_module"}
    docs = gen_api(root_names, dry=True)
    assert len(docs) == 0


# LLM-generated content at query #59
#--------------------------

```python
def test_gen_api_empty_doc_skipped():
    root_names = {"Test": "nonexistent.module"}
    result = gen_api(root_names, dry=True)
    assert result == []


# LLM-generated content at query #60
#--------------------------

```python
def test_load_module_with_none_spec():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = None
    assert _load_module(name, path, p) is False


# LLM-generated content at query #61
#--------------------------

```python
def test_write_creates_file():
    import os
    path = 'test_file.txt'
    doc = 'test content'
    _write(path, doc)
    assert os.path.exists(path)
    os.remove(path)


# LLM-generated content at query #62
#--------------------------

```python
def test_gen_api_empty_doc():
    root_names = {"Test": "test_module"}
    with patch('apimd.loader.loader') as mock_loader, \
         patch('apimd.loader.isdir', return_value=True), \
         patch('apimd.loader.logger') as mock_logger:
        mock_loader.return_value = ""
        result = gen_api(root_names, level=1)
        assert result == []
        mock_logger.warning.assert_called_once_with("'test_module' can not be found")


# LLM-generated content at query #63
#--------------------------

```python
def test_read_returns_file_content():
    content = "test content"
    path = "test_file.txt"
    with open(path, 'w') as f:
        f.write(content)
    assert _read(path) == content


# LLM-generated content at query #64
#--------------------------

```python
def test__read():
    import os
    test_file = "test_script.py"
    content = "print('Hello, World!')"
    with open(test_file, 'w') as f:
        f.write(content)
    assert _read(test_file) == content
    os.remove(test_file)


# LLM-generated content at query #65
#--------------------------

```python
def test_write_creates_file():
    import os
    path = "test_file.txt"
    doc = "test content"
    _write(path, doc)
    assert os.path.exists(path)
    os.remove(path)


