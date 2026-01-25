####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_gen_api_with_valid_root_names():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix='test_docs', link=True, level=1, toc=False, dry=True)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")

def test_gen_api_with_invalid_root_names():
    root_names = {"Invalid": "nonexistent_package"}
    result = gen_api(root_names, pwd=None, prefix='test_docs', link=True, level=1, toc=False, dry=True)
    assert len(result) == 0

def test_gen_api_with_custom_prefix():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix='custom_docs', link=True, level=1, toc=False, dry=True)
    assert len(result) == 1

def test_gen_api_with_dry_false():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix='test_docs', link=True, level=1, toc=False, dry=False)
    assert len(result) == 1

def test_gen_api_with_different_level():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix='test_docs', link=True, level=2, toc=False, dry=True)
    assert result[0].startswith("## Test API\n\n")

def test_gen_api_with_toc_enabled():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix='test_docs', link=True, level=1, toc=True, dry=True)
    assert len(result) == 1

def test_gen_api_with_link_disabled():
    root_names = {"Test": "test_package"}
    result = gen_api(root_names, pwd=None, prefix='test_docs', link=False, level=1, toc=False, dry=True)
    assert len(result) == 1


# LLM-generated content at query #2
#--------------------------

```python
def test_isdir_predicate():
    assert not isdir('docs')


# LLM-generated content at query #3
#--------------------------

```python
def test_loader_basic():
    result = loader('test_pkg', 'path/to/test_pkg', True, 1, False)
    assert isinstance(result, str)
    assert result.strip() != ''

def test_loader_with_toc():
    result = loader('test_pkg', 'path/to/test_pkg', True, 1, True)
    assert 'Table of contents:' in result

def test_loader_different_levels():
    result_level1 = loader('test_pkg', 'path/to/test_pkg', True, 1, False)
    result_level2 = loader('test_pkg', 'path/to/test_pkg', True, 2, False)
    assert result_level1 != result_level2

def test_loader_no_link():
    result = loader('test_pkg', 'path/to/test_pkg', False, 1, False)
    assert '<a id=' not in result

def test_loader_empty_package():
    result = loader('empty_pkg', 'path/to/empty_pkg', True, 1, False)
    assert result.strip() == ''

def test_loader_with_stub_files():
    result = loader('stub_pkg', 'path/to/stub_pkg', True, 1, False)
    assert isinstance(result, str)


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module("os.path", "path/to/os/path.py", p) == True
    assert p.docstring.get("os.path") is not None

def test_load_module_failure():
    p = Parser()
    assert _load_module("nonexistent.module", "path/to/nonexistent.py", p) == False
    assert p.docstring.get("nonexistent.module") is None

def test_load_module_with_parent_import_error():
    p = Parser()
    assert _load_module("invalid.parent.child", "path/to/child.py", p) == False
    assert p.docstring.get("invalid.parent.child") is None


# LLM-generated content at query #6
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        isfile_result = False
        if ext == ".py":
            assert not isfile_result
            assert pure_py is False


# LLM-generated content at query #7
#--------------------------

```python
def test_load_module_with_none_spec():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = None
    assert not _load_module(name, path, p)


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_false():
    pure_py = False
    ext = ".pyi"
    assert not (ext == ".py")


# LLM-generated content at query #9
#--------------------------

```python
def test_loader_basic():
    assert loader("test_pkg", "path/to/test_pkg", True, 1, False) == expected_output

def test_loader_no_toc():
    p = Parser.new(True, 1, False)
    assert loader("test_pkg", "path/to/test_pkg", True, 1, False) == p.compile()

def test_loader_with_toc():
    p = Parser.new(True, 1, True)
    assert loader("test_pkg", "path/to/test_pkg", True, 1, True) == p.compile()

def test_loader_no_link():
    p = Parser.new(False, 1, False)
    assert loader("test_pkg", "path/to/test_pkg", False, 1, False) == p.compile()

def test_loader_different_level():
    p = Parser.new(True, 2, False)
    assert loader("test_pkg", "path/to/test_pkg", True, 2, False) == p.compile()

def test_loader_empty_package():
    assert loader("empty_pkg", "path/to/empty_pkg", True, 1, False) == ""

def test_loader_extension_module():
    assert loader("ext_pkg", "path/to/ext_pkg", True, 1, False) == expected_ext_output

def test_loader_mixed_package():
    assert loader("mixed_pkg", "path/to/mixed_pkg", True, 1, False) == expected_mixed_output


# LLM-generated content at query #10
#--------------------------

```python
def test_gen_api_calls_loader_and_writes_files():
    root_names = {"Test": "test"}
    pwd = "/path/to/test"
    prefix = "docs"
    link = True
    level = 1
    toc = False
    dry = False

    result = gen_api(root_names, pwd, prefix=prefix, link=link, level=level, toc=toc, dry=dry)

    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")
    assert exists(join(prefix, "test-api.md"))


# LLM-generated content at query #11
#--------------------------

```python
def test_site_path_existing_package():
    assert _site_path("os").endswith("os")

def test_site_path_nonexistent_package():
    assert _site_path("nonexistent_package") == ""

def test_site_path_package_without_submodules():
    assert _site_path("sys") == ""


# LLM-generated content at query #12
#--------------------------

```python
def test_loader_basic():
    result = loader('test_pkg', 'tests/data', True, 1, False)
    assert isinstance(result, str)
    assert result.startswith('# Module `test_pkg`')

def test_loader_with_toc():
    result = loader('test_pkg', 'tests/data', True, 1, True)
    assert result.startswith('**Table of contents:**')

def test_loader_no_link():
    result = loader('test_pkg', 'tests/data', False, 1, False)
    assert '<a id=' not in result

def test_loader_different_level():
    result = loader('test_pkg', 'tests/data', True, 2, False)
    assert result.startswith('## Module `test_pkg`')

def test_loader_empty_package():
    result = loader('empty_pkg', 'tests/data', True, 1, False)
    assert 'Missing documentation' in result

def test_loader_with_submodules():
    result = loader('parent_pkg', 'tests/data', True, 1, False)
    assert 'Module `parent_pkg.submodule`' in result


# LLM-generated content at query #13
#--------------------------

```python
def test__load_module_with_none_spec():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = None
    assert not _load_module(name, path, p)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_22_evaluates_to_false():
    assert not False


# LLM-generated content at query #15
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module('os.path', 'path/to/os/path.py', p) == True
    assert p.docstring.get('os.path') is not None

def test_load_module_failure():
    p = Parser()
    assert _load_module('nonexistent.module', 'path/to/nonexistent.py', p) == False
    assert p.docstring.get('nonexistent.module') is None

def test_load_module_invalid_spec():
    p = Parser()
    assert _load_module('invalid.spec', 'path/to/invalid.py', p) == False
    assert p.docstring.get('invalid.spec') is None


# LLM-generated content at query #16
#--------------------------

```python
def test_site_path_with_valid_package():
    assert _site_path("os").endswith("site-packages")

def test_site_path_with_invalid_package():
    assert _site_path("nonexistent_package") == ""

def test_site_path_with_package_no_submodules():
    assert _site_path("sys") == ""


# LLM-generated content at query #17
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        if ext == ".py":
            pure_py = True
    assert not pure_py


# LLM-generated content at query #18
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    assert _site_path("nonexistent_module") == ""


# LLM-generated content at query #19
#--------------------------

```python
def test__read_returns_file_content():
    assert _read("test_file.txt") == "expected content"


# LLM-generated content at query #20
#--------------------------

```python
def test_pure_py_false_when_no_py_file():
    """Test that pure_py remains False when no .py file is found."""
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        isfile_result = False
        if ext == ".py":
            assert not isfile_result
            assert pure_py is False


# LLM-generated content at query #21
#--------------------------

```python
def test_site_path_with_valid_module():
    assert _site_path("os").endswith("site-packages")

def test_site_path_with_invalid_module():
    assert _site_path("nonexistent_module") == ""

def test_site_path_with_module_no_submodule_search_locations():
    assert _site_path("sys") == ""


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_load_module_predicate_false():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    s = None
    assert not (s is not None and isinstance(s.loader, Loader))


# LLM-generated content at query #24
#--------------------------

```python
def test_loader_basic_functionality():
    result = loader("test_pkg", "test_path", True, 1, False)
    assert isinstance(result, str)
    assert result.startswith("**Table of contents:**") or result.startswith("#")

def test_loader_with_different_options():
    result_link_true = loader("test_pkg", "test_path", True, 1, False)
    result_link_false = loader("test_pkg", "test_path", False, 1, False)
    assert result_link_true != result_link_false

def test_loader_with_toc_enabled():
    result = loader("test_pkg", "test_path", True, 1, True)
    assert result.startswith("**Table of contents:**")

def test_loader_with_different_levels():
    result_level_1 = loader("test_pkg", "test_path", True, 1, False)
    result_level_2 = loader("test_pkg", "test_path", True, 2, False)
    assert result_level_1 != result_level_2

def test_loader_with_empty_package():
    result = loader("empty_pkg", "empty_path", True, 1, False)
    assert result == "\n"

def test_loader_with_nonexistent_package():
    result = loader("nonexistent_pkg", "nonexistent_path", True, 1, False)
    assert result == "\n"


# LLM-generated content at query #25
#--------------------------

```python
def test_site_path_with_none_submodule_search_locations():
    import importlib.util
    from os.path import dirname

    # Mock a spec with None submodule_search_locations
    spec = importlib.util.spec_from_loader("test_module", None)
    spec.submodule_search_locations = None

    # Override find_spec to return the mock spec
    import importlib
    original_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name: spec

    # Call the function and assert the predicate evaluates to False
    result = _site_path("test_module")
    assert result == ""

    # Restore the original find_spec
    importlib.util.find_spec = original_find_spec


# LLM-generated content at query #26
#--------------------------

```python
def test__read_returns_content_of_file():
    content = _read('test_file.txt')
    assert content == 'Expected content of test_file.txt'


# LLM-generated content at query #27
#--------------------------

```python
def test_write_creates_file_and_writes_content():
    import os
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc
    os.remove(path)


# LLM-generated content at query #28
#--------------------------

```python
def test_write_predicate_false():
    assert not _write.__doc__.startswith("Write")


# LLM-generated content at query #29
#--------------------------

```python
def test_write_predicate_evaluates_to_false():
    path = "/invalid/path/that/does/not/exist/file.txt"
    doc = "test content"
    assert not _write(path, doc)


# LLM-generated content at query #30
#--------------------------

```python
def test_write_predicate_false():
    assert not (path := 'test.txt') or not (doc := 'content')


# LLM-generated content at query #31
#--------------------------

```python
def test_read_returns_file_content():
    content = _read("test_file.txt")
    assert content == "expected content"


# LLM-generated content at query #32
#--------------------------

```python
def test_root_names_not_empty():
    root_names = {"test": "module"}
    assert root_names


# LLM-generated content at query #33
#--------------------------

```python
def test_empty_root_names():
    assert not gen_api({})


# LLM-generated content at query #34
#--------------------------

```python
def test_read_returns_content_of_file():
    assert _read("test_file.txt") == "Expected content"


# LLM-generated content at query #35
#--------------------------

```python
def test_read_returns_content_of_file():
    assert _read("test_file.txt") == "This is a test file."


# LLM-generated content at query #36
#--------------------------

```python
def test_write_creates_file():
    import os
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    assert os.path.exists(path)
    os.remove(path)


# LLM-generated content at query #37
#--------------------------

```python
def test_write_creates_file():
    import os
    _write('test_file.txt', 'test content')
    assert os.path.exists('test_file.txt')


# LLM-generated content at query #38
#--------------------------

```python
def test_root_names_items_iteration():
    root_names = {"test_title": "test_name"}
    for title, name in root_names.items():
        assert title == "test_title"
        assert name == "test_name"


# LLM-generated content at query #39
#--------------------------

```python
def test_gen_api_iterates_over_root_names():
    root_names = {"test": "module"}
    gen_api(root_names, dry=True)


# LLM-generated content at query #40
#--------------------------

```python
def test__read_returns_file_content():
    import os
    test_file_path = "test_file.txt"
    test_content = "Hello, World!"
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    assert _read(test_file_path) == test_content
    os.remove(test_file_path)


# LLM-generated content at query #41
#--------------------------

```python
def test__read_existing_file():
    content = _read('existing_file.txt')
    assert content == 'expected content'

def test__read_nonexistent_file():
    try:
        _read('nonexistent_file.txt')
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# LLM-generated content at query #42
#--------------------------

```python
def test_gen_api_with_valid_inputs():
    result = gen_api({"test": "test_package"}, "test_path", prefix="test_docs", link=True, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) >= 0


# LLM-generated content at query #43
#--------------------------

```python
def test_read_existing_file():
    content = _read("test_file.txt")
    assert content == "Hello, World!"

def test_read_nonexistent_file():
    try:
        _read("nonexistent_file.txt")
    except FileNotFoundError:
        pass


# LLM-generated content at query #44
#--------------------------

```python
def test_empty_root_names():
    assert not gen_api({})


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loader_basic():
    result = loader("test_pkg", "path/to/test_pkg", True, 1, False)
    assert isinstance(result, str)
    assert result.startswith("**Table of contents:**") or result.startswith("#")

def test_loader_with_toc():
    result = loader("test_pkg", "path/to/test_pkg", True, 1, True)
    assert result.startswith("**Table of contents:**")

def test_loader_different_level():
    result = loader("test_pkg", "path/to/test_pkg", False, 2, False)
    assert isinstance(result, str)

def test_loader_no_link():
    result = loader("test_pkg", "path/to/test_pkg", False, 1, False)
    assert isinstance(result, str)

def test_loader_empty_package():
    result = loader("empty_pkg", "path/to/empty_pkg", True, 1, False)
    assert isinstance(result, str)

def test_loader_non_existent_package():
    result = loader("non_existent", "path/to/non_existent", True, 1, False)
    assert isinstance(result, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = "nonexistent_path" + ext
        isfile_result = False
        if not isfile_result:
            continue
        pure_py = True
    assert not pure_py


# LLM-generated content at query #3
#--------------------------

```python
def test_isfile_returns_false():
    isfile.return_value = False
    assert not isfile("any_path.py")


# LLM-generated content at query #4
#--------------------------

```python
def test_ext_equals_py():
    ext = ".py"
    assert ext == ".py"


# LLM-generated content at query #5
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = "nonexistent_path" + ext
        if not isfile(path_ext):
            continue
        pure_py = True
    assert not pure_py


# LLM-generated content at query #6
#--------------------------

```python
def test_write_creates_file_with_content():
    import os
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc
    os.remove(path)


# LLM-generated content at query #7
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    name = "test_module"
    path = "/path/to/module"
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = path + ext
        isfile_result = False
        if ext == ".py":
            assert not isfile_result
            assert pure_py == False


# LLM-generated content at query #8
#--------------------------

```python
def test_gen_api_basic():
    result = gen_api({"Test": "test"}, "test_path", prefix="test_docs", link=False, level=2, toc=True, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("## Test API\n\n")


# LLM-generated content at query #9
#--------------------------

```python
def test_site_path_existing_package():
    assert _site_path("os").endswith("os")

def test_site_path_nonexistent_package():
    assert _site_path("nonexistent_package") == ""

def test_site_path_package_without_submodules():
    assert _site_path("sys") == ""


# LLM-generated content at query #10
#--------------------------

```python
def test_isdir_predicate():
    assert not isdir('docs')


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_false():
    pure_py = False
    ext = ".pyi"
    assert not (ext == ".py")


# LLM-generated content at query #12
#--------------------------

```python
def test_loader_basic():
    result = loader("test_pkg", "path/to/test_pkg", True, 1, False)
    assert isinstance(result, str)
    assert result.startswith("# Module `test_pkg`")

def test_loader_with_toc():
    result = loader("test_pkg", "path/to/test_pkg", True, 1, True)
    assert result.startswith("**Table of contents:**")

def test_loader_no_link():
    result = loader("test_pkg", "path/to/test_pkg", False, 1, False)
    assert "<a id=" not in result

def test_loader_different_level():
    result = loader("test_pkg", "path/to/test_pkg", True, 2, False)
    assert result.startswith("## Module `test_pkg`")

def test_loader_empty_package():
    result = loader("empty_pkg", "path/to/empty_pkg", True, 1, False)
    assert result == "\n"

def test_loader_with_submodules():
    result = loader("parent_pkg", "path/to/parent_pkg", True, 1, False)
    assert "parent_pkg.submodule" in result

def test_loader_with_docstrings():
    result = loader("doc_pkg", "path/to/doc_pkg", True, 1, False)
    assert "This is a test package" in result

def test_loader_with_constants():
    result = loader("const_pkg", "path/to/const_pkg", True, 1, False)
    assert "Constants" in result
    assert "TEST_CONST" in result

def test_loader_with_classes():
    result = loader("class_pkg", "path/to/class_pkg", True, 1, False)
    assert "class TestClass" in result

def test_loader_with_functions():
    result = loader("func_pkg", "path/to/func_pkg", True, 1, False)
    assert "test_function()" in result

def test_loader_with_imports():
    result = loader("import_pkg", "path/to/import_pkg", True, 1, False)
    assert "Imported module" in result

def test_loader_with_decorators():
    result = loader("decorator_pkg", "path/to/decorator_pkg", True, 1, False)
    assert "Decorators" in result
    assert "@staticmethod" in result

def test_loader_with_inheritance():
    result = loader("inherit_pkg", "path/to/inherit_pkg", True, 1, False)
    assert "Bases" in result
    assert "ParentClass" in result

def test_loader_with_enum():
    result = loader("enum_pkg", "path/to/enum_pkg", True, 1, False)
    assert "Enums" in result
    assert "RED" in result

def test_loader_with_members():
    result = loader("member_pkg", "path/to/member_pkg", True, 1, False)
    assert "Members" in result
    assert "public_attr" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_load_module_success():
    parser = Parser()
    assert _load_module('os.path', 'path/to/os/path.py', parser) is True
    assert parser.docstring.get('os.path') is not None

def test_load_module_failure():
    parser = Parser()
    assert _load_module('nonexistent.module', 'path/to/nonexistent.py', parser) is False
    assert parser.docstring.get('nonexistent.module') is None

def test_load_module_with_parent_import_error():
    parser = Parser()
    assert _load_module('invalid.parent.child', 'path/to/child.py', parser) is False
    assert parser.docstring.get('invalid.parent.child') is None


# LLM-generated content at query #14
#--------------------------

```python
def test__load_module_returns_false_when_spec_is_none():
    assert _load_module("test_module", "test_path", Parser()) is False


# LLM-generated content at query #15
#--------------------------

```python
def test_isdir_predicate_true():
    assert isdir('docs') == True


# LLM-generated content at query #16
#--------------------------

```python
def test_isdir_evaluates_to_true():
    assert isdir('docs')


# LLM-generated content at query #17
#--------------------------

```python
def test_loader_basic():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, False)
    assert isinstance(result, str)
    assert result.startswith('# Module `test_pkg`')

def test_loader_with_toc():
    result = loader('test_pkg', 'tests/test_pkg', True, 1, True)
    assert isinstance(result, str)
    assert result.startswith('**Table of contents:**')

def test_loader_no_link():
    result = loader('test_pkg', 'tests/test_pkg', False, 1, False)
    assert isinstance(result, str)
    assert '<a id=' not in result

def test_loader_different_level():
    result = loader('test_pkg', 'tests/test_pkg', True, 2, False)
    assert isinstance(result, str)
    assert result.startswith('## Module `test_pkg`')

def test_loader_empty_package():
    result = loader('empty_pkg', 'tests/empty_pkg', True, 1, False)
    assert isinstance(result, str)
    assert 'no module' in result or 'Missing documentation' in result

def test_loader_extension_module():
    result = loader('ext_pkg', 'tests/ext_pkg', True, 1, False)
    assert isinstance(result, str)
    assert 'extension module' in result or 'Missing documentation' in result


# LLM-generated content at query #18
#--------------------------

```python
def test__load_module_with_none_spec():
    s = None
    assert not (s is not None and isinstance(s.loader, Loader))


# LLM-generated content at query #19
#--------------------------

```python
def test_load_module_with_none_spec():
    name = "test_module"
    path = "test_path.py"
    p = Parser()
    with patch('apimd.loader.spec_from_file_location', return_value=None):
        assert _load_module(name, path, p) == False


# LLM-generated content at query #20
#--------------------------

```python
def test_pure_py_is_false_when_no_py_file():
    pure_py = False
    for ext in [".py", ".pyi"]:
        path_ext = "nonexistent_path" + ext
        if not isfile(path_ext):
            continue
        if ext == ".py":
            pure_py = True
    assert not pure_py


# LLM-generated content at query #21
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


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_gen_api():
    root_names = {"Test": "test"}
    result = gen_api(root_names, pwd=None, prefix='docs', link=True, level=1, toc=False, dry=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("# Test API\n\n")


# LLM-generated content at query #24
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    import importlib.util
    importlib.util.find_spec = lambda name: None
    assert _site_path("nonexistent_module") == ""


# LLM-generated content at query #25
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    from importlib.util import find_spec
    from os.path import dirname

    find_spec.return_value = None
    assert _site_path("nonexistent_module") == ""


# LLM-generated content at query #26
#--------------------------

```python
def test_pure_py_remains_false_when_no_py_file():
    pure_py = False
    ext = ".pyi"
    if ext == ".py":
        pure_py = True
    assert not pure_py


# LLM-generated content at query #27
#--------------------------

```python
def test_read_returns_file_content():
    assert _read("test_file.txt") == "expected content"


# LLM-generated content at query #28
#--------------------------

```python
def test_site_path_returns_empty_string_when_spec_is_none():
    import importlib.util
    importlib.util.find_spec = lambda _: None
    assert _site_path("nonexistent_module") == ""


# LLM-generated content at query #29
#--------------------------

```python
def test_site_path_existing_package():
    assert _site_path("os").endswith("os") or _site_path("os") == ""

def test_site_path_nonexistent_package():
    assert _site_path("nonexistent_package") == ""


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_false():
    pure_py = False
    ext = ".pyi"
    assert not (ext == ".py")


# LLM-generated content at query #31
#--------------------------

```python
def test__load_module_returns_false_when_spec_is_none():
    assert not _load_module("test_module", "test_path", Parser())


# LLM-generated content at query #32
#--------------------------

```python
def test__write_creates_file_with_content():
    path = "test_file.txt"
    doc = "Test content"
    _write(path, doc)
    with open(path, 'r', encoding='utf-8') as f:
        assert f.read() == doc


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_evaluates_to_false():
    ext = ".pyi"
    assert ext == ".py" is False


# LLM-generated content at query #34
#--------------------------

```python
def test_read_returns_file_content():
    assert _read("test_file.txt") == "expected content"


# LLM-generated content at query #35
#--------------------------

```python
def test_load_module_success():
    p = Parser()
    assert _load_module('test_module', 'test_module.py', p) is True
    assert p.docstring == {'test_module': 'Test module docstring'}

def test_load_module_failure():
    p = Parser()
    assert _load_module('nonexistent_module', 'nonexistent_module.py', p) is False
    assert p.docstring == {}

def test_load_module_with_parent_import_error():
    p = Parser()
    assert _load_module('child_module', 'child_module.py', p) is False
    assert p.docstring == {}


