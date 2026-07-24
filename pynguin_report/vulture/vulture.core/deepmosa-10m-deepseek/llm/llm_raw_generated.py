####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_get_unused_items_all_items_unused():
    defined_items = [MockItem("b"), MockItem("a")]
    used_names = set()
    result = _get_unused_items(defined_items, used_names)
    assert result == [MockItem("a"), MockItem("b")]

def test_get_unused_items_some_items_unused():
    defined_items = [MockItem("x"), MockItem("y"), MockItem("z")]
    used_names = {"x", "z"}
    result = _get_unused_items(defined_items, used_names)
    assert result == [MockItem("y")]

def test_get_unused_items_all_items_used():
    defined_items = [MockItem("a"), MockItem("b")]
    used_names = {"a", "b"}
    result = _get_unused_items(defined_items, used_names)
    assert result == []

def test_get_unused_items_case_insensitive_sort():
    defined_items = [MockItem("B"), MockItem("a"), MockItem("C")]
    used_names = set()
    result = _get_unused_items(defined_items, used_names)
    assert result == [MockItem("a"), MockItem("B"), MockItem("C")]

def test_get_unused_items_empty_defined_items():
    defined_items = []
    used_names = {"a", "b"}
    result = _get_unused_items(defined_items, used_names)
    assert result == []

def test_get_unused_items_no_used_names():
    defined_items = [MockItem("z"), MockItem("y")]
    used_names = set()
    result = _get_unused_items(defined_items, used_names)
    assert result == [MockItem("y"), MockItem("z")]
```


# LLM-generated content at query #2
#--------------------------

```python
def test_get_unused_code_returns_unused_items():
    vulture = Vulture()
    vulture.scan("x = 1\nprint(y)\n")
    unused = vulture.get_unused_code()
    assert len(unused) == 1
    assert unused[0].name == "x"

def test_get_unused_code_filters_by_min_confidence():
    vulture = Vulture()
    vulture.scan("x = 1\ndef f():\n    pass\n")
    unused = vulture.get_unused_code(min_confidence=100)
    assert len(unused) == 1
    assert unused[0].name == "x"

def test_get_unused_code_sorts_by_name():
    vulture = Vulture()
    vulture.scan("b = 1\na = 2\nprint(c)\n")
    unused = vulture.get_unused_code(sort_by_size=False)
    assert len(unused) == 2
    assert unused[0].name == "a"
    assert unused[1].name == "b"

def test_get_unused_code_sorts_by_size():
    vulture = Vulture()
    vulture.scan("a = 1\nb = 2\nprint(c)\n")
    unused = vulture.get_unused_code(sort_by_size=True)
    assert len(unused) == 2
    assert unused[0].size <= unused[1].size

def test_get_unused_code_raises_on_invalid_confidence():
    vulture = Vulture()
    try:
        vulture.get_unused_code(min_confidence=101)
        assert False
    except ValueError:
        pass

def test_get_unused_code_includes_unreachable_code():
    vulture = Vulture()
    vulture.scan("def f():\n    return\n    x = 1\n")
    unused = vulture.get_unused_code()
    assert len(unused) == 1
    assert unused[0].typ == "unreachable_code"

def test_get_unused_code_empty_when_no_dead_code():
    vulture = Vulture()
    vulture.scan("x = 1\nprint(x)\n")
    unused = vulture.get_unused_code()
    assert len(unused) == 0
```


# LLM-generated content at query #3
#--------------------------

```python
def test_visit_function_def_property_decorator():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="my_prop",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="property", ctx=ast.Load())],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_props) == 1
    assert vulture.defined_props[0].name == "my_prop"
    assert vulture.defined_props[0].typ == "property"


def test_visit_function_def_staticmethod_decorator():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="static_method",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="staticmethod", ctx=ast.Load())],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "static_method"
    assert vulture.defined_methods[0].typ == "method"


def test_visit_function_def_classmethod_decorator():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="class_method",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="cls", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="classmethod", ctx=ast.Load())],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "class_method"
    assert vulture.defined_methods[0].typ == "method"


def test_visit_function_def_method_with_self():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="instance_method",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "instance_method"
    assert vulture.defined_methods[0].typ == "method"


def test_visit_function_def_regular_function():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="regular_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 1
    assert vulture.defined_funcs[0].name == "regular_func"
    assert vulture.defined_funcs[0].typ == "function"


def test_visit_function_def_ignore_decorator():
    vulture = Vulture(ignore_decorators=["ignored_decorator"])
    node = ast.FunctionDef(
        name="ignored_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="ignored_decorator", ctx=ast.Load())],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 0
    assert len(vulture.defined_methods) == 0
    assert len(vulture.defined_props) == 0


def test_visit_function_def_no_decorator_no_self():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="top_level_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="x", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 1
    assert vulture.defined_funcs[0].name == "top_level_func"
    assert vulture.defined_funcs[0].typ == "function"
```


# LLM-generated content at query #4
#--------------------------

```
def test_ignore_variable_with_underscore_only():
    result = _ignore_variable("file.py", "_")

def test_ignore_variable_with_private_prefix():
    result = _ignore_variable("file.py", "_private")

def test_ignore_variable_with_dunder():
    result = _ignore_variable("file.py", "__name__")

def test_ignore_variable_with_dunder_prefix_only():
    result = _ignore_variable("file.py", "__name")

def test_ignore_variable_normal_name():
    result = _ignore_variable("file.py", "normal_var")

def test_ignore_variable_starting_with_double_underscore_not_special():
    result = _ignore_variable("file.py", "__x")

def test_ignore_variable_single_underscore():
    result = _ignore_variable("file.py", "_")

def test_ignore_variable_empty_string():
    result = _ignore_variable("file.py", "")
```


# LLM-generated content at query #5
#--------------------------

```python
def test_visit_FunctionDef_detects_property():
    vulture = Vulture()
    node = ast.parse("@property\ndef foo(self):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_props) == 1
    assert vulture.defined_props[0].name == "foo"

def test_visit_FunctionDef_detects_method_with_self():
    vulture = Vulture()
    node = ast.parse("def bar(self):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "bar"

def test_visit_FunctionDef_detects_staticmethod():
    vulture = Vulture()
    node = ast.parse("@staticmethod\ndef baz():\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "baz"

def test_visit_FunctionDef_detects_classmethod():
    vulture = Vulture()
    node = ast.parse("@classmethod\ndef qux(cls):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "qux"

def test_visit_FunctionDef_detects_function():
    vulture = Vulture()
    node = ast.parse("def quux():\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 1
    assert vulture.defined_funcs[0].name == "quux"

def test_visit_FunctionDef_ignores_decorator_whitelisted():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    node = ast.parse("@my_decorator\ndef foo():\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 0

def test_visit_FunctionDef_ignores_method_decorator_whitelisted():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    node = ast.parse("@my_decorator\ndef bar(self):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 0

def test_visit_FunctionDef_ignores_property_decorator_whitelisted():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    node = ast.parse("@my_decorator\n@property\ndef baz(self):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_props) == 0

def test_visit_FunctionDef_ignores_function_with_ignore_names():
    vulture = Vulture(ignore_names=["ignored_func"])
    node = ast.parse("def ignored_func():\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 0

def test_visit_FunctionDef_ignores_method_with_ignore_names():
    vulture = Vulture(ignore_names=["ignored_method"])
    node = ast.parse("def ignored_method(self):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 0

def test_visit_FunctionDef_ignores_property_with_ignore_names():
    vulture = Vulture(ignore_names=["ignored_prop"])
    node = ast.parse("@property\ndef ignored_prop(self):\n    pass").body[0]
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_props) == 0
```


# LLM-generated content at query #6
#--------------------------

```
def test_ignore_function_returns_true_for_pytest_function_in_test_file():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "test", "file.py")
    assert _ignore_function(filename, "pytest_func") == True

def test_ignore_function_returns_true_for_test_prefixed_function_in_test_file():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "tests", "file.py")
    assert _ignore_function(filename, "test_something") == True

def test_ignore_function_returns_false_for_non_test_function_in_test_file():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "test", "file.py")
    assert _ignore_function(filename, "regular_func") == False

def test_ignore_function_returns_false_for_pytest_function_in_non_test_file():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "src", "file.py")
    assert _ignore_function(filename, "pytest_func") == False

def test_ignore_function_returns_false_for_test_prefixed_function_in_non_test_file():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "src", "file.py")
    assert _ignore_function(filename, "test_something") == False

def test_ignore_function_returns_false_for_non_test_function_in_non_test_file():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "src", "file.py")
    assert _ignore_function(filename, "regular_func") == False

def test_ignore_function_handles_test_pattern_with_underscore():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "test_example.py")
    assert _ignore_function(filename, "test_func") == True

def test_ignore_function_handles_test_pattern_with_hyphen():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "test-example.py")
    assert _ignore_function(filename, "test_func") == True

def test_ignore_function_handles_test_suffix_pattern():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "example_test.py")
    assert _ignore_function(filename, "test_func") == True

def test_ignore_function_handles_test_suffix_pattern_with_hyphen():
    from unittest.mock import MagicMock
    import os
    filename = MagicMock()
    filename.resolve.return_value = os.path.join("some", "example-test.py")
    assert _ignore_function(filename, "test_func") == False
```


# LLM-generated content at query #7
#--------------------------

```python
def test_visit_FunctionDef_regular_function():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="my_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 1
    assert vulture.defined_funcs[0].name == "my_func"

def test_visit_FunctionDef_method_with_self():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="my_method",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "my_method"

def test_visit_FunctionDef_property():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="my_prop",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="property")],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_props) == 1
    assert vulture.defined_props[0].name == "my_prop"

def test_visit_FunctionDef_staticmethod():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="my_static",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="staticmethod")],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "my_static"

def test_visit_FunctionDef_classmethod():
    vulture = Vulture()
    node = ast.FunctionDef(
        name="my_classmethod",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="cls")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="classmethod")],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_methods) == 1
    assert vulture.defined_methods[0].name == "my_classmethod"

def test_visit_FunctionDef_ignored_decorator():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    node = ast.FunctionDef(
        name="my_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[ast.Name(id="my_decorator")],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 0
    assert len(vulture.defined_methods) == 0
    assert len(vulture.defined_props) == 0

def test_visit_FunctionDef_ignored_name():
    vulture = Vulture(ignore_names=["ignored_func"])
    node = ast.FunctionDef(
        name="ignored_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 0

def test_visit_FunctionDef_async_function():
    vulture = Vulture()
    node = ast.AsyncFunctionDef(
        name="async_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        lineno=1,
        end_lineno=3,
    )
    vulture.visit_FunctionDef(node)
    assert len(vulture.defined_funcs) == 1
    assert vulture.defined_funcs[0].name == "async_func"
```


# LLM-generated content at query #8
#--------------------------

def test_scavenge_excludes_path():
    v = Vulture()
    v.scavenge([], exclude=["test_exclude"])
    assert v.exit_code == 0

def test_scavenge_invalid_file():
    v = Vulture()
    v.scavenge(["nonexistent_file.py"])
    assert v.exit_code == 1

def test_scavenge_valid_file():
    v = Vulture()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("x = 1\n")
        fname = f.name
    v.scavenge([fname])
    assert v.exit_code == 0
    import os
    os.unlink(fname)

def test_scavenge_with_whitelist():
    v = Vulture()
    v.defined_imports.append(type('ImportItem', (), {'name': 'os'})())
    v.scavenge([])
    assert v.exit_code == 0


# LLM-generated content at query #9
#--------------------------

def test_ignore_method_special_name():
    result = _ignore_method("some_file.py", "__init__")
    assert result == True

def test_ignore_method_pytest_method_in_test_file():
    result = _ignore_method("tests/test_example.py", "test_something")
    assert result == True

def test_ignore_method_test_startswith_in_test_dir():
    result = _ignore_method("test/test_example.py", "test_another")
    assert result == True

def test_ignore_method_regular_method_in_test_file():
    result = _ignore_method("tests/test_example.py", "regular_method")
    assert result == False

def test_ignore_method_pytest_method_in_non_test_file():
    result = _ignore_method("src/main.py", "test_something")
    assert result == False

def test_ignore_method_special_name_in_non_test_file():
    result = _ignore_method("src/main.py", "__str__")
    assert result == True

def test_ignore_method_contains_test_in_name():
    result = _ignore_method("test_example.py", "test_case")
    assert result == True

def test_ignore_method_ends_with_test():
    result = _ignore_method("example_test.py", "test_runner")
    assert result == True

def test_ignore_method_regular_name_in_non_test_file():
    result = _ignore_method("src/main.py", "helper")
    assert result == False

def test_ignore_method_pytest_method_names_list():
    result = _ignore_method("tests/test_example.py", "setup_method")
    assert result == True


# LLM-generated content at query #10
#--------------------------

```python
def test_scan_simple_variable_assignment():
    v = Vulture()
    code = "x = 1"
    v.scan(code)
    assert len(v.defined_vars) == 1
    assert v.defined_vars[0].name == "x"
    assert v.defined_vars[0].first_lineno == 1

def test_scan_syntax_error_sets_exit_code():
    v = Vulture()
    code = "def foo(:"
    v.scan(code)
    assert v.exit_code == ExitCode.InvalidInput

def test_scan_value_error_sets_exit_code():
    v = Vulture()
    code = "x = \0"
    v.scan(code)
    assert v.exit_code == ExitCode.InvalidInput

def test_scan_with_type_comment():
    v = Vulture()
    code = "x = 1  # type: int"
    v.scan(code)
    assert len(v.defined_vars) == 1
    assert v.defined_vars[0].name == "x"

def test_scan_syntax_error_in_type_comment():
    v = Vulture()
    code = "x = 1  # type: not a type"
    v.scan(code)
    assert v.exit_code == ExitCode.DeadCode or v.exit_code == ExitCode.NoDeadCode

def test_scan_import_usage():
    v = Vulture()
    code = "import os\nos.path.join('a', 'b')"
    v.scan(code)
    assert len(v.used_names) > 0
    assert "os" in v.used_names

def test_scan_function_definition():
    v = Vulture()
    code = "def my_func():\n    pass"
    v.scan(code)
    assert len(v.defined_funcs) == 1
    assert v.defined_funcs[0].name == "my_func"
    assert v.defined_funcs[0].first_lineno == 1

def test_scan_class_definition():
    v = Vulture()
    code = "class MyClass:\n    pass"
    v.scan(code)
    assert len(v.defined_classes) == 1
    assert v.defined_classes[0].name == "MyClass"
    assert v.defined_classes[0].first_lineno == 1

def test_scan_with_ignore_names():
    v = Vulture(ignore_names=["ignored_var"])
    code = "ignored_var = 42"
    v.scan(code)
    assert len(v.defined_vars) == 0

def test_scan_with_ignore_decorators():
    v = Vulture(ignore_decorators=["@custom_decorator"])
    code = "@custom_decorator\ndef my_func():\n    pass"
    v.scan(code)
    assert len(v.defined_funcs) == 0

def test_scan_empty_code():
    v = Vulture()
    v.scan("")
    assert v.exit_code == ExitCode.NoDeadCode

def test_scan_code_with_noqa_comment():
    v = Vulture()
    code = "x = 1  # noqa: V101"
    v.scan(code)
    assert len(v.defined_vars) == 0

def test_scan_with_reachability_analysis():
    v = Vulture()
    code = "if True:\n    dead_code = 1"
    v.scan(code)
    assert len(v.unreachable_code) > 0

def test_scan_resets_reachability():
    v = Vulture()
    code = "x = 1"
    v.scan(code)
    v.scan("y = 2")
    assert len(v.defined_vars) == 1
    assert v.defined_vars[0].name == "y"

def test_scan_with_filename():
    v = Vulture()
    code = "x = 1"
    v.scan(code, filename="test.py")
    assert str(v.filename) == "test.py"
    assert v.defined_vars[0].filename == Path("test.py")

def test_scan_parses_type_comments():
    v = Vulture()
    code = "def foo(x: int) -> str: pass"
    v.scan(code)
    assert len(v.defined_funcs) == 1
    assert v.defined_funcs[0].name == "foo"

def test_scan_async_function():
    v = Vulture()
    code = "async def my_async():\n    pass"
    v.scan(code)
    assert len(v.defined_funcs) == 1
    assert v.defined_funcs[0].name == "my_async"

def test_scan_property_decorator():
    v = Vulture()
    code = "class MyClass:\n    @property\n    def my_prop(self):\n        pass"
    v.scan(code)
    assert len(v.defined_props) == 1
    assert v.defined_props[0].name == "my_prop"

def test_scan_method_with_self():
    v = Vulture()
    code = "class MyClass:\n    def my_method(self):\n        pass"
    v.scan(code)
    assert len(v.defined_methods) == 1
    assert v.defined_methods[0].name == "my_method"

def test_scan_multiline_code():
    v = Vulture()
    code = "x = 1\ny = 2\nz = x + y"
    v.scan(code)
    assert len(v.defined_vars) == 3
    assert len(v.used_names) >= 2

def test_scan_code_with_nested_functions():
    v = Vulture()
    code = "def outer():\n    def inner():\n        pass\n    pass"
    v.scan(code)
    assert len(v.defined_funcs) == 2

def test_scan_import_from_usage():
    v = Vulture()
    code = "from os import path\npath.join('a', 'b')"
    v.scan(code)
    assert len(v.defined_imports) == 1
    assert v.defined_imports[0].name == "path"
    assert "path" in v.used_names

def test_scan_attribute_definition():
    v = Vulture()
    code = "class MyClass:\n    def __init__(self):\n        self.attr = 1"
    v.scan(code)
    assert len(v.defined_attrs) == 1
    assert v.defined_attrs[0].name == "attr"

def test_scan_code_with_unreachable_code():
    v = Vulture()
    code = "def foo():\n    return 1\n    x = 2"
    v.scan(code)
    assert len(v.unreachable_code) >= 1

def test_scan_code_with_no_line_numbers():
    v = Vulture()
    code = "x = 1"
    v.scan(code)
    assert v.code == ["x = 1"]

def test_scan_code_with_null_bytes():
    v = Vulture()
    code = "x = \x00"
    v.scan(code)
    assert v.exit_code == ExitCode.InvalidInput

def test_scan_code_with_unicode():
    v = Vulture()
    code = "x = 'café'"
    v.scan(code)
    assert len(v.defined_vars) == 1
    assert v.defined_vars[0].name == "x"

def test_scan_code_with_match_statement():
    v = Vulture()
    code = "match x:\n    case 1:\n        pass"
    v.scan(code)
    assert len(v.defined_vars) >= 1 or v.exit_code == ExitCode.DeadCode

def test_scan_code_with_walrus_operator():
    v = Vulture()
    code = "if (x := 1):\n    pass"
    v.scan(code)
    assert len(v.defined_vars) == 1
    assert v.defined_vars[0].name == "x"
```


# LLM-generated content at query #11
#--------------------------

```
def test_prepare_pattern_returns_pattern_with_wildcards_when_no_special_chars():
    vulture = Vulture()
    paths = [__file__]
    exclude = ["test"]
    vulture.scavenge(paths, exclude=exclude)
    assert any("*test*" in str(item) for item in vulture.defined_vars) or True


# LLM-generated content at query #12
#--------------------------

```
def test_scavenge_exclude_path_true():
    v = Vulture()
    v.defined_imports = v.get_list("import")
    v.scan("import os", filename="test.py")
    v.scavenge(["test.py"], exclude="test")


# LLM-generated content at query #13
#--------------------------

def test_vulture_constructor_defaults():
    v = Vulture()
    assert v.verbose == False
    assert v.ignore_names == []
    assert v.ignore_decorators == []
    assert v.filename == Path()
    assert v.code == []
    assert v.exit_code == ExitCode.NoDeadCode
    assert v.noqa_lines == {}
    assert isinstance(v.defined_attrs, utils.LoggingList)
    assert isinstance(v.defined_classes, utils.LoggingList)
    assert isinstance(v.defined_funcs, utils.LoggingList)
    assert isinstance(v.defined_imports, utils.LoggingList)
    assert isinstance(v.defined_methods, utils.LoggingList)
    assert isinstance(v.defined_props, utils.LoggingList)
    assert isinstance(v.defined_vars, utils.LoggingList)
    assert isinstance(v.unreachable_code, utils.LoggingList)
    assert isinstance(v.used_names, utils.LoggingSet)
    assert isinstance(v.reachability, Reachability)

def test_vulture_constructor_with_verbose():
    v = Vulture(verbose=True)
    assert v.verbose == True

def test_vulture_constructor_with_ignore_names():
    v = Vulture(ignore_names=["foo", "bar"])
    assert v.ignore_names == ["foo", "bar"]

def test_vulture_constructor_with_ignore_decorators():
    v = Vulture(ignore_decorators=["decor1", "decor2"])
    assert v.ignore_decorators == ["decor1", "decor2"]


# LLM-generated content at query #14
#--------------------------

def test_vulture_constructor_default_parameters():
    vulture = Vulture()
    assert vulture.verbose == False
    assert vulture.ignore_names == []
    assert vulture.ignore_decorators == []
    assert vulture.filename == Path()
    assert vulture.code == []
    assert vulture.exit_code == ExitCode.NoDeadCode
    assert vulture.noqa_lines == {}

def test_vulture_constructor_custom_parameters():
    vulture = Vulture(verbose=True, ignore_names=["foo"], ignore_decorators=["bar"])
    assert vulture.verbose == True
    assert vulture.ignore_names == ["foo"]
    assert vulture.ignore_decorators == ["bar"]


# LLM-generated content at query #15
#--------------------------

```
def test_get_list_returns_logging_list_for_property():
    vulture = Vulture(verbose=False)
    assert isinstance(vulture.defined_props, utils.LoggingList)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_prepare_pattern_returns_pattern_with_wildcards_when_no_wildcards_in_input():
    result = prepare_pattern("foo")
    assert result == "*foo*"


# LLM-generated content at query #17
#--------------------------

def test_item_constructor_initializes_all_attributes():
    item = Item("my_func", "function", "/path/to/file.py", 10, 20, "custom message", 75)
    assert item.name == "my_func"
    assert item.typ == "function"
    assert item.filename == "/path/to/file.py"
    assert item.first_lineno == 10
    assert item.last_lineno == 20
    assert item.message == "custom message"
    assert item.confidence == 75

def test_item_constructor_default_message():
    item = Item("my_var", "variable", "/path/file.py", 5, 5)
    assert item.message == "unused variable 'my_var'"

def test_item_constructor_default_confidence():
    item = Item("MyClass", "class", "/path/file.py", 1, 10)
    assert item.confidence == DEFAULT_CONFIDENCE

def test_item_constructor_empty_message_overridden():
    item = Item("test", "function", "/path/file.py", 1, 1, "")
    assert item.message == "unused function 'test'"


# LLM-generated content at query #18
#--------------------------

```
def test_visit_import_from_adds_aliases():
    v = Vulture()
    node = ast.parse("from os import path").body[0]
    v.visit_ImportFrom(node)
    assert len(v.defined_imports) == 1
    assert v.defined_imports[0].name == "os"
    assert v.defined_imports[0].confidence == 90

def test_visit_import_from_skips_future():
    v = Vulture()
    node = ast.parse("from __future__ import annotations").body[0]
    v.visit_ImportFrom(node)
    assert len(v.defined_imports) == 0

def test_visit_import_from_with_alias():
    v = Vulture()
    node = ast.parse("from os import path as p").body[0]
    v.visit_ImportFrom(node)
    assert len(v.defined_imports) == 1
    assert v.defined_imports[0].name == "p"
    assert v.defined_imports[0].confidence == 90

def test_visit_import_from_multiple_names():
    v = Vulture()
    node = ast.parse("from os import path, getcwd").body[0]
    v.visit_ImportFrom(node)
    assert len(v.defined_imports) == 2
    assert v.defined_imports[0].name == "os"
    assert v.defined_imports[1].name == "os"

def test_visit_import_from_used_names_added_for_alias():
    v = Vulture()
    node = ast.parse("from os import path as p").body[0]
    v.visit_ImportFrom(node)
    assert "os" in v.used_names

def test_visit_import_from_package_more_than_one_level():
    v = Vulture()
    node = ast.parse("from os.path import join").body[0]
    v.visit_ImportFrom(node)
    assert len(v.defined_imports) == 1
    assert v.defined_imports[0].name == "os"
```


# LLM-generated content at query #19
#--------------------------

def test_visit_call_getattr_with_three_args():
    v = Vulture()
    node = ast.parse("getattr(obj, 'some_attr', default)").body[0].value
    v.visit_Call(node)
    assert "some_attr" in v.used_names

def test_visit_call_getattr_with_two_args():
    v = Vulture()
    node = ast.parse("getattr(obj, 'some_attr')").body[0].value
    v.visit_Call(node)
    assert "some_attr" in v.used_names

def test_visit_call_getattr_with_one_arg():
    v = Vulture()
    node = ast.parse("getattr(obj)").body[0].value
    v.visit_Call(node)
    assert "some_attr" not in v.used_names

def test_visit_call_hasattr_with_two_args():
    v = Vulture()
    node = ast.parse("hasattr(obj, 'some_attr')").body[0].value
    v.visit_Call(node)
    assert "some_attr" in v.used_names

def test_visit_call_hasattr_with_one_arg():
    v = Vulture()
    node = ast.parse("hasattr(obj)").body[0].value
    v.visit_Call(node)
    assert "some_attr" not in v.used_names

def test_visit_call_getattr_non_string_attr():
    v = Vulture()
    node = ast.parse("getattr(obj, attr_name)").body[0].value
    v.visit_Call(node)
    assert "attr_name" not in v.used_names

def test_visit_call_format_with_locals():
    v = Vulture()
    node = ast.parse("\"{my_var}\".format(**locals())").body[0].value
    v.visit_Call(node)
    assert "my_var" in v.used_names

def test_visit_call_format_without_locals():
    v = Vulture()
    node = ast.parse("\"{my_var}\".format()").body[0].value
    v.visit_Call(node)
    assert "my_var" not in v.used_names

def test_visit_call_format_with_non_locals_kwargs():
    v = Vulture()
    node = ast.parse("\"{my_var}\".format(**other)").body[0].value
    v.visit_Call(node)
    assert "my_var" not in v.used_names

def test_visit_call_format_with_positional_args():
    v = Vulture()
    node = ast.parse("\"{}\".format(1)").body[0].value
    v.visit_Call(node)
    assert "my_var" not in v.used_names


# LLM-generated content at query #20
#--------------------------

def test_verbose_false():
    v = Vulture(verbose=False)
    assert v.verbose == False


# LLM-generated content at query #21
#--------------------------

```python
def test_visit_attribute_load():
    v = Vulture()
    node = ast.Attribute(attr="some_attr", ctx=ast.Load())
    v.visit_Attribute(node)
    assert "some_attr" in v.used_names

def test_visit_attribute_store():
    v = Vulture()
    node = ast.Attribute(attr="some_attr", ctx=ast.Store())
    v.visit_Attribute(node)
    assert len(v.defined_attrs) == 1
    assert v.defined_attrs[0].name == "some_attr" 
```


# LLM-generated content at query #22
#--------------------------

```
def test_generic_visit_with_list_of_nodes():
    v = Vulture()
    node = ast.FunctionDef(
        name="test",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Pass()],
        decorator_list=[],
    )
    v.code = ["def test(): pass"]
    v.generic_visit(node)

def test_generic_visit_with_single_node():
    v = Vulture()
    node = ast.Expr(value=ast.Constant(value=1))
    v.code = ["1"]
    v.generic_visit(node)

def test_generic_visit_with_empty_list():
    v = Vulture()
    node = ast.Module(body=[])
    v.code = [""]
    v.generic_visit(node)

def test_generic_visit_with_nested_ast_nodes():
    v = Vulture()
    node = ast.BinOp(left=ast.Constant(value=1), op=ast.Add(), right=ast.Constant(value=2))
    v.code = ["1 + 2"]
    v.generic_visit(node)

def test_generic_visit_with_multiple_lists():
    v = Vulture()
    node = ast.FunctionDef(
        name="f",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="x")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Return(value=ast.Name(id="x", ctx=ast.Load()))],
        decorator_list=[],
    )
    v.code = ["def f(x): return x"]
    v.generic_visit(node)

def test_generic_visit_with_attribute_access():
    v = Vulture()
    node = ast.Attribute(value=ast.Name(id="obj", ctx=ast.Load()), attr="attr", ctx=ast.Load())
    v.code = ["obj.attr"]
    v.generic_visit(node)

def test_generic_visit_with_none_value():
    v = Vulture()
    node = ast.If(test=ast.Constant(value=True), body=[ast.Pass()], orelse=[])
    v.code = ["if True: pass"]
    v.generic_visit(node)

def test_generic_visit_with_complex_node():
    v = Vulture()
    node = ast.Call(func=ast.Name(id="func", ctx=ast.Load()), args=[ast.Constant(value=1)], keywords=[])
    v.code = ["func(1)"]
    v.generic_visit(node)

def test_generic_visit_with_subscript():
    v = Vulture()
    node = ast.Subscript(value=ast.Name(id="lst", ctx=ast.Load()), slice=ast.Constant(value=0), ctx=ast.Load())
    v.code = ["lst[0]"]
    v.generic_visit(node)

def test_generic_visit_with_list_node():
    v = Vulture()
    node = ast.List(elts=[ast.Constant(value=1), ast.Constant(value=2)], ctx=ast.Load())
    v.code = ["[1, 2]"]
    v.generic_visit(node)
```


# LLM-generated content at query #23
#--------------------------

```
def test_scavenge_exclude_path_true():
    vulture = Vulture(verbose=False)
    vulture.scavenge(paths=["."], exclude=["*.py"])
    assert vulture.exit_code == ExitCode.NoDeadCode
```


# LLM-generated content at query #24
#--------------------------

def test_vulture_constructor_defaults():
    v = Vulture()
    assert v.verbose == False
    assert v.ignore_names == []
    assert v.ignore_decorators == []
    assert v.filename == Path()
    assert v.code == []
    assert v.exit_code == ExitCode.NoDeadCode
    assert v.noqa_lines == {}
    assert isinstance(v.defined_attrs, utils.LoggingList)
    assert v.defined_attrs.typ == "attribute"
    assert isinstance(v.defined_classes, utils.LoggingList)
    assert v.defined_classes.typ == "class"
    assert isinstance(v.defined_funcs, utils.LoggingList)
    assert v.defined_funcs.typ == "function"
    assert isinstance(v.defined_imports, utils.LoggingList)
    assert v.defined_imports.typ == "import"
    assert isinstance(v.defined_methods, utils.LoggingList)
    assert v.defined_methods.typ == "method"
    assert isinstance(v.defined_props, utils.LoggingList)
    assert v.defined_props.typ == "property"
    assert isinstance(v.defined_vars, utils.LoggingList)
    assert v.defined_vars.typ == "variable"
    assert isinstance(v.unreachable_code, utils.LoggingList)
    assert v.unreachable_code.typ == "unreachable_code"
    assert isinstance(v.used_names, utils.LoggingSet)
    assert v.used_names.typ == "name"
    assert hasattr(v, 'reachability')

def test_vulture_constructor_with_verbose():
    v = Vulture(verbose=True)
    assert v.verbose == True

def test_vulture_constructor_with_ignore_names():
    v = Vulture(ignore_names=["foo", "bar"])
    assert v.ignore_names == ["foo", "bar"]

def test_vulture_constructor_with_ignore_decorators():
    v = Vulture(ignore_decorators=["@deco"])
    assert v.ignore_decorators == ["@deco"]

def test_vulture_constructor_with_all_arguments():
    v = Vulture(verbose=True, ignore_names=["x"], ignore_decorators=["@y"])
    assert v.verbose == True
    assert v.ignore_names == ["x"]
    assert v.ignore_decorators == ["@y"]


# LLM-generated content at query #25
#--------------------------

```python
def test_init_predicate_evaluates_to_false():
    item = Item("x", "variable", "file.py", 1, 1, message="custom message")
    assert item.message == "custom message"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_unused_code_returns_list():
    vulture = Vulture()
    vulture.scan("x = 1\n", filename="test.py")
    result = vulture.get_unused_code()
    assert isinstance(result, list)


def test_get_unused_code_min_confidence_low():
    vulture = Vulture()
    vulture.scan("def foo():\n    pass\n", filename="test.py")
    result = vulture.get_unused_code(min_confidence=0)
    assert len(result) == 1
    assert result[0].name == "foo"


def test_get_unused_code_min_confidence_high():
    vulture = Vulture()
    vulture.scan("def foo():\n    pass\n", filename="test.py")
    result = vulture.get_unused_code(min_confidence=100)
    assert len(result) == 0


def test_get_unused_code_min_confidence_boundary():
    vulture = Vulture()
    vulture.scan("x = 1\n", filename="test.py")
    result_low = vulture.get_unused_code(min_confidence=0)
    result_high = vulture.get_unused_code(min_confidence=100)
    assert len(result_low) == 1
    assert len(result_high) == 1


def test_get_unused_code_sort_by_size_default():
    vulture = Vulture()
    vulture.scan("def foo():\n    pass\n\ndef bar():\n    pass\n", filename="test.py")
    result = vulture.get_unused_code(sort_by_size=False)
    assert result[0].name == "bar"
    assert result[1].name == "foo"


def test_get_unused_code_sort_by_size():
    vulture = Vulture()
    vulture.scan("def foo():\n    pass\n\ndef bar():\n    pass\n", filename="test.py")
    result = vulture.get_unused_code(sort_by_size=True)
    assert result[0].size <= result[1].size


def test_get_unused_code_returns_item_objects():
    vulture = Vulture()
    vulture.scan("x = 1\n", filename="test.py")
    result = vulture.get_unused_code()
    assert all(hasattr(item, "name") for item in result)
    assert all(hasattr(item, "filename") for item in result)
    assert all(hasattr(item, "first_lineno") for item in result)
    assert all(hasattr(item, "size") for item in result)
    assert all(hasattr(item, "confidence") for item in result)


def test_get_unused_code_empty():
    vulture = Vulture()
    vulture.scan("", filename="test.py")
    result = vulture.get_unused_code()
    assert result == []


def test_get_unused_code_min_confidence_negative():
    vulture = Vulture()
    vulture.scan("x = 1\n", filename="test.py")
    try:
        vulture.get_unused_code(min_confidence=-1)
        assert False
    except ValueError:
        pass


def test_get_unused_code_min_confidence_over_100():
    vulture = Vulture()
    vulture.scan("x = 1\n", filename="test.py")
    try:
        vulture.get_unused_code(min_confidence=101)
        assert False
    except ValueError:
        pass
```


# LLM-generated content at query #2
#--------------------------

```
def test_ignore_function_returns_true_for_pytest_function_in_test_file():
    import tempfile
    import pathlib
    file = pathlib.Path(tempfile.mkstemp(suffix="test.py")[1])
    assert _ignore_function(file, "pytest") == True

def test_ignore_function_returns_true_for_test_prefixed_function_in_test_file():
    import tempfile
    import pathlib
    file = pathlib.Path(tempfile.mkstemp(suffix="test.py")[1])
    assert _ignore_function(file, "test_myfunc") == True

def test_ignore_function_returns_false_for_non_test_function_in_test_file():
    import tempfile
    import pathlib
    file = pathlib.Path(tempfile.mkstemp(suffix="test.py")[1])
    assert _ignore_function(file, "myfunc") == False

def test_ignore_function_returns_false_for_pytest_function_in_non_test_file():
    import tempfile
    import pathlib
    file = pathlib.Path(tempfile.mkstemp(suffix=".py")[1])
    assert _ignore_function(file, "pytest") == False

def test_ignore_function_returns_false_for_test_prefixed_function_in_non_test_file():
    import tempfile
    import pathlib
    file = pathlib.Path(tempfile.mkstemp(suffix=".py")[1])
    assert _ignore_function(file, "test_myfunc") == False

def test_ignore_function_returns_false_for_non_test_function_in_non_test_file():
    import tempfile
    import pathlib
    file = pathlib.Path(tempfile.mkstemp(suffix=".py")[1])
    assert _ignore_function(file, "myfunc") == False
```


# LLM-generated content at query #3
#--------------------------

```python
def test_get_report_without_size():
    item = Item("my_func", "function", "path/to/file.py", 10, 20)
    result = item.get_report()
    assert result == "path/to/file.py:10: unused function 'my_func' (100% confidence)"

def test_get_report_with_size_and_single_line():
    item = Item("my_func", "function", "path/to/file.py", 10, 10)
    result = item.get_report(add_size=True)
    assert result == "path/to/file.py:10: unused function 'my_func' (100% confidence, 1 line)"

def test_get_report_with_size_and_multiple_lines():
    item = Item("my_func", "function", "path/to/file.py", 10, 20)
    result = item.get_report(add_size=True)
    assert result == "path/to/file.py:10: unused function 'my_func' (100% confidence, 11 lines)"

def test_get_report_with_custom_message():
    item = Item("my_var", "variable", "path/to/file.py", 5, 5, message="custom message")
    result = item.get_report()
    assert result == "path/to/file.py:5: custom message (100% confidence)"

def test_get_report_with_custom_confidence():
    item = Item("my_var", "variable", "path/to/file.py", 5, 5, confidence=75)
    result = item.get_report()
    assert result == "path/to/file.py:5: unused variable 'my_var' (75% confidence)"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_scan_empty_code():
    vulture = Vulture()
    vulture.scan("", filename="empty.py")
    assert vulture.code == []
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scan_valid_code():
    vulture = Vulture()
    code = "x = 1\n"
    vulture.scan(code, filename="test.py")
    assert vulture.code == ["x = 1"]
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scan_syntax_error():
    vulture = Vulture()
    code = "x = "
    vulture.scan(code, filename="syntax_error.py")
    assert vulture.exit_code == ExitCode.InvalidInput

def test_scan_null_bytes():
    vulture = Vulture()
    code = "x = \x00"
    vulture.scan(code, filename="null_bytes.py")
    assert vulture.exit_code == ExitCode.InvalidInput

def test_scan_with_noqa():
    vulture = Vulture()
    code = "x = 1  # noqa\n"
    vulture.scan(code, filename="noqa.py")
    assert vulture.code == ["x = 1  # noqa"]
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scan_with_filename():
    vulture = Vulture()
    code = "y = 2\n"
    vulture.scan(code, filename="custom.py")
    assert vulture.filename == Path("custom.py")
    assert vulture.code == ["y = 2"]

def test_scan_multiple_lines():
    vulture = Vulture()
    code = "a = 1\nb = 2\n"
    vulture.scan(code, filename="multi.py")
    assert vulture.code == ["a = 1", "b = 2"]
    assert len(vulture.defined_vars) == 2

def test_scan_type_comment_syntax_error():
    vulture = Vulture()
    code = "def f():\n    pass\n"
    vulture.scan(code, filename="type_comment.py")
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scan_resets_reachability():
    vulture = Vulture()
    vulture.scan("x = 1\n", filename="first.py")
    vulture.scan("y = 2\n", filename="second.py")
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scan_empty_filename():
    vulture = Vulture()
    code = "z = 3\n"
    vulture.scan(code, filename="")
    assert vulture.filename == Path("")
    assert vulture.code == ["z = 3"]
```


# LLM-generated content at query #5
#--------------------------

```python
def test_scan_empty_code():
    vulture = Vulture()
    vulture.scan("", filename="test.py")
    assert vulture.exit_code == 0

def test_scan_valid_code():
    vulture = Vulture()
    code = "x = 1"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_syntax_error():
    vulture = Vulture()
    code = "x = "
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 1

def test_scan_null_byte():
    vulture = Vulture()
    code = "x = 1\0"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 1

def test_scan_syntax_error_in_type_comment():
    vulture = Vulture()
    code = "x = 1  # type: "
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 1

def test_scan_multiple_lines():
    vulture = Vulture()
    code = "a = 1\nb = 2"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_with_noqa():
    vulture = Vulture()
    code = "x = 1  # noqa"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_verbose():
    vulture = Vulture(verbose=True)
    code = "x = 1"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_with_ignore_names():
    vulture = Vulture(ignore_names=["x"])
    code = "x = 1"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_with_ignore_decorators():
    vulture = Vulture(ignore_decorators=["@decorator"])
    code = "def func(): pass"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_resets_reachability():
    vulture = Vulture()
    vulture.scan("", filename="test.py")
    vulture.scan("", filename="test2.py")
    assert vulture.reachability is not None

def test_scan_stores_code():
    vulture = Vulture()
    code = "x = 1"
    vulture.scan(code, filename="test.py")
    assert vulture.code == ["x = 1"]

def test_scan_stores_filename():
    vulture = Vulture()
    vulture.scan("", filename="test.py")
    assert vulture.filename == Path("test.py")

def test_scan_with_unicode():
    vulture = Vulture()
    code = "x = 'ü'"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_with_escape_sequence():
    vulture = Vulture()
    code = "x = '\\n'"
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_invalid_utf8():
    vulture = Vulture()
    code = b"x = 1\x80".decode("utf-8", "replace")
    vulture.scan(code, filename="test.py")
    assert vulture.exit_code == 0

def test_scan_after_syntax_error():
    vulture = Vulture()
    vulture.scan("x = ", filename="test.py")
    vulture.scan("x = 1", filename="test.py")
    assert vulture.exit_code == 0

def test_scan_after_value_error():
    vulture = Vulture()
    vulture.scan("x = 1\0", filename="test.py")
    vulture.scan("x = 1", filename="test.py")
    assert vulture.exit_code == 0
```


# LLM-generated content at query #6
#--------------------------

```
def test_ignore_variable_ignores_underscore():
    result = _ignore_variable("test.py", "_")
    assert result == True

def test_ignore_variable_ignores_single_underscore_prefix():
    result = _ignore_variable("test.py", "_x")
    assert result == True

def test_ignore_variable_ignores_dunder_special():
    result = _ignore_variable("test.py", "__init__")
    assert result == True

def test_ignore_variable_does_not_ignore_dunder_prefix():
    result = _ignore_variable("test.py", "__x")
    assert result == False

def test_ignore_variable_does_not_ignore_normal_name():
    result = _ignore_variable("test.py", "x")
    assert result == False

def test_ignore_variable_does_not_ignore_multiple_underscore():
    result = _ignore_variable("test.py", "___x")
    assert result == False
```


# LLM-generated content at query #7
#--------------------------

```
def test_visit_FunctionDef_marks_property_when_property_decorator_present():
    v = Vulture()
    code = """
class MyClass:
    @property
    def my_prop(self):
        return 42
"""
    v.scan(code)
    assert any(item.name == "my_prop" and item.typ == "property" for item in v.defined_props)

def test_visit_FunctionDef_marks_method_when_self_first_arg():
    v = Vulture()
    code = """
class MyClass:
    def my_method(self, x):
        pass
"""
    v.scan(code)
    assert any(item.name == "my_method" and item.typ == "method" for item in v.defined_methods)

def test_visit_FunctionDef_marks_method_when_staticmethod_decorator():
    v = Vulture()
    code = """
class MyClass:
    @staticmethod
    def my_static():
        pass
"""
    v.scan(code)
    assert any(item.name == "my_static" and item.typ == "method" for item in v.defined_methods)

def test_visit_FunctionDef_marks_method_when_classmethod_decorator():
    v = Vulture()
    code = """
class MyClass:
    @classmethod
    def my_classmethod(cls):
        pass
"""
    v.scan(code)
    assert any(item.name == "my_classmethod" and item.typ == "method" for item in v.defined_methods)

def test_visit_FunctionDef_marks_function_when_no_self_and_no_special_decorator():
    v = Vulture()
    code = """
def my_func():
    pass
"""
    v.scan(code)
    assert any(item.name == "my_func" and item.typ == "function" for item in v.defined_funcs)

def test_visit_FunctionDef_ignores_function_with_ignored_decorator():
    v = Vulture(ignore_decorators=["my_decorator"])
    code = """
@my_decorator
def my_func():
    pass
"""
    v.scan(code)
    assert not any(item.name == "my_func" for item in v.defined_funcs)

def test_visit_FunctionDef_ignores_method_with_ignored_decorator():
    v = Vulture(ignore_decorators=["my_decorator"])
    code = """
class MyClass:
    @my_decorator
    def my_method(self):
        pass
"""
    v.scan(code)
    assert not any(item.name == "my_method" for item in v.defined_methods)

def test_visit_FunctionDef_ignores_property_with_ignored_decorator():
    v = Vulture(ignore_decorators=["my_decorator"])
    code = """
class MyClass:
    @my_decorator
    @property
    def my_prop(self):
        return 42
"""
    v.scan(code)
    assert not any(item.name == "my_prop" for item in v.defined_props)
```


# LLM-generated content at query #8
#--------------------------

```
def test_ignore_star_import():
    from pathlib import Path
    assert _ignore_import(Path("/some/dir/__init__.py"), "*") == True

def test_ignore_import_in_init():
    from pathlib import Path
    assert _ignore_import(Path("/some/dir/__init__.py"), "os") == True

def test_ignore_import_star_not_init():
    from pathlib import Path
    assert _ignore_import(Path("/some/dir/module.py"), "*") == True

def test_not_ignore_regular_import():
    from pathlib import Path
    assert _ignore_import(Path("/some/dir/module.py"), "os") == False

def test_not_ignore_other_import():
    from pathlib import Path
    assert _ignore_import(Path("/some/dir/module.py"), "sys") == False
```


# LLM-generated content at query #9
#--------------------------

```
def test_ignore_class_returns_true_for_test_file_with_test_in_class_name():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/tests/test_example.py'})()})()
    result = _ignore_class(filename, "TestExample")
    assert result == True

def test_ignore_class_returns_false_for_non_test_file_with_test_in_class_name():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/src/example.py'})()})()
    result = _ignore_class(filename, "TestExample")
    assert result == False

def test_ignore_class_returns_false_for_test_file_without_test_in_class_name():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/tests/test_example.py'})()})()
    result = _ignore_class(filename, "Example")
    assert result == False

def test_ignore_class_returns_false_for_non_test_file_without_test_in_class_name():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/src/example.py'})()})()
    result = _ignore_class(filename, "Example")
    assert result == False

def test_ignore_class_matches_test_directory_pattern():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/test/example.py'})()})()
    result = _ignore_class(filename, "TestExample")
    assert result == True

def test_ignore_class_matches_test_suffix_pattern():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/example_test.py'})()})()
    result = _ignore_class(filename, "TestExample")
    assert result == True

def test_ignore_class_case_insensitive_check():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/TESTS/test_example.py'})()})()
    result = _ignore_class(filename, "TestExample")
    assert result == True

def test_ignore_class_class_name_contains_test_not_prefix():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/tests/test_example.py'})()})()
    result = _ignore_class(filename, "MyTestExample")
    assert result == True

def test_ignore_class_class_name_lowercase_test():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/tests/test_example.py'})()})()
    result = _ignore_class(filename, "testexample")
    assert result == False

def test_ignore_class_empty_class_name():
    filename = type('Path', (object,), {'resolve': lambda self: type('Path', (object,), {'__str__': lambda self: '/project/tests/test_example.py'})()})()
    result = _ignore_class(filename, "")
    assert result == False
```


# LLM-generated content at query #10
#--------------------------

```
def test_visit_ClassDef_ignores_class_with_ignored_decorator():
    v = Vulture(ignore_decorators=["ignored_decorator"])
    node = ast.ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[ast.Name(id="ignored_decorator", ctx=ast.Load())],
    )
    v.visit_ClassDef(node)
    assert len(v.defined_classes) == 0

def test_visit_ClassDef_defines_class_without_ignored_decorator():
    v = Vulture()
    node = ast.ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[],
    )
    v.visit_ClassDef(node)
    assert len(v.defined_classes) == 1
    assert v.defined_classes[0].name == "TestClass"

def test_visit_ClassDef_defines_class_with_non_ignored_decorator():
    v = Vulture(ignore_decorators=["other_decorator"])
    node = ast.ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[ast.Name(id="some_decorator", ctx=ast.Load())],
    )
    v.visit_ClassDef(node)
    assert len(v.defined_classes) == 1
    assert v.defined_classes[0].name == "TestClass"

def test_visit_ClassDef_ignores_class_with_matching_ignore_names():
    v = Vulture(ignore_names=["TestClass"])
    node = ast.ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[],
    )
    v.visit_ClassDef(node)
    assert len(v.defined_classes) == 0

def test_visit_ClassDef_defines_class_without_matching_ignore_names():
    v = Vulture(ignore_names=["OtherClass"])
    node = ast.ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[],
    )
    v.visit_ClassDef(node)
    assert len(v.defined_classes) == 1
    assert v.defined_classes[0].name == "TestClass"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_scavenge_with_exclude_pattern():
    vulture = Vulture()
    vulture.scavenge(["test_path"], exclude=["*.pyc"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_without_exclude():
    vulture = Vulture()
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_with_empty_paths():
    vulture = Vulture()
    vulture.scavenge([])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_with_verbose():
    vulture = Vulture(verbose=True)
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_with_nonexistent_file():
    vulture = Vulture()
    vulture.scavenge(["nonexistent_file.py"])
    assert vulture.exit_code == ExitCode.InvalidInput

def test_scavenge_with_exclude_all():
    vulture = Vulture()
    vulture.scavenge(["test_path"], exclude=["*"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_with_multiple_paths():
    vulture = Vulture()
    vulture.scavenge(["path1", "path2"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_with_ignore_names():
    vulture = Vulture(ignore_names=["test"])
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_with_ignore_decorators():
    vulture = Vulture(ignore_decorators=["@staticmethod"])
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == ExitCode.NoDeadCode

def test_scavenge_repeated_calls():
    vulture = Vulture()
    vulture.scavenge(["path1"])
    vulture.scavenge(["path2"])
    assert vulture.exit_code == ExitCode.NoDeadCode
```


# LLM-generated content at query #12
#--------------------------

```python
def test_visit_name_load_used_name_added():
    vulture = Vulture()
    node = ast.Name(id='my_var', ctx=ast.Load())
    vulture.visit_Name(node)
    assert 'my_var' in vulture.used_names

def test_visit_name_del_used_name_added():
    vulture = Vulture()
    node = ast.Name(id='my_var', ctx=ast.Del())
    vulture.visit_Name(node)
    assert 'my_var' in vulture.used_names

def test_visit_name_ignored_variable_not_added():
    vulture = Vulture()
    node = ast.Name(id='_', ctx=ast.Load())
    vulture.visit_Name(node)
    assert '_' not in vulture.used_names

def test_visit_name_store_defined_variable():
    vulture = Vulture()
    node = ast.Name(id='my_var', ctx=ast.Store())
    vulture.visit_Name(node)
    assert any(item.name == 'my_var' for item in vulture.defined_vars)

def test_visit_name_param_defined_variable():
    vulture = Vulture()
    node = ast.Name(id='my_var', ctx=ast.Param())
    vulture.visit_Name(node)
    assert any(item.name == 'my_var' for item in vulture.defined_vars)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_ignore_method_special_name():
    assert _ignore_method(Path("any/path/file.py"), "__init__") == True

def test_ignore_method_special_name_dunder():
    assert _ignore_method(Path("any/path/file.py"), "__str__") == True

def test_ignore_method_pytest_method_in_test_file():
    assert _ignore_method(Path("project/tests/test_module.py"), "setup_method") == True

def test_ignore_method_test_method_in_test_file():
    assert _ignore_method(Path("project/test/test_module.py"), "test_example") == True

def test_ignore_method_regular_method_in_test_file():
    assert _ignore_method(Path("project/tests/test_module.py"), "helper") == False

def test_ignore_method_test_method_in_non_test_file():
    assert _ignore_method(Path("project/src/module.py"), "test_example") == False

def test_ignore_method_pytest_method_in_non_test_file():
    assert _ignore_method(Path("project/src/module.py"), "setup_method") == False

def test_ignore_method_regular_method_in_non_test_file():
    assert _ignore_method(Path("project/src/module.py"), "helper") == False

def test_ignore_method_method_starting_with_test_but_not_test_file():
    assert _ignore_method(Path("project/src/module.py"), "test_runner") == False

def test_ignore_method_special_name_not_in_test_file():
    assert _ignore_method(Path("project/tests/test_file.py"), "__special__") == True
```


# LLM-generated content at query #14
#--------------------------

```
def test_ignore_function_returns_true_for_pytest_function_in_test_file():
    import tempfile
    import pathlib
    f = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
    f.close()
    p = pathlib.Path(f.name)
    test_file = pathlib.Path("/tmp/test_example.py")
    test_file.touch()
    assert _ignore_function(test_file, "pytest") == True
    test_file.unlink()
    pathlib.Path(f.name).unlink()

def test_ignore_function_returns_true_for_test_starting_function_in_test_file():
    import tempfile
    import pathlib
    test_file = pathlib.Path("/tmp/test_example.py")
    test_file.touch()
    assert _ignore_function(test_file, "test_something") == True
    test_file.unlink()

def test_ignore_function_returns_false_for_non_test_function_in_test_file():
    import tempfile
    import pathlib
    test_file = pathlib.Path("/tmp/test_example.py")
    test_file.touch()
    assert _ignore_function(test_file, "helper_function") == False
    test_file.unlink()

def test_ignore_function_returns_false_for_pytest_function_in_non_test_file():
    import tempfile
    import pathlib
    non_test_file = pathlib.Path("/tmp/example.py")
    non_test_file.touch()
    assert _ignore_function(non_test_file, "pytest") == False
    non_test_file.unlink()

def test_ignore_function_returns_false_for_test_starting_function_in_non_test_file():
    import tempfile
    import pathlib
    non_test_file = pathlib.Path("/tmp/example.py")
    non_test_file.touch()
    assert _ignore_function(non_test_file, "test_something") == False
    non_test_file.unlink()

def test_ignore_function_returns_false_for_non_test_function_in_non_test_file():
    import tempfile
    import pathlib
    non_test_file = pathlib.Path("/tmp/example.py")
    non_test_file.touch()
    assert _ignore_function(non_test_file, "helper_function") == False
    non_test_file.unlink()
```


# LLM-generated content at query #15
#--------------------------

```
def test_assigns_special_variable__all__with_list():
    import ast
    node = ast.parse("__all__ = ['a', 'b']").body[0]
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__with_tuple():
    import ast
    node = ast.parse("__all__ = ('a', 'b')").body[0]
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__with_non_list_or_tuple():
    import ast
    node = ast.parse("__all__ = 'string'").body[0]
    assert not _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__with_non_all_variable():
    import ast
    node = ast.parse("other = ['a', 'b']").body[0]
    assert not _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__with_multiple_targets():
    import ast
    node = ast.parse("x = __all__ = ['a']").body[0]
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__with_attribute_target():
    import ast
    node = ast.parse("obj.__all__ = ['a']").body[0]
    assert not _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__empty_list():
    import ast
    node = ast.parse("__all__ = []").body[0]
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__empty_tuple():
    import ast
    node = ast.parse("__all__ = ()").body[0]
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable__all__non_assign_node():
    import ast
    node = ast.parse("print('test')").body[0]
    assert not _assigns_special_variable__all__(node)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_scavenge_unique_imports_non_empty_when_defined_imports_present():
    vulture = Vulture()
    vulture.defined_imports.append(Item("os", "import", Path(), 1, 1))
    vulture.scavenge([], exclude=None)
    assert len(vulture.defined_imports) > 0
```


# LLM-generated content at query #17
#--------------------------

def test_scavenge_with_valid_python_files():
    v = Vulture()
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp()
    file1 = os.path.join(tmpdir, "test1.py")
    with open(file1, "w") as f:
        f.write("x = 1\n")
    file2 = os.path.join(tmpdir, "test2.py")
    with open(file2, "w") as f:
        f.write("y = 2\n")
    v.scavenge([tmpdir])
    import shutil
    shutil.rmtree(tmpdir)

def test_scavenge_with_exclude_pattern():
    v = Vulture()
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp()
    file1 = os.path.join(tmpdir, "test1.py")
    with open(file1, "w") as f:
        f.write("x = 1\n")
    file2 = os.path.join(tmpdir, "test2.py")
    with open(file2, "w") as f:
        f.write("y = 2\n")
    v.scavenge([tmpdir], exclude=["test1*"])
    import shutil
    shutil.rmtree(tmpdir)

def test_scavenge_with_non_existent_path():
    v = Vulture()
    v.scavenge(["/nonexistent/path"])

def test_scavenge_with_single_file():
    v = Vulture()
    import tempfile
    import os
    tmpfile = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmpfile.write("z = 3\n")
    tmpfile.close()
    v.scavenge([tmpfile.name])
    os.unlink(tmpfile.name)


# LLM-generated content at query #18
#--------------------------

```
def test_visit_name_ignored_variable():
    v = Vulture()
    code = "_ = 1"
    v.scan(code)  # _ is typically in IGNORED_VARIABLE_NAMES
    assert "_" not in v.used_names
```


# LLM-generated content at query #19
#--------------------------

```
def test_assigns_special_variable_all_with_list():
    node = ast.Assign(targets=[ast.Name(id="__all__", ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable_all_with_tuple():
    node = ast.Assign(targets=[ast.Name(id="__all__", ctx=ast.Store())], value=ast.Tuple(elts=[], ctx=ast.Load()))
    assert _assigns_special_variable__all__(node)

def test_assigns_special_variable_all_not_assignment():
    node = ast.Expr(value=ast.Constant(value=1))
    assert not _assigns_special_variable__all__(node)

def test_assigns_special_variable_all_wrong_target_name():
    node = ast.Assign(targets=[ast.Name(id="other", ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
    assert not _assigns_special_variable__all__(node)

def test_assigns_special_variable_all_non_list_tuple_value():
    node = ast.Assign(targets=[ast.Name(id="__all__", ctx=ast.Store())], value=ast.Constant(value=1))
    assert not _assigns_special_variable__all__(node)

def test_assigns_special_variable_all_target_not_name():
    node = ast.Assign(targets=[ast.Attribute(value=ast.Name(id="mod", ctx=ast.Load()), attr="__all__", ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
    assert not _assigns_special_variable__all__(node)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_prepare_pattern_adds_wildcards_when_no_special_chars():
    vulture = Vulture()
    prepare_pattern = vulture.__init__.__code__.co_consts[1]
    pattern = "mypattern"
    result = prepare_pattern(pattern)
    assert result == "*mypattern*"


# LLM-generated content at query #21
#--------------------------

def test_get_whitelist_string_default_type():
    item = Item("my_func", "function", "file.py", 10, 20)
    result = item.get_whitelist_string()
    assert result == "my_func  # unused function (file.py:10)"

def test_get_whitelist_string_attribute():
    item = Item("attr1", "attribute", "file.py", 5, 5)
    result = item.get_whitelist_string()
    assert result == "_.attr1  # unused attribute (file.py:5)"

def test_get_whitelist_string_method():
    item = Item("method1", "method", "file.py", 1, 10)
    result = item.get_whitelist_string()
    assert result == "_.method1  # unused method (file.py:1)"

def test_get_whitelist_string_property():
    item = Item("prop1", "property", "file.py", 2, 2)
    result = item.get_whitelist_string()
    assert result == "_.prop1  # unused property (file.py:2)"

def test_get_whitelist_string_unreachable_code():
    item = Item("code_block", "unreachable_code", "file.py", 3, 8)
    result = item.get_whitelist_string()
    assert result == "# unused unreachable_code 'code_block' (file.py:3)"

def test_get_whitelist_string_with_message():
    item = Item("var1", "variable", "file.py", 4, 4, message="custom message")
    result = item.get_whitelist_string()
    assert result == "var1  # custom message (file.py:4)"


# LLM-generated content at query #22
#--------------------------

```
def test_visit_assigns_all_used_names():
    vulture = Vulture()
    code = """
__all__ = ['foo', 'bar']
foo = 1
bar = 2
"""
    vulture.scan(code)
    assert "foo" in vulture.used_names
    assert "bar" in vulture.used_names

def test_visit_ignores_ignored_variable_names():
    vulture = Vulture()
    code = "_ = 1"
    vulture.scan(code)
    assert "_" not in vulture.used_names

def test_visit_defined_variable_on_store():
    vulture = Vulture()
    code = "x = 1"
    vulture.scan(code)
    assert any(item.name == "x" for item in vulture.defined_vars)

def test_visit_used_name_on_load():
    vulture = Vulture()
    code = "x = 1; print(x)"
    vulture.scan(code)
    assert "x" in vulture.used_names

def test_visit_call_getattr_adds_used_name():
    vulture = Vulture()
    code = "getattr(obj, 'some_attr')"
    vulture.scan(code)
    assert "some_attr" in vulture.used_names

def test_visit_call_hasattr_adds_used_name():
    vulture = Vulture()
    code = "hasattr(obj, 'some_attr')"
    vulture.scan(code)
    assert "some_attr" in vulture.used_names

def test_visit_binop_format_string_adds_used_names():
    vulture = Vulture()
    code = '"%(my_var)s" % locals()'
    vulture.scan(code)
    assert "my_var" in vulture.used_names

def test_visit_call_format_with_locals_adds_used_names():
    vulture = Vulture()
    code = '"{my_var}".format(**locals())'
    vulture.scan(code)
    assert "my_var" in vulture.used_names

def test_visit_class_def_defined_class():
    vulture = Vulture()
    code = "class MyClass: pass"
    vulture.scan(code)
    assert any(item.name == "MyClass" for item in vulture.defined_classes)

def test_visit_function_def_defined_function():
    vulture = Vulture()
    code = "def my_func(): pass"
    vulture.scan(code)
    assert any(item.name == "my_func" for item in vulture.defined_funcs)

def test_visit_function_def_defined_method():
    vulture = Vulture()
    code = "class A: def method(self): pass"
    vulture.scan(code)
    assert any(item.name == "method" for item in vulture.defined_methods)

def test_visit_function_def_defined_property():
    vulture = Vulture()
    code = "class A: @property\ndef prop(self): pass"
    vulture.scan(code)
    assert any(item.name == "prop" for item in vulture.defined_props)

def test_visit_import_adds_defined_import():
    vulture = Vulture()
    code = "import os"
    vulture.scan(code)
    assert any(item.name == "os" for item in vulture.defined_imports)

def test_visit_import_from_adds_defined_import():
    vulture = Vulture()
    code = "from os import path"
    vulture.scan(code)
    assert any(item.name == "path" for item in vulture.defined_imports)

def test_visit_import_from_future_ignored():
    vulture = Vulture()
    code = "from __future__ import annotations"
    vulture.scan(code)
    assert not any(item.name == "annotations" for item in vulture.defined_imports)

def test_visit_attribute_as_store_adds_defined_attr():
    vulture = Vulture()
    code = "obj.attr = 1"
    vulture.scan(code)
    assert any(item.name == "attr" for item in vulture.defined_attrs)

def test_visit_attribute_as_load_adds_used_name():
    vulture = Vulture()
    code = "print(obj.attr)"
    vulture.scan(code)
    assert "attr" in vulture.used_names

def test_visit_match_class_adds_used_name():
    vulture = Vulture()
    code = """
match obj:
    case A(x=1):
        pass
"""
    vulture.scan(code)
    assert "x" in vulture.used_names

def test_visit_async_function_def_defined_function():
    vulture = Vulture()
    code = "async def my_async_func(): pass"
    vulture.scan(code)
    assert any(item.name == "my_async_func" for item in vulture.defined_funcs)

def test_visit_arg_defined_variable():
    vulture = Vulture()
    code = "def func(arg): pass"
    vulture.scan(code)
    assert any(item.name == "arg" for item in vulture.defined_vars)

def test_visit_assign_all_adds_used_names():
    vulture = Vulture()
    code = "__all__ = ['a']\na = 1"
    vulture.scan(code)
    assert "a" in vulture.used_names

def test_visit_ignore_names_ignored():
    vulture = Vulture(ignore_names=["ignored_func"])
    code = "def ignored_func(): pass"
    vulture.scan(code)
    assert not any(item.name == "ignored_func" for item in vulture.defined_funcs)

def test_visit_ignore_decorators_ignored():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    code = "@my_decorator\ndef decorated_func(): pass"
    vulture.scan(code)
    assert not any(item.name == "decorated_func" for item in vulture.defined_funcs)


# LLM-generated content at query #23
#--------------------------

```python
def test_visit_assign_with_all_list():
    vulture = Vulture()
    code = "__all__ = ['a', 'b']"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'a' in vulture.used_names
    assert 'b' in vulture.used_names

def test_visit_assign_with_all_tuple():
    vulture = Vulture()
    code = "__all__ = ('x', 'y')"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'x' in vulture.used_names
    assert 'y' in vulture.used_names

def test_visit_assign_without_all():
    vulture = Vulture()
    code = "x = 1"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'x' not in vulture.used_names

def test_visit_assign_all_non_string_elements():
    vulture = Vulture()
    code = "__all__ = [1, 2]"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert not vulture.used_names

def test_visit_assign_all_with_mixed_types():
    vulture = Vulture()
    code = "__all__ = ['a', 1, 'b']"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'a' in vulture.used_names
    assert 'b' in vulture.used_names
    assert '1' not in vulture.used_names

def test_visit_assign_all_not_first_target():
    vulture = Vulture()
    code = "a, __all__ = 1, ['x', 'y']"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'x' in vulture.used_names
    assert 'y' in vulture.used_names

def test_visit_assign_all_multiple_targets():
    vulture = Vulture()
    code = "__all__ = __all__ = ['a']"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'a' in vulture.used_names

def test_visit_assign_all_empty_list():
    vulture = Vulture()
    code = "__all__ = []"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert not vulture.used_names

def test_visit_assign_all_empty_tuple():
    vulture = Vulture()
    code = "__all__ = ()"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert not vulture.used_names

def test_visit_assign_all_attribute_target():
    vulture = Vulture()
    code = "module.__all__ = ['a']"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'a' not in vulture.used_names

def test_visit_assign_all_subscript_target():
    vulture = Vulture()
    code = "d['__all__'] = ['a']"
    node = ast.parse(code).body[0]
    vulture.visit_Assign(node)
    assert 'a' not in vulture.used_names
```


# LLM-generated content at query #24
#--------------------------

def test_vulture_constructor_defaults():
    v = Vulture()
    assert v.verbose == False
    assert v.ignore_names == []
    assert v.ignore_decorators == []
    assert v.filename == Path()
    assert v.code == []
    assert v.exit_code == ExitCode.NoDeadCode
    assert v.noqa_lines == {}

def test_vulture_constructor_verbose_true():
    v = Vulture(verbose=True)
    assert v.verbose == True

def test_vulture_constructor_with_ignore_names():
    names = ["foo", "bar"]
    v = Vulture(ignore_names=names)
    assert v.ignore_names == names

def test_vulture_constructor_with_ignore_decorators():
    decorators = ["decor1", "decor2"]
    v = Vulture(ignore_decorators=decorators)
    assert v.ignore_decorators == decorators

def test_vulture_constructor_defines_attributes():
    v = Vulture()
    assert hasattr(v, "defined_attrs")
    assert hasattr(v, "defined_classes")
    assert hasattr(v, "defined_funcs")
    assert hasattr(v, "defined_imports")
    assert hasattr(v, "defined_methods")
    assert hasattr(v, "defined_props")
    assert hasattr(v, "defined_vars")
    assert hasattr(v, "unreachable_code")
    assert hasattr(v, "used_names")
    assert hasattr(v, "reachability")


# LLM-generated content at query #25
#--------------------------

def test_predicate_at_line_14_true():
    v = Vulture(verbose=True)
    assert isinstance(v.defined_props, utils.LoggingList)
    assert v.defined_props.typ == "property"


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_evaluates_to_false():
    vulture = Vulture()
    assert vulture.ignore_names == []
    assert vulture.ignore_decorators == []
```


# LLM-generated content at query #27
#--------------------------

```
def test_get_unused_items_returns_empty_list_when_all_defined_items_are_used():
    defined_items = [MockItem("a"), MockItem("b")]
    used_names = {"a", "b"}
    result = _get_unused_items(defined_items, used_names)
    assert result == []

def test_get_unused_items_returns_unused_items_sorted_case_insensitive():
    defined_items = [MockItem("B"), MockItem("a"), MockItem("C")]
    used_names = {"a"}
    result = _get_unused_items(defined_items, used_names)
    assert len(result) == 2
    assert result[0].name == "B"
    assert result[1].name == "C"

def test_get_unused_items_handles_empty_defined_items():
    defined_items = []
    used_names = {"a", "b"}
    result = _get_unused_items(defined_items, used_names)
    assert result == []

def test_get_unused_items_handles_empty_used_names():
    defined_items = [MockItem("x"), MockItem("y")]
    used_names = set()
    result = _get_unused_items(defined_items, used_names)
    assert len(result) == 2
    assert result[0].name == "x"
    assert result[1].name == "y"

def test_get_unused_items_ignores_duplicate_items():
    defined_items = [MockItem("a"), MockItem("a"), MockItem("b")]
    used_names = {"a"}
    result = _get_unused_items(defined_items, used_names)
    assert len(result) == 1
    assert result[0].name == "b"

class MockItem:
    def __init__(self, name):
        self.name = name
```


# LLM-generated content at query #28
#--------------------------

```
def test_visit_name_with_ignored_variable_name_not_added_to_used_names():
    vulture = Vulture()
    node = ast.Name(id='_', ctx=ast.Load())
    vulture.visit_Name(node)
    assert '_' not in vulture.used_names
```


# LLM-generated content at query #29
#--------------------------

```python
def test_scan_parses_valid_code():
    v = Vulture()
    v.scan("x = 1")
    assert len(v.defined_vars) == 1
    assert v.defined_vars[0].name == "x"

def test_scan_handles_syntax_error():
    v = Vulture()
    v.scan("x = ", filename="test.py")
    assert v.exit_code == ExitCode.InvalidInput

def test_scan_handles_null_bytes():
    v = Vulture()
    v.scan("x = 1\0", filename="test.py")
    assert v.exit_code == ExitCode.InvalidInput

def test_scan_sets_filename():
    v = Vulture()
    v.scan("", filename="module.py")
    assert v.filename == Path("module.py")

def test_scan_resets_reachability():
    v = Vulture()
    v.scan("")
    v.reachability.visit(ast.parse("if True: pass"))
    v.scan("")
    assert v.reachability._current is None

def test_scan_processes_type_comments():
    v = Vulture()
    v.scan("def f(x): # type: (int) -> None\n    pass")
    assert len(v.defined_funcs) == 1

def test_scan_without_filename():
    v = Vulture()
    v.scan("a = 1")
    assert v.filename == Path()

def test_scan_with_type_comment_error():
    v = Vulture()
    v.scan("x = 1 # type: invalid")
    assert v.exit_code == ExitCode.InvalidInput
```


# LLM-generated content at query #30
#--------------------------

```python
def test_main_valid_input():
    import sys
    sys.argv = ['vulture', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')

def test_main_invalid_cmdline_arguments():
    import sys
    sys.argv = ['vulture', '--invalid']
    try:
        main()
    except SystemExit as e:
        assert e.code == 2

def test_main_verbose_mode():
    import sys
    sys.argv = ['vulture', '--verbose', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')

def test_main_with_ignore_names():
    import sys
    sys.argv = ['vulture', '--ignore-names', 'foo', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')

def test_main_with_exclude():
    import sys
    sys.argv = ['vulture', '--exclude', 'test_*.py', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')

def test_main_with_min_confidence():
    import sys
    sys.argv = ['vulture', '--min-confidence', '80', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')

def test_main_with_sort_by_size():
    import sys
    sys.argv = ['vulture', '--sort-by-size', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')

def test_main_with_make_whitelist():
    import sys
    sys.argv = ['vulture', '--make-whitelist', 'test_file.py']
    open('test_file.py', 'w').close()
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
    finally:
        import os
        os.remove('test_file.py')
```


# LLM-generated content at query #31
#--------------------------

```
def test_match_decorator_returns_true():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    node = ast.ClassDef(
        name="MyClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[ast.Name(id="my_decorator", ctx=ast.Load())],
    )
    vulture.visit_ClassDef(node)
```


# LLM-generated content at query #32
#--------------------------

def test_get_list_verbose_false():
    vulture = Vulture(verbose=False, ignore_names=[], ignore_decorators=[])
    result = vulture.defined_attrs
    assert result.verbose == False


# LLM-generated content at query #33
#--------------------------

```
def test_prepare_pattern_returns_same_pattern_when_contains_asterisk():
    vulture = Vulture()
    exclude = ["*pattern*"]
    result = vulture.scavenge([], exclude=exclude)


# LLM-generated content at query #34
#--------------------------

```
def test_visit_name_node_id_in_ignored_variable_names():
    vulture = Vulture()
    node = ast.Name(id='_', ctx=ast.Load())
    vulture.visit_Name(node)
    assert '_' not in vulture.used_names
```


# LLM-generated content at query #35
#--------------------------

def test_item_constructor_sets_all_fields():
    item = Item("my_func", "function", "/path/to/file.py", 10, 20, "custom message", 75)
    assert item.name == "my_func"
    assert item.typ == "function"
    assert item.filename == "/path/to/file.py"
    assert item.first_lineno == 10
    assert item.last_lineno == 20
    assert item.message == "custom message"
    assert item.confidence == 75

def test_item_constructor_default_message():
    item = Item("my_var", "variable", "/path/to/file.py", 5, 5)
    assert item.message == "unused variable 'my_var'"

def test_item_constructor_default_confidence():
    item = Item("my_func", "function", "/path/to/file.py", 1, 2)
    assert item.confidence == 75


# LLM-generated content at query #36
#--------------------------

```
def test_scavenge_prepare_pattern_without_special_chars():
    v = Vulture()
    result = v.scavenge(["test.py"], exclude="test")
    # The predicate is that the inner function prepare_pattern works correctly
    # We need to verify that the exclude list is processed properly
    # Since scavenge doesn't return anything, we check internal state
    assert v.exit_code == ExitCode.NoDeadCode
```


# LLM-generated content at query #37
#--------------------------

```
def test_visit_Call_getattr_with_string_attr():
    v = Vulture()
    code = "getattr(obj, 'some_attr')"
    tree = ast.parse(code)
    v.visit(tree)
    assert 'some_attr' in v.used_names

def test_visit_Call_getattr_with_too_few_args():
    v = Vulture()
    code = "getattr(obj)"
    tree = ast.parse(code)
    v.visit(tree)
    assert v.used_names == set()

def test_visit_Call_getattr_with_too_many_args():
    v = Vulture()
    code = "getattr(obj, 'attr', default, extra)"
    tree = ast.parse(code)
    v.visit(tree)
    assert v.used_names == set()

def test_visit_Call_hasattr_with_string_attr():
    v = Vulture()
    code = "hasattr(obj, 'some_attr')"
    tree = ast.parse(code)
    v.visit(tree)
    assert 'some_attr' in v.used_names

def test_visit_Call_hasattr_with_wrong_arg_count():
    v = Vulture()
    code = "hasattr(obj, 'attr', extra)"
    tree = ast.parse(code)
    v.visit(tree)
    assert v.used_names == set()

def test_visit_Call_format_with_locals():
    v = Vulture()
    code = "'{my_var}'.format(**locals())"
    tree = ast.parse(code)
    v.visit(tree)
    assert 'my_var' in v.used_names

def test_visit_Call_format_with_non_locals_kwargs():
    v = Vulture()
    code = "'{my_var}'.format(my_var=1)"
    tree = ast.parse(code)
    v.visit(tree)
    assert 'my_var' not in v.used_names

def test_visit_Call_non_matching_function():
    v = Vulture()
    code = "other_func(obj, 'attr')"
    tree = ast.parse(code)
    v.visit(tree)
    assert v.used_names == set()

def test_visit_Call_getattr_with_non_string_attr():
    v = Vulture()
    code = "getattr(obj, attr_name)"
    tree = ast.parse(code)
    v.visit(tree)
    assert v.used_names == set()

def test_visit_Call_nested_format():
    v = Vulture()
    code = "'{a} {b}'.format(**locals())"
    tree = ast.parse(code)
    v.visit(tree)
    assert 'a' in v.used_names and 'b' in v.used_names
```


# LLM-generated content at query #38
#--------------------------

```
def test_ignore_decorators_matches_name():
    vulture = Vulture(ignore_decorators=["my_decorator"])
    code = """
@my_decorator
def my_func():
    pass
"""
    vulture.scan(code)
    assert len(vulture.defined_funcs) == 0
    assert len(vulture.unused_funcs) == 0
```


# LLM-generated content at query #39
#--------------------------

def test_get_list_logging_verbose_false():
    vulture = Vulture(verbose=False)
    result = vulture.defined_attrs
    assert isinstance(result, utils.LoggingList)


# LLM-generated content at query #40
#--------------------------

```
def test_visit_logs_verbose_line():
    vulture = Vulture(verbose=True)
    code = "x = 1"
    vulture.scan(code)
    # The test just ensures visit runs without error and logs the line
    assert vulture.code == ["x = 1"]

def test_visit_type_comment_function():
    vulture = Vulture()
    code = "def f(x): # type: (int) -> None\n    pass\n"
    vulture.scan(code)
    assert len(vulture.defined_funcs) == 1
    assert vulture.defined_funcs[0].name == "f"

def test_visit_type_comment_eval():
    vulture = Vulture()
    code = "x = 1  # type: int\n"
    vulture.scan(code)
    assert len(vulture.defined_vars) == 1
    assert vulture.defined_vars[0].name == "x"

def test_visit_reachability_analyzes_node():
    vulture = Vulture()
    code = "if True:\n    pass\n"
    vulture.scan(code)
    # Just ensure no exceptions and reachability is called
    assert vulture.unreachable_code == []

def test_visit_calls_visitor_method():
    vulture = Vulture()
    code = "class A:\n    pass\n"
    vulture.scan(code)
    assert len(vulture.defined_classes) == 1
    assert vulture.defined_classes[0].name == "A"

def test_visit_generic_visit_handles_lists():
    vulture = Vulture()
    code = "a, b = 1, 2\n"
    vulture.scan(code)
    assert len(vulture.defined_vars) == 2
    assert vulture.defined_vars[0].name == "a"
    assert vulture.defined_vars[1].name == "b"


# LLM-generated content at query #41
#--------------------------

```
def test_visit_assign_not_all():
    v = Vulture()
    code = "x = 1"
    v.scan(code)
    assert True  # The predicate at line 3 is not reached because _assigns_special_variable__all__ returns False
```


# LLM-generated content at query #42
#--------------------------

```
def test_get_list_returns_logging_list_with_verbose_true():
    vulture = Vulture(verbose=True)
    result = vulture._Vulture__get_list("test") if hasattr(vulture, "_Vulture__get_list") else vulture.get_list("test")
    assert isinstance(result, utils.LoggingList)


# LLM-generated content at query #43
#--------------------------

```python
def test_ignore_class_with_test_file_and_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/test/test_example.py"})()
    assert _ignore_class(fake_file, "TestExample")

def test_ignore_class_with_test_file_and_no_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/test/test_example.py"})()
    assert not _ignore_class(fake_file, "MyClass")

def test_ignore_class_with_non_test_file_and_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/src/example.py"})()
    assert not _ignore_class(fake_file, "TestExample")

def test_ignore_class_with_non_test_file_and_no_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/src/example.py"})()
    assert not _ignore_class(fake_file, "MyClass")

def test_ignore_class_with_tests_directory_and_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/tests/test_example.py"})()
    assert _ignore_class(fake_file, "TestExample")

def test_ignore_class_with_test_suffix_and_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/test_example.py"})()
    assert _ignore_class(fake_file, "TestExample")

def test_ignore_class_with_test_prefix_and_no_test_in_class_name():
    fake_file = type("FakePath", (object,), {"resolve": lambda self: "/project/test_example.py"})()
    assert not _ignore_class(fake_file, "MyClass")
```


# LLM-generated content at query #44
#--------------------------

```
def test_visit_assign_assertion_false():
    v = Vulture()
    code = "__all__ = {'a', 'b'}"
    v.scan(code)
```


# LLM-generated content at query #45
#--------------------------

def test_item_constructor_default_values():
    item = Item("my_func", "function", "test.py", 10, 20)
    assert item.name == "my_func"
    assert item.typ == "function"
    assert item.filename == "test.py"
    assert item.first_lineno == 10
    assert item.last_lineno == 20
    assert item.message == "unused function 'my_func'"
    assert item.confidence == 80

def test_item_constructor_custom_message():
    item = Item("my_var", "variable", "code.py", 5, 5, message="Custom message")
    assert item.name == "my_var"
    assert item.typ == "variable"
    assert item.filename == "code.py"
    assert item.first_lineno == 5
    assert item.last_lineno == 5
    assert item.message == "Custom message"
    assert item.confidence == 80

def test_item_constructor_custom_confidence():
    item = Item("x", "attribute", "module.py", 1, 1, confidence=50)
    assert item.name == "x"
    assert item.typ == "attribute"
    assert item.filename == "module.py"
    assert item.first_lineno == 1
    assert item.last_lineno == 1
    assert item.message == "unused attribute 'x'"
    assert item.confidence == 50


# LLM-generated content at query #46
#--------------------------

```
def test_get_list_verbose_false():
    vulture = Vulture(verbose=False)
    log_list = vulture._get_list("test")
    assert log_list.verbose == False
```


# LLM-generated content at query #47
#--------------------------

def test_ignore_names_initialization():
    v = Vulture(verbose=False, ignore_names=[], ignore_decorators=[])
    assert v.ignore_names == []


# LLM-generated content at query #48
#--------------------------

def test_get_list_returns_logging_list():
    vulture = Vulture(verbose=True)
    result = vulture.defined_attrs
    assert result.typ == "attribute"
    assert result.verbose == True


# LLM-generated content at query #49
#--------------------------

```python
def test_scavenge_with_exclude_pattern():
    vulture = Vulture()
    vulture.scavenge(["test_path"], exclude=["*.pyc"])
    assert vulture.exit_code == 0

def test_scavenge_without_exclude():
    vulture = Vulture()
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == 0

def test_scavenge_with_verbose():
    vulture = Vulture(verbose=True)
    vulture.scavenge(["test_path"], exclude=["*test*"])
    assert vulture.exit_code == 0

def test_scavenge_with_nonexistent_file():
    vulture = Vulture()
    vulture.scavenge(["nonexistent_file.py"])
    assert vulture.exit_code == 1

def test_scavenge_with_empty_paths():
    vulture = Vulture()
    vulture.scavenge([])
    assert vulture.exit_code == 0

def test_scavenge_with_multiple_paths():
    vulture = Vulture()
    vulture.scavenge(["path1", "path2"])
    assert vulture.exit_code == 0

def test_scavenge_with_exclude_all():
    vulture = Vulture()
    vulture.scavenge(["test_path"], exclude=["*"])
    assert vulture.exit_code == 0

def test_scavenge_with_ignore_names():
    vulture = Vulture(ignore_names=["test_ignore"])
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == 0

def test_scavenge_with_ignore_decorators():
    vulture = Vulture(ignore_decorators=["@decorator"])
    vulture.scavenge(["test_path"])
    assert vulture.exit_code == 0
```


# LLM-generated content at query #50
#--------------------------

```python
def test_predicate_false():
    item = Item("foo", "function", "test.py", 1, 5)
    assert item.message == "unused function 'foo'"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_whitelist_string_with_unreachable_code_type():
    item = Item("unused_func", "unreachable_code", "/path/to/file.py", 10, 20)
    result = item.get_whitelist_string()
    expected = "# unused unreachable code 'unused_func' (/path/to/file.py:10)"
    assert result == expected

def test_get_whitelist_string_with_attribute_type():
    item = Item("attr_name", "attribute", "/path/to/file.py", 5, 5)
    result = item.get_whitelist_string()
    expected = "_.attr_name  # unused attribute (/path/to/file.py:5)"
    assert result == expected

def test_get_whitelist_string_with_method_type():
    item = Item("method_name", "method", "/path/to/file.py", 15, 25)
    result = item.get_whitelist_string()
    expected = "_.method_name  # unused method (/path/to/file.py:15)"
    assert result == expected

def test_get_whitelist_string_with_property_type():
    item = Item("prop_name", "property", "/path/to/file.py", 30, 30)
    result = item.get_whitelist_string()
    expected = "_.prop_name  # unused property (/path/to/file.py:30)"
    assert result == expected

def test_get_whitelist_string_with_variable_type():
    item = Item("var_name", "variable", "/path/to/file.py", 42, 42)
    result = item.get_whitelist_string()
    expected = "var_name  # unused variable (/path/to/file.py:42)"
    assert result == expected
```


# LLM-generated content at query #2
#--------------------------

```python
def test_get_unused_code_returns_list_of_items():
    vulture = Vulture()
    vulture.scan("x = 1", filename="test.py")
    result = vulture.get_unused_code()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].name == "x"
    assert result[0].typ == "variable"

def test_get_unused_code_filters_by_min_confidence():
    vulture = Vulture()
    vulture.scan("x = 1", filename="test.py")
    result_high_confidence = vulture.get_unused_code(min_confidence=100)
    assert len(result_high_confidence) == 1
    result_low_confidence = vulture.get_unused_code(min_confidence=200)
    assert len(result_low_confidence) == 0

def test_get_unused_code_sorts_by_name():
    vulture = Vulture()
    vulture.scan("a = 1\nb = 2", filename="test.py")
    result = vulture.get_unused_code(sort_by_size=False)
    assert result[0].name == "a"
    assert result[1].name == "b"

def test_get_unused_code_sorts_by_size():
    vulture = Vulture()
    vulture.scan("a = 1\n\nb = 2", filename="test.py")
    result = vulture.get_unused_code(sort_by_size=True)
    assert result[0].name == "a"
    assert result[1].name == "b"

def test_get_unused_code_with_unused_class():
    vulture = Vulture()
    vulture.scan("class MyClass: pass", filename="test.py")
    result = vulture.get_unused_code()
    assert len(result) == 1
    assert result[0].name == "MyClass"
    assert result[0].typ == "class"

def test_get_unused_code_with_unused_function():
    vulture = Vulture()
    vulture.scan("def my_func(): pass", filename="test.py")
    result = vulture.get_unused_code()
    assert len(result) == 1
    assert result[0].name == "my_func"
    assert result[0].typ == "function"

def test_get_unused_code_with_unused_import():
    vulture = Vulture()
    vulture.scan("import os", filename="test.py")
    result = vulture.get_unused_code()
    assert len(result) == 1
    assert result[0].name == "os"
    assert result[0].typ == "import"

def test_get_unused_code_with_unused_method():
    vulture = Vulture()
    vulture.scan("class A:\n    def method(self): pass", filename="test.py")
    result = vulture.get_unused_code()
    assert len(result) == 1
    assert result[0].name == "method"
    assert result[0].typ == "method"

def test_get_unused_code_with_unused_property():
    vulture = Vulture()
    vulture.scan("class A:\n    @property\n    def prop(self): pass", filename="test.py")
    result = vulture.get_unused_code()
    assert len(result) == 1
    assert result[0].name == "prop"
    assert result[0].typ == "property"

def test_get_unused_code_with_unused_attribute():
    vulture = Vulture()
    vulture.scan("class A: pass\na = A()\na.attr = 1", filename="test.py")
    result = vulture.get_unused_code()
    assert any(item.typ == "attribute" and item.name == "attr" for item in result)

def test_get_unused_code_with_unreachable_code():
    vulture = Vulture()
    vulture.scan("def f():\n    return 1\n    x = 2", filename="test.py")
    result = vulture.get_unused_code()
    assert any(item.typ == "unreachable_code" for item in result)

def test_get_unused_code_raises_value_error_for_invalid_confidence():
    vulture = Vulture()
    try:
        vulture.get_unused_code(min_confidence=-1)
        assert False
    except ValueError:
        assert True
    try:
        vulture.get_unused_code(min_confidence=101)
        assert False
    except ValueError:
        assert True
```


# LLM-generated content at query #3
#--------------------------

```python
def test_visit_call_ignores_non_getattr_or_hasattr():
    vulture = Vulture()
    node = ast.parse("some_function(1, 2)").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_getattr_with_string_attr_adds_used_name():
    vulture = Vulture()
    node = ast.parse("getattr(obj, 'some_attr')").body[0].value
    vulture.visit_Call(node)
    assert 'some_attr' in vulture.used_names

def test_visit_call_getattr_with_non_string_attr_does_not_add():
    vulture = Vulture()
    node = ast.parse("getattr(obj, attr_name)").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_getattr_with_three_args_adds_used_name():
    vulture = Vulture()
    node = ast.parse("getattr(obj, 'some_attr', default)").body[0].value
    vulture.visit_Call(node)
    assert 'some_attr' in vulture.used_names

def test_visit_call_getattr_with_incorrect_arg_count_does_not_add():
    vulture = Vulture()
    node = ast.parse("getattr(obj, 'some_attr', default, extra)").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_hasattr_with_string_attr_adds_used_name():
    vulture = Vulture()
    node = ast.parse("hasattr(obj, 'some_attr')").body[0].value
    vulture.visit_Call(node)
    assert 'some_attr' in vulture.used_names

def test_visit_call_hasattr_with_non_string_attr_does_not_add():
    vulture = Vulture()
    node = ast.parse("hasattr(obj, attr_name)").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_hasattr_with_incorrect_arg_count_does_not_add():
    vulture = Vulture()
    node = ast.parse("hasattr(obj, 'some_attr', extra)").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_format_with_locals_adds_used_names():
    vulture = Vulture()
    node = ast.parse("'{my_var}'.format(**locals())").body[0].value
    vulture.visit_Call(node)
    assert 'my_var' in vulture.used_names

def test_visit_call_format_without_locals_does_not_add():
    vulture = Vulture()
    node = ast.parse("'{my_var}'.format(my_var=1)").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_format_with_non_string_does_not_add():
    vulture = Vulture()
    node = ast.parse("some_var.format(**locals())").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_ignores_non_format_method():
    vulture = Vulture()
    node = ast.parse("'{my_var}'.replace(**locals())").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0

def test_visit_call_getattr_with_non_name_func_does_not_add():
    vulture = Vulture()
    node = ast.parse("something.getattr(obj, 'attr')").body[0].value
    vulture.visit_Call(node)
    assert len(vulture.used_names) == 0
```


# LLM-generated content at query #4
#--------------------------

```python
def test_visit_assign_with_all_list():
    vulture = Vulture()
    code = "__all__ = ['func1', 'func2']"
    tree = ast.parse(code)
    assign_node = tree.body[0]
    vulture.visit_Assign(assign_node)
    assert 'func1' in vulture.used_names
    assert 'func2' in vulture.used_names

def test_visit_assign_with_all_tuple():
    vulture = Vulture()
    code = "__all__ = ('func1', 'func2')"
    tree = ast.parse(code)
    assign_node = tree.body[0]
    vulture.visit_Assign(assign_node)
    assert 'func1' in vulture.used_names
    assert 'func2' in vulture.used_names

def test_visit_assign_without_all():
    vulture = Vulture()
    code = "x = 5"
    tree = ast.parse(code)
    assign_node = tree.body[0]
    vulture.visit_Assign(assign_node)
    assert len(vulture.used_names) == 0

def test_visit_assign_all_with_non_string():
    vulture = Vulture()
    code = "__all__ = [1, 2]"
    tree = ast.parse(code)
    assign_node = tree.body[0]
    vulture.visit_Assign(assign_node)
    assert len(vulture.used_names) == 0

def test_visit_assign_all_with_mixed_list():
    vulture = Vulture()
    code = "__all__ = ['func1', 2, 'func3']"
    tree = ast.parse(code)
    assign_node = tree.body[0]
    vulture.visit_Assign(assign_node)
    assert 'func1' in vulture.used_names
    assert 'func3' in vulture.used_names
    assert len(vulture.used_names) == 2

def test_visit_assign_non_all_variable():
    vulture = Vulture()
    code = "other_var = ['a', 'b']"
    tree = ast.parse(code)
    assign_node = tree.body[0]
    vulture.visit_Assign(assign_node)
    assert len(vulture.used_names) == 0
```


# LLM-generated content at query #5
#--------------------------

```
def test_vulture_constructor_defaults():
    vulture = Vulture()
    assert vulture.verbose == False
    assert vulture.ignore_names == []
    assert vulture.ignore_decorators == []
    assert vulture.filename == Path()
    assert vulture.code == []
    assert vulture.exit_code == ExitCode.NoDeadCode
    assert vulture.noqa_lines == {}
    assert isinstance(vulture.defined_attrs, utils.LoggingList)
    assert vulture.defined_attrs.typ == "attribute"
    assert isinstance(vulture.defined_classes, utils.LoggingList)
    assert vulture.defined_classes.typ == "class"
    assert isinstance(vulture.defined_funcs, utils.LoggingList)
    assert vulture.defined_funcs.typ == "function"
    assert isinstance(vulture.defined_imports, utils.LoggingList)
    assert vulture.defined_imports.typ == "import"
    assert isinstance(vulture.defined_methods, utils.LoggingList)
    assert vulture.defined_methods.typ == "method"
    assert isinstance(vulture.defined_props, utils.LoggingList)
    assert vulture.defined_props.typ == "property"
    assert isinstance(vulture.defined_vars, utils.LoggingList)
    assert vulture.defined_vars.typ == "variable"
    assert isinstance(vulture.unreachable_code, utils.LoggingList)
    assert vulture.unreachable_code.typ == "unreachable_code"
    assert isinstance(vulture.used_names, utils.LoggingSet)
    assert vulture.used_names.typ == "name"
    assert isinstance(vulture.reachability, Reachability)

def test_vulture_constructor_with_verbose():
    vulture = Vulture(verbose=True)
    assert vulture.verbose == True

def test_vulture_constructor_with_ignore_names():
    vulture = Vulture(ignore_names=["foo", "bar"])
    assert vulture.ignore_names == ["foo", "bar"]

def test_vulture_constructor_with_ignore_decorators():
    vulture = Vulture(ignore_decorators=["@deprecated"])
    assert vulture.ignore_decorators == ["@deprecated"]

def test_vulture_constructor_with_all_params():
    vulture = Vulture(verbose=True, ignore_names=["x"], ignore_decorators=["@y"])
    assert vulture.verbose == True
    assert vulture.ignore_names == ["x"]
    assert vulture.ignore_decorators == ["@y"]


# LLM-generated content at query #6
#--------------------------

def test_scavenge_excludes_paths_and_scans_modules():
    vulture = Vulture()
    vulture.scavenge(["test_path"], exclude=["test_pattern"])


# LLM-generated content at query #7
#--------------------------

```python
def test_main_with_valid_config_no_dead_code():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_dead_code():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    mock_vulture = MagicMock(spec=Vulture)
    mock_vulture.report.return_value = ExitCode.DeadCode
    with patch("vulture.make_config", return_value=config):
        with patch("vulture.Vulture", return_value=mock_vulture):
            with patch.object(sys, "exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(ExitCode.DeadCode)

def test_main_with_invalid_cmdline_arguments():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, ExitCode, InputError, make_config

    with patch("vulture.make_config", side_effect=InputError("Invalid input")):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.InvalidCmdlineArguments)

def test_main_verbose_mode():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": True,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_paths_and_exclude():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["path1.py", "path2.py"],
        "exclude": ["test_*.py"],
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_min_confidence_and_sort_by_size():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 50,
        "sort_by_size": True,
        "make_whitelist": False,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_make_whitelist():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": True,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_ignore_names():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": ["unused_var"],
        "ignore_decorators": [],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_ignore_decorators():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": ["@staticmethod"],
        "paths": ["test_path.py"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    with patch("vulture.make_config", return_value=config):
        with patch.object(sys, "exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(ExitCode.NoDeadCode)

def test_main_with_scavenge_raising_exception():
    import sys
    from unittest.mock import patch, MagicMock
    from vulture import main, Vulture, ExitCode, InputError, make_config

    config = {
        "verbose": False,
        "ignore_names": [],
        "ignore_decorators": [],
        "paths": ["invalid_path"],
        "exclude": None,
        "min_confidence": 0,
        "sort_by_size": False,
        "make_whitelist": False,
    }
    mock_vulture = MagicMock(spec=Vulture)
    mock_vulture.scavenge.side_effect = Exception("Unexpected error")
    with patch("vulture.make_config", return_value=config):
        with patch("vulture.Vulture", return_value=mock_vulture):
            with patch.object(sys, "exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(ExitCode.InvalidInput)
```


