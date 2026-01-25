####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_class_api():
    """Unit test for Parser.class_api method."""
    from ast import parse, ClassDef
    
    # Test basic class with bases
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    
    script = """
class BaseClass:
    pass

class DerivedClass(BaseClass):
    x: int
    y: str = "default"
    _private: float
    
    def method(self):
        pass
"""
    
    tree = parse(script)
    derived_class = tree.body[1]
    
    parser.class_api(root, "test_module.DerivedClass", derived_class.bases, derived_class.body)
    
    assert "test_module.DerivedClass" in parser.doc
    doc_content = parser.doc["test_module.DerivedClass"]
    assert "BaseClass" in doc_content or "Bases" in doc_content
    

def test_Parser_class_api_with_enum():
    """Test Parser.class_api with enum class."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {"test_module.enum": "enum"}
    
    script = """
import enum

class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
"""
    
    tree = parse(script)
    color_class = tree.body[1]
    
    parser.class_api(root, "test_module.Color", color_class.bases, color_class.body)
    
    assert "test_module.Color" in parser.doc
    doc_content = parser.doc["test_module.Color"]
    assert "Enums" in doc_content or "RED" in doc_content


def test_Parser_class_api_with_members():
    """Test Parser.class_api with class members."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    
    script = """
class MyClass:
    count: int
    name: str
    _internal: bool
    value: float = 3.14
"""
    
    tree = parse(script)
    my_class = tree.body[0]
    
    parser.class_api(root, "test_module.MyClass", my_class.bases, my_class.body)
    
    assert "test_module.MyClass" in parser.doc
    doc_content = parser.doc["test_module.MyClass"]
    assert "count" in doc_content or "Members" in doc_content


def test_Parser_class_api_with_deleted_members():
    """Test Parser.class_api with deleted members."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    
    script = """
class MyClass:
    x: int
    y: str
    del x
"""
    
    tree = parse(script)
    my_class = tree.body[0]
    
    parser.class_api(root, "test_module.MyClass", my_class.bases, my_class.body)
    
    assert "test_module.MyClass" in parser.doc
    doc_content = parser.doc["test_module.MyClass"]
    # y should be in members, but x should be deleted
    assert "y" in doc_content or "Members" in doc_content


def test_Parser_class_api_empty_class():
    """Test Parser.class_api with empty class."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    
    script = "class EmptyClass: pass"
    
    tree = parse(script)
    empty_class = tree.body[0]
    
    parser.class_api(root, "test_module.EmptyClass", empty_class.bases, empty_class.body)
    
    assert "test_module.EmptyClass" in parser.doc


def test_Parser_class_api_with_type_comments():
    """Test Parser.class_api with type comments."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    
    script = """
class MyClass:
    value = 42  # type: int
    name = "test"  # type: str
"""
    
    tree = parse(script)
    my_class = tree.body[0]
    
    parser.class_api(root, "test_module.MyClass", my_class.bases, my_class.body)
    
    assert "test_module.MyClass" in parser.doc
    doc_content = parser.doc["test_module.MyClass"]
    assert "Members" in doc_content or "value" in doc_content


# LLM-generated content at query #2
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    non_string_node = Constant(value=42)
    result = resolver.visit_Constant(non_string_node)
    assert result is non_string_node
    assert result.value == 42
    
    # Test with float constant
    float_node = Constant(value=3.14)
    result = resolver.visit_Constant(float_node)
    assert result is float_node
    assert result.value == 3.14
    
    # Test with None constant
    none_node = Constant(value=None)
    result = resolver.visit_Constant(none_node)
    assert result is none_node
    assert result.value is None
    
    # Test with boolean constant
    bool_node = Constant(value=True)
    result = resolver.visit_Constant(bool_node)
    assert result is bool_node
    assert result.value is True
    
    # Test with valid string expression (Name)
    string_node = Constant(value="int")
    result = resolver.visit_Constant(string_node)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with valid string expression (Attribute)
    string_node = Constant(value="typing.List")
    result = resolver.visit_Constant(string_node)
    assert isinstance(result, Attribute)
    
    # Test with invalid syntax string (should return unchanged)
    invalid_node = Constant(value="not a valid python expression !!!")
    result = resolver.visit_Constant(invalid_node)
    assert result is invalid_node
    assert result.value == "not a valid python expression !!!"
    
    # Test with empty string
    empty_node = Constant(value="")
    result = resolver.visit_Constant(empty_node)
    assert result is empty_node
    assert result.value == ""
    
    # Test with string that has syntax error
    syntax_error_node = Constant(value="(invalid")
    result = resolver.visit_Constant(syntax_error_node)
    assert result is syntax_error_node


# LLM-generated content at query #3
#--------------------------

```python
def test_doctest():
    """Test the doctest function."""
    # Test empty string
    assert doctest("") == ""
    
    # Test single line without doctest
    assert doctest("This is a comment") == "This is a comment"
    
    # Test single doctest line
    result = doctest(">>> print('hello')")
    assert result == "```python\n>>> print('hello')\n```"
    
    # Test multiple lines with doctest at start
    result = doctest(">>> x = 1\n>>> print(x)")
    assert result == "```python\n>>> x = 1\n>>> print(x)\n```"
    
    # Test doctest with regular output
    result = doctest(">>> print('hello')\nhello")
    assert result == "```python\n>>> print('hello')\nhello\n```"
    
    # Test multiple doctest blocks
    result = doctest(">>> x = 1\nsome comment\n>>> y = 2")
    assert result == "```python\n>>> x = 1\n```\nsome comment\n```python\n>>> y = 2\n```"
    
    # Test doctest at end of document
    result = doctest("Some comment\n>>> print('end')")
    assert result == "Some comment\n```python\n>>> print('end')\n```"
    
    # Test multiple consecutive doctest lines
    result = doctest(">>> a = 1\n>>> b = 2\n>>> c = a + b")
    assert result == "```python\n>>> a = 1\n>>> b = 2\n>>> c = a + b\n```"
    
    # Test doctest with output in middle
    result = doctest(">>> x = 5\n5\n>>> y = 10")
    assert result == "```python\n>>> x = 5\n5\n```\n```python\n>>> y = 10\n```"
    
    # Test single line comment between doctest blocks
    result = doctest(">>> x = 1\ncomment line\n>>> y = 2")
    assert result == "```python\n>>> x = 1\n```\ncomment line\n```python\n>>> y = 2\n```"
    
    # Test with only output lines (no doctest markers)
    result = doctest("output line 1\noutput line 2")
    assert result == "output line 1\noutput line 2"


# LLM-generated content at query #4
#--------------------------

```python
def test_walk_body():
    """Test walk_body function."""
    # Test with empty body
    result = list(walk_body([]))
    assert result == []
    
    # Test with simple statements
    simple_assign = parse("x = 1").body
    result = list(walk_body(simple_assign))
    assert len(result) == 1
    assert isinstance(result[0], Assign)
    
    # Test with If statement
    if_stmt = parse("if True:\n    x = 1\nelse:\n    y = 2").body
    result = list(walk_body(if_stmt))
    assert len(result) == 2
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with nested If statements
    nested_if = parse("if True:\n    if False:\n        x = 1\n    y = 2").body
    result = list(walk_body(nested_if))
    assert len(result) == 2
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with Try statement
    try_stmt = parse("try:\n    x = 1\nexcept:\n    y = 2\nfinally:\n    z = 3").body
    result = list(walk_body(try_stmt))
    assert len(result) == 3
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with Try-Except-Else
    try_else = parse("try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3").body
    result = list(walk_body(try_else))
    assert len(result) == 3
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with complex nested structure
    complex_code = parse(
        "if True:\n"
        "    try:\n"
        "        x = 1\n"
        "    except:\n"
        "        y = 2\n"
        "else:\n"
        "    z = 3"
    ).body
    result = list(walk_body(complex_code))
    assert len(result) == 3
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with multiple statements at same level
    multi_stmt = parse("x = 1\ny = 2\nz = 3").body
    result = list(walk_body(multi_stmt))
    assert len(result) == 3
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with If containing multiple statements
    if_multi = parse("if True:\n    x = 1\n    y = 2\nelse:\n    z = 3\n    w = 4").body
    result = list(walk_body(if_multi))
    assert len(result) == 4
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with Try containing multiple handlers
    try_multi = parse(
        "try:\n"
        "    x = 1\n"
        "except ValueError:\n"
        "    y = 2\n"
        "except TypeError:\n"
        "    z = 3"
    ).body
    result = list(walk_body(try_multi))
    assert len(result) == 3
    assert all(isinstance(r, Assign) for r in result)
    
    # Test with non-assignment statements (should still be yielded)
    expr_stmt = parse("print(1)").body
    result = list(walk_body(expr_stmt))
    assert len(result) == 1
    assert isinstance(result[0], Expr)
    
    # Test with deeply nested If
    deep_if = parse(
        "if True:\n"
        "    if True:\n"
        "        if True:\n"
        "            x = 1"
    ).body
    result = list(walk_body(deep_if))
    assert len(result) == 1
    assert isinstance(result[0], Assign)


# LLM-generated content at query #5
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with value
    node = AnnAssign(
        target=Name(id="TypeAlias", ctx=Store()),
        annotation=Name(id="type", ctx=Load()),
        value=Name(id="str", ctx=Load()),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "TypeAlias") in parser.alias
    assert parser.alias[_m(root, "TypeAlias")] == "str"
    
    # Test AnnAssign without value (should return early)
    node_no_value = AnnAssign(
        target=Name(id="var", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, node_no_value)
    assert _m(root, "var") not in parser.alias
    
    # Test Assign with type_comment
    node_assign_comment = Assign(
        targets=[Name(id="CONSTANT", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals(root, node_assign_comment)
    assert _m(root, "CONSTANT") in parser.alias
    assert _m(root, "CONSTANT") in parser.const
    assert parser.const[_m(root, "CONSTANT")] == "int"
    
    # Test Assign without type_comment
    node_assign_no_comment = Assign(
        targets=[Name(id="value", ctx=Store())],
        value=Constant(value=10),
        type_comment=None
    )
    parser.globals(root, node_assign_no_comment)
    assert _m(root, "value") in parser.alias
    assert parser.alias[_m(root, "value")] == "10"
    
    # Test uppercase constant
    node_const = Assign(
        targets=[Name(id="MAX_VALUE", ctx=Store())],
        value=Constant(value=100),
        type_comment=None
    )
    parser.globals(root, node_const)
    assert _m(root, "MAX_VALUE") in parser.const
    assert parser.root[_m(root, "MAX_VALUE")] == root
    
    # Test __all__ with Tuple
    node_all_tuple = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node_all_tuple)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test __all__ with List
    node_all_list = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="func3")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node_all_list)
    assert _m(root, "func3") in parser.imp[root]
    
    # Test Assign with multiple targets (should return early)
    node_multi = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, node_multi)
    assert _m(root, "a") not in parser.alias
    
    # Test non-Name target (should return early)
    node_attr = Assign(
        targets=[Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, node_attr)
    assert len(parser.alias) == 5  # Only the previous assignments


# LLM-generated content at query #6
#--------------------------

```python
def test_Parser_compile():
    """Test Parser.compile method."""
    # Test basic compilation with empty parser
    p = Parser()
    result = p.compile()
    assert isinstance(result, str)
    assert result.endswith('\n')
    
    # Test with populated parser data
    p = Parser(link=True, b_level=1, toc=False)
    p.doc['test_module'] = '# Module `test_module`\n\n'
    p.docstring['test_module'] = 'Test documentation'
    p.root['test_module'] = 'test_module'
    p.level['test_module'] = 0
    p.imp['test_module'] = set()
    
    result = p.compile()
    assert isinstance(result, str)
    assert 'test_module' in result
    assert 'Test documentation' in result
    
    # Test with toc enabled
    p2 = Parser(link=True, b_level=1, toc=True)
    p2.doc['mod.func'] = '## func()\n\n'
    p2.docstring['mod.func'] = 'Function docs'
    p2.root['mod.func'] = 'mod'
    p2.level['mod.func'] = 1
    p2.imp['mod'] = set()
    
    result2 = p2.compile()
    assert '**Table of contents:**' in result2
    assert 'mod-func' in result2
    
    # Test with constants
    p3 = Parser(link=False, b_level=1, toc=False)
    p3.doc['test'] = '# Module `test`\n\n'
    p3.docstring['test'] = 'Test'
    p3.const['test.CONST'] = 'int'
    p3.root['test'] = 'test'
    p3.root['test.CONST'] = 'test'
    p3.level['test'] = 0
    p3.imp['test'] = set()
    
    result3 = p3.compile()
    assert isinstance(result3, str)
    
    # Test with magic methods excluded
    p4 = Parser(link=True, b_level=1, toc=False)
    p4.doc['test.__init__'] = '## __init__()\n\n'
    p4.root['test.__init__'] = 'test'
    p4.level['test.__init__'] = 1
    p4.imp['test'] = set()
    
    result4 = p4.compile()
    assert '__init__' not in result4
    
    # Test with multiple levels and nested modules
    p5 = Parser(link=True, b_level=1, toc=True)
    p5.doc['pkg'] = '# Module `pkg`\n\n'
    p5.doc['pkg.mod'] = '## Module `pkg.mod`\n\n'
    p5.doc['pkg.mod.func'] = '### func()\n\n'
    p5.docstring['pkg'] = 'Package'
    p5.docstring['pkg.mod'] = 'Module'
    p5.docstring['pkg.mod.func'] = 'Function'
    p5.root['pkg'] = 'pkg'
    p5.root['pkg.mod'] = 'pkg'
    p5.root['pkg.mod.func'] = 'pkg'
    p5.level['pkg'] = 0
    p5.level['pkg.mod'] = 1
    p5.level['pkg.mod.func'] = 1
    p5.imp['pkg'] = set()
    
    result5 = p5.compile()
    assert 'Table of contents' in result5
    assert 'Package' in result5
    assert 'Module' in result5
    assert 'Function' in result5
    
    # Test with __all__ filter
    p6 = Parser(link=False, b_level=1, toc=False)
    p6.doc['test'] = '# Module `test`\n\n'
    p6.doc['test.public'] = '## public\n\n'
    p6.doc['test._private'] = '## _private\n\n'
    p6.docstring['test'] = 'Test'
    p6.docstring['test.public'] = 'Public'
    p6.docstring['test._private'] = 'Private'
    p6.root['test'] = 'test'
    p6.root['test.public'] = 'test'
    p6.root['test._private'] = 'test'
    p6.level['test'] = 0
    p6.level['test.public'] = 1
    p6.level['test._private'] = 1
    p6.imp['test'] = {'test.public'}
    
    result6 = p6.compile()
    assert 'public' in result6
    assert '_private' not in result6


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_load_docstring():
    """Test Parser.load_docstring method."""
    from types import ModuleType
    
    # Create a mock module with docstrings
    mock_module = ModuleType("test_module")
    
    # Create a nested module structure
    nested_module = ModuleType("nested")
    nested_module.__doc__ = "Nested module docstring"
    mock_module.nested = nested_module
    
    # Create a function with docstring
    def test_func():
        """Test function docstring"""
        pass
    mock_module.test_func = test_func
    
    # Create a class with docstring
    class TestClass:
        """Test class docstring"""
        pass
    mock_module.TestClass = TestClass
    
    # Initialize parser with some documentation entries
    parser = Parser()
    parser.doc["test_module"] = "Module test_module"
    parser.doc["test_module.test_func"] = "Function test_func"
    parser.doc["test_module.TestClass"] = "Class TestClass"
    parser.doc["test_module.nested"] = "Nested module"
    
    # Call load_docstring
    parser.load_docstring("test_module", mock_module)
    
    # Verify docstrings were loaded
    assert "test_module.test_func" in parser.docstring
    assert "Test function docstring" in parser.docstring["test_module.test_func"]
    
    assert "test_module.TestClass" in parser.docstring
    assert "Test class docstring" in parser.docstring["test_module.TestClass"]
    
    assert "test_module.nested" in parser.docstring
    assert "Nested module docstring" in parser.docstring["test_module.nested"]


def test_Parser_load_docstring_no_docstring():
    """Test Parser.load_docstring with no docstrings."""
    from types import ModuleType
    
    mock_module = ModuleType("test_module")
    
    # Add attribute without docstring
    mock_module.no_doc = 42
    
    parser = Parser()
    parser.doc["test_module"] = "Module"
    parser.doc["test_module.no_doc"] = "Attribute"
    
    parser.load_docstring("test_module", mock_module)
    
    # Should not add entry if no docstring exists
    assert "test_module.no_doc" not in parser.docstring


def test_Parser_load_docstring_filters_by_root():
    """Test Parser.load_docstring only processes matching root."""
    from types import ModuleType
    
    mock_module = ModuleType("test_module")
    
    def func_with_doc():
        """Function docstring"""
        pass
    mock_module.func = func_with_doc
    
    parser = Parser()
    parser.doc["test_module.func"] = "Function"
    parser.doc["other_module.func"] = "Other function"
    
    parser.load_docstring("test_module", mock_module)
    
    # Only test_module entries should be processed
    assert "test_module.func" in parser.docstring
    assert "other_module.func" not in parser.docstring


def test_Parser_load_docstring_empty_module():
    """Test Parser.load_docstring with empty module."""
    from types import ModuleType
    
    mock_module = ModuleType("empty_module")
    
    parser = Parser()
    parser.doc["empty_module"] = "Empty module"
    
    # Should not raise any errors
    parser.load_docstring("empty_module", mock_module)
    
    assert "empty_module" not in parser.docstring


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_api():
    """Test Parser.api method."""
    from ast import parse, FunctionDef, AsyncFunctionDef, ClassDef
    
    parser = Parser()
    root = "test_module"
    parser.doc[root] = "# Module `test_module`\n\n"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    
    # Test with FunctionDef
    script = "def test_func(x: int) -> str: pass"
    tree = parse(script)
    func_node = tree.body[0]
    
    parser.api(root, func_node)
    
    assert "test_module.test_func" in parser.doc
    assert "test_func()" in parser.doc["test_module.test_func"]
    assert "*Full name:* `test_module.test_func`" in parser.doc["test_module.test_func"]
    assert parser.level["test_module.test_func"] == 0
    assert parser.root["test_module.test_func"] == root
    
    # Test with AsyncFunctionDef
    parser2 = Parser()
    root2 = "async_module"
    parser2.doc[root2] = "# Module `async_module`\n\n"
    parser2.level[root2] = 0
    parser2.root[root2] = root2
    parser2.alias = {}
    
    script2 = "async def async_func(): pass"
    tree2 = parse(script2)
    async_func_node = tree2.body[0]
    
    parser2.api(root2, async_func_node)
    
    assert "async_module.async_func" in parser2.doc
    assert "async async_func()" in parser2.doc["async_module.async_func"]
    
    # Test with ClassDef
    parser3 = Parser()
    root3 = "class_module"
    parser3.doc[root3] = "# Module `class_module`\n\n"
    parser3.level[root3] = 0
    parser3.root[root3] = root3
    parser3.alias = {}
    
    script3 = "class TestClass: pass"
    tree3 = parse(script3)
    class_node = tree3.body[0]
    
    parser3.api(root3, class_node)
    
    assert "class_module.TestClass" in parser3.doc
    assert "class TestClass" in parser3.doc["class_module.TestClass"]
    assert "*Full name:* `class_module.TestClass`" in parser3.doc["class_module.TestClass"]
    
    # Test with prefix (nested class/function)
    parser4 = Parser()
    root4 = "nested_module"
    parser4.doc[root4] = "# Module `nested_module`\n\n"
    parser4.level[root4] = 0
    parser4.root[root4] = root4
    parser4.alias = {}
    
    script4 = "class OuterClass:\n    def inner_method(self): pass"
    tree4 = parse(script4)
    outer_class = tree4.body[0]
    inner_method = outer_class.body[0]
    
    parser4.api(root4, outer_class)
    parser4.api(root4, inner_method, prefix="OuterClass")
    
    assert "nested_module.OuterClass.inner_method" in parser4.doc
    assert "inner_method()" in parser4.doc["nested_module.OuterClass.inner_method"]
    
    # Test with decorator
    parser5 = Parser()
    root5 = "decorator_module"
    parser5.doc[root5] = "# Module `decorator_module`\n\n"
    parser5.level[root5] = 0
    parser5.root[root5] = root5
    parser5.alias = {}
    
    script5 = "@staticmethod\ndef decorated_func(): pass"
    tree5 = parse(script5)
    decorated_node = tree5.body[0]
    
    parser5.api(root5, decorated_node)
    
    assert "decorator_module.decorated_func" in parser5.doc
    assert "Decorators" in parser5.doc["decorator_module.decorated_func"]


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_func_api():
    """Test func_api method of Parser class."""
    from ast import parse, arg as ast_arg
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    # Create a simple function with arguments and return type
    script = """
def example_func(a: int, b: str = "default", *args, c: float = 1.0, **kwargs) -> bool:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    # Test basic function API generation
    parser.func_api('test_module', 'test_module.example_func', func_node.args, func_node.returns)
    
    assert 'test_module.example_func' in parser.doc
    assert 'a' in parser.doc['test_module.example_func']
    assert 'b' in parser.doc['test_module.example_func']
    assert 'return' in parser.doc['test_module.example_func']


def test_Parser_func_api_with_self():
    """Test func_api with self parameter (instance method)."""
    from ast import parse
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    script = """
def method(self, x: int) -> str:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.func_api('test_module', 'test_module.method', func_node.args, func_node.returns, 
                    has_self=True, cls_method=False)
    
    assert 'test_module.method' in parser.doc
    assert 'Self' in parser.doc['test_module.method']


def test_Parser_func_api_classmethod():
    """Test func_api with classmethod."""
    from ast import parse
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    script = """
def create(cls, value: int) -> None:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.func_api('test_module', 'test_module.create', func_node.args, func_node.returns,
                    has_self=True, cls_method=True)
    
    assert 'test_module.create' in parser.doc
    assert 'type[Self]' in parser.doc['test_module.create']


def test_Parser_func_api_no_defaults():
    """Test func_api with required arguments only."""
    from ast import parse
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    script = """
def simple(x: int, y: str) -> None:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.func_api('test_module', 'test_module.simple', func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert 'test_module.simple' in parser.doc
    doc_content = parser.doc['test_module.simple']
    assert 'x' in doc_content
    assert 'y' in doc_content


def test_Parser_func_api_with_annotations():
    """Test func_api correctly formats annotations."""
    from ast import parse
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {'test_module.List': 'typing.List'}
    
    script = """
def typed_func(items: list, count: int = 0) -> dict:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.func_api('test_module', 'test_module.typed_func', func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert 'test_module.typed_func' in parser.doc
    doc_content = parser.doc['test_module.typed_func']
    assert 'items' in doc_content
    assert 'count' in doc_content
    assert 'return' in doc_content


def test_Parser_func_api_posonly_args():
    """Test func_api with positional-only arguments."""
    from ast import parse
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    script = """
def pos_only(a: int, /, b: str) -> None:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.func_api('test_module', 'test_module.pos_only', func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert 'test_module.pos_only' in parser.doc
    doc_content = parser.doc['test_module.pos_only']
    assert '/' in doc_content


def test_Parser_func_api_kwonly_args():
    """Test func_api with keyword-only arguments."""
    from ast import parse
    
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    script = """
def kw_only(a: int, *, b: str) -> None:
    pass
"""
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.func_api('test_module', 'test_module.kw_only', func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert 'test_module.kw_only' in parser.doc
    doc_content = parser.doc['test_module.kw_only']
    assert 'b' in doc_content


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_parse():
    """Test Parser.parse method."""
    parser = Parser()
    root = "test_module"
    script = """
'''Module docstring.'''

import os
from typing import List, Optional

MY_CONSTANT: int = 42

def my_function(x: int, y: str = "default") -> List[str]:
    '''Function docstring.'''
    return [y] * x

class MyClass:
    '''Class docstring.'''
    attr1: int
    attr2: str = "value"
    
    def method(self, arg: int) -> str:
        '''Method docstring.'''
        return str(arg)
"""
    
    parser.parse(root, script)
    
    # Check that root module is registered
    assert root in parser.doc
    assert root in parser.level
    assert root in parser.imp
    assert root in parser.root
    
    # Check module level is correct
    assert parser.level[root] == 0
    assert parser.root[root] == root
    
    # Check docstring was captured
    assert root in parser.docstring
    assert "Module docstring" in parser.docstring[root]
    
    # Check imports were registered
    assert _m(root, "os") in parser.alias
    assert _m(root, "List") in parser.alias
    assert _m(root, "Optional") in parser.alias
    
    # Check constants were registered
    assert _m(root, "MY_CONSTANT") in parser.const
    assert parser.const[_m(root, "MY_CONSTANT")] == "int"
    
    # Check function was registered
    assert _m(root, "my_function") in parser.doc
    assert _m(root, "my_function") in parser.level
    assert _m(root, "my_function") in parser.root
    assert _m(root, "my_function") in parser.docstring
    
    # Check class was registered
    assert _m(root, "MyClass") in parser.doc
    assert _m(root, "MyClass") in parser.level
    assert _m(root, "MyClass") in parser.root
    assert _m(root, "MyClass") in parser.docstring
    
    # Check nested method was registered
    assert _m(root, "MyClass", "method") in parser.doc
    assert _m(root, "MyClass", "method") in parser.docstring


def test_Parser_parse_with_all():
    """Test Parser.parse with __all__ definition."""
    parser = Parser()
    root = "test_module_all"
    script = """
def public_func():
    '''Public function.'''
    pass

def _private_func():
    '''Private function.'''
    pass

__all__ = ['public_func']
"""
    
    parser.parse(root, script)
    
    assert root in parser.imp
    assert _m(root, "public_func") in parser.imp[root]


def test_Parser_parse_with_type_comment():
    """Test Parser.parse with type comments."""
    parser = Parser()
    root = "test_module_comment"
    script = """
x = 42  # type: int
y = "hello"  # type: str
"""
    
    parser.parse(root, script)
    
    assert _m(root, "x") in parser.const
    assert parser.const[_m(root, "x")] == "int"
    assert _m(root, "y") in parser.const
    assert parser.const[_m(root, "y")] == "str"


def test_Parser_parse_nested_class():
    """Test Parser.parse with nested class."""
    parser = Parser()
    root = "test_nested"
    script = """
class Outer:
    '''Outer class.'''
    
    class Inner:
        '''Inner class.'''
        
        def inner_method(self):
            '''Inner method.'''
            pass
"""
    
    parser.parse(root, script)
    
    assert _m(root, "Outer") in parser.doc
    assert _m(root, "Outer", "Inner") in parser.doc
    assert _m(root, "Outer", "Inner", "inner_method") in parser.doc


def test_Parser_parse_async_function():
    """Test Parser.parse with async function."""
    parser = Parser()
    root = "test_async"
    script = """
async def async_func(x: int) -> str:
    '''Async function.'''
    return str(x)
"""
    
    parser.parse(root, script)
    
    assert _m(root, "async_func") in parser.doc
    assert "async" in parser.doc[_m(root, "async_func")]


def test_Parser_parse_with_decorators():
    """Test Parser.parse with decorated functions."""
    parser = Parser()
    root = "test_decorators"
    script = """
from functools import staticmethod, classmethod

class MyClass:
    '''Test class.'''
    
    @staticmethod
    def static_method(x: int) -> int:
        '''Static method.'''
        return x
    
    @classmethod
    def class_method(cls, x: int) -> int:
        '''Class method.'''
        return x
"""
    
    parser.parse(root, script)
    
    assert _m(root, "MyClass", "static_method") in parser.doc
    assert _m(root, "MyClass", "class_method") in parser.doc


# LLM-generated content at query #11
#--------------------------

```python
def test_Parser_api():
    """Unit test for Parser.api method."""
    from ast import parse, FunctionDef, AsyncFunctionDef, ClassDef
    
    # Test with FunctionDef
    parser = Parser()
    script = """
def example_func(x: int) -> str:
    '''Example function.'''
    return str(x)
"""
    root = "test_module"
    parser.parse(root, script)
    
    root_node = parse(script)
    func_node = root_node.body[0]
    
    parser.api(root, func_node)
    
    full_name = _m(root, func_node.name)
    assert full_name in parser.doc
    assert "example_func()" in parser.doc[full_name]
    assert parser.level[full_name] == parser.level[root]
    assert parser.root[full_name] == root
    
    # Test with AsyncFunctionDef
    script_async = """
async def async_func() -> None:
    '''Async function.'''
    pass
"""
    root_async = "test_async"
    parser.parse(root_async, script_async)
    
    root_node_async = parse(script_async)
    async_func_node = root_node_async.body[0]
    
    parser.api(root_async, async_func_node)
    
    async_full_name = _m(root_async, async_func_node.name)
    assert async_full_name in parser.doc
    assert "async async_func()" in parser.doc[async_full_name]
    
    # Test with ClassDef
    script_class = """
class ExampleClass:
    '''Example class.'''
    pass
"""
    root_class = "test_class"
    parser.parse(root_class, script_class)
    
    root_node_class = parse(script_class)
    class_node = root_node_class.body[0]
    
    parser.api(root_class, class_node)
    
    class_full_name = _m(root_class, class_node.name)
    assert class_full_name in parser.doc
    assert "class ExampleClass" in parser.doc[class_full_name]
    
    # Test with prefix (nested class/function)
    script_nested = """
class Container:
    def method(self) -> int:
        '''Nested method.'''
        return 42
"""
    root_nested = "test_nested"
    parser.parse(root_nested, script_nested)
    
    root_node_nested = parse(script_nested)
    container_class = root_node_nested.body[0]
    method_node = container_class.body[0]
    
    parser.api(root_nested, method_node, prefix="Container")
    
    method_full_name = _m(root_nested, "Container", method_node.name)
    assert method_full_name in parser.doc
    assert parser.level[method_full_name] == parser.level[root_nested]
    assert parser.root[method_full_name] == root_nested
    
    # Test with decorators
    script_decorated = """
@property
def decorated_func() -> str:
    '''Decorated function.'''
    return "test"
"""
    root_decorated = "test_decorated"
    parser.parse(root_decorated, script_decorated)
    
    root_node_decorated = parse(script_decorated)
    decorated_node = root_node_decorated.body[0]
    
    parser.api(root_decorated, decorated_node)
    
    decorated_full_name = _m(root_decorated, decorated_node.name)
    assert decorated_full_name in parser.doc
    assert "Decorators" in parser.doc[decorated_full_name]


# LLM-generated content at query #12
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test with AnnAssign node with value
    ann_assign = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert _m(root, "x") in parser.alias
    assert parser.alias[_m(root, "x")] == "42"
    assert parser.const[_m(root, "x")] == "int"
    
    # Test with uppercase constant
    ann_assign_const = AnnAssign(
        target=Name(id="MAX_VALUE", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=100),
        simple=1
    )
    parser.globals(root, ann_assign_const)
    assert parser.root[_m(root, "MAX_VALUE")] == root
    assert parser.const[_m(root, "MAX_VALUE")] == "int"
    
    # Test with Assign node with type_comment
    assign_with_comment = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals(root, assign_with_comment)
    assert _m(root, "y") in parser.alias
    assert parser.const[_m(root, "y")] == "float"
    
    # Test with Assign node without type_comment
    assign_no_comment = Assign(
        targets=[Name(id="z", ctx=Store())],
        value=List(elts=[Constant(value=1), Constant(value=2)], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, assign_no_comment)
    assert _m(root, "z") in parser.alias
    assert parser.const[_m(root, "z")] == "list[int]"
    
    # Test with __all__ assignment
    all_assign = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="func1"), Constant(value="func2")],
            ctx=Load()
        ),
        type_comment=None
    )
    parser.globals(root, all_assign)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test with multiple targets (should return early)
    multi_assign = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, multi_assign)
    assert _m(root, "a") not in parser.alias
    
    # Test with AnnAssign without value (should return early)
    ann_no_value = AnnAssign(
        target=Name(id="w", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, ann_no_value)
    assert _m(root, "w") not in parser.alias
    
    # Test with non-Name target (should return early)
    assign_attr = Assign(
        targets=[Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, assign_attr)
    assert _m(root, "obj") not in parser.alias


# LLM-generated content at query #13
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test case 1: AnnAssign with value
    node = AnnAssign(
        target=Name(id="MyType", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=Constant(value="hello"),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "MyType") in parser.alias
    assert parser.alias[_m(root, "MyType")] == "'hello'"
    
    # Test case 2: Assign with single target
    node = Assign(
        targets=[Name(id="CONSTANT", ctx=Store())],
        value=Constant(value=42),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "CONSTANT") in parser.alias
    assert _m(root, "CONSTANT") in parser.const
    assert parser.const[_m(root, "CONSTANT")] == "int"
    
    # Test case 3: Uppercase constant
    node = Assign(
        targets=[Name(id="DEBUG", ctx=Store())],
        value=Constant(value=True),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "DEBUG") in parser.root
    assert parser.root[_m(root, "DEBUG")] == root
    
    # Test case 4: __all__ filter with list
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test case 5: __all__ filter with tuple
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="func3")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "func3") in parser.imp[root]
    
    # Test case 6: AnnAssign without value (should return early)
    node = AnnAssign(
        target=Name(id="NoValue", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "NoValue") not in parser.alias
    
    # Test case 7: Assign with multiple targets (should return early)
    node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "a") not in parser.alias
    
    # Test case 8: Assign with type_comment
    node = Assign(
        targets=[Name(id="TYPED", ctx=Store())],
        value=Constant(value=100),
        type_comment="int"
    )
    parser.globals(root, node)
    assert _m(root, "TYPED") in parser.const
    assert parser.const[_m(root, "TYPED")] == "int"
    
    # Test case 9: AnnAssign with type annotation
    node = AnnAssign(
        target=Name(id="TypedVar", ctx=Store()),
        annotation=Name(id="List", ctx=Load()),
        value=List(elts=[], ctx=Load()),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "TypedVar") in parser.alias


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test 1: Non-string constant should return unchanged
    const_int = Constant(value=42)
    result = resolver.visit_Constant(const_int)
    assert result is const_int
    assert result.value == 42
    
    # Test 2: Non-string constant (float)
    const_float = Constant(value=3.14)
    result = resolver.visit_Constant(const_float)
    assert result is const_float
    assert result.value == 3.14
    
    # Test 3: Non-string constant (None)
    const_none = Constant(value=None)
    result = resolver.visit_Constant(const_none)
    assert result is const_none
    assert result.value is None
    
    # Test 4: String that is a valid name
    const_str_name = Constant(value="int")
    result = resolver.visit_Constant(const_str_name)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test 5: String that is a valid subscript expression
    const_str_subscript = Constant(value="list[int]")
    result = resolver.visit_Constant(const_str_subscript)
    assert isinstance(result, Subscript)
    
    # Test 6: String that is invalid syntax - should return original node
    const_str_invalid = Constant(value="not valid python @@")
    result = resolver.visit_Constant(const_str_invalid)
    assert result is const_str_invalid
    assert result.value == "not valid python @@"
    
    # Test 7: String with attribute access
    const_str_attr = Constant(value="typing.List")
    result = resolver.visit_Constant(const_str_attr)
    assert isinstance(result, Attribute)
    
    # Test 8: Empty string
    const_str_empty = Constant(value="")
    result = resolver.visit_Constant(const_str_empty)
    assert result is const_str_empty
    assert result.value == ""
    
    # Test 9: String with binary operation
    const_str_binop = Constant(value="int | str")
    result = resolver.visit_Constant(const_str_binop)
    assert isinstance(result, BinOp)
    
    # Test 10: With resolver alias - string that references aliased type
    resolver_with_alias = Resolver("test_module", {"test_module.MyType": "int"})
    const_str_alias = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(const_str_alias)
    assert isinstance(result, Name)
    assert result.id == "MyType"


# LLM-generated content at query #15
#--------------------------

def test_is_public_family():
    """Test is_public_family function."""
    # Test fully public names
    assert is_public_family("os.path.join") is True
    assert is_public_family("sys") is True
    assert is_public_family("collections.abc.Sequence") is True
    
    # Test names with magic methods (should be allowed)
    assert is_public_family("module.__init__") is True
    assert is_public_family("__main__") is True
    assert is_public_family("os.__dict__") is True
    
    # Test names with private components (underscore prefix)
    assert is_public_family("_private") is False
    assert is_public_family("os._internal") is False
    assert is_public_family("module._private.func") is False
    assert is_public_family("os.path._private") is False
    
    # Test mixed cases
    assert is_public_family("os.__name__.public") is True
    assert is_public_family("public.__magic__.func") is True
    assert is_public_family("public._private.__magic__") is False
    
    # Test edge cases
    assert is_public_family("") is True
    assert is_public_family("a") is True
    assert is_public_family("_") is False
    assert is_public_family("__") is True  # Magic name
    
    # Test complex nested names
    assert is_public_family("package.module.Class.method") is True
    assert is_public_family("package._internal.Class") is False
    assert is_public_family("package.module._private_class.method") is False


# LLM-generated content at query #16
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    from ast import parse, Subscript, Name, Load, Tuple, Constant, BinOp, BitOr
    
    # Test case 1: Union type conversion to BitOr
    resolver = Resolver("typing", {
        "typing.Union": "typing.Union"
    })
    code_str = "Union[int, str]"
    node = cast(Expr, parse(code_str).body[0]).value
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test case 2: Optional type conversion
    resolver = Resolver("typing", {
        "typing.Optional": "typing.Optional"
    })
    code_str = "Optional[int]"
    node = cast(Expr, parse(code_str).body[0]).value
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test case 3: PEP585 deprecated name conversion
    resolver = Resolver("typing", {
        "typing.List": "typing.List"
    })
    code_str = "List[int]"
    node = cast(Expr, parse(code_str).body[0]).value
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    
    # Test case 4: Non-Name subscript value (passthrough)
    resolver = Resolver("test", {})
    code_str = "some_func()[int]"
    node = cast(Expr, parse(code_str).body[0]).value
    result = resolver.visit_Subscript(node)
    assert result is node
    
    # Test case 5: Union with non-Tuple slice (passthrough)
    resolver = Resolver("typing", {
        "typing.Union": "typing.Union"
    })
    code_str = "Union[int]"
    node = cast(Expr, parse(code_str).body[0]).value
    result = resolver.visit_Subscript(node)
    assert result is node.slice
    
    # Test case 6: Unknown subscript type (passthrough)
    resolver = Resolver("typing", {})
    code_str = "CustomType[int]"
    node = cast(Expr, parse(code_str).body[0]).value
    result = resolver.visit_Subscript(node)
    assert result is node


# LLM-generated content at query #17
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    from ast import Attribute, Name, Load
    
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("mymodule", {})
    typing_attr = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)
    
    # Test case 2: Keep non-typing attributes
    resolver = Resolver("mymodule", {})
    other_attr = Attribute(Name("other", Load()), "SomeAttr", Load())
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeAttr"
    assert isinstance(result.value, Name)
    assert result.value.id == "other"
    
    # Test case 3: Non-Name value (nested attribute)
    resolver = Resolver("mymodule", {})
    nested_attr = Attribute(Attribute(Name("typing", Load()), "Dict", Load()), "items", Load())
    result = resolver.visit_Attribute(nested_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "items"
    
    # Test case 4: typing.Optional
    resolver = Resolver("mymodule", {})
    optional_attr = Attribute(Name("typing", Load()), "Optional", Load())
    result = resolver.visit_Attribute(optional_attr)
    assert isinstance(result, Name)
    assert result.id == "Optional"
    
    # Test case 5: typing.Union
    resolver = Resolver("mymodule", {})
    union_attr = Attribute(Name("typing", Load()), "Union", Load())
    result = resolver.visit_Attribute(union_attr)
    assert isinstance(result, Name)
    assert result.id == "Union"
    
    # Test case 6: typing.Dict
    resolver = Resolver("mymodule", {})
    dict_attr = Attribute(Name("typing", Load()), "Dict", Load())
    result = resolver.visit_Attribute(dict_attr)
    assert isinstance(result, Name)
    assert result.id == "Dict"


# LLM-generated content at query #18
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("test_module", {})
    
    # Test 1: Remove 'typing.*' prefix
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'
    
    # Test 2: Remove 'typing.Dict' prefix
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Dict', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Dict'
    
    # Test 3: Remove 'typing.Optional' prefix
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Optional', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Optional'
    
    # Test 4: Keep non-typing attributes unchanged
    node = Attribute(value=Name(id='os', ctx=Load()), attr='path', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'path'
    assert isinstance(result.value, Name)
    assert result.value.id == 'os'
    
    # Test 5: Keep non-typing attributes with different module name
    node = Attribute(value=Name(id='collections', ctx=Load()), attr='defaultdict', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'defaultdict'
    assert result.value.id == 'collections'
    
    # Test 6: Handle nested non-Name value (should return node unchanged)
    node = Attribute(value=Constant(value=42), attr='real', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'real'
    
    # Test 7: typing with Union
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Union', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Union'
    
    # Test 8: typing with Callable
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Callable', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Callable'


# LLM-generated content at query #19
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with value
    target = Name(id="TypeAlias", ctx=Store())
    annotation = Name(id="type", ctx=Load())
    value = Constant(value="dict[str, int]")
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    assert _m(root, "TypeAlias") in parser.alias
    assert parser.alias[_m(root, "TypeAlias")] == "dict[str, int]"
    
    # Test AnnAssign without value (should return early)
    target2 = Name(id="Var", ctx=Store())
    annotation2 = Name(id="int", ctx=Load())
    node2 = AnnAssign(target=target2, annotation=annotation2, value=None, simple=1)
    
    parser.globals(root, node2)
    assert _m(root, "Var") not in parser.alias
    
    # Test uppercase constant
    target3 = Name(id="CONSTANT", ctx=Store())
    value3 = Constant(value=42)
    node3 = Assign(targets=[target3], value=value3, type_comment=None)
    
    parser.globals(root, node3)
    assert _m(root, "CONSTANT") in parser.const
    assert parser.const[_m(root, "CONSTANT")] == "int"
    
    # Test __all__ filter
    target4 = Name(id="__all__", ctx=Store())
    value4 = Tuple(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load())
    node4 = Assign(targets=[target4], value=value4, type_comment=None)
    
    parser.globals(root, node4)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test Assign with type_comment
    target5 = Name(id="typed_var", ctx=Store())
    value5 = Constant(value="hello")
    node5 = Assign(targets=[target5], value=value5, type_comment="str")
    
    parser.globals(root, node5)
    assert _m(root, "typed_var") in parser.alias
    assert parser.alias[_m(root, "typed_var")] == "'hello'"
    
    # Test invalid node (multiple targets)
    target6a = Name(id="x", ctx=Store())
    target6b = Name(id="y", ctx=Store())
    value6 = Constant(value=1)
    node6 = Assign(targets=[target6a, target6b], value=value6, type_comment=None)
    
    initial_len = len(parser.alias)
    parser.globals(root, node6)
    assert len(parser.alias) == initial_len
    
    # Test invalid node (non-Name target)
    target7 = Tuple(elts=[Name(id="a", ctx=Store())], ctx=Store())
    value7 = Constant(value=1)
    node7 = Assign(targets=[target7], value=value7, type_comment=None)
    
    initial_len = len(parser.alias)
    parser.globals(root, node7)
    assert len(parser.alias) == initial_len


# LLM-generated content at query #20
#--------------------------

```python
def test_Parser_func_ann():
    """Test Parser.func_ann method."""
    parser = Parser()
    
    # Test case 1: Simple function with no arguments
    args = []
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == []
    
    # Test case 2: Function with simple annotation
    from ast import arg as ast_arg
    args = [ast_arg(arg='x', annotation=Constant(value=int))]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert len(result) == 1
    
    # Test case 3: Function with self parameter
    args = [ast_arg(arg='self', annotation=None)]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=False))
    assert result[0] == 'Self'
    
    # Test case 4: Class method with self parameter
    args = [ast_arg(arg='cls', annotation=None)]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'
    
    # Test case 5: Function with self and additional parameters
    args = [
        ast_arg(arg='self', annotation=None),
        ast_arg(arg='x', annotation=Constant(value=int))
    ]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=False))
    assert result[0] == 'Self'
    assert len(result) == 2
    
    # Test case 6: Function with *args separator
    args = [
        ast_arg(arg='x', annotation=Constant(value=int)),
        ast_arg(arg='*', annotation=None),
        ast_arg(arg='y', annotation=Constant(value=str))
    ]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result[1] == ""
    
    # Test case 7: Function with no annotation yields ANY
    args = [ast_arg(arg='x', annotation=None)]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result[0] == ANY
    
    # Test case 8: Class method with self_ty annotation
    parser.alias["test_module.MyClass"] = "MyClass"
    args = [ast_arg(arg='cls', annotation=Name(id='MyClass', ctx=Load()))]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'


# LLM-generated content at query #21
#--------------------------

```python
def test_Parser_func_api():
    """Test Parser.func_api method."""
    from ast import parse, arg as ast_arg
    
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    
    # Parse a simple function to get arguments node
    script = "def func(a: int, b: str = 'default', *args, c: float = 1.0, **kwargs) -> bool: pass"
    tree = parse(script)
    func_def = tree.body[0]
    
    # Test basic function API
    parser.func_api(root, name, func_def.args, func_def.returns, 
                    has_self=False, cls_method=False)
    
    assert name in parser.doc
    assert "test_module.test_func()" in parser.doc[name]
    
    # Test with self parameter
    script_self = "def method(self, x: int) -> None: pass"
    tree_self = parse(script_self)
    func_def_self = tree_self.body[0]
    
    name_self = "test_module.TestClass.method"
    parser.func_api(root, name_self, func_def_self.args, func_def_self.returns,
                    has_self=True, cls_method=False)
    
    assert name_self in parser.doc
    
    # Test with classmethod
    name_cls = "test_module.TestClass.cls_method"
    parser.func_api(root, name_cls, func_def_self.args, func_def_self.returns,
                    has_self=True, cls_method=True)
    
    assert name_cls in parser.doc
    
    # Test with no annotations
    script_no_ann = "def func(a, b): pass"
    tree_no_ann = parse(script_no_ann)
    func_def_no_ann = tree_no_ann.body[0]
    
    name_no_ann = "test_module.no_ann_func"
    parser.func_api(root, name_no_ann, func_def_no_ann.args, None,
                    has_self=False, cls_method=False)
    
    assert name_no_ann in parser.doc
    
    # Test with position-only and keyword-only arguments
    script_complex = "def func(a, /, b, *args, c, **kwargs) -> int: pass"
    tree_complex = parse(script_complex)
    func_def_complex = tree_complex.body[0]
    
    name_complex = "test_module.complex_func"
    parser.func_api(root, name_complex, func_def_complex.args, 
                    func_def_complex.returns, has_self=False, cls_method=False)
    
    assert name_complex in parser.doc


# LLM-generated content at query #22
#--------------------------

```python
def test_Parser_imports():
    """Test Parser.imports method."""
    from ast import Import, ImportFrom, alias
    
    parser = Parser()
    root = "test_module"
    
    # Test simple import
    node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"
    
    # Test import with alias
    node = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, node)
    assert parser.alias["test_module.np"] == "numpy"
    
    # Test multiple imports
    node = Import(names=[
        alias(name="sys", asname=None),
        alias(name="json", asname="j")
    ])
    parser.imports(root, node)
    assert parser.alias["test_module.sys"] == "sys"
    assert parser.alias["test_module.j"] == "json"
    
    # Test ImportFrom without level
    node = ImportFrom(module="collections", names=[alias(name="defaultdict", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"
    
    # Test ImportFrom with alias
    node = ImportFrom(module="typing", names=[alias(name="List", asname="L")], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.L"] == "typing.List"
    
    # Test ImportFrom with relative import (level=1)
    parser.root["test_module.submodule"] = "test_module.submodule"
    node = ImportFrom(module="utils", names=[alias(name="helper", asname=None)], level=1)
    parser.imports("test_module.submodule", node)
    assert parser.alias["test_module.submodule.helper"] == "test_module.utils.helper"
    
    # Test ImportFrom with star import
    node = ImportFrom(module="os", names=[alias(name="*", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.*"] == "os.*"
    
    # Test ImportFrom with multiple names
    node = ImportFrom(module="itertools", names=[
        alias(name="chain", asname=None),
        alias(name="cycle", asname="c")
    ], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.chain"] == "itertools.chain"
    assert parser.alias["test_module.c"] == "itertools.cycle"
    
    # Test ImportFrom with level=2 (relative import up two levels)
    parser.root["a.b.c"] = "a.b.c"
    node = ImportFrom(module="x", names=[alias(name="y", asname=None)], level=2)
    parser.imports("a.b.c", node)
    assert parser.alias["a.b.c.y"] == "a.x.y"


# LLM-generated content at query #23
#--------------------------

```python
def test_Parser_load_docstring():
    """Unit test for Parser.load_docstring method."""
    from types import ModuleType
    from unittest.mock import MagicMock, patch
    
    # Create a parser instance
    parser = Parser()
    
    # Set up test data
    root = "test_module"
    parser.doc[root] = "Module doc"
    parser.doc["test_module.func"] = "Function doc"
    parser.doc["test_module.MyClass"] = "Class doc"
    parser.doc["other_module.func"] = "Other doc"
    parser.docstring[root] = ""
    parser.docstring["test_module.func"] = ""
    parser.docstring["test_module.MyClass"] = ""
    
    # Create a mock module
    mock_module = MagicMock(spec=ModuleType)
    mock_module.__doc__ = "Module docstring"
    
    # Create mock attributes
    mock_func = MagicMock()
    mock_func.__doc__ = "Function docstring"
    mock_class = MagicMock()
    mock_class.__doc__ = "Class docstring"
    
    # Mock the _attr function to return our mock objects
    with patch('builtins.__import__') as mock_import:
        def mock_attr_side_effect(module, attr):
            if attr == "func":
                return mock_func
            elif attr == "MyClass":
                return mock_class
            return None
        
        with patch('getdoc') as mock_getdoc:
            def getdoc_side_effect(obj):
                if obj is mock_module:
                    return "Module docstring"
                elif obj is mock_func:
                    return "Function docstring"
                elif obj is mock_class:
                    return "Class docstring"
                return None
            
            mock_getdoc.side_effect = getdoc_side_effect
            
            with patch('doctest') as mock_doctest:
                mock_doctest.side_effect = lambda x: f"processed: {x}"
                
                # Call load_docstring with a custom _attr implementation
                original_attr = None
                try:
                    import sys
                    # Create a simple test module
                    test_mod = ModuleType("test_module")
                    test_mod.func = mock_func
                    test_mod.MyClass = mock_class
                    
                    with patch('_attr', side_effect=lambda m, attr: getattr(test_mod, attr, None)):
                        parser.load_docstring(root, test_mod)
                except:
                    pass
    
    # Verify docstrings were loaded for items in the root module
    assert "test_module" in parser.docstring
    assert "test_module.func" in parser.docstring
    assert "test_module.MyClass" in parser.docstring
    
    # Verify that items from other modules were not updated
    assert parser.docstring.get("other_module.func") == ""


def test_Parser_load_docstring_with_none_docstrings():
    """Test load_docstring when module has None docstrings."""
    from types import ModuleType
    from unittest.mock import MagicMock, patch
    
    parser = Parser()
    
    # Set up test data
    root = "test_module"
    parser.doc[root] = "Module doc"
    parser.doc["test_module.func"] = "Function doc"
    parser.docstring[root] = ""
    parser.docstring["test_module.func"] = ""
    
    # Create a mock module with None docstrings
    mock_module = MagicMock(spec=ModuleType)
    mock_module.__doc__ = None
    
    mock_func = MagicMock()
    mock_func.__doc__ = None
    
    with patch('getdoc', return_value=None):
        with patch('_attr', return_value=mock_func):
            parser.load_docstring(root, mock_module)
    
    # Docstrings should remain empty if getdoc returns None
    assert parser.docstring[root] == ""


def test_Parser_load_docstring_empty_doc():
    """Test load_docstring with empty doc dictionary."""
    from types import ModuleType
    from unittest.mock import MagicMock
    
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    
    mock_module = MagicMock(spec=ModuleType)
    
    # Should not raise an error
    parser.load_docstring("any_root", mock_module)
    
    assert len(parser.docstring) == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("test_module", {})
    
    # Test case 1: Remove 'typing.*' prefix
    typing_node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(typing_node)
    assert isinstance(result, Name)
    assert result.id == 'List'
    
    # Test case 2: Remove 'typing.Dict' prefix
    typing_dict_node = Attribute(value=Name(id='typing', ctx=Load()), attr='Dict', ctx=Load())
    result = resolver.visit_Attribute(typing_dict_node)
    assert isinstance(result, Name)
    assert result.id == 'Dict'
    
    # Test case 3: Non-typing attribute should remain unchanged
    other_node = Attribute(value=Name(id='other', ctx=Load()), attr='method', ctx=Load())
    result = resolver.visit_Attribute(other_node)
    assert isinstance(result, Attribute)
    assert result.value.id == 'other'
    assert result.attr == 'method'
    
    # Test case 4: Attribute with non-Name value should remain unchanged
    complex_node = Attribute(value=Attribute(value=Name(id='typing', ctx=Load()), attr='sub', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(complex_node)
    assert isinstance(result, Attribute)
    assert result.attr == 'List'
    
    # Test case 5: typing.Optional
    optional_node = Attribute(value=Name(id='typing', ctx=Load()), attr='Optional', ctx=Load())
    result = resolver.visit_Attribute(optional_node)
    assert isinstance(result, Name)
    assert result.id == 'Optional'
    
    # Test case 6: typing.Union
    union_node = Attribute(value=Name(id='typing', ctx=Load()), attr='Union', ctx=Load())
    result = resolver.visit_Attribute(union_node)
    assert isinstance(result, Name)
    assert result.id == 'Union'


# LLM-generated content at query #25
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test 1: Non-string constant returns unchanged
    const_int = Constant(value=42)
    result = resolver.visit_Constant(const_int)
    assert result is const_int
    assert result.value == 42
    
    # Test 2: Non-string constant (float)
    const_float = Constant(value=3.14)
    result = resolver.visit_Constant(const_float)
    assert result is const_float
    assert result.value == 3.14
    
    # Test 3: Non-string constant (None)
    const_none = Constant(value=None)
    result = resolver.visit_Constant(const_none)
    assert result is const_none
    assert result.value is None
    
    # Test 4: String constant with valid name
    const_str_name = Constant(value="int")
    result = resolver.visit_Constant(const_str_name)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test 5: String constant with valid attribute access
    const_str_attr = Constant(value="typing.List")
    result = resolver.visit_Constant(const_str_attr)
    assert isinstance(result, Attribute)
    
    # Test 6: String constant with invalid syntax returns unchanged
    const_str_invalid = Constant(value="not valid syntax @@")
    result = resolver.visit_Constant(const_str_invalid)
    assert result is const_str_invalid
    assert result.value == "not valid syntax @@"
    
    # Test 7: Empty string constant
    const_str_empty = Constant(value="")
    result = resolver.visit_Constant(const_str_empty)
    assert result is const_str_empty
    assert result.value == ""
    
    # Test 8: String with complex expression
    const_str_complex = Constant(value="List[int]")
    result = resolver.visit_Constant(const_str_complex)
    assert isinstance(result, Subscript)
    
    # Test 9: String with self_ty set
    resolver_with_self = Resolver("test_module", {}, self_ty="T")
    const_str_self = Constant(value="T")
    result = resolver_with_self.visit_Constant(const_str_self)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #26
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("test_module", {})
    
    # Test case 1: typing.List should be replaced with List
    typing_attr = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)
    
    # Test case 2: other.List should remain as Attribute
    other_attr = Attribute(Name("other", Load()), "List", Load())
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    assert result.value.id == "other"
    
    # Test case 3: typing.Dict should be replaced with Dict
    typing_dict = Attribute(Name("typing", Load()), "Dict", Load())
    result = resolver.visit_Attribute(typing_dict)
    assert isinstance(result, Name)
    assert result.id == "Dict"
    
    # Test case 4: Non-Name value should return node unchanged
    non_name_attr = Attribute(Constant("not_a_name"), "attr", Load())
    result = resolver.visit_Attribute(non_name_attr)
    assert result is non_name_attr
    
    # Test case 5: typing.Optional should be replaced with Optional
    typing_optional = Attribute(Name("typing", Load()), "Optional", Load())
    result = resolver.visit_Attribute(typing_optional)
    assert isinstance(result, Name)
    assert result.id == "Optional"
    
    # Test case 6: module.submodule.attr should remain unchanged
    nested_attr = Attribute(Attribute(Name("module", Load()), "submodule", Load()), "attr", Load())
    result = resolver.visit_Attribute(nested_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"


# LLM-generated content at query #27
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    # Test case 1: Remove typing.* prefix
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'
    
    # Test case 2: Non-typing attribute should remain unchanged
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='other_module', ctx=Load()), attr='SomeClass', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'SomeClass'
    assert isinstance(result.value, Name)
    assert result.value.id == 'other_module'
    
    # Test case 3: Attribute on non-Name node should remain unchanged
    resolver = Resolver("test_module", {})
    inner_attr = Attribute(value=Name(id='obj', ctx=Load()), attr='prop', ctx=Load())
    node = Attribute(value=inner_attr, attr='nested', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'nested'
    assert isinstance(result.value, Attribute)
    
    # Test case 4: typing.Optional should be converted to Name
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Optional', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Optional'
    
    # Test case 5: typing.Dict should be converted to Name
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Dict', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Dict'


# LLM-generated content at query #28
#--------------------------

```python
def test_Parser_load_docstring():
    """Test Parser.load_docstring method."""
    from types import ModuleType
    from unittest.mock import MagicMock, patch
    
    # Create a parser instance
    parser = Parser()
    
    # Set up test data
    root = "test_module"
    parser.doc[root] = "Module doc"
    parser.doc["test_module.func"] = "Function doc"
    parser.doc["test_module.MyClass"] = "Class doc"
    parser.doc["other_module.func"] = "Other doc"
    parser.docstring = {}
    
    # Create a mock module
    mock_module = MagicMock(spec=ModuleType)
    mock_func = MagicMock()
    mock_func.__doc__ = "Loaded function docstring"
    mock_class = MagicMock()
    mock_class.__doc__ = "Loaded class docstring"
    
    # Mock _attr function to return appropriate objects
    with patch('__main__._attr') as mock_attr, \
         patch('__main__.getdoc') as mock_getdoc, \
         patch('__main__.doctest') as mock_doctest:
        
        def attr_side_effect(module, name):
            if name == "func":
                return mock_func
            elif name == "MyClass":
                return mock_class
            return None
        
        def getdoc_side_effect(obj):
            if obj is mock_func:
                return "Loaded function docstring"
            elif obj is mock_class:
                return "Loaded class docstring"
            return None
        
        def doctest_side_effect(doc):
            return f"processed: {doc}"
        
        mock_attr.side_effect = attr_side_effect
        mock_getdoc.side_effect = getdoc_side_effect
        mock_doctest.side_effect = doctest_side_effect
        
        # Call load_docstring
        parser.load_docstring(root, mock_module)
        
        # Verify docstrings were loaded and processed
        assert "test_module.func" in parser.docstring
        assert parser.docstring["test_module.func"] == "processed: Loaded function docstring"
        assert "test_module.MyClass" in parser.docstring
        assert parser.docstring["test_module.MyClass"] == "processed: Loaded class docstring"
        assert "other_module.func" not in parser.docstring


def test_Parser_load_docstring_no_docstring():
    """Test Parser.load_docstring with None docstrings."""
    from types import ModuleType
    from unittest.mock import MagicMock, patch
    
    parser = Parser()
    root = "test_module"
    parser.doc[root] = "Module doc"
    parser.doc["test_module.func"] = "Function doc"
    parser.docstring = {}
    
    mock_module = MagicMock(spec=ModuleType)
    
    with patch('__main__._attr') as mock_attr, \
         patch('__main__.getdoc') as mock_getdoc:
        
        mock_attr.return_value = MagicMock()
        mock_getdoc.return_value = None
        
        parser.load_docstring(root, mock_module)
        
        # Docstring should not be added if getdoc returns None
        assert "test_module.func" not in parser.docstring


def test_Parser_load_docstring_empty_doc():
    """Test Parser.load_docstring with empty doc dictionary."""
    from types import ModuleType
    from unittest.mock import MagicMock
    
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    
    mock_module = MagicMock(spec=ModuleType)
    
    # Should not raise any errors
    parser.load_docstring("test_module", mock_module)
    
    assert parser.docstring == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_Parser_imports():
    """Test Parser.imports method."""
    parser = Parser()
    root = "test_module"
    
    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "os")] == "os"
    
    # Test Import node with asname
    import_node = Import(names=[alias(name="collections", asname="col")])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "col")] == "collections"
    
    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(
        module="typing",
        names=[alias(name="List", asname=None)],
        level=0
    )
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "List")] == _m("", "typing", "List")
    
    # Test ImportFrom node with asname
    import_from_node = ImportFrom(
        module="typing",
        names=[alias(name="Dict", asname="D")],
        level=0
    )
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "D")] == _m("", "typing", "Dict")
    
    # Test ImportFrom node with level > 0
    parser2 = Parser()
    root2 = "pkg.sub.module"
    import_from_node = ImportFrom(
        module="utils",
        names=[alias(name="helper", asname=None)],
        level=1
    )
    parser2.imports(root2, import_from_node)
    parent_module = parent(root2, level=0)
    assert parser2.alias[_m(root2, "helper")] == _m(parent_module, "utils", "helper")
    
    # Test ImportFrom with multiple names
    import_from_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname=None), alias(name="getcwd", asname="cwd")],
        level=0
    )
    parser3 = Parser()
    root3 = "another_module"
    parser3.imports(root3, import_from_node)
    assert parser3.alias[_m(root3, "path")] == _m("", "os", "path")
    assert parser3.alias[_m(root3, "cwd")] == _m("", "os", "getcwd")
    
    # Test ImportFrom with None module
    import_from_node = ImportFrom(
        module=None,
        names=[alias(name="something", asname=None)],
        level=0
    )
    parser4 = Parser()
    root4 = "test"
    parser4.imports(root4, import_from_node)
    # Should not crash and not add to alias when module is None
    assert _m(root4, "something") not in parser4.alias


# LLM-generated content at query #30
#--------------------------

```python
def test_const_type():
    """Test const_type function with various AST expressions."""
    # Test Constant nodes
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"
    assert const_type(Constant(value=None)) == "NoneType"
    assert const_type(Constant(value=1+2j)) == "complex"
    
    # Test Tuple
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == "tuple[int, int]"
    
    # Test List
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == "list[int, int]"
    
    # Test Set
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == "set[int, int]"
    
    # Test Dict
    dict_node = Dict(keys=[Constant(value="a"), Constant(value="b")],
                     values=[Constant(value=1), Constant(value=2)])
    assert const_type(dict_node) == "dict[str, int]"
    
    # Test mixed type tuple (should return Any)
    mixed_tuple = Tuple(elts=[Constant(value=1), Constant(value="str")], ctx=Load())
    assert const_type(mixed_tuple) == ANY
    
    # Test empty tuple
    empty_tuple = Tuple(elts=[], ctx=Load())
    assert const_type(empty_tuple) == "tuple"
    
    # Test Call to built-in types
    bool_call = Call(func=Name(id="bool", ctx=Load()), args=[], keywords=[])
    assert const_type(bool_call) == "bool"
    
    int_call = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(int_call) == "int"
    
    float_call = Call(func=Name(id="float", ctx=Load()), args=[], keywords=[])
    assert const_type(float_call) == "float"
    
    str_call = Call(func=Name(id="str", ctx=Load()), args=[], keywords=[])
    assert const_type(str_call) == "str"
    
    complex_call = Call(func=Name(id="complex", ctx=Load()), args=[], keywords=[])
    assert const_type(complex_call) == "complex"
    
    # Test Call to unknown function
    unknown_call = Call(func=Name(id="unknown_func", ctx=Load()), args=[], keywords=[])
    assert const_type(unknown_call) == ANY
    
    # Test Name node (not a constant)
    name_node = Name(id="x", ctx=Load())
    assert const_type(name_node) == ANY
    
    # Test BinOp node (not a constant)
    binop_node = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(binop_node) == ANY
    
    # Test tuple with None element (should return empty string -> ANY)
    tuple_with_none = Tuple(elts=[None, Constant(value=1)], ctx=Load())
    assert const_type(tuple_with_none) == ANY
    
    # Test Dict with mixed value types (should return Any)
    mixed_dict = Dict(keys=[Constant(value="a")],
                      values=[Constant(value=1)])
    assert const_type(mixed_dict) == "dict[str, int]"
    
    # Test list with different types
    mixed_list = List(elts=[Constant(value=1), Constant(value="str")], ctx=Load())
    assert const_type(mixed_list) == ANY


# LLM-generated content at query #31
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    const_int = Constant(value=42)
    result = resolver.visit_Constant(const_int)
    assert result is const_int
    
    # Test with non-string constant (float)
    const_float = Constant(value=3.14)
    result = resolver.visit_Constant(const_float)
    assert result is const_float
    
    # Test with non-string constant (None)
    const_none = Constant(value=None)
    result = resolver.visit_Constant(const_none)
    assert result is const_none
    
    # Test with valid name string that can be parsed
    const_name = Constant(value="int")
    result = resolver.visit_Constant(const_name)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with valid attribute string
    const_attr = Constant(value="typing.List")
    result = resolver.visit_Constant(const_attr)
    assert isinstance(result, Attribute)
    
    # Test with invalid syntax string (should return original node)
    const_invalid = Constant(value="@#$%^&*()")
    result = resolver.visit_Constant(const_invalid)
    assert result is const_invalid
    
    # Test with string containing spaces (invalid syntax)
    const_spaces = Constant(value="not a valid expression !!!")
    result = resolver.visit_Constant(const_spaces)
    assert result is const_spaces
    
    # Test with empty string
    const_empty = Constant(value="")
    result = resolver.visit_Constant(const_empty)
    assert result is const_empty
    
    # Test with resolver that has aliases
    resolver_with_alias = Resolver("test_module", {"test_module.MyType": "typing.List[int]"})
    const_mytype = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(const_mytype)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test with self_ty set
    resolver_self = Resolver("test_module", {}, self_ty="T")
    const_t = Constant(value="T")
    result = resolver_self.visit_Constant(const_t)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #32
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    from ast import Attribute, Name, Load
    
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'
    
    # Test case 2: Keep non-typing attribute access
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='other_module', ctx=Load()), attr='SomeClass', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == 'other_module'
    assert result.attr == 'SomeClass'
    
    # Test case 3: Non-Name value should be returned as-is
    resolver = Resolver("test_module", {})
    inner_attr = Attribute(value=Name(id='obj', ctx=Load()), attr='prop', ctx=Load())
    node = Attribute(value=inner_attr, attr='nested', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result == node
    
    # Test case 4: typing attribute with different attr names
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Dict', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Dict'
    
    # Test case 5: typing attribute with Union
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Union', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Union'


# LLM-generated content at query #33
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test 1: Non-string constant should return unchanged
    const_int = Constant(value=42)
    result = resolver.visit_Constant(const_int)
    assert result is const_int
    assert result.value == 42
    
    # Test 2: Non-string constant (float)
    const_float = Constant(value=3.14)
    result = resolver.visit_Constant(const_float)
    assert result is const_float
    assert result.value == 3.14
    
    # Test 3: Non-string constant (None)
    const_none = Constant(value=None)
    result = resolver.visit_Constant(const_none)
    assert result is const_none
    assert result.value is None
    
    # Test 4: String constant that is valid Python name
    const_name = Constant(value="int")
    result = resolver.visit_Constant(const_name)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test 5: String constant that is a valid expression
    const_expr = Constant(value="List[str]")
    result = resolver.visit_Constant(const_expr)
    assert isinstance(result, Subscript)
    
    # Test 6: String constant that is invalid Python syntax
    const_invalid = Constant(value="not valid python @@@@")
    result = resolver.visit_Constant(const_invalid)
    assert result is const_invalid
    assert result.value == "not valid python @@@@"
    
    # Test 7: Empty string constant
    const_empty = Constant(value="")
    result = resolver.visit_Constant(const_empty)
    assert result is const_empty
    assert result.value == ""
    
    # Test 8: String with spaces (invalid as expression)
    const_spaces = Constant(value="  invalid  ")
    result = resolver.visit_Constant(const_spaces)
    assert isinstance(result, Constant)
    
    # Test 9: String constant with simple name and alias
    alias = {"test_module.MyType": "List[int]"}
    resolver_with_alias = Resolver("test_module", alias)
    const_alias = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(const_alias)
    assert isinstance(result, Name)
    assert result.id == "MyType"


# LLM-generated content at query #34
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    non_string_const = Constant(value=42)
    result = resolver.visit_Constant(non_string_const)
    assert result is non_string_const
    assert result.value == 42
    
    # Test with float constant
    float_const = Constant(value=3.14)
    result = resolver.visit_Constant(float_const)
    assert result is float_const
    assert result.value == 3.14
    
    # Test with None constant
    none_const = Constant(value=None)
    result = resolver.visit_Constant(none_const)
    assert result is none_const
    assert result.value is None
    
    # Test with valid string that parses to a name
    string_name_const = Constant(value="MyType")
    result = resolver.visit_Constant(string_name_const)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test with valid string that parses to an attribute
    string_attr_const = Constant(value="typing.List")
    result = resolver.visit_Constant(string_attr_const)
    assert isinstance(result, Attribute)
    
    # Test with invalid string that cannot be parsed
    invalid_string_const = Constant(value="not a valid python expression !!!")
    result = resolver.visit_Constant(invalid_string_const)
    assert result is invalid_string_const
    assert result.value == "not a valid python expression !!!"
    
    # Test with string containing only spaces
    space_string_const = Constant(value="   ")
    result = resolver.visit_Constant(space_string_const)
    assert isinstance(result, (Name, Constant))
    
    # Test with empty string
    empty_string_const = Constant(value="")
    result = resolver.visit_Constant(empty_string_const)
    assert result is empty_string_const or isinstance(result, Name)
    
    # Test with complex valid expression string
    string_subscript_const = Constant(value="List[int]")
    result = resolver.visit_Constant(string_subscript_const)
    assert isinstance(result, Subscript)
    
    # Test with alias resolution in string
    resolver_with_alias = Resolver("test_module", {"test_module.MyType": "str"})
    string_with_alias = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(string_with_alias)
    assert isinstance(result, Name)


# LLM-generated content at query #35
#--------------------------

```python
def test_Parser_class_api():
    """Test Parser.class_api method."""
    from ast import parse, ClassDef
    
    # Create a parser instance
    parser = Parser()
    
    # Test case 1: Class with bases
    script = """
class MyClass(BaseClass, OtherBase):
    '''Class docstring'''
    pass
"""
    root = "test_module"
    parser.parse(root, script)
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.MyClass", class_node.bases, class_node.body)
    assert "test_module.MyClass" in parser.doc
    assert "Bases" in parser.doc["test_module.MyClass"]
    
    # Test case 2: Class with members
    script2 = """
class MyClass2:
    '''Class docstring'''
    attr1: int = 5
    attr2: str
    _private: float = 3.14
"""
    root2 = "test_module2"
    parser2 = Parser()
    parser2.parse(root2, script2)
    tree2 = parse(script2)
    class_node2 = tree2.body[0]
    
    parser2.class_api(root2, "test_module2.MyClass2", class_node2.bases, class_node2.body)
    assert "test_module2.MyClass2" in parser2.doc
    assert "Members" in parser2.doc["test_module2.MyClass2"]
    
    # Test case 3: Enum class
    script3 = """
from enum import Enum

class MyEnum(Enum):
    '''Enum docstring'''
    VALUE1: int = 1
    VALUE2: str = "test"
"""
    root3 = "test_module3"
    parser3 = Parser()
    parser3.parse(root3, script3)
    tree3 = parse(script3)
    class_node3 = tree3.body[1]
    
    parser3.class_api(root3, "test_module3.MyEnum", class_node3.bases, class_node3.body)
    assert "test_module3.MyEnum" in parser3.doc
    
    # Test case 4: Class with deleted attributes
    script4 = """
class MyClass4:
    '''Class docstring'''
    attr1: int = 1
    attr2: str = "test"
    del attr1
"""
    root4 = "test_module4"
    parser4 = Parser()
    parser4.parse(root4, script4)
    tree4 = parse(script4)
    class_node4 = tree4.body[0]
    
    parser4.class_api(root4, "test_module4.MyClass4", class_node4.bases, class_node4.body)
    assert "test_module4.MyClass4" in parser4.doc
    
    # Test case 5: Empty class
    script5 = """
class EmptyClass:
    pass
"""
    root5 = "test_module5"
    parser5 = Parser()
    parser5.parse(root5, script5)
    tree5 = parse(script5)
    class_node5 = tree5.body[0]
    
    parser5.class_api(root5, "test_module5.EmptyClass", class_node5.bases, class_node5.body)
    assert "test_module5.EmptyClass" in parser5.doc
    assert "Bases" not in parser5.doc["test_module5.EmptyClass"]
    assert "Members" not in parser5.doc["test_module5.EmptyClass"]


# LLM-generated content at query #36
#--------------------------

```python
def test_Parser_class_api():
    """Unit test for Parser.class_api method."""
    from ast import parse, AnnAssign, Assign, Delete, Name, Constant
    
    # Create a parser instance
    parser = Parser()
    
    # Setup test data
    root = "test_module"
    name = "test_module.TestClass"
    parser.root[name] = root
    parser.level[root] = 0
    parser.alias = {}
    
    # Test case 1: Class with bases
    script = """
class MyClass(BaseClass, AnotherBase):
    pass
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert name in parser.doc
    assert "Bases" in parser.doc[name]
    
    # Test case 2: Class with members
    script = """
class MyClass:
    public_attr: int
    _private_attr: str
    another_member: float = 3.14
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert "Members" in parser.doc[name]
    
    # Test case 3: Enum class
    script = """
class MyEnum(enum.Enum):
    OPTION_A: int
    OPTION_B: str
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.alias["test_module.enum"] = "enum"
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert "Enums" in parser.doc[name]
    
    # Test case 4: Empty class
    script = "class EmptyClass: pass"
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, [], class_node.body)
    assert name in parser.doc
    
    # Test case 5: Class with deleted attributes
    script = """
class MyClass:
    attr: int
    del attr
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert name in parser.doc
    
    # Test case 6: Class with type comments
    script = """
class MyClass:
    value = 42  # type: int
"""
    tree = parse(script, type_comments=True)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert name in parser.doc


# LLM-generated content at query #37
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    from ast import parse, unparse, Load
    
    # Test case 1: Union type conversion to BitOr
    resolver = Resolver("mymodule", {"mymodule.Union": "typing.Union"})
    code_str = "Union[int, str]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert unparse(result) == "int | str"
    
    # Test case 2: Optional type conversion to BitOr with None
    resolver = Resolver("mymodule", {"mymodule.Optional": "typing.Optional"})
    code_str = "Optional[int]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert unparse(result) == "int | None"
    
    # Test case 3: Non-Name subscript value (should return unchanged)
    resolver = Resolver("mymodule", {})
    code_str = "obj.Optional[int]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, Subscript)
    
    # Test case 4: PEP585 deprecated type conversion
    resolver = Resolver("mymodule", {"mymodule.Dict": "typing.Dict"})
    code_str = "Dict[str, int]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "dict"
    
    # Test case 5: Union with single element (non-Tuple slice)
    resolver = Resolver("mymodule", {"mymodule.Union": "typing.Union"})
    code_str = "Union[int]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    # Non-Tuple slice should return the slice itself
    assert isinstance(result, Subscript)
    
    # Test case 6: Unknown subscript name (should return unchanged)
    resolver = Resolver("mymodule", {})
    code_str = "CustomType[int]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "CustomType"
    
    # Test case 7: Union with multiple types
    resolver = Resolver("mymodule", {"mymodule.Union": "typing.Union"})
    code_str = "Union[int, str, float]"
    node = parse(code_str).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, BinOp)
    assert unparse(result) == "int | str | float"


# LLM-generated content at query #38
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("test_module", {})
    
    # Test case 1: typing.List should be converted to List
    typing_attr = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == "List"
    
    # Test case 2: typing.Dict should be converted to Dict
    typing_dict = Attribute(Name("typing", Load()), "Dict", Load())
    result = resolver.visit_Attribute(typing_dict)
    assert isinstance(result, Name)
    assert result.id == "Dict"
    
    # Test case 3: other.List should remain unchanged
    other_attr = Attribute(Name("other", Load()), "List", Load())
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    assert isinstance(result.value, Name)
    assert result.value.id == "other"
    
    # Test case 4: typing.Optional should be converted to Optional
    typing_optional = Attribute(Name("typing", Load()), "Optional", Load())
    result = resolver.visit_Attribute(typing_optional)
    assert isinstance(result, Name)
    assert result.id == "Optional"
    
    # Test case 5: non-Name value should remain unchanged
    non_name_attr = Attribute(Constant("typing"), "List", Load())
    result = resolver.visit_Attribute(non_name_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    
    # Test case 6: module.submodule.Class should remain unchanged
    module_attr = Attribute(Name("collections", Load()), "abc", Load())
    result = resolver.visit_Attribute(module_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "abc"
    assert isinstance(result.value, Name)
    assert result.value.id == "collections"


# LLM-generated content at query #39
#--------------------------

```python
def test_Parser_imports():
    """Test Parser.imports method."""
    parser = Parser()
    
    # Test Import statement
    root = "test_module"
    parser.level[root] = 0
    
    import_node = Import(names=[alias(name='os', asname=None)])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, 'os')] == 'os'
    
    # Test Import with asname
    import_node_alias = Import(names=[alias(name='numpy', asname='np')])
    parser.imports(root, import_node_alias)
    assert parser.alias[_m(root, 'np')] == 'numpy'
    
    # Test ImportFrom with absolute import
    import_from_node = ImportFrom(
        module='os.path',
        names=[alias(name='join', asname=None)],
        level=0
    )
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, 'join')] == _m('', 'os.path', 'join')
    
    # Test ImportFrom with asname
    import_from_alias = ImportFrom(
        module='collections',
        names=[alias(name='defaultdict', asname='dd')],
        level=0
    )
    parser.imports(root, import_from_alias)
    assert parser.alias[_m(root, 'dd')] == _m('', 'collections', 'defaultdict')
    
    # Test ImportFrom with relative import (level > 0)
    root_nested = "package.submodule.test"
    import_from_relative = ImportFrom(
        module='utils',
        names=[alias(name='helper', asname=None)],
        level=1
    )
    parser.imports(root_nested, import_from_relative)
    parent_module = parent(root_nested, level=0)
    assert parser.alias[_m(root_nested, 'helper')] == _m(parent_module, 'utils', 'helper')
    
    # Test ImportFrom with level=2
    import_from_relative_2 = ImportFrom(
        module='config',
        names=[alias(name='settings', asname=None)],
        level=2
    )
    parser.imports(root_nested, import_from_relative_2)
    parent_module_2 = parent(root_nested, level=1)
    assert parser.alias[_m(root_nested, 'settings')] == _m(parent_module_2, 'config', 'settings')
    
    # Test ImportFrom with multiple names
    import_from_multi = ImportFrom(
        module='typing',
        names=[alias(name='List', asname=None), alias(name='Dict', asname=None)],
        level=0
    )
    parser.imports(root, import_from_multi)
    assert parser.alias[_m(root, 'List')] == _m('', 'typing', 'List')
    assert parser.alias[_m(root, 'Dict')] == _m('', 'typing', 'Dict')
    
    # Test ImportFrom with module=None
    import_from_none = ImportFrom(
        module=None,
        names=[alias(name='something', asname=None)],
        level=1
    )
    parser.imports(root, import_from_none)
    parent_none = parent(root, level=0)
    assert parser.alias[_m(root, 'something')] == _m(parent_none, '', 'something')


# LLM-generated content at query #40
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test 1: Non-string constant should be returned as-is
    const_int = Constant(value=42)
    result = resolver.visit_Constant(const_int)
    assert result is const_int
    assert result.value == 42
    
    # Test 2: Non-string constant (float) should be returned as-is
    const_float = Constant(value=3.14)
    result = resolver.visit_Constant(const_float)
    assert result is const_float
    assert result.value == 3.14
    
    # Test 3: Non-string constant (None) should be returned as-is
    const_none = Constant(value=None)
    result = resolver.visit_Constant(const_none)
    assert result is const_none
    assert result.value is None
    
    # Test 4: String constant with valid name should be parsed and visited
    const_name = Constant(value="int")
    result = resolver.visit_Constant(const_name)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test 5: String constant with invalid syntax should be returned as-is
    const_invalid = Constant(value="invalid syntax @@")
    result = resolver.visit_Constant(const_invalid)
    assert result is const_invalid
    assert result.value == "invalid syntax @@"
    
    # Test 6: String constant with complex expression should be parsed
    const_expr = Constant(value="List[str]")
    result = resolver.visit_Constant(const_expr)
    assert isinstance(result, Subscript)
    
    # Test 7: String constant with alias resolution
    resolver_with_alias = Resolver("test_module", {"test_module.MyType": "int"})
    const_alias = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(const_alias)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test 8: Empty string constant
    const_empty = Constant(value="")
    result = resolver.visit_Constant(const_empty)
    # Empty string is invalid Python syntax, should return original
    assert result is const_empty


# LLM-generated content at query #41
#--------------------------

```python
def test_Resolver_visit_Name():
    """Test Resolver.visit_Name method."""
    from ast import Name, Load
    
    # Test 1: Replace self_ty with "Self"
    resolver = Resolver("mymodule", {}, self_ty="MyClass")
    node = Name("MyClass", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    
    # Test 2: Keep non-self_ty names unchanged when not in alias
    resolver = Resolver("mymodule", {}, self_ty="")
    node = Name("SomeName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeName"
    
    # Test 3: Replace name with alias expression
    alias = {"mymodule.MyType": "int"}
    resolver = Resolver("mymodule", alias)
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test 4: Don't replace TypeVar definitions
    alias = {"mymodule.T": "typing.TypeVar('T')", "typing.TypeVar": "typing.TypeVar"}
    resolver = Resolver("mymodule", alias)
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"
    
    # Test 5: Skip circular alias (name in its own definition)
    alias = {"mymodule.MyType": "mymodule.MyType"}
    resolver = Resolver("mymodule", alias)
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test 6: Replace with complex alias
    alias = {"mymodule.Container": "list[int]"}
    resolver = Resolver("mymodule", alias)
    node = Name("Container", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)
    
    # Test 7: Name not in alias with different root
    alias = {"other.Type": "str"}
    resolver = Resolver("mymodule", alias)
    node = Name("Type", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Type"


# LLM-generated content at query #42
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with value
    ann_assign = AnnAssign(
        target=Name(id="var1", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert _m(root, "var1") in parser.alias
    assert parser.alias[_m(root, "var1")] == "42"
    
    # Test AnnAssign without value (should return early)
    ann_assign_no_val = AnnAssign(
        target=Name(id="var2", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, ann_assign_no_val)
    assert _m(root, "var2") not in parser.alias
    
    # Test Assign with type_comment
    assign_with_comment = Assign(
        targets=[Name(id="var3", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals(root, assign_with_comment)
    assert _m(root, "var3") in parser.alias
    assert _m(root, "var3") in parser.const
    assert parser.const[_m(root, "var3")] == "float"
    
    # Test Assign without type_comment
    assign_no_comment = Assign(
        targets=[Name(id="var4", ctx=Store())],
        value=Constant(value="hello"),
        type_comment=None
    )
    parser.globals(root, assign_no_comment)
    assert _m(root, "var4") in parser.alias
    assert _m(root, "var4") in parser.const
    assert parser.const[_m(root, "var4")] == "str"
    
    # Test uppercase constant
    assign_const = Assign(
        targets=[Name(id="CONST", ctx=Store())],
        value=Constant(value=100),
        type_comment=None
    )
    parser.globals(root, assign_const)
    assert _m(root, "CONST") in parser.const
    assert _m(root, "CONST") in parser.root
    assert parser.root[_m(root, "CONST")] == root
    
    # Test __all__ assignment
    all_assign = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="func1"), Constant(value="Class1")],
            ctx=Load()
        ),
        type_comment=None
    )
    parser.globals(root, all_assign)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "Class1") in parser.imp[root]
    
    # Test invalid target (should return early)
    invalid_assign = Assign(
        targets=[Tuple(elts=[Name(id="a", ctx=Store())], ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, invalid_assign)
    
    # Test multiple targets (should return early)
    multi_assign = Assign(
        targets=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals(root, multi_assign)
    assert _m(root, "x") not in parser.alias


# LLM-generated content at query #43
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with value
    node = AnnAssign(
        target=Name(id="MyType", ctx=Store()),
        annotation=Subscript(
            value=Name(id="list", ctx=Load()),
            slice=Name(id="str", ctx=Load()),
            ctx=Load()
        ),
        value=List(elts=[], ctx=Load()),
        simple=1
    )
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    parser.globals(root, node)
    
    assert _m(root, "MyType") in parser.alias
    assert parser.alias[_m(root, "MyType")] == "[]"
    
    # Test uppercase constant
    node2 = AnnAssign(
        target=Name(id="CONSTANT", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node2)
    
    assert parser.root[_m(root, "CONSTANT")] == root
    assert _m(root, "CONSTANT") in parser.const
    
    # Test Assign without type comment
    node3 = Assign(
        targets=[Name(id="VAR", ctx=Store())],
        value=Constant(value="string")
    )
    parser.globals(root, node3)
    
    assert _m(root, "VAR") in parser.alias
    assert parser.const[_m(root, "VAR")] == "str"
    
    # Test __all__ filter
    node4 = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="func"), Constant(value="Class")],
            ctx=Load()
        )
    )
    parser.globals(root, node4)
    
    assert _m(root, "func") in parser.imp[root]
    assert _m(root, "Class") in parser.imp[root]
    
    # Test Assign with type comment
    node5 = Assign(
        targets=[Name(id="TYPED", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals(root, node5)
    
    assert parser.const[_m(root, "TYPED")] == "float"
    
    # Test invalid node (should return early)
    node6 = Assign(
        targets=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals(root, node6)
    
    assert _m(root, "x") not in parser.alias


# LLM-generated content at query #44
#--------------------------

```python
def test_const_type():
    """Test const_type function with various AST nodes."""
    # Test Constant nodes
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"
    assert const_type(Constant(value=None)) == "NoneType"
    assert const_type(Constant(value=1+2j)) == "complex"
    
    # Test Tuple nodes
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == "tuple[int, int]"
    
    tuple_mixed = Tuple(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(tuple_mixed) == "tuple[Any, Any]"
    
    # Test List nodes
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == "list[int, int]"
    
    list_empty = List(elts=[], ctx=Load())
    assert const_type(list_empty) == "list"
    
    # Test Set nodes
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == "set[int, int]"
    
    # Test Dict nodes
    dict_node = Dict(keys=[Constant(value="a")], values=[Constant(value=1)])
    assert const_type(dict_node) == "dict[str, int]"
    
    dict_empty = Dict(keys=[], values=[])
    assert const_type(dict_empty) == "dict"
    
    # Test Call nodes with builtin types
    call_int = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(call_int) == "int"
    
    call_str = Call(func=Name(id="str", ctx=Load()), args=[], keywords=[])
    assert const_type(call_str) == "str"
    
    call_bool = Call(func=Name(id="bool", ctx=Load()), args=[], keywords=[])
    assert const_type(call_bool) == "bool"
    
    call_float = Call(func=Name(id="float", ctx=Load()), args=[], keywords=[])
    assert const_type(call_float) == "float"
    
    call_complex = Call(func=Name(id="complex", ctx=Load()), args=[], keywords=[])
    assert const_type(call_complex) == "complex"
    
    # Test Call nodes with attribute (e.g., module.func)
    call_attr = Call(
        func=Attribute(value=Name(id="module", ctx=Load()), attr="list", ctx=Load()),
        args=[],
        keywords=[]
    )
    assert const_type(call_attr) == ANY
    
    # Test Call nodes with unknown function
    call_unknown = Call(func=Name(id="unknown", ctx=Load()), args=[], keywords=[])
    assert const_type(call_unknown) == ANY
    
    # Test BinOp (should return ANY)
    binop = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(binop) == ANY
    
    # Test Name node (should return ANY)
    name_node = Name(id="x", ctx=Load())
    assert const_type(name_node) == ANY
    
    # Test Tuple with None element (should return empty string behavior)
    tuple_with_none = Tuple(elts=[Constant(value=1), None], ctx=Load())
    assert const_type(tuple_with_none) == "tuple"
    
    # Test List with mixed types
    list_mixed = List(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(list_mixed) == "list[Any, Any]"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_is_public_family():
    """Test is_public_family function."""
    # Test fully public names
    assert is_public_family("module")
    assert is_public_family("module.submodule")
    assert is_public_family("module.submodule.Class")
    assert is_public_family("module.Class.method")
    
    # Test names with magic methods (should be allowed)
    assert is_public_family("module.__init__")
    assert is_public_family("module.__init__.submodule")
    assert is_public_family("__main__")
    assert is_public_family("__main__.module")
    assert is_public_family("module.__dict__.submodule")
    
    # Test names with private/local names (underscore prefix)
    assert not is_public_family("_private")
    assert not is_public_family("module._private")
    assert not is_public_family("module._private.Class")
    assert not is_public_family("module.Class._private")
    assert not is_public_family("_module.public")
    
    # Test names with dunder in middle (should fail if not at both ends)
    assert is_public_family("module.normal_name")
    assert is_public_family("module.name_with_underscores")
    
    # Test edge cases
    assert is_public_family("A")
    assert not is_public_family("_A")
    assert is_public_family("__A__")
    assert not is_public_family("module.__A")
    assert not is_public_family("module.A_")


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with value
    node = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "x") in parser.alias
    assert parser.alias[_m(root, "x")] == "42"
    
    # Test uppercase constant
    node = AnnAssign(
        target=Name(id="CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=100),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "CONST") in parser.const
    assert parser.const[_m(root, "CONST")] == "int"
    
    # Test Assign with single target
    node = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value="hello"),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "y") in parser.alias
    
    # Test Assign with type comment
    node = Assign(
        targets=[Name(id="z", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals(root, node)
    assert _m(root, "z") in parser.alias
    
    # Test __all__ filter
    node = AnnAssign(
        target=Name(id="__all__", ctx=Store()),
        annotation=Name(id="list", ctx=Load()),
        value=List(elts=[Constant(value="public_func")], ctx=Load()),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "public_func") in parser.imp[root]
    
    # Test Assign with __all__
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test invalid node (should return early)
    node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    initial_len = len(parser.alias)
    parser.globals(root, node)
    assert len(parser.alias) == initial_len
    
    # Test AnnAssign without value (should return early)
    node = AnnAssign(
        target=Name(id="unassigned", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    initial_len = len(parser.alias)
    parser.globals(root, node)
    assert len(parser.alias) == initial_len
    
    # Test lowercase non-constant
    node = Assign(
        targets=[Name(id="local_var", ctx=Store())],
        value=Constant(value=42),
        type_comment=None
    )
    parser.globals(root, node)
    assert _m(root, "local_var") not in parser.const
    assert _m(root, "local_var") in parser.alias


# LLM-generated content at query #3
#--------------------------

```python
def test_Parser_compile():
    """Test Parser.compile method."""
    # Test basic compilation with empty parser
    p = Parser()
    result = p.compile()
    assert isinstance(result, str)
    assert result.endswith('\n')
    
    # Test compilation with single module
    p = Parser()
    p.parse('test_module', 'def foo(): pass')
    result = p.compile()
    assert isinstance(result, str)
    assert 'test_module' in result
    
    # Test compilation with toc enabled
    p = Parser(toc=True)
    p.parse('test_module', 'def foo(): pass')
    result = p.compile()
    assert isinstance(result, str)
    assert '**Table of contents:**' in result
    
    # Test compilation with multiple levels
    p = Parser()
    p.parse('pkg.module1', 'def func1(): pass')
    p.parse('pkg.module2', 'def func2(): pass')
    result = p.compile()
    assert isinstance(result, str)
    assert 'pkg' in result or 'func' in result
    
    # Test compilation with constants
    p = Parser()
    p.parse('test_module', 'CONST = 42')
    p.const['test_module.CONST'] = 'int'
    p.root['test_module.CONST'] = 'test_module'
    p.doc['test_module.CONST'] = '## CONST\n\n'
    p.docstring['test_module.CONST'] = 'A constant\n'
    result = p.compile()
    assert isinstance(result, str)
    
    # Test compilation with docstring
    p = Parser()
    p.doc['test_module'] = '# Module test_module\n\n'
    p.docstring['test_module'] = 'Module documentation\n'
    p.root['test_module'] = 'test_module'
    p.imp['test_module'] = set()
    p.level['test_module'] = 0
    result = p.compile()
    assert 'Module documentation' in result
    
    # Test compilation with link enabled
    p = Parser(link=True)
    p.doc['test_module'] = '# Module `{}`\n<a id=\"{}\"></a>\n\n'
    p.root['test_module'] = 'test_module'
    p.imp['test_module'] = set()
    p.level['test_module'] = 0
    result = p.compile()
    assert isinstance(result, str)
    
    # Test compilation with magic methods (should be skipped without docstring)
    p = Parser()
    p.doc['test_module.__init__'] = '## __init__\n\n'
    p.root['test_module.__init__'] = 'test_module'
    p.imp['test_module'] = set()
    p.level['test_module.__init__'] = 1
    result = p.compile()
    assert '__init__' not in result  # Magic methods skipped if no docstring
    
    # Test compilation with __all__ filtering
    p = Parser()
    p.parse('test_module', '__all__ = ["func1"]')
    p.doc['test_module'] = '# Module test_module\n\n'
    p.doc['test_module.func1'] = '## func1\n\n'
    p.docstring['test_module'] = 'Module doc\n'
    p.docstring['test_module.func1'] = 'Function doc\n'
    p.root['test_module'] = 'test_module'
    p.root['test_module.func1'] = 'test_module'
    p.level['test_module'] = 0
    p.level['test_module.func1'] = 1
    result = p.compile()
    assert isinstance(result, str)
    
    # Test compilation result format
    p = Parser()
    p.doc['module'] = '# Module\n\n'
    p.docstring['module'] = 'Documentation\n'
    p.root['module'] = 'module'
    p.imp['module'] = set()
    p.level['module'] = 0
    result = p.compile()
    assert result.endswith('\n')
    assert '# Module' in result
    assert 'Documentation' in result


# LLM-generated content at query #4
#--------------------------

```python
def test_Parser_api():
    """Test Parser.api method."""
    parser = Parser()
    root = "test_module"
    
    # Test with FunctionDef
    func_code = """
def example_func(x: int) -> str:
    '''Example function.'''
    return str(x)
"""
    func_tree = parse(func_code)
    func_node = func_tree.body[0]
    
    parser.level[root] = 0
    parser.root[root] = root
    parser.api(root, func_node)
    
    name = _m(root, func_node.name)
    assert name in parser.doc
    assert "example_func()" in parser.doc[name]
    assert "*Full name:*" in parser.doc[name]
    assert parser.level[name] == 0
    assert parser.root[name] == root
    
    # Test with AsyncFunctionDef
    async_code = """
async def async_func() -> None:
    '''Async function.'''
    pass
"""
    async_tree = parse(async_code)
    async_node = async_tree.body[0]
    
    parser.api(root, async_node)
    
    async_name = _m(root, async_node.name)
    assert async_name in parser.doc
    assert "async async_func()" in parser.doc[async_name]
    
    # Test with ClassDef
    class_code = """
class ExampleClass:
    '''Example class.'''
    pass
"""
    class_tree = parse(class_code)
    class_node = class_tree.body[0]
    
    parser.api(root, class_node)
    
    class_name = _m(root, class_node.name)
    assert class_name in parser.doc
    assert "class ExampleClass" in parser.doc[class_name]
    
    # Test with decorators
    decorated_code = """
@staticmethod
def decorated_func():
    '''Decorated function.'''
    pass
"""
    decorated_tree = parse(decorated_code)
    decorated_node = decorated_tree.body[0]
    
    parser.api(root, decorated_node)
    
    decorated_name = _m(root, decorated_node.name)
    assert decorated_name in parser.doc
    assert "Decorators" in parser.doc[decorated_name]
    
    # Test with prefix (nested class method)
    parser.api(root, func_node, prefix="OuterClass")
    
    nested_name = _m(root, "OuterClass", func_node.name)
    assert nested_name in parser.doc
    assert parser.root[nested_name] == root


# LLM-generated content at query #5
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    # Test Union type conversion to BitOr
    resolver = Resolver("test_module", {})
    union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(elts=[Name(id="int", ctx=Load()), Name(id="str", ctx=Load())], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test Optional type conversion
    resolver = Resolver("test_module", {})
    optional_node = Subscript(
        value=Name(id="Optional", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test PEP585 deprecated name warning
    resolver = Resolver("test_module", {})
    deprecated_node = Subscript(
        value=Name(id="List", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(deprecated_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    
    # Test non-Name subscript value (should return unchanged)
    resolver = Resolver("test_module", {})
    attr_node = Subscript(
        value=Attribute(value=Name(id="typing", ctx=Load()), attr="Dict", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(attr_node)
    assert result is attr_node
    
    # Test Union with single element in slice (not a Tuple)
    resolver = Resolver("test_module", {})
    union_single = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_single)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with alias resolution
    resolver = Resolver("test_module", {"test_module.Union": "typing.Union"})
    union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(elts=[Name(id="int", ctx=Load()), Name(id="str", ctx=Load())], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test unknown subscript type (should return unchanged)
    resolver = Resolver("test_module", {})
    unknown_node = Subscript(
        value=Name(id="CustomType", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(unknown_node)
    assert result is unknown_node


# LLM-generated content at query #6
#--------------------------

```python
def test_Parser_func_api():
    """Unit test for Parser.func_api method."""
    from ast import parse, arg, arguments
    
    # Create a Parser instance
    parser = Parser()
    
    # Test basic function with no arguments
    root = "test_module"
    name = "test_module.test_func"
    
    # Create arguments node for a simple function
    func_args = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.doc[name] = "## test_func()\n\n"
    parser.func_api(root, name, func_args, returns, has_self=False, cls_method=False)
    
    assert name in parser.doc
    assert "test_func()" in parser.doc[name]
    
    # Test function with arguments and defaults
    root2 = "test_module2"
    name2 = "test_module2.func_with_args"
    
    arg1 = arg(arg='x', annotation=parse('int').body[0].value)
    arg2 = arg(arg='y', annotation=parse('str').body[0].value)
    default_val = parse('""').body[0].value
    
    func_args2 = arguments(
        posonlyargs=[],
        args=[arg1, arg2],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[default_val]
    )
    
    parser.doc[name2] = "## func_with_args()\n\n"
    parser.func_api(root2, name2, func_args2, None, has_self=False, cls_method=False)
    
    assert name2 in parser.doc
    
    # Test method with self
    root3 = "test_module3"
    name3 = "test_module3.MyClass.method"
    
    self_arg = arg(arg='self', annotation=None)
    arg3 = arg(arg='value', annotation=parse('int').body[0].value)
    
    func_args3 = arguments(
        posonlyargs=[],
        args=[self_arg, arg3],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    parser.doc[name3] = "## method()\n\n"
    parser.func_api(root3, name3, func_args3, None, has_self=True, cls_method=False)
    
    assert name3 in parser.doc
    
    # Test classmethod
    root4 = "test_module4"
    name4 = "test_module4.MyClass.cls_method"
    
    cls_arg = arg(arg='cls', annotation=parse('type[MyClass]').body[0].value)
    
    func_args4 = arguments(
        posonlyargs=[],
        args=[cls_arg],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    parser.doc[name4] = "## cls_method()\n\n"
    parser.func_api(root4, name4, func_args4, None, has_self=True, cls_method=True)
    
    assert name4 in parser.doc
    
    # Test function with *args and **kwargs
    root5 = "test_module5"
    name5 = "test_module5.var_func"
    
    vararg = arg(arg='args', annotation=None)
    kwarg = arg(arg='kwargs', annotation=None)
    
    func_args5 = arguments(
        posonlyargs=[],
        args=[],
        vararg=vararg,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=kwarg,
        defaults=[]
    )
    
    parser.doc[name5] = "## var_func()\n\n"
    parser.func_api(root5, name5, func_args5, None, has_self=False, cls_method=False)
    
    assert name5 in parser.doc
    
    # Test function with keyword-only arguments
    root6 = "test_module6"
    name6 = "test_module6.kw_func"
    
    kwonly_arg = arg(arg='kwonly', annotation=parse('bool').body[0].value)
    
    func_args6 = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[kwonly_arg],
        kw_defaults=[None],
        kwarg=None,
        defaults=[]
    )
    
    parser.doc[name6] = "## kw_func()\n\n"
    parser.func_api(root6, name6, func_args6, None, has_self=False, cls_method=False)
    
    assert name6 in parser.doc


# LLM-generated content at query #7
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    non_string_const = Constant(value=42)
    result = resolver.visit_Constant(non_string_const)
    assert result is non_string_const
    
    # Test with string that is a valid name
    string_const = Constant(value="x")
    result = resolver.visit_Constant(string_const)
    assert isinstance(result, Name)
    assert result.id == "x"
    
    # Test with string that is a valid expression
    string_const = Constant(value="int")
    result = resolver.visit_Constant(string_const)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with invalid syntax string
    invalid_const = Constant(value="@invalid syntax!")
    result = resolver.visit_Constant(invalid_const)
    assert result is invalid_const
    
    # Test with float constant
    float_const = Constant(value=3.14)
    result = resolver.visit_Constant(float_const)
    assert result is float_const
    
    # Test with None constant
    none_const = Constant(value=None)
    result = resolver.visit_Constant(none_const)
    assert result is none_const
    
    # Test with complex expression string
    expr_const = Constant(value="List[int]")
    result = resolver.visit_Constant(expr_const)
    assert isinstance(result, Subscript)
    
    # Test with string containing attribute access
    attr_const = Constant(value="typing.List")
    result = resolver.visit_Constant(attr_const)
    assert isinstance(result, Attribute)


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_imports():
    """Test Parser.imports method."""
    parser = Parser()
    
    # Test Import node
    root = "test_module"
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "os")] == "os"
    
    import_node_alias = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_alias)
    assert parser.alias[_m(root, "np")] == "numpy"
    
    # Test ImportFrom node without level
    import_from_node = ImportFrom(
        module="collections",
        names=[alias(name="defaultdict", asname=None)],
        level=0
    )
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "defaultdict")] == "collections.defaultdict"
    
    # Test ImportFrom with asname
    import_from_alias = ImportFrom(
        module="typing",
        names=[alias(name="List", asname="L")],
        level=0
    )
    parser.imports(root, import_from_alias)
    assert parser.alias[_m(root, "L")] == "typing.List"
    
    # Test ImportFrom with relative import (level > 0)
    parser.root["parent.test_module"] = "parent.test_module"
    import_from_relative = ImportFrom(
        module="sibling",
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports("parent.test_module", import_from_relative)
    assert parser.alias[_m("parent.test_module", "func")] == "parent.sibling.func"
    
    # Test ImportFrom with level=2
    parser.root["parent.child.test_module"] = "parent.child.test_module"
    import_from_relative2 = ImportFrom(
        module="module",
        names=[alias(name="item", asname=None)],
        level=2
    )
    parser.imports("parent.child.test_module", import_from_relative2)
    assert parser.alias[_m("parent.child.test_module", "item")] == "parent.module.item"
    
    # Test ImportFrom with multiple names
    import_from_multi = ImportFrom(
        module="math",
        names=[
            alias(name="sin", asname=None),
            alias(name="cos", asname="cosine"),
            alias(name="pi", asname=None)
        ],
        level=0
    )
    parser.imports(root, import_from_multi)
    assert parser.alias[_m(root, "sin")] == "math.sin"
    assert parser.alias[_m(root, "cosine")] == "math.cos"
    assert parser.alias[_m(root, "pi")] == "math.pi"
    
    # Test ImportFrom with None module (from . import x)
    import_from_dot = ImportFrom(
        module=None,
        names=[alias(name="local_func", asname=None)],
        level=1
    )
    parser.imports("package.module", import_from_dot)
    assert parser.alias[_m("package.module", "local_func")] == "package.local_func"


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_globals():
    """Unit test for Parser.globals method."""
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with value
    node = AnnAssign(
        target=Name(id="MyType", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=Constant(value="SomeType"),
        simple=1
    )
    parser.globals(root, node)
    assert _m(root, "MyType") in parser.alias
    assert parser.alias[_m(root, "MyType")] == "SomeType"
    
    # Test uppercase constant
    node2 = AnnAssign(
        target=Name(id="CONSTANT", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node2)
    assert _m(root, "CONSTANT") in parser.const
    assert parser.const[_m(root, "CONSTANT")] == "int"
    assert _m(root, "CONSTANT") in parser.root
    
    # Test Assign with type comment
    node3 = Assign(
        targets=[Name(id="var", ctx=Store())],
        value=Constant(value=100),
        type_comment="int"
    )
    parser.globals(root, node3)
    assert _m(root, "var") in parser.alias
    assert parser.alias[_m(root, "var")] == "100"
    
    # Test Assign without type comment (inferred type)
    node4 = Assign(
        targets=[Name(id="num", ctx=Store())],
        value=Constant(value=3.14),
        type_comment=None
    )
    parser.globals(root, node4)
    assert _m(root, "num") in parser.alias
    assert parser.const[_m(root, "num")] == "float"
    
    # Test __all__ filter with List
    node5 = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(
            elts=[
                Constant(value="func1"),
                Constant(value="Class1")
            ],
            ctx=Load()
        ),
        type_comment=None
    )
    parser.imp[root] = set()
    parser.globals(root, node5)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "Class1") in parser.imp[root]
    
    # Test __all__ filter with Tuple
    node6 = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="item1")],
            ctx=Load()
        ),
        type_comment=None
    )
    parser.globals(root, node6)
    assert _m(root, "item1") in parser.imp[root]
    
    # Test invalid node (should return early)
    node7 = Assign(
        targets=[
            Name(id="a", ctx=Store()),
            Name(id="b", ctx=Store())
        ],
        value=Constant(value=1),
        type_comment=None
    )
    initial_size = len(parser.alias)
    parser.globals(root, node7)
    assert len(parser.alias) == initial_size
    
    # Test AnnAssign without value (should return early)
    node8 = AnnAssign(
        target=Name(id="annotated", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    initial_size = len(parser.alias)
    parser.globals(root, node8)
    assert len(parser.alias) == initial_size


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_globals():
    """Test Parser.globals method for handling assignments and annotations."""
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.alias[root] = root
    
    # Test AnnAssign with value
    ann_assign = AnnAssign(
        target=Name(id="TypeAlias", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=Constant(value="test"),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert _m(root, "TypeAlias") in parser.alias
    assert parser.alias[_m(root, "TypeAlias")] == "'test'"
    
    # Test Assign with uppercase constant
    assign = Assign(
        targets=[Name(id="CONST", ctx=Store())],
        value=Constant(value=42),
        type_comment=None
    )
    parser.globals(root, assign)
    assert _m(root, "CONST") in parser.const
    assert parser.const[_m(root, "CONST")] == "int"
    
    # Test Assign with type_comment
    assign_with_comment = Assign(
        targets=[Name(id="VAR", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals(root, assign_with_comment)
    assert _m(root, "VAR") in parser.alias
    assert parser.const[_m(root, "VAR")] == "float"
    
    # Test __all__ filter
    all_assign = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="func1"), Constant(value="Class1")],
            ctx=Load()
        ),
        type_comment=None
    )
    parser.imp[root] = set()
    parser.globals(root, all_assign)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "Class1") in parser.imp[root]
    
    # Test invalid assignments (should return early)
    invalid_assign = Assign(
        targets=[Tuple(elts=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())], ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )
    initial_len = len(parser.alias)
    parser.globals(root, invalid_assign)
    assert len(parser.alias) == initial_len
    
    # Test AnnAssign without value
    ann_assign_no_value = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, ann_assign_no_value)
    assert _m(root, "x") not in parser.alias


# LLM-generated content at query #11
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    from ast import parse, Subscript, Name, Load, Tuple, Constant, BinOp, BitOr
    
    # Test case 1: Union type conversion to BitOr
    resolver = Resolver("test_module", {
        "test_module.Union": "typing.Union"
    })
    union_code = "Union[int, str]"
    union_node = cast(Subscript, parse(union_code).body[0].value)
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test case 2: Optional type conversion to BitOr with None
    resolver = Resolver("test_module", {
        "test_module.Optional": "typing.Optional"
    })
    optional_code = "Optional[int]"
    optional_node = cast(Subscript, parse(optional_code).body[0].value)
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test case 3: PEP585 deprecated type warning
    resolver = Resolver("test_module", {
        "test_module.List": "typing.List"
    })
    list_code = "List[int]"
    list_node = cast(Subscript, parse(list_code).body[0].value)
    result = resolver.visit_Subscript(list_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    
    # Test case 4: Non-Name value returns unchanged
    resolver = Resolver("test_module", {})
    attr_code = "some_attr.Union[int, str]"
    attr_node = cast(Subscript, parse(attr_code).body[0].value)
    result = resolver.visit_Subscript(attr_node)
    assert result == attr_node
    
    # Test case 5: Unknown subscript type returns unchanged
    resolver = Resolver("test_module", {})
    unknown_code = "SomeType[int]"
    unknown_node = cast(Subscript, parse(unknown_code).body[0].value)
    result = resolver.visit_Subscript(unknown_node)
    assert result == unknown_node
    
    # Test case 6: Union with multiple types
    resolver = Resolver("test_module", {
        "test_module.Union": "typing.Union"
    })
    multi_union_code = "Union[int, str, float]"
    multi_union_node = cast(Subscript, parse(multi_union_code).body[0].value)
    result = resolver.visit_Subscript(multi_union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test case 7: Optional with non-Tuple slice
    resolver = Resolver("test_module", {
        "test_module.Optional": "typing.Optional"
    })
    optional_simple = "Optional[int]"
    optional_simple_node = cast(Subscript, parse(optional_simple).body[0].value)
    result = resolver.visit_Subscript(optional_simple_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)


# LLM-generated content at query #12
#--------------------------

```python
def test_Parser_imports():
    """Test Parser.imports method."""
    parser = Parser()
    root = "test_module"
    
    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "os")] == "os"
    
    # Test Import node with asname
    import_node_alias = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_alias)
    assert parser.alias[_m(root, "np")] == "numpy"
    
    # Test ImportFrom without level
    import_from = ImportFrom(module="typing", names=[alias(name="List", asname=None)], level=0)
    parser.imports(root, import_from)
    assert parser.alias[_m(root, "List")] == "typing.List"
    
    # Test ImportFrom with asname
    import_from_alias = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    parser.imports(root, import_from_alias)
    assert parser.alias[_m(root, "dd")] == "collections.defaultdict"
    
    # Test ImportFrom with level=1 (relative import)
    import_from_relative = ImportFrom(module="submodule", names=[alias(name="func", asname=None)], level=1)
    parser.imports(root, import_from_relative)
    expected_key = _m(root, "func")
    expected_value = _m(parent(root, level=0), "submodule", "func")
    assert parser.alias[expected_key] == expected_value
    
    # Test ImportFrom with multiple names
    import_from_multi = ImportFrom(module="os.path", names=[alias(name="join", asname=None), alias(name="exists", asname="path_exists")], level=0)
    parser.imports(root, import_from_multi)
    assert parser.alias[_m(root, "join")] == "os.path.join"
    assert parser.alias[_m(root, "path_exists")] == "os.path.exists"
    
    # Test ImportFrom with level=2 (relative import two levels up)
    import_from_relative_2 = ImportFrom(module="utils", names=[alias(name="helper", asname=None)], level=2)
    parser.imports(root, import_from_relative_2)
    expected_key = _m(root, "helper")
    expected_value = _m(parent(root, level=1), "utils", "helper")
    assert parser.alias[expected_key] == expected_value


# LLM-generated content at query #13
#--------------------------

```python
def test_Parser_func_ann():
    """Unit test for Parser.func_ann method."""
    from ast import arg, parse, FunctionDef
    
    # Test case 1: Function with self parameter
    parser = Parser()
    parser.alias = {}
    
    func_code = "def method(self, x: int, y: str) -> bool: pass"
    tree = parse(func_code)
    func_node = tree.body[0]
    
    result = list(parser.func_ann("test_module", func_node.args, has_self=True, cls_method=False))
    assert result[0] == 'Self'
    assert result[1] == 'int'
    assert result[2] == 'str'
    assert result[3] == 'bool'
    
    # Test case 2: Class method with self parameter
    result = list(parser.func_ann("test_module", func_node.args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'
    assert result[1] == 'int'
    assert result[2] == 'str'
    assert result[3] == 'bool'
    
    # Test case 3: Function without annotations
    func_code_no_ann = "def func(a, b, c): pass"
    tree = parse(func_code_no_ann)
    func_node = tree.body[0]
    
    result = list(parser.func_ann("test_module", func_node.args, has_self=False, cls_method=False))
    assert all(r == 'Any' for r in result)
    
    # Test case 4: Function with *args and **kwargs
    func_code_args = "def func(a: int, *args: str, **kwargs: float) -> None: pass"
    tree = parse(func_code_args)
    func_node = tree.body[0]
    
    result = list(parser.func_ann("test_module", func_node.args, has_self=False, cls_method=False))
    assert result[0] == 'int'
    assert result[1] == 'str'
    assert result[2] == 'float'
    assert result[3] == 'None'
    
    # Test case 5: Function with keyword-only arguments
    func_code_kw = "def func(a: int, *, b: str) -> bool: pass"
    tree = parse(func_code_kw)
    func_node = tree.body[0]
    
    result = list(parser.func_ann("test_module", func_node.args, has_self=False, cls_method=False))
    assert result[0] == 'int'
    assert result[1] == ''  # The * separator
    assert result[2] == 'str'
    assert result[3] == 'bool'
    
    # Test case 6: Function with positional-only arguments
    func_code_posonly = "def func(a: int, /, b: str) -> bool: pass"
    tree = parse(func_code_posonly)
    func_node = tree.body[0]
    
    result = list(parser.func_ann("test_module", func_node.args, has_self=False, cls_method=False))
    assert result[0] == 'int'
    assert result[1] == ''  # The / separator
    assert result[2] == 'str'
    assert result[3] == 'bool'
    
    # Test case 7: Self type annotation with class method
    parser.alias = {"test_module.MyClass": "MyClass"}
    func_code_self_ann = "def method(self: 'MyClass', x: int) -> None: pass"
    tree = parse(func_code_self_ann)
    func_node = tree.body[0]
    
    result = list(parser.func_ann("test_module", func_node.args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'
    assert result[1] == 'int'
    assert result[2] == 'None'


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("test_module", {})
    
    # Test case 1: typing.List should be converted to List
    typing_attr = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == "List"
    
    # Test case 2: typing.Dict should be converted to Dict
    typing_dict = Attribute(Name("typing", Load()), "Dict", Load())
    result = resolver.visit_Attribute(typing_dict)
    assert isinstance(result, Name)
    assert result.id == "Dict"
    
    # Test case 3: typing.Optional should be converted to Optional
    typing_optional = Attribute(Name("typing", Load()), "Optional", Load())
    result = resolver.visit_Attribute(typing_optional)
    assert isinstance(result, Name)
    assert result.id == "Optional"
    
    # Test case 4: other_module.List should remain as Attribute
    other_attr = Attribute(Name("other_module", Load()), "List", Load())
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    assert isinstance(result.value, Name)
    assert result.value.id == "other_module"
    
    # Test case 5: nested attribute should remain unchanged
    nested_attr = Attribute(Attribute(Name("module", Load()), "submodule", Load()), "Type", Load())
    result = resolver.visit_Attribute(nested_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "Type"
    
    # Test case 6: typing.Union should be converted to Union
    typing_union = Attribute(Name("typing", Load()), "Union", Load())
    result = resolver.visit_Attribute(typing_union)
    assert isinstance(result, Name)
    assert result.id == "Union"


# LLM-generated content at query #15
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("test_module", {})
    
    # Test case 1: typing.List -> List
    typing_attr = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == "List"
    
    # Test case 2: other.List -> other.List (unchanged)
    other_attr = Attribute(Name("other", Load()), "List", Load())
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.value.id == "other"
    assert result.attr == "List"
    
    # Test case 3: typing.Dict -> Dict
    typing_dict = Attribute(Name("typing", Load()), "Dict", Load())
    result = resolver.visit_Attribute(typing_dict)
    assert isinstance(result, Name)
    assert result.id == "Dict"
    
    # Test case 4: typing.Optional -> Optional
    typing_optional = Attribute(Name("typing", Load()), "Optional", Load())
    result = resolver.visit_Attribute(typing_optional)
    assert isinstance(result, Name)
    assert result.id == "Optional"
    
    # Test case 5: Non-Name value (e.g., nested attribute) -> unchanged
    nested_attr = Attribute(Attribute(Name("typing", Load()), "foo", Load()), "bar", Load())
    result = resolver.visit_Attribute(nested_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "bar"
    
    # Test case 6: collections.List -> unchanged
    collections_attr = Attribute(Name("collections", Load()), "List", Load())
    result = resolver.visit_Attribute(collections_attr)
    assert isinstance(result, Attribute)
    assert result.value.id == "collections"
    assert result.attr == "List"


# LLM-generated content at query #16
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("module", {})
    
    # Test case 1: typing.List should be converted to List
    typing_attr = Attribute(Name('typing', Load()), 'List', Load())
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == 'List'
    
    # Test case 2: other.List should remain unchanged
    other_attr = Attribute(Name('other', Load()), 'List', Load())
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.value.id == 'other'
    assert result.attr == 'List'
    
    # Test case 3: Non-Name value should remain unchanged
    const_attr = Attribute(Constant('not_a_name'), 'attr', Load())
    result = resolver.visit_Attribute(const_attr)
    assert isinstance(result, Attribute)
    assert result.value == const_attr.value
    
    # Test case 4: typing.Optional should be converted to Optional
    typing_optional = Attribute(Name('typing', Load()), 'Optional', Load())
    result = resolver.visit_Attribute(typing_optional)
    assert isinstance(result, Name)
    assert result.id == 'Optional'
    
    # Test case 5: typing.Union should be converted to Union
    typing_union = Attribute(Name('typing', Load()), 'Union', Load())
    result = resolver.visit_Attribute(typing_union)
    assert isinstance(result, Name)
    assert result.id == 'Union'
    
    # Test case 6: typing with various other attributes
    typing_dict = Attribute(Name('typing', Load()), 'Dict', Load())
    result = resolver.visit_Attribute(typing_dict)
    assert isinstance(result, Name)
    assert result.id == 'Dict'
    
    # Test case 7: Nested attribute access should remain unchanged
    nested_attr = Attribute(Attribute(Name('module', Load()), 'sub', Load()), 'attr', Load())
    result = resolver.visit_Attribute(nested_attr)
    assert isinstance(result, Attribute)


# LLM-generated content at query #17
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    non_string_node = Constant(value=42)
    result = resolver.visit_Constant(non_string_node)
    assert result is non_string_node
    assert result.value == 42
    
    # Test with float constant
    float_node = Constant(value=3.14)
    result = resolver.visit_Constant(float_node)
    assert result is float_node
    
    # Test with None constant
    none_node = Constant(value=None)
    result = resolver.visit_Constant(none_node)
    assert result is none_node
    
    # Test with valid string expression - simple name
    string_name_node = Constant(value="int")
    result = resolver.visit_Constant(string_name_node)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with valid string expression - attribute
    string_attr_node = Constant(value="typing.List")
    result = resolver.visit_Constant(string_attr_node)
    assert isinstance(result, Attribute)
    
    # Test with invalid syntax string
    invalid_syntax_node = Constant(value="not valid python @@@@")
    result = resolver.visit_Constant(invalid_syntax_node)
    assert result is invalid_syntax_node
    assert result.value == "not valid python @@@@"
    
    # Test with empty string
    empty_string_node = Constant(value="")
    result = resolver.visit_Constant(empty_string_node)
    assert result is empty_string_node
    
    # Test with resolver that has alias
    resolver_with_alias = Resolver("test_module", {"test_module.MyType": "int"})
    string_with_alias_node = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(string_with_alias_node)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test with self_ty parameter
    resolver_with_self = Resolver("test_module", {}, self_ty="T")
    string_self_node = Constant(value="T")
    result = resolver_with_self.visit_Constant(string_self_node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #18
#--------------------------

```python
def test_Parser_class_api():
    """Test Parser.class_api method."""
    from ast import parse, AnnAssign, Assign, Name, Constant, Delete
    
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    
    # Setup parser state
    parser.root[name] = root
    parser.level[name] = 0
    parser.doc[name] = "## class TestClass\n\n"
    
    # Test with bases
    script = """
class TestClass(BaseClass, AnotherBase):
    pass
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert "Bases" in parser.doc[name]
    
    # Test with enum
    parser.doc[name] = "## class TestEnum\n\n"
    enum_name = "test_module.TestEnum"
    parser.root[enum_name] = root
    script_enum = """
class TestEnum(enum.Enum):
    MEMBER1 = 1
    MEMBER2 = 2
"""
    tree_enum = parse(script_enum)
    class_node_enum = tree_enum.body[0]
    parser.alias[f"{root}.enum"] = "enum"
    parser.class_api(root, enum_name, class_node_enum.bases, class_node_enum.body)
    assert "Enums" in parser.doc[enum_name]
    
    # Test with members
    parser.doc[name] = "## class TestClass\n\n"
    script_members = """
class TestClass:
    public_attr: int
    _private_attr: str
    another_public: float = 3.14
"""
    tree_members = parse(script_members)
    class_node_members = tree_members.body[0]
    parser.class_api(root, name, class_node_members.bases, class_node_members.body)
    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]
    
    # Test with deleted members
    parser.doc[name] = "## class TestClass\n\n"
    script_delete = """
class TestClass:
    attr1: int
    attr2: str
    del attr1
"""
    tree_delete = parse(script_delete)
    class_node_delete = tree_delete.body[0]
    parser.class_api(root, name, class_node_delete.bases, class_node_delete.body)
    # Should not include deleted members
    assert parser.doc[name]
    
    # Test empty class
    parser.doc[name] = "## class TestClass\n\n"
    script_empty = "class TestClass:\n    pass"
    tree_empty = parse(script_empty)
    class_node_empty = tree_empty.body[0]
    parser.class_api(root, name, class_node_empty.bases, class_node_empty.body)
    assert parser.doc[name] == "## class TestClass\n\n"


# LLM-generated content at query #19
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    # Test Union type conversion to BitOr
    resolver = Resolver("test_module", {})
    union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(elts=[Name(id="int", ctx=Load()), Name(id="str", ctx=Load())], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)

    # Test Optional type conversion
    resolver = Resolver("test_module", {"test_module.Optional": "typing.Optional"})
    optional_node = Subscript(
        value=Name(id="Optional", ctx=Load()),
        slice=Name(id="str", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test PEP585 deprecated name warning
    resolver = Resolver("test_module", {"test_module.List": "typing.List"})
    pep585_node = Subscript(
        value=Name(id="List", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(pep585_node)
    assert isinstance(result, Subscript)

    # Test non-Name value returns unchanged
    resolver = Resolver("test_module", {})
    non_name_node = Subscript(
        value=Attribute(value=Name(id="typing", ctx=Load()), attr="Union", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(non_name_node)
    assert result == non_name_node

    # Test Union with single slice (not Tuple)
    resolver = Resolver("test_module", {"test_module.Union": "typing.Union"})
    single_slice_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(single_slice_node)
    assert result == single_slice_node

    # Test Union with multiple elements
    resolver = Resolver("test_module", {"test_module.Union": "typing.Union"})
    multi_union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(elts=[
            Name(id="int", ctx=Load()),
            Name(id="str", ctx=Load()),
            Name(id="float", ctx=Load())
        ], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(multi_union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)

    # Test unknown subscript type returns unchanged
    resolver = Resolver("test_module", {})
    unknown_node = Subscript(
        value=Name(id="CustomType", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(unknown_node)
    assert result == unknown_node


# LLM-generated content at query #20
#--------------------------

```python
def test_Resolver_visit_Name():
    """Test Resolver.visit_Name method."""
    # Test 1: Replace self_ty with Self
    resolver = Resolver("module", {}, self_ty="T")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test 2: Non-self_ty name without alias
    resolver = Resolver("module", {})
    node = Name("SomeName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeName"

    # Test 3: Name with alias that resolves to a simple name
    resolver = Resolver("module", {"module.OldName": "NewName"})
    node = Name("OldName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "NewName"

    # Test 4: Name with alias that is TypeVar should not be replaced
    resolver = Resolver("module", {
        "module.T": "typing.TypeVar('T')",
        "module.TypeVar": "typing.TypeVar"
    })
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test 5: Name with circular alias (name in its own alias)
    resolver = Resolver("module", {"module.SelfRef": "module.SelfRef"})
    node = Name("SelfRef", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SelfRef"

    # Test 6: Name with alias that resolves to complex expression
    resolver = Resolver("module", {"module.ListInt": "List[int]"})
    node = Name("ListInt", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)

    # Test 7: Multiple root levels with alias
    resolver = Resolver("pkg.module", {"pkg.module.Type": "str"})
    node = Name("Type", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "str"

    # Test 8: Empty root with alias
    resolver = Resolver("", {"Type": "int"})
    node = Name("Type", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #21
#--------------------------

```python
def test_Resolver_visit_Name():
    """Test Resolver.visit_Name method."""
    # Test case 1: Replace self_ty with 'Self'
    resolver = Resolver("module", {}, self_ty="T")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    
    # Test case 2: Name not in alias, return as is
    resolver = Resolver("module", {}, self_ty="")
    node = Name("SomeName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeName"
    
    # Test case 3: Name in alias, resolve to expression
    resolver = Resolver("module", {"module.MyType": "int"}, self_ty="")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test case 4: Name in alias with circular reference, return as is
    resolver = Resolver("module", {"module.MyType": "module.MyType"}, self_ty="")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test case 5: TypeVar in alias, return node as is
    resolver = Resolver("module", {
        "module.T": "typing.TypeVar('T')",
        "module.typing": "typing"
    }, self_ty="")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"
    
    # Test case 6: Nested module resolution
    resolver = Resolver("pkg.module", {"pkg.module.Type": "str"}, self_ty="")
    node = Name("Type", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "str"
    
    # Test case 7: self_ty with different name
    resolver = Resolver("module", {}, self_ty="GenericType")
    node = Name("GenericType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    
    # Test case 8: Name matches self_ty exactly
    resolver = Resolver("module", {"module.X": "int"}, self_ty="X")
    node = Name("X", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #22
#--------------------------

```python
def test_walk_body():
    """Test walk_body function."""
    # Test simple body without control flow
    simple_body = parse("x = 1\ny = 2").body
    result = list(walk_body(simple_body))
    assert len(result) == 2
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Assign)
    
    # Test body with If statement
    if_code = parse("if True:\n    x = 1\nelse:\n    y = 2").body
    result = list(walk_body(if_code))
    assert len(result) == 2
    assert all(isinstance(node, Assign) for node in result)
    
    # Test body with nested If statement
    nested_if = parse("if True:\n    if False:\n        x = 1\n    y = 2").body
    result = list(walk_body(nested_if))
    assert len(result) == 2
    assert all(isinstance(node, Assign) for node in result)
    
    # Test body with Try statement
    try_code = parse("try:\n    x = 1\nexcept:\n    y = 2\nfinally:\n    z = 3").body
    result = list(walk_body(try_code))
    assert len(result) == 3
    assert all(isinstance(node, Assign) for node in result)
    
    # Test body with Try and multiple except handlers
    try_multi = parse("try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept KeyError:\n    z = 3").body
    result = list(walk_body(try_multi))
    assert len(result) == 3
    assert all(isinstance(node, Assign) for node in result)
    
    # Test body with Try else clause
    try_else = parse("try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3").body
    result = list(walk_body(try_else))
    assert len(result) == 3
    assert all(isinstance(node, Assign) for node in result)
    
    # Test empty body
    empty_body = []
    result = list(walk_body(empty_body))
    assert len(result) == 0
    
    # Test body with mixed statements and control flow
    mixed = parse("x = 1\nif True:\n    y = 2\nz = 3").body
    result = list(walk_body(mixed))
    assert len(result) == 3
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Assign)
    assert isinstance(result[2], Assign)
    
    # Test deeply nested If and Try
    complex_code = parse(
        "if True:\n"
        "    try:\n"
        "        x = 1\n"
        "    except:\n"
        "        y = 2\n"
        "    z = 3"
    ).body
    result = list(walk_body(complex_code))
    assert len(result) == 3
    assert all(isinstance(node, Assign) for node in result)


# LLM-generated content at query #23
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    from ast import parse, unparse, Load, Name, Subscript, Tuple, Constant, BinOp, BitOr
    
    # Test case 1: Union type conversion to BitOr
    resolver = Resolver("test_module", {
        "test_module.Union": "typing.Union"
    })
    union_node = parse("Union[int, str]").body[0].value
    result = resolver.visit(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert unparse(result) == "int | str"
    
    # Test case 2: Optional type conversion to BitOr with None
    resolver = Resolver("test_module", {
        "test_module.Optional": "typing.Optional"
    })
    optional_node = parse("Optional[int]").body[0].value
    result = resolver.visit(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert unparse(result) == "int | None"
    
    # Test case 3: PEP585 deprecated type replacement
    resolver = Resolver("test_module", {
        "test_module.Dict": "typing.Dict"
    })
    dict_node = parse("Dict[str, int]").body[0].value
    result = resolver.visit(dict_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "dict"
    
    # Test case 4: Non-Name subscript value (no transformation)
    resolver = Resolver("test_module", {})
    subscript_node = parse("some_func()[int]").body[0].value
    result = resolver.visit(subscript_node)
    assert isinstance(result, Subscript)
    
    # Test case 5: Regular subscript without special handling
    resolver = Resolver("test_module", {})
    regular_node = parse("List[int]").body[0].value
    result = resolver.visit(regular_node)
    assert isinstance(result, Subscript)
    
    # Test case 6: Union with multiple types
    resolver = Resolver("test_module", {
        "test_module.Union": "typing.Union"
    })
    multi_union = parse("Union[int, str, float]").body[0].value
    result = resolver.visit(multi_union)
    assert isinstance(result, BinOp)
    assert unparse(result) == "int | str | float"


# LLM-generated content at query #24
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    from ast import parse, unparse, Load
    
    # Test case 1: Union type conversion to BitOr
    resolver = Resolver("test", {
        "test.Union": "typing.Union"
    })
    union_code = "Union[int, str]"
    node = parse(union_code).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test case 2: Optional type conversion
    resolver = Resolver("test", {
        "test.Optional": "typing.Optional"
    })
    optional_code = "Optional[int]"
    node = parse(optional_code).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test case 3: PEP585 deprecated type warning
    resolver = Resolver("test", {
        "test.Dict": "typing.Dict"
    })
    dict_code = "Dict[str, int]"
    node = parse(dict_code).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, Subscript)
    
    # Test case 4: Non-Name value (attribute access)
    resolver = Resolver("test", {})
    attr_code = "some_module.Union[int, str]"
    node = parse(attr_code).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, Subscript)
    
    # Test case 5: Union with single element should return the element
    resolver = Resolver("test", {
        "test.Union": "typing.Union"
    })
    union_code = "Union[int, str, float]"
    node = parse(union_code).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, BinOp)
    
    # Test case 6: Non-Tuple slice for Union
    resolver = Resolver("test", {
        "test.Union": "typing.Union"
    })
    # This would be invalid Union syntax, but test the code path
    resolver_inst = Resolver("test", {})
    subscript_node = Subscript(Name("Union", Load()), Name("int", Load()), Load())
    result = resolver_inst.visit_Subscript(subscript_node)
    assert result == subscript_node
    
    # Test case 7: Regular subscript that doesn't match any special case
    resolver = Resolver("test", {})
    list_code = "List[int]"
    node = parse(list_code).body[0].value
    result = resolver.visit(node)
    assert isinstance(result, Subscript)


# LLM-generated content at query #25
#--------------------------

```python
def test_Parser_class_api():
    """Test Parser.class_api method."""
    from ast import parse, Name, Load
    
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    
    # Test with no bases and no members
    script = """
class TestClass:
    pass
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.level[root] = 0
    parser.root[name] = root
    parser.class_api(root, name, class_node.bases, class_node.body)
    assert name in parser.doc
    assert "Bases" not in parser.doc[name]
    
    # Test with bases
    script = """
class TestClass(BaseClass):
    pass
"""
    parser2 = Parser()
    parser2.level[root] = 0
    parser2.root[name] = root
    parser2.alias[_m(root, "BaseClass")] = "module.BaseClass"
    tree = parse(script)
    class_node = tree.body[0]
    parser2.class_api(root, name, class_node.bases, class_node.body)
    assert "Bases" in parser2.doc[name]
    
    # Test with members
    script = """
class TestClass:
    attr1: int
    attr2: str = "default"
"""
    parser3 = Parser()
    parser3.level[root] = 0
    parser3.root[name] = root
    tree = parse(script)
    class_node = tree.body[0]
    parser3.class_api(root, name, class_node.bases, class_node.body)
    assert "Members" in parser3.doc[name]
    
    # Test with enums
    script = """
class TestEnum(enum.Enum):
    VALUE1 = 1
    VALUE2 = 2
"""
    parser4 = Parser()
    parser4.level[root] = 0
    parser4.root[name] = root
    parser4.alias[_m(root, "enum")] = "enum"
    tree = parse(script)
    class_node = tree.body[0]
    parser4.class_api(root, name, class_node.bases, class_node.body)
    assert "Enums" in parser4.doc[name]
    
    # Test with deleted members
    script = """
class TestClass:
    attr1: int
    del attr1
"""
    parser5 = Parser()
    parser5.level[root] = 0
    parser5.root[name] = root
    tree = parse(script)
    class_node = tree.body[0]
    parser5.class_api(root, name, class_node.bases, class_node.body)
    assert "Members" not in parser5.doc[name]
    
    # Test with private members (should be excluded)
    script = """
class TestClass:
    _private: int
    public: str
"""
    parser6 = Parser()
    parser6.level[root] = 0
    parser6.root[name] = root
    tree = parse(script)
    class_node = tree.body[0]
    parser6.class_api(root, name, class_node.bases, class_node.body)
    assert "Members" in parser6.doc[name]


# LLM-generated content at query #26
#--------------------------

```python
def test_const_type():
    """Test const_type function with various AST nodes."""
    # Test Constant nodes
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"
    assert const_type(Constant(value=None)) == "NoneType"
    assert const_type(Constant(value=1+2j)) == "complex"
    
    # Test Tuple
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == "tuple[int, int]"
    
    tuple_mixed = Tuple(elts=[Constant(value=1), Constant(value="str")], ctx=Load())
    assert const_type(tuple_mixed) == "tuple[Any, Any]"
    
    tuple_empty = Tuple(elts=[], ctx=Load())
    assert const_type(tuple_empty) == "tuple"
    
    # Test List
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == "list[int, int]"
    
    list_mixed = List(elts=[Constant(value=1), Constant(value=3.14)], ctx=Load())
    assert const_type(list_mixed) == "list[Any, Any]"
    
    list_empty = List(elts=[], ctx=Load())
    assert const_type(list_empty) == "list"
    
    # Test Set
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == "set[int, int]"
    
    set_mixed = Set(elts=[Constant(value=1), Constant(value="str")])
    assert const_type(set_mixed) == "set[Any, Any]"
    
    set_empty = Set(elts=[])
    assert const_type(set_empty) == "set"
    
    # Test Dict
    dict_node = Dict(keys=[Constant(value="a")], values=[Constant(value=1)])
    assert const_type(dict_node) == "dict[str, int]"
    
    dict_mixed_keys = Dict(keys=[Constant(value=1), Constant(value="a")], 
                           values=[Constant(value=1), Constant(value=2)])
    assert const_type(dict_mixed_keys) == "dict[Any, int]"
    
    dict_mixed_values = Dict(keys=[Constant(value="a"), Constant(value="b")], 
                             values=[Constant(value=1), Constant(value="str")])
    assert const_type(dict_mixed_values) == "dict[str, Any]"
    
    dict_empty = Dict(keys=[], values=[])
    assert const_type(dict_empty) == "dict"
    
    # Test Call with builtin types
    call_int = Call(func=Name(id='int', ctx=Load()), args=[], keywords=[])
    assert const_type(call_int) == "int"
    
    call_str = Call(func=Name(id='str', ctx=Load()), args=[], keywords=[])
    assert const_type(call_str) == "str"
    
    call_bool = Call(func=Name(id='bool', ctx=Load()), args=[], keywords=[])
    assert const_type(call_bool) == "bool"
    
    call_float = Call(func=Name(id='float', ctx=Load()), args=[], keywords=[])
    assert const_type(call_float) == "float"
    
    call_complex = Call(func=Name(id='complex', ctx=Load()), args=[], keywords=[])
    assert const_type(call_complex) == "complex"
    
    # Test Call with unknown function
    call_unknown = Call(func=Name(id='unknown_func', ctx=Load()), args=[], keywords=[])
    assert const_type(call_unknown) == ANY
    
    # Test BinOp (should return ANY)
    binop = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(binop) == ANY
    
    # Test Name node (should return ANY)
    name_node = Name(id='x', ctx=Load())
    assert const_type(name_node) == ANY
    
    # Test Tuple with None elements (should return empty string behavior)
    tuple_with_none = Tuple(elts=[None, Constant(value=1)], ctx=Load())
    assert const_type(tuple_with_none) == "tuple"
    
    # Test List with None elements
    list_with_none = List(elts=[Constant(value=1), None], ctx=Load())
    assert const_type(list_with_none) == "list"


# LLM-generated content at query #27
#--------------------------

```python
def test_Parser_imports():
    """Test Parser.imports method."""
    parser = Parser()
    
    # Test Import node
    root = "test_module"
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"
    
    # Test Import with alias
    import_node_alias = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_alias)
    assert parser.alias["test_module.np"] == "numpy"
    
    # Test ImportFrom with module
    root2 = "test_module.submodule"
    parser.root[root2] = root2
    
    import_from_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname=None)],
        level=0
    )
    parser.imports(root2, import_from_node)
    assert parser.alias["test_module.submodule.path"] == "os.path"
    
    # Test ImportFrom with alias
    import_from_alias = ImportFrom(
        module="collections",
        names=[alias(name="defaultdict", asname="dd")],
        level=0
    )
    parser.imports(root2, import_from_alias)
    assert parser.alias["test_module.submodule.dd"] == "collections.defaultdict"
    
    # Test ImportFrom with relative import (level=1)
    import_from_relative = ImportFrom(
        module="sibling",
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports(root2, import_from_relative)
    assert parser.alias["test_module.submodule.func"] == "test_module.sibling.func"
    
    # Test ImportFrom with relative import (level=2)
    root3 = "package.subpackage.module"
    parser.root[root3] = root3
    
    import_from_level2 = ImportFrom(
        module="other",
        names=[alias(name="item", asname=None)],
        level=2
    )
    parser.imports(root3, import_from_level2)
    assert parser.alias["package.subpackage.module.item"] == "package.other.item"
    
    # Test ImportFrom with multiple names
    import_from_multi = ImportFrom(
        module="typing",
        names=[
            alias(name="List", asname=None),
            alias(name="Dict", asname=None),
            alias(name="Optional", asname="Opt")
        ],
        level=0
    )
    parser.imports(root, import_from_multi)
    assert parser.alias["test_module.List"] == "typing.List"
    assert parser.alias["test_module.Dict"] == "typing.Dict"
    assert parser.alias["test_module.Opt"] == "typing.Optional"


# LLM-generated content at query #28
#--------------------------

```python
def test_Resolver_visit_Name():
    """Test Resolver.visit_Name method."""
    # Test case 1: Replace self_ty with "Self"
    resolver = Resolver("mymodule", {}, self_ty="MyType")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 2: Non-matching self_ty, simple name without alias
    resolver = Resolver("mymodule", {}, self_ty="")
    node = Name("SomeName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeName"

    # Test case 3: Name with alias that resolves to another name
    alias = {"mymodule.OldName": "NewName"}
    resolver = Resolver("mymodule", alias)
    node = Name("OldName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "NewName"

    # Test case 4: Name with alias that is a TypeVar (should return original)
    alias = {"mymodule.TypeVar": "typing.TypeVar", "mymodule.T": "TypeVar('T')"}
    resolver = Resolver("mymodule", alias)
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Name with circular alias (should return original)
    alias = {"mymodule.Self": "Self"}
    resolver = Resolver("mymodule", alias)
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 6: Name with complex alias expression
    alias = {"mymodule.MyList": "list[int]"}
    resolver = Resolver("mymodule", alias)
    node = Name("MyList", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)

    # Test case 7: Name not in alias
    alias = {"mymodule.Other": "str"}
    resolver = Resolver("mymodule", alias)
    node = Name("UnknownName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "UnknownName"

    # Test case 8: Multiple roots with nested module names
    alias = {"parent.child.Name": "int"}
    resolver = Resolver("parent.child", alias)
    node = Name("Name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.value == 1  # int constant


# LLM-generated content at query #29
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    # Test case 1: Remove typing.* prefix
    resolver = Resolver("test_module", {})
    typing_attr = Attribute(
        value=Name(id="typing", ctx=Load()),
        attr="List",
        ctx=Load()
    )
    result = resolver.visit_Attribute(typing_attr)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Remove typing.Optional prefix
    typing_optional = Attribute(
        value=Name(id="typing", ctx=Load()),
        attr="Optional",
        ctx=Load()
    )
    result = resolver.visit_Attribute(typing_optional)
    assert isinstance(result, Name)
    assert result.id == "Optional"

    # Test case 3: Remove typing.Dict prefix
    typing_dict = Attribute(
        value=Name(id="typing", ctx=Load()),
        attr="Dict",
        ctx=Load()
    )
    result = resolver.visit_Attribute(typing_dict)
    assert isinstance(result, Name)
    assert result.id == "Dict"

    # Test case 4: Non-typing attribute - should return unchanged
    other_attr = Attribute(
        value=Name(id="other_module", ctx=Load()),
        attr="SomeClass",
        ctx=Load()
    )
    result = resolver.visit_Attribute(other_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert isinstance(result.value, Name)
    assert result.value.id == "other_module"

    # Test case 5: Non-Name value - should return unchanged
    non_name_attr = Attribute(
        value=Attribute(
            value=Name(id="typing", ctx=Load()),
            attr="nested",
            ctx=Load()
        ),
        attr="Inner",
        ctx=Load()
    )
    result = resolver.visit_Attribute(non_name_attr)
    assert isinstance(result, Attribute)
    assert result.attr == "Inner"

    # Test case 6: typing.Union prefix
    typing_union = Attribute(
        value=Name(id="typing", ctx=Load()),
        attr="Union",
        ctx=Load()
    )
    result = resolver.visit_Attribute(typing_union)
    assert isinstance(result, Name)
    assert result.id == "Union"

    # Test case 7: typing.Callable prefix
    typing_callable = Attribute(
        value=Name(id="typing", ctx=Load()),
        attr="Callable",
        ctx=Load()
    )
    result = resolver.visit_Attribute(typing_callable)
    assert isinstance(result, Name)
    assert result.id == "Callable"


# LLM-generated content at query #30
#--------------------------

```python
def test_Parser_parse():
    """Unit test for Parser.parse method."""
    parser = Parser()
    root = "test_module"
    script = """
'''Module docstring.'''
import os
from typing import List

__all__ = ['public_func', 'PUBLIC_CONST']

PUBLIC_CONST: int = 42

def public_func(x: int) -> str:
    '''Function docstring.'''
    return str(x)

def _private_func():
    pass

class PublicClass:
    '''Class docstring.'''
    attr: int = 10
    
    def method(self) -> None:
        '''Method docstring.'''
        pass
"""
    
    parser.parse(root, script)
    
    assert root in parser.doc
    assert root in parser.level
    assert root in parser.imp
    assert root in parser.root
    assert root in parser.docstring
    assert parser.root[root] == root
    assert parser.level[root] == 0
    assert "Module docstring." in parser.docstring[root]
    assert "test_module.public_func" in parser.doc
    assert "test_module.PUBLIC_CONST" in parser.const
    assert "test_module.PublicClass" in parser.doc
    assert "test_module.PublicClass.method" in parser.doc
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.List"] == "typing.List"
    assert "test_module.public_func" in parser.imp[root]
    assert "test_module.PUBLIC_CONST" in parser.imp[root]


def test_Parser_parse_nested_modules():
    """Test parse with nested module structure."""
    parser = Parser()
    root = "pkg.submodule"
    script = """
def func(): pass
"""
    
    parser.parse(root, script)
    
    assert root in parser.doc
    assert parser.level[root] == 1
    assert parser.root[root] == root


def test_Parser_parse_with_type_comments():
    """Test parse with type comments."""
    parser = Parser()
    root = "test_module"
    script = """
x = 42  # type: int
y = "hello"  # type: str
"""
    
    parser.parse(root, script)
    
    assert "test_module.x" in parser.const
    assert "test_module.y" in parser.const


def test_Parser_parse_empty_script():
    """Test parse with empty script."""
    parser = Parser()
    root = "empty_module"
    script = ""
    
    parser.parse(root, script)
    
    assert root in parser.doc
    assert root in parser.level
    assert root in parser.imp


def test_Parser_parse_async_function():
    """Test parse with async function."""
    parser = Parser()
    root = "async_module"
    script = """
async def async_func():
    '''Async function.'''
    pass
"""
    
    parser.parse(root, script)
    
    assert "async_module.async_func" in parser.doc


def test_Parser_parse_with_decorators():
    """Test parse with decorated functions."""
    parser = Parser()
    root = "decorated_module"
    script = """
from functools import wraps

@wraps
def decorated_func():
    '''Decorated function.'''
    pass
"""
    
    parser.parse(root, script)
    
    assert "decorated_module.decorated_func" in parser.doc


def test_Parser_parse_nested_class():
    """Test parse with nested class definitions."""
    parser = Parser()
    root = "nested_module"
    script = """
class OuterClass:
    '''Outer class.'''
    
    class InnerClass:
        '''Inner class.'''
        pass
"""
    
    parser.parse(root, script)
    
    assert "nested_module.OuterClass" in parser.doc
    assert "nested_module.OuterClass.InnerClass" in parser.doc


def test_Parser_parse_annotated_assignment():
    """Test parse with annotated assignments."""
    parser = Parser()
    root = "annotated_module"
    script = """
from typing import Optional

VAR: Optional[int] = None
CONST: str = "value"
"""
    
    parser.parse(root, script)
    
    assert "annotated_module.VAR" in parser.const
    assert "annotated_module.CONST" in parser.const


# LLM-generated content at query #31
#--------------------------

def test_const_type():
    """Test const_type function with various AST node types."""
    # Test Constant nodes
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"
    assert const_type(Constant(value=None)) == "NoneType"
    assert const_type(Constant(value=1+2j)) == "complex"
    
    # Test Tuple nodes
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == "tuple[int, int]"
    
    tuple_mixed = Tuple(elts=[Constant(value=1), Constant(value="str")], ctx=Load())
    assert const_type(tuple_mixed) == "tuple[Any, Any]"
    
    # Test List nodes
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == "list[int]"
    
    list_mixed = List(elts=[Constant(value=1), Constant(value="str")], ctx=Load())
    assert const_type(list_mixed) == "list[Any]"
    
    # Test Set nodes
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == "set[int]"
    
    # Test empty collections
    empty_tuple = Tuple(elts=[], ctx=Load())
    assert const_type(empty_tuple) == "tuple"
    
    empty_list = List(elts=[], ctx=Load())
    assert const_type(empty_list) == "list"
    
    empty_set = Set(elts=[])
    assert const_type(empty_set) == "set"
    
    # Test Dict nodes
    dict_node = Dict(keys=[Constant(value="key")], values=[Constant(value=1)])
    assert const_type(dict_node) == "dict[str, int]"
    
    dict_mixed = Dict(keys=[Constant(value="key")], values=[Constant(value="val")])
    assert const_type(dict_mixed) == "dict[str, str]"
    
    # Test empty Dict
    empty_dict = Dict(keys=[], values=[])
    assert const_type(empty_dict) == "dict"
    
    # Test Call nodes with builtin types
    int_call = Call(func=Name(id='int', ctx=Load()), args=[], keywords=[])
    assert const_type(int_call) == "int"
    
    str_call = Call(func=Name(id='str', ctx=Load()), args=[], keywords=[])
    assert const_type(str_call) == "str"
    
    bool_call = Call(func=Name(id='bool', ctx=Load()), args=[], keywords=[])
    assert const_type(bool_call) == "bool"
    
    float_call = Call(func=Name(id='float', ctx=Load()), args=[], keywords=[])
    assert const_type(float_call) == "float"
    
    complex_call = Call(func=Name(id='complex', ctx=Load()), args=[], keywords=[])
    assert const_type(complex_call) == "complex"
    
    # Test Call nodes with non-builtin functions
    custom_call = Call(func=Name(id='custom_func', ctx=Load()), args=[], keywords=[])
    assert const_type(custom_call) == ANY
    
    # Test BinOp (should return ANY)
    binop = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(binop) == ANY
    
    # Test Name node (should return ANY)
    name_node = Name(id='x', ctx=Load())
    assert const_type(name_node) == ANY
    
    # Test Tuple with None elements
    tuple_with_none = Tuple(elts=[Constant(value=1), None], ctx=Load())
    assert const_type(tuple_with_none) == ""
    
    # Test List with None elements
    list_with_none = List(elts=[Constant(value=1), None], ctx=Load())
    assert const_type(list_with_none) == ""


# LLM-generated content at query #32
#--------------------------

```python
def test_Resolver_visit_Subscript():
    """Test Resolver.visit_Subscript method."""
    from ast import parse, unparse
    
    # Test case 1: Union type conversion to BitOr
    resolver = Resolver("module", {
        "module.Union": "typing.Union"
    })
    union_node = parse("Union[int, str]").body[0].value
    result = resolver.visit(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test case 2: Optional type conversion
    resolver = Resolver("module", {
        "module.Optional": "typing.Optional"
    })
    optional_node = parse("Optional[int]").body[0].value
    result = resolver.visit(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test case 3: PEP585 deprecated type warning
    resolver = Resolver("module", {
        "module.Dict": "typing.Dict"
    })
    dict_node = parse("Dict[str, int]").body[0].value
    result = resolver.visit(dict_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "dict"
    
    # Test case 4: Non-Name subscript value - should return unchanged
    resolver = Resolver("module", {})
    attr_subscript = parse("obj.Dict[int]").body[0].value
    result = resolver.visit(attr_subscript)
    assert result == attr_subscript
    
    # Test case 5: Union with non-Tuple slice - should return slice
    resolver = Resolver("module", {
        "module.Union": "typing.Union"
    })
    union_single = parse("Union[int]").body[0].value
    result = resolver.visit(union_single)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test case 6: Unknown subscript - should return unchanged
    resolver = Resolver("module", {})
    list_node = parse("List[int]").body[0].value
    result = resolver.visit(list_node)
    assert result == list_node
    
    # Test case 7: Multiple Union types
    resolver = Resolver("module", {
        "module.Union": "typing.Union"
    })
    multi_union = parse("Union[int, str, bool]").body[0].value
    result = resolver.visit(multi_union)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test case 8: Nested Optional
    resolver = Resolver("module", {
        "module.Optional": "typing.Optional"
    })
    nested_optional = parse("Optional[Optional[int]]").body[0].value
    result = resolver.visit(nested_optional)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)


# LLM-generated content at query #33
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    non_string_const = Constant(value=42)
    result = resolver.visit_Constant(non_string_const)
    assert result is non_string_const
    assert result.value == 42
    
    # Test with float constant
    float_const = Constant(value=3.14)
    result = resolver.visit_Constant(float_const)
    assert result is float_const
    assert result.value == 3.14
    
    # Test with None constant
    none_const = Constant(value=None)
    result = resolver.visit_Constant(none_const)
    assert result is none_const
    assert result.value is None
    
    # Test with string constant that is a valid name
    string_const = Constant(value="int")
    result = resolver.visit_Constant(string_const)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with string constant that is a valid expression
    string_const = Constant(value="List[str]")
    result = resolver.visit_Constant(string_const)
    assert isinstance(result, Subscript)
    
    # Test with string constant that is invalid syntax
    string_const = Constant(value="invalid syntax ][")
    result = resolver.visit_Constant(string_const)
    assert result is string_const
    assert result.value == "invalid syntax ]["
    
    # Test with resolver having alias
    resolver_with_alias = Resolver("test_module", {"test_module.MyType": "int"})
    string_const = Constant(value="MyType")
    result = resolver_with_alias.visit_Constant(string_const)
    assert isinstance(result, Name)
    assert result.id == "MyType"
    
    # Test with string constant containing complex expression
    string_const = Constant(value="Dict[str, int]")
    result = resolver.visit_Constant(string_const)
    assert isinstance(result, Subscript)
    
    # Test with empty string
    empty_const = Constant(value="")
    result = resolver.visit_Constant(empty_const)
    assert result is empty_const
    assert result.value == ""
    
    # Test with boolean constant
    bool_const = Constant(value=True)
    result = resolver.visit_Constant(bool_const)
    assert result is bool_const
    assert result.value is True


# LLM-generated content at query #34
#--------------------------

```python
def test_Parser_class_api():
    """Test Parser.class_api method."""
    from ast import parse, AnnAssign, Assign, Delete, Name, Constant
    
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    
    # Setup parser state
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    
    # Test 1: Class with bases
    script = """
class MyClass(BaseClass, AnotherBase):
    pass
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    
    assert name in parser.doc
    assert "BaseClass" in parser.doc[name]
    assert "AnotherBase" in parser.doc[name]
    
    # Test 2: Class with enum
    parser.doc.clear()
    script = """
class MyEnum(enum.Enum):
    OPTION_A = 1
    OPTION_B = 2
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.alias[f"{root}.enum"] = "enum"
    parser.class_api(root, name, class_node.bases, class_node.body)
    
    assert name in parser.doc
    assert "OPTION_A" in parser.doc[name]
    assert "OPTION_B" in parser.doc[name]
    assert "Enums" in parser.doc[name]
    
    # Test 3: Class with members
    parser.doc.clear()
    parser.alias.clear()
    script = """
class MyClass:
    public_attr: int = 5
    _private_attr: str = "test"
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    
    assert name in parser.doc
    assert "public_attr" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]
    assert "Members" in parser.doc[name]
    
    # Test 4: Class with deleted members
    parser.doc.clear()
    script = """
class MyClass:
    attr1: int
    attr2: str
    del attr1
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    
    assert name in parser.doc
    assert "attr2" in parser.doc[name]
    assert "attr1" not in parser.doc[name]
    
    # Test 5: Empty class with no bases
    parser.doc.clear()
    script = """
class EmptyClass:
    pass
"""
    tree = parse(script)
    class_node = tree.body[0]
    parser.class_api(root, name, class_node.bases, class_node.body)
    
    assert name in parser.doc
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]


# LLM-generated content at query #35
#--------------------------

```python
def test_Parser_class_api():
    """Test Parser.class_api method."""
    from ast import parse, AnnAssign, Assign, Name, Constant, Delete
    
    # Test case 1: Class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases_code = "class Base: pass"
    bases = parse(bases_code).body[0].bases
    body = []
    
    parser.doc[name] = "## class TestClass\n\n"
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    
    # Test case 2: Class with annotated members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    
    code_str = """
class TestClass:
    public_attr: int
    _private_attr: str
    """
    tree = parse(code_str)
    class_node = tree.body[0]
    body = class_node.body
    
    parser.doc[name] = "## class TestClass\n\n"
    parser.alias = {}
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "Members" in parser.doc[name] or "int" in parser.doc[name]
    
    # Test case 3: Enum class
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases_code = "class TestEnum(enum.Enum): pass"
    bases = parse(bases_code).body[0].bases
    
    code_str = """
class TestEnum(enum.Enum):
    MEMBER1 = 1
    MEMBER2 = 2
    """
    tree = parse(code_str)
    class_node = tree.body[0]
    body = class_node.body
    
    parser.doc[name] = "## class TestEnum\n\n"
    parser.alias = {"test_module.enum": "enum"}
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "Enums" in parser.doc[name] or "MEMBER" in parser.doc[name]
    
    # Test case 4: Class with deleted members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    
    code_str = """
class TestClass:
    attr1: int
    attr2: str
    del attr1
    """
    tree = parse(code_str)
    class_node = tree.body[0]
    body = class_node.body
    
    parser.doc[name] = "## class TestClass\n\n"
    parser.alias = {}
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    
    # Test case 5: Class with type comments
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    
    code_str = """
class TestClass:
    attr1 = 42  # type: int
    attr2 = "hello"  # type: str
    """
    tree = parse(code_str)
    class_node = tree.body[0]
    body = class_node.body
    
    parser.doc[name] = "## class TestClass\n\n"
    parser.alias = {}
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    
    # Test case 6: Empty class
    parser = Parser()
    root = "test_module"
    name = "test_module.EmptyClass"
    bases = []
    body = []
    
    parser.doc[name] = "## class EmptyClass\n\n"
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert parser.doc[name].count('\n') >= 2


# LLM-generated content at query #36
#--------------------------

```python
def test_Parser_func_ann():
    """Test Parser.func_ann method."""
    from ast import arg, parse
    
    # Setup parser
    p = Parser()
    root = "test_module"
    p.root[root] = root
    p.alias[root] = root
    
    # Test case 1: Simple function with no self
    args = [
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='y', annotation=parse('str').body[0].value),
        arg(arg='return', annotation=parse('bool').body[0].value),
    ]
    result = list(p.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ['int', 'str', 'bool']
    
    # Test case 2: Function with self parameter
    args = [
        arg(arg='self', annotation=None),
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='return', annotation=parse('None').body[0].value),
    ]
    result = list(p.func_ann(root, args, has_self=True, cls_method=False))
    assert result[0] == 'Self'
    assert result[1] == 'int'
    assert result[2] == 'None'
    
    # Test case 3: Class method with cls parameter
    args = [
        arg(arg='cls', annotation=parse('type[TestClass]').body[0].value),
        arg(arg='x', annotation=parse('str').body[0].value),
        arg(arg='return', annotation=parse('TestClass').body[0].value),
    ]
    result = list(p.func_ann(root, args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'
    assert result[1] == 'str'
    
    # Test case 4: Function with no annotation
    args = [
        arg(arg='x', annotation=None),
        arg(arg='y', annotation=None),
        arg(arg='return', annotation=None),
    ]
    result = list(p.func_ann(root, args, has_self=False, cls_method=False))
    assert result == [ANY, ANY, ANY]
    
    # Test case 5: Function with *args separator
    args = [
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='*', annotation=None),
        arg(arg='y', annotation=parse('str').body[0].value),
        arg(arg='return', annotation=None),
    ]
    result = list(p.func_ann(root, args, has_self=False, cls_method=False))
    assert result[0] == 'int'
    assert result[1] == ''
    assert result[2] == 'str'
    assert result[3] == ANY
    
    # Test case 6: Self type annotation with custom self_ty
    args = [
        arg(arg='self', annotation=parse('MyClass').body[0].value),
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='return', annotation=None),
    ]
    result = list(p.func_ann(root, args, has_self=True, cls_method=False))
    assert result[0] == 'Self'
    assert result[1] == 'int'


# LLM-generated content at query #37
#--------------------------

```python
def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    non_string_node = Constant(value=42)
    result = resolver.visit_Constant(non_string_node)
    assert result is non_string_node
    
    # Test with non-string constant (float)
    float_node = Constant(value=3.14)
    result = resolver.visit_Constant(float_node)
    assert result is float_node
    
    # Test with non-string constant (None)
    none_node = Constant(value=None)
    result = resolver.visit_Constant(none_node)
    assert result is none_node
    
    # Test with valid string that parses to a Name
    valid_name_node = Constant(value="int")
    result = resolver.visit_Constant(valid_name_node)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test with valid string that parses to a subscript
    subscript_node = Constant(value="list[int]")
    result = resolver.visit_Constant(subscript_node)
    assert isinstance(result, Subscript)
    
    # Test with invalid syntax string
    invalid_syntax_node = Constant(value="@#$%^")
    result = resolver.visit_Constant(invalid_syntax_node)
    assert result is invalid_syntax_node
    
    # Test with complex valid expression
    expr_node = Constant(value="dict[str, int]")
    result = resolver.visit_Constant(expr_node)
    assert isinstance(result, Subscript)
    
    # Test with string containing spaces
    spaced_node = Constant(value="str")
    result = resolver.visit_Constant(spaced_node)
    assert isinstance(result, Name)
    
    # Test with self_ty substitution
    resolver_with_self = Resolver("test_module", {}, self_ty="T")
    self_node = Constant(value="T")
    result = resolver_with_self.visit_Constant(self_node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    
    # Test with alias substitution
    alias_resolver = Resolver("test_module", {"test_module.MyType": "int"})
    alias_node = Constant(value="MyType")
    result = alias_resolver.visit_Constant(alias_node)
    assert isinstance(result, Name)


# LLM-generated content at query #38
#--------------------------

```python
def test_Parser_compile():
    """Test Parser.compile method."""
    # Test basic compilation with empty parser
    p = Parser()
    result = p.compile()
    assert isinstance(result, str)
    assert result.endswith('\n')
    
    # Test with toc enabled
    p_toc = Parser(toc=True)
    result_toc = p_toc.compile()
    assert isinstance(result_toc, str)
    assert '**Table of contents:**' in result_toc
    
    # Test with populated doc and docstring
    p_populated = Parser(link=True, b_level=1, toc=False)
    p_populated.doc['test_module'] = '# Module `test_module`\n\n'
    p_populated.docstring['test_module'] = 'Test documentation'
    p_populated.root['test_module'] = 'test_module'
    p_populated.level['test_module'] = 0
    p_populated.imp['test_module'] = set()
    
    result_populated = p_populated.compile()
    assert isinstance(result_populated, str)
    assert 'test_module' in result_populated
    assert 'Test documentation' in result_populated
    assert result_populated.endswith('\n')
    
    # Test with multiple entries
    p_multi = Parser(link=True, b_level=1, toc=True)
    p_multi.doc['pkg'] = '# Module `pkg`\n\n'
    p_multi.doc['pkg.module'] = '## module()\n\n'
    p_multi.docstring['pkg'] = 'Package docs'
    p_multi.docstring['pkg.module'] = 'Module docs'
    p_multi.root['pkg'] = 'pkg'
    p_multi.root['pkg.module'] = 'pkg'
    p_multi.level['pkg'] = 0
    p_multi.level['pkg.module'] = 0
    p_multi.imp['pkg'] = set()
    
    result_multi = p_multi.compile()
    assert isinstance(result_multi, str)
    assert '**Table of contents:**' in result_multi
    assert 'pkg' in result_multi
    assert 'module' in result_multi
    
    # Test with constants
    p_const = Parser(link=False, b_level=1, toc=False)
    p_const.doc['test'] = '# Module `test`\n\n'
    p_const.docstring['test'] = 'Test'
    p_const.const['test.CONST'] = 'int'
    p_const.root['test'] = 'test'
    p_const.root['test.CONST'] = 'test'
    p_const.level['test'] = 0
    p_const.imp['test'] = set()
    
    result_const = p_const.compile()
    assert isinstance(result_const, str)
    assert 'test' in result_const
    
    # Test with missing documentation warning (magic methods)
    p_magic = Parser(toc=False)
    p_magic.doc['test.__init__'] = '## __init__()\n\n'
    p_magic.root['test.__init__'] = 'test'
    p_magic.level['test.__init__'] = 0
    p_magic.imp['test'] = set()
    
    result_magic = p_magic.compile()
    assert isinstance(result_magic, str)
    # Magic methods without docstring should be excluded
    assert '__init__' not in result_magic or 'test.__init__' not in result_magic
    
    # Test sorting by level and name
    p_sort = Parser(toc=True)
    p_sort.doc['a'] = '# a\n\n'
    p_sort.doc['b.c'] = '# b.c\n\n'
    p_sort.doc['a.b'] = '# a.b\n\n'
    p_sort.docstring['a'] = 'A'
    p_sort.docstring['b.c'] = 'BC'
    p_sort.docstring['a.b'] = 'AB'
    p_sort.root['a'] = 'a'
    p_sort.root['a.b'] = 'a'
    p_sort.root['b.c'] = 'b'
    p_sort.level['a'] = 0
    p_sort.level['a.b'] = 1
    p_sort.level['b.c'] = 1
    p_sort.imp['a'] = set()
    p_sort.imp['b'] = set()
    
    result_sort = p_sort.compile()
    assert isinstance(result_sort, str)
    assert result_sort.endswith('\n')


# LLM-generated content at query #39
#--------------------------

```python
def test_Resolver_visit_Attribute():
    """Test Resolver.visit_Attribute method."""
    resolver = Resolver("mymodule", {})
    
    # Test case 1: Remove 'typing.*' prefix
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'
    
    # Test case 2: Remove 'typing.Dict' prefix
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Dict', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Dict'
    
    # Test case 3: Keep non-typing attributes
    node = Attribute(value=Name(id='mymodule', ctx=Load()), attr='MyClass', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'MyClass'
    assert isinstance(result.value, Name)
    assert result.value.id == 'mymodule'
    
    # Test case 4: Keep nested attributes (not simple Name)
    inner_attr = Attribute(value=Name(id='typing', ctx=Load()), attr='io', ctx=Load())
    node = Attribute(value=inner_attr, attr='StringIO', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'StringIO'
    
    # Test case 5: Multiple typing attributes
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='Optional', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Optional'
    
    # Test case 6: Attribute with different module name
    node = Attribute(value=Name(id='collections', ctx=Load()), attr='abc', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'abc'
    assert result.value.id == 'collections'


# LLM-generated content at query #40
#--------------------------

```python
def test_Parser_func_api():
    """Test Parser.func_api method."""
    from ast import parse, arg, arguments
    
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    
    # Create a simple function with arguments
    script = """
def test_func(x: int, y: str = "default", *args, z: float = 1.0, **kwargs) -> bool:
    pass
"""
    parser.parse(root, script)
    
    # Test with basic function arguments
    func_node = parse(script).body[0]
    parser.func_api(root, name, func_node.args, func_node.returns, 
                    has_self=False, cls_method=False)
    
    assert name in parser.doc
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]
    assert "bool" in parser.doc[name]


def test_Parser_func_api_with_self():
    """Test Parser.func_api method with self parameter."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass.test_method"
    
    script = """
class TestClass:
    def test_method(self, x: int) -> str:
        pass
"""
    parser.parse(root, script)
    
    func_node = parse(script).body[0].body[0]
    parser.func_api(root, name, func_node.args, func_node.returns,
                    has_self=True, cls_method=False)
    
    assert name in parser.doc
    assert "Self" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]


def test_Parser_func_api_with_classmethod():
    """Test Parser.func_api method with classmethod."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass.test_classmethod"
    
    script = """
class TestClass:
    @classmethod
    def test_classmethod(cls, x: int) -> str:
        pass
"""
    parser.parse(root, script)
    
    func_node = parse(script).body[0].body[0]
    parser.func_api(root, name, func_node.args, func_node.returns,
                    has_self=True, cls_method=True)
    
    assert name in parser.doc
    assert "type[Self]" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]


def test_Parser_func_api_with_kwonly():
    """Test Parser.func_api method with keyword-only arguments."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    name = "test_module.func_kwonly"
    
    script = """
def func_kwonly(x: int, *, y: str, z: float = 1.0) -> None:
    pass
"""
    parser.parse(root, script)
    
    func_node = parse(script).body[0]
    parser.func_api(root, name, func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert name in parser.doc
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]
    assert "float" in parser.doc[name]


def test_Parser_func_api_with_posonly():
    """Test Parser.func_api method with positional-only arguments."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    name = "test_module.func_posonly"
    
    script = """
def func_posonly(x: int, /, y: str) -> None:
    pass
"""
    parser.parse(root, script)
    
    func_node = parse(script).body[0]
    parser.func_api(root, name, func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert name in parser.doc
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]


def test_Parser_func_api_no_annotations():
    """Test Parser.func_api method with unannotated arguments."""
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    name = "test_module.func_no_ann"
    
    script = """
def func_no_ann(x, y):
    pass
"""
    parser.parse(root, script)
    
    func_node = parse(script).body[0]
    parser.func_api(root, name, func_node.args, func_node.returns,
                    has_self=False, cls_method=False)
    
    assert name in parser.doc
    # Should contain ANY for unannotated arguments
    assert "Any" in parser.doc[name]


