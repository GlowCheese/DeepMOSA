####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_api():
    # Test case 1: Function definition
    parser = Parser()
    root = "test_module"
    node = FunctionDef(
        name="test_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert "test_func()" in parser.doc["test_module.test_func"]

    # Test case 2: Async function definition
    parser = Parser()
    node = AsyncFunctionDef(
        name="test_async_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.test_async_func" in parser.doc
    assert "async test_async_func()" in parser.doc["test_module.test_async_func"]

    # Test case 3: Class definition
    parser = Parser()
    node = ClassDef(
        name="TestClass",
        bases=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]

    # Test case 4: Function with decorator
    parser = Parser()
    node = FunctionDef(
        name="decorated_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="decorator", ctx=Load())]
    )
    parser.api(root, node)
    assert "decorated_func()" in parser.doc["test_module.decorated_func"]
    assert "@decorator" in parser.doc["test_module.decorated_func"]

    # Test case 5: Class with bases
    parser = Parser()
    node = ClassDef(
        name="ChildClass",
        bases=[Name(id="ParentClass", ctx=Load())],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.ChildClass" in parser.doc
    assert "class ChildClass" in parser.doc["test_module.ChildClass"]
    assert "ParentClass" in parser.doc["test_module.ChildClass"]

    # Test case 6: Nested class
    parser = Parser()
    outer_class = ClassDef(
        name="OuterClass",
        bases=[],
        body=[
            ClassDef(
                name="InnerClass",
                bases=[],
                body=[],
                decorator_list=[]
            )
        ],
        decorator_list=[]
    )
    parser.api(root, outer_class)
    assert "test_module.OuterClass" in parser.doc
    assert "test_module.OuterClass.InnerClass" in parser.doc
    assert "class InnerClass" in parser.doc["test_module.OuterClass.InnerClass"]


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_is_public():
    # Test case 1: Public name in __all__
    parser = Parser()
    parser.imp = {'module': {'public_name'}}
    parser.root = {'module.public_name': 'module'}
    parser.doc = {'module.public_name': ''}
    assert parser.is_public('module.public_name') == True

    # Test case 2: Non-public name not in __all__
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._private_name': 'module'}
    assert parser.is_public('module._private_name') == False

    # Test case 3: Public name with no __all__ (should be public by convention)
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module.public_name': 'module'}
    parser.doc = {'module.public_name': ''}
    assert parser.is_public('module.public_name') == True

    # Test case 4: Name listed in parent's __all__
    parser = Parser()
    parser.imp = {'module': {'submodule'}}
    parser.root = {'module.submodule.item': 'module'}
    parser.doc = {'module.submodule.item': ''}
    assert parser.is_public('module.submodule.item') == True

    # Test case 5: Name not listed in __all__ but has public children
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module.submodule': 'module', 'module.submodule.public_item': 'module'}
    parser.doc = {'module.submodule.public_item': ''}
    assert parser.is_public('module.submodule') == True

    # Test case 6: Private name with public children (should still be private)
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._private': 'module', 'module._private.public_item': 'module'}
    parser.doc = {'module._private.public_item': ''}
    assert parser.is_public('module._private') == False


# LLM-generated content at query #3
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Non-typing attribute
    resolver = Resolver("test", {})
    node = Attribute(Name("other", Load()), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"
    assert result.value.id == "other"

    # Test case 3: Non-Name value
    resolver = Resolver("test", {})
    node = Attribute(Constant(1), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"
    assert isinstance(result.value, Constant)


# LLM-generated content at query #4
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Constant(value=2))
    body = [stmt1, stmt2]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=3))],
        orelse=[Expr(value=Constant(value=4))]
    )
    body = [if_node]
    result = list(walk_body(body))
    assert len(result) == 2
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Expr)

    # Test Try statement
    try_node = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=5))],
        handlers=[],
        orelse=[Expr(value=Constant(value=6))],
        finalbody=[Expr(value=Constant(value=7))]
    )
    body = [try_node]
    result = list(walk_body(body))
    assert len(result) == 3
    assert all(isinstance(node, (Assign, Expr)) for node in result)

    # Test nested If and Try
    nested = If(
        test=Constant(value=True),
        body=[
            Try(
                body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=8))],
                handlers=[],
                orelse=[],
                finalbody=[]
            )
        ],
        orelse=[]
    )
    body = [nested]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test mixed statements
    mixed = [
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=9)),
        If(
            test=Constant(value=True),
            body=[Expr(value=Constant(value=10))],
            orelse=[]
        ),
        Try(
            body=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=11))],
            handlers=[],
            orelse=[],
            finalbody=[]
        )
    ]
    result = list(walk_body(mixed))
    assert len(result) == 3
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Expr)
    assert isinstance(result[2], Assign)


# LLM-generated content at query #5
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"

    # Test case 1: Simple function with no arguments and no return type
    node = FunctionDef(
        name="simple_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        returns=None,
        decorator_list=[],
    )
    parser.func_api(root, "test_module.simple_func", node.args, node.returns, has_self=False, cls_method=False)
    expected_doc = parser.doc["test_module.simple_func"]
    assert "return" in expected_doc
    assert "Any" in expected_doc

    # Test case 2: Function with positional arguments and return type
    node = FunctionDef(
        name="func_with_args",
        args=arguments(
            posonlyargs=[],
            args=[
                arg(arg="arg1", annotation=Name(id="int", ctx=Load())),
                arg(arg="arg2", annotation=Name(id="str", ctx=Load())),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        returns=Name(id="bool", ctx=Load()),
        decorator_list=[],
    )
    parser.func_api(root, "test_module.func_with_args", node.args, node.returns, has_self=False, cls_method=False)
    expected_doc = parser.doc["test_module.func_with_args"]
    assert "arg1" in expected_doc
    assert "arg2" in expected_doc
    assert "return" in expected_doc
    assert "int" in expected_doc
    assert "str" in expected_doc
    assert "bool" in expected_doc

    # Test case 3: Function with default arguments
    node = FunctionDef(
        name="func_with_defaults",
        args=arguments(
            posonlyargs=[],
            args=[
                arg(arg="arg1", annotation=Name(id="int", ctx=Load())),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[Constant(value=10)],
        ),
        returns=None,
        decorator_list=[],
    )
    parser.func_api(root, "test_module.func_with_defaults", node.args, node.returns, has_self=False, cls_method=False)
    expected_doc = parser.doc["test_module.func_with_defaults"]
    assert "arg1" in expected_doc
    assert "10" in expected_doc

    # Test case 4: Function with *args and **kwargs
    node = FunctionDef(
        name="func_with_varargs",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=arg(arg="args", annotation=None),
            kwarg=arg(arg="kwargs", annotation=None),
        ),
        returns=None,
        decorator_list=[],
    )
    parser.func_api(root, "test_module.func_with_varargs", node.args, node.returns, has_self=False, cls_method=False)
    expected_doc = parser.doc["test_module.func_with_varargs"]
    assert "*args" in expected_doc
    assert "**kwargs" in expected_doc

    # Test case 5: Function with self parameter (method)
    node = FunctionDef(
        name="method",
        args=arguments(
            posonlyargs=[],
            args=[
                arg(arg="self", annotation=None),
                arg(arg="arg1", annotation=Name(id="int", ctx=Load())),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        returns=None,
        decorator_list=[],
    )
    parser.func_api(root, "test_module.method", node.args, node.returns, has_self=True, cls_method=False)
    expected_doc = parser.doc["test_module.method"]
    assert "Self" in expected_doc
    assert "arg1" in expected_doc

    # Test case 6: Function with classmethod decorator
    node = FunctionDef(
        name="classmethod_func",
        args=arguments(
            posonlyargs=[],
            args=[
                arg(arg="cls", annotation=None),
                arg(arg="arg1", annotation=Name(id="int", ctx=Load())),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        returns=None,
        decorator_list=[Name(id="classmethod", ctx=Load())],
    )
    parser.func_api(root, "test_module.classmethod_func", node.args, node.returns, has_self=True, cls_method=True)
    expected_doc = parser.doc["test_module.classmethod_func"]
    assert "type[Self]" in expected_doc
    assert "arg1" in expected_doc


# LLM-generated content at query #6
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("attr1", Store()), Constant(1), None, None),
        AnnAssign(Name("attr2", Store()), Constant("hello"), None, None),
        Assign([Name("attr3", Store())], Constant(3.14)),
        Delete([Name("attr1", Del())])
    ]

    # Execute
    parser.class_api(root, name, bases, body)

    # Verify
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "attr3" in parser.doc[name]
    assert "attr1" not in parser.doc[name]


# LLM-generated content at query #7
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test Union type resolution
    resolver = Resolver("test", {"test.typing.Union": "typing.Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional type resolution
    resolver = Resolver("test", {"test.typing.Optional": "typing.Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test PEP585 type resolution
    resolver = Resolver("test", {"test.typing.List": "typing.List"})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

    # Test non-PEP585 type resolution
    resolver = Resolver("test", {"test.typing.Dict": "typing.Dict"})
    node = Subscript(Name("Dict", Load()), Tuple([Name("str", Load()), Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"
    assert isinstance(result.slice, Tuple)
    assert len(result.slice.elts) == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no self and no class method
    args = [arg("x", Name("int", Load())), arg("return", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "str"]

    # Test case 2: Function with self and not a class method
    args = [arg("self", Name("TestClass", Load())), arg("x", Name("int", Load())), arg("return", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "int", "str"]

    # Test case 3: Function with self and is a class method
    args = [arg("cls", Name("TestClass", Load())), arg("x", Name("int", Load())), arg("return", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int", "str"]

    # Test case 4: Function with *args
    args = [arg("x", Name("int", Load())), arg("*", None), arg("return", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "", "str"]

    # Test case 5: Function with **kwargs
    args = [arg("x", Name("int", Load())), arg("**kwargs", Name("dict", Load())), arg("return", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "dict", "str"]

    # Test case 6: Function with no annotations
    args = [arg("x", None), arg("return", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any"]


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test.module"

    # Test Import node
    import_node = Import(names=[alias(name="os"), alias(name="sys", asname="system")])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "os")] == "os"
    assert parser.alias[_m(root, "system")] == "sys"

    # Test ImportFrom node with level 0
    import_from_node = ImportFrom(module="collections", names=[alias(name="defaultdict")], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "defaultdict")] == "collections.defaultdict"

    # Test ImportFrom node with level 1
    import_from_node_level = ImportFrom(module="os.path", names=[alias(name="join")], level=1)
    parser.imports(root, import_from_node_level)
    assert parser.alias[_m(root, "join")] == "test.os.path.join"

    # Test ImportFrom node with asname
    import_from_asname_node = ImportFrom(module="typing", names=[alias(name="List", asname="list")], level=0)
    parser.imports(root, import_from_asname_node)
    assert parser.alias[_m(root, "list")] == "typing.List"


# LLM-generated content at query #10
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("test_module", {"test_module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Handle self_ty replacement
    resolver = Resolver("test_module", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: No replacement when name not in alias
    resolver = Resolver("test_module", {"test_module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("test_module", {"test_module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("test_module", {"test_module.name": "test_module.other", "test_module.other": "final"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"


# LLM-generated content at query #11
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with string that is not a valid name
    node = Constant("invalid name")
    assert resolver.visit_Constant(node) == node

    # Test with string that is a valid name
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"

    # Test with string that is a valid name and in alias
    resolver = Resolver("test", {"test.valid_name": "int"})
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #12
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 123

    # Test with string constant that is not a valid name
    resolver = Resolver("test", {})
    node = Constant("invalid name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "invalid name"

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.valid_name": "int"})
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"


# LLM-generated content at query #13
#--------------------------

```python
def test_Parser_func_ann():
    # Test case 1: Simple function with no self and no class method
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('y', Name('str', Load()))]
    result = list(parser.func_ann('test', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

    # Test case 2: Function with self and not a class method
    parser = Parser()
    args = [arg('self', Name('TestClass', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('test', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

    # Test case 3: Function with self and is a class method
    parser = Parser()
    args = [arg('cls', Name('TestClass', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('test', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

    # Test case 4: Function with *args
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('*', None), arg('y', Name('str', Load()))]
    result = list(parser.func_ann('test', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']

    # Test case 5: Function with **kwargs
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('**kwargs', Name('Any', Load()))]
    result = list(parser.func_ann('test', args, has_self=False, cls_method=False))
    assert result == ['int', 'Any']

    # Test case 6: Function with no annotations
    parser = Parser()
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('test', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("typing.List", Load()), Constant(1), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    union_node = Subscript(Name("typing.Union", Load()),
                          Tuple([Constant(1), Constant(2)], Load()), Load())
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert result.op.__class__.__name__ == "BitOr"

    # Test Optional conversion
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    optional_node = Subscript(Name("typing.Optional", Load()),
                             Constant(1), Load())
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert result.op.__class__.__name__ == "BitOr"
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {})
    node = Subscript(Name("some.OtherType", Load()), Constant(1), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #15
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test AnnAssign with type annotation
    node_ann = AnnAssign(
        target=Name("CONST_A", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node_ann)
    assert parser.alias["test_module.CONST_A"] == "42"
    assert parser.const["test_module.CONST_A"] == "int"
    assert parser.root["test_module.CONST_A"] == root

    # Test Assign with type comment
    node_assign = Assign(
        targets=[Name("CONST_B", Store())],
        value=Constant(3.14),
        type_comment="float"
    )
    parser.globals(root, node_assign)
    assert parser.alias["test_module.CONST_B"] == "3.14"
    assert parser.const["test_module.CONST_B"] == "float"
    assert parser.root["test_module.CONST_B"] == root

    # Test Assign without type comment (should infer type)
    node_assign_no_comment = Assign(
        targets=[Name("CONST_C", Store())],
        value=List(elts=[Constant(1), Constant(2)], ctx=Load())
    )
    parser.globals(root, node_assign_no_comment)
    assert parser.alias["test_module.CONST_C"] == "[1, 2]"
    assert parser.const["test_module.CONST_C"] == "list[int]"
    assert parser.root["test_module.CONST_C"] == root

    # Test __all__ handling
    node_all = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[
            Constant("func1"),
            Constant("func2")
        ], ctx=Load())
    )
    parser.globals(root, node_all)
    assert parser.imp[root] == {"test_module.func1", "test_module.func2"}

    # Test non-constant assignment (should not be added to const)
    node_non_const = Assign(
        targets=[Name("non_const", Store())],
        value=Constant(100)
    )
    parser.globals(root, node_non_const)
    assert "test_module.non_const" not in parser.const
    assert "test_module.non_const" not in parser.root

    # Test multiple targets (should be ignored)
    node_multi = Assign(
        targets=[
            Name("var1", Store()),
            Name("var2", Store())
        ],
        value=Constant(200)
    )
    parser.globals(root, node_multi)
    assert "test_module.var1" not in parser.alias
    assert "test_module.var2" not in parser.alias


# LLM-generated content at query #16
#--------------------------

```python
def test_Parser_class_api():
    # Test case 1: Basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 2: Class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

    # Test case 3: Class with annotated members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        AnnAssign(
            target=Name("attr2", Store()),
            annotation=Name("str", Load()),
            value=None,
            simple=1
        )
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]

    # Test case 4: Class with assigned members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name("attr1", Store())],
            value=Constant(1),
            type_comment="int"
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant("hello"),
            type_comment="str"
        )
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]

    # Test case 5: Enum class
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name("enum.Enum", Load())]
    body = [
        Assign(
            targets=[Name("VALUE1", Store())],
            value=Constant(1)
        ),
        Assign(
            targets=[Name("VALUE2", Store())],
            value=Constant(2)
        )
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestEnum" in parser.doc[name]
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 6: Class with deleted members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        Delete(
            targets=[Name("attr1", Del())]
        )
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #17
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 replacement
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union replacement
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional replacement
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 case
    resolver = Resolver("test", {})
    node = Subscript(Name("CustomType", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "CustomType"


# LLM-generated content at query #18
#--------------------------

```python
def test_Parser_compile():
    # Test basic compilation
    p = Parser()
    p.parse("test_module", "def foo(): pass")
    assert "## Module `test_module`" in p.compile()
    assert "### foo()" in p.compile()

    # Test with imports
    p = Parser()
    p.parse("test_module", "from typing import List\nx: List[int] = []")
    assert "### x" in p.compile()
    assert "list[int]" in p.compile()

    # Test with class
    p = Parser()
    p.parse("test_module", """
class MyClass:
    def method(self): pass
""")
    assert "### class MyClass" in p.compile()
    assert "#### method()" in p.compile()

    # Test with constants
    p = Parser()
    p.parse("test_module", "CONSTANT = 42")
    assert "### CONSTANT" in p.compile()
    assert "int" in p.compile()

    # Test with private members
    p = Parser()
    p.parse("test_module", """
def _private(): pass
class MyClass:
    def __init__(self): pass
""")
    assert "_private" not in p.compile()
    assert "__init__" not in p.compile()

    # Test with __all__
    p = Parser()
    p.parse("test_module", """
__all__ = ['public_func']
def public_func(): pass
def _private_func(): pass
""")
    assert "public_func" in p.compile()
    assert "_private_func" not in p.compile()

    # Test with TOC
    p = Parser(toc=True)
    p.parse("test_module", "def foo(): pass")
    assert "**Table of contents:**" in p.compile()

    # Test with missing docstring
    p = Parser()
    p.parse("test_module", "def foo(): pass")
    with pytest.warns(UserWarning):
        p.compile()


# LLM-generated content at query #19
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Replace self_ty with "Self"
    resolver = Resolver("module", {}, self_ty="self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Return original node if in alias but points to itself
    resolver = Resolver("module", {"module.name": "module.name"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 5: Support TypeVar
    resolver = Resolver("module", {"module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"


# LLM-generated content at query #20
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()

    # Test Import node
    import_node = Import(names=[alias(name='os', asname=None), alias(name='sys', asname='system')])
    parser.imports('test_module', import_node)
    assert parser.alias['test_module.os'] == 'os'
    assert parser.alias['test_module.system'] == 'sys'

    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(module='collections', names=[alias(name='defaultdict', asname=None)], level=0)
    parser.imports('test_module', import_from_node)
    assert parser.alias['test_module.defaultdict'] == 'collections.defaultdict'

    # Test ImportFrom node with level=1
    import_from_node = ImportFrom(module='typing', names=[alias(name='List', asname='MyList')], level=1)
    parser.imports('test.sub_module', import_from_node)
    assert parser.alias['test.sub_module.MyList'] == 'test.typing.List'

    # Test ImportFrom node with level=2
    import_from_node = ImportFrom(module='utils', names=[alias(name='helper', asname=None)], level=2)
    parser.imports('test.sub.sub_module', import_from_node)
    assert parser.alias['test.sub.sub_module.helper'] == 'test.utils.helper'


# LLM-generated content at query #21
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with string that is not a name
    node = Constant("not a name")
    assert resolver.visit_Constant(node) == node

    # Test with string that is a name
    node = Constant("test_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "test_name"

    # Test with string that is a name and has alias
    resolver = Resolver("test", {"test.alias_name": "int"})
    node = Constant("alias_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #22
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert f"## class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestClass"
    bases = [parse_expression("BaseClass"), parse_expression("AnotherBase")]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "AnotherBase" in parser.doc[name]

    # Test class with members (annotated assignments)
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name("public_attr", Load()),
            annotation=Name("int", Load()),
            value=None
        ),
        AnnAssign(
            target=Name("_private_attr", Load()),
            annotation=Name("str", Load()),
            value=None
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]

    # Test class with members (assignments with type comments)
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name("public_attr", Store())],
            value=Constant(42),
            type_comment="int"
        ),
        Assign(
            targets=[Name("_private_attr", Store())],
            value=Constant("hello"),
            type_comment="str"
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]

    # Test enum class
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestEnum"
    bases = [parse_expression("enum.Enum")]
    body = [
        AnnAssign(
            target=Name("VALUE1", Load()),
            annotation=Name("int", Load()),
            value=None
        ),
        AnnAssign(
            target=Name("VALUE2", Load()),
            annotation=Name("int", Load()),
            value=None
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test class with deleted members
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name("public_attr", Load()),
            annotation=Name("int", Load()),
            value=None
        ),
        Delete(
            targets=[Name("public_attr", Del())]
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #23
#--------------------------

```python
def test_Parser_imports():
    # Test Import node
    p = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os"), alias(name="sys", asname="system")])
    p.imports(root, node)
    assert p.alias["test_module.os"] == "os"
    assert p.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    p = Parser()
    node = ImportFrom(module="os.path", names=[alias(name="join")], level=0)
    p.imports(root, node)
    assert p.alias["test_module.join"] == "os.path.join"

    # Test ImportFrom node with level 1
    p = Parser()
    p.root[root] = root
    node = ImportFrom(module="sibling", names=[alias(name="func")], level=1)
    p.imports(root, node)
    assert p.alias["test_module.func"] == "test.sibling.func"

    # Test ImportFrom node with asname
    p = Parser()
    node = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    p.imports(root, node)
    assert p.alias["test_module.dd"] == "collections.defaultdict"


# LLM-generated content at query #24
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    body = [Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))]
    assert list(walk_body(body)) == body

    # Test nested If statements
    body = [
        If(
            test=Constant(value=True),
            body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
            orelse=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 2
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Assign)

    # Test Try with handlers
    body = [
        Try(
            body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))],
            handlers=[],
            orelse=[Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5))],
            finalbody=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 3
    assert all(isinstance(node, Assign) for node in result)

    # Test mixed nested structures
    body = [
        If(
            test=Constant(value=True),
            body=[
                Try(
                    body=[Assign(targets=[Name(id='d', ctx=Load())], value=Constant(value=7))],
                    handlers=[],
                    orelse=[],
                    finalbody=[]
                )
            ],
            orelse=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)


# LLM-generated content at query #25
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {"test_module.alias": "int"})
    # Test with non-string constant
    node = Constant(42)
    assert resolver.visit_Constant(node) == node
    # Test with string constant that is not a valid expression
    node = Constant("invalid expression")
    assert resolver.visit_Constant(node) == node
    # Test with string constant that is a valid expression
    node = Constant("alias")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"
    # Test with string constant that is a valid expression with attribute
    resolver = Resolver("test_module", {"test_module.alias.attr": "str"})
    node = Constant("alias.attr")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"
    assert isinstance(result.value, Name)
    assert result.value.id == "alias"


# LLM-generated content at query #26
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n"
    assert parser.doc[name] == expected_doc

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = []

    parser.alias[_m(root, "BaseClass")] = "base_module.BaseClass"
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n" + \
                   table("Bases", items=["base_module.BaseClass"])
    assert parser.doc[name] == expected_doc

    # Test class with enum members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name("Enum", Load())]
    body = [
        AnnAssign(target=Name("MEMBER1", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("MEMBER2", Store()), annotation=Name("int", Load()), value=None)
    ]

    parser.alias[_m(root, "Enum")] = "enum.Enum"
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestEnum\n\n*Full name:* `{name}`\n\n" + \
                   table("Bases", items=["enum.Enum"]) + \
                   table("Enums", items=["MEMBER1", "MEMBER2"])
    assert parser.doc[name] == expected_doc

    # Test class with public members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("public_attr", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("_private_attr", Store()), annotation=Name("int", Load()), value=None),
        Assign(targets=[Name("another_attr", Store())], value=Constant(42))
    ]

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n" + \
                   table('Members', 'Type', items=[("another_attr", "int"), ("public_attr", "int")])
    assert parser.doc[name] == expected_doc

    # Test class with deleted members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("attr1", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("attr2", Store()), annotation=Name("str", Load()), value=None),
        Delete(targets=[Name("attr1", Del())])
    ]

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n" + \
                   table('Members', 'Type', items=[("attr2", "str")])
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #27
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global names with its expression recursively
    resolver = Resolver("test", {"test.name": "int"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test case 2: Return node if name is not in alias
    resolver = Resolver("test", {"test.other": "str"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 3: Replace self_ty with "Self"
    resolver = Resolver("test", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 4: Support TypeVar
    resolver = Resolver("test", {"test.TypeVar": "typing.TypeVar"}, "Self")
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"


# LLM-generated content at query #28
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    args = arguments(
        posonlyargs=[arg("a", None)],
        args=[arg("b", None)],
        defaults=[Constant(1)],
        vararg=None,
        kwonlyargs=[arg("c", None)],
        kw_defaults=[Constant(2)],
        kwarg=None
    )
    returns = Constant("result")

    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)

    expected_doc = (
        f"## test_func()\n\n"
        f"*Full name:* `{name}`\n"
        f"<a id=\"test-module-test-func\"></a>\n\n"
        f"| | |\n"
        f"|---|---|\n"
        f"| a | Any |\n"
        f"| / |  |\n"
        f"| b | Any |\n"
        f"| c | Any |\n"
        f"| return | Any |\n"
    )
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #29
#--------------------------

```python
def test_Parser_imports():
    # Test Import node
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname="system")])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    node = ImportFrom(module="os.path", names=[alias(name="join", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.join"] == "os.path.join"

    # Test ImportFrom node with level 1
    node = ImportFrom(module="sibling", names=[alias(name="func", asname="f")], level=1)
    parser.imports(root, node)
    assert parser.alias["test_module.f"] == "test.sibling.func"


# LLM-generated content at query #30
#--------------------------

```python
def test_Parser_class_api():
    # Test case 1: Class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)

    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test-module-testclass\"></a>\n\n"
    assert parser.doc[name] == expected_doc

    # Test case 2: Class with bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = []

    parser.class_api(root, name, bases, body)

    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test-module-testclass\"></a>\n\n" + \
                   table("Bases", items=["BaseClass"])
    assert parser.doc[name] == expected_doc

    # Test case 3: Class with no bases and body with public members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("public_attr", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("_private_attr", Store()), annotation=Name("str", Load()), value=None),
    ]

    parser.class_api(root, name, bases, body)

    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test-module-testclass\"></a>\n\n" + \
                   table('Members', 'Type', items=[("public_attr", "int")])
    assert parser.doc[name] == expected_doc

    # Test case 4: Class with no bases and body with enum members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("enum.Enum", Load())]
    body = [
        AnnAssign(target=Name("ENUM_MEMBER1", Store()), annotation=None, value=None),
        AnnAssign(target=Name("ENUM_MEMBER2", Store()), annotation=None, value=None),
    ]

    parser.class_api(root, name, bases, body)

    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test-module-testclass\"></a>\n\n" + \
                   table("Bases", items=["enum.Enum"]) + \
                   table("Enums", items=["ENUM_MEMBER1", "ENUM_MEMBER2"])
    assert parser.doc[name] == expected_doc

    # Test case 5: Class with no bases and body with mixed members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("public_attr", Store()), annotation=Name("int", Load()), value=None),
        Assign(targets=[Name("assigned_attr", Store())], value=Constant(42)),
        Delete(targets=[Name("deleted_attr", Store())]),
    ]

    parser.class_api(root, name, bases, body)

    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test-module-testclass\"></a>\n\n" + \
                   table('Members', 'Type', items=[("public_attr", "int"), ("assigned_attr", "int")])
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #31
#--------------------------

```python
def test_Parser_api():
    # Test case 1: Function definition
    p = Parser()
    code = """
def foo(x: int, y: str) -> bool:
    '''This is a function.'''
    return True
"""
    p.parse("test_module", code)
    assert "test_module.foo" in p.doc
    assert "foo()" in p.doc["test_module.foo"]
    assert "Full name: `test_module.foo`" in p.doc["test_module.foo"]

    # Test case 2: Class definition
    p = Parser()
    code = """
class Bar:
    '''This is a class.'''
    def __init__(self, value: int):
        self.value = value
"""
    p.parse("test_module", code)
    assert "test_module.Bar" in p.doc
    assert "class Bar" in p.doc["test_module.Bar"]
    assert "Full name: `test_module.Bar`" in p.doc["test_module.Bar"]

    # Test case 3: Async function definition
    p = Parser()
    code = """
async def baz(x: float) -> str:
    '''This is an async function.'''
    return "hello"
"""
    p.parse("test_module", code)
    assert "test_module.baz" in p.doc
    assert "async baz()" in p.doc["test_module.baz"]
    assert "Full name: `test_module.baz`" in p.doc["test_module.baz"]

    # Test case 4: Nested class and function
    p = Parser()
    code = """
class Outer:
    class Inner:
        def method(self):
            '''Inner method.'''
            pass
    def outer_method(self):
        '''Outer method.'''
        pass
"""
    p.parse("test_module", code)
    assert "test_module.Outer" in p.doc
    assert "test_module.Outer.Inner" in p.doc
    assert "test_module.Outer.outer_method" in p.doc
    assert "test_module.Outer.Inner.method" in p.doc
    assert "class Inner" in p.doc["test_module.Outer.Inner"]
    assert "method()" in p.doc["test_module.Outer.Inner.method"]

    # Test case 5: Function with decorators
    p = Parser()
    code = """
@decorator1
@decorator2
def decorated_func():
    '''Decorated function.'''
    pass
"""
    p.parse("test_module", code)
    assert "test_module.decorated_func" in p.doc
    assert "decorated_func()" in p.doc["test_module.decorated_func"]
    assert "Decorators" in p.doc["test_module.decorated_func"]
    assert "@decorator1" in p.doc["test_module.decorated_func"]
    assert "@decorator2" in p.doc["test_module.decorated_func"]


# LLM-generated content at query #32
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is not a valid name
    resolver = Resolver("test", {})
    node = Constant("not a valid name")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "test.alias"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #33
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Non-typing attribute
    resolver = Resolver("test", {})
    node = Attribute(Name("other", Load()), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"
    assert isinstance(result.value, Name)
    assert result.value.id == "other"

    # Test case 3: Non-Name value
    resolver = Resolver("test", {})
    node = Attribute(Constant(42), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"
    assert isinstance(result.value, Constant)
    assert result.value.value == 42


# LLM-generated content at query #34
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 replacement
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union replacement
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert isinstance(result.right, Name)
    assert result.left.id == "int"
    assert result.right.id == "str"

    # Test Optional replacement
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert isinstance(result.right, Constant)
    assert result.left.id == "int"
    assert result.right.value is None

    # Test non-PEP585 case
    resolver = Resolver("test", {})
    node = Subscript(Name("SomeType", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "SomeType"


# LLM-generated content at query #35
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant(42),
            type_comment="float"
        ),
        Delete(targets=[Name("attr3", Del())])
    ]

    parser.class_api(root, name, bases, body)

    expected_doc = (
        "Bases\n"
        "-----\n"
        "BaseClass\n\n"
        "Members\n"
        "-------\n"
        "attr1\n"
        "int\n\n"
        "attr2\n"
        "float\n\n"
    )

    assert parser.doc[name] == expected_doc


# LLM-generated content at query #36
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "replacement"

    # Test case 2: Handle self type
    resolver = Resolver("module", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("module", {"module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "TypeVar"

    # Test case 5: Replace with complex expression
    resolver = Resolver("module", {"module.name": "Union[int, str]"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, BinOp)
    assert unparse(result) == "int | str"


# LLM-generated content at query #37
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test", {"test.foo": "bar"})

    # Test with non-string constant
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 42

    # Test with string constant that is not a valid name
    node = Constant("invalid name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "invalid name"

    # Test with string constant that is a valid name
    node = Constant("foo")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "bar"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("foo")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "foo"


# LLM-generated content at query #38
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))
    body = [stmt1, stmt2]
    assert list(walk_body(body)) == body

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
        orelse=[Expr(value=Call(func=Name(id='exit', ctx=Load()), args=[], keywords=[]))]
    )
    body_with_if = [stmt1, if_node, stmt2]
    expected = [
        stmt1,
        if_node.body[0],
        if_node.orelse[0],
        stmt2
    ]
    assert list(walk_body(body_with_if)) == expected

    # Test Try statement
    try_node = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
        handlers=[],
        orelse=[Expr(value=Call(func=Name(id='log', ctx=Load()), args=[], keywords=[]))],
        finalbody=[Expr(value=Call(func=Name(id='cleanup', ctx=Load()), args=[], keywords=[]))]
    )
    body_with_try = [stmt1, try_node, stmt2]
    expected = [
        stmt1,
        try_node.body[0],
        try_node.orelse[0],
        try_node.finalbody[0],
        stmt2
    ]
    assert list(walk_body(body_with_try)) == expected

    # Test nested If and Try
    nested_if = If(
        test=Constant(value=True),
        body=[try_node],
        orelse=[]
    )
    body_nested = [nested_if]
    expected = [
        try_node.body[0],
        try_node.orelse[0],
        try_node.finalbody[0]
    ]
    assert list(walk_body(body_nested)) == expected


# LLM-generated content at query #39
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test", {"typing.List": "list"})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Non-typing attribute
    resolver = Resolver("test", {"other.Module": "module"})
    node = Attribute(Name("other", Load()), "Module", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other"
    assert result.attr == "Module"

    # Test case 3: Non-Name value
    resolver = Resolver("test", {})
    node = Attribute(Constant(1), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Constant)
    assert result.attr == "attr"


# LLM-generated content at query #40
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("root", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is not a name
    node = Constant("not a name")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is a name
    node = Constant("some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "some_name"

    # Test with string constant that is a name with alias
    resolver = Resolver("root", {"root.some_name": "alias_name"})
    node = Constant("some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias_name"

    # Test with string constant that is a name with self_ty
    resolver = Resolver("root", {}, self_ty="Self")
    node = Constant("Self")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #41
#--------------------------

```python
def test_Parser_imports():
    # Test Import node
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os"), alias(name="sys", asname="system")])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    node = ImportFrom(module="collections", names=[alias(name="defaultdict")], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"

    # Test ImportFrom node with level 1
    parser.level["test_module"] = 1
    node = ImportFrom(module="os.path", names=[alias(name="join")], level=1)
    parser.imports(root, node)
    assert parser.alias["test_module.join"] == "os.path.join"

    # Test ImportFrom node with asname
    node = ImportFrom(module="typing", names=[alias(name="List", asname="list")], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.list"] == "typing.List"


# LLM-generated content at query #42
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(42)
    assert resolver.visit_Constant(node) == node

    # Test with string that is not a valid name
    node = Constant("invalid name")
    assert resolver.visit_Constant(node) == node

    # Test with valid name string
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"

    # Test with nested valid name string
    resolver = Resolver("test", {"test.nested": "int"})
    node = Constant("nested")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #43
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    node = Constant(123)
    resolver = Resolver("test", {})
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string that is not a valid name
    node = Constant("invalid name")
    resolver = Resolver("test", {})
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with valid name string
    node = Constant("valid_name")
    resolver = Resolver("test", {"test.valid_name": "new_name"})
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "new_name"

    # Test with nested name string
    node = Constant("nested.name")
    resolver = Resolver("test", {"test.nested.name": "new_nested_name"})
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "new_nested_name"


# LLM-generated content at query #44
#--------------------------

```python
def test_Parser_load_docstring():
    # Setup
    parser = Parser()
    parser.doc = {
        'test_module': '# Module `test_module`',
        'test_module.TestClass': '# class TestClass',
        'test_module.test_function': '# test_function()'
    }
    parser.docstring = {}

    # Mock module
    class MockModule:
        __doc__ = "Module docstring"
        TestClass = type('TestClass', (), {'__doc__': "Class docstring"})
        test_function = lambda: None
        test_function.__doc__ = "Function docstring"

    # Test
    parser.load_docstring('test_module', MockModule)

    # Assertions
    assert parser.docstring['test_module'] == "Module docstring"
    assert parser.docstring['test_module.TestClass'] == "Class docstring"
    assert parser.docstring['test_module.test_function'] == "Function docstring"


# LLM-generated content at query #45
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test basic name replacement
    resolver = Resolver("test", {"test.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test self type replacement
    resolver = Resolver("test", {}, "SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test TypeVar handling
    resolver = Resolver("test", {"test.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test no replacement case
    resolver = Resolver("test", {"test.other": "value"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test nested replacement
    resolver = Resolver("test", {"test.nested": "test.other", "test.other": "final"})
    node = Name("nested", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"


# LLM-generated content at query #46
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 42

    # Test with string constant that is not a valid name
    resolver = Resolver("test", {})
    node = Constant("not a valid name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "not a valid name"

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #47
#--------------------------

```python
def test_Parser_load_docstring():
    # Setup
    parser = Parser()
    parser.doc = {
        'test_module': '# Module `test_module`',
        'test_module.TestClass': '## class TestClass',
        'test_module.test_function': '## test_function()'
    }
    parser.docstring = {}

    # Create a mock module
    class MockModule:
        __doc__ = "Module docstring"
        TestClass = type('TestClass', (), {'__doc__': "Class docstring"})
        test_function = lambda: None
        test_function.__doc__ = "Function docstring"

    # Execute
    parser.load_docstring('test_module', MockModule)

    # Verify
    assert parser.docstring['test_module'] == "Module docstring"
    assert parser.docstring['test_module.TestClass'] == "Class docstring"
    assert parser.docstring['test_module.test_function'] == "Function docstring"


# LLM-generated content at query #48
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os"), alias(name="sys", asname="system")])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    import_from_node = ImportFrom(module="collections", names=[alias(name="defaultdict")], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"

    # Test ImportFrom node with level 1
    import_from_node_level = ImportFrom(module="os.path", names=[alias(name="join")], level=1)
    parser.imports(root, import_from_node_level)
    assert parser.alias["test_module.join"] == "os.path.join"

    # Test ImportFrom node with asname
    import_from_asname = ImportFrom(module="typing", names=[alias(name="List", asname="list")], level=0)
    parser.imports(root, import_from_asname)
    assert parser.alias["test_module.list"] == "typing.List"


# LLM-generated content at query #49
#--------------------------

```python
def test_Parser_globals():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"

    # Test AnnAssign with type annotation
    node_annassign = AnnAssign(
        target=Name("CONSTANT", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node_annassign)
    assert parser.alias["test_module.CONSTANT"] == "42"
    assert parser.const["test_module.CONSTANT"] == "int"
    assert parser.root["test_module.CONSTANT"] == root

    # Test AnnAssign without value (should not process)
    node_annassign_no_value = AnnAssign(
        target=Name("VAR", Load()),
        annotation=Name("str", Load()),
        value=None,
        simple=1
    )
    parser.globals(root, node_annassign_no_value)
    assert "test_module.VAR" not in parser.alias

    # Test Assign with type comment
    node_assign_type_comment = Assign(
        targets=[Name("ANOTHER_CONST", Load())],
        value=Constant(3.14),
        type_comment="float"
    )
    parser.globals(root, node_assign_type_comment)
    assert parser.alias["test_module.ANOTHER_CONST"] == "3.14"
    assert parser.const["test_module.ANOTHER_CONST"] == "float"

    # Test Assign without type comment (should infer type)
    node_assign_no_comment = Assign(
        targets=[Name("INFERRED", Load())],
        value=List(elts=[Constant(1), Constant(2)], ctx=Load())
    )
    parser.globals(root, node_assign_no_comment)
    assert parser.alias["test_module.INFERRED"] == "[1, 2]"
    assert parser.const["test_module.INFERRED"] == "list[int]"

    # Test __all__ processing
    node_all = Assign(
        targets=[Name("__all__", Load())],
        value=List(elts=[
            Constant("public_func"),
            Constant("PublicClass")
        ], ctx=Load())
    )
    parser.globals(root, node_all)
    assert parser.imp[root] == {"test_module.public_func", "test_module.PublicClass"}

    # Test non-constant assignment (should not add to const)
    node_non_const = Assign(
        targets=[Name("variable", Load())],
        value=Constant("test")
    )
    parser.globals(root, node_non_const)
    assert "test_module.variable" not in parser.const

    # Test multiple targets (should not process)
    node_multi_target = Assign(
        targets=[Name("a", Load()), Name("b", Load())],
        value=Constant(1)
    )
    parser.globals(root, node_multi_target)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #50
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"test.typing.Custom": "Custom"})
    node = Subscript(Name("Custom", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Custom"


# LLM-generated content at query #51
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(42)
    assert resolver.visit_Constant(node) == node

    # Test with string constant that is not a valid name
    resolver = Resolver("test", {})
    node = Constant("not a valid name")
    assert resolver.visit_Constant(node) == node

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #52
#--------------------------

```python
def test_walk_body():
    # Test simple body
    body = [Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test body with If statement
    body = [
        If(
            test=Constant(value=True),
            body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
            orelse=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 2
    assert all(isinstance(node, Assign) for node in result)

    # Test body with Try statement
    body = [
        Try(
            body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))],
            handlers=[],
            orelse=[Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5))],
            finalbody=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 3
    assert all(isinstance(node, Assign) for node in result)

    # Test nested If statements
    body = [
        If(
            test=Constant(value=True),
            body=[
                If(
                    test=Constant(value=False),
                    body=[Assign(targets=[Name(id='d', ctx=Load())], value=Constant(value=7))],
                    orelse=[]
                )
            ],
            orelse=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test empty body
    body = []
    result = list(walk_body(body))
    assert len(result) == 0


# LLM-generated content at query #53
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string Constant
    resolver = Resolver("test", {}, "Self")
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with string Constant that is not a valid name
    node = Constant("invalid name")
    assert resolver.visit_Constant(node) == node

    # Test with string Constant that is a valid name
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"

    # Test with string Constant that is a valid name with alias
    resolver = Resolver("test", {"test.valid_name": "int"}, "Self")
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #54
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {})
    node = Subscript(Name("SomeType", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "SomeType"


# LLM-generated content at query #55
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()

    # Test Import node
    import_node = Import(names=[alias(name='os', asname=None)])
    parser.imports('test_module', import_node)
    assert parser.alias['test_module.os'] == 'os'

    import_node_with_asname = Import(names=[alias(name='numpy', asname='np')])
    parser.imports('test_module', import_node_with_asname)
    assert parser.alias['test_module.np'] == 'numpy'

    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(module='collections', names=[alias(name='defaultdict')], level=0)
    parser.imports('test_module', import_from_node)
    assert parser.alias['test_module.defaultdict'] == 'collections.defaultdict'

    # Test ImportFrom node with level>0 and asname
    import_from_node_with_level = ImportFrom(module='os.path', names=[alias(name='join', asname='path_join')], level=1)
    parser.imports('test_module', import_from_node_with_level)
    assert parser.alias['test_module.path_join'] == 'os.path.join'


# LLM-generated content at query #56
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert f"class {name}" in parser.doc[name]

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = [parse("BaseClass").body[0].value]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

    # Test class with enum members
    parser = Parser()
    root = "test_module"
    name = "TestEnum"
    bases = [parse("enum.Enum").body[0].value]
    body = [
        AnnAssign(target=Name("MEMBER1", Load()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("MEMBER2", Load()), annotation=Name("int", Load()), value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "MEMBER1" in parser.doc[name]
    assert "MEMBER2" in parser.doc[name]

    # Test class with public members
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("public_attr", Load()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("_private_attr", Load()), annotation=Name("int", Load()), value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]

    # Test class with deleted members
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("attr1", Load()), annotation=Name("int", Load()), value=None),
        Delete(targets=[Name("attr1", Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "attr1" not in parser.doc[name]


# LLM-generated content at query #57
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with string constant that is not a valid expression
    node = Constant("invalid expression")
    assert resolver.visit_Constant(node) == node

    # Test with string constant that is a valid expression
    node = Constant("test_var")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "test_var"

    # Test with string constant that is a valid expression with alias
    resolver = Resolver("test", {"test.test_var": "int"})
    node = Constant("test_var")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #58
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Constant(value="hello"))
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
        orelse=[Expr(value=Constant(value="world"))]
    )
    assert list(walk_body([if_node])) == [
        Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2)),
        Expr(value=Constant(value="world"))
    ]

    # Test Try statement
    try_node = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
        handlers=[],
        orelse=[Expr(value=Constant(value="try_else"))],
        finalbody=[Expr(value=Constant(value="finally"))]
    )
    assert list(walk_body([try_node])) == [
        Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3)),
        Expr(value=Constant(value="try_else")),
        Expr(value=Constant(value="finally"))
    ]

    # Test nested If statements
    nested_if = If(
        test=Constant(value=True),
        body=[
            If(
                test=Constant(value=False),
                body=[Expr(value=Constant(value="nested"))],
                orelse=[]
            )
        ],
        orelse=[]
    )
    assert list(walk_body([nested_if])) == [Expr(value=Constant(value="nested"))]

    # Test Try with handlers
    try_with_handler = Try(
        body=[Expr(value=Constant(value="try_body"))],
        handlers=[
            {
                'type': Name(id='Exception', ctx=Load()),
                'name': 'e',
                'body': [Expr(value=Constant(value="handler"))]
            }
        ],
        orelse=[],
        finalbody=[]
    )
    # Note: The actual handler structure is more complex, this is simplified for testing
    # In real implementation, you'd need proper handler node construction
    assert list(walk_body([try_with_handler])) == [
        Expr(value=Constant(value="try_body")),
        Expr(value=Constant(value="handler"))
    ]


# LLM-generated content at query #59
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(42)
    assert resolver.visit_Constant(node) == node

    # Test with string that is not a name
    node = Constant("not a name")
    assert resolver.visit_Constant(node) == node

    # Test with string that is a name
    node = Constant("test_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "test_name"

    # Test with string that is a name and has alias
    resolver = Resolver("test", {"test.alias_name": "int"})
    node = Constant("alias_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #60
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test Union resolution
    resolver = Resolver("test", {"test.typing.Union": "test.typing.Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name) and result.right.id == "str"

    # Test Optional resolution
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant) and result.right.value is None

    # Test PEP585 resolution with warning
    resolver = Resolver("test", {"test.typing.List": "test.typing.List"})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    with pytest.warns(UserWarning):
        result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name) and result.value.id == "list"
    assert isinstance(result.slice, Name) and result.slice.id == "int"

    # Test non-PEP585 case
    node = Subscript(Name("CustomType", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result is node


# LLM-generated content at query #61
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test 1: Replace global names with its expression recursively
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test 2: Handle self_ty replacement
    resolver = Resolver("module", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test 3: No replacement when name not in alias
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test 4: Handle TypeVar case
    resolver = Resolver("module", {"module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test 5: Recursive replacement
    resolver = Resolver("module", {"module.name": "replacement", "module.replacement": "final"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"


# LLM-generated content at query #62
#--------------------------

```python
def test_const_type():
    # Test Constant
    node = Constant(value=42)
    assert const_type(node) == "int"

    # Test Tuple
    node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "tuple[int, int]"

    # Test List
    node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "list[int, int]"

    # Test Set
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "set[int, int]"

    # Test Dict
    node = Dict(keys=[Constant(value="a"), Constant(value="b")], values=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "dict[str, int]"

    # Test Call with Name
    node = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "int"

    # Test Call with Attribute
    node = Call(func=Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "Any"

    # Test unsupported node type
    node = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(node) == "Any"

    # Test empty elements
    node = List(elts=[], ctx=Load())
    assert const_type(node) == "Any"

    # Test mixed types in elements
    node = List(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(node) == "Any"


# LLM-generated content at query #63
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test_module", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Non-typing attribute remains unchanged
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert result.value.id == "other_module"

    # Test case 3: Nested attribute access
    node = Attribute(Attribute(Name("typing", Load()), "List", Load()), "append", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "append"
    assert isinstance(result.value, Name)
    assert result.value.id == "List"


# LLM-generated content at query #64
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test_module", {"test_module.typing": "typing"})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Keep non-typing attribute
    resolver = Resolver("test_module", {})
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert result.value.id == "other_module"

    # Test case 3: Non-Name value
    resolver = Resolver("test_module", {})
    node = Attribute(Constant(42), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"


# LLM-generated content at query #65
#--------------------------

```python
def test_Parser_load_docstring():
    # Setup
    parser = Parser()
    parser.doc = {
        'module.submodule': 'Documentation for submodule',
        'module.submodule.function': 'Documentation for function'
    }
    parser.docstring = {}

    # Create a mock module with docstrings
    class MockModule:
        def __init__(self):
            self.submodule = MockSubmodule()

    class MockSubmodule:
        def __init__(self):
            self.function = "This is the function's docstring"

        def __doc__(self):
            return "This is the submodule's docstring"

    mock_module = MockModule()
    mock_module.__doc__ = "This is the module's docstring"

    # Execute
    parser.load_docstring('module', mock_module)

    # Verify
    assert parser.docstring['module'] == doctest("This is the module's docstring")
    assert parser.docstring['module.submodule'] == doctest("This is the submodule's docstring")
    assert parser.docstring['module.submodule.function'] == doctest("This is the function's docstring")


# LLM-generated content at query #66
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test_module-testclass\"></a>\n\n"
    assert parser.doc[name] == expected_doc

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = []

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test_module-testclass\"></a>\n\n" + \
                   "| Bases |\n|-------|\n| `BaseClass` |\n"
    assert parser.doc[name] == expected_doc

    # Test class with members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("member1", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("member2", Store()), annotation=Name("str", Load()), value=None),
    ]

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test_module-testclass\"></a>\n\n" + \
                   "| Members | Type |\n|---------|------|\n| `member1` | `int` |\n| `member2` | `str` |\n"
    assert parser.doc[name] == expected_doc

    # Test enum class
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name("enum.Enum", Load())]
    body = [
        AnnAssign(target=Name("OPTION1", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("OPTION2", Store()), annotation=Name("int", Load()), value=None),
    ]

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestEnum\n\n*Full name:* `{name}`\n<a id=\"test_module-testenum\"></a>\n\n" + \
                   "| Enums |\n|-------|\n| `OPTION1` |\n| `OPTION2` |\n"
    assert parser.doc[name] == expected_doc

    # Test class with deleted members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("member1", Store()), annotation=Name("int", Load()), value=None),
        Delete(targets=[Name("member1", Del())]),
    ]

    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n<a id=\"test_module-testclass\"></a>\n\n"
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #67
#--------------------------

```python
def test_Resolver_visit_Subscript():
    resolver = Resolver("test_module", {"test_module.typing.Union": "Union", "test_module.typing.Optional": "Optional", "test_module.typing.List": "list"})

    # Test Union
    union_node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional
    optional_node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test PEP585
    list_node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(list_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

    # Test non-pep585
    other_node = Subscript(Name("Other", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(other_node)
    assert result is other_node


# LLM-generated content at query #68
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"

    import_node_with_asname = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_with_asname)
    assert parser.alias["test_module.np"] == "numpy"

    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(module="sys", names=[alias(name="path", asname=None)], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias["test_module.path"] == "sys.path"

    import_from_node_with_asname = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    parser.imports(root, import_from_node_with_asname)
    assert parser.alias["test_module.dd"] == "collections.defaultdict"

    # Test ImportFrom node with level>0
    import_from_node_with_level = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    parser.imports(root, import_from_node_with_level)
    assert parser.alias["test_module.path"] == "os.path"

    # Test multiple imports
    parser.alias.clear()
    multiple_imports = Import(names=[alias(name="sys", asname=None), alias(name="os", asname="operating_system")])
    parser.imports(root, multiple_imports)
    assert parser.alias["test_module.sys"] == "sys"
    assert parser.alias["test_module.operating_system"] == "os"


# LLM-generated content at query #69
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"typing.Dict": "Dict"})
    node = Subscript(Name("Dict", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"


# LLM-generated content at query #70
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {"test_module.A": "int"})

    # Test with non-string constant
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 42

    # Test with string that is not a valid name
    node = Constant("not a name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "not a name"

    # Test with string that is a valid name
    node = Constant("A")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string that is a valid attribute access
    node = Constant("A.B")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Attribute)
    assert result.attr == "B"
    assert isinstance(result.value, Name)
    assert result.value.id == "int"


# LLM-generated content at query #71
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with a non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with a string that is not a valid expression
    resolver = Resolver("test", {})
    node = Constant("invalid expression")
    assert resolver.visit_Constant(node) == node

    # Test with a string that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with a string that is a valid attribute
    resolver = Resolver("test", {"test.module.name": "str"})
    node = Constant("module.name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "str"

    # Test with a string that is a valid subscript
    resolver = Resolver("test", {"test.List": "list"})
    node = Constant("List[int]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"


# LLM-generated content at query #72
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 123

    # Test with string constant that is not a valid expression
    resolver = Resolver("test", {})
    node = Constant("invalid expression")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "invalid expression"

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid attribute
    resolver = Resolver("test", {"test.attr": "str"})
    node = Constant("attr")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "str"


# LLM-generated content at query #73
#--------------------------

```python
def test_Parser_class_api():
    # Test case 1: Class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 2: Class with bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse_expression("BaseClass")]
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 3: Class with no bases and body with AnnAssign
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [AnnAssign(target=Name("attr1", Load()), annotation=Name("int", Load()), value=None)]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "int" in parser.doc[name]

    # Test case 4: Class with no bases and body with Assign
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [Assign(targets=[Name("attr1", Load())], value=Constant(1))]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "int" in parser.doc[name]

    # Test case 5: Class with no bases and body with Delete
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [Delete(targets=[Name("attr1", Load())])]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 6: Class with enum base and body with AnnAssign
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse_expression("enum.Enum")]
    body = [AnnAssign(target=Name("ATTR1", Load()), annotation=Name("int", Load()), value=None)]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "enum.Enum" in parser.doc[name]
    assert "Enums" in parser.doc[name]
    assert "ATTR1" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 7: Class with enum base and body with Assign
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse_expression("enum.Enum")]
    body = [Assign(targets=[Name("ATTR1", Load())], value=Constant(1))]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "enum.Enum" in parser.doc[name]
    assert "Enums" in parser.doc[name]
    assert "ATTR1" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 8: Class with enum base and body with Delete
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse_expression("enum.Enum")]
    body = [Delete(targets=[Name("ATTR1", Load())])]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "enum.Enum" in parser.doc[name]
    assert "Enums" not in parser.doc[name]
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #74
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test_module", {"test_module.typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test_module", {"test_module.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional conversion
    resolver = Resolver("test_module", {"test_module.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-typing subscript
    resolver = Resolver("test_module", {})
    node = Subscript(Name("SomeClass", Load()), Name("attr", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result is node

    # Test non-Tuple slice
    resolver = Resolver("test_module", {"test_module.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #75
#--------------------------

```python
def test_const_type():
    # Test Constant node with different types
    assert const_type(Constant(value=42)) == 'int'
    assert const_type(Constant(value=3.14)) == 'float'
    assert const_type(Constant(value="hello")) == 'str'
    assert const_type(Constant(value=True)) == 'bool'
    assert const_type(Constant(value=None)) == 'NoneType'

    # Test Tuple node
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == 'tuple[int, int]'

    # Test List node
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == 'list[int, int]'

    # Test Set node
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == 'set[int, int]'

    # Test Dict node
    dict_node = Dict(keys=[Constant(value="a"), Constant(value="b")],
                     values=[Constant(value=1), Constant(value=2)])
    assert const_type(dict_node) == 'dict[str, int]'

    # Test Call node with type conversion functions
    call_node = Call(func=Name(id='int', ctx=Load()), args=[], keywords=[])
    assert const_type(call_node) == 'int'

    # Test unsupported node types
    assert const_type(Name(id='x', ctx=Load())) == 'Any'
    assert const_type(BinOp(left=Constant(value=1), op=Add(), right=Constant(value=2))) == 'Any'


# LLM-generated content at query #76
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n"
    assert parser.doc[name] == expected_doc

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = []
    parser.alias[_m(root, "BaseClass")] = "BaseClass"
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n"
    expected_doc += "| Bases |\n|-------|\n| `BaseClass` |\n"
    assert parser.doc[name] == expected_doc

    # Test class with enum members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name("enum.Enum", Load())]
    body = [
        AnnAssign(target=Name("MEMBER1", Store()), annotation=Constant(1), value=None),
        AnnAssign(target=Name("MEMBER2", Store()), annotation=Constant(2), value=None),
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestEnum\n\n*Full name:* `{name}`\n\n"
    expected_doc += "| Bases |\n|-------|\n| `enum.Enum` |\n"
    expected_doc += "| Enums |\n|-------|\n| `MEMBER1` |\n| `MEMBER2` |\n"
    assert parser.doc[name] == expected_doc

    # Test class with public members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("public_attr", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("_private_attr", Store()), annotation=Name("str", Load()), value=None),
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n"
    expected_doc += "| Members | Type |\n|----------|------|\n| `public_attr` | `int` |\n"
    assert parser.doc[name] == expected_doc

    # Test class with mixed members and enum
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("enum.Enum", Load())]
    body = [
        AnnAssign(target=Name("MEMBER1", Store()), annotation=Constant(1), value=None),
        AnnAssign(target=Name("public_attr", Store()), annotation=Name("int", Load()), value=None),
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n"
    expected_doc += "| Bases |\n|-------|\n| `enum.Enum` |\n"
    expected_doc += "| Enums |\n|-------|\n| `MEMBER1` |\n"
    assert parser.doc[name] == expected_doc

    # Test class with deleted members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name("public_attr", Store()), annotation=Name("int", Load()), value=None),
        Delete(targets=[Name("public_attr", Del())]),
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = f"## class TestClass\n\n*Full name:* `{name}`\n\n"
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #77
#--------------------------

```python
def test_Parser_func_api():
    # Test case 1: Simple function with no arguments and no return annotation
    parser = Parser()
    root = "test_module"
    name = "test_module.simple_func"
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_doc = f"### simple_func()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("return", items=[ANY])
    assert parser.doc[name] == expected_doc

    # Test case 2: Function with positional arguments and return annotation
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_args"
    pos_arg1 = arg("arg1", Name("int", Load()))
    pos_arg2 = arg("arg2", Name("str", Load()))
    args = arguments(posonlyargs=[], args=[pos_arg1, pos_arg2], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = Name("bool", Load())
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_doc = f"### func_with_args()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("arg1", "arg2", "return", items=[["int", "str", "bool"]])
    assert parser.doc[name] == expected_doc

    # Test case 3: Function with default arguments
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_defaults"
    pos_arg1 = arg("arg1", Name("int", Load()))
    pos_arg2 = arg("arg2", Name("str", Load()))
    default1 = Constant(10)
    default2 = Constant("default")
    args = arguments(posonlyargs=[], args=[pos_arg1, pos_arg2], kwonlyargs=[], kw_defaults=[], defaults=[default1, default2])
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_doc = f"### func_with_defaults()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("arg1", "arg2", "return", items=[["int", "str", ANY], ["10", "\"default\"", ""]])
    assert parser.doc[name] == expected_doc

    # Test case 4: Function with *args and **kwargs
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_varargs"
    pos_arg = arg("arg", Name("int", Load()))
    vararg = arg("args", None)
    kwarg = arg("kwargs", None)
    args = arguments(posonlyargs=[], args=[pos_arg], kwonlyargs=[], kw_defaults=[], defaults=[], vararg=vararg, kwarg=kwarg)
    returns = Name("None", Load())
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_doc = f"### func_with_varargs()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("arg", "*args", "**kwargs", "return", items=[["int", "", "", "None"]])
    assert parser.doc[name] == expected_doc

    # Test case 5: Method with self and classmethod
    parser = Parser()
    root = "test_module"
    name = "test_module.MyClass.method"
    self_arg = arg("self", Name("MyClass", Load()))
    other_arg = arg("other", Name("int", Load()))
    args = arguments(posonlyargs=[], args=[self_arg, other_arg], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = Name("str", Load())
    parser.func_api(root, name, args, returns, has_self=True, cls_method=True)
    expected_doc = f"### method()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("other", "return", items=[["int", "str"]])
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #78
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global names with its expression recursively
    resolver = Resolver("test_module", {"test_module.name": "replaced_name"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replaced_name"

    # Test case 2: Handle self_ty replacement
    resolver = Resolver("test_module", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("test_module", {"test_module.other": "other_name"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("test_module", {"test_module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test case 5: Handle nested alias resolution
    resolver = Resolver("test_module", {"test_module.A": "test_module.B", "test_module.B": "final_name"})
    node = Name("A", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final_name"


# LLM-generated content at query #79
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {"test_module.SomeClass": "SomeClass"})
    # Test with non-string constant
    node = Constant(42)
    assert resolver.visit_Constant(node) == node
    # Test with string that is not a valid name
    node = Constant("not a valid name!")
    assert resolver.visit_Constant(node) == node
    # Test with string that is a valid name
    node = Constant("SomeClass")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "SomeClass"
    # Test with string that is a valid name with module path
    node = Constant("test_module.SomeClass")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "SomeClass"


# LLM-generated content at query #80
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("attr1", Store()), Constant(1), None, None),
        AnnAssign(Name("attr2", Store()), Constant("hello"), None, None),
        Assign([Name("attr3", Store())], Constant(3.14)),
        Delete([Name("attr2", Store())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" not in parser.doc[name]
    assert "attr3" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "float" in parser.doc[name]


# LLM-generated content at query #81
#--------------------------

```python
def test_const_type():
    # Test Constant node
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"

    # Test Tuple node
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == "tuple[int, int]"

    # Test List node
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == "list[int, int]"

    # Test Set node
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == "set[int, int]"

    # Test Dict node
    dict_node = Dict(keys=[Constant(value="a"), Constant(value="b")],
                     values=[Constant(value=1), Constant(value=2)])
    assert const_type(dict_node) == "dict[str, int]"

    # Test Call node with Name
    call_node = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(call_node) == "int"

    # Test Call node with Attribute
    attr_node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    call_node = Call(func=attr_node, args=[], keywords=[])
    assert const_type(call_node) == "list"

    # Test unsupported node
    binop_node = BinOp(left=Constant(value=1), op=Add(), right=Constant(value=2))
    assert const_type(binop_node) == "Any"

    # Test empty node
    assert const_type(None) == "Any"


# LLM-generated content at query #82
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("module", {"module.name": "expression"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "expression"

    # Test case 2: Handle self_ty
    resolver = Resolver("module", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("module", {"module.other": "other_expression"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar
    resolver = Resolver("module", {"module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test case 5: Handle nested alias
    resolver = Resolver("module", {"module.name": "module.other", "module.other": "expression"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "expression"


# LLM-generated content at query #83
#--------------------------

```python
def test_Parser_func_api():
    # Test case 1: Simple function with no arguments and no return type
    parser = Parser()
    root = "test_module"
    name = f"{root}.test_func"
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"{name}\n\n*Full name:* `{name}`\n\n"
    expected_output += table("return")
    assert parser.doc[name] == expected_output

    # Test case 2: Function with positional arguments and return type
    parser = Parser()
    root = "test_module"
    name = f"{root}.test_func"
    args = arguments(
        posonlyargs=[arg("x", None), arg("y", None)],
        args=[arg("z", None)],
        kwonlyargs=[arg("a", None)],
        kw_defaults=[Constant(1)],
        defaults=[Constant(2), Constant(3)]
    )
    returns = Constant("str")
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"{name}\n\n*Full name:* `{name}`\n\n"
    expected_output += table("x", "y", "/", "z", "*", "a", "return", items=[["", "", "", "", "", "int", "str"]])
    assert parser.doc[name] == expected_output

    # Test case 3: Function with self and class method
    parser = Parser()
    root = "test_module"
    name = f"{root}.TestClass.test_method"
    args = arguments(
        posonlyargs=[arg("self", None)],
        args=[arg("x", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = Constant("int")
    parser.func_api(root, name, args, returns, has_self=True, cls_method=True)
    expected_output = f"{name}\n\n*Full name:* `{name}`\n\n"
    expected_output += table("self", "x", "return", items=[["type[Self]", "int", "int"]])
    assert parser.doc[name] == expected_output

    # Test case 4: Function with *args and **kwargs
    parser = Parser()
    root = "test_module"
    name = f"{root}.test_func"
    args = arguments(
        posonlyargs=[],
        args=[arg("x", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    args.vararg = arg("*args", None)
    args.kwarg = arg("**kwargs", None)
    returns = Constant("None")
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"{name}\n\n*Full name:* `{name}`\n\n"
    expected_output += table("x", "*args", "**kwargs", "return", items=[["", "", "", "None"]])
    assert parser.doc[name] == expected_output


# LLM-generated content at query #84
#--------------------------

```python
def test_Parser_load_docstring():
    # Setup
    parser = Parser()
    parser.doc = {
        'test_module': '# Module `test_module`',
        'test_module.func': '# func()',
        'test_module.Class': '# class Class'
    }
    parser.docstring = {}
    parser.root = {
        'test_module': 'test_module',
        'test_module.func': 'test_module',
        'test_module.Class': 'test_module'
    }

    # Mock module
    class MockModule:
        __doc__ = "Module docstring"
        func = lambda: None
        func.__doc__ = "Function docstring"
        Class = type('Class', (), {'__doc__': "Class docstring"})

    # Test
    parser.load_docstring('test_module', MockModule)

    # Assertions
    assert parser.docstring['test_module'] == doctest("Module docstring")
    assert parser.docstring['test_module.func'] == doctest("Function docstring")
    assert parser.docstring['test_module.Class'] == doctest("Class docstring")


# LLM-generated content at query #85
#--------------------------

```python
def test_Resolver_visit_Subscript():
    resolver = Resolver("test", {"test.typing.Union": "Union", "test.typing.Optional": "Optional", "test.typing.List": "list"})
    # Test Union
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"
    # Test Optional
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    # Test PEP585
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"
    # Test non-PEP585
    node = Subscript(Name("Dict", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"


# LLM-generated content at query #86
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []
    parser.alias[_m(root, "BaseClass")] = "base_module.BaseClass"
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "base_module.BaseClass" in parser.doc[name]

    # Test class with members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "int" in parser.doc[name]

    # Test enum class
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="Enum", ctx=Load())]
    body = [
        AnnAssign(
            target=Name(id="VALUE1", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None
        )
    ]
    parser.alias[_m(root, "Enum")] = "enum.Enum"
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test class with deleted attribute
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None
        ),
        Delete(targets=[Name(id="public_attr", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]

    # Test class with type comment in assignment
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name(id="public_attr", ctx=Store())],
            value=Constant(value=42),
            type_comment="int"
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "int" in parser.doc[name]


# LLM-generated content at query #87
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is not a valid expression
    resolver = Resolver("test", {})
    node = Constant("invalid expression")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid attribute
    resolver = Resolver("test", {"test.module.Class": "str"})
    node = Constant("module.Class")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "str"


# LLM-generated content at query #88
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("attr1", Store()), Constant(1), None, None),
        AnnAssign(Name("attr2", Store()), Constant("hello"), None, None),
        Assign([Name("attr3", Store())], Constant(3.14), None),
        Delete([Name("attr2", Del())])
    ]

    # Execute
    parser.class_api(root, name, bases, body)

    # Verify
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" not in parser.doc[name]  # Deleted
    assert "attr3" in parser.doc[name]
    assert "3.14" in parser.doc[name]  # Type inferred for attr3


# LLM-generated content at query #89
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    parser.alias = {}
    parser.root = {}
    parser.level = {}
    parser.doc = {}
    parser.docstring = {}
    parser.imp = {}
    parser.const = {}

    # Test case 1: Simple function with no self and no class method
    args = [
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

    # Test case 2: Function with self and no class method
    args = [
        arg('self', Name('module.Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str']

    # Test case 3: Function with self and class method
    args = [
        arg('cls', Name('module.Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int', 'str']

    # Test case 4: Function with *args
    args = [
        arg('x', Name('int', Load())),
        arg('*', None),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']

    # Test case 5: Function with no annotations
    args = [
        arg('x', None),
        arg('return', None)
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']


# LLM-generated content at query #90
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(42)
    assert resolver.visit_Constant(node) == node

    # Test with string that is not a valid name
    node = Constant("not a name")
    assert resolver.visit_Constant(node) == node

    # Test with string that is a valid name
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"

    # Test with string that is a valid name and in alias
    resolver = Resolver("test", {"test.valid_name": "int"})
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #91
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()

    # Test case 1: Simple function with no self and no class method
    args = [
        arg('x', Name('int', Load())),
        arg('y', Name('str', Load())),
        arg('return', Name('bool', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', 'str', 'bool']

    # Test case 2: Function with self and not a class method
    args = [
        arg('self', Name('Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str']

    # Test case 3: Function with self and is a class method
    args = [
        arg('cls', Name('Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int', 'str']

    # Test case 4: Function with *args and **kwargs
    args = [
        arg('x', Name('int', Load())),
        arg('*', None),
        arg('y', Name('str', Load())),
        arg('**', None),
        arg('return', Name('bool', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str', '', 'bool']

    # Test case 5: Function with no annotations
    args = [
        arg('x', None),
        arg('y', None),
        arg('return', None)
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any', 'Any']

    # Test case 6: Function with mixed annotations
    args = [
        arg('self', Name('Class', Load())),
        arg('x', None),
        arg('y', Name('str', Load())),
        arg('return', None)
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'Any', 'str', 'Any']


# LLM-generated content at query #92
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string that is not a valid expression
    node = Constant("not a valid expression")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string that is a valid attribute
    resolver = Resolver("test", {"test.module.Class": "test.module.Class"})
    node = Constant("module.Class")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Attribute)
    assert result.attr == "Class"
    assert isinstance(result.value, Name)
    assert result.value.id == "module"


# LLM-generated content at query #93
#--------------------------

```python
def test_Parser_imports():
    # Test Import node
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os"), alias(name="sys", asname="system")])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    node = ImportFrom(module="collections", names=[alias(name="defaultdict")], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"

    # Test ImportFrom node with level 1
    node = ImportFrom(module="os.path", names=[alias(name="join")], level=1)
    parser.imports(root, node)
    assert parser.alias["test_module.join"] == "test_module.os.path.join"


# LLM-generated content at query #94
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 transformation
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union transformation
    resolver = Resolver("test", {"typing.Union": "Union"})
    union_node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value == "str"

    # Test Optional transformation
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    optional_node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-transformable case
    resolver = Resolver("test", {})
    node = Subscript(Name("SomeClass", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #95
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root["test_module"] = "test_module"

    # Test with a simple class
    class_node = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.class_api("test_module", "test_module.TestClass", class_node.bases, class_node.body)
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]

    # Test with a class that has bases
    class_node_with_bases = ClassDef(
        name="TestClassWithBases",
        bases=[Name(id="BaseClass", ctx=Load())],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.alias["test_module.BaseClass"] = "BaseClass"
    parser.class_api("test_module", "test_module.TestClassWithBases", class_node_with_bases.bases, class_node_with_bases.body)
    assert "Bases" in parser.doc["test_module.TestClassWithBases"]
    assert "BaseClass" in parser.doc["test_module.TestClassWithBases"]

    # Test with a class that has members
    class_node_with_members = ClassDef(
        name="TestClassWithMembers",
        bases=[],
        keywords=[],
        body=[
            AnnAssign(
                target=Name(id="member1", ctx=Store()),
                annotation=Name(id="int", ctx=Load()),
                value=None,
                simple=1
            ),
            AnnAssign(
                target=Name(id="member2", ctx=Store()),
                annotation=Name(id="str", ctx=Load()),
                value=None,
                simple=1
            )
        ],
        decorator_list=[]
    )
    parser.class_api("test_module", "test_module.TestClassWithMembers", class_node_with_members.bases, class_node_with_members.body)
    assert "Members" in parser.doc["test_module.TestClassWithMembers"]
    assert "member1" in parser.doc["test_module.TestClassWithMembers"]
    assert "member2" in parser.doc["test_module.TestClassWithMembers"]

    # Test with an enum class
    class_node_enum = ClassDef(
        name="TestEnum",
        bases=[Name(id="Enum", ctx=Load())],
        keywords=[],
        body=[
            Assign(
                targets=[Name(id="VALUE1", ctx=Store())],
                value=Constant(value=1)
            ),
            Assign(
                targets=[Name(id="VALUE2", ctx=Store())],
                value=Constant(value=2)
            )
        ],
        decorator_list=[]
    )
    parser.alias["test_module.Enum"] = "enum.Enum"
    parser.class_api("test_module", "test_module.TestEnum", class_node_enum.bases, class_node_enum.body)
    assert "Enums" in parser.doc["test_module.TestEnum"]
    assert "VALUE1" in parser.doc["test_module.TestEnum"]
    assert "VALUE2" in parser.doc["test_module.TestEnum"]

    # Test with a class that has deleted members
    class_node_with_deleted = ClassDef(
        name="TestClassWithDeleted",
        bases=[],
        keywords=[],
        body=[
            AnnAssign(
                target=Name(id="member1", ctx=Store()),
                annotation=Name(id="int", ctx=Load()),
                value=None,
                simple=1
            ),
            Delete(
                targets=[Name(id="member1", ctx=Del())]
            )
        ],
        decorator_list=[]
    )
    parser.class_api("test_module", "test_module.TestClassWithDeleted", class_node_with_deleted.bases, class_node_with_deleted.body)
    assert "Members" not in parser.doc["test_module.TestClassWithDeleted"]


# LLM-generated content at query #96
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with alias
    resolver = Resolver("test", {"test.name": "alias.name"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "alias.name"

    # Test case 2: Return original node if no alias
    resolver = Resolver("test", {})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 3: Handle self_ty replacement
    resolver = Resolver("test", {}, self_ty="SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 4: Recursive alias replacement
    resolver = Resolver("test", {"test.name": "alias.other", "alias.other": "final.name"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final.name"

    # Test case 5: TypeVar should remain unchanged
    resolver = Resolver("test", {"test.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #97
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test_module", {"typing.List": "list"})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Non-typing attribute remains unchanged
    resolver = Resolver("test_module", {"typing.List": "list"})
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert isinstance(result.value, Name)
    assert result.value.id == "other_module"

    # Test case 3: Nested attribute (non-Name value)
    resolver = Resolver("test_module", {"typing.List": "list"})
    node = Attribute(Attribute(Name("typing", Load()), "List", Load()), "nested", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "nested"
    assert isinstance(result.value, Name)
    assert result.value.id == "List"


# LLM-generated content at query #98
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant(42),
            type_comment="float"
        ),
        Delete(targets=[Name("attr3", Del())])
    ]

    # Expected output
    expected_doc = (
        "# class TestClass\n\n"
        "*Full name:* `test_module.TestClass`\n"
        "<a id=\"test_module-testclass\"></a>\n\n"
        "Bases\n"
        "-----\n"
        "BaseClass\n\n"
        "Members\n"
        "-------\n"
        "Name\n"
        "Type\n"
        "attr1\n"
        "int\n"
        "attr2\n"
        "float\n"
    )

    # Execute
    parser.class_api(root, name, bases, body)

    # Verify
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #99
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))
    body = [stmt1, stmt2]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[stmt1],
        orelse=[stmt2]
    )
    body = [if_node]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test Try statement
    try_node = Try(
        body=[stmt1],
        handlers=[],
        orelse=[stmt2],
        finalbody=[]
    )
    body = [try_node]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test nested If statements
    nested_if = If(
        test=Constant(value=True),
        body=[if_node],
        orelse=[]
    )
    body = [nested_if]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test Try with handlers
    handler = ExceptHandler(
        type=Name(id='Exception', ctx=Load()),
        name=None,
        body=[stmt1]
    )
    try_node = Try(
        body=[stmt2],
        handlers=[handler],
        orelse=[],
        finalbody=[]
    )
    body = [try_node]
    assert list(walk_body(body)) == [stmt2, stmt1]

    # Test mixed statements
    mixed_body = [
        stmt1,
        if_node,
        stmt2,
        try_node
    ]
    assert list(walk_body(mixed_body)) == [stmt1, stmt1, stmt2, stmt2, stmt1]


# LLM-generated content at query #100
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("test_module", {"test_module.name": "replacement"})
    name_node = Name("name", Load())
    result = resolver.visit_Name(name_node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Handle self_ty replacement
    resolver = Resolver("test_module", {}, self_ty="SelfType")
    name_node = Name("SelfType", Load())
    result = resolver.visit_Name(name_node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: No replacement when name not in alias
    resolver = Resolver("test_module", {"test_module.other": "other_replacement"})
    name_node = Name("name", Load())
    result = resolver.visit_Name(name_node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("test_module", {"test_module.T": "typing.TypeVar('T')"})
    name_node = Name("T", Load())
    result = resolver.visit_Name(name_node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("test_module", {"test_module.a": "b", "test_module.b": "c"})
    name_node = Name("a", Load())
    result = resolver.visit_Name(name_node)
    assert isinstance(result, Name)
    assert result.id == "c"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_func_ann():
    # Test case 1: Simple function with no self and no annotations
    parser = Parser()
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']

    # Test case 2: Function with self and annotations
    parser = Parser()
    args = [arg('self', Name('int', Load())), arg('x', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'str']

    # Test case 3: Class method with self and annotations
    parser = Parser()
    args = [arg('self', Name('int', Load())), arg('x', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'str']

    # Test case 4: Function with *args and **kwargs
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('*', None), arg('y', Name('str', Load())), arg('**', None)]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str', '']

    # Test case 5: Function with mixed annotations and no annotations
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('y', None), arg('z', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', 'Any', 'str']


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no body
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test class with bases
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("BaseClass").body[0].value]
    body = []

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

    # Test class with annotated members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name("attr1", Load()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        AnnAssign(
            target=Name("attr2", Load()),
            annotation=Name("str", Load()),
            value=None,
            simple=1
        )
    ]

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]

    # Test class with assigned members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name("attr1", Store())],
            value=Constant(1)
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant("hello")
        )
    ]

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]

    # Test enum class
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("enum.Enum").body[0].value]
    body = [
        AnnAssign(
            target=Name("VALUE1", Load()),
            annotation=None,
            value=Constant(1),
            simple=1
        ),
        AnnAssign(
            target=Name("VALUE2", Load()),
            annotation=None,
            value=Constant(2),
            simple=1
        )
    ]

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestEnum" in parser.doc[name]
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test class with deleted members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name("attr1", Load()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        Delete(
            targets=[Name("attr1", Del())]
        )
    ]

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #3
#--------------------------

```python
def test_Parser_globals():
    # Test case 1: AnnAssign with type annotation
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name("x", Load()),
        annotation=Name("int", Load()),
        value=Constant(10)
    )
    parser.globals(root, node)
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "10"
    assert "test_module.x" in parser.const
    assert parser.const["test_module.x"] == "int"

    # Test case 2: Assign with type comment
    parser = Parser()
    node = Assign(
        targets=[Name("y", Store())],
        value=Constant("hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert "test_module.y" in parser.alias
    assert parser.alias["test_module.y"] == "'hello'"
    assert "test_module.y" in parser.const
    assert parser.const["test_module.y"] == "str"

    # Test case 3: Assign without type comment (constant type inference)
    parser = Parser()
    node = Assign(
        targets=[Name("z", Store())],
        value=List(elts=[Constant(1), Constant(2), Constant(3)])
    )
    parser.globals(root, node)
    assert "test_module.z" in parser.alias
    assert parser.alias["test_module.z"] == "[1, 2, 3]"
    assert "test_module.z" in parser.const
    assert parser.const["test_module.z"] == "list[int]"

    # Test case 4: __all__ handling
    parser = Parser()
    node = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("func1"), Constant("func2")])
    )
    parser.globals(root, node)
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]

    # Test case 5: Non-constant assignment (should not be added to const)
    parser = Parser()
    node = Assign(
        targets=[Name("a", Store())],
        value=Name("b", Load())
    )
    parser.globals(root, node)
    assert "test_module.a" in parser.alias
    assert "test_module.a" not in parser.const

    # Test case 6: Multiple targets (should not be processed)
    parser = Parser()
    node = Assign(
        targets=[Name("a", Store()), Name("b", Store())],
        value=Constant(10)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #4
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
        orelse=[Expr(value=Call(func=Name(id='exit', ctx=Load()), args=[], keywords=[]))]
    )
    result = list(walk_body([if_node]))
    assert len(result) == 2
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Expr)

    # Test Try statement
    try_node = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
        handlers=[],
        orelse=[Expr(value=Call(func=Name(id='log', ctx=Load()), args=[], keywords=[]))],
        finalbody=[Expr(value=Call(func=Name(id='cleanup', ctx=Load()), args=[], keywords=[]))]
    )
    result = list(walk_body([try_node]))
    assert len(result) == 3
    assert all(isinstance(node, (Assign, Expr)) for node in result)

    # Test nested If statements
    nested_if = If(
        test=Constant(value=True),
        body=[
            If(
                test=Constant(value=False),
                body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))],
                orelse=[]
            )
        ],
        orelse=[]
    )
    result = list(walk_body([nested_if]))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test mixed statements
    mixed = [
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5)),
        If(
            test=Constant(value=True),
            body=[Expr(value=Call(func=Name(id='func', ctx=Load()), args=[], keywords=[]))],
            orelse=[]
        ),
        Try(
            body=[],
            handlers=[],
            orelse=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))],
            finalbody=[]
        )
    ]
    result = list(walk_body(mixed))
    assert len(result) == 3
    assert all(isinstance(node, (Assign, Expr)) for node in result)


# LLM-generated content at query #5
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("test_module", {"test_module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Return node if name is not in alias
    resolver = Resolver("test_module", {})
    node = Name("unknown_name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "unknown_name"

    # Test case 3: Replace self_ty with "Self"
    resolver = Resolver("test_module", {}, self_ty="self")
    node = Name("self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 4: Return node if it's a TypeVar
    resolver = Resolver("test_module", {"test_module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"


# LLM-generated content at query #6
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("root", {}, "Self")
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 123

    # Test with string constant that is not a name
    resolver = Resolver("root", {}, "Self")
    node = Constant("not_a_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "not_a_name"

    # Test with string constant that is a name
    resolver = Resolver("root", {}, "Self")
    node = Constant("some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "some_name"

    # Test with string constant that is a name with alias
    resolver = Resolver("root", {"root.some_name": "alias_name"}, "Self")
    node = Constant("some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias_name"

    # Test with string constant that is a TypeVar call
    resolver = Resolver("root", {"root.TypeVar": "typing.TypeVar"}, "Self")
    node = Constant("TypeVar('T')")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "TypeVar('T')"


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_parse():
    # Test basic module parsing
    parser = Parser()
    script = """
    '''Module docstring.'''
    x = 1
    y: int = 2
    def foo(a: int) -> str:
        '''Function docstring.'''
        return str(a)
    class Bar:
        '''Class docstring.'''
        z: float = 3.0
    """
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    assert "test_module.Bar" in parser.doc
    assert "test_module.Bar.z" in parser.const

    # Test import handling
    script_with_imports = """
    from typing import List
    import os
    x: List[int] = [1, 2, 3]
    """
    parser.parse("test_imports", script_with_imports)
    assert "typing.List" in parser.alias.values()
    assert "os" in parser.alias.values()

    # Test type alias
    script_with_alias = """
    MyType = dict[str, int]
    x: MyType = {"a": 1}
    """
    parser.parse("test_alias", script_with_alias)
    assert "MyType" in parser.alias
    assert parser.alias["MyType"] == "dict[str, int]"

    # Test constants
    script_with_const = """
    CONSTANT = 42
    ANOTHER_CONST = "hello"
    """
    parser.parse("test_const", script_with_const)
    assert "CONSTANT" in parser.const
    assert "ANOTHER_CONST" in parser.const
    assert parser.const["CONSTANT"] == "int"
    assert parser.const["ANOTHER_CONST"] == "str"

    # Test __all__ handling
    script_with_all = """
    __all__ = ["public_func", "PublicClass"]
    def public_func():
        pass
    def _private_func():
        pass
    class PublicClass:
        pass
    class _PrivateClass:
        pass
    """
    parser.parse("test_all", script_with_all)
    assert "public_func" in parser.imp["test_all"]
    assert "PublicClass" in parser.imp["test_all"]
    assert "_private_func" not in parser.imp["test_all"]
    assert "_PrivateClass" not in parser.imp["test_all"]

    # Test decorator handling
    script_with_decorator = """
    def decorator(func):
        return func
    @decorator
    def decorated_func():
        pass
    """
    parser.parse("test_decorator", script_with_decorator)
    assert "decorated_func" in parser.doc
    doc = parser.doc["test_decorator.decorated_func"]
    assert "@decorator" in doc

    # Test class inheritance
    script_with_inheritance = """
    class Base:
        pass
    class Derived(Base):
        pass
    """
    parser.parse("test_inheritance", script_with_inheritance)
    doc = parser.doc["test_inheritance.Derived"]
    assert "Base" in doc

    # Test enum handling
    script_with_enum = """
    from enum import Enum
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    """
    parser.parse("test_enum", script_with_enum)
    doc = parser.doc["test_enum.Color"]
    assert "Enums" in doc
    assert "RED" in doc
    assert "GREEN" in doc
    assert "BLUE" in doc

    # Test function with annotations
    script_with_annotations = """
    def complex_func(
        a: int,
        b: str = "default",
        *args: float,
        c: bool = True,
        **kwargs: dict
    ) -> list:
        pass
    """
    parser.parse("test_annotations", script_with_annotations)
    doc = parser.doc["test_annotations.complex_func"]
    assert "a" in doc
    assert "b" in doc
    assert "args" in doc
    assert "c" in doc
    assert "kwargs" in doc
    assert "return" in doc

    # Test async function
    script_with_async = """
    async def async_func():
        pass
    """
    parser.parse("test_async", script_with_async)
    doc = parser.doc["test_async.async_func"]
    assert "async" in doc

    # Test nested class
    script_with_nested = """
    class Outer:
        class Inner:
            pass
    """
    parser.parse("test_nested", script_with_nested)
    assert "test_nested.Outer.Inner" in parser.doc

    # Test property deletion
    script_with_deletion = """
    class Test:
        x: int = 1
        del x
    """
    parser.parse("test_deletion", script_with_deletion)
    doc = parser.doc["test_deletion.Test"]
    assert "x" not in doc


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_globals():
    # Test case 1: AnnAssign with type annotation
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="test_var", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == root

    # Test case 2: Assign with type comment
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == root

    # Test case 3: Assign without type comment (constant type inference)
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == root

    # Test case 4: __all__ handling
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[
            Constant(value="public_func"),
            Constant(value="public_class")
        ])
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func", "test_module.public_class"}

    # Test case 5: Non-Name target (should not process)
    parser = Parser()
    node = Assign(
        targets=[Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())])],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias

    # Test case 6: Multiple targets (should not process)
    parser = Parser()
    node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias

    # Test case 7: Non-constant value (should not add to const)
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Name(id="some_func", ctx=Load())
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "some_func"
    assert parser.const.get("test_module.test_var", "NOT_FOUND") == "NOT_FOUND"


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    parser.alias = {}
    parser.const = {}

    # Test AnnAssign with type annotation
    node_ann = AnnAssign(
        target=Name("TEST_CONST", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node_ann)
    assert parser.alias["test_module.TEST_CONST"] == "42"
    assert parser.const["test_module.TEST_CONST"] == "int"
    assert parser.root["test_module.TEST_CONST"] == root

    # Test Assign with type comment
    node_assign = Assign(
        targets=[Name("ANOTHER_CONST", Store())],
        value=Constant(3.14),
        type_comment="float"
    )
    parser.globals(root, node_assign)
    assert parser.alias["test_module.ANOTHER_CONST"] == "3.14"
    assert parser.const["test_module.ANOTHER_CONST"] == "float"
    assert parser.root["test_module.ANOTHER_CONST"] == root

    # Test Assign without type comment (should use const_type)
    node_assign_no_comment = Assign(
        targets=[Name("INFERRED_CONST", Store())],
        value=List(elts=[Constant(1), Constant(2)], ctx=Load())
    )
    parser.globals(root, node_assign_no_comment)
    assert parser.alias["test_module.INFERRED_CONST"] == "[1, 2]"
    assert parser.const["test_module.INFERRED_CONST"] == "list[int]"

    # Test __all__ handling
    node_all = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[
            Constant("public_func"),
            Constant("PublicClass")
        ], ctx=Load())
    )
    parser.globals(root, node_all)
    assert "test_module.public_func" in parser.imp[root]
    assert "test_module.PublicClass" in parser.imp[root]

    # Test non-constant assignment (should not add to const)
    node_non_const = Assign(
        targets=[Name("non_const", Store())],
        value=Constant(100)
    )
    parser.globals(root, node_non_const)
    assert "test_module.non_const" in parser.alias
    assert "test_module.non_const" not in parser.const

    # Test complex assignment (should be ignored)
    node_complex = Assign(
        targets=[Tuple(elts=[Name("a", Store()), Name("b", Store())], ctx=Store())],
        value=Constant(1)
    )
    parser.globals(root, node_complex)
    assert "test_module.a" not in parser.alias


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_imports():
    # Test case 1: Import with no alias
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os")])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"

    # Test case 2: Import with alias
    parser = Parser()
    node = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, node)
    assert parser.alias["test_module.np"] == "numpy"

    # Test case 3: ImportFrom with no level and no alias
    parser = Parser()
    node = ImportFrom(module="sys", names=[alias(name="path")], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.path"] == "sys.path"

    # Test case 4: ImportFrom with level and alias
    parser = Parser()
    node = ImportFrom(module="os", names=[alias(name="path", asname="osp")], level=1)
    parser.imports(root, node)
    assert parser.alias["test_module.osp"] == "test_module.os.path"


# LLM-generated content at query #11
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()

    # Test AnnAssign with type annotation
    node_ann = AnnAssign(
        target=Name("x", Load()),
        annotation=Name("int", Load()),
        value=Constant(42)
    )
    parser.globals("module", node_ann)
    assert parser.alias["module.x"] == "42"
    assert parser.const["module.x"] == "int"

    # Test Assign with type comment
    node_assign = Assign(
        targets=[Name("y", Store())],
        value=Constant(3.14),
        type_comment="float"
    )
    parser.globals("module", node_assign)
    assert parser.alias["module.y"] == "3.14"
    assert parser.const["module.y"] == "float"

    # Test Assign without type comment (constant type inference)
    node_assign_no_comment = Assign(
        targets=[Name("z", Store())],
        value=List(elts=[Constant(1), Constant(2)])
    )
    parser.globals("module", node_assign_no_comment)
    assert parser.alias["module.z"] == "[1, 2]"
    assert parser.const["module.z"] == "list[int]"

    # Test __all__ handling
    node_all = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("public_func")])
    )
    parser.globals("module", node_all)
    assert parser.imp["module"] == {"module.public_func"}

    # Test non-constant assignment (should not add to const)
    node_non_const = Assign(
        targets=[Name("a", Store())],
        value=Name("b", Load())
    )
    parser.globals("module", node_non_const)
    assert "module.a" not in parser.const

    # Test multiple targets (should not process)
    node_multi = Assign(
        targets=[Name("c", Store()), Name("d", Store())],
        value=Constant(100)
    )
    parser.globals("module", node_multi)
    assert "module.c" not in parser.alias
    assert "module.d" not in parser.alias


# LLM-generated content at query #12
#--------------------------

```python
def test_doctest():
    # Test with no doctest
    assert doctest("This is a simple docstring.") == "This is a simple docstring."

    # Test with single line doctest
    input_doc = ">>> print('Hello, World!')"
    expected = "```python\n>>> print('Hello, World!')\n```"
    assert doctest(input_doc) == expected

    # Test with multi-line doctest
    input_doc = """This is a docstring with a doctest:

>>> x = 5
>>> y = 10
>>> x + y
15

And some more text."""
    expected = """This is a docstring with a doctest:



# LLM-generated content at query #13
#--------------------------

```python
def test_Parser_globals():
    # Test case 1: AnnAssign with type annotation
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name("x", Load()),
        annotation=Name("int", Load()),
        value=Constant(5)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.x"] == "5"
    assert parser.const["test_module.x"] == "int"
    assert parser.root["test_module.x"] == root

    # Test case 2: Assign with type comment
    parser = Parser()
    node = Assign(
        targets=[Name("y", Store())],
        value=Constant("hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.y"] == "'hello'"
    assert parser.const["test_module.y"] == "str"
    assert parser.root["test_module.y"] == root

    # Test case 3: Assign without type comment (constant type inference)
    parser = Parser()
    node = Assign(
        targets=[Name("z", Store())],
        value=List(elts=[Constant(1), Constant(2)])
    )
    parser.globals(root, node)
    assert parser.alias["test_module.z"] == "[1, 2]"
    assert parser.const["test_module.z"] == "list[int]"
    assert parser.root["test_module.z"] == root

    # Test case 4: __all__ handling
    parser = Parser()
    node = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("func1"), Constant("func2")])
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.func1", "test_module.func2"}

    # Test case 5: Non-uppercase constant (should not add to const)
    parser = Parser()
    node = Assign(
        targets=[Name("non_const", Store())],
        value=Constant(42)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.non_const"] == "42"
    assert "test_module.non_const" not in parser.const

    # Test case 6: Multiple targets (should not process)
    parser = Parser()
    node = Assign(
        targets=[Name("a", Store()), Name("b", Store())],
        value=Constant(10)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"typing.Dict": "Dict"})
    node = Subscript(Name("Dict", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"


# LLM-generated content at query #15
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 123

    # Test with string constant that is not a valid expression
    resolver = Resolver("test", {})
    node = Constant("invalid expression")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "invalid expression"

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "new_name"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "new_name"

    # Test with string constant that is a valid attribute
    resolver = Resolver("test", {"test.obj.attr": "new_attr"})
    node = Constant("obj.attr")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Attribute)
    assert result.attr == "new_attr"


# LLM-generated content at query #16
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("root", {"root.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Return node if name is not in alias
    resolver = Resolver("root", {"root.other": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 3: Replace self_ty with "Self"
    resolver = Resolver("root", {}, self_ty="self")
    node = Name("self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 4: Support TypeVar
    resolver = Resolver("root", {"root.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("root", {"root.name": "root.other", "root.other": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"


# LLM-generated content at query #17
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-typing subscript (should return unchanged)
    resolver = Resolver("test", {})
    node = Subscript(Name("MyClass", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "MyClass"


# LLM-generated content at query #18
#--------------------------

```python
def test_Parser_is_public():
    # Test case 1: Public name (starts with non-underscore)
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.root = {'root.public_name': 'root'}
    parser.doc = {'root.public_name': ''}
    assert parser.is_public('root.public_name') == True

    # Test case 2: Private name (starts with underscore)
    parser = Parser()
    parser.imp = {'root': {'_private_name'}}
    parser.root = {'root._private_name': 'root'}
    parser.doc = {'root._private_name': ''}
    assert parser.is_public('root._private_name') == False

    # Test case 3: Name in __all__
    parser = Parser()
    parser.imp = {'root': {'name_in_all'}}
    parser.root = {'root.name_in_all': 'root'}
    parser.doc = {'root.name_in_all': ''}
    assert parser.is_public('root.name_in_all') == True

    # Test case 4: Child of name in __all__
    parser = Parser()
    parser.imp = {'root': {'parent_name'}}
    parser.root = {'root.parent_name.child': 'root'}
    parser.doc = {'root.parent_name.child': ''}
    assert parser.is_public('root.parent_name.child') == True

    # Test case 5: Name not in __all__ and not public style
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root._private_name': 'root'}
    parser.doc = {'root._private_name': ''}
    assert parser.is_public('root._private_name') == False

    # Test case 6: Name with parent in __all__
    parser = Parser()
    parser.imp = {'root': {'parent_name'}}
    parser.root = {'root.parent_name': 'root'}
    parser.root = {'root.parent_name.child': 'root'}
    parser.doc = {'root.parent_name.child': ''}
    assert parser.is_public('root.parent_name.child') == True

    # Test case 7: Name with grandparent in __all__
    parser = Parser()
    parser.imp = {'root': {'grandparent_name'}}
    parser.root = {'root.grandparent_name': 'root'}
    parser.root = {'root.grandparent_name.parent_name': 'root'}
    parser.root = {'root.grandparent_name.parent_name.child': 'root'}
    parser.doc = {'root.grandparent_name.parent_name.child': ''}
    assert parser.is_public('root.grandparent_name.parent_name.child') == True


# LLM-generated content at query #19
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("module", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("module", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value == "str"

    # Test Optional conversion
    resolver = Resolver("module", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("module", {"typing.Dict": "Dict"})
    node = Subscript(Name("Dict", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"

    # Test non-Tuple slice in Union
    resolver = Resolver("module", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Constant)
    assert result.value == "int"


# LLM-generated content at query #20
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Handle self_ty
    resolver = Resolver("module", {}, "SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("module", {"module.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test case 5: Handle nested module name
    resolver = Resolver("module.submodule", {"module.submodule.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"


# LLM-generated content at query #21
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no self and no class method
    args = [
        arg("x", Name("int", Load())),
        arg("y", Name("str", Load())),
        arg("return", Name("bool", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "str", "bool"]

    # Test case 2: Function with self and no class method
    args = [
        arg("self", Name("TestClass", Load())),
        arg("x", Name("int", Load())),
        arg("return", Name("str", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "int", "str"]

    # Test case 3: Function with self and class method
    args = [
        arg("cls", Name("TestClass", Load())),
        arg("x", Name("int", Load())),
        arg("return", Name("str", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int", "str"]

    # Test case 4: Function with *args and **kwargs
    args = [
        arg("x", Name("int", Load())),
        arg("*", None),
        arg("y", Name("str", Load())),
        arg("**kwargs", Name("Any", Load())),
        arg("return", Name("bool", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "Any", "bool"]

    # Test case 5: Function with no annotations
    args = [
        arg("x", None),
        arg("y", None),
        arg("return", None)
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any", "Any"]


# LLM-generated content at query #22
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 substitution
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union substitution
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value == "str"

    # Test Optional substitution
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.value == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-subscriptable case
    resolver = Resolver("test", {})
    node = Subscript(Name("Unknown", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #23
#--------------------------

```python
def test_Parser_api():
    # Setup
    parser = Parser()
    root = "test_module"
    parser.doc[root] = ""
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    parser.alias = {}
    parser.const = {}

    # Test function definition
    func_node = FunctionDef(
        name="test_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, func_node)
    assert "test_func" in parser.doc
    assert "test_module.test_func" in parser.doc
    assert "test_func()" in parser.doc["test_module.test_func"]

    # Test async function definition
    async_func_node = AsyncFunctionDef(
        name="test_async_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, async_func_node)
    assert "test_async_func" in parser.doc
    assert "test_module.test_async_func" in parser.doc
    assert "async test_async_func()" in parser.doc["test_module.test_async_func"]

    # Test class definition
    class_node = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, class_node)
    assert "TestClass" in parser.doc
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]

    # Test nested class definition
    nested_class_node = ClassDef(
        name="NestedClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    class_node.body.append(nested_class_node)
    parser.api(root, class_node)
    assert "NestedClass" in parser.doc
    assert "test_module.TestClass.NestedClass" in parser.doc
    assert "class NestedClass" in parser.doc["test_module.TestClass.NestedClass"]

    # Test function with decorators
    decorated_func_node = FunctionDef(
        name="decorated_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="decorator", ctx=Load())]
    )
    parser.api(root, decorated_func_node)
    assert "decorated_func" in parser.doc
    assert "@decorator" in parser.doc["test_module.decorated_func"]

    # Test class with bases
    class_with_bases_node = ClassDef(
        name="ClassWithBases",
        bases=[Name(id="BaseClass", ctx=Load())],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, class_with_bases_node)
    assert "ClassWithBases" in parser.doc
    assert "Bases" in parser.doc["test_module.ClassWithBases"]
    assert "BaseClass" in parser.doc["test_module.ClassWithBases"]


# LLM-generated content at query #24
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname="system")])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    import_from_node = ImportFrom(module="collections", names=[alias(name="defaultdict", asname=None)], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"

    # Test ImportFrom node with level 1
    import_from_node_level = ImportFrom(module="os.path", names=[alias(name="join", asname="path_join")], level=1)
    parser.imports(root, import_from_node_level)
    assert parser.alias["test_module.path_join"] == "test_module.os.path.join"

    # Test ImportFrom node with level > 1
    import_from_node_level2 = ImportFrom(module="pkg.subpkg", names=[alias(name="func", asname=None)], level=2)
    parser.imports(root, import_from_node_level2)
    assert parser.alias["test_module.func"] == "test_module.pkg.subpkg.func"


# LLM-generated content at query #25
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test removing 'typing.*' prefix
    resolver = Resolver("test_module", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test non-typing attribute remains unchanged
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert result.value.id == "other_module"

    # Test non-Name value remains unchanged
    node = Attribute(Constant(42), "value", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "value"


# LLM-generated content at query #26
#--------------------------

```python
def test_Parser_compile():
    # Test basic module compilation
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
CONSTANT = 42
def function():
    \"\"\"Function docstring.\"\"\"
    pass
class Class:
    \"\"\"Class docstring.\"\"\"
    pass
""")
    result = parser.compile()
    assert "Test module docstring" in result
    assert "CONSTANT" in result
    assert "function()" in result
    assert "Function docstring" in result
    assert "class Class" in result
    assert "Class docstring" in result

    # Test with table of contents
    parser_toc = Parser.new(link=True, level=1, toc=True)
    parser_toc.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
CONSTANT = 42
def function():
    \"\"\"Function docstring.\"\"\"
    pass
class Class:
    \"\"\"Class docstring.\"\"\"
    pass
""")
    result_toc = parser_toc.compile()
    assert "Table of contents" in result_toc
    assert "+ [test_module](#test_module)" in result_toc
    assert "+ [test_module.function](#test_module-function)" in result_toc
    assert "+ [test_module.Class](#test_module-class)" in result_toc

    # Test with imports and aliases
    parser_alias = Parser.new(link=False, level=1, toc=False)
    parser_alias.parse("test_module", """
from typing import List
MyList = List[int]
def function() -> MyList:
    \"\"\"Function with alias.\"\"\"
    pass
""")
    result_alias = parser_alias.compile()
    assert "MyList" in result_alias
    assert "function()" in result_alias
    assert "Function with alias" in result_alias

    # Test with class members
    parser_class = Parser.new(link=False, level=1, toc=False)
    parser_class.parse("test_module", """
class TestClass:
    \"\"\"Test class.\"\"\"
    member: int
    def method(self) -> None:
        \"\"\"Test method.\"\"\"
        pass
""")
    result_class = parser_class.compile()
    assert "class TestClass" in result_class
    assert "Test class" in result_class
    assert "member" in result_class
    assert "method()" in result_class
    assert "Test method" in result_class


# LLM-generated content at query #27
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test_module", {"typing.List": "list"}, "Self")
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional conversion
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    node = Subscript(Name("NonPEP585", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "NonPEP585"

    # Test non-Tuple slice
    node = Subscript(Name("Union", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #28
#--------------------------

```python
def test_Parser_parse():
    # Test basic module parsing
    parser = Parser()
    script = """
    '''Module docstring.'''
    x = 1
    y: int = 2
    def foo(a: int) -> str:
        '''Function docstring.'''
        return str(a)
    class Bar:
        '''Class docstring.'''
        z: float = 3.0
    """
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    assert "test_module.Bar" in parser.doc
    assert "test_module.Bar.z" in parser.const

    # Test with imports
    parser = Parser()
    script = """
    from typing import List
    import os
    x: List[int] = [1, 2, 3]
    """
    parser.parse("test_imports", script)
    assert "typing.List" in parser.alias.values()
    assert "os" in parser.alias.values()

    # Test with class and method
    parser = Parser()
    script = """
    class MyClass:
        def method(self, x: int) -> str:
            return str(x)
    """
    parser.parse("test_class", script)
    assert "test_class.MyClass" in parser.doc
    assert "test_class.MyClass.method" in parser.doc

    # Test with decorator
    parser = Parser()
    script = """
    def decorator(func):
        return func
    @decorator
    def decorated_func():
        pass
    """
    parser.parse("test_decorator", script)
    assert "test_decorator.decorated_func" in parser.doc
    assert "@decorator" in parser.doc["test_decorator.decorated_func"]

    # Test with enum
    parser = Parser()
    script = """
    from enum import Enum
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    """
    parser.parse("test_enum", script)
    assert "test_enum.Color" in parser.doc
    assert "Enums" in parser.doc["test_enum.Color"]


# LLM-generated content at query #29
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no self and no annotations
    args = [arg("x", None), arg("y", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any"]

    # Test case 2: Function with self and annotations
    args = [arg("self", Name("TestClass", Load())), arg("x", Name("int", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "int"]

    # Test case 3: Class method with self and annotations
    args = [arg("cls", Name("TestClass", Load())), arg("x", Name("int", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int"]

    # Test case 4: Function with *args and **kwargs
    args = [
        arg("x", Name("int", Load())),
        arg("*", None),
        arg("y", Name("str", Load())),
        arg("**kwargs", None)
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "Any"]

    # Test case 5: Function with mixed annotations and defaults
    args = [
        arg("x", Name("int", Load())),
        arg("y", None),
        arg("z", Name("str", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "Any", "str"]


# LLM-generated content at query #30
#--------------------------

```python
def test_Parser_func_ann():
    # Test case 1: Simple function with no self and no class method
    parser = Parser()
    args = [
        arg('x', Name('int', Load())),
        arg('y', Name('str', Load())),
        arg('return', Name('bool', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', 'str', 'bool']

    # Test case 2: Function with self and not a class method
    parser = Parser()
    args = [
        arg('self', Name('Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str']

    # Test case 3: Function with self and is a class method
    parser = Parser()
    args = [
        arg('cls', Name('Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int', 'str']

    # Test case 4: Function with *args and **kwargs
    parser = Parser()
    args = [
        arg('x', Name('int', Load())),
        arg('*', None),
        arg('y', Name('str', Load())),
        arg('**', None),
        arg('return', Name('bool', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str', '', 'bool']

    # Test case 5: Function with no annotations
    parser = Parser()
    args = [
        arg('x', None),
        arg('y', None),
        arg('return', None)
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any', 'Any']


# LLM-generated content at query #31
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser.new(link=False, level=1, toc=False)

    # Test simple function with no arguments
    root = "test_module"
    name = "test_module.simple_func"
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "return" in parser.doc[name]

    # Test function with positional arguments
    args = arguments(
        posonlyargs=[arg("x", None), arg("y", None)],
        args=[arg("z", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "z" in parser.doc[name]

    # Test function with keyword arguments
    args = arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[arg("a", None), arg("b", None)],
        kw_defaults=[],
        defaults=[]
    )
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "a" in parser.doc[name]
    assert "b" in parser.doc[name]

    # Test function with defaults
    args = arguments(
        posonlyargs=[],
        args=[arg("x", None), arg("y", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[Constant(1), Constant(2)]
    )
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "1" in parser.doc[name]
    assert "2" in parser.doc[name]

    # Test function with varargs and kwargs
    args = arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=arg("args", None),
        kwarg=arg("kwargs", None)
    )
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "*args" in parser.doc[name]
    assert "**kwargs" in parser.doc[name]

    # Test function with return annotation
    returns = Name("int", Load())
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "int" in parser.doc[name]

    # Test function with self and cls_method
    args = arguments(
        posonlyargs=[],
        args=[arg("self", None), arg("x", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    parser.func_api(root, name, args, returns, has_self=True, cls_method=True)
    assert "type[Self]" in parser.doc[name]


# LLM-generated content at query #32
#--------------------------

```python
def test_Parser_func_api():
    # Test case 1: Simple function with no arguments and no return type
    parser = Parser()
    root = "test_module"
    name = "test_module.simple_func"
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"## simple_func()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| | |\n| --- | --- |\n| return | Any |\n"
    assert parser.doc[name] == expected_output

    # Test case 2: Function with positional arguments and return type
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_args"
    arg1 = arg(arg="arg1", annotation=Name(id="int", ctx=Load()))
    arg2 = arg(arg="arg2", annotation=Name(id="str", ctx=Load()))
    args = arguments(posonlyargs=[], args=[arg1, arg2], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = Name(id="bool", ctx=Load())
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"## func_with_args()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| arg1 | arg2 | return |\n| --- | --- | --- |\n| int | str | bool |\n"
    assert parser.doc[name] == expected_output

    # Test case 3: Function with default arguments
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_defaults"
    arg1 = arg(arg="arg1", annotation=Name(id="int", ctx=Load()))
    arg2 = arg(arg="arg2", annotation=Name(id="str", ctx=Load()))
    args = arguments(posonlyargs=[], args=[arg1, arg2], kwonlyargs=[], defaults=[Constant(value="default")], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"## func_with_defaults()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| arg1 | arg2 | return |\n| --- | --- | --- |\n| int | str | Any |\n|  | 'default' |  |\n"
    assert parser.doc[name] == expected_output

    # Test case 4: Function with variable arguments
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_varargs"
    arg1 = arg(arg="arg1", annotation=Name(id="int", ctx=Load()))
    vararg = arg(arg="args", annotation=None)
    args = arguments(posonlyargs=[], args=[arg1], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=vararg)
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"## func_with_varargs()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| arg1 | *args | return |\n| --- | --- | --- |\n| int |  | Any |\n"
    assert parser.doc[name] == expected_output

    # Test case 5: Function with keyword-only arguments
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_kwonly"
    arg1 = arg(arg="arg1", annotation=Name(id="int", ctx=Load()))
    kwarg1 = arg(arg="kwarg1", annotation=Name(id="str", ctx=Load()))
    kwarg2 = arg(arg="kwarg2", annotation=Name(id="bool", ctx=Load()))
    args = arguments(posonlyargs=[], args=[arg1], kwonlyargs=[kwarg1, kwarg2], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = f"## func_with_kwonly()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| arg1 | * | kwarg1 | kwarg2 | return |\n| --- | --- | --- | --- | --- |\n| int |  | str | bool | Any |\n"
    assert parser.doc[name] == expected_output

    # Test case 6: Function with self parameter (method)
    parser = Parser()
    root = "test_module"
    name = "test_module.Class.method"
    self_arg = arg(arg="self", annotation=None)
    arg1 = arg(arg="arg1", annotation=Name(id="int", ctx=Load()))
    args = arguments(posonlyargs=[], args=[self_arg, arg1], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api(root, name, args, returns, has_self=True, cls_method=False)
    expected_output = f"## method()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| arg1 | return |\n| --- | --- |\n| int | Any |\n"
    assert parser.doc[name] == expected_output

    # Test case 7: Class method
    parser = Parser()
    root = "test_module"
    name = "test_module.Class.class_method"
    cls_arg = arg(arg="cls", annotation=None)
    arg1 = arg(arg="arg1", annotation=Name(id="int", ctx=Load()))
    args = arguments(posonlyargs=[], args=[cls_arg, arg1], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api(root, name, args, returns, has_self=True, cls_method=True)
    expected_output = f"## class_method()\n\n*Full name:* `{name}`\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n| arg1 | return |\n| --- | --- |\n| int | Any |\n"
    assert parser.doc[name] == expected_output


# LLM-generated content at query #33
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Handle self type
    resolver = Resolver("module", {}, "SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if no replacement found
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("module", {"module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("module", {"module.A": "module.B", "module.B": "final"})
    node = Name("A", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"


# LLM-generated content at query #34
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Replace self_ty with "Self"
    resolver = Resolver("module", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("module", {"module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("module", {"module.A": "module.B", "module.B": "replacement"})
    node = Name("A", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"


# LLM-generated content at query #35
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no self and no class method
    args = [arg("x", None), arg("y", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any"]

    # Test case 2: Function with self and no class method
    args = [arg("self", None), arg("x", None)]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "Any"]

    # Test case 3: Function with self and class method
    args = [arg("cls", None), arg("x", None)]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "Any"]

    # Test case 4: Function with annotations
    args = [arg("x", Name("int", Load())), arg("y", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "str"]

    # Test case 5: Function with mixed annotations and no annotations
    args = [arg("x", Name("int", Load())), arg("y", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "Any"]

    # Test case 6: Function with *args and **kwargs
    args = [arg("x", None), arg("*", None), arg("y", None), arg("**kwargs", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "", "Any", "Any"]


# LLM-generated content at query #36
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Remove 'typing.*' prefix
    resolver = Resolver("test_module", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test case 2: Non-typing attribute remains unchanged
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert result.value.id == "other_module"

    # Test case 3: Complex attribute chain
    node = Attribute(Attribute(Name("typing", Load()), "List", Load()), "append", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "List"
    assert result.attr == "append"

    # Test case 4: Non-Name value in Attribute
    node = Attribute(Constant(42), "some_attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Constant)
    assert result.attr == "some_attr"


# LLM-generated content at query #37
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"typing.Dict": "Dict"})
    node = Subscript(Name("Dict", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"

    # Test non-Subscript node
    resolver = Resolver("test", {})
    node = Name("test", Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Name)
    assert result.id == "test"


# LLM-generated content at query #38
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test removing 'typing.*' prefix
    resolver = Resolver("test_module", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test non-typing attribute remains unchanged
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert result.value.id == "other_module"

    # Test nested attribute access
    node = Attribute(Name("typing", Load()), "Union", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Union"


# LLM-generated content at query #39
#--------------------------

```python
def test_Parser_globals():
    # Test case 1: Type alias with annotation
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name("MyType", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.MyType"] == "42"
    assert parser.const["test_module.MyType"] == "int"

    # Test case 2: Constant assignment without annotation
    parser = Parser()
    node = Assign(
        targets=[Name("CONST_VALUE", Load())],
        value=Constant(3.14),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.alias["test_module.CONST_VALUE"] == "3.14"
    assert parser.const["test_module.CONST_VALUE"] == "float"

    # Test case 3: __all__ processing
    parser = Parser()
    node = Assign(
        targets=[Name("__all__", Load())],
        value=List(elts=[
            Constant("public_func"),
            Constant("PublicClass")
        ]),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.imp["test_module"] == {"test_module.public_func", "test_module.PublicClass"}

    # Test case 4: Non-constant assignment (should not be added to const)
    parser = Parser()
    node = Assign(
        targets=[Name("non_const", Load())],
        value=Name("some_value", Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert "test_module.non_const" in parser.alias
    assert "test_module.non_const" not in parser.const

    # Test case 5: Invalid node type (should return early)
    parser = Parser()
    node = Pass()  # Invalid node type for globals processing
    parser.globals(root, node)
    assert not parser.alias
    assert not parser.const


# LLM-generated content at query #40
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test AnnAssign with type annotation
    node = AnnAssign(
        target=Name(id="VAR1", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.VAR1"] == "42"
    assert parser.const["test_module.VAR1"] == "int"
    assert parser.root["test_module.VAR1"] == root

    # Test Assign with type comment
    node = Assign(
        targets=[Name(id="VAR2", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.VAR2"] == "'hello'"
    assert parser.const["test_module.VAR2"] == "str"
    assert parser.root["test_module.VAR2"] == root

    # Test Assign without type comment (constant inference)
    node = Assign(
        targets=[Name(id="VAR3", ctx=Store())],
        value=List(elts=[Constant(value=1), Constant(value=2)])
    )
    parser.globals(root, node)
    assert parser.alias["test_module.VAR3"] == "[1, 2]"
    assert parser.const["test_module.VAR3"] == "list[int]"
    assert parser.root["test_module.VAR3"] == root

    # Test __all__ handling
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[
            Constant(value="public_func"),
            Constant(value="PublicClass")
        ])
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func", "test_module.PublicClass"}

    # Test non-Name target (should not process)
    node = Assign(
        targets=[Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())])],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias

    # Test non-constant value in __all__ (should not process)
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Name(id="some_var", ctx=Load())
    )
    parser.globals(root, node)
    assert root not in parser.imp


# LLM-generated content at query #41
#--------------------------

```python
def test_Parser_api():
    # Test case 1: Function definition
    parser = Parser()
    root = "test_module"
    node = FunctionDef(
        name="test_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="arg1", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert "test_func()" in parser.doc["test_module.test_func"]
    assert "Full name: `test_module.test_func`" in parser.doc["test_module.test_func"]

    # Test case 2: Class definition
    parser = Parser()
    node = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]
    assert "Full name: `test_module.TestClass`" in parser.doc["test_module.TestClass"]

    # Test case 3: Async function definition
    parser = Parser()
    node = AsyncFunctionDef(
        name="async_test_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.async_test_func" in parser.doc
    assert "async async_test_func()" in parser.doc["test_module.async_test_func"]
    assert "Full name: `test_module.async_test_func`" in parser.doc["test_module.async_test_func"]

    # Test case 4: Function with decorator
    parser = Parser()
    node = FunctionDef(
        name="decorated_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="decorator", ctx=Load())]
    )
    parser.api(root, node)
    assert "test_module.decorated_func" in parser.doc
    assert "@decorator" in parser.doc["test_module.decorated_func"]

    # Test case 5: Class with base
    parser = Parser()
    node = ClassDef(
        name="DerivedClass",
        bases=[Name(id="BaseClass", ctx=Load())],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.DerivedClass" in parser.doc
    assert "Bases" in parser.doc["test_module.DerivedClass"]
    assert "BaseClass" in parser.doc["test_module.DerivedClass"]


# LLM-generated content at query #42
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    args = arguments(
        posonlyargs=[arg("a", None), arg("b", None)],
        args=[arg("c", None), arg("d", None)],
        defaults=[Constant(1), Constant(2)],
        vararg=arg("args", None),
        kwonlyargs=[arg("e", None), arg("f", None)],
        kw_defaults=[Constant(3), Constant(4)],
        kwarg=arg("kwargs", None),
    )
    returns = None

    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)

    expected_doc = (
        f"## test_func()\n\n"
        f"*Full name:* `{name}`\n"
        f"<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
        f"| | | |\n"
        f"| --- | --- | --- |\n"
        f"| a | {ANY} | |\n"
        f"| b | {ANY} | |\n"
        f"| / | | |\n"
        f"| c | {ANY} | 1 |\n"
        f"| d | {ANY} | 2 |\n"
        f"| *args | {ANY} | |\n"
        f"| * | | |\n"
        f"| e | {ANY} | 3 |\n"
        f"| f | {ANY} | 4 |\n"
        f"| **kwargs | {ANY} | |\n"
        f"| return | {ANY} | |\n"
    )

    assert parser.doc[name] == expected_doc


# LLM-generated content at query #43
#--------------------------

```python
def test_Parser_compile():
    # Test basic module documentation
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
x = 1
y = "hello"
def func(a: int) -> str:
    \"\"\"Function docstring.\"\"\"
    return str(a)
""")
    result = p.compile()
    assert "Module `test_module`" in result
    assert "func()" in result
    assert "Full name: `test_module.func`" in result
    assert "a: int" in result
    assert "return: str" in result
    assert "Function docstring." in result

    # Test with TOC
    p = Parser.new(link=True, level=1, toc=True)
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
def func(a: int) -> str:
    \"\"\"Function docstring.\"\"\"
    return str(a)
""")
    result = p.compile()
    assert "**Table of contents:**" in result
    assert "+ [test_module](#test_module)" in result
    assert "+ [test_module.func](#test_module-func)" in result

    # Test with class
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
class MyClass:
    \"\"\"Class docstring.\"\"\"
    x: int = 1
    def method(self, a: str) -> None:
        \"\"\"Method docstring.\"\"\"
        pass
""")
    result = p.compile()
    assert "class MyClass" in result
    assert "Full name: `test_module.MyClass`" in result
    assert "x: int" in result
    assert "method()" in result
    assert "self: Self" in result
    assert "a: str" in result
    assert "return: None" in result
    assert "Method docstring." in result

    # Test with constants
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
CONST = 42
def func() -> None:
    pass
""")
    result = p.compile()
    assert "Constants" in result
    assert "CONST" in result
    assert "Type" in result
    assert "int" in result

    # Test with imports
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
from typing import List
x: List[int] = []
""")
    result = p.compile()
    assert "x: list[int]" in result

    # Test with private members
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
def _private_func() -> None:
    pass
class MyClass:
    _private_attr: int = 1
    def __init__(self) -> None:
        pass
""")
    result = p.compile()
    assert "_private_func" not in result
    assert "_private_attr" not in result
    assert "__init__" not in result

    # Test with __all__
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
__all__ = ['public_func']
def public_func() -> None:
    pass
def _private_func() -> None:
    pass
""")
    result = p.compile()
    assert "public_func" in result
    assert "_private_func" not in result

    # Test with enum
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
""")
    result = p.compile()
    assert "Enums" in result
    assert "RED" in result
    assert "GREEN" in result
    assert "BLUE" in result

    # Test with type alias
    p = Parser()
    p.parse("test_module", """
\"\"\"Test module docstring.\"\"\"
from typing import List
IntList = List[int]
x: IntList = []
""")
    result = p.compile()
    assert "x: list[int]" in result


# LLM-generated content at query #44
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

    # Test If statement
    if_stmt = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
        orelse=[Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))]
    )
    assert list(walk_body([if_stmt])) == [
        Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2)),
        Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))
    ]

    # Test Try statement
    try_stmt = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
        handlers=[],
        orelse=[Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))],
        finalbody=[Assign(targets=[Name(id='w', ctx=Load())], value=Constant(value=4))]
    )
    assert list(walk_body([try_stmt])) == [
        Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3)),
        Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[])),
        Assign(targets=[Name(id='w', ctx=Load())], value=Constant(value=4))
    ]

    # Test nested If statements
    nested_if = If(
        test=Constant(value=True),
        body=[
            If(
                test=Constant(value=False),
                body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=5))],
                orelse=[]
            )
        ],
        orelse=[]
    )
    assert list(walk_body([nested_if])) == [
        If(
            test=Constant(value=False),
            body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=5))],
            orelse=[]
        )
    ]

    # Test mixed statements
    mixed = [
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=6)),
        If(
            test=Constant(value=True),
            body=[Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))],
            orelse=[]
        ),
        Try(
            body=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=7))],
            handlers=[],
            orelse=[],
            finalbody=[]
        )
    ]
    assert list(walk_body(mixed)) == [
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=6)),
        Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[])),
        Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=7))
    ]


# LLM-generated content at query #45
#--------------------------

```python
def test_Parser_api():
    # Test case 1: FunctionDef
    parser = Parser()
    root = "test_module"
    node = FunctionDef(
        name="test_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert "test_func()" in parser.doc["test_module.test_func"]

    # Test case 2: AsyncFunctionDef
    parser = Parser()
    node = AsyncFunctionDef(
        name="test_async_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.test_async_func" in parser.doc
    assert "async test_async_func()" in parser.doc["test_module.test_async_func"]

    # Test case 3: ClassDef
    parser = Parser()
    node = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]

    # Test case 4: ClassDef with prefix
    parser = Parser()
    node = ClassDef(
        name="InnerClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node, prefix="OuterClass")
    assert "test_module.OuterClass.InnerClass" in parser.doc
    assert "class InnerClass" in parser.doc["test_module.OuterClass.InnerClass"]

    # Test case 5: FunctionDef with decorator
    parser = Parser()
    node = FunctionDef(
        name="test_decorated_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="decorator", ctx=Load())]
    )
    parser.api(root, node)
    assert "test_module.test_decorated_func" in parser.doc
    assert "test_decorated_func()" in parser.doc["test_module.test_decorated_func"]
    assert "Decorators" in parser.doc["test_module.test_decorated_func"]

    # Test case 6: ClassDef with bases
    parser = Parser()
    node = ClassDef(
        name="TestClass",
        bases=[Name(id="BaseClass", ctx=Load())],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.TestClass" in parser.doc
    assert "Bases" in parser.doc["test_module.TestClass"]

    # Test case 7: FunctionDef with docstring
    parser = Parser()
    node = FunctionDef(
        name="test_func_with_doc",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[Expr(value=Constant(value="This is a test function."))],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.test_func_with_doc" in parser.doc
    assert "This is a test function." in parser.docstring["test_module.test_func_with_doc"]

    # Test case 8: ClassDef with docstring
    parser = Parser()
    node = ClassDef(
        name="TestClassWithDoc",
        bases=[],
        keywords=[],
        body=[Expr(value=Constant(value="This is a test class."))],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_module.TestClassWithDoc" in parser.doc
    assert "This is a test class." in parser.docstring["test_module.TestClassWithDoc"]


# LLM-generated content at query #46
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("attr1", Store()), Constant(1), None, None),
        Assign([Name("attr2", Store())], Constant("value")),
        Delete([Name("attr3", Store())])
    ]

    # Test with bases
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

    # Test without bases
    parser.class_api(root, name, [], body)
    assert "Bases" not in parser.doc[name]

    # Test enum detection
    enum_bases = [Name("enum.Enum", Load())]
    parser.class_api(root, name, enum_bases, body)
    assert "Enums" in parser.doc[name]

    # Test member detection
    parser.class_api(root, name, [], body)
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "attr3" not in parser.doc[name]

    # Test private member filtering
    private_body = [
        AnnAssign(Name("_private", Store()), Constant(1), None, None),
        AnnAssign(Name("__dunder", Store()), Constant(2), None, None)
    ]
    parser.class_api(root, name, [], private_body)
    assert "_private" not in parser.doc[name]
    assert "__dunder" not in parser.doc[name]

    # Test public member detection
    public_body = [
        AnnAssign(Name("public", Store()), Constant(1), None, None),
        AnnAssign(Name("Public", Store()), Constant(2), None, None)
    ]
    parser.class_api(root, name, [], public_body)
    assert "public" in parser.doc[name]
    assert "Public" in parser.doc[name]


# LLM-generated content at query #47
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("test_module", {"test_module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Handle self_ty
    resolver = Resolver("test_module", {}, self_ty="self")
    node = Name("self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if not in alias
    resolver = Resolver("test_module", {"test_module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("test_module", {"test_module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Handle nested alias resolution
    resolver = Resolver("test_module", {"test_module.A": "test_module.B", "test_module.B": "final"})
    node = Name("A", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"


# LLM-generated content at query #48
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global names with its expression
    resolver = Resolver("test_module", {"test_module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Replace self_ty with "Self"
    resolver = Resolver("test_module", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: No replacement when name not in alias
    resolver = Resolver("test_module", {"test_module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Support TypeVar
    resolver = Resolver("test_module", {"test_module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("test_module", {"test_module.name": "test_module.other", "test_module.other": "final"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"


# LLM-generated content at query #49
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"

    import_node_with_as = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_with_as)
    assert parser.alias["test_module.np"] == "numpy"

    # Test ImportFrom node with level 0
    import_from_node = ImportFrom(module="sys", names=[alias(name="path")], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias["test_module.path"] == "sys.path"

    # Test ImportFrom node with level 1
    parser.level[root] = 1
    import_from_node_level = ImportFrom(module="os", names=[alias(name="listdir")], level=1)
    parser.imports(root, import_from_node_level)
    assert parser.alias["test_module.listdir"] == "os.listdir"

    # Test ImportFrom node with asname
    import_from_node_as = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    parser.imports(root, import_from_node_as)
    assert parser.alias["test_module.dd"] == "collections.defaultdict"


# LLM-generated content at query #50
#--------------------------

```python
def test_Parser_globals():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"

    # Test type alias
    node = AnnAssign(
        target=Name("TypeAlias", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TypeAlias"] == "42"
    assert parser.const["test_module.TypeAlias"] == "int"

    # Test constant
    node = Assign(
        targets=[Name("CONSTANT", Load())],
        value=Constant(3.14),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.alias["test_module.CONSTANT"] == "3.14"
    assert parser.const["test_module.CONSTANT"] == "float"

    # Test __all__ filter
    node = Assign(
        targets=[Name("__all__", Load())],
        value=List(elts=[Constant("public_func"), Constant("public_class")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func", "test_module.public_class"}

    # Test non-constant assignment
    node = Assign(
        targets=[Name("variable", Load())],
        value=Constant(100),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.alias["test_module.variable"] == "100"
    assert "test_module.variable" not in parser.const

    # Test type comment
    node = Assign(
        targets=[Name("TypedVar", Load())],
        value=Constant("hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TypedVar"] == "'hello'"
    assert parser.const["test_module.TypedVar"] == "str"


# LLM-generated content at query #51
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global names with its expression
    resolver = Resolver("module", {"module.name": "replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "replacement"

    # Test case 2: Handle self_ty replacement
    resolver = Resolver("module", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "Self"

    # Test case 3: No replacement when name not in alias
    resolver = Resolver("module", {"module.other": "other_replacement"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "name"

    # Test case 4: Handle TypeVar case
    resolver = Resolver("module", {"module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "T"

    # Test case 5: Recursive replacement
    resolver = Resolver("module", {"module.name": "module.other", "module.other": "final"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert unparse(result) == "final"


# LLM-generated content at query #52
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 replacement
    resolver = Resolver("test", {"test.Union": "typing.Union", "test.Optional": "typing.Optional"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional replacement
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test PEP585 warning
    resolver = Resolver("test", {"test.List": "typing.List"})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    with pytest.warns(UserWarning):
        result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

    # Test non-PEP585 case
    resolver = Resolver("test", {"test.Custom": "typing.Custom"})
    node = Subscript(Name("Custom", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result is node


# LLM-generated content at query #53
#--------------------------

```python
def test_Parser_func_api():
    # Test case 1: Simple function with no arguments and no return annotation
    parser = Parser()
    root = "test_module"
    name = "test_module.simple_func"
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = "|||\n|-|-|-|\n|return|Any|\n"
    assert parser.doc[name].endswith(expected_output)

    # Test case 2: Function with positional arguments and return annotation
    parser = Parser()
    root = "test_module"
    name = "test_module.complex_func"
    args = arguments(
        posonlyargs=[arg("a", None), arg("b", None)],
        args=[arg("c", None)],
        kwonlyargs=[arg("d", None)],
        kw_defaults=[Constant(1)],
        defaults=[Constant(2)]
    )
    returns = Constant("str")
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = "|||||\n|-|-|-|-|-|\n|a|b|/|c|*|\n|d|return|\n|Any|Any|Any|Any|Any|\n|1|str|\n"
    assert parser.doc[name].endswith(expected_output)

    # Test case 3: Function with self parameter (instance method)
    parser = Parser()
    root = "test_module"
    name = "test_module.MyClass.method"
    args = arguments(
        posonlyargs=[],
        args=[arg("self", None), arg("x", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = None
    parser.func_api(root, name, args, returns, has_self=True, cls_method=False)
    expected_output = "|||\n|-|-|-|\n|self|x|return|\n|Self|Any|Any|\n"
    assert parser.doc[name].endswith(expected_output)

    # Test case 4: Function with cls parameter (class method)
    parser = Parser()
    root = "test_module"
    name = "test_module.MyClass.class_method"
    args = arguments(
        posonlyargs=[],
        args=[arg("cls", None), arg("x", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = None
    parser.func_api(root, name, args, returns, has_self=True, cls_method=True)
    expected_output = "|||\n|-|-|-|\n|cls|x|return|\n|type[Self]|Any|Any|\n"
    assert parser.doc[name].endswith(expected_output)

    # Test case 5: Function with *args and **kwargs
    parser = Parser()
    root = "test_module"
    name = "test_module.var_args_func"
    args = arguments(
        posonlyargs=[],
        args=[arg("x", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=arg("args", None),
        kwarg=arg("kwargs", None)
    )
    returns = None
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_output = "||||\n|-|-|-|-|\n|x|*args|**kwargs|return|\n|Any|Any|Any|Any|\n"
    assert parser.doc[name].endswith(expected_output)


# LLM-generated content at query #54
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Constant(value=2))
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

    # Test nested If statements
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=3))],
        orelse=[Expr(value=Constant(value=4))]
    )
    assert list(walk_body([if_node])) == [
        Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=3)),
        Expr(value=Constant(value=4))
    ]

    # Test Try statement with all components
    try_node = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=5))],
        handlers=[{
            'type': Name(id='Exception', ctx=Load()),
            'name': 'e',
            'body': [Expr(value=Constant(value=6))]
        }],
        orelse=[Expr(value=Constant(value=7))],
        finalbody=[Expr(value=Constant(value=8))]
    )
    assert list(walk_body([try_node])) == [
        Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=5)),
        Expr(value=Constant(value=6)),
        Expr(value=Constant(value=7)),
        Expr(value=Constant(value=8))
    ]

    # Test mixed nested structures
    mixed_node = If(
        test=Constant(value=True),
        body=[
            Try(
                body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=9))],
                handlers=[],
                orelse=[],
                finalbody=[]
            ),
            Expr(value=Constant(value=10))
        ],
        orelse=[
            If(
                test=Constant(value=False),
                body=[Expr(value=Constant(value=11))],
                orelse=[]
            )
        ]
    )
    assert list(walk_body([mixed_node])) == [
        Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=9)),
        Expr(value=Constant(value=10)),
        Expr(value=Constant(value=11))
    ]


# LLM-generated content at query #55
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test AnnAssign with type annotation
    node_ann = AnnAssign(
        target=Name("CONST_VAR", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node_ann)
    assert parser.alias.get("test_module.CONST_VAR") == "42"
    assert parser.const.get("test_module.CONST_VAR") == "int"
    assert parser.root.get("test_module.CONST_VAR") == root

    # Test Assign with type comment
    node_assign = Assign(
        targets=[Name("ANOTHER_CONST", Store())],
        value=Constant("hello"),
        type_comment="str"
    )
    parser.globals(root, node_assign)
    assert parser.alias.get("test_module.ANOTHER_CONST") == "'hello'"
    assert parser.const.get("test_module.ANOTHER_CONST") == "str"
    assert parser.root.get("test_module.ANOTHER_CONST") == root

    # Test Assign without type comment (constant type inference)
    node_assign_no_comment = Assign(
        targets=[Name("INFERRED_CONST", Store())],
        value=List(elts=[Constant(1), Constant(2), Constant(3)])
    )
    parser.globals(root, node_assign_no_comment)
    assert parser.alias.get("test_module.INFERRED_CONST") == "[1, 2, 3]"
    assert parser.const.get("test_module.INFERRED_CONST") == "list[int]"

    # Test __all__ handling
    node_all = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("public_func"), Constant("PublicClass")])
    )
    parser.globals(root, node_all)
    assert "test_module.public_func" in parser.imp[root]
    assert "test_module.PublicClass" in parser.imp[root]

    # Test non-constant assignment (should not be added to const)
    node_non_const = Assign(
        targets=[Name("non_const", Store())],
        value=Constant(42)
    )
    parser.globals(root, node_non_const)
    assert parser.const.get("test_module.non_const") == "Any"

    # Test multiple targets (should be ignored)
    node_multi = Assign(
        targets=[Name("a", Store()), Name("b", Store())],
        value=Constant(42)
    )
    parser.globals(root, node_multi)
    assert parser.alias.get("test_module.a") is None
    assert parser.alias.get("test_module.b") is None


# LLM-generated content at query #56
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: No self, no class method, no annotations
    args = [arg("x", None), arg("y", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any"]

    # Test case 2: With self, no class method, no annotations
    args = [arg("self", None), arg("x", None)]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "Any"]

    # Test case 3: With self, class method, no annotations
    args = [arg("cls", None), arg("x", None)]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "Any"]

    # Test case 4: With annotations
    args = [arg("x", Name("int", Load())), arg("y", Name("str", Load()))]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "str"]

    # Test case 5: Mixed annotations and no annotations
    args = [arg("self", Name("TestClass", Load())), arg("x", None)]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "Any"]

    # Test case 6: With self_ty
    args = [arg("self", None), arg("x", Name("Self", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False, self_ty="TestClass"))
    assert result == ["Self", "TestClass"]

    # Test case 7: Varargs and kwargs
    args = [arg("x", None), arg("*", None), arg("y", None), arg("**kwargs", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "", "Any", "Any"]


# LLM-generated content at query #57
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Constant("int")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"typing.Dict": "Dict"})
    node = Subscript(Name("Dict", Load()), Tuple([Constant("str"), Constant("int")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "Dict"

    # Test warning for deprecated name
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Constant("int")], Load()), Load())
    with pytest.warns(UserWarning):
        resolver.visit_Subscript(node)


# LLM-generated content at query #58
#--------------------------

```python
def test_Parser_class_api():
    # Test case 1: Class with bases and members
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("public_attr", Store()), Constant(42), None, None),
        AnnAssign(Name("_private_attr", Store()), Constant("private"), None, None),
        Assign([Name("class_attr", Store())], Constant(3.14))
    ]
    parser.doc[name] = "# TestClass\n\n"
    parser.level[name] = 1
    parser.root[name] = root
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]
    assert "class_attr" in parser.doc[name]

    # Test case 2: Enum class
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name("enum.Enum", Load())]
    body = [
        Assign([Name("VALUE1", Store())], Constant(1)),
        Assign([Name("VALUE2", Store())], Constant(2)),
        AnnAssign(Name("_private", Store()), Constant("private"), None, None)
    ]
    parser.doc[name] = "# TestEnum\n\n"
    parser.level[name] = 1
    parser.root[name] = root
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]
    assert "_private" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

    # Test case 3: Class with deleted attributes
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(Name("attr1", Store()), Constant(1), None, None),
        AnnAssign(Name("attr2", Store()), Constant(2), None, None),
        Delete([Name("attr1", Del())])
    ]
    parser.doc[name] = "# TestClass\n\n"
    parser.level[name] = 1
    parser.root[name] = root
    parser.class_api(root, name, bases, body)
    assert "attr1" not in parser.doc[name]
    assert "attr2" in parser.doc[name]

    # Test case 4: Empty class
    parser = Parser()
    root = "test_module"
    name = "test_module.EmptyClass"
    bases = []
    body = []
    parser.doc[name] = "# EmptyClass\n\n"
    parser.level[name] = 1
    parser.root[name] = root
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]


# LLM-generated content at query #59
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no self and no annotations
    args = [arg("x", None), arg("y", None)]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any"]

    # Test case 2: Function with self and no annotations
    args = [arg("self", None), arg("x", None)]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "Any"]

    # Test case 3: Function with self and annotations
    args = [arg("self", Name("TestClass", Load())), arg("x", Name("int", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "int"]

    # Test case 4: Class method with self and annotations
    args = [arg("cls", Name("TestClass", Load())), arg("x", Name("int", Load()))]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int"]

    # Test case 5: Function with *args and **kwargs
    args = [
        arg("x", Name("int", Load())),
        arg("*", None),
        arg("y", Name("str", Load())),
        arg("**", None),
        arg("z", Name("float", Load())),
        arg("return", Name("bool", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "", "float", "bool"]

    # Test case 6: Function with mixed annotations and no annotations
    args = [
        arg("a", Name("int", Load())),
        arg("b", None),
        arg("c", Name("str", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "Any", "str"]


# LLM-generated content at query #60
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(target=Name("attr1", Store()), annotation=Name("int", Load()), value=None),
        AnnAssign(target=Name("attr2", Store()), annotation=Name("str", Load()), value=None),
        Assign(targets=[Name("attr3", Store())], value=Constant(42)),
        Delete(targets=[Name("attr1", Del())])
    ]

    # Mock the necessary methods
    parser.resolve = lambda r, n, *args: n.id if isinstance(n, Name) else "Any"
    parser.doc = {}
    parser.level = {root: 0}
    parser.root = {root: root}
    parser.const = {}
    parser.docstring = {}

    # Execute
    parser.class_api(root, name, bases, body)

    # Verify
    assert name in parser.doc
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "attr3" in parser.doc[name]
    assert "Any" in parser.doc[name]
    assert "attr1" not in parser.doc[name]


# LLM-generated content at query #61
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Attribute with 'typing' prefix
    node = Attribute(Name('typing', Load()), 'List', Load())
    resolver = Resolver('test_module', {'test_module.typing': 'typing'})
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'

    # Test case 2: Attribute without 'typing' prefix
    node = Attribute(Name('other_module', Load()), 'SomeClass', Load())
    resolver = Resolver('test_module', {})
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == 'SomeClass'
    assert result.value.id == 'other_module'

    # Test case 3: Attribute with non-Name value
    node = Attribute(Constant(42), 'attr', Load())
    resolver = Resolver('test_module', {})
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.value == 42
    assert result.attr == 'attr'


# LLM-generated content at query #62
#--------------------------

```python
def test_Parser_func_ann():
    # Test case 1: Simple function with no self and no annotations
    parser = Parser()
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']

    # Test case 2: Function with self and no annotations
    parser = Parser()
    args = [arg('self', None), arg('x', None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'Any']

    # Test case 3: Function with self and annotations
    parser = Parser()
    args = [arg('self', Name('int', Load())), arg('x', Name('str', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'str']

    # Test case 4: Class method with self and annotations
    parser = Parser()
    args = [arg('cls', Name('int', Load())), arg('x', Name('str', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'str']

    # Test case 5: Function with *args and **kwargs
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('*', None), arg('y', Name('str', Load())), arg('**', None), arg('return', Name('bool', Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str', '', 'bool']

    # Test case 6: Function with self_ty provided
    parser = Parser()
    args = [arg('self', None), arg('x', Name('Self', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False, self_ty='CustomType'))
    assert result == ['Self', 'CustomType']


# LLM-generated content at query #63
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("test_module", {"test_module.name": "replacement"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"

    # Test case 2: Replace self_ty with "Self"
    resolver = Resolver("test_module", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 3: Return original node if name is not in alias
    resolver = Resolver("test_module", {"test_module.other": "other_replacement"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 4: Return original node if name is in alias but points to itself
    resolver = Resolver("test_module", {"test_module.name": "test_module.name"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 5: Support TypeVar (should return original node)
    resolver = Resolver("test_module", {"test_module.T": "typing.TypeVar('T')"}, "Self")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #64
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="sys", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias["test_module.sys"] == "sys"

    import_node_with_asname = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_with_asname)
    assert parser.alias["test_module.np"] == "numpy"

    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(module="collections", names=[alias(name="defaultdict", asname=None)], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"

    import_from_node_with_asname = ImportFrom(module="os.path", names=[alias(name="join", asname="path_join")], level=0)
    parser.imports(root, import_from_node_with_asname)
    assert parser.alias["test_module.path_join"] == "os.path.join"

    # Test ImportFrom node with level>0
    parser.level[root] = 2
    import_from_node_with_level = ImportFrom(module="sibling", names=[alias(name="helper", asname=None)], level=1)
    parser.imports(root, import_from_node_with_level)
    assert parser.alias["test_module.sibling.helper"] == "test_module.sibling.helper"


# LLM-generated content at query #65
#--------------------------

```python
def test_const_type():
    # Test Constant node
    node = Constant(value=42)
    assert const_type(node) == "int"

    # Test Tuple node
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

    # Test List node
    node = List(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "list[int, int]"

    # Test Set node
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "set[int, int]"

    # Test Dict node
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "dict[int, str]"

    # Test Call node with Name
    node = Call(func=Name(id="int"))
    assert const_type(node) == "int"

    # Test Call node with Attribute
    node = Call(func=Attribute(value=Name(id="module"), attr="int"))
    assert const_type(node) == "int"

    # Test unsupported node
    node = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(node) == "Any"


# LLM-generated content at query #66
#--------------------------

```python
def test_Parser_func_ann():
    # Test case 1: Simple function with no self and no class method
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('return', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

    # Test case 2: Function with self and not a class method
    parser = Parser()
    args = [arg('self', Name('Class', Load())), arg('x', Name('int', Load())), arg('return', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str']

    # Test case 3: Function with self and is a class method
    parser = Parser()
    args = [arg('cls', Name('Class', Load())), arg('x', Name('int', Load())), arg('return', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int', 'str']

    # Test case 4: Function with *args and **kwargs
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('*', None), arg('y', Name('str', Load())), arg('**', None), arg('return', Name('bool', Load()))]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str', '', 'bool']

    # Test case 5: Function with no annotations
    parser = Parser()
    args = [arg('x', None), arg('return', None)]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']


# LLM-generated content at query #67
#--------------------------

```python
def test_Parser_func_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    args = arguments(
        posonlyargs=[arg("a", None), arg("b", None)],
        args=[arg("c", None), arg("d", None)],
        defaults=[Constant(1), Constant(2)],
        vararg=arg("args", None),
        kwonlyargs=[arg("e", None), arg("f", None)],
        kw_defaults=[Constant(3), Constant(4)],
        kwarg=arg("kwargs", None)
    )
    returns = Constant("result")

    # Test case 1: Regular function
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_doc = f"## test_func()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("a", "b", "/", "c", "d", "*args", "*", "e", "f", "**kwargs", "return",
                         items=[["Any", "Any", "", "Any", "Any", "Any", "", "Any", "Any", "Any", "Any"]])
    assert parser.doc[name] == expected_doc

    # Test case 2: Function with self
    parser.func_api(root, name, args, returns, has_self=True, cls_method=False)
    expected_doc = f"## test_func()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("a", "b", "/", "c", "d", "*args", "*", "e", "f", "**kwargs", "return",
                         items=[["Self", "Any", "", "Any", "Any", "Any", "", "Any", "Any", "Any", "Any"]])
    assert parser.doc[name] == expected_doc

    # Test case 3: Class method
    parser.func_api(root, name, args, returns, has_self=True, cls_method=True)
    expected_doc = f"## test_func()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("a", "b", "/", "c", "d", "*args", "*", "e", "f", "**kwargs", "return",
                         items=[["type[Self]", "Any", "", "Any", "Any", "Any", "", "Any", "Any", "Any", "Any"]])
    assert parser.doc[name] == expected_doc

    # Test case 4: Function with defaults
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    expected_doc = f"## test_func()\n\n*Full name:* `{name}`\n\n<a id=\"{name.lower().replace('.', '-')}\"></a>\n\n"
    expected_doc += table("a", "b", "/", "c", "d", "*args", "*", "e", "f", "**kwargs", "return",
                         items=[
                             ["Any", "Any", "", "Any", "Any", "Any", "", "Any", "Any", "Any", "Any"],
                             ["", "", "", "1", "2", "", "", "3", "4", "", ""]
                         ])
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #68
#--------------------------

```python
def test_walk_body():
    # Test simple body
    body = [Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test body with If statement
    body = [
        If(
            test=Constant(value=True),
            body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
            orelse=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test body with Try statement
    body = [
        Try(
            body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
            handlers=[],
            orelse=[],
            finalbody=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test nested If statements
    body = [
        If(
            test=Constant(value=True),
            body=[
                If(
                    test=Constant(value=False),
                    body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))],
                    orelse=[]
                )
            ],
            orelse=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 1
    assert isinstance(result[0], Assign)

    # Test Try with handlers
    body = [
        Try(
            body=[Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5))],
            handlers=[
                {
                    'type': Name(id='Exception', ctx=Load()),
                    'body': [Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))]
                }
            ],
            orelse=[],
            finalbody=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 2
    assert all(isinstance(node, Assign) for node in result)

    # Test empty body
    body = []
    result = list(walk_body(body))
    assert len(result) == 0


# LLM-generated content at query #69
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert isinstance(result.right, Name)

    # Test Optional conversion
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {})
    node = Subscript(Name("SomeType", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "SomeType"


# LLM-generated content at query #70
#--------------------------

```python
def test_const_type():
    # Test Constant node
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"

    # Test Tuple, List, Set nodes
    assert const_type(Tuple(elts=[Constant(value=1), Constant(value=2)])) == "tuple[int, int]"
    assert const_type(Tuple(elts=[Constant(value=1), Constant(value="a")])) == "tuple[Any, Any]"
    assert const_type(List(elts=[Constant(value=1), Constant(value=2)])) == "list[int, int]"
    assert const_type(Set(elts=[Constant(value=1), Constant(value=2)])) == "set[int, int]"

    # Test Dict node
    assert const_type(Dict(keys=[Constant(value="a")], values=[Constant(value=1)])) == "dict[str, int]"
    assert const_type(Dict(keys=[Constant(value="a"), Constant(value=1)], values=[Constant(value=1), Constant(value="b")])) == "dict[Any, Any]"

    # Test Call node with Name or Attribute
    assert const_type(Call(func=Name(id="int"))) == "int"
    assert const_type(Call(func=Attribute(value=Name(id="x"), attr="int"))) == "int"

    # Test unsupported node types
    assert const_type(Name(id="x")) == "Any"
    assert const_type(BinOp()) == "Any"


# LLM-generated content at query #71
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test Union
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test PEP585
    resolver = Resolver("test", {"typing.List": "List"})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

    # Test non-typing attribute
    resolver = Resolver("test", {})
    node = Subscript(Name("test", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "test"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"


# LLM-generated content at query #72
#--------------------------

```python
def test_Parser_load_docstring():
    # Setup
    parser = Parser()
    root = "test_module"
    parser.doc = {
        f"{root}.ClassA": "ClassA doc",
        f"{root}.ClassA.method1": "method1 doc",
        f"{root}.ClassB": "ClassB doc",
        f"{root}.function1": "function1 doc"
    }
    parser.docstring = {}

    # Create a mock module with docstrings
    class MockModule:
        class ClassA:
            """ClassA docstring"""
            def method1(self):
                """method1 docstring"""
                pass

        class ClassB:
            """ClassB docstring"""
            pass

        def function1():
            """function1 docstring"""
            pass

    mock_module = MockModule()

    # Execute
    parser.load_docstring(root, mock_module)

    # Verify
    assert parser.docstring[f"{root}.ClassA"] == "ClassA docstring"
    assert parser.docstring[f"{root}.ClassA.method1"] == "method1 docstring"
    assert parser.docstring[f"{root}.ClassB"] == "ClassB docstring"
    assert parser.docstring[f"{root}.function1"] == "function1 docstring"


# LLM-generated content at query #73
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 substitution
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant(1), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union substitution
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant(1), Constant(2)], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)

    # Test Optional substitution
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant(1), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-substitution case
    resolver = Resolver("test", {"typing.Dict": "dict"})
    node = Subscript(Name("SomeType", Load()), Constant(1), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "SomeType"


# LLM-generated content at query #74
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"

    import_node_with_asname = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_with_asname)
    assert parser.alias["test_module.np"] == "numpy"

    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(module="sys", names=[alias(name="path", asname=None)], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias["test_module.path"] == "sys.path"

    import_from_node_with_asname = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    parser.imports(root, import_from_node_with_asname)
    assert parser.alias["test_module.dd"] == "collections.defaultdict"

    # Test ImportFrom node with level>0
    import_from_node_with_level = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    parser.imports(root, import_from_node_with_level)
    assert parser.alias["test_module.path"] == "test_module.os.path"


# LLM-generated content at query #75
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no arguments and no return
    func_node = FunctionDef(
        name="test_func1",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.test_func1", func_node.args, func_node.returns, has_self=False, cls_method=False)
    expected_output = parser.doc["test_module.test_func1"]
    assert "return" in expected_output
    assert "Any" in expected_output

    # Test case 2: Function with positional arguments and return type
    func_node = FunctionDef(
        name="test_func2",
        args=arguments(
            posonlyargs=[arg(arg="arg1", annotation=Name(id="int", ctx=Load()))],
            args=[arg(arg="arg2", annotation=Name(id="str", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=Name(id="bool", ctx=Load()),
        decorator_list=[]
    )
    parser.func_api(root, "test_module.test_func2", func_node.args, func_node.returns, has_self=False, cls_method=False)
    expected_output = parser.doc["test_module.test_func2"]
    assert "arg1" in expected_output
    assert "arg2" in expected_output
    assert "int" in expected_output
    assert "str" in expected_output
    assert "bool" in expected_output
    assert "return" in expected_output

    # Test case 3: Function with default values
    func_node = FunctionDef(
        name="test_func3",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="arg1", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[Constant(value=10)]
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.test_func3", func_node.args, func_node.returns, has_self=False, cls_method=False)
    expected_output = parser.doc["test_module.test_func3"]
    assert "arg1" in expected_output
    assert "10" in expected_output
    assert "int" in expected_output

    # Test case 4: Function with *args and **kwargs
    func_node = FunctionDef(
        name="test_func4",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=arg(arg="args", annotation=None),
            kwarg=arg(arg="kwargs", annotation=None)
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.test_func4", func_node.args, func_node.returns, has_self=False, cls_method=False)
    expected_output = parser.doc["test_module.test_func4"]
    assert "*args" in expected_output
    assert "**kwargs" in expected_output
    assert "return" in expected_output

    # Test case 5: Function with self and class method
    func_node = FunctionDef(
        name="test_func5",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="self", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=None,
        decorator_list=[Name(id="classmethod", ctx=Load())]
    )
    parser.func_api(root, "test_module.test_func5", func_node.args, func_node.returns, has_self=True, cls_method=True)
    expected_output = parser.doc["test_module.test_func5"]
    assert "Self" in expected_output
    assert "type[Self]" in expected_output


# LLM-generated content at query #76
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test removing 'typing.*' prefix
    resolver = Resolver("test_module", {"typing.List": "list"})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

    # Test non-typing attribute
    node = Attribute(Name("other_module", Load()), "SomeClass", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "SomeClass"
    assert result.value.id == "other_module"

    # Test nested attribute (should not modify)
    node = Attribute(Attribute(Name("module", Load()), "submodule", Load()), "Class", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "Class"
    assert isinstance(result.value, Attribute)


# LLM-generated content at query #77
#--------------------------

```python
def test_const_type():
    # Test Constant node
    node = Constant(value=42)
    assert const_type(node) == "int"

    # Test Tuple node
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

    # Test List node
    node = List(elts=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "list[str, str]"

    # Test Set node
    node = Set(elts=[Constant(value=1.0), Constant(value=2.0)])
    assert const_type(node) == "set[float, float]"

    # Test Dict node
    node = Dict(keys=[Constant(value="x"), Constant(value="y")], values=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "dict[str, str, int, int]"

    # Test Call node with Name
    node = Call(func=Name(id="int"))
    assert const_type(node) == "int"

    # Test Call node with Attribute
    node = Call(func=Attribute(value=Name(id="typing"), attr="List"))
    assert const_type(node) == "typing.List"

    # Test unsupported node
    node = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(node) == "Any"


# LLM-generated content at query #78
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "os")] == "os"

    import_node = Import(names=[alias(name="sys", asname="system")])
    parser.imports(root, import_node)
    assert parser.alias[_m(root, "system")] == "sys"

    # Test ImportFrom node with level=0
    import_from_node = ImportFrom(module="os.path", names=[alias(name="join")], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "join")] == "os.path.join"

    # Test ImportFrom node with level=1
    import_from_node = ImportFrom(module="path", names=[alias(name="join")], level=1)
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "join")] == _m("test", "path.join")

    # Test ImportFrom node with asname
    import_from_node = ImportFrom(module="os", names=[alias(name="path", asname="ospath")], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias[_m(root, "ospath")] == "os.path"


# LLM-generated content at query #79
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"typing.Dict": "Dict"})
    node = Subscript(Name("Dict", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"


# LLM-generated content at query #80
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace global name with its expression
    resolver = Resolver("test", {"test.name": "int"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test case 2: Return node if name not in alias
    resolver = Resolver("test", {"test.other": "str"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test case 3: Replace self_ty with "Self"
    resolver = Resolver("test", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 4: Support TypeVar
    resolver = Resolver("test", {"test.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test case 5: Replace nested name
    resolver = Resolver("test", {"test.nested.name": "float"})
    node = Name("nested.name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "float"


# LLM-generated content at query #81
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("attr1", Store()), Constant(1), None, None),
        AnnAssign(Name("attr2", Store()), Constant("hello"), None, None),
        Assign([Name("attr3", Store())], Constant(3.14)),
        Delete([Name("attr2", Store())])
    ]

    # Mock the necessary methods
    parser.resolve = lambda r, n: n.id if isinstance(n, Name) else "Any"
    parser.root = {root: root}
    parser.level = {root: 0}
    parser.doc = {name: ""}
    parser.docstring = {}
    parser.imp = {root: set()}

    # Execute
    parser.class_api(root, name, bases, body)

    # Verify
    expected_doc = (
        "Bases\n---\n| BaseClass |\n\n"
        "Members\n-------\n| Name | Type |\n|------|------|\n| `attr1` | `attr1` |\n| `attr3` | `Any` |\n"
    )
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #82
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    union_node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    optional_node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 case
    resolver = Resolver("test", {})
    node = Subscript(Name("SomeType", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #83
#--------------------------

```python
def test_Parser_api():
    # Test case 1: FunctionDef
    parser = Parser()
    root = "test_module"
    node = FunctionDef(
        name="test_function",
        args=arguments(
            posonlyargs=[arg(arg="a")],
            args=[arg(arg="b")],
            kwonlyargs=[arg(arg="c")],
            kw_defaults=[Constant(value=1)],
            defaults=[Constant(value=2)]
        ),
        body=[Pass()],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "test_function" in parser.doc
    assert "test_module.test_function" in parser.doc
    assert "test_function()" in parser.doc["test_module.test_function"]

    # Test case 2: AsyncFunctionDef
    parser = Parser()
    node = AsyncFunctionDef(
        name="async_test_function",
        args=arguments(
            args=[arg(arg="a")],
            defaults=[Constant(value=1)]
        ),
        body=[Pass()],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "async_test_function" in parser.doc
    assert "test_module.async_test_function" in parser.doc
    assert "async async_test_function()" in parser.doc["test_module.async_test_function"]

    # Test case 3: ClassDef
    parser = Parser()
    node = ClassDef(
        name="TestClass",
        bases=[],
        body=[Pass()],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "TestClass" in parser.doc
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]

    # Test case 4: ClassDef with prefix
    parser = Parser()
    node = ClassDef(
        name="InnerClass",
        bases=[],
        body=[Pass()],
        decorator_list=[]
    )
    parser.api(root, node, prefix="OuterClass")
    assert "OuterClass.InnerClass" in parser.doc
    assert "test_module.OuterClass.InnerClass" in parser.doc
    assert "class OuterClass_InnerClass" in parser.doc["test_module.OuterClass.InnerClass"]

    # Test case 5: FunctionDef with decorators
    parser = Parser()
    node = FunctionDef(
        name="decorated_function",
        args=arguments(),
        body=[Pass()],
        decorator_list=[Name(id="decorator", ctx=Load())]
    )
    parser.api(root, node)
    assert "decorated_function" in parser.doc
    assert "test_module.decorated_function" in parser.doc
    assert "@decorator" in parser.doc["test_module.decorated_function"]

    # Test case 6: ClassDef with bases
    parser = Parser()
    node = ClassDef(
        name="DerivedClass",
        bases=[Name(id="BaseClass", ctx=Load())],
        body=[Pass()],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "DerivedClass" in parser.doc
    assert "test_module.DerivedClass" in parser.doc
    assert "Bases" in parser.doc["test_module.DerivedClass"]
    assert "BaseClass" in parser.doc["test_module.DerivedClass"]

    # Test case 7: ClassDef with members
    parser = Parser()
    node = ClassDef(
        name="ClassWithMembers",
        bases=[],
        body=[
            AnnAssign(
                target=Name(id="member", ctx=Store()),
                annotation=Name(id="int", ctx=Load()),
                value=None
            )
        ],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "ClassWithMembers" in parser.doc
    assert "test_module.ClassWithMembers" in parser.doc
    assert "Members" in parser.doc["test_module.ClassWithMembers"]
    assert "member" in parser.doc["test_module.ClassWithMembers"]
    assert "int" in parser.doc["test_module.ClassWithMembers"]

    # Test case 8: ClassDef with nested class
    parser = Parser()
    node = ClassDef(
        name="OuterClass",
        bases=[],
        body=[
            ClassDef(
                name="InnerClass",
                bases=[],
                body=[Pass()],
                decorator_list=[]
            )
        ],
        decorator_list=[]
    )
    parser.api(root, node)
    assert "OuterClass" in parser.doc
    assert "test_module.OuterClass" in parser.doc
    assert "InnerClass" in parser.doc
    assert "test_module.OuterClass.InnerClass" in parser.doc


# LLM-generated content at query #84
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant(42),
            type_comment="int"
        ),
        Delete(targets=[Name("attr3", Del())])
    ]

    # Mock the necessary methods
    parser.resolve = lambda r, n: n.id if isinstance(n, Name) else "Any"
    parser.doc = {name: ""}
    parser.level = {name: 0}
    parser.root = {name: root}
    parser.const = {}
    parser.imp = {root: set()}

    # Execute
    parser.class_api(root, name, bases, body)

    # Verify
    expected_doc = (
        "Bases\n---\n`BaseClass`\n\n"
        "Members\n-------\nName | Type\n--- | ---\n`attr1` | `int`\n`attr2` | `int`\n"
    )
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #85
#--------------------------

```python
def test_const_type():
    # Test Constant node
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"

    # Test Tuple node
    tuple_node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(tuple_node) == "tuple[int, int]"

    # Test List node
    list_node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(list_node) == "list[int, int]"

    # Test Set node
    set_node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(set_node) == "set[int, int]"

    # Test Dict node
    dict_node = Dict(keys=[Constant(value="a"), Constant(value="b")],
                     values=[Constant(value=1), Constant(value=2)])
    assert const_type(dict_node) == "dict[str, int]"

    # Test Call node with Name
    call_node = Call(func=Name(id="int", ctx=Load()),
                     args=[Constant(value="42")], keywords=[])
    assert const_type(call_node) == "int"

    # Test Call node with Attribute
    call_node = Call(func=Attribute(value=Name(id="builtins", ctx=Load()),
                                    attr="str", ctx=Load()),
                     args=[Constant(value=42)], keywords=[])
    assert const_type(call_node) == "str"

    # Test unsupported node type
    unsupported_node = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(unsupported_node) == "Any"

    # Test empty node
    assert const_type(None) == "Any"


# LLM-generated content at query #86
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test type alias
    node = AnnAssign(
        target=Name("MyType", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.MyType"] == "42"
    assert parser.const["test_module.MyType"] == "int"

    # Test constant
    node = Assign(
        targets=[Name("CONST", Load())],
        value=Constant(3.14),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.alias["test_module.CONST"] == "3.14"
    assert parser.const["test_module.CONST"] == "float"

    # Test __all__ filter
    node = Assign(
        targets=[Name("__all__", Load())],
        value=List(elts=[Constant("public_func"), Constant("public_var")], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func", "test_module.public_var"}

    # Test non-constant assignment
    node = Assign(
        targets=[Name("var", Load())],
        value=Name("value", Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert "test_module.var" not in parser.const

    # Test non-Name target
    node = Assign(
        targets=[Subscript(Name("list", Load()), Constant(0), Load())],
        value=Constant(1),
        type_comment=None
    )
    parser.globals(root, node)
    assert "test_module.list" not in parser.alias


