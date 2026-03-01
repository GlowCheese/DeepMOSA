####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_api():
    parser = Parser()
    parser.parse("test_module", "")
    
    # Test function definition
    func_node = FunctionDef(
        name="test_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=Name(id="str", ctx=Load())
    )
    
    parser.api("test_module", func_node)
    assert "test_module.test_func" in parser.doc
    assert "test_func()" in parser.doc["test_module.test_func"]
    assert "*Full name:* `test_module.test_func`" in parser.doc["test_module.test_func"]
    
    # Test async function
    async_func = AsyncFunctionDef(
        name="async_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=None
    )
    
    parser.api("test_module", async_func)
    assert "test_module.async_func" in parser.doc
    assert "async async_func()" in parser.doc["test_module.async_func"]
    
    # Test class definition
    class_node = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    
    parser.api("test_module", class_node)
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]
    
    # Test nested class
    nested_class = ClassDef(
        name="NestedClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    
    parser.api("test_module", nested_class, prefix="TestClass")
    assert "test_module.TestClass.NestedClass" in parser.doc
    assert "class NestedClass" in parser.doc["test_module.TestClass.NestedClass"]
    
    # Test function with decorators
    decorated_func = FunctionDef(
        name="decorated_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="staticmethod", ctx=Load())],
        returns=None
    )
    
    parser.api("test_module", decorated_func)
    assert "@staticmethod" in parser.doc["test_module.decorated_func"]
    
    # Test with self_ty parameter
    class_with_method = ClassDef(
        name="ClassWithMethod",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    
    parser.api("test_module", class_with_method)
    
    method_node = FunctionDef(
        name="instance_method",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="self", annotation=None)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=None
    )
    
    parser.api("test_module", method_node, prefix="ClassWithMethod")
    assert "test_module.ClassWithMethod.instance_method" in parser.doc
    
    # Test class method
    class_method = FunctionDef(
        name="class_method",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="cls", annotation=None)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="classmethod", ctx=Load())],
        returns=None
    )
    
    parser.api("test_module", class_method, prefix="ClassWithMethod")
    assert "@classmethod" in parser.doc["test_module.ClassWithMethod.class_method"]
    
    # Test with docstring
    func_with_doc = FunctionDef(
        name="func_with_doc",
        args=arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[Expr(value=Constant(value="Test docstring"))],
        decorator_list=[],
        returns=None
    )
    
    parser.api("test_module", func_with_doc)
    assert "test_module.func_with_doc" in parser.doc
    
    # Test with link option enabled
    parser_with_link = Parser(link=True)
    parser_with_link.parse("test_module", "")
    parser_with_link.api("test_module", func_node)
    
    assert "<a id=" in parser_with_link.doc["test_module.test_func"]
    
    # Test with different base level
    parser_level_2 = Parser(b_level=2)
    parser_level_2.parse("test_module", "")
    parser_level_2.api("test_module", func_node)
    
    assert "## test_func()" in parser_level_2.doc["test_module.test_func"]


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_is_public():
    # Test 1: Public name without __all__ filter
    parser = Parser()
    parser.root = {"module.name": "module"}
    parser.imp = {"module": set()}
    assert parser.is_public("module.name") == True
    
    # Test 2: Private name (starts with underscore)
    parser = Parser()
    parser.root = {"module._private": "module"}
    parser.imp = {"module": set()}
    assert parser.is_public("module._private") == False
    
    # Test 3: Name in __all__ list
    parser = Parser()
    parser.root = {"module.name": "module"}
    parser.imp = {"module": {"module.name"}}
    assert parser.is_public("module.name") == True
    
    # Test 4: Parent in __all__ list
    parser = Parser()
    parser.root = {"module.sub.name": "module"}
    parser.imp = {"module": {"module.sub"}}
    assert parser.is_public("module.sub.name") == True
    
    # Test 5: Not in __all__ when __all__ exists
    parser = Parser()
    parser.root = {"module.name": "module"}
    parser.imp = {"module": {"module.other"}}
    assert parser.is_public("module.name") == False
    
    # Test 6: Module root with __all__ containing children
    parser = Parser()
    parser.root = {"module": "module"}
    parser.imp = {"module": {"module.child"}}
    assert parser.is_public("module") == True
    
    # Test 7: Module root without __all__ but with public children
    parser = Parser()
    parser.root = {"module": "module", "module.public_child": "module"}
    parser.imp = {"module": set()}
    parser.doc = {"module.public_child": ""}
    assert parser.is_public("module") == True
    
    # Test 8: Module root without __all__ and no public children
    parser = Parser()
    parser.root = {"module": "module"}
    parser.imp = {"module": set()}
    parser.doc = {}
    assert parser.is_public("module") == False
    
    # Test 9: Name with magic method (double underscore)
    parser = Parser()
    parser.root = {"module.__magic__": "module"}
    parser.imp = {"module": set()}
    assert parser.is_public("module.__magic__") == False
    
    # Test 10: Mixed case public name
    parser = Parser()
    parser.root = {"module.PublicName": "module"}
    parser.imp = {"module": set()}
    assert parser.is_public("module.PublicName") == True


# LLM-generated content at query #3
#--------------------------

```python
def test_Resolver_visit_Attribute():
    from ast import Attribute, Name, Load, Constant, parse, Expr, Call, Subscript
    from typing import cast

    # Test 1: Attribute with typing module should be converted to Name
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

    # Test 2: Attribute with non-typing module should remain unchanged
    node = Attribute(value=Name(id="collections", ctx=Load()), attr="abc", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Name)
    assert result.value.id == "collections"
    assert result.attr == "abc"

    # Test 3: Attribute with nested structure should remain unchanged
    node = Attribute(
        value=Attribute(value=Name(id="typing", ctx=Load()), attr="Union", ctx=Load()),
        attr="Optional",
        ctx=Load(),
    )
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Attribute)
    assert result.value.value.id == "typing"
    assert result.value.attr == "Union"
    assert result.attr == "Optional"

    # Test 4: Attribute with non-Name value should remain unchanged
    node = Attribute(value=Constant(value="test"), attr="attr", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Constant)
    assert result.value.value == "test"
    assert result.attr == "attr"

    # Test 5: Complex scenario with alias mapping
    resolver = Resolver("test_module", {"test_module.typing": "typing"})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Dict", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Dict"


# LLM-generated content at query #4
#--------------------------

```python
def test_walk_body():
    # Test with simple body
    body = [Expr(Constant(1)), Expr(Constant(2))]
    result = list(walk_body(body))
    assert result == body
    
    # Test with If statement
    if_node = If(
        test=Constant(True),
        body=[Expr(Constant(3)), Expr(Constant(4))],
        orelse=[Expr(Constant(5))]
    )
    body = [Expr(Constant(1)), if_node, Expr(Constant(2))]
    result = list(walk_body(body))
    assert result == [
        Expr(Constant(1)),
        Expr(Constant(3)),
        Expr(Constant(4)),
        Expr(Constant(5)),
        Expr(Constant(2))
    ]
    
    # Test with nested If statements
    nested_if = If(
        test=Constant(False),
        body=[Expr(Constant(6))],
        orelse=[If(
            test=Constant(True),
            body=[Expr(Constant(7))],
            orelse=[]
        )]
    )
    body = [Expr(Constant(1)), nested_if]
    result = list(walk_body(body))
    assert result == [
        Expr(Constant(1)),
        Expr(Constant(6)),
        Expr(Constant(7))
    ]
    
    # Test with Try statement
    try_node = Try(
        body=[Expr(Constant(8)), Expr(Constant(9))],
        handlers=[arg(
            arg='e',
            annotation=None,
            type_comment=None
        )],
        orelse=[Expr(Constant(10))],
        finalbody=[Expr(Constant(11))]
    )
    body = [Expr(Constant(1)), try_node, Expr(Constant(2))]
    result = list(walk_body(body))
    assert result == [
        Expr(Constant(1)),
        Expr(Constant(8)),
        Expr(Constant(9)),
        Expr(Constant(10)),
        Expr(Constant(11)),
        Expr(Constant(2))
    ]
    
    # Test with empty body
    result = list(walk_body([]))
    assert result == []
    
    # Test with mixed control flow
    mixed_body = [
        Expr(Constant(1)),
        If(
            test=Constant(True),
            body=[Try(
                body=[Expr(Constant(2))],
                handlers=[],
                orelse=[],
                finalbody=[]
            )],
            orelse=[]
        ),
        Expr(Constant(3))
    ]
    result = list(walk_body(mixed_body))
    assert result == [
        Expr(Constant(1)),
        Expr(Constant(2)),
        Expr(Constant(3))
    ]


# LLM-generated content at query #5
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    parser.root = {"test_module": "test_module"}
    parser.alias = {}
    
    # Test basic function with no arguments
    name = "test_module.func1"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "func1()" in parser.doc[name]
    assert "*Full name:* `test_module.func1`" in parser.doc[name]
    
    # Test function with arguments and annotations
    name = "test_module.func2"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="x", annotation=Name(id="int", ctx=Load())),
            arg(arg="y", annotation=Name(id="str", ctx=Load()))
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name(id="bool", ctx=Load())
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "func2()" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "return" in parser.doc[name]
    
    # Test function with self argument
    name = "test_module.Class.method"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="self", annotation=None),
            arg(arg="value", annotation=Name(id="int", ctx=Load()))
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=False)
    
    assert "method()" in parser.doc[name]
    
    # Test function with defaults
    name = "test_module.func3"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="a", annotation=None),
            arg(arg="b", annotation=None)
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[
            Constant(value=1),
            Constant(value="default")
        ]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "func3()" in parser.doc[name]
    
    # Test function with vararg and kwarg
    name = "test_module.func4"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg(arg="args", annotation=None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg(arg="kwargs", annotation=None),
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "*args" in parser.doc[name]
    assert "**kwargs" in parser.doc[name]
    
    # Test class method
    name = "test_module.Class.class_method"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="cls", annotation=None),
            arg(arg="param", annotation=Name(id="str", ctx=Load()))
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=True)
    
    assert "class_method()" in parser.doc[name]
    
    # Test function with kwonlyargs
    name = "test_module.func5"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[
            arg(arg="key", annotation=Name(id="str", ctx=Load())),
            arg(arg="value", annotation=Name(id="int", ctx=Load()))
        ],
        kw_defaults=[
            Constant(value="default_key"),
            Constant(value=0)
        ],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "key" in parser.doc[name]
    assert "value" in parser.doc[name]
    
    # Test function with posonlyargs
    name = "test_module.func6"
    node = arguments(
        posonlyargs=[
            arg(arg="pos1", annotation=Name(id="int", ctx=Load())),
            arg(arg="pos2", annotation=Name(id="str", ctx=Load()))
        ],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name(id="None", ctx=Load())
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "pos1" in parser.doc[name]
    assert "pos2" in parser.doc[name]
    assert "/" in parser.doc[name]


# LLM-generated content at query #6
#--------------------------

```python
def test_Resolver_visit_Attribute():
    # Test case 1: Attribute with typing module prefix
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

    # Test case 2: Attribute with non-typing module prefix
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id="collections", ctx=Load()), attr="abc", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result is node  # Should return unchanged node

    # Test case 3: Attribute with non-Name value
    resolver = Resolver("test_module", {})
    sub_node = Attribute(value=Name(id="typing", ctx=Load()), attr="Union", ctx=Load())
    node = Attribute(value=sub_node, attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result is node  # Should return unchanged node

    # Test case 4: Attribute with different context
    resolver = Resolver("test_module", {})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Dict", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Dict"
    assert isinstance(result.ctx, Load)

    # Test case 5: Multiple attribute resolution
    resolver = Resolver("test_module", {})
    node1 = Attribute(value=Name(id="typing", ctx=Load()), attr="Optional", ctx=Load())
    node2 = Attribute(value=Name(id="typing", ctx=Load()), attr="Callable", ctx=Load())
    
    result1 = resolver.visit_Attribute(node1)
    result2 = resolver.visit_Attribute(node2)
    
    assert isinstance(result1, Name)
    assert result1.id == "Optional"
    assert isinstance(result2, Name)
    assert result2.id == "Callable"


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_globals():
    # Test AnnAssign with Name target
    parser = Parser()
    root = "test_module"
    
    # Create AnnAssign node
    target = Name(id="MY_CONST", ctx=Store())
    annotation = Name(id="int", ctx=Load())
    value = Constant(value=42)
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert parser.alias["test_module.MY_CONST"] == "42"
    assert parser.const["test_module.MY_CONST"] == "int"
    assert parser.root["test_module.MY_CONST"] == "test_module"
    
    # Test Assign with Name target and type_comment
    parser = Parser()
    target = Name(id="my_var", ctx=Store())
    value = Constant(value="hello")
    node = Assign(targets=[target], value=value, type_comment="str")
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert parser.alias["test_module.my_var"] == "'hello'"
    assert "test_module.my_var" not in parser.const
    
    # Test Assign with Name target without type_comment
    parser = Parser()
    target = Name(id="my_var", ctx=Store())
    value = Constant(value=3.14)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert parser.alias["test_module.my_var"] == "3.14"
    assert parser.const.get("test_module.my_var", ANY) == ANY
    
    # Test __all__ assignment with Tuple
    parser = Parser()
    target = Name(id="__all__", ctx=Store())
    elts = [Constant(value="func1"), Constant(value="ClassA")]
    value = Tuple(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert parser.imp[root] == {"test_module.func1", "test_module.ClassA"}
    
    # Test __all__ assignment with List
    parser = Parser()
    target = Name(id="__all__", ctx=Store())
    elts = [Constant(value="CONSTANT")]
    value = List(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert parser.imp[root] == {"test_module.CONSTANT"}
    
    # Test non-uppercase constant
    parser = Parser()
    target = Name(id="lowercase", ctx=Store())
    value = Constant(value=100)
    node = Assign(targets=[target], value=value, type_comment="int")
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert "test_module.lowercase" not in parser.const
    
    # Test uppercase constant with existing value
    parser = Parser()
    target = Name(id="EXISTING", ctx=Store())
    value = Constant(value="new_value")
    node = Assign(targets=[target], value=value, type_comment="str")
    
    parser.parse(root, "")
    parser.const["test_module.EXISTING"] = "old_type"
    parser.globals(root, node)
    
    assert parser.const["test_module.EXISTING"] == "str"
    
    # Test AnnAssign without value
    parser = Parser()
    target = Name(id="no_value", ctx=Store())
    annotation = Name(id="Optional[int]", ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=None, simple=1)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert "test_module.no_value" not in parser.alias
    
    # Test Assign with multiple targets
    parser = Parser()
    target1 = Name(id="a", ctx=Store())
    target2 = Name(id="b", ctx=Store())
    value = Constant(value=1)
    node = Assign(targets=[target1, target2], value=value)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias
    
    # Test Assign with non-Name target
    parser = Parser()
    target = Attribute(value=Name(id="module", ctx=Load()), attr="attr", ctx=Store())
    value = Constant(value=2)
    node = Assign(targets=[target], value=value)
    
    parser.parse(root, "")
    parser.globals(root, node)
    
    assert "test_module.attr" not in parser.alias


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    parser.alias = {}
    
    # Test basic function annotation
    args = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=Name(id="bool", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int", "str", "bool"]
    
    # Test with self parameter (non-classmethod)
    args_with_self = [
        arg(arg="self", annotation=None),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_with_self, has_self=True, cls_method=False))
    assert result == ["Self", "int", "None"]
    
    # Test with self parameter (classmethod)
    args_with_cls = [
        arg(arg="cls", annotation=Name(id="type", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_with_cls, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int", "None"]
    
    # Test with annotated self parameter
    args_annotated_self = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=None)
    ]
    result = list(parser.func_ann("test_module", args_annotated_self, has_self=True, cls_method=False))
    assert result == ["Self", "int", "Any"]
    
    # Test with vararg separator
    args_with_star = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="*", annotation=None),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=None)
    ]
    result = list(parser.func_ann("test_module", args_with_star, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "Any"]
    
    # Test with unresolved annotation (should return ANY)
    args_unresolved = [
        arg(arg="x", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result = list(parser.func_ann("test_module", args_unresolved, has_self=False, cls_method=False))
    assert result == ["Any", "Any"]
    
    # Test with alias resolution
    parser.alias = {"test_module.int": "builtins.int"}
    args_alias = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="str", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_alias, has_self=False, cls_method=False))
    assert result == ["int", "str"]
    
    # Test with self_ty parameter in annotation resolution
    args_self_ty = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="other", annotation=Name(id="Self", ctx=Load())),
        arg(arg="return", annotation=None)
    ]
    result = list(parser.func_ann("test_module", args_self_ty, has_self=True, cls_method=False))
    assert result == ["Self", "Self", "Any"]


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_parse():
    parser = Parser()
    
    # Test basic module parsing
    parser.parse("test_module", "def foo(): pass")
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    
    # Test with imports
    parser.parse("test_module2", "import os\ndef bar(): pass")
    assert "test_module2" in parser.doc
    assert "test_module2.bar" in parser.doc
    assert "os" in parser.alias
    
    # Test with class definition
    parser.parse("test_module3", "class MyClass:\n    def method(self): pass")
    assert "test_module3.MyClass" in parser.doc
    assert "test_module3.MyClass.method" in parser.doc
    
    # Test with async function
    parser.parse("test_module4", "async def async_func(): pass")
    assert "test_module4.async_func" in parser.doc
    assert "async" in parser.doc["test_module4.async_func"]
    
    # Test with module docstring
    parser.parse("test_module5", '"""Module docstring."""\ndef func(): pass')
    assert "test_module5" in parser.docstring
    
    # Test with assignments
    parser.parse("test_module6", "CONSTANT = 42\n__all__ = ['foo']")
    assert "test_module6.CONSTANT" in parser.const
    
    # Test with type alias
    parser.parse("test_module7", "from typing import List\nVector = List[float]")
    assert "test_module7.Vector" in parser.alias
    
    # Test nested module
    parser.parse("parent.child", "def nested(): pass")
    assert "parent.child" in parser.doc
    assert "parent.child.nested" in parser.doc
    
    # Test with decorators
    parser.parse("test_module8", "@staticmethod\ndef static_method(): pass")
    assert "test_module8.static_method" in parser.doc
    assert "@staticmethod" in parser.doc["test_module8.static_method"]
    
    # Test empty module
    parser.parse("empty_module", "")
    assert "empty_module" in parser.doc


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    
    # Test basic function with no arguments
    name = "test_module.func1"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func1()" in parser.doc[name]
    assert "*Full name:* `test_module.func1`" in parser.doc[name]
    
    # Test function with positional arguments
    name = "test_module.func2"
    node = arguments(
        posonlyargs=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
        args=[arg(arg="y", annotation=Name(id="str", ctx=Load()))],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name(id="bool", ctx=Load())
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func2()" in parser.doc[name]
    
    # Test function with defaults
    name = "test_module.func3"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="a", annotation=Name(id="int", ctx=Load())),
            arg(arg="b", annotation=Name(id="str", ctx=Load()))
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=1), Constant(value="default")]
    )
    returns = Name(id="None", ctx=Load())
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func3()" in parser.doc[name]
    
    # Test function with vararg and kwarg
    name = "test_module.func4"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg(arg="args", annotation=Name(id="Any", ctx=Load())),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg(arg="kwargs", annotation=Name(id="Any", ctx=Load())),
        defaults=[]
    )
    returns = Name(id="Any", ctx=Load())
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func4()" in parser.doc[name]
    
    # Test method with self
    name = "test_module.Class.method"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="self", annotation=None),
            arg(arg="x", annotation=Name(id="int", ctx=Load()))
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name(id="None", ctx=Load())
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=False)
    assert "method()" in parser.doc[name]
    
    # Test classmethod
    name = "test_module.Class.class_method"
    node = arguments(
        posonlyargs=[],
        args=[
            arg(arg="cls", annotation=Name(id="type", ctx=Load())),
            arg(arg="x", annotation=Name(id="int", ctx=Load()))
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name(id="Any", ctx=Load())
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=True)
    assert "class_method()" in parser.doc[name]
    
    # Test function with kwonlyargs
    name = "test_module.func5"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[
            arg(arg="x", annotation=Name(id="int", ctx=Load())),
            arg(arg="y", annotation=Name(id="str", ctx=Load()))
        ],
        kw_defaults=[Constant(value=1), Constant(value="test")],
        kwarg=None,
        defaults=[]
    )
    returns = None
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func5()" in parser.doc[name]
    
    # Test function with posonlyargs separator
    name = "test_module.func6"
    node = arguments(
        posonlyargs=[
            arg(arg="a", annotation=Name(id="int", ctx=Load())),
            arg(arg="b", annotation=Name(id="int", ctx=Load()))
        ],
        args=[arg(arg="c", annotation=Name(id="str", ctx=Load()))],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func6()" in parser.doc[name]


# LLM-generated content at query #11
#--------------------------

```python
def test_Resolver_visit_Attribute():
    from ast import Attribute, Name, Load, Constant, Subscript, Tuple, BinOp, BitOr
    from typing import cast

    resolver = Resolver("test_module", {})

    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

    node = Attribute(Name("collections", Load()), "abc", Load())
    result = resolver.visit_Attribute(node)
    assert result is node

    node = Attribute(Constant("test"), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert result is node

    resolver = Resolver("test_module", {"test_module.typing": "typing"})
    node = Attribute(Name("typing", Load()), "Union", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Union"

    node = Attribute(Name("other_module", Load()), "Something", Load())
    result = resolver.visit_Attribute(node)
    assert result is node


# LLM-generated content at query #12
#--------------------------

```python
def test_Resolver_visit_Subscript():
    from ast import parse, Subscript, Name, Tuple, BinOp, BitOr, Constant, Load
    from typing import cast

    # Test Union transformation
    resolver = Resolver("test_module", {})
    node = cast(Subscript, parse("Union[int, str]").body[0].value)
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert unparse(result) == "(int | str)"

    # Test Optional transformation
    node = cast(Subscript, parse("Optional[int]").body[0].value)
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert unparse(result) == "(int | None)"

    # Test PEP585 deprecated name warning
    import io
    from contextlib import redirect_stderr
    stderr_capture = io.StringIO()
    
    with redirect_stderr(stderr_capture):
        node = cast(Subscript, parse("List[int]").body[0].value)
        result = resolver.visit_Subscript(node)
    
    assert "deprecated name typing.List" in stderr_capture.getvalue()
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test non-typing name remains unchanged
    node = cast(Subscript, parse("Custom[int]").body[0].value)
    result = resolver.visit_Subscript(node)
    assert result is node

    # Test Union with single element
    node = cast(Subscript, parse("Union[int]").body[0].value)
    result = resolver.visit_Subscript(node)
    assert unparse(result) == "int"

    # Test with alias mapping
    resolver = Resolver("test_module", {"test_module.Union": "typing.Union"})
    node = cast(Subscript, parse("Union[int, str]").body[0].value)
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)

    # Test non-Tuple slice in Union
    node = cast(Subscript, parse("Union[int]").body[0].value)
    node.slice = Name("int", Load())  # Replace with non-Tuple
    result = resolver.visit_Subscript(node)
    assert result is node.slice


# LLM-generated content at query #13
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    parser.parse("test_module", "")
    
    # Test basic class with no bases and no members
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.doc[name] = ""
    parser.class_api("test_module", name, bases, body)
    assert "Bases" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]
    assert "Members" not in parser.doc[name]
    
    # Test class with bases
    parser.doc[name] = ""
    bases = [Name("BaseClass", Load()), Name("AnotherBase", Load())]
    parser.class_api("test_module", name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "AnotherBase" in parser.doc[name]
    
    # Test class with enum bases
    parser.doc[name] = ""
    bases = [Attribute(Name("enum", Load()), "Enum", Load())]
    body = [
        AnnAssign(Name("RED", Store()), Constant(1), None, 1),
        AnnAssign(Name("GREEN", Store()), Constant(2), None, 1),
        Assign([Name("BLUE", Store())], Constant(3), None)
    ]
    parser.class_api("test_module", name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "RED" in parser.doc[name]
    assert "GREEN" in parser.doc[name]
    assert "BLUE" in parser.doc[name]
    
    # Test class with public members
    parser.doc[name] = ""
    bases = []
    body = [
        AnnAssign(Name("public_attr", Store()), Name("int", Load()), None, 1),
        AnnAssign(Name("_private_attr", Store()), Name("str", Load()), None, 1),
        Assign([Name("another_public", Store())], Constant("value"), None),
        Assign([Name("_another_private", Store())], Constant(42), None)
    ]
    parser.class_api("test_module", name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "another_public" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]
    assert "_another_private" not in parser.doc[name]
    
    # Test class with type comments
    parser.doc[name] = ""
    body = [
        Assign([Name("with_type", Store())], Constant(1), "int")
    ]
    parser.class_api("test_module", name, bases, body)
    assert "Members" in parser.doc[name]
    assert "with_type" in parser.doc[name]
    assert "int" in parser.doc[name]
    
    # Test class with member deletion
    parser.doc[name] = ""
    body = [
        AnnAssign(Name("attr1", Store()), Name("int", Load()), None, 1),
        AnnAssign(Name("attr2", Store()), Name("str", Load()), None, 1),
        Delete([Name("attr1", Del())])
    ]
    parser.class_api("test_module", name, bases, body)
    assert "Members" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "attr1" not in parser.doc[name]
    
    # Test enum with member deletion
    parser.doc[name] = ""
    bases = [Attribute(Name("enum", Load()), "Enum", Load())]
    body = [
        AnnAssign(Name("VAL1", Store()), Constant(1), None, 1),
        AnnAssign(Name("VAL2", Store()), Constant(2), None, 1),
        Delete([Name("VAL1", Del())])
    ]
    parser.class_api("test_module", name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "VAL2" in parser.doc[name]
    assert "VAL1" not in parser.doc[name]
    
    # Test mixed enum and regular members
    parser.doc[name] = ""
    bases = [Attribute(Name("enum", Load()), "Enum", Load())]
    body = [
        AnnAssign(Name("ENUM_VAL", Store()), Constant(1), None, 1),
        AnnAssign(Name("regular_attr", Store()), Name("int", Load()), None, 1)
    ]
    parser.class_api("test_module", name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "ENUM_VAL" in parser.doc[name]
    assert "Members" not in parser.doc[name]
    assert "regular_attr" not in parser.doc[name]
    
    # Test complex base resolution
    parser.doc[name] = ""
    parser.alias["test_module.CustomEnum"] = "enum.Enum"
    bases = [Name("CustomEnum", Load())]
    body = [AnnAssign(Name("OPTION", Store()), Constant(1), None, 1)]
    parser.class_api("test_module", name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "OPTION" in parser.doc[name]


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Attribute():
    resolver = Resolver("test_module", {})
    
    # Test with typing module attribute
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)
    
    # Test with non-typing module attribute
    node = Attribute(value=Name(id="collections", ctx=Load()), attr="abc", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "collections"
    assert result.attr == "abc"
    assert isinstance(result.ctx, Load)
    
    # Test with nested attribute (should not transform)
    node = Attribute(
        value=Attribute(value=Name(id="typing", ctx=Load()), attr="extensions", ctx=Load()),
        attr="Protocol",
        ctx=Load()
    )
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "extensions"
    assert result.attr == "Protocol"
    assert isinstance(result.ctx, Load)
    
    # Test with non-Name value
    node = Attribute(
        value=Call(func=Name(id="get_module", ctx=Load()), args=[], keywords=[]),
        attr="List",
        ctx=Load()
    )
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Call)
    assert result.attr == "List"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #15
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    parser.root["test_module"] = "test_module"
    parser.alias = {}
    
    # Test basic function with no arguments
    name = "test_module.func1"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func1()" in parser.doc[name]
    assert "*Full name:* `test_module.func1`" in parser.doc[name]
    
    # Test function with positional arguments
    name = "test_module.func2"
    node = arguments(
        posonlyargs=[arg("x", None), arg("y", None)],
        args=[arg("z", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(1), Constant(2)]
    )
    returns = Name("int", Load())
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func2()" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "z" in parser.doc[name]
    assert "return" in parser.doc[name]
    
    # Test function with self argument
    name = "test_module.Class.method"
    node = arguments(
        posonlyargs=[],
        args=[arg("self", None), arg("x", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(10)]
    )
    returns = Name("str", Load())
    
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=False)
    assert "method()" in parser.doc[name]
    assert "Self" in parser.doc[name]  # self should be resolved to Self
    
    # Test class method
    name = "test_module.Class.class_method"
    node = arguments(
        posonlyargs=[],
        args=[arg("cls", Name("Type", Load())), arg("x", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name("None", Load())
    
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=True)
    assert "class_method()" in parser.doc[name]
    assert "type[Self]" in parser.doc[name]  # cls should be resolved to type[Self]
    
    # Test function with varargs and kwargs
    name = "test_module.func3"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg("args", None),
        kwonlyargs=[arg("x", None), arg("y", None)],
        kw_defaults=[Constant(1), Constant(2)],
        kwarg=arg("kwargs", None),
        defaults=[]
    )
    returns = Constant(None)
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func3()" in parser.doc[name]
    assert "*args" in parser.doc[name]
    assert "**kwargs" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    
    # Test function with type annotations
    name = "test_module.func4"
    node = arguments(
        posonlyargs=[],
        args=[arg("x", Name("int", Load())), arg("y", Name("str", Load()))],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Subscript(Name("list", Load()), Name("int", Load()), Load())
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func4()" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "str" in parser.doc[name]
    assert "list[int]" in parser.doc[name]
    
    # Test function with complex defaults
    name = "test_module.func5"
    node = arguments(
        posonlyargs=[],
        args=[arg("x", None), arg("y", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[List([Constant(1), Constant(2)], Load()), Dict([], [])]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func5()" in parser.doc[name]
    assert "[1, 2]" in parser.doc[name] or "list" in parser.doc[name]
    
    # Test function with kwonlyargs and asterisk
    name = "test_module.func6"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg("x", None), arg("y", None)],
        kw_defaults=[Constant(1), None],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func6()" in parser.doc[name]
    assert "*" in parser.doc[name]  * should appear before kwonlyargs
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]


# LLM-generated content at query #16
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    parser.parse("test_module", "")
    
    # Test basic class with no bases and no members
    class_name = "test_module.TestClass"
    parser.doc[class_name] = "### class TestClass\n\n*Full name:* `{}`\n\n"
    parser.level[class_name] = 0
    parser.root[class_name] = "test_module"
    
    bases = []
    body = []
    parser.class_api("test_module", class_name, bases, body)
    
    assert "Bases" not in parser.doc[class_name]
    assert "Enums" not in parser.doc[class_name]
    assert "Members" not in parser.doc[class_name]
    
    # Test class with bases
    class_name2 = "test_module.TestClass2"
    parser.doc[class_name2] = "### class TestClass2\n\n*Full name:* `{}`\n\n"
    parser.level[class_name2] = 0
    parser.root[class_name2] = "test_module"
    
    bases = [Name("BaseClass", Load()), Name("AnotherBase", Load())]
    body = []
    parser.class_api("test_module", class_name2, bases, body)
    
    assert "Bases" in parser.doc[class_name2]
    assert "BaseClass" in parser.doc[class_name2]
    assert "AnotherBase" in parser.doc[class_name2]
    
    # Test class with enum bases
    class_name3 = "test_module.TestEnum"
    parser.doc[class_name3] = "### class TestEnum\n\n*Full name:* `{}`\n\n"
    parser.level[class_name3] = 0
    parser.root[class_name3] = "test_module"
    
    bases = [Attribute(Name("enum", Load()), "Enum", Load())]
    body = [
        AnnAssign(Name("RED", Store()), Constant(1), None, 1),
        AnnAssign(Name("GREEN", Store()), Constant(2), None, 1),
        AnnAssign(Name("BLUE", Store()), Constant(3), None, 1)
    ]
    parser.class_api("test_module", class_name3, bases, body)
    
    assert "Enums" in parser.doc[class_name3]
    assert "RED" in parser.doc[class_name3]
    assert "GREEN" in parser.doc[class_name3]
    assert "BLUE" in parser.doc[class_name3]
    
    # Test class with members (AnnAssign)
    class_name4 = "test_module.TestClassWithMembers"
    parser.doc[class_name4] = "### class TestClassWithMembers\n\n*Full name:* `{}`\n\n"
    parser.level[class_name4] = 0
    parser.root[class_name4] = "test_module"
    
    bases = []
    body = [
        AnnAssign(Name("public_attr", Store()), Name("int", Load()), None, 1),
        AnnAssign(Name("_private_attr", Store()), Name("str", Load()), None, 1),
        AnnAssign(Name("protected_attr", Store()), Name("float", Load()), None, 1)
    ]
    parser.class_api("test_module", class_name4, bases, body)
    
    assert "Members" in parser.doc[class_name4]
    assert "public_attr" in parser.doc[class_name4]
    assert "_private_attr" not in parser.doc[class_name4]
    assert "protected_attr" not in parser.doc[class_name4]
    assert "int" in parser.doc[class_name4]
    
    # Test class with members (Assign with type comment)
    class_name5 = "test_module.TestClassWithAssign"
    parser.doc[class_name5] = "### class TestClassWithAssign\n\n*Full name:* `{}`\n\n"
    parser.level[class_name5] = 0
    parser.root[class_name5] = "test_module"
    
    bases = []
    body = [
        Assign([Name("attr1", Store())], Constant("value"), None, "# type: str"),
        Assign([Name("attr2", Store())], Constant(123), None, "# type: int")
    ]
    parser.class_api("test_module", class_name5, bases, body)
    
    assert "Members" in parser.doc[class_name5]
    assert "attr1" in parser.doc[class_name5]
    assert "attr2" in parser.doc[class_name5]
    assert "str" in parser.doc[class_name5]
    assert "int" in parser.doc[class_name5]
    
    # Test class with Delete statements
    class_name6 = "test_module.TestClassWithDelete"
    parser.doc[class_name6] = "### class TestClassWithDelete\n\n*Full name:* `{}`\n\n"
    parser.level[class_name6] = 0
    parser.root[class_name6] = "test_module"
    
    bases = [Attribute(Name("enum", Load()), "Enum", Load())]
    body = [
        AnnAssign(Name("ITEM1", Store()), Constant(1), None, 1),
        AnnAssign(Name("ITEM2", Store()), Constant(2), None, 1),
        Delete([Name("ITEM2", Del())]),
        AnnAssign(Name("public_attr", Store()), Name("int", Load()), None, 1),
        Delete([Name("public_attr", Del())])
    ]
    parser.class_api("test_module", class_name6, bases, body)
    
    assert "Enums" in parser.doc[class_name6]
    assert "ITEM1" in parser.doc[class_name6]
    assert "ITEM2" not in parser.doc[class_name6]
    assert "Members" not in parser.doc[class_name6]
    
    # Test mixed enum and regular members
    class_name7 = "test_module.TestMixed"
    parser.doc[class_name7] = "### class TestMixed\n\n*Full name:* `{}`\n\n"
    parser.level[class_name7] = 0
    parser.root[class_name7] = "test_module"
    
    bases = [Name("object", Load())]
    body = [
        AnnAssign(Name("CONSTANT", Store()), Constant(100), None, 1),
        AnnAssign(Name("regular_attr", Store()), Name("str", Load()), None, 1)
    ]
    parser.class_api("test_module", class_name7, bases, body)
    
    assert "Enums" not in parser.doc[class_name7]
    assert "Members" in parser.doc[class_name7]
    assert "regular_attr" in parser.doc[class_name7]
    assert "CONSTANT" not in parser.doc[class_name7]


# LLM-generated content at query #17
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    int_node = Constant(value=42)
    result = resolver.visit_Constant(int_node)
    assert result is int_node
    
    # Test with string constant that is not valid Python
    invalid_str_node = Constant(value="not valid python code")
    result = resolver.visit_Constant(invalid_str_node)
    assert result is invalid_str_node
    
    # Test with valid Python string expression
    valid_str_node = Constant(value="List[int]")
    result = resolver.visit_Constant(valid_str_node)
    assert isinstance(result, Name)
    assert result.id == "List"
    
    # Test with string containing attribute access
    attr_str_node = Constant(value="typing.List")
    result = resolver.visit_Constant(attr_str_node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Name)
    assert result.value.id == "typing"
    assert result.attr == "List"
    
    # Test with empty string
    empty_str_node = Constant(value="")
    result = resolver.visit_Constant(empty_str_node)
    assert result is empty_str_node
    
    # Test with string that parses to a complex expression
    complex_str_node = Constant(value="Union[str, int]")
    result = resolver.visit_Constant(complex_str_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "str"
    assert isinstance(result.right, Name)
    assert result.right.id == "int"
    assert isinstance(result.op, BitOr)


# LLM-generated content at query #18
#--------------------------

```python
def test_Resolver_visit_Subscript():
    resolver = Resolver("test_module", {})
    
    # Test PEP585 deprecation warning
    subscript_node = Subscript(
        value=Name(id="typing.List", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(subscript_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    
    # Test typing.Union conversion to BitOr
    union_subscript = Subscript(
        value=Name(id="typing.Union", ctx=Load()),
        slice=Tuple(elts=[
            Name(id="int", ctx=Load()),
            Name(id="str", ctx=Load())
        ], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_subscript)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test typing.Optional conversion
    optional_subscript = Subscript(
        value=Name(id="typing.Optional", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(optional_subscript)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test non-typing subscript remains unchanged
    custom_subscript = Subscript(
        value=Name(id="CustomType", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(custom_subscript)
    assert result is custom_subscript
    
    # Test Union with non-Tuple slice
    union_non_tuple = Subscript(
        value=Name(id="typing.Union", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_non_tuple)
    assert result is union_non_tuple
    
    # Test with alias mapping
    resolver_with_alias = Resolver("test_module", {
        "test_module.typing.List": "typing.List"
    })
    result = resolver_with_alias.visit_Subscript(subscript_node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"


# LLM-generated content at query #19
#--------------------------

```python
def test_Resolver_visit_Attribute():
    resolver = Resolver("test_module", {})
    
    # Test with typing.* prefix
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)
    
    # Test without typing.* prefix
    node = Attribute(value=Name(id="collections", ctx=Load()), attr="abc", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Name)
    assert result.value.id == "collections"
    assert result.attr == "abc"
    
    # Test with non-Name value
    node = Attribute(value=Attribute(value=Name(id="typing", ctx=Load()), attr="extensions", ctx=Load()), attr="Protocol", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Attribute)
    assert result.value.value.id == "typing"
    assert result.value.attr == "extensions"
    assert result.attr == "Protocol"


# LLM-generated content at query #20
#--------------------------

```python
def test_Resolver_visit_Constant():
    from ast import Constant, parse, Expr, Name, Load, Attribute, Call, List, Tuple, Set, Dict
    from typing import cast
    
    # Test with non-string constant
    resolver = Resolver("test_module", {})
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result is node
    
    # Test with string constant that's not a valid name
    node = Constant(value="not a valid name syntax")
    result = resolver.visit_Constant(node)
    assert result is node
    
    # Test with string constant that's a simple name
    resolver = Resolver("test_module", {"test_module.MyClass": "MyClass"})
    node = Constant(value="MyClass")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "MyClass"
    assert isinstance(result.ctx, Load)
    
    # Test with string constant that's an attribute
    resolver = Resolver("test_module", {"test_module.submodule.Class": "submodule.Class"})
    node = Constant(value="submodule.Class")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Attribute)
    assert result.attr == "Class"
    
    # Test with string constant that has alias
    resolver = Resolver("test_module", {"test_module.OldName": "NewName"})
    node = Constant(value="OldName")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "NewName"
    
    # Test with string constant that's a complex expression
    resolver = Resolver("test_module", {})
    node = Constant(value="List[int]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "List"
    
    # Test with string constant that's a call expression
    resolver = Resolver("test_module", {})
    node = Constant(value="TypeVar('T')")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Call)
    assert isinstance(result.func, Name)
    assert result.func.id == "TypeVar"
    
    # Test with empty string constant
    node = Constant(value="")
    result = resolver.visit_Constant(node)
    assert result is node
    
    # Test with string constant containing special characters
    node = Constant(value="my-name")
    result = resolver.visit_Constant(node)
    assert result is node
    
    # Test with self_ty parameter
    resolver = Resolver("test_module", {}, self_ty="SelfType")
    node = Constant(value="SelfType")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #21
#--------------------------

```python
def test_Resolver_visit_Attribute():
    resolver = Resolver("test_module", {})
    
    # Test case 1: Attribute with typing prefix should be converted to Name
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)
    
    # Test case 2: Attribute without typing prefix should remain unchanged
    node = Attribute(value=Name(id="collections", ctx=Load()), attr="abc", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Name)
    assert result.value.id == "collections"
    assert result.attr == "abc"
    assert isinstance(result.ctx, Load)
    
    # Test case 3: Attribute with non-Name value should remain unchanged
    node = Attribute(value=Constant(value="test"), attr="attr", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Constant)
    assert result.value.value == "test"
    assert result.attr == "attr"
    assert isinstance(result.ctx, Load)
    
    # Test case 4: Test with alias mapping
    resolver = Resolver("test_module", {"test_module.typing": "typing"})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Dict", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Dict"
    
    # Test case 5: Test with different context
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Optional", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Optional"


# LLM-generated content at query #22
#--------------------------

```python
def test_walk_body():
    # Test with simple body
    body = [Expr(Constant(1)), Expr(Constant(2))]
    result = list(walk_body(body))
    assert result == body

    # Test with If statement
    if_node = If(
        test=Constant(True),
        body=[Expr(Constant(3)), Expr(Constant(4))],
        orelse=[Expr(Constant(5))]
    )
    body = [Expr(Constant(1)), if_node, Expr(Constant(2))]
    result = list(walk_body(body))
    assert len(result) == 5
    assert result[0] == body[0]
    assert result[1] == if_node.body[0]
    assert result[2] == if_node.body[1]
    assert result[3] == if_node.orelse[0]
    assert result[4] == body[2]

    # Test with nested If statements
    nested_if = If(
        test=Constant(False),
        body=[Expr(Constant(6))],
        orelse=[
            If(
                test=Constant(True),
                body=[Expr(Constant(7))],
                orelse=[]
            )
        ]
    )
    body = [nested_if]
    result = list(walk_body(body))
    assert len(result) == 2
    assert result[0] == nested_if.body[0]
    assert result[1] == nested_if.orelse[0].body[0]

    # Test with Try statement
    try_node = Try(
        body=[Expr(Constant(8))],
        handlers=[arg(name='e', annotation=None)],
        orelse=[Expr(Constant(9))],
        finalbody=[Expr(Constant(10))]
    )
    body = [try_node]
    result = list(walk_body(body))
    assert len(result) == 3
    assert result[0] == try_node.body[0]
    assert result[1] == try_node.orelse[0]
    assert result[2] == try_node.finalbody[0]

    # Test with Try statement with handler body
    try_node = Try(
        body=[Expr(Constant(11))],
        handlers=[arg(name='e', annotation=None, body=[Expr(Constant(12))])],
        orelse=[],
        finalbody=[]
    )
    body = [try_node]
    result = list(walk_body(body))
    assert len(result) == 2
    assert result[0] == try_node.body[0]
    assert result[1] == try_node.handlers[0].body[0]

    # Test with mixed complex structure
    if_node = If(
        test=Constant(True),
        body=[
            Try(
                body=[Expr(Constant(13))],
                handlers=[],
                orelse=[],
                finalbody=[]
            )
        ],
        orelse=[]
    )
    body = [Expr(Constant(14)), if_node, Expr(Constant(15))]
    result = list(walk_body(body))
    assert len(result) == 3
    assert result[0] == body[0]
    assert result[1] == if_node.body[0].body[0]
    assert result[2] == body[2]

    # Test with empty body
    result = list(walk_body([]))
    assert result == []

    # Test with only If statement having empty bodies
    if_node = If(
        test=Constant(True),
        body=[],
        orelse=[]
    )
    body = [if_node]
    result = list(walk_body(body))
    assert result == []


# LLM-generated content at query #23
#--------------------------

```python
def test_Parser_api():
    parser = Parser()
    parser.parse("test_module", """
def simple_func():
    \"\"\"Simple function.\"\"\"
    pass

async def async_func():
    \"\"\"Async function.\"\"\"
    pass

class TestClass:
    \"\"\"Test class.\"\"\"
    pass
""")
    
    assert "test_module.simple_func" in parser.doc
    assert "### simple_func()" in parser.doc["test_module.simple_func"]
    assert "*Full name:* `test_module.simple_func`" in parser.doc["test_module.simple_func"]
    
    assert "test_module.async_func" in parser.doc
    assert "### async async_func()" in parser.doc["test_module.async_func"]
    
    assert "test_module.TestClass" in parser.doc
    assert "### class TestClass" in parser.doc["test_module.TestClass"]
    
    assert "Simple function." in parser.docstring["test_module.simple_func"]
    assert "Async function." in parser.docstring["test_module.async_func"]
    assert "Test class." in parser.docstring["test_module.TestClass"]
    
    parser = Parser()
    parser.parse("nested", """
class Outer:
    class Inner:
        \"\"\"Inner class.\"\"\"
        pass
    
    def method(self):
        \"\"\"Method.\"\"\"
        pass
""")
    
    assert "nested.Outer" in parser.doc
    assert "nested.Outer.Inner" in parser.doc
    assert "nested.Outer.method" in parser.doc
    
    assert "#### class Inner" in parser.doc["nested.Outer.Inner"]
    assert "#### method()" in parser.doc["nested.Outer.method"]
    
    parser = Parser()
    parser.parse("decorated", """
@decorator1
@decorator2
def decorated_func():
    pass
""")
    
    assert "decorated.decorated_func" in parser.doc
    assert "@decorator1" in parser.doc["decorated.decorated_func"]
    assert "@decorator2" in parser.doc["decorated.decorated_func"]
    
    parser = Parser(link=True)
    parser.parse("linked", """
def linked_func():
    pass
""")
    
    assert '<a id="linked.linked_func"></a>' in parser.doc["linked.linked_func"]
    
    parser = Parser(link=False)
    parser.parse("unlinked", """
def unlinked_func():
    pass
""")
    
    assert '<a id="unlinked.unlinked_func"></a>' not in parser.doc["unlinked.unlinked_func"]


# LLM-generated content at query #24
#--------------------------

```python
def test_Resolver_visit_Name():
    from ast import Name, Load, parse, Expr, Call, Constant
    from typing import cast

    # Test 1: Replace with self type
    resolver = Resolver("module", {}, "SelfType")
    node = Name(id="SelfType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)

    # Test 2: Name exists in alias mapping
    alias = {"module.name": "other_module.OtherName"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "OtherName"
    assert isinstance(result.ctx, Load)

    # Test 3: Name exists in alias but maps to itself (no change)
    alias = {"module.name": "module.name"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"
    assert isinstance(result.ctx, Load)

    # Test 4: Name not in alias mapping
    resolver = Resolver("module", {}, "")
    node = Name(id="unknown", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node

    # Test 5: TypeVar in alias should not be replaced
    alias = {"module.T": "typing.TypeVar"}
    resolver = Resolver("module", alias, "")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"
    assert isinstance(result.ctx, Load)

    # Test 6: Complex alias expression
    alias = {"module.name": "other_module.submodule.ClassName"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Attribute)
    assert result.attr == "ClassName"
    assert isinstance(result.value, Attribute)
    assert result.value.attr == "submodule"
    assert isinstance(result.value.value, Name)
    assert result.value.value.id == "other_module"

    # Test 7: Nested alias with call expression
    alias = {"module.name": "some_func()"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Call)
    assert isinstance(result.func, Name)
    assert result.func.id == "some_func"

    # Test 8: Different root module
    alias = {"other_module.name": "replacement"}
    resolver = Resolver("other_module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "replacement"
    assert isinstance(result.ctx, Load)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    parser.root["test_module"] = "test_module"
    parser.alias = {}
    
    # Test basic function annotation
    args = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=Name(id="bool", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int", "str", "bool"]
    
    # Test with Self parameter (has_self=True)
    args_with_self = [
        arg(arg="self", annotation=Name(id="Self", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_with_self, has_self=True, cls_method=False))
    assert result == ["Self", "int", "None"]
    
    # Test with classmethod (cls_method=True)
    args_classmethod = [
        arg(arg="cls", annotation=Subscript(value=Name(id="type", ctx=Load()), slice=Name(id="Self", ctx=Load()), ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_classmethod, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int", "None"]
    
    # Test with vararg separator (*)
    args_with_star = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="*", annotation=None),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_with_star, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "None"]
    
    # Test with missing annotations (should return ANY)
    args_no_ann = [
        arg(arg="x", annotation=None),
        arg(arg="y", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result = list(parser.func_ann("test_module", args_no_ann, has_self=False, cls_method=False))
    assert result == [ANY, ANY, ANY]
    
    # Test with alias resolution
    parser.alias["test_module.CustomType"] = "typing.CustomType"
    args_with_alias = [
        arg(arg="x", annotation=Name(id="CustomType", ctx=Load())),
        arg(arg="return", annotation=Name(id="int", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_with_alias, has_self=False, cls_method=False))
    assert result == ["CustomType", "int"]
    
    # Test with self_ty parameter in resolve
    args_self_ty = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_self_ty, has_self=True, cls_method=False))
    assert result == ["Self", "int", "None"]
    
    # Test mixed annotations with vararg and kwarg
    args_complex = [
        arg(arg="self", annotation=Name(id="Self", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="*args", annotation=Name(id="Any", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="**kwargs", annotation=Name(id="dict", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args_complex, has_self=True, cls_method=False))
    assert result == ["Self", "int", "Any", "str", "dict", "None"]


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_class_api():
    # Test basic class with no bases and no members
    parser = Parser()
    parser.parse("test_module", "class EmptyClass:\n    pass")
    
    name = "test_module.EmptyClass"
    assert name in parser.doc
    assert "class EmptyClass" in parser.doc[name]
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]
    
    # Test class with bases
    parser = Parser()
    parser.parse("test_module", "class ChildClass(Base1, Base2):\n    pass")
    
    name = "test_module.ChildClass"
    assert name in parser.doc
    assert "class ChildClass" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "Base1" in parser.doc[name]
    assert "Base2" in parser.doc[name]
    
    # Test class with annotated members
    parser = Parser()
    parser.parse("test_module", """
class MyClass:
    x: int
    y: str
    _private: float
""")
    
    name = "test_module.MyClass"
    assert name in parser.doc
    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "str" in parser.doc[name]
    assert "_private" not in parser.doc[name]
    
    # Test class with assigned members
    parser = Parser()
    parser.parse("test_module", """
class MyClass:
    CONSTANT = 42
    name = "test"
    _hidden = True
""")
    
    name = "test_module.MyClass"
    assert name in parser.doc
    assert "Members" in parser.doc[name]
    assert "CONSTANT" in parser.doc[name]
    assert "name" in parser.doc[name]
    assert "_hidden" not in parser.doc[name]
    
    # Test enum class
    parser = Parser()
    parser.parse("test_module", """
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
""")
    
    name = "test_module.Color"
    assert name in parser.doc
    assert "Enums" in parser.doc[name]
    assert "RED" in parser.doc[name]
    assert "GREEN" in parser.doc[name]
    assert "BLUE" in parser.doc[name]
    assert "Members" not in parser.doc[name]
    
    # Test class with deleted members
    parser = Parser()
    parser.parse("test_module", """
class MyClass:
    x: int
    y: str
    
    def __init__(self):
        del self.x
""")
    
    name = "test_module.MyClass"
    assert name in parser.doc
    assert "Members" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "x" not in parser.doc[name]
    
    # Test class with type comment assignments
    parser = Parser()
    parser.parse("test_module", """
class MyClass:
    x = 42  # type: int
    y = "hello"  # type: str
""")
    
    name = "test_module.MyClass"
    assert name in parser.doc
    assert "Members" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "str" in parser.doc[name]
    
    # Test class with mixed bases including enum
    parser = Parser()
    parser.parse("test_module", """
from enum import IntEnum
class MyEnum(IntEnum, BaseClass):
    VALUE1 = 1
    VALUE2 = 2
""")
    
    name = "test_module.MyEnum"
    assert name in parser.doc
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]
    assert "Bases" in parser.doc[name]
    assert "IntEnum" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    
    # Test class with no public members
    parser = Parser()
    parser.parse("test_module", """
class PrivateClass:
    _private1: int
    _private2: str
    __very_private = True
""")
    
    name = "test_module.PrivateClass"
    assert name in parser.doc
    assert "Members" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]


# LLM-generated content at query #3
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with Name target
    node = AnnAssign(
        target=Name(id="TYPE_ALIAS", ctx=Store()),
        annotation=Name(id="List", ctx=Load()),
        value=Name(id="list", ctx=Load()),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TYPE_ALIAS"] == "list"
    assert parser.const["test_module.TYPE_ALIAS"] == "List"
    
    # Test Assign with single Name target and type comment
    node = Assign(
        targets=[Name(id="CONSTANT", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.CONSTANT"] == "42"
    assert parser.const["test_module.CONSTANT"] == "int"
    
    # Test Assign with single Name target without type comment
    node = Assign(
        targets=[Name(id="ANOTHER", ctx=Store())],
        value=List(elts=[Constant(value=1), Constant(value=2)], ctx=Load()),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.alias["test_module.ANOTHER"] == "[1, 2]"
    assert parser.const["test_module.ANOTHER"] == "list[int]"
    
    # Test __all__ assignment with Tuple
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="func1"), Constant(value="Class1")], ctx=Load())
    )
    parser.globals(root, node)
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.Class1" in parser.imp[root]
    
    # Test __all__ assignment with List
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="var1"), Constant(value="var2")], ctx=Load())
    )
    parser.globals(root, node)
    assert "test_module.var1" in parser.imp[root]
    assert "test_module.var2" in parser.imp[root]
    
    # Test non-uppercase assignment (should not be added to const)
    node = AnnAssign(
        target=Name(id="lowercase", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=Constant(value="test"),
        simple=1
    )
    parser.globals(root, node)
    assert "test_module.lowercase" not in parser.const
    
    # Test multiple targets assignment (should be ignored)
    node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias
    
    # Test non-Name target (should be ignored)
    node = AnnAssign(
        target=Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=5),
        simple=0
    )
    parser.globals(root, node)
    assert "test_module.obj.attr" not in parser.alias


# LLM-generated content at query #4
#--------------------------

```python
def test_walk_body():
    # Test with simple body
    body = [Expr(Constant(1)), Expr(Constant(2))]
    result = list(walk_body(body))
    assert result == body

    # Test with If statement
    if_node = If(
        test=Constant(True),
        body=[Expr(Constant(3)), Expr(Constant(4))],
        orelse=[Expr(Constant(5))]
    )
    body = [Expr(Constant(1)), if_node, Expr(Constant(2))]
    result = list(walk_body(body))
    assert len(result) == 5
    assert result[0] == body[0]
    assert result[1] == if_node.body[0]
    assert result[2] == if_node.body[1]
    assert result[3] == if_node.orelse[0]
    assert result[4] == body[2]

    # Test with nested If statements
    nested_if = If(
        test=Constant(False),
        body=[Expr(Constant(6))],
        orelse=[If(
            test=Constant(True),
            body=[Expr(Constant(7))],
            orelse=[]
        )]
    )
    body = [nested_if]
    result = list(walk_body(body))
    assert len(result) == 2
    assert result[0] == nested_if.body[0]
    assert result[1] == nested_if.orelse[0].body[0]

    # Test with Try statement
    try_node = Try(
        body=[Expr(Constant(8))],
        handlers=[arg(arg='e', annotation=None)],
        orelse=[Expr(Constant(9))],
        finalbody=[Expr(Constant(10))]
    )
    body = [try_node]
    result = list(walk_body(body))
    assert len(result) == 3
    assert result[0] == try_node.body[0]
    assert result[1] == try_node.orelse[0]
    assert result[2] == try_node.finalbody[0]

    # Test with Try statement with handler body
    try_node = Try(
        body=[Expr(Constant(11))],
        handlers=[arg(arg='e', annotation=None, body=[Expr(Constant(12))])],
        orelse=[],
        finalbody=[]
    )
    body = [try_node]
    result = list(walk_body(body))
    assert len(result) == 2
    assert result[0] == try_node.body[0]
    assert result[1] == try_node.handlers[0].body[0]

    # Test with mixed statements
    if_node = If(
        test=Constant(True),
        body=[Expr(Constant(13))],
        orelse=[]
    )
    try_node = Try(
        body=[Expr(Constant(14))],
        handlers=[],
        orelse=[Expr(Constant(15))],
        finalbody=[Expr(Constant(16))]
    )
    body = [Expr(Constant(17)), if_node, try_node]
    result = list(walk_body(body))
    assert len(result) == 5
    assert result[0] == body[0]
    assert result[1] == if_node.body[0]
    assert result[2] == try_node.body[0]
    assert result[3] == try_node.orelse[0]
    assert result[4] == try_node.finalbody[0]

    # Test with empty body
    result = list(walk_body([]))
    assert result == []

    # Test with only If statement having empty bodies
    if_node = If(
        test=Constant(True),
        body=[],
        orelse=[]
    )
    result = list(walk_body([if_node]))
    assert result == []


# LLM-generated content at query #5
#--------------------------

```python
def test_Resolver_visit_Name():
    from ast import Name, Load, parse, Expr, Call, Constant
    from typing import cast

    # Test 1: Replace with self type
    resolver = Resolver("module", {}, "SelfType")
    node = Name(id="SelfType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)

    # Test 2: Replace with alias
    alias = {"module.name": "alias_name"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "alias_name"
    assert isinstance(result.ctx, Load)

    # Test 3: Alias points to itself (no change)
    alias = {"module.name": "module.name"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"
    assert isinstance(result.ctx, Load)

    # Test 4: No alias found
    resolver = Resolver("module", {}, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node

    # Test 5: TypeVar should not be replaced
    alias = {"module.TypeVar": "typing.TypeVar"}
    resolver = Resolver("module", alias, "")
    node = Name(id="TypeVar", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"
    assert isinstance(result.ctx, Load)

    # Test 6: Complex alias expression
    alias = {"module.name": "List[int]"}
    resolver = Resolver("module", alias, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "List"
    assert isinstance(result.slice, Constant)
    assert result.slice.value == "int"

    # Test 7: Nested alias resolution
    alias = {"module.outer": "inner", "module.inner": "final"}
    resolver = Resolver("module", alias, "")
    node = Name(id="outer", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "final"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #6
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    int_node = Constant(value=42)
    result = resolver.visit_Constant(int_node)
    assert result is int_node
    
    # Test with string that is not valid Python
    invalid_str_node = Constant(value="not valid python code")
    result = resolver.visit_Constant(invalid_str_node)
    assert result is invalid_str_node
    
    # Test with valid Python string expression
    valid_str_node = Constant(value="List[int]")
    result = resolver.visit_Constant(valid_str_node)
    assert isinstance(result, Expr)
    
    # Test with string containing valid name
    name_str_node = Constant(value="SomeType")
    result = resolver.visit_Constant(name_str_node)
    assert isinstance(result, Name)
    assert result.id == "SomeType"
    
    # Test with string containing attribute access
    attr_str_node = Constant(value="typing.List")
    result = resolver.visit_Constant(attr_str_node)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    
    # Test with alias resolution
    resolver.alias = {"test_module.SomeType": "typing.List[int]"}
    alias_str_node = Constant(value="SomeType")
    result = resolver.visit_Constant(alias_str_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "List"
    
    # Test with self type replacement
    resolver.self_ty = "SelfType"
    self_str_node = Constant(value="SelfType")
    result = resolver.visit_Constant(self_str_node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    parser.root = {"test_module": "test_module"}
    parser.alias = {}
    
    # Test basic function annotation
    args = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=Name(id="bool", ctx=Load()))
    ]
    
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int", "str", "bool"]
    
    # Test with Self parameter (has_self=True)
    args_with_self = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    
    result = list(parser.func_ann("test_module", args_with_self, has_self=True, cls_method=False))
    assert result == ["Self", "int", "None"]
    
    # Test with classmethod (cls_method=True)
    result = list(parser.func_ann("test_module", args_with_self, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int", "None"]
    
    # Test with vararg separator (*)
    args_with_star = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="*", annotation=None),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    
    result = list(parser.func_ann("test_module", args_with_star, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "None"]
    
    # Test with missing annotation (should return ANY)
    args_missing_ann = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=None),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    
    result = list(parser.func_ann("test_module", args_missing_ann, has_self=False, cls_method=False))
    assert result == ["int", "Any", "None"]
    
    # Test with self_ty parameter in resolve
    parser.alias = {"test_module.MyClass": "MyClass"}
    args_self_ty = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="MyClass", ctx=Load()))
    ]
    
    result = list(parser.func_ann("test_module", args_self_ty, has_self=True, cls_method=False))
    assert result == ["Self", "int", "MyClass"]
    
    # Test with complex annotation
    args_complex = [
        arg(arg="items", annotation=Subscript(
            value=Name(id="List", ctx=Load()),
            slice=Name(id="str", ctx=Load()),
            ctx=Load()
        )),
        arg(arg="return", annotation=Subscript(
            value=Name(id="Dict", ctx=Load()),
            slice=Tuple(
                elts=[Name(id="str", ctx=Load()), Name(id="int", ctx=Load())],
                ctx=Load()
            ),
            ctx=Load()
        ))
    ]
    
    result = list(parser.func_ann("test_module", args_complex, has_self=False, cls_method=False))
    assert result == ["List[str]", "Dict[str, int]"]
    
    # Test empty args
    result = list(parser.func_ann("test_module", [], has_self=False, cls_method=False))
    assert result == []


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    parser.root["test_module"] = "test_module"
    parser.alias = {}
    
    # Test basic function with no arguments
    name = "test_module.func1"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "func1()" in parser.doc[name]
    assert "*Full name:* `test_module.func1`" in parser.doc[name]
    
    # Test function with positional arguments
    name = "test_module.func2"
    node = arguments(
        posonlyargs=[arg("x", None), arg("y", None)],
        args=[arg("z", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "z" in parser.doc[name]
    assert "return" in parser.doc[name]
    
    # Test function with defaults
    name = "test_module.func3"
    node = arguments(
        posonlyargs=[],
        args=[arg("a", None), arg("b", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(1), Constant("test")]
    )
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "a" in parser.doc[name]
    assert "b" in parser.doc[name]
    
    # Test function with vararg and kwarg
    name = "test_module.func4"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg("args", None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg("kwargs", None),
        defaults=[]
    )
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "*args" in parser.doc[name]
    assert "**kwargs" in parser.doc[name]
    
    # Test function with kwonlyargs
    name = "test_module.func5"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg("x", None), arg("y", None)],
        kw_defaults=[Constant(1), Constant(2)],
        kwarg=None,
        defaults=[]
    )
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    
    # Test function with posonly separator
    name = "test_module.func6"
    node = arguments(
        posonlyargs=[arg("a", None)],
        args=[arg("b", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "/" in parser.doc[name]
    
    # Test method with self
    name = "test_module.Class.method"
    node = arguments(
        posonlyargs=[],
        args=[arg("self", None), arg("x", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=False)
    assert "self" in parser.doc[name]
    assert "x" in parser.doc[name]
    
    # Test classmethod
    name = "test_module.Class.class_method"
    node = arguments(
        posonlyargs=[],
        args=[arg("cls", None), arg("x", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=True)
    assert "cls" in parser.doc[name]
    assert "x" in parser.doc[name]
    
    # Test function with return annotation
    name = "test_module.func7"
    node = arguments(
        posonlyargs=[],
        args=[arg("x", None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name("int", Load())
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "return" in parser.doc[name]
    
    # Test function with all argument types
    name = "test_module.func8"
    node = arguments(
        posonlyargs=[arg("a", None)],
        args=[arg("b", None), arg("c", None)],
        vararg=arg("args", None),
        kwonlyargs=[arg("d", None), arg("e", None)],
        kw_defaults=[Constant(10), Constant(20)],
        kwarg=arg("kwargs", None),
        defaults=[Constant(1)]
    )
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    assert "a" in parser.doc[name]
    assert "b" in parser.doc[name]
    assert "c" in parser.doc[name]
    assert "*args" in parser.doc[name]
    assert "d" in parser.doc[name]
    assert "e" in parser.doc[name]
    assert "**kwargs" in parser.doc[name]
    assert "return" in parser.doc[name]


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_api():
    parser = Parser()
    parser.parse("test_module", """
def simple_function():
    \"\"\"Simple function docstring.\"\"\"
    pass

async def async_function():
    \"\"\"Async function docstring.\"\"\"
    pass

class TestClass:
    \"\"\"Test class docstring.\"\"\"
    def method(self):
        pass
    
    @classmethod
    def class_method(cls):
        pass
    
    @staticmethod
    def static_method():
        pass
    
    class InnerClass:
        \"\"\"Inner class docstring.\"\"\"
        pass
""")
    
    assert "test_module.simple_function" in parser.doc
    assert "simple_function()" in parser.doc["test_module.simple_function"]
    assert "Full name" in parser.doc["test_module.simple_function"]
    
    assert "test_module.async_function" in parser.doc
    assert "async async_function()" in parser.doc["test_module.async_function"]
    
    assert "test_module.TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]
    
    assert "test_module.TestClass.method" in parser.doc
    assert "test_module.TestClass.class_method" in parser.doc
    assert "test_module.TestClass.static_method" in parser.doc
    
    assert "test_module.TestClass.InnerClass" in parser.doc
    assert "class InnerClass" in parser.doc["test_module.TestClass.InnerClass"]
    
    assert "test_module.simple_function" in parser.docstring
    assert "Simple function docstring" in parser.docstring["test_module.simple_function"]
    
    assert "test_module.TestClass" in parser.docstring
    assert "Test class docstring" in parser.docstring["test_module.TestClass"]
    
    assert "test_module.TestClass.InnerClass" in parser.docstring
    assert "Inner class docstring" in parser.docstring["test_module.TestClass.InnerClass"]
    
    parser2 = Parser(link=True, b_level=2, toc=False)
    parser2.parse("another_module", """
class BaseClass:
    pass

class DerivedClass(BaseClass):
    \"\"\"Derived class with base.\"\"\"
    pass
""")
    
    assert "another_module.DerivedClass" in parser2.doc
    assert "class DerivedClass" in parser2.doc["another_module.DerivedClass"]
    assert "Bases" in parser2.doc["another_module.DerivedClass"]
    
    parser3 = Parser()
    parser3.parse("module_with_decorators", """
@property
def computed_property():
    \"\"\"Property docstring.\"\"\"
    return 42

@classmethod
@staticmethod
def multi_decorated():
    pass
""")
    
    assert "module_with_decorators.computed_property" in parser3.doc
    assert "@property" in parser3.doc["module_with_decorators.computed_property"]
    
    assert "module_with_decorators.multi_decorated" in parser3.doc
    assert "@classmethod" in parser3.doc["module_with_decorators.multi_decorated"]
    assert "@staticmethod" in parser3.doc["module_with_decorators.multi_decorated"]


# LLM-generated content at query #10
#--------------------------

```python
def test_Resolver_visit_Subscript():
    resolver = Resolver("test_module", {})
    
    # Test PEP585 deprecation warning
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        node = Subscript(Name("List", Load()), Constant(1), Load())
        result = resolver.visit_Subscript(node)
        assert len(w) == 1
        assert "deprecated name typing.List" in str(w[0].message)
        assert isinstance(result, Subscript)
        assert isinstance(result.value, Name)
        assert result.value.id == "list"
    
    # Test Union conversion to BitOr
    resolver.alias["test_module.Union"] = "typing.Union"
    node = Subscript(
        Name("Union", Load()),
        Tuple([Name("int", Load()), Name("str", Load())], Load()),
        Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Name)
    assert result.right.id == "str"
    
    # Test Optional conversion
    resolver.alias["test_module.Optional"] = "typing.Optional"
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test non-typing subscript remains unchanged
    node = Subscript(Name("Custom", Load()), Name("T", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result is node
    
    # Test subscript with non-Name value
    attr = Attribute(Name("module", Load()), "Class", Load())
    node = Subscript(attr, Name("T", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result is node
    
    # Test Union with single element (should return slice directly)
    node = Subscript(
        Name("Union", Load()),
        Name("int", Load()),
        Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Name)
    assert result.id == "int"
    
    # Test Union with more than two elements
    node = Subscript(
        Name("Union", Load()),
        Tuple([
            Name("int", Load()),
            Name("str", Load()),
            Name("bool", Load())
        ], Load()),
        Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    # Should create nested BinOp structure
    assert isinstance(result.right, Name)
    assert result.right.id == "bool"
    assert isinstance(result.left, BinOp)


# LLM-generated content at query #11
#--------------------------

```python
def test_Parser_load_docstring():
    import sys
    from types import ModuleType
    from typing import get_type_hints
    
    # Mock module with docstrings
    class MockClass:
        """Class docstring."""
        def method(self):
            """Method docstring."""
            pass
        
        @classmethod
        def class_method(cls):
            """Class method docstring."""
            pass
    
    mock_module = ModuleType('test_module')
    mock_module.MockClass = MockClass
    mock_module.__doc__ = "Module docstring"
    
    # Create parser and parse initial structure
    parser = Parser()
    parser.parse('test_module', '''
class MockClass:
    """Class docstring."""
    def method(self):
        """Method docstring."""
        pass
    
    @classmethod
    def class_method(cls):
        """Class method docstring."""
        pass
''')
    
    # Test loading docstrings
    parser.load_docstring('test_module', mock_module)
    
    # Verify module docstring was loaded
    assert 'test_module' in parser.docstring
    assert parser.docstring['test_module'] == 'Module docstring'
    
    # Verify class docstring was loaded
    assert 'test_module.MockClass' in parser.docstring
    assert parser.docstring['test_module.MockClass'] == 'Class docstring'
    
    # Test with missing attribute
    parser.parse('other_module', '''
def some_function():
    pass
''')
    
    other_module = ModuleType('other_module')
    other_module.__doc__ = "Other module doc"
    
    parser.load_docstring('other_module', other_module)
    assert 'other_module' in parser.docstring
    assert parser.docstring['other_module'] == 'Other module doc'
    
    # Test with nested structure
    parser.parse('nested.module', '''
class Outer:
    class Inner:
        """Inner class doc."""
        pass
''')
    
    class Inner:
        """Inner class doc."""
        pass
    
    class Outer:
        Inner = Inner
    
    nested_module = ModuleType('nested.module')
    nested_module.Outer = Outer
    nested_module.__doc__ = "Nested module"
    
    parser.load_docstring('nested.module', nested_module)
    assert 'nested.module' in parser.docstring
    assert parser.docstring['nested.module'] == 'Nested module'
    assert 'nested.module.Outer.Inner' in parser.docstring
    assert parser.docstring['nested.module.Outer.Inner'] == 'Inner class doc.'
    
    # Test with no docstring
    parser.parse('empty_module', '''
class NoDoc:
    pass
''')
    
    class NoDoc:
        pass
    
    empty_module = ModuleType('empty_module')
    empty_module.NoDoc = NoDoc
    
    parser.load_docstring('empty_module', empty_module)
    assert 'empty_module' not in parser.docstring or parser.docstring.get('empty_module') == ''
    
    # Test doctest processing
    parser.parse('doctest_module', '''
def func_with_doctest():
    """This is a doctest.
    
    >>> func_with_doctest()
    'result'
    """
    return 'result'
''')
    
    def func_with_doctest():
        """This is a doctest.
        
        >>> func_with_doctest()
        'result'
        """
        return 'result'
    
    doctest_module = ModuleType('doctest_module')
    doctest_module.func_with_doctest = func_with_doctest
    
    parser.load_docstring('doctest_module', doctest_module)
    assert 'doctest_module.func_with_doctest' in parser.docstring
    assert '>>> func_with_doctest()' in parser.docstring['doctest_module.func_with_doctest']


# LLM-generated content at query #12
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test with self_ty parameter
    resolver = Resolver("test_module", {}, "SelfType")
    node = Name(id="SelfType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)

    # Test with alias mapping
    alias = {"test_module.name1": "typing.List[int]"}
    resolver = Resolver("test_module", alias, "")
    node = Name(id="name1", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "List"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

    # Test with TypeVar in alias (should return original node)
    alias = {"test_module.T": "typing.TypeVar('T')"}
    resolver = Resolver("test_module", alias, "")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node

    # Test without alias mapping
    resolver = Resolver("test_module", {}, "")
    node = Name(id="some_name", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node

    # Test with alias pointing to same name (should return original node)
    alias = {"test_module.name2": "test_module.name2"}
    resolver = Resolver("test_module", alias, "")
    node = Name(id="name2", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node

    # Test with nested alias resolution
    alias = {
        "test_module.outer": "typing.Dict[str, int]",
        "test_module.inner": "test_module.outer"
    }
    resolver = Resolver("test_module", alias, "")
    node = Name(id="inner", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"
    assert isinstance(result.slice, Tuple)
    assert len(result.slice.elts) == 2
    assert isinstance(result.slice.elts[0], Name)
    assert result.slice.elts[0].id == "str"
    assert isinstance(result.slice.elts[1], Name)
    assert result.slice.elts[1].id == "int"


# LLM-generated content at query #13
#--------------------------

```python
def test_doctest():
    # Test with no doctest lines
    doc = "This is a regular docstring."
    result = doctest(doc)
    assert result == doc

    # Test with single doctest line
    doc = ">>> print('hello')"
    result = doctest(doc)
    assert result == "```python\n>>> print('hello')\n```"

    # Test with multiple doctest lines
    doc = ">>> x = 1\n>>> print(x)"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n>>> print(x)\n```"

    # Test with mixed content
    doc = "Some text.\n>>> code()\nMore text."
    result = doctest(doc)
    assert result == "Some text.\n```python\n>>> code()\n```\nMore text."

    # Test with multiline doctest
    doc = ">>> for i in range(3):\n...     print(i)"
    result = doctest(doc)
    assert result == "```python\n>>> for i in range(3):\n...     print(i)\n```"

    # Test with consecutive doctest blocks
    doc = ">>> first()\nText\n>>> second()"
    result = doctest(doc)
    assert result == "```python\n>>> first()\n```\nText\n```python\n>>> second()\n```"

    # Test empty string
    doc = ""
    result = doctest(doc)
    assert result == ""

    # Test with only whitespace
    doc = "   \n\t\n"
    result = doctest(doc)
    assert result == "   \n\t\n"


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test_module", {})
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result is node

    # Test with string that is not valid Python expression
    resolver = Resolver("test_module", {})
    node = Constant(value="not a valid python expression!")
    result = resolver.visit_Constant(node)
    assert result is node

    # Test with string that is valid Python expression
    resolver = Resolver("test_module", {})
    node = Constant(value="test_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "test_name"

    # Test with string that contains valid expression with alias
    resolver = Resolver("test_module", {"test_module.test_name": "other_name"})
    node = Constant(value="test_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "other_name"

    # Test with string that is a TypeVar expression
    resolver = Resolver("test_module", {"test_module.TypeVar": "typing.TypeVar"})
    node = Constant(value="TypeVar")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test with empty string
    resolver = Resolver("test_module", {})
    node = Constant(value="")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == ""

    # Test with string that parses to more complex expression
    resolver = Resolver("test_module", {})
    node = Constant(value="a + b")
    result = resolver.visit_Constant(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "a"
    assert isinstance(result.right, Name)
    assert result.right.id == "b"


# LLM-generated content at query #15
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with Name target
    ann_assign = AnnAssign(
        target=Name(id="TYPE_ALIAS", ctx=Store()),
        annotation=Name(id="List", ctx=Load()),
        value=Name(id="list", ctx=Load()),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert parser.alias["test_module.TYPE_ALIAS"] == "list"
    assert parser.const["test_module.TYPE_ALIAS"] == "List"
    assert parser.root["test_module.TYPE_ALIAS"] == root
    
    # Test Assign with single Name target and type comment
    assign_with_type = Assign(
        targets=[Name(id="CONSTANT", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals(root, assign_with_type)
    assert parser.alias["test_module.CONSTANT"] == "42"
    assert parser.const["test_module.CONSTANT"] == "int"
    
    # Test Assign with single Name target without type comment
    assign_no_type = Assign(
        targets=[Name(id="ANOTHER", ctx=Store())],
        value=Constant(value="string")
    )
    parser.globals(root, assign_no_type)
    assert parser.alias["test_module.ANOTHER"] == "'string'"
    assert parser.const["test_module.ANOTHER"] == "str"
    
    # Test __all__ assignment with Tuple
    all_tuple = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="func1"), Constant(value="Class1")], ctx=Load())
    )
    parser.globals(root, all_tuple)
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.Class1" in parser.imp[root]
    
    # Test __all__ assignment with List
    all_list = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="func2"), Constant(value="attr")], ctx=Load())
    )
    parser.globals(root, all_list)
    assert "test_module.func2" in parser.imp[root]
    assert "test_module.attr" in parser.imp[root]
    
    # Test non-uppercase assignment (should not be added to const)
    lowercase = Assign(
        targets=[Name(id="variable", ctx=Store())],
        value=Constant(value=3.14)
    )
    parser.globals(root, lowercase)
    assert "test_module.variable" in parser.alias
    assert "test_module.variable" not in parser.const
    
    # Test AnnAssign without value (should be ignored)
    ann_no_value = AnnAssign(
        target=Name(id="NO_VALUE", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, ann_no_value)
    assert "test_module.NO_VALUE" not in parser.alias
    
    # Test Assign with multiple targets (should be ignored)
    multi_target = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals(root, multi_target)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias
    
    # Test Assign with non-Name target (should be ignored)
    attr_target = Assign(
        targets=[Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store())],
        value=Constant(value=2)
    )
    parser.globals(root, attr_target)
    assert "test_module.obj.attr" not in parser.alias


# LLM-generated content at query #16
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with Name target
    node = AnnAssign(
        target=Name(id="MY_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.MY_CONST"] == "42"
    assert parser.const["test_module.MY_CONST"] == "int"
    assert parser.root["test_module.MY_CONST"] == root
    
    # Test Assign with Name target and type comment
    node = Assign(
        targets=[Name(id="another_var", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.another_var"] == "'hello'"
    assert parser.const.get("test_module.another_var") is None
    
    # Test Assign with Name target without type comment
    node = Assign(
        targets=[Name(id="no_type", ctx=Store())],
        value=Constant(value=3.14)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.no_type"] == "3.14"
    assert parser.const.get("test_module.no_type") is None
    
    # Test __all__ assignment with Tuple
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[
            Constant(value="func1"),
            Constant(value="ClassA")
        ], ctx=Load())
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.func1", "test_module.ClassA"}
    
    # Test __all__ assignment with List
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[
            Constant(value="var1"),
            Constant(value="CONST2")
        ], ctx=Load())
    )
    parser.globals(root, node)
    assert "test_module.var1" in parser.imp[root]
    assert "test_module.CONST2" in parser.imp[root]
    
    # Test uppercase constant with complex value
    node = AnnAssign(
        target=Name(id="COMPLEX_CONST", ctx=Store()),
        annotation=Name(id="list", ctx=Load()),
        value=List(elts=[Constant(value=1), Constant(value=2)], ctx=Load()),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.COMPLEX_CONST"] == "[1, 2]"
    assert parser.const["test_module.COMPLEX_CONST"] == "list"
    
    # Test non-Name target (should be ignored)
    node = Assign(
        targets=[Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store())],
        value=Constant(value=5)
    )
    parser.globals(root, node)
    assert "test_module.obj.attr" not in parser.alias
    
    # Test multiple targets (should be ignored)
    node = Assign(
        targets=[
            Name(id="a", ctx=Store()),
            Name(id="b", ctx=Store())
        ],
        value=Constant(value=10)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias
    
    # Test AnnAssign without value (should be ignored)
    node = AnnAssign(
        target=Name(id="no_value", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    parser.globals(root, node)
    assert "test_module.no_value" not in parser.alias
    
    # Test constant overwriting
    node1 = AnnAssign(
        target=Name(id="OVERWRITE", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=1),
        simple=1
    )
    parser.globals(root, node1)
    assert parser.const["test_module.OVERWRITE"] == "int"
    
    node2 = AnnAssign(
        target=Name(id="OVERWRITE", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=Constant(value="new"),
        simple=1
    )
    parser.globals(root, node2)
    assert parser.const["test_module.OVERWRITE"] == "str"


# LLM-generated content at query #17
#--------------------------

```python
def test_Resolver_visit_Subscript():
    from ast import parse, Subscript, Name, Tuple, BinOp, BitOr, Constant, Load
    from typing import cast

    # Test Union transformation
    root = "test_module"
    alias = {}
    resolver = Resolver(root, alias)
    
    # Create Union[int, str] annotation
    union_node = cast(Subscript, parse("Union[int, str]").body[0].value)
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.right, Name) and result.right.id == "str"
    
    # Test Optional transformation
    optional_node = cast(Subscript, parse("Optional[int]").body[0].value)
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.right, Constant) and result.right.value is None
    
    # Test PEP585 deprecation warning
    import io
    import sys
    from unittest.mock import patch
    
    list_node = cast(Subscript, parse("List[int]").body[0].value)
    with patch.object(logger, 'warning') as mock_warning:
        result = resolver.visit_Subscript(list_node)
        mock_warning.assert_called_once()
        assert "List" in mock_warning.call_args[0][0]
        assert "list" in mock_warning.call_args[0][0]
    
    # Test non-typing subscript remains unchanged
    custom_node = cast(Subscript, parse("Custom[int]").body[0].value)
    result = resolver.visit_Subscript(custom_node)
    assert result is custom_node
    
    # Test Union with single element (should return slice directly)
    union_single = cast(Subscript, parse("Union[int]").body[0].value)
    result = resolver.visit_Subscript(union_single)
    assert isinstance(result, Name) and result.id == "int"
    
    # Test Union with more than two elements
    union_multi = cast(Subscript, parse("Union[int, str, float]").body[0].value)
    result = resolver.visit_Subscript(union_multi)
    assert isinstance(result, BinOp)
    # Should create nested BinOp structure: (int | str) | float
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name) and result.right.id == "float"
    assert isinstance(result.left, BinOp)
    assert isinstance(result.left.op, BitOr)
    assert isinstance(result.left.left, Name) and result.left.left.id == "int"
    assert isinstance(result.left.right, Name) and result.left.right.id == "str"


# LLM-generated content at query #18
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    parser.alias = {}
    
    # Test basic function annotation
    args = [arg(arg="x", annotation=Name(id="int", ctx=Load()))]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int"]
    
    # Test function with self parameter
    args = [
        arg(arg="self", annotation=None),
        arg(arg="x", annotation=Name(id="str", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=False))
    assert result == ["Self", "str"]
    
    # Test class method
    args = [
        arg(arg="cls", annotation=Name(id="type", ctx=Load())),
        arg(arg="x", annotation=Name(id="float", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "float"]
    
    # Test function with vararg separator
    args = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="*", annotation=None),
        arg(arg="y", annotation=Name(id="str", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int", "", "str"]
    
    # Test function without annotations
    args = [arg(arg="x", annotation=None)]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["Any"]
    
    # Test function with return annotation
    args = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="return", annotation=Name(id="str", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int", "str"]
    
    # Test with alias resolution
    parser.alias = {"test_module.int": "builtins.int"}
    args = [arg(arg="x", annotation=Name(id="int", ctx=Load()))]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int"]
    
    # Test with self type annotation
    args = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=True, cls_method=False))
    assert result == ["Self", "int"]
    
    # Test multiple parameters
    args = [
        arg(arg="a", annotation=Name(id="int", ctx=Load())),
        arg(arg="b", annotation=Name(id="str", ctx=Load())),
        arg(arg="c", annotation=Name(id="float", ctx=Load()))
    ]
    result = list(parser.func_ann("test_module", args, has_self=False, cls_method=False))
    assert result == ["int", "str", "float"]


# LLM-generated content at query #19
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    parser.parse("test_module", "")
    
    # Test Import with alias
    import_node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports("test_module", import_node)
    assert parser.alias["test_module.operating_system"] == "os"
    
    # Test Import without alias
    import_node = Import(names=[alias(name="sys", asname=None)])
    parser.imports("test_module", import_node)
    assert parser.alias["test_module.sys"] == "sys"
    
    # Test ImportFrom with relative level
    import_from = ImportFrom(
        module="submodule",
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports("parent.child", import_from)
    assert parser.alias["parent.child.func"] == "parent.submodule.func"
    
    # Test ImportFrom with absolute import
    import_from = ImportFrom(
        module="package.module",
        names=[alias(name="Class", asname="Cls")],
        level=0
    )
    parser.imports("test_module", import_from)
    assert parser.alias["test_module.Cls"] == "package.module.Class"
    
    # Test ImportFrom with multiple names
    import_from = ImportFrom(
        module="collections",
        names=[
            alias(name="defaultdict", asname=None),
            alias(name="OrderedDict", asname="ODict")
        ],
        level=0
    )
    parser.imports("test_module", import_from)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"
    assert parser.alias["test_module.ODict"] == "collections.OrderedDict"
    
    # Test that existing aliases are preserved
    parser.alias["test_module.existing"] = "old.value"
    import_node = Import(names=[alias(name="new_module", asname="existing")])
    parser.imports("test_module", import_node)
    assert parser.alias["test_module.existing"] == "new_module"


# LLM-generated content at query #20
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    int_node = Constant(value=42)
    result = resolver.visit_Constant(int_node)
    assert result is int_node
    
    # Test with string that's not a valid name
    invalid_str_node = Constant(value="not a valid name syntax")
    result = resolver.visit_Constant(invalid_str_node)
    assert result is invalid_str_node
    
    # Test with valid name string
    valid_str_node = Constant(value="test_name")
    result = resolver.visit_Constant(valid_str_node)
    assert isinstance(result, Name)
    assert result.id == "test_name"
    assert isinstance(result.ctx, Load)
    
    # Test with alias mapping
    resolver = Resolver("test_module", {"test_module.test_name": "mapped_name"})
    valid_str_node = Constant(value="test_name")
    result = resolver.visit_Constant(valid_str_node)
    assert isinstance(result, Name)
    assert result.id == "mapped_name"
    
    # Test with self type replacement
    resolver = Resolver("test_module", {}, self_ty="SelfType")
    self_str_node = Constant(value="SelfType")
    result = resolver.visit_Constant(self_str_node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    
    # Test with complex expression string
    complex_str_node = Constant(value="Union[int, str]")
    result = resolver.visit_Constant(complex_str_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"


# LLM-generated content at query #21
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test_module", {})
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result is node

    # Test with string that is not valid Python expression
    resolver = Resolver("test_module", {})
    node = Constant(value="not a valid python expression!")
    result = resolver.visit_Constant(node)
    assert result is node

    # Test with valid Python expression string
    resolver = Resolver("test_module", {})
    node = Constant(value="some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "some_name"

    # Test with string containing complex expression
    resolver = Resolver("test_module", {})
    node = Constant(value="Union[int, str]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test with alias mapping
    resolver = Resolver("test_module", {"test_module.some_name": "mapped_name"})
    node = Constant(value="some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "mapped_name"

    # Test with self type replacement
    resolver = Resolver("test_module", {}, self_ty="SelfType")
    node = Constant(value="SelfType")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test with TypeVar in alias
    resolver = Resolver("test_module", {"test_module.T": "typing.TypeVar('T')"})
    node = Constant(value="T")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #22
#--------------------------

```python
def test_const_type():
    # Test Constant nodes
    assert const_type(Constant(value=42)) == "int"
    assert const_type(Constant(value=3.14)) == "float"
    assert const_type(Constant(value="hello")) == "str"
    assert const_type(Constant(value=True)) == "bool"
    assert const_type(Constant(value=None)) == "NoneType"
    
    # Test Tuple nodes
    assert const_type(Tuple(elts=[Constant(value=1), Constant(value=2)])) == "tuple[int, int]"
    assert const_type(Tuple(elts=[Constant(value="a"), Constant(value="b")])) == "tuple[str, str]"
    assert const_type(Tuple(elts=[Constant(value=1), Constant(value="a")])) == "tuple[Any, Any]"
    assert const_type(Tuple(elts=[])) == "tuple"
    
    # Test List nodes
    assert const_type(List(elts=[Constant(value=1.0), Constant(value=2.0)])) == "list[float, float]"
    assert const_type(List(elts=[Constant(value=True), Constant(value=False)])) == "list[bool, bool]"
    assert const_type(List(elts=[])) == "list"
    
    # Test Set nodes
    assert const_type(Set(elts=[Constant(value=1), Constant(value=2)])) == "set[int, int]"
    assert const_type(Set(elts=[])) == "set"
    
    # Test Dict nodes
    assert const_type(Dict(keys=[Constant(value="key")], values=[Constant(value=1)])) == "dict[str, int]"
    assert const_type(Dict(keys=[Constant(value=1)], values=[Constant(value="value")])) == "dict[int, str]"
    assert const_type(Dict(keys=[], values=[])) == "dict"
    
    # Test Call nodes with built-in types
    assert const_type(Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])) == "int"
    assert const_type(Call(func=Name(id="str", ctx=Load()), args=[], keywords=[])) == "str"
    assert const_type(Call(func=Name(id="bool", ctx=Load()), args=[], keywords=[])) == "bool"
    assert const_type(Call(func=Name(id="float", ctx=Load()), args=[], keywords=[])) == "float"
    assert const_type(Call(func=Name(id="complex", ctx=Load()), args=[], keywords=[])) == "complex"
    
    # Test Call nodes with PEP585 types
    assert const_type(Call(func=Name(id="list", ctx=Load()), args=[], keywords=[])) == "list"
    assert const_type(Call(func=Name(id="dict", ctx=Load()), args=[], keywords=[])) == "dict"
    assert const_type(Call(func=Name(id="tuple", ctx=Load()), args=[], keywords=[])) == "tuple"
    
    # Test Call nodes with Attribute (e.g., typing.List)
    assert const_type(Call(func=Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load()), args=[], keywords=[])) == "List"
    
    # Test other expression types return "Any"
    assert const_type(Name(id="x", ctx=Load())) == "Any"
    assert const_type(BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))) == "Any"
    assert const_type(Subscript(value=Name(id="List", ctx=Load()), slice=Constant(value=1), ctx=Load())) == "Any"


# LLM-generated content at query #23
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    int_node = Constant(value=42)
    result = resolver.visit_Constant(int_node)
    assert result is int_node
    
    # Test with string that is not valid Python
    invalid_str_node = Constant(value="not valid python code")
    result = resolver.visit_Constant(invalid_str_node)
    assert result is invalid_str_node
    
    # Test with valid Python string expression
    valid_str_node = Constant(value="List[int]")
    result = resolver.visit_Constant(valid_str_node)
    assert isinstance(result, Expr)
    
    # Test with string that resolves to Name node
    name_str_node = Constant(value="SomeType")
    result = resolver.visit_Constant(name_str_node)
    assert isinstance(result, Name)
    assert result.id == "SomeType"
    
    # Test with alias resolution
    resolver.alias = {"test_module.SomeType": "typing.List[int]"}
    result = resolver.visit_Constant(name_str_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "List"
    
    # Test with self type replacement
    resolver.self_ty = "SomeType"
    self_str_node = Constant(value="SomeType")
    result = resolver.visit_Constant(self_str_node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #24
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    parser.parse("test_module", "")
    
    # Test basic function with no arguments
    name = "test_module.func1"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "func1()" in parser.doc[name]
    assert "*Full name:* `test_module.func1`" in parser.doc[name]
    
    # Test function with positional arguments
    name = "test_module.func2"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="x", annotation=None), arg(arg="y", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "return" in parser.doc[name]
    
    # Test function with self argument
    name = "test_module.Class.method"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="self", annotation=None), arg(arg="x", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=False)
    
    # Test function with classmethod
    name = "test_module.Class.class_method"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="cls", annotation=None), arg(arg="x", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=True, cls_method=True)
    
    # Test function with defaults
    name = "test_module.func3"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="x", annotation=None), arg(arg="y", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=1), Constant(value=2)]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    # Test function with vararg and kwarg
    name = "test_module.func4"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="x", annotation=None)],
        vararg=arg(arg="args", annotation=None),
        kwonlyargs=[arg(arg="kw1", annotation=None)],
        kw_defaults=[Constant(value="default")],
        kwarg=arg(arg="kwargs", annotation=None),
        defaults=[]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "*args" in parser.doc[name]
    assert "**kwargs" in parser.doc[name]
    assert "kw1" in parser.doc[name]
    
    # Test function with return annotation
    name = "test_module.func5"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="x", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = Name(id="int", ctx=Load())
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    # Test function with posonlyargs
    name = "test_module.func6"
    node = arguments(
        posonlyargs=[arg(arg="x", annotation=None)],
        args=[arg(arg="y", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "/" in parser.doc[name]
    
    # Test function with kwonlyargs and no vararg
    name = "test_module.func7"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="x", annotation=None)],
        vararg=None,
        kwonlyargs=[arg(arg="kw1", annotation=None), arg(arg="kw2", annotation=None)],
        kw_defaults=[Constant(value=1), Constant(value=2)],
        kwarg=None,
        defaults=[]
    )
    
    parser.func_api("test_module", name, node, returns, has_self=False, cls_method=False)
    
    assert "*" in parser.doc[name]


# LLM-generated content at query #25
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test_module", {})
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result is node
    
    # Test with string that is not valid Python expression
    node = Constant(value="not a valid expression")
    result = resolver.visit_Constant(node)
    assert result is node
    
    # Test with string that is valid Python expression
    node = Constant(value="some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "some_name"
    
    # Test with string expression when root and alias are set
    resolver = Resolver("test_module", {"test_module.some_name": "other_name"})
    node = Constant(value="some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "other_name"
    
    # Test with complex string expression
    node = Constant(value="some_name[0]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "some_name"


# LLM-generated content at query #26
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    int_node = Constant(value=42)
    result = resolver.visit_Constant(int_node)
    assert result is int_node
    
    # Test with string that is not a valid name
    invalid_str_node = Constant(value="not a valid name syntax")
    result = resolver.visit_Constant(invalid_str_node)
    assert result is invalid_str_node
    
    # Test with valid name string
    valid_name_node = Constant(value="SomeType")
    result = resolver.visit_Constant(valid_name_node)
    assert isinstance(result, Name)
    assert result.id == "SomeType"
    
    # Test with alias mapping
    resolver.alias = {"test_module.SomeType": "typing.List[int]"}
    result = resolver.visit_Constant(valid_name_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "List"
    
    # Test with self type replacement
    resolver.self_ty = "SomeType"
    result = resolver.visit_Constant(valid_name_node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    
    # Test with complex expression string
    complex_node = Constant(value="Union[int, str]")
    result = resolver.visit_Constant(complex_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"


