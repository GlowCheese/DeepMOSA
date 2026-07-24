####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_imports_method():
    parser = Parser()
    parser.parse("root_module", "")
    import_node = Import(names=[ast.alias(name="module_name", asname="alias_name")])
    parser.imports("root_module", import_node)
    assert parser.alias == {"root_module.alias_name": "module_name"}
    import_from_node = ImportFrom(module="module_name", names=[ast.alias(name="function_name", asname="alias_name")], level=0)
    parser.imports("root_module", import_from_node)
    assert parser.alias == {"root_module.alias_name": "module_name.function_name", "root_module.alias_name": "module_name.function_name"}
    import_from_node_level = ImportFrom(module="module_name", names=[ast.alias(name="function_name", asname="alias_name")], level=1)
    parser.imports("root_module", import_from_node_level)
    assert parser.alias == {"root_module.alias_name": "module_name.function_name", "root_module.alias_name": "module_name.function_name"}


# LLM-generated content at query #2
#--------------------------

```python
def test_table_single_row():
    result = table('a', 'b', items=[['c', 'd']])
    expected = "| a | b |\n|:---:|:---:|\n| c | d |\n\n"
    assert result == expected

def test_table_multiple_rows():
    result = table('a', 'b', items=[['c', 'd'], ['e', 'f']])
    expected = "| a | b |\n|:---:|:---:|\n| c | d |\n| e | f |\n\n"
    assert result == expected

def test_table_single_item():
    result = table('a', 'b', items=[['c']])
    expected = "| a | b |\n|:---:|:---:|\n| c |\n\n"
    assert result == expected

def test_table_empty_items():
    result = table('a', 'b', items=[])
    expected = "| a | b |\n|:---:|:---:|\n\n"
    assert result == expected

def test_table_with_single_string_item():
    result = table('a', 'b', items=['c'])
    expected = "| a | b |\n|:---:|:---:|\n| c |\n\n"
    assert result == expected

def test_table_with_varying_length_titles():
    result = table('abc', 'd', items=[['e', 'f']])
    expected = "| abc | d |\n|:-----:|:---:|\n| e | f |\n\n"
    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_parser_default_constructor():
    p = Parser()
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


def test_parser_new_classmethod():
    p = Parser.new(link=False, level=2, toc=True)
    assert p.link is False
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


def test_parser_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


def test_parser_post_init_without_toc():
    p = Parser(toc=False)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is False


# LLM-generated content at query #4
#--------------------------

```python
def test_visit_Constant_with_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_invalid_syntax_string():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_valid_expression_string():
    resolver = Resolver("root", {})
    node = Constant("some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "some_name"
    assert isinstance(result.ctx, Load)

def test_visit_Constant_with_self_ty_replacement():
    resolver = Resolver("root", {}, self_ty="self_type")
    node = Constant("self_type")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #5
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    root = "test_module"
    node = AnnAssign(Name(id="x"), Constant(value=42), annotation=Name(id="int"))
    parser.globals(root, node)
    assert parser.alias["test_module.x"] == "42"
    assert parser.const["test_module.x"] == "int"

def test_globals_with_assign():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="y")], value=Constant(value="hello"), type_comment="str")
    parser.globals(root, node)
    assert parser.alias["test_module.y"] == "'hello'"
    assert parser.const["test_module.y"] == "str"

def test_globals_with_assign_no_type_comment():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="z")], value=Constant(value=3.14))
    parser.globals(root, node)
    assert parser.alias["test_module.z"] == "3.14"
    assert parser.const["test_module.z"] == "float"

def test_globals_with_assign_tuple():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="a")], value=Tuple(elts=[Constant(value=1), Constant(value=2)]))
    parser.globals(root, node)
    assert parser.alias["test_module.a"] == "(1, 2)"
    assert parser.const["test_module.a"] == "tuple[int, int]"

def test_globals_with_assign_list():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="b")], value=List(elts=[Constant(value="a"), Constant(value="b")]))
    parser.globals(root, node)
    assert parser.alias["test_module.b"] == "['a', 'b']"
    assert parser.const["test_module.b"] == "list[str, str]"

def test_globals_with_assign_set():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="c")], value=Set(elts=[Constant(value=True), Constant(value=False)]))
    parser.globals(root, node)
    assert parser.alias["test_module.c"] == "{True, False}"
    assert parser.const["test_module.c"] == "set[bool, bool]"

def test_globals_with_assign_dict():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="d")], value=Dict(keys=[Constant(value="key")], values=[Constant(value="value")]))
    parser.globals(root, node)
    assert parser.alias["test_module.d"] == "{'key': 'value'}"
    assert parser.const["test_module.d"] == "dict[str, str]"

def test_globals_with_assign_call():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="e")], value=Call(func=Name(id="int"), args=[Constant(value="42")]))
    parser.globals(root, node)
    assert parser.alias["test_module.e"] == "int('42')"
    assert parser.const["test_module.e"] == "int"

def test_globals_with_assign_not_name_target():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Attribute(value=Name(id="obj"), attr="attr")], value=Constant(value=1))
    parser.globals(root, node)
    assert parser.alias == {}
    assert parser.const == {}

def test_globals_with_multiple_targets():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="x"), Name(id="y")], value=Constant(value=1))
    parser.globals(root, node)
    assert parser.alias == {}
    assert parser.const == {}

def test_globals_with_ann_assign_no_value():
    parser = Parser()
    root = "test_module"
    node = AnnAssign(Name(id="x"), None, annotation=Name(id="int"))
    parser.globals(root, node)
    assert parser.alias == {}
    assert parser.const == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_compile_with_toc():
    parser = Parser(toc=True)
    parser.doc = {'module': '# Module `module`\n\n', 'module.func': '## func()\n\n'}
    parser.docstring = {'module.func': 'Docstring for func'}
    parser.level = {'module': 0, 'module.func': 1}
    parser.root = {'module': 'module', 'module.func': 'module'}
    parser.const = {}
    parser.imp = {'module': set()}
    result = parser.compile()
    expected = "**Table of contents:**\n    + [`module`](#module)\n        + [`module.func`](#module-func)\n\n# Module `module`\n\n\n\n## func()\n\nDocstring for func\n"
    assert result == expected

def test_compile_without_toc():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n\n', 'module.func': '## func()\n\n'}
    parser.docstring = {'module.func': 'Docstring for func'}
    parser.level = {'module': 0, 'module.func': 1}
    parser.root = {'module': 'module', 'module.func': 'module'}
    parser.const = {}
    parser.imp = {'module': set()}
    result = parser.compile()
    expected = "# Module `module`\n\n\n\n## func()\n\nDocstring for func\n"
    assert result == expected

def test_compile_with_constants():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n\n'}
    parser.docstring = {}
    parser.level = {'module': 0}
    parser.root = {'module': 'module'}
    parser.const = {'module.CONST': 'int'}
    parser.imp = {'module': set()}
    result = parser.compile()
    expected = "# Module `module`\n\n\n| Constants | Type |\n| --- | --- |\n| `CONST` | `int` |\n"
    assert result == expected

def test_compile_with_missing_documentation():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n\n', 'module.__magic': '## __magic()\n\n'}
    parser.docstring = {}
    parser.level = {'module': 0, 'module.__magic': 1}
    parser.root = {'module': 'module', 'module.__magic': 'module'}
    parser.const = {}
    parser.imp = {'module': set()}
    result = parser.compile()
    expected = "# Module `module`\n\n\n"
    assert result == expected

def test_compile_with_immediate_family():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n\n', 'module.func': '## func()\n\n', 'module.alias': '## alias()\n\n'}
    parser.docstring = {'module.func': 'Docstring for func'}
    parser.level = {'module': 0, 'module.func': 1, 'module.alias': 1}
    parser.root = {'module': 'module', 'module.func': 'module', 'module.alias': 'module'}
    parser.const = {}
    parser.imp = {'module': set()}
    parser.alias = {'module.alias': 'module.func'}
    result = parser.compile()
    expected = "# Module `module`\n\n\n\n## func()\n\nDocstring for func\n"
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_parse_method_with_default_values():
    parser = Parser()
    parser.parse("root", "import os")
    assert parser.doc["root"] == "# Module `root`\n\n"
    assert parser.level["root"] == 0
    assert parser.imp["root"] == set()
    assert parser.root["root"] == "root"

def test_parse_method_with_link_true():
    parser = Parser(link=True)
    parser.parse("root", "import os")
    assert parser.doc["root"] == "# Module `root`\n<a id=\"root\"></a>\n\n"

def test_parse_method_with_toc_true():
    parser = Parser(toc=True)
    parser.parse("root", "import os")
    assert parser.link == True

def test_parse_method_with_level_set():
    parser = Parser(b_level=2)
    parser.parse("root", "import os")
    assert parser.doc["root"] == "## Module `root`\n\n"

def test_parse_method_with_import_statement():
    parser = Parser()
    parser.parse("root", "import os")
    assert parser.alias["root.os"] == "os"

def test_parse_method_with_import_from_statement():
    parser = Parser()
    parser.parse("root", "from os import path")
    assert parser.alias["root.path"] == "os.path"

def test_parse_method_with_assignment_statement():
    parser = Parser()
    parser.parse("root", "x = 10")
    assert parser.alias["root.x"] == "10"

def test_parse_method_with_ann_assignment_statement():
    parser = Parser()
    parser.parse("root", "x: int = 10")
    assert parser.alias["root.x"] == "10"

def test_parse_method_with_function_def_statement():
    parser = Parser()
    parser.parse("root", "def func(): pass")
    assert parser.doc["root.func"] == "### func()\n\n*Full name:* `root.func`\n\n"

def test_parse_method_with_class_def_statement():
    parser = Parser()
    parser.parse("root", "class MyClass: pass")
    assert parser.doc["root.MyClass"] == "### class MyClass\n\n*Full name:* `root.MyClass`\n\n"

def test_parse_method_with_docstring():
    parser = Parser()
    parser.parse("root", '"""This is a docstring"""')
    assert parser.docstring["root"] == "This is a docstring"


# LLM-generated content at query #8
#--------------------------

```python
def test_class_api():
    p = Parser()
    p.parse('root', 'class A: pass')
    assert p.doc == {'root': '## Module `root`\n<a id="root"></a>\n\n', 'root.A': '### class A\n\n*Full name:* `root.A`\n<a id="root.A"></a>\n\n'}

    p = Parser()
    p.parse('root', 'class A(B, C): pass')
    assert p.doc == {'root': '## Module `root`\n<a id="root"></a>\n\n', 'root.A': '### class A\n\n*Full name:* `root.A`\n<a id="root.A"></a>\n\nBases\n| B | C |\n'}

    p = Parser()
    p.parse('root', 'class A:\n    x: int\n    y: str')
    assert p.doc == {'root': '## Module `root`\n<a id="root"></a>\n\n', 'root.A': '### class A\n\n*Full name:* `root.A`\n<a id="root.A"></a>\n\nMembers\n| x | y |\nType\n| int | str |\n'}

    p = Parser()
    p.parse('root', 'class A:\n    class B: pass')
    assert p.doc == {'root': '## Module `root`\n<a id="root"></a>\n\n', 'root.A': '### class A\n\n*Full name:* `root.A`\n<a id="root.A"></a>\n\n', 'root.A.B': '#### class B\n\n*Full name:* `root.A.B`\n<a id="root.A.B"></a>\n\n'}

    p = Parser()
    p.parse('root', 'class A:\n    def f(): pass')
    assert p.doc == {'root': '## Module `root`\n<a id="root"></a>\n\n', 'root.A': '### class A\n\n*Full name:* `root.A`\n<a id="root.A"></a>\n\n', 'root.A.f': '#### f()\n\n*Full name:* `root.A.f`\n<a id="root.A.f"></a>\n\n'}


# LLM-generated content at query #9
#--------------------------

```
def test_walk_body_with_single_stmt():
    stmt = "stmt"
    assert list(walk_body([stmt])) == [stmt]

def test_walk_body_with_if_stmt():
    if_stmt = If(body=["body_stmt"], orelse=["orelse_stmt"])
    assert list(walk_body([if_stmt])) == ["body_stmt", "orelse_stmt"]

def test_walk_body_with_try_stmt():
    try_stmt = Try(body=["body_stmt"], handlers=[Handler(body=["handler_body"])], orelse=["orelse_stmt"], finalbody=["finalbody_stmt"])
    assert list(walk_body([try_stmt])) == ["body_stmt", "handler_body", "orelse_stmt", "finalbody_stmt"]

def test_walk_body_with_nested_stmts():
    nested_if = If(body=["nested_body"], orelse=["nested_orelse"])
    try_stmt = Try(body=[nested_if], handlers=[Handler(body=["handler_body"])], orelse=["orelse_stmt"], finalbody=["finalbody_stmt"])
    assert list(walk_body([try_stmt])) == ["nested_body", "nested_orelse", "handler_body", "orelse_stmt", "finalbody_stmt"]

def test_walk_body_with_multiple_stmts():
    stmt1 = "stmt1"
    stmt2 = "stmt2"
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]


# LLM-generated content at query #10
#--------------------------

```python
def test_class_api_with_non_public_family_attribute():
    parser = Parser()
    node = AnnAssign(target=Name(id='_private'), annotation=None, value=None)
    parser.class_api('root', 'name', [], [node])
    assert '_private' not in parser.doc['name']


# LLM-generated content at query #11
#--------------------------

```python
def test_globals_predicate_evaluates_to_false():
    parser = Parser()
    node = Assign(targets=[Name(id="x", ctx=Load()), Name(id="y", ctx=Load())], value=Constant(value=42))
    parser.globals("root", node)


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_23_evaluates_to_false():
    node = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=42), type_comment="int")
    parser = Parser()
    parser.globals(root="test", node=node)


# LLM-generated content at query #13
#--------------------------

```python
def test_parser_constructor_default_values():
    p = Parser()
    assert p.link == True
    assert p.b_level == 1
    assert p.toc == False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link == False
    assert p.b_level == 2
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link == True

def test_parser_post_init_without_toc():
    p = Parser(toc=False)
    assert p.link == True


# LLM-generated content at query #14
#--------------------------

```python
def test_class_api_with_enums():
    parser = Parser()
    class_body = [
        AnnAssign(target=Name(id='ENUM_VALUE_1'), annotation=Name(id='int'), value=None),
        AnnAssign(target=Name(id='ENUM_VALUE_2'), annotation=Name(id='int'), value=None),
        AnnAssign(target=Name(id='ENUM_VALUE_3'), annotation=Name(id='int'), value=None)
    ]
    parser.class_api('root', 'name', [Name(id='enum.Enum')], class_body)
    assert 'Enums' in parser.doc['name']


# LLM-generated content at query #15
#--------------------------

```python
def test_class_api_predicate_evaluates_to_false():
    root = "module_name"
    name = "ClassName"
    bases = []
    body = []
    parser = Parser()
    parser.class_api(root, name, bases, body)
    assert not isinstance(body[0], AnnAssign) if body else True


# LLM-generated content at query #16
#--------------------------

```python
def test_is_public_family_with_public_name():
    assert is_public_family("public.name") == True

def test_is_public_family_with_private_name():
    assert is_public_family("_private.name") == False

def test_is_public_family_with_magic_name():
    assert is_public_family("__magic__") == True

def test_is_public_family_with_mixed_name():
    assert is_public_family("public._private") == False

def test_is_public_family_with_nested_magic_name():
    assert is_public_family("module.__magic__.name") == True

def test_is_public_family_with_nested_private_name():
    assert is_public_family("module._private.name") == False


# LLM-generated content at query #17
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root.name': 'root'}
    assert parser.is_public('root.name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root._name': 'root'}
    assert parser.is_public('root._name') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root.__name__': 'root'}
    assert parser.is_public('root.__name__') == True

def test_is_public_with_name_in_all():
    parser = Parser()
    parser.imp = {'root': {'root.name'}}
    parser.root = {'root.name': 'root'}
    assert parser.is_public('root.name') == True

def test_is_public_with_parent_in_all():
    parser = Parser()
    parser.imp = {'root': {'root'}}
    parser.root = {'root.name': 'root'}
    assert parser.is_public('root.name') == True

def test_is_public_with_root_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root': 'root'}
    assert parser.is_public('root') == True


# LLM-generated content at query #18
#--------------------------

```
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[arg(arg='x', annotation=None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| x |' in p.doc['name']

def test_func_api_with_args_and_defaults():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[None, Constant(value=1)])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| x | y |' in p.doc['name']
    assert '|   | 1 |' in p.doc['name']

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=arg(arg='args', annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| *args |' in p.doc['name']

def test_func_api_with_kwonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg(arg='x', annotation=None)], kw_defaults=[None], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| * | x |' in p.doc['name']

def test_func_api_with_kwarg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=arg(arg='kwargs', annotation=None), defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| **kwargs |' in p.doc['name']

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, Name(id='int', ctx=Load()), has_self=False, cls_method=False)
    assert '| return |' in p.doc['name']
    assert '| int |' in p.doc['name']

def test_func_api_with_self_and_cls_method():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert '| type[Self] |' in p.doc['name']

def test_func_api_with_self_and_no_cls_method():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=False)
    assert '| Self |' in p.doc['name']


# LLM-generated content at query #19
#--------------------------

```python
def test_api_method_with_function_def():
    parser = Parser()
    parser.parse("root", "def example_function(): pass")
    assert "example_function" in parser.doc

def test_api_method_with_async_function_def():
    parser = Parser()
    parser.parse("root", "async def example_async_function(): pass")
    assert "example_async_function" in parser.doc

def test_api_method_with_class_def():
    parser = Parser()
    parser.parse("root", "class ExampleClass: pass")
    assert "ExampleClass" in parser.doc

def test_api_method_with_function_def_with_decorators():
    parser = Parser()
    parser.parse("root", "@decorator\ndef example_function(): pass")
    assert "example_function" in parser.doc
    assert "Decorators" in parser.doc["root.example_function"]

def test_api_method_with_class_def_with_bases():
    parser = Parser()
    parser.parse("root", "class ExampleClass(BaseClass): pass")
    assert "ExampleClass" in parser.doc
    assert "Bases" in parser.doc["root.ExampleClass"]

def test_api_method_with_class_def_with_members():
    parser = Parser()
    parser.parse("root", "class ExampleClass:\n    member: int = 1")
    assert "ExampleClass" in parser.doc
    assert "Members" in parser.doc["root.ExampleClass"]

def test_api_method_with_class_def_with_enums():
    parser = Parser()
    parser.parse("root", "class ExampleClass(enum.Enum):\n    ENUM = 1")
    assert "ExampleClass" in parser.doc
    assert "Enums" in parser.doc["root.ExampleClass"]

def test_api_method_with_function_def_with_self():
    parser = Parser()
    parser.parse("root", "class ExampleClass:\n    def method(self): pass")
    assert "ExampleClass.method" in parser.doc

def test_api_method_with_function_def_with_class_method():
    parser = Parser()
    parser.parse("root", "class ExampleClass:\n    @classmethod\n    def method(cls): pass")
    assert "ExampleClass.method" in parser.doc

def test_api_method_with_function_def_with_static_method():
    parser = Parser()
    parser.parse("root", "class ExampleClass:\n    @staticmethod\n    def method(): pass")
    assert "ExampleClass.method" in parser.doc


# LLM-generated content at query #20
#--------------------------

```python
def test_is_public_returns_true_when_s_is_root_and_all_l_contains_s():
    parser = Parser()
    parser.imp = {'module': {'module'}}
    parser.root = {'module': 'module'}
    assert parser.is_public('module') == True

def test_is_public_returns_true_when_parent_s_in_all_l():
    parser = Parser()
    parser.imp = {'module': {'module.submodule'}}
    parser.root = {'module.submodule': 'module'}
    assert parser.is_public('module.submodule') == True

def test_is_public_returns_true_when_s_is_public_family_and_all_l_is_empty():
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module.public_function': 'module'}
    assert parser.is_public('module.public_function') == True


# LLM-generated content at query #21
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(target=Name(id='x'), annotation=Name(id='int'), value=Constant(value=42))
    parser.globals('root', node)
    assert parser.alias['root.x'] == '42'
    assert parser.const['root.x'] == 'int'

def test_globals_with_assign():
    parser = Parser()
    node = Assign(targets=[Name(id='y')], value=Constant(value=100))
    parser.globals('root', node)
    assert parser.alias['root.y'] == '100'
    assert parser.const['root.y'] == 'int'

def test_globals_with_assign_and_type_comment():
    parser = Parser()
    node = Assign(targets=[Name(id='z')], value=Constant(value='test'), type_comment='str')
    parser.globals('root', node)
    assert parser.alias['root.z'] == 'test'
    assert parser.const['root.z'] == 'str'

def test_globals_with_assign_and_tuple_value():
    parser = Parser()
    node = Assign(targets=[Name(id='a')], value=Tuple(elts=[Constant(value=1), Constant(value=2)]))
    parser.globals('root', node)
    assert parser.alias['root.a'] == '(1, 2)'
    assert parser.const['root.a'] == 'tuple[int, int]'

def test_globals_with_assign_and_list_value():
    parser = Parser()
    node = Assign(targets=[Name(id='b')], value=List(elts=[Constant(value=3), Constant(value=4)]))
    parser.globals('root', node)
    assert parser.alias['root.b'] == '[3, 4]'
    assert parser.const['root.b'] == 'list[int, int]'

def test_globals_with_assign_and_set_value():
    parser = Parser()
    node = Assign(targets=[Name(id='c')], value=Set(elts=[Constant(value=5), Constant(value=6)]))
    parser.globals('root', node)
    assert parser.alias['root.c'] == '{5, 6}'
    assert parser.const['root.c'] == 'set[int, int]'

def test_globals_with_assign_and_dict_value():
    parser = Parser()
    node = Assign(targets=[Name(id='d')], value=Dict(keys=[Constant(value='key')], values=[Constant(value='value')]))
    parser.globals('root', node)
    assert parser.alias['root.d'] == "{'key': 'value'}"
    assert parser.const['root.d'] == 'dict[str, str]'

def test_globals_with_assign_and_call_value():
    parser = Parser()
    node = Assign(targets=[Name(id='e')], value=Call(func=Name(id='str'), args=[Constant(value='test')]))
    parser.globals('root', node)
    assert parser.alias['root.e'] == 'str(test)'
    assert parser.const['root.e'] == 'str'

def test_globals_with_assign_and_multiple_targets():
    parser = Parser()
    node = Assign(targets=[Name(id='f'), Name(id='g')], value=Constant(value=123))
    parser.globals('root', node)
    assert parser.alias == {}
    assert parser.const == {}

def test_globals_with_assign_and_non_name_target():
    parser = Parser()
    node = Assign(targets=[Attribute(value=Name(id='obj'), attr='attr')], value=Constant(value=456))
    parser.globals('root', node)
    assert parser.alias == {}
    assert parser.const == {}

def test_globals_with_assign_and_non_constant_value():
    parser = Parser()
    node = Assign(targets=[Name(id='h')], value=Name(id='x'))
    parser.globals('root', node)
    assert parser.alias['root.h'] == 'x'
    assert parser.const['root.h'] == 'Any'

def test_globals_with_assign_and_non_public_name():
    parser = Parser()
    node = Assign(targets=[Name(id='_private')], value=Constant(value=789))
    parser.globals('root', node)
    assert parser.alias['root._private'] == '789'
    assert parser.const == {}

def test_globals_with_assign_and_uppercase_name():
    parser = Parser()
    node = Assign(targets=[Name(id='MY_CONST')], value=Constant(value=999))
    parser.globals('root', node)
    assert parser.alias['root.MY_CONST'] == '999'
    assert parser.const['root.MY_CONST'] == 'int'

def test_globals_with_assign_and_all_name():
    parser = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Tuple(elts=[Constant(value='public_func'), Constant(value='PublicClass')]))
    parser.globals('root', node)
    assert parser.alias == {}
    assert parser.const == {}
    assert parser.imp['root'] == {'root.public_func', 'root.PublicClass'}


# LLM-generated content at query #22
#--------------------------

```python
def test_defaults_with_none_values():
    args = [None, "test", None]
    result = list(_defaults(args))
    assert result == [" ", "`test`", " "]

def test_defaults_with_non_none_values():
    args = ["value1", "value2"]
    result = list(_defaults(args))
    assert result == ["`value1`", "`value2`"]

def test_defaults_with_empty_sequence():
    args = []
    result = list(_defaults(args))
    assert result == []

def test_defaults_with_special_characters():
    args = ["test|value", "test&value"]
    result = list(_defaults(args))
    assert result == ["<code>test&#124;value</code>", "<code>test&value</code>"]


# LLM-generated content at query #23
#--------------------------

```python
def test__e_type_empty_input():
    assert _e_type() == ""

def test__e_type_single_empty_element():
    assert _e_type([]) == ""

def test__e_type_single_element_with_non_constant():
    class NonConstant:
        pass
    assert _e_type([NonConstant()]) == ""

def test__e_type_single_element_with_constants_of_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int, int]"

def test__e_type_single_element_with_constants_of_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_multiple_elements_with_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)], [Constant("a"), Constant("b")]) == "[int, int], [str, str]"

def test__e_type_multiple_elements_with_mixed_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")], [Constant("b"), Constant(2)]) == "[Any], [Any]"


# LLM-generated content at query #24
#--------------------------

```python
def test_imports_predicate_false():
    parser = Parser()
    node = Import(names=[alias(name='module_name', asname='alias_name')])
    parser.imports('root', node)
    assert parser.alias == {'root.alias_name': 'module_name'}


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    parser = Parser()
    parser.const = {"root.key": "existing_value"}
    node = Assign(targets=[Name(id="KEY", ctx=Store())], value=Constant(value=42))
    parser.globals("root", node)
    assert parser.const["root.KEY"] != "existing_value"


# LLM-generated content at query #26
#--------------------------

```python
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1))
    p.globals('root', node)
    assert p.alias['root.x'] == '1'
    assert p.const['root.x'] == 'int'

def test_globals_with_assign():
    p = Parser()
    node = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=2))
    p.globals('root', node)
    assert p.alias['root.y'] == '2'
    assert p.const['root.y'] == 'int'

def test_globals_with_assign_and_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3), type_comment='float')
    p.globals('root', node)
    assert p.alias['root.z'] == '3'
    assert p.const['root.z'] == 'float'

def test_globals_with_non_name_target():
    p = Parser()
    node = Assign(targets=[Subscript(value=Name(id='a', ctx=Load()), slice=Constant(value=0), ctx=Store())], value=Constant(value=4))
    p.globals('root', node)
    assert 'root.a' not in p.alias
    assert 'root.a' not in p.const

def test_globals_with_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='b', ctx=Store()), Name(id='c', ctx=Store())], value=Constant(value=5))
    p.globals('root', node)
    assert 'root.b' not in p.alias
    assert 'root.c' not in p.alias

def test_globals_with_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='CONST', ctx=Store())], value=Constant(value=6))
    p.globals('root', node)
    assert p.alias['root.CONST'] == '6'
    assert p.const['root.CONST'] == 'int'

def test_globals_with___all__():
    p = Parser()
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=List(elts=[Constant(value='x'), Constant(value='y')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.x', 'root.y'}


# LLM-generated content at query #27
#--------------------------

```python
def test_visit_Constant_with_valid_string():
    resolver = Resolver("root", {})
    node = Constant(value="valid_string")
    result = resolver.visit_Constant(node)
    assert result == resolver.visit(cast(Expr, parse("valid_string").body[0]).value)


# LLM-generated content at query #28
#--------------------------

```python
def test_is_enum_evaluates_to_true_when_bases_contain_enum():
    parser = Parser()
    class_name = "TestClass"
    root = "test_module"
    bases = ["enum.Enum"]
    body = []
    parser.class_api(root, class_name, bases, body)
    assert any(s.startswith('enum.') for s in parser.doc[class_name])


# LLM-generated content at query #29
#--------------------------

```python
def test_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(Name("attr1", Store()), Name("int", Load()), None, None),
        Assign([Name("attr2", Store())], Constant(42), None),
        Assign([Name("_private_attr", Store())], Constant(42), None),
        Delete([Name("attr2", Store())]),
        Assign([Name("attr3", Store())], Constant(42), None),
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "test_module.BaseClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr3" in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]
    assert "attr2" not in parser.doc[name]


# LLM-generated content at query #30
#--------------------------

```
def test_predicate_at_line_19_evaluates_to_false():
    parser = Parser()
    assign_node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=42))
    result = isinstance(assign_node, Assign) and len(assign_node.targets) == 1 and isinstance(assign_node.targets[0], Name)
    assert result is False


# LLM-generated content at query #31
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver("root", {}, "SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.Alias": "AliasType"}, "")
    node = Name("Alias", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "AliasType"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_typevar_alias():
    resolver = Resolver("root", {"root.TypeVar": "typing.TypeVar"}, "")
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_no_alias():
    resolver = Resolver("root", {}, "")
    node = Name("NoAlias", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "NoAlias"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_self_reference_alias():
    resolver = Resolver("root", {"root.Alias": "root.Alias"}, "")
    node = Name("Alias", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Alias"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #32
#--------------------------

```python
def test_func_ann_with_self_and_cls_method():
    parser = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='a', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'Any']

def test_func_ann_with_self_and_no_cls_method():
    parser = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='a', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'Any']

def test_func_ann_without_self():
    parser = Parser()
    args = [arg(arg='a', annotation=None), arg(arg='b', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']

def test_func_ann_with_annotation():
    parser = Parser()
    args = [arg(arg='a', annotation=Name(id='int', ctx=Load())), arg(arg='b', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', 'Any']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg(arg='*', annotation=None), arg(arg='a', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['', 'Any']


# LLM-generated content at query #33
#--------------------------

```python
def test_e_type_empty_sequence():
    assert _e_type([]) == ""

def test_e_type_none_element():
    assert _e_type([None]) == ""

def test_e_type_non_constant_element():
    assert _e_type([1]) == ""

def test_e_type_mixed_types():
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test_e_type_single_constant():
    assert _e_type([Constant(1)]) == "[int]"

def test_e_type_multiple_constants_same_type():
    assert _e_type([Constant(1), Constant(2)]) == "[int, int]"


# LLM-generated content at query #34
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "module"
    name = "module.MyClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "module"
    name = "module.MyClass"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        Assign(targets=[Name(id="ENUM_VALUE", ctx=Store())], value=Constant(value=1))
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "ENUM_VALUE" in parser.doc[name]

def test_class_api_with_members():
    parser = Parser()
    root = "module"
    name = "module.MyClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="member", ctx=Store()), annotation=Name(id="int", ctx=Load()))
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "member" in parser.doc[name]
    assert "int" in parser.doc[name]

def test_class_api_with_deleted_member():
    parser = Parser()
    root = "module"
    name = "module.MyClass"
    bases = []
    body = [
        Assign(targets=[Name(id="member", ctx=Store())], value=Constant(value=1)),
        Delete(targets=[Name(id="member", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #35
#--------------------------

```python
def test_is_public_returns_true_when_s_is_root_and_all_l_contains_s():
    parser = Parser()
    parser.imp = {"root": {"root"}}
    parser.root = {"root": "root"}
    parser.doc = {}
    parser.const = {}
    assert parser.is_public("root") == True

def test_is_public_returns_true_when_parent_s_is_in_all_l():
    parser = Parser()
    parser.imp = {"root": {"parent"}}
    parser.root = {"root.child": "root"}
    parser.doc = {}
    parser.const = {}
    assert parser.is_public("root.child") == True

def test_is_public_returns_true_when_s_is_public_family_and_all_l_is_empty():
    parser = Parser()
    parser.imp = {"root": set()}
    parser.root = {"root.public_method": "root"}
    parser.doc = {}
    parser.const = {}
    assert parser.is_public("root.public_method") == True


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    parser = Parser()
    root = "module"
    name = "module.Class"
    body = [
        Assign(targets=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())], value=Constant(value=10)),
        Assign(targets=[Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())], value=Constant(value=20))
    ]
    parser.class_api(root, name, [], body)


# LLM-generated content at query #37
#--------------------------

```python
def test_globals_ann_assign():
    node = AnnAssign(target=Name(id='x'), annotation=Name(id='int'), value=Constant(value=10))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias['root.x'] == '10'
    assert parser.const['root.x'] == 'int'

def test_globals_assign():
    node = Assign(targets=[Name(id='x')], value=Constant(value=10))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias['root.x'] == '10'
    assert parser.const['root.x'] == 'int'

def test_globals_assign_with_type_comment():
    node = Assign(targets=[Name(id='x')], value=Constant(value=10), type_comment='int')
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias['root.x'] == '10'
    assert parser.const['root.x'] == 'int'

def test_globals_assign_multiple_targets():
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=10))
    parser = Parser()
    parser.globals('root', node)
    assert 'root.x' not in parser.alias
    assert 'root.y' not in parser.alias

def test_globals_assign_non_name_target():
    node = Assign(targets=[Attribute(value=Name(id='obj'), attr='x')], value=Constant(value=10))
    parser = Parser()
    parser.globals('root', node)
    assert 'root.obj.x' not in parser.alias

def test_globals_assign_non_constant_value():
    node = Assign(targets=[Name(id='x')], value=BinOp(left=Constant(value=1), op=Add(), right=Constant(value=2)))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias['root.x'] == '1 + 2'
    assert parser.const['root.x'] == 'Any'

def test_globals_assign_all():
    node = Assign(targets=[Name(id='__all__')], value=List(elts=[Constant(value='x'), Constant(value='y')]))
    parser = Parser()
    parser.globals('root', node)
    assert parser.imp['root'] == {'root.x', 'root.y'}


# LLM-generated content at query #38
#--------------------------

```python
def test_visit_name_returns_self_when_node_id_matches_self_ty():
    resolver = Resolver(root="root", alias={}, self_ty="self_ty")
    node = Name(id="self_ty", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #39
#--------------------------

```python
def test_visit_Subscript_with_Union():
    resolver = Resolver(root="root", alias={})
    node = Subscript(value=Name(id="typing.Union", ctx=Load()), slice=Tuple(elts=[Name(id="A", ctx=Load()), Name(id="B", ctx=Load())], ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(node)
    expected = BinOp(Name(id="A", ctx=Load()), BitOr(), Name(id="B", ctx=Load()))
    assert result == expected

def test_visit_Subscript_with_Optional():
    resolver = Resolver(root="root", alias={})
    node = Subscript(value=Name(id="typing.Optional", ctx=Load()), slice=Name(id="A", ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(node)
    expected = BinOp(Name(id="A", ctx=Load()), BitOr(), Constant(None))
    assert result == expected

def test_visit_Subscript_with_PEP585():
    resolver = Resolver(root="root", alias={})
    node = Subscript(value=Name(id="typing.List", ctx=Load()), slice=Name(id="A", ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(node)
    expected = Subscript(Name(id="list", ctx=Load()), Name(id="A", ctx=Load()), Load())
    assert result == expected

def test_visit_Subscript_with_other_type():
    resolver = Resolver(root="root", alias={})
    node = Subscript(value=Name(id="typing.Dict", ctx=Load()), slice=Tuple(elts=[Name(id="A", ctx=Load()), Name(id="B", ctx=Load())], ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #40
#--------------------------

```python
def test_imports_with_import_statement():
    p = Parser()
    p.parse('test_module', 'import os')
    assert p.alias == {'test_module.os': 'os'}

def test_imports_with_import_as_statement():
    p = Parser()
    p.parse('test_module', 'import os as operating_system')
    assert p.alias == {'test_module.operating_system': 'os'}

def test_imports_with_from_import_statement():
    p = Parser()
    p.parse('test_module', 'from os import path')
    assert p.alias == {'test_module.path': 'os.path'}

def test_imports_with_from_import_as_statement():
    p = Parser()
    p.parse('test_module', 'from os import path as p')
    assert p.alias == {'test_module.p': 'os.path'}

def test_imports_with_relative_import():
    p = Parser()
    p.parse('test.module', 'from .. import utils')
    assert p.alias == {'test.module.utils': 'test.utils'}

def test_imports_with_relative_from_import():
    p = Parser()
    p.parse('test.module.sub', 'from .. import utils')
    assert p.alias == {'test.module.sub.utils': 'test.module.utils'}

def test_imports_with_multiple_imports():
    p = Parser()
    p.parse('test_module', 'import os, sys as system')
    assert p.alias == {'test_module.os': 'os', 'test_module.system': 'sys'}

def test_imports_with_multiple_from_imports():
    p = Parser()
    p.parse('test_module', 'from os import path, sep as separator')
    assert p.alias == {'test_module.path': 'os.path', 'test_module.separator': 'os.sep'}


# LLM-generated content at query #41
#--------------------------

```python
def test_class_api_is_public_family_false():
    parser = Parser()
    parser.class_api("root", "name", [], [])
    assert not parser.doc.get("name", "")


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_evaluates_to_true_when_types_differ():
    elements = [[Constant(1), Constant(2)], [Constant("a"), Constant("b")]]
    _e_type(*elements)


# LLM-generated content at query #43
#--------------------------

```python
def test_func_api_with_kwonlyargs():
    root = "module"
    name = "module.function"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg(arg="kwarg", annotation=None)],
        kw_defaults=[None],
        kwarg=None,
        defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False
    parser = Parser()
    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert '*' in parser.doc[name]


# LLM-generated content at query #44
#--------------------------

```python
def test_func_api_has_default_is_false():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=1)]
    )
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert not all(d is None for d in [None, Constant(value=1), None])


# LLM-generated content at query #45
#--------------------------

```python
def test_visit_name_with_alias_and_not_in_alias_of_itself():
    resolver = Resolver("root", {"root.name": "alias_value"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, AST)


# LLM-generated content at query #46
#--------------------------

```python
def test_is_public_family_evaluates_to_false_in_class_api():
    parser = Parser()
    body = [
        Assign(targets=[Name(id='_private_attr', ctx=Store())], value=Constant(value=42)),
        Assign(targets=[Name(id='__magic_attr__', ctx=Store())], value=Constant(value=42)),
    ]
    parser.class_api('root', 'name', [], body)
    assert 'private_attr' not in parser.doc['name']
    assert 'magic_attr' not in parser.doc['name']


# LLM-generated content at query #47
#--------------------------

```
def test_is_magic_predicate_evaluates_to_true():
    parser = Parser()
    parser.doc = {'__init__': 'docstring'}
    parser.docstring = {}
    parser.root = {'__init__': '__init__'}
    parser.level = {'__init__': 0}
    parser.imp = {'__init__': set()}
    parser.const = {}
    parser.toc = False
    result = parser.compile()
    assert result == '\n'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(
        target=Name(id='VAR', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    p.globals('root', node)
    assert p.alias['root.VAR'] == '42'
    assert p.const['root.VAR'] == 'int'
    assert p.root['root.VAR'] == 'root'

def test_globals_with_assign():
    p = Parser()
    node = Assign(
        targets=[Name(id='VAR', ctx=Store())],
        value=Constant(value=42),
        type_comment='int'
    )
    p.globals('root', node)
    assert p.alias['root.VAR'] == '42'
    assert p.const['root.VAR'] == 'int'
    assert p.root['root.VAR'] == 'root'

def test_globals_with_assign_no_type_comment():
    p = Parser()
    node = Assign(
        targets=[Name(id='VAR', ctx=Store())],
        value=Constant(value=42)
    )
    p.globals('root', node)
    assert p.alias['root.VAR'] == '42'
    assert p.const['root.VAR'] == 'int'
    assert p.root['root.VAR'] == 'root'

def test_globals_with_non_uppercase_var():
    p = Parser()
    node = Assign(
        targets=[Name(id='var', ctx=Store())],
        value=Constant(value=42),
        type_comment='int'
    )
    p.globals('root', node)
    assert p.alias['root.var'] == '42'
    assert 'root.var' not in p.const
    assert 'root.var' not in p.root

def test_globals_with___all__():
    p = Parser()
    node = Assign(
        targets=[Name(id='__all__', ctx=Store())],
        value=List(elts=[Constant(value='VAR')], ctx=Load())
    )
    p.globals('root', node)
    assert 'root.VAR' in p.imp['root']

def test_globals_with_invalid___all__():
    p = Parser()
    node = Assign(
        targets=[Name(id='__all__', ctx=Store())],
        value=Constant(value=42)
    )
    p.globals('root', node)
    assert not p.imp['root']

def test_globals_with_multiple_targets():
    p = Parser()
    node = Assign(
        targets=[Name(id='VAR1', ctx=Store()), Name(id='VAR2', ctx=Store())],
        value=Constant(value=42)
    )
    p.globals('root', node)
    assert 'root.VAR1' not in p.alias
    assert 'root.VAR2' not in p.alias

def test_globals_with_non_name_target():
    p = Parser()
    node = Assign(
        targets=[Subscript(value=Name(id='list', ctx=Load()), slice=Constant(value=0), ctx=Store())],
        value=Constant(value=42)
    )
    p.globals('root', node)
    assert not p.alias


# LLM-generated content at query #2
#--------------------------

```python
def test_const_type_with_int_constant():
    node = Constant(value=42)
    result = const_type(node)
    assert result == "int"

def test_const_type_with_str_constant():
    node = Constant(value="hello")
    result = const_type(node)
    assert result == "str"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[], ctx=Load())
    result = const_type(node)
    assert result == "tuple[]"

def test_const_type_with_tuple_of_ints():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    result = const_type(node)
    assert result == "tuple[int, int]"

def test_const_type_with_empty_list():
    node = List(elts=[], ctx=Load())
    result = const_type(node)
    assert result == "list[]"

def test_const_type_with_list_of_strs():
    node = List(elts=[Constant(value="a"), Constant(value="b")], ctx=Load())
    result = const_type(node)
    assert result == "list[str, str]"

def test_const_type_with_empty_set():
    node = Set(elts=[])
    result = const_type(node)
    assert result == "set[]"

def test_const_type_with_set_of_floats():
    node = Set(elts=[Constant(value=1.0), Constant(value=2.0)])
    result = const_type(node)
    assert result == "set[float, float]"

def test_const_type_with_empty_dict():
    node = Dict(keys=[], values=[])
    result = const_type(node)
    assert result == "dict[]"

def test_const_type_with_dict_of_int_to_str():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    result = const_type(node)
    assert result == "dict[int, int, str, str]"

def test_const_type_with_call_to_int():
    node = Call(func=Name(id="int", ctx=Load()), args=[Constant(value=42)], keywords=[])
    result = const_type(node)
    assert result == "int"

def test_const_type_with_call_to_str():
    node = Call(func=Name(id="str", ctx=Load()), args=[Constant(value="hello")], keywords=[])
    result = const_type(node)
    assert result == "str"

def test_const_type_with_call_to_unknown_function():
    node = Call(func=Name(id="unknown", ctx=Load()), args=[Constant(value=42)], keywords=[])
    result = const_type(node)
    assert result == "Any"


# LLM-generated content at query #3
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name(id='B', ctx=Load())], [])
    assert 'B' in parser.doc['test_module.A']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], [])
    assert 'enum.Enum' in parser.doc['test_module.A']

def test_class_api_with_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1)]
    parser.class_api('test_module', 'test_module.A', [], body)
    assert 'x' in parser.doc['test_module.A']

def test_class_api_with_enum_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=1))]
    parser.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], body)
    assert 'X' in parser.doc['test_module.A']

def test_class_api_with_deleted_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [
        AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1),
        Delete(targets=[Name(id='x', ctx=Del())])
    ]
    parser.class_api('test_module', 'test_module.A', [], body)
    assert 'x' not in parser.doc['test_module.A']

def test_class_api_with_private_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [AnnAssign(target=Name(id='_x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1)]
    parser.class_api('test_module', 'test_module.A', [], body)
    assert '_x' not in parser.doc['test_module.A']


# LLM-generated content at query #4
#--------------------------

```python
def test_parse_method_with_empty_script():
    p = Parser()
    p.parse("test_module", "")
    assert p.doc == {"test_module": "## Module `test_module`\n\n"}
    assert p.level == {"test_module": 0}
    assert p.imp == {"test_module": set()}
    assert p.root == {"test_module": "test_module"}
    assert p.docstring == {}

def test_parse_method_with_simple_script():
    p = Parser()
    script = "x = 1\ny = 2"
    p.parse("test_module", script)
    assert p.doc == {"test_module": "## Module `test_module`\n\n"}
    assert p.level == {"test_module": 0}
    assert p.imp == {"test_module": set()}
    assert p.root == {"test_module": "test_module"}
    assert p.docstring == {}

def test_parse_method_with_docstring():
    p = Parser()
    script = "'''Some docstring'''\nx = 1"
    p.parse("test_module", script)
    assert p.doc == {"test_module": "## Module `test_module`\n\n"}
    assert p.level == {"test_module": 0}
    assert p.imp == {"test_module": set()}
    assert p.root == {"test_module": "test_module"}
    assert p.docstring == {"test_module": "'''Some docstring'''"}

def test_parse_method_with_import():
    p = Parser()
    script = "import os\nx = 1"
    p.parse("test_module", script)
    assert p.doc == {"test_module": "## Module `test_module`\n\n"}
    assert p.level == {"test_module": 0}
    assert p.imp == {"test_module": set()}
    assert p.root == {"test_module": "test_module"}
    assert p.alias == {"test_module.os": "os"}
    assert p.docstring == {}

def test_parse_method_with_function_def():
    p = Parser()
    script = "def foo():\n    pass"
    p.parse("test_module", script)
    assert p.doc == {"test_module": "## Module `test_module`\n\n", "test_module.foo": "### foo()\n\n*Full name:* `test_module.foo`\n\n"}
    assert p.level == {"test_module": 0, "test_module.foo": 0}
    assert p.imp == {"test_module": set()}
    assert p.root == {"test_module": "test_module", "test_module.foo": "test_module"}
    assert p.docstring == {}


# LLM-generated content at query #5
#--------------------------

```python
def test_annassign_with_name_target_and_non_none_value():
    parser = Parser()
    node = AnnAssign(target=Name(id='x'), value=Constant(value=42), annotation=Name(id='int'))
    assert isinstance(node, AnnAssign) and isinstance(node.target, Name) and node.value is not None


# LLM-generated content at query #6
#--------------------------

```
def test_imports_simple_import():
    p = Parser()
    p.parse('test_module', "import os")
    assert p.alias == {'test_module.os': 'os'}

def test_imports_import_with_alias():
    p = Parser()
    p.parse('test_module', "import os as operating_system")
    assert p.alias == {'test_module.operating_system': 'os'}

def test_imports_from_import():
    p = Parser()
    p.parse('test_module', "from os import path")
    assert p.alias == {'test_module.path': 'os.path'}

def test_imports_from_import_with_alias():
    p = Parser()
    p.parse('test_module', "from os import path as p")
    assert p.alias == {'test_module.p': 'os.path'}

def test_imports_from_import_with_relative_level():
    p = Parser()
    p.parse('test.module', "from ..sub import func")
    assert p.alias == {'test.module.func': 'test.sub.func'}

def test_imports_multiple_imports():
    p = Parser()
    p.parse('test_module', "import os, sys as system")
    assert p.alias == {'test_module.os': 'os', 'test_module.system': 'sys'}


# LLM-generated content at query #7
#--------------------------

```python
def test_globals_predicate_evaluates_to_false():
    parser = Parser()
    node = Assign(targets=[Name(id='x', ctx=Store()), Name(id='y', ctx=Store())], value=Constant(value=42))
    parser.globals('root', node)
    assert 'x' not in parser.alias
    assert 'y' not in parser.alias


# LLM-generated content at query #8
#--------------------------

```python
def test_visit_Subscript_with_Union():
    resolver = Resolver("root", {})
    node = Subscript(Name("Union", Load()), Tuple([Name("A", Load()), Name("B", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)

def test_visit_Subscript_with_Optional():
    resolver = Resolver("root", {})
    node = Subscript(Name("Optional", Load()), Name("A", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

def test_visit_Subscript_with_PEP585_deprecated():
    resolver = Resolver("root", {})
    node = Subscript(Name("List", Load()), Name("A", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

def test_visit_Subscript_with_unknown_type():
    resolver = Resolver("root", {})
    node = Subscript(Name("Unknown", Load()), Name("A", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #9
#--------------------------

```python
def test_api_method_with_function_def():
    parser = Parser()
    parser.parse('root', 'def func(): pass')
    parser.api('root', FunctionDef(name='func', args=arguments(), body=[], decorator_list=[], returns=None))
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.func': '### func()\n\n*Full name:* `root.func`\n\n'}
    assert parser.docstring == {}

def test_api_method_with_async_function_def():
    parser = Parser()
    parser.parse('root', 'async def func(): pass')
    parser.api('root', AsyncFunctionDef(name='func', args=arguments(), body=[], decorator_list=[], returns=None))
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.func': '### async func()\n\n*Full name:* `root.func`\n\n'}
    assert parser.docstring == {}

def test_api_method_with_class_def():
    parser = Parser()
    parser.parse('root', 'class Cls: pass')
    parser.api('root', ClassDef(name='Cls', bases=[], body=[], decorator_list=[]))
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.Cls': '### class Cls\n\n*Full name:* `root.Cls`\n\n'}
    assert parser.docstring == {}

def test_api_method_with_decorators():
    parser = Parser()
    parser.parse('root', '@decorator\ndef func(): pass')
    parser.api('root', FunctionDef(name='func', args=arguments(), body=[], decorator_list=[Name(id='decorator')], returns=None))
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.func': '### func()\n\n*Full name:* `root.func`\n\n| Decorators |\n|:---:|\n| `@decorator` |\n\n'}
    assert parser.docstring == {}

def test_api_method_with_class_method():
    parser = Parser()
    parser.parse('root', '@classmethod\ndef func(): pass')
    parser.api('root', FunctionDef(name='func', args=arguments(), body=[], decorator_list=[Name(id='classmethod')], returns=None), prefix='Cls')
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.Cls.func': '#### func()\n\n*Full name:* `root.Cls.func`\n\n| Decorators |\n|:---:|\n| `@classmethod` |\n\n'}
    assert parser.docstring == {}

def test_api_method_with_static_method():
    parser = Parser()
    parser.parse('root', '@staticmethod\ndef func(): pass')
    parser.api('root', FunctionDef(name='func', args=arguments(), body=[], decorator_list=[Name(id='staticmethod')], returns=None), prefix='Cls')
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.Cls.func': '#### func()\n\n*Full name:* `root.Cls.func`\n\n| Decorators |\n|:---:|\n| `@staticmethod` |\n\n'}
    assert parser.docstring == {}

def test_api_method_with_docstring():
    parser = Parser()
    parser.parse('root', 'def func():\n    """docstring"""')
    parser.api('root', FunctionDef(name='func', args=arguments(), body=[], decorator_list=[], returns=None))
    assert parser.doc == {'root': '## Module `root`\n\n', 'root.func': '### func()\n\n*Full name:* `root.func`\n\n'}
    assert parser.docstring == {'root.func': '```python\n"""docstring"""\n```'}


# LLM-generated content at query #10
#--------------------------

```python
def test_func_api():
    parser = Parser()
    root = "module"
    name = "module.func"
    node = arguments(
        posonlyargs=[arg(arg="posonly")],
        args=[arg(arg="arg1"), arg(arg="arg2")],
        vararg=arg(arg="vararg"),
        kwonlyargs=[arg(arg="kwonly1"), arg(arg="kwonly2")],
        kw_defaults=[None, None],
        kwarg=arg(arg="kwarg"),
        defaults=[None, None]
    )
    returns = None
    parser.func_api(root, name, node, returns, has_self=True, cls_method=True)
    expected = (
        "#### func()\n\n"
        "*Full name:* `module.func`\n\n"
        "| posonly | / | arg1 | arg2 | *vararg | * | kwonly1 | kwonly2 | **kwarg | return |\n"
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        "| Self |  | Self | Self | Self |  | Self | Self | Self |  |\n"
    )
    assert parser.doc[name] == expected


# LLM-generated content at query #11
#--------------------------

```
def test_globals_const_condition_false():
    parser = Parser()
    root = "test_root"
    node = Assign(
        targets=[Name(id="TEST", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.const["test_root.TEST"] = "int"
    parser.globals(root, node)
    assert parser.const["test_root.TEST"] == "int"


# LLM-generated content at query #12
#--------------------------

```python
def test_class_api_AnnAssign_Name_condition():
    parser = Parser()
    root = "test_root"
    name = "test_name"
    bases = []
    body = [AnnAssign(target=Name(id="test_attr"), annotation=Name(id="int"))]
    parser.class_api(root, name, bases, body)
    assert "test_attr" in parser.doc[name]


# LLM-generated content at query #13
#--------------------------

```python
def test_parser_constructor_default_values():
    p = Parser()
    assert p.link == True
    assert p.b_level == 1
    assert p.toc == False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link == False
    assert p.b_level == 2
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link == True


# LLM-generated content at query #14
#--------------------------

```python
def test_api_method_without_link():
    parser = Parser(link=False)
    fake_node = FunctionDef(name="test_function", body=[], decorator_list=[])
    parser.api("root", fake_node)
    assert "<a id=" not in parser.doc["root.test_function"]


# LLM-generated content at query #15
#--------------------------

```python
def test_visit_Subscript_handles_typing_Union():
    resolver = Resolver("root", {})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

def test_visit_Subscript_handles_typing_Optional():
    resolver = Resolver("root", {})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

def test_visit_Subscript_handles_PEP585_deprecated_names():
    resolver = Resolver("root", {})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

def test_visit_Subscript_returns_node_for_non_typing_Union_or_Optional():
    resolver = Resolver("root", {})
    node = Subscript(Name("Dict", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #16
#--------------------------

```python
def test_func_api_has_default_false():
    parser = Parser()
    node = arguments(
        posonlyargs=[arg(arg='posonlyarg', annotation=None)],
        args=[arg(arg='arg1', annotation=None)],
        vararg=None,
        kwonlyargs=[arg(arg='kwonlyarg', annotation=None)],
        kw_defaults=[None],
        kwarg=None,
        defaults=[Name(id='default_value', ctx=Load())]
    )
    parser.func_api(root='root', name='name', node=node, returns=None, has_self=False, cls_method=False)
    assert '[ann]' not in parser.doc['name']  # Ensure the predicate evaluates to False


# LLM-generated content at query #17
#--------------------------

```
def test__defaults_with_none_values():
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_non_none_values():
    args = ["a", "b", "c"]
    result = list(_defaults(args))
    assert result == ["`a`", "`b`", "`c`"]

def test__defaults_with_mixed_values():
    args = [None, "a", None, "b"]
    result = list(_defaults(args))
    assert result == [" ", "`a`", " ", "`b`"]

def test__defaults_with_ampersand_value():
    args = ["a & b"]
    result = list(_defaults(args))
    assert result == ["<code>a & b</code>"]

def test__defaults_with_pipe_value():
    args = ["a | b"]
    result = list(_defaults(args))
    assert result == ["<code>a &#124; b</code>"]


# LLM-generated content at query #18
#--------------------------

```python
def test_visit_Attribute_with_typing_prefix():
    resolver = Resolver("root", {})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_without_typing_prefix():
    resolver = Resolver("root", {})
    node = Attribute(value=Name(id="module", ctx=Load()), attr="SomeClass", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result == node


# LLM-generated content at query #19
#--------------------------

```python
def test_is_public_with_public_family_name():
    parser = Parser()
    parser.root['test.public'] = 'test'
    assert parser.is_public('test.public') == True

def test_is_public_with_private_family_name():
    parser = Parser()
    parser.root['test._private'] = 'test'
    assert parser.is_public('test._private') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.root['test.__magic__'] = 'test'
    assert parser.is_public('test.__magic__') == False

def test_is_public_with_name_in_all_list():
    parser = Parser()
    parser.root['test.module'] = 'test'
    parser.imp['test'] = {'test.module.public'}
    assert parser.is_public('test.module.public') == True

def test_is_public_with_name_not_in_all_list():
    parser = Parser()
    parser.root['test.module'] = 'test'
    parser.imp['test'] = {'test.module.other'}
    assert parser.is_public('test.module.public') == False

def test_is_public_with_root_module():
    parser = Parser()
    parser.root['test'] = 'test'
    parser.imp['test'] = set()
    assert parser.is_public('test') == True

def test_is_public_with_child_of_root_in_all_list():
    parser = Parser()
    parser.root['test.module'] = 'test'
    parser.imp['test'] = {'test.module'}
    assert parser.is_public('test.module.child') == True

def test_is_public_with_parent_in_all_list():
    parser = Parser()
    parser.root['test.module.child'] = 'test'
    parser.imp['test'] = {'test.module'}
    assert parser.is_public('test.module.child') == True


# LLM-generated content at query #20
#--------------------------

```python
def test_visit_Subscript_with_non_name_value():
    resolver = Resolver("", {})
    node = Subscript(value=Constant(value=42), slice=Name(id="x", ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #21
#--------------------------

```python
def test_parser_default_constructor():
    p = Parser()
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


def test_parser_new_method():
    p = Parser.new(link=False, level=2, toc=True)
    assert p.link is False
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


def test_parser_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link is True
    assert p.toc is True


def test_parser_post_init_without_toc():
    p = Parser(toc=False)
    assert p.link is True
    assert p.toc is False


# LLM-generated content at query #22
#--------------------------

```python
def test_compile_with_toc():
    parser = Parser(toc=True)
    parser.doc = {"module.name": "## Module `module.name`\n\n"}
    parser.docstring = {"module.name": "Docstring content"}
    parser.level = {"module.name": 1}
    parser.root = {"module.name": "module"}
    parser.const = {}
    parser.imp = {"module": set()}
    result = parser.compile()
    expected = "**Table of contents:**\n    + [module.name](#module-name)\n\n## Module `module.name`\n\nDocstring content\n"
    assert result == expected

def test_compile_without_toc():
    parser = Parser(toc=False)
    parser.doc = {"module.name": "## Module `module.name`\n\n"}
    parser.docstring = {"module.name": "Docstring content"}
    parser.level = {"module.name": 1}
    parser.root = {"module.name": "module"}
    parser.const = {}
    parser.imp = {"module": set()}
    result = parser.compile()
    expected = "## Module `module.name`\n\nDocstring content\n"
    assert result == expected

def test_compile_with_constants():
    parser = Parser(toc=False)
    parser.doc = {"module.name": "## Module `module.name`\n\n"}
    parser.docstring = {"module.name": "Docstring content"}
    parser.level = {"module.name": 1}
    parser.root = {"module.name": "module"}
    parser.const = {"module.name.constant": "str"}
    parser.imp = {"module": {"module.name"}}
    result = parser.compile()
    expected = "## Module `module.name`\n\n| Constants | Type |\n|-----------|------|\n| constant | str |\n\nDocstring content\n"
    assert result == expected

def test_compile_with_missing_docstring():
    parser = Parser(toc=False)
    parser.doc = {"module.name": "## Module `module.name`\n\n"}
    parser.docstring = {}
    parser.level = {"module.name": 1}
    parser.root = {"module.name": "module"}
    parser.const = {}
    parser.imp = {"module": set()}
    result = parser.compile()
    expected = "## Module `module.name`\n\n"
    assert result == expected

def test_compile_with_magic_method():
    parser = Parser(toc=False)
    parser.doc = {"module.__magic__": "## Module `module.__magic__`\n\n"}
    parser.docstring = {"module.__magic__": "Docstring content"}
    parser.level = {"module.__magic__": 1}
    parser.root = {"module.__magic__": "module"}
    parser.const = {}
    parser.imp = {"module": set()}
    result = parser.compile()
    expected = ""
    assert result == expected


# LLM-generated content at query #23
#--------------------------

```python
def test_func_ann_with_self_and_cls_method():
    p = Parser()
    args = [arg(arg='self', annotation=Name(id='Self', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]']

def test_func_ann_with_self_no_cls_method():
    p = Parser()
    args = [arg(arg='self', annotation=Name(id='Self', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self']

def test_func_ann_with_annotation():
    p = Parser()
    args = [arg(arg='x', annotation=Name(id='int', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int']

def test_func_ann_without_annotation():
    p = Parser()
    args = [arg(arg='x', annotation=None)]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['typing.Any']

def test_func_ann_with_star_arg():
    p = Parser()
    args = [arg(arg='*', annotation=None)]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['']

def test_func_ann_with_multiple_args():
    p = Parser()
    args = [
        arg(arg='self', annotation=Name(id='Self', ctx=Load())),
        arg(arg='x', annotation=Name(id='int', ctx=Load())),
        arg(arg='y', annotation=None),
        arg(arg='*', annotation=None),
        arg(arg='z', annotation=Name(id='str', ctx=Load()))
    ]
    result = list(p.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'typing.Any', '', 'str']


# LLM-generated content at query #24
#--------------------------

```
def test_api_function_def():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[])
    p.api(root, node)
    assert "### test\\_func()" in p.doc["test_module.test_func"]
    assert "*Full name:* `test_module.test_func`" in p.doc["test_module.test_func"]

def test_api_async_function_def():
    p = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(), body=[], decorator_list=[])
    p.api(root, node)
    assert "### async test\\_async\\_func()" in p.doc["test_module.test_async_func"]
    assert "*Full name:* `test_module.test_async_func`" in p.doc["test_module.test_async_func"]

def test_api_class_def():
    p = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], body=[], decorator_list=[])
    p.api(root, node)
    assert "### class TestClass" in p.doc["test_module.TestClass"]
    assert "*Full name:* `test_module.TestClass`" in p.doc["test_module.TestClass"]

def test_api_with_decorators():
    p = Parser()
    root = "test_module"
    decorator = Name(id="decorator", ctx=Load())
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[decorator])
    p.api(root, node)
    assert "@decorator" in p.doc["test_module.test_func"]

def test_api_with_prefix():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[], returns=None)
    p.api(root, node, prefix="TestClass")
    assert "#### TestClass.test\\_func()" in p.doc["test_module.TestClass.test_func"]
    assert "*Full name:* `test_module.TestClass.test_func`" in p.doc["test_module.TestClass.test_func"]

def test_api_with_link():
    p = Parser(link=True)
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[])
    p.api(root, node)
    assert "<a id=\"test-module-test-func\"></a>" in p.doc["test_module.test_func"]

def test_api_with_docstring():
    p = Parser()
    root = "test_module"
    docstring = "This is a test function"
    node = FunctionDef(name="test_func", args=arguments(), body=[Expr(value=Constant(value=docstring))], decorator_list=[])
    p.api(root, node)
    assert "This is a test function" in p.docstring["test_module.test_func"]


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_false_when_node_is_not_assign():
    parser = Parser()
    node = AnnAssign(target=Name(id='x'), value=Constant(value=1), annotation=Name(id='int'))
    parser.globals('root', node)
    assert len(parser.alias) == 0

def test_predicate_evaluates_to_false_when_node_has_multiple_targets():
    parser = Parser()
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=1))
    parser.globals('root', node)
    assert len(parser.alias) == 0

def test_predicate_evaluates_to_false_when_target_is_not_name():
    parser = Parser()
    node = Assign(targets=[Attribute(value=Name(id='obj'), attr='attr')], value=Constant(value=1))
    parser.globals('root', node)
    assert len(parser.alias) == 0


# LLM-generated content at query #26
#--------------------------

```python
def test_func_ann_with_self_and_cls_method():
    parser = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='x', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'typing.Any']

def test_func_ann_without_self():
    parser = Parser()
    args = [arg(arg='x', annotation=None), arg(arg='y', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['typing.Any', 'typing.Any']

def test_func_ann_with_annotations():
    parser = Parser()
    args = [arg(arg='x', annotation=Name(id='int')), arg(arg='y', annotation=Name(id='str'))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

def test_func_ann_with_vararg():
    parser = Parser()
    args = [arg(arg='*', annotation=None), arg(arg='x', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['', 'typing.Any']

def test_func_ann_with_self_and_annotation():
    parser = Parser()
    args = [arg(arg='self', annotation=Name(id='MyClass')), arg(arg='x', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'typing.Any']


# LLM-generated content at query #27
#--------------------------

```python
def test_func_api_line_16_true():
    node = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg(arg='kwarg')], kw_defaults=[None], kwarg=None, defaults=[])
    parser = Parser()
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)


# LLM-generated content at query #28
#--------------------------

```python
def test_e_type_empty_elements():
    assert _e_type() == ""

def test_e_type_single_element_empty_sequence():
    assert _e_type([]) == ""

def test_e_type_single_element_non_constant():
    class NonConstant:
        pass
    assert _e_type([NonConstant()]) == ""

def test_e_type_single_element_single_constant():
    class Constant:
        pass
    assert _e_type([[Constant()]]) == "[Constant]"

def test_e_type_single_element_multiple_constants_same_type():
    class Constant:
        pass
    assert _e_type([[Constant(), Constant()]]) == "[Constant]"

def test_e_type_single_element_multiple_constants_different_types():
    class Constant1:
        pass
    class Constant2:
        pass
    assert _e_type([[Constant1(), Constant2()]]) == "[Any]"

def test_e_type_multiple_elements_same_type():
    class Constant:
        pass
    assert _e_type([[Constant()], [Constant()]]) == "[Constant, Constant]"

def test_e_type_multiple_elements_different_types():
    class Constant1:
        pass
    class Constant2:
        pass
    assert _e_type([[Constant1()], [Constant2()]]) == "[Constant1, Constant2]"

def test_e_type_multiple_elements_mixed_types():
    class Constant1:
        pass
    class Constant2:
        pass
    assert _e_type([[Constant1(), Constant2()], [Constant1()]]) == "[Any, Constant1]"

def test_e_type_multiple_elements_with_empty_sequence():
    class Constant:
        pass
    assert _e_type([[Constant()], []]) == ""


# LLM-generated content at query #29
#--------------------------

```python
def test_is_public_family_public_module():
    assert is_public_family("public.module.name") == True

def test_is_public_family_private_module():
    assert is_public_family("_private.module.name") == False

def test_is_public_family_magic_name():
    assert is_public_family("module.__magic__") == True

def test_is_public_family_mixed_public_and_magic():
    assert is_public_family("public.module.__magic__") == True

def test_is_public_family_mixed_private_and_magic():
    assert is_public_family("_private.module.__magic__") == False

def test_is_public_family_multilevel_magic():
    assert is_public_family("module.__magic__.__another_magic__") == True

def test_is_public_family_multilevel_private():
    assert is_public_family("module._private._another_private") == False

def test_is_public_family_multilevel_mixed():
    assert is_public_family("module.__magic__._private") == False

def test_is_public_family_single_public():
    assert is_public_family("public") == True

def test_is_public_family_single_private():
    assert is_public_family("_private") == False

def test_is_public_family_single_magic():
    assert is_public_family("__magic__") == True

def test_is_public_family_empty_string():
    assert is_public_family("") == True


# LLM-generated content at query #30
#--------------------------

```
def test_globals_const_get_returns_not_any():
    parser = Parser()
    parser.const["test"] = "some_value"
    node = Assign(targets=[Name(id="TEST", ctx=Store())], value=Constant(value=42))
    parser.globals("root", node)
    assert "TEST" not in parser.const


# LLM-generated content at query #31
#--------------------------

```
def test__e_type_empty_element():
    elements = [[]]
    result = _e_type(*elements)
    assert result == ""


# LLM-generated content at query #32
#--------------------------

```python
def test_class_api():
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    parser.resolve = lambda root, node: str(node)
    parser.class_api('root', 'root.Class', [], [])
    assert parser.doc['root.Class'] == "### class Class\n\n*Full name:* `root.Class`\n\n"
    assert parser.docstring == {}

    parser.doc = {}
    parser.docstring = {}
    bases = ['Base1', 'Base2']
    parser.class_api('root', 'root.Class', bases, [])
    assert parser.doc['root.Class'] == "### class Class\n\n*Full name:* `root.Class`\n\n| Bases |\n|:---:|\n| `Base1` |\n| `Base2` |\n\n"
    assert parser.docstring == {}

    parser.doc = {}
    parser.docstring = {}
    body = [AnnAssign(target=Name(id='attr1'), annotation=Name(id='int'), value=None)]
    parser.class_api('root', 'root.Class', [], body)
    assert parser.doc['root.Class'] == "### class Class\n\n*Full name:* `root.Class`\n\n| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n\n"
    assert parser.docstring == {}

    parser.doc = {}
    parser.docstring = {}
    body = [Assign(targets=[Name(id='attr1')], value=Constant(value=1))]
    parser.class_api('root', 'root.Class', ['enum.Enum'], body)
    assert parser.doc['root.Class'] == "### class Class\n\n*Full name:* `root.Class`\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n\n| Enums |\n|:---:|\n| attr1 |\n\n"
    assert parser.docstring == {}


# LLM-generated content at query #33
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver("root", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.name": "alias_value"}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "alias_value"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_typevar():
    resolver = Resolver("root", {"root.TypeVar": "typing.TypeVar"}, "")
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_no_alias():
    resolver = Resolver("root", {}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #34
#--------------------------

```python
def test_class_api_is_enum_true_when_base_starts_with_enum():
    parser = Parser()
    class_node = ClassDef(name='TestClass', bases=[Name(id='enum.Enum')], body=[], keywords=[], decorator_list=[])
    parser.class_api(root='test_root', name='test_name', bases=[Name(id='enum.Enum')], body=[])
    assert parser.doc['test_name'] == "### class TestClass\n\n*Full name:* `test_name`\n\n"


# LLM-generated content at query #35
#--------------------------

```python
def test_parser_constructor_default_values():
    p = Parser()
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link is False
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_with_toc_enables_link():
    p = Parser(toc=True)
    assert p.link is True
    assert p.toc is True


# LLM-generated content at query #36
#--------------------------

```python
def test_isinstance_node_Delete():
    parser = Parser()
    node = Delete(targets=[Name(id='x', ctx=Del())])
    assert isinstance(node, Delete)


# LLM-generated content at query #37
#--------------------------

```python
def test_func_ann_yield_self_for_first_arg_with_has_self():
    parser = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='other', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result[0] == 'Self'

def test_func_ann_yield_type_self_for_first_arg_with_has_self_and_cls_method():
    parser = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='other', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'

def test_func_ann_yield_empty_string_for_star_arg():
    parser = Parser()
    args = [arg(arg='*', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result[0] == ""

def test_func_ann_yield_resolved_annotation_for_arg_with_annotation():
    parser = Parser()
    args = [arg(arg='param', annotation='str')]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result[0] == parser.resolve('root', 'str')

def test_func_ann_yield_any_for_arg_without_annotation():
    parser = Parser()
    args = [arg(arg='param', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result[0] == ANY


# LLM-generated content at query #38
#--------------------------

```python
def test_is_public_returns_true_for_root_module():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root'] = 'root'
    assert parser.is_public('root') == True

def test_is_public_returns_true_for_public_name():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == True

def test_is_public_returns_true_for_name_in_all():
    parser = Parser()
    parser.imp['root'] = {'root.name'}
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == True

def test_is_public_returns_true_for_parent_in_all():
    parser = Parser()
    parser.imp['root'] = {'root'}
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == True

def test_is_public_returns_false_for_private_name():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root._name'] = 'root'
    assert parser.is_public('root._name') == False

def test_is_public_returns_false_for_name_not_in_all():
    parser = Parser()
    parser.imp['root'] = {'other'}
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == False


# LLM-generated content at query #39
#--------------------------

```
def test_func_api_with_posonlyargs():
    p = Parser()
    args = [arg('a', None), arg('b', None)]
    posonlyargs = [arg('x', None), arg('y', None)]
    node = arguments(posonlyargs=posonlyargs, args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', node, None, has_self=False, cls_method=False)


def test_func_api_with_vararg():
    p = Parser()
    args = [arg('a', None), arg('b', None)]
    vararg = arg('*args', None)
    node = arguments(posonlyargs=[], args=args, vararg=vararg, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', node, None, has_self=False, cls_method=False)


def test_func_api_with_kwonlyargs():
    p = Parser()
    args = [arg('a', None), arg('b', None)]
    kwonlyargs = [arg('x', None), arg('y', None)]
    kw_defaults = [None, None]
    node = arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, kwarg=None, defaults=[])
    p.func_api('root', 'name', node, None, has_self=False, cls_method=False)


def test_func_api_with_kwarg():
    p = Parser()
    args = [arg('a', None), arg('b', None)]
    kwarg = arg('**kwargs', None)
    node = arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=kwarg, defaults=[])
    p.func_api('root', 'name', node, None, has_self=False, cls_method=False)


def test_func_api_with_has_self():
    p = Parser()
    args = [arg('self', None), arg('a', None)]
    node = arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', node, None, has_self=True, cls_method=False)


def test_func_api_with_cls_method():
    p = Parser()
    args = [arg('cls', None), arg('a', None)]
    node = arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', node, None, has_self=True, cls_method=True)


def test_func_api_with_returns():
    p = Parser()
    args = [arg('a', None), arg('b', None)]
    returns = Name(id='int', ctx=Load())
    node = arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', node, returns, has_self=False, cls_method=False)


# LLM-generated content at query #40
#--------------------------

```python
def test_globals_predicate_at_line_23_evaluates_to_false():
    class Node:
        def __init__(self, targets, value, type_comment):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment
    
    class Name:
        def __init__(self, id):
            self.id = id

    node = Node(targets=[Name(id="x")], value="value", type_comment="type_comment")
    parser = Parser()
    parser.globals(root="root", node=node)
    assert parser.alias["root.x"] == "value"
    assert parser.const == {}


# LLM-generated content at query #41
#--------------------------

```python
def test_walk_body_with_if():
    class If:
        def __init__(self, body, orelse):
            self.body = body
            self.orelse = orelse

    class Stmt:
        pass

    stmt1 = Stmt()
    stmt2 = Stmt()
    stmt3 = Stmt()
    stmt4 = Stmt()

    if_node = If(body=[stmt1, stmt2], orelse=[stmt3])
    body = [if_node, stmt4]

    result = list(walk_body(body))
    assert result == [stmt1, stmt2, stmt3, stmt4]

def test_walk_body_with_try():
    class Try:
        def __init__(self, body, handlers, orelse, finalbody):
            self.body = body
            self.handlers = handlers
            self.orelse = orelse
            self.finalbody = finalbody

    class Handler:
        def __init__(self, body):
            self.body = body

    class Stmt:
        pass

    stmt1 = Stmt()
    stmt2 = Stmt()
    stmt3 = Stmt()
    stmt4 = Stmt()
    stmt5 = Stmt()
    stmt6 = Stmt()

    handler1 = Handler(body=[stmt2])
    handler2 = Handler(body=[stmt3])
    try_node = Try(body=[stmt1], handlers=[handler1, handler2], orelse=[stmt4], finalbody=[stmt5])
    body = [try_node, stmt6]

    result = list(walk_body(body))
    assert result == [stmt1, stmt2, stmt3, stmt4, stmt5, stmt6]

def test_walk_body_without_control_structures():
    class Stmt:
        pass

    stmt1 = Stmt()
    stmt2 = Stmt()
    body = [stmt1, stmt2]

    result = list(walk_body(body))
    assert result == [stmt1, stmt2]


# LLM-generated content at query #42
#--------------------------

```python
def test_func_ann_with_self_and_annotation():
    parser = Parser()
    args = [arg(arg='self', annotation=Name(id='Self', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self']


# LLM-generated content at query #43
#--------------------------

```python
def test_is_magic_predicate_evaluates_to_true():
    parser = Parser()
    parser.doc = {'__init__': 'docstring'}
    parser.root = {'__init__': 'root'}
    parser.level = {'__init__': 1}
    parser.imp = {'root': set()}
    parser.const = {}
    parser.docstring = {}
    parser.compile()


# LLM-generated content at query #44
#--------------------------

```python
def test_func_api_with_self_and_cls_method():
    parser = Parser()
    parser.func_api('root', 'name', arguments([], [], None, [], [], None, []), None, has_self=True, cls_method=True)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n\n"

def test_func_api_without_self_and_cls_method():
    parser = Parser()
    parser.func_api('root', 'name', arguments([], [], None, [], [], None, []), None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n\n"

def test_func_api_with_args_and_defaults():
    parser = Parser()
    args = arguments([arg('arg1', None)], [], None, [], [None], None, [])
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| arg1 |\n|:---:|\n| ANY |\n|   |\n\n"

def test_func_api_with_vararg_and_kwarg():
    parser = Parser()
    args = arguments([], [], arg('*args', None), [], [], arg('**kwargs', None), [])
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| *args | **kwargs |\n|:---:|:---:|\n| ANY | ANY |\n|   |   |\n\n"


# LLM-generated content at query #45
#--------------------------

```python
def test_globals_with_assign_multiple_targets():
    parser = Parser()
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=42))
    parser.globals('root', node)
    assert 'root.x' not in parser.alias
    assert 'root.y' not in parser.alias

def test_globals_with_assign_non_name_target():
    parser = Parser()
    node = Assign(targets=[Attribute(value=Name(id='obj'), attr='attr')], value=Constant(value=42))
    parser.globals('root', node)
    assert 'root.obj.attr' not in parser.alias

def test_globals_with_ann_assign_non_name_target():
    parser = Parser()
    node = AnnAssign(target=Attribute(value=Name(id='obj'), attr='attr'), annotation=Name(id='int'), value=Constant(value=42))
    parser.globals('root', node)
    assert 'root.obj.attr' not in parser.alias


