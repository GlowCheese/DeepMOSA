####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__m_with_single_name():
    assert _m("module") == "module"

def test__m_with_multiple_names():
    assert _m("module", "submodule", "component") == "module.submodule.component"

def test__m_with_empty_strings():
    assert _m("", "module", "", "submodule") == "module.submodule"

def test__m_with_all_empty_strings():
    assert _m("", "", "") == ""

def test__m_with_no_arguments():
    assert _m() == ""


# LLM-generated content at query #2
#--------------------------

```python
def test_imports_with_Import_node():
    parser = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname="system")])
    parser.imports(root, node)
    assert parser.alias["test.module.os"] == "os"
    assert parser.alias["test.module.system"] == "sys"

def test_imports_with_ImportFrom_node_no_level():
    parser = Parser()
    root = "test.module"
    node = ImportFrom(module="os.path", names=[alias(name="join", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test.module.join"] == "os.path.join"

def test_imports_with_ImportFrom_node_with_level():
    parser = Parser()
    root = "test.module.sub"
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    parser.imports(root, node)
    assert parser.alias["test.module.sub.path"] == "test.module.os.path"


# LLM-generated content at query #3
#--------------------------

```python
def test_compile_empty_parser():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.compile() == '\n'

def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc['test'] = '# Test'
    p.level['test'] = 0
    p.root['test'] = 'test'
    assert p.compile() == '**Table of contents:**\n\n+ [`test`](#test)\n\n# Test\n'

def test_compile_with_docstring():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test'] = '# Test'
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.docstring['test'] = 'This is a test.'
    assert p.compile() == '# Test\n\nThis is a test.\n'

def test_compile_with_magic_name():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['__init__'] = '# __init__'
    p.level['__init__'] = 0
    p.root['__init__'] = '__init__'
    assert p.compile() == '\n'

def test_compile_with_non_public_name():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['_private'] = '# _private'
    p.level['_private'] = 0
    p.root['_private'] = '_private'
    assert p.compile() == '\n'

def test_compile_with_nested_module():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test'] = '# Test'
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.doc['test.sub'] = '## Sub'
    p.level['test.sub'] = 1
    p.root['test.sub'] = 'test'
    assert p.compile() == '# Test\n\n## Sub\n'

def test_compile_with_constants():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test'] = '# Test'
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.const['test.CONST'] = 'int'
    p.imp['test'] = {'test.CONST'}
    assert p.compile() == '# Test\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n'

def test_compile_with_link():
    p = Parser.new(link=True, level=1, toc=False)
    p.doc['test'] = '# Test\n<a id="test"></a>'
    p.level['test'] = 0
    p.root['test'] = 'test'
    assert p.compile() == '# Test\n<a id="test"></a>\n'

def test_compile_with_missing_docstring():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test'] = '# Test'
    p.level['test'] = 0
    p.root['test'] = 'test'
    with pytest.warns(UserWarning, match="Missing documentation for test"):
        assert p.compile() == '# Test\n'


# LLM-generated content at query #4
#--------------------------

```python
def test_visit_Constant_with_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_invalid_syntax():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_valid_name():
    resolver = Resolver("root", {"root.name": "alias"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias"


# LLM-generated content at query #5
#--------------------------

```python
def test_parser_default_constructor():
    parser = Parser()
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_custom_constructor():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_new_method():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is False
    assert parser.b_level == 3
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_post_init_toc_true():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #6
#--------------------------

```python
def test_globals_with_annassign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="test_var"),
        annotation=Name(id="int"),
        value=Constant(value=42)
    )
    parser.globals("root", node)
    assert parser.alias["root.test_var"] == "42"
    assert parser.const["root.TEST_VAR"] == "int"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42)
    )
    parser.globals("root", node)
    assert parser.alias["root.test_var"] == "42"
    assert parser.const["root.TEST_VAR"] == "int"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="public_func")])
    )
    parser.globals("root", node)
    assert "root.public_func" in parser.imp["root"]

def test_globals_ignore_non_name_target():
    parser = Parser()
    node = Assign(
        targets=[Tuple(elts=[Name(id="a"), Name(id="b")])],
        value=Constant(value=42)
    )
    parser.globals("root", node)
    assert "root.a" not in parser.alias
    assert "root.b" not in parser.alias


# LLM-generated content at query #7
#--------------------------

```python
def test_func_api_no_args_no_return():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api('root', 'name', args, returns, has_self=False, cls_method=False)
    expected = "| return |\n|:---:|\n|  |\n\n"
    assert parser.doc['name'] == expected

def test_func_api_with_args():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('x', None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api('root', 'name', args, returns, has_self=False, cls_method=False)
    expected = "| x | return |\n|:---:|:---:|\n|  |  |\n\n"
    assert parser.doc['name'] == expected

def test_func_api_with_defaults():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('x', None)], kwonlyargs=[], kw_defaults=[], defaults=[Constant(value=1)])
    returns = None
    parser.func_api('root', 'name', args, returns, has_self=False, cls_method=False)
    expected = "| x | return |\n|:---:|:---:|\n| `1` |  |\n\n"
    assert parser.doc['name'] == expected

def test_func_api_with_self():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('self', None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api('root', 'name', args, returns, has_self=True, cls_method=False)
    expected = "| self | return |\n|:---:|:---:|\n| `Self` |  |\n\n"
    assert parser.doc['name'] == expected

def test_func_api_with_cls_method():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('cls', None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api('root', 'name', args, returns, has_self=True, cls_method=True)
    expected = "| cls | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n"
    assert parser.doc['name'] == expected

def test_func_api_with_annotations():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('x', Name(id='int'))], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = Name(id='str')
    parser.func_api('root', 'name', args, returns, has_self=False, cls_method=False)
    expected = "| x | return |\n|:---:|:---:|\n| `int` | `str` |\n\n"
    assert parser.doc['name'] == expected


# LLM-generated content at query #8
#--------------------------

```python
def test_is_public_with_all_listed():
    parser = Parser()
    parser.imp = {'pkg': {'mod1', 'mod1.submod'}}
    parser.root = {'pkg.mod1': 'pkg', 'pkg.mod1.submod': 'pkg.mod1'}
    parser.doc = {'pkg.mod1': '', 'pkg.mod1.submod': ''}
    assert parser.is_public('pkg.mod1')
    assert parser.is_public('pkg.mod1.submod')

def test_is_public_without_all_listed():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg.mod1': 'pkg', 'pkg.mod1.submod': 'pkg.mod1'}
    parser.doc = {'pkg.mod1': '', 'pkg.mod1.submod': ''}
    assert parser.is_public('pkg.mod1')
    assert parser.is_public('pkg.mod1.submod')

def test_is_public_private_name():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg._private': 'pkg'}
    parser.doc = {'pkg._private': ''}
    assert not parser.is_public('pkg._private')

def test_is_public_not_in_all():
    parser = Parser()
    parser.imp = {'pkg': {'mod1'}}
    parser.root = {'pkg.mod2': 'pkg'}
    parser.doc = {'pkg.mod2': ''}
    assert not parser.is_public('pkg.mod2')

def test_is_public_magic_name():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg.__init__': 'pkg'}
    parser.doc = {'pkg.__init__': ''}
    assert parser.is_public('pkg.__init__')


# LLM-generated content at query #9
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42),
        type_comment=None
    )
    parser.globals(root, node)
    assert parser.imp[root] == set()


# LLM-generated content at query #10
#--------------------------

```python
def test_parser_constructor_default():
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

def test_parser_constructor_with_toc():
    p = Parser.new(link=False, level=2, toc=True)
    assert p.link == True  # link should be True when toc is True
    assert p.b_level == 2
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_func_api_with_vararg():
    parser = Parser()
    root = "test_module"
    name = "test_function"
    node = arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=arg(arg="args", annotation=None),
        kwarg=None
    )
    returns = None
    has_self = False
    cls_method = False
    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert True


# LLM-generated content at query #12
#--------------------------

```python
def test__e_type_with_empty_elements():
    assert _e_type() == ""

def test__e_type_with_none_element():
    assert _e_type([None]) == ""

def test__e_type_with_non_constant_element():
    assert _e_type([1, "a"]) == ""

def test__e_type_with_single_constant_element():
    assert _e_type([Constant(1)]) == "[int]"

def test__e_type_with_multiple_constant_elements_of_same_type():
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_with_multiple_constant_elements_of_different_types():
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_with_multiple_sequences_of_constants():
    assert _e_type([Constant(1), Constant(2)], [Constant("a"), Constant("b")]) == "[int, str]"


# LLM-generated content at query #13
#--------------------------

```python
def test_is_public_family_magic_name():
    assert is_public_family('__init__') == True
    assert is_public_family('__main__') == True
    assert is_public_family('__name__') == True

def test_is_public_family_public_name():
    assert is_public_family('os.path') == True
    assert is_public_family('sys.argv') == True
    assert is_public_family('math.sqrt') == True

def test_is_public_family_private_name():
    assert is_public_family('_private') == False
    assert is_public_family('os._path') == False
    assert is_public_family('sys._argv') == False

def test_is_public_family_local_name():
    assert is_public_family('_local') == False
    assert is_public_family('module._local') == False
    assert is_public_family('package.module._local') == False

def test_is_public_family_mixed_name():
    assert is_public_family('os.__init__') == True
    assert is_public_family('sys.__main__') == True
    assert is_public_family('math.__name__') == True
    assert is_public_family('os._path.__init__') == False
    assert is_public_family('sys._argv.__main__') == False
    assert is_public_family('math._sqrt.__name__') == False


# LLM-generated content at query #14
#--------------------------

```python
def test_imports_with_Import_node():
    p = Parser()
    node = Import(names=[alias(name='os', asname=None)])
    p.imports('pkg', node)
    assert p.alias['pkg.os'] == 'os'


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_35_evaluates_to_false():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Tuple(elts=[Constant(value='foo')]))
    p.globals('root', node)
    assert '__all__' in p.imp['root']


# LLM-generated content at query #16
#--------------------------

```python
def test_visit_Name_self_ty():
    resolver = Resolver("root", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.name": "alias.value"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "value"

def test_visit_Name_without_alias():
    resolver = Resolver("root", {}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_TypeVar_alias():
    resolver = Resolver("root", {"root.name": "typing.TypeVar('T')"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_non_TypeVar_call_alias():
    resolver = Resolver("root", {"root.name": "some_func()"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Call)


# LLM-generated content at query #17
#--------------------------

```python
def test_load_docstring_basic():
    p = Parser()
    p.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.func': 'func()'}
    p.docstring = {}
    m = type('module', (), {'func': lambda: None})()
    m.func.__doc__ = "Function doc"
    p.load_docstring('pkg.module', m)
    assert p.docstring['pkg.module.func'] == "```python\nFunction doc\n```"

def test_load_docstring_nested():
    p = Parser()
    p.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.Class.method': 'method()'}
    p.docstring = {}
    m = type('module', (), {'Class': type('Class', (), {'method': lambda: None})()})()
    m.Class.method.__doc__ = "Method doc"
    p.load_docstring('pkg.module', m)
    assert p.docstring['pkg.module.Class.method'] == "```python\nMethod doc\n```"

def test_load_docstring_none_doc():
    p = Parser()
    p.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.func': 'func()'}
    p.docstring = {}
    m = type('module', (), {'func': lambda: None})()
    p.load_docstring('pkg.module', m)
    assert p.docstring == {}

def test_load_docstring_partial_match():
    p = Parser()
    p.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.func': 'func()', 'pkg.other.func': 'func()'}
    p.docstring = {}
    m = type('module', (), {'func': lambda: None})()
    m.func.__doc__ = "Function doc"
    p.load_docstring('pkg.module', m)
    assert p.docstring == {'pkg.module.func': "```python\nFunction doc\n```"}
    assert 'pkg.other.func' not in p.docstring


# LLM-generated content at query #18
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=1)
    )
    parser.globals("test", node)
    assert parser.alias["test.x"] == "1"
    assert parser.const["test.x"] == "int"
    assert parser.root["test.x"] == "test"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value="hello")
    )
    parser.globals("test", node)
    assert parser.alias["test.y"] == "'hello'"
    assert parser.const["test.y"] == "str"
    assert parser.root["test.y"] == "test"

def test_globals_with_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="z", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals("test", node)
    assert parser.alias["test.z"] == "3.14"
    assert parser.const["test.z"] == "float"
    assert parser.root["test.z"] == "test"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("test", node)
    assert parser.imp["test"] == {"test.func1", "test.func2"}

def test_globals_with_non_uppercase():
    parser = Parser()
    node = Assign(
        targets=[Name(id="non_upper", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals("test", node)
    assert parser.alias["test.non_upper"] == "42"
    assert "test.non_upper" not in parser.const
    assert "test.non_upper" not in parser.root


# LLM-generated content at query #19
#--------------------------

```python
def test_is_public_with_all_listed():
    parser = Parser()
    parser.imp = {'pkg': {'func', 'cls'}}
    parser.root = {'pkg.func': 'pkg', 'pkg.cls': 'pkg'}
    parser.doc = {'pkg.func': '', 'pkg.cls': ''}
    assert parser.is_public('pkg.func') == True
    assert parser.is_public('pkg.cls') == True

def test_is_public_without_all_listed():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg.func': 'pkg', 'pkg.cls': 'pkg'}
    parser.doc = {'pkg.func': '', 'pkg.cls': ''}
    assert parser.is_public('pkg.func') == True
    assert parser.is_public('pkg.cls') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg._private': 'pkg'}
    parser.doc = {'pkg._private': ''}
    assert parser.is_public('pkg._private') == False

def test_is_public_with_nested_public():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg.sub': 'pkg', 'pkg.sub.func': 'pkg.sub'}
    parser.doc = {'pkg.sub': '', 'pkg.sub.func': ''}
    assert parser.is_public('pkg.sub.func') == True

def test_is_public_with_nested_private():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.root = {'pkg.sub': 'pkg', 'pkg.sub._private': 'pkg.sub'}
    parser.doc = {'pkg.sub': '', 'pkg.sub._private': ''}
    assert parser.is_public('pkg.sub._private') == False


# LLM-generated content at query #20
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
    assert p.link is True  # toc=True forces link=True
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_class_api_no_bases_no_members():
    parser = Parser()
    node = ClassDef(name='TestClass', bases=[], body=[], decorator_list=[])
    parser.class_api('root', 'root.TestClass', node.bases, node.body)
    assert parser.doc['root.TestClass'] == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n'

def test_class_api_with_bases():
    parser = Parser()
    node = ClassDef(name='TestClass', bases=[Name(id='Base1'), Name(id='Base2')], body=[], decorator_list=[])
    parser.class_api('root', 'root.TestClass', node.bases, node.body)
    assert parser.doc['root.TestClass'] == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n| Bases |\n|:---:|\n| `Base1` |\n| `Base2` |\n'

def test_class_api_with_members():
    parser = Parser()
    node = ClassDef(name='TestClass', bases=[], body=[
        AnnAssign(target=Name(id='attr1'), annotation=Name(id='int'), value=None),
        AnnAssign(target=Name(id='attr2'), annotation=Name(id='str'), value=None)
    ], decorator_list=[])
    parser.class_api('root', 'root.TestClass', node.bases, node.body)
    assert parser.doc['root.TestClass'] == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n'

def test_class_api_with_enum():
    parser = Parser()
    node = ClassDef(name='TestEnum', bases=[Name(id='enum.Enum')], body=[
        AnnAssign(target=Name(id='A'), annotation=None, value=Constant(value=1)),
        AnnAssign(target=Name(id='B'), annotation=None, value=Constant(value=2))
    ], decorator_list=[])
    parser.class_api('root', 'root.TestEnum', node.bases, node.body)
    assert parser.doc['root.TestEnum'] == '### class TestEnum\n\n*Full name:* `root.TestEnum`\n<a id="root-testenum"></a>\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n| Enums |\n|:---:|\n| A |\n| B |\n'

def test_class_api_with_deleted_members():
    parser = Parser()
    node = ClassDef(name='TestClass', bases=[], body=[
        AnnAssign(target=Name(id='attr1'), annotation=Name(id='int'), value=None),
        Delete(targets=[Name(id='attr1')])
    ], decorator_list=[])
    parser.class_api('root', 'root.TestClass', node.bases, node.body)
    assert parser.doc['root.TestClass'] == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n'

def test_class_api_private_members_ignored():
    parser = Parser()
    node = ClassDef(name='TestClass', bases=[], body=[
        AnnAssign(target=Name(id='_private'), annotation=Name(id='int'), value=None),
        AnnAssign(target=Name(id='public'), annotation=Name(id='str'), value=None)
    ], decorator_list=[])
    parser.class_api('root', 'root.TestClass', node.bases, node.body)
    assert parser.doc['root.TestClass'] == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n| Members | Type |\n|:---:|:---:|\n| `public` | `str` |\n'


# LLM-generated content at query #22
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="VALUE1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        AnnAssign(target=Name(id="VALUE2", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]

def test_class_api_with_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        AnnAssign(target=Name(id="attr2", ctx=Store()), annotation=Name(id="str", ctx=Load()), value=None),
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]

def test_class_api_with_deleted_member():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Delete(targets=[Name(id="attr1", ctx=Del())]),
    ]
    parser.class_api(root, name, bases, body)
    assert "attr1" not in parser.doc[name]

def test_class_api_with_private_member():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="_private_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
    ]
    parser.class_api(root, name, bases, body)
    assert "_private_attr" not in parser.doc[name]

def test_class_api_with_assign_member():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        Assign(targets=[Name(id="attr1", ctx=Store())], value=Constant(value=42)),
    ]
    parser.class_api(root, name, bases, body)
    assert "attr1" in parser.doc[name]
    assert "int" in parser.doc[name]


# LLM-generated content at query #23
#--------------------------

```python
def test_load_docstring_basic():
    p = Parser()
    p.doc = {'pkg.submod.func': 'Function doc', 'pkg.submod': 'Module doc'}
    p.docstring = {}
    p.root = {'pkg.submod.func': 'pkg.submod', 'pkg.submod': 'pkg.submod'}
    m = type('test_module', (), {'func': lambda: None})()
    m.func.__doc__ = "New function doc"
    p.load_docstring('pkg.submod', m)
    assert p.docstring == {'pkg.submod.func': "```python\nNew function doc\n```"}

def test_load_docstring_nested():
    p = Parser()
    p.doc = {'pkg.submod.Class.method': 'Method doc', 'pkg.submod.Class': 'Class doc'}
    p.docstring = {}
    p.root = {'pkg.submod.Class.method': 'pkg.submod', 'pkg.submod.Class': 'pkg.submod'}
    m = type('test_module', (), {'Class': type('Class', (), {'method': lambda: None})})()
    m.Class.method.__doc__ = "New method doc"
    p.load_docstring('pkg.submod', m)
    assert p.docstring == {'pkg.submod.Class.method': "```python\nNew method doc\n```"}

def test_load_docstring_none_doc():
    p = Parser()
    p.doc = {'pkg.submod.func': 'Function doc'}
    p.docstring = {}
    p.root = {'pkg.submod.func': 'pkg.submod'}
    m = type('test_module', (), {'func': lambda: None})()
    m.func.__doc__ = None
    p.load_docstring('pkg.submod', m)
    assert p.docstring == {}


# LLM-generated content at query #24
#--------------------------

```python
def test_is_enum_predicate():
    p = Parser()
    p.doc = {}
    p.alias = {}
    p.const = {}
    p.root = {}
    p.level = {}
    p.imp = {}
    p.b_level = 1
    p.link = True
    p.toc = False
    p.docstring = {}

    class MockExpr:
        def __init__(self, value):
            self.value = value

    class MockStmt:
        def __init__(self, type, target=None, annotation=None, value=None, type_comment=None):
            self.type = type
            self.target = target
            self.annotation = annotation
            self.value = value
            self.type_comment = type_comment

    class MockName:
        def __init__(self, id):
            self.id = id

    class MockAnnAssign(MockStmt):
        def __init__(self, target, annotation, value=None):
            super().__init__('AnnAssign', target, annotation, value)

    class MockAssign(MockStmt):
        def __init__(self, targets, value, type_comment=None):
            super().__init__('Assign', targets[0] if targets else None, None, value, type_comment)
            self.targets = targets

    class MockDelete(MockStmt):
        def __init__(self, targets):
            super().__init__('Delete')
            self.targets = targets

    # Mock walk_body to return a list of nodes
    def mock_walk_body(body):
        return body

    # Mock table to return a string
    def mock_table(*args, items=None):
        return "table"

    # Mock code to return the input as string
    def mock_code(s):
        return str(s)

    # Mock is_public_family to return True
    def mock_is_public_family(s):
        return True

    # Mock const_type to return a string
    def mock_const_type(value):
        return "type"

    # Mock resolve to return a string
    def mock_resolve(root, node, self_ty=""):
        return "resolved_type"

    # Mock get_docstring to return None
    def mock_get_docstring(node):
        return None

    # Mock doctest to return the input
    def mock_doctest(doc):
        return doc

    # Mock esc_underscore to return the input
    def mock_esc_underscore(s):
        return s

    # Mock is_magic to return False
    def mock_is_magic(s):
        return False

    # Mock parent to return a string
    def mock_parent(s, level=1):
        return s.rsplit('.', level)[0]

    # Mock _m to return a string
    def mock_m(*args):
        return '.'.join(args)

    # Mock logger to do nothing
    class MockLogger:
        def warning(self, msg):
            pass

    # Replace the methods
    p.walk_body = mock_walk_body
    p.table = mock_table
    p.code = mock_code
    p.is_public_family = mock_is_public_family
    p.const_type = mock_const_type
    p.resolve = mock_resolve
    p.get_docstring = mock_get_docstring
    p.doctest = mock_doctest
    p.esc_underscore = mock_esc_underscore
    p.is_magic = mock_is_magic
    p.parent = mock_parent
    p._m = mock_m
    p.logger = MockLogger()

    # Create a mock node for AnnAssign
    node = MockAnnAssign(MockName('attr'), MockExpr('annotation'))

    # Call class_api with is_enum=True
    p.class_api('root', 'name', [MockExpr('enum.Enum')], [node])

    # Check that enums contains 'attr'
    assert 'attr' in p.enums


# LLM-generated content at query #25
#--------------------------

```python
def test_is_public_with_all_listed():
    p = Parser()
    p.imp = {'pkg': {'mod1', 'mod2'}}
    p.root = {'pkg.mod1': 'pkg', 'pkg.mod2': 'pkg'}
    p.doc = {'pkg.mod1': '', 'pkg.mod2': ''}
    p.const = {}
    assert p.is_public('pkg.mod1') == True
    assert p.is_public('pkg.mod2') == True

def test_is_public_without_all():
    p = Parser()
    p.imp = {'pkg': set()}
    p.root = {'pkg.mod1': 'pkg'}
    p.doc = {'pkg.mod1': ''}
    p.const = {}
    assert p.is_public('pkg.mod1') == True

def test_is_public_private_name():
    p = Parser()
    p.imp = {'pkg': set()}
    p.root = {'pkg._mod1': 'pkg'}
    p.doc = {'pkg._mod1': ''}
    p.const = {}
    assert p.is_public('pkg._mod1') == False

def test_is_public_magic_name():
    p = Parser()
    p.imp = {'pkg': set()}
    p.root = {'pkg.__init__': 'pkg'}
    p.doc = {'pkg.__init__': ''}
    p.const = {}
    assert p.is_public('pkg.__init__') == True

def test_is_public_not_in_all():
    p = Parser()
    p.imp = {'pkg': {'mod1'}}
    p.root = {'pkg.mod2': 'pkg'}
    p.doc = {'pkg.mod2': ''}
    p.const = {}
    assert p.is_public('pkg.mod2') == False


# LLM-generated content at query #26
#--------------------------

```python
def test_is_public_with_imp_and_valid_child():
    p = Parser()
    p.imp = {'pkg': {'mod'}}
    p.doc = {'pkg.mod': 'doc'}
    p.const = {}
    p.root = {'pkg.mod': 'pkg'}
    assert p.is_public('pkg') == True


# LLM-generated content at query #27
#--------------------------

```python
def test_const_type_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_constant_float():
    node = Constant(value=3.14)
    assert const_type(node) == "float"

def test_const_type_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_tuple_homogeneous():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

def test_const_type_tuple_heterogeneous():
    node = Tuple(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "tuple[Any, Any]"

def test_const_type_list_homogeneous():
    node = List(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "list[int]"

def test_const_type_list_heterogeneous():
    node = List(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "list[Any]"

def test_const_type_set_homogeneous():
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "set[int]"

def test_const_type_set_heterogeneous():
    node = Set(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "set[Any]"

def test_const_type_dict_homogeneous():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "dict[int, str]"

def test_const_type_dict_heterogeneous():
    node = Dict(keys=[Constant(value=1), Constant(value="a")], values=[Constant(value="b"), Constant(value=2)])
    assert const_type(node) == "dict[Any, Any]"

def test_const_type_call_bool():
    node = Call(func=Name(id="bool"), args=[])
    assert const_type(node) == "bool"

def test_const_type_call_int():
    node = Call(func=Name(id="int"), args=[])
    assert const_type(node) == "int"

def test_const_type_call_unknown():
    node = Call(func=Name(id="unknown"), args=[])
    assert const_type(node) == "Any"

def test_const_type_empty_tuple():
    node = Tuple(elts=[])
    assert const_type(node) == "tuple[]"

def test_const_type_empty_list():
    node = List(elts=[])
    assert const_type(node) == "list[]"

def test_const_type_empty_set():
    node = Set(elts=[])
    assert const_type(node) == "set[]"

def test_const_type_empty_dict():
    node = Dict(keys=[], values=[])
    assert const_type(node) == "dict[, ]"

def test_const_type_none_in_elements():
    node = Tuple(elts=[Constant(value=1), None])
    assert const_type(node) == "Any"

def test_const_type_non_constant_in_tuple():
    node = Tuple(elts=[Constant(value=1), Name(id="x")])
    assert const_type(node) == "Any"


# LLM-generated content at query #28
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is True  # toc overrides link
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_walk_body_empty_sequence():
    assert list(walk_body([])) == []

def test_walk_body_single_statement():
    stmt = stmt()
    assert list(walk_body([stmt])) == [stmt]

def test_walk_body_if_statement():
    if_node = If([stmt()], [stmt()])
    assert list(walk_body([if_node])) == [stmt(), stmt()]

def test_walk_body_try_statement():
    try_node = Try([stmt()], [ExceptHandler([stmt()])], [stmt()], [stmt()])
    assert list(walk_body([try_node])) == [stmt(), stmt(), stmt(), stmt()]

def test_walk_body_nested_if_statements():
    inner_if = If([stmt()], [stmt()])
    outer_if = If([inner_if], [stmt()])
    assert list(walk_body([outer_if])) == [stmt(), stmt(), stmt()]

def test_walk_body_mixed_statements():
    if_node = If([stmt()], [stmt()])
    try_node = Try([stmt()], [ExceptHandler([stmt()])], [], [])
    assert list(walk_body([if_node, try_node])) == [stmt(), stmt(), stmt(), stmt()]


# LLM-generated content at query #30
#--------------------------

```python
def test_imports_with_importfrom_node():
    p = Parser()
    node = ImportFrom(module='os.path', names=[alias(name='join', asname='j')], level=0)
    p.imports('pkg', node)
    assert p.alias['pkg.j'] == 'os.path.join'


# LLM-generated content at query #31
#--------------------------

```python
def test_parser_constructor_defaults():
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

def test_parser_constructor_with_values():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is False
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_post_init_toc_sets_link():
    p = Parser(toc=True)
    assert p.link is True
    assert p.toc is True


# LLM-generated content at query #32
#--------------------------

```python
def test_globals_predicate_false():
    p = Parser()
    node = Assign(targets=[Name(id='x')], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.x' not in p.imp['root']


# LLM-generated content at query #33
#--------------------------

```python
def test_api_function():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'def func(): pass')
    assert parser.doc['test.func'] == '### func()\n\n*Full name:* `test.func`\n\n'
    assert parser.level['test.func'] == 0
    assert parser.root['test.func'] == 'test'

def test_api_async_function():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'async def async_func(): pass')
    assert parser.doc['test.async_func'] == '### async async_func()\n\n*Full name:* `test.async_func`\n\n'
    assert parser.level['test.async_func'] == 0
    assert parser.root['test.async_func'] == 'test'

def test_api_class():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'class MyClass: pass')
    assert parser.doc['test.MyClass'] == '### class MyClass\n\n*Full name:* `test.MyClass`\n\n'
    assert parser.level['test.MyClass'] == 0
    assert parser.root['test.MyClass'] == 'test'

def test_api_with_prefix():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', '''
class MyClass:
    def method(self):
        pass
''')
    assert parser.doc['test.MyClass.method'] == '#### method()\n\n*Full name:* `test.MyClass.method`\n\n'
    assert parser.level['test.MyClass.method'] == 0
    assert parser.root['test.MyClass.method'] == 'test'

def test_api_with_decorators():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', '''
@decorator
def decorated_func():
    pass
''')
    assert 'Decorators' in parser.doc['test.decorated_func']
    assert parser.doc['test.decorated_func'].startswith('### decorated_func()\n\n*Full name:* `test.decorated_func`\n\n')

def test_api_with_docstring():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', '''
def documented_func():
    """This is a docstring."""
    pass
''')
    assert parser.doc['test.documented_func'].startswith('### documented_func()\n\n*Full name:* `test.documented_func`\n\n')
    assert 'This is a docstring.' in parser.docstring['test.documented_func']


# LLM-generated content at query #34
#--------------------------

```python
def test_parse_basic_module():
    p = Parser()
    script = """
\"\"\"Module docstring.\"\"\"
x = 1
def foo():
    \"\"\"Function docstring.\"\"\"
    pass
"""
    p.parse("test_module", script)
    assert "test_module" in p.doc
    assert "test_module.foo" in p.doc
    assert "test_module.x" in p.alias
    assert p.docstring["test_module"] == "```python\n\"\"\"Module docstring.\"\"\"\n```"
    assert p.docstring["test_module.foo"] == "```python\n\"\"\"Function docstring.\"\"\"\n```"

def test_parse_with_imports():
    p = Parser()
    script = """
from typing import List
import os
x = 1
"""
    p.parse("test_module", script)
    assert "typing.List" in p.alias["test_module.List"]
    assert "os" in p.alias["test_module.os"]
    assert "test_module.x" in p.alias

def test_parse_class_and_methods():
    p = Parser()
    script = """
class MyClass:
    \"\"\"Class docstring.\"\"\"
    def method(self):
        \"\"\"Method docstring.\"\"\"
        pass
    @staticmethod
    def static_method():
        pass
"""
    p.parse("test_module", script)
    assert "test_module.MyClass" in p.doc
    assert "test_module.MyClass.method" in p.doc
    assert "test_module.MyClass.static_method" in p.doc
    assert p.docstring["test_module.MyClass"] == "```python\n\"\"\"Class docstring.\"\"\"\n```"
    assert p.docstring["test_module.MyClass.method"] == "```python\n\"\"\"Method docstring.\"\"\"\n```"

def test_parse_with_annotations():
    p = Parser()
    script = """
from typing import Optional
def foo(x: int, y: Optional[str] = None) -> bool:
    \"\"\"Function with annotations.\"\"\"
    return True
"""
    p.parse("test_module", script)
    assert "test_module.foo" in p.doc
    assert p.docstring["test_module.foo"] == "```python\n\"\"\"Function with annotations.\"\"\"\n```"

def test_parse_with_decorators():
    p = Parser()
    script = """
def decorator(func):
    return func
@decorator
def foo():
    \"\"\"Decorated function.\"\"\"
    pass
"""
    p.parse("test_module", script)
    assert "test_module.foo" in p.doc
    assert "@decorator" in p.doc["test_module.foo"]
    assert p.docstring["test_module.foo"] == "```python\n\"\"\"Decorated function.\"\"\"\n```"

def test_parse_empty_module():
    p = Parser()
    script = ""
    p.parse("test_module", script)
    assert "test_module" in p.doc
    assert p.doc["test_module"] == "### Module `test_module`\n\n"

def test_parse_with_all():
    p = Parser()
    script = """
__all__ = ['public_func', 'PublicClass']
def public_func():
    pass
def _private_func():
    pass
class PublicClass:
    pass
class _PrivateClass:
    pass
"""
    p.parse("test_module", script)
    assert "test_module.public_func" in p.imp["test_module"]
    assert "test_module.PublicClass" in p.imp["test_module"]
    assert "test_module._private_func" not in p.imp["test_module"]
    assert "test_module._PrivateClass" not in p.imp["test_module"]


# LLM-generated content at query #35
#--------------------------

```python
def test_class_api_with_enum_base():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("enum.Enum").body[0].value]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]


# LLM-generated content at query #36
#--------------------------

```python
def test_visit_Subscript_with_non_Name_value():
    resolver = Resolver("root", {})
    node = Subscript(Constant(1), Constant(0), Load())
    assert resolver.visit_Subscript(node) == node

def test_visit_Subscript_with_typing_Union():
    resolver = Resolver("root", {"root.typing.Union": "typing.Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    expected = BinOp(Name("int", Load()), BitOr(), Name("str", Load()))
    assert resolver.visit_Subscript(node) == expected

def test_visit_Subscript_with_typing_Optional():
    resolver = Resolver("root", {"root.typing.Optional": "typing.Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    expected = BinOp(Name("int", Load()), BitOr(), Constant(None))
    assert resolver.visit_Subscript(node) == expected

def test_visit_Subscript_with_PEP585_type():
    resolver = Resolver("root", {"root.typing.List": "typing.List"})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    expected = Subscript(Name("list", Load()), Name("int", Load()), Load())
    assert resolver.visit_Subscript(node) == expected

def test_visit_Subscript_with_unknown_type():
    resolver = Resolver("root", {})
    node = Subscript(Name("Unknown", Load()), Name("int", Load()), Load())
    assert resolver.visit_Subscript(node) == node


# LLM-generated content at query #37
#--------------------------

```python
def test_func_api_with_vararg():
    parser = Parser()
    root = "test_module"
    name = "test_function"
    node = arguments(posonlyargs=[], args=[], defaults=[], vararg=arg('args', None), kwonlyargs=[], kw_defaults=[], kwarg=None)
    returns = None
    has_self = False
    cls_method = False

    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)

    assert parser.doc[name] is not None


# LLM-generated content at query #38
#--------------------------

```python
def test_attr_single_level():
    class TestClass:
        attr = "value"
    obj = TestClass()
    assert _attr(obj, "attr") == "value"

def test_attr_nested():
    class Inner:
        value = 42
    class Outer:
        inner = Inner()
    obj = Outer()
    assert _attr(obj, "inner.value") == 42

def test_attr_missing():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "nonexistent") is None

def test_attr_missing_nested():
    class Inner:
        pass
    class Outer:
        inner = Inner()
    obj = Outer()
    assert _attr(obj, "inner.nonexistent") is None

def test_attr_empty_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "") is None


# LLM-generated content at query #39
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("class Base: pass").body[0]]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("from enum import Enum\nclass Base(Enum): pass").body[0]]
    body = [parse("A = 1\nB = 2").body[0]]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("x: int = 1\ny: str = 'test'").body[0]]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("_x: int = 1\n__y: str = 'test'").body[0]]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("x: int = 1\ndel x").body[0]]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #40
#--------------------------

```python
def test_globals_type_comment_not_none():
    parser = Parser()
    node = Assign(targets=[Name(id='x')], value=Constant(value=1), type_comment='int')
    parser.globals('root', node)
    assert parser.alias['root.x'] == '1'
    assert parser.const['root.x'] == 'int'


# LLM-generated content at query #41
#--------------------------

```python
def test_link_false_no_anchor_tag():
    p = Parser.new(link=False, level=1, toc=False)
    p.parse('test', 'def foo(): pass')
    assert '<a id=' not in p.doc['test.foo']


# LLM-generated content at query #42
#--------------------------

```python
def test_globals_with_annassign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="test_var"),
        annotation=Constant(value="int"),
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_assign_no_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("test_module", node)
    assert parser.imp["test_module"] == {"test_module.func1", "test_module.func2"}

def test_globals_with_non_uppercase():
    parser = Parser()
    node = Assign(
        targets=[Name(id="non_upper")],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.non_upper"] == "42"
    assert "test_module.non_upper" not in parser.const
    assert "test_module.non_upper" not in parser.root

def test_globals_with_multiple_targets():
    parser = Parser()
    node = Assign(
        targets=[Name(id="var1"), Name(id="var2")],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert "test_module.var1" not in parser.alias
    assert "test_module.var2" not in parser.alias


# LLM-generated content at query #43
#--------------------------

```python
def test__attr_with_nonexistent_nested_attribute():
    class MockObject:
        pass

    obj = MockObject()
    assert _attr(obj, "nonexistent.nested.attribute") is None


# LLM-generated content at query #44
#--------------------------

```python
def test_parser_default_constructor():
    parser = Parser()
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_custom_constructor():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_new_method():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is False
    assert parser.b_level == 3
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_post_init_toc_true():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #45
#--------------------------

```python
def test_func_ann_no_annotation():
    parser = Parser()
    args = [arg('x'), arg('y')]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['ANY', 'ANY']

def test_func_ann_with_annotation():
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('y', Name('str', Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

def test_func_ann_with_self():
    parser = Parser()
    args = [arg('self', Name('Self', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_cls_method():
    parser = Parser()
    args = [arg('cls', Name('type[Self]', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_with_star_args():
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('*', None), arg('y', Name('str', Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']


# LLM-generated content at query #46
#--------------------------

```python
def test_attr_simple_attribute():
    class TestClass:
        def __init__(self):
            self.x = 42
    obj = TestClass()
    assert _attr(obj, 'x') == 42

def test_attr_nested_attribute():
    class Inner:
        def __init__(self):
            self.y = 100
    class Outer:
        def __init__(self):
            self.inner = Inner()
    obj = Outer()
    assert _attr(obj, 'inner.y') == 100

def test_attr_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, 'nonexistent') is None

def test_attr_nonexistent_nested_attribute():
    class Inner:
        pass
    class Outer:
        def __init__(self):
            self.inner = Inner()
    obj = Outer()
    assert _attr(obj, 'inner.nonexistent') is None

def test_attr_empty_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, '') is None


# LLM-generated content at query #47
#--------------------------

```python
def test_visit_Name_with_TypeVar():
    resolver = Resolver("root", {"root.TypeVar": "typing.TypeVar"})
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"


# LLM-generated content at query #48
#--------------------------

```python
def test_imports_with_none_asname():
    p = Parser()
    node = ImportFrom(module="sys", names=[alias(name="exit", asname=None)], level=0)
    p.imports("test", node)
    assert "test.exit" not in p.alias


# LLM-generated content at query #49
#--------------------------

```python
def test_isinstance_async_function_def():
    parser = Parser()
    node = AsyncFunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
    parser.api("root", node)
    assert isinstance(node, AsyncFunctionDef)


# LLM-generated content at query #50
#--------------------------

```python
def test_load_docstring_updates_docstring_when_doc_exists():
    parser = Parser()
    parser.doc = {"pkg.submod": "Module doc", "pkg.submod.func": "Function doc"}
    parser.docstring = {}
    mock_module = type('mock_module', (), {'submod': type('mock_submod', (), {'func': lambda: None})})
    mock_module.submod.func.__doc__ = "Function documentation"
    parser.load_docstring("pkg.submod", mock_module)
    assert parser.docstring["pkg.submod.func"] == doctest("Function documentation")


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_evaluates_to_false():
    obj = object()
    assert _attr(obj, 'nonexistent') is None


# LLM-generated content at query #52
#--------------------------

```python
def test_visit_Subscript_with_typing_Union():
    resolver = Resolver("test", {"typing.Union": "typing.Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"


# LLM-generated content at query #53
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name(id="another_attr", ctx=Store())],
            value=Constant(value=42),
            type_comment="float"
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "another_attr" in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyEnum"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        Assign(
            targets=[Name(id="FIRST", ctx=Store())],
            value=Constant(value=1)
        ),
        Assign(
            targets=[Name(id="SECOND", ctx=Store())],
            value=Constant(value=2)
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "FIRST" in parser.doc[name]
    assert "SECOND" in parser.doc[name]
    assert "Members" not in parser.doc[name]

def test_class_api_with_deleted_member():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        ),
        Delete(targets=[Name(id="public_attr", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]
    assert "public_attr" not in parser.doc[name]

def test_class_api_with_private_member():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="_private_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]
    assert "_private_attr" not in parser.doc[name]

def test_class_api_with_no_bases_or_members():
    parser = Parser()
    root = "test.module"
    name = "test.module.EmptyClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]


# LLM-generated content at query #54
#--------------------------

```python
def test_is_public_predicate_false():
    p = Parser()
    p.imp = {}
    p.root = {'test': 'test'}
    assert not p.is_public('test')


# LLM-generated content at query #55
#--------------------------

```python
def test_load_docstring():
    p = Parser()
    p.doc = {'pkg.module': 'doc', 'pkg.module.func': 'doc'}
    p.docstring = {}
    m = ModuleType('pkg.module')
    m.func = lambda: None
    m.func.__doc__ = "Function docstring"
    p.load_docstring('pkg.module', m)
    assert p.docstring == {'pkg.module': '```python\n```', 'pkg.module.func': '```python\nFunction docstring\n```'}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_func_api_with_no_args_and_no_return():
    parser = Parser()
    args = arguments([], None, [], [], None, [], None)
    parser.func_api('root', 'func', args, None, has_self=False, cls_method=False)
    expected = "| return |\n|:-----:|\n|  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_args_and_no_defaults():
    parser = Parser()
    args = arguments([arg('a', None), arg('b', None)], None, [], [], None, [], None)
    parser.func_api('root', 'func', args, None, has_self=False, cls_method=False)
    expected = "| a | b | return |\n|:---:|:---:|:-----:|\n|  |  |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_args_and_defaults():
    parser = Parser()
    args = arguments([arg('a', None), arg('b', None)], None, [], [], None, [], None)
    defaults = [None, Constant(1)]
    parser.func_api('root', 'func', args, None, has_self=False, cls_method=False)
    expected = "| a | b | return |\n|:---:|:---:|:-----:|\n|  | `1` |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_self_and_cls_method():
    parser = Parser()
    args = arguments([arg('self', None)], None, [], [], None, [], None)
    parser.func_api('root', 'func', args, None, has_self=True, cls_method=True)
    expected = "| self | return |\n|:----:|:-----:|\n| type[Self] |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_self_and_not_cls_method():
    parser = Parser()
    args = arguments([arg('self', None)], None, [], [], None, [], None)
    parser.func_api('root', 'func', args, None, has_self=True, cls_method=False)
    expected = "| self | return |\n|:----:|:-----:|\n| Self |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_varargs():
    parser = Parser()
    args = arguments([], None, [], [], arg('args', None), [], None)
    parser.func_api('root', 'func', args, None, has_self=False, cls_method=False)
    expected = "| *args | return |\n|:-----:|:-----:|\n|  |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_kwargs():
    parser = Parser()
    args = arguments([], None, [], [], None, [], arg('kwargs', None))
    parser.func_api('root', 'func', args, None, has_self=False, cls_method=False)
    expected = "| **kwargs | return |\n|:-------:|:-----:|\n|  |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_annotations():
    parser = Parser()
    args = arguments([arg('a', Constant('int'))], None, [], [], None, [], None)
    parser.func_api('root', 'func', args, None, has_self=False, cls_method=False)
    expected = "| a | return |\n|:---:|:-----:|\n| `int` |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_return_annotation():
    parser = Parser()
    args = arguments([], None, [], [], None, [], None)
    parser.func_api('root', 'func', args, Constant('str'), has_self=False, cls_method=False)
    expected = "| return |\n|:-----:|\n| `str` |\n\n"
    assert parser.doc['root.func'] == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_class_api_basic():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]

def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("BaseClass", mode='eval').body]
    body = []
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

def test_class_api_with_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        parse("x: int = 1", mode='exec').body[0],
        parse("y: str = 'hello'", mode='exec').body[0]
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("enum.Enum", mode='eval').body]
    body = [
        parse("A = 1", mode='exec').body[0],
        parse("B = 2", mode='exec').body[0]
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "Enums" in parser.doc[name]
    assert "A" in parser.doc[name]
    assert "B" in parser.doc[name]

def test_class_api_with_deleted_member():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        parse("x: int = 1", mode='exec').body[0],
        parse("del x", mode='exec').body[0]
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "x" not in parser.doc[name]

def test_class_api_private_members_ignored():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        parse("_private: int = 1", mode='exec').body[0],
        parse("public: int = 2", mode='exec').body[0]
    ]
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "_private" not in parser.doc[name]
    assert "public" in parser.doc[name]


# LLM-generated content at query #3
#--------------------------

```python
def test_parser_constructor_defaults():
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

def test_parser_constructor_with_parameters():
    p = Parser.new(link=False, level=2, toc=True)
    assert p.link is True  # toc=True forces link=True
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_visit_Name_self_type():
    resolver = Resolver(root="module", alias={}, self_ty="T")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver(root="module", alias={"module.Name": "typing.List"})
    node = Name("Name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "List"

def test_visit_Name_no_alias():
    resolver = Resolver(root="module", alias={})
    node = Name("Name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Name"

def test_visit_Name_typevar_alias():
    resolver = Resolver(root="module", alias={"module.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #5
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("BaseClass", mode='eval').body]
    body = [
        parse("x: int = 1", mode='eval').body,
        parse("y: str = 'hello'", mode='eval').body,
        parse("del z", mode='eval').body
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "z" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("enum.Enum", mode='eval').body]
    body = [
        parse("A = 1", mode='eval').body,
        parse("B = 2", mode='eval').body
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "A" in parser.doc[name]
    assert "B" in parser.doc[name]

def test_class_api_empty():
    parser = Parser()
    root = "test_module"
    name = "test_module.EmptyClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert parser.doc[name] == ""


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_line_19_false():
    p = Parser()
    p.parse('test_module', 'x = 1')
    assert not isinstance(p.root['test_module'].body[0], Assign)


# LLM-generated content at query #7
#--------------------------

```python
def test__attr_simple_attribute():
    class TestClass:
        x = 10
    obj = TestClass()
    assert _attr(obj, 'x') == 10

def test__attr_nested_attribute():
    class Inner:
        y = 20
    class Outer:
        inner = Inner()
    obj = Outer()
    assert _attr(obj, 'inner.y') == 20

def test__attr_nonexistent_attribute():
    class TestClass:
        x = 10
    obj = TestClass()
    assert _attr(obj, 'z') is None

def test__attr_nonexistent_nested_attribute():
    class Inner:
        y = 20
    class Outer:
        inner = Inner()
    obj = Outer()
    assert _attr(obj, 'inner.z') is None

def test__attr_chain_with_none():
    class TestClass:
        x = None
    obj = TestClass()
    assert _attr(obj, 'x.y') is None


# LLM-generated content at query #8
#--------------------------

```python
def test_const_type_with_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_with_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_with_tuple_of_ints():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

def test_const_type_with_tuple_of_mixed_types():
    node = Tuple(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "tuple[Any, Any]"

def test_const_type_with_list_of_floats():
    node = List(elts=[Constant(value=1.1), Constant(value=2.2)])
    assert const_type(node) == "list[float, float]"

def test_const_type_with_set_of_bools():
    node = Set(elts=[Constant(value=True), Constant(value=False)])
    assert const_type(node) == "set[bool, bool]"

def test_const_type_with_dict_of_str_to_int():
    node = Dict(keys=[Constant(value="a"), Constant(value="b")], values=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "dict[str, str, int, int]"

def test_const_type_with_dict_of_mixed_key_types():
    node = Dict(keys=[Constant(value=1), Constant(value="a")], values=[Constant(value=2), Constant(value="b")])
    assert const_type(node) == "dict[Any, Any, Any, Any]"

def test_const_type_with_call_to_bool():
    node = Call(func=Name(id="bool"), args=[Constant(value=1)])
    assert const_type(node) == "bool"

def test_const_type_with_call_to_int():
    node = Call(func=Name(id="int"), args=[Constant(value="42")])
    assert const_type(node) == "int"

def test_const_type_with_unknown_call():
    node = Call(func=Name(id="unknown"), args=[])
    assert const_type(node) == "Any"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[])
    assert const_type(node) == "tuple[]"

def test_const_type_with_none_in_tuple():
    node = Tuple(elts=[None, Constant(value=1)])
    assert const_type(node) == "Any"

def test_const_type_with_non_constant_in_tuple():
    node = Tuple(elts=[Constant(value=1), Name(id="x")])
    assert const_type(node) == "Any"


# LLM-generated content at query #9
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse_expression("BaseClass")]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse_expression("enum.Enum")]
    body = [parse("VALUE1 = 1"), parse("VALUE2 = 2")]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]

def test_class_api_with_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("member: int = 1")]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]

def test_class_api_with_deleted_member():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("member: int = 1"), parse("del member")]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #10
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Constant(value=int),
        value=Constant(value=1),
        simple=1
    )
    parser.globals("root", node)
    assert parser.alias["root.x"] == "1"
    assert parser.const["root.x"] == "int"
    assert parser.root["root.x"] == "root"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value="hello")
    )
    parser.globals("root", node)
    assert parser.alias["root.y"] == "'hello'"
    assert parser.const["root.y"] == "str"
    assert parser.root["root.y"] == "root"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="foo"), Constant(value="bar")])
    )
    parser.globals("root", node)
    assert parser.imp["root"] == {"root.foo", "root.bar"}

def test_globals_ignores_complex_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="z", ctx=Store()), Name(id="w", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals("root", node)
    assert "root.z" not in parser.alias
    assert "root.w" not in parser.alias

def test_globals_ignores_non_constant_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Name(id="some_var", ctx=Load())
    )
    parser.globals("root", node)
    assert parser.imp["root"] == set()


# LLM-generated content at query #11
#--------------------------

```python
def test_parser_constructor_defaults():
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

def test_parser_constructor_toc_forces_link():
    p = Parser(link=False, toc=True)
    assert p.link is True
    assert p.toc is True


# LLM-generated content at query #12
#--------------------------

```python
def test_visit_Name_with_TypeVar():
    resolver = Resolver(root="test", alias={"test.T": "typing.TypeVar('T')"}, self_ty="")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #13
#--------------------------

```python
def test_compile_empty():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.compile() == '\n'

def test_compile_single_module():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    assert p.compile() == '# Module `module`\n'

def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    assert p.compile() == '**Table of contents:**\n\n+ [`module`](#module)\n\n# Module `module`\n<a id="module"></a>\n'

def test_compile_with_docstring():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    p.docstring['module'] = 'This is a test module.'
    assert p.compile() == '# Module `module`\n\nThis is a test module.\n'

def test_compile_with_const():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    p.const['module.CONST'] = 'int'
    assert p.compile() == '# Module `module`\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n'

def test_compile_non_public():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    p.doc['module._private'] = '## `_private`()'
    p.level['module._private'] = 1
    p.root['module._private'] = 'module'
    assert p.compile() == '# Module `module`\n'

def test_compile_magic_method():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    p.doc['module.__init__'] = '## `__init__`()'
    p.level['module.__init__'] = 1
    p.root['module.__init__'] = 'module'
    assert p.compile() == '# Module `module`\n'

def test_compile_multiple_modules():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module1'] = '# Module `{}`'
    p.level['module1'] = 0
    p.root['module1'] = 'module1'
    p.imp['module1'] = set()
    p.doc['module2'] = '# Module `{}`'
    p.level['module2'] = 0
    p.root['module2'] = 'module2'
    p.imp['module2'] = set()
    assert p.compile() == '# Module `module1`\n\n# Module `module2`\n'

def test_compile_with_link():
    p = Parser.new(link=True, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    assert p.compile() == '# Module `module`\n<a id="module"></a>\n'

def test_compile_with_alias():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = set()
    p.alias['module.alias'] = 'module.original'
    p.doc['module.original'] = '## `original`()'
    p.level['module.original'] = 1
    p.root['module.original'] = 'module'
    assert p.compile() == '# Module `module`\n\n## `alias`()\n'

def test_compile_with_all():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc['module'] = '# Module `{}`'
    p.level['module'] = 0
    p.root['module'] = 'module'
    p.imp['module'] = {'module'}
    p.docstring['module'] = 'This is a test module.'
    p.const['module.CONST'] = 'int'
    p.doc['module.func'] = '## `func`()'
    p.level['module.func'] = 1
    p.root['module.func'] = 'module'
    p.docstring['module.func'] = 'This is a test function.'
    assert p.compile() == '''**Table of contents:**

+ [`module`](#module)
    + [`func`](#module-func)

# Module `module`
<a id="module"></a>

| Constants | Type |
|-----------|------|
| `CONST` | `int` |

This is a test module.

## `func`()
<a id="module-func"></a>

This is a test function.
'''


# LLM-generated content at query #14
#--------------------------

```python
def test_parse_basic_module():
    parser = Parser()
    script = """
    '''Module docstring.'''
    x = 1
    """
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module" in parser.level
    assert "test_module" in parser.root
    assert "test_module.x" in parser.alias
    assert "test_module" in parser.docstring

def test_parse_with_imports():
    parser = Parser()
    script = """
    import os
    from sys import path
    """
    parser.parse("test_module", script)
    assert "test_module.os" in parser.alias
    assert "test_module.path" in parser.alias

def test_parse_with_functions():
    parser = Parser()
    script = """
    def foo():
        '''Function docstring.'''
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.doc
    assert "test_module.foo" in parser.docstring

def test_parse_with_classes():
    parser = Parser()
    script = """
    class Bar:
        '''Class docstring.'''
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.Bar" in parser.doc
    assert "test_module.Bar" in parser.docstring

def test_parse_with_annotations():
    parser = Parser()
    script = """
    x: int = 1
    """
    parser.parse("test_module", script)
    assert "test_module.x" in parser.alias
    assert "test_module.x" in parser.const


# LLM-generated content at query #15
#--------------------------

```python
def test_is_public_family_magic_name():
    assert is_public_family('__init__') == True
    assert is_public_family('__main__') == True
    assert is_public_family('__name__') == True

def test_is_public_family_public_name():
    assert is_public_family('public') == True
    assert is_public_family('public.module') == True
    assert is_public_family('public.module.submodule') == True

def test_is_public_family_private_name():
    assert is_public_family('_private') == False
    assert is_public_family('public._private') == False
    assert is_public_family('_private.module') == False
    assert is_public_family('public.module._private') == False

def test_is_public_family_mixed_name():
    assert is_public_family('public.__init__') == True
    assert is_public_family('__init__.public') == True
    assert is_public_family('public.__init__.module') == True
    assert is_public_family('__init__.public.__main__') == True


# LLM-generated content at query #16
#--------------------------

```python
def test_annassign_not_instance_of_name():
    p = Parser()
    node = AnnAssign(target=Constant(value=1), annotation=Name(id='int'), value=None)
    assert not (isinstance(node, AnnAssign) and isinstance(node.target, Name))


# LLM-generated content at query #17
#--------------------------

```python
def test_delete_non_name_target():
    p = Parser()
    p.parse('test', 'class A:\n    x = 1\n    del x[0]')
    assert p.doc['test.A'] == '#' * (p.b_level + 3) + " class A\n\n*Full name:* `test.A`\n<a id=\"test-a\"></a>\n\n"


# LLM-generated content at query #18
#--------------------------

```python
def test_visit_Name_with_TypeVar_in_alias():
    resolver = Resolver("root", {"root.T": "typing.TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #19
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.doc = {'root.public_name': ''}
    assert parser.is_public('root.public_name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root._private_name': ''}
    assert parser.is_public('root._private_name') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root.__magic__': ''}
    assert parser.is_public('root.__magic__') == True

def test_is_public_with_nested_public_name():
    parser = Parser()
    parser.imp = {'root': {'nested'}}
    parser.doc = {'root.nested.public_name': ''}
    assert parser.is_public('root.nested.public_name') == True

def test_is_public_with_nested_private_name():
    parser = Parser()
    parser.imp = {'root': {'nested'}}
    parser.doc = {'root.nested._private_name': ''}
    assert parser.is_public('root.nested._private_name') == False

def test_is_public_with_all_listed_name():
    parser = Parser()
    parser.imp = {'root': {'listed_name'}}
    parser.doc = {'root.listed_name': ''}
    assert parser.is_public('root.listed_name') == True

def test_is_public_with_parent_listed_in_all():
    parser = Parser()
    parser.imp = {'root': {'parent'}}
    parser.doc = {'root.parent.child': ''}
    assert parser.is_public('root.parent.child') == True

def test_is_public_with_empty_all():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root.public_name': ''}
    assert parser.is_public('root.public_name') == True

def test_is_public_with_const_in_all():
    parser = Parser()
    parser.imp = {'root': {'CONST_NAME'}}
    parser.const = {'root.CONST_NAME': 'int'}
    assert parser.is_public('root.CONST_NAME') == True


# LLM-generated content at query #20
#--------------------------

```python
def test_parser_default_constructor():
    parser = Parser()
    assert parser.link == True
    assert parser.b_level == 1
    assert parser.toc == False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_custom_constructor():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link == False
    assert parser.b_level == 2
    assert parser.toc == True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_new_method():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link == False
    assert parser.b_level == 3
    assert parser.toc == True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_post_init_toc_true():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link == True
    assert parser.b_level == 1
    assert parser.toc == True

def test_parser_post_init_toc_false():
    parser = Parser.new(link=False, level=1, toc=False)
    assert parser.link == False
    assert parser.b_level == 1
    assert parser.toc == False


# LLM-generated content at query #21
#--------------------------

```python
def test_api_function():
    parser = Parser()
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.doc = {}
    parser.docstring = {}
    parser.imp = {'root': set()}
    parser.alias = {}
    parser.const = {}

    # Mock AST node for a function
    node = FunctionDef(
        name='test_func',
        args=arguments(
            posonlyargs=[],
            args=[arg('x', None), arg('y', None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=None,
        type_comment=None
    )

    parser.api('root', node)

    assert 'root.test_func' in parser.doc
    assert parser.doc['root.test_func'].startswith('### test_func()')
    assert 'Full name' in parser.doc['root.test_func']

def test_api_async_function():
    parser = Parser()
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.doc = {}
    parser.docstring = {}
    parser.imp = {'root': set()}
    parser.alias = {}
    parser.const = {}

    # Mock AST node for an async function
    node = AsyncFunctionDef(
        name='test_async_func',
        args=arguments(
            posonlyargs=[],
            args=[arg('x', None), arg('y', None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=None,
        type_comment=None
    )

    parser.api('root', node)

    assert 'root.test_async_func' in parser.doc
    assert parser.doc['root.test_async_func'].startswith('### async test_async_func()')
    assert 'Full name' in parser.doc['root.test_async_func']

def test_api_class():
    parser = Parser()
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.doc = {}
    parser.docstring = {}
    parser.imp = {'root': set()}
    parser.alias = {}
    parser.const = {}

    # Mock AST node for a class
    node = ClassDef(
        name='TestClass',
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )

    parser.api('root', node)

    assert 'root.TestClass' in parser.doc
    assert parser.doc['root.TestClass'].startswith('### class TestClass')
    assert 'Full name' in parser.doc['root.TestClass']


# LLM-generated content at query #22
#--------------------------

```python
def test__defaults_with_none_values():
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_empty_strings():
    args = ["", "", ""]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_ampersand():
    args = ["a&b", "c&d"]
    result = list(_defaults(args))
    assert result == ["<code>a&#38;b</code>", "<code>c&#38;d</code>"]

def test__defaults_with_pipe():
    args = ["a|b", "c|d"]
    result = list(_defaults(args))
    assert result == ["`a&#124;b`", "`c&#124;d`"]

def test__defaults_with_mixed_values():
    args = ["a|b", None, "c&d", ""]
    result = list(_defaults(args))
    assert result == ["`a&#124;b`", " ", "<code>c&#38;d</code>", " "]


# LLM-generated content at query #23
#--------------------------

```python
def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.parse('test', 'def foo(): pass')
    result = p.compile()
    assert result.startswith('**Table of contents:**')


# LLM-generated content at query #24
#--------------------------

```python
def test_compile_skips_magic_names_without_docstring():
    p = Parser()
    p.doc = {'__init__': '## __init__\n\n*Full name:* `{}`\n\n'}
    p.root = {'__init__': ''}
    p.level = {'__init__': 0}
    p.docstring = {}
    p.imp = {}
    p.const = {}
    p.toc = False
    assert p.compile() == ''


# LLM-generated content at query #25
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x"),
        annotation=Name(id="int"),
        value=Constant(value=1),
        simple=1
    )
    parser.globals("test", node)
    assert parser.alias["test.x"] == "1"
    assert parser.const["test.x"] == "int"
    assert parser.root["test.x"] == "test"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y")],
        value=Constant(value="hello")
    )
    parser.globals("test", node)
    assert parser.alias["test.y"] == "'hello'"
    assert parser.const["test.y"] == "str"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="foo"), Constant(value="bar")])
    )
    parser.globals("test", node)
    assert parser.imp["test"] == {"test.foo", "test.bar"}

def test_globals_with_non_constant():
    parser = Parser()
    node = Assign(
        targets=[Name(id="z")],
        value=Name(id="some_var")
    )
    parser.globals("test", node)
    assert parser.alias["test.z"] == "some_var"
    assert parser.const.get("test.z") is None


# LLM-generated content at query #26
#--------------------------

```python
def test_nested_attribute_none():
    class MockObject:
        pass

    obj = MockObject()
    result = _attr(obj, "nonexistent_attribute")
    assert result is None


# LLM-generated content at query #27
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("class Base: pass").body[0]]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "| Base |" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("from enum import Enum\nclass Enum: pass").body[0]]
    body = [parse("A = 1\nB = 2").body[0]]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "| A |" in parser.doc[name]
    assert "| B |" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("x: int = 1\ny: str = 'hello'").body[0]]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "| x | int |" in parser.doc[name]
    assert "| y | str |" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("x: int = 1\ndel x").body[0]]
    parser.class_api(root, name, bases, body)
    assert "x" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [parse("_private: int = 1").body[0]]
    parser.class_api(root, name, bases, body)
    assert "_private" not in parser.doc[name]


# LLM-generated content at query #28
#--------------------------

```python
def test_isinstance_functiondef():
    p = Parser()
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
    assert isinstance(node, FunctionDef)


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_line_38_false():
    p = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="test_var")],
        value=List(elts=[Constant(value=123)])
    )
    p.globals(root, node)
    assert len(p.imp[root]) == 0


# LLM-generated content at query #30
#--------------------------

```python
def test_isinstance_call_and_name_or_attribute():
    node = Call(func=Name(id='test'), args=[])
    assert isinstance(node, Call) and isinstance(node.func, (Name, Attribute))


# LLM-generated content at query #31
#--------------------------

```python
def test_load_docstring():
    p = Parser()
    m = ModuleType('test_module')
    m.__doc__ = "Module docstring"
    m.test_func.__doc__ = "Function docstring"
    p.doc = {'test_module': '', 'test_module.test_func': ''}
    p.load_docstring('test_module', m)
    assert p.docstring['test_module'] == doctest("Module docstring")
    assert p.docstring['test_module.test_func'] == doctest("Function docstring")


# LLM-generated content at query #32
#--------------------------

```python
def test_none_attr_returns_none():
    class Dummy:
        pass
    assert _attr(Dummy(), 'nonexistent') is None


# LLM-generated content at query #33
#--------------------------

```python
def test_parse_basic_module():
    parser = Parser()
    script = """
\"\"\"Module docstring.\"\"\"
x = 1
def foo():
    \"\"\"Function docstring.\"\"\"
    pass
"""
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    assert "test_module.x" in parser.alias
    assert parser.docstring["test_module"] == "```python\n\"\"\"Module docstring.\"\"\"\n```"
    assert parser.docstring["test_module.foo"] == "```python\n\"\"\"Function docstring.\"\"\"\n```"

def test_parse_with_imports():
    parser = Parser()
    script = """
from os import path
import sys
x = path.join('a', 'b')
"""
    parser.parse("test_module", script)
    assert "os.path" in parser.alias["test_module.path"]
    assert "sys" in parser.alias["test_module.sys"]

def test_parse_class_and_method():
    parser = Parser()
    script = """
class MyClass:
    \"\"\"Class docstring.\"\"\"
    def method(self):
        \"\"\"Method docstring.\"\"\"
        pass
"""
    parser.parse("test_module", script)
    assert "test_module.MyClass" in parser.doc
    assert "test_module.MyClass.method" in parser.doc
    assert parser.docstring["test_module.MyClass"] == "```python\n\"\"\"Class docstring.\"\"\"\n```"
    assert parser.docstring["test_module.MyClass.method"] == "```python\n\"\"\"Method docstring.\"\"\"\n```"

def test_parse_with_decorators():
    parser = Parser()
    script = """
@decorator
def foo():
    pass
"""
    parser.parse("test_module", script)
    assert "@decorator" in parser.doc["test_module.foo"]

def test_parse_with_annotations():
    parser = Parser()
    script = """
def foo(x: int, y: str) -> bool:
    pass
"""
    parser.parse("test_module", script)
    assert "int" in parser.doc["test_module.foo"]
    assert "str" in parser.doc["test_module.foo"]
    assert "bool" in parser.doc["test_module.foo"]

def test_parse_with_all():
    parser = Parser()
    script = """
__all__ = ['foo', 'bar']
def foo():
    pass
def bar():
    pass
def baz():
    pass
"""
    parser.parse("test_module", script)
    assert "foo" in parser.imp["test_module"]
    assert "bar" in parser.imp["test_module"]
    assert "baz" not in parser.imp["test_module"]


# LLM-generated content at query #34
#--------------------------

```python
def test_link_false():
    parser = Parser(link=False)
    assert parser.link is False


# LLM-generated content at query #35
#--------------------------

```python
def test__e_type_empty_elements():
    assert _e_type() == ""

def test__e_type_with_none_element():
    assert _e_type([None]) == ""

def test__e_type_with_non_constant_element():
    assert _e_type([expr()]) == ""

def test__e_type_with_single_constant_element():
    assert _e_type([Constant(1)]) == "[int]"

def test__e_type_with_multiple_constant_elements_same_type():
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_with_multiple_constant_elements_different_types():
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_with_multiple_sequences():
    assert _e_type([Constant(1)], [Constant("a")]) == "[int, str]"


# LLM-generated content at query #36
#--------------------------

```python
def test_node_type_comment_is_not_none():
    parser = Parser()
    node = Assign(
        targets=[Name(id='x')],
        value=Constant(value=42),
        type_comment='int'
    )
    parser.class_api('root', 'root.Class', [], [node])
    assert 'mem' not in parser.__dict__ or 'x' not in parser.__dict__.get('mem', {})


# LLM-generated content at query #37
#--------------------------

```python
def test_annassign_with_non_name_target():
    p = Parser()
    node = AnnAssign(target=Attribute(value=Name(id='x'), attr='y'), annotation=Name(id='int'), value=None)
    assert not (isinstance(node, AnnAssign) and isinstance(node.target, Name))


# LLM-generated content at query #38
#--------------------------

```python
def test_visit_Constant_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_invalid_syntax():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_valid_name():
    resolver = Resolver("root", {"root.name": "alias"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias"


# LLM-generated content at query #39
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is True  # toc=True overrides link
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #40
#--------------------------

```python
def test_imports_with_Import_node():
    parser = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname="system")])
    parser.imports(root, node)
    assert parser.alias["test.module.os"] == "os"
    assert parser.alias["test.module.system"] == "sys"

def test_imports_with_ImportFrom_node_no_level():
    parser = Parser()
    root = "test.module"
    node = ImportFrom(module="os.path", names=[alias(name="join", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test.module.join"] == "os.path.join"

def test_imports_with_ImportFrom_node_with_level():
    parser = Parser()
    root = "test.module.submodule"
    node = ImportFrom(module="sibling", names=[alias(name="func", asname="f")], level=1)
    parser.imports(root, node)
    assert parser.alias["test.module.submodule.f"] == "test.module.sibling.func"


# LLM-generated content at query #41
#--------------------------

```python
def test_imports_with_level():
    parser = Parser()
    node = ImportFrom(module="os", names=[alias(name="path", asname="p")], level=1)
    parser.imports("pkg.subpkg", node)
    assert parser.alias["pkg.subpkg.p"] == "pkg.path"


# LLM-generated content at query #42
#--------------------------

```python
def test_compile_empty():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.compile() == "\n"

def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc["test"] = "# Test"
    p.level["test"] = 0
    p.root["test"] = "test"
    assert p.compile() == "**Table of contents:**\n\n+ [`test`](#test)\n\n# Test\n"

def test_compile_with_docstring():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["test"] = "# Test"
    p.docstring["test"] = "This is a test."
    p.level["test"] = 0
    p.root["test"] = "test"
    assert p.compile() == "# Test\n\nThis is a test.\n"

def test_compile_with_magic_name():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["__init__"] = "# __init__"
    p.level["__init__"] = 0
    p.root["__init__"] = "__init__"
    assert p.compile() == "\n"

def test_compile_with_const():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["test"] = "# Test"
    p.const["test.CONST"] = "int"
    p.root["test.CONST"] = "test"
    p.level["test"] = 0
    p.root["test"] = "test"
    p.imp["test"] = {"test.CONST"}
    assert p.compile() == "# Test\n\n| Constants | Type |\n| --- | --- |\n| `CONST` | `int` |\n"

def test_compile_with_non_public():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["_private"] = "# _private"
    p.level["_private"] = 0
    p.root["_private"] = "_private"
    assert p.compile() == "\n"

def test_compile_with_alias():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["test"] = "# Test"
    p.alias["test.alias"] = "test"
    p.level["test"] = 0
    p.root["test"] = "test"
    p.level["test.alias"] = 0
    p.root["test.alias"] = "test"
    assert p.compile() == "# Test\n\n# Test\n"


# LLM-generated content at query #43
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="test_var"),
        annotation=Name(id="int"),
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("test_module", node)
    assert parser.imp["test_module"] == {"test_module.func1", "test_module.func2"}

def test_globals_with_invalid_node():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var"), Name(id="another_var")],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert "test_module.test_var" not in parser.alias
    assert "test_module.another_var" not in parser.alias


# LLM-generated content at query #44
#--------------------------

```python
def test_empty_body():
    assert list(walk_body([])) == []

def test_single_statement():
    stmt = Assign(targets=[Name(id='x')], value=Constant(value=1))
    assert list(walk_body([stmt])) == [stmt]

def test_if_statement_with_body():
    if_node = If(test=Constant(value=True), body=[Assign(targets=[Name(id='x')], value=Constant(value=1))], orelse=[])
    assert list(walk_body([if_node])) == [Assign(targets=[Name(id='x')], value=Constant(value=1))]

def test_if_statement_with_orelse():
    if_node = If(
        test=Constant(value=True),
        body=[],
        orelse=[Assign(targets=[Name(id='x')], value=Constant(value=1))]
    )
    assert list(walk_body([if_node])) == [Assign(targets=[Name(id='x')], value=Constant(value=1))]

def test_try_statement_with_body():
    try_node = Try(
        body=[Assign(targets=[Name(id='x')], value=Constant(value=1))],
        handlers=[],
        orelse=[],
        finalbody=[]
    )
    assert list(walk_body([try_node])) == [Assign(targets=[Name(id='x')], value=Constant(value=1))]

def test_try_statement_with_handlers():
    try_node = Try(
        body=[],
        handlers=[ExceptHandler(body=[Assign(targets=[Name(id='x')], value=Constant(value=1))])],
        orelse=[],
        finalbody=[]
    )
    assert list(walk_body([try_node])) == [Assign(targets=[Name(id='x')], value=Constant(value=1))]

def test_try_statement_with_orelse():
    try_node = Try(
        body=[],
        handlers=[],
        orelse=[Assign(targets=[Name(id='x')], value=Constant(value=1))],
        finalbody=[]
    )
    assert list(walk_body([try_node])) == [Assign(targets=[Name(id='x')], value=Constant(value=1))]

def test_try_statement_with_finalbody():
    try_node = Try(
        body=[],
        handlers=[],
        orelse=[],
        finalbody=[Assign(targets=[Name(id='x')], value=Constant(value=1))]
    )
    assert list(walk_body([try_node])) == [Assign(targets=[Name(id='x')], value=Constant(value=1))]

def test_nested_if_statements():
    inner_if = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y')], value=Constant(value=2))],
        orelse=[]
    )
    outer_if = If(
        test=Constant(value=True),
        body=[inner_if],
        orelse=[]
    )
    assert list(walk_body([outer_if])) == [inner_if, Assign(targets=[Name(id='y')], value=Constant(value=2))]

def test_mixed_statements():
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='x')], value=Constant(value=1))],
        orelse=[]
    )
    try_node = Try(
        body=[Assign(targets=[Name(id='y')], value=Constant(value=2))],
        handlers=[],
        orelse=[],
        finalbody=[]
    )
    assert list(walk_body([if_node, try_node])) == [
        Assign(targets=[Name(id='x')], value=Constant(value=1)),
        try_node,
        Assign(targets=[Name(id='y')], value=Constant(value=2))
    ]


# LLM-generated content at query #45
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="test_var", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.TEST_VAR"] == "int"
    assert parser.root["test_module.TEST_VAR"] == "test_module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.TEST_VAR"] == "int"
    assert parser.root["test_module.TEST_VAR"] == "test_module"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[
            Constant(value="public_func"),
            Constant(value="PublicClass")
        ])
    )
    parser.globals("test_module", node)
    assert parser.imp["test_module"] == {"test_module.public_func", "test_module.PublicClass"}

def test_globals_ignores_non_simple_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals("test_module", node)
    assert "test_module.x" not in parser.alias
    assert "test_module.y" not in parser.alias


# LLM-generated content at query #46
#--------------------------

```python
def test_is_public_predicate_false():
    p = Parser()
    p.imp = {'root': {'child'}}
    p.root = {'root.child': 'root'}
    p.doc = {}
    p.const = {}
    assert not p.is_public('root.child')


# LLM-generated content at query #47
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Tuple(elts=[Constant(value='foo')]))
    parser.globals('root', node)
    assert '__all__' not in parser.imp['root']


# LLM-generated content at query #48
#--------------------------

```python
def test_handlers_is_sequence_of_except_handlers():
    node = Try(body=[], handlers=[ExceptHandler()], orelse=[], finalbody=[])
    assert isinstance(node.handlers, Sequence)
    assert all(isinstance(h, ExceptHandler) for h in node.handlers)


# LLM-generated content at query #49
#--------------------------

```python
def test_imports_predicate_false():
    p = Parser()
    node = ImportFrom(module=None, names=[], level=0)
    p.imports("root", node)
    assert len(p.alias) == 0


# LLM-generated content at query #50
#--------------------------

```python
def test_globals_ann_assign_with_type_annotation():
    parser = Parser()
    node = AnnAssign(target=Name(id="x"), annotation=Name(id="int"), value=Constant(value=1))
    parser.globals("module", node)
    assert parser.alias["module.x"] == "1"
    assert parser.const["module.x"] == "int"
    assert parser.root["module.x"] == "module"

def test_globals_assign_without_type_comment():
    parser = Parser()
    node = Assign(targets=[Name(id="y")], value=Constant(value="hello"))
    parser.globals("module", node)
    assert parser.alias["module.y"] == "'hello'"
    assert parser.const["module.y"] == "str"
    assert parser.root["module.y"] == "module"

def test_globals_assign_with_type_comment():
    parser = Parser()
    node = Assign(targets=[Name(id="z")], value=Constant(value=3.14), type_comment="float")
    parser.globals("module", node)
    assert parser.alias["module.z"] == "3.14"
    assert parser.const["module.z"] == "float"
    assert parser.root["module.z"] == "module"

def test_globals_all_filter():
    parser = Parser()
    node = Assign(targets=[Name(id="__all__")], value=List(elts=[Constant(value="func1"), Constant(value="func2")]))
    parser.globals("module", node)
    assert parser.imp["module"] == {"module.func1", "module.func2"}

def test_globals_ignores_complex_assignments():
    parser = Parser()
    node = Assign(targets=[Name(id="a"), Name(id="b")], value=Constant(value=42))
    parser.globals("module", node)
    assert "module.a" not in parser.alias
    assert "module.b" not in parser.alias

def test_globals_ignores_non_uppercase_constants():
    parser = Parser()
    node = Assign(targets=[Name(id="not_constant")], value=Constant(value=100))
    parser.globals("module", node)
    assert "module.not_constant" not in parser.const


# LLM-generated content at query #51
#--------------------------

```python
def test_is_public_family_returns_false_for_private_attribute():
    p = Parser()
    assert not is_public_family('_private_attr')


# LLM-generated content at query #52
#--------------------------

```python
def test_const_type_with_constant():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_with_tuple():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

def test_const_type_with_list():
    node = List(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "list[int, int]"

def test_const_type_with_set():
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "set[int, int]"

def test_const_type_with_dict():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value='a'), Constant(value='b')])
    assert const_type(node) == "dict[int, str]"

def test_const_type_with_call_bool():
    node = Call(func=Name(id='bool', ctx=Load()))
    assert const_type(node) == "bool"

def test_const_type_with_call_int():
    node = Call(func=Name(id='int', ctx=Load()))
    assert const_type(node) == "int"

def test_const_type_with_call_float():
    node = Call(func=Name(id='float', ctx=Load()))
    assert const_type(node) == "float"

def test_const_type_with_call_complex():
    node = Call(func=Name(id='complex', ctx=Load()))
    assert const_type(node) == "complex"

def test_const_type_with_call_str():
    node = Call(func=Name(id='str', ctx=Load()))
    assert const_type(node) == "str"

def test_const_type_with_unknown_call():
    node = Call(func=Name(id='unknown', ctx=Load()))
    assert const_type(node) == "Any"

def test_const_type_with_non_constant_elements():
    node = Tuple(elts=[Constant(value=1), Name(id='x', ctx=Load())])
    assert const_type(node) == "Any"

def test_const_type_with_mixed_types():
    node = List(elts=[Constant(value=1), Constant(value='a')])
    assert const_type(node) == "list[Any, Any]"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[])
    assert const_type(node) == "tuple[]"

def test_const_type_with_empty_list():
    node = List(elts=[])
    assert const_type(node) == "list[]"

def test_const_type_with_empty_set():
    node = Set(elts=[])
    assert const_type(node) == "set[]"

def test_const_type_with_empty_dict():
    node = Dict(keys=[], values=[])
    assert const_type(node) == "dict[]"


# LLM-generated content at query #53
#--------------------------

```python
def test_visit_Constant_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

def test_visit_Constant_invalid_syntax():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax")
    assert resolver.visit_Constant(node) == node

def test_visit_Constant_valid_name():
    resolver = Resolver("root", {"root.Name": "alias.Name"})
    node = Constant("Name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Name"


# LLM-generated content at query #54
#--------------------------

```python
def test_enums_table_generation():
    parser = Parser()
    parser.doc = {}
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.level = {}
    parser.imp = {}
    parser.b_level = 1
    parser.link = True
    parser.toc = False

    class MockExpr:
        pass

    class MockStmt:
        pass

    class MockName:
        def __init__(self, id):
            self.id = id

    class MockAnnAssign:
        def __init__(self, target, annotation):
            self.target = target
            self.annotation = annotation

    class MockAssign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    class MockDelete:
        def __init__(self, targets):
            self.targets = targets

    def mock_resolve(root, node, self_ty=""):
        return "enum.Enum"

    parser.resolve = mock_resolve

    body = [
        MockAnnAssign(MockName("RED"), None),
        MockAnnAssign(MockName("GREEN"), None),
        MockAnnAssign(MockName("BLUE"), None)
    ]

    parser.class_api("root", "Color", [MockExpr()], body)

    assert "Enums" in parser.doc["root.Color"]
    assert "RED" in parser.doc["root.Color"]
    assert "GREEN" in parser.doc["root.Color"]
    assert "BLUE" in parser.doc["root.Color"]


# LLM-generated content at query #55
#--------------------------

```python
def test_func_ann_with_has_self_and_cls_method():
    parser = Parser.new(link=False, level=1, toc=False)
    args = [arg('self', None), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_with_has_self_no_cls_method():
    parser = Parser.new(link=False, level=1, toc=False)
    args = [arg('self', None), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_star_arg():
    parser = Parser.new(link=False, level=1, toc=False)
    args = [arg('x', None), arg('*', None), arg('y', Name('str', Load()))]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['str', '', 'str']

def test_func_ann_with_no_annotation():
    parser = Parser.new(link=False, level=1, toc=False)
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']

def test_func_ann_with_self_type_annotation():
    parser = Parser.new(link=False, level=1, toc=False)
    args = [arg('self', Name('MyClass', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_cls_method_and_self_type():
    parser = Parser.new(link=False, level=1, toc=False)
    args = [arg('cls', Name('type[MyClass]', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']


# LLM-generated content at query #56
#--------------------------

```python
def test_func_api_basic():
    parser = Parser()
    root = "module"
    name = "module.function"
    node = arguments(posonlyargs=[], args=[arg("x", None)], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.doc[name] = ""
    parser.func_api(root, name, node, returns, has_self=False, cls_method=False)
    assert parser.doc[name] == "| x | return |\n|:---:|:---:|\n| `str` | `str` |\n\n"

def test_func_api_with_defaults():
    parser = Parser()
    root = "module"
    name = "module.function"
    node = arguments(posonlyargs=[], args=[arg("x", None), arg("y", None)], kwonlyargs=[], defaults=[Constant(value=1)], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.doc[name] = ""
    parser.func_api(root, name, node, returns, has_self=False, cls_method=False)
    assert parser.doc[name] == "| x | y | return |\n|:---:|:---:|:---:|\n| `str` | `str` | `str` |\n|  | `1` |  |\n\n"

def test_func_api_with_self():
    parser = Parser()
    root = "module"
    name = "module.Class.method"
    node = arguments(posonlyargs=[], args=[arg("self", None), arg("x", None)], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.doc[name] = ""
    parser.func_api(root, name, node, returns, has_self=True, cls_method=False)
    assert parser.doc[name] == "| self | x | return |\n|:---:|:---:|:---:|\n| `Self` | `str` | `str` |\n\n"

def test_func_api_with_classmethod():
    parser = Parser()
    root = "module"
    name = "module.Class.method"
    node = arguments(posonlyargs=[], args=[arg("cls", None), arg("x", None)], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.doc[name] = ""
    parser.func_api(root, name, node, returns, has_self=True, cls_method=True)
    assert parser.doc[name] == "| cls | x | return |\n|:---:|:---:|:---:|\n| `type[Self]` | `str` | `str` |\n\n"

def test_func_api_with_varargs():
    parser = Parser()
    root = "module"
    name = "module.function"
    node = arguments(posonlyargs=[], args=[arg("x", None)], kwonlyargs=[], defaults=[], kw_defaults=[], kwarg=arg("kwargs", None), vararg=arg("args", None))
    returns = None
    parser.doc[name] = ""
    parser.func_api(root, name, node, returns, has_self=False, cls_method=False)
    assert parser.doc[name] == "| x | *args | **kwargs | return |\n|:---:|:---:|:---:|:---:|\n| `str` | `str` | `str` | `str` |\n\n"


# LLM-generated content at query #57
#--------------------------

```python
def test_func_ann_yields_type_self_when_cls_method_is_true():
    parser = Parser()
    args = [arg(arg='self', annotation=Name(id='SomeClass', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'


# LLM-generated content at query #58
#--------------------------

```python
def test_const_type_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_tuple_homogeneous():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

def test_const_type_tuple_heterogeneous():
    node = Tuple(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "tuple[Any, Any]"

def test_const_type_list_homogeneous():
    node = List(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "list[int, int]"

def test_const_type_list_heterogeneous():
    node = List(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "list[Any, Any]"

def test_const_type_set_homogeneous():
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "set[int, int]"

def test_const_type_set_heterogeneous():
    node = Set(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "set[Any, Any]"

def test_const_type_dict_homogeneous():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "dict[int, str]"

def test_const_type_dict_heterogeneous_keys():
    node = Dict(keys=[Constant(value=1), Constant(value="a")], values=[Constant(value="b"), Constant(value="c")])
    assert const_type(node) == "dict[Any, str]"

def test_const_type_dict_heterogeneous_values():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value=1)])
    assert const_type(node) == "dict[int, Any]"

def test_const_type_call_bool():
    node = Call(func=Name(id="bool"))
    assert const_type(node) == "bool"

def test_const_type_call_int():
    node = Call(func=Name(id="int"))
    assert const_type(node) == "int"

def test_const_type_call_unknown():
    node = Call(func=Name(id="unknown"))
    assert const_type(node) == "Any"

def test_const_type_non_constant():
    node = Name(id="x")
    assert const_type(node) == "Any"


# LLM-generated content at query #59
#--------------------------

```python
def test_is_public_family_false():
    p = Parser()
    assert not is_public_family('_private')


# LLM-generated content at query #60
#--------------------------

```python
def test_parser_constructor_default_values():
    parser = Parser()
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_constructor_custom_values():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_constructor_toc_overrides_link():
    parser = Parser(link=False, toc=True)
    assert parser.link is True
    assert parser.toc is True


# LLM-generated content at query #61
#--------------------------

```python
def test_func_api_with_kwarg():
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    node = arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=arg('kwargs', None)
    )
    returns = None
    parser.func_api(root, name, node, returns, has_self=False, cls_method=False)
    assert parser.doc[name].endswith("\n\n| **kwargs | Any |\n")


# LLM-generated content at query #62
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()),
                  annotation=Name(id="int", ctx=Load()),
                  value=None),
        Assign(targets=[Name(id="another_attr", ctx=Store())],
               value=Constant(value=42)),
        Delete(targets=[Name(id="deleted_attr", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "another_attr" in parser.doc[name]
    assert "deleted_attr" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="VALUE1", ctx=Store()),
                  annotation=None,
                  value=None),
        AnnAssign(target=Name(id="VALUE2", ctx=Store()),
                  annotation=None,
                  value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]
    assert "Members" not in parser.doc[name]

def test_class_api_empty():
    parser = Parser()
    root = "test_module"
    name = "test_module.EmptyClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]
    assert "Enums" not in parser.doc[name]


# LLM-generated content at query #63
#--------------------------

```python
def test_parser_default_constructor():
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

def test_parser_custom_constructor():
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
    p = Parser.new(link=False, level=2, toc=True)
    assert p.link == True
    assert p.b_level == 2
    assert p.toc == True


# LLM-generated content at query #64
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {'pkg': {'public_name'}}
    parser.doc = {'pkg.public_name': ''}
    assert parser.is_public('pkg.public_name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.doc = {'pkg._private_name': ''}
    assert parser.is_public('pkg._private_name') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.doc = {'pkg.__magic__': ''}
    assert parser.is_public('pkg.__magic__') == True

def test_is_public_with_nested_public_name():
    parser = Parser()
    parser.imp = {'pkg': {'subpkg'}}
    parser.doc = {'pkg.subpkg.public_name': ''}
    assert parser.is_public('pkg.subpkg.public_name') == True

def test_is_public_with_nested_private_name():
    parser = Parser()
    parser.imp = {'pkg': {'subpkg'}}
    parser.doc = {'pkg.subpkg._private_name': ''}
    assert parser.is_public('pkg.subpkg._private_name') == False

def test_is_public_with_const_in_all():
    parser = Parser()
    parser.imp = {'pkg': {'CONST'}}
    parser.const = {'pkg.CONST': 'int'}
    assert parser.is_public('pkg.CONST') == True

def test_is_public_without_all_but_public_family():
    parser = Parser()
    parser.imp = {'pkg': set()}
    parser.doc = {'pkg.public_name': ''}
    assert parser.is_public('pkg.public_name') == True

def test_is_public_with_parent_in_all():
    parser = Parser()
    parser.imp = {'pkg': {'subpkg'}}
    parser.doc = {'pkg.subpkg': ''}
    assert parser.is_public('pkg.subpkg') == True


# LLM-generated content at query #65
#--------------------------

```python
def test_func_ann_cls_method_false():
    p = Parser()
    args = [arg('self', None)]
    result = list(p.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self']


# LLM-generated content at query #66
#--------------------------

```python
def test_parser_default_constructor():
    parser = Parser()
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_custom_constructor():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_new_method():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is False
    assert parser.b_level == 3
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_post_init_with_toc():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #67
#--------------------------

```python
def test_imports_with_level():
    p = Parser()
    node = ImportFrom(module="sys", names=[alias(name="path", asname=None)], level=1)
    p.imports("pkg.subpkg", node)
    assert p.alias["pkg.subpkg.path"] == "sys.path"


# LLM-generated content at query #68
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum_members():
    parser = Parser()
    root = "module"
    name = "module.EnumClass"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="MEMBER1", ctx=Store()), annotation=Constant(value=1), value=None),
        AnnAssign(target=Name(id="MEMBER2", ctx=Store()), annotation=Constant(value=2), value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "MEMBER1" in parser.doc[name]
    assert "MEMBER2" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Assign(targets=[Name(id="another_attr", ctx=Store())], value=Constant(value=42))
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "another_attr" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Delete(targets=[Name(id="public_attr", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "public_attr" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [
        AnnAssign(target=Name(id="_private_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #69
#--------------------------

```python
def test__attr_with_nonexistent_attribute():
    class Dummy:
        pass
    assert _attr(Dummy(), "nonexistent") is None


# LLM-generated content at query #70
#--------------------------

```python
def test_func_api_with_kwonlyargs():
    parser = Parser()
    root = "test_module"
    name = "test_function"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg("kw1", None), arg("kw2", None)],
        kw_defaults=[],
        kwarg=None
    )
    returns = None
    has_self = False
    cls_method = False

    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)

    assert parser.doc[name].endswith("()\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n".format(name, name.lower().replace('.', '-')))


