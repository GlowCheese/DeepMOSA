####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_imports_with_import_statement():
    parser = Parser()
    parser.parse('root', 'import module')
    assert parser.alias == {'root.module': 'module'}

def test_imports_with_import_as_statement():
    parser = Parser()
    parser.parse('root', 'import module as mod')
    assert parser.alias == {'root.mod': 'module'}

def test_imports_with_from_import_statement():
    parser = Parser()
    parser.parse('root', 'from package import module')
    assert parser.alias == {'root.module': 'package.module'}

def test_imports_with_from_import_as_statement():
    parser = Parser()
    parser.parse('root', 'from package import module as mod')
    assert parser.alias == {'root.mod': 'package.module'}

def test_imports_with_from_import_with_level():
    parser = Parser()
    parser.parse('root.sub', 'from ..package import module')
    assert parser.alias == {'root.sub.module': 'package.module'}


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_with_valid_script():
    parser = Parser()
    script = '''
import os
from sys import path

CONSTANT = 42

def func():
    pass

class Cls:
    pass
    '''
    parser.parse('module', script)
    assert 'module' in parser.doc
    assert 'module.func' in parser.doc
    assert 'module.Cls' in parser.doc
    assert 'module.CONSTANT' in parser.const

def test_parse_with_empty_script():
    parser = Parser()
    script = ''
    parser.parse('module', script)
    assert 'module' in parser.doc
    assert len(parser.doc) == 1
    assert len(parser.const) == 0

def test_parse_with_imports():
    parser = Parser()
    script = '''
import os
from sys import path
    '''
    parser.parse('module', script)
    assert 'module.os' in parser.alias
    assert 'module.path' in parser.alias

def test_parse_with_constants():
    parser = Parser()
    script = '''
CONSTANT = 42
ANOTHER_CONSTANT = "value"
    '''
    parser.parse('module', script)
    assert 'module.CONSTANT' in parser.const
    assert 'module.ANOTHER_CONSTANT' in parser.const

def test_parse_with_functions():
    parser = Parser()
    script = '''
def func():
    pass

async def async_func():
    pass
    '''
    parser.parse('module', script)
    assert 'module.func' in parser.doc
    assert 'module.async_func' in parser.doc

def test_parse_with_classes():
    parser = Parser()
    script = '''
class Cls:
    pass

class AnotherCls:
    pass
    '''
    parser.parse('module', script)
    assert 'module.Cls' in parser.doc
    assert 'module.AnotherCls' in parser.doc

def test_parse_with_docstrings():
    parser = Parser()
    script = '''
def func():
    """This is a docstring."""
    pass

class Cls:
    """This is a class docstring."""
    pass
    '''
    parser.parse('module', script)
    assert 'module.func' in parser.docstring
    assert 'module.Cls' in parser.docstring


# LLM-generated content at query #3
#--------------------------

```
def test_walk_body_simple_statement():
    body = [stmt()]
    result = list(walk_body(body))
    assert result == body

def test_walk_body_if_statement():
    if_node = If(test=expr(), body=[stmt()], orelse=[stmt()])
    body = [if_node]
    result = list(walk_body(body))
    assert result == if_node.body + if_node.orelse

def test_walk_body_try_statement():
    try_node = Try(body=[stmt()], handlers=[ExceptHandler(body=[stmt()])], orelse=[stmt()], finalbody=[stmt()])
    body = [try_node]
    result = list(walk_body(body))
    assert result == try_node.body + try_node.handlers[0].body + try_node.orelse + try_node.finalbody

def test_walk_body_nested_statements():
    if_node = If(test=expr(), body=[stmt()], orelse=[stmt()])
    try_node = Try(body=[if_node], handlers=[ExceptHandler(body=[stmt()])], orelse=[stmt()], finalbody=[stmt()])
    body = [try_node]
    result = list(walk_body(body))
    expected_result = if_node.body + if_node.orelse + try_node.handlers[0].body + try_node.orelse + try_node.finalbody
    assert result == expected_result


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```
def test__attr_with_single_attribute():
    class TestClass:
        x = 1
    obj = TestClass()
    assert _attr(obj, 'x') == 1

def test__attr_with_nested_attribute():
    class Inner:
        y = 2
    class Outer:
        inner = Inner()
    obj = Outer()
    assert _attr(obj, 'inner.y') == 2

def test__attr_with_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, 'nonexistent') is None

def test__attr_with_nonexistent_nested_attribute():
    class Inner:
        pass
    class Outer:
        inner = Inner()
    obj = Outer()
    assert _attr(obj, 'inner.nonexistent') is None

def test__attr_with_empty_attr_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, '') is None

def test__attr_with_none_object():
    assert _attr(None, 'any.attribute') is None


# LLM-generated content at query #6
#--------------------------

```python
def test_imports_predicate_evaluates_to_false():
    parser = Parser()
    node = Import(names=[alias(name='module_name')])
    parser.imports('root', node)
    assert parser.alias == {'root.module_name': 'module_name'}


# LLM-generated content at query #7
#--------------------------

```
def test_api_method_with_function_def():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[], returns=None)
    p.api(root, node)
    assert "test_func()" in p.doc["test_module.test_func"]
    assert "*Full name:* `test_module.test_func`" in p.doc["test_module.test_func"]

def test_api_method_with_async_function_def():
    p = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(), body=[], decorator_list=[], returns=None)
    p.api(root, node)
    assert "async test_async_func()" in p.doc["test_module.test_async_func"]
    assert "*Full name:* `test_module.test_async_func`" in p.doc["test_module.test_async_func"]

def test_api_method_with_class_def():
    p = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], body=[], decorator_list=[])
    p.api(root, node)
    assert "class TestClass" in p.doc["test_module.TestClass"]
    assert "*Full name:* `test_module.TestClass`" in p.doc["test_module.TestClass"]

def test_api_method_with_decorators():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[Name(id="decorator")], returns=None)
    p.api(root, node)
    assert "@decorator" in p.doc["test_module.test_func"]

def test_api_method_with_prefix():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[], returns=None)
    p.api(root, node, prefix="TestClass")
    assert "test_module.TestClass.test_func" in p.doc["test_module.TestClass.test_func"]


# LLM-generated content at query #8
#--------------------------

```
def test_class_api_with_bases():
    parser = Parser()
    parser.doc = {'root.Class': ''}
    parser.class_api('root', 'root.Class', [Name(id='Base', ctx=Load())], [])
    assert parser.doc['root.Class'] == '| Bases |\n|:---:|\n| `Base` |\n\n'

def test_class_api_with_enums():
    parser = Parser()
    parser.doc = {'root.Class': ''}
    parser.class_api('root', 'root.Class', [Name(id='enum.Enum', ctx=Load())], [
        Assign(targets=[Name(id='ENUM_VALUE', ctx=Store())], value=Constant(value=1))
    ])
    assert parser.doc['root.Class'] == '| Enums |\n|:---:|\n| ENUM_VALUE |\n\n'

def test_class_api_with_members():
    parser = Parser()
    parser.doc = {'root.Class': ''}
    parser.class_api('root', 'root.Class', [], [
        AnnAssign(target=Name(id='member', ctx=Store()), annotation=Name(id='int', ctx=Load()))
    ])
    assert parser.doc['root.Class'] == '| Members | Type |\n|:---:|:---:|\n| `member` | `int` |\n\n'

def test_class_api_with_private_members():
    parser = Parser()
    parser.doc = {'root.Class': ''}
    parser.class_api('root', 'root.Class', [], [
        AnnAssign(target=Name(id='_private', ctx=Store()), annotation=Name(id='int', ctx=Load()))
    ])
    assert parser.doc['root.Class'] == ''

def test_class_api_with_deleted_members():
    parser = Parser()
    parser.doc = {'root.Class': ''}
    parser.class_api('root', 'root.Class', [], [
        AnnAssign(target=Name(id='member', ctx=Store()), annotation=Name(id='int', ctx=Load())),
        Delete(targets=[Name(id='member', ctx=Del())])
    ])
    assert parser.doc['root.Class'] == ''


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    parser = Parser()
    root = "test_root"
    name = "test_name"
    body = [Assign(targets=[Name(id="test_attr", ctx=Store())], value=Constant(value=42))]
    parser.class_api(root, name, [], body)
    assert isinstance(body[0], Assign)
    assert len(body[0].targets) == 1
    assert isinstance(body[0].targets[0], Name)


# LLM-generated content at query #10
#--------------------------

```python
def test__e_type_empty_input():
    assert _e_type() == ""

def test__e_type_single_empty_element():
    assert _e_type([]) == ""

def test__e_type_multiple_empty_elements():
    assert _e_type([], []) == ""

def test__e_type_single_element_with_non_constant():
    class NonConstant: pass
    assert _e_type([NonConstant()]) == ""

def test__e_type_single_element_with_constants_of_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_single_element_with_constants_of_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_multiple_elements_with_constants_of_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)], [Constant(3), Constant(4)]) == "[int, int]"

def test__e_type_multiple_elements_with_constants_of_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")], [Constant("b"), Constant(2)]) == "[Any, Any]"

def test__e_type_mixed_elements_with_constants_and_non_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    class NonConstant: pass
    assert _e_type([Constant(1), NonConstant()], [Constant(2), Constant(3)]) == ""


# LLM-generated content at query #11
#--------------------------

```python
def test_globals_with_ann_assign():
    class Constant:
        def __init__(self, value):
            self.value = value

    class Name:
        def __init__(self, id):
            self.id = id

    class AnnAssign:
        def __init__(self, target, annotation, value):
            self.target = target
            self.annotation = annotation
            self.value = value

    parser = Parser()
    target = Name("TEST")
    annotation = "int"
    value = Constant(42)
    node = AnnAssign(target, annotation, value)
    parser.globals("root", node)
    assert parser.const["root.TEST"] == "int"

def test_globals_with_assign():
    class Constant:
        def __init__(self, value):
            self.value = value

    class Name:
        def __init__(self, id):
            self.id = id

    class Assign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    parser = Parser()
    target = Name("TEST")
    value = Constant(42)
    node = Assign([target], value)
    parser.globals("root", node)
    assert parser.const["root.TEST"] == "int"

def test_globals_with_assign_type_comment():
    class Name:
        def __init__(self, id):
            self.id = id

    class Assign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    parser = Parser()
    target = Name("TEST")
    value = "dummy"
    type_comment = "str"
    node = Assign([target], value, type_comment)
    parser.globals("root", node)
    assert parser.const["root.TEST"] == "str"

def test_globals_with_assign_invalid_targets():
    class Constant:
        def __init__(self, value):
            self.value = value

    class Name:
        def __init__(self, id):
            self.id = id

    class Assign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    parser = Parser()
    target1 = Name("TEST1")
    target2 = Name("TEST2")
    value = Constant(42)
    node = Assign([target1, target2], value)
    parser.globals("root", node)
    assert "root.TEST1" not in parser.const
    assert "root.TEST2" not in parser.const

def test_globals_with_assign_non_name_target():
    class Constant:
        def __init__(self, value):
            self.value = value

    class Assign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    parser = Parser()
    target = "TEST"
    value = Constant(42)
    node = Assign([target], value)
    parser.globals("root", node)
    assert "root.TEST" not in parser.const

def test_globals_with_assign_non_uppercase_target():
    class Constant:
        def __init__(self, value):
            self.value = value

    class Name:
        def __init__(self, id):
            self.id = id

    class Assign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    parser = Parser()
    target = Name("test")
    value = Constant(42)
    node = Assign([target], value)
    parser.globals("root", node)
    assert "root.test" not in parser.const

def test_globals_with_assign_all():
    class Constant:
        def __init__(self, value):
            self.value = value

    class Name:
        def __init__(self, id):
            self.id = id

    class Assign:
        def __init__(self, targets, value, type_comment=None):
            self.targets = targets
            self.value = value
            self.type_comment = type_comment

    parser = Parser()
    target = Name("__all__")
    value = [Constant("TEST1"), Constant("TEST2")]
    node = Assign([target], value)
    parser.globals("root", node)
    assert "root.TEST1" in parser.imp["root"]
    assert "root.TEST2" in parser.imp["root"]


# LLM-generated content at query #12
#--------------------------

```
def test_e_type_empty_elements():
    assert _e_type() == ""


# LLM-generated content at query #13
#--------------------------

```
def test_attr_predicate_evaluates_to_false():
    class TestClass:
        def __init__(self):
            self.nested = None

    obj = TestClass()
    result = _attr(obj, 'nested.nonexistent')
    assert result is None


# LLM-generated content at query #14
#--------------------------

```
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[arg(arg='x', annotation=None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| / |\n|:---:|\n|  |\n\n"

def test_func_api_with_args_and_defaults():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[None, Name(id='None', ctx=Load())])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| x | y |\n|:---:|:---:|\n|  |  |\n|  | None |\n\n"

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=arg(arg='args', annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| *args |\n|:---:|\n|  |\n\n"

def test_func_api_with_kwonlyargs_and_kwarg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg(arg='x', annotation=None)], kw_defaults=[None], kwarg=arg(arg='kwargs', annotation=None), defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| * | x | **kwargs |\n|:---:|:---:|:---:|\n|  |  |  |\n|  |  |  |\n\n"

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, Name(id='int', ctx=Load()), has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| return |\n|:---:|\n| `int` |\n\n"

def test_func_api_with_self_and_classmethod():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| self |\n|:---:|\n| `type[Self]` |\n\n"


# LLM-generated content at query #15
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
    parser.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], [Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=1))])
    assert 'X' in parser.doc['test_module.A']

def test_class_api_with_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert 'x' in parser.doc['test_module.A'] and 'int' in parser.doc['test_module.A']

def test_class_api_with_deleted_member():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None), Delete(targets=[Name(id='x', ctx=Del())])])
    assert 'x' not in parser.doc['test_module.A']

def test_class_api_with_private_member():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='_x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert '_x' not in parser.doc['test_module.A']


# LLM-generated content at query #16
#--------------------------

```python
def test_node_posonlyargs_evaluates_to_true():
    node = lambda: None
    node.posonlyargs = ['arg1']
    parser = Parser()
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert len(parser.doc['name']) > 0


# LLM-generated content at query #17
#--------------------------

```python
def test_func_ann_with_self_and_cls_method():
    parser = Parser()
    args = [arg(arg='self', annotation=Name(id='Self')), arg(arg='arg1', annotation=Name(id='int'))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_with_self_and_no_cls_method():
    parser = Parser()
    args = [arg(arg='self', annotation=Name(id='Self')), arg(arg='arg1', annotation=Name(id='int'))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_without_self():
    parser = Parser()
    args = [arg(arg='arg1', annotation=Name(id='int')), arg(arg='arg2', annotation=Name(id='str'))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg(arg='*', annotation=None), arg(arg='arg1', annotation=Name(id='int'))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['', 'int']

def test_func_ann_with_no_annotation():
    parser = Parser()
    args = [arg(arg='arg1', annotation=None), arg(arg='arg2', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['typing.Any', 'typing.Any']


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_class_api_with_bases_and_enum():
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {"root": set()}
    parser.root = {"name": "root"}
    parser.level = {"name": 0}
    bases = [Name(id="BaseClass", ctx=Load())]
    body = [AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None, simple=1)]
    parser.class_api("root", "name", bases, body)
    assert "name" in parser.doc
    assert "Bases" in parser.doc["name"]
    assert "attr1" in parser.doc["name"]

def test_class_api_without_bases_and_enum():
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {"root": set()}
    parser.root = {"name": "root"}
    parser.level = {"name": 0}
    bases = []
    body = [AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None, simple=1)]
    parser.class_api("root", "name", bases, body)
    assert "name" in parser.doc
    assert "Bases" not in parser.doc["name"]
    assert "attr1" in parser.doc["name"]

def test_class_api_with_enum():
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {"root": set()}
    parser.root = {"name": "root"}
    parser.level = {"name": 0}
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None, simple=1)]
    parser.class_api("root", "name", bases, body)
    assert "name" in parser.doc
    assert "Enums" in parser.doc["name"]
    assert "attr1" in parser.doc["name"]

def test_class_api_with_members():
    parser = Parser()
    parser.doc = {}
    parser.docstring = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {"root": set()}
    parser.root = {"name": "root"}
    parser.level = {"name": 0}
    bases = []
    body = [
        AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None, simple=1),
        Assign(targets=[Name(id="attr2", ctx=Store())], value=Constant(value=42), type_comment=None)
    ]
    parser.class_api("root", "name", bases, body)
    assert "name" in parser.doc
    assert "Members" in parser.doc["name"]
    assert "attr1" in parser.doc["name"]
    assert "attr2" in parser.doc["name"]


# LLM-generated content at query #20
#--------------------------

```python
def test_is_instance_of_AnnAssign_and_Name():
    parser = Parser()
    node = AnnAssign(target=Name(id='test'), annotation=None)
    result = isinstance(node, AnnAssign) and isinstance(node.target, Name)
    assert result == True


# LLM-generated content at query #21
#--------------------------

```
def test_api_predicate_evaluates_to_true():
    parser = Parser()
    root = "test_root"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
    prefix = ""
    parser.api(root, node, prefix=prefix)
    assert parser.doc[f"{root}.{node.name}"].startswith("### test_func()\n\n")


# LLM-generated content at query #22
#--------------------------

```
def test_predicate_evaluates_to_false():
    class MockObject:
        def __init__(self):
            self.nested = None

    obj = MockObject()
    result = _attr(obj, 'nested.missing_attr')
    assert result is None


# LLM-generated content at query #23
#--------------------------

```
def test__e_type_empty_elements():
    assert _e_type() == ""

def test__e_type_none_element():
    assert _e_type(None) == ""

def test__e_type_empty_sequence_element():
    assert _e_type([]) == ""

def test__e_type_non_constant_element():
    class NonConstant: pass
    assert _e_type([NonConstant()]) == ""

def test__e_type_mixed_type_elements():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"


# LLM-generated content at query #24
#--------------------------

```python
def test_class_api_without_bases_and_members():
    parser = Parser()
    parser.class_api("root", "root.Class", [], [])
    assert parser.doc["root.Class"] == "### class Class\n\n*Full name:* `root.Class`\n\n<a id=\"root.Class\"></a>\n\n"

def test_class_api_with_bases_and_no_members():
    parser = Parser()
    bases = ["Base1", "Base2"]
    parser.class_api("root", "root.Class", bases, [])
    assert parser.doc["root.Class"] == "### class Class\n\n*Full name:* `root.Class`\n\n<a id=\"root.Class\"></a>\n\n| Bases |\n|:---:|\n| Base1 |\n| Base2 |\n\n"

def test_class_api_with_bases_and_members():
    parser = Parser()
    bases = ["Base1", "Base2"]
    body = [Assign(targets=[Name(id="member", ctx=Store())], value=Constant(value=42))]
    parser.class_api("root", "root.Class", bases, body)
    assert parser.doc["root.Class"] == "### class Class\n\n*Full name:* `root.Class`\n\n<a id=\"root.Class\"></a>\n\n| Bases |\n|:---:|\n| Base1 |\n| Base2 |\n\n| Members | Type |\n|:---:|:---:|\n| member | int |\n\n"

def test_class_api_with_bases_and_enums():
    parser = Parser()
    bases = ["enum.Enum"]
    body = [Assign(targets=[Name(id="ENUM_VALUE", ctx=Store())], value=Constant(value=1))]
    parser.class_api("root", "root.Class", bases, body)
    assert parser.doc["root.Class"] == "### class Class\n\n*Full name:* `root.Class`\n\n<a id=\"root.Class\"></a>\n\n| Bases |\n|:---:|\n| enum.Enum |\n\n| Enums |\n|:---:|\n| ENUM_VALUE |\n\n"


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_true_for_try_node():
    class Try:
        def __init__(self, body, handlers, orelse, finalbody):
            self.body = body
            self.handlers = handlers
            self.orelse = orelse
            self.finalbody = finalbody

    try_node = Try(body=[], handlers=[], orelse=[], finalbody=[])
    assert isinstance(try_node, Try)


# LLM-generated content at query #26
#--------------------------

```
def test_func_api_with_kwarg():
    parser = Parser()
    root = "test_root"
    name = "test_func"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg(arg="kwargs", annotation=None),
        defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False
    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert "**kwargs" in parser.doc[name]


# LLM-generated content at query #27
#--------------------------

```python
def test_globals_with_multiple_targets():
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias == {}

def test_globals_with_non_name_target():
    node = Assign(targets=[Attribute(value=Name(id='obj'), attr='attr')], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias == {}

def test_globals_with_annassign_and_non_name_target():
    node = AnnAssign(target=Attribute(value=Name(id='obj'), attr='attr'), annotation=Name(id='int'), value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias == {}

def test_globals_with_assign_and_non_name_target():
    node = Assign(targets=[Attribute(value=Name(id='obj'), attr='attr')], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias == {}

def test_globals_with_assign_and_multiple_targets():
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias == {}

def test_globals_with_annassign_and_null_value():
    node = AnnAssign(target=Name(id='x'), annotation=Name(id='int'), value=None)
    parser = Parser()
    parser.globals('root', node)
    assert parser.alias == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_e_type_empty_sequence():
    assert _e_type([]) == "[]"

def test_e_type_none_elements():
    assert _e_type([None]) == ""

def test_e_type_single_constant():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)]) == "[int]"

def test_e_type_multiple_constants_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test_e_type_multiple_constants_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test_e_type_nested_sequence_with_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([[Constant(1), Constant(2)]]) == "[int]"

def test_e_type_nested_sequence_with_non_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([[Constant(1), "not a constant"]]) == ""

def test_e_type_multiple_sequences():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)], [Constant(2)]) == "[int, int]"

def test_e_type_multiple_sequences_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)], [Constant("a")]) == "[int, str]"


# LLM-generated content at query #29
#--------------------------

```python
def test_api_predicate_evaluates_to_true_for_functiondef():
    parser = Parser()
    root = "test_root"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[])
    parser.api(root, node)

def test_api_predicate_evaluates_to_true_for_asyncfunctiondef():
    parser = Parser()
    root = "test_root"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(), body=[], decorator_list=[])
    parser.api(root, node)

def test_api_predicate_evaluates_to_true_for_classdef():
    parser = Parser()
    root = "test_root"
    node = ClassDef(name="TestClass", bases=[], body=[], decorator_list=[])
    parser.api(root, node)


# LLM-generated content at query #30
#--------------------------

```python
def test_const_type_with_constant_int():
    node = Constant(value=42)
    result = const_type(node)
    assert result == "int"

def test_const_type_with_constant_str():
    node = Constant(value="hello")
    result = const_type(node)
    assert result == "str"

def test_const_type_with_constant_float():
    node = Constant(value=3.14)
    result = const_type(node)
    assert result == "float"

def test_const_type_with_tuple_of_constants():
    node = Tuple(elts=[Constant(value=1), Constant(value=2), Constant(value=3)])
    result = const_type(node)
    assert result == "tuple[int, int, int]"

def test_const_type_with_list_of_constants():
    node = List(elts=[Constant(value="a"), Constant(value="b")])
    result = const_type(node)
    assert result == "list[str, str]"

def test_const_type_with_set_of_constants():
    node = Set(elts=[Constant(value=True), Constant(value=False)])
    result = const_type(node)
    assert result == "set[bool, bool]"

def test_const_type_with_dict_of_constants():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    result = const_type(node)
    assert result == "dict[int, int, str, str]"

def test_const_type_with_call_to_builtin_function():
    node = Call(func=Name(id="int"), args=[Constant(value=42)])
    result = const_type(node)
    assert result == "int"

def test_const_type_with_call_to_non_builtin_function():
    node = Call(func=Name(id="custom_func"), args=[Constant(value=42)])
    result = const_type(node)
    assert result == "Any"

def test_const_type_with_non_constant_node():
    node = Name(id="x")
    result = const_type(node)
    assert result == "Any"


# LLM-generated content at query #31
#--------------------------

```
def test_visit_Subscript_handles_typing_Union():
    resolver = Resolver("root", {})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

def test_visit_Subscript_handles_typing_Optional():
    resolver = Resolver("root", {})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value is None

def test_visit_Subscript_handles_PEP585_deprecated_names():
    resolver = Resolver("root", {})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

def test_visit_Subscript_returns_node_for_non_typing_Union_or_Optional():
    resolver = Resolver("root", {})
    node = Subscript(Name("Dict", Load()), Tuple([Name("str", Load()), Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result == node

def test_visit_Subscript_returns_node_for_non_Name_value():
    resolver = Resolver("root", {})
    node = Subscript(Attribute(Name("typing", Load()), "List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #32
#--------------------------

```
def test_func_api_has_default_false():
    parser = Parser()
    args_node = arguments(
        posonlyargs=[],
        args=[arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=1)]
    )
    parser.func_api('root', 'name', args_node, None, has_self=False, cls_method=False)
    assert 'x' in parser.doc['name']


# LLM-generated content at query #33
#--------------------------

```python
def test_func_api_without_self_and_class_method():
    parser = Parser.new(link=True, level=1, toc=False)
    args = [arg('arg1', None), arg('arg2', None)]
    default = [None, None]
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| arg1 | arg2 |\n|:---:|:---:|\n| ANY | ANY |\n\n"

def test_func_api_with_self_and_class_method():
    parser = Parser.new(link=True, level=1, toc=False)
    args = [arg('self', None), arg('arg1', None)]
    default = [None, None]
    parser.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| self | arg1 |\n|:---:|:---:|\n| type[Self] | ANY |\n\n"

def test_func_api_with_vararg_and_kwarg():
    parser = Parser.new(link=True, level=1, toc=False)
    args = [arg('*args', None), arg('**kwargs', None)]
    default = [None, None]
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| *args | **kwargs |\n|:---:|:---:|\n|  |  |\n\n"

def test_func_api_with_posonlyargs_and_kwonlyargs():
    parser = Parser.new(link=True, level=1, toc=False)
    args = [arg('arg1', None), arg('/', None), arg('arg2', None), arg('*', None), arg('arg3', None)]
    default = [None, None, None, None, None]
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| arg1 | / | arg2 | * | arg3 |\n|:---:|:---:|:---:|:---:|:---:|\n| ANY |  | ANY |  | ANY |\n\n"

def test_func_api_with_default_values():
    parser = Parser.new(link=True, level=1, toc=False)
    args = [arg('arg1', None), arg('arg2', None)]
    default = [Constant(value=1), Constant(value=2)]
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == "### name()\n\n*Full name:* `name`\n<a id=\"name\"></a>\n\n| arg1 | arg2 |\n|:---:|:---:|\n| ANY | ANY |\n| `1` | `2` |\n\n"


# LLM-generated content at query #34
#--------------------------

```python
def test_globals_predicate_evaluates_to_false_when_not_all_or_not_tuple_list():
    parser = Parser()
    root = "test_root"
    
    # Case 1: left.id is not '__all__'
    node = Assign(targets=[Name(id='not_all', ctx=Store())], value=List(elts=[], ctx=Load()))
    parser.globals(root, node)
    assert '__all__' not in parser.alias
    
    # Case 2: node.value is not Tuple or List
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=Name(id='x', ctx=Load()))
    parser.globals(root, node)
    assert not parser.imp[root]
    
    # Case 3: left.id is not '__all__' AND node.value is not Tuple or List
    node = Assign(targets=[Name(id='other', ctx=Store())], value=Name(id='y', ctx=Load()))
    parser.globals(root, node)
    assert '__all__' not in parser.alias and not parser.imp[root]


# LLM-generated content at query #35
#--------------------------

```
def test_func_ann_with_annotation_and_not_self_or_star():
    parser = Parser()
    arg_node = arg(arg='test_arg', annotation=Name(id='int', ctx=Load()))
    args = [arg_node]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int']


# LLM-generated content at query #36
#--------------------------

```python
def test_load_docstring():
    parser = Parser()
    parser.doc['mod'] = '# Module `mod`\n\n'
    parser.doc['mod.func'] = '# func()\n\n'
    parser.docstring['mod.func'] = ''

    class MockModule:
        def func():
            """Docstring for func"""
            pass

    parser.load_docstring('mod', MockModule)
    assert parser.docstring['mod.func'] == '```python\nDocstring for func\n```'


# LLM-generated content at query #37
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

def test_parser_new_method_with_parameters():
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


# LLM-generated content at query #38
#--------------------------

```python
def test_e_type_with_non_constant_elements():
    elements = [[MockNonConstantClass()]]
    result = _e_type(*elements)
    assert result == ""


# LLM-generated content at query #39
#--------------------------

```
def test__e_type_with_constant_elements():
    class Constant:
        def __init__(self, value):
            self.value = value

    def _type_name(value):
        return type(value).__name__

    elements = [[Constant(1), Constant(2), Constant(3)]]
    result = _e_type(*elements)
    assert result == "[int]"

def test__e_type_with_empty_elements():
    result = _e_type()
    assert result == ""

def test__e_type_with_none_element():
    elements = [None]
    result = _e_type(*elements)
    assert result == ""

def test__e_type_with_mixed_constant_types():
    class Constant:
        def __init__(self, value):
            self.value = value

    def _type_name(value):
        return type(value).__name__

    elements = [[Constant(1), Constant("a")]]
    result = _e_type(*elements)
    assert result == ""

def test__e_type_with_non_constant_elements():
    elements = [[1, 2, 3]]
    result = _e_type(*elements)
    assert result == ""


# LLM-generated content at query #40
#--------------------------

```
def test_compile_with_toc():
    parser = Parser(toc=True)
    parser.doc = {'module': '# Module `module`\n<a id="module"></a>\n\n'}
    parser.docstring = {'module': 'Module docstring'}
    parser.level = {'module': 0}
    parser.root = {'module': 'module'}
    parser.imp = {'module': set()}
    parser.const = {}
    expected_output = '**Table of contents:**\n+ [module](#module)\n\n# Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert parser.compile() == expected_output

def test_compile_without_toc():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n<a id="module"></a>\n\n'}
    parser.docstring = {'module': 'Module docstring'}
    parser.level = {'module': 0}
    parser.root = {'module': 'module'}
    parser.imp = {'module': set()}
    parser.const = {}
    expected_output = '# Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert parser.compile() == expected_output

def test_compile_with_constants():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n<a id="module"></a>\n\n'}
    parser.docstring = {'module': 'Module docstring'}
    parser.level = {'module': 0}
    parser.root = {'module': 'module'}
    parser.imp = {'module': set()}
    parser.const = {'module.CONST': 'int'}
    expected_output = '# Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert parser.compile() == expected_output

def test_compile_with_missing_docstring():
    parser = Parser(toc=False)
    parser.doc = {'module': '# Module `module`\n<a id="module"></a>\n\n'}
    parser.docstring = {}
    parser.level = {'module': 0}
    parser.root = {'module': 'module'}
    parser.imp = {'module': set()}
    parser.const = {}
    expected_output = '# Module `module`\n<a id="module"></a>\n\n'
    assert parser.compile() == expected_output

def test_compile_with_multiple_modules():
    parser = Parser(toc=True)
    parser.doc = {
        'module1': '# Module `module1`\n<a id="module1"></a>\n\n',
        'module2': '# Module `module2`\n<a id="module2"></a>\n\n'
    }
    parser.docstring = {
        'module1': 'Module1 docstring',
        'module2': 'Module2 docstring'
    }
    parser.level = {'module1': 0, 'module2': 0}
    parser.root = {'module1': 'module1', 'module2': 'module2'}
    parser.imp = {'module1': set(), 'module2': set()}
    parser.const = {}
    expected_output = '**Table of contents:**\n+ [module1](#module1)\n+ [module2](#module2)\n\n# Module `module1`\n<a id="module1"></a>\n\nModule1 docstring\n\n# Module `module2`\n<a id="module2"></a>\n\nModule2 docstring\n'
    assert parser.compile() == expected_output


# LLM-generated content at query #41
#--------------------------

```
def test_const_type_call_with_name_or_attribute_func():
    node = Call(func=Name(id='int', ctx=Load()))
    assert const_type(node) == 'int'
    node = Call(func=Attribute(value=Name(id='builtins', ctx=Load()), attr='str', ctx=Load()))
    assert const_type(node) == 'str'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_globals_handles_ann_assign_with_name_target_and_value():
    p = Parser()
    node = AnnAssign(target=Name(id='x'), value=Constant(value=1), annotation=Name(id='int'))
    p.globals('root', node)
    assert p.alias['root.x'] == '1'
    assert p.const['root.x'] == 'int'

def test_globals_handles_assign_with_name_target_and_value():
    p = Parser()
    node = Assign(targets=[Name(id='y')], value=Constant(value='test'))
    p.globals('root', node)
    assert p.alias['root.y'] == "'test'"
    assert p.const['root.y'] == 'str'

def test_globals_handles_assign_with_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z')], value=Constant(value=3.14), type_comment='float')
    p.globals('root', node)
    assert p.alias['root.z'] == '3.14'
    assert p.const['root.z'] == 'float'

def test_globals_handles_uppercase_name_as_constant():
    p = Parser()
    node = Assign(targets=[Name(id='PI')], value=Constant(value=3.14159))
    p.globals('root', node)
    assert p.alias['root.PI'] == '3.14159'
    assert p.const['root.PI'] == 'float'

def test_globals_handles___all__assignment_with_tuple():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Tuple(elts=[Constant(value='x'), Constant(value='y')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.x', 'root.y'}

def test_globals_handles___all__assignment_with_list():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=List(elts=[Constant(value='a'), Constant(value='b')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.a', 'root.b'}

def test_globals_ignores_non_name_targets():
    p = Parser()
    node = Assign(targets=[Subscript()], value=Constant(value=1))
    p.globals('root', node)
    assert not p.alias
    assert not p.const
    assert not p.imp['root']

def test_globals_ignores_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='a'), Name(id='b')], value=Constant(value=1))
    p.globals('root', node)
    assert not p.alias
    assert not p.const
    assert not p.imp['root']

def test_globals_ignores_non_tuple_list___all__():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Constant(value='invalid'))
    p.globals('root', node)
    assert not p.imp['root']


# LLM-generated content at query #2
#--------------------------

```python
def test_visit_Attribute_simple_attribute():
    resolver = Resolver("root", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_non_typing_attribute():
    resolver = Resolver("root", {})
    node = Attribute(Name("other_module", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other_module"
    assert result.attr == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_nested_attribute():
    resolver = Resolver("root", {})
    node = Attribute(Attribute(Name("typing", Load()), "Optional", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Name)
    assert result.value.id == "Optional"
    assert result.attr == "List"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #3
#--------------------------

```python
def test_compile_empty_parser():
    p = Parser()
    assert p.compile() == ""

def test_compile_single_module():
    p = Parser()
    p.doc["module"] = "# Module `{}`\n\n"
    p.root["module"] = "module"
    p.level["module"] = 0
    p.imp["module"] = set()
    assert p.compile() == "# Module `module`\n\n\n"

def test_compile_with_toc():
    p = Parser(toc=True)
    p.doc["module"] = "# Module `{}`\n\n"
    p.root["module"] = "module"
    p.level["module"] = 0
    p.imp["module"] = set()
    assert p.compile() == "**Table of contents:**\n+ [`module`](#module)\n\n# Module `module`\n\n\n"

def test_compile_with_constants():
    p = Parser()
    p.doc["module"] = "# Module `{}`\n\n"
    p.root["module"] = "module"
    p.level["module"] = 0
    p.imp["module"] = set()
    p.const["module.CONST"] = "int"
    p.root["module.CONST"] = "module"
    assert p.compile() == "# Module `module`\n\n| Constants | Type |\n|:----------|:-----|\n| `CONST` | `int` |\n\n\n"

def test_compile_with_private_members():
    p = Parser()
    p.doc["module._private"] = "# _private()\n\n"
    p.root["module._private"] = "module"
    p.level["module._private"] = 0
    p.imp["module"] = set()
    assert p.compile() == ""

def test_compile_with_magic_methods():
    p = Parser()
    p.doc["module.__magic__"] = "# __magic__()\n\n"
    p.root["module.__magic__"] = "module"
    p.level["module.__magic__"] = 0
    p.imp["module"] = set()
    assert p.compile() == ""

def test_compile_with_nested_modules():
    p = Parser()
    p.doc["module"] = "# Module `{}`\n\n"
    p.doc["module.sub"] = "## sub()\n\n"
    p.root["module"] = "module"
    p.root["module.sub"] = "module"
    p.level["module"] = 0
    p.level["module.sub"] = 1
    p.imp["module"] = set()
    assert p.compile() == "# Module `module`\n\n\n## sub()\n\n\n"


# LLM-generated content at query #4
#--------------------------

```
def test_const_type_with_constant_int():
    node = Constant(42)
    assert const_type(node) == "int"


def test_const_type_with_constant_float():
    node = Constant(3.14)
    assert const_type(node) == "float"


def test_const_type_with_constant_str():
    node = Constant("hello")
    assert const_type(node) == "str"


def test_const_type_with_empty_list():
    node = List([], None)
    assert const_type(node) == "list[]"


def test_const_type_with_list_of_ints():
    node = List([Constant(1), Constant(2), Constant(3)], None)
    assert const_type(node) == "list[int]"


def test_const_type_with_list_of_mixed_types():
    node = List([Constant(1), Constant("two"), Constant(3.0)], None)
    assert const_type(node) == "list[Any]"


def test_const_type_with_empty_tuple():
    node = Tuple([], None)
    assert const_type(node) == "tuple[]"


def test_const_type_with_tuple_of_strs():
    node = Tuple([Constant("a"), Constant("b"), Constant("c")], None)
    assert const_type(node) == "tuple[str]"


def test_const_type_with_empty_set():
    node = Set([])
    assert const_type(node) == "set[]"


def test_const_type_with_set_of_floats():
    node = Set([Constant(1.1), Constant(2.2), Constant(3.3)])
    assert const_type(node) == "set[float]"


def test_const_type_with_empty_dict():
    node = Dict([], [])
    assert const_type(node) == "dict[]"


def test_const_type_with_dict_of_int_to_str():
    node = Dict([Constant(1), Constant(2)], [Constant("a"), Constant("b")])
    assert const_type(node) == "dict[int, str]"


def test_const_type_with_dict_of_mixed_types():
    node = Dict([Constant(1), Constant("two")], [Constant(3.0), Constant(4)])
    assert const_type(node) == "dict[Any, Any]"


def test_const_type_with_builtin_int_call():
    node = Call(Name("int", None), [Constant("42")], [])
    assert const_type(node) == "int"


def test_const_type_with_builtin_str_call():
    node = Call(Name("str", None), [Constant(42)], [])
    assert const_type(node) == "str"


def test_const_type_with_unknown_call():
    node = Call(Name("unknown", None), [], [])
    assert const_type(node) == "Any"


# LLM-generated content at query #5
#--------------------------

```python
def test_table_with_single_item():
    result = table('Header1', 'Header2', items=[['Item1', 'Item2']])
    expected = "| Header1 | Header2 |\n|:-------:|:-------:|\n| Item1 | Item2 |\n\n"
    assert result == expected

def test_table_with_multiple_items():
    result = table('A', 'B', items=[['1', '2'], ['3', '4']])
    expected = "| A | B |\n|:---:|:---:|\n| 1 | 2 |\n| 3 | 4 |\n\n"
    assert result == expected

def test_table_with_single_column():
    result = table('Column1', items=[['Value1'], ['Value2']])
    expected = "| Column1 |\n|:-------:|\n| Value1 |\n| Value2 |\n\n"
    assert result == expected

def test_table_with_short_values():
    result = table('X', 'Y', items=[['a', 'b'], ['c', 'd']])
    expected = "| X | Y |\n|:---:|:---:|\n| a | b |\n| c | d |\n\n"
    assert result == expected

def test_table_with_mixed_length_values():
    result = table('LongHeader', 'Short', items=[['Value', 'X'], ['AnotherValue', 'Y']])
    expected = "| LongHeader | Short |\n|:----------:|:-----:|\n| Value | X |\n| AnotherValue | Y |\n\n"
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[arg(arg='x', annotation=None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert "| x | return |" in p.doc['name']

def test_func_api_with_args_and_defaults():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[Constant(value=1)])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert "| x | y | return |" in p.doc['name']
    assert "|   | 1 |   |" in p.doc['name']

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=arg(arg='args', annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert "| *args | return |" in p.doc['name']

def test_func_api_with_kwonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg(arg='x', annotation=None)], kw_defaults=[Constant(value=1)], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert "| * | x | return |" in p.doc['name']
    assert "|   | 1 |   |" in p.doc['name']

def test_func_api_with_kwarg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=arg(arg='kwargs', annotation=None), defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert "| **kwargs | return |" in p.doc['name']

def test_func_api_with_has_self():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=False)
    assert "| Self | return |" in p.doc['name']

def test_func_api_with_cls_method():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='cls', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert "| type[Self] | return |" in p.doc['name']

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, Name(id='int'), has_self=False, cls_method=False)
    assert "| return |" in p.doc['name']
    assert "| int |" in p.doc['name']


# LLM-generated content at query #7
#--------------------------

```python
def test_doctest_wraps_doctest_as_markdown_python_code():
    doc = ">>> print('Hello, world!')\nHello, world!"
    expected = "```python\n>>> print('Hello, world!')\n```\nHello, world!"
    assert doctest(doc) == expected

def test_doctest_handles_multiple_doctest_lines():
    doc = ">>> a = 1\n>>> b = 2\n>>> print(a + b)\n3"
    expected = "```python\n>>> a = 1\n>>> b = 2\n>>> print(a + b)\n```\n3"
    assert doctest(doc) == expected

def test_doctest_handles_doctest_followed_by_text():
    doc = ">>> print('Hello')\nHello\nThis is a test."
    expected = "```python\n>>> print('Hello')\n```\nHello\nThis is a test."
    assert doctest(doc) == expected

def test_doctest_handles_doctest_in_the_middle_of_text():
    doc = "This is a test.\n>>> print('Hello')\nHello\nThis is another test."
    expected = "This is a test.\n```python\n>>> print('Hello')\n```\nHello\nThis is another test."
    assert doctest(doc) == expected

def test_doctest_handles_empty_input():
    doc = ""
    expected = ""
    assert doctest(doc) == expected

def test_doctest_handles_no_doctest_lines():
    doc = "This is a test.\nThis is another test."
    expected = "This is a test.\nThis is another test."
    assert doctest(doc) == expected

def test_doctest_handles_trailing_doctest_line():
    doc = ">>> print('Hello')"
    expected = "```python\n>>> print('Hello')\n```"
    assert doctest(doc) == expected


# LLM-generated content at query #8
#--------------------------

```python
def test_api_method_function_def():
    parser = Parser()
    function_def = FunctionDef(name='test_function', args=arguments(), body=[], decorator_list=[], returns=None)
    parser.api('root', function_def)
    assert 'test_function()' in parser.doc['root.test_function']

def test_api_method_async_function_def():
    parser = Parser()
    async_function_def = AsyncFunctionDef(name='test_async_function', args=arguments(), body=[], decorator_list=[], returns=None)
    parser.api('root', async_function_def)
    assert 'async test_async_function()' in parser.doc['root.test_async_function']

def test_api_method_class_def():
    parser = Parser()
    class_def = ClassDef(name='TestClass', bases=[], body=[], decorator_list=[])
    parser.api('root', class_def)
    assert 'class TestClass' in parser.doc['root.TestClass']

def test_api_method_with_prefix():
    parser = Parser()
    function_def = FunctionDef(name='test_function', args=arguments(), body=[], decorator_list=[], returns=None)
    parser.api('root', function_def, prefix='prefix')
    assert 'prefix.test_function()' in parser.doc['root.prefix.test_function']

def test_api_method_with_decorators():
    parser = Parser()
    function_def = FunctionDef(name='test_function', args=arguments(), body=[], decorator_list=['@decorator'], returns=None)
    parser.api('root', function_def)
    assert '@decorator' in parser.doc['root.test_function']

def test_api_method_with_docstring():
    parser = Parser()
    function_def = FunctionDef(name='test_function', args=arguments(), body=[], decorator_list=[], returns=None)
    function_def.body = [Expr(value=Constant(value='Docstring'))]
    parser.api('root', function_def)
    assert 'Docstring' in parser.docstring['root.test_function']

def test_api_method_with_nested_class():
    parser = Parser()
    class_def = ClassDef(name='TestClass', bases=[], body=[], decorator_list=[])
    function_def = FunctionDef(name='test_function', args=arguments(), body=[], decorator_list=[], returns=None)
    class_def.body = [function_def]
    parser.api('root', class_def)
    assert 'test_function()' in parser.doc['root.TestClass.test_function']


# LLM-generated content at query #9
#--------------------------

```python
def test__e_type_with_no_elements():
    assert _e_type() == ""

def test__e_type_with_empty_elements():
    assert _e_type([]) == ""

def test__e_type_with_non_constant_elements():
    class NotConstant:
        pass
    assert _e_type([NotConstant()]) == ""

def test__e_type_with_single_constant_element():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)]) == "[int]"

def test__e_type_with_multiple_constant_elements_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int, int]"

def test__e_type_with_multiple_constant_elements_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_with_nested_constant_elements_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)], [Constant(3), Constant(4)]) == "[int, int], [int, int]"

def test__e_type_with_nested_constant_elements_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")], [Constant(3.0), Constant(4)]) == "[Any], [Any]"


# LLM-generated content at query #10
#--------------------------

```python
def test_const_type_with_constant():
    node = Constant(42)
    result = const_type(node)
    assert result == "int"

def test_const_type_with_tuple():
    node = Tuple([Constant(1), Constant(2), Constant(3)])
    result = const_type(node)
    assert result == "tuple[int, int, int]"

def test_const_type_with_list():
    node = List([Constant("a"), Constant("b"), Constant("c")])
    result = const_type(node)
    assert result == "list[str, str, str]"

def test_const_type_with_set():
    node = Set([Constant(1.0), Constant(2.0), Constant(3.0)])
    result = const_type(node)
    assert result == "set[float, float, float]"

def test_const_type_with_dict():
    node = Dict([Constant("key1"), Constant("key2")], [Constant(1), Constant(2)])
    result = const_type(node)
    assert result == "dict[str, str, int, int]"

def test_const_type_with_call():
    node = Call(Name("int"), [Constant("42")])
    result = const_type(node)
    assert result == "int"

def test_const_type_with_invalid_call():
    node = Call(Name("unknown_func"), [])
    result = const_type(node)
    assert result == "Any"

def test_const_type_with_non_constant():
    node = Name("x")
    result = const_type(node)
    assert result == "Any"

def test_const_type_with_empty_tuple():
    node = Tuple([])
    result = const_type(node)
    assert result == "tuple[]"

def test_const_type_with_empty_list():
    node = List([])
    result = const_type(node)
    assert result == "list[]"

def test_const_type_with_empty_set():
    node = Set([])
    result = const_type(node)
    assert result == "set[]"

def test_const_type_with_empty_dict():
    node = Dict([], [])
    result = const_type(node)
    assert result == "dict[]"

def test_const_type_with_mixed_tuple():
    node = Tuple([Constant(1), Constant("a")])
    result = const_type(node)
    assert result == "tuple[Any, Any]"

def test_const_type_with_mixed_list():
    node = List([Constant(1), Constant("a")])
    result = const_type(node)
    assert result == "list[Any, Any]"

def test_const_type_with_mixed_set():
    node = Set([Constant(1), Constant("a")])
    result = const_type(node)
    assert result == "set[Any, Any]"

def test_const_type_with_mixed_dict():
    node = Dict([Constant(1), Constant("a")], [Constant("b"), Constant(2)])
    result = const_type(node)
    assert result == "dict[Any, Any, Any, Any]"


# LLM-generated content at query #11
#--------------------------

```
def test_predicate_at_line_16_evaluates_to_true():
    doc = ">>> print('hello')"
    lines = doc.splitlines()
    i = 0
    signed = lines[i].startswith(">>> ")
    assert signed and i == len(lines) - 1


# LLM-generated content at query #12
#--------------------------

```
def test_class_api_with_bases():
    parser = Parser()
    parser.doc = {'root.name': ''}
    parser.class_api('root', 'root.name', [Name(id='Base', ctx=Load())], [])
    assert 'Bases' in parser.doc['root.name']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.doc = {'root.name': ''}
    parser.class_api('root', 'root.name', [Name(id='enum.Enum', ctx=Load())], [Assign(targets=[Name(id='ATTR', ctx=Store())], value=Constant(value=1))])
    assert 'Enums' in parser.doc['root.name']

def test_class_api_with_members():
    parser = Parser()
    parser.doc = {'root.name': ''}
    parser.class_api('root', 'root.name', [], [AnnAssign(target=Name(id='attr', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert 'Members' in parser.doc['root.name']

def test_class_api_with_deleted_member():
    parser = Parser()
    parser.doc = {'root.name': ''}
    parser.class_api('root', 'root.name', [], [AnnAssign(target=Name(id='attr', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None), Delete(targets=[Name(id='attr', ctx=Del())])])
    assert 'Members' not in parser.doc['root.name']

def test_class_api_with_non_public_member():
    parser = Parser()
    parser.doc = {'root.name': ''}
    parser.class_api('root', 'root.name', [], [AnnAssign(target=Name(id='_attr', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert 'Members' not in parser.doc['root.name']


# LLM-generated content at query #13
#--------------------------

```python
def test_parse():
    parser = Parser()
    parser.parse("root", "import os\nimport sys\n")
    assert parser.imp["root"] == set()
    assert parser.alias == {"root.os": "os", "root.sys": "sys"}

    parser.parse("root", "x = 10\ny = 20")
    assert parser.alias == {"root.os": "os", "root.sys": "sys", "root.x": "10", "root.y": "20"}

    parser.parse("root", "class MyClass:\n    pass")
    assert "root.MyClass" in parser.doc

    parser.parse("root", "def my_func():\n    pass")
    assert "root.my_func" in parser.doc

    parser.parse("root", "async def my_async_func():\n    pass")
    assert "root.my_async_func" in parser.doc


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    parser = Parser()
    node = Delete(targets=[Name(id='attr')])
    result = [isinstance(d, Name) for d in node.targets]
    assert all(result)


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_31_evaluates_to_true():
    parser = Parser()
    node = Delete(targets=[Name(id='attr1')])
    result = isinstance(node.targets[0], Name)
    assert result == True


# LLM-generated content at query #16
#--------------------------

```
def test_is_magic_predicate_evaluates_to_true():
    parser = Parser()
    parser.doc = {'__init__': 'docstring'}
    parser.docstring = {}
    parser.root = {'__init__': 'root'}
    parser.level = {'__init__': 1}
    parser.imp = {'root': set()}
    parser.const = {}
    result = parser.compile()
    assert '__init__' not in result


# LLM-generated content at query #17
#--------------------------

```python
def test_walk_body_with_single_node():
    body = [stmt()]
    result = list(walk_body(body))
    assert result == body

def test_walk_body_with_if_node():
    if_node = If(body=[stmt()], orelse=[stmt()])
    body = [if_node]
    result = list(walk_body(body))
    assert result == if_node.body + if_node.orelse

def test_walk_body_with_try_node():
    try_node = Try(body=[stmt()], handlers=[ExceptHandler(body=[stmt()])], orelse=[stmt()], finalbody=[stmt()])
    body = [try_node]
    result = list(walk_body(body))
    assert result == try_node.body + [stmt() for h in try_node.handlers] + try_node.orelse + try_node.finalbody

def test_walk_body_with_nested_nodes():
    inner_if = If(body=[stmt()], orelse=[stmt()])
    try_node = Try(body=[inner_if], handlers=[ExceptHandler(body=[stmt()])], orelse=[stmt()], finalbody=[stmt()])
    body = [try_node]
    result = list(walk_body(body))
    expected = inner_if.body + inner_if.orelse + [stmt() for h in try_node.handlers] + try_node.orelse + try_node.finalbody
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test_parse_method():
    p = Parser()
    doc = p.doc
    level = p.level
    imp = p.imp
    root = p.root
    p.parse("example_module", """
import os
from sys import path
def example_function():
    pass
class ExampleClass:
    pass
""")
    assert doc["example_module"] == "## Module `example_module`\n\n"
    assert level["example_module"] == 0
    assert imp["example_module"] == set()
    assert root["example_module"] == "example_module"


# LLM-generated content at query #19
#--------------------------

```python
def test_is_public_with_docstring():
    parser = Parser()
    parser.doc = {"module.name": ""}
    parser.docstring = {"module.name": "docstring"}
    parser.root = {"module.name": "module"}
    parser.level = {"module.name": 1}
    parser.imp = {"module": set()}
    assert parser.is_public("module.name") == True

def test_is_public_without_docstring():
    parser = Parser()
    parser.doc = {"module.name": ""}
    parser.docstring = {}
    parser.root = {"module.name": "module"}
    parser.level = {"module.name": 1}
    parser.imp = {"module": set()}
    assert parser.is_public("module.name") == True


# LLM-generated content at query #20
#--------------------------

```
def test__e_type_returns_empty_string_when_elements_empty():
    assert _e_type() == ""

def test__e_type_returns_empty_string_when_element_is_none():
    assert _e_type(None) == ""

def test__e_type_returns_empty_string_when_element_contains_non_constant():
    class NonConstant:
        pass
    assert _e_type([NonConstant()]) == ""

def test__e_type_returns_type_name_when_single_element_single_constant():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)]) == "[int]"

def test__e_type_returns_any_when_multiple_constants_with_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_returns_type_name_when_multiple_constants_with_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_returns_multiple_types_when_multiple_elements():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)], [Constant("a")]) == "[int, str]"


# LLM-generated content at query #21
#--------------------------

```python
def test_globals_with_assign_node_with_type_comment():
    parser = Parser()
    node = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=42), type_comment='int')
    parser.globals('root', node)
    assert parser.alias['root.x'] == '42'
    assert parser.const == {}


# LLM-generated content at query #22
#--------------------------

```python
def test_walk_body_yields_non_control_flow_nodes():
    body = [
        ast.Assign(targets=[ast.Name(id='x')], value=ast.Num(n=1)),
        ast.If(test=ast.Name(id='cond'), body=[ast.Assign(targets=[ast.Name(id='y')], value=ast.Num(n=2))], orelse=[]),
        ast.Expr(value=ast.Call(func=ast.Name(id='print'), args=[ast.Name(id='x')], keywords=[]))
    ]
    result = list(walk_body(body))
    assert len(result) == 2
    assert isinstance(result[0], ast.Assign)
    assert isinstance(result[1], ast.Expr)


# LLM-generated content at query #23
#--------------------------

```
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| x | / |' in p.doc['name']

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
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)], kw_defaults=[None, Constant(value=1)], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| * | x | y |' in p.doc['name']
    assert '|   |   | 1 |' in p.doc['name']

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

def test_func_api_with_self_param():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=False)
    assert '| Self |' in p.doc['name']

def test_func_api_with_cls_param():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='cls', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert '| type[Self] |' in p.doc['name']


# LLM-generated content at query #24
#--------------------------

```
def test_class_api_with_bases():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [Name(id='object', ctx=Load())], [])
    assert 'Bases' in p.doc['test_module.A']

def test_class_api_with_enum_bases():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], [Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=1))])
    assert 'Enums' in p.doc['test_module.A']

def test_class_api_with_members():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert 'Members' in p.doc['test_module.A']

def test_class_api_with_deleted_member():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None), Delete(targets=[Name(id='x', ctx=Del())])])
    assert 'Members' not in p.doc['test_module.A']

def test_class_api_with_private_member():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='_x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert 'Members' not in p.doc['test_module.A']


# LLM-generated content at query #25
#--------------------------

```python
def test_globals_node_type_comment_is_not_none():
    root = "module"
    node = Assign(targets=[Name(id="x", ctx=Store())], value=Constant(value=42), type_comment="int")
    parser = Parser()
    parser.globals(root, node)
    assert parser.alias["module.x"] == "42"
    assert parser.const.get("module.x", ANY) == "int"


# LLM-generated content at query #26
#--------------------------

```python
def test_const_type_with_call_to_builtin_func():
    node = Call(func=Name(id='int', ctx=Load()), args=[], keywords=[])
    assert const_type(node) == 'int'

def test_const_type_with_call_to_builtin_attribute():
    node = Call(func=Attribute(value=Name(id='builtins', ctx=Load()), attr='int', ctx=Load()), args=[], keywords=[])
    assert const_type(node) == 'int'


# LLM-generated content at query #27
#--------------------------

```
def test_empty_element_in_elements():
    assert _e_type([]) == ""


# LLM-generated content at query #28
#--------------------------

```python
def test_api_method_handles_class_def_with_nested_functions():
    parser = Parser()
    class_def = ClassDef(name="TestClass", bases=[], body=[FunctionDef(name="test_func", args=arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[])], decorator_list=[], type_comment=None)
    parser.api("root", class_def)
    assert "test_func" in parser.doc


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_25_evaluates_to_true():
    parser = Parser()
    node = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api("root", node, prefix="prefix")
    assert parser.doc["root.prefix.test"] == "### test()\n\n*Full name:* `root.prefix.test`\n\n"


# LLM-generated content at query #30
#--------------------------

```python
def test_walk_body_with_no_nested_structure():
    body = ["stmt1", "stmt2", "stmt3"]
    result = list(walk_body(body))
    assert result == ["stmt1", "stmt2", "stmt3"]

def test_walk_body_with_if_statement():
    body = ["stmt1", {"type": "If", "body": ["stmt2"], "orelse": ["stmt3"]}, "stmt4"]
    result = list(walk_body(body))
    assert result == ["stmt1", "stmt2", "stmt3", "stmt4"]

def test_walk_body_with_try_statement():
    body = ["stmt1", {"type": "Try", "body": ["stmt2"], "handlers": [{"body": ["stmt3"]}], "orelse": ["stmt4"], "finalbody": ["stmt5"]}, "stmt6"]
    result = list(walk_body(body))
    assert result == ["stmt1", "stmt2", "stmt3", "stmt4", "stmt5", "stmt6"]

def test_walk_body_with_nested_if_and_try():
    body = ["stmt1", {"type": "If", "body": [{"type": "Try", "body": ["stmt2"], "handlers": [{"body": ["stmt3"]}], "orelse": ["stmt4"], "finalbody": ["stmt5"]}], "orelse": ["stmt6"]}, "stmt7"]
    result = list(walk_body(body))
    assert result == ["stmt1", "stmt2", "stmt3", "stmt4", "stmt5", "stmt6", "stmt7"]

def test_walk_body_with_empty_body():
    body = []
    result = list(walk_body(body))
    assert result == []


# LLM-generated content at query #31
#--------------------------

```python
def test_parse_sets_docstring_when_docstring_exists():
    parser = Parser()
    script = '"""Module docstring"""\n'
    parser.parse('root', script)
    assert parser.docstring['root'] == doctest('Module docstring')


# LLM-generated content at query #32
#--------------------------

```
def test_decorators_list_not_empty():
    parser = Parser()
    root = "test_root"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[Name(id="decorator", ctx=Load())])
    parser.api(root, node)
    assert "Decorators" in parser.doc[f"{root}.test_func"]


# LLM-generated content at query #33
#--------------------------

```
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
    assert result == ['Any']

def test_func_ann_with_star_arg():
    p = Parser()
    args = [arg(arg='*', annotation=None)]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['']

def test_func_ann_multiple_args():
    p = Parser()
    args = [
        arg(arg='self', annotation=Name(id='Self', ctx=Load())),
        arg(arg='x', annotation=Name(id='int', ctx=Load())),
        arg(arg='y', annotation=None),
        arg(arg='*', annotation=None),
        arg(arg='z', annotation=Name(id='str', ctx=Load()))
    ]
    result = list(p.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'Any', '', 'str']


# LLM-generated content at query #34
#--------------------------

```python
def test_api_method_creates_function_def_documentation():
    parser = Parser()
    parser.parse('module', 'def example_function(): pass')
    assert 'module.example_function' in parser.doc

def test_api_method_creates_async_function_def_documentation():
    parser = Parser()
    parser.parse('module', 'async def example_async_function(): pass')
    assert 'module.example_async_function' in parser.doc

def test_api_method_creates_class_def_documentation():
    parser = Parser()
    parser.parse('module', 'class ExampleClass: pass')
    assert 'module.ExampleClass' in parser.doc

def test_api_method_includes_decorators_in_documentation():
    parser = Parser()
    parser.parse('module', '@decorator\ndef example_function(): pass')
    assert '@decorator' in parser.doc['module.example_function']

def test_api_method_includes_function_parameters_in_documentation():
    parser = Parser()
    parser.parse('module', 'def example_function(param1, param2): pass')
    assert 'param1' in parser.doc['module.example_function']
    assert 'param2' in parser.doc['module.example_function']

def test_api_method_includes_return_type_in_documentation():
    parser = Parser()
    parser.parse('module', 'def example_function() -> str: pass')
    assert 'return' in parser.doc['module.example_function']

def test_api_method_includes_class_bases_in_documentation():
    parser = Parser()
    parser.parse('module', 'class ExampleClass(BaseClass): pass')
    assert 'BaseClass' in parser.doc['module.ExampleClass']

def test_api_method_includes_class_members_in_documentation():
    parser = Parser()
    parser.parse('module', 'class ExampleClass:\n    member: int = 1')
    assert 'member' in parser.doc['module.ExampleClass']
    assert 'int' in parser.doc['module.ExampleClass']

def test_api_method_handles_nested_classes():
    parser = Parser()
    parser.parse('module', 'class OuterClass:\n    class InnerClass: pass')
    assert 'module.OuterClass.InnerClass' in parser.doc

def test_api_method_handles_nested_functions():
    parser = Parser()
    parser.parse('module', 'def outer_function():\n    def inner_function(): pass')
    assert 'module.outer_function.inner_function' in parser.doc


# LLM-generated content at query #35
#--------------------------

```python
def test_is_magic_name():
    parser = Parser()
    parser.doc = {"__init__": "Initializer", "__str__": "String representation"}
    parser.docstring = {}
    parser.root = {"__init__": "module", "__str__": "module"}
    parser.level = {"__init__": 1, "__str__": 1}
    parser.imp = {}
    parser.const = {}
    parser.toc = True
    result = parser.compile()
    assert "__init__" not in result
    assert "__str__" not in result


# LLM-generated content at query #36
#--------------------------

```python
def test_func_api_with_kwonlyargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg('kwarg', None)], kw_defaults=[None], kwarg=None, defaults=[])
    parser.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '*' in parser.doc['name']


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    parser = Parser()
    assign_node = Assign(targets=[Name(id="test_attr", ctx=Store())], value=Constant(value=42), type_comment=None)
    assert isinstance(assign_node, Assign)
    assert len(assign_node.targets) == 1
    assert isinstance(assign_node.targets[0], Name)


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_true():
    from ast import arguments, arg
    node = arguments(posonlyargs=[arg(arg='posonly')], args=[], defaults=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None)
    parser = Parser()
    parser.func_api(root='root', name='func', node=node, returns=None, has_self=False, cls_method=False)
    assert len(parser.doc['func']) > 0


# LLM-generated content at query #39
#--------------------------

```python
def test_api_without_link():
    parser = Parser(link=False)
    class MockNode:
        name = "test_node"
        decorator_list = []
        body = []
    parser.parse("test_root", "")
    parser.api("test_root", MockNode())
    assert "\n<a id=\"{}\"></a>" not in parser.doc["test_root.test_node"]


# LLM-generated content at query #40
#--------------------------

```
def test_class_api_with_bases_and_members():
    parser = Parser()
    parser.doc = {}
    parser.level = {}
    parser.root = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {}
    parser.docstring = {}
    parser.parse('root', 'class A: pass')
    parser.class_api('root', 'root.A', [], [])
    assert 'root.A' in parser.doc
    assert 'class A' in parser.doc['root.A']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.doc = {}
    parser.level = {}
    parser.root = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {}
    parser.docstring = {}
    parser.parse('root', 'class A(enum.Enum): pass')
    parser.class_api('root', 'root.A', [Name(id='enum.Enum', ctx=Load())], [])
    assert 'root.A' in parser.doc
    assert 'Bases' in parser.doc['root.A']

def test_class_api_with_public_members():
    parser = Parser()
    parser.doc = {}
    parser.level = {}
    parser.root = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {}
    parser.docstring = {}
    parser.parse('root', 'class A: x: int = 1')
    node = ClassDef(name='A', bases=[], keywords=[], body=[AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1)], decorator_list=[])
    parser.class_api('root', 'root.A', [], node.body)
    assert 'root.A' in parser.doc
    assert 'Members' in parser.doc['root.A']

def test_class_api_with_enum_members():
    parser = Parser()
    parser.doc = {}
    parser.level = {}
    parser.root = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {}
    parser.docstring = {}
    parser.parse('root', 'class A(enum.Enum): X = 1')
    node = ClassDef(name='A', bases=[Name(id='enum.Enum', ctx=Load())], keywords=[], body=[Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=1), type_comment=None)], decorator_list=[])
    parser.class_api('root', 'root.A', node.bases, node.body)
    assert 'root.A' in parser.doc
    assert 'Enums' in parser.doc['root.A']

def test_class_api_with_deleted_members():
    parser = Parser()
    parser.doc = {}
    parser.level = {}
    parser.root = {}
    parser.alias = {}
    parser.const = {}
    parser.imp = {}
    parser.docstring = {}
    parser.parse('root', 'class A: x: int = 1; del x')
    node = ClassDef(name='A', bases=[], keywords=[], body=[AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1), Delete(targets=[Name(id='x', ctx=Del())])], decorator_list=[])
    parser.class_api('root', 'root.A', [], node.body)
    assert 'root.A' in parser.doc
    assert 'Members' not in parser.doc['root.A']


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_at_line_23_evaluates_to_false():
    parser = Parser()
    root = "test_module"
    name = "test_class"
    body = [
        Assign(
            targets=[Name(id="test_attr", ctx=Store())],
            value=Constant(value=42),
            type_comment=None
        )
    ]
    bases = ["object"]
    parser.class_api(root, name, bases, body)
    assert "test_attr" not in parser.doc[name]


# LLM-generated content at query #42
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


# LLM-generated content at query #43
#--------------------------

```
def test_class_api_with_bases():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [Name(id='B', ctx=Load())], [])
    assert 'B' in p.doc['test_module.A']

def test_class_api_with_enum_bases():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    p.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], [])
    assert 'enum.Enum' in p.doc['test_module.A']

def test_class_api_with_members():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    body = [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1)]
    p.class_api('test_module', 'test_module.A', [], body)
    assert 'x' in p.doc['test_module.A'] and 'int' in p.doc['test_module.A']

def test_class_api_with_enums():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    body = [Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=1), type_comment=None)]
    p.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], body)
    assert 'X' in p.doc['test_module.A']

def test_class_api_with_deleted_member():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    body = [
        Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=1), type_comment=None),
        Delete(targets=[Name(id='x', ctx=Del())])
    ]
    p.class_api('test_module', 'test_module.A', [], body)
    assert 'x' not in p.doc['test_module.A']

def test_class_api_with_private_member():
    p = Parser()
    p.parse('test_module', 'class A: pass')
    body = [Assign(targets=[Name(id='_x', ctx=Store())], value=Constant(value=1), type_comment=None)]
    p.class_api('test_module', 'test_module.A', [], body)
    assert '_x' not in p.doc['test_module.A']


# LLM-generated content at query #44
#--------------------------

```python
def test_visit_Attribute_with_typing():
    node = Attribute(Name("typing", Load()), "List", Load())
    resolver = Resolver("root", {})
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #45
#--------------------------

```python
def test_func_api_with_vararg():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg(arg='args', annotation=None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    parser.func_api('root', 'func_name', node, None, has_self=False, cls_method=False)
    assert '*args' in parser.doc['func_name']


# LLM-generated content at query #46
#--------------------------

```
def test_func_api_without_self():
    p = Parser()
    args = arguments(
        posonlyargs=[arg(arg='a', annotation=None), arg(arg='b', annotation=None)],
        args=[arg(arg='c', annotation=None), arg(arg='d', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| a | b | c | d | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|:---:|:---:|' in p.doc['name']

def test_func_api_with_self():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='self', annotation=None), arg(arg='a', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=True, cls_method=False)
    assert '| self | a | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|' in p.doc['name']

def test_func_api_with_classmethod():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='cls', annotation=None), arg(arg='a', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert '| cls | a | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|' in p.doc['name']

def test_func_api_with_varargs():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='a', annotation=None)],
        vararg=arg(arg='args', annotation=None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| a | *args | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|' in p.doc['name']

def test_func_api_with_kwargs():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='a', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg(arg='kwargs', annotation=None),
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| a | **kwargs | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|' in p.doc['name']

def test_func_api_with_defaults():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='a', annotation=None), arg(arg='b', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=1), Constant(value=2)]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '| a | b | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|' in p.doc['name']
    assert '| 1 | 2 |  |' in p.doc['name']

def test_func_api_with_annotations():
    p = Parser()
    p.alias = {'root.int': 'int', 'root.str': 'str'}
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='a', annotation=Name(id='int')), arg(arg='b', annotation=Name(id='str'))],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, Name(id='None'), has_self=False, cls_method=False)
    assert '| a | b | return |' in p.doc['name']
    assert '|:---:|:---:|:---:|' in p.doc['name']
    assert '| `int` | `str` | `None` |' in p.doc['name']


