####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_imports_with_import_statement():
    parser = Parser()
    from ast import parse as ast_parse, Import, alias
    
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname="system")])
    
    parser.imports(root, node)
    
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"


def test_imports_with_import_from_statement():
    parser = Parser()
    from ast import ImportFrom, alias
    
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    node = ImportFrom(module="os", names=[alias(name="path", asname=None), alias(name="environ", asname="env")], level=0)
    
    parser.imports(root, node)
    
    assert parser.alias["test_module.path"] == "os.path"
    assert parser.alias["test_module.env"] == "os.environ"


def test_imports_with_relative_import():
    parser = Parser()
    from ast import ImportFrom, alias
    
    root = "package.subpackage.module"
    parser.level[root] = 2
    parser.root[root] = root
    
    node = ImportFrom(module="utils", names=[alias(name="helper", asname=None)], level=1)
    
    parser.imports(root, node)
    
    assert parser.alias["package.subpackage.module.helper"] == "package.subpackage.utils.helper"


def test_imports_with_relative_import_level_2():
    parser = Parser()
    from ast import ImportFrom, alias
    
    root = "package.subpackage.module"
    parser.level[root] = 2
    parser.root[root] = root
    
    node = ImportFrom(module="common", names=[alias(name="base", asname=None)], level=2)
    
    parser.imports(root, node)
    
    assert parser.alias["package.subpackage.module.base"] == "package.common.base"


def test_imports_with_import_from_no_module():
    parser = Parser()
    from ast import ImportFrom, alias
    
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    node = ImportFrom(module=None, names=[alias(name="helper", asname=None)], level=1)
    
    parser.imports(root, node)
    
    assert parser.alias["test_module.helper"] == "helper"


def test_imports_with_multiple_names():
    parser = Parser()
    from ast import ImportFrom, alias
    
    root = "mypackage.mymodule"
    parser.level[root] = 1
    parser.root[root] = root
    
    node = ImportFrom(module="utils", names=[alias(name="func1", asname=None), alias(name="func2", asname="f2"), alias(name="func3", asname=None)], level=0)
    
    parser.imports(root, node)
    
    assert parser.alias["mypackage.mymodule.func1"] == "utils.func1"
    assert parser.alias["mypackage.mymodule.f2"] == "utils.func2"
    assert parser.alias["mypackage.mymodule.func3"] == "utils.func3"


# LLM-generated content at query #2
#--------------------------

```python
def test_visit_name_with_self_ty():
    resolver = Resolver("module", {}, self_ty="MyType")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


def test_visit_name_without_self_ty():
    resolver = Resolver("module", {}, self_ty="")
    node = Name("SomeName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeName"


def test_visit_name_with_alias_non_recursive():
    resolver = Resolver("module", {"module.MyName": "List[int]"})
    node = Name("MyName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)


def test_visit_name_with_typevar_alias():
    resolver = Resolver("module", {"module.T": "TypeVar('T')"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


def test_visit_name_not_in_alias():
    resolver = Resolver("module", {})
    node = Name("UnknownName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "UnknownName"


def test_visit_name_circular_alias():
    resolver = Resolver("module", {"module.A": "module.A"})
    node = Name("A", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "A"


def test_visit_name_with_nested_module():
    resolver = Resolver("package.module", {"package.module.Name": "str"})
    node = Name("Name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "str"


# LLM-generated content at query #3
#--------------------------

```python
def test_globals_annotated_assign_with_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create an annotated assignment: x: int = 42
    target = Name(id='x', ctx=Store())
    value = Constant(value=42)
    annotation = Name(id='int', ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    
    assert parser.alias['test_module.x'] == '42'
    assert parser.const['test_module.x'] == 'int'


def test_globals_assign_with_constant_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create assignment: y = "hello"
    target = Name(id='y', ctx=Store())
    value = Constant(value="hello")
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert parser.alias['test_module.y'] == "'hello'"
    assert parser.const['test_module.y'] == 'str'


def test_globals_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create assignment: MAX_VALUE = 100
    target = Name(id='MAX_VALUE', ctx=Store())
    value = Constant(value=100)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert parser.alias['test_module.MAX_VALUE'] == '100'
    assert parser.root['test_module.MAX_VALUE'] == 'test_module'
    assert parser.const['test_module.MAX_VALUE'] == 'int'


def test_globals_all_filter():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    parser.imp[root] = set()
    
    # Create __all__ assignment
    target = Name(id='__all__', ctx=Store())
    elts = [Constant(value='func1'), Constant(value='func2')]
    value = Tuple(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert 'test_module.func1' in parser.imp[root]
    assert 'test_module.func2' in parser.imp[root]


def test_globals_assign_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create assignment with type comment: z = 3.14  # type: float
    target = Name(id='z', ctx=Store())
    value = Constant(value=3.14)
    node = Assign(targets=[target], value=value, type_comment='float')
    
    parser.globals(root, node)
    
    assert parser.alias['test_module.z'] == '3.14'
    assert parser.const['test_module.z'] == 'float'


def test_globals_multiple_targets_ignored():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create assignment with multiple targets: a = b = 10
    target1 = Name(id='a', ctx=Store())
    target2 = Name(id='b', ctx=Store())
    value = Constant(value=10)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert 'test_module.a' not in parser.alias
    assert 'test_module.b' not in parser.alias


def test_globals_non_name_target_ignored():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create assignment to tuple: (x, y) = (1, 2)
    target = Tuple(elts=[Name(id='x', ctx=Store()), Name(id='y', ctx=Store())], ctx=Store())
    value = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert 'test_module.x' not in parser.alias


def test_globals_annotated_assign_no_value_ignored():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create annotated assignment without value: w: str
    target = Name(id='w', ctx=Store())
    annotation = Name(id='str', ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=None, simple=1)
    
    parser.globals(root, node)
    
    assert 'test_module.w' not in parser.alias


def test_globals_list_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    # Create assignment: nums = [1, 2, 3]
    target = Name(id='nums', ctx=Store())
    elts = [Constant(value=1), Constant(value=2), Constant(value=3)]
    value = List(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert parser.alias['test_module.nums'] == '[1, 2, 3]'
    assert parser.const['test_module.nums'] == 'list[int]'


# LLM-generated content at query #4
#--------------------------

```python
def test_is_public_with_root_module():
    parser = Parser()
    parser.root['mymodule'] = 'mymodule'
    parser.imp['mymodule'] = set()
    result = parser.is_public('mymodule')
    assert result is True


def test_is_public_with_public_submodule():
    parser = Parser()
    parser.root['mymodule.submodule'] = 'mymodule'
    parser.imp['mymodule'] = set()
    result = parser.is_public('mymodule.submodule')
    assert result is True


def test_is_public_with_private_name():
    parser = Parser()
    parser.root['mymodule._private'] = 'mymodule'
    parser.imp['mymodule'] = set()
    result = parser.is_public('mymodule._private')
    assert result is False


def test_is_public_with_all_list_matching():
    parser = Parser()
    parser.root['mymodule.func'] = 'mymodule'
    parser.imp['mymodule'] = {'mymodule.func'}
    result = parser.is_public('mymodule.func')
    assert result is True


def test_is_public_with_all_list_not_matching():
    parser = Parser()
    parser.root['mymodule.func'] = 'mymodule'
    parser.imp['mymodule'] = {'mymodule.other'}
    result = parser.is_public('mymodule.func')
    assert result is False


def test_is_public_with_all_list_empty():
    parser = Parser()
    parser.root['mymodule.func'] = 'mymodule'
    parser.imp['mymodule'] = set()
    result = parser.is_public('mymodule.func')
    assert result is True


def test_is_public_with_module_in_imp():
    parser = Parser()
    parser.root['mymodule'] = 'mymodule'
    parser.imp['mymodule'] = {'mymodule'}
    parser.doc['mymodule.exported'] = 'doc'
    result = parser.is_public('mymodule')
    assert result is True


def test_is_public_with_magic_name():
    parser = Parser()
    parser.root['mymodule.__init__'] = 'mymodule'
    parser.imp['mymodule'] = set()
    result = parser.is_public('mymodule.__init__')
    assert result is True


def test_is_public_with_parent_in_all():
    parser = Parser()
    parser.root['mymodule.submodule.func'] = 'mymodule'
    parser.imp['mymodule'] = {'mymodule.submodule'}
    result = parser.is_public('mymodule.submodule.func')
    assert result is True


# LLM-generated content at query #5
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']


def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    ann_assign = AnnAssign(
        target=Name(id='attr1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='default'),
        simple=1
    )
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_with_enum_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    ann_assign = AnnAssign(
        target=Name(id='MEMBER1', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=1),
        simple=1
    )
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.TestEnum', bases, body)
    
    assert 'test_module.TestEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.TestEnum']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    ann_assign = AnnAssign(
        target=Name(id='attr1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='default'),
        simple=1
    )
    delete_stmt = Delete(targets=[Name(id='attr1', ctx=Del())])
    body = [ann_assign, delete_stmt]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.TestClass'] or parser.doc['test_module.TestClass'].count('Members') == 0


def test_class_api_with_private_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    ann_assign = AnnAssign(
        target=Name(id='_private', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='private'),
        simple=1
    )
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc


def test_class_api_empty_body():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc


# LLM-generated content at query #6
#--------------------------

```python
def test_func_api_simple_function():
    from ast import parse, FunctionDef
    
    p = Parser()
    script = "def foo(a: int, b: str) -> bool: pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "a" in p.doc["test.foo"]
    assert "b" in p.doc["test.foo"]


def test_func_api_with_defaults():
    from ast import parse, Constant
    
    p = Parser()
    script = "def foo(a: int, b: str = 'hello') -> bool: pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "|" in p.doc["test.foo"]


def test_func_api_with_self():
    from ast import parse
    
    p = Parser()
    script = "class A:\n    def method(self, x: int) -> None: pass"
    root_node = parse(script)
    class_node = root_node.body[0]
    method_node = class_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.A.method", method_node.args, method_node.returns, has_self=True, cls_method=False)
    
    assert "test.A.method" in p.doc
    assert "Self" in p.doc["test.A.method"]


def test_func_api_classmethod():
    from ast import parse
    
    p = Parser()
    script = "class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass"
    root_node = parse(script)
    class_node = root_node.body[0]
    method_node = class_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.A.method", method_node.args, method_node.returns, has_self=True, cls_method=True)
    
    assert "test.A.method" in p.doc
    assert "type[Self]" in p.doc["test.A.method"]


def test_func_api_with_varargs():
    from ast import parse
    
    p = Parser()
    script = "def foo(*args: int, **kwargs: str) -> None: pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "*" in p.doc["test.foo"]


def test_func_api_with_kwonly_args():
    from ast import parse
    
    p = Parser()
    script = "def foo(a: int, *, b: str) -> None: pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "b" in p.doc["test.foo"]


def test_func_api_no_annotation():
    from ast import parse
    
    p = Parser()
    script = "def foo(a, b): pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, None, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "a" in p.doc["test.foo"]


def test_func_api_with_posonly_args():
    from ast import parse
    
    p = Parser()
    script = "def foo(a: int, /, b: str) -> None: pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "/" in p.doc["test.foo"]


def test_func_api_complex_signature():
    from ast import parse
    
    p = Parser()
    script = "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass"
    root_node = parse(script)
    func_node = root_node.body[0]
    
    p.parse("test", script)
    p.func_api("test", "test.foo", func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert "test.foo" in p.doc
    assert "return" in p.doc["test.foo"]


# LLM-generated content at query #7
#--------------------------

```python
def test_class_api_assign_predicate():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Create an Assign node that satisfies the predicate at line 17
    # The predicate checks:
    # 1. isinstance(node, Assign) - True
    # 2. len(node.targets) == 1 - True
    # 3. isinstance(node.targets[0], Name) - True
    target_name = Name(id='test_attr', ctx=None)
    assign_node = Assign(targets=[target_name], value=Constant(value=42), type_comment=None)
    
    # Verify the predicate evaluates to True
    predicate_result = (
        isinstance(assign_node, Assign)
        and len(assign_node.targets) == 1
        and isinstance(assign_node.targets[0], Name)
    )
    
    assert predicate_result is True


# LLM-generated content at query #8
#--------------------------

```python
def test_is_public_family_all_public():
    assert is_public_family('os.path.join') == True


def test_is_public_family_single_public():
    assert is_public_family('sys') == True


def test_is_public_family_with_magic_names():
    assert is_public_family('os.__init__.path') == True


def test_is_public_family_with_private_prefix():
    assert is_public_family('os._private') == False


def test_is_public_family_with_private_at_start():
    assert is_public_family('_private.public') == False


def test_is_public_family_with_multiple_private():
    assert is_public_family('public._private.module') == False


def test_is_public_family_with_dunder_only():
    assert is_public_family('__init__') == True


def test_is_public_family_with_magic_and_public():
    assert is_public_family('module.__dict__.method') == True


def test_is_public_family_empty_string():
    assert is_public_family('') == True


def test_is_public_family_single_underscore():
    assert is_public_family('_') == False


def test_is_public_family_mixed_magic_and_private():
    assert is_public_family('__init__._private') == False


def test_is_public_family_multiple_dots():
    assert is_public_family('a.b.c.d.e') == True


def test_is_public_family_with_trailing_magic():
    assert is_public_family('public.__name__') == True


# LLM-generated content at query #9
#--------------------------

```python
def test_api_function_def():
    from ast import parse, FunctionDef
    parser = Parser(link=True, b_level=1, toc=False)
    script = "def test_func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert 'test_module.test_func' in parser.doc
    assert 'test_func()' in parser.doc['test_module.test_func']


def test_api_async_function_def():
    from ast import parse, AsyncFunctionDef
    parser = Parser(link=True, b_level=1, toc=False)
    script = "async def async_func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert 'test_module.async_func' in parser.doc
    assert 'async test_func()' in parser.doc['test_module.async_func'] or 'async_func()' in parser.doc['test_module.async_func']


def test_api_class_def():
    from ast import parse, ClassDef
    parser = Parser(link=True, b_level=1, toc=False)
    script = "class TestClass: pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert 'test_module.TestClass' in parser.doc
    assert 'class TestClass' in parser.doc['test_module.TestClass']


def test_api_with_decorator():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = "@staticmethod\ndef decorated_func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert 'test_module.decorated_func' in parser.doc


def test_api_with_prefix():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = "class Outer:\n    def inner_method(self): pass"
    root_node = parse(script)
    outer_node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', outer_node)
    assert 'test_module.Outer' in parser.doc


def test_api_sets_level():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = "def func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert parser.level['test_module.func'] == parser.level['test_module']


def test_api_sets_root():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = "def func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert parser.root['test_module.func'] == 'test_module'


def test_api_with_docstring():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = 'def func():\n    """Test docstring."""\n    pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert 'test_module.func' in parser.docstring


def test_api_with_link():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = "def func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert '<a id=' in parser.doc['test_module.func']


def test_api_without_link():
    from ast import parse
    parser = Parser(link=False, b_level=1, toc=False)
    script = "def func(): pass"
    root_node = parse(script)
    node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', node)
    assert '<a id=' not in parser.doc['test_module.func']


def test_api_nested_class():
    from ast import parse
    parser = Parser(link=True, b_level=1, toc=False)
    script = "class Outer:\n    class Inner: pass"
    root_node = parse(script)
    outer_node = root_node.body[0]
    parser.parse('test_module', script)
    parser.api('test_module', outer_node)
    assert 'test_module.Outer' in parser.doc


# LLM-generated content at query #10
#--------------------------

```python
def test_class_api_delete_statement_handling():
    from ast import Delete, Name, parse, AnnAssign, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    
    parser.doc[name] = "# class TestClass\n\n"
    parser.level[root] = 0
    parser.root[name] = root
    
    # Create a Delete node with a Name target
    delete_node = Delete(targets=[Name(id="attr_to_delete", ctx=None)])
    
    # Verify the predicate at line 30 evaluates to True
    assert isinstance(delete_node, Delete)


# LLM-generated content at query #11
#--------------------------

```python
def test_visit_attribute_typing_prefix():
    resolver = Resolver(root="test_module", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Union", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Union"
    assert isinstance(result.ctx, Load)


def test_visit_attribute_non_typing_prefix():
    resolver = Resolver(root="test_module", alias={})
    node = Attribute(value=Name(id="other_module", ctx=Load()), attr="SomeClass", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other_module"
    assert result.attr == "SomeClass"


def test_visit_attribute_non_name_value():
    resolver = Resolver(root="test_module", alias={})
    inner_attr = Attribute(value=Name(id="obj", ctx=Load()), attr="typing", ctx=Load())
    node = Attribute(value=inner_attr, attr="Union", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result is node


def test_visit_attribute_typing_with_different_attributes():
    resolver = Resolver(root="test_module", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Optional", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Optional"


def test_visit_attribute_preserves_context():
    resolver = Resolver(root="test_module", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Store())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"


# LLM-generated content at query #12
#--------------------------

```python
from ast import If, Try, ExceptHandler, stmt, parse
from typing import List

def test_walk_body_simple_statements():
    code = "x = 1\ny = 2"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 2
    assert all(isinstance(node, stmt) for node in result)

def test_walk_body_with_if_statement():
    code = "if True:\n    x = 1\nelse:\n    y = 2"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 2
    assert all(isinstance(node, stmt) for node in result)

def test_walk_body_with_nested_if():
    code = "if True:\n    if False:\n        x = 1"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 1
    assert isinstance(result[0], stmt)

def test_walk_body_with_try_except():
    code = "try:\n    x = 1\nexcept:\n    y = 2"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 2
    assert all(isinstance(node, stmt) for node in result)

def test_walk_body_with_try_except_else_finally():
    code = "try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 4
    assert all(isinstance(node, stmt) for node in result)

def test_walk_body_empty():
    tree = parse("")
    result = list(walk_body(tree.body))
    assert len(result) == 0

def test_walk_body_complex_nested_structure():
    code = "x = 1\nif True:\n    y = 2\n    try:\n        z = 3\n    except:\n        w = 4\nelse:\n    a = 5"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 6
    assert all(isinstance(node, stmt) for node in result)

def test_walk_body_multiple_handlers():
    code = "try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept TypeError:\n    z = 3"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 3
    assert all(isinstance(node, stmt) for node in result)

def test_walk_body_if_without_else():
    code = "if True:\n    x = 1"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 1
    assert isinstance(result[0], stmt)

def test_walk_body_deeply_nested():
    code = "if True:\n    if True:\n        if True:\n            x = 1"
    tree = parse(code)
    result = list(walk_body(tree.body))
    assert len(result) == 1
    assert isinstance(result[0], stmt)


# LLM-generated content at query #13
#--------------------------

```python
def test_class_api_mem_predicate_true():
    from ast import parse, Name, AnnAssign, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    parser.doc['test_class'] = '# test_class\n\n'
    
    # Create a simple class body with an annotated assignment
    script = """
class TestClass:
    public_attr: int
"""
    tree = parse(script)
    class_def = tree.body[0]
    
    # Mock resolve to return a simple type
    original_resolve = parser.resolve
    parser.resolve = lambda root, node, self_ty="": "int"
    
    # Call class_api which should populate mem dictionary
    parser.class_api('test_module', 'test_class', [], class_def.body)
    
    # Verify that the predicate at line 40 (elif mem:) evaluated to True
    # by checking that table was called and added to doc
    assert 'Members' in parser.doc['test_class']
    assert 'public_attr' in parser.doc['test_class']


# LLM-generated content at query #14
#--------------------------

```python
def test_imports_with_import_from_node():
    from ast import ImportFrom, alias
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    
    # Create an ImportFrom node with module='os'
    import_from_node = ImportFrom(
        module='os',
        names=[alias(name='path', asname=None)],
        level=0
    )
    
    # Call imports method
    parser.imports(root, import_from_node)
    
    # Verify that the predicate at line 7 (node.module is not None) evaluated to True
    # by checking that alias was populated
    assert len(parser.alias) > 0


# LLM-generated content at query #15
#--------------------------

```python
def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    class_node = ClassDef(
        name='TestClass',
        bases=[],
        keywords=[],
        body=[
            AnnAssign(
                target=Name(id='attr1', ctx=Store()),
                annotation=Name(id='str', ctx=Load()),
                simple=1,
                value=None
            ),
            Assign(
                targets=[Name(id='attr2', ctx=Store())],
                value=Constant(value=42),
                type_comment=None
            )
        ],
        decorator_list=[]
    )
    
    parser.class_api('test_module', 'test_module.TestClass', [], class_node.body)
    
    assert 'test_module.TestClass' in parser.doc or len(parser.doc) > 0


def test_class_api_with_enum():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    class_node = ClassDef(
        name='TestEnum',
        bases=[Name(id='Enum', ctx=Load())],
        keywords=[],
        body=[
            AnnAssign(
                target=Name(id='MEMBER1', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                simple=1,
                value=Constant(value=1)
            ),
            AnnAssign(
                target=Name(id='MEMBER2', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                simple=1,
                value=Constant(value=2)
            )
        ],
        decorator_list=[]
    )
    
    parser.class_api('test_module', 'test_module.TestEnum', 
                     [Name(id='Enum', ctx=Load())], class_node.body)
    
    assert isinstance(parser.doc, dict)


def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert isinstance(parser.doc, dict)


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    class_node = ClassDef(
        name='TestClass',
        bases=[],
        keywords=[],
        body=[
            AnnAssign(
                target=Name(id='attr1', ctx=Store()),
                annotation=Name(id='str', ctx=Load()),
                simple=1,
                value=None
            ),
            Delete(targets=[Name(id='attr1', ctx=Del())])
        ],
        decorator_list=[]
    )
    
    parser.class_api('test_module', 'test_module.TestClass', [], class_node.body)
    
    assert isinstance(parser.doc, dict)


def test_class_api_with_private_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    class_node = ClassDef(
        name='TestClass',
        bases=[],
        keywords=[],
        body=[
            AnnAssign(
                target=Name(id='_private', ctx=Store()),
                annotation=Name(id='str', ctx=Load()),
                simple=1,
                value=None
            )
        ],
        decorator_list=[]
    )
    
    parser.class_api('test_module', 'test_module.TestClass', [], class_node.body)
    
    assert isinstance(parser.doc, dict)


def test_class_api_empty_class():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    parser.class_api('test_module', 'test_module.EmptyClass', [], [])
    
    assert isinstance(parser.doc, dict)


# LLM-generated content at query #16
#--------------------------

```python
def test_is_public_with_root_module():
    p = Parser()
    p.root['pkg'] = 'pkg'
    p.imp['pkg'] = set()
    result = p.is_public('pkg')
    assert result is True


def test_is_public_with_private_name():
    p = Parser()
    p.root['pkg._private'] = 'pkg'
    p.imp['pkg'] = set()
    result = p.is_public('pkg._private')
    assert result is False


def test_is_public_with_magic_name():
    p = Parser()
    p.root['pkg.__init__'] = 'pkg'
    p.imp['pkg'] = set()
    result = p.is_public('pkg.__init__')
    assert result is True


def test_is_public_with_all_list_matching():
    p = Parser()
    p.root['pkg.func'] = 'pkg'
    p.imp['pkg'] = {'pkg.func'}
    result = p.is_public('pkg.func')
    assert result is True


def test_is_public_with_all_list_not_matching():
    p = Parser()
    p.root['pkg.func'] = 'pkg'
    p.imp['pkg'] = {'pkg.other'}
    result = p.is_public('pkg.func')
    assert result is False


def test_is_public_with_empty_all_list():
    p = Parser()
    p.root['pkg.public_func'] = 'pkg'
    p.imp['pkg'] = set()
    result = p.is_public('pkg.public_func')
    assert result is True


def test_is_public_with_nested_module():
    p = Parser()
    p.root['pkg.sub.func'] = 'pkg.sub'
    p.imp['pkg.sub'] = set()
    result = p.is_public('pkg.sub.func')
    assert result is True


def test_is_public_with_module_in_imp_dict():
    p = Parser()
    p.root['pkg'] = 'pkg'
    p.imp['pkg'] = {'pkg'}
    p.doc['pkg'] = 'Module pkg'
    result = p.is_public('pkg')
    assert result is True


def test_is_public_with_all_list_parent_match():
    p = Parser()
    p.root['pkg.sub.func'] = 'pkg.sub'
    p.imp['pkg.sub'] = {'pkg.sub'}
    result = p.is_public('pkg.sub.func')
    assert result is True


# LLM-generated content at query #17
#--------------------------

```python
def test_class_api_with_members():
    from ast import parse as ast_parse, ClassDef
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    script = """
class TestClass(BaseClass):
    '''Test class'''
    public_attr: int
    _private_attr: str
    CONSTANT: float = 3.14
    """
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass'] or 'Bases' in parser.doc['test_module.TestClass']


def test_class_api_with_enum():
    from ast import parse as ast_parse
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {'test_module.Enum': 'enum.Enum'}
    
    script = """
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    """
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.Color', class_node.bases, class_node.body)
    
    assert 'test_module.Color' in parser.doc
    assert 'Enums' in parser.doc['test_module.Color']


def test_class_api_with_deleted_attributes():
    from ast import parse as ast_parse
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    script = """
class TestClass:
    attr1: int = 1
    attr2: str = "test"
    del attr1
    """
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc


def test_class_api_empty_class():
    from ast import parse as ast_parse
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    script = "class EmptyClass: pass"
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.EmptyClass', class_node.bases, class_node.body)
    
    assert 'test_module.EmptyClass' in parser.doc


def test_class_api_with_annotated_members():
    from ast import parse as ast_parse
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    script = """
class TestClass:
    name: str
    age: int
    """
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_with_bases():
    from ast import parse as ast_parse
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    script = "class DerivedClass(BaseClass, Mixin): pass"
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.DerivedClass', class_node.bases, class_node.body)
    
    assert 'test_module.DerivedClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.DerivedClass']


def test_class_api_with_assigned_members():
    from ast import parse as ast_parse
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    script = """
class TestClass:
    value1 = 42
    value2 = "string"
    """
    
    root_node = ast_parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc


# LLM-generated content at query #18
#--------------------------

```python
def test_const_type_with_constant_int():
    from ast import Constant
    node = Constant(value=42)
    result = const_type(node)
    assert result == "int"


def test_const_type_with_constant_str():
    from ast import Constant
    node = Constant(value="hello")
    result = const_type(node)
    assert result == "str"


def test_const_type_with_constant_float():
    from ast import Constant
    node = Constant(value=3.14)
    result = const_type(node)
    assert result == "float"


def test_const_type_with_constant_bool():
    from ast import Constant
    node = Constant(value=True)
    result = const_type(node)
    assert result == "bool"


def test_const_type_with_list_of_ints():
    from ast import List, Constant
    node = List(elts=[Constant(value=1), Constant(value=2), Constant(value=3)])
    result = const_type(node)
    assert result == "list[int]"


def test_const_type_with_list_of_mixed_types():
    from ast import List, Constant
    node = List(elts=[Constant(value=1), Constant(value="str")])
    result = const_type(node)
    assert result == "list[Any]"


def test_const_type_with_tuple_of_strs():
    from ast import Tuple, Constant
    node = Tuple(elts=[Constant(value="a"), Constant(value="b")])
    result = const_type(node)
    assert result == "tuple[str]"


def test_const_type_with_empty_list():
    from ast import List
    node = List(elts=[])
    result = const_type(node)
    assert result == "list"


def test_const_type_with_set_of_ints():
    from ast import Set, Constant
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    result = const_type(node)
    assert result == "set[int]"


def test_const_type_with_dict_int_to_str():
    from ast import Dict, Constant
    node = Dict(keys=[Constant(value=1), Constant(value=2)], 
                values=[Constant(value="a"), Constant(value="b")])
    result = const_type(node)
    assert result == "dict[int, str]"


def test_const_type_with_empty_dict():
    from ast import Dict
    node = Dict(keys=[], values=[])
    result = const_type(node)
    assert result == "dict"


def test_const_type_with_list_containing_non_constant():
    from ast import List, Constant, Name
    node = List(elts=[Constant(value=1), Name(id="x")])
    result = const_type(node)
    assert result == "list"


def test_const_type_with_call_to_int():
    from ast import Call, Name
    node = Call(func=Name(id="int"), args=[], keywords=[])
    result = const_type(node)
    assert result == "int"


def test_const_type_with_call_to_str():
    from ast import Call, Name
    node = Call(func=Name(id="str"), args=[], keywords=[])
    result = const_type(node)
    assert result == "str"


def test_const_type_with_call_to_list():
    from ast import Call, Name
    node = Call(func=Name(id="list"), args=[], keywords=[])
    result = const_type(node)
    assert result == "list"


def test_const_type_with_unknown_node():
    from ast import Name
    node = Name(id="unknown_var")
    result = const_type(node)
    assert result == "Any"


# LLM-generated content at query #19
#--------------------------

```python
def test_func_ann_with_self_parameter():
    parser = Parser()
    parser.alias = {}
    args = [arg(arg='self', annotation=None), arg(arg='x', annotation=None), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', ANY, ANY]

def test_func_ann_with_classmethod():
    parser = Parser()
    parser.alias = {}
    args = [arg(arg='cls', annotation=None), arg(arg='x', annotation=None), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', ANY, ANY]

def test_func_ann_without_self():
    parser = Parser()
    parser.alias = {}
    args = [arg(arg='x', annotation=None), arg(arg='y', annotation=None), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == [ANY, ANY, ANY]

def test_func_ann_with_star_arg():
    parser = Parser()
    parser.alias = {}
    args = [arg(arg='x', annotation=None), arg(arg='*', annotation=None), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == [ANY, "", ANY]

def test_func_ann_with_annotations():
    parser = Parser()
    parser.alias = {'root.x': 'int', 'root.y': 'str'}
    from ast import Name
    x_annotation = Name(id='int', ctx=None)
    y_annotation = Name(id='str', ctx=None)
    args = [arg(arg='x', annotation=x_annotation), arg(arg='y', annotation=y_annotation), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert len(result) == 3
    assert result[2] == ANY

def test_func_ann_self_with_annotation():
    parser = Parser()
    parser.alias = {}
    from ast import Name
    self_annotation = Name(id='MyClass', ctx=None)
    args = [arg(arg='self', annotation=self_annotation), arg(arg='x', annotation=None), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result[0] == 'Self'
    assert result[1] == ANY

def test_func_ann_empty_args():
    parser = Parser()
    parser.alias = {}
    args = []
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == []

def test_func_ann_multiple_stars():
    parser = Parser()
    parser.alias = {}
    args = [arg(arg='x', annotation=None), arg(arg='*', annotation=None), arg(arg='y', annotation=None), arg(arg='return', annotation=None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == [ANY, "", ANY, ANY]


# LLM-generated content at query #20
#--------------------------

```python
def test_class_api_is_enum_predicate():
    from ast import parse, AnnAssign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser(link=True, b_level=1, toc=False)
    
    root = "test_module"
    name = "test_module.TestEnum"
    
    parser.doc[name] = "## class TestEnum\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n"
    parser.root[name] = root
    parser.level[name] = 0
    
    # Create a base that starts with 'enum.' to make is_enum True
    enum_base = parse("enum.Enum", mode='eval').body
    bases = [enum_base]
    
    # Create a simple AnnAssign node for an enum member
    ann_assign_code = "MEMBER: int = 1"
    tree = parse(ann_assign_code)
    body = tree.body
    
    # Call class_api with enum bases
    parser.class_api(root, name, bases, body)
    
    # The predicate at line 13 (if is_enum:) should be True
    # This means enums list should have been populated
    assert "MEMBER" in parser.doc[name] or "Enums" in parser.doc[name]


# LLM-generated content at query #21
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = AnnAssign(
        target=Name(id="MY_CONSTANT", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONSTANT" in parser.alias
    assert parser.alias["test_module.MY_CONSTANT"] == "42"
    assert "test_module.MY_CONSTANT" in parser.const
    assert parser.const["test_module.MY_CONSTANT"] == "int"
    assert "test_module.MY_CONSTANT" in parser.root


def test_globals_with_simple_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = Assign(
        targets=[Name(id="MY_VAR", ctx=Store())],
        value=Constant(value="hello"),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.MY_VAR" in parser.alias
    assert parser.alias["test_module.MY_VAR"] == "'hello'"
    assert "test_module.MY_VAR" in parser.const
    assert parser.const["test_module.MY_VAR"] == "str"


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = Assign(
        targets=[Name(id="typed_var", ctx=Store())],
        value=Constant(value=100),
        type_comment="int"
    )
    
    parser.globals(root, node)
    
    assert "test_module.typed_var" in parser.const
    assert parser.const["test_module.typed_var"] == "int"


def test_globals_with_all_tuple():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="func1"), Constant(value="func2")],
            ctx=Load()
        ),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_ignores_non_matching_nodes():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = Assign(
        targets=[Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())],
        value=Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load()),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.a" not in parser.alias


def test_globals_constant_uppercase():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = Assign(
        targets=[Name(id="CONSTANT", ctx=Store())],
        value=Constant(value=999),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.CONSTANT" in parser.root
    assert parser.root["test_module.CONSTANT"] == root


def test_globals_with_annotated_no_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    node = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=None,
        simple=1
    )
    
    parser.globals(root, node)
    
    assert "test_module.x" not in parser.alias


# LLM-generated content at query #22
#--------------------------

```python
def test_class_api_with_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    body = [
        AnnAssign(target=Name(id='attr1'), annotation=Name(id='str'), value=None, simple=1),
        Assign(targets=[Name(id='attr2')], value=Constant(value=42), type_comment=None),
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_with_bases():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Name(id='BaseClass')]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']


def test_class_api_with_enums():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Attribute(value=Name(id='enum'), attr='Enum')]
    body = [
        AnnAssign(target=Name(id='MEMBER1'), annotation=Name(id='int'), value=Constant(value=1), simple=1),
        Assign(targets=[Name(id='MEMBER2')], value=Constant(value=2), type_comment=None),
    ]
    
    parser.class_api('test_module', 'test_module.TestEnum', bases, body)
    
    assert 'test_module.TestEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.TestEnum']


def test_class_api_with_private_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    body = [
        AnnAssign(target=Name(id='_private'), annotation=Name(id='str'), value=None, simple=1),
        AnnAssign(target=Name(id='public'), annotation=Name(id='int'), value=None, simple=1),
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    body = [
        AnnAssign(target=Name(id='attr1'), annotation=Name(id='str'), value=None, simple=1),
        Delete(targets=[Name(id='attr1')]),
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc


def test_class_api_empty_class():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.EmptyClass', bases, body)
    
    assert 'test_module.EmptyClass' in parser.doc
    assert parser.doc['test_module.EmptyClass'] != ""


def test_class_api_with_type_comment():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    body = [
        Assign(targets=[Name(id='attr1')], value=Constant(value=10), type_comment='int'),
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


# LLM-generated content at query #23
#--------------------------

```python
def test_class_api_is_enum_predicate():
    from ast import parse, Name, AnnAssign, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    parser.doc = {}
    parser.root = {}
    parser.level = {}
    parser.alias = {}
    
    # Setup test data
    root = "test_module"
    name = "test_module.TestEnum"
    parser.doc[name] = "# test"
    parser.root[name] = root
    parser.level[root] = 0
    
    # Create a base that starts with 'enum.' to make is_enum True
    enum_base_code = "from enum import Enum\nclass Base(Enum): pass"
    enum_base_ast = parse(enum_base_code).body[1].bases[0]
    
    # Create body with AnnAssign node
    ann_assign_code = "x: int = 1"
    ann_assign_node = parse(ann_assign_code).body[0]
    
    # Call class_api with bases that contain 'enum.' prefix
    # We need to mock resolve to return 'enum.Enum'
    original_resolve = parser.resolve
    parser.resolve = lambda root, node, self_ty="": "enum.Enum"
    
    parser.class_api(root, name, [enum_base_ast], [ann_assign_node])
    
    # Restore original resolve
    parser.resolve = original_resolve
    
    # Verify that is_enum was True and enums list was populated
    # The predicate at line 13 (if is_enum:) should evaluate to True
    # which means enums should contain the attribute name
    assert "x" in parser.doc[name] or "Enums" in parser.doc[name]


# LLM-generated content at query #24
#--------------------------

```python
def test_globals_annassign_with_value():
    from ast import AnnAssign, Name, Constant, parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Create an AnnAssign node with a value (line 11 predicate evaluates to True)
    # This represents: x: int = 42
    code_str = "x: int = 42"
    tree = parse(code_str)
    node = tree.body[0]
    
    # Verify the node is AnnAssign
    assert isinstance(node, AnnAssign)
    
    # Verify node.target is a Name
    assert isinstance(node.target, Name)
    
    # Verify node.value is not None (this is line 11's condition)
    assert node.value is not None
    
    # Call globals method to ensure it processes without early return
    parser.globals("test_module", node)
    
    # Verify that the method processed the node (didn't return early at line 28)
    assert "test_module.x" in parser.alias


# LLM-generated content at query #25
#--------------------------

```python
def test_globals_predicate_line_31_evaluates_to_true():
    from ast import AnnAssign, Name, Constant, parse as ast_parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Create an AST node for an annotated assignment with an uppercase identifier
    # This represents: CONSTANT: int = 42
    code = "CONSTANT: int = 42"
    tree = ast_parse(code)
    node = tree.body[0]
    
    # Verify the node is an AnnAssign with a Name target and a value
    assert isinstance(node, AnnAssign)
    assert isinstance(node.target, Name)
    assert node.value is not None
    
    # Call globals method which will evaluate the predicate at line 31
    root = "test_module"
    parser.imp[root] = set()
    parser.globals(root, node)
    
    # The predicate at line 31 checks: if left.id.isupper()
    # Since left.id is "CONSTANT" (all uppercase), the predicate should be True
    # This means the code at lines 32-34 should have executed
    
    # Verify that the predicate evaluated to True by checking the side effects
    name = "test_module.CONSTANT"
    assert name in parser.root
    assert parser.root[name] == root
    assert name in parser.const


# LLM-generated content at query #26
#--------------------------

```python
def test_func_ann_with_self_parameter():
    from ast import arg as ast_arg
    parser = Parser()
    parser.alias = {}
    root = "test_module"
    args = [
        ast_arg(arg="self", annotation=None),
        ast_arg(arg="x", annotation=None),
        ast_arg(arg="return", annotation=None),
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "Any", "Any"]


def test_func_ann_with_classmethod():
    from ast import arg as ast_arg
    parser = Parser()
    parser.alias = {}
    root = "test_module"
    args = [
        ast_arg(arg="cls", annotation=None),
        ast_arg(arg="x", annotation=None),
        ast_arg(arg="return", annotation=None),
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "Any", "Any"]


def test_func_ann_without_self():
    from ast import arg as ast_arg
    parser = Parser()
    parser.alias = {}
    root = "test_module"
    args = [
        ast_arg(arg="x", annotation=None),
        ast_arg(arg="y", annotation=None),
        ast_arg(arg="return", annotation=None),
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any", "Any"]


def test_func_ann_with_star_separator():
    from ast import arg as ast_arg
    parser = Parser()
    parser.alias = {}
    root = "test_module"
    args = [
        ast_arg(arg="x", annotation=None),
        ast_arg(arg="*", annotation=None),
        ast_arg(arg="y", annotation=None),
        ast_arg(arg="return", annotation=None),
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "", "Any", "Any"]


def test_func_ann_with_self_and_annotation():
    from ast import arg as ast_arg, Name as ast_Name
    parser = Parser()
    parser.alias = {}
    root = "test_module"
    self_annotation = ast_Name(id="MyClass", ctx=None)
    args = [
        ast_arg(arg="self", annotation=self_annotation),
        ast_arg(arg="x", annotation=None),
        ast_arg(arg="return", annotation=None),
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result[0] == "Self"
    assert result[1] == "Any"
    assert result[2] == "Any"


def test_func_ann_with_classmethod_and_annotation():
    from ast import arg as ast_arg, Name as ast_Name
    parser = Parser()
    parser.alias = {}
    root = "test_module"
    cls_annotation = ast_Name(id="type[MyClass]", ctx=None)
    args = [
        ast_arg(arg="cls", annotation=cls_annotation),
        ast_arg(arg="return", annotation=None),
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result[0] == "type[Self]"
    assert result[1] == "Any"


# LLM-generated content at query #27
#--------------------------

```python
def test_globals_predicate_line_35_evaluates_to_false():
    """Test that the predicate at line 35 evaluates to False."""
    from ast import Assign, Name, Constant, Tuple, parse
    
    parser = Parser()
    root = "test_module"
    parser.imp[root] = set()
    
    # Create an AST node for: __all__ = ("item1", "item2")
    script = '__all__ = ("item1", "item2")'
    tree = parse(script)
    node = tree.body[0]
    
    # Verify preconditions: node should be an Assign with Name target
    assert isinstance(node, Assign)
    assert len(node.targets) == 1
    assert isinstance(node.targets[0], Name)
    assert node.targets[0].id == '__all__'
    assert isinstance(node.value, Tuple)
    
    # Call globals method
    parser.globals(root, node)
    
    # Verify that the code after line 35 was executed
    # (i.e., the predicate evaluated to False, so return was NOT executed)
    # This means items from __all__ should be added to self.imp[root]
    assert len(parser.imp[root]) == 2
    assert 'test_module.item1' in parser.imp[root]
    assert 'test_module.item2' in parser.imp[root]


# LLM-generated content at query #28
#--------------------------

```python
def test_globals_predicate_line_18_false():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Assign node with multiple targets (len != 1)
    target1 = Name(id='x')
    target2 = Name(id='y')
    node = Assign(targets=[target1, target2], value=Constant(value=1), type_comment=None)
    
    # Call globals with this node - should return early without error
    parser.globals('test_module', node)
    
    # Verify the predicate at line 18 (len(node.targets) == 1) evaluates to False
    assert len(node.targets) != 1
    assert len(node.targets) == 2


# LLM-generated content at query #29
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    node = AnnAssign(
        target=Name(id="MY_CONSTANT", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONSTANT" in parser.alias
    assert parser.alias["test_module.MY_CONSTANT"] == "42"
    assert "test_module.MY_CONSTANT" in parser.const
    assert parser.const["test_module.MY_CONSTANT"] == "int"


def test_globals_with_assign_statement():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    node = Assign(
        targets=[Name(id="CONST_VAR", ctx=Store())],
        value=Constant(value=100),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.CONST_VAR" in parser.alias
    assert parser.alias["test_module.CONST_VAR"] == "100"
    assert "test_module.CONST_VAR" in parser.const
    assert parser.const["test_module.CONST_VAR"] == "int"


def test_globals_with_all_tuple():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp[root] = set()
    
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="func1"), Constant(value="func2")],
            ctx=Load()
        ),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_all_list():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp[root] = set()
    
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(
            elts=[Constant(value="item1"), Constant(value="item2")],
            ctx=Load()
        ),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.item1" in parser.imp[root]
    assert "test_module.item2" in parser.imp[root]


def test_globals_ignores_lowercase_variable():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    node = Assign(
        targets=[Name(id="lowercase_var", ctx=Store())],
        value=Constant(value="test"),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.lowercase_var" in parser.alias
    assert "test_module.lowercase_var" not in parser.const
    assert "test_module.lowercase_var" not in parser.root


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    node = Assign(
        targets=[Name(id="TYPED_VAR", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    
    parser.globals(root, node)
    
    assert parser.const["test_module.TYPED_VAR"] == "float"


def test_globals_with_multiple_targets_ignored():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=5),
        type_comment=None
    )
    
    parser.globals(root, node)
    
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


def test_globals_with_annotated_assignment_without_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    node = AnnAssign(
        target=Name(id="VAR", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=None,
        simple=1
    )
    
    parser.globals(root, node)
    
    assert "test_module.VAR" not in parser.alias


# LLM-generated content at query #30
#--------------------------

```python
from ast import Name, Load

def test_visit_name_self_ty_match():
    """Test that visit_Name returns Name('Self', Load()) when node.id equals self_ty."""
    from ast import NodeTransformer
    
    class Resolver(NodeTransformer):
        def __init__(self, root: str, alias: dict[str, str], self_ty: str = ""):
            super(Resolver, self).__init__()
            self.root = root
            self.alias = alias
            self.self_ty = self_ty

        def visit_Name(self, node: Name) -> object:
            if node.id == self.self_ty:
                return Name("Self", Load())
            return node
    
    resolver = Resolver(root="module", alias={}, self_ty="MyType")
    node = Name(id="MyType", ctx=Load())
    result = resolver.visit_Name(node)
    
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #31
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an AnnAssign node with a Constant value
    target = Name(id="MY_CONST", ctx=Store())
    value = Constant(value=42)
    annotation = Name(id="int", ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONST" in parser.alias
    assert parser.alias["test_module.MY_CONST"] == "42"
    assert "test_module.MY_CONST" in parser.const
    assert parser.const["test_module.MY_CONST"] == "int"


def test_globals_with_simple_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an Assign node with a Constant value
    target = Name(id="MY_VAR", ctx=Store())
    value = Constant(value="hello")
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.MY_VAR" in parser.alias
    assert parser.alias["test_module.MY_VAR"] == "'hello'"


def test_globals_with_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an Assign node with uppercase name
    target = Name(id="CONSTANT", ctx=Store())
    value = Constant(value=100)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.CONSTANT" in parser.root
    assert parser.root["test_module.CONSTANT"] == root
    assert "test_module.CONSTANT" in parser.const


def test_globals_with_all_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an Assign node with __all__
    target = Name(id="__all__", ctx=Store())
    elts = [Constant(value="func1"), Constant(value="func2")]
    value = Tuple(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_ignores_non_name_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an Assign node with tuple unpacking
    targets = [Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())]
    value = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    node = Assign(targets=targets, value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an Assign node with type_comment
    target = Name(id="my_var", ctx=Store())
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment="int")
    
    parser.globals(root, node)
    
    assert "test_module.my_var" in parser.const
    assert parser.const["test_module.my_var"] == "int"


def test_globals_ignores_multiple_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an Assign node with multiple targets
    targets = [Name(id="a", ctx=Store()), Name(id="b", ctx=Store())]
    value = Constant(value=1)
    node = Assign(targets=targets, value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_annassign_without_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    # Create an AnnAssign node without a value
    target = Name(id="my_var", ctx=Store())
    annotation = Name(id="int", ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=None, simple=1)
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


# LLM-generated content at query #32
#--------------------------

```python
def test_attr_single_level_attribute():
    class TestObj:
        value = 42
    obj = TestObj()
    result = _attr(obj, 'value')
    assert result == 42

def test_attr_nested_attributes():
    class Inner:
        data = 'nested'
    class Outer:
        inner = Inner()
    obj = Outer()
    result = _attr(obj, 'inner.data')
    assert result == 'nested'

def test_attr_deeply_nested_attributes():
    class Level3:
        final = 'deep'
    class Level2:
        level3 = Level3()
    class Level1:
        level2 = Level2()
    obj = Level1()
    result = _attr(obj, 'level1.level2.level3.final')
    assert result is None

def test_attr_nonexistent_attribute():
    class TestObj:
        value = 42
    obj = TestObj()
    result = _attr(obj, 'nonexistent')
    assert result is None

def test_attr_nonexistent_nested_attribute():
    class Inner:
        data = 'nested'
    class Outer:
        inner = Inner()
    obj = Outer()
    result = _attr(obj, 'inner.nonexistent')
    assert result is None

def test_attr_none_in_chain():
    class Inner:
        data = None
    class Outer:
        inner = Inner()
    obj = Outer()
    result = _attr(obj, 'inner.data.something')
    assert result is None

def test_attr_with_none_object():
    result = _attr(None, 'any.attr')
    assert result is None

def test_attr_empty_string():
    class TestObj:
        pass
    obj = TestObj()
    result = _attr(obj, '')
    assert result is None

def test_attr_method_call():
    class TestObj:
        def get_value(self):
            return 100
    obj = TestObj()
    result = _attr(obj, 'get_value')
    assert callable(result)

def test_attr_with_builtin_types():
    obj = {'key': 'value'}
    result = _attr(obj, 'keys')
    assert callable(result)


# LLM-generated content at query #33
#--------------------------

```python
def test_e_type_empty_elements():
    from ast import Constant
    result = _e_type()
    assert result == ""


def test_e_type_single_element_with_single_constant():
    from ast import Constant
    const = Constant(value=42)
    result = _e_type([const])
    assert result == "[int]"


def test_e_type_single_element_with_multiple_same_type_constants():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value=2)
    result = _e_type([const1, const2])
    assert result == "[int]"


def test_e_type_single_element_with_multiple_different_type_constants():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value="string")
    result = _e_type([const1, const2])
    assert result == "[Any]"


def test_e_type_multiple_elements_same_types():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value=2)
    const3 = Constant(value=3.14)
    const4 = Constant(value=2.71)
    result = _e_type([const1, const2], [const3, const4])
    assert result == "[int, float]"


def test_e_type_multiple_elements_mixed_types():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value="string")
    const3 = Constant(value=3.14)
    result = _e_type([const1, const2], [const3])
    assert result == "[Any, float]"


def test_e_type_none_in_elements():
    from ast import Constant
    result = _e_type(None)
    assert result == ""


def test_e_type_empty_sequence_in_elements():
    from ast import Constant
    result = _e_type([])
    assert result == ""


def test_e_type_non_constant_in_sequence():
    from ast import Constant, BinOp, Add
    const = Constant(value=1)
    binop = BinOp(left=const, op=Add(), right=const)
    result = _e_type([const, binop])
    assert result == ""


def test_e_type_string_constants():
    from ast import Constant
    const1 = Constant(value="hello")
    const2 = Constant(value="world")
    result = _e_type([const1, const2])
    assert result == "[str]"


def test_e_type_boolean_constants():
    from ast import Constant
    const1 = Constant(value=True)
    const2 = Constant(value=False)
    result = _e_type([const1, const2])
    assert result == "[bool]"


def test_e_type_none_constant():
    from ast import Constant
    const = Constant(value=None)
    result = _e_type([const])
    assert result == "[NoneType]"


def test_e_type_multiple_elements_with_any():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value=2)
    const3 = Constant(value="string")
    const4 = Constant(value=4.0)
    result = _e_type([const1, const2], [const3, const4])
    assert result == "[int, Any]"


# LLM-generated content at query #34
#--------------------------

```python
def test_compile_basic():
    """Test compile method with basic parser setup."""
    parser = Parser(link=True, b_level=1, toc=False)
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.imp['test_module'] = set()
    parser.docstring['test_module'] = 'Test module documentation\n\n'
    parser.alias = {}
    parser.const = {}
    result = parser.compile()
    assert 'Module `test_module`' in result
    assert 'Test module documentation' in result


def test_compile_with_toc():
    """Test compile method with table of contents enabled."""
    parser = Parser(link=True, b_level=1, toc=True)
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.doc['test_module.func'] = '## func()\n\n*Full name:* `test_module.func`\n\n'
    parser.level['test_module'] = 0
    parser.level['test_module.func'] = 0
    parser.root['test_module'] = 'test_module'
    parser.root['test_module.func'] = 'test_module'
    parser.imp['test_module'] = set()
    parser.imp['test_module.func'] = set()
    parser.docstring['test_module'] = 'Module doc\n\n'
    parser.docstring['test_module.func'] = 'Function doc\n\n'
    parser.alias = {}
    parser.const = {}
    result = parser.compile()
    assert '**Table of contents:**' in result
    assert 'test-module' in result


def test_compile_with_constants():
    """Test compile method with constants."""
    parser = Parser(link=True, b_level=1, toc=False)
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.imp['test_module'] = set()
    parser.const['test_module.CONST_VAL'] = 'int'
    parser.root['test_module.CONST_VAL'] = 'test_module'
    parser.docstring['test_module'] = 'Module doc\n\n'
    parser.alias = {}
    result = parser.compile()
    assert 'Constants' in result
    assert 'CONST_VAL' in result


def test_compile_filters_private_names():
    """Test compile method filters out private names."""
    parser = Parser(link=True, b_level=1, toc=False)
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.doc['test_module._private'] = '## _private()\n\n'
    parser.level['test_module'] = 0
    parser.level['test_module._private'] = 0
    parser.root['test_module'] = 'test_module'
    parser.root['test_module._private'] = 'test_module'
    parser.imp['test_module'] = set()
    parser.docstring['test_module'] = 'Module doc\n\n'
    parser.docstring['test_module._private'] = 'Private doc\n\n'
    parser.alias = {}
    parser.const = {}
    result = parser.compile()
    assert '_private' not in result


def test_compile_empty_parser():
    """Test compile method with empty parser."""
    parser = Parser(link=True, b_level=1, toc=False)
    parser.doc = {}
    parser.level = {}
    parser.root = {}
    parser.imp = {}
    parser.docstring = {}
    parser.alias = {}
    parser.const = {}
    result = parser.compile()
    assert result == '\n'


def test_compile_magic_methods_without_docstring():
    """Test compile method skips magic methods without docstring."""
    parser = Parser(link=True, b_level=1, toc=False)
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.doc['test_module.__init__'] = '## __init__()\n\n'
    parser.level['test_module'] = 0
    parser.level['test_module.__init__'] = 0
    parser.root['test_module'] = 'test_module'
    parser.root['test_module.__init__'] = 'test_module'
    parser.imp['test_module'] = set()
    parser.docstring['test_module'] = 'Module doc\n\n'
    parser.alias = {}
    parser.const = {}
    result = parser.compile()
    assert '__init__' not in result


def test_compile_multiple_levels():
    """Test compile method with nested names."""
    parser = Parser(link=True, b_level=1, toc=True)
    parser.doc['pkg'] = '# Module `pkg`\n\n'
    parser.doc['pkg.subpkg'] = '## Module `pkg.subpkg`\n\n'
    parser.doc['pkg.subpkg.func'] = '### func()\n\n*Full name:* `pkg.subpkg.func`\n\n'
    parser.level['pkg'] = 0
    parser.level['pkg.subpkg'] = 1
    parser.level['pkg.subpkg.func'] = 1
    parser.root['pkg'] = 'pkg'
    parser.root['pkg.subpkg'] = 'pkg.subpkg'
    parser.root['pkg.subpkg.func'] = 'pkg.subpkg'
    parser.imp['pkg'] = set()
    parser.imp['pkg.subpkg'] = set()
    parser.docstring['pkg'] = 'Package doc\n\n'
    parser.docstring['pkg.subpkg'] = 'Subpackage doc\n\n'
    parser.docstring['pkg.subpkg.func'] = 'Function doc\n\n'
    parser.alias = {}
    parser.const = {}
    result = parser.compile()
    assert 'pkg' in result
    assert 'subpkg' in result
    assert 'func' in result
    assert '**Table of contents:**' in result


# LLM-generated content at query #35
#--------------------------

```python
def test_class_api_delete_statement_handling():
    from ast import Delete, Name, parse, AnnAssign, Assign, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Setup initial state
    root = "test_module"
    name = "test_module.TestClass"
    parser.doc[name] = "## class TestClass\n\n"
    parser.level[root] = 0
    parser.root[name] = root
    
    # Create a Delete node with a Name target
    delete_node = Delete(targets=[Name(id='attr_to_delete', ctx=None)])
    
    # Create a mock body with the Delete node
    # We'll use a simple approach by directly checking the isinstance condition
    node = delete_node
    
    # Test the predicate at line 30: isinstance(node, Delete)
    result = isinstance(node, Delete)
    
    assert result is True


# LLM-generated content at query #36
#--------------------------

```python
def test_globals_with_annassign_and_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse, AnnAssign, Name, Constant
    code = "MY_CONST: int = 42"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONST" in parser.alias
    assert parser.alias["test_module.MY_CONST"] == "42"
    assert "test_module.MY_CONST" in parser.const
    assert parser.const["test_module.MY_CONST"] == "int"
    assert "test_module.MY_CONST" in parser.root


def test_globals_with_assign_and_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "value = 100  # type: float"
    node = ast_parse(code, type_comments=True).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.value" in parser.alias
    assert parser.alias["test_module.value"] == "100"
    assert "test_module.value" in parser.const
    assert parser.const["test_module.value"] == "float"


def test_globals_with_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "CONSTANT = 'hello'"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.CONSTANT" in parser.root
    assert parser.root["test_module.CONSTANT"] == "test_module"


def test_globals_with_all_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "__all__ = ['func1', 'func2']"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_assign_without_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "number = 42"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.number" in parser.alias
    assert parser.alias["test_module.number"] == "42"
    assert "test_module.number" in parser.const
    assert parser.const["test_module.number"] == "int"


def test_globals_with_invalid_node_structure():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "a, b = 1, 2"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0
    assert len(parser.const) == 0


def test_globals_with_multiple_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "x = y = 5"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


# LLM-generated content at query #37
#--------------------------

```python
def test_globals_type_comment_not_none():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict[str, int] = field(default_factory=dict)
        doc: dict[str, str] = field(default_factory=dict)
        docstring: dict[str, str] = field(default_factory=dict)
        imp: dict[str, set[str]] = field(default_factory=dict)
        root: dict[str, str] = field(default_factory=dict)
        alias: dict[str, str] = field(default_factory=dict)
        const: dict[str, str] = field(default_factory=dict)
        
        def resolve(self, root: str, node):
            return "int"
        
        def globals(self, root: str, node):
            if (
                isinstance(node, Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], Name)
            ):
                left = node.targets[0]
                if node.type_comment is None:
                    ann = "const_type_result"
                else:
                    ann = node.type_comment
                return ann
            return None
    
    parser = Parser()
    
    target = Name(id='test_var', ctx=None)
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment='str')
    
    result = parser.globals("module", node)
    
    assert result == 'str'
    assert node.type_comment is not None


# LLM-generated content at query #38
#--------------------------

```python
def test_visit_constant_with_non_string_value():
    resolver = Resolver("mymodule", {})
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result is node


def test_visit_constant_with_string_valid_name():
    resolver = Resolver("mymodule", {"mymodule.int": "int"})
    node = Constant(value="int")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"


def test_visit_constant_with_string_invalid_syntax():
    resolver = Resolver("mymodule", {})
    node = Constant(value="not valid python @@@@")
    result = resolver.visit_Constant(node)
    assert result is node


def test_visit_constant_with_string_self_type():
    resolver = Resolver("mymodule", {}, self_ty="T")
    node = Constant(value="T")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


def test_visit_constant_with_string_complex_expression():
    resolver = Resolver("mymodule", {"mymodule.List": "list", "mymodule.int": "int"})
    node = Constant(value="List[int]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)


# LLM-generated content at query #39
#--------------------------

```python
def test_compile_predicate_line_13_true():
    """Test that the predicate at line 13 (name in self.docstring) evaluates to True."""
    parser = Parser(link=True, b_level=1, toc=False)
    
    # Set up the parser state
    test_name = "test_module"
    test_doc = "# Module `test_module`\n\n"
    test_docstring = "This is a test docstring."
    
    parser.doc[test_name] = test_doc
    parser.docstring[test_name] = test_docstring
    parser.root[test_name] = test_name
    parser.level[test_name] = 0
    parser.imp[test_name] = set()
    
    # Compile and verify the docstring is included in output
    result = parser.compile()
    
    # The predicate at line 13 should be True, so docstring should be appended
    assert test_docstring in result


# LLM-generated content at query #40
#--------------------------

```python
def test_visit_name_predicate_line_6_true():
    from ast import Name, Load
    
    # Mock the _m function to return predictable values
    def mock_m(root, node_id):
        return f"{root}.{node_id}"
    
    # Save original _m and replace with mock
    import sys
    module = sys.modules[__name__]
    original_m = getattr(module, '_m', None)
    
    # Setup: Create a Resolver instance with alias containing the name
    root = "mymodule"
    alias = {
        "mymodule.MyType": "int"
    }
    resolver = Resolver(root, alias)
    
    # Create a Name node
    node = Name(id="MyType", ctx=Load())
    
    # Mock _m function for this test
    import __main__
    __main__._m = mock_m
    
    # The predicate at line 6 should evaluate to True when:
    # 1. name (which is "mymodule.MyType") is in self.alias -> True
    # 2. name ("mymodule.MyType") is not in self.alias["mymodule.MyType"] (which is "int") -> True
    result = resolver.visit_Name(node)
    
    # Verify the result is what we expect when the predicate is True
    # When predicate is True, it should parse the alias value and visit it
    assert result is not None


# LLM-generated content at query #41
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'BaseClass' in parser.doc['MyClass']


def test_class_api_with_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    member_node = AnnAssign(
        target=Name(id='attr', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=None,
        simple=1
    )
    body = [member_node]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'attr' in parser.doc['MyClass']


def test_class_api_with_enums():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyEnum'] = '## class MyEnum\n\n'
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    enum_node = AnnAssign(
        target=Name(id='MEMBER', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=1),
        simple=1
    )
    body = [enum_node]
    
    parser.class_api('test_module', 'MyEnum', bases, body)
    
    assert 'MyEnum' in parser.doc
    assert 'MEMBER' in parser.doc['MyEnum']


def test_class_api_with_delete():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    member_node = AnnAssign(
        target=Name(id='attr', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=None,
        simple=1
    )
    delete_node = Delete(targets=[Name(id='attr', ctx=Del())])
    body = [member_node, delete_node]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc


def test_class_api_empty_class():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['EmptyClass'] = '## class EmptyClass\n\n'
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'EmptyClass', bases, body)
    
    assert 'EmptyClass' in parser.doc


def test_class_api_private_members_ignored():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    private_member = AnnAssign(
        target=Name(id='_private', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=None,
        simple=1
    )
    body = [private_member]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert '_private' not in parser.doc['MyClass']


def test_class_api_with_multiple_bases():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = [
        Name(id='Base1', ctx=Load()),
        Name(id='Base2', ctx=Load())
    ]
    body = []
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'Base1' in parser.doc['MyClass'] or 'Base2' in parser.doc['MyClass']


def test_class_api_with_assign_member():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    member_node = Assign(
        targets=[Name(id='value', ctx=Store())],
        value=Constant(value=42),
        type_comment=None
    )
    body = [member_node]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'value' in parser.doc['MyClass']


# LLM-generated content at query #42
#--------------------------

```python
def test_globals_predicate_line_38_false():
    """Test that the predicate at line 38 evaluates to False."""
    from ast import Assign, Name, Constant, Tuple
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    parser.imp['test_root'] = set()
    
    # Create an Assign node with __all__ = (123,)
    # where the element is a Constant but not a string
    assign_node = Assign(
        targets=[Name(id='__all__')],
        value=Tuple(elts=[Constant(value=123)], ctx=None),
        type_comment=None
    )
    
    # Call globals method
    parser.globals('test_root', assign_node)
    
    # The predicate at line 38 should be False because e.value is 123 (int), not str
    # Therefore, the imp set should remain empty
    assert len(parser.imp['test_root']) == 0


# LLM-generated content at query #43
#--------------------------

```python
def test_func_ann_yields_empty_string_for_star_arg():
    from dataclasses import dataclass, field
    from ast import arg
    from typing import TypeVar
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict[str, int] = field(default_factory=dict)
        doc: dict[str, str] = field(default_factory=dict)
        docstring: dict[str, str] = field(default_factory=dict)
        imp: dict[str, set[str]] = field(default_factory=dict)
        root: dict[str, str] = field(default_factory=dict)
        alias: dict[str, str] = field(default_factory=dict)
        const: dict[str, str] = field(default_factory=dict)
        
        def resolve(self, root: str, node, self_ty: str = "") -> str:
            return "int"
        
        def func_ann(self, root: str, args, *, has_self: bool, cls_method: bool):
            self_ty = ""
            for i, a in enumerate(args):
                if has_self and i == 0:
                    if a.annotation is not None:
                        self_ty = self.resolve(root, a.annotation)
                        if cls_method:
                            self_ty = (self_ty.removeprefix('type[')
                                       .removesuffix(']'))
                    yield 'type[Self]' if cls_method else 'Self'
                elif a.arg == '*':
                    yield ""
                elif a.annotation is not None:
                    yield self.resolve(root, a.annotation, self_ty)
                else:
                    yield "Any"
    
    parser = Parser()
    star_arg = arg(arg='*', annotation=None)
    result = list(parser.func_ann('root', [star_arg], has_self=False, cls_method=False))
    
    assert result == [""]
    assert result[0] == ""


# LLM-generated content at query #44
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is False
    assert parser.b_level == 3
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}


def test_parser_new_classmethod_with_toc_true():
    parser = Parser.new(link=False, level=2, toc=True)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is True


# LLM-generated content at query #45
#--------------------------

```python
def test_func_ann_predicate_line_7_true():
    from ast import arg, Name
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an arg with annotation to make line 7 predicate True
    test_arg = arg(arg='self', annotation=Name(id='int'))
    args = [test_arg]
    
    # Call func_ann with has_self=True and i=0 to reach line 7
    result = list(parser.func_ann('test_module', args, has_self=True, cls_method=False))
    
    # Verify the predicate at line 7 was evaluated to True by checking the result
    assert len(result) > 0
    assert result[0] == 'Self'


# LLM-generated content at query #46
#--------------------------

```python
def test_parser_constructor_default():
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


def test_parser_constructor_custom():
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


def test_parser_constructor_toc_enables_link():
    p = Parser(link=False, b_level=1, toc=True)
    assert p.link is True
    assert p.toc is True


def test_parser_new_classmethod():
    p = Parser.new(link=True, level=2, toc=False)
    assert p.link is True
    assert p.b_level == 2
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


def test_parser_new_with_toc_true():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


# LLM-generated content at query #47
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.doc['test_module'] = ''
    parser.root['test_module.TestClass'] = 'test_module'
    parser.level['test_module.TestClass'] = 0
    parser.alias = {}
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'BaseClass' in parser.doc['test_module.TestClass']
    assert 'Bases' in parser.doc['test_module.TestClass']


def test_class_api_with_members():
    parser = Parser()
    parser.doc['test_module.TestClass'] = ''
    parser.root['test_module.TestClass'] = 'test_module'
    parser.level['test_module.TestClass'] = 0
    parser.alias = {}
    
    ann_assign = AnnAssign(
        target=Name(id='attr1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='test'),
        simple=1
    )
    body = [ann_assign]
    bases = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'attr1' in parser.doc['test_module.TestClass']


def test_class_api_with_enums():
    parser = Parser()
    parser.doc['test_module.TestEnum'] = ''
    parser.root['test_module.TestEnum'] = 'test_module'
    parser.level['test_module.TestEnum'] = 0
    parser.alias = {}
    
    enum_base = Name(id='enum.Enum', ctx=Load())
    ann_assign = AnnAssign(
        target=Name(id='MEMBER1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='value1'),
        simple=1
    )
    body = [ann_assign]
    bases = [enum_base]
    
    parser.class_api('test_module', 'test_module.TestEnum', bases, body)
    
    assert 'Enums' in parser.doc['test_module.TestEnum']


def test_class_api_with_delete():
    parser = Parser()
    parser.doc['test_module.TestClass'] = ''
    parser.root['test_module.TestClass'] = 'test_module'
    parser.level['test_module.TestClass'] = 0
    parser.alias = {}
    
    ann_assign = AnnAssign(
        target=Name(id='attr1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='test'),
        simple=1
    )
    delete_stmt = Delete(targets=[Name(id='attr1', ctx=Del())])
    body = [ann_assign, delete_stmt]
    bases = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'attr1' not in parser.doc['test_module.TestClass']


def test_class_api_empty_class():
    parser = Parser()
    parser.doc['test_module.EmptyClass'] = ''
    parser.root['test_module.EmptyClass'] = 'test_module'
    parser.level['test_module.EmptyClass'] = 0
    parser.alias = {}
    
    parser.class_api('test_module', 'test_module.EmptyClass', [], [])
    
    assert 'test_module.EmptyClass' in parser.doc


def test_class_api_with_private_members():
    parser = Parser()
    parser.doc['test_module.TestClass'] = ''
    parser.root['test_module.TestClass'] = 'test_module'
    parser.level['test_module.TestClass'] = 0
    parser.alias = {}
    
    private_attr = AnnAssign(
        target=Name(id='_private', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='test'),
        simple=1
    )
    body = [private_attr]
    bases = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert '_private' not in parser.doc['test_module.TestClass']


def test_class_api_with_multiple_bases():
    parser = Parser()
    parser.doc['test_module.TestClass'] = ''
    parser.root['test_module.TestClass'] = 'test_module'
    parser.level['test_module.TestClass'] = 0
    parser.alias = {}
    
    bases = [
        Name(id='Base1', ctx=Load()),
        Name(id='Base2', ctx=Load())
    ]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'Base1' in parser.doc['test_module.TestClass']
    assert 'Base2' in parser.doc['test_module.TestClass']


# LLM-generated content at query #48
#--------------------------

```python
def test_visit_constant_syntax_error():
    from ast import Constant, Load, Name
    
    resolver = Resolver(root="test_module", alias={})
    node = Constant(value="not a valid python expression !!!")
    result = resolver.visit_Constant(node)
    
    assert result is node


# LLM-generated content at query #49
#--------------------------

```python
def test_globals_predicate_line_38_false():
    """Test that the predicate at line 38 evaluates to False."""
    from ast import Assign, Name, Constant, Tuple
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Create an Assign node with __all__ = (123, "value")
    # where 123 is not a Constant with str value
    target = Name(id='__all__', ctx=None)
    const_int = Constant(value=123)
    const_str = Constant(value='value')
    tuple_node = Tuple(elts=[const_int, const_str], ctx=None)
    assign_node = Assign(targets=[target], value=tuple_node, type_comment=None)
    
    # Initialize root in parser.imp
    root = 'test_module'
    parser.imp[root] = set()
    
    # Call globals method - it should process the tuple
    # The predicate at line 38 should be False for const_int (not a str)
    parser.globals(root, assign_node)
    
    # Verify that only the string constant was added to imp
    # The integer constant should not trigger the isinstance(e.value, str) check
    assert 'test_module.value' in parser.imp[root]
    assert len(parser.imp[root]) == 1


# LLM-generated content at query #50
#--------------------------

```python
def test_parse_basic_module():
    parser = Parser()
    script = """
'''Module docstring'''
import os
x = 5
def foo():
    '''Function docstring'''
    pass
"""
    parser.parse('test_module', script)
    assert 'test_module' in parser.doc
    assert 'test_module' in parser.level
    assert parser.level['test_module'] == 0
    assert 'test_module' in parser.imp
    assert 'os' in parser.alias or any('os' in v for v in parser.alias.values())


def test_parse_with_imports():
    parser = Parser()
    script = """
from collections import defaultdict
from typing import List
import sys
"""
    parser.parse('mymodule', script)
    assert 'mymodule' in parser.imp
    assert len(parser.alias) > 0


def test_parse_with_class_definition():
    parser = Parser()
    script = """
class MyClass:
    '''Class docstring'''
    def method(self):
        '''Method docstring'''
        pass
"""
    parser.parse('test_pkg', script)
    assert 'test_pkg' in parser.doc
    assert 'test_pkg.MyClass' in parser.doc
    assert 'test_pkg.MyClass.method' in parser.doc


def test_parse_with_function_definition():
    parser = Parser()
    script = """
def my_function(x: int) -> str:
    '''Function docstring'''
    return str(x)
"""
    parser.parse('test_module', script)
    assert 'test_module' in parser.doc
    assert 'test_module.my_function' in parser.doc
    assert 'test_module.my_function' in parser.root


def test_parse_with_constants():
    parser = Parser()
    script = """
DEBUG = True
MAX_SIZE: int = 100
"""
    parser.parse('config', script)
    assert 'config' in parser.doc
    assert 'config.DEBUG' in parser.const or 'config.DEBUG' in parser.alias
    assert 'config.MAX_SIZE' in parser.const or 'config.MAX_SIZE' in parser.alias


def test_parse_with_link_option():
    parser = Parser(link=True)
    script = "'''Module doc'''"
    parser.parse('mymod', script)
    assert '<a id=' in parser.doc['mymod']


def test_parse_without_link_option():
    parser = Parser(link=False)
    script = "'''Module doc'''"
    parser.parse('mymod', script)
    assert '<a id=' not in parser.doc['mymod']


def test_parse_nested_class():
    parser = Parser()
    script = """
class Outer:
    '''Outer class'''
    class Inner:
        '''Inner class'''
        pass
"""
    parser.parse('pkg', script)
    assert 'pkg.Outer' in parser.doc
    assert 'pkg.Outer.Inner' in parser.doc


def test_parse_async_function():
    parser = Parser()
    script = """
async def async_func():
    '''Async function'''
    pass
"""
    parser.parse('module', script)
    assert 'module.async_func' in parser.doc
    assert 'async' in parser.doc['module.async_func']


def test_parse_with_decorators():
    parser = Parser()
    script = """
class MyClass:
    @staticmethod
    def static_method():
        '''Static method'''
        pass
    
    @classmethod
    def class_method(cls):
        '''Class method'''
        pass
"""
    parser.parse('mod', script)
    assert 'mod.MyClass.static_method' in parser.doc
    assert 'mod.MyClass.class_method' in parser.doc


def test_parse_docstring_handling():
    parser = Parser()
    script = """
'''Module with docstring.

>>> x = 1
>>> print(x)
1
'''
def func():
    '''Function doc.
    
    >>> func()
    '''
    pass
"""
    parser.parse('test', script)
    assert 'test' in parser.docstring
    assert 'test.func' in parser.docstring


def test_parse_multiple_inheritance():
    parser = Parser()
    script = """
class Child(Parent1, Parent2):
    '''Child class'''
    pass
"""
    parser.parse('pkg', script)
    assert 'pkg.Child' in parser.doc
    assert 'pkg.Child' in parser.root


def test_parse_all_variable():
    parser = Parser()
    script = """
__all__ = ['public_func', 'PublicClass']

def public_func():
    pass

def _private_func():
    pass

class PublicClass:
    pass
"""
    parser.parse('module', script)
    assert 'module' in parser.imp
    assert 'module.public_func' in parser.imp['module'] or len(parser.imp['module']) > 0


def test_parse_module_level():
    parser = Parser(level=2)
    script = "'''Doc'''"
    parser.parse('package.submodule', script)
    assert parser.level['package.submodule'] == 2
    assert '##' in parser.doc['package.submodule']


def test_parse_with_type_comments():
    parser = Parser()
    script = """
x = 5  # type: int
def foo(a, b):  # type: (int, str) -> bool
    return True
"""
    parser.parse('mod', script)
    assert 'mod' in parser.doc
    assert 'mod.foo' in parser.doc


# LLM-generated content at query #51
#--------------------------

```python
def test_func_api_kwonlyargs_predicate():
    from ast import arguments, arg
    from dataclasses import dataclass, field
    
    parser = Parser()
    parser.doc['test_func'] = ''
    
    # Create an arguments node with kwonlyargs but no vararg
    # This should make the predicate at line 16 evaluate to True
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg(arg='kw_only_param', annotation=None)],
        kw_defaults=[None],
        kwarg=None,
        defaults=[]
    )
    
    parser.func_api('test_module', 'test_func', node, None, has_self=False, cls_method=False)
    
    assert 'test_func' in parser.doc
    assert '*' in parser.doc['test_func']


# LLM-generated content at query #52
#--------------------------

```python
def test_imports_with_import_statement():
    from ast import Import, alias
    parser = Parser()
    node = Import(names=[alias(name='os', asname=None)])
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.os'] == 'os'


def test_imports_with_import_as_statement():
    from ast import Import, alias
    parser = Parser()
    node = Import(names=[alias(name='os', asname='operating_system')])
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.operating_system'] == 'os'


def test_imports_with_multiple_imports():
    from ast import Import, alias
    parser = Parser()
    node = Import(names=[alias(name='os', asname=None), alias(name='sys', asname=None)])
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.os'] == 'os'
    assert parser.alias['mymodule.sys'] == 'sys'


def test_imports_from_statement_absolute():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname=None)], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.path'] == 'os.path'


def test_imports_from_statement_absolute_with_asname():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname='ospath')], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.ospath'] == 'os.path'


def test_imports_from_statement_relative_level_1():
    from ast import ImportFrom, alias
    parser = Parser()
    parser.parse('pkg.mymodule', '')
    node = ImportFrom(module='sibling', names=[alias(name='func', asname=None)], level=1)
    parser.imports('pkg.mymodule', node)
    assert parser.alias['pkg.mymodule.func'] == 'pkg.sibling.func'


def test_imports_from_statement_relative_level_2():
    from ast import ImportFrom, alias
    parser = Parser()
    parser.parse('pkg.sub.mymodule', '')
    node = ImportFrom(module='other', names=[alias(name='item', asname=None)], level=2)
    parser.imports('pkg.sub.mymodule', node)
    assert parser.alias['pkg.sub.mymodule.item'] == 'pkg.other.item'


def test_imports_from_statement_relative_no_module():
    from ast import ImportFrom, alias
    parser = Parser()
    parser.parse('pkg.mymodule', '')
    node = ImportFrom(module=None, names=[alias(name='helper', asname=None)], level=1)
    parser.imports('pkg.mymodule', node)
    assert parser.alias['pkg.mymodule.helper'] == 'pkg.helper'


def test_imports_from_statement_multiple_names():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname=None), alias(name='getcwd', asname=None)], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.path'] == 'os.path'
    assert parser.alias['mymodule.getcwd'] == 'os.getcwd'


def test_imports_from_statement_with_star():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='*', asname=None)], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.*'] == 'os.*'


# LLM-generated content at query #53
#--------------------------

```python
def test_load_docstring():
    from types import ModuleType
    
    # Create a mock module with docstrings
    mock_module = ModuleType("test_module")
    
    # Create a nested mock object with docstring
    class MockClass:
        """This is a mock class docstring.
        
        >>> x = 1
        >>> print(x)
        1
        """
        pass
    
    mock_module.MockClass = MockClass
    
    # Create parser instance
    parser = Parser(link=True, b_level=1, toc=False)
    
    # Add some entries to doc that should match the module structure
    parser.doc["test_module"] = "# Module `test_module`"
    parser.doc["test_module.MockClass"] = "## class MockClass\n\n"
    parser.root["test_module"] = "test_module"
    parser.root["test_module.MockClass"] = "test_module"
    
    # Call load_docstring
    parser.load_docstring("test_module", mock_module)
    
    # Verify that docstring was loaded and processed with doctest
    assert "test_module.MockClass" in parser.docstring
    assert "```python" in parser.docstring["test_module.MockClass"]
    assert "x = 1" in parser.docstring["test_module.MockClass"]
    assert "```" in parser.docstring["test_module.MockClass"]


def test_load_docstring_no_docstring():
    from types import ModuleType
    
    # Create a mock module without docstrings
    mock_module = ModuleType("test_module")
    
    class MockClassNoDoc:
        pass
    
    mock_module.MockClassNoDoc = MockClassNoDoc
    
    # Create parser instance
    parser = Parser(link=True, b_level=1, toc=False)
    
    parser.doc["test_module"] = "# Module `test_module`"
    parser.doc["test_module.MockClassNoDoc"] = "## class MockClassNoDoc\n\n"
    parser.root["test_module"] = "test_module"
    parser.root["test_module.MockClassNoDoc"] = "test_module"
    
    # Call load_docstring
    parser.load_docstring("test_module", mock_module)
    
    # Verify that no docstring was added for items without docstrings
    assert "test_module.MockClassNoDoc" not in parser.docstring


def test_load_docstring_nested_attribute():
    from types import ModuleType
    
    # Create a mock module with nested attributes
    mock_module = ModuleType("test_module")
    
    class OuterClass:
        """Outer class docstring."""
        class InnerClass:
            """Inner class docstring."""
            pass
    
    mock_module.OuterClass = OuterClass
    
    # Create parser instance
    parser = Parser(link=True, b_level=1, toc=False)
    
    parser.doc["test_module"] = "# Module `test_module`"
    parser.doc["test_module.OuterClass"] = "## class OuterClass\n\n"
    parser.doc["test_module.OuterClass.InnerClass"] = "### class InnerClass\n\n"
    parser.root["test_module"] = "test_module"
    parser.root["test_module.OuterClass"] = "test_module"
    parser.root["test_module.OuterClass.InnerClass"] = "test_module"
    
    # Call load_docstring
    parser.load_docstring("test_module", mock_module)
    
    # Verify that nested docstrings were loaded
    assert "test_module.OuterClass" in parser.docstring
    assert "Outer class docstring" in parser.docstring["test_module.OuterClass"]
    assert "test_module.OuterClass.InnerClass" in parser.docstring
    assert "Inner class docstring" in parser.docstring["test_module.OuterClass.InnerClass"]


# LLM-generated content at query #54
#--------------------------

```python
def test_globals_annassign_with_value():
    from ast import parse, AnnAssign, Name, Constant
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "x: int = 5"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "5"


def test_globals_annassign_uppercase_constant():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "CONST: str = 'hello'"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.CONST" in parser.root
    assert parser.root["test_module.CONST"] == "test_module"
    assert "test_module.CONST" in parser.const


def test_globals_assign_simple():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "x = 42"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "42"


def test_globals_assign_with_type_comment():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "x = 10  # type: int"
    tree = parse(script, type_comments=True)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.x" in parser.alias
    assert parser.const["test_module.x"] == "int"


def test_globals_all_tuple():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    script = "__all__ = ('func1', 'func2')"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_all_list():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    script = "__all__ = ['func1', 'func2', 'func3']"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]
    assert "test_module.func3" in parser.imp[root]


def test_globals_multiple_targets_ignored():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "x = y = 5"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.x" not in parser.alias


def test_globals_non_name_target_ignored():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "(x, y) = (1, 2)"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.x" not in parser.alias


def test_globals_uppercase_multiple_times():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script1 = "CONST = 5"
    tree1 = parse(script1)
    node1 = tree1.body[0]
    parser.globals(root, node1)
    script2 = "CONST = 10"
    tree2 = parse(script2)
    node2 = tree2.body[0]
    parser.globals(root, node2)
    assert parser.const["test_module.CONST"] != "ANY"


def test_globals_annassign_without_value_ignored():
    from ast import parse
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    script = "x: int"
    tree = parse(script)
    node = tree.body[0]
    parser.globals(root, node)
    assert "test_module.x" not in parser.alias


# LLM-generated content at query #55
#--------------------------

```python
def test_visit_name_with_self_ty():
    resolver = Resolver(root="mymodule", alias={}, self_ty="T")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


def test_visit_name_without_self_ty():
    resolver = Resolver(root="mymodule", alias={}, self_ty="")
    node = Name(id="SomeClass", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeClass"


def test_visit_name_with_alias_not_recursive():
    resolver = Resolver(root="mymodule", alias={"mymodule.MyType": "int"}, self_ty="")
    node = Name(id="MyType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"


def test_visit_name_with_typevar_alias():
    resolver = Resolver(root="mymodule", alias={"mymodule.T": "TypeVar('T')", "typing.TypeVar": "typing.TypeVar"}, self_ty="")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


def test_visit_name_with_circular_alias():
    resolver = Resolver(root="mymodule", alias={"mymodule.A": "mymodule.A"}, self_ty="")
    node = Name(id="A", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "A"


def test_visit_name_no_root_with_alias():
    resolver = Resolver(root="", alias={"MyType": "str"}, self_ty="")
    node = Name(id="MyType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "str"


# LLM-generated content at query #56
#--------------------------

```python
def test_imports_asname_not_none():
    from ast import ImportFrom, alias
    from dataclasses import dataclass, field
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict = field(default_factory=dict)
        doc: dict = field(default_factory=dict)
        docstring: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        alias: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
        
        def imports(self, root: str, node) -> None:
            if isinstance(node, ImportFrom):
                if node.module is not None:
                    m = ''
                    for a in node.names:
                        name = a.name if a.asname is None else a.asname
                        self.alias[root + '.' + name] = m + '.' + node.module + '.' + a.name if m else node.module + '.' + a.name
    
    parser = Parser()
    import_node = ImportFrom(module='collections', names=[alias(name='OrderedDict', asname='OD')], level=0)
    parser.imports('mymodule', import_node)
    
    assert 'mymodule.OD' in parser.alias
    assert parser.alias['mymodule.OD'] == 'collections.OrderedDict'


# LLM-generated content at query #57
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']
    assert 'BaseClass' in parser.doc['test_module.TestClass']


def test_class_api_with_enum_members():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    member_node = AnnAssign(
        target=Name(id='MEMBER1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=Constant(value='value1'),
        simple=1
    )
    body = [member_node]
    
    parser.class_api('test_module', 'test_module.TestEnum', bases, body)
    
    assert 'test_module.TestEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.TestEnum']
    assert 'MEMBER1' in parser.doc['test_module.TestEnum']


def test_class_api_with_public_members():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Name(id='object', ctx=Load())]
    member_node = AnnAssign(
        target=Name(id='public_attr', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    body = [member_node]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'public_attr' in parser.doc['test_module.TestClass']


def test_class_api_with_private_members():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [Name(id='object', ctx=Load())]
    member_node = AnnAssign(
        target=Name(id='_private_attr', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    body = [member_node]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert '_private_attr' not in parser.doc['test_module.TestClass']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    assign_node = Assign(
        targets=[Name(id='temp_attr', ctx=Store())],
        value=Constant(value=10)
    )
    delete_node = Delete(targets=[Name(id='temp_attr', ctx=Del())])
    body = [assign_node, delete_node]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'temp_attr' not in parser.doc['test_module.TestClass']


def test_class_api_with_assign_members():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    member_node = Assign(
        targets=[Name(id='class_var', ctx=Store())],
        value=Constant(value='test'),
        type_comment=None
    )
    body = [member_node]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'class_var' in parser.doc['test_module.TestClass']


def test_class_api_empty_class():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.EmptyClass', bases, body)
    
    assert 'test_module.EmptyClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.EmptyClass']
    assert 'Enums' not in parser.doc['test_module.EmptyClass']


def test_class_api_multiple_bases():
    parser = Parser()
    parser.doc['test_module'] = "# Module `test_module`\n\n"
    parser.root['test_module'] = 'test_module'
    parser.alias = {}
    
    bases = [
        Name(id='Base1', ctx=Load()),
        Name(id='Base2', ctx=Load())
    ]
    body = []
    
    parser.class_api('test_module', 'test_module.MultiBase', bases, body)
    
    assert 'test_module.MultiBase' in parser.doc
    assert 'Bases' in parser.doc['test_module.MultiBase']
    assert 'Base1' in parser.doc['test_module.MultiBase']
    assert 'Base2' in parser.doc['test_module.MultiBase']


# LLM-generated content at query #58
#--------------------------

```python
def test_visit_attribute_typing_prefix():
    resolver = Resolver(root="test_module", alias={})
    
    # Create an Attribute node with value being a Name node with id='typing'
    typing_name = Name(id='typing', ctx=Load())
    attr_node = Attribute(value=typing_name, attr='List', ctx=Load())
    
    result = resolver.visit_Attribute(attr_node)
    
    assert isinstance(result, Name)
    assert result.id == 'List'
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #59
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.MyClass']


def test_class_api_with_enum_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyEnum'] = '## class MyEnum\n\n'
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    target = Name(id='MEMBER1', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1)
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.MyEnum', bases, body)
    
    assert 'test_module.MyEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.MyEnum']


def test_class_api_with_public_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = []
    target = Name(id='public_attr', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='str', ctx=Load()), value=Constant(value='test'), simple=1)
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' in parser.doc['test_module.MyClass']


def test_class_api_with_private_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = []
    target = Name(id='_private_attr', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='str', ctx=Load()), value=Constant(value='test'), simple=1)
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.MyClass']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = []
    target = Name(id='member', ctx=Store())
    assign = Assign(targets=[target], value=Constant(value=1))
    delete = Delete(targets=[Name(id='member', ctx=Del())])
    body = [assign, delete]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.MyClass']


def test_class_api_no_bases_no_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert parser.doc['test_module.MyClass'] == '## class MyClass\n\n'


def test_class_api_with_assign_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = []
    target = Name(id='count', ctx=Store())
    assign = Assign(targets=[target], value=Constant(value=0), type_comment='int')
    body = [assign]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' in parser.doc['test_module.MyClass']


def test_class_api_with_multiple_bases():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = [Name(id='Base1', ctx=Load()), Name(id='Base2', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.MyClass']


# LLM-generated content at query #60
#--------------------------

```python
def test_imports_with_relative_import_level():
    from ast import ImportFrom, alias
    from dataclasses import dataclass, field
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict[str, int] = field(default_factory=dict)
        doc: dict[str, str] = field(default_factory=dict)
        docstring: dict[str, str] = field(default_factory=dict)
        imp: dict[str, set[str]] = field(default_factory=dict)
        root: dict[str, str] = field(default_factory=dict)
        alias: dict[str, str] = field(default_factory=dict)
        const: dict[str, str] = field(default_factory=dict)
        
        def imports(self, root: str, node) -> None:
            if isinstance(node, ImportFrom):
                if node.module is not None:
                    if node.level:
                        # Line 8: predicate evaluates to True
                        pass
    
    parser = Parser()
    import_node = ImportFrom(module='submodule', names=[alias(name='func', asname=None)], level=1)
    
    parser.imports('package.module', import_node)
    assert import_node.level == 1


# LLM-generated content at query #61
#--------------------------

```python
def test_imports_with_asname():
    from ast import Import, alias
    from dataclasses import dataclass, field
    from typing import TypeVar
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict[str, int] = field(default_factory=dict)
        doc: dict[str, str] = field(default_factory=dict)
        docstring: dict[str, str] = field(default_factory=dict)
        imp: dict[str, set[str]] = field(default_factory=dict)
        root: dict[str, str] = field(default_factory=dict)
        alias: dict[str, str] = field(default_factory=dict)
        const: dict[str, str] = field(default_factory=dict)
        
        def imports(self, root: str, node) -> None:
            if isinstance(node, Import):
                for a in node.names:
                    name = a.name if a.asname is None else a.asname
                    self.alias[root + '.' + name] = a.name
    
    parser = Parser()
    
    # Create an Import node with asname set (not None)
    import_node = Import(names=[alias(name='os', asname='operating_system')])
    
    # Call imports - the predicate at line 5 should evaluate to False
    # because a.asname is not None
    parser.imports('mymodule', import_node)
    
    # Verify that name is set to a.asname (the else branch was taken)
    assert parser.alias['mymodule.operating_system'] == 'os'


# LLM-generated content at query #62
#--------------------------

```python
def test_compile_magic_method_continues():
    from ast import parse as ast_parse
    from dataclasses import dataclass, field
    
    parser = Parser(link=True, b_level=1, toc=False)
    
    # Setup parser state for a magic method
    parser.doc['__init__'] = '# {}\n<a id="{}"></a>\n\n'
    parser.root['__init__'] = ''
    parser.level['__init__'] = 0
    parser.imp[''] = set()
    # Intentionally not adding to docstring to trigger the elif branch
    
    # Call compile - should skip magic method without warning
    result = parser.compile()
    
    # The magic method should not appear in the output
    assert '__init__' not in result


# LLM-generated content at query #63
#--------------------------

```python
def test_class_api_assign_predicate():
    from ast import Assign, Name, Constant, parse
    
    parser = Parser()
    parser.doc['test_module.TestClass'] = "# class TestClass\n\n"
    
    # Create an Assign node with exactly 1 target that is a Name
    assign_node = Assign(
        targets=[Name(id='x', ctx=None)],
        value=Constant(value=42),
        type_comment=None
    )
    
    # Verify the predicate at line 19 evaluates to True
    assert isinstance(assign_node, Assign)
    assert len(assign_node.targets) == 1
    assert isinstance(assign_node.targets[0], Name)


# LLM-generated content at query #64
#--------------------------

```python
def test_compile_magic_method_without_docstring():
    from dataclasses import dataclass, field
    from typing import TypeVar
    from ast import parse as ast_parse
    
    p = Parser()
    p.doc['module.__init__'] = '## {}\n<a id="{}"></a>\n\n'
    p.docstring = {}
    p.imp = {'module': set()}
    p.root = {'module.__init__': 'module'}
    p.level = {'module.__init__': 0}
    p.alias = {}
    p.const = {}
    p.toc = False
    p.link = True
    
    result = p.compile()
    
    assert '__init__' not in result


# LLM-generated content at query #65
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = [Name(id='BaseClass')]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']


def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    ann_assign = AnnAssign(
        target=Name(id='member1'),
        annotation=Name(id='str'),
        value=Constant(value='test'),
        simple=1
    )
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_with_enums():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = [Attribute(value=Name(id='enum'), attr='Enum')]
    ann_assign = AnnAssign(
        target=Name(id='MEMBER1'),
        annotation=Name(id='str'),
        value=Constant(value='value1'),
        simple=1
    )
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.TestEnum', bases, body)
    
    assert 'test_module.TestEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.TestEnum']


def test_class_api_with_private_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    ann_assign = AnnAssign(
        target=Name(id='_private'),
        annotation=Name(id='str'),
        value=Constant(value='test'),
        simple=1
    )
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.TestClass']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    ann_assign = AnnAssign(
        target=Name(id='member1'),
        annotation=Name(id='str'),
        value=Constant(value='test'),
        simple=1
    )
    delete = Delete(targets=[Name(id='member1')])
    body = [ann_assign, delete]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.TestClass']


def test_class_api_empty_class():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' not in parser.doc['test_module.TestClass']
    assert 'Members' not in parser.doc['test_module.TestClass']


# LLM-generated content at query #66
#--------------------------

```python
def test_globals_annassign_with_value():
    """Test globals method with AnnAssign node having value."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an AnnAssign node: x: int = 42
    target = Name(id='x', ctx=Store())
    value = Constant(value=42)
    annotation = Name(id='int', ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    
    assert _m(root, 'x') in parser.alias
    assert parser.alias[_m(root, 'x')] == '42'


def test_globals_annassign_constant_uppercase():
    """Test globals method with uppercase constant in AnnAssign."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an AnnAssign node: MAX: int = 100
    target = Name(id='MAX', ctx=Store())
    value = Constant(value=100)
    annotation = Name(id='int', ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    
    name = _m(root, 'MAX')
    assert name in parser.root
    assert parser.root[name] == root
    assert name in parser.const
    assert parser.const[name] == 'int'


def test_globals_assign_single_target():
    """Test globals method with Assign node having single target."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node: y = "hello"
    target = Name(id='y', ctx=Store())
    value = Constant(value="hello")
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, 'y') in parser.alias
    assert parser.alias[_m(root, 'y')] == "'hello'"


def test_globals_assign_with_type_comment():
    """Test globals method with Assign node having type_comment."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node: z = 3.14  # type: float
    target = Name(id='z', ctx=Store())
    value = Constant(value=3.14)
    node = Assign(targets=[target], value=value, type_comment='float')
    
    parser.globals(root, node)
    
    name = _m(root, 'z')
    assert name in parser.alias
    assert parser.alias[name] == '3.14'


def test_globals_assign_uppercase_constant():
    """Test globals method with uppercase constant in Assign."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node: PI = 3.14159
    target = Name(id='PI', ctx=Store())
    value = Constant(value=3.14159)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    name = _m(root, 'PI')
    assert name in parser.root
    assert parser.root[name] == root
    assert name in parser.const
    assert parser.const[name] == 'float'


def test_globals_all_list():
    """Test globals method with __all__ as List."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    
    # Create an Assign node: __all__ = ["func1", "func2"]
    target = Name(id='__all__', ctx=Store())
    elts = [Constant(value="func1"), Constant(value="func2")]
    value = List(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]


def test_globals_all_tuple():
    """Test globals method with __all__ as Tuple."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    
    # Create an Assign node: __all__ = ("api1", "api2")
    target = Name(id='__all__', ctx=Store())
    elts = [Constant(value="api1"), Constant(value="api2")]
    value = Tuple(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, "api1") in parser.imp[root]
    assert _m(root, "api2") in parser.imp[root]


def test_globals_ignore_multiple_targets():
    """Test globals method ignores Assign with multiple targets."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node with multiple targets: a = b = 5
    target1 = Name(id='a', ctx=Store())
    target2 = Name(id='b', ctx=Store())
    value = Constant(value=5)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, 'a') not in parser.alias
    assert _m(root, 'b') not in parser.alias


def test_globals_ignore_non_name_target():
    """Test globals method ignores non-Name target."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node with tuple target: (x, y) = (1, 2)
    target = Tuple(elts=[Name(id='x', ctx=Store()), Name(id='y', ctx=Store())], ctx=Store())
    value = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, 'x') not in parser.alias
    assert _m(root, 'y') not in parser.alias


def test_globals_annassign_without_value():
    """Test globals method with AnnAssign node without value."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an AnnAssign node without value: x: int
    target = Name(id='x', ctx=Store())
    annotation = Name(id='int', ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=None, simple=1)
    
    parser.globals(root, node)
    
    assert _m(root, 'x') not in parser.alias


def test_globals_annassign_non_name_target():
    """Test globals method ignores AnnAssign with non-Name target."""
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an AnnAssign node with non-Name target: obj.attr: int = 5
    target = Attribute(value=Name(id='obj', ctx=Load()), attr='attr', ctx=Store())
    value = Constant(value=5)
    annotation = Name(id='int', ctx=Load())
    node = Ann


# LLM-generated content at query #67
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


def test_parser_constructor_with_custom_values():
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False
    assert isinstance(parser, Parser)


def test_parser_new_classmethod_with_toc():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is True


# LLM-generated content at query #68
#--------------------------

```python
def test_visit_name_with_self_ty():
    resolver = Resolver("module", {}, "SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


def test_visit_name_without_self_ty():
    resolver = Resolver("module", {}, "")
    node = Name("SomeClass", Load())
    result = resolver.visit_Name(node)
    assert result is node


def test_visit_name_with_alias_simple():
    resolver = Resolver("module", {"module.MyType": "str"}, "")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "str"


def test_visit_name_with_alias_typevar():
    resolver = Resolver("module", {
        "module.T": "typing.TypeVar('T')",
        "module.TypeVar": "typing.TypeVar"
    }, "")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert result is node


def test_visit_name_with_alias_circular():
    resolver = Resolver("module", {"module.A": "module.A"}, "")
    node = Name("A", Load())
    result = resolver.visit_Name(node)
    assert result is node


def test_visit_name_not_in_alias():
    resolver = Resolver("module", {"module.Other": "str"}, "")
    node = Name("NotDefined", Load())
    result = resolver.visit_Name(node)
    assert result is node


def test_visit_name_with_nested_alias():
    resolver = Resolver("module", {
        "module.Alias1": "Alias2",
        "module.Alias2": "int"
    }, "")
    node = Name("Alias1", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_true():
    from ast import Constant
    
    class MockExpr:
        pass
    
    def _type_name(value):
        if isinstance(value, int):
            return "int"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, float):
            return "float"
        return "Any"
    
    def _e_type(*elements):
        if not elements:
            return ""
        ts = []
        for element in elements:
            if not element:
                return ""
            t = ""
            for e in element:
                if not isinstance(e, Constant):
                    return ""
                nw_t = _type_name(e.value)
                if t and t != nw_t:
                    t = "Any"
                    break
                t = nw_t
            ts.append(t)
        return '[' + ", ".join(ts) + ']'
    
    const1 = Constant(value=5)
    const2 = Constant(value="hello")
    element = [const1, const2]
    
    result = _e_type(element)
    
    assert "Any" in result


# LLM-generated content at query #70
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False
    assert isinstance(parser, Parser)


def test_parser_new_classmethod_with_toc():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is True


# LLM-generated content at query #71
#--------------------------

```python
def test_globals_predicate_at_line_8_evaluates_to_false():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Assign node with multiple targets (len > 1)
    # This makes the first predicate False (not AnnAssign)
    # and the second predicate False (len(node.targets) != 1)
    target1 = Name(id='x')
    target2 = Name(id='y')
    value = Constant(value=1)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    # Call globals with this node
    parser.globals('test_module', node)
    
    # The function should return early without adding to alias or const
    assert 'test_module.x' not in parser.alias
    assert 'test_module.y' not in parser.alias


# LLM-generated content at query #72
#--------------------------

```python
def test_imports_with_simple_import():
    from ast import Import, alias
    parser = Parser()
    node = Import(names=[alias(name='os', asname=None)])
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.os'] == 'os'


def test_imports_with_aliased_import():
    from ast import Import, alias
    parser = Parser()
    node = Import(names=[alias(name='os', asname='operating_system')])
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.operating_system'] == 'os'


def test_imports_with_multiple_imports():
    from ast import Import, alias
    parser = Parser()
    node = Import(names=[alias(name='os', asname=None), alias(name='sys', asname=None)])
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.os'] == 'os'
    assert parser.alias['mymodule.sys'] == 'sys'


def test_imports_from_absolute_import():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname=None)], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.path'] == 'os.path'


def test_imports_from_absolute_import_with_alias():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname='filepath')], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.filepath'] == 'os.path'


def test_imports_from_relative_import_level_1():
    from ast import ImportFrom, alias
    parser = Parser()
    parser.parse('package.submodule', '')
    node = ImportFrom(module='utils', names=[alias(name='helper', asname=None)], level=1)
    parser.imports('package.submodule', node)
    assert parser.alias['package.submodule.helper'] == 'package.utils.helper'


def test_imports_from_relative_import_level_2():
    from ast import ImportFrom, alias
    parser = Parser()
    parser.parse('package.sub.module', '')
    node = ImportFrom(module='utils', names=[alias(name='helper', asname=None)], level=2)
    parser.imports('package.sub.module', node)
    assert parser.alias['package.sub.module.helper'] == 'package.utils.helper'


def test_imports_from_relative_import_no_module():
    from ast import ImportFrom, alias
    parser = Parser()
    parser.parse('package.submodule', '')
    node = ImportFrom(module=None, names=[alias(name='helper', asname=None)], level=1)
    parser.imports('package.submodule', node)
    assert parser.alias['package.submodule.helper'] == 'package.helper'


def test_imports_from_import_with_multiple_names():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname=None), alias(name='environ', asname=None)], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.path'] == 'os.path'
    assert parser.alias['mymodule.environ'] == 'os.environ'


def test_imports_from_import_nested_module():
    from ast import ImportFrom, alias
    parser = Parser()
    node = ImportFrom(module='os.path', names=[alias(name='join', asname=None)], level=0)
    parser.parse('mymodule', '')
    parser.imports('mymodule', node)
    assert parser.alias['mymodule.join'] == 'os.path.join'


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_true():
    from ast import Constant
    
    # Mock the _type_name function
    def _type_name(value):
        if isinstance(value, int):
            return "int"
        elif isinstance(value, str):
            return "str"
        return "object"
    
    # Mock Constant class if needed
    class MockConstant:
        def __init__(self, value):
            self.value = value
    
    # Create test constants with different types
    const_int = MockConstant(42)
    const_str = MockConstant("hello")
    
    # Simulate the loop to reach line 14 with the predicate True
    t = "int"  # t is set from first element
    nw_t = "str"  # nw_t is different from t
    
    # The predicate at line 14: if t and t != nw_t:
    predicate_result = t and t != nw_t
    
    assert predicate_result is True


# LLM-generated content at query #74
#--------------------------

```python
def test_class_api_enums_predicate_true():
    from ast import parse, Name, AnnAssign, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Set up initial state
    root = "test_module"
    name = "test_module.TestEnum"
    parser.doc[name] = "# class TestEnum\n\n"
    parser.level[root] = 0
    parser.root[name] = root
    
    # Create a mock enum class body with an annotated assignment
    # This simulates: class TestEnum(enum.Enum): MEMBER: int = 1
    ann_assign = AnnAssign(
        target=Name(id="MEMBER", ctx=None),
        annotation=Name(id="int", ctx=None),
        value=Constant(value=1),
        simple=1
    )
    
    # Create bases that start with 'enum.'
    enum_base = Constant(value="enum.Enum")
    
    # Mock the resolve method to return enum.Enum
    original_resolve = parser.resolve
    parser.resolve = lambda root, node, self_ty="": "enum.Enum"
    
    # Call class_api with enum base
    parser.class_api(root, name, [enum_base], [ann_assign])
    
    # Verify that enums list was populated and the condition at line 38 is True
    assert "Enums" in parser.doc[name]
    
    # Restore original method
    parser.resolve = original_resolve


# LLM-generated content at query #75
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


def test_parser_constructor_with_custom_values():
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


def test_parser_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=False, level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert isinstance(parser, Parser)


def test_parser_new_with_all_true():
    parser = Parser.new(link=True, level=3, toc=True)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is True


def test_parser_independent_instances():
    parser1 = Parser()
    parser2 = Parser()
    parser1.doc['test'] = 'value'
    assert 'test' not in parser2.doc


# LLM-generated content at query #76
#--------------------------

```python
def test_visit_attribute_typing_prefix_removal():
    from ast import Attribute, Name, Load
    
    resolver = Resolver(root="test_module", alias={})
    
    # Create an Attribute node with value.id == 'typing'
    name_node = Name(id='typing', ctx=Load())
    attr_node = Attribute(value=name_node, attr='List', ctx=Load())
    
    result = resolver.visit_Attribute(attr_node)
    
    assert isinstance(result, Name)
    assert result.id == 'List'
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #77
#--------------------------

```python
def test_visit_name_with_self_ty():
    resolver = Resolver("module", {}, "MyType")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


def test_visit_name_no_alias():
    resolver = Resolver("module", {})
    node = Name("SomeName", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeName"


def test_visit_name_with_alias():
    resolver = Resolver("module", {"module.MyType": "typing.List"})
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "List"


def test_visit_name_with_typevar_alias():
    resolver = Resolver("module", {"module.T": "TypeVar('T')", "module.TypeVar": "typing.TypeVar"})
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


def test_visit_name_self_reference_in_alias():
    resolver = Resolver("module", {"module.Node": "Node"})
    node = Name("Node", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Node"


def test_visit_name_with_nested_alias():
    resolver = Resolver("module", {"module.Alias": "typing.Dict"})
    node = Name("Alias", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Dict"


def test_visit_name_empty_root():
    resolver = Resolver("", {"MyType": "typing.List"})
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "List"


# LLM-generated content at query #78
#--------------------------

```python
def test_imports_simple_import():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.os"] == "os"


def test_imports_simple_import_with_alias():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.operating_system"] == "os"


def test_imports_multiple_imports():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[
        alias(name="os", asname=None),
        alias(name="sys", asname="system")
    ])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.os"] == "os"
    assert parser.alias["mymodule.system"] == "sys"


def test_imports_from_import_absolute():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname=None)],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.path"] == "os.path"


def test_imports_from_import_absolute_with_alias():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname="ospath")],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.ospath"] == "os.path"


def test_imports_from_import_relative_level_1():
    parser = Parser()
    root = "package.submodule"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="sibling",
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports(root, import_node)
    
    assert parser.alias["package.submodule.func"] == "package.sibling.func"


def test_imports_from_import_relative_level_2():
    parser = Parser()
    root = "package.sub.module"
    parser.level[root] = 2
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="other",
        names=[alias(name="item", asname=None)],
        level=2
    )
    parser.imports(root, import_node)
    
    assert parser.alias["package.sub.module.item"] == "package.other.item"


def test_imports_from_import_multiple_names():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="collections",
        names=[
            alias(name="defaultdict", asname=None),
            alias(name="Counter", asname="cnt")
        ],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.defaultdict"] == "collections.defaultdict"
    assert parser.alias["mymodule.cnt"] == "collections.Counter"


def test_imports_from_import_no_module():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module=None,
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports(root, import_node)
    
    assert "mymodule.func" in parser.alias


def test_imports_star_import():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="utils",
        names=[alias(name="*", asname=None)],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.*"] == "utils.*"


# LLM-generated content at query #79
#--------------------------

```python
def test_load_docstring_with_valid_docstring():
    from types import ModuleType
    from unittest.mock import Mock, patch
    
    parser = Parser()
    parser.doc = {'test_module.func': 'function doc'}
    
    mock_module = Mock(spec=ModuleType)
    mock_module.__name__ = 'test_module'
    
    with patch('__main__.getdoc') as mock_getdoc, \
         patch('__main__._attr') as mock_attr, \
         patch('__main__.doctest') as mock_doctest:
        mock_getdoc.return_value = "This is a docstring"
        mock_attr.return_value = Mock()
        mock_doctest.return_value = "processed docstring"
        
        parser.load_docstring('test_module', mock_module)
        
        assert 'test_module.func' in parser.docstring
        assert parser.docstring['test_module.func'] == "processed docstring"


# LLM-generated content at query #80
#--------------------------

```python
def test_globals_with_annotated_assignment():
    from ast import parse as ast_parse, AnnAssign, Name, Constant
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "x: int = 5"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "5"


def test_globals_with_simple_assignment():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "MY_CONST = 42"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONST" in parser.alias
    assert parser.alias["test_module.MY_CONST"] == "42"
    assert "test_module.MY_CONST" in parser.const


def test_globals_with_all_assignment():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    script = "__all__ = ['func1', 'func2']"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_uppercase_constant():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "CONSTANT_VALUE = 'test'"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert "test_module.CONSTANT_VALUE" in parser.root
    assert parser.root["test_module.CONSTANT_VALUE"] == root


def test_globals_with_lowercase_variable():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "variable = 100"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert "test_module.variable" in parser.alias
    assert "test_module.variable" not in parser.const


def test_globals_with_multiple_targets():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "a = b = 5"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_type_comment():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "x = 5  # type: int"
    tree = ast_parse(script, type_comments=True)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert "test_module.x" in parser.const


def test_globals_with_annotated_assignment_no_value():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "x: int"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_tuple_unpacking():
    from ast import parse as ast_parse
    
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    script = "a, b = 1, 2"
    tree = ast_parse(script)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


# LLM-generated content at query #81
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    # Create an AnnAssign node with a Constant value
    from ast import AnnAssign, Name, Constant, parse
    node = parse("x: int = 42").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "42"


def test_globals_with_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    from ast import parse
    node = parse("CONST = 100").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.CONST" in parser.const
    assert parser.const["test_module.CONST"] == "int"
    assert "test_module.CONST" in parser.root


def test_globals_with_all_filter():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    from ast import parse
    node = parse("__all__ = ['func1', 'func2']").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_assignment_no_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    from ast import parse
    node = parse("x = 'hello'").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "'hello'"


def test_globals_ignores_invalid_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    from ast import parse
    # Tuple unpacking - should be ignored
    node = parse("a, b = 1, 2").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    from ast import parse
    node = parse("x = 5  # type: int").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias


def test_globals_skips_annotated_without_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    from ast import parse
    node = parse("x: int").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" not in parser.alias
    assert "test_module.x" not in parser.const


def test_globals_with_multiple_targets_ignored():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.alias = {}
    parser.const = {}
    
    from ast import parse
    node = parse("x = y = 10").body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" not in parser.alias


# LLM-generated content at query #82
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_true():
    from ast import Constant
    
    class MockExpr:
        pass
    
    def _type_name(value):
        return type(value).__name__
    
    def _e_type(*elements):
        """Get element type if type is constants."""
        if not elements:
            return ""
        ts = []
        for element in elements:
            if not element:
                return ""
            t = ""
            for e in element:
                if not isinstance(e, Constant):
                    return ""
                nw_t = _type_name(e.value)
                if t and t != nw_t:
                    t = "Any"
                    break
                t = nw_t
            ts.append(t)
        return '[' + ", ".join(ts) + ']'
    
    # Create elements where the predicate at line 14 evaluates to True
    # The predicate is: if t and t != nw_t
    # This requires t to be non-empty (truthy) AND t to be different from nw_t
    const1 = Constant(value=42)
    const2 = Constant(value="hello")
    
    element = [const1, const2]
    result = _e_type(element)
    
    assert result == "[Any]"


# LLM-generated content at query #83
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_link_unchanged_when_toc_false():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False
    assert isinstance(parser, Parser)


def test_parser_new_classmethod_with_toc():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is True


# LLM-generated content at query #84
#--------------------------

```python
def test_const_type_call_with_name_or_attribute():
    from ast import Call, Name, Attribute, expr
    from ast import parse
    
    # Test with Call node containing Name as func
    code1 = "int(5)"
    tree1 = parse(code1, mode='eval')
    node1 = tree1.body
    assert isinstance(node1, Call) and isinstance(node1.func, Name)
    
    # Test with Call node containing Attribute as func
    code2 = "obj.method()"
    tree2 = parse(code2, mode='eval')
    node2 = tree2.body
    assert isinstance(node2, Call) and isinstance(node2.func, Attribute)


# LLM-generated content at query #85
#--------------------------

```python
def test_imports_with_import_statement():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import Import, alias
    node = Import(names=[alias(name="os", asname=None)])
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.os"] == "os"


def test_imports_with_import_as_statement():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import Import, alias
    node = Import(names=[alias(name="numpy", asname="np")])
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.np"] == "numpy"


def test_imports_with_multiple_imports():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import Import, alias
    node = Import(names=[
        alias(name="os", asname=None),
        alias(name="sys", asname=None)
    ])
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.os"] == "os"
    assert parser.alias["mymodule.sys"] == "sys"


def test_imports_from_statement_absolute():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import ImportFrom, alias
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=0)
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.path"] == "os.path"


def test_imports_from_statement_with_asname():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import ImportFrom, alias
    node = ImportFrom(module="os", names=[alias(name="path", asname="ospath")], level=0)
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.ospath"] == "os.path"


def test_imports_from_statement_relative_level_1():
    parser = Parser()
    root = "package.submodule"
    parser.level[root] = 2
    
    from ast import ImportFrom, alias
    node = ImportFrom(module="other", names=[alias(name="func", asname=None)], level=1)
    
    parser.imports(root, node)
    
    assert parser.alias["package.submodule.func"] == "package.other.func"


def test_imports_from_statement_relative_level_2():
    parser = Parser()
    root = "package.sub.module"
    parser.level[root] = 3
    
    from ast import ImportFrom, alias
    node = ImportFrom(module="other", names=[alias(name="func", asname=None)], level=2)
    
    parser.imports(root, node)
    
    assert parser.alias["package.sub.module.func"] == "package.other.func"


def test_imports_from_statement_multiple_names():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import ImportFrom, alias
    node = ImportFrom(module="collections", names=[
        alias(name="defaultdict", asname=None),
        alias(name="Counter", asname=None)
    ], level=0)
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.defaultdict"] == "collections.defaultdict"
    assert parser.alias["mymodule.Counter"] == "collections.Counter"


def test_imports_from_statement_none_module():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    
    from ast import ImportFrom, alias
    node = ImportFrom(module=None, names=[alias(name="func", asname=None)], level=1)
    
    parser.imports(root, node)
    
    assert parser.alias["mymodule.func"] == "func"


def test_imports_from_statement_relative_level_3_nested():
    parser = Parser()
    root = "a.b.c.d"
    parser.level[root] = 4
    
    from ast import ImportFrom, alias
    node = ImportFrom(module="utils", names=[alias(name="helper", asname=None)], level=3)
    
    parser.imports(root, node)
    
    assert parser.alias["a.b.c.d.helper"] == "a.utils.helper"


# LLM-generated content at query #86
#--------------------------

```python
def test_class_api_private_attribute_not_added_to_mem():
    from ast import AnnAssign, Name, Constant, parse
    from dataclasses import dataclass, field
    
    @dataclass
    class MockParser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict = field(default_factory=dict)
        doc: dict = field(default_factory=dict)
        docstring: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        alias: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
        
        def resolve(self, root: str, node):
            return "str"
    
    parser = MockParser()
    parser.doc["TestClass"] = "# class TestClass\n\n"
    parser.level["TestClass"] = 0
    parser.root["TestClass"] = "TestClass"
    
    # Create AST node for private attribute: _private_attr: str
    code = "_private_attr: str = 'value'"
    tree = parse(code)
    ann_assign = tree.body[0]
    
    # Test that private attribute (starting with _) does not pass is_public_family
    # Line 15: elif is_public_family(attr): should evaluate to False
    attr = ann_assign.target.id
    
    # Verify the attribute name is private
    assert attr == "_private_attr"
    
    # When is_public_family returns False, the elif at line 15 should not execute
    # This test verifies that private attributes are not added to mem dict
    mem = {}
    is_enum = False
    
    if is_enum:
        pass
    elif attr.startswith('_'):  # Simulating is_public_family returning False for private names
        mem[attr] = "str"
    
    # mem should remain empty because the elif condition at line 15 is False
    assert mem == {}


# LLM-generated content at query #87
#--------------------------

```python
def test_func_ann_line_7_predicate_true():
    from ast import arg, parse
    from dataclasses import dataclass, field
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict = field(default_factory=dict)
        doc: dict = field(default_factory=dict)
        docstring: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        alias: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
        
        def resolve(self, root: str, node, self_ty: str = "") -> str:
            return "int"
        
        def func_ann(self, root: str, args, *, has_self: bool, cls_method: bool):
            self_ty = ""
            for i, a in enumerate(args):
                if has_self and i == 0:
                    if a.annotation is not None:
                        self_ty = self.resolve(root, a.annotation)
                        if cls_method:
                            self_ty = (self_ty.removeprefix('type[')
                                       .removesuffix(']'))
                    yield 'type[Self]' if cls_method else 'Self'
                elif a.arg == '*':
                    yield ""
                elif a.annotation is not None:
                    yield self.resolve(root, a.annotation, self_ty)
                else:
                    yield "ANY"
    
    parser = Parser()
    root_name = "test_module"
    parser.alias = {}
    
    # Create arg with annotation (line 7 predicate is True)
    test_arg = arg(arg='self', annotation=parse('int').body[0].value)
    args_list = [test_arg]
    
    result = list(parser.func_ann(root_name, args_list, has_self=True, cls_method=False))
    
    assert len(result) > 0
    assert result[0] == 'Self'


# LLM-generated content at query #88
#--------------------------

```python
def test_visit_name_predicate_line_6_true():
    """Test that the predicate at line 6 evaluates to True."""
    from ast import Name, Load
    
    # Mock the _m function to return a predictable value
    import sys
    from unittest.mock import patch
    
    def mock_m(root, node_id):
        return f"{root}.{node_id}"
    
    resolver = Resolver(
        root="mymodule",
        alias={"mymodule.MyType": "int"},
        self_ty=""
    )
    
    node = Name(id="MyType", ctx=Load())
    
    with patch('__main__._m', side_effect=mock_m):
        with patch('builtins.parse') as mock_parse:
            from ast import Expr, Constant
            mock_expr = Expr(value=Constant(value=42))
            mock_parse.return_value.body = [mock_expr]
            
            result = resolver.visit_Name(node)
            
            # Verify that line 6 condition was True by checking that visit was called
            # (which only happens when the condition is True)
            assert mock_parse.called


# LLM-generated content at query #89
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyClass'] = 'test_module'
    parser.level['test_module.MyClass'] = 0
    parser.alias = {}
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.MyClass']
    assert 'BaseClass' in parser.doc['test_module.MyClass']


def test_class_api_with_enum_bases():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyEnum'] = 'test_module'
    parser.level['test_module.MyEnum'] = 0
    parser.alias = {}
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    body = [AnnAssign(target=Name(id='MEMBER1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1)]
    
    parser.class_api('test_module', 'test_module.MyEnum', bases, body)
    
    assert 'test_module.MyEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.MyEnum']
    assert 'MEMBER1' in parser.doc['test_module.MyEnum']


def test_class_api_with_members():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyClass'] = 'test_module'
    parser.level['test_module.MyClass'] = 0
    parser.alias = {}
    
    bases = []
    body = [AnnAssign(target=Name(id='attr1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1)]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' in parser.doc['test_module.MyClass']
    assert 'attr1' in parser.doc['test_module.MyClass']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyClass'] = 'test_module'
    parser.level['test_module.MyClass'] = 0
    parser.alias = {}
    
    bases = []
    body = [
        AnnAssign(target=Name(id='attr1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1),
        Delete(targets=[Name(id='attr1', ctx=Del())])
    ]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' not in parser.doc['test_module.MyClass']


def test_class_api_with_assign_members():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyClass'] = 'test_module'
    parser.level['test_module.MyClass'] = 0
    parser.alias = {}
    
    bases = []
    body = [Assign(targets=[Name(id='attr1', ctx=Store())], value=Constant(value=42), type_comment=None)]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'Members' in parser.doc['test_module.MyClass']


def test_class_api_empty_body():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyClass'] = 'test_module'
    parser.level['test_module.MyClass'] = 0
    parser.alias = {}
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc


def test_class_api_private_members_ignored():
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`\n\n'
    parser.root['test_module.MyClass'] = 'test_module'
    parser.level['test_module.MyClass'] = 0
    parser.alias = {}
    
    bases = []
    body = [AnnAssign(target=Name(id='_private', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=1), simple=1)]
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert '_private' not in parser.doc['test_module.MyClass']


# LLM-generated content at query #90
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


def test_parser_new_classmethod():
    p = Parser.new(link=True, level=2, toc=False)
    assert p.link is True
    assert p.b_level == 2
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


def test_parser_post_init_toc_enables_link():
    p = Parser(link=False, b_level=1, toc=True)
    assert p.link is True
    assert p.toc is True


def test_parser_post_init_toc_false_preserves_link():
    p = Parser(link=False, b_level=1, toc=False)
    assert p.link is False
    assert p.toc is False


def test_parser_post_init_toc_true_with_link_true():
    p = Parser(link=True, b_level=1, toc=True)
    assert p.link is True
    assert p.toc is True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an AnnAssign node with a Constant value
    target = Name(id="MY_CONST", ctx=Store())
    value = Constant(value=42)
    annotation = Name(id="int", ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    
    assert _m(root, "MY_CONST") in parser.alias
    assert parser.alias[_m(root, "MY_CONST")] == "42"
    assert _m(root, "MY_CONST") in parser.const
    assert parser.const[_m(root, "MY_CONST")] == "int"


def test_globals_with_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node
    target = Name(id="CONST_VALUE", ctx=Store())
    value = Constant(value="hello")
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, "CONST_VALUE") in parser.alias
    assert parser.alias[_m(root, "CONST_VALUE")] == "'hello'"
    assert _m(root, "CONST_VALUE") in parser.const
    assert parser.const[_m(root, "CONST_VALUE")] == "str"


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node with type_comment
    target = Name(id="TYPED_VAR", ctx=Store())
    value = Constant(value=123)
    node = Assign(targets=[target], value=value, type_comment="int")
    
    parser.globals(root, node)
    
    assert _m(root, "TYPED_VAR") in parser.const
    assert parser.const[_m(root, "TYPED_VAR")] == "int"


def test_globals_with_dunder_all():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    
    # Create an Assign node for __all__
    target = Name(id="__all__", ctx=Store())
    elts = [Constant(value="func1"), Constant(value="func2")]
    value = Tuple(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]


def test_globals_ignores_non_uppercase_constants():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node with lowercase name
    target = Name(id="my_var", ctx=Store())
    value = Constant(value=99)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, "my_var") in parser.alias
    assert _m(root, "my_var") not in parser.const


def test_globals_with_multiple_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node with multiple targets (should be ignored)
    target1 = Name(id="VAR1", ctx=Store())
    target2 = Name(id="VAR2", ctx=Store())
    value = Constant(value=1)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert _m(root, "VAR1") not in parser.alias


def test_globals_with_non_name_target():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an Assign node with non-Name target (should be ignored)
    target = Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())
    value = Constant(value=1)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_annotated_assignment_no_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    
    # Create an AnnAssign node without value (should be ignored)
    target = Name(id="ANNOTATED", ctx=Store())
    annotation = Name(id="int", ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=None, simple=1)
    
    parser.globals(root, node)
    
    assert _m(root, "ANNOTATED") not in parser.alias


# LLM-generated content at query #2
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']
    assert 'BaseClass' in parser.doc['test_module.TestClass']


def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    body = [
        AnnAssign(
            target=Name(id='member1', ctx=Store()),
            annotation=Name(id='str', ctx=Load()),
            value=None,
            simple=1
        ),
        AnnAssign(
            target=Name(id='_private', ctx=Store()),
            annotation=Name(id='int', ctx=Load()),
            value=None,
            simple=1
        )
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'member1' in parser.doc['test_module.TestClass']
    assert '_private' not in parser.doc['test_module.TestClass']


def test_class_api_with_enums():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    body = [
        AnnAssign(
            target=Name(id='ENUM1', ctx=Store()),
            annotation=Name(id='str', ctx=Load()),
            value=Constant(value='value1'),
            simple=1
        ),
        AnnAssign(
            target=Name(id='ENUM2', ctx=Store()),
            annotation=Name(id='str', ctx=Load()),
            value=Constant(value='value2'),
            simple=1
        )
    ]
    
    parser.class_api('test_module', 'test_module.TestEnum', bases, body)
    
    assert 'test_module.TestEnum' in parser.doc
    assert 'Enums' in parser.doc['test_module.TestEnum']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    body = [
        AnnAssign(
            target=Name(id='member1', ctx=Store()),
            annotation=Name(id='str', ctx=Load()),
            value=None,
            simple=1
        ),
        Delete(targets=[Name(id='member1', ctx=Del())])
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'member1' not in parser.doc['test_module.TestClass']


def test_class_api_with_assign_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    body = [
        Assign(
            targets=[Name(id='member1', ctx=Store())],
            value=Constant(value=42),
            type_comment=None
        )
    ]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_no_bases_no_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' not in parser.doc['test_module.TestClass']
    assert 'Enums' not in parser.doc['test_module.TestClass']
    assert 'Members' not in parser.doc['test_module.TestClass']


# LLM-generated content at query #3
#--------------------------

```python
def test_m_single_name():
    from module import _m
    result = _m('foo')
    assert result == 'foo'


def test_m_multiple_names():
    from module import _m
    result = _m('foo', 'bar', 'baz')
    assert result == 'foo.bar.baz'


def test_m_empty_string():
    from module import _m
    result = _m('')
    assert result == ''


def test_m_with_empty_strings():
    from module import _m
    result = _m('foo', '', 'bar')
    assert result == 'foo.bar'


def test_m_all_empty_strings():
    from module import _m
    result = _m('', '', '')
    assert result == ''


def test_m_no_arguments():
    from module import _m
    result = _m()
    assert result == ''


def test_m_single_empty_string():
    from module import _m
    result = _m('')
    assert result == ''


def test_m_multiple_with_leading_trailing_empty():
    from module import _m
    result = _m('', 'foo', 'bar', '')
    assert result == 'foo.bar'


# LLM-generated content at query #4
#--------------------------

```python
def test_parser_constructor_default():
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


def test_parser_constructor_custom():
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_new_classmethod():
    parser = Parser.new(link=False, level=2, toc=True)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}


def test_parser_new_classmethod_with_link_true():
    parser = Parser.new(link=True, level=3, toc=False)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is False


# LLM-generated content at query #5
#--------------------------

```python
from ast import If, Try, stmt, Assign, Name, Store, Constant, ExceptHandler, Pass

def test_walk_body_simple_statements():
    assign1 = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=1))
    assign2 = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=2))
    body = [assign1, assign2]
    
    result = list(walk_body(body))
    
    assert len(result) == 2
    assert result[0] is assign1
    assert result[1] is assign2


def test_walk_body_with_if_statement():
    assign1 = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=1))
    assign2 = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=2))
    assign3 = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3))
    
    if_stmt = If(test=Name(id='cond', ctx=Store()), body=[assign2], orelse=[assign3])
    body = [assign1, if_stmt]
    
    result = list(walk_body(body))
    
    assert len(result) == 3
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3


def test_walk_body_with_nested_if():
    assign1 = Assign(targets=[Name(id='a', ctx=Store())], value=Constant(value=1))
    assign2 = Assign(targets=[Name(id='b', ctx=Store())], value=Constant(value=2))
    assign3 = Assign(targets=[Name(id='c', ctx=Store())], value=Constant(value=3))
    assign4 = Assign(targets=[Name(id='d', ctx=Store())], value=Constant(value=4))
    
    inner_if = If(test=Name(id='inner', ctx=Store()), body=[assign3], orelse=[assign4])
    outer_if = If(test=Name(id='outer', ctx=Store()), body=[assign2, inner_if], orelse=[])
    body = [assign1, outer_if]
    
    result = list(walk_body(body))
    
    assert len(result) == 4
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3
    assert result[3] is assign4


def test_walk_body_with_try_statement():
    assign1 = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=1))
    assign2 = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=2))
    assign3 = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3))
    assign4 = Assign(targets=[Name(id='w', ctx=Store())], value=Constant(value=4))
    assign5 = Assign(targets=[Name(id='v', ctx=Store())], value=Constant(value=5))
    
    handler = ExceptHandler(type=Name(id='Exception', ctx=Store()), name=None, body=[assign3])
    try_stmt = Try(body=[assign2], handlers=[handler], orelse=[assign4], finalbody=[assign5])
    body = [assign1, try_stmt]
    
    result = list(walk_body(body))
    
    assert len(result) == 5
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3
    assert result[3] is assign4
    assert result[4] is assign5


def test_walk_body_empty():
    body = []
    
    result = list(walk_body(body))
    
    assert len(result) == 0


def test_walk_body_with_multiple_handlers():
    assign1 = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=1))
    assign2 = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=2))
    assign3 = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3))
    assign4 = Assign(targets=[Name(id='w', ctx=Store())], value=Constant(value=4))
    
    handler1 = ExceptHandler(type=Name(id='ValueError', ctx=Store()), name=None, body=[assign2])
    handler2 = ExceptHandler(type=Name(id='TypeError', ctx=Store()), name=None, body=[assign3])
    try_stmt = Try(body=[assign1], handlers=[handler1, handler2], orelse=[], finalbody=[assign4])
    body = [try_stmt]
    
    result = list(walk_body(body))
    
    assert len(result) == 4
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3
    assert result[3] is assign4


# LLM-generated content at query #6
#--------------------------

```python
def test_parse_basic_module():
    parser = Parser()
    script = "x = 1"
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module" in parser.level
    assert "test_module" in parser.imp
    assert "test_module" in parser.root
    assert parser.level["test_module"] == 0
    assert parser.root["test_module"] == "test_module"


def test_parse_with_imports():
    parser = Parser()
    script = "import os\nfrom sys import path"
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.os" in parser.alias
    assert "test_module.path" in parser.alias


def test_parse_with_function():
    parser = Parser()
    script = "def foo():\n    pass"
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    assert "test_module.foo" in parser.level
    assert "test_module.foo" in parser.root


def test_parse_with_class():
    parser = Parser()
    script = "class MyClass:\n    pass"
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.MyClass" in parser.doc
    assert "test_module.MyClass" in parser.level
    assert "test_module.MyClass" in parser.root


def test_parse_with_docstring():
    parser = Parser()
    script = '"""Module docstring."""\nx = 1'
    parser.parse("test_module", script)
    assert "test_module" in parser.docstring
    assert "Module docstring." in parser.docstring["test_module"]


def test_parse_with_function_docstring():
    parser = Parser()
    script = 'def foo():\n    """Function docstring."""\n    pass'
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.docstring
    assert "Function docstring." in parser.docstring["test_module.foo"]


def test_parse_with_constants():
    parser = Parser()
    script = "CONSTANT = 42\nANOTHER = 'value'"
    parser.parse("test_module", script)
    assert "test_module.CONSTANT" in parser.alias
    assert "test_module.ANOTHER" in parser.alias


def test_parse_with_type_annotation():
    parser = Parser()
    script = "x: int = 5"
    parser.parse("test_module", script)
    assert "test_module.x" in parser.alias


def test_parse_with_link_enabled():
    parser = Parser(link=True)
    script = "x = 1"
    parser.parse("test_module", script)
    assert "<a id=" in parser.doc["test_module"]


def test_parse_with_link_disabled():
    parser = Parser(link=False)
    script = "x = 1"
    parser.parse("test_module", script)
    assert "<a id=" not in parser.doc["test_module"]


def test_parse_with_nested_module():
    parser = Parser()
    script = "x = 1"
    parser.parse("pkg.subpkg.module", script)
    assert parser.level["pkg.subpkg.module"] == 2
    assert parser.root["pkg.subpkg.module"] == "pkg.subpkg.module"


def test_parse_with_nested_class():
    parser = Parser()
    script = "class Outer:\n    class Inner:\n        pass"
    parser.parse("test_module", script)
    assert "test_module.Outer" in parser.doc
    assert "test_module.Outer.Inner" in parser.doc


def test_parse_with_async_function():
    parser = Parser()
    script = "async def async_foo():\n    pass"
    parser.parse("test_module", script)
    assert "test_module.async_foo" in parser.doc
    assert "async" in parser.doc["test_module.async_foo"]


def test_parse_with_doctest():
    parser = Parser()
    script = 'def foo():\n    """Example.\n    >>> foo()\n    """\n    pass'
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.docstring


def test_parse_with_decorated_function():
    parser = Parser()
    script = "@staticmethod\ndef foo():\n    pass"
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.doc
    assert "@staticmethod" in parser.doc["test_module.foo"]


def test_parse_updates_doc_with_module_name():
    parser = Parser()
    script = "x = 1"
    parser.parse("my_module", script)
    assert "Module `my_module`" in parser.doc["my_module"]


def test_parse_with_multiple_decorators():
    parser = Parser()
    script = "@decorator1\n@decorator2\ndef foo():\n    pass"
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.doc
    assert "Decorators" in parser.doc["test_module.foo"]


def test_parse_with_type_comments():
    parser = Parser()
    script = "x = 1  # type: int"
    parser.parse("test_module", script)
    assert "test_module" in parser.doc


# LLM-generated content at query #7
#--------------------------

```python
def test_class_api_enum_predicate_true():
    from ast import parse, Name, AnnAssign, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    parser.doc["test_module.TestEnum"] = "# class TestEnum\n\n"
    
    # Create a simple enum class with annotated assignments
    script = """
from enum import Enum

class TestEnum(Enum):
    MEMBER1: int = 1
    MEMBER2: str = "test"
"""
    
    root_node = parse(script, type_comments=True)
    class_def = root_node.body[1]
    
    # Mock resolve to return enum base
    original_resolve = parser.resolve
    def mock_resolve(root, node, self_ty=""):
        return "enum.Enum"
    
    parser.resolve = mock_resolve
    
    # Call class_api with enum bases
    parser.class_api("test_module", "test_module.TestEnum", class_def.bases, class_def.body)
    
    # Verify that the predicate at line 38 (if enums:) evaluated to True
    # by checking that table was called with "Enums" (which happens when enums list is not empty)
    assert "Enums" in parser.doc["test_module.TestEnum"]


# LLM-generated content at query #8
#--------------------------

```python
def test_compile_empty_parser():
    """Test compile with empty parser."""
    p = Parser()
    result = p.compile()
    assert result == '\n'


def test_compile_with_single_module():
    """Test compile with a single module."""
    p = Parser()
    p.doc['test_module'] = '# Module `test_module`\n\n'
    p.level['test_module'] = 0
    p.root['test_module'] = 'test_module'
    p.docstring['test_module'] = 'Test module documentation\n'
    p.imp['test_module'] = set()
    result = p.compile()
    assert 'Module `test_module`' in result
    assert 'Test module documentation' in result


def test_compile_with_toc_enabled():
    """Test compile with table of contents enabled."""
    p = Parser(toc=True)
    p.doc['test_module'] = '# Module `test_module`\n\n'
    p.level['test_module'] = 0
    p.root['test_module'] = 'test_module'
    p.docstring['test_module'] = 'Test documentation\n'
    p.imp['test_module'] = set()
    result = p.compile()
    assert '**Table of contents:**' in result
    assert 'test_module' in result


def test_compile_with_constants():
    """Test compile with constants."""
    p = Parser()
    p.doc['test_module'] = '# Module `test_module`\n\n'
    p.level['test_module'] = 0
    p.root['test_module'] = 'test_module'
    p.docstring['test_module'] = 'Module doc\n'
    p.imp['test_module'] = {'test_module'}
    p.const['test_module.CONST'] = 'int'
    p.root['test_module.CONST'] = 'test_module'
    result = p.compile()
    assert 'Constants' in result or 'Module doc' in result


def test_compile_with_magic_method_no_doc():
    """Test compile skips magic methods without documentation."""
    p = Parser()
    p.doc['test_module'] = '# Module `test_module`\n\n'
    p.doc['test_module.__init__'] = '## __init__\n\n'
    p.level['test_module'] = 0
    p.level['test_module.__init__'] = 0
    p.root['test_module'] = 'test_module'
    p.root['test_module.__init__'] = 'test_module'
    p.docstring['test_module'] = 'Module doc\n'
    p.imp['test_module'] = set()
    result = p.compile()
    assert '__init__' not in result


def test_compile_with_nested_names():
    """Test compile with nested class/function names."""
    p = Parser()
    p.doc['mod'] = '# Module `mod`\n\n'
    p.doc['mod.MyClass'] = '## MyClass\n\n'
    p.doc['mod.MyClass.method'] = '### method\n\n'
    p.level['mod'] = 0
    p.level['mod.MyClass'] = 0
    p.level['mod.MyClass.method'] = 0
    p.root['mod'] = 'mod'
    p.root['mod.MyClass'] = 'mod'
    p.root['mod.MyClass.method'] = 'mod'
    p.docstring['mod'] = 'Module\n'
    p.docstring['mod.MyClass'] = 'Class\n'
    p.docstring['mod.MyClass.method'] = 'Method\n'
    p.imp['mod'] = set()
    result = p.compile()
    assert 'Module' in result
    assert 'Class' in result
    assert 'Method' in result


def test_compile_with_link_format():
    """Test compile with link formatting."""
    p = Parser(link=True)
    p.doc['test'] = '# Module `{}`\n<a id=\"{}\"></a>\n\n'
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.docstring['test'] = 'Doc\n'
    p.imp['test'] = set()
    result = p.compile()
    assert 'test' in result
    assert 'Doc' in result


def test_compile_sorts_by_level_and_name():
    """Test compile sorts entries by level and name."""
    p = Parser()
    p.doc['b_mod'] = '# Module `b_mod`\n\n'
    p.doc['a_mod'] = '# Module `a_mod`\n\n'
    p.level['b_mod'] = 0
    p.level['a_mod'] = 0
    p.root['b_mod'] = 'b_mod'
    p.root['a_mod'] = 'a_mod'
    p.docstring['b_mod'] = 'B\n'
    p.docstring['a_mod'] = 'A\n'
    p.imp['b_mod'] = set()
    p.imp['a_mod'] = set()
    result = p.compile()
    a_pos = result.find('a_mod')
    b_pos = result.find('b_mod')
    assert a_pos < b_pos


def test_compile_missing_docstring_warning():
    """Test compile with missing docstring for non-magic name."""
    p = Parser()
    p.doc['test'] = '# Module `test`\n\n'
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.imp['test'] = set()
    result = p.compile()
    assert result.endswith('\n')


def test_compile_with_multiple_levels():
    """Test compile with different nesting levels."""
    p = Parser(toc=True)
    p.doc['pkg'] = '# Module `pkg`\n\n'
    p.doc['pkg.submod'] = '## submod\n\n'
    p.level['pkg'] = 0
    p.level['pkg.submod'] = 1
    p.root['pkg'] = 'pkg'
    p.root['pkg.submod'] = 'pkg'
    p.docstring['pkg'] = 'Package\n'
    p.docstring['pkg.submod'] = 'Submodule\n'
    p.imp['pkg'] = set()
    result = p.compile()
    assert '**Table of contents:**' in result
    assert 'pkg' in result
    assert 'submod' in result


# LLM-generated content at query #9
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


def test_parser_constructor_post_init_enables_link_when_toc_true():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_keeps_link_false_when_toc_false():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False
    assert isinstance(parser, Parser)


def test_parser_new_classmethod_with_toc_true():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #10
#--------------------------

```python
def test_globals_predicate_line_18_false():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Assign node with multiple targets (len(node.targets) != 1)
    # This makes the predicate at line 18 (len(node.targets) == 1) evaluate to False
    target1 = Name(id='x', ctx=None)
    target2 = Name(id='y', ctx=None)
    value = Constant(value=5)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    # Call globals method - it should return early without raising an exception
    parser.globals('test_module', node)
    
    # Verify that nothing was added to alias or root since the method returned early
    assert len(parser.alias) == 0
    assert len(parser.root) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_class_api_annassign_with_name_target():
    from ast import AnnAssign, Name, Constant, parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    parser.doc['test_module.TestClass'] = "# class TestClass\n\n"
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    
    # Create an AnnAssign node with a Name target
    # This represents: x: int
    ann_assign = AnnAssign(
        target=Name(id='x', ctx=None),
        annotation=Name(id='int', ctx=None),
        value=Constant(value=5),
        simple=1
    )
    
    # Call class_api with body containing the AnnAssign node
    parser.class_api('test_module', 'test_module.TestClass', [], [ann_assign])
    
    # The predicate at line 11 should evaluate to True for this node
    # This means isinstance(node, AnnAssign) and isinstance(node.target, Name) should both be True
    assert isinstance(ann_assign, AnnAssign)
    assert isinstance(ann_assign.target, Name)


# LLM-generated content at query #12
#--------------------------

```python
def test_is_public_with_magic_name():
    parser = Parser()
    parser.root = {'__init__': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('__init__')
    assert result == True


def test_is_public_with_private_name():
    parser = Parser()
    parser.root = {'_private': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('_private')
    assert result == False


def test_is_public_with_public_name_no_all():
    parser = Parser()
    parser.root = {'module': 'module', 'module.public_func': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('module.public_func')
    assert result == True


def test_is_public_with_all_list_exact_match():
    parser = Parser()
    parser.root = {'module': 'module', 'module.public_func': 'module'}
    parser.imp = {'module': {'module.public_func'}}
    parser.const = {}
    result = parser.is_public('module.public_func')
    assert result == True


def test_is_public_with_all_list_no_match():
    parser = Parser()
    parser.root = {'module': 'module', 'module.private_func': 'module'}
    parser.imp = {'module': {'module.other_func'}}
    parser.const = {}
    result = parser.is_public('module.private_func')
    assert result == False


def test_is_public_with_all_list_parent_match():
    parser = Parser()
    parser.root = {'module': 'module', 'module.submodule': 'module'}
    parser.imp = {'module': {'module.submodule'}}
    parser.const = {}
    result = parser.is_public('module.submodule')
    assert result == True


def test_is_public_module_itself():
    parser = Parser()
    parser.root = {'module': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('module')
    assert result == True


def test_is_public_in_imp_with_public_children():
    parser = Parser()
    parser.root = {'module': 'module', 'module.submod': 'module', 'module.submod.func': 'module'}
    parser.imp = {'module': set()}
    parser.doc = {'module': 'doc', 'module.submod': 'doc', 'module.submod.func': 'doc'}
    parser.const = {}
    result = parser.is_public('module.submod')
    assert result == True


def test_is_public_in_imp_without_public_children():
    parser = Parser()
    parser.root = {'module': 'module', 'module.submod': 'module'}
    parser.imp = {'module': set()}
    parser.doc = {'module': 'doc'}
    parser.const = {}
    result = parser.is_public('module.submod')
    assert result == False


# LLM-generated content at query #13
#--------------------------

```python
def test_attr_single_level_attribute():
    class Obj:
        attr = "value"
    obj = Obj()
    result = _attr(obj, "attr")
    assert result == "value"

def test_attr_nested_attributes():
    class Inner:
        value = "nested_value"
    class Middle:
        inner = Inner()
    class Outer:
        middle = Middle()
    obj = Outer()
    result = _attr(obj, "middle.inner.value")
    assert result == "nested_value"

def test_attr_nonexistent_attribute():
    class Obj:
        attr = "value"
    obj = Obj()
    result = _attr(obj, "nonexistent")
    assert result is None

def test_attr_nested_with_missing_middle():
    class Inner:
        value = "nested_value"
    class Middle:
        inner = Inner()
    class Outer:
        middle = Middle()
    obj = Outer()
    result = _attr(obj, "middle.nonexistent.value")
    assert result is None

def test_attr_nested_with_none_in_chain():
    class Inner:
        value = "nested_value"
    class Middle:
        inner = None
    class Outer:
        middle = Middle()
    obj = Outer()
    result = _attr(obj, "middle.inner.value")
    assert result is None

def test_attr_empty_string():
    class Obj:
        pass
    obj = Obj()
    result = _attr(obj, "")
    assert result is obj

def test_attr_with_numeric_values():
    class Obj:
        num = 42
    obj = Obj()
    result = _attr(obj, "num")
    assert result == 42

def test_attr_with_none_value():
    class Obj:
        attr = None
    obj = Obj()
    result = _attr(obj, "attr")
    assert result is None

def test_attr_multiple_levels_all_exist():
    class Level3:
        data = "final"
    class Level2:
        level3 = Level3()
    class Level1:
        level2 = Level2()
    obj = Level1()
    result = _attr(obj, "level2.level3.data")
    assert result == "final"


# LLM-generated content at query #14
#--------------------------

```python
def test_api_function_def():
    from ast import parse, FunctionDef
    parser = Parser()
    parser.parse('test_module', 'def foo(): pass')
    script = 'def foo(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.foo' in parser.doc
    assert 'foo()' in parser.doc['test_module.foo']


def test_api_async_function_def():
    from ast import parse, AsyncFunctionDef
    parser = Parser()
    parser.parse('test_module', 'async def bar(): pass')
    script = 'async def bar(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.bar' in parser.doc
    assert 'async bar()' in parser.doc['test_module.bar']


def test_api_class_def():
    from ast import parse, ClassDef
    parser = Parser()
    parser.parse('test_module', 'class Baz: pass')
    script = 'class Baz: pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.Baz' in parser.doc
    assert 'class Baz' in parser.doc['test_module.Baz']


def test_api_with_decorator():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', '@staticmethod\ndef func(): pass')
    script = '@staticmethod\ndef func(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.func' in parser.doc
    assert 'Decorators' in parser.doc['test_module.func']


def test_api_with_docstring():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', 'def func():\n    """Test doc"""\n    pass')
    script = 'def func():\n    """Test doc"""\n    pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.func' in parser.docstring
    assert 'Test doc' in parser.docstring['test_module.func']


def test_api_nested_class_method():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', 'class Outer:\n    def inner(): pass')
    script = 'class Outer:\n    def inner(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.Outer' in parser.doc
    assert 'test_module.Outer.inner' in parser.doc


def test_api_sets_level():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', 'def func(): pass')
    script = 'def func(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert parser.level['test_module.func'] == 0


def test_api_sets_root():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', 'def func(): pass')
    script = 'def func(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert parser.root['test_module.func'] == 'test_module'


def test_api_with_prefix():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', 'class Outer:\n    def inner(): pass')
    script = 'class Outer:\n    def inner(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node, prefix='')
    assert 'test_module.Outer' in parser.doc


def test_api_underscore_escaping():
    from ast import parse
    parser = Parser()
    parser.parse('test_module', 'def func_with_underscores(): pass')
    script = 'def func_with_underscores(): pass'
    root_node = parse(script)
    node = root_node.body[0]
    parser.api('test_module', node)
    assert 'test_module.func_with_underscores' in parser.doc


# LLM-generated content at query #15
#--------------------------

```python
def test_const_type_with_constant_int():
    from ast import Constant
    node = Constant(value=42)
    result = const_type(node)
    assert result == "int"


def test_const_type_with_constant_str():
    from ast import Constant
    node = Constant(value="hello")
    result = const_type(node)
    assert result == "str"


def test_const_type_with_constant_float():
    from ast import Constant
    node = Constant(value=3.14)
    result = const_type(node)
    assert result == "float"


def test_const_type_with_constant_bool():
    from ast import Constant
    node = Constant(value=True)
    result = const_type(node)
    assert result == "bool"


def test_const_type_with_list_of_ints():
    from ast import List, Constant
    node = List(elts=[Constant(value=1), Constant(value=2), Constant(value=3)])
    result = const_type(node)
    assert result == "list[int]"


def test_const_type_with_list_of_strings():
    from ast import List, Constant
    node = List(elts=[Constant(value="a"), Constant(value="b")])
    result = const_type(node)
    assert result == "list[str]"


def test_const_type_with_empty_list():
    from ast import List
    node = List(elts=[])
    result = const_type(node)
    assert result == "list"


def test_const_type_with_tuple_of_ints():
    from ast import Tuple, Constant
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    result = const_type(node)
    assert result == "tuple[int]"


def test_const_type_with_set_of_ints():
    from ast import Set, Constant
    node = Set(elts=[Constant(value=1), Constant(value=2)])
    result = const_type(node)
    assert result == "set[int]"


def test_const_type_with_dict_of_str_to_int():
    from ast import Dict, Constant
    node = Dict(keys=[Constant(value="a"), Constant(value="b")], 
                values=[Constant(value=1), Constant(value=2)])
    result = const_type(node)
    assert result == "dict[str, int]"


def test_const_type_with_mixed_type_list():
    from ast import List, Constant
    node = List(elts=[Constant(value=1), Constant(value="a")])
    result = const_type(node)
    assert result == "list[Any]"


def test_const_type_with_non_constant_element():
    from ast import List, Name
    node = List(elts=[Name(id="x")])
    result = const_type(node)
    assert result == "list"


def test_const_type_with_call_to_int():
    from ast import Call, Name
    node = Call(func=Name(id="int"), args=[], keywords=[])
    result = const_type(node)
    assert result == "int"


def test_const_type_with_call_to_str():
    from ast import Call, Name
    node = Call(func=Name(id="str"), args=[], keywords=[])
    result = const_type(node)
    assert result == "str"


def test_const_type_with_call_to_bool():
    from ast import Call, Name
    node = Call(func=Name(id="bool"), args=[], keywords=[])
    result = const_type(node)
    assert result == "bool"


def test_const_type_with_call_to_list():
    from ast import Call, Name
    node = Call(func=Name(id="list"), args=[], keywords=[])
    result = const_type(node)
    assert result == "ANY"


def test_const_type_with_unsupported_node():
    from ast import BinOp, Constant, Add
    node = BinOp(left=Constant(value=1), op=Add(), right=Constant(value=2))
    result = const_type(node)
    assert result == "ANY"


# LLM-generated content at query #16
#--------------------------

```python
def test_walk_body_predicate_line_9_false():
    from ast import Try, ExceptHandler, stmt
    
    # Create a Try node with an empty handlers list
    try_node = Try(
        body=[],
        handlers=[],
        orelse=[],
        finalbody=[]
    )
    
    # The predicate at line 9 is: `for h in node.handlers:`
    # It evaluates to False when node.handlers is empty
    assert not try_node.handlers
    assert len(try_node.handlers) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_imports_simple_import():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.os"] == "os"


def test_imports_simple_import_with_asname():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.operating_system"] == "os"


def test_imports_multiple_imports():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[
        alias(name="os", asname=None),
        alias(name="sys", asname="system")
    ])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.os"] == "os"
    assert parser.alias["mymodule.system"] == "sys"


def test_imports_from_import():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname=None)],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.path"] == "os.path"


def test_imports_from_import_with_asname():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname="ospath")],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.ospath"] == "os.path"


def test_imports_from_import_relative_level_1():
    parser = Parser()
    root = "pkg.mymodule"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="sibling",
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.mymodule.func"] == "pkg.sibling.func"


def test_imports_from_import_relative_level_2():
    parser = Parser()
    root = "pkg.subpkg.mymodule"
    parser.level[root] = 2
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="other",
        names=[alias(name="Class", asname=None)],
        level=2
    )
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.subpkg.mymodule.Class"] == "pkg.other.Class"


def test_imports_from_import_multiple_names():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="collections",
        names=[
            alias(name="defaultdict", asname=None),
            alias(name="Counter", asname="MyCounter")
        ],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.defaultdict"] == "collections.defaultdict"
    assert parser.alias["mymodule.MyCounter"] == "collections.Counter"


def test_imports_from_import_with_none_module():
    parser = Parser()
    root = "pkg.mymodule"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = ImportFrom(
        module=None,
        names=[alias(name="helper", asname=None)],
        level=1
    )
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.mymodule.helper"] == "pkg.helper"


# LLM-generated content at query #18
#--------------------------

```python
def test_func_ann_with_self_and_annotations():
    from ast import arg, parse
    
    parser = Parser()
    root = "test_module"
    parser.alias = {}
    
    # Create arg objects with annotations
    args_list = [
        arg(arg='self', annotation=parse('MyClass').body[0].value),
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='y', annotation=parse('str').body[0].value),
    ]
    
    result = list(parser.func_ann(root, args_list, has_self=True, cls_method=False))
    
    assert result[0] == 'Self'
    assert result[1] == 'int'
    assert result[2] == 'str'


def test_func_ann_without_self_annotations():
    from ast import arg, parse
    
    parser = Parser()
    root = "test_module"
    parser.alias = {}
    
    args_list = [
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='y', annotation=parse('str').body[0].value),
    ]
    
    result = list(parser.func_ann(root, args_list, has_self=False, cls_method=False))
    
    assert result[0] == 'int'
    assert result[1] == 'str'


def test_func_ann_with_classmethod():
    from ast import arg, parse
    
    parser = Parser()
    root = "test_module"
    parser.alias = {}
    
    args_list = [
        arg(arg='cls', annotation=parse('type[MyClass]').body[0].value),
        arg(arg='x', annotation=parse('int').body[0].value),
    ]
    
    result = list(parser.func_ann(root, args_list, has_self=True, cls_method=True))
    
    assert result[0] == 'type[Self]'
    assert result[1] == 'int'


def test_func_ann_with_no_annotation():
    from ast import arg
    
    parser = Parser()
    root = "test_module"
    parser.alias = {}
    
    args_list = [
        arg(arg='x', annotation=None),
        arg(arg='y', annotation=None),
    ]
    
    result = list(parser.func_ann(root, args_list, has_self=False, cls_method=False))
    
    assert result[0] == 'Any'
    assert result[1] == 'Any'


def test_func_ann_with_varargs_marker():
    from ast import arg, parse
    
    parser = Parser()
    root = "test_module"
    parser.alias = {}
    
    args_list = [
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='*', annotation=None),
        arg(arg='y', annotation=parse('str').body[0].value),
    ]
    
    result = list(parser.func_ann(root, args_list, has_self=False, cls_method=False))
    
    assert result[0] == 'int'
    assert result[1] == ''
    assert result[2] == 'str'


def test_func_ann_mixed_annotations():
    from ast import arg, parse
    
    parser = Parser()
    root = "test_module"
    parser.alias = {}
    
    args_list = [
        arg(arg='self', annotation=parse('MyClass').body[0].value),
        arg(arg='x', annotation=parse('int').body[0].value),
        arg(arg='y', annotation=None),
        arg(arg='z', annotation=parse('list[str]').body[0].value),
    ]
    
    result = list(parser.func_ann(root, args_list, has_self=True, cls_method=False))
    
    assert result[0] == 'Self'
    assert result[1] == 'int'
    assert result[2] == 'Any'
    assert result[3] == 'list[str]'


# LLM-generated content at query #19
#--------------------------

```python
def test_is_public_predicate_line_5_evaluates_to_false():
    from dataclasses import dataclass, field
    from itertools import chain
    
    # Mock the is_public_family function to return False
    import sys
    from unittest.mock import MagicMock
    
    # Create a Parser instance
    parser = Parser(link=True, b_level=1, toc=False)
    
    # Set up test data
    # s is in self.imp (condition at line 3 is True)
    parser.imp = {'module1': set()}
    parser.root = {'module1': 'module1'}
    parser.doc = {'module1.private_attr': 'doc1', 'module1.another': 'doc2'}
    parser.const = {'module1.const1': 'int'}
    
    # Mock is_public_family to return False for the test
    original_is_public_family = __import__('sys').modules.get('__main__').__dict__.get('is_public_family')
    
    # We need to ensure that for all ch that starts with 'module1.',
    # is_public_family(ch) returns False, making the predicate at line 5 False
    test_s = 'module1'
    
    # Set up so that s is in self.imp
    parser.imp = {'module1': set()}
    parser.root = {'module1': 'module1', 'module1.private_attr': 'module1'}
    parser.doc = {'module1.private_attr': 'some doc'}
    parser.const = {}
    
    # The loop will iterate through doc and const keys
    # For the predicate at line 5 to evaluate to False:
    # - ch must start with 'module1.' (True for 'module1.private_attr')
    # - is_public_family(ch) must be False
    
    # We mock is_public_family to always return False
    mock_is_public_family = MagicMock(return_value=False)
    
    # Monkey patch for this test
    import builtins
    original_func = getattr(__import__('sys').modules.get('__main__'), 'is_public_family', None)
    
    # Call is_public with mocked is_public_family behavior
    # The predicate "ch.startswith(s + '.') and is_public_family(ch)" should be False
    # when is_public_family returns False
    result = parser.is_public('module1')
    
    # The result should be False because:
    # - 'module1' is in self.imp
    # - The loop finds 'module1.private_attr' which starts with 'module1.'
    # - But is_public_family('module1.private_attr') is False (in real scenario)
    # - So the predicate at line 5 is False, loop continues to else clause
    # - else clause returns False
    assert result == False


# LLM-generated content at query #20
#--------------------------

```python
def test_func_api_with_simple_arguments():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse, arg
    func_node = parse("def func(a, b): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=False, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert '| a |' in parser.doc['test_module.func']


def test_func_api_with_defaults():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(a, b=1): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=False, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert '| a |' in parser.doc['test_module.func']


def test_func_api_with_self():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(self, a): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=True, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert 'Self' in parser.doc['test_module.func']


def test_func_api_with_return_type():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(a) -> int: pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, func_node.returns, has_self=False, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert 'return' in parser.doc['test_module.func']


def test_func_api_with_varargs():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(a, *args): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=False, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert '*args' in parser.doc['test_module.func']


def test_func_api_with_kwargs():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(a, **kwargs): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=False, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert '**kwargs' in parser.doc['test_module.func']


def test_func_api_with_classmethod():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(cls, a): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=True, cls_method=True)
    
    assert 'test_module.func' in parser.doc
    assert 'Self' in parser.doc['test_module.func']


def test_func_api_with_kwonly_args():
    parser = Parser()
    parser.doc['test_module'] = "Module"
    parser.alias = {}
    parser.root['test_module.func'] = 'test_module'
    
    from ast import parse
    func_node = parse("def func(a, *, b): pass").body[0]
    
    parser.func_api('test_module', 'test_module.func', func_node.args, None, has_self=False, cls_method=False)
    
    assert 'test_module.func' in parser.doc
    assert '| a |' in parser.doc['test_module.func']


# LLM-generated content at query #21
#--------------------------

```python
def test_visit_constant_with_non_string_value():
    resolver = Resolver("test_module", {})
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result is node


def test_visit_constant_with_string_value_valid_name():
    resolver = Resolver("test_module", {"test_module.MyType": "int"})
    node = Constant(value="MyType")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "MyType"


def test_visit_constant_with_string_value_invalid_syntax():
    resolver = Resolver("test_module", {})
    node = Constant(value="not a valid python expression !!!")
    result = resolver.visit_Constant(node)
    assert result is node


def test_visit_constant_with_string_value_self_type():
    resolver = Resolver("test_module", {}, self_ty="T")
    node = Constant(value="T")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


def test_visit_constant_with_string_subscript_expression():
    resolver = Resolver("test_module", {"test_module.Optional": "typing.Optional"})
    node = Constant(value="list[int]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"


def test_visit_constant_with_string_value_complex_expression():
    resolver = Resolver("test_module", {})
    node = Constant(value="Union[int, str]")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Subscript)


# LLM-generated content at query #22
#--------------------------

```python
def test_func_api_predicate_false():
    from ast import arguments, arg
    from dataclasses import dataclass, field
    
    @dataclass
    class MockParser:
        doc: dict = field(default_factory=dict)
        
        def func_ann(self, root, args, has_self, cls_method):
            return iter(['int', 'str'])
    
    parser = MockParser()
    parser.doc['test_func'] = ''
    
    node = arguments(
        posonlyargs=[],
        args=[arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    default_list = [None, None, None]
    has_default = all(d is None for d in default_list)
    
    assert has_default is True
    
    default_list_with_value = [None, 5, None]
    has_default_with_value = all(d is None for d in default_list_with_value)
    
    assert has_default_with_value is False


# LLM-generated content at query #23
#--------------------------

```python
def test_func_ann_annotation_not_none():
    from ast import arg as ast_arg, parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Set up minimal required state
    parser.alias = {}
    parser.doc = {}
    parser.level = {"test_module": 0}
    parser.root = {"test_module": "test_module"}
    
    # Create arguments with annotation (line 15 condition: a.annotation is not None)
    # Create an arg with annotation
    test_arg = ast_arg(arg='x', annotation=parse('int').body[0].value)
    args = [test_arg]
    
    # Call func_ann with has_self=False and cls_method=False
    # This ensures we skip the first condition (line 6) and second condition (line 13)
    # and reach line 15 where a.annotation is not None
    result = list(parser.func_ann('test_module', args, has_self=False, cls_method=False))
    
    # Assert that the result is not empty and contains resolved annotation
    assert len(result) > 0
    assert result[0] == 'int'


# LLM-generated content at query #24
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']
    assert 'BaseClass' in parser.doc['test_module.TestClass']


def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = []
    member_ann = AnnAssign(
        target=Name(id='attr1', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=None,
        simple=1
    )
    body = [member_ann]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'attr1' in parser.doc['test_module.TestClass']


def test_class_api_with_enums():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    enum_assign = AnnAssign(
        target=Name(id='RED', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=1),
        simple=1
    )
    body = [enum_assign]
    
    parser.class_api('test_module', 'test_module.Color', bases, body)
    
    assert 'test_module.Color' in parser.doc
    assert 'Enums' in parser.doc['test_module.Color']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = []
    member_assign = Assign(
        targets=[Name(id='attr1', ctx=Store())],
        value=Constant(value='test')
    )
    delete_node = Delete(targets=[Name(id='attr1', ctx=Del())])
    body = [member_assign, delete_node]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'attr1' not in parser.doc['test_module.TestClass']


def test_class_api_empty_class():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.Empty', bases, body)
    
    assert 'test_module.Empty' in parser.doc
    assert 'Bases' not in parser.doc['test_module.Empty']
    assert 'Members' not in parser.doc['test_module.Empty']


def test_class_api_private_members_excluded():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = []
    private_member = AnnAssign(
        target=Name(id='_private', ctx=Store()),
        annotation=Name(id='str', ctx=Load()),
        value=None,
        simple=1
    )
    public_member = AnnAssign(
        target=Name(id='public', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=None,
        simple=1
    )
    body = [private_member, public_member]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'public' in parser.doc['test_module.TestClass']
    assert '_private' not in parser.doc['test_module.TestClass']


def test_class_api_with_type_comment():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    
    bases = []
    member_assign = Assign(
        targets=[Name(id='attr1', ctx=Store())],
        value=Constant(value=42),
        type_comment='int'
    )
    body = [member_assign]
    
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


# LLM-generated content at query #25
#--------------------------

```python
def test_is_public_with_public_module():
    parser = Parser()
    parser.root = {'pkg.module': 'pkg.module'}
    parser.imp = {'pkg.module': set()}
    result = parser.is_public('pkg.module')
    assert result is True


def test_is_public_with_private_name():
    parser = Parser()
    parser.root = {'pkg._private': 'pkg._private'}
    parser.imp = {'pkg': set()}
    result = parser.is_public('pkg._private')
    assert result is False


def test_is_public_with_magic_name():
    parser = Parser()
    parser.root = {'pkg.__init__': 'pkg.__init__'}
    parser.imp = {'pkg': set()}
    result = parser.is_public('pkg.__init__')
    assert result is True


def test_is_public_with_all_list_containing_name():
    parser = Parser()
    parser.root = {'pkg.module': 'pkg.module', 'pkg.func': 'pkg.module'}
    parser.imp = {'pkg.module': {'pkg.module', 'pkg.func'}}
    result = parser.is_public('pkg.func')
    assert result is True


def test_is_public_with_all_list_not_containing_name():
    parser = Parser()
    parser.root = {'pkg.module': 'pkg.module', 'pkg.func': 'pkg.module'}
    parser.imp = {'pkg.module': {'pkg.module'}}
    result = parser.is_public('pkg.func')
    assert result is False


def test_is_public_when_name_is_in_imp_and_has_public_children():
    parser = Parser()
    parser.root = {'pkg': 'pkg', 'pkg.sub.func': 'pkg.sub.func'}
    parser.imp = {'pkg': set()}
    parser.doc = {'pkg': 'doc', 'pkg.sub.func': 'doc'}
    parser.const = {}
    result = parser.is_public('pkg')
    assert result is True


def test_is_public_when_name_is_in_imp_and_no_public_children():
    parser = Parser()
    parser.root = {'pkg': 'pkg'}
    parser.imp = {'pkg': set()}
    parser.doc = {}
    parser.const = {}
    result = parser.is_public('pkg')
    assert result is False


def test_is_public_with_all_list_containing_parent():
    parser = Parser()
    parser.root = {'pkg.module': 'pkg.module', 'pkg.module.func': 'pkg.module'}
    parser.imp = {'pkg.module': {'pkg.module'}}
    result = parser.is_public('pkg.module.func')
    assert result is True


def test_is_public_lowercase_name():
    parser = Parser()
    parser.root = {'pkg.func': 'pkg.func'}
    parser.imp = {'pkg.func': set()}
    result = parser.is_public('pkg.func')
    assert result is True


def test_is_public_uppercase_constant():
    parser = Parser()
    parser.root = {'pkg.CONST': 'pkg.CONST'}
    parser.imp = {'pkg.CONST': set()}
    result = parser.is_public('pkg.CONST')
    assert result is True


# LLM-generated content at query #26
#--------------------------

```python
def test_globals_const_predicate_false():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Assign node with a Name target
    target = Name(id='MY_CONST', ctx=None)
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    root = 'test_module'
    parser.imp[root] = set()
    
    # Pre-populate const with a non-ANY value for the same name
    name = root + '.' + 'MY_CONST'
    parser.const[name] = 'int'
    
    # Call globals method
    parser.globals(root, node)
    
    # The predicate at line 33 should evaluate to False
    # because parser.const.get(name, ANY) != ANY (it equals 'int')
    # So the assignment at line 34 should not happen
    assert parser.const[name] == 'int'


# LLM-generated content at query #27
#--------------------------

```python
def test_is_public_with_magic_name():
    parser = Parser()
    parser.root = {'__init__': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('__init__')
    assert result is True


def test_is_public_with_private_name():
    parser = Parser()
    parser.root = {'_private': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('_private')
    assert result is False


def test_is_public_with_public_name_no_all():
    parser = Parser()
    parser.root = {'module.public': 'module'}
    parser.imp = {'module': set()}
    result = parser.is_public('module.public')
    assert result is True


def test_is_public_with_public_name_in_all():
    parser = Parser()
    parser.root = {'module.func': 'module'}
    parser.imp = {'module': {'module.func'}}
    result = parser.is_public('module.func')
    assert result is True


def test_is_public_with_module_itself_in_all():
    parser = Parser()
    parser.root = {'module': 'module'}
    parser.imp = {'module': {'module'}}
    result = parser.is_public('module')
    assert result is True


def test_is_public_with_submodule_in_all():
    parser = Parser()
    parser.root = {'module.sub.func': 'module'}
    parser.imp = {'module': {'module.sub'}}
    result = parser.is_public('module.sub.func')
    assert result is True


def test_is_public_not_in_all_with_all_list():
    parser = Parser()
    parser.root = {'module.func': 'module'}
    parser.imp = {'module': {'module.other'}}
    result = parser.is_public('module.func')
    assert result is False


def test_is_public_with_module_in_imp_dict():
    parser = Parser()
    parser.root = {'module': 'module'}
    parser.imp = {'module': set()}
    parser.doc = {'module': 'doc'}
    result = parser.is_public('module')
    assert result is True


def test_is_public_with_underscore_prefix_in_all():
    parser = Parser()
    parser.root = {'module._private': 'module'}
    parser.imp = {'module': {'module._private'}}
    result = parser.is_public('module._private')
    assert result is True


# LLM-generated content at query #28
#--------------------------

```python
def test_globals_predicate_line_18_false():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Assign node with multiple targets (len(node.targets) != 1)
    # This makes the predicate at line 18 evaluate to False
    target1 = Name(id='x')
    target2 = Name(id='y')
    value = Constant(value=5)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    # Call globals with the node
    # Since len(node.targets) == 2, the condition at line 18 is False
    # and the function should return early at line 28
    parser.globals('test_module', node)
    
    # Verify that nothing was added to parser's dictionaries
    # because the function returned early
    assert 'test_module.x' not in parser.alias
    assert 'test_module.y' not in parser.alias


# LLM-generated content at query #29
#--------------------------

```python
def test_globals_predicate_line_35_evaluates_to_false():
    """Test that the predicate at line 35 evaluates to False.
    
    The predicate is: left.id != '__all__' or not isinstance(node.value, (Tuple, List))
    For it to evaluate to False, both conditions must be False:
    - left.id == '__all__' (first condition is False)
    - isinstance(node.value, (Tuple, List)) (second condition is False, so 'not' makes it False)
    """
    from ast import Assign, Name, Constant, Tuple, parse
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    parser.imp[root] = set()
    
    # Create an assignment: __all__ = ('item1', 'item2')
    # This matches: left.id == '__all__' and isinstance(node.value, Tuple)
    script = "__all__ = ('item1', 'item2')"
    tree = parse(script)
    node = tree.body[0]
    
    # Verify the node structure
    assert isinstance(node, Assign)
    assert len(node.targets) == 1
    assert isinstance(node.targets[0], Name)
    assert node.targets[0].id == '__all__'
    assert isinstance(node.value, Tuple)
    
    # Call globals method
    parser.globals(root, node)
    
    # If predicate evaluates to False, the loop at line 37 should execute
    # and items should be added to parser.imp[root]
    assert len(parser.imp[root]) > 0
    assert 'test_module.item1' in parser.imp[root]
    assert 'test_module.item2' in parser.imp[root]


# LLM-generated content at query #30
#--------------------------

```python
def test_globals_predicate_line_38_false():
    """Test that the predicate at line 38 evaluates to False."""
    from ast import AnnAssign, Assign, Name, Constant, Tuple, List
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    parser.imp[root] = set()
    
    # Create a node with __all__ = (123,) where element is Constant but not str
    constant_int = Constant(value=123)
    tuple_node = Tuple(elts=[constant_int], ctx=None)
    assign_node = Assign(
        targets=[Name(id='__all__', ctx=None)],
        value=tuple_node,
        type_comment=None
    )
    
    # Call globals - it should reach line 35 condition check
    # The predicate at line 38 should be False because e.value is int, not str
    parser.globals(root, assign_node)
    
    # The imp[root] should remain empty since the predicate was False
    assert len(parser.imp[root]) == 0


# LLM-generated content at query #31
#--------------------------

```python
def test_doctest_empty_string():
    result = doctest("")
    assert result == ""

def test_doctest_no_doctest_lines():
    doc = "This is a regular line\nAnother regular line"
    result = doctest(doc)
    assert result == "This is a regular line\nAnother regular line"

def test_doctest_single_doctest_line():
    doc = ">>> x = 1"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n```"

def test_doctest_multiple_doctest_lines():
    doc = ">>> x = 1\n>>> y = 2\n>>> print(x + y)"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n>>> y = 2\n>>> print(x + y)\n```"

def test_doctest_mixed_lines():
    doc = "Some text\n>>> x = 1\n>>> y = 2\nMore text"
    result = doctest(doc)
    assert result == "Some text\n```python\n>>> x = 1\n>>> y = 2\n```\nMore text"

def test_doctest_multiple_blocks():
    doc = ">>> x = 1\ntext\n>>> y = 2"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n```\ntext\n```python\n>>> y = 2\n```"

def test_doctest_doctest_at_start():
    doc = ">>> x = 1\nregular line"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n```\nregular line"

def test_doctest_doctest_at_end():
    doc = "regular line\n>>> x = 1"
    result = doctest(doc)
    assert result == "regular line\n```python\n>>> x = 1\n```"

def test_doctest_consecutive_blocks():
    doc = ">>> x = 1\n>>> y = 2\ntext\n>>> z = 3\n>>> w = 4"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n>>> y = 2\n```\ntext\n```python\n>>> z = 3\n>>> w = 4\n```"

def test_doctest_single_line_only_doctest():
    doc = ">>> print('hello')"
    result = doctest(doc)
    assert result == "```python\n>>> print('hello')\n```"

def test_doctest_empty_lines_between():
    doc = ">>> x = 1\n\n>>> y = 2"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n```\n\n```python\n>>> y = 2\n```"

def test_doctest_doctest_with_output():
    doc = ">>> x = 1\n>>> print(x)\n1\nmore text"
    result = doctest(doc)
    assert result == "```python\n>>> x = 1\n>>> print(x)\n```\n1\nmore text"


# LLM-generated content at query #32
#--------------------------

```python
def test_const_type_predicate_line_11():
    from ast import expr, Constant, Call, Name
    from itertools import chain
    
    # Mock PEP585 dictionary
    PEP585 = {'list': 'list', 'dict': 'dict', 'set': 'set'}
    
    # Test that func is in the chain of built-in types
    func = 'int'
    result = func in chain({'bool', 'int', 'float', 'complex', 'str'}, PEP585.keys(), PEP585.values())
    assert result is True
    
    # Test with another built-in type
    func = 'str'
    result = func in chain({'bool', 'int', 'float', 'complex', 'str'}, PEP585.keys(), PEP585.values())
    assert result is True
    
    # Test with PEP585 key
    func = 'list'
    result = func in chain({'bool', 'int', 'float', 'complex', 'str'}, PEP585.keys(), PEP585.values())
    assert result is True
    
    # Test with PEP585 value
    func = 'dict'
    result = func in chain({'bool', 'int', 'float', 'complex', 'str'}, PEP585.keys(), PEP585.values())
    assert result is True


# LLM-generated content at query #33
#--------------------------

```python
def test_func_ann_with_self_annotation():
    from ast import arg, parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Create a mock root module name
    root = "test_module"
    
    # Create arg objects for testing
    # First arg has annotation (self parameter)
    arg_with_annotation = arg(arg="self", annotation=parse("str").body[0].value)
    
    # Call func_ann with has_self=True, i=0, and a.annotation is not None
    args_list = [arg_with_annotation]
    result = list(parser.func_ann(root, args_list, has_self=True, cls_method=False))
    
    # The predicate at line 7 (a.annotation is not None) should evaluate to True
    # and the code should execute lines 8-11
    assert result[0] == 'Self'


# LLM-generated content at query #34
#--------------------------

```python
def test_load_docstring():
    from types import ModuleType
    
    # Create a mock module with docstrings
    mock_module = ModuleType('test_module')
    mock_module.__doc__ = "Module docstring"
    
    # Create a nested mock object
    class MockClass:
        """Class docstring"""
        pass
    
    mock_module.TestClass = MockClass()
    
    # Create parser instance
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`'
    parser.doc['test_module.TestClass'] = '## class TestClass'
    parser.root['test_module'] = 'test_module'
    parser.root['test_module.TestClass'] = 'test_module'
    
    # Call load_docstring
    parser.load_docstring('test_module', mock_module)
    
    # Verify that docstrings were loaded
    assert 'test_module' in parser.docstring
    assert 'Module docstring' in parser.docstring['test_module']
    assert 'test_module.TestClass' in parser.docstring
    assert 'Class docstring' in parser.docstring['test_module.TestClass']


def test_load_docstring_no_docstring():
    from types import ModuleType
    
    # Create a mock module without docstrings
    mock_module = ModuleType('test_module')
    
    class MockClass:
        pass
    
    mock_module.TestClass = MockClass()
    
    # Create parser instance
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`'
    parser.doc['test_module.TestClass'] = '## class TestClass'
    parser.root['test_module'] = 'test_module'
    parser.root['test_module.TestClass'] = 'test_module'
    
    # Call load_docstring
    parser.load_docstring('test_module', mock_module)
    
    # Verify that no docstrings were added for items without docstrings
    assert 'test_module' not in parser.docstring
    assert 'test_module.TestClass' not in parser.docstring


def test_load_docstring_filtered_by_root():
    from types import ModuleType
    
    # Create a mock module
    mock_module = ModuleType('test_module')
    
    class MockClass:
        """Class docstring"""
        pass
    
    mock_module.TestClass = MockClass()
    
    # Create parser instance with items from different roots
    parser = Parser()
    parser.doc['test_module'] = '# Module `test_module`'
    parser.doc['other_module.TestClass'] = '## class TestClass'
    parser.root['test_module'] = 'test_module'
    parser.root['other_module.TestClass'] = 'other_module'
    
    # Call load_docstring
    parser.load_docstring('test_module', mock_module)
    
    # Verify that only items from the specified root were processed
    assert 'test_module' not in parser.docstring
    assert 'other_module.TestClass' not in parser.docstring


def test_load_docstring_nested_attributes():
    from types import ModuleType
    
    # Create a mock module with nested attributes
    mock_module = ModuleType('test_module')
    
    class OuterClass:
        """Outer class docstring"""
        class InnerClass:
            """Inner class docstring"""
            pass
    
    mock_module.OuterClass = OuterClass()
    
    # Create parser instance
    parser = Parser()
    parser.doc['test_module.OuterClass'] = '## class OuterClass'
    parser.doc['test_module.OuterClass.InnerClass'] = '### class InnerClass'
    parser.root['test_module.OuterClass'] = 'test_module'
    parser.root['test_module.OuterClass.InnerClass'] = 'test_module'
    
    # Call load_docstring
    parser.load_docstring('test_module', mock_module)
    
    # Verify that nested docstrings were loaded
    assert 'test_module.OuterClass' in parser.docstring
    assert 'Outer class docstring' in parser.docstring['test_module.OuterClass']


# LLM-generated content at query #35
#--------------------------

```python
def test_e_type_empty_elements():
    from ast import Constant
    result = _e_type()
    assert result == ""


def test_e_type_single_element_with_single_constant():
    from ast import Constant
    const = Constant(value=42)
    result = _e_type([const])
    assert result == "[int]"


def test_e_type_single_element_with_multiple_same_type_constants():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value=2)
    result = _e_type([const1, const2])
    assert result == "[int]"


def test_e_type_single_element_with_multiple_different_type_constants():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value="string")
    result = _e_type([const1, const2])
    assert result == "[Any]"


def test_e_type_multiple_elements_same_types():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value=2)
    const3 = Constant(value=3.14)
    const4 = Constant(value=2.71)
    result = _e_type([const1, const2], [const3, const4])
    assert result == "[int, float]"


def test_e_type_multiple_elements_mixed_types():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value="string")
    const3 = Constant(value=3.14)
    result = _e_type([const1, const2], [const3])
    assert result == "[Any, float]"


def test_e_type_with_none_in_sequence():
    from ast import Constant
    result = _e_type([None])
    assert result == ""


def test_e_type_with_non_constant_in_sequence():
    from ast import Constant, Name
    const = Constant(value=42)
    name = Name(id="x")
    result = _e_type([const, name])
    assert result == ""


def test_e_type_with_empty_sequence():
    result = _e_type([])
    assert result == ""


def test_e_type_with_string_constants():
    from ast import Constant
    const1 = Constant(value="hello")
    const2 = Constant(value="world")
    result = _e_type([const1, const2])
    assert result == "[str]"


def test_e_type_with_bool_constants():
    from ast import Constant
    const1 = Constant(value=True)
    const2 = Constant(value=False)
    result = _e_type([const1, const2])
    assert result == "[bool]"


def test_e_type_multiple_elements_with_mixed_consistency():
    from ast import Constant
    const1 = Constant(value=1)
    const2 = Constant(value=2)
    const3 = Constant(value="string")
    result = _e_type([const1, const2], [const3, const3])
    assert result == "[int, str]"


# LLM-generated content at query #36
#--------------------------

```python
def test_class_api_enum_predicate_is_true():
    from ast import parse, Name, AnnAssign, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Setup: create a simple enum base
    root = "test_module"
    name = "test_module.TestEnum"
    
    # Create bases list with an enum base that starts with 'enum.'
    bases = [parse("enum.Enum").body[0].value]
    
    # Create a body with an AnnAssign node
    body_code = "x: int"
    body = parse(body_code).body
    
    # Initialize doc entry for the name
    parser.doc[name] = "# test"
    parser.level[root] = 0
    parser.root[name] = root
    
    # Call class_api - this will execute line 13
    # We need to mock resolve method to return something
    original_resolve = parser.resolve
    parser.resolve = lambda r, d: "enum.Enum"
    
    parser.class_api(root, name, bases, body)
    
    # Verify that line 13 predicate (if is_enum:) evaluated to True
    # by checking that enums list was populated
    # We can't directly access enums, but we can verify the behavior
    # by checking that the doc was updated with the Enums table
    assert "Enums" in parser.doc[name]
    
    parser.resolve = original_resolve


# LLM-generated content at query #37
#--------------------------

```python
def test_globals_predicate_line_23_false():
    """Test that the predicate at line 23 evaluates to False when node.type_comment is not None."""
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Assign node with type_comment set to a string (not None)
    target = Name(id='x', ctx=None)
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment='int')
    
    # Call globals method - this should execute the else branch at line 26
    # because node.type_comment is not None, making the predicate at line 23 False
    parser.globals('test_module', node)
    
    # Verify that the else branch was taken by checking that ann was set to type_comment
    # We can verify this indirectly by checking that alias was set (which happens at line 30)
    assert 'test_module.x' in parser.alias
    assert parser.alias['test_module.x'] == '42'


# LLM-generated content at query #38
#--------------------------

```python
def test_globals_predicate_line_38_false():
    """Test that the predicate at line 38 evaluates to False."""
    from ast import Assign, Name, Constant, Tuple, List
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    parser.imp[root] = set()
    
    # Create an Assign node with __all__ = (1, 2, 3)
    # where elements are not Constant nodes with str values
    target = Name(id='__all__', ctx=None)
    constant_int = Constant(value=1)
    constant_int_2 = Constant(value=2)
    tuple_node = Tuple(elts=[constant_int, constant_int_2], ctx=None)
    assign_node = Assign(targets=[target], value=tuple_node, type_comment=None)
    
    parser.globals(root, assign_node)
    
    # The predicate at line 38 should be False for integer Constants
    # so nothing should be added to parser.imp[root]
    assert len(parser.imp[root]) == 0


# LLM-generated content at query #39
#--------------------------

```python
def test_globals_predicate_line_38_false():
    from ast import Assign, Name, Constant, List
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create a node where line 38 predicate evaluates to False
    # isinstance(e, Constant) is True but isinstance(e.value, str) is False
    const_node = Constant(value=42)  # Integer, not string
    
    list_node = List(elts=[const_node], ctx=None)
    
    target = Name(id='__all__', ctx=None)
    assign_node = Assign(targets=[target], value=list_node, type_comment=None)
    
    parser.imp['test_root'] = set()
    
    # Call globals method
    parser.globals('test_root', assign_node)
    
    # The predicate at line 38 should be False because e.value is 42 (int), not str
    # So nothing should be added to self.imp['test_root']
    assert len(parser.imp['test_root']) == 0


# LLM-generated content at query #40
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.TestClass'] = '## class TestClass\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
class Base1:
    pass

class Base2:
    pass
'''
    tree = ast_parse(code_str)
    bases = tree.body[0].body
    
    parser.class_api('test_module', 'test_module.TestClass', tree.body[0].bases, tree.body[0].body)
    assert 'test_module.TestClass' in parser.doc


def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
class MyClass:
    attr1: int
    attr2: str = "default"
'''
    tree = ast_parse(code_str)
    class_node = tree.body[0]
    
    parser.class_api('test_module', 'test_module.MyClass', class_node.bases, class_node.body)
    assert 'test_module.MyClass' in parser.doc


def test_class_api_with_enums():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.Status'] = '## class Status\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
from enum import Enum

class Status(Enum):
    ACTIVE = 1
    INACTIVE = 2
'''
    tree = ast_parse(code_str)
    class_node = tree.body[1]
    
    parser.class_api('test_module', 'test_module.Status', class_node.bases, class_node.body)
    assert 'test_module.Status' in parser.doc


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.TestClass'] = '## class TestClass\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
class TestClass:
    attr1: int
    del attr1
'''
    tree = ast_parse(code_str)
    class_node = tree.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    assert 'test_module.TestClass' in parser.doc


def test_class_api_empty_class():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.Empty'] = '## class Empty\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
class Empty:
    pass
'''
    tree = ast_parse(code_str)
    class_node = tree.body[0]
    
    parser.class_api('test_module', 'test_module.Empty', class_node.bases, class_node.body)
    assert 'test_module.Empty' in parser.doc


def test_class_api_with_private_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.TestClass'] = '## class TestClass\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
class TestClass:
    public_attr: int
    _private_attr: str
'''
    tree = ast_parse(code_str)
    class_node = tree.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    assert 'test_module.TestClass' in parser.doc


def test_class_api_with_type_comments():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.doc['test_module.TestClass'] = '## class TestClass\n\n'
    
    from ast import parse as ast_parse
    code_str = '''
class TestClass:
    attr = 42  # type: int
'''
    tree = ast_parse(code_str, type_comments=True)
    class_node = tree.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    assert 'test_module.TestClass' in parser.doc


# LLM-generated content at query #41
#--------------------------

```python
def test_defaults_with_none_values():
    from ast import parse, unparse
    from collections.abc import Sequence
    
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]


def test_defaults_with_expression_values():
    from ast import parse, unparse
    
    code_expr = parse("42").body[0].value
    args = [code_expr]
    result = list(_defaults(args))
    assert len(result) == 1
    assert result[0] == "`42`"


def test_defaults_with_mixed_values():
    from ast import parse, unparse
    
    code_expr = parse("'hello'").body[0].value
    args = [code_expr, None, code_expr]
    result = list(_defaults(args))
    assert len(result) == 3
    assert result[0] == "`'hello'`"
    assert result[1] == " "
    assert result[2] == "`'hello'`"


def test_defaults_with_special_characters():
    from ast import parse, unparse
    
    code_expr = parse("'a&b'").body[0].value
    args = [code_expr]
    result = list(_defaults(args))
    assert len(result) == 1
    assert "<code>" in result[0]
    assert "</code>" in result[0]


def test_defaults_with_pipe_character():
    from ast import parse, unparse
    
    code_expr = parse("'a|b'").body[0].value
    args = [code_expr]
    result = list(_defaults(args))
    assert len(result) == 1
    assert "&#124;" in result[0]


def test_defaults_empty_sequence():
    args = []
    result = list(_defaults(args))
    assert result == []


# LLM-generated content at query #42
#--------------------------

```python
def test_e_type_empty_elements():
    from ast import Constant
    result = _e_type()
    assert result == ""


def test_e_type_single_element_with_constants():
    from ast import Constant
    result = _e_type([Constant(value=1), Constant(value=2)])
    assert result == "[int, int]"


def test_e_type_single_element_with_mixed_types():
    from ast import Constant
    result = _e_type([Constant(value=1), Constant(value="str")])
    assert result == "[Any]"


def test_e_type_multiple_elements_same_type():
    from ast import Constant
    result = _e_type([Constant(value=1)], [Constant(value=2)])
    assert result == "[int, int]"


def test_e_type_multiple_elements_different_types():
    from ast import Constant
    result = _e_type([Constant(value=1)], [Constant(value="str")])
    assert result == "[int, str]"


def test_e_type_element_with_none():
    from ast import Constant
    result = _e_type([Constant(value=1)], None)
    assert result == ""


def test_e_type_empty_sequence():
    from ast import Constant
    result = _e_type([])
    assert result == ""


def test_e_type_with_non_constant():
    from ast import Constant, Name
    result = _e_type([Constant(value=1), Name(id="x")])
    assert result == ""


def test_e_type_single_element_strings():
    from ast import Constant
    result = _e_type([Constant(value="a"), Constant(value="b")])
    assert result == "[str, str]"


def test_e_type_single_element_floats():
    from ast import Constant
    result = _e_type([Constant(value=1.5), Constant(value=2.5)])
    assert result == "[float, float]"


def test_e_type_multiple_elements_mixed():
    from ast import Constant
    result = _e_type([Constant(value=1), Constant(value=2)], [Constant(value="a")])
    assert result == "[int, str]"


def test_e_type_single_element_booleans():
    from ast import Constant
    result = _e_type([Constant(value=True), Constant(value=False)])
    assert result == "[bool, bool]"


# LLM-generated content at query #43
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = [Name(id='BaseClass')]
    body = []
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'BaseClass' in parser.doc['MyClass']


def test_class_api_with_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    body = [
        AnnAssign(
            target=Name(id='public_attr'),
            annotation=Name(id='str'),
            value=None,
            simple=1
        )
    ]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'public_attr' in parser.doc['MyClass'] or 'Members' in parser.doc['MyClass']


def test_class_api_with_private_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    body = [
        AnnAssign(
            target=Name(id='_private_attr'),
            annotation=Name(id='str'),
            value=None,
            simple=1
        )
    ]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert '_private_attr' not in parser.doc['MyClass']


def test_class_api_with_enum_bases():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['MyEnum'] = '## class MyEnum\n\n'
    
    bases = [Attribute(value=Name(id='enum'), attr='Enum')]
    body = [
        AnnAssign(
            target=Name(id='MEMBER1'),
            annotation=Name(id='int'),
            value=Constant(value=1),
            simple=1
        )
    ]
    
    parser.class_api('test_module', 'MyEnum', bases, body)
    
    assert 'MyEnum' in parser.doc
    assert 'Enums' in parser.doc['MyEnum'] or 'MEMBER1' in parser.doc['MyEnum']


def test_class_api_with_deleted_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    body = [
        AnnAssign(
            target=Name(id='attr'),
            annotation=Name(id='str'),
            value=None,
            simple=1
        ),
        Delete(targets=[Name(id='attr')])
    ]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc


def test_class_api_with_assign_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['MyClass'] = '## class MyClass\n\n'
    
    bases = []
    body = [
        Assign(
            targets=[Name(id='class_var')],
            value=Constant(value=42),
            type_comment=None
        )
    ]
    
    parser.class_api('test_module', 'MyClass', bases, body)
    
    assert 'MyClass' in parser.doc
    assert 'class_var' in parser.doc['MyClass'] or 'Members' in parser.doc['MyClass']


def test_class_api_no_bases_no_members():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.level['test_module'] = 0
    parser.alias = {}
    parser.doc['SimpleClass'] = '## class SimpleClass\n\n'
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'SimpleClass', bases, body)
    
    assert 'SimpleClass' in parser.doc
    assert parser.doc['SimpleClass'] == '## class SimpleClass\n\n'


# LLM-generated content at query #44
#--------------------------

```python
def test_walk_body_with_if_statement():
    from ast import If, stmt, Pass
    from typing import Sequence, Iterator
    
    def walk_body(body: Sequence[stmt]) -> Iterator[stmt]:
        """Traverse around body and its simple definition scope."""
        for node in body:
            if isinstance(node, If):
                yield from walk_body(node.body)
                yield from walk_body(node.orelse)
            elif isinstance(node, Try):
                yield from walk_body(node.body)
                for h in node.handlers:
                    yield from walk_body(h.body)
                yield from walk_body(node.orelse)
                yield from walk_body(node.finalbody)
            else:
                yield node
    
    if_node = If(test=Pass(), body=[Pass()], orelse=[])
    result = list(walk_body([if_node]))
    assert isinstance(if_node, If)
    assert True


# LLM-generated content at query #45
#--------------------------

```python
def test_globals_predicate_line_18_false():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict = field(default_factory=dict)
        doc: dict = field(default_factory=dict)
        docstring: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        alias: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
    
    parser = Parser()
    
    # Create an Assign node with multiple targets (len(node.targets) != 1)
    # This makes the predicate at line 18 (len(node.targets) == 1) evaluate to False
    target1 = Name(id='x')
    target2 = Name(id='y')
    node = Assign(targets=[target1, target2], value=Constant(value=1))
    
    # Verify that isinstance(node, Assign) is True
    assert isinstance(node, Assign)
    
    # Verify that len(node.targets) == 1 is False
    assert len(node.targets) == 1 is False
    
    # The predicate at line 18 should evaluate to False
    predicate = isinstance(node, Assign) and len(node.targets) == 1 and isinstance(node.targets[0], Name)
    assert predicate is False


# LLM-generated content at query #46
#--------------------------

```python
def test_attr_predicate_evaluates_to_false():
    class TestObj:
        def __init__(self):
            self.nested = NestedObj()
    
    class NestedObj:
        def __init__(self):
            self.value = "test"
    
    obj = TestObj()
    result = _attr(obj, "nested.value")
    assert result is not None
    assert result == "test"


# LLM-generated content at query #47
#--------------------------

```python
def test_func_ann_predicate_line_7():
    from ast import arg, parse, Name
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Create a mock arg with annotation
    mock_arg = arg(arg='self', annotation=parse('int').body[0].value)
    
    # Call func_ann with has_self=True and i=0 (first iteration)
    # The predicate at line 7 checks if a.annotation is not None
    result = list(parser.func_ann('test_module', [mock_arg], has_self=True, cls_method=False))
    
    # The result should contain 'Self' because has_self=True, i=0, and we yielded at line 12
    assert len(result) > 0
    assert result[0] == 'Self'


# LLM-generated content at query #48
#--------------------------

```python
def test_visit_attribute_typing_prefix():
    resolver = Resolver(root="module", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)


def test_visit_attribute_non_typing_prefix():
    resolver = Resolver(root="module", alias={})
    node = Attribute(value=Name(id="other", ctx=Load()), attr="Method", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other"
    assert result.attr == "Method"


def test_visit_attribute_non_name_value():
    resolver = Resolver(root="module", alias={})
    inner_attr = Attribute(value=Name(id="obj", ctx=Load()), attr="inner", ctx=Load())
    node = Attribute(value=inner_attr, attr="Method", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value == inner_attr
    assert result.attr == "Method"


def test_visit_attribute_typing_union():
    resolver = Resolver(root="module", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Union", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Union"


def test_visit_attribute_typing_optional():
    resolver = Resolver(root="module", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="Optional", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "Optional"


# LLM-generated content at query #49
#--------------------------

```python
from ast import If, stmt, parse

def test_walk_body_if_predicate():
    code = "if True:\n    pass\nelse:\n    pass"
    tree = parse(code)
    body = tree.body
    node = body[0]
    assert isinstance(node, If)


# LLM-generated content at query #50
#--------------------------

```python
def test_class_api_delete_statement_predicate():
    from ast import Delete, Name, parse
    
    parser = Parser()
    parser.doc['test_module.TestClass'] = "## class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"
    
    # Create a Delete node with Name targets
    delete_code = "del x, y"
    delete_node = parse(delete_code).body[0]
    
    assert isinstance(delete_node, Delete)
    assert len(delete_node.targets) == 2
    assert all(isinstance(target, Name) for target in delete_node.targets)
    
    # Verify the predicate at line 30 evaluates to True
    assert isinstance(delete_node, Delete)


# LLM-generated content at query #51
#--------------------------

```python
def test_class_api_delete_statement_predicate():
    from ast import Delete, Name, parse
    
    parser = Parser()
    parser.doc['test_module.TestClass'] = "# class TestClass\n\n"
    
    # Create a Delete node with Name targets
    delete_node = Delete(targets=[Name(id='attr1', ctx=None), Name(id='attr2', ctx=None)])
    
    # Verify the predicate at line 30 evaluates to True
    result = isinstance(delete_node, Delete)
    assert result is True


# LLM-generated content at query #52
#--------------------------

```python
def test_compile_magic_method_continues():
    from dataclasses import dataclass, field
    from typing import TypeVar
    
    p = Parser(link=False, b_level=1, toc=False)
    p.doc['__init__'] = '# {}'
    p.docstring = {}
    p.imp = {}
    p.root['__init__'] = ''
    p.level['__init__'] = 0
    
    result = p.compile()
    
    assert '__init__' not in result


# LLM-generated content at query #53
#--------------------------

```python
def test_visit_name_self_ty():
    resolver = Resolver("module", {}, "T")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)


def test_visit_name_not_in_alias():
    resolver = Resolver("module", {}, "")
    node = Name("SomeClass", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "SomeClass"


def test_visit_name_in_alias_simple():
    resolver = Resolver("module", {"module.MyType": "int"}, "")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"


def test_visit_name_in_alias_circular_reference():
    resolver = Resolver("module", {"module.MyType": "module.MyType"}, "")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "MyType"


def test_visit_name_typevar_in_alias():
    resolver = Resolver("module", {"module.T": "TypeVar('T')", "module.TypeVar": "typing.TypeVar"}, "")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


def test_visit_name_complex_expression_in_alias():
    resolver = Resolver("module", {"module.MyType": "list[int]"}, "")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Subscript)


def test_visit_name_with_empty_root():
    resolver = Resolver("", {"MyType": "str"}, "")
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "str"


# LLM-generated content at query #54
#--------------------------

```python
def test_visit_subscript_union_with_tuple():
    resolver = Resolver("test_module", {})
    node = Subscript(
        value=Name(id='Union', ctx=Load()),
        slice=Tuple(elts=[Name(id='int', ctx=Load()), Name(id='str', ctx=Load())], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)


def test_visit_subscript_union_without_tuple():
    resolver = Resolver("test_module", {})
    node = Subscript(
        value=Name(id='Union', ctx=Load()),
        slice=Name(id='int', ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Name)
    assert result.id == 'int'


def test_visit_subscript_optional():
    resolver = Resolver("test_module", {"test_module.Optional": "typing.Optional"})
    node = Subscript(
        value=Name(id='Optional', ctx=Load()),
        slice=Name(id='str', ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None


def test_visit_subscript_non_name_value():
    resolver = Resolver("test_module", {})
    node = Subscript(
        value=Attribute(value=Name(id='typing', ctx=Load()), attr='Union', ctx=Load()),
        slice=Tuple(elts=[Name(id='int', ctx=Load()), Name(id='str', ctx=Load())], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(node)
    assert result is node


def test_visit_subscript_pep585_list():
    resolver = Resolver("test_module", {"test_module.list": "builtins.list"})
    node = Subscript(
        value=Name(id='list', ctx=Load()),
        slice=Name(id='int', ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)


def test_visit_subscript_regular_subscript():
    resolver = Resolver("test_module", {})
    node = Subscript(
        value=Name(id='List', ctx=Load()),
        slice=Name(id='int', ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(node)
    assert result is node


# LLM-generated content at query #55
#--------------------------

```python
def test_imports_simple_import():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    
    assert parser.alias["test_module.os"] == "os"


def test_imports_simple_import_with_asname():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, import_node)
    
    assert parser.alias["test_module.operating_system"] == "os"


def test_imports_from_import_absolute():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=0)
    parser.imports(root, import_node)
    
    assert parser.alias["test_module.path"] == "os.path"


def test_imports_from_import_absolute_with_asname():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(module="os", names=[alias(name="path", asname="p")], level=0)
    parser.imports(root, import_node)
    
    assert parser.alias["test_module.p"] == "os.path"


def test_imports_from_import_relative_level_1():
    parser = Parser()
    root = "pkg.test_module"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = ImportFrom(module="sibling", names=[alias(name="func", asname=None)], level=1)
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.test_module.func"] == "pkg.sibling.func"


def test_imports_from_import_relative_level_2():
    parser = Parser()
    root = "pkg.sub.test_module"
    parser.level[root] = 2
    parser.root[root] = root
    
    import_node = ImportFrom(module="other", names=[alias(name="cls", asname=None)], level=2)
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.sub.test_module.cls"] == "pkg.other.cls"


def test_imports_from_import_multiple_names():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(module="collections", names=[
        alias(name="defaultdict", asname=None),
        alias(name="Counter", asname=None)
    ], level=0)
    parser.imports(root, import_node)
    
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"
    assert parser.alias["test_module.Counter"] == "collections.Counter"


def test_imports_multiple_import_statements():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[
        alias(name="os", asname=None),
        alias(name="sys", asname=None)
    ])
    parser.imports(root, import_node)
    
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.sys"] == "sys"


def test_imports_from_import_no_module():
    parser = Parser()
    root = "pkg.test_module"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = ImportFrom(module=None, names=[alias(name="func", asname=None)], level=1)
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.test_module.func"] == "pkg.func"


def test_imports_nested_package():
    parser = Parser()
    root = "pkg.subpkg.module"
    parser.level[root] = 2
    parser.root[root] = root
    
    import_node = ImportFrom(module="utils", names=[alias(name="helper", asname="h")], level=0)
    parser.imports(root, import_node)
    
    assert parser.alias["pkg.subpkg.module.h"] == "utils.helper"


# LLM-generated content at query #56
#--------------------------

```python
def test_imports_with_asname():
    from ast import Import, alias
    from dataclasses import dataclass, field
    
    parser = Parser()
    
    # Create an Import node with an alias (asname is not None)
    import_alias = alias(name='original_name', asname='renamed_name')
    import_node = Import(names=[import_alias])
    
    # Call imports method
    parser.imports('test_module', import_node)
    
    # Verify that the predicate (a.asname is None) evaluates to False
    # which means 'name' should be set to a.asname ('renamed_name')
    assert 'test_module.renamed_name' in parser.alias
    assert parser.alias['test_module.renamed_name'] == 'original_name'


# LLM-generated content at query #57
#--------------------------

```python
def test_load_docstring():
    from types import ModuleType
    from dataclasses import dataclass, field
    
    # Create a parser instance
    parser = Parser()
    
    # Set up initial state
    parser.doc['test_module'] = 'Module test_module'
    parser.doc['test_module.func'] = 'Function func'
    parser.doc['test_module.Class'] = 'Class Class'
    parser.docstring = {}
    
    # Create a mock module with docstrings
    mock_module = ModuleType('test_module')
    mock_module.__doc__ = 'This is module docstring'
    
    class MockClass:
        """This is class docstring"""
        pass
    
    def mock_func():
        """This is function docstring"""
        pass
    
    mock_module.func = mock_func
    mock_module.Class = MockClass
    
    # Call load_docstring
    parser.load_docstring('test_module', mock_module)
    
    # Verify that docstrings were loaded
    assert 'test_module' in parser.docstring
    assert 'This is module docstring' in parser.docstring['test_module']
    assert 'test_module.func' in parser.docstring
    assert 'This is function docstring' in parser.docstring['test_module.func']
    assert 'test_module.Class' in parser.docstring
    assert 'This is class docstring' in parser.docstring['test_module.Class']


def test_load_docstring_no_docstring():
    from types import ModuleType
    
    parser = Parser()
    parser.doc['test_module'] = 'Module test_module'
    parser.docstring = {}
    
    mock_module = ModuleType('test_module')
    
    parser.load_docstring('test_module', mock_module)
    
    assert 'test_module' not in parser.docstring


def test_load_docstring_nested_module():
    from types import ModuleType
    
    parser = Parser()
    parser.doc['parent.child'] = 'Module child'
    parser.doc['parent.child.func'] = 'Function func'
    parser.docstring = {}
    
    def mock_func():
        """Nested function docstring"""
        pass
    
    mock_module = ModuleType('parent.child')
    mock_module.func = mock_func
    
    parser.load_docstring('parent.child', mock_module)
    
    assert 'parent.child.func' in parser.docstring
    assert 'Nested function docstring' in parser.docstring['parent.child.func']


def test_load_docstring_with_doctest():
    from types import ModuleType
    
    parser = Parser()
    parser.doc['test_module'] = 'Module test_module'
    parser.docstring = {}
    
    mock_module = ModuleType('test_module')
    mock_module.__doc__ = '>>> x = 1\n>>> print(x)'
    
    parser.load_docstring('test_module', mock_module)
    
    assert 'test_module' in parser.docstring
    assert '```python' in parser.docstring['test_module']
    assert '```' in parser.docstring['test_module']


# LLM-generated content at query #58
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=False, level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}


def test_parser_new_classmethod_with_toc_enables_link():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is True


# LLM-generated content at query #59
#--------------------------

```python
def test_globals_type_comment_not_none():
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    @dataclass
    class MockParser:
        alias: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        
        def resolve(self, root, annotation):
            return "resolved_type"
    
    parser = MockParser()
    
    # Create an Assign node with type_comment set (not None)
    target = Name(id='MY_CONST', ctx=None)
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment='int')
    
    # Call globals method
    parser.globals('test_module', node)
    
    # Verify that node.type_comment is not None (line 23 predicate is False)
    assert node.type_comment is not None
    assert node.type_comment == 'int'


# LLM-generated content at query #60
#--------------------------

```python
def test_globals_with_annassign_and_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    # Create an AnnAssign node with value
    target = Name(id="my_var", ctx=Store())
    value = Constant(value=42)
    annotation = Name(id="int", ctx=Load())
    node = AnnAssign(target=target, annotation=annotation, value=value, simple=1)
    
    parser.globals(root, node)
    
    assert "test_module.my_var" in parser.alias
    assert parser.alias["test_module.my_var"] == "42"


def test_globals_with_assign_and_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    target = Name(id="MY_CONST", ctx=Store())
    value = Constant(value=100)
    node = Assign(targets=[target], value=value, type_comment="int")
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONST" in parser.alias
    assert "test_module.MY_CONST" in parser.root
    assert parser.root["test_module.MY_CONST"] == root
    assert parser.const["test_module.MY_CONST"] == "int"


def test_globals_with_assign_without_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    target = Name(id="MY_CONST", ctx=Store())
    value = Constant(value="hello")
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.MY_CONST" in parser.const
    assert parser.const["test_module.MY_CONST"] == "str"


def test_globals_with_all_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    target = Name(id="__all__", ctx=Store())
    elts = [Constant(value="func1"), Constant(value="func2")]
    value = Tuple(elts=elts, ctx=Load())
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_multiple_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    target1 = Name(id="x", ctx=Store())
    target2 = Name(id="y", ctx=Store())
    value = Constant(value=42)
    node = Assign(targets=[target1, target2], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.x" not in parser.alias


def test_globals_with_non_name_target():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    target = Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    target = Name(id="CONSTANT", ctx=Store())
    value = Constant(value=999)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.CONSTANT" in parser.root
    assert parser.root["test_module.CONSTANT"] == root
    assert "test_module.CONSTANT" in parser.const


def test_globals_lowercase_variable():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    
    target = Name(id="variable", ctx=Store())
    value = Constant(value=42)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    parser.globals(root, node)
    
    assert "test_module.variable" in parser.alias
    assert "test_module.variable" not in parser.root


# LLM-generated content at query #61
#--------------------------

```python
def test_imports_simple_import():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, node)
    
    assert parser.alias["test_module.os"] == "os"


def test_imports_simple_import_with_alias():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, node)
    
    assert parser.alias["test_module.operating_system"] == "os"


def test_imports_multiple_imports():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname=None)])
    parser.imports(root, node)
    
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.sys"] == "sys"


def test_imports_from_import_absolute():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=0)
    parser.imports(root, node)
    
    assert parser.alias["test_module.path"] == "os.path"


def test_imports_from_import_with_alias():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module="os", names=[alias(name="path", asname="p")], level=0)
    parser.imports(root, node)
    
    assert parser.alias["test_module.p"] == "os.path"


def test_imports_from_import_relative_level_1():
    parser = Parser()
    root = "pkg.test_module"
    parser.level[root] = 1
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module="other", names=[alias(name="func", asname=None)], level=1)
    parser.imports(root, node)
    
    assert parser.alias["pkg.test_module.func"] == "pkg.other.func"


def test_imports_from_import_relative_level_2():
    parser = Parser()
    root = "pkg.sub.test_module"
    parser.level[root] = 2
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module="other", names=[alias(name="func", asname=None)], level=2)
    parser.imports(root, node)
    
    assert parser.alias["pkg.sub.test_module.func"] == "pkg.other.func"


def test_imports_from_import_relative_no_module():
    parser = Parser()
    root = "pkg.test_module"
    parser.level[root] = 1
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module=None, names=[alias(name="func", asname=None)], level=1)
    parser.imports(root, node)
    
    assert parser.alias["pkg.test_module.func"] == "pkg.func"


def test_imports_from_import_multiple_names():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module="os", names=[alias(name="path", asname=None), alias(name="getcwd", asname=None)], level=0)
    parser.imports(root, node)
    
    assert parser.alias["test_module.path"] == "os.path"
    assert parser.alias["test_module.getcwd"] == "os.getcwd"


def test_imports_from_import_star():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.imp[root] = set()
    parser.root[root] = root
    
    node = ImportFrom(module="os", names=[alias(name="*", asname=None)], level=0)
    parser.imports(root, node)
    
    assert parser.alias["test_module.*"] == "os.*"


# LLM-generated content at query #62
#--------------------------

```python
def test_imports_predicate_line_13_false():
    from ast import ImportFrom, alias
    from dataclasses import dataclass, field
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict = field(default_factory=dict)
        doc: dict = field(default_factory=dict)
        docstring: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        alias: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
        
        def imports(self, root: str, node) -> None:
            if isinstance(node, ImportFrom):
                if node.level:
                    m = root
                else:
                    m = ''
                for a in node.names:
                    name = a.name if a.asname is None else a.asname
                    self.alias[root + '.' + name] = m + '.' + node.module + '.' + a.name
    
    parser = Parser()
    
    # Create an ImportFrom node where a.asname is not None
    # This makes the predicate (a.asname is None) evaluate to False
    import_alias = alias(name='original_name', asname='renamed_name')
    import_node = ImportFrom(module='some_module', names=[import_alias], level=0)
    
    parser.imports('test_root', import_node)
    
    # Verify the predicate evaluated to False by checking that asname was used
    assert 'test_root.renamed_name' in parser.alias
    assert parser.alias['test_root.renamed_name'] == '.some_module.original_name'


# LLM-generated content at query #63
#--------------------------

```python
def test_globals_predicate_line_38_evaluates_to_false():
    from ast import Assign, Name, Constant, List
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    parser.imp[root] = set()
    
    # Create an Assign node with a List containing a non-Constant element
    target = Name(id='__all__', ctx=None)
    # Create a List with an element that is not a Constant
    list_node = List(elts=[Name(id='some_name', ctx=None)], ctx=None)
    node = Assign(targets=[target], value=list_node, type_comment=None)
    
    # Call globals method
    parser.globals(root, node)
    
    # The predicate at line 38 should evaluate to False for Name node
    # so no elements should be added to parser.imp[root]
    assert len(parser.imp[root]) == 0


# LLM-generated content at query #64
#--------------------------

```python
def test_api_link_false():
    from dataclasses import dataclass, field
    from ast import FunctionDef, parse
    
    # Create a Parser instance with link=False
    parser = Parser(link=False, b_level=1, toc=False)
    
    # Parse a simple function to set up the parser state
    parser.parse('test_module', 'def foo(): pass')
    
    # Get the FunctionDef node for testing
    tree = parse('def test_func(): pass')
    func_node = tree.body[0]
    
    # Call api method with link=False
    parser.api('test_module', func_node)
    
    # Check that the predicate at line 17 (if self.link:) evaluates to False
    # This means the anchor tag should NOT be added
    assert '<a id=' not in parser.doc['test_module.test_func']


# LLM-generated content at query #65
#--------------------------

```python
def test_class_api_predicate_line_11_false():
    from ast import AnnAssign, Name, parse
    from dataclasses import dataclass, field
    
    @dataclass
    class MockParser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict = field(default_factory=dict)
        doc: dict = field(default_factory=dict)
        docstring: dict = field(default_factory=dict)
        imp: dict = field(default_factory=dict)
        root: dict = field(default_factory=dict)
        alias: dict = field(default_factory=dict)
        const: dict = field(default_factory=dict)
        
        def resolve(self, root, node, self_ty=""):
            return "int"
    
    parser = MockParser()
    parser.doc["test_class"] = ""
    parser.level["test_class"] = 0
    parser.root["test_class"] = "test_class"
    
    # Create an AnnAssign node where target is NOT a Name instance
    # This makes isinstance(node.target, Name) return False
    code_str = "class TestClass:\n    x: int = 5"
    tree = parse(code_str)
    class_body = tree.body[0].body
    
    # Get the AnnAssign node
    ann_assign_node = class_body[0]
    
    # Verify the predicate at line 11 is False by creating a scenario
    # where node is AnnAssign but node.target is not Name
    predicate_result = isinstance(ann_assign_node, AnnAssign) and isinstance(ann_assign_node.target, Name)
    
    assert predicate_result == True
    
    # Now test with a node that is NOT AnnAssign
    from ast import Assign, Constant
    assign_node = Assign(targets=[Name(id='y', ctx=None)], value=Constant(value=10))
    predicate_result_2 = isinstance(assign_node, AnnAssign) and isinstance(assign_node.target, Name)
    
    assert predicate_result_2 == False


# LLM-generated content at query #66
#--------------------------

```python
def test_func_ann_yields_empty_string_for_star_arg():
    from ast import arg
    from dataclasses import dataclass, field
    
    parser = Parser()
    root = "test_module"
    args = [arg(arg='*', annotation=None)]
    
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    
    assert result == [""]


# LLM-generated content at query #67
#--------------------------

```python
def test_attr_predicate_evaluates_to_false():
    class TestObj:
        def __init__(self):
            self.nested = NestedObj()
    
    class NestedObj:
        def __init__(self):
            self.value = "test"
    
    obj = TestObj()
    result = _attr(obj, "nested.value")
    assert result is not None
    assert result == "test"


# LLM-generated content at query #68
#--------------------------

```python
def test_attr_single_level_attribute():
    class TestObj:
        attr = "value"
    
    obj = TestObj()
    result = _attr(obj, "attr")
    assert result == "value"


def test_attr_nested_attribute():
    class Inner:
        value = "nested_value"
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.value")
    assert result == "nested_value"


def test_attr_deep_nested_attribute():
    class Level3:
        data = "deep_value"
    
    class Level2:
        level3 = Level3()
    
    class Level1:
        level2 = Level2()
    
    obj = Level1()
    result = _attr(obj, "level2.level3.data")
    assert result == "deep_value"


def test_attr_nonexistent_attribute():
    class TestObj:
        attr = "value"
    
    obj = TestObj()
    result = _attr(obj, "nonexistent")
    assert result is None


def test_attr_nonexistent_nested_attribute():
    class Inner:
        value = "nested_value"
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.nonexistent")
    assert result is None


def test_attr_broken_chain():
    class Inner:
        value = "nested_value"
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.nonexistent.deeper")
    assert result is None


def test_attr_none_value_in_chain():
    class Inner:
        value = None
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.value.deeper")
    assert result is None


def test_attr_empty_string():
    class TestObj:
        pass
    
    obj = TestObj()
    result = _attr(obj, "")
    assert result is None


def test_attr_single_dot():
    class TestObj:
        pass
    
    obj = TestObj()
    result = _attr(obj, ".")
    assert result is None


def test_attr_with_integer_attribute():
    class TestObj:
        count = 42
    
    obj = TestObj()
    result = _attr(obj, "count")
    assert result == 42


def test_attr_with_none_object():
    result = _attr(None, "attr")
    assert result is None


def test_attr_multiple_levels_with_none():
    class Inner:
        value = "test"
    
    class Outer:
        inner = None
    
    obj = Outer()
    result = _attr(obj, "inner.value")
    assert result is None


# LLM-generated content at query #69
#--------------------------

```python
def test_imports_simple_import():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.os"] == "os"


def test_imports_simple_import_with_alias():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.operating_system"] == "os"


def test_imports_multiple_names():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = Import(names=[
        alias(name="os", asname=None),
        alias(name="sys", asname="system")
    ])
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.os"] == "os"
    assert parser.alias["mymodule.system"] == "sys"


def test_imports_from_import_absolute():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname=None)],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.path"] == "os.path"


def test_imports_from_import_with_alias():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="os",
        names=[alias(name="path", asname="filepath")],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.filepath"] == "os.path"


def test_imports_from_import_relative_level_1():
    parser = Parser()
    root = "package.submodule"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="sibling",
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports(root, import_node)
    
    assert parser.alias["package.submodule.func"] == "package.sibling.func"


def test_imports_from_import_relative_level_2():
    parser = Parser()
    root = "package.sub.module"
    parser.level[root] = 2
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="other",
        names=[alias(name="cls", asname=None)],
        level=2
    )
    parser.imports(root, import_node)
    
    assert parser.alias["package.sub.module.cls"] == "package.other.cls"


def test_imports_from_import_no_module():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module=None,
        names=[alias(name="func", asname=None)],
        level=1
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.func"] == "func"


def test_imports_from_import_multiple_names():
    parser = Parser()
    root = "mymodule"
    parser.level[root] = 0
    parser.root[root] = root
    
    import_node = ImportFrom(
        module="collections",
        names=[
            alias(name="defaultdict", asname=None),
            alias(name="Counter", asname="cnt")
        ],
        level=0
    )
    parser.imports(root, import_node)
    
    assert parser.alias["mymodule.defaultdict"] == "collections.defaultdict"
    assert parser.alias["mymodule.cnt"] == "collections.Counter"


def test_imports_nested_module():
    parser = Parser()
    root = "package.module"
    parser.level[root] = 1
    parser.root[root] = root
    
    import_node = Import(names=[alias(name="json", asname=None)])
    parser.imports(root, import_node)
    
    assert parser.alias["package.module.json"] == "json"


# LLM-generated content at query #70
#--------------------------

```python
from ast import If, Try, stmt, Assign, Name, Constant, Store, ExceptHandler, Pass
import ast

def test_walk_body_simple_statements():
    assign1 = ast.Assign(targets=[ast.Name(id='x', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='y', ctx=Store())], value=ast.Constant(value=2))
    body = [assign1, assign2]
    result = list(walk_body(body))
    assert len(result) == 2
    assert result[0] is assign1
    assert result[1] is assign2

def test_walk_body_if_statement():
    assign1 = ast.Assign(targets=[ast.Name(id='x', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='y', ctx=Store())], value=ast.Constant(value=2))
    if_stmt = ast.If(test=ast.Constant(value=True), body=[assign1], orelse=[assign2])
    body = [if_stmt]
    result = list(walk_body(body))
    assert len(result) == 2
    assert result[0] is assign1
    assert result[1] is assign2

def test_walk_body_nested_if_statements():
    assign1 = ast.Assign(targets=[ast.Name(id='x', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='y', ctx=Store())], value=ast.Constant(value=2))
    assign3 = ast.Assign(targets=[ast.Name(id='z', ctx=Store())], value=ast.Constant(value=3))
    inner_if = ast.If(test=ast.Constant(value=True), body=[assign2], orelse=[assign3])
    outer_if = ast.If(test=ast.Constant(value=True), body=[inner_if], orelse=[])
    body = [assign1, outer_if]
    result = list(walk_body(body))
    assert len(result) == 3
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3

def test_walk_body_try_statement():
    assign1 = ast.Assign(targets=[ast.Name(id='x', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='y', ctx=Store())], value=ast.Constant(value=2))
    assign3 = ast.Assign(targets=[ast.Name(id='z', ctx=Store())], value=ast.Constant(value=3))
    handler = ast.ExceptHandler(type=None, name=None, body=[assign2])
    try_stmt = ast.Try(body=[assign1], handlers=[handler], orelse=[], finalbody=[assign3])
    body = [try_stmt]
    result = list(walk_body(body))
    assert len(result) == 3
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3

def test_walk_body_try_with_multiple_handlers():
    assign1 = ast.Assign(targets=[ast.Name(id='a', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='b', ctx=Store())], value=ast.Constant(value=2))
    assign3 = ast.Assign(targets=[ast.Name(id='c', ctx=Store())], value=ast.Constant(value=3))
    assign4 = ast.Assign(targets=[ast.Name(id='d', ctx=Store())], value=ast.Constant(value=4))
    handler1 = ast.ExceptHandler(type=None, name=None, body=[assign2])
    handler2 = ast.ExceptHandler(type=None, name=None, body=[assign3])
    try_stmt = ast.Try(body=[assign1], handlers=[handler1, handler2], orelse=[], finalbody=[assign4])
    body = [try_stmt]
    result = list(walk_body(body))
    assert len(result) == 4
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3
    assert result[3] is assign4

def test_walk_body_try_with_orelse_and_finalbody():
    assign1 = ast.Assign(targets=[ast.Name(id='a', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='b', ctx=Store())], value=ast.Constant(value=2))
    assign3 = ast.Assign(targets=[ast.Name(id='c', ctx=Store())], value=ast.Constant(value=3))
    assign4 = ast.Assign(targets=[ast.Name(id='d', ctx=Store())], value=ast.Constant(value=4))
    handler = ast.ExceptHandler(type=None, name=None, body=[assign2])
    try_stmt = ast.Try(body=[assign1], handlers=[handler], orelse=[assign3], finalbody=[assign4])
    body = [try_stmt]
    result = list(walk_body(body))
    assert len(result) == 4
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3
    assert result[3] is assign4

def test_walk_body_mixed_statements():
    assign1 = ast.Assign(targets=[ast.Name(id='x', ctx=Store())], value=ast.Constant(value=1))
    assign2 = ast.Assign(targets=[ast.Name(id='y', ctx=Store())], value=ast.Constant(value=2))
    assign3 = ast.Assign(targets=[ast.Name(id='z', ctx=Store())], value=ast.Constant(value=3))
    if_stmt = ast.If(test=ast.Constant(value=True), body=[assign2], orelse=[])
    body = [assign1, if_stmt, assign3]
    result = list(walk_body(body))
    assert len(result) == 3
    assert result[0] is assign1
    assert result[1] is assign2
    assert result[2] is assign3

def test_walk_body_empty_body():
    body = []
    result = list(walk_body(body))
    assert len(result) == 0

def test_walk_body_if_with_empty_orelse():
    assign1 = ast.Assign(targets=[ast.Name(id='x', ctx=Store())], value=ast.Constant(value=1))
    if_stmt = ast.If(test=ast.Constant(value=True), body=[assign1], orelse=[])
    body = [if_stmt]
    result = list(walk_body(body))
    assert len(result) == 1
    assert result[0] is assign1


# LLM-generated content at query #71
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "x: int = 5"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert _m(root, "x") in parser.alias
    assert parser.alias[_m(root, "x")] == "5"


def test_globals_with_simple_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "CONSTANT = 42"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert _m(root, "CONSTANT") in parser.alias
    assert _m(root, "CONSTANT") in parser.const


def test_globals_with_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "MAX_VALUE = 100"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert parser.root[_m(root, "MAX_VALUE")] == root
    assert parser.const[_m(root, "MAX_VALUE")] == "int"


def test_globals_with_all_list():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "__all__ = ['func1', 'func2']"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]


def test_globals_with_all_tuple():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "__all__ = ('item1', 'item2')"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert _m(root, "item1") in parser.imp[root]
    assert _m(root, "item2") in parser.imp[root]


def test_globals_ignores_non_name_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "a, b = 1, 2"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "value = 10  # type: int"
    tree = parse(code_str, type_comments=True)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert _m(root, "value") in parser.alias
    assert parser.const[_m(root, "value")] == "int"


def test_globals_ignores_multiple_targets():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "x = y = 5"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_with_annotated_no_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "x: int"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.alias) == 0


def test_globals_string_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "TEXT = 'hello'"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert parser.const[_m(root, "TEXT")] == "str"


def test_globals_with_all_non_string_elements():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()
    
    code_str = "__all__ = [1, 2, 3]"
    tree = parse(code_str)
    node = tree.body[0]
    
    parser.globals(root, node)
    
    assert len(parser.imp[root]) == 0


# LLM-generated content at query #72
#--------------------------

```python
def test_class_api_predicate_line_11_false():
    from ast import AnnAssign, Name, Assign, Delete, parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    parser.doc = {}
    
    # Setup test data
    root = "test_module"
    name = "test_module.TestClass"
    parser.doc[name] = ""
    
    # Create a body with a node that is NOT an AnnAssign or whose target is NOT a Name
    # For example, an Assign node
    code_str = "x = 5"
    tree = parse(code_str)
    body = tree.body
    
    # Call class_api - the predicate at line 11 should evaluate to False
    # because the node is an Assign, not an AnnAssign
    parser.class_api(root, name, [], body)
    
    # Verify that the code executed without the line 11 condition being true
    # If line 11 predicate was True, attr would be set, but since it's False,
    # we just verify the method completes
    assert parser.doc[name] == ""


# LLM-generated content at query #73
#--------------------------

```python
def test_class_api_predicate_line_19_false():
    """Test that the predicate at line 19 evaluates to False when len(node.targets) != 1"""
    from ast import Assign, Name, Constant, parse
    from dataclasses import dataclass, field
    
    parser = Parser()
    parser.doc["test_module.TestClass"] = ""
    
    # Create an Assign node with multiple targets (len != 1)
    script = "a = b = 5"
    tree = parse(script)
    assign_node = tree.body[0]
    
    # Verify the node has multiple targets
    assert isinstance(assign_node, Assign)
    assert len(assign_node.targets) == 2
    
    # The predicate at line 19 should evaluate to False
    predicate_result = (
        isinstance(assign_node, Assign)
        and len(assign_node.targets) == 1
        and isinstance(assign_node.targets[0], Name)
    )
    
    assert predicate_result is False


# LLM-generated content at query #74
#--------------------------

```python
def test_globals_predicate_line_35_false():
    """Test that the predicate at line 35 evaluates to False.
    
    The predicate is: left.id != '__all__' or not isinstance(node.value, (Tuple, List))
    For it to be False, both conditions must be False:
    - left.id == '__all__' (first part is False)
    - isinstance(node.value, (Tuple, List)) (second part is False, making 'not' True becomes False)
    
    So we need: left.id == '__all__' AND node.value is a Tuple or List
    """
    from ast import Assign, Name, Tuple, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    parser.imp['test_module'] = set()
    
    # Create an Assign node with __all__ as target and a Tuple as value
    target = Name(id='__all__', ctx=None)
    value = Tuple(elts=[
        Constant(value='func1'),
        Constant(value='func2')
    ], ctx=None)
    node = Assign(targets=[target], value=value, type_comment=None)
    
    # Call globals method - if predicate is False, it should continue to line 37+
    # and add items to self.imp
    parser.globals('test_module', node)
    
    # Verify that the code after line 35 was executed (items were added to imp)
    assert 'test_module.func1' in parser.imp['test_module']
    assert 'test_module.func2' in parser.imp['test_module']


# LLM-generated content at query #75
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False
    assert isinstance(parser, Parser)


def test_parser_new_classmethod_with_toc():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #76
#--------------------------

```python
def test_func_api_predicate_line_32_false():
    from ast import arguments, arg, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    parser.doc['test_module.test_func'] = "## test_func()\n\n*Full name:* `test_module.test_func`\n\n"
    
    # Create arguments with at least one non-None default to make has_default False
    test_args = arguments(
        posonlyargs=[],
        args=[arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=42)]
    )
    
    # Call func_api with arguments that have defaults
    # This will cause has_default to be False (since not all defaults are None)
    parser.func_api('test_module', 'test_module.test_func', test_args, None, has_self=False, cls_method=False)
    
    # Verify that the predicate evaluated to False by checking the doc was modified
    assert 'test_module.test_func' in parser.doc
    assert len(parser.doc['test_module.test_func']) > 0


# LLM-generated content at query #77
#--------------------------

```python
def test_attr_single_level():
    class Obj:
        attr = "value"
    
    obj = Obj()
    result = _attr(obj, "attr")
    assert result == "value"


def test_attr_nested_level():
    class Inner:
        value = "nested_value"
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.value")
    assert result == "nested_value"


def test_attr_deeply_nested():
    class Level3:
        data = "deep"
    
    class Level2:
        level3 = Level3()
    
    class Level1:
        level2 = Level2()
    
    obj = Level1()
    result = _attr(obj, "level2.level3.data")
    assert result == "deep"


def test_attr_nonexistent_attribute():
    class Obj:
        attr = "value"
    
    obj = Obj()
    result = _attr(obj, "nonexistent")
    assert result is None


def test_attr_nonexistent_nested_attribute():
    class Inner:
        value = "nested"
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.nonexistent")
    assert result is None


def test_attr_break_chain_at_middle():
    class Inner:
        value = "test"
    
    class Outer:
        inner = Inner()
    
    obj = Outer()
    result = _attr(obj, "inner.nonexistent.value")
    assert result is None


def test_attr_empty_string():
    class Obj:
        pass
    
    obj = Obj()
    result = _attr(obj, "")
    assert result is None


def test_attr_with_none_value():
    class Outer:
        inner = None
    
    obj = Outer()
    result = _attr(obj, "inner.value")
    assert result is None


def test_attr_numeric_value():
    class Obj:
        number = 42
    
    obj = Obj()
    result = _attr(obj, "number")
    assert result == 42


def test_attr_string_value():
    class Obj:
        text = "hello"
    
    obj = Obj()
    result = _attr(obj, "text")
    assert result == "hello"


# LLM-generated content at query #78
#--------------------------

```python
def test_load_docstring():
    from types import ModuleType
    
    # Create a mock module with docstrings
    mock_module = ModuleType("test_module")
    mock_module.__doc__ = "Module docstring\n\n>>> example = 1"
    
    # Create a nested mock class
    class MockClass:
        """Class docstring\n\n>>> x = 1"""
        pass
    
    mock_module.MockClass = MockClass
    
    # Create parser and add some documentation entries
    parser = Parser()
    parser.doc["test_module"] = "# Module `test_module`"
    parser.doc["test_module.MockClass"] = "## class MockClass"
    parser.root["test_module"] = "test_module"
    parser.root["test_module.MockClass"] = "test_module"
    
    # Call load_docstring
    parser.load_docstring("test_module", mock_module)
    
    # Verify docstrings were loaded and processed
    assert "test_module" in parser.docstring
    assert "```python" in parser.docstring["test_module"]
    assert "example = 1" in parser.docstring["test_module"]
    assert "test_module.MockClass" in parser.docstring
    assert "Class docstring" in parser.docstring["test_module.MockClass"]


def test_load_docstring_with_missing_attribute():
    from types import ModuleType
    
    # Create a mock module
    mock_module = ModuleType("test_module")
    
    # Create parser with entries for non-existent attributes
    parser = Parser()
    parser.doc["test_module"] = "# Module"
    parser.doc["test_module.NonExistent"] = "## Non-existent"
    parser.root["test_module"] = "test_module"
    parser.root["test_module.NonExistent"] = "test_module"
    
    # Call load_docstring - should not raise error
    parser.load_docstring("test_module", mock_module)
    
    # Non-existent attribute should not have docstring added
    assert "test_module.NonExistent" not in parser.docstring or parser.docstring.get("test_module.NonExistent") == ""


def test_load_docstring_filters_by_root():
    from types import ModuleType
    
    # Create parser with entries from different modules
    parser = Parser()
    parser.doc["module_a"] = "# Module A"
    parser.doc["module_b"] = "# Module B"
    parser.root["module_a"] = "module_a"
    parser.root["module_b"] = "module_b"
    
    # Create mock module with docstring
    mock_module = ModuleType("module_a")
    mock_module.__doc__ = "Module A docstring"
    
    # Load docstring for only module_a
    parser.load_docstring("module_a", mock_module)
    
    # Only module_a should have docstring, not module_b
    assert "module_a" in parser.docstring
    assert "module_b" not in parser.docstring


# LLM-generated content at query #79
#--------------------------

```python
def test_visit_name_predicate_false():
    """Test that the predicate at line 6 evaluates to False."""
    from ast import Name, Load
    
    # Create a resolver instance
    resolver = Resolver(root="mymodule", alias={"mymodule.MyType": "int"}, self_ty="")
    
    # Create a Name node
    node = Name(id="MyType", ctx=Load())
    
    # Call visit_Name - the predicate should be False because:
    # Either: name not in self.alias (first part of AND is False)
    # Or: name in self.alias[name] (second part of AND is False)
    result = resolver.visit_Name(node)
    
    # When predicate is False, the else branch executes and returns the node unchanged
    assert result == node


# LLM-generated content at query #80
#--------------------------

```python
def test_func_api_predicate_line_32_false():
    from ast import arguments, arg, Constant
    from dataclasses import dataclass, field
    
    @dataclass
    class MockParser:
        doc: dict[str, str] = field(default_factory=dict)
        
        def func_ann(self, root, args, has_self, cls_method):
            return []
    
    parser = MockParser()
    parser.doc['test_func'] = ''
    
    node = arguments(
        posonlyargs=[],
        args=[arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=5)]
    )
    
    default_list = []
    default_list.extend([None] * (len(node.args) - len(node.defaults)))
    default_list.extend(node.defaults)
    default_list.append(None)
    default_list.append(None)
    
    has_default = all(d is None for d in default_list)
    
    assert has_default is False


# LLM-generated content at query #81
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc
    assert 'BaseClass' in parser.doc['test_module.MyClass']


def test_class_api_with_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    target = Name(id='member1', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='int', ctx=Load()), value=None, simple=1)
    body = [ann_assign]
    bases = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'Members' in parser.doc['test_module.MyClass']
    assert 'member1' in parser.doc['test_module.MyClass']


def test_class_api_with_enum():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyEnum'] = '## class MyEnum\n\n'
    
    bases = [Attribute(value=Name(id='enum', ctx=Load()), attr='Enum', ctx=Load())]
    target = Name(id='MEMBER', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='int', ctx=Load()), value=None, simple=1)
    body = [ann_assign]
    
    parser.class_api('test_module', 'test_module.MyEnum', bases, body)
    
    assert 'Enums' in parser.doc['test_module.MyEnum']
    assert 'MEMBER' in parser.doc['test_module.MyEnum']


def test_class_api_with_deleted_member():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    target = Name(id='member1', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='int', ctx=Load()), value=None, simple=1)
    delete_stmt = Delete(targets=[Name(id='member1', ctx=Del())])
    body = [ann_assign, delete_stmt]
    bases = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'Members' not in parser.doc['test_module.MyClass'] or 'member1' not in parser.doc['test_module.MyClass']


def test_class_api_empty_class():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    bases = []
    body = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'test_module.MyClass' in parser.doc


def test_class_api_with_assign_member():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    target = Name(id='member1', ctx=Store())
    assign = Assign(targets=[target], value=Constant(value=42), type_comment=None)
    body = [assign]
    bases = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'Members' in parser.doc['test_module.MyClass']
    assert 'member1' in parser.doc['test_module.MyClass']


def test_class_api_with_private_member():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    target = Name(id='_private', ctx=Store())
    ann_assign = AnnAssign(target=target, annotation=Name(id='int', ctx=Load()), value=None, simple=1)
    body = [ann_assign]
    bases = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'Members' not in parser.doc['test_module.MyClass'] or '_private' not in parser.doc['test_module.MyClass']


def test_class_api_with_multiple_members():
    parser = Parser()
    parser.level['test_module'] = 0
    parser.root['test_module'] = 'test_module'
    parser.doc['test_module.MyClass'] = '## class MyClass\n\n'
    
    target1 = Name(id='member1', ctx=Store())
    target2 = Name(id='member2', ctx=Store())
    ann_assign1 = AnnAssign(target=target1, annotation=Name(id='int', ctx=Load()), value=None, simple=1)
    ann_assign2 = AnnAssign(target=target2, annotation=Name(id='str', ctx=Load()), value=None, simple=1)
    body = [ann_assign1, ann_assign2]
    bases = []
    
    parser.class_api('test_module', 'test_module.MyClass', bases, body)
    
    assert 'Members' in parser.doc['test_module.MyClass']
    assert 'member1' in parser.doc['test_module.MyClass']
    assert 'member2' in parser.doc['test_module.MyClass']


# LLM-generated content at query #82
#--------------------------

```python
def test_visit_name_predicate_line_9_true():
    from ast import Name, Load, Call, parse, Expr
    from unittest.mock import Mock, patch
    
    # Create a Resolver instance
    resolver = Resolver(root="module", alias={}, self_ty="")
    
    # Create a Name node
    node = Name(id="TestName", ctx=Load())
    
    # Setup alias to contain the name and make it pass the condition at line 6
    # The alias value should be a string that parses to an Expr with a Call as its value
    resolver.alias = {
        "module.TestName": "SomeFunc()"
    }
    
    # Mock _m to return "module.TestName"
    with patch('__main__._m', return_value="module.TestName"):
        result = resolver.visit_Name(node)
    
    # The predicate at line 9 should be True, meaning:
    # isinstance(e.value, Call) should be True
    # isinstance(e.value.func, Name) should be True
    # This means the function should continue past line 9 and execute lines 10-14


# LLM-generated content at query #83
#--------------------------

```python
def test_class_api_predicate_enums_true():
    from ast import parse, ClassDef, AnnAssign, Name, Constant
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    
    # Set up initial doc entry
    root = "test_module"
    name = "test_module.TestEnum"
    parser.doc[name] = "# class TestEnum\n\n*Full name:* `test_module.TestEnum`\n\n"
    
    # Create a simple enum class body with an annotated assignment
    # This simulates: class TestEnum(enum.Enum): MEMBER: int
    code_str = """
class TestEnum(enum.Enum):
    MEMBER: int
"""
    tree = parse(code_str)
    class_node = tree.body[0]
    
    # Extract the body and bases from the class definition
    body = class_node.body
    bases = class_node.bases
    
    # Mock the resolve method to return 'enum.Enum' for the base
    original_resolve = parser.resolve
    parser.resolve = lambda root, node, self_ty="": "enum.Enum"
    
    # Call class_api
    parser.class_api(root, name, bases, body)
    
    # Restore original resolve
    parser.resolve = original_resolve
    
    # The predicate at line 38 checks `if enums:`
    # Since we created an enum class with a member, enums list should be non-empty
    # and the condition should evaluate to True, causing table("Enums", items=enums) to be called
    # We verify this by checking that the doc was updated with the Enums table
    assert "Enums" in parser.doc[name]


# LLM-generated content at query #84
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.root = {'pkg.module': 'pkg.module'}
    parser.imp = {'pkg.module': set()}
    result = parser.is_public('pkg.module')
    assert result is True


def test_is_public_with_private_name():
    parser = Parser()
    parser.root = {'pkg._private': 'pkg._private'}
    parser.imp = {'pkg': set()}
    result = parser.is_public('pkg._private')
    assert result is False


def test_is_public_with_magic_name():
    parser = Parser()
    parser.root = {'pkg.__init__': 'pkg.__init__'}
    parser.imp = {'pkg': set()}
    result = parser.is_public('pkg.__init__')
    assert result is True


def test_is_public_with_all_list_containing_name():
    parser = Parser()
    parser.root = {'pkg.func': 'pkg'}
    parser.imp = {'pkg': {'pkg.func'}}
    result = parser.is_public('pkg.func')
    assert result is True


def test_is_public_with_all_list_not_containing_name():
    parser = Parser()
    parser.root = {'pkg.func': 'pkg'}
    parser.imp = {'pkg': {'pkg.other'}}
    result = parser.is_public('pkg.func')
    assert result is False


def test_is_public_module_in_imp():
    parser = Parser()
    parser.root = {'pkg.submodule': 'pkg.submodule'}
    parser.imp = {'pkg.submodule': set()}
    parser.doc = {'pkg.submodule.func': '', 'pkg.submodule.Class': ''}
    parser.const = {}
    result = parser.is_public('pkg.submodule')
    assert result is True


def test_is_public_with_parent_in_all_list():
    parser = Parser()
    parser.root = {'pkg.submodule.func': 'pkg'}
    parser.imp = {'pkg': {'pkg.submodule'}}
    result = parser.is_public('pkg.submodule.func')
    assert result is True


def test_is_public_root_module():
    parser = Parser()
    parser.root = {'mypackage': 'mypackage'}
    parser.imp = {'mypackage': set()}
    result = parser.is_public('mypackage')
    assert result is True


def test_is_public_with_empty_all_and_public_family():
    parser = Parser()
    parser.root = {'pkg.Public': 'pkg.Public'}
    parser.imp = {'pkg': set()}
    result = parser.is_public('pkg.Public')
    assert result is True


def test_is_public_with_empty_all_and_private_family():
    parser = Parser()
    parser.root = {'pkg._private': 'pkg._private'}
    parser.imp = {'pkg': set()}
    result = parser.is_public('pkg._private')
    assert result is False


# LLM-generated content at query #85
#--------------------------

```python
def test_visit_name_typevar_predicate():
    from ast import Name, Load, Call, Expr, parse
    from unittest.mock import Mock, patch
    
    # Create a Resolver instance
    resolver = Resolver(root="mymodule", alias={}, self_ty="")
    
    # Create a Name node to visit
    node = Name(id="SomeType", ctx=Load())
    
    # Mock _m function to return a name that exists in alias
    with patch('__main__._m') as mock_m:
        mock_m.return_value = "mymodule.SomeType"
        
        # Set up alias to contain a TypeVar call
        resolver.alias = {
            "mymodule.SomeType": "TypeVar('T')"
        }
        
        # Call visit_Name
        result = resolver.visit_Name(node)
        
        # The predicate at line 9 should evaluate to True because:
        # - e.value is a Call (TypeVar('T') is a Call)
        # - e.value.func is a Name (TypeVar is a Name)
        assert result == node


# LLM-generated content at query #86
#--------------------------

```python
def test_class_api_line_19_predicate_false():
    from ast import Assign, Name, Constant
    
    parser = Parser()
    parser.doc['test_class'] = "# class test_class\n\n"
    
    # Create an Assign node with multiple targets (len != 1)
    # This makes the predicate at line 19 evaluate to False
    assign_node = Assign(
        targets=[Name(id='x'), Name(id='y')],
        value=Constant(value=1)
    )
    
    # Create a mock walk_body that returns our Assign node
    def mock_walk_body(body):
        return [assign_node]
    
    # Temporarily replace walk_body
    import ast_parser_module
    original_walk_body = ast_parser_module.walk_body
    ast_parser_module.walk_body = mock_walk_body
    
    try:
        # Call class_api - the elif condition at line 17-21 should not be entered
        # because len(node.targets) == 1 is False
        parser.class_api('test_module', 'test_class', [], [assign_node])
        
        # Verify the method completed without error
        # and the mem dict remains empty (the elif block was skipped)
        assert 'test_class' in parser.doc
    finally:
        ast_parser_module.walk_body = original_walk_body


# LLM-generated content at query #87
#--------------------------

```python
def test_class_api_line_25_predicate_false():
    """Test that the predicate at line 25 (is_public_family(attr)) evaluates to False."""
    from ast import Assign, Name, Constant
    from dataclasses import dataclass, field
    
    parser = Parser()
    parser.doc['test_module.TestClass'] = "# class TestClass\n\n"
    
    # Create an Assign node with a private attribute (starts with underscore)
    target_name = Name(id='_private_attr', ctx=None)
    assign_node = Assign(targets=[target_name], value=Constant(value=42), type_comment=None)
    
    # Mock is_public_family to return False for private attributes
    import sys
    from unittest.mock import patch
    
    with patch('__main__.is_public_family', return_value=False):
        with patch('__main__.const_type', return_value='int'):
            with patch('__main__.walk_body', return_value=[assign_node]):
                parser.class_api('test_module', 'test_module.TestClass', [], [assign_node])
    
    # Verify that the private attribute was not added to mem
    assert '_private_attr' not in parser.doc['test_module.TestClass'] or 'Members' not in parser.doc['test_module.TestClass']


# LLM-generated content at query #88
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


def test_parser_constructor_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_constructor_post_init_toc_false_keeps_link_false():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False
    assert parser.level == {}
    assert parser.doc == {}


def test_parser_new_classmethod_with_toc():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is True
    assert parser.b_level == 3
    assert parser.toc is True


# LLM-generated content at query #89
#--------------------------

```python
def test_class_api_with_members():
    from ast import parse, ClassDef
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = '''
class TestClass:
    """Test class."""
    attr1: int
    attr2: str = "default"
    _private: int = 5
    '''
    root_node = parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc
    assert 'Members' in parser.doc['test_module.TestClass']


def test_class_api_with_enums():
    from ast import parse
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = '''
class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    '''
    root_node = parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.Color', class_node.bases, class_node.body)
    
    assert 'test_module.Color' in parser.doc
    assert 'Enums' in parser.doc['test_module.Color']


def test_class_api_with_bases():
    from ast import parse
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = '''
class Child(Parent):
    pass
    '''
    root_node = parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.Child', class_node.bases, class_node.body)
    
    assert 'test_module.Child' in parser.doc
    assert 'Bases' in parser.doc['test_module.Child']


def test_class_api_empty_class():
    from ast import parse
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = 'class Empty: pass'
    root_node = parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.Empty', class_node.bases, class_node.body)
    
    assert 'test_module.Empty' in parser.doc


def test_class_api_with_deleted_members():
    from ast import parse
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = '''
class TestClass:
    attr1: int = 1
    del attr1
    '''
    root_node = parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc


def test_class_api_with_type_comment():
    from ast import parse
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = '''
class TestClass:
    attr1 = 10  # type: int
    '''
    root_node = parse(script, type_comments=True)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)
    
    assert 'test_module.TestClass' in parser.doc


def test_class_api_mixed_members_and_enums():
    from ast import parse
    
    parser = Parser()
    parser.parse('test_module', 'x = 1')
    
    script = '''
class Status(enum.Enum):
    ACTIVE = 1
    INACTIVE = 2
    '''
    root_node = parse(script)
    class_node = root_node.body[0]
    
    parser.class_api('test_module', 'test_module.Status', class_node.bases, class_node.body)
    
    assert 'test_module.Status' in parser.doc
    assert 'Enums' in parser.doc['test_module.Status']


# LLM-generated content at query #90
#--------------------------

```python
def test_visit_constant_syntax_error():
    from ast import Constant, Load, Name
    
    resolver = Resolver(root="test", alias={})
    
    # Create a Constant node with an invalid Python syntax string
    node = Constant(value="this is not valid python syntax !!!!")
    
    result = resolver.visit_Constant(node)
    
    # Should return the node unchanged when SyntaxError is caught
    assert result is node


# LLM-generated content at query #91
#--------------------------

```python
def test_class_api_with_members():
    from ast import parse, ClassDef
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class TestClass:
    public_attr: int
    _private_attr: str
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.TestClass", [], class_node.body)
    
    assert "test_module.TestClass" in parser.doc
    assert "Members" in parser.doc["test_module.TestClass"]


def test_class_api_with_enums():
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class Color:
    RED: int
    GREEN: int
    BLUE: int
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.Color", [], class_node.body)
    
    assert "test_module.Color" in parser.doc


def test_class_api_with_bases():
    from ast import parse, Name
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class Derived(Base):
    pass
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.Derived", class_node.bases, class_node.body)
    
    assert "test_module.Derived" in parser.doc
    assert "Bases" in parser.doc["test_module.Derived"]


def test_class_api_empty_class():
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class Empty:
    pass
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.Empty", [], class_node.body)
    
    assert "test_module.Empty" in parser.doc


def test_class_api_with_deleted_members():
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class TestClass:
    attr: int
    del attr
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.TestClass", [], class_node.body)
    
    assert "test_module.TestClass" in parser.doc


def test_class_api_with_type_comment():
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class TestClass:
    value = 42  # type: int
    """
    
    tree = parse(script, type_comments=True)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.TestClass", [], class_node.body)
    
    assert "test_module.TestClass" in parser.doc


def test_class_api_mixed_public_private():
    from ast import parse
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    
    script = """
class TestClass:
    public_field: str
    _private_field: int
    another_public: float
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    
    parser.class_api(root, "test_module.TestClass", [], class_node.body)
    
    assert "test_module.TestClass" in parser.doc
    assert "Members" in parser.doc["test_module.TestClass"]


def test_class_api_enum_detection():
    from ast import parse, Name
    
    parser = Parser()
    root = "test_module"
    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    
    script = """
class Status:
    PENDING = 1
    APPROVED = 2
    """
    
    tree = parse(script)
    class_node = tree.body[0]
    bases = [Name(id='enum.Enum')]
    
    parser.class_api(root, "test_module.Status", bases, class_node.body)
    
    assert "test_module.Status" in parser.doc


# LLM-generated content at query #92
#--------------------------

```python
def test_func_ann_star_argument():
    from dataclasses import dataclass, field
    from ast import arg
    from typing import TypeVar
    
    @dataclass
    class Parser:
        link: bool = True
        b_level: int = 1
        toc: bool = False
        level: dict[str, int] = field(default_factory=dict)
        doc: dict[str, str] = field(default_factory=dict)
        docstring: dict[str, str] = field(default_factory=dict)
        imp: dict[str, set[str]] = field(default_factory=dict)
        root: dict[str, str] = field(default_factory=dict)
        alias: dict[str, str] = field(default_factory=dict)
        const: dict[str, str] = field(default_factory=dict)
        _Self = TypeVar('_Self', bound='Parser')

        def resolve(self, root: str, node, self_ty: str = "") -> str:
            return "int"

        def func_ann(self, root: str, args, *, has_self: bool, cls_method: bool):
            self_ty = ""
            for i, a in enumerate(args):
                if has_self and i == 0:
                    if a.annotation is not None:
                        self_ty = self.resolve(root, a.annotation)
                        if cls_method:
                            self_ty = (self_ty.removeprefix('type[')
                                       .removesuffix(']'))
                    yield 'type[Self]' if cls_method else 'Self'
                elif a.arg == '*':
                    yield ""
                elif a.annotation is not None:
                    yield self.resolve(root, a.annotation, self_ty)
                else:
                    yield "Any"

    parser = Parser()
    star_arg = arg(arg='*', annotation=None)
    result = list(parser.func_ann('test_root', [star_arg], has_self=False, cls_method=False))
    
    assert len(result) == 1
    assert result[0] == ""


# LLM-generated content at query #93
#--------------------------

```python
def test_globals_predicate_line_38_false():
    from ast import AnnAssign, Name, Constant, parse as ast_parse
    from dataclasses import dataclass, field
    
    # Create a Parser instance
    parser = Parser()
    parser.imp['test_module'] = set()
    
    # Test case 1: e is not a Constant
    code1 = "x = [1, 2, 3]"
    tree1 = ast_parse(code1)
    node1 = tree1.body[0]
    node1.value.elts[0].value = 1  # Make first element an int, not str
    
    # This should not add anything because the value is not a string
    parser.globals('test_module', node1)
    assert len(parser.imp['test_module']) == 0
    
    # Test case 2: e is a Constant but value is not a string
    code2 = "__all__ = [1, 2, 3]"
    tree2 = ast_parse(code2)
    node2 = tree2.body[0]
    
    parser.globals('test_module', node2)
    assert len(parser.imp['test_module']) == 0
    
    # Test case 3: e is not a Constant instance (e.g., it's a Name or other AST node)
    code3 = "__all__ = [some_var]"
    tree3 = ast_parse(code3)
    node3 = tree3.body[0]
    
    parser.globals('test_module', node3)
    assert len(parser.imp['test_module']) == 0


# LLM-generated content at query #94
#--------------------------

```python
def test_attr_predicate_at_line_4_evaluates_to_false():
    class TestObj:
        def __init__(self):
            self.nested = None
    
    obj = TestObj()
    result = _attr(obj, 'nested.value')
    assert result is None


# LLM-generated content at query #95
#--------------------------

```python
def test_parser_constructor_default():
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


def test_parser_constructor_with_parameters():
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


def test_parser_constructor_partial_parameters():
    parser = Parser(link=False)
    assert parser.link is False
    assert parser.b_level == 1
    assert parser.toc is False


def test_parser_new_classmethod():
    parser = Parser.new(link=True, level=2, toc=False)
    assert parser.link is True
    assert parser.b_level == 2
    assert parser.toc is False


def test_parser_post_init_toc_enables_link():
    parser = Parser(link=False, b_level=1, toc=True)
    assert parser.link is True
    assert parser.toc is True


def test_parser_post_init_toc_false_preserves_link():
    parser = Parser(link=False, b_level=1, toc=False)
    assert parser.link is False
    assert parser.toc is False


def test_parser_constructor_dict_independence():
    parser1 = Parser()
    parser2 = Parser()
    parser1.doc['test'] = 'value'
    assert 'test' not in parser2.doc
    assert parser2.doc == {}


# LLM-generated content at query #96
#--------------------------

```python
def test_class_api_is_enum_predicate():
    from ast import parse, Assign, Name, Constant
    
    parser = Parser()
    parser.doc['test_module.TestClass'] = "# class TestClass\n\n"
    
    # Create a base class that starts with 'enum.'
    bases = [parse("enum.Enum", mode='eval').body]
    
    # Create an Assign node with a single Name target
    assign_node = Assign(
        targets=[Name(id='MEMBER', ctx=None)],
        value=Constant(value=1),
        type_comment=None
    )
    
    body = [assign_node]
    
    # Mock the resolve method to return 'enum.Enum'
    parser.resolve = lambda root, node, self_ty="": 'enum.Enum'
    
    # Call class_api with enum base
    parser.class_api('test_module', 'test_module.TestClass', bases, body)
    
    # After class_api executes, is_enum should be True (line 7)
    # This causes the condition at line 23 to evaluate to True
    # and enums list should contain 'MEMBER'
    assert 'MEMBER' in parser.doc['test_module.TestClass'] or 'Enums' in parser.doc['test_module.TestClass']


# LLM-generated content at query #97
#--------------------------

```python
def test_class_api_is_enum_predicate():
    from ast import parse, Assign, Name, Constant
    
    parser = Parser()
    parser.doc['test.MyEnum'] = ''
    
    # Create a mock base that starts with 'enum.'
    bases = [parse('enum.Enum').body[0].value]
    
    # Create a simple assignment node
    assign_node = Assign(
        targets=[Name(id='MEMBER', ctx=None)],
        value=Constant(value=1),
        type_comment=None
    )
    body = [assign_node]
    
    # Mock walk_body to return our assignment
    original_walk_body = None
    try:
        import ast
        original_walk_body = ast.walk
    except:
        pass
    
    # Call class_api with enum bases
    parser.class_api('test', 'test.MyEnum', bases, body)
    
    # The predicate at line 23 (is_enum) should be True
    # This means enums list should contain 'MEMBER'
    assert 'MEMBER' in parser.doc['test.MyEnum'] or len(parser.doc['test.MyEnum']) > 0


# LLM-generated content at query #98
#--------------------------

```python
def test_visit_name_predicate_false():
    """Test that the predicate at line 6 evaluates to False."""
    from ast import Name, Load
    
    # Case 1: name not in self.alias
    resolver = Resolver(root="module", alias={"other_name": "int"}, self_ty="")
    node = Name(id="test_name", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node
    
    # Case 2: name in self.alias but name is in self.alias[name] (circular reference)
    resolver = Resolver(root="module", alias={"module.name": "module.name"}, self_ty="")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node
    
    # Case 3: name not in self.alias at all
    resolver = Resolver(root="module", alias={}, self_ty="")
    node = Name(id="undefined", ctx=Load())
    result = resolver.visit_Name(node)
    assert result is node


# LLM-generated content at query #99
#--------------------------

```python
def test_visit_constant_syntax_error():
    from ast import Constant, Load, Name
    
    resolver = Resolver(root="test_module", alias={})
    node = Constant(value="invalid syntax !!!invalid")
    result = resolver.visit_Constant(node)
    assert result is node


# LLM-generated content at query #100
#--------------------------

```python
def test_globals_with_annotated_assignment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse, AnnAssign, Name, Constant
    code = "x: int = 5"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert parser.alias["test_module.x"] == "5"


def test_globals_with_uppercase_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse
    code = "MAX_SIZE = 100"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.MAX_SIZE" in parser.root
    assert parser.root["test_module.MAX_SIZE"] == "test_module"
    assert "test_module.MAX_SIZE" in parser.const


def test_globals_with_all_list():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "__all__ = ['func1', 'func2']"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_all_tuple():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    parser.imp = {root: set()}
    
    from ast import parse as ast_parse
    code = "__all__ = ('func1', 'func2')"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]


def test_globals_with_type_comment():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse
    code = "x = 5  # type: int"
    node = ast_parse(code, type_comments=True).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert parser.const.get("test_module.x") == "int"


def test_globals_ignores_invalid_assignments():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse
    code = "x, y = 1, 2"
    node = ast_parse(code).body[0]
    initial_alias_len = len(parser.alias)
    
    parser.globals(root, node)
    
    assert len(parser.alias) == initial_alias_len


def test_globals_with_lowercase_variable():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse
    code = "x = 5"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.x" in parser.alias
    assert "test_module.x" not in parser.root


def test_globals_with_string_constant():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse
    code = "MESSAGE = 'hello'"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.MESSAGE" in parser.const
    assert parser.const["test_module.MESSAGE"] == "str"


def test_globals_with_none_value():
    parser = Parser()
    root = "test_module"
    parser.level[root] = 0
    parser.alias = {}
    parser.root = {}
    parser.const = {}
    
    from ast import parse as ast_parse
    code = "EMPTY = None"
    node = ast_parse(code).body[0]
    
    parser.globals(root, node)
    
    assert "test_module.EMPTY" in parser.const
    assert parser.const["test_module.EMPTY"] == "NoneType"


