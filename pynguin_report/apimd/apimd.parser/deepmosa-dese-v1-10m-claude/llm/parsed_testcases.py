####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 6/14 statements.
# Partially parsed test_imports_with_import_from_statement. Retrieved 8/16 statements.
# Partially parsed test_imports_with_relative_import. Retrieved 6/13 statements.
# Partially parsed test_imports_with_relative_import_level_2. Retrieved 6/13 statements.
# Partially parsed test_imports_with_import_from_no_module. Retrieved 5/12 statements.
# Partially parsed test_imports_with_multiple_names. Retrieved 9/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 'environ'
    var_6 = 'env'
    var_7 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.subpackage.module'
    var_2 = 'utils'
    var_3 = 'helper'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.subpackage.module'
    var_2 = 'common'
    var_3 = 'base'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = None
    var_3 = 'helper'
    var_4 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mypackage.mymodule'
    var_2 = 'utils'
    var_3 = 'func1'
    var_4 = None
    var_5 = 'func2'
    var_6 = 'f2'
    var_7 = 'func3'
    var_8 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 5/8 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 6/9 statements.
# Partially parsed test_visit_name_with_alias_non_recursive. Retrieved 7/10 statements.
# Partially parsed test_visit_name_with_typevar_alias. Retrieved 7/10 statements.
# Partially parsed test_visit_name_not_in_alias. Retrieved 5/8 statements.
# Partially parsed test_visit_name_circular_alias. Retrieved 6/9 statements.
# Partially parsed test_visit_name_with_nested_module. Retrieved 7/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'MyType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeName'
    var_5 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyName'
    var_2 = 'List[int]'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyName'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = "TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'T'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'UnknownName'
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.A'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    var_4 = 'A'
    var_5 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'package.module'
    var_1 = 'package.module.Name'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Name'
    var_6 = module_1.Load()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_globals_annotated_assign_with_value. Retrieved 9/17 statements.
# Partially parsed test_globals_assign_with_constant_value. Retrieved 6/15 statements.
# Partially parsed test_globals_uppercase_constant. Retrieved 6/15 statements.
# Partially parsed test_globals_all_filter. Retrieved 11/21 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_multiple_targets_ignored. Retrieved 7/18 statements.
# Partially parsed test_globals_non_name_target_ignored. Retrieved 12/26 statements.
# Partially parsed test_globals_annotated_assign_no_value_ignored. Retrieved 8/16 statements.
# Partially parsed test_globals_list_constant. Retrieved 13/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'y'
    var_3 = 'hello'
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MAX_VALUE'
    var_3 = 100
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = module_1.Constant()
    var_5 = 'func2'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.Tuple()
    var_10 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'z'
    var_3 = 3.14
    var_4 = module_1.Constant()
    var_5 = 'float'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 10
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = 2
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.Tuple()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'w'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'nums'
    var_3 = 1
    var_4 = module_1.Constant()
    var_5 = 2
    var_6 = module_1.Constant()
    var_7 = 3
    var_8 = module_1.Constant()
    var_9 = [var_4, var_6, var_8]
    var_10 = module_1.Load()
    var_11 = module_1.List()
    var_12 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_public_submodule. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_matching. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_not_matching. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_all_list_empty. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_module_in_imp. Retrieved 3/6 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_parent_in_all. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.submodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule._private'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.other'
    var_2 = 'mymodule.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.__init__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.submodule'
    var_2 = 'mymodule.submodule.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 11/19 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 16/24 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 14/28 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 11/19 statements.
# Partially parsed test_class_api_empty_body. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'default'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'MEMBER1'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = 'test_module'
    var_15 = 'test_module.TestEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'default'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'
    var_11 = var_0.doc[var_10]
    var_12 = 'Members'
    var_13 = 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'private'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(a: int, b: str) -> bool: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = var_4.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "def foo(a: int, b: str = 'hello') -> bool: pass"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = var_4.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class A:\n    def method(self, x: int) -> None: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.body[var_3]
    var_6 = 'test'
    var_7 = var_0.parse(var_6, var_1)
    var_8 = 'test.A.method'
    var_9 = var_5.args
    var_10 = var_5.returns
    var_11 = True
    var_12 = False
    var_13 = var_0.func_api(var_6, var_8, var_9, var_10, has_self=var_11, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.body[var_3]
    var_6 = 'test'
    var_7 = var_0.parse(var_6, var_1)
    var_8 = 'test.A.method'
    var_9 = var_5.args
    var_10 = var_5.returns
    var_11 = True
    var_12 = var_0.func_api(var_6, var_8, var_9, var_10, has_self=var_11, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(*args: int, **kwargs: str) -> None: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = var_4.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(a: int, *, b: str) -> None: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = var_4.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(a, b): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = None
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(a: int, /, b: str) -> None: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = var_4.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    var_7 = 'test.foo'
    var_8 = var_4.args
    var_9 = var_4.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_5, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_class_api_assign_predicate. Retrieved 14/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_attr'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = module_1.Assign()
    var_8 = var_7.targets
    var_9 = len(var_8)
    var_10 = 1
    var_11 = var_9 == var_10
    var_12 = 0
    var_13 = var_7.targets[var_12]



# Parsed testcases at query #8
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.path.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.__init__.path'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public._private.module'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__dict__.method'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a.b.c.d.e'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.__name__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def test_func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'async def async_func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'class TestClass: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = '@staticmethod\ndef decorated_func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'class Outer:\n    def inner_method(self): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func():\n    """Test docstring."""\n    pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'def func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_0]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'class Outer:\n    class Inner: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    var_8 = var_2.api(var_6, var_5)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_class_api_delete_statement_handling. Retrieved 8/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'attr_to_delete'
    var_4 = None
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = module_1.Delete()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_visit_attribute_typing_prefix. Retrieved 11/13 statements.
# Partially parsed test_visit_attribute_non_typing_prefix. Retrieved 10/11 statements.
# Partially parsed test_visit_attribute_typing_with_different_attributes. Retrieved 10/11 statements.
# Partially parsed test_visit_attribute_preserves_context. Retrieved 7/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'Union'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other_module'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'SomeClass'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'obj'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'typing'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = 'Union'
    var_10 = module_1.Load()
    var_11 = module_1.Attribute()
    var_12 = var_2.visit_Attribute(var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'Optional'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_walk_body_simple_statements. Retrieved 6/8 statements.
# Partially parsed test_walk_body_with_if_statement. Retrieved 6/8 statements.
# Partially parsed test_walk_body_with_nested_if. Retrieved 8/9 statements.
# Partially parsed test_walk_body_with_try_except. Retrieved 6/8 statements.
# Partially parsed test_walk_body_with_try_except_else_finally. Retrieved 6/8 statements.
# Partially parsed test_walk_body_complex_nested_structure. Retrieved 6/8 statements.
# Partially parsed test_walk_body_multiple_handlers. Retrieved 6/8 statements.
# Partially parsed test_walk_body_if_without_else. Retrieved 8/9 statements.
# Partially parsed test_walk_body_deeply_nested. Retrieved 8/9 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1\ny = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    if False:\n        x = 1'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 4

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 0

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1\nif True:\n    y = 2\n    try:\n        z = 3\n    except:\n        w = 4\nelse:\n    a = 5'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 6

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept TypeError:\n    z = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    x = 1'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    if True:\n        if True:\n            x = 1'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_class_api_mem_predicate_true. Retrieved 12/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.resolve
    var_6 = 'int'
    var_7 = 'test_module'
    var_8 = 'test_class'
    var_9 = []
    var_10 = var_4.body
    var_11 = var_0.class_api(var_7, var_8, var_9, var_10)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_imports_with_import_from_node. Retrieved 8/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias
    var_7 = len(var_6)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 21/34 statements.
# Partially parsed test_class_api_with_enum. Retrieved 25/38 statements.
# Partially parsed test_class_api_with_bases. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 15/29 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 15/25 statements.
# Partially parsed test_class_api_empty_class. Retrieved 7/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = 'attr1'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = None
    var_10 = 'attr2'
    var_11 = 42
    var_12 = module_1.Constant()
    var_13 = []
    var_14 = 'test_module'
    var_15 = 'test_module.TestClass'
    var_16 = []
    var_17 = var_0.doc
    var_18 = len(var_17)
    var_19 = 0
    var_20 = var_18 > var_19

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestEnum'
    var_2 = 'Enum'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = [var_4]
    var_6 = []
    var_7 = 'MEMBER1'
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = 1
    var_12 = module_1.Constant()
    var_13 = 'MEMBER2'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = 2
    var_17 = module_1.Constant()
    var_18 = []
    var_19 = 'test_module'
    var_20 = 'test_module.TestEnum'
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = [var_22]
    var_24 = var_0.doc

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)
    var_9 = var_0.doc

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = 'attr1'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = None
    var_10 = []
    var_11 = 'test_module'
    var_12 = 'test_module.TestClass'
    var_13 = []
    var_14 = var_0.doc

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = '_private'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = None
    var_10 = []
    var_11 = 'test_module'
    var_12 = 'test_module.TestClass'
    var_13 = []
    var_14 = var_0.doc

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = var_0.doc



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_matching. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_not_matching. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_empty_all_list. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_nested_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_module_in_imp_dict. Retrieved 3/6 statements.
# Partially parsed test_is_public_with_all_list_parent_match. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg._private'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.__init__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.other'
    var_2 = 'pkg.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.public_func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub.func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = 'pkg.sub.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_enum. Retrieved 7/16 statements.
# Partially parsed test_class_api_with_deleted_attributes. Retrieved 5/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_annotated_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_bases. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_assigned_members. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass TestClass(BaseClass):\n    '''Test class'''\n    public_attr: int\n    _private_attr: str\n    CONSTANT: float = 3.14\n    "
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.Enum'
    var_2 = 'enum.Enum'
    var_3 = '\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    '
    var_4 = 0
    var_5 = 'test_module'
    var_6 = 'test_module.Color'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int = 1\n    attr2: str = "test"\n    del attr1\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class EmptyClass: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    name: str\n    age: int\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class DerivedClass(BaseClass, Mixin): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.DerivedClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    value1 = 42\n    value2 = "string"\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'



# Parsed testcases at query #18
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'float'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = 3
    var_5 = module_0.Constant()
    var_6 = [var_1, var_3, var_5]
    var_7 = module_0.List()
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'list[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'str'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Tuple()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'tuple[str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.List()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Set()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'set[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = 'a'
    var_6 = module_0.Constant()
    var_7 = 'b'
    var_8 = module_0.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_0.Dict()
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'dict[int, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0.Dict()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'dict'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'x'
    var_3 = module_0.Name()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown_var'
    var_1 = module_0.Name()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'Any'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_func_ann_with_self_parameter. Retrieved 14/15 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 13/14 statements.
# Partially parsed test_func_ann_without_self. Retrieved 13/14 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 13/14 statements.
# Partially parsed test_func_ann_with_annotations. Retrieved 20/22 statements.
# Partially parsed test_func_ann_self_with_annotation. Retrieved 16/18 statements.
# Partially parsed test_func_ann_empty_args. Retrieved 6/7 statements.
# Partially parsed test_func_ann_multiple_stars. Retrieved 15/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'x'
    var_5 = module_1.arg()
    var_6 = 'return'
    var_7 = module_1.arg()
    var_8 = [var_3, var_5, var_7]
    var_9 = 'root'
    var_10 = True
    var_11 = False
    var_12 = var_0.func_ann(var_9, var_8, has_self=var_10, cls_method=var_11)
    var_13 = list(var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'x'
    var_5 = module_1.arg()
    var_6 = 'return'
    var_7 = module_1.arg()
    var_8 = [var_3, var_5, var_7]
    var_9 = 'root'
    var_10 = True
    var_11 = var_0.func_ann(var_9, var_8, has_self=var_10, cls_method=var_10)
    var_12 = list(var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'y'
    var_5 = module_1.arg()
    var_6 = 'return'
    var_7 = module_1.arg()
    var_8 = [var_3, var_5, var_7]
    var_9 = 'root'
    var_10 = False
    var_11 = var_0.func_ann(var_9, var_8, has_self=var_10, cls_method=var_10)
    var_12 = list(var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = '*'
    var_5 = module_1.arg()
    var_6 = 'return'
    var_7 = module_1.arg()
    var_8 = [var_3, var_5, var_7]
    var_9 = 'root'
    var_10 = False
    var_11 = var_0.func_ann(var_9, var_8, has_self=var_10, cls_method=var_10)
    var_12 = list(var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.x'
    var_2 = 'root.y'
    var_3 = 'int'
    var_4 = 'str'
    var_5 = None
    var_6 = module_1.Name()
    var_7 = module_1.Name()
    var_8 = 'x'
    var_9 = module_1.arg()
    var_10 = 'y'
    var_11 = module_1.arg()
    var_12 = 'return'
    var_13 = module_1.arg()
    var_14 = [var_9, var_11, var_13]
    var_15 = 'root'
    var_16 = False
    var_17 = var_0.func_ann(var_15, var_14, has_self=var_16, cls_method=var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 3

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MyClass'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = 'self'
    var_5 = module_1.arg()
    var_6 = 'x'
    var_7 = module_1.arg()
    var_8 = 'return'
    var_9 = module_1.arg()
    var_10 = [var_5, var_7, var_9]
    var_11 = 'root'
    var_12 = True
    var_13 = False
    var_14 = var_0.func_ann(var_11, var_10, has_self=var_12, cls_method=var_13)
    var_15 = list(var_14)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'root'
    var_3 = False
    var_4 = var_0.func_ann(var_2, var_1, has_self=var_3, cls_method=var_3)
    var_5 = list(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = '*'
    var_5 = module_1.arg()
    var_6 = 'y'
    var_7 = module_1.arg()
    var_8 = 'return'
    var_9 = module_1.arg()
    var_10 = [var_3, var_5, var_7, var_9]
    var_11 = 'root'
    var_12 = False
    var_13 = var_0.func_ann(var_11, var_10, has_self=var_12, cls_method=var_12)
    var_14 = list(var_13)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_class_api_is_enum_predicate. Retrieved 14/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = 'test_module.TestEnum'
    var_5 = 'enum.Enum'
    var_6 = 'eval'
    var_7 = module_1.parse(var_5, mode=var_6)
    var_8 = var_7.body
    var_9 = [var_8]
    var_10 = 'MEMBER: int = 1'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body
    var_13 = var_2.class_api(var_3, var_4, var_9, var_12)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 10/19 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 7/17 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 7/17 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 12/22 statements.
# Partially parsed test_globals_ignores_non_matching_nodes. Retrieved 13/28 statements.
# Partially parsed test_globals_constant_uppercase. Retrieved 7/17 statements.
# Partially parsed test_globals_with_annotated_no_value. Retrieved 9/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MY_CONSTANT'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 42
    var_8 = module_1.Constant()
    var_9 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MY_VAR'
    var_4 = 'hello'
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'typed_var'
    var_4 = 100
    var_5 = module_1.Constant()
    var_6 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'func1'
    var_5 = module_1.Constant()
    var_6 = 'func2'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.Tuple()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = 2
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Load()
    var_11 = module_1.Tuple()
    var_12 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'CONSTANT'
    var_4 = 999
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = None
    var_8 = 1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 19/22 statements.
# Partially parsed test_class_api_with_bases. Retrieved 8/11 statements.
# Partially parsed test_class_api_with_enums. Retrieved 24/27 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 18/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 16/19 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 13/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = module_1.AnnAssign()
    var_9 = 'attr2'
    var_10 = module_1.Name()
    var_11 = [var_10]
    var_12 = 42
    var_13 = module_1.Constant()
    var_14 = module_1.Assign()
    var_15 = [var_8, var_14]
    var_16 = 'test_module'
    var_17 = 'test_module.TestClass'
    var_18 = var_0.class_api(var_16, var_17, var_1, var_15)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = []
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_0.class_api(var_5, var_6, var_3, var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Name()
    var_3 = 'Enum'
    var_4 = module_1.Attribute()
    var_5 = [var_4]
    var_6 = 'MEMBER1'
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Name()
    var_10 = 1
    var_11 = module_1.Constant()
    var_12 = module_1.AnnAssign()
    var_13 = 'MEMBER2'
    var_14 = module_1.Name()
    var_15 = [var_14]
    var_16 = 2
    var_17 = module_1.Constant()
    var_18 = None
    var_19 = module_1.Assign()
    var_20 = [var_12, var_19]
    var_21 = 'test_module'
    var_22 = 'test_module.TestEnum'
    var_23 = var_0.class_api(var_21, var_22, var_5, var_20)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = module_1.AnnAssign()
    var_9 = 'public'
    var_10 = module_1.Name()
    var_11 = 'int'
    var_12 = module_1.Name()
    var_13 = module_1.AnnAssign()
    var_14 = [var_8, var_13]
    var_15 = 'test_module'
    var_16 = 'test_module.TestClass'
    var_17 = var_0.class_api(var_15, var_16, var_1, var_14)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = module_1.AnnAssign()
    var_9 = module_1.Name()
    var_10 = [var_9]
    var_11 = module_1.Delete()
    var_12 = [var_8, var_11]
    var_13 = 'test_module'
    var_14 = 'test_module.TestClass'
    var_15 = var_0.class_api(var_13, var_14, var_1, var_12)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 10
    var_6 = module_1.Constant()
    var_7 = 'int'
    var_8 = module_1.Assign()
    var_9 = [var_8]
    var_10 = 'test_module'
    var_11 = 'test_module.TestClass'
    var_12 = var_0.class_api(var_10, var_11, var_1, var_9)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_class_api_is_enum_predicate. Retrieved 17/28 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'from enum import Enum\nclass Base(Enum): pass'
    var_4 = 0
    var_5 = 1
    var_6 = module_1.parse(var_3)
    var_7 = var_6.body[var_5]
    var_8 = var_7.bases[var_4]
    var_9 = 'x: int = 1'
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body[var_4]
    var_12 = var_0.resolve
    var_13 = 'enum.Enum'
    var_14 = [var_8]
    var_15 = [var_11]
    var_16 = var_0.class_api(var_1, var_2, var_14, var_15)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_globals_annassign_with_value. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int = 42'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.target
    var_6 = 'test_module'
    var_7 = var_0.globals(var_6, var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_globals_predicate_line_31_evaluates_to_true. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONSTANT: int = 42'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.CONSTANT'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_func_ann_with_self_parameter. Retrieved 8/16 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 7/15 statements.
# Partially parsed test_func_ann_without_self. Retrieved 7/15 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 8/17 statements.
# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 9/18 statements.
# Partially parsed test_func_ann_with_classmethod_and_annotation. Retrieved 7/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = None
    var_4 = 'x'
    var_5 = 'return'
    var_6 = True
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'cls'
    var_3 = None
    var_4 = 'x'
    var_5 = 'return'
    var_6 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = 'return'
    var_6 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = '*'
    var_5 = 'y'
    var_6 = 'return'
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MyClass'
    var_3 = None
    var_4 = 'self'
    var_5 = 'x'
    var_6 = 'return'
    var_7 = True
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'type[MyClass]'
    var_3 = None
    var_4 = 'cls'
    var_5 = 'return'
    var_6 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_globals_predicate_line_35_evaluates_to_false. Retrieved 14/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 35 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '__all__ = ("item1", "item2")'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_6.targets
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_6.targets[var_5]
    var_10 = var_6.value
    var_11 = var_1.globals(var_2, var_6)
    var_12 = var_1.imp[var_2]
    var_13 = len(var_12)
    assert var_13 == 2



# Parsed testcases at query #28
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = None
    var_9 = module_1.Assign()
    var_10 = 'test_module'
    var_11 = var_0.globals(var_10, var_9)
    var_12 = var_9.targets
    var_13 = len(var_12)
    var_14 = var_9.targets
    var_15 = len(var_14)
    assert var_15 == 2



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 9/17 statements.
# Partially parsed test_globals_with_assign_statement. Retrieved 6/15 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 11/21 statements.
# Partially parsed test_globals_with_all_list. Retrieved 11/21 statements.
# Partially parsed test_globals_ignores_lowercase_variable. Retrieved 6/15 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_with_multiple_targets_ignored. Retrieved 7/18 statements.
# Partially parsed test_globals_with_annotated_assignment_without_value. Retrieved 8/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONSTANT'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST_VAR'
    var_3 = 100
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = module_1.Constant()
    var_5 = 'func2'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.Tuple()
    var_10 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'item1'
    var_4 = module_1.Constant()
    var_5 = 'item2'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.List()
    var_10 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'lowercase_var'
    var_3 = 'test'
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TYPED_VAR'
    var_3 = 3.14
    var_4 = module_1.Constant()
    var_5 = 'float'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_visit_name_self_ty_match. Retrieved 9/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "Test that visit_Name returns Name('Self', Load()) when node.id equals self_ty."
    var_1 = 'module'
    var_2 = {}
    var_3 = 'MyType'
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = var_4.visit_Name(var_6)
    var_8 = var_7.ctx



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 10/19 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 7/17 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 7/17 statements.
# Partially parsed test_globals_with_all_assignment. Retrieved 12/22 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 15/30 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 7/17 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 10/22 statements.
# Partially parsed test_globals_with_annassign_without_value. Retrieved 11/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MY_CONST'
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MY_VAR'
    var_4 = 'hello'
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'CONSTANT'
    var_4 = 100
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'func1'
    var_5 = module_1.Constant()
    var_6 = 'func2'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.Tuple()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = 2
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Load()
    var_11 = module_1.Tuple()
    var_12 = None
    var_13 = var_0.alias
    var_14 = len(var_13)
    assert var_14 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'my_var'
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = None
    var_8 = var_0.alias
    var_9 = len(var_8)
    assert var_9 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'my_var'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = None
    var_8 = 1
    var_9 = var_0.alias
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_none_in_chain. Retrieved 2/7 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_method_call. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'value'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'inner.data'

def test_case_0():
    var_0 = 'deep'
    var_1 = 'level1.level2.level3.final'

def test_case_0():
    var_0 = 42
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = None
    var_1 = 'inner.data.something'

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'any.attr'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'get_value'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'keys'
    var_4 = module_0._attr(var_2, var_3)
    var_5 = callable(var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_e_type_single_element_with_single_constant. Retrieved 3/5 statements.
# Partially parsed test_e_type_single_element_with_multiple_same_type_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_single_element_with_multiple_different_type_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_multiple_elements_same_types. Retrieved 10/12 statements.
# Partially parsed test_e_type_multiple_elements_mixed_types. Retrieved 8/10 statements.
# Partially parsed test_e_type_none_in_elements. Retrieved 1/3 statements.
# Partially parsed test_e_type_empty_sequence_in_elements. Retrieved 1/3 statements.
# Partially parsed test_e_type_non_constant_in_sequence. Retrieved 2/7 statements.
# Partially parsed test_e_type_string_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_boolean_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_none_constant. Retrieved 3/5 statements.
# Partially parsed test_e_type_multiple_elements_with_any. Retrieved 10/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

import ast as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = [var_1]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'string'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = 2.71
    var_7 = module_0.Constant()
    var_8 = [var_1, var_3]
    var_9 = [var_5, var_7]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'string'
    var_3 = module_0.Constant()
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = [var_1, var_3]
    var_7 = [var_5]

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = []

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()

import ast as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Constant()
    var_2 = 'world'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Constant()
    var_2 = False
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Constant()
    var_2 = [var_1]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = 'string'
    var_5 = module_0.Constant()
    var_6 = 4.0
    var_7 = module_0.Constant()
    var_8 = [var_1, var_3]
    var_9 = [var_5, var_7]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_compile_basic. Retrieved 5/12 statements.
# Partially parsed test_compile_with_toc. Retrieved 4/16 statements.
# Partially parsed test_compile_with_constants. Retrieved 5/13 statements.
# Partially parsed test_compile_filters_private_names. Retrieved 5/16 statements.
# Partially parsed test_compile_empty_parser. Retrieved 5/12 statements.
# Partially parsed test_compile_magic_methods_without_docstring. Retrieved 5/15 statements.
# Partially parsed test_compile_multiple_levels. Retrieved 4/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method with basic parser setup.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method with table of contents enabled.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1, var_1)
    var_3 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method with constants.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method filters out private names.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method with empty parser.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()
    assert var_4 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method skips magic methods without docstring.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile method with nested names.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1, var_1)
    var_3 = var_2.compile()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_class_api_delete_statement_handling. Retrieved 9/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'attr_to_delete'
    var_4 = None
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = module_1.Delete()
    var_8 = var_7



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_globals_with_annassign_and_value. Retrieved 5/14 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_assign_without_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_invalid_node_structure. Retrieved 9/18 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 7/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MY_CONST: int = 42'
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'value = 100  # type: float'
    var_4 = 0
    var_5 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "CONSTANT = 'hello'"
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['func1', 'func2']"
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'number = 42'
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a, b = 1, 2'
    var_4 = 0
    var_5 = var_0.alias
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_0.const
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x = y = 5'
    var_4 = 0
    var_5 = var_0.alias
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #37
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = [var_3]
    var_7 = 'str'
    var_8 = module_1.Assign()
    var_9 = 'module'
    var_10 = var_0.globals(var_9, var_8)
    assert var_10 == 'str'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_visit_constant_with_string_valid_name. Retrieved 7/8 statements.
# Partially parsed test_visit_constant_with_string_self_type. Retrieved 6/7 statements.
# Partially parsed test_visit_constant_with_string_complex_expression. Retrieved 11/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.int'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = module_1.Constant()
    var_6 = var_4.visit_Constant(var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not valid python @@@@'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Constant()
    var_5 = var_3.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.List'
    var_2 = 'mymodule.int'
    var_3 = 'list'
    var_4 = 'int'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = 'List[int]'
    var_8 = module_1.Constant()
    var_9 = var_6.visit_Constant(var_8)
    var_10 = var_9.value



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_compile_predicate_line_13_true. Retrieved 8/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 (name in self.docstring) evaluates to True.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = 'test_module'
    var_5 = '# Module `test_module`\n\n'
    var_6 = 'This is a test docstring.'
    var_7 = var_3.compile()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_true. Retrieved 11/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '_m'
    var_1 = None
    var_2 = 'mymodule'
    var_3 = 'mymodule.MyType'
    var_4 = 'int'
    var_5 = {var_3: var_4}
    var_6 = module_0.Resolver(var_2, var_5)
    var_7 = 'MyType'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = var_6.visit_Name(var_9)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/18 statements.
# Partially parsed test_class_api_with_enums. Retrieved 16/24 statements.
# Partially parsed test_class_api_with_delete. Retrieved 10/22 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_private_members_ignored. Retrieved 10/18 statements.
# Partially parsed test_class_api_with_multiple_bases. Retrieved 12/15 statements.
# Partially parsed test_class_api_with_assign_member. Retrieved 8/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'MyClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'MEMBER'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = 'test_module'
    var_15 = 'MyEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'MyClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Base2'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = [var_3, var_6]
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'MyClass'
    var_11 = var_0.class_api(var_9, var_10, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'value'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'MyClass'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 15/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '__all__'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 123
    var_6 = module_1.Constant()
    var_7 = [var_6]
    var_8 = None
    var_9 = module_1.Tuple()
    var_10 = module_1.Assign()
    var_11 = 'test_root'
    var_12 = var_1.globals(var_11, var_10)
    var_13 = var_1.imp[var_11]
    var_14 = len(var_13)
    assert var_14 == 0



# Parsed testcases at query #43
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'root'
    var_5 = [var_3]
    var_6 = False
    var_7 = var_0.func_ann(var_4, var_5, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True



# Parsed testcases at query #45
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'int'
    var_3 = module_1.Name()
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = 'test_module'
    var_7 = True
    var_8 = False
    var_9 = var_0.func_ann(var_6, var_5, has_self=var_7, cls_method=var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_with_toc_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/13 statements.
# Partially parsed test_class_api_with_members. Retrieved 11/20 statements.
# Partially parsed test_class_api_with_enums. Retrieved 14/23 statements.
# Partially parsed test_class_api_with_delete. Retrieved 11/24 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/10 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 11/20 statements.
# Partially parsed test_class_api_with_multiple_bases. Retrieved 12/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = 'str'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'test'
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum.Enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'MEMBER1'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'value1'
    var_9 = module_1.Constant()
    var_10 = 1
    var_11 = [var_3]
    var_12 = 'test_module'
    var_13 = 'test_module.TestEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = 'str'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'test'
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'str'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'test'
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Base2'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = [var_3, var_6]
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'
    var_11 = var_0.class_api(var_9, var_10, var_7, var_8)



# Parsed testcases at query #48
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not a valid python expression !!!'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 17/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '__all__'
    var_3 = None
    var_4 = module_1.Name()
    var_5 = 123
    var_6 = module_1.Constant()
    var_7 = 'value'
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Tuple()
    var_11 = [var_4]
    var_12 = module_1.Assign()
    var_13 = 'test_module'
    var_14 = var_1.globals(var_13, var_12)
    var_15 = var_1.imp[var_13]
    var_16 = len(var_15)
    assert var_16 == 1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_parse_basic_module. Retrieved 5/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function docstring'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n'
    var_2 = 'mymodule'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.alias
    var_5 = len(var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n"
    var_2 = 'test_pkg'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ndef my_function(x: int) -> str:\n    '''Function docstring'''\n    return str(x)\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nDEBUG = True\nMAX_SIZE: int = 100\n'
    var_2 = 'config'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = "'''Module doc'''"
    var_3 = 'mymod'
    var_4 = var_1.parse(var_3, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = "'''Module doc'''"
    var_3 = 'mymod'
    var_4 = var_1.parse(var_3, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass Outer:\n    '''Outer class'''\n    class Inner:\n        '''Inner class'''\n        pass\n"
    var_2 = 'pkg'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nasync def async_func():\n    '''Async function'''\n    pass\n"
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass MyClass:\n    @staticmethod\n    def static_method():\n        '''Static method'''\n        pass\n    \n    @classmethod\n    def class_method(cls):\n        '''Class method'''\n        pass\n"
    var_2 = 'mod'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n'''Module with docstring.\n\n>>> x = 1\n>>> print(x)\n1\n'''\ndef func():\n    '''Function doc.\n    \n    >>> func()\n    '''\n    pass\n"
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n"
    var_2 = 'pkg'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n"
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.imp[var_2]
    var_5 = len(var_4)
    var_6 = 0
    var_7 = var_5 > var_6

import apimd.parser as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Parser(level=var_0)
    var_2 = "'''Doc'''"
    var_3 = 'package.submodule'
    var_4 = var_1.parse(var_3, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n'
    var_2 = 'mod'
    var_3 = var_0.parse(var_2, var_1)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_func_api_kwonlyargs_predicate. Retrieved 14/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'kw_only_param'
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = [var_3]
    var_8 = []
    var_9 = module_1.arguments(*var_2)
    var_10 = 'test_module'
    var_11 = 'test_func'
    var_12 = False
    var_13 = var_0.func_api(var_10, var_11, var_9, var_3, has_self=var_12, cls_method=var_12)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 6/11 statements.
# Partially parsed test_imports_with_import_as_statement. Retrieved 6/11 statements.
# Partially parsed test_imports_with_multiple_imports. Retrieved 7/13 statements.
# Partially parsed test_imports_from_statement_absolute. Retrieved 8/13 statements.
# Partially parsed test_imports_from_statement_absolute_with_asname. Retrieved 8/13 statements.
# Partially parsed test_imports_from_statement_relative_level_1. Retrieved 8/13 statements.
# Partially parsed test_imports_from_statement_relative_level_2. Retrieved 8/13 statements.
# Partially parsed test_imports_from_statement_relative_no_module. Retrieved 7/12 statements.
# Partially parsed test_imports_from_statement_multiple_names. Retrieved 9/15 statements.
# Partially parsed test_imports_from_statement_with_star. Retrieved 8/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'mymodule'
    var_4 = ''
    var_5 = var_0.parse(var_3, var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'mymodule'
    var_4 = ''
    var_5 = var_0.parse(var_3, var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'sys'
    var_4 = 'mymodule'
    var_5 = ''
    var_6 = var_0.parse(var_4, var_5)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = ''
    var_7 = var_0.parse(var_5, var_6)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'ospath'
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = ''
    var_7 = var_0.parse(var_5, var_6)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mymodule'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'sibling'
    var_5 = 'func'
    var_6 = None
    var_7 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub.mymodule'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'other'
    var_5 = 'item'
    var_6 = None
    var_7 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mymodule'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = None
    var_5 = 'helper'
    var_6 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 'getcwd'
    var_5 = 0
    var_6 = 'mymodule'
    var_7 = ''
    var_8 = var_0.parse(var_6, var_7)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = '*'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = ''
    var_7 = var_0.parse(var_5, var_6)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_load_docstring. Retrieved 4/15 statements.
# Partially parsed test_load_docstring_no_docstring. Retrieved 4/14 statements.
# Partially parsed test_load_docstring_nested_attribute. Retrieved 4/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_globals_annassign_with_value. Retrieved 7/9 statements.
# Partially parsed test_globals_annassign_uppercase_constant. Retrieved 7/9 statements.
# Partially parsed test_globals_assign_simple. Retrieved 7/9 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 8/10 statements.
# Partially parsed test_globals_all_tuple. Retrieved 7/10 statements.
# Partially parsed test_globals_all_list. Retrieved 7/10 statements.
# Partially parsed test_globals_multiple_targets_ignored. Retrieved 7/9 statements.
# Partially parsed test_globals_non_name_target_ignored. Retrieved 7/9 statements.
# Partially parsed test_globals_uppercase_multiple_times. Retrieved 11/13 statements.
# Partially parsed test_globals_annassign_without_value_ignored. Retrieved 7/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "CONST: str = 'hello'"
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 42'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 10  # type: int'
    var_3 = True
    var_4 = module_1.parse(var_2, type_comments=var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ('func1', 'func2')"
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ['func1', 'func2', 'func3']"
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = y = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '(x, y) = (1, 2)'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'CONST = 10'
    var_8 = module_1.parse(var_7)
    var_9 = var_8.body[var_4]
    var_10 = var_0.globals(var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 8/10 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 8/9 statements.
# Partially parsed test_visit_name_with_alias_not_recursive. Retrieved 10/11 statements.
# Partially parsed test_visit_name_with_typevar_alias. Retrieved 11/12 statements.
# Partially parsed test_visit_name_with_circular_alias. Retrieved 9/10 statements.
# Partially parsed test_visit_name_no_root_with_alias. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)
    var_7 = var_6.ctx

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeClass'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = var_3.visit_Name(var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = var_5.visit_Name(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.T'
    var_2 = 'typing.TypeVar'
    var_3 = "TypeVar('T')"
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = ''
    var_6 = module_0.Resolver(var_0, var_4, var_5)
    var_7 = 'T'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = var_6.visit_Name(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.A'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'A'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = var_4.visit_Name(var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'MyType'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3, var_0)
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = var_4.visit_Name(var_6)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_imports_asname_not_none. Retrieved 6/30 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'collections'
    var_2 = 'OrderedDict'
    var_3 = 'OD'
    var_4 = 0
    var_5 = 'mymodule'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/12 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 17/25 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 14/22 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 14/22 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 7/20 statements.
# Partially parsed test_class_api_with_assign_members. Retrieved 8/17 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_multiple_bases. Retrieved 12/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'MEMBER1'
    var_9 = 'str'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 'value1'
    var_13 = module_1.Constant()
    var_14 = 1
    var_15 = 'test_module'
    var_16 = 'test_module.TestEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'object'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 'public_attr'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 42
    var_10 = module_1.Constant()
    var_11 = 1
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'object'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = '_private_attr'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 42
    var_10 = module_1.Constant()
    var_11 = 1
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'temp_attr'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'class_var'
    var_3 = 'test'
    var_4 = module_1.Constant()
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Base2'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = [var_3, var_6]
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.MultiBase'
    var_11 = var_0.class_api(var_9, var_10, var_7, var_8)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_visit_attribute_typing_prefix. Retrieved 11/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/12 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 16/24 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 11/19 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 11/19 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 7/20 statements.
# Partially parsed test_class_api_no_bases_no_members. Retrieved 6/9 statements.
# Partially parsed test_class_api_with_assign_members. Retrieved 8/17 statements.
# Partially parsed test_class_api_with_multiple_bases. Retrieved 12/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'MEMBER1'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = 'test_module'
    var_15 = 'test_module.MyEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'public_attr'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'test'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private_attr'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'test'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member'
    var_3 = 1
    var_4 = module_1.Constant()
    var_5 = 'test_module'
    var_6 = 'test_module.MyClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.MyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'count'
    var_3 = 0
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Base2'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = [var_3, var_6]
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'
    var_11 = var_0.class_api(var_9, var_10, var_7, var_8)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_imports_with_relative_import_level. Retrieved 6/28 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'submodule'
    var_2 = 'func'
    var_3 = None
    var_4 = 1
    var_5 = 'package.module'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 4/27 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'mymodule'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_compile_magic_method_continues. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.compile()



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_class_api_assign_predicate. Retrieved 12/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = module_1.Assign()
    var_8 = var_7.targets
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_7.targets[var_10]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_compile_magic_method_without_docstring. Retrieved 6/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = set()
    var_3 = 'module.__init__'
    var_4 = 0
    var_5 = var_0.compile()



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 8/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 14/17 statements.
# Partially parsed test_class_api_with_enums. Retrieved 18/21 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 14/17 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 17/20 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = []
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_0.class_api(var_5, var_6, var_3, var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = 'test'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_1.AnnAssign()
    var_10 = [var_9]
    var_11 = 'test_module'
    var_12 = 'test_module.TestClass'
    var_13 = var_0.class_api(var_11, var_12, var_1, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Name()
    var_3 = 'Enum'
    var_4 = module_1.Attribute()
    var_5 = [var_4]
    var_6 = 'MEMBER1'
    var_7 = module_1.Name()
    var_8 = 'str'
    var_9 = module_1.Name()
    var_10 = 'value1'
    var_11 = module_1.Constant()
    var_12 = 1
    var_13 = module_1.AnnAssign()
    var_14 = [var_13]
    var_15 = 'test_module'
    var_16 = 'test_module.TestEnum'
    var_17 = var_0.class_api(var_15, var_16, var_5, var_14)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = 'test'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_1.AnnAssign()
    var_10 = [var_9]
    var_11 = 'test_module'
    var_12 = 'test_module.TestClass'
    var_13 = var_0.class_api(var_11, var_12, var_1, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = 'test'
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_1.AnnAssign()
    var_10 = module_1.Name()
    var_11 = [var_10]
    var_12 = module_1.Delete()
    var_13 = [var_9, var_12]
    var_14 = 'test_module'
    var_15 = 'test_module.TestClass'
    var_16 = var_0.class_api(var_14, var_15, var_1, var_13)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_globals_annassign_with_value. Retrieved 10/18 statements.
# Partially parsed test_globals_annassign_constant_uppercase. Retrieved 10/16 statements.
# Partially parsed test_globals_assign_single_target. Retrieved 7/16 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 7/14 statements.
# Partially parsed test_globals_assign_uppercase_constant. Retrieved 7/14 statements.
# Partially parsed test_globals_all_list. Retrieved 12/21 statements.
# Partially parsed test_globals_all_tuple. Retrieved 12/21 statements.
# Partially parsed test_globals_ignore_multiple_targets. Retrieved 8/18 statements.
# Partially parsed test_globals_ignore_non_name_target. Retrieved 13/26 statements.
# Partially parsed test_globals_annassign_without_value. Retrieved 9/15 statements.
# Partially parsed test_globals_annassign_non_name_target. Retrieved 12/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with AnnAssign node having value.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with uppercase constant in AnnAssign.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'MAX'
    var_4 = 100
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with Assign node having single target.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'y'
    var_4 = 'hello'
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with Assign node having type_comment.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'z'
    var_4 = 3.14
    var_5 = module_1.Constant()
    var_6 = 'float'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with uppercase constant in Assign.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'PI'
    var_4 = 3.14159
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with __all__ as List.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '__all__'
    var_4 = 'func1'
    var_5 = module_1.Constant()
    var_6 = 'func2'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.List()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with __all__ as Tuple.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '__all__'
    var_4 = 'api1'
    var_5 = module_1.Constant()
    var_6 = 'api2'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.Tuple()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method ignores Assign with multiple targets.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method ignores non-Name target.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = 2
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Load()
    var_11 = module_1.Tuple()
    var_12 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method with AnnAssign node without value.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = None
    var_8 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test globals method ignores AnnAssign with non-Name target.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'obj'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'attr'
    var_7 = 5
    var_8 = module_1.Constant()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 5/10 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 6/8 statements.
# Partially parsed test_visit_name_with_alias_simple. Retrieved 8/11 statements.
# Partially parsed test_visit_name_with_alias_typevar. Retrieved 10/12 statements.
# Partially parsed test_visit_name_with_alias_circular. Retrieved 7/9 statements.
# Partially parsed test_visit_name_not_in_alias. Retrieved 8/10 statements.
# Partially parsed test_visit_name_with_nested_alias. Retrieved 10/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeClass'
    var_5 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = 'module.TypeVar'
    var_3 = "typing.TypeVar('T')"
    var_4 = 'typing.TypeVar'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = ''
    var_7 = module_0.Resolver(var_0, var_5, var_6)
    var_8 = 'T'
    var_9 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.A'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'A'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.Other'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'NotDefined'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.Alias1'
    var_2 = 'module.Alias2'
    var_3 = 'Alias2'
    var_4 = 'int'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = ''
    var_7 = module_0.Resolver(var_0, var_5, var_6)
    var_8 = 'Alias1'
    var_9 = module_1.Load()



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_true. Retrieved 5/35 statements.


import ast as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Constant()
    var_2 = 'hello'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #71
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = [var_2, var_4]
    var_8 = None
    var_9 = module_1.Assign()
    var_10 = 'test_module'
    var_11 = var_0.globals(var_10, var_9)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_imports_with_simple_import. Retrieved 6/11 statements.
# Partially parsed test_imports_with_aliased_import. Retrieved 6/11 statements.
# Partially parsed test_imports_with_multiple_imports. Retrieved 7/13 statements.
# Partially parsed test_imports_from_absolute_import. Retrieved 8/13 statements.
# Partially parsed test_imports_from_absolute_import_with_alias. Retrieved 8/13 statements.
# Partially parsed test_imports_from_relative_import_level_1. Retrieved 8/13 statements.
# Partially parsed test_imports_from_relative_import_level_2. Retrieved 8/13 statements.
# Partially parsed test_imports_from_relative_import_no_module. Retrieved 7/12 statements.
# Partially parsed test_imports_from_import_with_multiple_names. Retrieved 9/15 statements.
# Partially parsed test_imports_from_import_nested_module. Retrieved 8/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'mymodule'
    var_4 = ''
    var_5 = var_0.parse(var_3, var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'mymodule'
    var_4 = ''
    var_5 = var_0.parse(var_3, var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'sys'
    var_4 = 'mymodule'
    var_5 = ''
    var_6 = var_0.parse(var_4, var_5)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = ''
    var_7 = var_0.parse(var_5, var_6)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'filepath'
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = ''
    var_7 = var_0.parse(var_5, var_6)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'utils'
    var_5 = 'helper'
    var_6 = None
    var_7 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.sub.module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'utils'
    var_5 = 'helper'
    var_6 = None
    var_7 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = None
    var_5 = 'helper'
    var_6 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 'environ'
    var_5 = 0
    var_6 = 'mymodule'
    var_7 = ''
    var_8 = var_0.parse(var_6, var_7)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os.path'
    var_2 = 'join'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = ''
    var_7 = var_0.parse(var_5, var_6)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_true. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = 'int'
    var_3 = 'str'
    var_4 = var_2 != var_3
    var_5 = var_2 and var_4
    assert var_5 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_class_api_enums_predicate_true. Retrieved 17/24 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'MEMBER'
    var_4 = None
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = module_1.AnnAssign()
    var_11 = 'enum.Enum'
    var_12 = module_1.Constant()
    var_13 = var_0.resolve
    var_14 = [var_12]
    var_15 = [var_10]
    var_16 = var_0.class_api(var_1, var_2, var_14, var_15)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_with_all_true. Retrieved 2/3 statements.
# Partially parsed test_parser_independent_instances. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = True
    var_1 = 3

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = module_0.Parser()



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_visit_attribute_typing_prefix_removal. Retrieved 11/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 5/10 statements.
# Partially parsed test_visit_name_no_alias. Retrieved 5/8 statements.
# Partially parsed test_visit_name_with_alias. Retrieved 7/10 statements.
# Partially parsed test_visit_name_with_typevar_alias. Retrieved 9/12 statements.
# Partially parsed test_visit_name_self_reference_in_alias. Retrieved 6/9 statements.
# Partially parsed test_visit_name_with_nested_alias. Retrieved 7/10 statements.
# Partially parsed test_visit_name_empty_root. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'MyType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'SomeName'
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyType'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = 'module.TypeVar'
    var_3 = "TypeVar('T')"
    var_4 = 'typing.TypeVar'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = 'T'
    var_8 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.Node'
    var_2 = 'Node'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.Alias'
    var_2 = 'typing.Dict'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Alias'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'MyType'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = module_1.Load()



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/10 statements.
# Partially parsed test_imports_simple_import_with_alias. Retrieved 4/10 statements.
# Partially parsed test_imports_multiple_imports. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_absolute. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_absolute_with_alias. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 8/15 statements.
# Partially parsed test_imports_from_import_no_module. Retrieved 5/11 statements.
# Partially parsed test_imports_star_import. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.sub.module'
    var_2 = 'other'
    var_3 = 'item'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = None
    var_5 = 'Counter'
    var_6 = 'cnt'
    var_7 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = None
    var_3 = 'func'
    var_4 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'utils'
    var_3 = '*'
    var_4 = None
    var_5 = 0



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_load_docstring_with_valid_docstring. Retrieved 4/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.func'
    var_2 = 'function doc'
    var_3 = 'test_module'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 4/10 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 4/10 statements.
# Partially parsed test_globals_with_all_assignment. Retrieved 4/11 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 4/10 statements.
# Partially parsed test_globals_with_lowercase_variable. Retrieved 4/10 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 6/12 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_with_annotated_assignment_no_value. Retrieved 6/12 statements.
# Partially parsed test_globals_with_tuple_unpacking. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 5'
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST = 42'
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ['func1', 'func2']"
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "CONSTANT_VALUE = 'test'"
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'variable = 100'
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a = b = 5'
    var_3 = 0
    var_4 = var_0.alias
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 5  # type: int'
    var_3 = True
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int'
    var_3 = 0
    var_4 = var_0.alias
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a, b = 1, 2'
    var_3 = 0
    var_4 = var_0.alias
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 7/12 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 7/12 statements.
# Partially parsed test_globals_with_all_filter. Retrieved 8/14 statements.
# Partially parsed test_globals_with_assignment_no_type_comment. Retrieved 7/12 statements.
# Partially parsed test_globals_ignores_invalid_targets. Retrieved 7/12 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 7/12 statements.
# Partially parsed test_globals_skips_annotated_without_value. Retrieved 7/12 statements.
# Partially parsed test_globals_with_multiple_targets_ignored. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'x: int = 42'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'CONST = 100'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = "__all__ = ['func1', 'func2']"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_0.globals(var_1, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = "x = 'hello'"
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'a, b = 1, 2'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'x = 5  # type: int'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'x: int'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'x = y = 10'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_0.globals(var_1, var_5)



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_true. Retrieved 5/30 statements.


import ast as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = 'hello'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_const_type_call_with_name_or_attribute. Retrieved 9/15 statements.


import ast as module_0

def test_case_0():
    var_0 = 'int(5)'
    var_1 = 'eval'
    var_2 = module_0.parse(var_0, mode=var_1)
    var_3 = var_2.body
    var_4 = var_3.func
    var_5 = 'obj.method()'
    var_6 = module_0.parse(var_5, mode=var_1)
    var_7 = var_6.body
    var_8 = var_7.func



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 4/10 statements.
# Partially parsed test_imports_with_import_as_statement. Retrieved 4/10 statements.
# Partially parsed test_imports_with_multiple_imports. Retrieved 5/12 statements.
# Partially parsed test_imports_from_statement_absolute. Retrieved 6/12 statements.
# Partially parsed test_imports_from_statement_with_asname. Retrieved 6/12 statements.
# Partially parsed test_imports_from_statement_relative_level_1. Retrieved 6/12 statements.
# Partially parsed test_imports_from_statement_relative_level_2. Retrieved 6/12 statements.
# Partially parsed test_imports_from_statement_multiple_names. Retrieved 7/14 statements.
# Partially parsed test_imports_from_statement_none_module. Retrieved 5/11 statements.
# Partially parsed test_imports_from_statement_relative_level_3_nested. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'numpy'
    var_3 = 'np'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.sub.module'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = None
    var_5 = 'Counter'
    var_6 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = None
    var_3 = 'func'
    var_4 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a.b.c.d'
    var_2 = 'utils'
    var_3 = 'helper'
    var_4 = None
    var_5 = 3



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_class_api_private_attribute_not_added_to_mem. Retrieved 7/30 statements.


import ast as module_0

def test_case_0():
    var_0 = "_private_attr: str = 'value'"
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = var_3.target.id
    assert var_4 == '_private_attr'
    var_5 = {}
    var_6 = False



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_func_ann_line_7_predicate_true. Retrieved 15/45 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 0
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = [var_8]
    var_10 = True
    var_11 = False
    var_12 = var_0.func_ann(var_1, var_9, has_self=var_10, cls_method=var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_true. Retrieved 14/23 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to True.'
    var_1 = 'mymodule'
    var_2 = 'mymodule.MyType'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = module_0.Resolver(var_1, var_4, var_5)
    var_7 = 'MyType'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 42
    var_11 = module_1.Constant()
    var_12 = module_1.Expr()
    var_13 = var_6.visit_Name(var_9)



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/13 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 16/25 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/19 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/23 statements.
# Partially parsed test_class_api_with_assign_members. Retrieved 8/18 statements.
# Partially parsed test_class_api_empty_body. Retrieved 6/10 statements.
# Partially parsed test_class_api_private_members_ignored. Retrieved 10/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'MEMBER1'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = 'test_module'
    var_15 = 'test_module.MyEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.MyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0, var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 9/20 statements.
# Partially parsed test_globals_with_assignment. Retrieved 6/18 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_with_dunder_all. Retrieved 11/20 statements.
# Partially parsed test_globals_ignores_non_uppercase_constants. Retrieved 6/14 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 7/16 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 9/20 statements.
# Partially parsed test_globals_with_annotated_assignment_no_value. Retrieved 8/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST_VALUE'
    var_3 = 'hello'
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TYPED_VAR'
    var_3 = 123
    var_4 = module_1.Constant()
    var_5 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = module_1.Constant()
    var_5 = 'func2'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.Tuple()
    var_10 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_var'
    var_3 = 99
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR1'
    var_3 = 'VAR2'
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = None
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'ANNOTATED'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 14/25 statements.
# Partially parsed test_class_api_with_enums. Retrieved 22/33 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/22 statements.
# Partially parsed test_class_api_with_assign_members. Retrieved 8/17 statements.
# Partially parsed test_class_api_no_bases_no_members. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = '_private'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'ENUM1'
    var_9 = 'str'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 'value1'
    var_13 = module_1.Constant()
    var_14 = 1
    var_15 = 'ENUM2'
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = 'value2'
    var_19 = module_1.Constant()
    var_20 = 'test_module'
    var_21 = 'test_module.TestEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_m_single_name. Retrieved 1/3 statements.
# Partially parsed test_m_multiple_names. Retrieved 3/5 statements.
# Partially parsed test_m_empty_string. Retrieved 1/3 statements.
# Partially parsed test_m_with_empty_strings. Retrieved 3/5 statements.
# Partially parsed test_m_all_empty_strings. Retrieved 1/3 statements.
# Partially parsed test_m_single_empty_string. Retrieved 1/3 statements.
# Partially parsed test_m_multiple_with_leading_trailing_empty. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'foo'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = 'baz'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'foo'
    var_1 = ''
    var_2 = 'bar'

def test_case_0():
    var_0 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._m()
    assert var_0 == ''

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = ''
    var_1 = 'foo'
    var_2 = 'bar'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_link_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = True
    var_1 = 3
    var_2 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_walk_body_simple_statements. Retrieved 6/18 statements.
# Partially parsed test_walk_body_with_if_statement. Retrieved 10/31 statements.
# Partially parsed test_walk_body_with_nested_if. Retrieved 15/44 statements.
# Partially parsed test_walk_body_with_try_statement. Retrieved 17/50 statements.
# Partially parsed test_walk_body_with_multiple_handlers. Retrieved 16/48 statements.


import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'z'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = 'cond'

import ast as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'b'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'c'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = 'd'
    var_10 = 4
    var_11 = module_0.Constant()
    var_12 = 'inner'
    var_13 = 'outer'
    var_14 = []

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'z'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = 'w'
    var_10 = 4
    var_11 = module_0.Constant()
    var_12 = 'v'
    var_13 = 5
    var_14 = module_0.Constant()
    var_15 = 'Exception'
    var_16 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'z'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = 'w'
    var_10 = 4
    var_11 = module_0.Constant()
    var_12 = 'ValueError'
    var_13 = None
    var_14 = 'TypeError'
    var_15 = []



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nfrom sys import path'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass:\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""Module docstring."""\nx = 1'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    """Function docstring."""\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "CONSTANT = 42\nANOTHER = 'value'"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int = 5'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'x = 1'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = 'x = 1'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1'
    var_2 = 'pkg.subpkg.module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Outer:\n    class Inner:\n        pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    """Example.\n    >>> foo()\n    """\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@staticmethod\ndef foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1'
    var_2 = 'my_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@decorator1\n@decorator2\ndef foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1  # type: int'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_class_api_enum_predicate_true. Retrieved 11/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nfrom enum import Enum\n\nclass TestEnum(Enum):\n    MEMBER1: int = 1\n    MEMBER2: str = "test"\n'
    var_2 = True
    var_3 = module_1.parse(var_1, type_comments=var_2)
    var_4 = var_3.body[var_2]
    var_5 = var_0.resolve
    var_6 = 'test_module'
    var_7 = 'test_module.TestEnum'
    var_8 = var_4.bases
    var_9 = var_4.body
    var_10 = var_0.class_api(var_6, var_7, var_8, var_9)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_compile_with_single_module. Retrieved 3/8 statements.
# Partially parsed test_compile_with_toc_enabled. Retrieved 4/9 statements.
# Partially parsed test_compile_with_constants. Retrieved 4/11 statements.
# Partially parsed test_compile_with_magic_method_no_doc. Retrieved 3/11 statements.
# Partially parsed test_compile_with_nested_names. Retrieved 3/16 statements.
# Partially parsed test_compile_with_link_format. Retrieved 4/9 statements.
# Partially parsed test_compile_sorts_by_level_and_name. Retrieved 5/17 statements.
# Partially parsed test_compile_missing_docstring_warning. Retrieved 4/9 statements.
# Partially parsed test_compile_with_multiple_levels. Retrieved 4/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with empty parser.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()
    assert var_2 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with a single module.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with table of contents enabled.'
    var_1 = True
    var_2 = module_0.Parser(toc=var_1)
    var_3 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with constants.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile skips magic methods without documentation.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with nested class/function names.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with link formatting.'
    var_1 = True
    var_2 = module_0.Parser(var_1)
    var_3 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile sorts entries by level and name.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()
    var_3 = 'a_mod'
    var_4 = 'b_mod'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with missing docstring for non-magic name.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()
    var_3 = '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with different nesting levels.'
    var_1 = True
    var_2 = module_0.Parser(toc=var_1)
    var_3 = var_2.compile()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #10
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = 'y'
    var_5 = module_1.Name()
    var_6 = 5
    var_7 = module_1.Constant()
    var_8 = [var_3, var_5]
    var_9 = module_1.Assign()
    var_10 = 'test_module'
    var_11 = var_0.globals(var_10, var_9)
    var_12 = var_0.alias
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = var_0.root
    var_15 = len(var_14)
    assert var_15 == 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_class_api_annassign_with_name_target. Retrieved 16/23 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = 'int'
    var_5 = module_1.Name()
    var_6 = 5
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_1.AnnAssign()
    var_10 = 'test_module'
    var_11 = 'test_module.TestClass'
    var_12 = []
    var_13 = [var_9]
    var_14 = var_0.class_api(var_10, var_11, var_12, var_13)
    var_15 = var_9.target



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_public_name_no_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_exact_match. Retrieved 5/8 statements.
# Partially parsed test_is_public_with_all_list_no_match. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_all_list_parent_match. Retrieved 5/8 statements.
# Partially parsed test_is_public_module_itself. Retrieved 4/6 statements.
# Partially parsed test_is_public_in_imp_with_public_children. Retrieved 7/11 statements.
# Partially parsed test_is_public_in_imp_without_public_children. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = 'module'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'module'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.public_func'
    var_3 = set()
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.public_func'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.private_func'
    var_3 = 'module.other_func'
    var_4 = {var_3}
    var_5 = var_0.is_public(var_2)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.submodule'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.submod'
    var_3 = 'module.submod.func'
    var_4 = set()
    var_5 = 'doc'
    var_6 = var_0.is_public(var_2)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.submod'
    var_3 = set()
    var_4 = 'doc'
    var_5 = var_0.is_public(var_2)
    assert var_5 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_with_missing_middle. Retrieved 2/9 statements.
# Partially parsed test_attr_nested_with_none_in_chain. Retrieved 3/9 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_numeric_values. Retrieved 2/5 statements.
# Partially parsed test_attr_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_attr_multiple_levels_all_exist. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'middle.inner.value'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'middle.nonexistent.value'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = None
    var_2 = 'middle.inner.value'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 42
    var_1 = 'num'

def test_case_0():
    var_0 = None
    var_1 = 'attr'

def test_case_0():
    var_0 = 'final'
    var_1 = 'level2.level3.data'



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'def foo(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'async def bar(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'async def bar(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class Baz: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'class Baz: pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '@staticmethod\ndef func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '@staticmethod\ndef func(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def func():\n    """Test doc"""\n    pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'def func():\n    """Test doc"""\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class Outer:\n    def inner(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'class Outer:\n    def inner(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'def func(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'def func(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class Outer:\n    def inner(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'class Outer:\n    def inner(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = ''
    var_9 = var_0.api(var_1, var_7, prefix=var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def func_with_underscores(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'def func_with_underscores(): pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.api(var_1, var_7)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_const_type_with_unsupported_node. Retrieved 4/8 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'float'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = 3
    var_5 = module_0.Constant()
    var_6 = [var_1, var_3, var_5]
    var_7 = module_0.List()
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'list[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list[str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.List()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Tuple()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'tuple[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Set()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'set[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = 1
    var_6 = module_0.Constant()
    var_7 = 2
    var_8 = module_0.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_0.Dict()
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'dict[str, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'a'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = module_0.List()
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = module_0.Call(*var_2)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'ANY'

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()



# Parsed testcases at query #16
#--------------------------




import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_0.Try()
    var_5 = var_4.handlers
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/10 statements.
# Partially parsed test_imports_simple_import_with_asname. Retrieved 4/10 statements.
# Partially parsed test_imports_multiple_imports. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_with_asname. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 8/15 statements.
# Partially parsed test_imports_from_import_with_none_module. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mymodule'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.subpkg.mymodule'
    var_2 = 'other'
    var_3 = 'Class'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = None
    var_5 = 'Counter'
    var_6 = 'MyCounter'
    var_7 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mymodule'
    var_2 = None
    var_3 = 'helper'
    var_4 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_func_ann_with_self_and_annotations. Retrieved 26/28 statements.
# Partially parsed test_func_ann_without_self_annotations. Retrieved 20/22 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 19/21 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 11/13 statements.
# Partially parsed test_func_ann_with_varargs_marker. Retrieved 23/25 statements.
# Partially parsed test_func_ann_mixed_annotations. Retrieved 29/31 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 0
    var_4 = 'MyClass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = 'x'
    var_10 = 'int'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_3]
    var_13 = var_12.value
    var_14 = module_1.arg()
    var_15 = 'y'
    var_16 = 'str'
    var_17 = module_1.parse(var_16)
    var_18 = var_17.body[var_3]
    var_19 = var_18.value
    var_20 = module_1.arg()
    var_21 = [var_8, var_14, var_20]
    var_22 = True
    var_23 = False
    var_24 = var_0.func_ann(var_1, var_21, has_self=var_22, cls_method=var_23)
    var_25 = list(var_24)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 0
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = 'y'
    var_10 = 'str'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_3]
    var_13 = var_12.value
    var_14 = module_1.arg()
    var_15 = [var_8, var_14]
    var_16 = False
    var_17 = False
    var_18 = var_0.func_ann(var_1, var_15, has_self=var_16, cls_method=var_17)
    var_19 = list(var_18)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'cls'
    var_3 = 0
    var_4 = 'type[MyClass]'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = 'x'
    var_10 = 'int'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_3]
    var_13 = var_12.value
    var_14 = module_1.arg()
    var_15 = [var_8, var_14]
    var_16 = True
    var_17 = var_0.func_ann(var_1, var_15, has_self=var_16, cls_method=var_16)
    var_18 = list(var_17)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = 'y'
    var_6 = module_1.arg()
    var_7 = [var_4, var_6]
    var_8 = False
    var_9 = var_0.func_ann(var_1, var_7, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 0
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = '*'
    var_10 = None
    var_11 = module_1.arg()
    var_12 = 'y'
    var_13 = 'str'
    var_14 = module_1.parse(var_13)
    var_15 = var_14.body[var_3]
    var_16 = var_15.value
    var_17 = module_1.arg()
    var_18 = [var_8, var_11, var_17]
    var_19 = False
    var_20 = False
    var_21 = var_0.func_ann(var_1, var_18, has_self=var_19, cls_method=var_20)
    var_22 = list(var_21)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 0
    var_4 = 'MyClass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = 'x'
    var_10 = 'int'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_3]
    var_13 = var_12.value
    var_14 = module_1.arg()
    var_15 = 'y'
    var_16 = None
    var_17 = module_1.arg()
    var_18 = 'z'
    var_19 = 'list[str]'
    var_20 = module_1.parse(var_19)
    var_21 = var_20.body[var_3]
    var_22 = var_21.value
    var_23 = module_1.arg()
    var_24 = [var_8, var_14, var_17, var_23]
    var_25 = True
    var_26 = False
    var_27 = var_0.func_ann(var_1, var_24, has_self=var_25, cls_method=var_26)
    var_28 = list(var_27)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_public_predicate_line_5_evaluates_to_false. Retrieved 21/39 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'module1'
    var_4 = set()
    var_5 = 'module1.private_attr'
    var_6 = 'module1.another'
    var_7 = 'doc1'
    var_8 = 'doc2'
    var_9 = 'module1.const1'
    var_10 = 'int'
    var_11 = 'sys'
    var_12 = __import__(var_11)
    var_13 = '__main__'
    var_14 = 'is_public_family'
    var_15 = 'module1'
    var_16 = set()
    var_17 = 'some doc'
    var_18 = __import__(var_11)
    var_19 = None
    var_20 = var_2.is_public(var_3)
    assert var_20 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_func_api_with_simple_arguments. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_return_type. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 12/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(a, b): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(a, b=1): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(self, a): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = True
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(a) -> int: pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = var_4.returns
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(a, *args): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(a, **kwargs): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(cls, a): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = True
    var_10 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'def func(a, *, b): pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = var_4.args
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_visit_constant_with_string_value_valid_name. Retrieved 8/9 statements.
# Partially parsed test_visit_constant_with_string_value_self_type. Retrieved 6/7 statements.
# Partially parsed test_visit_constant_with_string_subscript_expression. Retrieved 9/11 statements.
# Partially parsed test_visit_constant_with_string_value_complex_expression. Retrieved 6/7 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyType'
    var_6 = module_1.Constant()
    var_7 = var_4.visit_Constant(var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not a valid python expression !!!'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Constant()
    var_5 = var_3.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.Optional'
    var_2 = 'typing.Optional'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'list[int]'
    var_6 = module_1.Constant()
    var_7 = var_4.visit_Constant(var_6)
    var_8 = var_7.value

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union[int, str]'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_func_api_predicate_false. Retrieved 12/24 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = None
    var_3 = module_0.arg()
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_0.arguments(*var_4)
    var_9 = [var_2, var_2, var_2]
    var_10 = 5
    var_11 = [var_2, var_10, var_2]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_ann_annotation_not_none. Retrieved 10/21 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_2]
    var_7 = var_6.value
    var_8 = False
    var_9 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/17 statements.
# Partially parsed test_class_api_with_enums. Retrieved 16/23 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 7/19 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/8 statements.
# Partially parsed test_class_api_private_members_excluded. Retrieved 14/24 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 8/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'RED'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = 'test_module'
    var_15 = 'test_module.Color'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'test'
    var_4 = module_1.Constant()
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.Empty'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'str'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'public'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_public_with_public_module. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_containing_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_not_containing_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_when_name_is_in_imp_and_has_public_children. Retrieved 6/10 statements.
# Partially parsed test_is_public_when_name_is_in_imp_and_no_public_children. Retrieved 4/8 statements.
# Partially parsed test_is_public_with_all_list_containing_parent. Retrieved 5/7 statements.
# Partially parsed test_is_public_lowercase_name. Retrieved 4/6 statements.
# Partially parsed test_is_public_uppercase_constant. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg._private'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.__init__'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.func'
    var_3 = {var_1, var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.func'
    var_3 = {var_1}
    var_4 = var_0.is_public(var_2)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'pkg.sub.func'
    var_3 = set()
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = {var_1}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.func'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.CONST'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_globals_const_predicate_false. Retrieved 13/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONST'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = [var_3]
    var_7 = module_1.Assign()
    var_8 = 'test_module'
    var_9 = '.'
    var_10 = var_8 + var_9
    var_11 = var_10 + var_1
    var_12 = var_0.globals(var_8, var_7)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_public_name_no_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_public_name_in_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_module_itself_in_all. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_submodule_in_all. Retrieved 6/8 statements.
# Partially parsed test_is_public_not_in_all_with_all_list. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_module_in_imp_dict. Retrieved 5/8 statements.
# Partially parsed test_is_public_with_underscore_prefix_in_all. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = 'module'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'module'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.public'
    var_2 = 'module'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.func'
    var_2 = 'module'
    var_3 = {var_1}
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.sub.func'
    var_2 = 'module'
    var_3 = 'module.sub'
    var_4 = {var_3}
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.func'
    var_2 = 'module'
    var_3 = 'module.other'
    var_4 = {var_3}
    var_5 = var_0.is_public(var_1)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = set()
    var_3 = 'doc'
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module._private'
    var_2 = 'module'
    var_3 = {var_1}
    var_4 = var_0.is_public(var_1)
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = [var_2, var_4]
    var_8 = None
    var_9 = module_1.Assign()
    var_10 = 'test_module'
    var_11 = var_0.globals(var_10, var_9)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_globals_predicate_line_35_evaluates_to_false. Retrieved 14/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "Test that the predicate at line 35 evaluates to False.\n    \n    The predicate is: left.id != '__all__' or not isinstance(node.value, (Tuple, List))\n    For it to evaluate to False, both conditions must be False:\n    - left.id == '__all__' (first condition is False)\n    - isinstance(node.value, (Tuple, List)) (second condition is False, so 'not' makes it False)\n    "
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = "__all__ = ('item1', 'item2')"
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_6.targets
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_6.targets[var_5]
    var_10 = var_6.value
    var_11 = var_1.globals(var_2, var_6)
    var_12 = var_1.imp[var_2]
    var_13 = len(var_12)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 15/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 123
    var_4 = module_1.Constant()
    var_5 = [var_4]
    var_6 = None
    var_7 = module_1.Tuple()
    var_8 = '__all__'
    var_9 = module_1.Name()
    var_10 = [var_9]
    var_11 = module_1.Assign()
    var_12 = var_1.globals(var_2, var_11)
    var_13 = var_1.imp[var_2]
    var_14 = len(var_13)
    assert var_14 == 0



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a regular line\nAnother regular line'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a regular line\nAnother regular line'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n>>> y = 2\n>>> print(x + y)'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n>>> y = 2\n>>> print(x + y)\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Some text\n>>> x = 1\n>>> y = 2\nMore text'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'Some text\n```python\n>>> x = 1\n>>> y = 2\n```\nMore text'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\ntext\n>>> y = 2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```\ntext\n```python\n>>> y = 2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\nregular line'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```\nregular line'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'regular line\n>>> x = 1'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'regular line\n```python\n>>> x = 1\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n>>> y = 2\ntext\n>>> z = 3\n>>> w = 4'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n>>> y = 2\n```\ntext\n```python\n>>> z = 3\n>>> w = 4\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "```python\n>>> print('hello')\n```"

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n\n>>> y = 2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```\n\n```python\n>>> y = 2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n>>> print(x)\n1\nmore text'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n>>> print(x)\n```\n1\nmore text'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_const_type_predicate_line_11. Retrieved 17/35 statements.


def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = 'set'
    var_3 = {var_0: var_0, var_1: var_1, var_2: var_2}
    var_4 = 'int'
    var_5 = 'bool'
    var_6 = 'int'
    var_7 = 'float'
    var_8 = 'complex'
    var_9 = 'str'
    var_10 = {var_5, var_6, var_7, var_8, var_9}
    var_11 = 'str'
    var_12 = {var_5, var_6, var_7, var_8, var_9}
    var_13 = 'list'
    var_14 = {var_5, var_6, var_7, var_8, var_9}
    var_15 = 'dict'
    var_16 = {var_5, var_6, var_7, var_8, var_9}



# Parsed testcases at query #33
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 0
    var_4 = 'str'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = module_1.arg()
    var_9 = [var_8]
    var_10 = True
    var_11 = False
    var_12 = var_0.func_ann(var_1, var_9, has_self=var_10, cls_method=var_11)
    var_13 = list(var_12)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_docstring. Retrieved 2/14 statements.
# Partially parsed test_load_docstring_no_docstring. Retrieved 2/12 statements.
# Partially parsed test_load_docstring_filtered_by_root. Retrieved 2/13 statements.
# Partially parsed test_load_docstring_nested_attributes. Retrieved 2/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_e_type_single_element_with_single_constant. Retrieved 3/5 statements.
# Partially parsed test_e_type_single_element_with_multiple_same_type_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_single_element_with_multiple_different_type_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_multiple_elements_same_types. Retrieved 10/12 statements.
# Partially parsed test_e_type_multiple_elements_mixed_types. Retrieved 8/10 statements.
# Partially parsed test_e_type_with_none_in_sequence. Retrieved 2/4 statements.
# Partially parsed test_e_type_with_non_constant_in_sequence. Retrieved 5/7 statements.
# Partially parsed test_e_type_with_empty_sequence. Retrieved 1/2 statements.
# Partially parsed test_e_type_with_string_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_with_bool_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_multiple_elements_with_mixed_consistency. Retrieved 8/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

import ast as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = [var_1]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'string'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = 2.71
    var_7 = module_0.Constant()
    var_8 = [var_1, var_3]
    var_9 = [var_5, var_7]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'string'
    var_3 = module_0.Constant()
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = [var_1, var_3]
    var_7 = [var_5]

def test_case_0():
    var_0 = None
    var_1 = [var_0]

import ast as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = 'x'
    var_3 = module_0.Name()
    var_4 = [var_1, var_3]

def test_case_0():
    var_0 = []

import ast as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Constant()
    var_2 = 'world'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Constant()
    var_2 = False
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = 'string'
    var_5 = module_0.Constant()
    var_6 = [var_1, var_3]
    var_7 = [var_5, var_5]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_class_api_enum_predicate_is_true. Retrieved 14/21 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 0
    var_4 = 'enum.Enum'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = [var_7]
    var_9 = 'x: int'
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body
    var_12 = var_0.resolve
    var_13 = var_0.class_api(var_1, var_2, var_8, var_11)



# Parsed testcases at query #37
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 23 evaluates to False when node.type_comment is not None.'
    var_1 = module_0.Parser()
    var_2 = 'x'
    var_3 = None
    var_4 = module_1.Name()
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = [var_4]
    var_8 = 'int'
    var_9 = module_1.Assign()
    var_10 = 'test_module'
    var_11 = var_1.globals(var_10, var_9)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 17/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '__all__'
    var_4 = None
    var_5 = module_1.Name()
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = 2
    var_9 = module_1.Constant()
    var_10 = [var_7, var_9]
    var_11 = module_1.Tuple()
    var_12 = [var_5]
    var_13 = module_1.Assign()
    var_14 = var_1.globals(var_2, var_13)
    var_15 = var_1.imp[var_2]
    var_16 = len(var_15)
    assert var_16 == 0



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 14/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 42
    var_2 = module_1.Constant()
    var_3 = [var_2]
    var_4 = None
    var_5 = module_1.List()
    var_6 = '__all__'
    var_7 = module_1.Name()
    var_8 = [var_7]
    var_9 = module_1.Assign()
    var_10 = 'test_root'
    var_11 = var_0.globals(var_10, var_9)
    var_12 = var_0.imp[var_10]
    var_13 = len(var_12)
    assert var_13 == 0



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/17 statements.
# Partially parsed test_class_api_with_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_enums. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_type_comments. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Base1:\n    pass\n\nclass Base2:\n    pass\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass MyClass:\n    attr1: int\n    attr2: str = "default"\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.MyClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nfrom enum import Enum\n\nclass Status(Enum):\n    ACTIVE = 1\n    INACTIVE = 2\n'
    var_2 = 1
    var_3 = 'test_module'
    var_4 = 'test_module.Status'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int\n    del attr1\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Empty:\n    pass\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Empty'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr = 42  # type: int\n'
    var_2 = True
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'



# Parsed testcases at query #41
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = '42'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = "'hello'"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = None
    var_6 = [var_4, var_5, var_4]
    var_7 = module_1._defaults(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = "'a&b'"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = "'a|b'"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_e_type_single_element_with_constants. Retrieved 5/7 statements.
# Partially parsed test_e_type_single_element_with_mixed_types. Retrieved 5/7 statements.
# Partially parsed test_e_type_multiple_elements_same_type. Retrieved 6/8 statements.
# Partially parsed test_e_type_multiple_elements_different_types. Retrieved 6/8 statements.
# Partially parsed test_e_type_element_with_none. Retrieved 4/6 statements.
# Partially parsed test_e_type_empty_sequence. Retrieved 1/3 statements.
# Partially parsed test_e_type_with_non_constant. Retrieved 5/7 statements.
# Partially parsed test_e_type_single_element_strings. Retrieved 5/7 statements.
# Partially parsed test_e_type_single_element_floats. Retrieved 5/7 statements.
# Partially parsed test_e_type_multiple_elements_mixed. Retrieved 8/10 statements.
# Partially parsed test_e_type_single_element_booleans. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'str'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = [var_1]
    var_3 = 2
    var_4 = module_0.Constant()
    var_5 = [var_4]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = [var_1]
    var_3 = 'str'
    var_4 = module_0.Constant()
    var_5 = [var_4]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = [var_1]
    var_3 = None

def test_case_0():
    var_0 = []

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'x'
    var_3 = module_0.Name()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1.5
    var_1 = module_0.Constant()
    var_2 = 2.5
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = 'a'
    var_6 = module_0.Constant()
    var_7 = [var_6]

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Constant()
    var_2 = False
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 8/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 13/17 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 13/17 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 17/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 16/20 statements.
# Partially parsed test_class_api_with_assign_members. Retrieved 13/17 statements.
# Partially parsed test_class_api_no_bases_no_members. Retrieved 6/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = []
    var_5 = 'test_module'
    var_6 = 'MyClass'
    var_7 = var_0.class_api(var_5, var_6, var_3, var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'public_attr'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = module_1.AnnAssign()
    var_9 = [var_8]
    var_10 = 'test_module'
    var_11 = 'MyClass'
    var_12 = var_0.class_api(var_10, var_11, var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private_attr'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = module_1.AnnAssign()
    var_9 = [var_8]
    var_10 = 'test_module'
    var_11 = 'MyClass'
    var_12 = var_0.class_api(var_10, var_11, var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Name()
    var_3 = 'Enum'
    var_4 = module_1.Attribute()
    var_5 = [var_4]
    var_6 = 'MEMBER1'
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Name()
    var_10 = 1
    var_11 = module_1.Constant()
    var_12 = module_1.AnnAssign()
    var_13 = [var_12]
    var_14 = 'test_module'
    var_15 = 'MyEnum'
    var_16 = var_0.class_api(var_14, var_15, var_5, var_13)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr'
    var_3 = module_1.Name()
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = module_1.AnnAssign()
    var_9 = module_1.Name()
    var_10 = [var_9]
    var_11 = module_1.Delete()
    var_12 = [var_8, var_11]
    var_13 = 'test_module'
    var_14 = 'MyClass'
    var_15 = var_0.class_api(var_13, var_14, var_1, var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'class_var'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = None
    var_8 = module_1.Assign()
    var_9 = [var_8]
    var_10 = 'test_module'
    var_11 = 'MyClass'
    var_12 = var_0.class_api(var_10, var_11, var_1, var_9)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'SimpleClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_walk_body_with_if_statement. Retrieved 1/24 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_globals_predicate_line_18_false. Retrieved 16/33 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = var_8.targets
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = var_8.targets
    var_12 = len(var_11)
    var_13 = var_12 == var_6
    var_14 = 0
    var_15 = var_8.targets[var_14]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #47
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 0
    var_3 = 'int'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_5.value
    var_7 = module_1.arg()
    var_8 = 'test_module'
    var_9 = [var_7]
    var_10 = True
    var_11 = False
    var_12 = var_0.func_ann(var_8, var_9, has_self=var_10, cls_method=var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_visit_attribute_typing_prefix. Retrieved 11/13 statements.
# Partially parsed test_visit_attribute_non_typing_prefix. Retrieved 10/11 statements.
# Partially parsed test_visit_attribute_non_name_value. Retrieved 13/14 statements.
# Partially parsed test_visit_attribute_typing_union. Retrieved 10/11 statements.
# Partially parsed test_visit_attribute_typing_optional. Retrieved 10/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'Method'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'obj'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'inner'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = 'Method'
    var_10 = module_1.Load()
    var_11 = module_1.Attribute()
    var_12 = var_2.visit_Attribute(var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'Union'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'Optional'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_walk_body_if_predicate. Retrieved 5/6 statements.


import ast as module_0

def test_case_0():
    var_0 = 'if True:\n    pass\nelse:\n    pass'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = 0
    var_4 = var_2[var_3]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_class_api_delete_statement_predicate. Retrieved 8/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'del x, y'
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_4.targets
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_4.targets



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_class_api_delete_statement_predicate. Retrieved 8/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = None
    var_3 = module_1.Name()
    var_4 = 'attr2'
    var_5 = module_1.Name()
    var_6 = [var_3, var_5]
    var_7 = module_1.Delete()



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_compile_magic_method_continues. Retrieved 4/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_visit_name_self_ty. Retrieved 5/10 statements.
# Partially parsed test_visit_name_not_in_alias. Retrieved 6/9 statements.
# Partially parsed test_visit_name_in_alias_simple. Retrieved 8/11 statements.
# Partially parsed test_visit_name_in_alias_circular_reference. Retrieved 7/10 statements.
# Partially parsed test_visit_name_typevar_in_alias. Retrieved 10/13 statements.
# Partially parsed test_visit_name_complex_expression_in_alias. Retrieved 8/11 statements.
# Partially parsed test_visit_name_with_empty_root. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeClass'
    var_5 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'MyType'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = 'module.TypeVar'
    var_3 = "TypeVar('T')"
    var_4 = 'typing.TypeVar'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = ''
    var_7 = module_0.Resolver(var_0, var_5, var_6)
    var_8 = 'T'
    var_9 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'list[int]'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'MyType'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3, var_0)
    var_5 = module_1.Load()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_visit_subscript_union_with_tuple. Retrieved 19/21 statements.
# Partially parsed test_visit_subscript_union_without_tuple. Retrieved 12/13 statements.
# Partially parsed test_visit_subscript_optional. Retrieved 16/19 statements.
# Partially parsed test_visit_subscript_pep585_list. Retrieved 14/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'str'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = [var_8, var_11]
    var_13 = module_1.Load()
    var_14 = module_1.Tuple()
    var_15 = module_1.Load()
    var_16 = module_1.Subscript()
    var_17 = var_2.visit_Subscript(var_16)
    var_18 = var_17.op

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Load()
    var_10 = module_1.Subscript()
    var_11 = var_2.visit_Subscript(var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.Optional'
    var_2 = 'typing.Optional'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Optional'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.Load()
    var_12 = module_1.Subscript()
    var_13 = var_4.visit_Subscript(var_12)
    var_14 = var_13.op
    var_15 = var_13.right

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'Union'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = 'str'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = [var_11, var_14]
    var_16 = module_1.Load()
    var_17 = module_1.Tuple()
    var_18 = module_1.Load()
    var_19 = module_1.Subscript()
    var_20 = var_2.visit_Subscript(var_19)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.list'
    var_2 = 'builtins.list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'list'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.Load()
    var_12 = module_1.Subscript()
    var_13 = var_4.visit_Subscript(var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'List'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Load()
    var_10 = module_1.Subscript()
    var_11 = var_2.visit_Subscript(var_10)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/10 statements.
# Partially parsed test_imports_simple_import_with_asname. Retrieved 4/10 statements.
# Partially parsed test_imports_from_import_absolute. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_absolute_with_asname. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 7/14 statements.
# Partially parsed test_imports_multiple_import_statements. Retrieved 5/12 statements.
# Partially parsed test_imports_from_import_no_module. Retrieved 5/11 statements.
# Partially parsed test_imports_nested_package. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'p'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.test_module'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub.test_module'
    var_2 = 'other'
    var_3 = 'cls'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = None
    var_5 = 'Counter'
    var_6 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.test_module'
    var_2 = None
    var_3 = 'func'
    var_4 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.subpkg.module'
    var_2 = 'utils'
    var_3 = 'helper'
    var_4 = 'h'
    var_5 = 0



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'original_name'
    var_2 = 'renamed_name'
    var_3 = 'test_module'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_load_docstring. Retrieved 2/19 statements.
# Partially parsed test_load_docstring_no_docstring. Retrieved 2/7 statements.
# Partially parsed test_load_docstring_nested_module. Retrieved 2/12 statements.
# Partially parsed test_load_docstring_with_doctest. Retrieved 2/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent.child'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc_enables_link. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_globals_type_comment_not_none. Retrieved 9/20 statements.


import ast as module_0

def test_case_0():
    var_0 = 'MY_CONST'
    var_1 = None
    var_2 = module_0.Name()
    var_3 = 42
    var_4 = module_0.Constant()
    var_5 = [var_2]
    var_6 = 'int'
    var_7 = module_0.Assign()
    var_8 = 'test_module'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_globals_with_annassign_and_value. Retrieved 9/15 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 6/13 statements.
# Partially parsed test_globals_with_assign_without_type_comment. Retrieved 6/13 statements.
# Partially parsed test_globals_with_all_assignment. Retrieved 11/19 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 7/16 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 9/21 statements.
# Partially parsed test_globals_uppercase_constant. Retrieved 6/13 statements.
# Partially parsed test_globals_lowercase_variable. Retrieved 6/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_var'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 100
    var_4 = module_1.Constant()
    var_5 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 'hello'
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = module_1.Constant()
    var_5 = 'func2'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.Tuple()
    var_10 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = None
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT'
    var_3 = 999
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'variable'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = None



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/11 statements.
# Partially parsed test_imports_simple_import_with_alias. Retrieved 4/11 statements.
# Partially parsed test_imports_multiple_imports. Retrieved 5/13 statements.
# Partially parsed test_imports_from_import_absolute. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_with_alias. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_relative_no_module. Retrieved 5/12 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 7/15 statements.
# Partially parsed test_imports_from_import_star. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'p'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.test_module'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub.test_module'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.test_module'
    var_2 = None
    var_3 = 'func'
    var_4 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 'getcwd'
    var_6 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = '*'
    var_4 = None
    var_5 = 0



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_imports_predicate_line_13_false. Retrieved 6/31 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'original_name'
    var_2 = 'renamed_name'
    var_3 = 'some_module'
    var_4 = 0
    var_5 = 'test_root'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_globals_predicate_line_38_evaluates_to_false. Retrieved 14/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = None
    var_4 = module_1.Name()
    var_5 = 'some_name'
    var_6 = module_1.Name()
    var_7 = [var_6]
    var_8 = module_1.List()
    var_9 = [var_4]
    var_10 = module_1.Assign()
    var_11 = var_0.globals(var_1, var_10)
    var_12 = var_0.imp[var_1]
    var_13 = len(var_12)
    assert var_13 == 0



# Parsed testcases at query #64
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'test_module'
    var_4 = 'def foo(): pass'
    var_5 = var_2.parse(var_3, var_4)
    var_6 = 'def test_func(): pass'
    var_7 = module_1.parse(var_6)
    var_8 = var_7.body[var_0]
    var_9 = var_2.api(var_3, var_8)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_class_api_predicate_line_11_false. Retrieved 15/41 statements.


import ast as module_0

def test_case_0():
    var_0 = 'class TestClass:\n    x: int = 5'
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = var_3.body
    var_5 = var_4[var_2]
    var_6 = var_5.target
    var_7 = 'y'
    var_8 = None
    var_9 = module_0.Name()
    var_10 = [var_9]
    var_11 = 10
    var_12 = module_0.Constant()
    var_13 = module_0.Assign()
    var_14 = var_13.target



# Parsed testcases at query #66
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = False
    var_7 = var_0.func_ann(var_1, var_5, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_deep_nested_attribute. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_broken_chain. Retrieved 2/7 statements.
# Partially parsed test_attr_none_value_in_chain. Retrieved 2/7 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_single_dot. Retrieved 1/5 statements.
# Partially parsed test_attr_with_integer_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_multiple_levels_with_none. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent.deeper'

def test_case_0():
    var_0 = None
    var_1 = 'inner.value.deeper'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '.'

def test_case_0():
    var_0 = 42
    var_1 = 'count'

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'attr'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = 'inner.value'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/10 statements.
# Partially parsed test_imports_simple_import_with_alias. Retrieved 4/10 statements.
# Partially parsed test_imports_multiple_names. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_absolute. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_with_alias. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_no_module. Retrieved 5/11 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 8/15 statements.
# Partially parsed test_imports_nested_module. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'filepath'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.sub.module'
    var_2 = 'other'
    var_3 = 'cls'
    var_4 = None
    var_5 = 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = None
    var_3 = 'func'
    var_4 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = None
    var_5 = 'Counter'
    var_6 = 'cnt'
    var_7 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.module'
    var_2 = 'json'
    var_3 = None



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_walk_body_simple_statements. Retrieved 6/18 statements.
# Partially parsed test_walk_body_if_statement. Retrieved 8/23 statements.
# Partially parsed test_walk_body_nested_if_statements. Retrieved 14/35 statements.
# Partially parsed test_walk_body_try_statement. Retrieved 11/33 statements.
# Partially parsed test_walk_body_try_with_multiple_handlers. Retrieved 14/42 statements.
# Partially parsed test_walk_body_try_with_orelse_and_finalbody. Retrieved 13/40 statements.
# Partially parsed test_walk_body_mixed_statements. Retrieved 12/30 statements.
# Partially parsed test_walk_body_if_with_empty_orelse. Retrieved 6/16 statements.


import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = True
    var_7 = module_0.Constant()

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'z'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = True
    var_10 = module_0.Constant()
    var_11 = True
    var_12 = module_0.Constant()
    var_13 = []

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'z'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = None
    var_10 = []

import ast as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'b'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'c'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = 'd'
    var_10 = 4
    var_11 = module_0.Constant()
    var_12 = None
    var_13 = []

import ast as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'b'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'c'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = 'd'
    var_10 = 4
    var_11 = module_0.Constant()
    var_12 = None

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'y'
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = 'z'
    var_7 = 3
    var_8 = module_0.Constant()
    var_9 = True
    var_10 = module_0.Constant()
    var_11 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = True
    var_4 = module_0.Constant()
    var_5 = []



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 8/13 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 8/15 statements.
# Partially parsed test_globals_with_all_list. Retrieved 9/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 9/14 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 9/12 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 9/15 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 9/12 statements.
# Partially parsed test_globals_with_annotated_no_value. Retrieved 9/12 statements.
# Partially parsed test_globals_string_constant. Retrieved 8/13 statements.
# Partially parsed test_globals_with_all_non_string_elements. Retrieved 9/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'x'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT = 42'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'CONSTANT'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MAX_VALUE = 100'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'MAX_VALUE'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ['func1', 'func2']"
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'func1'
    var_8 = 'func2'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ('item1', 'item2')"
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'item1'
    var_8 = 'item2'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a, b = 1, 2'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'value = 10  # type: int'
    var_3 = True
    var_4 = module_1.parse(var_2, type_comments=var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'value'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = y = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "TEXT = 'hello'"
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = 'TEXT'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__ = [1, 2, 3]'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = var_0.imp[var_1]
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_class_api_predicate_line_11_false. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'x = 5'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body
    var_6 = []
    var_7 = var_0.class_api(var_1, var_2, var_6, var_5)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_class_api_predicate_line_19_false. Retrieved 13/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False when len(node.targets) != 1'
    var_1 = module_0.Parser()
    var_2 = 'a = b = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_5.targets
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_5.targets
    var_9 = len(var_8)
    var_10 = 1
    var_11 = var_9 == var_10
    var_12 = var_5.targets[var_4]



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_globals_predicate_line_35_false. Retrieved 15/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "Test that the predicate at line 35 evaluates to False.\n    \n    The predicate is: left.id != '__all__' or not isinstance(node.value, (Tuple, List))\n    For it to be False, both conditions must be False:\n    - left.id == '__all__' (first part is False)\n    - isinstance(node.value, (Tuple, List)) (second part is False, making 'not' True becomes False)\n    \n    So we need: left.id == '__all__' AND node.value is a Tuple or List\n    "
    var_1 = module_0.Parser()
    var_2 = '__all__'
    var_3 = None
    var_4 = module_1.Name()
    var_5 = 'func1'
    var_6 = module_1.Constant()
    var_7 = 'func2'
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Tuple()
    var_11 = [var_4]
    var_12 = module_1.Assign()
    var_13 = 'test_module'
    var_14 = var_1.globals(var_13, var_12)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_func_api_predicate_line_32_false. Retrieved 18/21 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = 42
    var_9 = module_1.Constant()
    var_10 = [var_9]
    var_11 = module_1.arguments(*var_5)
    var_12 = 'test_module'
    var_13 = 'test_module.test_func'
    var_14 = False
    var_15 = var_0.func_api(var_12, var_13, var_11, var_3, has_self=var_14, cls_method=var_14)
    var_16 = var_0.doc[var_13]
    var_17 = len(var_16)



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_level. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_break_chain_at_middle. Retrieved 2/7 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_attr_numeric_value. Retrieved 2/5 statements.
# Partially parsed test_attr_string_value. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = 'test'
    var_1 = 'inner.nonexistent.value'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = None
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 42
    var_1 = 'number'

def test_case_0():
    var_0 = 'hello'
    var_1 = 'text'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_load_docstring. Retrieved 2/14 statements.
# Partially parsed test_load_docstring_with_missing_attribute. Retrieved 4/13 statements.
# Partially parsed test_load_docstring_filters_by_root. Retrieved 2/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.NonExistent'
    var_3 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_a'



# Parsed testcases at query #79
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to False.'
    var_1 = 'mymodule'
    var_2 = 'mymodule.MyType'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = module_0.Resolver(var_1, var_4, var_5)
    var_7 = 'MyType'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = var_6.visit_Name(var_9)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_func_api_predicate_line_32_false. Retrieved 20/34 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = None
    var_3 = module_0.arg()
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = 5
    var_8 = module_0.Constant()
    var_9 = [var_8]
    var_10 = module_0.arguments(*var_4)
    var_11 = []
    var_12 = [var_2]
    var_13 = var_10.args
    var_14 = len(var_13)
    var_15 = var_10.defaults
    var_16 = len(var_15)
    var_17 = var_14 - var_16
    var_18 = var_12 * var_17
    var_19 = var_10.defaults



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 9/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/18 statements.
# Partially parsed test_class_api_with_enum. Retrieved 16/24 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 10/22 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_with_assign_member. Retrieved 8/17 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 10/18 statements.
# Partially parsed test_class_api_with_multiple_members. Retrieved 14/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'
    var_8 = var_0.class_api(var_6, var_7, var_4, var_5)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = None
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'Enum'
    var_5 = module_1.Load()
    var_6 = module_1.Attribute()
    var_7 = [var_6]
    var_8 = 'MEMBER'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = None
    var_13 = 1
    var_14 = 'test_module'
    var_15 = 'test_module.MyEnum'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = None
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.MyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = None
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = None
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 'member2'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = None
    var_7 = 1
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = []
    var_12 = 'test_module'
    var_13 = 'test_module.MyClass'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_visit_name_predicate_line_9_true. Retrieved 10/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'TestName'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'module.TestName'
    var_8 = 'SomeFunc()'
    var_9 = var_3.visit_Name(var_6)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_class_api_predicate_enums_true. Retrieved 12/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = '\nclass TestEnum(enum.Enum):\n    MEMBER: int\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_6.body
    var_8 = var_6.bases
    var_9 = var_0.resolve
    var_10 = 'enum.Enum'
    var_11 = var_0.class_api(var_1, var_2, var_8, var_7)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_containing_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_not_containing_name. Retrieved 6/8 statements.
# Partially parsed test_is_public_module_in_imp. Retrieved 7/11 statements.
# Partially parsed test_is_public_with_parent_in_all_list. Retrieved 6/8 statements.
# Partially parsed test_is_public_root_module. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_empty_all_and_public_family. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_empty_all_and_private_family. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg._private'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.__init__'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.func'
    var_2 = 'pkg'
    var_3 = {var_1}
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.func'
    var_2 = 'pkg'
    var_3 = 'pkg.other'
    var_4 = {var_3}
    var_5 = var_0.is_public(var_1)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submodule'
    var_2 = set()
    var_3 = 'pkg.submodule.func'
    var_4 = 'pkg.submodule.Class'
    var_5 = ''
    var_6 = var_0.is_public(var_1)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submodule.func'
    var_2 = 'pkg'
    var_3 = 'pkg.submodule'
    var_4 = {var_3}
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mypackage'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.Public'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg._private'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_visit_name_typevar_predicate. Retrieved 10/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeType'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'mymodule.SomeType'
    var_8 = "TypeVar('T')"
    var_9 = var_3.visit_Name(var_6)



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_class_api_line_19_predicate_false. Retrieved 14/23 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'test_module'
    var_10 = 'test_class'
    var_11 = []
    var_12 = [var_8]
    var_13 = var_0.class_api(var_9, var_10, var_11, var_12)



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_class_api_line_25_predicate_false. Retrieved 14/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (is_public_family(attr)) evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '_private_attr'
    var_3 = None
    var_4 = module_1.Name()
    var_5 = [var_4]
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'
    var_11 = []
    var_12 = [var_8]
    var_13 = var_1.class_api(var_9, var_10, var_11, var_12)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #89
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass TestClass:\n    """Test class."""\n    attr1: int\n    attr2: str = "default"\n    _private: int = 5\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.TestClass'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_0.class_api(var_1, var_8, var_9, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.Color'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_0.class_api(var_1, var_8, var_9, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass Child(Parent):\n    pass\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.Child'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_0.class_api(var_1, var_8, var_9, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'class Empty: pass'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.Empty'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_0.class_api(var_1, var_8, var_9, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass TestClass:\n    attr1: int = 1\n    del attr1\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.TestClass'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_0.class_api(var_1, var_8, var_9, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass TestClass:\n    attr1 = 10  # type: int\n    '
    var_5 = True
    var_6 = module_1.parse(var_4, type_comments=var_5)
    var_7 = 0
    var_8 = var_6.body[var_7]
    var_9 = 'test_module.TestClass'
    var_10 = var_8.bases
    var_11 = var_8.body
    var_12 = var_0.class_api(var_1, var_9, var_10, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass Status(enum.Enum):\n    ACTIVE = 1\n    INACTIVE = 2\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.Status'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_0.class_api(var_1, var_8, var_9, var_10)



# Parsed testcases at query #90
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'this is not valid python syntax !!!!'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_enums. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_bases. Retrieved 10/13 statements.
# Partially parsed test_class_api_empty_class. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 11/14 statements.
# Partially parsed test_class_api_mixed_public_private. Retrieved 10/13 statements.
# Partially parsed test_class_api_enum_detection. Retrieved 12/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Color:\n    RED: int\n    GREEN: int\n    BLUE: int\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.Color'
    var_7 = []
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Derived(Base):\n    pass\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.Derived'
    var_7 = var_5.bases
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Empty:\n    pass\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.Empty'
    var_7 = []
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    attr: int\n    del attr\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    value = 42  # type: int\n    '
    var_3 = True
    var_4 = module_1.parse(var_2, type_comments=var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.TestClass'
    var_8 = []
    var_9 = var_6.body
    var_10 = var_0.class_api(var_1, var_7, var_8, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    public_field: str\n    _private_field: int\n    another_public: float\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Status:\n    PENDING = 1\n    APPROVED = 2\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'enum.Enum'
    var_7 = module_1.Name()
    var_8 = [var_7]
    var_9 = 'test_module.Status'
    var_10 = var_5.body
    var_11 = var_0.class_api(var_1, var_9, var_8, var_10)



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_func_ann_star_argument. Retrieved 10/41 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'test_root'
    var_5 = [var_3]
    var_6 = False
    var_7 = var_0.func_ann(var_4, var_5, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 12/25 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = [1, 2, 3]'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = var_0.imp[var_3]
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = '__all__ = [1, 2, 3]'
    var_7 = var_0.imp[var_3]
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = '__all__ = [some_var]'
    var_10 = var_0.imp[var_3]
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_attr_predicate_at_line_4_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_constructor_dict_independence. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = module_0.Parser()



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_class_api_is_enum_predicate. Retrieved 17/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum.Enum'
    var_2 = 'eval'
    var_3 = module_1.parse(var_1, mode=var_2)
    var_4 = var_3.body
    var_5 = [var_4]
    var_6 = 'MEMBER'
    var_7 = None
    var_8 = module_1.Name()
    var_9 = [var_8]
    var_10 = 1
    var_11 = module_1.Constant()
    var_12 = module_1.Assign()
    var_13 = [var_12]
    var_14 = 'test_module'
    var_15 = 'test_module.TestClass'
    var_16 = var_0.class_api(var_14, var_15, var_5, var_13)



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_class_api_is_enum_predicate. Retrieved 22/27 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 0
    var_2 = 'enum.Enum'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = [var_5]
    var_7 = 'MEMBER'
    var_8 = None
    var_9 = module_1.Name()
    var_10 = [var_9]
    var_11 = 1
    var_12 = module_1.Constant()
    var_13 = module_1.Assign()
    var_14 = [var_13]
    var_15 = None
    var_16 = 'test'
    var_17 = 'test.MyEnum'
    var_18 = var_0.class_api(var_16, var_17, var_6, var_14)
    var_19 = var_0.doc[var_17]
    var_20 = len(var_19)
    var_21 = var_20 > var_1



# Parsed testcases at query #98
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to False.'
    var_1 = 'module'
    var_2 = 'other_name'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = module_0.Resolver(var_1, var_4, var_5)
    var_7 = 'test_name'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = var_6.visit_Name(var_9)
    var_11 = 'module.name'
    var_12 = {var_11: var_11}
    var_13 = module_0.Resolver(var_1, var_12, var_5)
    var_14 = 'name'
    var_15 = module_1.Load()
    var_16 = module_1.Name()
    var_17 = var_13.visit_Name(var_16)
    var_18 = {}
    var_19 = module_0.Resolver(var_1, var_18, var_5)
    var_20 = 'undefined'
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = var_19.visit_Name(var_22)



# Parsed testcases at query #99
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax !!!invalid'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 4/12 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 4/12 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/14 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_ignores_invalid_assignments. Retrieved 8/16 statements.
# Partially parsed test_globals_with_lowercase_variable. Retrieved 4/12 statements.
# Partially parsed test_globals_with_string_constant. Retrieved 4/12 statements.
# Partially parsed test_globals_with_none_value. Retrieved 4/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 5'
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MAX_SIZE = 100'
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['func1', 'func2']"
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ('func1', 'func2')"
    var_4 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 5  # type: int'
    var_3 = 0
    var_4 = True
    var_5 = 'test_module.x'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x, y = 1, 2'
    var_3 = 0
    var_4 = var_0.alias
    var_5 = len(var_4)
    var_6 = var_0.alias
    var_7 = len(var_6)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 5'
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "MESSAGE = 'hello'"
    var_3 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'EMPTY = None'
    var_3 = 0



