####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 6/12 statements.
# Partially parsed test_imports_with_from_import_statement. Retrieved 6/12 statements.
# Partially parsed test_imports_with_relative_import. Retrieved 6/12 statements.
# Partially parsed test_imports_with_relative_import_one_level. Retrieved 6/12 statements.
# Partially parsed test_imports_with_from_import_multiple_names. Retrieved 6/12 statements.
# Partially parsed test_imports_with_import_no_alias. Retrieved 6/12 statements.
# Partially parsed test_imports_with_nested_module_from_import. Retrieved 6/12 statements.
# Partially parsed test_imports_empty_module. Retrieved 8/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'import os\nimport sys as system'
    var_4 = module_1.parse(var_3)
    var_5 = 'test_module'
    var_6 = var_0.alias['test_module.os']
    assert var_6 == 'os'
    var_7 = var_0.alias['test_module.system']
    assert var_7 == 'sys'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'from os import path\nfrom sys import argv as args'
    var_4 = module_1.parse(var_3)
    var_5 = 'test_module'
    var_6 = var_0.alias['test_module.path']
    assert var_6 == 'os.path'
    var_7 = var_0.alias['test_module.args']
    assert var_7 == 'sys.argv'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub.module'
    var_2 = 2
    var_3 = 'from ..utils import helper'
    var_4 = module_1.parse(var_3)
    var_5 = 'pkg.sub.module'
    var_6 = var_0.alias['pkg.sub.module.helper']
    assert var_6 == 'pkg.utils.helper'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 1
    var_3 = 'from .utils import func'
    var_4 = module_1.parse(var_3)
    var_5 = 'pkg.module'
    var_6 = var_0.alias['pkg.module.func']
    assert var_6 == 'pkg.utils.func'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'from os import path, getcwd, chdir as change_dir'
    var_4 = module_1.parse(var_3)
    var_5 = 'test_module'
    var_6 = var_0.alias['test_module.path']
    assert var_6 == 'os.path'
    var_7 = var_0.alias['test_module.getcwd']
    assert var_7 == 'os.getcwd'
    var_8 = var_0.alias['test_module.change_dir']
    assert var_8 == 'os.chdir'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'import json\nimport collections'
    var_4 = module_1.parse(var_3)
    var_5 = 'test_module'
    var_6 = var_0.alias['test_module.json']
    assert var_6 == 'json'
    var_7 = var_0.alias['test_module.collections']
    assert var_7 == 'collections'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'app.models'
    var_2 = 1
    var_3 = 'from .utils.helpers import process_data'
    var_4 = module_1.parse(var_3)
    var_5 = 'app.models'
    var_6 = var_0.alias['app.models.process_data']
    assert var_6 == 'app.utils.helpers.process_data'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'empty_module'
    var_2 = 0
    var_3 = ''
    var_4 = module_1.parse(var_3)
    var_5 = 'empty_module'
    var_6 = var_0.alias
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_docstring. Retrieved 3/26 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module'
    var_3 = bool(var_1 in var_0.docstring)
    assert var_3 is True
    var_4 = 'Module docstring'
    var_5 = bool('Module docstring' in var_0.docstring[var_1])
    assert var_5 is True
    var_6 = 'test_module.func1'
    var_7 = bool('test_module.func1' in var_0.docstring)
    assert var_7 is True
    var_8 = 'Function docstring'
    var_9 = bool('Function docstring' in var_0.docstring['test_module.func1'])
    assert var_9 is True
    var_10 = 'test_module.Class1'
    var_11 = bool('test_module.Class1' in var_0.docstring)
    assert var_11 is True
    var_12 = 'Class docstring'
    var_13 = bool('Class docstring' in var_0.docstring['test_module.Class1'])
    assert var_13 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_api_function_def. Retrieved 5/12 statements.
# Partially parsed test_api_async_function_def. Retrieved 5/12 statements.
# Partially parsed test_api_class_def. Retrieved 5/12 statements.
# Partially parsed test_api_with_decorator. Retrieved 5/12 statements.
# Partially parsed test_api_with_prefix. Retrieved 6/13 statements.
# Partially parsed test_api_nested_class. Retrieved 6/15 statements.
# Partially parsed test_api_with_anchor_link. Retrieved 5/12 statements.
# Partially parsed test_api_underscore_escaping. Retrieved 5/12 statements.
# Partially parsed test_api_level_calculation. Retrieved 6/13 statements.
# Partially parsed test_api_with_docstring. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def foo(): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.foo'
    var_6 = bool('test_module.foo' in var_1.doc)
    assert var_6 is True
    var_7 = '# foo()'
    var_8 = bool('# foo()' in var_1.doc['test_module.foo'])
    assert var_8 is True
    var_9 = '*Full name:* `test_module.foo`'
    var_10 = bool('*Full name:* `test_module.foo`' in var_1.doc['test_module.foo'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'async def bar(): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.bar'
    var_6 = bool('test_module.bar' in var_1.doc)
    assert var_6 is True
    var_7 = '# async bar()'
    var_8 = bool('# async bar()' in var_1.doc['test_module.bar'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'class MyClass: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.MyClass'
    var_6 = bool('test_module.MyClass' in var_1.doc)
    assert var_6 is True
    var_7 = '# class MyClass'
    var_8 = bool('# class MyClass' in var_1.doc['test_module.MyClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '@staticmethod\ndef decorated(): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.decorated'
    var_6 = bool('test_module.decorated' in var_1.doc)
    assert var_6 is True
    var_7 = 'Decorators'
    var_8 = bool('Decorators' in var_1.doc['test_module.decorated'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def method(self): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'OuterClass'
    var_6 = 'test_module.OuterClass.method'
    var_7 = bool('test_module.OuterClass.method' in var_1.doc)
    assert var_7 is True
    var_8 = 'OuterClass.method()'
    var_9 = bool('OuterClass.method()' in var_1.doc['test_module.OuterClass.method'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'class Outer:\n    class Inner: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'Outer'
    var_6 = 'test_module.Outer'
    var_7 = bool('test_module.Outer' in var_1.doc)
    assert var_7 is True
    var_8 = 'test_module.Outer.Inner'
    var_9 = bool('test_module.Outer.Inner' in var_1.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def linked_func(): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = '<a id="{}">'
    var_6 = bool('<a id="{}">' in var_1.doc['test_module.linked_func'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def func_with_underscores(): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'func\\_with\\_underscores'
    var_6 = bool('func\\_with\\_underscores' in var_1.doc['test_module.func_with_underscores'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = module_0.Parser(var_0, var_1)
    var_3 = 'def func(): pass'
    var_4 = 0
    var_5 = 'pkg.module'
    var_6 = var_2.level['pkg.module.func']
    assert var_6 == 1
    var_7 = var_2.root['pkg.module.func']
    assert var_7 == 'pkg.module'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def func():\n    """This is a docstring."""\n    pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.func'
    var_6 = bool('test_module.func' in var_1.docstring)
    assert var_6 is True
    var_7 = 'This is a docstring.'
    var_8 = bool('This is a docstring.' in var_1.docstring['test_module.func'])
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_enum. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_bases. Retrieved 5/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    constant: int = 42\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = 'test_module.TestClass'
    var_6 = bool('test_module.TestClass' in var_0.doc)
    assert var_6 is True
    var_7 = 'Members'
    var_8 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Color(enum.Enum):\n    RED: int = 1\n    GREEN: int = 2\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Color'
    var_5 = 'test_module.Color'
    var_6 = bool('test_module.Color' in var_0.doc)
    assert var_6 is True
    var_7 = 'Enums'
    var_8 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Child(Parent):\n    pass\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Child'
    var_5 = 'test_module.Child'
    var_6 = bool('test_module.Child' in var_0.doc)
    assert var_6 is True
    var_7 = 'Bases'
    var_8 = bool('Bases' in var_0.doc['test_module.Child'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Empty:\n    pass\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Empty'
    var_5 = 'test_module.Empty'
    var_6 = bool('test_module.Empty' in var_0.doc)
    assert var_6 is True
    var_7 = var_0.doc['test_module.Empty']
    var_8 = bool(var_0.doc['test_module.Empty'] != '')
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr1\n    '
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = 'test_module.TestClass'
    var_6 = bool('test_module.TestClass' in var_0.doc)
    assert var_6 is True
    var_7 = 'attr1'
    var_8 = bool('attr1' not in var_0.doc['test_module.TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    value = 100  # type: int\n    '
    var_2 = True
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_class_api_annassign_with_name_target. Retrieved 6/26 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.target



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_tuple_of_strs. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_set_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_mixed_types_in_list. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_dict_int_keys_str_values. Retrieved 4/13 statements.
# Partially parsed test_const_type_with_empty_dict. Retrieved 2/5 statements.
# Partially parsed test_const_type_with_call_to_int. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_call_to_str. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_call_to_bool. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_call_to_float. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_call_to_complex. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_unsupported_node. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 'a'
    var_5 = []
    var_6 = 'b'
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = '42'
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = 42
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 1
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'float'
    var_1 = []
    var_2 = '3.14'
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'complex'
    var_1 = []
    var_2 = '1+2j'
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parse_level_calculation. Retrieved 4/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 5'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.level)
    assert var_7 is True
    var_8 = 'test_module'
    var_9 = bool('test_module' in var_0.imp)
    assert var_9 is True
    var_10 = 'test_module'
    var_11 = bool('test_module' in var_0.root)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""Module docstring.\n\n>>> x = 1\n>>> print(x)\n1\n"""\nx = 5'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.docstring)
    assert var_5 is True
    var_6 = '```python'
    var_7 = bool('```python' in var_0.docstring['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nfrom sys import path'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.imp)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST_VALUE = 42\nMY_CONST: int = 100'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.CONST_VALUE'
    var_5 = bool('test_module.CONST_VALUE' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.MY_CONST'
    var_7 = bool('test_module.MY_CONST' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def my_func():\n    """Function doc."""\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.my_func'
    var_5 = bool('test_module.my_func' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.my_func'
    var_7 = bool('test_module.my_func' in var_0.level)
    assert var_7 is True
    var_8 = 'test_module.my_func'
    var_9 = bool('test_module.my_func' in var_0.docstring)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass:\n    """Class doc."""\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.level)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_func():\n    """Async function."""\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.async_func'
    var_5 = bool('test_module.async_func' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Outer:\n    """Outer class."""\n    class Inner:\n        """Inner class."""\n        pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.Outer'
    var_5 = bool('test_module.Outer' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.Outer.Inner'
    var_7 = bool('test_module.Outer.Inner' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 5  # type: int'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 5'
    var_2 = 'pkg.module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.level['pkg.module']
    assert var_4 == 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 5'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = '# Module `{}`'
    var_5 = bool('# Module `{}`' in var_0.doc['test_module'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'x = 5'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = '<a id="{}"></a>'
    var_6 = bool('<a id="{}"></a>' in var_1.doc['test_module'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = 'x = 5'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = '<a id="{}"></a>'
    var_6 = bool('<a id="{}"></a>' not in var_1.doc['test_module'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "__all__ = ['func1', 'func2']"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.func1'
    var_5 = bool('test_module.func1' in var_0.imp['test_module'])
    assert var_5 is True
    var_6 = 'test_module.func2'
    var_7 = bool('test_module.func2' in var_0.imp['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@staticmethod\n@property\ndef my_func():\n    """Decorated function."""\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.my_func'
    var_5 = bool('test_module.my_func' in var_0.doc)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/4 statements.
# Partially parsed test_parser_new_classmethod_with_false_values. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2

def test_case_0():
    var_0 = False
    var_1 = 3



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_globals_ann_assign_with_value. Retrieved 6/16 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_assign_without_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_all_list. Retrieved 6/19 statements.
# Partially parsed test_globals_all_tuple. Retrieved 6/19 statements.
# Partially parsed test_globals_ignores_non_constant_values. Retrieved 7/20 statements.
# Partially parsed test_globals_multiple_targets_ignored. Retrieved 6/17 statements.
# Partially parsed test_globals_ann_assign_without_value_ignored. Retrieved 6/15 statements.
# Partially parsed test_globals_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_lowercase_no_root. Retrieved 6/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 'int'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 1
    var_8 = var_0.alias['test_module.MY_CONST']
    assert var_8 == '42'
    var_9 = var_0.const['test_module.MY_CONST']
    assert var_9 == 'int'
    var_10 = var_0.root['test_module.MY_CONST']
    var_11 = bool(var_0.root['test_module.MY_CONST'] == var_1)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_VAR'
    var_3 = 100
    var_4 = []
    var_5 = 'int'
    var_6 = var_0.alias['test_module.MY_VAR']
    assert var_6 == '100'
    var_7 = var_0.const['test_module.MY_VAR']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_NUM'
    var_3 = 3.14
    var_4 = []
    var_5 = None
    var_6 = var_0.alias['test_module.MY_NUM']
    assert var_6 == '3.14'
    var_7 = var_0.const['test_module.MY_NUM']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.func1'
    var_10 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.func2'
    var_12 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'ClassA'
    var_4 = []
    var_5 = 'ClassB'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.ClassA'
    var_10 = bool('test_module.ClassA' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.ClassB'
    var_12 = bool('test_module.ClassB' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'var'
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = var_0.imp[var_1]
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = []
    var_6 = None
    var_7 = 'test_module.a'
    var_8 = bool('test_module.a' not in var_0.alias)
    assert var_8 is True
    var_9 = 'test_module.b'
    var_10 = bool('test_module.b' not in var_0.alias)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_VAR'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module.MY_VAR'
    var_8 = bool('test_module.MY_VAR' not in var_0.alias)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT'
    var_3 = 'value'
    var_4 = []
    var_5 = None
    var_6 = var_0.root['test_module.CONSTANT']
    var_7 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_7 is True
    var_8 = var_0.const['test_module.CONSTANT']
    assert var_8 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'variable'
    var_3 = 42
    var_4 = []
    var_5 = None
    var_6 = 'test_module.variable'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_levels. Retrieved 2/7 statements.
# Partially parsed test_attr_multiple_nested_levels. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_nonexistent_first_level. Retrieved 2/5 statements.
# Partially parsed test_attr_none_in_chain. Retrieved 2/7 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_numeric_values. Retrieved 2/5 statements.
# Partially parsed test_attr_with_boolean_values. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'attr1'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value1'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = 'value1'
    var_1 = 'missing.nested.attr'

def test_case_0():
    var_0 = None
    var_1 = 'inner.value.something'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 42
    var_1 = 'number'

def test_case_0():
    var_0 = True
    var_1 = 'flag'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_compile_basic. Retrieved 5/10 statements.
# Partially parsed test_compile_with_toc. Retrieved 6/15 statements.
# Partially parsed test_compile_with_links. Retrieved 5/10 statements.
# Partially parsed test_compile_with_constants. Retrieved 5/13 statements.
# Partially parsed test_compile_magic_methods_skipped. Retrieved 5/13 statements.
# Partially parsed test_compile_private_names_excluded. Retrieved 5/13 statements.
# Partially parsed test_compile_nested_modules. Retrieved 6/15 statements.
# Partially parsed test_compile_multiple_functions. Retrieved 5/18 statements.
# Partially parsed test_compile_with_all_filter. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with basic documentation.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = '# Module `module`'
    var_6 = bool('# Module `module`' in var_4)
    assert var_6 is True
    var_7 = 'Module docstring'
    var_8 = bool('Module docstring' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with table of contents.'
    var_1 = False
    var_2 = 1
    var_3 = True
    var_4 = module_0.Parser(var_1, var_2, var_3)
    var_5 = var_4.compile()
    var_6 = '**Table of contents:**'
    var_7 = bool('**Table of contents:**' in var_5)
    assert var_7 is True
    var_8 = 'module-func'
    var_9 = bool('module-func' in var_5)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with link anchors.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()
    var_5 = '<a id="module"></a>'
    var_6 = bool('<a id="module"></a>' in var_4)
    assert var_6 is True
    var_7 = '# Module `module`'
    var_8 = bool('# Module `module`' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with constants table.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = 'Constants'
    var_6 = bool('Constants' in var_4)
    assert var_6 is True
    var_7 = 'CONST'
    var_8 = bool('CONST' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile skips magic methods without docstring.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = 'module.__init__'
    var_6 = bool('module.__init__' not in var_4)
    assert var_6 is True
    var_7 = '# Module `module`'
    var_8 = bool('# Module `module`' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile excludes private names.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = '_private'
    var_6 = bool('_private' not in var_4)
    assert var_6 is True
    var_7 = '# Module `module`'
    var_8 = bool('# Module `module`' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with nested module structure.'
    var_1 = False
    var_2 = 1
    var_3 = True
    var_4 = module_0.Parser(var_1, var_2, var_3)
    var_5 = var_4.compile()
    var_6 = '**Table of contents:**'
    var_7 = bool('**Table of contents:**' in var_5)
    assert var_7 is True
    var_8 = 'pkg'
    var_9 = bool('pkg' in var_5)
    assert var_9 is True
    var_10 = 'pkg.sub'
    var_11 = bool('pkg.sub' in var_5)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with empty parser.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    assert var_4 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with multiple functions.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = 'func1'
    var_6 = bool('func1' in var_4)
    assert var_6 is True
    var_7 = 'func2'
    var_8 = bool('func2' in var_4)
    assert var_8 is True
    var_9 = 'Function 1 doc'
    var_10 = bool('Function 1 doc' in var_4)
    assert var_10 is True
    var_11 = 'Function 2 doc'
    var_12 = bool('Function 2 doc' in var_4)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile respects __all__ filter.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = 'module.public'
    var_5 = var_3.compile()
    var_6 = 'public'
    var_7 = bool('public' in var_5)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_func_ann_with_self. Retrieved 8/15 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 8/17 statements.
# Partially parsed test_func_ann_without_self. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_annotation. Retrieved 8/17 statements.
# Partially parsed test_func_ann_with_varargs. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_self_type_annotation. Retrieved 10/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = []
    var_6 = 'return'
    var_7 = []
    var_8 = 'test_module'
    var_9 = True
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MyClass'
    var_2 = None
    var_3 = []
    var_4 = 'cls'
    var_5 = 'x'
    var_6 = []
    var_7 = 'return'
    var_8 = []
    var_9 = 'test_module'
    var_10 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'y'
    var_5 = []
    var_6 = 'return'
    var_7 = []
    var_8 = 'test_module'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'int'
    var_2 = None
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = 'x'
    var_7 = 'return'
    var_8 = 'test_module'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = []
    var_6 = 'return'
    var_7 = []
    var_8 = 'test_module'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MyClass'
    var_2 = None
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 'self'
    var_7 = 'x'
    var_8 = 'return'
    var_9 = []
    var_10 = 'test_module'
    var_11 = True
    var_12 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_globals_predicate_line_18_false. Retrieved 7/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when len(node.targets) != 1'
    var_1 = module_0.Parser()
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = 1
    var_8 = []
    var_9 = 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_levels. Retrieved 1/10 statements.
# Partially parsed test_attr_deep_nesting. Retrieved 1/14 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 1/9 statements.
# Partially parsed test_attr_none_in_chain. Retrieved 1/6 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_method. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'x'

def test_case_0():
    var_0 = 'b.value'

def test_case_0():
    var_0 = 'b.c.data'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'b.nonexistent'

def test_case_0():
    var_0 = 'b.c.d'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'get_value'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 6/17 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 5/15 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/15 statements.
# Partially parsed test_globals_with_all_list. Retrieved 7/22 statements.
# Partially parsed test_globals_ignores_lowercase_assignment. Retrieved 5/15 statements.
# Partially parsed test_globals_ignores_non_name_target. Retrieved 7/26 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 6/18 statements.
# Partially parsed test_globals_with_annotated_assignment_no_value. Retrieved 6/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module.MY_CONST'
    var_9 = bool('test_module.MY_CONST' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.MY_CONST']
    assert var_10 == '42'
    var_11 = 'test_module.MY_CONST'
    var_12 = bool('test_module.MY_CONST' in var_0.const)
    assert var_12 is True
    var_13 = var_0.const['test_module.MY_CONST']
    assert var_13 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 100
    var_4 = []
    var_5 = None
    var_6 = 'test_module.MY_CONST'
    var_7 = bool('test_module.MY_CONST' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.MY_CONST']
    assert var_8 == '100'
    var_9 = 'test_module.MY_CONST'
    var_10 = bool('test_module.MY_CONST' in var_0.const)
    assert var_10 is True
    var_11 = var_0.const['test_module.MY_CONST']
    assert var_11 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_VAR'
    var_3 = 'hello'
    var_4 = []
    var_5 = 'str'
    var_6 = 'test_module.MY_VAR'
    var_7 = bool('test_module.MY_VAR' in var_0.const)
    assert var_7 is True
    var_8 = var_0.const['test_module.MY_VAR']
    assert var_8 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'func1'
    var_5 = []
    var_6 = 'func2'
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.func1'
    var_11 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_11 is True
    var_12 = 'test_module.func2'
    var_13 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_var'
    var_3 = 42
    var_4 = []
    var_5 = None
    var_6 = 'test_module.my_var'
    var_7 = bool('test_module.my_var' in var_0.alias)
    assert var_7 is True
    var_8 = 'test_module.my_var'
    var_9 = bool('test_module.my_var' not in var_0.root)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = []
    var_6 = 2
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.a'
    var_11 = bool('test_module.a' not in var_0.alias)
    assert var_11 is True
    var_12 = 'test_module.b'
    var_13 = bool('test_module.b' not in var_0.alias)
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR1'
    var_3 = 'VAR2'
    var_4 = 42
    var_5 = []
    var_6 = None
    var_7 = 'test_module.VAR1'
    var_8 = bool('test_module.VAR1' not in var_0.alias)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module.MY_CONST'
    var_8 = bool('test_module.MY_CONST' not in var_0.alias)
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_visit_constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_invalid_syntax_string. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_valid_name_string. Retrieved 6/9 statements.
# Partially parsed test_visit_constant_with_self_type. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_complex_expression. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_subscript_string. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_none_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_empty_string. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not valid python @#$'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyType'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'int | str'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'list[int]'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = None
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = ''
    var_4 = []



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_public_submodule. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_private_submodule. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_all_list_containing_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_all_list_not_containing_name. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_all_list_containing_parent. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_module_in_imp_keys_no_children. Retrieved 4/8 statements.
# Partially parsed test_is_public_with_module_in_imp_keys_with_public_children. Retrieved 6/10 statements.
# Partially parsed test_is_public_magic_name_without_all. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_all_containing_magic_name. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg._private'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = 'pkg'
    var_3 = {var_1}
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = 'pkg'
    var_3 = 'pkg.other'
    var_4 = {var_3}
    var_5 = 'doc'
    var_6 = var_0.is_public(var_1)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub.deep'
    var_2 = 'pkg'
    var_3 = 'pkg.sub'
    var_4 = {var_3}
    var_5 = 'doc'
    var_6 = var_0.is_public(var_1)
    assert var_6 is True

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
    var_1 = 'pkg'
    var_2 = 'pkg.public'
    var_3 = set()
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.__init__'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.__init__'
    var_2 = 'pkg'
    var_3 = {var_1}
    var_4 = 'doc'
    var_5 = var_0.is_public(var_1)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_func_ann_annotation_not_none. Retrieved 10/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'str'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_5.value
    var_7 = 'param'
    var_8 = []
    var_9 = False
    var_10 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_visit_constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_string_value_valid_name. Retrieved 5/8 statements.
# Partially parsed test_visit_constant_with_string_value_invalid_syntax. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_string_value_self_type. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_string_value_complex_expression. Retrieved 4/9 statements.
# Partially parsed test_visit_constant_with_empty_string. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_string_value_alias_resolution. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.MyClass'
    var_2 = 'MyClass'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = '@#$%'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'int | str'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = ''
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'List'
    var_6 = []



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'value'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_func_ann_with_self_parameter. Retrieved 8/16 statements.
# Partially parsed test_func_ann_with_cls_method. Retrieved 7/15 statements.
# Partially parsed test_func_ann_without_self. Retrieved 7/15 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 6/13 statements.
# Partially parsed test_func_ann_with_annotation. Retrieved 7/15 statements.
# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 10/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = 'return'
    var_8 = []
    var_9 = True
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = 'return'
    var_8 = []
    var_9 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = 'return'
    var_8 = []
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = None
    var_4 = []
    var_5 = 'return'
    var_6 = []
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'int'
    var_3 = []
    var_4 = 'x'
    var_5 = 'return'
    var_6 = None
    var_7 = []
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MyClass'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = 'self'
    var_7 = 'x'
    var_8 = 'return'
    var_9 = None
    var_10 = []
    var_11 = True
    var_12 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_api_simple_function. Retrieved 10/17 statements.
# Partially parsed test_func_api_with_arguments. Retrieved 11/20 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 10/20 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/21 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 11/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.func'
    var_10 = False
    var_11 = 'test_module.func'
    var_12 = bool('test_module.func' in var_0.doc)
    assert var_12 is True
    var_13 = '|'
    var_14 = bool('|' in var_0.doc['test_module.func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = []
    var_4 = 'b'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = 'a'
    var_16 = bool('a' in var_0.doc['test_module.func'])
    assert var_16 is True
    var_17 = 'b'
    var_18 = bool('b' in var_0.doc['test_module.func'])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 10
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '10'
    var_15 = bool('10' in var_0.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'val'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass.method'
    var_12 = True
    var_13 = False
    var_14 = 'test_module.MyClass.method'
    var_15 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_15 is True
    var_16 = 'Self'
    var_17 = bool('Self' in var_0.doc['test_module.MyClass.method'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'args'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '*args'
    var_15 = bool('*args' in var_0.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'kwargs'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '**kwargs'
    var_15 = bool('**kwargs' in var_0.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'int'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = 'return'
    var_16 = bool('return' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = []
    var_4 = 'val'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass.method'
    var_12 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 4/12 statements.
# Partially parsed test_imports_with_import_as. Retrieved 4/12 statements.
# Partially parsed test_imports_with_multiple_imports. Retrieved 5/14 statements.
# Partially parsed test_imports_with_from_import. Retrieved 6/14 statements.
# Partially parsed test_imports_with_from_import_as. Retrieved 6/14 statements.
# Partially parsed test_imports_with_relative_import_level_1. Retrieved 6/14 statements.
# Partially parsed test_imports_with_relative_import_level_2. Retrieved 6/14 statements.
# Partially parsed test_imports_with_from_import_multiple_names. Retrieved 7/16 statements.
# Partially parsed test_imports_with_from_import_none_module. Retrieved 5/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'mymodule'
    var_4 = var_0.alias['mymodule.os']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'mymodule'
    var_4 = var_0.alias['mymodule.operating_system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'sys'
    var_4 = 'mymodule'
    var_5 = var_0.alias['mymodule.os']
    assert var_5 == 'os'
    var_6 = var_0.alias['mymodule.sys']
    assert var_6 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = var_0.alias['mymodule.path']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'p'
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = var_0.alias['mymodule.p']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'utils'
    var_2 = 'helper'
    var_3 = None
    var_4 = 1
    var_5 = 'package.module'
    var_6 = var_0.alias['package.module.helper']
    assert var_6 == 'package.utils.helper'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'utils'
    var_2 = 'helper'
    var_3 = None
    var_4 = 2
    var_5 = 'package.subpackage.module'
    var_6 = var_0.alias['package.subpackage.module.helper']
    assert var_6 == 'utils.helper'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 'sep'
    var_5 = 0
    var_6 = 'mymodule'
    var_7 = var_0.alias['mymodule.path']
    assert var_7 == 'os.path'
    var_8 = var_0.alias['mymodule.sep']
    assert var_8 == 'os.sep'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = None
    var_2 = 'helper'
    var_3 = 1
    var_4 = 'package.module'
    var_5 = var_0.alias['package.module.helper']
    assert var_5 == 'package.helper'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 5/14 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_invalid_annotation_target. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'x: int = 5'
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.x']
    assert var_7 == '5'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'y = 10'
    var_5 = 'test_module.y'
    var_6 = bool('test_module.y' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.y']
    assert var_7 == '10'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'MAX_VALUE = 100'
    var_5 = 'test_module.MAX_VALUE'
    var_6 = bool('test_module.MAX_VALUE' in var_0.const)
    assert var_6 is True
    var_7 = 'test_module.MAX_VALUE'
    var_8 = bool('test_module.MAX_VALUE' in var_0.root)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = "__all__ = ('func1', 'func2')"
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = "__all__ = ['item1', 'item2']"
    var_5 = 'test_module.item1'
    var_6 = bool('test_module.item1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.item2'
    var_8 = bool('test_module.item2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'a = b = 5'
    var_5 = 'test_module.a'
    var_6 = bool('test_module.a' not in var_0.alias)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'z = 42  # type: int'
    var_5 = 'test_module.z'
    var_6 = bool('test_module.z' in var_0.const)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = '(a, b): tuple = (1, 2)'
    var_5 = 'test_module.a'
    var_6 = bool('test_module.a' not in var_0.alias)
    assert var_6 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_compile_magic_method_predicate. Retrieved 4/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.compile()
    var_4 = '__init__'
    var_5 = bool('__init__' not in var_3)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_element_with_single_constant. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_element_with_multiple_same_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_single_element_with_multiple_different_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_same_type. Retrieved 4/12 statements.
# Partially parsed test_e_type_multiple_elements_different_types. Retrieved 3/11 statements.
# Partially parsed test_e_type_with_none_element. Retrieved 1/3 statements.
# Partially parsed test_e_type_with_empty_sequence. Retrieved 1/3 statements.
# Partially parsed test_e_type_with_non_constant. Retrieved 2/7 statements.
# Partially parsed test_e_type_with_string_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_with_float_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_mixed_numeric_types. Retrieved 2/7 statements.
# Partially parsed test_e_type_with_boolean_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_with_mixed_types. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 100
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'string'
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 100
    var_3 = []
    var_4 = 5
    var_5 = []
    var_6 = 10
    var_7 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'string'
    var_3 = []
    var_4 = 3.14
    var_5 = []

def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []
    var_2 = 'world'
    var_3 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []
    var_2 = 2.71
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 3.14
    var_3 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = False
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 100
    var_3 = []
    var_4 = 'string'
    var_5 = []
    var_6 = 'text'
    var_7 = []



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_func_ann_arg_equals_star. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = None
    var_4 = []
    var_5 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'nested'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_e_type_with_elements. Retrieved 2/36 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc_enables_link. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 4/8 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_alias_replacement. Retrieved 7/11 statements.
# Partially parsed test_visit_name_with_typevar_alias. Retrieved 9/13 statements.
# Partially parsed test_visit_name_no_alias_match. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_root_prefix. Retrieved 7/11 statements.
# Partially parsed test_visit_name_circular_alias_prevention. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeType'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = []

import apimd.parser as module_0

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
    var_9 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'UnknownType'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.List'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'List'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.X'
    var_2 = 'X'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = []



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_walk_body_simple_statements. Retrieved 6/9 statements.
# Partially parsed test_walk_body_with_if_statement. Retrieved 8/11 statements.


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
    var_6 = 'targets'
    var_7 = 'value'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    if False:\n        x = 1\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

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
    var_0 = 'try:\n    x = 1\nfinally:\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    try:\n        x = 1\n    except:\n        y = 2\nz = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

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
    var_0 = 'pass'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'original_name'
    var_2 = 'renamed_name'
    var_3 = 'test_module'
    var_4 = 'test_module.renamed_name'
    var_5 = bool('test_module.renamed_name' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.renamed_name']
    assert var_6 == 'original_name'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_api_predicate_line_17_false. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'def test_func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_0]
    var_6 = 'test_module'
    var_7 = var_2.api(var_6, var_5)
    var_8 = '\n<a id="{}"></a>'
    var_9 = bool('\n<a id="{}"></a>' not in var_2.doc['test_module.test_func'])
    assert var_9 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 2/39 statements.


def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = 100
    var_3 = [var_2]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_globals_predicate_line_33_false. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 33 evaluates to False when const already has the name.'
    var_1 = module_0.Parser()
    var_2 = 'TEST_VAR'
    var_3 = None
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'test_module'
    var_8 = var_1.const['test_module.TEST_VAR']
    assert var_8 == 'str'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_func_api_basic. Retrieved 11/21 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 11/23 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/22 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/21 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/21 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/21 statements.
# Partially parsed test_func_api_classmethod. Retrieved 11/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = '|'
    var_16 = bool('|' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 42
    var_2 = []
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = []
    var_7 = 'y'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'test_module'
    var_12 = 'test_module.func'
    var_13 = False
    var_14 = 'test_module.func'
    var_15 = bool('test_module.func' in var_0.doc)
    assert var_15 is True
    var_16 = '42'
    var_17 = bool('42' in var_0.doc['test_module.func'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass.method'
    var_12 = True
    var_13 = False
    var_14 = 'test_module.MyClass.method'
    var_15 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_15 is True
    var_16 = 'Self'
    var_17 = bool('Self' in var_0.doc['test_module.MyClass.method'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'args'
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = '*args'
    var_16 = bool('*args' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'kwargs'
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = '**kwargs'
    var_16 = bool('**kwargs' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'int'
    var_2 = []
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = 'return'
    var_16 = bool('return' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass.method'
    var_12 = True
    var_13 = 'test_module.MyClass.method'
    var_14 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_14 is True
    var_15 = 'type[Self]'
    var_16 = bool('type[Self]' in var_0.doc['test_module.MyClass.method'])
    assert var_16 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_class_api_enums_predicate. Retrieved 12/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = '\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.resolve
    var_8 = 'enum.Enum'
    var_9 = var_6.bases
    var_10 = var_6.body
    var_11 = var_0.class_api(var_1, var_2, var_9, var_10)
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc[var_2])
    assert var_13 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_func_api_basic_function. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_self. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 6/16 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 7/17 statements.
# Partially parsed test_func_api_no_annotations. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(x: int, y: str) -> bool: pass'
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = False
    var_6 = False
    var_7 = 'test.foo'
    var_8 = bool('test.foo' in var_0.doc)
    assert var_8 is True
    var_9 = '| x |'
    var_10 = bool('| x |' in var_0.doc['test.foo'])
    assert var_10 is True
    var_11 = '| y |'
    var_12 = bool('| y |' in var_0.doc['test.foo'])
    assert var_12 is True
    var_13 = '| return |'
    var_14 = bool('| return |' in var_0.doc['test.foo'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "def foo(x: int = 5, y: str = 'hello') -> bool: pass"
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = False
    var_6 = False
    var_7 = 'test.foo'
    var_8 = bool('test.foo' in var_0.doc)
    assert var_8 is True
    var_9 = '|'
    var_10 = bool('|' in var_0.doc['test.foo'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(x: int, *args, **kwargs) -> None: pass'
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = False
    var_6 = False
    var_7 = 'test.foo'
    var_8 = bool('test.foo' in var_0.doc)
    assert var_8 is True
    var_9 = '*args'
    var_10 = bool('*args' in var_0.doc['test.foo'])
    assert var_10 is True
    var_11 = '**kwargs'
    var_12 = bool('**kwargs' in var_0.doc['test.foo'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(self, x: int) -> str: pass'
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = True
    var_6 = False
    var_7 = 'test.foo'
    var_8 = bool('test.foo' in var_0.doc)
    assert var_8 is True
    var_9 = 'Self'
    var_10 = bool('Self' in var_0.doc['test.foo'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(cls, x: int) -> str: pass'
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = True
    var_6 = 'test.foo'
    var_7 = bool('test.foo' in var_0.doc)
    assert var_7 is True
    var_8 = 'type[Self]'
    var_9 = bool('type[Self]' in var_0.doc['test.foo'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(x: int, *, y: str) -> None: pass'
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = False
    var_6 = False
    var_7 = 'test.foo'
    var_8 = bool('test.foo' in var_0.doc)
    assert var_8 is True
    var_9 = '| x |'
    var_10 = bool('| x |' in var_0.doc['test.foo'])
    assert var_10 is True
    var_11 = '| y |'
    var_12 = bool('| y |' in var_0.doc['test.foo'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(x, y): pass'
    var_2 = 0
    var_3 = 'test'
    var_4 = 'test.foo'
    var_5 = None
    var_6 = False
    var_7 = False
    var_8 = 'test.foo'
    var_9 = bool('test.foo' in var_0.doc)
    assert var_9 is True
    var_10 = '| x |'
    var_11 = bool('| x |' in var_0.doc['test.foo'])
    assert var_11 is True
    var_12 = '| y |'
    var_13 = bool('| y |' in var_0.doc['test.foo'])
    assert var_13 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 8/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '__all__'
    var_4 = None
    var_5 = []
    var_6 = 123
    var_7 = []
    var_8 = var_1.imp[var_2]
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'p'
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = 'mymodule.p'
    var_7 = bool('mymodule.p' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['mymodule.p']
    assert var_8 == 'os.path'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_func_api_vararg_not_none. Retrieved 15/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'args'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.test_func'
    var_10 = False
    var_11 = 'test_module.test_func'
    var_12 = bool('test_module.test_func' in var_0.doc)
    assert var_12 is True
    var_13 = var_0.doc[var_9]
    var_14 = len(var_13)
    var_15 = 'Test function\n\n'
    var_16 = len(var_15)
    var_17 = bool(var_14 > var_16)
    assert var_17 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_true. Retrieved 11/27 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to True.'
    var_1 = 'mymodule'
    var_2 = 'mymodule.MyType'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = 'MyType'
    var_7 = []
    var_8 = 'mymodule.MyType'
    var_9 = var_8 in var_0
    var_10 = var_8 not in var_2
    var_11 = var_9 and var_10
    assert var_11 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_func_api_predicate_line_32_false. Retrieved 8/21 statements.


def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = [var_2, var_2]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_tuple_of_strings. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_set_of_floats. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_dict_int_str. Retrieved 4/13 statements.
# Partially parsed test_const_type_with_mixed_types_in_list. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_non_constant_element. Retrieved 1/6 statements.
# Partially parsed test_const_type_with_call_to_int. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_str. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_list. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_unknown_node. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []

def test_case_0():
    var_0 = 1.0
    var_1 = []
    var_2 = 2.0
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 'a'
    var_5 = []
    var_6 = 'b'
    var_7 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'list'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'unknown_var'
    var_1 = []



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_evaluates_to_true. Retrieved 11/28 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to True.'
    var_1 = 'mymodule'
    var_2 = 'mymodule.MyType'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = '_m'
    var_7 = None
    var_8 = 'MyType'
    var_9 = []
    var_10 = []
    var_11 = '_m'
    var_12 = '_m'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_class_api_mem_predicate_true. Retrieved 8/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'public_attr: str'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body
    var_6 = []
    var_7 = var_0.class_api(var_1, var_2, var_6, var_5)
    var_8 = 'Members'
    var_9 = bool('Members' in var_0.doc[var_2])
    assert var_9 is True
    var_10 = 'Type'
    var_11 = bool('Type' in var_0.doc[var_2])
    assert var_11 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 14/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'test_module'
    var_8 = 'test_module.test_func'
    var_9 = False
    var_10 = 'test_module.test_func'
    var_11 = bool('test_module.test_func' in var_0.doc)
    assert var_11 is True
    var_12 = var_0.doc[var_8]
    var_13 = len(var_12)
    var_14 = 'Test function\n'
    var_15 = len(var_14)
    var_16 = bool(var_13 > var_15)
    assert var_16 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_api_has_self_predicate_true. Retrieved 14/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_func'
    var_4 = []
    var_5 = 'self'
    var_6 = None
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'test_root'
    var_14 = 'TestClass'
    var_15 = 'test_root.TestClass.test_func'
    var_16 = bool('test_root.TestClass.test_func' in var_2.doc)
    assert var_16 is True
    var_17 = var_2.root['test_root.TestClass.test_func']
    assert var_17 == 'test_root'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_link_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = True
    var_1 = 3
    var_2 = False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_func_api_kwarg_not_none. Retrieved 11/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'kwargs'
    var_8 = 'test_module'
    var_9 = 'test_module.test_func'
    var_10 = False
    var_11 = '**kwargs'
    var_12 = bool('**kwargs' in var_0.doc['test_module.test_func'])
    assert var_12 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_public_family_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_private_family_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_matching. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_not_matching. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_submodule_in_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_module_as_import. Retrieved 3/6 statements.
# Partially parsed test_is_public_with_magic_name_no_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_empty_all_and_public_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_empty_all_and_private_name. Retrieved 3/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.public_func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule._private_func'
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
    var_1 = 'mymodule.other_func'
    var_2 = 'mymodule.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.submodule'
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
    var_1 = 'mymodule.func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule._private'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_visit_subscript_union_with_tuple. Retrieved 6/22 statements.
# Partially parsed test_visit_subscript_union_without_tuple. Retrieved 5/13 statements.
# Partially parsed test_visit_subscript_optional. Retrieved 5/16 statements.
# Partially parsed test_visit_subscript_pep585_deprecated. Retrieved 6/21 statements.
# Partially parsed test_visit_subscript_non_name_value. Retrieved 6/15 statements.
# Partially parsed test_visit_subscript_unknown_name. Retrieved 5/12 statements.
# Partially parsed test_visit_subscript_union_with_alias. Retrieved 8/21 statements.
# Partially parsed test_visit_subscript_optional_with_alias. Retrieved 7/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 'str'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Optional'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = []
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Dict'
    var_4 = []
    var_5 = 'str'
    var_6 = []
    var_7 = 'int'
    var_8 = []
    var_9 = []
    var_10 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Union'
    var_6 = []
    var_7 = 'int'
    var_8 = []
    var_9 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Unknown'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.Union'
    var_2 = 'typing.Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Union'
    var_6 = []
    var_7 = 'int'
    var_8 = []
    var_9 = 'str'
    var_10 = []
    var_11 = []
    var_12 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.Optional'
    var_2 = 'typing.Optional'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Optional'
    var_6 = []
    var_7 = 'str'
    var_8 = []
    var_9 = []
    var_10 = []



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_globals_predicate_line_8_false. Retrieved 9/44 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 42
    var_2 = []
    var_3 = 'test_module'
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {})
    assert var_5 is True
    var_6 = 'x'
    var_7 = []
    var_8 = 'y'
    var_9 = []
    var_10 = None
    var_11 = 'int'
    var_12 = []
    var_13 = 1
    var_14 = []
    var_15 = 0
    var_16 = var_0.alias
    var_17 = bool(var_0.alias == {})
    assert var_17 is True
    var_18 = []
    var_19 = []
    var_20 = var_0.alias
    var_21 = bool(var_0.alias == {})
    assert var_21 is True
    var_22 = []
    var_23 = var_0.alias
    var_24 = bool(var_0.alias == {})
    assert var_24 is True
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = var_0.alias
    var_29 = bool(var_0.alias == {})
    assert var_29 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/24 statements.
# Partially parsed test_class_api_with_bases. Retrieved 5/11 statements.
# Partially parsed test_class_api_with_enum. Retrieved 11/30 statements.
# Partially parsed test_class_api_with_delete. Retrieved 7/20 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 10/24 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/8 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'attr2'
    var_8 = 42
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.TestClass'
    var_12 = 'test_module.TestClass'
    var_13 = bool('test_module.TestClass' in var_0.doc)
    assert var_13 is True
    var_14 = 'Members'
    var_15 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'MEMBER1'
    var_6 = 'str'
    var_7 = []
    var_8 = None
    var_9 = 1
    var_10 = 'MEMBER2'
    var_11 = 'value'
    var_12 = []
    var_13 = 'test_module'
    var_14 = 'test_module.TestEnum'
    var_15 = 'test_module.TestEnum'
    var_16 = bool('test_module.TestEnum' in var_0.doc)
    assert var_16 is True
    var_17 = 'Enums'
    var_18 = bool('Enums' in var_0.doc['test_module.TestEnum'])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 1
    var_4 = []
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'str'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'public'
    var_8 = 42
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.TestClass'
    var_12 = 'test_module.TestClass'
    var_13 = bool('test_module.TestClass' in var_0.doc)
    assert var_13 is True
    var_14 = 'public'
    var_15 = bool('public' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr'
    var_3 = 10
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'int'
    var_11 = bool('int' in var_0.doc['test_module.TestClass'])
    assert var_11 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_func_ann_star_argument. Retrieved 5/40 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = '_Self'
    var_1 = 'Parser'
    var_2 = []
    var_3 = module_0.Parser()
    var_4 = '*'
    var_5 = None
    var_6 = []
    var_7 = 'test_root'
    var_8 = False



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_globals_predicate_line_35_evaluates_to_false. Retrieved 12/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "Test that the predicate at line 35 evaluates to False.\n    \n    The predicate is: left.id != '__all__' or not isinstance(node.value, (Tuple, List))\n    For this to be False, both conditions must be False:\n    - left.id == '__all__' (first part is False)\n    - isinstance(node.value, (Tuple, List)) (second part is False, so 'not' makes it False)\n    "
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = "__all__ = ('func1', 'func2')"
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_6.targets
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_6.targets[var_5]
    var_10 = var_6.targets[0].id
    assert var_10 == '__all__'
    var_11 = var_6.value
    var_12 = var_1.globals(var_2, var_6)
    var_13 = 'test_module.func1'
    var_14 = bool('test_module.func1' in var_1.imp[var_2])
    assert var_14 is True
    var_15 = 'test_module.func2'
    var_16 = bool('test_module.func2' in var_1.imp[var_2])
    assert var_16 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_class_api_delete_node_with_non_name_target. Retrieved 3/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = []
    var_3 = 'attr'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_parser_new_class_method. Retrieved 3/4 statements.
# Partially parsed test_parser_new_with_toc_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.b_level
    assert var_4 == 1
    var_5 = var_2.toc
    assert var_5 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_func_api_with_defaults. Retrieved 8/17 statements.
# Partially parsed test_func_api_with_self. Retrieved 7/16 statements.
# Partially parsed test_func_api_classmethod. Retrieved 6/15 statements.
# Partially parsed test_func_api_no_args. Retrieved 8/17 statements.
# Partially parsed test_func_api_posonly_args. Retrieved 8/17 statements.
# Partially parsed test_func_api_kwonly_args. Retrieved 8/17 statements.
# Partially parsed test_func_api_varargs_and_kwargs. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = "def func(a: int, b: str = 'default', *args, c: float = 1.0, **kwargs) -> bool: pass"
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.func'
    var_6 = False
    var_7 = False
    var_8 = 'test_module.func'
    var_9 = bool('test_module.func' in var_1.doc)
    assert var_9 is True
    var_10 = '|'
    var_11 = bool('|' in var_1.doc['test_module.func'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def method(self, x: int) -> str: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.MyClass.method'
    var_6 = False
    var_7 = 'test_module.MyClass.method'
    var_8 = bool('test_module.MyClass.method' in var_1.doc)
    assert var_8 is True
    var_9 = 'Self'
    var_10 = bool('Self' in var_1.doc['test_module.MyClass.method'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def create(cls, value: int): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.MyClass.create'
    var_6 = 'test_module.MyClass.create'
    var_7 = bool('test_module.MyClass.create' in var_1.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def simple() -> None: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.simple'
    var_6 = False
    var_7 = False
    var_8 = 'test_module.simple'
    var_9 = bool('test_module.simple' in var_1.doc)
    assert var_9 is True
    var_10 = 'return'
    var_11 = bool('return' in var_1.doc['test_module.simple'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def func(a: int, /, b: str) -> bool: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.func'
    var_6 = False
    var_7 = False
    var_8 = 'test_module.func'
    var_9 = bool('test_module.func' in var_1.doc)
    assert var_9 is True
    var_10 = '|'
    var_11 = bool('|' in var_1.doc['test_module.func'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def func(*, key: str, value: int) -> None: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.func'
    var_6 = False
    var_7 = False
    var_8 = 'test_module.func'
    var_9 = bool('test_module.func' in var_1.doc)
    assert var_9 is True
    var_10 = '*'
    var_11 = bool('*' in var_1.doc['test_module.func'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = 'def func(*args: str, **kwargs: int) -> None: pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.func'
    var_6 = False
    var_7 = False
    var_8 = 'test_module.func'
    var_9 = bool('test_module.func' in var_1.doc)
    assert var_9 is True
    var_10 = '*args'
    var_11 = bool('*args' in var_1.doc['test_module.func'])
    assert var_11 is True
    var_12 = '**kwargs'
    var_13 = bool('**kwargs' in var_1.doc['test_module.func'])
    assert var_13 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_api_function_def. Retrieved 8/13 statements.
# Partially parsed test_api_async_function_def. Retrieved 8/13 statements.
# Partially parsed test_api_class_def. Retrieved 8/13 statements.
# Partially parsed test_api_with_decorator. Retrieved 8/13 statements.
# Partially parsed test_api_with_prefix. Retrieved 9/14 statements.
# Partially parsed test_api_with_link_false. Retrieved 8/13 statements.
# Partially parsed test_api_nested_class. Retrieved 11/16 statements.
# Partially parsed test_api_underscore_escaping. Retrieved 8/13 statements.
# Partially parsed test_api_sets_level. Retrieved 9/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\ndef example_func():\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = 'test_module.example_func'
    var_9 = bool('test_module.example_func' in var_1.doc)
    assert var_9 is True
    var_10 = '## example_func()'
    var_11 = bool('## example_func()' in var_1.doc['test_module.example_func'])
    assert var_11 is True
    var_12 = '*Full name:* `test_module.example_func`'
    var_13 = bool('*Full name:* `test_module.example_func`' in var_1.doc['test_module.example_func'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\nasync def async_func():\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = 'test_module.async_func'
    var_9 = bool('test_module.async_func' in var_1.doc)
    assert var_9 is True
    var_10 = bool('async example_func()' in var_1.doc['test_module.async_func'] or 'async' in var_1.doc['test_module.async_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\nclass ExampleClass:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = 'test_module.ExampleClass'
    var_9 = bool('test_module.ExampleClass' in var_1.doc)
    assert var_9 is True
    var_10 = 'class ExampleClass'
    var_11 = bool('class ExampleClass' in var_1.doc['test_module.ExampleClass'])
    assert var_11 is True
    var_12 = '*Full name:* `test_module.ExampleClass`'
    var_13 = bool('*Full name:* `test_module.ExampleClass`' in var_1.doc['test_module.ExampleClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\n@property\ndef decorated_func():\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = 'test_module.decorated_func'
    var_9 = bool('test_module.decorated_func' in var_1.doc)
    assert var_9 is True
    var_10 = 'Decorators'
    var_11 = bool('Decorators' in var_1.doc['test_module.decorated_func'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\ndef method():\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'TestClass'
    var_8 = var_1.api(var_6, var_5, prefix=var_7)
    var_9 = 'test_module.TestClass.method'
    var_10 = bool('test_module.TestClass.method' in var_1.doc)
    assert var_10 is True
    var_11 = '### method()'
    var_12 = bool('### method()' in var_1.doc['test_module.TestClass.method'])
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1)
    var_3 = '\ndef func_no_link():\n    pass\n'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_0]
    var_6 = 'test_module'
    var_7 = var_2.api(var_6, var_5)
    var_8 = 'test_module.func_no_link'
    var_9 = bool('test_module.func_no_link' in var_2.doc)
    assert var_9 is True
    var_10 = '<a id='
    var_11 = bool('<a id=' not in var_2.doc['test_module.func_no_link'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\nclass OuterClass:\n    class InnerClass:\n        pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_5.body[var_4]
    var_7 = 'test_module'
    var_8 = var_1.api(var_7, var_5)
    var_9 = 'OuterClass'
    var_10 = var_1.api(var_7, var_6, prefix=var_9)
    var_11 = 'test_module.OuterClass'
    var_12 = bool('test_module.OuterClass' in var_1.doc)
    assert var_12 is True
    var_13 = 'test_module.OuterClass.InnerClass'
    var_14 = bool('test_module.OuterClass.InnerClass' in var_1.doc)
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\ndef func_with_underscores():\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = 'test_module.func_with_underscores'
    var_9 = bool('test_module.func_with_underscores' in var_1.doc)
    assert var_9 is True
    var_10 = 'func\\_with\\_underscores'
    var_11 = bool('func\\_with\\_underscores' in var_1.doc['test_module.func_with_underscores'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = module_0.Parser(var_0, var_1)
    var_3 = '\ndef example():\n    pass\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module'
    var_8 = var_2.api(var_7, var_6)
    var_9 = var_2.level['test_module.example']
    assert var_9 == 1



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 11/28 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/32 statements.
# Partially parsed test_class_api_with_delete. Retrieved 8/23 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 11/28 statements.
# Partially parsed test_class_api_with_assign_members. Retrieved 7/17 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'MyClass'
    var_6 = 'MyClass'
    var_7 = bool('MyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['MyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = []
    var_5 = 'test'
    var_6 = []
    var_7 = 1
    var_8 = 'attr2'
    var_9 = 'int'
    var_10 = []
    var_11 = 42
    var_12 = []
    var_13 = 'test_module'
    var_14 = 'MyClass'
    var_15 = 'MyClass'
    var_16 = bool('MyClass' in var_0.doc)
    assert var_16 is True
    var_17 = 'attr1'
    var_18 = bool('attr1' in var_0.doc['MyClass'])
    assert var_18 is True
    var_19 = 'attr2'
    var_20 = bool('attr2' in var_0.doc['MyClass'])
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'MEMBER1'
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'MEMBER2'
    var_11 = []
    var_12 = 2
    var_13 = []
    var_14 = 'test_module'
    var_15 = 'MyEnum'
    var_16 = 'MyEnum'
    var_17 = bool('MyEnum' in var_0.doc)
    assert var_17 is True
    var_18 = 'MEMBER1'
    var_19 = bool('MEMBER1' in var_0.doc['MyEnum'])
    assert var_19 is True
    var_20 = 'MEMBER2'
    var_21 = bool('MEMBER2' in var_0.doc['MyEnum'])
    assert var_21 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = []
    var_5 = 'test'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'MyClass'
    var_10 = 'MyClass'
    var_11 = bool('MyClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'str'
    var_4 = []
    var_5 = 'test'
    var_6 = []
    var_7 = 1
    var_8 = 'public'
    var_9 = 'int'
    var_10 = []
    var_11 = 42
    var_12 = []
    var_13 = 'test_module'
    var_14 = 'MyClass'
    var_15 = 'MyClass'
    var_16 = bool('MyClass' in var_0.doc)
    assert var_16 is True
    var_17 = 'public'
    var_18 = bool('public' in var_0.doc['MyClass'])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'test'
    var_4 = []
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'MyClass'
    var_8 = 'MyClass'
    var_9 = bool('MyClass' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'EmptyClass'
    var_7 = bool('EmptyClass' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_api_function_def. Retrieved 7/10 statements.
# Partially parsed test_api_async_function_def. Retrieved 7/10 statements.
# Partially parsed test_api_class_def. Retrieved 7/10 statements.
# Partially parsed test_api_with_decorators. Retrieved 7/11 statements.
# Partially parsed test_api_with_prefix. Retrieved 8/11 statements.
# Partially parsed test_api_with_docstring. Retrieved 7/10 statements.
# Partially parsed test_api_nested_class_methods. Retrieved 7/10 statements.
# Partially parsed test_api_with_link_enabled. Retrieved 8/11 statements.
# Partially parsed test_api_underscore_escaping. Retrieved 7/10 statements.
# Partially parsed test_api_classmethod. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def example_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.example_func'
    var_8 = bool('test_module.example_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'example_func()'
    var_10 = bool('example_func()' in var_0.doc['test_module.example_func'])
    assert var_10 is True
    var_11 = var_0.root['test_module.example_func']
    assert var_11 == 'test_module'
    var_12 = var_0.level['test_module.example_func']
    assert var_12 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.async_func'
    var_8 = bool('test_module.async_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'async async_func()'
    var_10 = bool('async async_func()' in var_0.doc['test_module.async_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class ExampleClass: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.ExampleClass'
    var_8 = bool('test_module.ExampleClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'class ExampleClass'
    var_10 = bool('class ExampleClass' in var_0.doc['test_module.ExampleClass'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@staticmethod\ndef decorated_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.decorated_func'
    var_8 = bool('test_module.decorated_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'Decorators'
    var_10 = bool('Decorators' in var_0.doc['test_module.decorated_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def method_func(self): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'TestClass'
    var_7 = var_0.api(var_5, var_4, prefix=var_6)
    var_8 = 'test_module.TestClass.method_func'
    var_9 = bool('test_module.TestClass.method_func' in var_0.doc)
    assert var_9 is True
    var_10 = var_0.root['test_module.TestClass.method_func']
    assert var_10 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func_with_doc():\n    """This is a docstring."""\n    pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.func_with_doc'
    var_8 = bool('test_module.func_with_doc' in var_0.docstring)
    assert var_8 is True
    var_9 = 'This is a docstring.'
    var_10 = bool('This is a docstring.' in var_0.docstring['test_module.func_with_doc'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class OuterClass:\n    def inner_method(self): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.OuterClass'
    var_8 = bool('test_module.OuterClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'test_module.OuterClass.inner_method'
    var_10 = bool('test_module.OuterClass.inner_method' in var_0.doc)
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'def func(): pass'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = '<a id='
    var_9 = bool('<a id=' in var_1.doc['test_module.func'])
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func_with_underscores(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'func\\_with\\_underscores()'
    var_8 = bool('func\\_with\\_underscores()' in var_0.doc['test_module.func_with_underscores'])
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@classmethod\ndef class_method(cls): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'TestClass'
    var_7 = var_0.api(var_5, var_4, prefix=var_6)
    var_8 = 'test_module.TestClass.class_method'
    var_9 = bool('test_module.TestClass.class_method' in var_0.doc)
    assert var_9 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_class_api_line_15_predicate_false. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_class'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = None
    var_6 = []
    var_7 = 42
    var_8 = []
    var_9 = 1
    var_10 = bool('_private_attr' not in var_0.doc[var_2] or var_0.doc[var_2] == '')
    assert var_10 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/16 statements.
# Partially parsed test_class_api_with_enums. Retrieved 8/20 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 8/19 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 8/16 statements.
# Partially parsed test_class_api_with_multiple_bases. Retrieved 6/13 statements.
# Partially parsed test_class_api_with_assigned_member. Retrieved 7/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.MyClass'
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['test_module.MyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'
    var_10 = 'test_module.MyClass'
    var_11 = bool('test_module.MyClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'member1'
    var_13 = bool('member1' in var_0.doc['test_module.MyClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = 'ENUM_VALUE'
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyEnum'
    var_12 = 'test_module.MyEnum'
    var_13 = bool('test_module.MyEnum' in var_0.doc)
    assert var_13 is True
    var_14 = 'ENUM_VALUE'
    var_15 = bool('ENUM_VALUE' in var_0.doc['test_module.MyEnum'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = None
    var_7 = 1
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'
    var_11 = 'test_module.MyClass'
    var_12 = bool('test_module.MyClass' in var_0.doc)
    assert var_12 is True
    var_13 = 'member1'
    var_14 = bool('member1' not in var_0.doc['test_module.MyClass'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.MyClass'
    var_10 = 'test_module.MyClass'
    var_11 = bool('test_module.MyClass' in var_0.doc)
    assert var_11 is True
    var_12 = '_private'
    var_13 = bool('_private' not in var_0.doc['test_module.MyClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = []
    var_3 = 'Base2'
    var_4 = []
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Base1'
    var_11 = bool('Base1' in var_0.doc['test_module.MyClass'])
    assert var_11 is True
    var_12 = 'Base2'
    var_13 = bool('Base2' in var_0.doc['test_module.MyClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = None
    var_7 = 'test_module'
    var_8 = 'test_module.MyClass'
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'member1'
    var_12 = bool('member1' in var_0.doc['test_module.MyClass'])
    assert var_12 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/17 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 8/17 statements.
# Partially parsed test_class_api_with_enum. Retrieved 8/20 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 8/20 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.MyClass'
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['test_module.MyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 10
    var_7 = []
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'
    var_11 = 'test_module.MyClass'
    var_12 = bool('test_module.MyClass' in var_0.doc)
    assert var_12 is True
    var_13 = 'member1'
    var_14 = bool('member1' in var_0.doc['test_module.MyClass'])
    assert var_14 is True
    var_15 = 'int'
    var_16 = bool('int' in var_0.doc['test_module.MyClass'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = 'test'
    var_7 = []
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'
    var_11 = 'test_module.MyClass'
    var_12 = bool('test_module.MyClass' in var_0.doc)
    assert var_12 is True
    var_13 = '_private'
    var_14 = bool('_private' not in var_0.doc['test_module.MyClass'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = 'MEMBER'
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyEnum'
    var_12 = 'test_module.MyEnum'
    var_13 = bool('test_module.MyEnum' in var_0.doc)
    assert var_13 is True
    var_14 = 'Enums'
    var_15 = bool('Enums' in var_0.doc['test_module.MyEnum'])
    assert var_15 is True
    var_16 = 'MEMBER'
    var_17 = bool('MEMBER' in var_0.doc['test_module.MyEnum'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 10
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass'
    var_12 = 'test_module.MyClass'
    var_13 = bool('test_module.MyClass' in var_0.doc)
    assert var_13 is True
    var_14 = 'member1'
    var_15 = bool('member1' not in var_0.doc['test_module.MyClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_func_ann_with_self_argument. Retrieved 15/25 statements.
# Partially parsed test_func_ann_without_self. Retrieved 14/24 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 14/24 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 17/28 statements.
# Partially parsed test_func_ann_without_annotation. Retrieved 12/20 statements.
# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 10/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ast'
    var_2 = __import__(var_1)
    var_3 = 'self'
    var_4 = __import__(var_1)
    var_5 = 'MyClass'
    var_6 = __import__(var_1)
    var_7 = __import__(var_1)
    var_8 = 'x'
    var_9 = __import__(var_1)
    var_10 = 'int'
    var_11 = __import__(var_1)
    var_12 = 'test_module'
    var_13 = True
    var_14 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ast'
    var_2 = __import__(var_1)
    var_3 = 'x'
    var_4 = __import__(var_1)
    var_5 = 'int'
    var_6 = __import__(var_1)
    var_7 = __import__(var_1)
    var_8 = 'y'
    var_9 = __import__(var_1)
    var_10 = 'str'
    var_11 = __import__(var_1)
    var_12 = 'test_module'
    var_13 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ast'
    var_2 = __import__(var_1)
    var_3 = 'cls'
    var_4 = __import__(var_1)
    var_5 = 'type[MyClass]'
    var_6 = __import__(var_1)
    var_7 = __import__(var_1)
    var_8 = 'x'
    var_9 = __import__(var_1)
    var_10 = 'int'
    var_11 = __import__(var_1)
    var_12 = 'test_module'
    var_13 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ast'
    var_2 = __import__(var_1)
    var_3 = 'x'
    var_4 = __import__(var_1)
    var_5 = 'int'
    var_6 = __import__(var_1)
    var_7 = __import__(var_1)
    var_8 = '*'
    var_9 = None
    var_10 = __import__(var_1)
    var_11 = 'y'
    var_12 = __import__(var_1)
    var_13 = 'str'
    var_14 = __import__(var_1)
    var_15 = 'test_module'
    var_16 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ast'
    var_2 = __import__(var_1)
    var_3 = 'x'
    var_4 = None
    var_5 = __import__(var_1)
    var_6 = 'y'
    var_7 = __import__(var_1)
    var_8 = 'int'
    var_9 = __import__(var_1)
    var_10 = 'test_module'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ast'
    var_2 = __import__(var_1)
    var_3 = 'self'
    var_4 = __import__(var_1)
    var_5 = 'MyClass'
    var_6 = __import__(var_1)
    var_7 = 'test_module'
    var_8 = True
    var_9 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_visit_subscript_returns_node_when_value_not_name. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = []



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_parser_new_class_method. Retrieved 3/4 statements.
# Partially parsed test_parser_new_with_different_parameters. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = True
    var_1 = 3
    var_2 = False



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_api_function_def. Retrieved 7/12 statements.
# Partially parsed test_api_async_function_def. Retrieved 7/12 statements.
# Partially parsed test_api_class_def. Retrieved 7/12 statements.
# Partially parsed test_api_with_prefix. Retrieved 8/13 statements.
# Partially parsed test_api_with_decorators. Retrieved 7/12 statements.
# Partially parsed test_api_with_docstring. Retrieved 7/12 statements.
# Partially parsed test_api_class_with_nested_methods. Retrieved 7/12 statements.
# Partially parsed test_api_sets_level_correctly. Retrieved 7/12 statements.
# Partially parsed test_api_sets_root_correctly. Retrieved 7/12 statements.
# Partially parsed test_api_with_underscore_name. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def example_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.example_func'
    var_8 = bool('test_module.example_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'example_func()'
    var_10 = bool('example_func()' in var_0.doc['test_module.example_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.async_func'
    var_8 = bool('test_module.async_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'async async_func()'
    var_10 = bool('async async_func()' in var_0.doc['test_module.async_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class ExampleClass: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.ExampleClass'
    var_8 = bool('test_module.ExampleClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'class ExampleClass'
    var_10 = bool('class ExampleClass' in var_0.doc['test_module.ExampleClass'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def inner_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'OuterClass'
    var_7 = var_0.api(var_5, var_4, prefix=var_6)
    var_8 = 'test_module.OuterClass.inner_func'
    var_9 = bool('test_module.OuterClass.inner_func' in var_0.doc)
    assert var_9 is True
    var_10 = var_0.level['test_module.OuterClass.inner_func']
    assert var_10 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@property\ndef prop_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.prop_func'
    var_8 = bool('test_module.prop_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'Decorators'
    var_10 = bool('Decorators' in var_0.doc['test_module.prop_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def documented_func():\n    """This is a docstring."""\n    pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.documented_func'
    var_8 = bool('test_module.documented_func' in var_0.docstring)
    assert var_8 is True
    var_9 = 'This is a docstring.'
    var_10 = bool('This is a docstring.' in var_0.docstring['test_module.documented_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class TestClass:\n    def method1(self): pass\n    def method2(self): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.TestClass'
    var_8 = bool('test_module.TestClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'test_module.TestClass.method1'
    var_10 = bool('test_module.TestClass.method1' in var_0.doc)
    assert var_10 is True
    var_11 = 'test_module.TestClass.method2'
    var_12 = bool('test_module.TestClass.method2' in var_0.doc)
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = var_0.level['test_module.func']
    assert var_7 == 2

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = var_0.root['test_module.func']
    assert var_7 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func_with_underscores(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.func_with_underscores'
    var_8 = bool('test_module.func_with_underscores' in var_0.doc)
    assert var_8 is True
    var_9 = 'func\\_with\\_underscores()'
    var_10 = bool('func\\_with\\_underscores()' in var_0.doc['test_module.func_with_underscores'])
    assert var_10 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_bases. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_enums. Retrieved 10/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 10/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True
    var_14 = 'public_attr'
    var_15 = bool('public_attr' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass BaseClass:\n    pass\n\nclass DerivedClass(BaseClass):\n    pass\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 1
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.DerivedClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.DerivedClass'
    var_11 = bool('test_module.DerivedClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Bases'
    var_13 = bool('Bases' in var_0.doc['test_module.DerivedClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nfrom enum import Enum\n\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 1
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.Color'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.Color'
    var_11 = bool('test_module.Color' in var_0.doc)
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_13 is True
    var_14 = 'RED'
    var_15 = bool('RED' in var_0.doc['test_module.Color'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class EmptyClass:\n    pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.EmptyClass'
    var_7 = []
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.EmptyClass'
    var_11 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr1\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'attr1'
    var_13 = bool('attr1' not in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    value = 42  # type: int\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_class_api_line_25_predicate_false. Retrieved 9/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = None
    var_6 = []
    var_7 = 42
    var_8 = []
    var_9 = var_0.doc[var_2]
    var_10 = 0



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_class_api_enum_predicate_true. Retrieved 8/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum.Enum'
    var_2 = []
    var_3 = 'MEMBER1'
    var_4 = None
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.TestEnum'
    var_12 = bool('MEMBER1' in var_0.doc['test_module.TestEnum'] or True)
    assert var_12 is True



# Parsed testcases at query #76
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.path.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.__name__.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public._private.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.join._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.__magic__.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__.__name__.__doc__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.__path__.join.__doc__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public..join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'collections.abc.Mapping'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'collections._abc.Mapping'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_func_ann_has_self_and_first_arg. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'self'
    var_4 = 'int'
    var_5 = []
    var_6 = 'x'
    var_7 = None
    var_8 = []
    var_9 = 'test_module'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_func_api_basic_function. Retrieved 12/22 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 12/24 statements.
# Partially parsed test_func_api_with_self. Retrieved 13/23 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 12/22 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 12/22 statements.
# Partially parsed test_func_api_with_return_type. Retrieved 12/22 statements.
# Partially parsed test_func_api_classmethod. Retrieved 12/22 statements.
# Partially parsed test_func_api_kwonly_args. Retrieved 2/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with a basic function.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_1.doc)
    assert var_13 is True
    var_14 = '|'
    var_15 = bool('|' in var_1.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with default arguments.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = 10
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_1.doc)
    assert var_14 is True
    var_15 = 'x'
    var_16 = bool('x' in var_1.doc['test_module.func'])
    assert var_16 is True
    var_17 = 'y'
    var_18 = bool('y' in var_1.doc['test_module.func'])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with self parameter (instance method).'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'self'
    var_4 = None
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass.method'
    var_11 = True
    var_12 = False
    var_13 = 'test_module.MyClass.method'
    var_14 = bool('test_module.MyClass.method' in var_1.doc)
    assert var_14 is True
    var_15 = 'Self'
    var_16 = bool('Self' in var_1.doc['test_module.MyClass.method'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with *args.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = 'args'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_1.doc)
    assert var_13 is True
    var_14 = '*args'
    var_15 = bool('*args' in var_1.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with **kwargs.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = 'kwargs'
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_1.doc)
    assert var_13 is True
    var_14 = '**kwargs'
    var_15 = bool('**kwargs' in var_1.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with return type annotation.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'int'
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_1.doc)
    assert var_14 is True
    var_15 = 'return'
    var_16 = bool('return' in var_1.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with classmethod decorator.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'cls'
    var_4 = None
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass.method'
    var_11 = True
    var_12 = 'test_module.MyClass.method'
    var_13 = bool('test_module.MyClass.method' in var_1.doc)
    assert var_13 is True
    var_14 = 'type[Self]'
    var_15 = bool('type[Self]' in var_1.doc['test_module.MyClass.method'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with keyword-only arguments.'
    var_1 = module_0.Parser()



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_imports_asname_not_none. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'original_name'
    var_2 = 'renamed_name'
    var_3 = 'some_module'
    var_4 = 0
    var_5 = 'test_root'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_class_api_delete_non_name_target. Retrieved 7/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = None
    var_3 = []
    var_4 = 'attr'
    var_5 = 'test_module'
    var_6 = 'test_class'
    var_7 = []
    var_8 = var_0.doc['test_class']
    assert var_8 == ''



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/9 statements.
# Partially parsed test_imports_simple_import_with_asname. Retrieved 4/9 statements.
# Partially parsed test_imports_multiple_names. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_absolute. Retrieved 6/11 statements.
# Partially parsed test_imports_from_import_with_asname. Retrieved 6/11 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/11 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/11 statements.
# Partially parsed test_imports_from_import_no_module. Retrieved 5/10 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 8/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None
    var_4 = var_0.alias['mymodule.os']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = var_0.alias['mymodule.operating_system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = var_0.alias['mymodule.os']
    assert var_6 == 'os'
    var_7 = var_0.alias['mymodule.system']
    assert var_7 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['mymodule.path']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'p'
    var_5 = 0
    var_6 = var_0.alias['mymodule.p']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.mymodule'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['package.mymodule.func']
    assert var_6 == 'package.other.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.subpackage.mymodule'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 2
    var_6 = var_0.alias['package.subpackage.mymodule.func']
    assert var_6 == 'package.other.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = None
    var_3 = 'func'
    var_4 = 1
    var_5 = 'mymodule.func'
    var_6 = bool('mymodule.func' in var_0.alias)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 'getcwd'
    var_6 = 'cwd'
    var_7 = 0
    var_8 = var_0.alias['mymodule.path']
    assert var_8 == 'os.path'
    var_9 = var_0.alias['mymodule.cwd']
    assert var_9 == 'os.getcwd'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_preserves_non_typing_attribute. Retrieved 5/10 statements.
# Partially parsed test_visit_Attribute_with_non_name_value. Retrieved 5/9 statements.
# Partially parsed test_visit_Attribute_preserves_typing_attribute_with_different_module. Retrieved 4/9 statements.
# Partially parsed test_visit_Attribute_typing_with_list_attr. Retrieved 5/11 statements.
# Partially parsed test_visit_Attribute_typing_with_dict_attr. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Union'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other_module'
    var_4 = []
    var_5 = 'SomeClass'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'string'
    var_4 = []
    var_5 = 'attr'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = []
    var_4 = 'Optional'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Dict'
    var_6 = []



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 6/16 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_list. Retrieved 6/19 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/16 statements.
# Partially parsed test_globals_with_invalid_node. Retrieved 10/21 statements.
# Partially parsed test_globals_annotated_without_value. Retrieved 10/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module.MY_CONST'
    var_9 = bool('test_module.MY_CONST' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.MY_CONST']
    assert var_10 == '42'
    var_11 = 'test_module.MY_CONST'
    var_12 = bool('test_module.MY_CONST' in var_0.const)
    assert var_12 is True
    var_13 = var_0.const['test_module.MY_CONST']
    assert var_13 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_var'
    var_3 = 'hello'
    var_4 = []
    var_5 = None
    var_6 = 'test_module.my_var'
    var_7 = bool('test_module.my_var' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.my_var']
    assert var_8 == "'hello'"

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT'
    var_3 = 100
    var_4 = []
    var_5 = None
    var_6 = 'test_module.CONSTANT'
    var_7 = bool('test_module.CONSTANT' in var_0.root)
    assert var_7 is True
    var_8 = var_0.root['test_module.CONSTANT']
    var_9 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_9 is True
    var_10 = 'test_module.CONSTANT'
    var_11 = bool('test_module.CONSTANT' in var_0.const)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.func1'
    var_10 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.func2'
    var_12 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_int'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module.my_int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = []
    var_6 = None
    var_7 = var_0.alias
    var_8 = len(var_7)
    var_9 = var_0.alias
    var_10 = len(var_9)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_var'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = var_0.alias
    var_8 = len(var_7)
    var_9 = var_0.alias
    var_10 = len(var_9)
    var_11 = bool(var_10 == var_8)
    assert var_11 is True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_true. Retrieved 8/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = []
    var_8 = []
    var_9 = 'value'
    var_10 = bool(var_2 or var_4)
    assert var_10 is True



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_load_docstring. Retrieved 2/19 statements.
# Partially parsed test_load_docstring_missing_attr. Retrieved 2/9 statements.
# Partially parsed test_load_docstring_with_function. Retrieved 3/2 statements.
# Partially parsed test_load_docstring_no_docstring. Retrieved 2/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = bool('test_module' in var_1.docstring)
    assert var_3 is True
    var_4 = 'test_module.TestClass'
    var_5 = bool('test_module.TestClass' in var_1.docstring)
    assert var_5 is True
    var_6 = 'test_module.TestClass.NestedClass'
    var_7 = bool('test_module.TestClass.NestedClass' in var_1.docstring)
    assert var_7 is True
    var_8 = var_1.docstring['test_module']
    assert var_8 == ''
    var_9 = 'test class docstring'
    var_10 = bool('test class docstring' in var_1.docstring['test_module.TestClass'])
    assert var_10 is True
    var_11 = 'nested class docstring'
    var_12 = bool('nested class docstring' in var_1.docstring['test_module.TestClass.NestedClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.NonExistent'
    var_3 = bool('test_module.NonExistent' not in var_1.docstring)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a test function docstring.'
    var_1 = 'test_module'
    var_2 = module_0.Parser()
    var_3 = 'test_module.test_function'
    var_4 = bool('test_module.test_function' in var_2.docstring)
    assert var_4 is True
    var_5 = 'test function docstring'
    var_6 = bool('test function docstring' in var_2.docstring['test_module.test_function'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a test function docstring.'
    var_1 = 'test_module'
    var_2 = module_0.Parser()
    var_3 = 'test_module.test_function'
    var_4 = bool('test_module.test_function' in var_2.docstring)
    assert var_4 is True
    var_5 = 'test function docstring'
    var_6 = bool('test function docstring' in var_2.docstring['test_module.test_function'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.NoDocClass'
    var_3 = bool('test_module.NoDocClass' not in var_1.docstring)
    assert var_3 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_imports_asname_not_none. Retrieved 6/30 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'p'
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = 'mymodule.p'
    var_7 = bool('mymodule.p' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['mymodule.p']
    assert var_8 == '.os.path'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_visit_constant_non_string_value. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = []



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 6/16 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_list. Retrieved 6/19 statements.
# Partially parsed test_globals_with_lowercase_name. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 6/17 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 7/25 statements.
# Partially parsed test_globals_with_annassign_without_value. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module.MY_CONST'
    var_9 = bool('test_module.MY_CONST' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.MY_CONST']
    assert var_10 == '42'
    var_11 = 'test_module.MY_CONST'
    var_12 = bool('test_module.MY_CONST' in var_0.const)
    assert var_12 is True
    var_13 = var_0.const['test_module.MY_CONST']
    assert var_13 == 'int'
    var_14 = var_0.root['test_module.MY_CONST']
    var_15 = bool(var_0.root['test_module.MY_CONST'] == var_1)
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT'
    var_3 = 'hello'
    var_4 = []
    var_5 = None
    var_6 = 'test_module.CONSTANT'
    var_7 = bool('test_module.CONSTANT' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.CONSTANT']
    assert var_8 == "'hello'"
    var_9 = 'test_module.CONSTANT'
    var_10 = bool('test_module.CONSTANT' in var_0.const)
    assert var_10 is True
    var_11 = var_0.const['test_module.CONSTANT']
    assert var_11 == 'str'
    var_12 = var_0.root['test_module.CONSTANT']
    var_13 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TYPED_VAR'
    var_3 = 123
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module.TYPED_VAR'
    var_7 = bool('test_module.TYPED_VAR' in var_0.const)
    assert var_7 is True
    var_8 = var_0.const['test_module.TYPED_VAR']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = []
    var_5 = 'Class1'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.func1'
    var_10 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.Class1'
    var_12 = bool('test_module.Class1' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'regular_var'
    var_3 = 42
    var_4 = []
    var_5 = None
    var_6 = 'test_module.regular_var'
    var_7 = bool('test_module.regular_var' in var_0.alias)
    assert var_7 is True
    var_8 = 'test_module.regular_var'
    var_9 = bool('test_module.regular_var' not in var_0.root)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 10
    var_5 = []
    var_6 = None
    var_7 = 'test_module.A'
    var_8 = bool('test_module.A' not in var_0.alias)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = []
    var_6 = 2
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.x'
    var_11 = bool('test_module.x' not in var_0.alias)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module.VAR'
    var_8 = bool('test_module.VAR' not in var_0.alias)
    assert var_8 is True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'test_module'
    var_4 = 'test_module.operating_system'
    var_5 = bool('test_module.operating_system' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.operating_system']
    assert var_6 == 'os'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_class_api_predicate_line_19_false. Retrieved 11/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False when len(node.targets) != 1'
    var_1 = module_0.Parser()
    var_2 = 'x = y = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_5.targets
    var_7 = len(var_6)
    var_8 = 1
    var_9 = var_7 == var_8
    var_10 = var_5.targets[var_4]



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_class_api_with_public_members. Retrieved 6/13 statements.
# Partially parsed test_class_api_with_bases. Retrieved 5/13 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 6/14 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 6/13 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/13 statements.
# Partially parsed test_class_api_with_typed_members. Retrieved 6/16 statements.
# Partially parsed test_class_api_with_const_members. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_3 = 0
    var_4 = 'test_module.TestClass'
    var_5 = []
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Base:\n    pass\n\nclass Derived(Base):\n    pass\n    '
    var_3 = 1
    var_4 = 'test_module.Derived'
    var_5 = 'test_module.Derived'
    var_6 = bool('test_module.Derived' in var_0.doc)
    assert var_6 is True
    var_7 = 'Base'
    var_8 = bool('Base' in var_0.doc['test_module.Derived'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Color:\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    '
    var_3 = 0
    var_4 = 'test_module.Color'
    var_5 = []
    var_6 = 'test_module.Color'
    var_7 = bool('test_module.Color' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr2\n    '
    var_3 = 0
    var_4 = 'test_module.TestClass'
    var_5 = []
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass EmptyClass:\n    pass\n    '
    var_3 = 0
    var_4 = 'test_module.EmptyClass'
    var_5 = []
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    count: int\n    name: str\n    value: float\n    '
    var_3 = 0
    var_4 = 'test_module.TestClass'
    var_5 = []
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    CONSTANT = 42\n    variable = "test"\n    '
    var_3 = 0
    var_4 = 'test_module.TestClass'
    var_5 = []
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #93
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_empty_sequence. Retrieved 1/3 statements.
# Partially parsed test_e_type_none_element_in_sequence. Retrieved 2/4 statements.
# Partially parsed test_e_type_single_constant_int. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_constant_str. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_constant_float. Retrieved 1/5 statements.
# Partially parsed test_e_type_multiple_same_type_constants. Retrieved 3/9 statements.
# Partially parsed test_e_type_multiple_different_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_sequences_same_type. Retrieved 3/10 statements.
# Partially parsed test_e_type_multiple_sequences_different_types. Retrieved 2/8 statements.
# Partially parsed test_e_type_mixed_sequences_with_same_type. Retrieved 4/12 statements.
# Partially parsed test_e_type_sequence_with_conflicting_types. Retrieved 2/7 statements.
# Partially parsed test_e_type_non_constant_element. Retrieved 1/5 statements.
# Partially parsed test_e_type_multiple_sequences_one_with_different_types. Retrieved 3/10 statements.
# Partially parsed test_e_type_single_sequence_with_bool. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_sequence_with_none_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]

def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'string'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'str'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []
    var_6 = 4
    var_7 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'mixed'
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 'str'
    var_5 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = None
    var_1 = []



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_constant_assignment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/11 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/11 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 5/10 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 5/10 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_string_constant. Retrieved 5/10 statements.
# Partially parsed test_globals_ignores_annotated_without_value. Retrieved 5/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int = 5'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.x']
    assert var_7 == '5'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONSTANT = 42'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.CONSTANT'
    var_6 = bool('test_module.CONSTANT' in var_0.alias)
    assert var_6 is True
    var_7 = 'test_module.CONSTANT'
    var_8 = bool('test_module.CONSTANT' in var_0.const)
    assert var_8 is True
    var_9 = var_0.root['test_module.CONSTANT']
    var_10 = bool(var_0.root['test_module.CONSTANT'] == var_2)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "__all__ = ['func1', 'func2']"
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_2])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_2])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "__all__ = ('func1', 'func2')"
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_2])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_2])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x, y = 1, 2'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' not in var_0.alias)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = y = 5'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' not in var_0.alias)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'value = 10  # type: int'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.value'
    var_6 = bool('test_module.value' in var_0.const)
    assert var_6 is True
    var_7 = var_0.const['test_module.value']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "name = 'hello'"
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.name'
    var_6 = bool('test_module.name' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.const['test_module.name']
    assert var_7 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' not in var_0.alias)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 7/9 statements.
# Partially parsed test_imports_with_import_as_statement. Retrieved 7/9 statements.
# Partially parsed test_imports_with_from_import_statement. Retrieved 7/9 statements.
# Partially parsed test_imports_with_from_import_as_statement. Retrieved 7/9 statements.
# Partially parsed test_imports_with_relative_import. Retrieved 7/9 statements.
# Partially parsed test_imports_with_relative_import_level_2. Retrieved 7/9 statements.
# Partially parsed test_imports_with_relative_import_with_module. Retrieved 7/9 statements.
# Partially parsed test_imports_with_multiple_imports. Retrieved 7/9 statements.
# Partially parsed test_imports_with_multiple_from_imports. Retrieved 7/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['test_module.os']
    assert var_7 == 'os'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os as operating_system'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['test_module.operating_system']
    assert var_7 == 'os'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from os import path'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['test_module.path']
    assert var_7 == 'os.path'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from os import path as p'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['test_module.p']
    assert var_7 == 'os.path'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from . import utils'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'pkg.module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['pkg.module.utils']
    assert var_7 == 'pkg.utils'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from .. import utils'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'pkg.sub.module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['pkg.sub.module.utils']
    assert var_7 == 'pkg.utils'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from .utils import helper'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'pkg.module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['pkg.module.helper']
    assert var_7 == 'pkg.utils.helper'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os, sys'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['test_module.os']
    assert var_7 == 'os'
    var_8 = var_0.alias['test_module.sys']
    assert var_8 == 'sys'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from os import path, sep'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias['test_module.path']
    assert var_7 == 'os.path'
    var_8 = var_0.alias['test_module.sep']
    assert var_8 == 'os.sep'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 6/15 statements.
# Partially parsed test_class_api_with_bases. Retrieved 6/15 statements.
# Partially parsed test_class_api_with_enums. Retrieved 6/15 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/15 statements.
# Partially parsed test_class_api_with_deleted_attributes. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = '## class TestClass\n\n'
    var_4 = '\nclass TestClass:\n    attr1: int\n    attr2: str = "default"\n    _private: int = 5\n    '
    var_5 = 0
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Members'
    var_9 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.MyClass'
    var_3 = '## class MyClass\n\n'
    var_4 = 'class MyClass(BaseClass): pass'
    var_5 = 0
    var_6 = 'Bases'
    var_7 = bool('Bases' in var_0.doc['test_module.MyClass'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.Color'
    var_3 = '## class Color\n\n'
    var_4 = '\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    '
    var_5 = 0
    var_6 = 'Enums'
    var_7 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.Empty'
    var_3 = '## class Empty\n\n'
    var_4 = 'class Empty: pass'
    var_5 = 0
    var_6 = 'test_module.Empty'
    var_7 = bool('test_module.Empty' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = '## class TestClass\n\n'
    var_4 = '\nclass TestClass:\n    attr1: int\n    del attr1\n    '
    var_5 = 0
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parser_new_factory_method. Retrieved 3/4 statements.
# Partially parsed test_parser_constructor_all_fields_independent. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = module_0.Parser()
    var_2 = 'test'
    var_3 = bool('test' not in var_1.doc)
    assert var_3 is True
    var_4 = var_0.doc
    var_5 = bool(var_0.doc is not var_1.doc)
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level is not var_1.level)
    assert var_7 is True
    var_8 = var_0.imp
    var_9 = bool(var_0.imp is not var_1.imp)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_visit_attribute_typing_prefix. Retrieved 5/11 statements.
# Partially parsed test_visit_attribute_non_typing_prefix. Retrieved 5/11 statements.
# Partially parsed test_visit_attribute_non_name_value. Retrieved 6/14 statements.
# Partially parsed test_visit_attribute_typing_with_different_attributes. Retrieved 5/11 statements.
# Partially parsed test_visit_attribute_typing_with_list. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Union'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other_module'
    var_4 = []
    var_5 = 'SomeClass'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'module'
    var_4 = []
    var_5 = 'submodule'
    var_6 = []
    var_7 = 'Class'
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Optional'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/13 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/20 statements.
# Partially parsed test_class_api_with_enum. Retrieved 8/25 statements.
# Partially parsed test_class_api_with_delete. Retrieved 8/24 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 8/20 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/18 statements.
# Partially parsed test_class_api_empty_body. Retrieved 6/10 statements.
# Partially parsed test_class_api_multiple_bases. Retrieved 6/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['test_module.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 'int'
    var_3 = []
    var_4 = 10
    var_5 = []
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'member1'
    var_13 = bool('member1' in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'MEMBER'
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.TestEnum'
    var_12 = 'test_module.TestEnum'
    var_13 = bool('test_module.TestEnum' in var_0.doc)
    assert var_13 is True
    var_14 = 'MEMBER'
    var_15 = bool('MEMBER' in var_0.doc['test_module.TestEnum'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 'int'
    var_3 = []
    var_4 = 10
    var_5 = []
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = bool('member1' not in var_0.doc['test_module.TestClass'] or 'member1' in var_0.doc['test_module.TestClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'int'
    var_3 = []
    var_4 = 10
    var_5 = []
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member1'
    var_2 = 10
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = []
    var_3 = 'Base2'
    var_4 = []
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_compile_with_single_module. Retrieved 2/7 statements.
# Partially parsed test_compile_with_magic_method_no_docstring. Retrieved 2/6 statements.
# Partially parsed test_compile_with_public_function. Retrieved 2/11 statements.
# Partially parsed test_compile_with_toc_enabled. Retrieved 3/8 statements.
# Partially parsed test_compile_with_constants. Retrieved 2/9 statements.
# Partially parsed test_compile_with_private_name. Retrieved 2/11 statements.
# Partially parsed test_compile_with_all_filter. Retrieved 3/12 statements.
# Partially parsed test_compile_warning_missing_docstring. Retrieved 2/10 statements.
# Partially parsed test_compile_sorted_by_level_and_name. Retrieved 4/16 statements.
# Partially parsed test_compile_with_link_in_name. Retrieved 3/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    var_2 = 'Module `test`'
    var_3 = bool('Module `test`' in var_1)
    assert var_3 is True
    var_4 = 'Test module documentation'
    var_5 = bool('Test module documentation' in var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    var_2 = 'Module `test`'
    var_3 = bool('Module `test`' in var_1)
    assert var_3 is True
    var_4 = 'Function documentation'
    var_5 = bool('Function documentation' in var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.compile()
    var_3 = '**Table of contents:**'
    var_4 = bool('**Table of contents:**' in var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    var_2 = bool('CONST' in var_1 or 'Module `test`' in var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    var_2 = '_private'
    var_3 = bool('_private' not in var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.public_func'
    var_2 = var_0.compile()
    var_3 = 'public_func'
    var_4 = bool('public_func' in var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    var_2 = '## `func`'
    var_3 = bool('## `func`' not in var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    var_2 = 'a_test'
    var_3 = 'z_test'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = var_1.compile()
    var_3 = bool('test-module' in var_2 or 'Module `test.module`' in var_2)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/16 statements.
# Partially parsed test_class_api_with_enum. Retrieved 7/19 statements.
# Partially parsed test_class_api_with_delete. Retrieved 8/19 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/16 statements.
# Partially parsed test_class_api_multiple_members. Retrieved 10/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_9 is True
    var_10 = 'BaseClass'
    var_11 = bool('BaseClass' in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = None
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True
    var_14 = 'member1'
    var_15 = bool('member1' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = 'ENUM_VALUE'
    var_5 = []
    var_6 = 'value'
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.TestEnum'
    var_10 = 'test_module.TestEnum'
    var_11 = bool('test_module.TestEnum' in var_0.doc)
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc['test_module.TestEnum'])
    assert var_13 is True
    var_14 = 'ENUM_VALUE'
    var_15 = bool('ENUM_VALUE' in var_0.doc['test_module.TestEnum'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.TestClass'
    var_11 = 'test_module.TestClass'
    var_12 = bool('test_module.TestClass' in var_0.doc)
    assert var_12 is True
    var_13 = 'member1'
    var_14 = bool('member1' not in var_0.doc['test_module.TestClass'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'class EmptyClass'
    var_9 = bool('class EmptyClass' in var_0.doc['test_module.EmptyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'int'
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 'test_module.TestClass'
    var_10 = bool('test_module.TestClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_12 is True
    var_13 = 'int'
    var_14 = bool('int' in var_0.doc['test_module.TestClass'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'str'
    var_5 = []
    var_6 = 'member2'
    var_7 = []
    var_8 = 'int'
    var_9 = []
    var_10 = None
    var_11 = 1
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'
    var_14 = 'test_module.TestClass'
    var_15 = bool('test_module.TestClass' in var_0.doc)
    assert var_15 is True
    var_16 = 'Members'
    var_17 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_17 is True
    var_18 = 'member1'
    var_19 = bool('member1' in var_0.doc['test_module.TestClass'])
    assert var_19 is True
    var_20 = 'member2'
    var_21 = bool('member2' in var_0.doc['test_module.TestClass'])
    assert var_21 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_class_api_delete_statement_predicate. Retrieved 3/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = []
    var_3 = 'attr2'
    var_4 = []



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested. Retrieved 2/7 statements.
# Partially parsed test_attr_break_in_chain. Retrieved 2/7 statements.
# Partially parsed test_attr_none_value. Retrieved 2/5 statements.
# Partially parsed test_attr_single_dot. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_multiple_levels_with_none_in_middle. Retrieved 3/7 statements.


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
    var_1 = 'inner.nonexistent.something'

def test_case_0():
    var_0 = None
    var_1 = 'attr'

def test_case_0():
    var_0 = 'value'
    var_1 = 'attr.'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'nested_value'
    var_1 = None
    var_2 = 'inner.value'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_compile_predicate_is_public_evaluates_to_true. Retrieved 6/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = "Test that the predicate 'if not self.is_public(name)' evaluates to True (skip continues)."
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.is_public
    var_5 = var_3.compile()
    assert var_5 == '\n'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc_true. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #13
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
    var_0 = 'os.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__dict__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__main__.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_internal.public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public._private.method'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.module._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__private__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'django.contrib.auth.models.User'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.__name__.method'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'package._module.Class'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_assign_predicate. Retrieved 8/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_class'
    var_2 = ''
    var_3 = 'x'
    var_4 = None
    var_5 = []
    var_6 = 42
    var_7 = []
    var_8 = 1
    var_9 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #17
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.level)
    assert var_7 is True
    var_8 = 'test_module'
    var_9 = bool('test_module' in var_0.imp)
    assert var_9 is True
    var_10 = 'test_module'
    var_11 = bool('test_module' in var_0.root)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nfrom sys import path'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.imp)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""Module docstring."""\nx = 1'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.docstring)
    assert var_5 is True
    var_6 = 'Module docstring'
    var_7 = bool('Module docstring' in var_0.docstring['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.root)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass:\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.root)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST = 42'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.CONST'
    var_5 = bool('test_module.CONST' in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int = 5'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Outer:\n    class Inner:\n        pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.Outer'
    var_5 = bool('test_module.Outer' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.Outer.Inner'
    var_7 = bool('test_module.Outer.Inner' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.async_foo'
    var_5 = bool('test_module.async_foo' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'x = 1'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' in var_1.doc['test_module'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = 'x = 1'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_1.doc['test_module'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1'
    var_2 = 'pkg.subpkg.module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.level['pkg.subpkg.module']
    assert var_4 == 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    pass\ndef bar():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.bar'
    var_7 = bool('test_module.bar' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    """Function doc."""\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.docstring)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass:\n    def method(self):\n        pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.method'
    var_7 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@property\ndef foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'try:\n    x = 1\nexcept:\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'if True:\n    x = 1'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_compile_skips_private_names. Retrieved 5/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that compile skips names where is_public returns False.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()
    var_5 = '_private_module'
    var_6 = bool('_private_module' not in var_4)
    assert var_6 is True
    var_7 = 'public_module'
    var_8 = bool('public_module' in var_4)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_api_function_def. Retrieved 7/13 statements.
# Partially parsed test_api_async_function_def. Retrieved 7/13 statements.
# Partially parsed test_api_class_def. Retrieved 7/13 statements.
# Partially parsed test_api_with_prefix. Retrieved 8/14 statements.
# Partially parsed test_api_with_decorator. Retrieved 7/13 statements.
# Partially parsed test_api_sets_full_name. Retrieved 7/13 statements.
# Partially parsed test_api_with_link. Retrieved 8/14 statements.
# Partially parsed test_api_without_link. Retrieved 7/13 statements.
# Partially parsed test_api_nested_class_methods. Retrieved 7/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(): pass'
    var_2 = 'test_module'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.api(var_2, var_5)
    var_7 = 'test_module.foo'
    var_8 = bool('test_module.foo' in var_0.doc)
    assert var_8 is True
    var_9 = 'foo()'
    var_10 = bool('foo()' in var_0.doc['test_module.foo'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def bar(): pass'
    var_2 = 'test_module'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.api(var_2, var_5)
    var_7 = 'test_module.bar'
    var_8 = bool('test_module.bar' in var_0.doc)
    assert var_8 is True
    var_9 = 'async bar()'
    var_10 = bool('async bar()' in var_0.doc['test_module.bar'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass: pass'
    var_2 = 'test_module'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.api(var_2, var_5)
    var_7 = 'test_module.MyClass'
    var_8 = bool('test_module.MyClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'class MyClass'
    var_10 = bool('class MyClass' in var_0.doc['test_module.MyClass'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def method(self): pass'
    var_2 = 'test_module'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'OuterClass'
    var_7 = var_0.api(var_2, var_5, prefix=var_6)
    var_8 = 'test_module.OuterClass.method'
    var_9 = bool('test_module.OuterClass.method' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@staticmethod\ndef foo(): pass'
    var_2 = 'test_module'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.api(var_2, var_5)
    var_7 = 'test_module.foo'
    var_8 = bool('test_module.foo' in var_0.doc)
    assert var_8 is True
    var_9 = 'Decorators'
    var_10 = bool('Decorators' in var_0.doc['test_module.foo'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(): pass'
    var_2 = 'mymodule'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.api(var_2, var_5)
    var_7 = '*Full name:* `mymodule.func`'
    var_8 = bool('*Full name:* `mymodule.func`' in var_0.doc['mymodule.func'])
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'def test(): pass'
    var_3 = 'module'
    var_4 = module_1.parse(var_2)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_1.api(var_3, var_6)
    var_8 = '<a id='
    var_9 = bool('<a id=' in var_1.doc['module.test'])
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = 'def test(): pass'
    var_3 = 'module'
    var_4 = module_1.parse(var_2)
    var_5 = var_4.body[var_0]
    var_6 = var_1.api(var_3, var_5)
    var_7 = '<a id='
    var_8 = bool('<a id=' not in var_1.doc['module.test'])
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Outer:\n    def inner(self): pass'
    var_2 = 'test_module'
    var_3 = module_1.parse(var_1)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.api(var_2, var_5)
    var_7 = 'test_module.Outer'
    var_8 = bool('test_module.Outer' in var_0.doc)
    assert var_8 is True
    var_9 = 'test_module.Outer.inner'
    var_10 = bool('test_module.Outer.inner' in var_0.doc)
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_class_api_predicate_line_11. Retrieved 11/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = 'x: int'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = var_7.target



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_func_api_with_simple_arguments. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_self. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 11/16 statements.
# Partially parsed test_func_api_no_annotation. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_return_type. Retrieved 11/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def simple_func(a: int, b: str) -> bool: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.simple_func'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_1, cls_method=var_1)
    var_11 = 'test_module.simple_func'
    var_12 = bool('test_module.simple_func' in var_2.doc)
    assert var_12 is True
    var_13 = '|'
    var_14 = bool('|' in var_2.doc['test_module.simple_func'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = "def func_with_defaults(a: int = 5, b: str = 'hello') -> None: pass"
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.func_with_defaults'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_1, cls_method=var_1)
    var_11 = 'test_module.func_with_defaults'
    var_12 = bool('test_module.func_with_defaults' in var_2.doc)
    assert var_12 is True
    var_13 = '|'
    var_14 = bool('|' in var_2.doc['test_module.func_with_defaults'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def method(self, x: int) -> str: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass.method'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_0, cls_method=var_1)
    var_11 = 'test_module.MyClass.method'
    var_12 = bool('test_module.MyClass.method' in var_2.doc)
    assert var_12 is True
    var_13 = 'Self'
    var_14 = bool('Self' in var_2.doc['test_module.MyClass.method'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def method(cls, x: int) -> str: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass.method'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_0, cls_method=var_0)
    var_11 = 'test_module.MyClass.method'
    var_12 = bool('test_module.MyClass.method' in var_2.doc)
    assert var_12 is True
    var_13 = 'type[Self]'
    var_14 = bool('type[Self]' in var_2.doc['test_module.MyClass.method'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func_varargs(*args: int, **kwargs: str) -> None: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.func_varargs'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_1, cls_method=var_1)
    var_11 = 'test_module.func_varargs'
    var_12 = bool('test_module.func_varargs' in var_2.doc)
    assert var_12 is True
    var_13 = bool('*args' in var_2.doc['test_module.func_varargs'] or 'args' in var_2.doc['test_module.func_varargs'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = "def func_kwonly(a: int, *, b: str = 'default') -> None: pass"
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.func_kwonly'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_1, cls_method=var_1)
    var_11 = 'test_module.func_kwonly'
    var_12 = bool('test_module.func_kwonly' in var_2.doc)
    assert var_12 is True
    var_13 = '|'
    var_14 = bool('|' in var_2.doc['test_module.func_kwonly'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func_no_ann(a, b): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.func_no_ann'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_1, cls_method=var_1)
    var_11 = 'test_module.func_no_ann'
    var_12 = bool('test_module.func_no_ann' in var_2.doc)
    assert var_12 is True
    var_13 = 'Any'
    var_14 = bool('Any' in var_2.doc['test_module.func_no_ann'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'def func_return(x: int) -> list[str]: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.func_return'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_2.func_api(var_6, var_7, var_8, var_9, has_self=var_1, cls_method=var_1)
    var_11 = 'test_module.func_return'
    var_12 = bool('test_module.func_return' in var_2.doc)
    assert var_12 is True
    var_13 = 'return'
    var_14 = bool('return' in var_2.doc['test_module.func_return'])
    assert var_14 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_defaults_with_constant_expressions. Retrieved 2/8 statements.
# Partially parsed test_defaults_with_mixed_none_and_expressions. Retrieved 3/9 statements.
# Partially parsed test_defaults_with_name_expression. Retrieved 1/6 statements.
# Partially parsed test_defaults_with_complex_expression. Retrieved 2/10 statements.
# Partially parsed test_defaults_returns_iterator. Retrieved 2/6 statements.
# Partially parsed test_defaults_with_pipe_character_in_expression. Retrieved 1/6 statements.
# Partially parsed test_defaults_with_ampersand_in_expression. Retrieved 1/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'hello'
    var_3 = []
    var_4 = '`42`'
    var_5 = "`'hello'`"

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = None
    var_3 = 2
    var_4 = []
    var_5 = '`1`'
    var_6 = '`2`'

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = '`x`'

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = '`'

def test_case_0():
    var_0 = None
    var_1 = 5
    var_2 = []

def test_case_0():
    var_0 = 'a|b'
    var_1 = []
    var_2 = '&#124;'

def test_case_0():
    var_0 = 'a&b'
    var_1 = []
    var_2 = '<code>'
    var_3 = '</code>'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_ann_with_self_parameter. Retrieved 12/19 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 9/15 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 5/11 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 12/19 statements.
# Partially parsed test_func_ann_multiple_args. Retrieved 20/28 statements.
# Partially parsed test_func_ann_with_self_type_annotation. Retrieved 10/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = 0
    var_7 = 'int'
    var_8 = module_1.parse(var_7)
    var_9 = var_8.body[var_6]
    var_10 = var_9.value
    var_11 = []
    var_12 = True
    var_13 = False

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
    var_8 = []
    var_9 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '*'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = 0
    var_7 = 'str'
    var_8 = module_1.parse(var_7)
    var_9 = var_8.body[var_6]
    var_10 = var_9.value
    var_11 = []
    var_12 = False
    var_13 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 0
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = []
    var_9 = 'b'
    var_10 = 'str'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_3]
    var_13 = var_12.value
    var_14 = []
    var_15 = 'c'
    var_16 = 'float'
    var_17 = module_1.parse(var_16)
    var_18 = var_17.body[var_3]
    var_19 = var_18.value
    var_20 = []
    var_21 = False
    var_22 = False

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
    var_8 = []
    var_9 = True
    var_10 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_compile_magic_method_predicate. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.compile()
    var_4 = '__init__'
    var_5 = bool('__init__' not in var_3)
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 4/8 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_alias_simple. Retrieved 7/11 statements.
# Partially parsed test_visit_name_with_typevar_alias. Retrieved 8/12 statements.
# Partially parsed test_visit_name_with_circular_alias. Retrieved 6/10 statements.
# Partially parsed test_visit_name_not_in_alias. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_complex_alias. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeClass'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.T'
    var_2 = 'typing.TypeVar'
    var_3 = "typing.TypeVar('T')"
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = ''
    var_6 = module_0.Resolver(var_0, var_4, var_5)
    var_7 = 'T'
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.X'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'X'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'UnknownType'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.MyList'
    var_2 = 'typing.List[int]'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyList'
    var_7 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_docstring. Retrieved 3/18 statements.
# Partially parsed test_load_docstring_with_doctest. Retrieved 2/7 statements.
# Partially parsed test_load_docstring_missing_attribute. Retrieved 2/8 statements.
# Partially parsed test_load_docstring_nested_attributes. Retrieved 2/14 statements.
# Partially parsed test_load_docstring_no_docstring. Retrieved 2/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.sub'
    var_2 = module_0.Parser()
    var_3 = 'test_module'
    var_4 = bool('test_module' in var_2.docstring)
    assert var_4 is True
    var_5 = 'Module docstring'
    var_6 = bool('Module docstring' in var_2.docstring['test_module'])
    assert var_6 is True
    var_7 = 'test_module.MockClass'
    var_8 = bool('test_module.MockClass' in var_2.docstring)
    assert var_8 is True
    var_9 = 'Class docstring'
    var_10 = bool('Class docstring' in var_2.docstring['test_module.MockClass'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = bool('test_module' in var_1.docstring)
    assert var_3 is True
    var_4 = '```python'
    var_5 = bool('```python' in var_1.docstring['test_module'])
    assert var_5 is True
    var_6 = '```'
    var_7 = bool('```' in var_1.docstring['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = bool('test_module' in var_1.docstring)
    assert var_3 is True
    var_4 = 'test_module.missing'
    var_5 = bool('test_module.missing' not in var_1.docstring)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.Outer'
    var_3 = bool('test_module.Outer' in var_1.docstring)
    assert var_3 is True
    var_4 = 'Outer class'
    var_5 = bool('Outer class' in var_1.docstring['test_module.Outer'])
    assert var_5 is True
    var_6 = 'test_module.Outer.Inner'
    var_7 = bool('test_module.Outer.Inner' in var_1.docstring)
    assert var_7 is True
    var_8 = 'Inner class'
    var_9 = bool('Inner class' in var_1.docstring['test_module.Outer.Inner'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.NoDoc'
    var_3 = bool('test_module.NoDoc' not in var_1.docstring)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_broken_chain_in_middle. Retrieved 2/7 statements.
# Partially parsed test_attr_with_none_intermediate_value. Retrieved 2/5 statements.
# Partially parsed test_attr_single_character_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_with_numeric_value. Retrieved 2/5 statements.
# Partially parsed test_attr_with_list_value. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'inner_value'
    var_1 = 'outer_attr.inner_attr'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.value'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'value'
    var_1 = 'outer_attr.nonexistent'

def test_case_0():
    var_0 = 'value'
    var_1 = 'outer_attr.nonexistent.inner_attr'

def test_case_0():
    var_0 = None
    var_1 = 'outer_attr.inner_attr'

def test_case_0():
    var_0 = 'single_char'
    var_1 = 'a'

def test_case_0():
    var_0 = 42
    var_1 = 'number'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_class_api_assign_predicate. Retrieved 5/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_attr'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 0



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_func_ann_annotation_not_none. Retrieved 10/45 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 0
    var_3 = 'int'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = var_5.value
    var_7 = []
    var_8 = 'test_root'
    var_9 = False
    var_10 = False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_visit_constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_invalid_syntax_string. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_valid_name_string. Retrieved 6/9 statements.
# Partially parsed test_visit_constant_with_self_type. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_subscript_expression. Retrieved 6/9 statements.
# Partially parsed test_visit_constant_with_complex_annotation. Retrieved 6/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not valid python ]['
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.str'
    var_2 = 'builtins.str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'str'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = 'MyClass'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.List'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'List[str]'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.Union'
    var_2 = 'typing.Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Union[int, str]'
    var_6 = []



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 8/14 statements.
# Partially parsed test_globals_with_all_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 9/15 statements.
# Partially parsed test_globals_ignores_invalid_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_annotated_no_value. Retrieved 8/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x: int = 5'
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.x']
    assert var_10 == '5'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'CONST = 42'
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.CONST'
    var_9 = bool('test_module.CONST' in var_0.const)
    assert var_9 is True
    var_10 = var_0.root['test_module.CONST']
    var_11 = bool(var_0.root['test_module.CONST'] == var_1)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['func1', 'func2']"
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.func1'
    var_9 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_9 is True
    var_10 = 'test_module.func2'
    var_11 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "value = 'hello'"
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.value'
    var_9 = bool('test_module.value' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.value']
    assert var_10 == "'hello'"

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x = 10  # type: int'
    var_4 = 0
    var_5 = True
    var_6 = module_1.parse(var_3, type_comments=var_5)
    var_7 = var_6.body[var_4]
    var_8 = var_0.globals(var_1, var_7)
    var_9 = 'test_module.x'
    var_10 = bool('test_module.x' in var_0.alias)
    assert var_10 is True
    var_11 = 'test_module.x'
    var_12 = bool('test_module.x' in var_0.const)
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a, b = 1, 2'
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.a'
    var_9 = bool('test_module.a' not in var_0.alias)
    assert var_9 is True
    var_10 = 'test_module.b'
    var_11 = bool('test_module.b' not in var_0.alias)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x: int'
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' not in var_0.alias)
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_class_api_assign_predicate. Retrieved 5/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 0



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_compile_docstring_condition. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 evaluates to True when name is in docstring.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = 'test_module.test_func'
    var_5 = var_3.docstring
    var_6 = var_4 in var_5
    assert var_6 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'doc'
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = False
    var_7 = module_0.Parser(var_5, var_5, var_6, var_2, var_4)
    var_8 = var_7.level
    var_9 = bool(var_7.level == var_2)
    assert var_9 is True
    var_10 = var_7.doc
    var_11 = bool(var_7.doc == var_4)
    assert var_11 is True
    var_12 = var_7.link
    assert var_12 is True
    var_13 = var_7.b_level
    assert var_13 == 1
    var_14 = var_7.toc
    assert var_14 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_is_public_with_magic_name. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_public_name_no_all. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_name_in_all. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_module_in_all. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_submodule_not_in_all. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_underscore_name_in_all. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_module_key_in_imp. Retrieved 7/11 statements.
# Partially parsed test_is_public_with_empty_all_public_family. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_empty_all_private_family. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = 'doc'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'doc'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.public_func'
    var_3 = 'doc'
    var_4 = set()
    var_5 = var_0.is_public(var_2)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.func'
    var_3 = 'doc'
    var_4 = {var_2}
    var_5 = var_0.is_public(var_2)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'doc'
    var_3 = {var_1}
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.sub'
    var_3 = 'doc'
    var_4 = {var_1}
    var_5 = var_0.is_public(var_2)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module._private'
    var_3 = 'doc'
    var_4 = {var_2}
    var_5 = var_0.is_public(var_2)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'pkg.mod'
    var_3 = 'pkg.mod.func'
    var_4 = 'doc'
    var_5 = {var_2}
    var_6 = var_0.is_public(var_2)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mod'
    var_2 = 'mod.func'
    var_3 = 'doc'
    var_4 = set()
    var_5 = var_0.is_public(var_2)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mod'
    var_2 = 'mod._func'
    var_3 = 'doc'
    var_4 = set()
    var_5 = var_0.is_public(var_2)
    assert var_5 is False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_class_api_predicate_line_11_false. Retrieved 23/43 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = 'x = 5'
    var_4 = 0
    var_5 = module_1.parse(var_3)
    var_6 = var_5.body[var_4]
    var_7 = []
    var_8 = [var_6]
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_10 = module_0.Parser()
    var_11 = 'test_module2'
    var_12 = 'TestClass2'
    var_13 = 'x, y: int'
    var_14 = 'x: int = 5'
    var_15 = module_1.parse(var_14)
    var_16 = var_15.body[var_4]
    var_17 = 'del x'
    var_18 = module_1.parse(var_17)
    var_19 = var_18.body[var_4]
    var_20 = []
    var_21 = [var_19]
    var_22 = var_10.class_api(var_11, var_12, var_20, var_21)



# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------

# Partially parsed test_func_api_predicate_line_32_false. Retrieved 10/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.test_func'
    var_11 = False
    var_12 = 'test_module.test_func'
    var_13 = bool('test_module.test_func' in var_0.doc)
    assert var_13 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_class_api_line_15_predicate_false. Retrieved 7/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that line 15 predicate (is_public_family(attr)) evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'TestClass'
    var_3 = '# class TestClass\n\n'
    var_4 = 0
    var_5 = '_private_attr: int'
    var_6 = []
    var_7 = bool('_private_attr' not in var_1.doc['TestClass'] or 'Members' not in var_1.doc['TestClass'])
    assert var_7 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 8/13 statements.
# Partially parsed test_class_api_with_bases. Retrieved 7/13 statements.
# Partially parsed test_class_api_with_enums. Retrieved 7/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 8/13 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 8/13 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 9/14 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 9/14 statements.
# Partially parsed test_class_api_multiple_bases. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass MyClass:\n    attr1: int\n    attr2: str = "default"\n    '
    var_5 = 0
    var_6 = 'test_module.MyClass'
    var_7 = []
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Members'
    var_11 = bool('Members' in var_0.doc['test_module.MyClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass MyClass(BaseClass):\n    pass\n    '
    var_5 = 0
    var_6 = 'test_module.MyClass'
    var_7 = 'test_module.MyClass'
    var_8 = bool('test_module.MyClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'Bases'
    var_10 = bool('Bases' in var_0.doc['test_module.MyClass'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import enum'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    '
    var_5 = 0
    var_6 = 'test_module.Color'
    var_7 = 'test_module.Color'
    var_8 = bool('test_module.Color' in var_0.doc)
    assert var_8 is True
    var_9 = 'Enums'
    var_10 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass EmptyClass:\n    pass\n    '
    var_5 = 0
    var_6 = 'test_module.EmptyClass'
    var_7 = []
    var_8 = 'test_module.EmptyClass'
    var_9 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass MyClass:\n    attr1: int\n    del attr1\n    '
    var_5 = 0
    var_6 = 'test_module.MyClass'
    var_7 = []
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass MyClass:\n    attr1 = 42  # type: int\n    '
    var_5 = True
    var_6 = 0
    var_7 = 'test_module.MyClass'
    var_8 = []
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.MyClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass MyClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_5 = 0
    var_6 = 'test_module.MyClass'
    var_7 = []
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = var_0.doc[var_6]
    var_11 = bool('public_attr' in var_10 or 'Members' not in var_10)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\nclass MyClass(Base1, Base2):\n    pass\n    '
    var_5 = 0
    var_6 = 'test_module.MyClass'
    var_7 = 'test_module.MyClass'
    var_8 = bool('test_module.MyClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'Bases'
    var_10 = bool('Bases' in var_0.doc['test_module.MyClass'])
    assert var_10 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_class_api_type_comment_is_not_none. Retrieved 11/24 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_class'
    var_3 = []
    var_4 = 'my_attr'
    var_5 = None
    var_6 = []
    var_7 = 42
    var_8 = []
    var_9 = 'int'
    var_10 = 'builtins'
    var_11 = __import__(var_10)
    var_12 = 'is_public_family'
    var_13 = bool('my_attr' in var_0.doc[var_2] or var_0.doc[var_2] == '')
    assert var_13 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_api_predicate_link_false. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'def foo(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_0]
    var_6 = 'test_module'
    var_7 = var_2.api(var_6, var_5)
    var_8 = '\n<a id="'
    var_9 = bool('\n<a id="' not in var_2.doc['test_module.foo'])
    assert var_9 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 4/8 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_alias_not_circular. Retrieved 7/11 statements.
# Partially parsed test_visit_name_with_circular_alias. Retrieved 6/10 statements.
# Partially parsed test_visit_name_with_typevar. Retrieved 9/13 statements.
# Partially parsed test_visit_name_without_alias. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_empty_root. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeType'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyType'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyType'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'MyType'
    var_6 = []

import apimd.parser as module_0

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
    var_9 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'UnknownType'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'MyType'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3, var_0)
    var_5 = []



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/16 statements.
# Partially parsed test_class_api_with_enums. Retrieved 12/18 statements.
# Partially parsed test_class_api_with_bases. Retrieved 10/16 statements.
# Partially parsed test_class_api_empty_class. Retrieved 10/16 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/16 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 11/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    attr1: int\n    attr2: str = "default"\n    _private: float\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.MyClass'
    var_7 = var_5.bases
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)
    var_10 = bool(var_6 in var_0.doc)
    assert var_10 is True
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc[var_6])
    assert var_12 is True
    var_13 = 'attr1'
    var_14 = bool('attr1' in var_0.doc[var_6])
    assert var_14 is True
    var_15 = 'attr2'
    var_16 = bool('attr2' in var_0.doc[var_6])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED: int\n    GREEN: int\n    BLUE: int\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 1
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.Color'
    var_7 = 'enum.Enum'
    var_8 = [var_7]
    var_9 = var_5.bases
    var_10 = var_5.body
    var_11 = var_0.class_api(var_1, var_6, var_9, var_10)
    var_12 = bool(var_6 in var_0.doc)
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Parent:\n    pass\n\nclass Child(Parent):\n    pass\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 1
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.Child'
    var_7 = var_5.bases
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)
    var_10 = bool(var_6 in var_0.doc)
    assert var_10 is True
    var_11 = 'Bases'
    var_12 = bool('Bases' in var_0.doc[var_6])
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class EmptyClass: pass'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.EmptyClass'
    var_7 = var_5.bases
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)
    var_10 = bool(var_6 in var_0.doc)
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    attr1: int\n    attr2: str\n    del attr2\n    '
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module.MyClass'
    var_7 = var_5.bases
    var_8 = var_5.body
    var_9 = var_0.class_api(var_1, var_6, var_7, var_8)
    var_10 = bool(var_6 in var_0.doc)
    assert var_10 is True
    var_11 = 'attr1'
    var_12 = bool('attr1' in var_0.doc[var_6])
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    value = 42  # type: int\n    '
    var_3 = True
    var_4 = module_1.parse(var_2, type_comments=var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.MyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_0.class_api(var_1, var_7, var_8, var_9)
    var_11 = bool(var_7 in var_0.doc)
    assert var_11 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_func_ann_yields_type_self_when_cls_method_true. Retrieved 5/39 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'root'
    var_5 = True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_visit_name_self_ty_predicate. Retrieved 4/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_default_params. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_dict_fields_are_independent. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = module_0.Parser()
    var_2 = 'test'
    var_3 = bool('test' not in var_1.doc)
    assert var_3 is True
    var_4 = var_1.doc
    var_5 = bool(var_1.doc == {})
    assert var_5 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 4/8 statements.
# Partially parsed test_visit_name_without_self_ty. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_alias_resolution. Retrieved 7/11 statements.
# Partially parsed test_visit_name_with_typevar_alias. Retrieved 9/13 statements.
# Partially parsed test_visit_name_with_circular_alias. Retrieved 6/10 statements.
# Partially parsed test_visit_name_unknown_name. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_nested_module. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'MyType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'SomeName'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyAlias'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyAlias'
    var_7 = []

import apimd.parser as module_0

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
    var_9 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.A'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'A'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'UnknownName'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'package.module'
    var_1 = 'package.module.Alias'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'Alias'
    var_7 = []



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_globals_predicate_line_35_evaluates_to_false. Retrieved 7/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 35 evaluates to False, allowing line 37 to execute.'
    var_1 = module_0.Parser()
    var_2 = '__all__'
    var_3 = None
    var_4 = []
    var_5 = 'item1'
    var_6 = []
    var_7 = 'item2'
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.item1'
    var_11 = bool('test_module.item1' in var_1.imp['test_module'])
    assert var_11 is True
    var_12 = 'test_module.item2'
    var_13 = bool('test_module.item2' in var_1.imp['test_module'])
    assert var_13 is True



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_element_with_single_constant. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_element_with_multiple_same_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_single_element_with_multiple_different_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_with_single_constants. Retrieved 2/8 statements.
# Partially parsed test_e_type_multiple_elements_with_multiple_constants. Retrieved 4/12 statements.
# Partially parsed test_e_type_element_with_none. Retrieved 2/4 statements.
# Partially parsed test_e_type_element_with_non_constant. Retrieved 2/7 statements.
# Partially parsed test_e_type_empty_sequence_in_elements. Retrieved 1/3 statements.
# Partially parsed test_e_type_mixed_types_in_single_element. Retrieved 2/7 statements.
# Partially parsed test_e_type_float_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_string_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_boolean_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_with_mixed_types. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'string'
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'hello'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 'a'
    var_5 = []
    var_6 = 'b'
    var_7 = []

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2.5
    var_3 = []

def test_case_0():
    var_0 = 1.5
    var_1 = []
    var_2 = 2.5
    var_3 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []
    var_2 = 'world'
    var_3 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = False
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 1.5
    var_3 = []
    var_4 = 'string'
    var_5 = []



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_visit_Attribute_typing_prefix. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_non_typing_prefix. Retrieved 5/11 statements.
# Partially parsed test_visit_Attribute_non_name_value. Retrieved 6/14 statements.
# Partially parsed test_visit_Attribute_typing_List. Retrieved 5/11 statements.
# Partially parsed test_visit_Attribute_typing_Dict. Retrieved 5/11 statements.
# Partially parsed test_visit_Attribute_preserves_context. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Union'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other_module'
    var_4 = []
    var_5 = 'SomeClass'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Dict'
    var_6 = []
    var_7 = 'items'
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Dict'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'Optional'
    var_6 = []



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_class_api_predicate_line_38_true. Retrieved 16/24 statements.


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
    var_9 = 'MEMBER1'
    var_10 = []
    var_11 = 'str'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_3]
    var_14 = var_13.value
    var_15 = 'value1'
    var_16 = []
    var_17 = 1
    var_18 = 'Enums'
    var_19 = bool('Enums' in var_0.doc[var_2])
    assert var_19 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_matching. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_list_not_matching. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_all_list_parent_matching. Retrieved 6/8 statements.
# Partially parsed test_is_public_module_in_imp_with_public_children. Retrieved 6/10 statements.
# Partially parsed test_is_public_module_in_imp_without_public_children. Retrieved 4/8 statements.
# Partially parsed test_is_public_public_family_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_nested_private. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
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
    var_1 = 'pkg.sub.func'
    var_2 = 'pkg'
    var_3 = 'pkg.sub'
    var_4 = {var_3}
    var_5 = var_0.is_public(var_1)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'pkg.child'
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
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.public_func'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module._internal'
    var_2 = 'pkg'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is False



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_globals_predicate_line_18_false. Retrieved 6/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when len(node.targets) != 1.'
    var_1 = module_0.Parser()
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = 1
    var_8 = []



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 4/12 statements.
# Partially parsed test_globals_with_assignment. Retrieved 4/12 statements.
# Partially parsed test_globals_with_lowercase_variable. Retrieved 4/12 statements.
# Partially parsed test_globals_with_all_list. Retrieved 4/13 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 4/13 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/13 statements.
# Partially parsed test_globals_invalid_assignment_target. Retrieved 4/12 statements.
# Partially parsed test_globals_multiple_targets. Retrieved 4/12 statements.
# Partially parsed test_globals_with_constant_value. Retrieved 4/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int = 5'
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.x']
    assert var_6 == '5'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONSTANT = 42'
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.CONSTANT'
    var_5 = bool('test_module.CONSTANT' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.CONSTANT'
    var_7 = bool('test_module.CONSTANT' in var_0.const)
    assert var_7 is True
    var_8 = var_0.const['test_module.CONSTANT']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "variable = 'hello'"
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.variable'
    var_5 = bool('test_module.variable' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.variable'
    var_7 = bool('test_module.variable' not in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "__all__ = ['func1', 'func2']"
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.func1'
    var_5 = bool('test_module.func1' in var_0.imp[var_2])
    assert var_5 is True
    var_6 = 'test_module.func2'
    var_7 = bool('test_module.func2' in var_0.imp[var_2])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "__all__ = ('func1', 'func2')"
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.func1'
    var_5 = bool('test_module.func1' in var_0.imp[var_2])
    assert var_5 is True
    var_6 = 'test_module.func2'
    var_7 = bool('test_module.func2' in var_0.imp[var_2])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'value = 10  # type: float'
    var_2 = 'test_module'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.value'
    var_6 = bool('test_module.value' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.const['test_module.value']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a, b = 1, 2'
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.a'
    var_5 = bool('test_module.a' not in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = y = 5'
    var_2 = 'test_module'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'NUM = 123'
    var_2 = 'test_module'
    var_3 = 0
    var_4 = var_0.root['test_module.NUM']
    var_5 = bool(var_0.root['test_module.NUM'] == var_2)
    assert var_5 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_func_api_basic_function. Retrieved 10/22 statements.
# Partially parsed test_func_api_with_self. Retrieved 9/21 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 8/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 10/22 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 10/19 statements.
# Partially parsed test_func_api_no_annotations. Retrieved 10/19 statements.
# Partially parsed test_func_api_kwonly_args. Retrieved 10/19 statements.
# Partially parsed test_func_api_posonly_args. Retrieved 10/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with basic function arguments.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = "def func(x: int, y: str = 'default') -> bool: pass"
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = 'bool'
    var_8 = False
    var_9 = False
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_2.doc)
    assert var_11 is True
    var_12 = '|'
    var_13 = bool('|' in var_2.doc['test_module.func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with self parameter.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = 'def method(self, x: int) -> None: pass'
    var_5 = 'test_module'
    var_6 = 'test_module.MyClass.method'
    var_7 = 'None'
    var_8 = False
    var_9 = 'test_module.MyClass.method'
    var_10 = bool('test_module.MyClass.method' in var_2.doc)
    assert var_10 is True
    var_11 = 'Self'
    var_12 = bool('Self' in var_2.doc['test_module.MyClass.method'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with classmethod decorator.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = 'def create(cls, value: int): pass'
    var_5 = 'test_module'
    var_6 = 'test_module.MyClass.create'
    var_7 = None
    var_8 = 'test_module.MyClass.create'
    var_9 = bool('test_module.MyClass.create' in var_2.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with *args and **kwargs.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = 'def func(*args, **kwargs) -> None: pass'
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = 'None'
    var_8 = False
    var_9 = False
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_2.doc)
    assert var_11 is True
    var_12 = '|'
    var_13 = bool('|' in var_2.doc['test_module.func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with default arguments.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = "def func(a: int, b: int = 10, c: str = 'hello'): pass"
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = None
    var_8 = False
    var_9 = False
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_2.doc)
    assert var_11 is True
    var_12 = 'return'
    var_13 = bool('return' in var_2.doc['test_module.func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with no type annotations.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = 'def func(x, y): pass'
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = None
    var_8 = False
    var_9 = False
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_2.doc)
    assert var_11 is True
    var_12 = '|'
    var_13 = bool('|' in var_2.doc['test_module.func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with keyword-only arguments.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = "def func(a: int, *, b: str = 'default'): pass"
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = None
    var_8 = False
    var_9 = False
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_2.doc)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test func_api with positional-only arguments.'
    var_1 = True
    var_2 = module_0.Parser(var_1, var_1)
    var_3 = 0
    var_4 = 'def func(a: int, /, b: str): pass'
    var_5 = 'test_module'
    var_6 = 'test_module.func'
    var_7 = None
    var_8 = False
    var_9 = False
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_2.doc)
    assert var_11 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_func_ann_predicate_line_15. Retrieved 10/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'param'
    var_3 = 0
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = []
    var_9 = False
    var_10 = False



# Parsed testcases at query #61
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '42'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "'hello'"
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'float'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'list[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "('a', 'b', 'c')"
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'tuple[str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '{1, 2, 3}'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'set[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "[1, 'a']"
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'list[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "{1: 'a', 2: 'b'}"
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'dict[int, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '[]'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool(1)'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "int('42')"
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'str(42)'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown_func()'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.parse(var_0)
    var_3 = var_2.body[var_1]
    var_4 = var_3.value
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'Any'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_class_api_line_25_predicate_false. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private_attr'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'test_class'
    var_7 = []
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc['test_class'])
    assert var_9 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_class_api_line_25_predicate_false. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (is_public_family(attr)) evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'TestClass'
    var_4 = []
    var_5 = '_private_attr'
    var_6 = None
    var_7 = []
    var_8 = 42
    var_9 = []
    var_10 = bool(var_3 not in var_1.doc or 'Members' not in var_1.doc[var_3])
    assert var_10 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 6/15 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 5/13 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/13 statements.
# Partially parsed test_globals_with_lowercase_name. Retrieved 5/13 statements.
# Partially parsed test_globals_with_all_list. Retrieved 6/18 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 6/18 statements.
# Partially parsed test_globals_ignores_invalid_nodes. Retrieved 6/16 statements.
# Partially parsed test_globals_with_annotated_assignment_without_value. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONSTANT'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module.MY_CONSTANT'
    var_9 = bool('test_module.MY_CONSTANT' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.MY_CONSTANT']
    assert var_10 == '42'
    var_11 = 'test_module.MY_CONSTANT'
    var_12 = bool('test_module.MY_CONSTANT' in var_0.const)
    assert var_12 is True
    var_13 = var_0.const['test_module.MY_CONSTANT']
    assert var_13 == 'int'
    var_14 = var_0.root['test_module.MY_CONSTANT']
    assert var_14 == 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT_VAR'
    var_3 = 'hello'
    var_4 = []
    var_5 = None
    var_6 = 'test_module.CONSTANT_VAR'
    var_7 = bool('test_module.CONSTANT_VAR' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.CONSTANT_VAR']
    assert var_8 == "'hello'"
    var_9 = 'test_module.CONSTANT_VAR'
    var_10 = bool('test_module.CONSTANT_VAR' in var_0.const)
    assert var_10 is True
    var_11 = var_0.const['test_module.CONSTANT_VAR']
    assert var_11 == 'str'
    var_12 = var_0.root['test_module.CONSTANT_VAR']
    assert var_12 == 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TYPED_VAR'
    var_3 = 100
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module.TYPED_VAR'
    var_7 = bool('test_module.TYPED_VAR' in var_0.const)
    assert var_7 is True
    var_8 = var_0.const['test_module.TYPED_VAR']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'regular_var'
    var_3 = 42
    var_4 = []
    var_5 = None
    var_6 = 'test_module.regular_var'
    var_7 = bool('test_module.regular_var' in var_0.alias)
    assert var_7 is True
    var_8 = 'test_module.regular_var'
    var_9 = bool('test_module.regular_var' not in var_0.root)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.func1'
    var_10 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.func2'
    var_12 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'ClassA'
    var_4 = []
    var_5 = 'ClassB'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.ClassA'
    var_10 = bool('test_module.ClassA' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.ClassB'
    var_12 = bool('test_module.ClassB' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'var1'
    var_3 = 'var2'
    var_4 = 42
    var_5 = []
    var_6 = None
    var_7 = 'test_module.var1'
    var_8 = bool('test_module.var1' not in var_0.alias)
    assert var_8 is True
    var_9 = 'test_module.var2'
    var_10 = bool('test_module.var2' not in var_0.alias)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'UNINITIALIZED'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module.UNINITIALIZED'
    var_8 = bool('test_module.UNINITIALIZED' not in var_0.alias)
    assert var_8 is True
    var_9 = 'test_module.UNINITIALIZED'
    var_10 = bool('test_module.UNINITIALIZED' not in var_0.const)
    assert var_10 is True



# Parsed testcases at query #65
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '42'
    var_1 = 'eval'
    var_2 = module_0.parse(var_0, mode=var_1)
    var_3 = var_2.body
    var_4 = None
    var_5 = [var_3, var_4, var_3]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = var_7[0]
    assert var_9 == '42'
    var_10 = var_7[1]
    assert var_10 == ' '
    var_11 = var_7[2]
    assert var_11 == '42'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_public_family_no_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_private_family_no_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_listed. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_not_listed. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_all_empty. Retrieved 3/5 statements.
# Partially parsed test_is_public_nested_module_in_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_nested_module_not_in_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_imp_key_matching. Retrieved 3/8 statements.
# Partially parsed test_is_public_magic_name_with_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_parent_in_all. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.public_func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule._private_func'
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
    var_1 = 'mymodule.submodule'
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
    var_1 = 'mymodule.submod'
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
    var_1 = 'mymodule.submod'
    var_2 = 'mymodule.submod.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_class_api_predicate_line_11_false. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = 'x = 5'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body
    var_6 = []
    var_7 = var_0.class_api(var_1, var_2, var_6, var_5)
    var_8 = var_0.doc[var_2]
    assert var_8 == ''



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.
# Partially parsed test_parser_constructor_empty_dicts. Retrieved 8/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = 2

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.level
    var_2 = var_0.doc
    var_3 = var_0.docstring
    var_4 = var_0.imp
    var_5 = var_0.root
    var_6 = var_0.alias
    var_7 = var_0.const



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_func_ann_with_self_and_regular_args. Retrieved 8/16 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 6/13 statements.
# Partially parsed test_func_ann_with_no_self. Retrieved 6/13 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 7/15 statements.
# Partially parsed test_func_ann_with_annotations. Retrieved 7/18 statements.
# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 8/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = 'y'
    var_8 = []
    var_9 = True
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = '*'
    var_6 = []
    var_7 = 'y'
    var_8 = []
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = []
    var_5 = 'y'
    var_6 = 'str'
    var_7 = []
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 'MyClass'
    var_4 = []
    var_5 = 'x'
    var_6 = 'int'
    var_7 = []
    var_8 = True
    var_9 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_class_api_enums_predicate. Retrieved 14/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = '\nclass TestEnum:\n    MEMBER1: int\n    MEMBER2: str\n'
    var_9 = module_1.parse(var_8)
    var_10 = 0
    var_11 = var_9.body[var_10]
    var_12 = var_11.body
    var_13 = var_0.class_api(var_1, var_2, var_7, var_12)
    var_14 = 'Enums'
    var_15 = bool('Enums' in var_0.doc[var_2])
    assert var_15 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_class_api_delete_non_name_target. Retrieved 5/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 1
    var_5 = []
    var_6 = var_0.doc[var_2]
    assert var_6 == '## class TestClass\n\n*Full name:* `test_module.TestClass`\n\n'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_api_link_false_predicate. Retrieved 16/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'test_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'test_module'
    var_14 = ''
    var_15 = 'test_module.test_func'
    var_16 = var_2.doc[var_15]
    var_17 = '<a id='
    var_18 = bool('<a id=' not in var_16)
    assert var_18 is True
    var_19 = bool('self.link' or var_2.link == False)
    assert var_19 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/14 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 5/14 statements.
# Partially parsed test_globals_with_annotated_no_value. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'x: int = 42'
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.x']
    assert var_7 == '42'
    var_8 = var_0.const['test_module.x']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = "y = 'hello'"
    var_5 = 'test_module.y'
    var_6 = bool('test_module.y' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.y']
    assert var_7 == "'hello'"
    var_8 = var_0.const['test_module.y']
    assert var_8 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'MAX_SIZE = 100'
    var_5 = 'test_module.MAX_SIZE'
    var_6 = bool('test_module.MAX_SIZE' in var_0.root)
    assert var_6 is True
    var_7 = var_0.root['test_module.MAX_SIZE']
    assert var_7 == 'test_module'
    var_8 = var_0.const['test_module.MAX_SIZE']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = "__all__ = ['func1', 'func2']"
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = "__all__ = ('func1', 'func2')"
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'z = 3.14  # type: float'
    var_5 = var_0.const['test_module.z']
    assert var_5 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'a = b = 10'
    var_5 = 'test_module.a'
    var_6 = bool('test_module.a' not in var_0.alias)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 0
    var_4 = 'w: str'
    var_5 = 'test_module.w'
    var_6 = bool('test_module.w' not in var_0.alias)
    assert var_6 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_class_api_predicate_line_19_false. Retrieved 8/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'y'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = None
    var_8 = 'test_module'
    var_9 = 'test_class'
    var_10 = []
    var_11 = 'test_class'
    var_12 = bool('test_class' in var_0.doc)
    assert var_12 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_compile_magic_method_continues. Retrieved 4/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.compile()
    var_4 = '__init__'
    var_5 = bool('__init__' not in var_3)
    assert var_5 is True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_class_api_enum_predicate. Retrieved 12/21 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = '\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.resolve
    var_8 = 'enum.Enum'
    var_9 = var_6.bases
    var_10 = var_6.body
    var_11 = var_0.class_api(var_1, var_2, var_9, var_10)
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc[var_2])
    assert var_13 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_func_api_predicate_false. Retrieved 12/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 5
    var_8 = []
    var_9 = 'root'
    var_10 = 'test_func'
    var_11 = False
    var_12 = 'test_func'
    var_13 = bool('test_func' in var_0.doc)
    assert var_13 is True
    var_14 = var_0.doc[var_10]
    var_15 = len(var_14)
    var_16 = bool(var_15 > 0)
    assert var_16 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_const_type_predicate_line_11_false. Retrieved 14/22 statements.


import ast as module_0

def test_case_0():
    var_0 = 'some_unknown_function()'
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = var_3.value
    var_5 = {}
    var_6 = var_4.func
    var_7 = module_0.unparse(var_6)
    var_8 = 'bool'
    var_9 = 'int'
    var_10 = 'float'
    var_11 = 'complex'
    var_12 = 'str'
    var_13 = {var_8, var_9, var_10, var_11, var_12}



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_tuple_of_strings. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_set_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_dict_int_str. Retrieved 4/13 statements.
# Partially parsed test_const_type_with_mixed_list_types. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_call_int. Retrieved 3/8 statements.
# Partially parsed test_const_type_with_call_str. Retrieved 3/8 statements.
# Partially parsed test_const_type_with_call_list. Retrieved 3/8 statements.
# Partially parsed test_const_type_with_non_constant_in_list. Retrieved 2/9 statements.
# Partially parsed test_const_type_with_none_constant. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 'a'
    var_5 = []
    var_6 = 'b'
    var_7 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'mixed'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'list'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = None
    var_1 = []



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_visit_Constant_syntax_error. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not valid python syntax !!!'
    var_4 = []



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_class_api_enum_predicate. Retrieved 20/30 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'RED: int = 1'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = []
    var_7 = [var_5]
    var_8 = 'class TestEnum(enum.Enum):\n    RED: int = 1'
    var_9 = module_1.parse(var_8)
    var_10 = var_9.body[var_1]
    var_11 = var_2.resolve
    var_12 = 'enum.Enum'
    var_13 = 'test_module'
    var_14 = 'test_module.TestEnum'
    var_15 = var_10.bases
    var_16 = var_10.body
    var_17 = var_2.class_api(var_13, var_14, var_15, var_16)
    var_18 = var_2.doc[var_14]
    var_19 = 'Enum'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_func_api_predicate_line_32_false. Retrieved 10/37 statements.


def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = [var_2]
    var_10 = 'return'
    var_11 = [var_10, var_2]



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_none. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_list_of_mixed_types. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_empty_tuple. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_tuple_of_strings. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_empty_set. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_set_of_floats. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_empty_dict. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_dict_of_constants. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_dict_mixed_keys. Retrieved 3/11 statements.
# Partially parsed test_const_type_with_call_to_int. Retrieved 2/5 statements.
# Partially parsed test_const_type_with_call_to_str. Retrieved 2/5 statements.
# Partially parsed test_const_type_with_call_to_bool. Retrieved 2/5 statements.
# Partially parsed test_const_type_with_call_to_list. Retrieved 2/5 statements.
# Partially parsed test_const_type_with_unknown_node. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_list_containing_non_constant. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_tuple_containing_non_constant. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = None
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'str'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1.0
    var_1 = []
    var_2 = 2.0
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 'key'
    var_1 = []
    var_2 = 1
    var_3 = []

def test_case_0():
    var_0 = 'key'
    var_1 = []
    var_2 = 1
    var_3 = []
    var_4 = []
    var_5 = 2
    var_6 = []

def test_case_0():
    var_0 = 'int(5)'
    var_1 = 'eval'

def test_case_0():
    var_0 = 'str(5)'
    var_1 = 'eval'

def test_case_0():
    var_0 = 'bool(1)'
    var_1 = 'eval'

def test_case_0():
    var_0 = 'list()'
    var_1 = 'eval'

def test_case_0():
    var_0 = 'x'
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'y'
    var_3 = []



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_e_type_with_elements. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to False when elements are provided.'
    var_1 = 1
    var_2 = []
    var_3 = 2
    var_4 = []



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_api_link_false. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'def my_func(): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_0]
    var_6 = 'test_module'
    var_7 = var_2.api(var_6, var_5)
    var_8 = '\n<a id="{}"></a>'
    var_9 = bool('\n<a id="{}"></a>' not in var_2.doc['test_module.my_func'])
    assert var_9 is True



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_func_ann_line_12_predicate_true. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'test_module'
    var_5 = True
    var_6 = False



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_func_ann_with_self_parameter. Retrieved 8/16 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 8/18 statements.
# Partially parsed test_func_ann_without_self. Retrieved 7/15 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 8/17 statements.
# Partially parsed test_func_ann_with_annotations. Retrieved 9/21 statements.
# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 9/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = 'return'
    var_5 = 'root'
    var_6 = True
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'type'
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = 'return'
    var_7 = 'root'
    var_8 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'return'
    var_5 = 'root'
    var_6 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = '*'
    var_4 = 'y'
    var_5 = 'return'
    var_6 = 'root'
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'int'
    var_2 = []
    var_3 = 'str'
    var_4 = []
    var_5 = 'x'
    var_6 = 'y'
    var_7 = 'return'
    var_8 = None
    var_9 = 'root'
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MyClass'
    var_2 = []
    var_3 = 'self'
    var_4 = 'x'
    var_5 = None
    var_6 = 'return'
    var_7 = 'root'
    var_8 = True
    var_9 = False



