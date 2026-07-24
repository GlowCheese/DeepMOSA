####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = module_0._m(*var_1)
    assert var_2 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'submodule'
    var_2 = 'component'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._m(*var_3)
    assert var_4 == 'module.submodule.component'

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'module'
    var_2 = 'submodule'
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0._m(*var_3)
    assert var_4 == 'module.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._m(*var_1)
    assert var_2 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._m(*var_0)
    assert var_1 == ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_imports_with_Import_node. Retrieved 6/11 statements.
# Partially parsed test_imports_with_ImportFrom_node_no_level. Retrieved 6/10 statements.
# Partially parsed test_imports_with_ImportFrom_node_with_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = var_0.alias['test.module.os']
    assert var_6 == 'os'
    var_7 = var_0.alias['test.module.system']
    assert var_7 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['test.module.join']
    assert var_6 == 'os.path.join'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.sub'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['test.module.sub.path']
    assert var_6 == 'test.module.os.path'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_compile_empty_parser. Retrieved 2/4 statements.
# Partially parsed test_compile_with_toc. Retrieved 1/6 statements.
# Partially parsed test_compile_with_docstring. Retrieved 2/8 statements.
# Partially parsed test_compile_with_magic_name. Retrieved 2/7 statements.
# Partially parsed test_compile_with_non_public_name. Retrieved 2/7 statements.
# Partially parsed test_compile_with_nested_module. Retrieved 2/10 statements.
# Partially parsed test_compile_with_constants. Retrieved 3/10 statements.
# Partially parsed test_compile_with_link. Retrieved 2/7 statements.
# Partially parsed test_compile_with_missing_docstring. Retrieved 2/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test.CONST'

def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = False
    assert var_0 == '# Test\n'
    var_1 = 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_visit_Constant_with_valid_name. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Constant(*var_6, **var_7)
    var_9 = var_4.visit_Constant(var_8)
    var_10 = var_9.id
    assert var_10 == 'alias'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = 42
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = {var_17: var_5, var_18: var_10, var_19: var_15}
    var_21 = module_1.AnnAssign(*var_16, **var_20)
    var_22 = 'root'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = var_0.alias['root.test_var']
    assert var_24 == '42'
    var_25 = var_0.const['root.TEST_VAR']
    assert var_25 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'root'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['root.test_var']
    assert var_19 == '42'
    var_20 = var_0.const['root.TEST_VAR']
    assert var_20 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'public_func'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.List(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_6, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'root'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = 'root.public_func'
    var_25 = bool('root.public_func' in var_0.imp['root'])
    assert var_25 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'b'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = [var_5, var_10]
    var_12 = []
    var_13 = 'elts'
    var_14 = {var_13: var_11}
    var_15 = module_1.Tuple(*var_12, **var_14)
    var_16 = [var_15]
    var_17 = 42
    var_18 = []
    var_19 = 'value'
    var_20 = {var_19: var_17}
    var_21 = module_1.Constant(*var_18, **var_20)
    var_22 = []
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_16, var_24: var_21}
    var_26 = module_1.Assign(*var_22, **var_25)
    var_27 = 'root'
    var_28 = var_0.globals(var_27, var_26)
    var_29 = 'root.a'
    var_30 = bool('root.a' not in var_0.alias)
    assert var_30 is True
    var_31 = 'root.b'
    var_32 = bool('root.b' not in var_0.alias)
    assert var_32 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_func_api_no_args_no_return. Retrieved 11/13 statements.
# Partially parsed test_func_api_with_args. Retrieved 14/16 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 16/18 statements.
# Partially parsed test_func_api_with_self. Retrieved 15/17 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 14/16 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 16/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = '| return |\n|:---:|\n|  |\n\n'
    var_12 = var_0.doc['name']
    var_13 = bool(var_0.doc['name'] == var_11)
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = None
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False
    var_16 = '| x | return |\n|:---:|:---:|\n|  |  |\n\n'
    var_17 = var_0.doc['name']
    var_18 = bool(var_0.doc['name'] == var_16)
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = 1
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_10}
    var_14 = module_1.Constant(*var_11, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = None
    var_18 = 'root'
    var_19 = 'name'
    var_20 = False
    var_21 = '| x | return |\n|:---:|:---:|\n| `1` |  |\n\n'
    var_22 = var_0.doc['name']
    var_23 = bool(var_0.doc['name'] == var_21)
    assert var_23 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = None
    var_13 = 'root'
    var_14 = 'name'
    var_15 = True
    var_16 = False
    var_17 = '| self | return |\n|:---:|:---:|\n| `Self` |  |\n\n'
    var_18 = var_0.doc['name']
    var_19 = bool(var_0.doc['name'] == var_17)
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = None
    var_13 = 'root'
    var_14 = 'name'
    var_15 = True
    var_16 = '| cls | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n'
    var_17 = var_0.doc['name']
    var_18 = bool(var_0.doc['name'] == var_16)
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = 'int'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = [var_2, var_7]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'str'
    var_17 = []
    var_18 = 'id'
    var_19 = {var_18: var_16}
    var_20 = module_1.Name(*var_17, **var_19)
    var_21 = 'root'
    var_22 = 'name'
    var_23 = False
    var_24 = '| x | return |\n|:---:|:---:|\n| `int` | `str` |\n\n'
    var_25 = var_0.doc['name']
    var_26 = bool(var_0.doc['name'] == var_24)
    assert var_26 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_public_with_all_listed. Retrieved 10/13 statements.
# Partially parsed test_is_public_without_all_listed. Retrieved 8/11 statements.
# Partially parsed test_is_public_private_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_not_in_all. Retrieved 7/10 statements.
# Partially parsed test_is_public_magic_name. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = 'mod1.submod'
    var_4 = {var_2, var_3}
    var_5 = 'pkg.mod1'
    var_6 = 'pkg.mod1.submod'
    var_7 = ''
    var_8 = var_0.is_public(var_5)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = var_0.is_public(var_6)
    var_11 = bool(var_10)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.mod1'
    var_4 = 'pkg.mod1.submod'
    var_5 = ''
    var_6 = var_0.is_public(var_3)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = var_0.is_public(var_4)
    var_9 = bool(var_8)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg._private'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    var_6 = bool(not var_5)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = {var_2}
    var_4 = 'pkg.mod2'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    var_7 = bool(not var_6)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.__init__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_1.Name(*var_3, **var_5)
    var_7 = [var_6]
    var_8 = 42
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = None
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = 'type_comment'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_1.Assign(*var_14, **var_18)
    var_20 = var_0.globals(var_1, var_19)
    var_21 = set()
    var_22 = var_0.imp[var_1]
    var_23 = bool(var_0.imp[var_1] == var_21)
    assert var_23 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parser_constructor_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_func_api_with_vararg. Retrieved 14/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'args'
    var_9 = None
    var_10 = []
    var_11 = 'arg'
    var_12 = 'annotation'
    var_13 = {var_11: var_8, var_12: var_9}
    var_14 = module_1.arg(*var_10, **var_13)
    var_15 = []
    var_16 = None
    var_17 = False
    var_18 = False
    var_19 = bool(True)
    assert var_19 is True



# Parsed testcases at query #12
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._e_type(*var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = [var_2]
    var_4 = module_0._e_type(*var_3)
    assert var_4 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = module_1._e_type(*var_5)
    assert var_6 == '[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = module_1._e_type(*var_9)
    assert var_10 == '[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = module_1._e_type(*var_9)
    assert var_10 == '[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Constant(*var_10, **var_11)
    var_13 = 'b'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Constant(*var_14, **var_15)
    var_17 = [var_12, var_16]
    var_18 = [var_8, var_17]
    var_19 = module_1._e_type(*var_18)
    assert var_19 == '[int, str]'



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = '__main__'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = '__name__'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.path'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = 'sys.argv'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'math.sqrt'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False
    var_2 = 'os._path'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is False
    var_4 = 'sys._argv'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_local'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False
    var_2 = 'module._local'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is False
    var_4 = 'package.module._local'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = 'sys.__main__'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'math.__name__'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = 'os._path.__init__'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is False
    var_8 = 'sys._argv.__main__'
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is False
    var_10 = 'math._sqrt.__name__'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_imports_with_Import_node. Retrieved 4/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'pkg'
    var_4 = var_0.alias['pkg.os']
    assert var_4 == 'os'



# Parsed testcases at query #15
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'foo'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.Tuple(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_6, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'root'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = '__all__'
    var_25 = bool('__all__' in var_0.imp['root'])
    assert var_25 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_visit_Name_self_ty. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 10/11 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 8/9 statements.
# Partially parsed test_visit_Name_with_TypeVar_alias. Retrieved 10/11 statements.
# Partially parsed test_visit_Name_with_non_TypeVar_call_alias. Retrieved 10/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = [var_2, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    var_10 = var_3.visit_Name(var_9)
    var_11 = var_10.id
    assert var_11 == 'Self'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias.value'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_1.Name(*var_10, **var_11)
    var_13 = var_5.visit_Name(var_12)
    var_14 = var_13.id
    assert var_14 == 'value'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'name'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_1.Name(*var_8, **var_9)
    var_11 = var_3.visit_Name(var_10)
    var_12 = var_11.id
    assert var_12 == 'name'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_1.Name(*var_10, **var_11)
    var_13 = var_5.visit_Name(var_12)
    var_14 = var_13.id
    assert var_14 == 'name'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'some_func()'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_1.Name(*var_10, **var_11)
    var_13 = var_5.visit_Name(var_12)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_docstring_basic. Retrieved 14/17 statements.
# Partially parsed test_load_docstring_nested. Retrieved 19/22 statements.
# Partially parsed test_load_docstring_none_doc. Retrieved 14/16 statements.
# Partially parsed test_load_docstring_partial_match. Retrieved 15/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'Module `pkg.module`'
    var_4 = 'func()'
    var_5 = 'module'
    var_6 = ()
    var_7 = 'func'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = {var_7: var_9}
    var_11 = type(var_5, var_6, var_10)
    var_12 = var_11()
    var_13 = var_0.load_docstring(var_1, var_12)
    var_14 = var_0.docstring['pkg.module.func']
    assert var_14 == '```python\nFunction doc\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.Class.method'
    var_3 = 'Module `pkg.module`'
    var_4 = 'method()'
    var_5 = 'module'
    var_6 = ()
    var_7 = 'Class'
    var_8 = ()
    var_9 = 'method'
    var_10 = None
    var_11 = lambda : var_10
    var_12 = {var_9: var_11}
    var_13 = type(var_7, var_8, var_12)
    var_14 = var_13()
    var_15 = {var_7: var_14}
    var_16 = type(var_5, var_6, var_15)
    var_17 = var_16()
    var_18 = var_0.load_docstring(var_1, var_17)
    var_19 = var_0.docstring['pkg.module.Class.method']
    assert var_19 == '```python\nMethod doc\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'Module `pkg.module`'
    var_4 = 'func()'
    var_5 = 'module'
    var_6 = ()
    var_7 = 'func'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = {var_7: var_9}
    var_11 = type(var_5, var_6, var_10)
    var_12 = var_11()
    var_13 = var_0.load_docstring(var_1, var_12)
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {})
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'pkg.other.func'
    var_4 = 'Module `pkg.module`'
    var_5 = 'func()'
    var_6 = 'module'
    var_7 = ()
    var_8 = 'func'
    var_9 = None
    var_10 = lambda : var_9
    var_11 = {var_8: var_10}
    var_12 = type(var_6, var_7, var_11)
    var_13 = var_12()
    var_14 = var_0.load_docstring(var_1, var_13)
    var_15 = var_0.docstring
    var_16 = bool(var_0.docstring == {'pkg.module.func': '```python\nFunction doc\n```'})
    assert var_16 is True
    var_17 = 'pkg.other.func'
    var_18 = bool('pkg.other.func' not in var_0.docstring)
    assert var_18 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 8/12 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/11 statements.
# Partially parsed test_globals_with_all. Retrieved 9/14 statements.
# Partially parsed test_globals_with_non_uppercase. Retrieved 5/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = 1
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = 'test'
    var_17 = var_0.alias['test.x']
    assert var_17 == '1'
    var_18 = var_0.const['test.x']
    assert var_18 == 'int'
    var_19 = var_0.root['test.x']
    assert var_19 == 'test'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'hello'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'test'
    var_8 = var_0.alias['test.y']
    assert var_8 == "'hello'"
    var_9 = var_0.const['test.y']
    assert var_9 == 'str'
    var_10 = var_0.root['test.y']
    assert var_10 == 'test'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3.14
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'float'
    var_8 = 'test'
    var_9 = var_0.alias['test.z']
    assert var_9 == '3.14'
    var_10 = var_0.const['test.z']
    assert var_10 == 'float'
    var_11 = var_0.root['test.z']
    assert var_11 == 'test'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'func1'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'func2'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_6, var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.List(*var_13, **var_15)
    var_17 = 'test'
    var_18 = var_0.imp['test']
    var_19 = bool(var_0.imp['test'] == {'test.func1', 'test.func2'})
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'non_upper'
    var_2 = 42
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'test'
    var_8 = var_0.alias['test.non_upper']
    assert var_8 == '42'
    var_9 = 'test.non_upper'
    var_10 = bool('test.non_upper' not in var_0.const)
    assert var_10 is True
    var_11 = 'test.non_upper'
    var_12 = bool('test.non_upper' not in var_0.root)
    assert var_12 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_public_with_all_listed. Retrieved 10/13 statements.
# Partially parsed test_is_public_without_all_listed. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_nested_public. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_nested_private. Retrieved 7/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'func'
    var_3 = 'cls'
    var_4 = {var_2, var_3}
    var_5 = 'pkg.func'
    var_6 = 'pkg.cls'
    var_7 = ''
    var_8 = var_0.is_public(var_5)
    assert var_8 is True
    var_9 = var_0.is_public(var_6)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.func'
    var_4 = 'pkg.cls'
    var_5 = ''
    var_6 = var_0.is_public(var_3)
    assert var_6 is True
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg._private'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.sub'
    var_4 = 'pkg.sub.func'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.sub'
    var_4 = 'pkg.sub._private'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True



# Parsed testcases at query #21
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'name'
    var_7 = 'bases'
    var_8 = 'body'
    var_9 = 'decorator_list'
    var_10 = {var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_1.ClassDef(*var_5, **var_10)
    var_12 = 'root'
    var_13 = 'root.TestClass'
    var_14 = var_11.bases
    var_15 = var_11.body
    var_16 = var_0.class_api(var_12, var_13, var_14, var_15)
    var_17 = var_0.doc['root.TestClass']
    assert var_17 == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = 'Base1'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_1.Name(*var_3, **var_5)
    var_7 = 'Base2'
    var_8 = []
    var_9 = 'id'
    var_10 = {var_9: var_7}
    var_11 = module_1.Name(*var_8, **var_10)
    var_12 = [var_6, var_11]
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'name'
    var_17 = 'bases'
    var_18 = 'body'
    var_19 = 'decorator_list'
    var_20 = {var_16: var_1, var_17: var_12, var_18: var_13, var_19: var_14}
    var_21 = module_1.ClassDef(*var_15, **var_20)
    var_22 = 'root'
    var_23 = 'root.TestClass'
    var_24 = var_21.bases
    var_25 = var_21.body
    var_26 = var_0.class_api(var_22, var_23, var_24, var_25)
    var_27 = var_0.doc['root.TestClass']
    assert var_27 == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n| Bases |\n|:---:|\n| `Base1` |\n| `Base2` |\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'attr1'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'int'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = None
    var_14 = []
    var_15 = 'target'
    var_16 = 'annotation'
    var_17 = 'value'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_1.AnnAssign(*var_14, **var_18)
    var_20 = 'attr2'
    var_21 = []
    var_22 = 'id'
    var_23 = {var_22: var_20}
    var_24 = module_1.Name(*var_21, **var_23)
    var_25 = 'str'
    var_26 = []
    var_27 = 'id'
    var_28 = {var_27: var_25}
    var_29 = module_1.Name(*var_26, **var_28)
    var_30 = []
    var_31 = 'target'
    var_32 = 'annotation'
    var_33 = 'value'
    var_34 = {var_31: var_24, var_32: var_29, var_33: var_13}
    var_35 = module_1.AnnAssign(*var_30, **var_34)
    var_36 = [var_19, var_35]
    var_37 = []
    var_38 = []
    var_39 = 'name'
    var_40 = 'bases'
    var_41 = 'body'
    var_42 = 'decorator_list'
    var_43 = {var_39: var_1, var_40: var_2, var_41: var_36, var_42: var_37}
    var_44 = module_1.ClassDef(*var_38, **var_43)
    var_45 = 'root'
    var_46 = 'root.TestClass'
    var_47 = var_44.bases
    var_48 = var_44.body
    var_49 = var_0.class_api(var_45, var_46, var_47, var_48)
    var_50 = var_0.doc['root.TestClass']
    assert var_50 == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestEnum'
    var_2 = 'enum.Enum'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_1.Name(*var_3, **var_5)
    var_7 = [var_6]
    var_8 = 'A'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = None
    var_14 = 1
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = []
    var_20 = 'target'
    var_21 = 'annotation'
    var_22 = 'value'
    var_23 = {var_20: var_12, var_21: var_13, var_22: var_18}
    var_24 = module_1.AnnAssign(*var_19, **var_23)
    var_25 = 'B'
    var_26 = []
    var_27 = 'id'
    var_28 = {var_27: var_25}
    var_29 = module_1.Name(*var_26, **var_28)
    var_30 = 2
    var_31 = []
    var_32 = 'value'
    var_33 = {var_32: var_30}
    var_34 = module_1.Constant(*var_31, **var_33)
    var_35 = []
    var_36 = 'target'
    var_37 = 'annotation'
    var_38 = 'value'
    var_39 = {var_36: var_29, var_37: var_13, var_38: var_34}
    var_40 = module_1.AnnAssign(*var_35, **var_39)
    var_41 = [var_24, var_40]
    var_42 = []
    var_43 = []
    var_44 = 'name'
    var_45 = 'bases'
    var_46 = 'body'
    var_47 = 'decorator_list'
    var_48 = {var_44: var_1, var_45: var_7, var_46: var_41, var_47: var_42}
    var_49 = module_1.ClassDef(*var_43, **var_48)
    var_50 = 'root'
    var_51 = 'root.TestEnum'
    var_52 = var_49.bases
    var_53 = var_49.body
    var_54 = var_0.class_api(var_50, var_51, var_52, var_53)
    var_55 = var_0.doc['root.TestEnum']
    assert var_55 == '### class TestEnum\n\n*Full name:* `root.TestEnum`\n<a id="root-testenum"></a>\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n| Enums |\n|:---:|\n| A |\n| B |\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'attr1'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'int'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = None
    var_14 = []
    var_15 = 'target'
    var_16 = 'annotation'
    var_17 = 'value'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_1.AnnAssign(*var_14, **var_18)
    var_20 = []
    var_21 = 'id'
    var_22 = {var_21: var_3}
    var_23 = module_1.Name(*var_20, **var_22)
    var_24 = [var_23]
    var_25 = []
    var_26 = 'targets'
    var_27 = {var_26: var_24}
    var_28 = module_1.Delete(*var_25, **var_27)
    var_29 = [var_19, var_28]
    var_30 = []
    var_31 = []
    var_32 = 'name'
    var_33 = 'bases'
    var_34 = 'body'
    var_35 = 'decorator_list'
    var_36 = {var_32: var_1, var_33: var_2, var_34: var_29, var_35: var_30}
    var_37 = module_1.ClassDef(*var_31, **var_36)
    var_38 = 'root'
    var_39 = 'root.TestClass'
    var_40 = var_37.bases
    var_41 = var_37.body
    var_42 = var_0.class_api(var_38, var_39, var_40, var_41)
    var_43 = var_0.doc['root.TestClass']
    assert var_43 == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = '_private'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'int'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = None
    var_14 = []
    var_15 = 'target'
    var_16 = 'annotation'
    var_17 = 'value'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_1.AnnAssign(*var_14, **var_18)
    var_20 = 'public'
    var_21 = []
    var_22 = 'id'
    var_23 = {var_22: var_20}
    var_24 = module_1.Name(*var_21, **var_23)
    var_25 = 'str'
    var_26 = []
    var_27 = 'id'
    var_28 = {var_27: var_25}
    var_29 = module_1.Name(*var_26, **var_28)
    var_30 = []
    var_31 = 'target'
    var_32 = 'annotation'
    var_33 = 'value'
    var_34 = {var_31: var_24, var_32: var_29, var_33: var_13}
    var_35 = module_1.AnnAssign(*var_30, **var_34)
    var_36 = [var_19, var_35]
    var_37 = []
    var_38 = []
    var_39 = 'name'
    var_40 = 'bases'
    var_41 = 'body'
    var_42 = 'decorator_list'
    var_43 = {var_39: var_1, var_40: var_2, var_41: var_36, var_42: var_37}
    var_44 = module_1.ClassDef(*var_38, **var_43)
    var_45 = 'root'
    var_46 = 'root.TestClass'
    var_47 = var_44.bases
    var_48 = var_44.body
    var_49 = var_0.class_api(var_45, var_46, var_47, var_48)
    var_50 = var_0.doc['root.TestClass']
    assert var_50 == '### class TestClass\n\n*Full name:* `root.TestClass`\n<a id="root-testclass"></a>\n\n| Members | Type |\n|:---:|:---:|\n| `public` | `str` |\n'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_class_api_with_enum. Retrieved 15/23 statements.
# Partially parsed test_class_api_with_members. Retrieved 13/21 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 9/18 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 9/14 statements.
# Partially parsed test_class_api_with_assign_member. Retrieved 7/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = var_0.class_api(var_1, var_2, var_12, var_13)
    var_15 = 'Bases'
    var_16 = bool('Bases' in var_0.doc[var_2])
    assert var_16 is True
    var_17 = 'BaseClass'
    var_18 = bool('BaseClass' in var_0.doc[var_2])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'VALUE1'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    var_23 = None
    var_24 = 'VALUE2'
    var_25 = []
    var_26 = {}
    var_27 = module_1.Load(*var_25, **var_26)
    var_28 = []
    var_29 = 'id'
    var_30 = 'ctx'
    var_31 = {var_29: var_14, var_30: var_27}
    var_32 = module_1.Name(*var_28, **var_31)
    var_33 = 'Enums'
    var_34 = bool('Enums' in var_0.doc[var_2])
    assert var_34 is True
    var_35 = 'VALUE1'
    var_36 = bool('VALUE1' in var_0.doc[var_2])
    assert var_36 is True
    var_37 = 'VALUE2'
    var_38 = bool('VALUE2' in var_0.doc[var_2])
    assert var_38 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'attr1'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 'attr2'
    var_16 = 'str'
    var_17 = []
    var_18 = {}
    var_19 = module_1.Load(*var_17, **var_18)
    var_20 = []
    var_21 = 'id'
    var_22 = 'ctx'
    var_23 = {var_21: var_16, var_22: var_19}
    var_24 = module_1.Name(*var_20, **var_23)
    var_25 = 'Members'
    var_26 = bool('Members' in var_0.doc[var_2])
    assert var_26 is True
    var_27 = 'attr1'
    var_28 = bool('attr1' in var_0.doc[var_2])
    assert var_28 is True
    var_29 = 'attr2'
    var_30 = bool('attr2' in var_0.doc[var_2])
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'attr1'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 'attr1'
    var_16 = bool('attr1' not in var_0.doc[var_2])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = '_private_attr'
    var_16 = bool('_private_attr' not in var_0.doc[var_2])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'attr1'
    var_5 = 42
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_1.Constant(*var_6, **var_8)
    var_10 = 'attr1'
    var_11 = bool('attr1' in var_0.doc[var_2])
    assert var_11 is True
    var_12 = 'int'
    var_13 = bool('int' in var_0.doc[var_2])
    assert var_13 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_docstring_basic. Retrieved 14/18 statements.
# Partially parsed test_load_docstring_nested. Retrieved 19/23 statements.
# Partially parsed test_load_docstring_none_doc. Retrieved 13/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.func'
    var_2 = 'pkg.submod'
    var_3 = 'Function doc'
    var_4 = 'Module doc'
    var_5 = 'test_module'
    var_6 = ()
    var_7 = 'func'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = {var_7: var_9}
    var_11 = type(var_5, var_6, var_10)
    var_12 = var_11()
    var_13 = var_0.load_docstring(var_2, var_12)
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {'pkg.submod.func': '```python\nNew function doc\n```'})
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.Class.method'
    var_2 = 'pkg.submod.Class'
    var_3 = 'Method doc'
    var_4 = 'Class doc'
    var_5 = 'pkg.submod'
    var_6 = 'test_module'
    var_7 = ()
    var_8 = 'Class'
    var_9 = ()
    var_10 = 'method'
    var_11 = None
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}
    var_14 = type(var_8, var_9, var_13)
    var_15 = {var_8: var_14}
    var_16 = type(var_6, var_7, var_15)
    var_17 = var_16()
    var_18 = var_0.load_docstring(var_5, var_17)
    var_19 = var_0.docstring
    var_20 = bool(var_0.docstring == {'pkg.submod.Class.method': '```python\nNew method doc\n```'})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.func'
    var_2 = 'Function doc'
    var_3 = 'pkg.submod'
    var_4 = 'test_module'
    var_5 = ()
    var_6 = 'func'
    var_7 = None
    var_8 = lambda : var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = var_0.load_docstring(var_3, var_11)
    var_13 = var_0.docstring
    var_14 = bool(var_0.docstring == {})
    assert var_14 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_enum_predicate. Retrieved 6/87 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr'
    var_2 = 'annotation'
    var_3 = 'root'
    var_4 = 'name'
    var_5 = 'enum.Enum'
    var_6 = 'attr'
    var_7 = bool('attr' in var_0.enums)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_public_with_all_listed. Retrieved 10/14 statements.
# Partially parsed test_is_public_without_all. Retrieved 6/10 statements.
# Partially parsed test_is_public_private_name. Retrieved 6/10 statements.
# Partially parsed test_is_public_magic_name. Retrieved 6/10 statements.
# Partially parsed test_is_public_not_in_all. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = 'mod2'
    var_4 = {var_2, var_3}
    var_5 = 'pkg.mod1'
    var_6 = 'pkg.mod2'
    var_7 = ''
    var_8 = var_0.is_public(var_5)
    assert var_8 is True
    var_9 = var_0.is_public(var_6)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.mod1'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg._mod1'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.__init__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = {var_2}
    var_4 = 'pkg.mod2'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_is_public_with_imp_and_valid_child. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod'
    var_3 = {var_2}
    var_4 = 'pkg.mod'
    var_5 = 'doc'
    var_6 = var_0.is_public(var_1)
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_const_type_call_bool. Retrieved 3/5 statements.
# Partially parsed test_const_type_call_int. Retrieved 3/5 statements.
# Partially parsed test_const_type_call_unknown. Retrieved 3/5 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 3.14
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'float'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Set(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'set[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Set(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'set[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 'a'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 'b'
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[int, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 'b'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 2
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[Any, Any]'

import ast as module_0

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = []

import ast as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = []

import ast as module_0

def test_case_0():
    var_0 = 'unknown'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = []

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.Tuple(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'tuple[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.List(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'list[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.Set(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'set[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'keys'
    var_4 = 'values'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Dict(*var_2, **var_5)
    var_7 = module_1.const_type(var_6)
    assert var_7 == 'dict[, ]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = []
    var_8 = 'elts'
    var_9 = {var_8: var_6}
    var_10 = module_0.Tuple(*var_7, **var_9)
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'x'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'Any'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_walk_body_try_statement. Retrieved 13/19 statements.
# Partially parsed test_walk_body_mixed_statements. Retrieved 16/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = module_1.walk_body(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [var_2])
    assert var_6 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.stmt(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.If(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = module_1.walk_body(var_11)
    var_13 = list(var_12)
    var_14 = []
    var_15 = {}
    var_16 = module_0.stmt(*var_14, **var_15)
    var_17 = []
    var_18 = {}
    var_19 = module_0.stmt(*var_17, **var_18)
    var_20 = [var_16, var_19]
    var_21 = bool(var_13 == var_20)
    assert var_21 is True

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.stmt(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = {}
    var_10 = module_0.stmt(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = {}
    var_14 = module_0.stmt(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = {}
    var_18 = module_0.stmt(*var_16, **var_17)
    var_19 = []
    var_20 = {}
    var_21 = module_0.stmt(*var_19, **var_20)
    var_22 = []
    var_23 = {}
    var_24 = module_0.stmt(*var_22, **var_23)
    var_25 = []
    var_26 = {}
    var_27 = module_0.stmt(*var_25, **var_26)
    var_28 = [var_18, var_21, var_24, var_27]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.stmt(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.If(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = {}
    var_14 = module_0.stmt(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = [var_11, var_15]
    var_17 = {}
    var_18 = module_0.If(*var_16, **var_17)
    var_19 = [var_18]
    var_20 = module_1.walk_body(var_19)
    var_21 = list(var_20)
    var_22 = []
    var_23 = {}
    var_24 = module_0.stmt(*var_22, **var_23)
    var_25 = []
    var_26 = {}
    var_27 = module_0.stmt(*var_25, **var_26)
    var_28 = []
    var_29 = {}
    var_30 = module_0.stmt(*var_28, **var_29)
    var_31 = [var_24, var_27, var_30]
    var_32 = bool(var_21 == var_31)
    assert var_32 is True

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.stmt(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.If(*var_8, **var_9)
    var_11 = []
    var_12 = {}
    var_13 = module_0.stmt(*var_11, **var_12)
    var_14 = [var_13]
    var_15 = []
    var_16 = {}
    var_17 = module_0.stmt(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = []
    var_22 = {}
    var_23 = module_0.stmt(*var_21, **var_22)
    var_24 = []
    var_25 = {}
    var_26 = module_0.stmt(*var_24, **var_25)
    var_27 = []
    var_28 = {}
    var_29 = module_0.stmt(*var_27, **var_28)
    var_30 = []
    var_31 = {}
    var_32 = module_0.stmt(*var_30, **var_31)
    var_33 = [var_23, var_26, var_29, var_32]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_imports_with_importfrom_node. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os.path'
    var_2 = 'join'
    var_3 = 'j'
    var_4 = 0
    var_5 = 'pkg'
    var_6 = var_0.alias['pkg.j']
    assert var_6 == 'os.path.join'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True
    var_3 = var_1.toc
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'root'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = 'root.x'
    var_20 = bool('root.x' not in var_0.imp['root'])
    assert var_20 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_api_function. Retrieved 5/6 statements.
# Partially parsed test_api_async_function. Retrieved 5/6 statements.
# Partially parsed test_api_class. Retrieved 5/6 statements.
# Partially parsed test_api_with_prefix. Retrieved 5/6 statements.
# Partially parsed test_api_with_decorators. Retrieved 7/10 statements.
# Partially parsed test_api_with_docstring. Retrieved 7/10 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'def func(): pass'
    var_4 = module_0.parse(var_2, var_3)

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'async def async_func(): pass'
    var_4 = module_0.parse(var_2, var_3)

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'class MyClass: pass'
    var_4 = module_0.parse(var_2, var_3)

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = '\nclass MyClass:\n    def method(self):\n        pass\n'
    var_4 = module_0.parse(var_2, var_3)

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = '\n@decorator\ndef decorated_func():\n    pass\n'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'Decorators'
    var_6 = 'test.decorated_func'
    var_7 = '### decorated_func()\n\n*Full name:* `test.decorated_func`\n\n'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = '\ndef documented_func():\n    """This is a docstring."""\n    pass\n'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'test.documented_func'
    var_6 = '### documented_func()\n\n*Full name:* `test.documented_func`\n\n'
    var_7 = 'This is a docstring.'



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.docstring['test_module']
    assert var_10 == '```python\n"""Module docstring."""\n```'
    var_11 = var_0.docstring['test_module.foo']
    assert var_11 == '```python\n"""Function docstring."""\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nfrom typing import List\nimport os\nx = 1\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'typing.List'
    var_5 = bool('typing.List' in var_0.alias['test_module.List'])
    assert var_5 is True
    var_6 = 'os'
    var_7 = bool('os' in var_0.alias['test_module.os'])
    assert var_7 is True
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.method'
    var_7 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.MyClass.static_method'
    var_9 = bool('test_module.MyClass.static_method' in var_0.doc)
    assert var_9 is True
    var_10 = var_0.docstring['test_module.MyClass']
    assert var_10 == '```python\n"""Class docstring."""\n```'
    var_11 = var_0.docstring['test_module.MyClass.method']
    assert var_11 == '```python\n"""Method docstring."""\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = var_0.docstring['test_module.foo']
    assert var_6 == '```python\n"""Function with annotations."""\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\ndef decorator(func):\n    return func\n@decorator\ndef foo():\n    """Decorated function."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = '@decorator'
    var_7 = bool('@decorator' in var_0.doc['test_module.foo'])
    assert var_7 is True
    var_8 = var_0.docstring['test_module.foo']
    assert var_8 == '```python\n"""Decorated function."""\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = ''
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = var_0.doc['test_module']
    assert var_6 == '### Module `test_module`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.public_func'
    var_5 = bool('test_module.public_func' in var_0.imp['test_module'])
    assert var_5 is True
    var_6 = 'test_module.PublicClass'
    var_7 = bool('test_module.PublicClass' in var_0.imp['test_module'])
    assert var_7 is True
    var_8 = 'test_module._private_func'
    var_9 = bool('test_module._private_func' not in var_0.imp['test_module'])
    assert var_9 is True
    var_10 = 'test_module._PrivateClass'
    var_11 = bool('test_module._PrivateClass' not in var_0.imp['test_module'])
    assert var_11 is True



# Parsed testcases at query #35
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 0
    var_4 = 'enum.Enum'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = [var_7]
    var_9 = []
    var_10 = var_0.class_api(var_1, var_2, var_8, var_9)
    var_11 = 'Enums'
    var_12 = bool('Enums' in var_0.doc[var_2])
    assert var_12 is True



# Parsed testcases at query #36
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = 0
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Constant(*var_8, **var_9)
    var_11 = []
    var_12 = {}
    var_13 = module_1.Load(*var_11, **var_12)
    var_14 = [var_6, var_10, var_13]
    var_15 = {}
    var_16 = module_1.Subscript(*var_14, **var_15)
    var_17 = var_2.visit_Subscript(var_16)
    var_18 = bool(var_17 == var_16)
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.typing.Union'
    var_2 = 'typing.Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Union'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = [var_12, var_15]
    var_17 = {}
    var_18 = module_1.Name(*var_16, **var_17)
    var_19 = 'str'
    var_20 = []
    var_21 = {}
    var_22 = module_1.Load(*var_20, **var_21)
    var_23 = [var_19, var_22]
    var_24 = {}
    var_25 = module_1.Name(*var_23, **var_24)
    var_26 = [var_18, var_25]
    var_27 = []
    var_28 = {}
    var_29 = module_1.Load(*var_27, **var_28)
    var_30 = [var_26, var_29]
    var_31 = {}
    var_32 = module_1.Tuple(*var_30, **var_31)
    var_33 = []
    var_34 = {}
    var_35 = module_1.Load(*var_33, **var_34)
    var_36 = [var_11, var_32, var_35]
    var_37 = {}
    var_38 = module_1.Subscript(*var_36, **var_37)
    var_39 = []
    var_40 = {}
    var_41 = module_1.Load(*var_39, **var_40)
    var_42 = [var_12, var_41]
    var_43 = {}
    var_44 = module_1.Name(*var_42, **var_43)
    var_45 = []
    var_46 = {}
    var_47 = module_1.BitOr(*var_45, **var_46)
    var_48 = []
    var_49 = {}
    var_50 = module_1.Load(*var_48, **var_49)
    var_51 = [var_19, var_50]
    var_52 = {}
    var_53 = module_1.Name(*var_51, **var_52)
    var_54 = [var_44, var_47, var_53]
    var_55 = {}
    var_56 = module_1.BinOp(*var_54, **var_55)
    var_57 = var_4.visit_Subscript(var_38)
    var_58 = bool(var_57 == var_56)
    assert var_58 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.typing.Optional'
    var_2 = 'typing.Optional'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Optional'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = [var_12, var_15]
    var_17 = {}
    var_18 = module_1.Name(*var_16, **var_17)
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = [var_11, var_18, var_21]
    var_23 = {}
    var_24 = module_1.Subscript(*var_22, **var_23)
    var_25 = []
    var_26 = {}
    var_27 = module_1.Load(*var_25, **var_26)
    var_28 = [var_12, var_27]
    var_29 = {}
    var_30 = module_1.Name(*var_28, **var_29)
    var_31 = []
    var_32 = {}
    var_33 = module_1.BitOr(*var_31, **var_32)
    var_34 = None
    var_35 = [var_34]
    var_36 = {}
    var_37 = module_1.Constant(*var_35, **var_36)
    var_38 = [var_30, var_33, var_37]
    var_39 = {}
    var_40 = module_1.BinOp(*var_38, **var_39)
    var_41 = var_4.visit_Subscript(var_24)
    var_42 = bool(var_41 == var_40)
    assert var_42 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.typing.List'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'List'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = [var_12, var_15]
    var_17 = {}
    var_18 = module_1.Name(*var_16, **var_17)
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = [var_11, var_18, var_21]
    var_23 = {}
    var_24 = module_1.Subscript(*var_22, **var_23)
    var_25 = 'list'
    var_26 = []
    var_27 = {}
    var_28 = module_1.Load(*var_26, **var_27)
    var_29 = [var_25, var_28]
    var_30 = {}
    var_31 = module_1.Name(*var_29, **var_30)
    var_32 = []
    var_33 = {}
    var_34 = module_1.Load(*var_32, **var_33)
    var_35 = [var_12, var_34]
    var_36 = {}
    var_37 = module_1.Name(*var_35, **var_36)
    var_38 = []
    var_39 = {}
    var_40 = module_1.Load(*var_38, **var_39)
    var_41 = [var_31, var_37, var_40]
    var_42 = {}
    var_43 = module_1.Subscript(*var_41, **var_42)
    var_44 = var_4.visit_Subscript(var_24)
    var_45 = bool(var_44 == var_43)
    assert var_45 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Unknown'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    var_10 = 'int'
    var_11 = []
    var_12 = {}
    var_13 = module_1.Load(*var_11, **var_12)
    var_14 = [var_10, var_13]
    var_15 = {}
    var_16 = module_1.Name(*var_14, **var_15)
    var_17 = []
    var_18 = {}
    var_19 = module_1.Load(*var_17, **var_18)
    var_20 = [var_9, var_16, var_19]
    var_21 = {}
    var_22 = module_1.Subscript(*var_20, **var_21)
    var_23 = var_2.visit_Subscript(var_22)
    var_24 = bool(var_23 == var_22)
    assert var_24 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_func_api_with_vararg. Retrieved 14/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'args'
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = None
    var_15 = False
    var_16 = False
    var_17 = var_0.doc[var_2]
    var_18 = bool(var_0.doc[var_2] is not None)
    assert var_18 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 2/5 statements.
# Partially parsed test_attr_nested. Retrieved 2/7 statements.
# Partially parsed test_attr_missing. Retrieved 1/5 statements.
# Partially parsed test_attr_missing_nested. Retrieved 1/7 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 42
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'inner.nonexistent'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #39
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 0
    var_4 = 'class Base: pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = [var_6]
    var_8 = []
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_10 = 'Bases'
    var_11 = bool('Bases' in var_0.doc[var_2])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 0
    var_4 = 'from enum import Enum\nclass Base(Enum): pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = [var_6]
    var_8 = 'A = 1\nB = 2'
    var_9 = module_1.parse(var_8)
    var_10 = var_9.body[var_3]
    var_11 = [var_10]
    var_12 = var_0.class_api(var_1, var_2, var_7, var_11)
    var_13 = 'Enums'
    var_14 = bool('Enums' in var_0.doc[var_2])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = "x: int = 1\ny: str = 'test'"
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = 'Members'
    var_11 = bool('Members' in var_0.doc[var_2])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = "_x: int = 1\n__y: str = 'test'"
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc[var_2])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = 'x: int = 1\ndel x'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc[var_2])
    assert var_11 is True



# Parsed testcases at query #40
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    var_19 = 'root'
    var_20 = var_0.globals(var_19, var_18)
    var_21 = var_0.alias['root.x']
    assert var_21 == '1'
    var_22 = var_0.const['root.x']
    assert var_22 == 'int'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_link_false_no_anchor_tag. Retrieved 5/6 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'def foo(): pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = '<a id='



# Parsed testcases at query #42
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_1.Constant(*var_7, **var_9)
    var_11 = 42
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = {var_17: var_5, var_18: var_10, var_19: var_15}
    var_21 = module_1.AnnAssign(*var_16, **var_20)
    var_22 = 'test_module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = var_0.alias['test_module.test_var']
    assert var_24 == '42'
    var_25 = var_0.const['test_module.test_var']
    assert var_25 == 'int'
    var_26 = var_0.root['test_module.test_var']
    assert var_26 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    var_19 = 'test_module'
    var_20 = var_0.globals(var_19, var_18)
    var_21 = var_0.alias['test_module.test_var']
    assert var_21 == '42'
    var_22 = var_0.const['test_module.test_var']
    assert var_22 == 'int'
    var_23 = var_0.root['test_module.test_var']
    assert var_23 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'test_module'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['test_module.test_var']
    assert var_19 == '42'
    var_20 = var_0.const['test_module.test_var']
    assert var_20 == 'int'
    var_21 = var_0.root['test_module.test_var']
    assert var_21 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'func1'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'func2'
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = [var_11, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = {var_19: var_17}
    var_21 = module_1.List(*var_18, **var_20)
    var_22 = []
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_6, var_24: var_21}
    var_26 = module_1.Assign(*var_22, **var_25)
    var_27 = 'test_module'
    var_28 = var_0.globals(var_27, var_26)
    var_29 = var_0.imp['test_module']
    var_30 = bool(var_0.imp['test_module'] == {'test_module.func1', 'test_module.func2'})
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'non_upper'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'test_module'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['test_module.non_upper']
    assert var_19 == '42'
    var_20 = 'test_module.non_upper'
    var_21 = bool('test_module.non_upper' not in var_0.const)
    assert var_21 is True
    var_22 = 'test_module.non_upper'
    var_23 = bool('test_module.non_upper' not in var_0.root)
    assert var_23 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var1'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'var2'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = [var_5, var_10]
    var_12 = 42
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_11, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'test_module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = 'test_module.var1'
    var_25 = bool('test_module.var1' not in var_0.alias)
    assert var_25 is True
    var_26 = 'test_module.var2'
    var_27 = bool('test_module.var2' not in var_0.alias)
    assert var_27 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test__attr_with_nonexistent_nested_attribute. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent.nested.attribute'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #45
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.arg(*var_2, **var_3)
    var_5 = 'y'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = [var_4, var_8]
    var_10 = 'root'
    var_11 = False
    var_12 = var_0.func_ann(var_10, var_9, has_self=var_11, cls_method=var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == ['ANY', 'ANY'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_1.Name(*var_6, **var_7)
    var_9 = [var_1, var_8]
    var_10 = {}
    var_11 = module_1.arg(*var_9, **var_10)
    var_12 = 'y'
    var_13 = 'str'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = [var_13, var_16]
    var_18 = {}
    var_19 = module_1.Name(*var_17, **var_18)
    var_20 = [var_12, var_19]
    var_21 = {}
    var_22 = module_1.arg(*var_20, **var_21)
    var_23 = [var_11, var_22]
    var_24 = 'root'
    var_25 = False
    var_26 = var_0.func_ann(var_24, var_23, has_self=var_25, cls_method=var_25)
    var_27 = list(var_26)
    var_28 = bool(var_27 == ['int', 'str'])
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_1.Name(*var_6, **var_7)
    var_9 = [var_1, var_8]
    var_10 = {}
    var_11 = module_1.arg(*var_9, **var_10)
    var_12 = 'x'
    var_13 = 'int'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = [var_13, var_16]
    var_18 = {}
    var_19 = module_1.Name(*var_17, **var_18)
    var_20 = [var_12, var_19]
    var_21 = {}
    var_22 = module_1.arg(*var_20, **var_21)
    var_23 = [var_11, var_22]
    var_24 = 'root'
    var_25 = True
    var_26 = False
    var_27 = var_0.func_ann(var_24, var_23, has_self=var_25, cls_method=var_26)
    var_28 = list(var_27)
    var_29 = bool(var_28 == ['Self', 'int'])
    assert var_29 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'type[Self]'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_1.Name(*var_6, **var_7)
    var_9 = [var_1, var_8]
    var_10 = {}
    var_11 = module_1.arg(*var_9, **var_10)
    var_12 = 'x'
    var_13 = 'int'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = [var_13, var_16]
    var_18 = {}
    var_19 = module_1.Name(*var_17, **var_18)
    var_20 = [var_12, var_19]
    var_21 = {}
    var_22 = module_1.arg(*var_20, **var_21)
    var_23 = [var_11, var_22]
    var_24 = 'root'
    var_25 = True
    var_26 = var_0.func_ann(var_24, var_23, has_self=var_25, cls_method=var_25)
    var_27 = list(var_26)
    var_28 = bool(var_27 == ['type[Self]', 'int'])
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_1.Name(*var_6, **var_7)
    var_9 = [var_1, var_8]
    var_10 = {}
    var_11 = module_1.arg(*var_9, **var_10)
    var_12 = '*'
    var_13 = None
    var_14 = [var_12, var_13]
    var_15 = {}
    var_16 = module_1.arg(*var_14, **var_15)
    var_17 = 'y'
    var_18 = 'str'
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = [var_18, var_21]
    var_23 = {}
    var_24 = module_1.Name(*var_22, **var_23)
    var_25 = [var_17, var_24]
    var_26 = {}
    var_27 = module_1.arg(*var_25, **var_26)
    var_28 = [var_11, var_16, var_27]
    var_29 = 'root'
    var_30 = False
    var_31 = var_0.func_ann(var_29, var_28, has_self=var_30, cls_method=var_30)
    var_32 = list(var_31)
    var_33 = bool(var_32 == ['int', '', 'str'])
    assert var_33 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 1/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 1/8 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'x'

def test_case_0():
    var_0 = 'inner.y'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'inner.nonexistent'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_visit_Name_with_TypeVar. Retrieved 9/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.TypeVar'
    var_2 = 'typing.TypeVar'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'TypeVar'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = var_4.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'TypeVar'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_imports_with_none_asname. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'exit'
    var_3 = None
    var_4 = 0
    var_5 = 'test'
    var_6 = 'test.exit'
    var_7 = bool('test.exit' not in var_0.alias)
    assert var_7 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_isinstance_async_function_def. Retrieved 10/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_load_docstring_updates_docstring_when_doc_exists. Retrieved 20/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod'
    var_2 = 'pkg.submod.func'
    var_3 = 'Module doc'
    var_4 = 'Function doc'
    var_5 = 'mock_module'
    var_6 = ()
    var_7 = 'submod'
    var_8 = 'mock_submod'
    var_9 = ()
    var_10 = 'func'
    var_11 = None
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}
    var_14 = type(var_8, var_9, var_13)
    var_15 = {var_7: var_14}
    var_16 = type(var_5, var_6, var_15)
    var_17 = var_0.load_docstring(var_1, var_16)
    var_18 = 'Function documentation'
    var_19 = module_0.doctest(var_18)
    var_20 = var_0.docstring['pkg.submod.func']
    var_21 = bool(var_0.docstring['pkg.submod.func'] == var_19)
    assert var_21 is True



# Parsed testcases at query #51
#--------------------------




import builtins as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'nonexistent'
    var_4 = module_1._attr(var_2, var_3)
    assert var_4 is None



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_visit_Subscript_with_typing_Union. Retrieved 22/26 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'typing.Union'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    var_4 = 'Union'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_1.Name(*var_8, **var_9)
    var_11 = 'int'
    var_12 = []
    var_13 = {}
    var_14 = module_1.Load(*var_12, **var_13)
    var_15 = [var_11, var_14]
    var_16 = {}
    var_17 = module_1.Name(*var_15, **var_16)
    var_18 = 'str'
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = [var_18, var_21]
    var_23 = {}
    var_24 = module_1.Name(*var_22, **var_23)
    var_25 = [var_17, var_24]
    var_26 = []
    var_27 = {}
    var_28 = module_1.Load(*var_26, **var_27)
    var_29 = [var_25, var_28]
    var_30 = {}
    var_31 = module_1.Tuple(*var_29, **var_30)
    var_32 = []
    var_33 = {}
    var_34 = module_1.Load(*var_32, **var_33)
    var_35 = [var_10, var_31, var_34]
    var_36 = {}
    var_37 = module_1.Subscript(*var_35, **var_36)
    var_38 = var_3.visit_Subscript(var_37)
    var_39 = var_38.left
    var_40 = var_38.left.id
    assert var_40 == 'int'
    var_41 = var_38.op
    var_42 = var_38.right
    var_43 = var_38.right.id
    assert var_43 == 'str'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 17/26 statements.
# Partially parsed test_class_api_with_enum. Retrieved 13/23 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 10/19 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 10/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'public_attr'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    var_23 = None
    var_24 = 1
    var_25 = 'another_attr'
    var_26 = 42
    var_27 = []
    var_28 = 'value'
    var_29 = {var_28: var_26}
    var_30 = module_1.Constant(*var_27, **var_29)
    var_31 = 'float'
    var_32 = 'Bases'
    var_33 = bool('Bases' in var_0.doc[var_2])
    assert var_33 is True
    var_34 = 'Members'
    var_35 = bool('Members' in var_0.doc[var_2])
    assert var_35 is True
    var_36 = 'Type'
    var_37 = bool('Type' in var_0.doc[var_2])
    assert var_37 is True
    var_38 = 'public_attr'
    var_39 = bool('public_attr' in var_0.doc[var_2])
    assert var_39 is True
    var_40 = 'another_attr'
    var_41 = bool('another_attr' in var_0.doc[var_2])
    assert var_41 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyEnum'
    var_3 = 'enum.Enum'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'FIRST'
    var_14 = 1
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = 'SECOND'
    var_20 = 2
    var_21 = []
    var_22 = 'value'
    var_23 = {var_22: var_20}
    var_24 = module_1.Constant(*var_21, **var_23)
    var_25 = 'Enums'
    var_26 = bool('Enums' in var_0.doc[var_2])
    assert var_26 is True
    var_27 = 'FIRST'
    var_28 = bool('FIRST' in var_0.doc[var_2])
    assert var_28 is True
    var_29 = 'SECOND'
    var_30 = bool('SECOND' in var_0.doc[var_2])
    assert var_30 is True
    var_31 = 'Members'
    var_32 = bool('Members' not in var_0.doc[var_2])
    assert var_32 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 1
    var_16 = 'Members'
    var_17 = bool('Members' not in var_0.doc[var_2])
    assert var_17 is True
    var_18 = 'public_attr'
    var_19 = bool('public_attr' not in var_0.doc[var_2])
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 1
    var_16 = 'Members'
    var_17 = bool('Members' not in var_0.doc[var_2])
    assert var_17 is True
    var_18 = '_private_attr'
    var_19 = bool('_private_attr' not in var_0.doc[var_2])
    assert var_19 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc[var_2])
    assert var_9 is True
    var_10 = 'Enums'
    var_11 = bool('Enums' not in var_0.doc[var_2])
    assert var_11 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_is_public_predicate_false. Retrieved 3/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = var_0.is_public(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_load_docstring. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'doc'
    var_4 = None
    var_5 = var_0.docstring
    var_6 = bool(var_0.docstring == {'pkg.module': '```python\n```', 'pkg.module.func': '```python\nFunction docstring\n```'})
    assert var_6 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = [var_1, var_2, var_3, var_4, var_2, var_5, var_2]
    var_7 = {}
    var_8 = module_1.arguments(*var_6, **var_7)
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False
    var_12 = var_0.func_api(var_9, var_10, var_8, var_2, has_self=var_11, cls_method=var_11)
    var_13 = '| return |\n|:-----:|\n|  |\n\n'
    var_14 = var_0.doc['root.func']
    var_15 = bool(var_0.doc['root.func'] == var_13)
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'b'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = [var_10, var_2, var_11, var_12, var_2, var_13, var_2]
    var_15 = {}
    var_16 = module_1.arguments(*var_14, **var_15)
    var_17 = 'root'
    var_18 = 'func'
    var_19 = False
    var_20 = var_0.func_api(var_17, var_18, var_16, var_2, has_self=var_19, cls_method=var_19)
    var_21 = '| a | b | return |\n|:---:|:---:|:-----:|\n|  |  |  |\n\n'
    var_22 = var_0.doc['root.func']
    var_23 = bool(var_0.doc['root.func'] == var_21)
    assert var_23 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'b'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = [var_10, var_2, var_11, var_12, var_2, var_13, var_2]
    var_15 = {}
    var_16 = module_1.arguments(*var_14, **var_15)
    var_17 = 1
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_1.Constant(*var_18, **var_19)
    var_21 = [var_2, var_20]
    var_22 = 'root'
    var_23 = 'func'
    var_24 = False
    var_25 = var_0.func_api(var_22, var_23, var_16, var_2, has_self=var_24, cls_method=var_24)
    var_26 = '| a | b | return |\n|:---:|:---:|:-----:|\n|  | `1` |  |\n\n'
    var_27 = var_0.doc['root.func']
    var_28 = bool(var_0.doc['root.func'] == var_26)
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_2, var_7, var_8, var_2, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'func'
    var_15 = True
    var_16 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)
    var_17 = '| self | return |\n|:----:|:-----:|\n| type[Self] |  |\n\n'
    var_18 = var_0.doc['root.func']
    var_19 = bool(var_0.doc['root.func'] == var_17)
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_2, var_7, var_8, var_2, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'func'
    var_15 = True
    var_16 = False
    var_17 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_16)
    var_18 = '| self | return |\n|:----:|:-----:|\n| Self |  |\n\n'
    var_19 = var_0.doc['root.func']
    var_20 = bool(var_0.doc['root.func'] == var_18)
    assert var_20 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = 'args'
    var_6 = [var_5, var_2]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = []
    var_10 = [var_1, var_2, var_3, var_4, var_8, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'func'
    var_15 = False
    var_16 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)
    var_17 = '| *args | return |\n|:-----:|:-----:|\n|  |  |\n\n'
    var_18 = var_0.doc['root.func']
    var_19 = bool(var_0.doc['root.func'] == var_17)
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_1, var_2, var_3, var_4, var_2, var_5, var_9]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'func'
    var_15 = False
    var_16 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)
    var_17 = '| **kwargs | return |\n|:-------:|:-----:|\n|  |  |\n\n'
    var_18 = var_0.doc['root.func']
    var_19 = bool(var_0.doc['root.func'] == var_17)
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'int'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Constant(*var_3, **var_4)
    var_6 = [var_1, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = None
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = [var_9, var_10, var_11, var_12, var_10, var_13, var_10]
    var_15 = {}
    var_16 = module_1.arguments(*var_14, **var_15)
    var_17 = 'root'
    var_18 = 'func'
    var_19 = False
    var_20 = var_0.func_api(var_17, var_18, var_16, var_10, has_self=var_19, cls_method=var_19)
    var_21 = '| a | return |\n|:---:|:-----:|\n| `int` |  |\n\n'
    var_22 = var_0.doc['root.func']
    var_23 = bool(var_0.doc['root.func'] == var_21)
    assert var_23 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = [var_1, var_2, var_3, var_4, var_2, var_5, var_2]
    var_7 = {}
    var_8 = module_1.arguments(*var_6, **var_7)
    var_9 = 'root'
    var_10 = 'func'
    var_11 = 'str'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Constant(*var_12, **var_13)
    var_15 = False
    var_16 = var_0.func_api(var_9, var_10, var_8, var_14, has_self=var_15, cls_method=var_15)
    var_17 = '| return |\n|:-----:|\n| `str` |\n\n'
    var_18 = var_0.doc['root.func']
    var_19 = bool(var_0.doc['root.func'] == var_17)
    assert var_19 is True



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = bool(var_2 in var_0.doc)
    assert var_6 is True
    var_7 = 'class TestClass'
    var_8 = bool('class TestClass' in var_0.doc[var_2])
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = []
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_10 = bool(var_2 in var_0.doc)
    assert var_10 is True
    var_11 = 'Bases'
    var_12 = bool('Bases' in var_0.doc[var_2])
    assert var_12 is True
    var_13 = 'BaseClass'
    var_14 = bool('BaseClass' in var_0.doc[var_2])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = 'x: int = 1'
    var_6 = 'exec'
    var_7 = module_1.parse(var_5, mode=var_6)
    var_8 = var_7.body[var_4]
    var_9 = "y: str = 'hello'"
    var_10 = module_1.parse(var_9, mode=var_6)
    var_11 = var_10.body[var_4]
    var_12 = [var_8, var_11]
    var_13 = var_0.class_api(var_1, var_2, var_3, var_12)
    var_14 = bool(var_2 in var_0.doc)
    assert var_14 is True
    var_15 = 'Members'
    var_16 = bool('Members' in var_0.doc[var_2])
    assert var_16 is True
    var_17 = 'Type'
    var_18 = bool('Type' in var_0.doc[var_2])
    assert var_18 is True
    var_19 = 'x'
    var_20 = bool('x' in var_0.doc[var_2])
    assert var_20 is True
    var_21 = 'y'
    var_22 = bool('y' in var_0.doc[var_2])
    assert var_22 is True

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
    var_8 = 0
    var_9 = 'A = 1'
    var_10 = 'exec'
    var_11 = module_1.parse(var_9, mode=var_10)
    var_12 = var_11.body[var_8]
    var_13 = 'B = 2'
    var_14 = module_1.parse(var_13, mode=var_10)
    var_15 = var_14.body[var_8]
    var_16 = [var_12, var_15]
    var_17 = var_0.class_api(var_1, var_2, var_7, var_16)
    var_18 = bool(var_2 in var_0.doc)
    assert var_18 is True
    var_19 = 'Enums'
    var_20 = bool('Enums' in var_0.doc[var_2])
    assert var_20 is True
    var_21 = 'A'
    var_22 = bool('A' in var_0.doc[var_2])
    assert var_22 is True
    var_23 = 'B'
    var_24 = bool('B' in var_0.doc[var_2])
    assert var_24 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = 'x: int = 1'
    var_6 = 'exec'
    var_7 = module_1.parse(var_5, mode=var_6)
    var_8 = var_7.body[var_4]
    var_9 = 'del x'
    var_10 = module_1.parse(var_9, mode=var_6)
    var_11 = var_10.body[var_4]
    var_12 = [var_8, var_11]
    var_13 = var_0.class_api(var_1, var_2, var_3, var_12)
    var_14 = bool(var_2 in var_0.doc)
    assert var_14 is True
    var_15 = 'x'
    var_16 = bool('x' not in var_0.doc[var_2])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = '_private: int = 1'
    var_6 = 'exec'
    var_7 = module_1.parse(var_5, mode=var_6)
    var_8 = var_7.body[var_4]
    var_9 = 'public: int = 2'
    var_10 = module_1.parse(var_9, mode=var_6)
    var_11 = var_10.body[var_4]
    var_12 = [var_8, var_11]
    var_13 = var_0.class_api(var_1, var_2, var_3, var_12)
    var_14 = bool(var_2 in var_0.doc)
    assert var_14 is True
    var_15 = '_private'
    var_16 = bool('_private' not in var_0.doc[var_2])
    assert var_16 is True
    var_17 = 'public'
    var_18 = bool('public' in var_0.doc[var_2])
    assert var_18 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_constructor_with_parameters. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_visit_Name_self_type. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 9/10 statements.
# Partially parsed test_visit_Name_no_alias. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_typevar_alias. Retrieved 9/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = [var_2, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    var_10 = var_3.visit_Name(var_9)
    var_11 = var_10.id
    assert var_11 == 'Self'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.Name'
    var_2 = 'typing.List'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Name'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = var_4.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'List'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Name'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    var_10 = var_2.visit_Name(var_9)
    var_11 = var_10.id
    assert var_11 == 'Name'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'T'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = var_4.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'T'



# Parsed testcases at query #5
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'x: int = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = "y: str = 'hello'"
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = 'del z'
    var_15 = module_1.parse(var_14, mode=var_4)
    var_16 = var_15.body
    var_17 = [var_10, var_13, var_16]
    var_18 = var_0.class_api(var_1, var_2, var_7, var_17)
    var_19 = 'Bases'
    var_20 = bool('Bases' in var_0.doc[var_2])
    assert var_20 is True
    var_21 = 'Members'
    var_22 = bool('Members' in var_0.doc[var_2])
    assert var_22 is True
    var_23 = 'Type'
    var_24 = bool('Type' in var_0.doc[var_2])
    assert var_24 is True
    var_25 = 'x'
    var_26 = bool('x' in var_0.doc[var_2])
    assert var_26 is True
    var_27 = 'y'
    var_28 = bool('y' in var_0.doc[var_2])
    assert var_28 is True
    var_29 = 'z'
    var_30 = bool('z' not in var_0.doc[var_2])
    assert var_30 is True

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
    var_8 = 'A = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = 'B = 2'
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = [var_10, var_13]
    var_15 = var_0.class_api(var_1, var_2, var_7, var_14)
    var_16 = 'Enums'
    var_17 = bool('Enums' in var_0.doc[var_2])
    assert var_17 is True
    var_18 = 'A'
    var_19 = bool('A' in var_0.doc[var_2])
    assert var_19 is True
    var_20 = 'B'
    var_21 = bool('B' in var_0.doc[var_2])
    assert var_21 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = var_0.doc[var_2]
    assert var_6 == ''



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_line_19_false. Retrieved 7/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = var_0.root[var_1]
    var_6 = var_5.body[var_4]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__attr_simple_attribute. Retrieved 2/5 statements.
# Partially parsed test__attr_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test__attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test__attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test__attr_chain_with_none. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'x'

def test_case_0():
    var_0 = 20
    var_1 = 'inner.y'

def test_case_0():
    var_0 = 10
    var_1 = 'z'

def test_case_0():
    var_0 = 20
    var_1 = 'inner.z'

def test_case_0():
    var_0 = None
    var_1 = 'x.y'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_const_type_with_call_to_bool. Retrieved 5/7 statements.
# Partially parsed test_const_type_with_call_to_int. Retrieved 5/7 statements.
# Partially parsed test_const_type_with_unknown_call. Retrieved 3/5 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1.1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2.2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[float, float]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = False
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Set(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'set[bool, bool]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'b'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 1
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 2
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[str, str, int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 2
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 'b'
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[Any, Any, Any, Any]'

import ast as module_0

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 1
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = []

import ast as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = '42'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = []

import ast as module_0

def test_case_0():
    var_0 = 'unknown'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = []

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.Tuple(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'tuple[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = []
    var_3 = 'value'
    var_4 = {var_3: var_1}
    var_5 = module_0.Constant(*var_2, **var_4)
    var_6 = [var_0, var_5]
    var_7 = []
    var_8 = 'elts'
    var_9 = {var_8: var_6}
    var_10 = module_0.Tuple(*var_7, **var_9)
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'x'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'Any'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/8 statements.
# Partially parsed test_class_api_with_enum. Retrieved 9/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = 'Bases'
    var_6 = bool('Bases' in var_0.doc[var_2])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = 'VALUE1 = 1'
    var_5 = module_1.parse(var_4)
    var_6 = 'VALUE2 = 2'
    var_7 = module_1.parse(var_6)
    var_8 = [var_5, var_7]
    var_9 = 'Enums'
    var_10 = bool('Enums' in var_0.doc[var_2])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'member: int = 1'
    var_5 = module_1.parse(var_4)
    var_6 = [var_5]
    var_7 = var_0.class_api(var_1, var_2, var_3, var_6)
    var_8 = 'Members'
    var_9 = bool('Members' in var_0.doc[var_2])
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'member: int = 1'
    var_5 = module_1.parse(var_4)
    var_6 = 'del member'
    var_7 = module_1.parse(var_6)
    var_8 = [var_5, var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc[var_2])
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 9/14 statements.
# Partially parsed test_globals_ignores_complex_assign. Retrieved 6/13 statements.
# Partially parsed test_globals_ignores_non_constant_all. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 1
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'root'
    var_8 = var_0.alias['root.x']
    assert var_8 == '1'
    var_9 = var_0.const['root.x']
    assert var_9 == 'int'
    var_10 = var_0.root['root.x']
    assert var_10 == 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'hello'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'root'
    var_8 = var_0.alias['root.y']
    assert var_8 == "'hello'"
    var_9 = var_0.const['root.y']
    assert var_9 == 'str'
    var_10 = var_0.root['root.y']
    assert var_10 == 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'foo'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'bar'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_6, var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.List(*var_13, **var_15)
    var_17 = 'root'
    var_18 = var_0.imp['root']
    var_19 = bool(var_0.imp['root'] == {'root.foo', 'root.bar'})
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 'w'
    var_3 = 42
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'root'
    var_9 = 'root.z'
    var_10 = bool('root.z' not in var_0.alias)
    assert var_10 is True
    var_11 = 'root.w'
    var_12 = bool('root.w' not in var_0.alias)
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'some_var'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = 'root'
    var_12 = set()
    var_13 = var_0.imp['root']
    var_14 = bool(var_0.imp['root'] == var_12)
    assert var_14 is True



# Parsed testcases at query #11
#--------------------------




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

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = module_0.Parser(var_0, toc=var_1)
    var_3 = var_2.link
    assert var_3 is True
    var_4 = var_2.toc
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_visit_Name_with_TypeVar. Retrieved 10/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.T'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'T'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_6, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = var_5.visit_Name(var_14)
    var_16 = var_15.id
    assert var_16 == 'T'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_compile_empty. Retrieved 2/4 statements.
# Partially parsed test_compile_single_module. Retrieved 2/8 statements.
# Partially parsed test_compile_with_toc. Retrieved 1/7 statements.
# Partially parsed test_compile_with_docstring. Retrieved 2/9 statements.
# Partially parsed test_compile_with_const. Retrieved 2/9 statements.
# Partially parsed test_compile_non_public. Retrieved 2/11 statements.
# Partially parsed test_compile_magic_method. Retrieved 2/11 statements.
# Partially parsed test_compile_multiple_modules. Retrieved 2/12 statements.
# Partially parsed test_compile_with_link. Retrieved 2/8 statements.
# Partially parsed test_compile_with_alias. Retrieved 2/12 statements.
# Partially parsed test_compile_with_all. Retrieved 2/14 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True
    var_1 = 'module'



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    '''Module docstring.'''\n    x = 1\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.level)
    assert var_7 is True
    var_8 = 'test_module'
    var_9 = bool('test_module' in var_0.root)
    assert var_9 is True
    var_10 = 'test_module.x'
    var_11 = bool('test_module.x' in var_0.alias)
    assert var_11 is True
    var_12 = 'test_module'
    var_13 = bool('test_module' in var_0.docstring)
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    import os\n    from sys import path\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.os'
    var_5 = bool('test_module.os' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.path'
    var_7 = bool('test_module.path' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    def foo():\n        '''Function docstring.'''\n        pass\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.docstring)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    class Bar:\n        '''Class docstring.'''\n        pass\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.Bar'
    var_5 = bool('test_module.Bar' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.Bar'
    var_7 = bool('test_module.Bar' in var_0.docstring)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    x: int = 1\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.x'
    var_7 = bool('test_module.x' in var_0.const)
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = '__main__'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = '__name__'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = 'public.module'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'public.module.submodule'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False
    var_2 = 'public._private'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is False
    var_4 = '_private.module'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is False
    var_6 = 'public.module._private'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = '__init__.public'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'public.__init__.module'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = '__init__.public.__main__'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_annassign_not_instance_of_name. Retrieved 8/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 1
    var_2 = []
    var_3 = 'value'
    var_4 = {var_3: var_1}
    var_5 = module_1.Constant(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = None
    var_12 = []
    var_13 = 'target'
    var_14 = 'annotation'
    var_15 = 'value'
    var_16 = {var_13: var_5, var_14: var_10, var_15: var_11}
    var_17 = module_1.AnnAssign(*var_12, **var_16)
    var_18 = var_17.target



# Parsed testcases at query #17
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'class A:\n    x = 1\n    del x[0]'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.doc['test.A']
    var_5 = bool(var_0.doc['test.A'] == '#' * (var_0.b_level + 3) + ' class A\n\n*Full name:* `test.A`\n<a id="test-a"></a>\n\n')
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_visit_Name_with_TypeVar_in_alias. Retrieved 9/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.T'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'T'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = var_4.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'T'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_nested_public_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_nested_private_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_all_listed_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_parent_listed_in_all. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_empty_all. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_const_in_all. Retrieved 7/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root._private_name'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.__magic__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'nested'
    var_3 = {var_2}
    var_4 = 'root.nested.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'nested'
    var_3 = {var_2}
    var_4 = 'root.nested._private_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'listed_name'
    var_3 = {var_2}
    var_4 = 'root.listed_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'root.parent.child'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.public_name'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'CONST_NAME'
    var_3 = {var_2}
    var_4 = 'root.CONST_NAME'
    var_5 = 'int'
    var_6 = var_0.is_public(var_4)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_false. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_api_function. Retrieved 20/31 statements.
# Partially parsed test_api_async_function. Retrieved 20/31 statements.
# Partially parsed test_api_class. Retrieved 14/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = set()
    var_4 = 'test_func'
    var_5 = []
    var_6 = 'x'
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = 'y'
    var_12 = [var_11, var_7]
    var_13 = {}
    var_14 = module_1.arg(*var_12, **var_13)
    var_15 = [var_10, var_14]
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = []
    var_22 = 'root.test_func'
    var_23 = bool('root.test_func' in var_0.doc)
    assert var_23 is True
    var_24 = 'root.test_func'
    var_25 = var_0.doc[var_24]
    var_26 = '### test_func()'
    var_27 = 'Full name'
    var_28 = bool('Full name' in var_0.doc['root.test_func'])
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = set()
    var_4 = 'test_async_func'
    var_5 = []
    var_6 = 'x'
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = 'y'
    var_12 = [var_11, var_7]
    var_13 = {}
    var_14 = module_1.arg(*var_12, **var_13)
    var_15 = [var_10, var_14]
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = []
    var_22 = 'root.test_async_func'
    var_23 = bool('root.test_async_func' in var_0.doc)
    assert var_23 is True
    var_24 = 'root.test_async_func'
    var_25 = var_0.doc[var_24]
    var_26 = '### async test_async_func()'
    var_27 = 'Full name'
    var_28 = bool('Full name' in var_0.doc['root.test_async_func'])
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = set()
    var_4 = 'TestClass'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'name'
    var_11 = 'bases'
    var_12 = 'keywords'
    var_13 = 'body'
    var_14 = 'decorator_list'
    var_15 = {var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7, var_14: var_8}
    var_16 = module_1.ClassDef(*var_9, **var_15)
    var_17 = var_0.api(var_1, var_16)
    var_18 = 'root.TestClass'
    var_19 = bool('root.TestClass' in var_0.doc)
    assert var_19 is True
    var_20 = 'root.TestClass'
    var_21 = var_0.doc[var_20]
    var_22 = '### class TestClass'
    var_23 = 'Full name'
    var_24 = bool('Full name' in var_0.doc['root.TestClass'])
    assert var_24 is True



# Parsed testcases at query #22
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a&b'
    var_1 = 'c&d'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['<code>a&#38;b</code>', '<code>c&#38;d</code>'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a|b'
    var_1 = 'c|d'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['`a&#124;b`', '`c&#124;d`'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a|b'
    var_1 = None
    var_2 = 'c&d'
    var_3 = ''
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._defaults(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == ['`a&#124;b`', ' ', '<code>c&#38;d</code>', ' '])
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_compile_with_toc. Retrieved 5/8 statements.


import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = 'def foo(): pass'
    var_3 = module_0.parse(var_1, var_2)
    var_4 = '**Table of contents:**'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_compile_skips_magic_names_without_docstring. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = '## __init__\n\n*Full name:* `{}`\n\n'
    var_3 = ''
    var_4 = 0
    var_5 = var_0.compile()
    assert var_5 == ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_globals_with_non_constant. Retrieved 10/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = 1
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = 'simple'
    var_21 = {var_17: var_5, var_18: var_10, var_19: var_15, var_20: var_11}
    var_22 = module_1.AnnAssign(*var_16, **var_21)
    var_23 = 'test'
    var_24 = var_0.globals(var_23, var_22)
    var_25 = var_0.alias['test.x']
    assert var_25 == '1'
    var_26 = var_0.const['test.x']
    assert var_26 == 'int'
    var_27 = var_0.root['test.x']
    assert var_27 == 'test'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'hello'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'test'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['test.y']
    assert var_19 == "'hello'"
    var_20 = var_0.const['test.y']
    assert var_20 == 'str'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'foo'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'bar'
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = [var_11, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = {var_19: var_17}
    var_21 = module_1.List(*var_18, **var_20)
    var_22 = []
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_6, var_24: var_21}
    var_26 = module_1.Assign(*var_22, **var_25)
    var_27 = 'test'
    var_28 = var_0.globals(var_27, var_26)
    var_29 = var_0.imp['test']
    var_30 = bool(var_0.imp['test'] == {'test.foo', 'test.bar'})
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'some_var'
    var_8 = []
    var_9 = 'id'
    var_10 = {var_9: var_7}
    var_11 = module_1.Name(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'test'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['test.z']
    assert var_19 == 'some_var'
    var_20 = 'test.z'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_nested_attribute_none. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent_attribute'



# Parsed testcases at query #27
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 0
    var_4 = 'class Base: pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = [var_6]
    var_8 = []
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_10 = 'Bases'
    var_11 = bool('Bases' in var_0.doc[var_2])
    assert var_11 is True
    var_12 = '| Base |'
    var_13 = bool('| Base |' in var_0.doc[var_2])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 0
    var_4 = 'from enum import Enum\nclass Enum: pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = [var_6]
    var_8 = 'A = 1\nB = 2'
    var_9 = module_1.parse(var_8)
    var_10 = var_9.body[var_3]
    var_11 = [var_10]
    var_12 = var_0.class_api(var_1, var_2, var_7, var_11)
    var_13 = 'Enums'
    var_14 = bool('Enums' in var_0.doc[var_2])
    assert var_14 is True
    var_15 = '| A |'
    var_16 = bool('| A |' in var_0.doc[var_2])
    assert var_16 is True
    var_17 = '| B |'
    var_18 = bool('| B |' in var_0.doc[var_2])
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = "x: int = 1\ny: str = 'hello'"
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = 'Members'
    var_11 = bool('Members' in var_0.doc[var_2])
    assert var_11 is True
    var_12 = '| x | int |'
    var_13 = bool('| x | int |' in var_0.doc[var_2])
    assert var_13 is True
    var_14 = '| y | str |'
    var_15 = bool('| y | str |' in var_0.doc[var_2])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = 'x: int = 1\ndel x'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = 'x'
    var_11 = bool('x' not in var_0.doc[var_2])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = '_private: int = 1'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = '_private'
    var_11 = bool('_private' not in var_0.doc[var_2])
    assert var_11 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_isinstance_functiondef. Retrieved 9/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []



# Parsed testcases at query #29
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_1.Name(*var_3, **var_5)
    var_7 = [var_6]
    var_8 = 123
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = 'elts'
    var_16 = {var_15: var_13}
    var_17 = module_1.List(*var_14, **var_16)
    var_18 = []
    var_19 = 'targets'
    var_20 = 'value'
    var_21 = {var_19: var_7, var_20: var_17}
    var_22 = module_1.Assign(*var_18, **var_21)
    var_23 = var_0.globals(var_1, var_22)
    var_24 = var_0.imp[var_1]
    var_25 = len(var_24)
    assert var_25 == 0



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_isinstance_call_and_name_or_attribute. Retrieved 3/8 statements.


import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = []



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_docstring. Retrieved 8/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = ''
    var_4 = 'Module docstring'
    var_5 = module_0.doctest(var_4)
    var_6 = var_0.docstring['test_module']
    var_7 = bool(var_0.docstring['test_module'] == var_5)
    assert var_7 is True
    var_8 = 'Function docstring'
    var_9 = module_0.doctest(var_8)
    var_10 = var_0.docstring['test_module.test_func']
    var_11 = bool(var_0.docstring['test_module.test_func'] == var_9)
    assert var_11 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_none_attr_returns_none. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #33
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.docstring['test_module']
    assert var_10 == '```python\n"""Module docstring."""\n```'
    var_11 = var_0.docstring['test_module.foo']
    assert var_11 == '```python\n"""Function docstring."""\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'os.path'
    var_5 = bool('os.path' in var_0.alias['test_module.path'])
    assert var_5 is True
    var_6 = 'sys'
    var_7 = bool('sys' in var_0.alias['test_module.sys'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.method'
    var_7 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_7 is True
    var_8 = var_0.docstring['test_module.MyClass']
    assert var_8 == '```python\n"""Class docstring."""\n```'
    var_9 = var_0.docstring['test_module.MyClass.method']
    assert var_9 == '```python\n"""Method docstring."""\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n@decorator\ndef foo():\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = '@decorator'
    var_5 = bool('@decorator' in var_0.doc['test_module.foo'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\ndef foo(x: int, y: str) -> bool:\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'int'
    var_5 = bool('int' in var_0.doc['test_module.foo'])
    assert var_5 is True
    var_6 = 'str'
    var_7 = bool('str' in var_0.doc['test_module.foo'])
    assert var_7 is True
    var_8 = 'bool'
    var_9 = bool('bool' in var_0.doc['test_module.foo'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'foo'
    var_5 = bool('foo' in var_0.imp['test_module'])
    assert var_5 is True
    var_6 = 'bar'
    var_7 = bool('bar' in var_0.imp['test_module'])
    assert var_7 is True
    var_8 = 'baz'
    var_9 = bool('baz' not in var_0.imp['test_module'])
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = var_1.link
    assert var_2 is False



# Parsed testcases at query #35
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._e_type(*var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.expr(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_1._e_type(*var_4)
    assert var_5 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = module_1._e_type(*var_5)
    assert var_6 == '[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = module_1._e_type(*var_9)
    assert var_10 == '[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = module_1._e_type(*var_9)
    assert var_10 == '[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Constant(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = [var_4, var_9]
    var_11 = module_1._e_type(*var_10)
    assert var_11 == '[int, str]'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_node_type_comment_is_not_none. Retrieved 15/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    var_19 = 'root'
    var_20 = 'root.Class'
    var_21 = []
    var_22 = [var_18]
    var_23 = var_0.class_api(var_19, var_20, var_21, var_22)
    var_24 = 'mem'
    var_25 = {}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_annassign_with_non_name_target. Retrieved 10/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'y'
    var_7 = []
    var_8 = 'value'
    var_9 = 'attr'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.Attribute(*var_7, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = 'id'
    var_15 = {var_14: var_12}
    var_16 = module_1.Name(*var_13, **var_15)
    var_17 = None
    var_18 = []
    var_19 = 'target'
    var_20 = 'annotation'
    var_21 = 'value'
    var_22 = {var_19: var_11, var_20: var_16, var_21: var_17}
    var_23 = module_1.AnnAssign(*var_18, **var_22)
    var_24 = var_23.target



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_visit_Constant_valid_name. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Constant(*var_6, **var_7)
    var_9 = var_4.visit_Constant(var_8)
    var_10 = var_9.id
    assert var_10 == 'alias'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_imports_with_Import_node. Retrieved 6/11 statements.
# Partially parsed test_imports_with_ImportFrom_node_no_level. Retrieved 6/10 statements.
# Partially parsed test_imports_with_ImportFrom_node_with_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = var_0.alias['test.module.os']
    assert var_6 == 'os'
    var_7 = var_0.alias['test.module.system']
    assert var_7 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['test.module.join']
    assert var_6 == 'os.path.join'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = 'f'
    var_5 = 1
    var_6 = var_0.alias['test.module.submodule.f']
    assert var_6 == 'test.module.sibling.func'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_imports_with_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'p'
    var_4 = 1
    var_5 = 'pkg.subpkg'
    var_6 = var_0.alias['pkg.subpkg.p']
    assert var_6 == 'pkg.path'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_compile_empty. Retrieved 2/4 statements.
# Partially parsed test_compile_with_toc. Retrieved 1/6 statements.
# Partially parsed test_compile_with_docstring. Retrieved 2/8 statements.
# Partially parsed test_compile_with_magic_name. Retrieved 2/7 statements.
# Partially parsed test_compile_with_const. Retrieved 3/11 statements.
# Partially parsed test_compile_with_non_public. Retrieved 2/7 statements.
# Partially parsed test_compile_with_alias. Retrieved 2/10 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test.CONST'

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1



# Parsed testcases at query #43
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = 42
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = {var_17: var_5, var_18: var_10, var_19: var_15}
    var_21 = module_1.AnnAssign(*var_16, **var_20)
    var_22 = 'test_module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = var_0.alias['test_module.test_var']
    assert var_24 == '42'
    var_25 = var_0.const['test_module.test_var']
    assert var_25 == 'int'
    var_26 = var_0.root['test_module.test_var']
    assert var_26 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'test_module'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['test_module.test_var']
    assert var_19 == '42'
    var_20 = var_0.const['test_module.test_var']
    assert var_20 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'func1'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'func2'
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = [var_11, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = {var_19: var_17}
    var_21 = module_1.List(*var_18, **var_20)
    var_22 = []
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_6, var_24: var_21}
    var_26 = module_1.Assign(*var_22, **var_25)
    var_27 = 'test_module'
    var_28 = var_0.globals(var_27, var_26)
    var_29 = var_0.imp['test_module']
    var_30 = bool(var_0.imp['test_module'] == {'test_module.func1', 'test_module.func2'})
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'another_var'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = [var_5, var_10]
    var_12 = 42
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_11, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'test_module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = 'test_module.test_var'
    var_25 = bool('test_module.test_var' not in var_0.alias)
    assert var_25 is True
    var_26 = 'test_module.another_var'
    var_27 = bool('test_module.another_var' not in var_0.alias)
    assert var_27 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_try_statement_with_handlers. Retrieved 15/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = module_1.walk_body(var_16)
    var_18 = list(var_17)
    var_19 = bool(var_18 == [var_15])
    assert var_19 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'x'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_0}
    var_14 = module_0.Constant(*var_11, **var_13)
    var_15 = []
    var_16 = 'targets'
    var_17 = 'value'
    var_18 = {var_16: var_10, var_17: var_14}
    var_19 = module_0.Assign(*var_15, **var_18)
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = 'test'
    var_24 = 'body'
    var_25 = 'orelse'
    var_26 = {var_23: var_4, var_24: var_20, var_25: var_21}
    var_27 = module_0.If(*var_22, **var_26)
    var_28 = [var_27]
    var_29 = module_1.walk_body(var_28)
    var_30 = list(var_29)
    var_31 = []
    var_32 = 'id'
    var_33 = {var_32: var_5}
    var_34 = module_0.Name(*var_31, **var_33)
    var_35 = [var_34]
    var_36 = []
    var_37 = 'value'
    var_38 = {var_37: var_0}
    var_39 = module_0.Constant(*var_36, **var_38)
    var_40 = []
    var_41 = 'targets'
    var_42 = 'value'
    var_43 = {var_41: var_35, var_42: var_39}
    var_44 = module_0.Assign(*var_40, **var_43)
    var_45 = [var_44]
    var_46 = bool(var_30 == var_45)
    assert var_46 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = []
    var_6 = 'x'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_0.Name(*var_7, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_0}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_11, var_18: var_15}
    var_20 = module_0.Assign(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = 'test'
    var_24 = 'body'
    var_25 = 'orelse'
    var_26 = {var_23: var_4, var_24: var_5, var_25: var_21}
    var_27 = module_0.If(*var_22, **var_26)
    var_28 = [var_27]
    var_29 = module_1.walk_body(var_28)
    var_30 = list(var_29)
    var_31 = []
    var_32 = 'id'
    var_33 = {var_32: var_6}
    var_34 = module_0.Name(*var_31, **var_33)
    var_35 = [var_34]
    var_36 = []
    var_37 = 'value'
    var_38 = {var_37: var_0}
    var_39 = module_0.Constant(*var_36, **var_38)
    var_40 = []
    var_41 = 'targets'
    var_42 = 'value'
    var_43 = {var_41: var_35, var_42: var_39}
    var_44 = module_0.Assign(*var_40, **var_43)
    var_45 = [var_44]
    var_46 = bool(var_30 == var_45)
    assert var_46 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = 'body'
    var_22 = 'handlers'
    var_23 = 'orelse'
    var_24 = 'finalbody'
    var_25 = {var_21: var_16, var_22: var_17, var_23: var_18, var_24: var_19}
    var_26 = module_0.Try(*var_20, **var_25)
    var_27 = [var_26]
    var_28 = module_1.walk_body(var_27)
    var_29 = list(var_28)
    var_30 = []
    var_31 = 'id'
    var_32 = {var_31: var_0}
    var_33 = module_0.Name(*var_30, **var_32)
    var_34 = [var_33]
    var_35 = []
    var_36 = 'value'
    var_37 = {var_36: var_6}
    var_38 = module_0.Constant(*var_35, **var_37)
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_34, var_41: var_38}
    var_43 = module_0.Assign(*var_39, **var_42)
    var_44 = [var_43]
    var_45 = bool(var_29 == var_44)
    assert var_45 is True

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_0.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_0.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_0.Assign(*var_12, **var_15)
    var_17 = [var_16]
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = 'id'
    var_22 = {var_21: var_1}
    var_23 = module_0.Name(*var_20, **var_22)
    var_24 = [var_23]
    var_25 = []
    var_26 = 'value'
    var_27 = {var_26: var_7}
    var_28 = module_0.Constant(*var_25, **var_27)
    var_29 = []
    var_30 = 'targets'
    var_31 = 'value'
    var_32 = {var_30: var_24, var_31: var_28}
    var_33 = module_0.Assign(*var_29, **var_32)
    var_34 = [var_33]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'x'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_0.Name(*var_3, **var_5)
    var_7 = [var_6]
    var_8 = 1
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_0.Constant(*var_9, **var_11)
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = {var_14: var_7, var_15: var_12}
    var_17 = module_0.Assign(*var_13, **var_16)
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = 'body'
    var_22 = 'handlers'
    var_23 = 'orelse'
    var_24 = 'finalbody'
    var_25 = {var_21: var_0, var_22: var_1, var_23: var_18, var_24: var_19}
    var_26 = module_0.Try(*var_20, **var_25)
    var_27 = [var_26]
    var_28 = module_1.walk_body(var_27)
    var_29 = list(var_28)
    var_30 = []
    var_31 = 'id'
    var_32 = {var_31: var_2}
    var_33 = module_0.Name(*var_30, **var_32)
    var_34 = [var_33]
    var_35 = []
    var_36 = 'value'
    var_37 = {var_36: var_8}
    var_38 = module_0.Constant(*var_35, **var_37)
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_34, var_41: var_38}
    var_43 = module_0.Assign(*var_39, **var_42)
    var_44 = [var_43]
    var_45 = bool(var_29 == var_44)
    assert var_45 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_0.Name(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = 1
    var_10 = []
    var_11 = 'value'
    var_12 = {var_11: var_9}
    var_13 = module_0.Constant(*var_10, **var_12)
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = {var_15: var_8, var_16: var_13}
    var_18 = module_0.Assign(*var_14, **var_17)
    var_19 = [var_18]
    var_20 = []
    var_21 = 'body'
    var_22 = 'handlers'
    var_23 = 'orelse'
    var_24 = 'finalbody'
    var_25 = {var_21: var_0, var_22: var_1, var_23: var_2, var_24: var_19}
    var_26 = module_0.Try(*var_20, **var_25)
    var_27 = [var_26]
    var_28 = module_1.walk_body(var_27)
    var_29 = list(var_28)
    var_30 = []
    var_31 = 'id'
    var_32 = {var_31: var_3}
    var_33 = module_0.Name(*var_30, **var_32)
    var_34 = [var_33]
    var_35 = []
    var_36 = 'value'
    var_37 = {var_36: var_9}
    var_38 = module_0.Constant(*var_35, **var_37)
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_34, var_41: var_38}
    var_43 = module_0.Assign(*var_39, **var_42)
    var_44 = [var_43]
    var_45 = bool(var_29 == var_44)
    assert var_45 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'y'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = 2
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_10, var_18: var_15}
    var_20 = module_0.Assign(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = 'test'
    var_25 = 'body'
    var_26 = 'orelse'
    var_27 = {var_24: var_4, var_25: var_21, var_26: var_22}
    var_28 = module_0.If(*var_23, **var_27)
    var_29 = []
    var_30 = 'value'
    var_31 = {var_30: var_0}
    var_32 = module_0.Constant(*var_29, **var_31)
    var_33 = [var_28]
    var_34 = []
    var_35 = []
    var_36 = 'test'
    var_37 = 'body'
    var_38 = 'orelse'
    var_39 = {var_36: var_32, var_37: var_33, var_38: var_34}
    var_40 = module_0.If(*var_35, **var_39)
    var_41 = [var_40]
    var_42 = module_1.walk_body(var_41)
    var_43 = list(var_42)
    var_44 = []
    var_45 = 'id'
    var_46 = {var_45: var_5}
    var_47 = module_0.Name(*var_44, **var_46)
    var_48 = [var_47]
    var_49 = []
    var_50 = 'value'
    var_51 = {var_50: var_11}
    var_52 = module_0.Constant(*var_49, **var_51)
    var_53 = []
    var_54 = 'targets'
    var_55 = 'value'
    var_56 = {var_54: var_48, var_55: var_52}
    var_57 = module_0.Assign(*var_53, **var_56)
    var_58 = [var_28, var_57]
    var_59 = bool(var_43 == var_58)
    assert var_59 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'x'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_0}
    var_14 = module_0.Constant(*var_11, **var_13)
    var_15 = []
    var_16 = 'targets'
    var_17 = 'value'
    var_18 = {var_16: var_10, var_17: var_14}
    var_19 = module_0.Assign(*var_15, **var_18)
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = 'test'
    var_24 = 'body'
    var_25 = 'orelse'
    var_26 = {var_23: var_4, var_24: var_20, var_25: var_21}
    var_27 = module_0.If(*var_22, **var_26)
    var_28 = 'y'
    var_29 = []
    var_30 = 'id'
    var_31 = {var_30: var_28}
    var_32 = module_0.Name(*var_29, **var_31)
    var_33 = [var_32]
    var_34 = 2
    var_35 = []
    var_36 = 'value'
    var_37 = {var_36: var_34}
    var_38 = module_0.Constant(*var_35, **var_37)
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_33, var_41: var_38}
    var_43 = module_0.Assign(*var_39, **var_42)
    var_44 = [var_43]
    var_45 = []
    var_46 = []
    var_47 = []
    var_48 = []
    var_49 = 'body'
    var_50 = 'handlers'
    var_51 = 'orelse'
    var_52 = 'finalbody'
    var_53 = {var_49: var_44, var_50: var_45, var_51: var_46, var_52: var_47}
    var_54 = module_0.Try(*var_48, **var_53)
    var_55 = [var_27, var_54]
    var_56 = module_1.walk_body(var_55)
    var_57 = list(var_56)
    var_58 = []
    var_59 = 'id'
    var_60 = {var_59: var_5}
    var_61 = module_0.Name(*var_58, **var_60)
    var_62 = [var_61]
    var_63 = []
    var_64 = 'value'
    var_65 = {var_64: var_0}
    var_66 = module_0.Constant(*var_63, **var_65)
    var_67 = []
    var_68 = 'targets'
    var_69 = 'value'
    var_70 = {var_68: var_62, var_69: var_66}
    var_71 = module_0.Assign(*var_67, **var_70)
    var_72 = []
    var_73 = 'id'
    var_74 = {var_73: var_28}
    var_75 = module_0.Name(*var_72, **var_74)
    var_76 = [var_75]
    var_77 = []
    var_78 = 'value'
    var_79 = {var_78: var_34}
    var_80 = module_0.Constant(*var_77, **var_79)
    var_81 = []
    var_82 = 'targets'
    var_83 = 'value'
    var_84 = {var_82: var_76, var_83: var_80}
    var_85 = module_0.Assign(*var_81, **var_84)
    var_86 = [var_71, var_54, var_85]
    var_87 = bool(var_57 == var_86)
    assert var_87 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 6/11 statements.
# Partially parsed test_globals_with_all. Retrieved 9/14 statements.
# Partially parsed test_globals_ignores_non_simple_assign. Retrieved 6/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = 'int'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = 42
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = 1
    var_17 = 'test_module'
    var_18 = var_0.alias['test_module.test_var']
    assert var_18 == '42'
    var_19 = var_0.const['test_module.TEST_VAR']
    assert var_19 == 'int'
    var_20 = var_0.root['test_module.TEST_VAR']
    assert var_20 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = 42
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'int'
    var_8 = 'test_module'
    var_9 = var_0.alias['test_module.test_var']
    assert var_9 == '42'
    var_10 = var_0.const['test_module.TEST_VAR']
    assert var_10 == 'int'
    var_11 = var_0.root['test_module.TEST_VAR']
    assert var_11 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'public_func'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'PublicClass'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_6, var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.List(*var_13, **var_15)
    var_17 = 'test_module'
    var_18 = var_0.imp['test_module']
    var_19 = bool(var_0.imp['test_module'] == {'test_module.public_func', 'test_module.PublicClass'})
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'test_module'
    var_9 = 'test_module.x'
    var_10 = bool('test_module.x' not in var_0.alias)
    assert var_10 is True
    var_11 = 'test_module.y'
    var_12 = bool('test_module.y' not in var_0.alias)
    assert var_12 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_is_public_predicate_false. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'child'
    var_3 = {var_2}
    var_4 = 'root.child'
    var_5 = var_0.is_public(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #47
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'foo'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.Tuple(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_6, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'root'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = '__all__'
    var_25 = bool('__all__' not in var_0.imp['root'])
    assert var_25 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_handlers_is_sequence_of_except_handlers. Retrieved 3/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []



# Parsed testcases at query #49
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = None
    var_2 = []
    var_3 = 0
    var_4 = []
    var_5 = 'module'
    var_6 = 'names'
    var_7 = 'level'
    var_8 = {var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_1.ImportFrom(*var_4, **var_8)
    var_10 = 'root'
    var_11 = var_0.imports(var_10, var_9)
    var_12 = var_0.alias
    var_13 = len(var_12)
    assert var_13 == 0



# Parsed testcases at query #50
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = 1
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = {var_17: var_5, var_18: var_10, var_19: var_15}
    var_21 = module_1.AnnAssign(*var_16, **var_20)
    var_22 = 'module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = var_0.alias['module.x']
    assert var_24 == '1'
    var_25 = var_0.const['module.x']
    assert var_25 == 'int'
    var_26 = var_0.root['module.x']
    assert var_26 == 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'hello'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'module'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['module.y']
    assert var_19 == "'hello'"
    var_20 = var_0.const['module.y']
    assert var_20 == 'str'
    var_21 = var_0.root['module.y']
    assert var_21 == 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 3.14
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'float'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    var_19 = 'module'
    var_20 = var_0.globals(var_19, var_18)
    var_21 = var_0.alias['module.z']
    assert var_21 == '3.14'
    var_22 = var_0.const['module.z']
    assert var_22 == 'float'
    var_23 = var_0.root['module.z']
    assert var_23 == 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'func1'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'func2'
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = [var_11, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = {var_19: var_17}
    var_21 = module_1.List(*var_18, **var_20)
    var_22 = []
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_6, var_24: var_21}
    var_26 = module_1.Assign(*var_22, **var_25)
    var_27 = 'module'
    var_28 = var_0.globals(var_27, var_26)
    var_29 = var_0.imp['module']
    var_30 = bool(var_0.imp['module'] == {'module.func1', 'module.func2'})
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'b'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = [var_5, var_10]
    var_12 = 42
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_11, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = 'module.a'
    var_25 = bool('module.a' not in var_0.alias)
    assert var_25 is True
    var_26 = 'module.b'
    var_27 = bool('module.b' not in var_0.alias)
    assert var_27 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'not_constant'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 100
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'module'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = 'module.not_constant'
    var_20 = bool('module.not_constant' not in var_0.const)
    assert var_20 is True



# Parsed testcases at query #51
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private_attr'
    var_2 = module_0.is_public_family(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #52
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Set(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'set[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 'a'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 'b'
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[int, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = 'func'
    var_11 = {var_10: var_8}
    var_12 = module_0.Call(*var_9, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = 'func'
    var_11 = {var_10: var_8}
    var_12 = module_0.Call(*var_9, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'float'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = 'func'
    var_11 = {var_10: var_8}
    var_12 = module_0.Call(*var_9, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'float'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'complex'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = 'func'
    var_11 = {var_10: var_8}
    var_12 = module_0.Call(*var_9, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'complex'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = 'func'
    var_11 = {var_10: var_8}
    var_12 = module_0.Call(*var_9, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = 'func'
    var_11 = {var_10: var_8}
    var_12 = module_0.Call(*var_9, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'x'
    var_6 = []
    var_7 = {}
    var_8 = module_0.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_0.Name(*var_9, **var_12)
    var_14 = [var_4, var_13]
    var_15 = []
    var_16 = 'elts'
    var_17 = {var_16: var_14}
    var_18 = module_0.Tuple(*var_15, **var_17)
    var_19 = module_1.const_type(var_18)
    assert var_19 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.Tuple(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'tuple[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.List(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'list[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_0.Set(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'set[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'keys'
    var_4 = 'values'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Dict(*var_2, **var_5)
    var_7 = module_1.const_type(var_6)
    assert var_7 == 'dict[]'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_visit_Constant_valid_name. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.Name'
    var_2 = 'alias.Name'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Name'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Constant(*var_6, **var_7)
    var_9 = var_4.visit_Constant(var_8)
    var_10 = var_9.id
    assert var_10 == 'Name'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_enums_table_generation. Retrieved 7/48 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'RED'
    var_2 = None
    var_3 = 'GREEN'
    var_4 = 'BLUE'
    var_5 = 'root'
    var_6 = 'Color'
    var_7 = 'Enums'
    var_8 = bool('Enums' in var_0.doc['root.Color'])
    assert var_8 is True
    var_9 = 'RED'
    var_10 = bool('RED' in var_0.doc['root.Color'])
    assert var_10 is True
    var_11 = 'GREEN'
    var_12 = bool('GREEN' in var_0.doc['root.Color'])
    assert var_12 is True
    var_13 = 'BLUE'
    var_14 = bool('BLUE' in var_0.doc['root.Color'])
    assert var_14 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_func_ann_with_has_self_and_cls_method. Retrieved 14/17 statements.
# Partially parsed test_func_ann_with_has_self_no_cls_method. Retrieved 13/16 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 14/17 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 9/12 statements.
# Partially parsed test_func_ann_with_self_type_annotation. Retrieved 15/18 statements.
# Partially parsed test_func_ann_with_cls_method_and_self_type. Retrieved 16/19 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.arg(*var_4, **var_5)
    var_7 = 'x'
    var_8 = 'int'
    var_9 = []
    var_10 = {}
    var_11 = module_0.Load(*var_9, **var_10)
    var_12 = [var_8, var_11]
    var_13 = {}
    var_14 = module_0.Name(*var_12, **var_13)
    var_15 = [var_7, var_14]
    var_16 = {}
    var_17 = module_0.arg(*var_15, **var_16)
    var_18 = [var_6, var_17]
    var_19 = 'module'
    var_20 = True
    var_21 = True

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.arg(*var_4, **var_5)
    var_7 = 'x'
    var_8 = 'int'
    var_9 = []
    var_10 = {}
    var_11 = module_0.Load(*var_9, **var_10)
    var_12 = [var_8, var_11]
    var_13 = {}
    var_14 = module_0.Name(*var_12, **var_13)
    var_15 = [var_7, var_14]
    var_16 = {}
    var_17 = module_0.arg(*var_15, **var_16)
    var_18 = [var_6, var_17]
    var_19 = 'module'
    var_20 = True

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'x'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.arg(*var_4, **var_5)
    var_7 = '*'
    var_8 = [var_7, var_3]
    var_9 = {}
    var_10 = module_0.arg(*var_8, **var_9)
    var_11 = 'y'
    var_12 = 'str'
    var_13 = []
    var_14 = {}
    var_15 = module_0.Load(*var_13, **var_14)
    var_16 = [var_12, var_15]
    var_17 = {}
    var_18 = module_0.Name(*var_16, **var_17)
    var_19 = [var_11, var_18]
    var_20 = {}
    var_21 = module_0.arg(*var_19, **var_20)
    var_22 = [var_6, var_10, var_21]
    var_23 = 'module'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'x'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.arg(*var_4, **var_5)
    var_7 = 'y'
    var_8 = [var_7, var_3]
    var_9 = {}
    var_10 = module_0.arg(*var_8, **var_9)
    var_11 = [var_6, var_10]
    var_12 = 'module'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'self'
    var_3 = 'MyClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.Load(*var_4, **var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.Name(*var_7, **var_8)
    var_10 = [var_2, var_9]
    var_11 = {}
    var_12 = module_0.arg(*var_10, **var_11)
    var_13 = 'x'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_0.Load(*var_15, **var_16)
    var_18 = [var_14, var_17]
    var_19 = {}
    var_20 = module_0.Name(*var_18, **var_19)
    var_21 = [var_13, var_20]
    var_22 = {}
    var_23 = module_0.arg(*var_21, **var_22)
    var_24 = [var_12, var_23]
    var_25 = 'module'
    var_26 = True

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'cls'
    var_3 = 'type[MyClass]'
    var_4 = []
    var_5 = {}
    var_6 = module_0.Load(*var_4, **var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.Name(*var_7, **var_8)
    var_10 = [var_2, var_9]
    var_11 = {}
    var_12 = module_0.arg(*var_10, **var_11)
    var_13 = 'x'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_0.Load(*var_15, **var_16)
    var_18 = [var_14, var_17]
    var_19 = {}
    var_20 = module_0.Name(*var_18, **var_19)
    var_21 = [var_13, var_20]
    var_22 = {}
    var_23 = module_0.arg(*var_21, **var_22)
    var_24 = [var_12, var_23]
    var_25 = 'module'
    var_26 = True
    var_27 = True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_func_api_basic. Retrieved 13/16 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 17/20 statements.
# Partially parsed test_func_api_with_self. Retrieved 16/19 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 15/18 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 17/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.function'
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = None
    var_15 = False
    var_16 = var_0.doc[var_2]
    assert var_16 == '| x | return |\n|:---:|:---:|\n| `str` | `str` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.function'
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = 'y'
    var_10 = [var_9, var_5]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_8, var_12]
    var_14 = []
    var_15 = 1
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_15}
    var_19 = module_1.Constant(*var_16, **var_18)
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = None
    var_24 = False
    var_25 = var_0.doc[var_2]
    assert var_25 == '| x | y | return |\n|:---:|:---:|:---:|\n| `str` | `str` | `str` |\n|  | `1` |  |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class.method'
    var_3 = []
    var_4 = 'self'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = 'x'
    var_10 = [var_9, var_5]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_8, var_12]
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = True
    var_20 = False
    var_21 = var_0.doc[var_2]
    assert var_21 == '| self | x | return |\n|:---:|:---:|:---:|\n| `Self` | `str` | `str` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class.method'
    var_3 = []
    var_4 = 'cls'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = 'x'
    var_10 = [var_9, var_5]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_8, var_12]
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = True
    var_20 = var_0.doc[var_2]
    assert var_20 == '| cls | x | return |\n|:---:|:---:|:---:|\n| `type[Self]` | `str` | `str` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.function'
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'kwargs'
    var_14 = [var_13, var_5]
    var_15 = {}
    var_16 = module_1.arg(*var_14, **var_15)
    var_17 = 'args'
    var_18 = [var_17, var_5]
    var_19 = {}
    var_20 = module_1.arg(*var_18, **var_19)
    var_21 = []
    var_22 = None
    var_23 = False
    var_24 = var_0.doc[var_2]
    assert var_24 == '| x | *args | **kwargs | return |\n|:---:|:---:|:---:|:---:|\n| `str` | `str` | `str` | `str` |\n\n'



# Parsed testcases at query #57
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'SomeClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = []
    var_12 = 'arg'
    var_13 = 'annotation'
    var_14 = {var_12: var_1, var_13: var_10}
    var_15 = module_1.arg(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = 'root'
    var_18 = True
    var_19 = var_0.func_ann(var_17, var_16, has_self=var_18, cls_method=var_18)
    var_20 = list(var_19)
    var_21 = var_20[0]
    assert var_21 == 'type[Self]'



# Parsed testcases at query #58
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Tuple(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'tuple[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.List(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'list[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Set(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'set[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_0.Set(*var_11, **var_13)
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'set[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 'a'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 'b'
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[int, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 'b'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = 'c'
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_0.Constant(*var_17, **var_19)
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_0.Dict(*var_22, **var_25)
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'dict[Any, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = 'a'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_0}
    var_19 = module_0.Constant(*var_16, **var_18)
    var_20 = [var_15, var_19]
    var_21 = []
    var_22 = 'keys'
    var_23 = 'values'
    var_24 = {var_22: var_10, var_23: var_20}
    var_25 = module_0.Dict(*var_21, **var_24)
    var_26 = module_1.const_type(var_25)
    assert var_26 == 'dict[int, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = 'func'
    var_7 = {var_6: var_4}
    var_8 = module_0.Call(*var_5, **var_7)
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = 'func'
    var_7 = {var_6: var_4}
    var_8 = module_0.Call(*var_5, **var_7)
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = 'func'
    var_7 = {var_6: var_4}
    var_8 = module_0.Call(*var_5, **var_7)
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'Any'



# Parsed testcases at query #59
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = module_0.is_public_family(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #60
#--------------------------




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
    var_1 = True
    var_2 = module_0.Parser(var_0, toc=var_1)
    var_3 = var_2.link
    assert var_3 is True
    var_4 = var_2.toc
    assert var_4 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_func_api_with_kwarg. Retrieved 15/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'kwargs'
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = []
    var_14 = None
    var_15 = False
    var_16 = var_0.doc[var_2]
    var_17 = '\n\n| **kwargs | Any |\n'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 16/29 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'public_attr'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    var_23 = None
    var_24 = 'another_attr'
    var_25 = 42
    var_26 = []
    var_27 = 'value'
    var_28 = {var_27: var_25}
    var_29 = module_1.Constant(*var_26, **var_28)
    var_30 = 'deleted_attr'
    var_31 = 'Bases'
    var_32 = bool('Bases' in var_0.doc[var_2])
    assert var_32 is True
    var_33 = 'Members'
    var_34 = bool('Members' in var_0.doc[var_2])
    assert var_34 is True
    var_35 = 'public_attr'
    var_36 = bool('public_attr' in var_0.doc[var_2])
    assert var_36 is True
    var_37 = 'another_attr'
    var_38 = bool('another_attr' in var_0.doc[var_2])
    assert var_38 is True
    var_39 = 'deleted_attr'
    var_40 = bool('deleted_attr' not in var_0.doc[var_2])
    assert var_40 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'VALUE1'
    var_14 = None
    var_15 = 'VALUE2'
    var_16 = 'Enums'
    var_17 = bool('Enums' in var_0.doc[var_2])
    assert var_17 is True
    var_18 = 'VALUE1'
    var_19 = bool('VALUE1' in var_0.doc[var_2])
    assert var_19 is True
    var_20 = 'VALUE2'
    var_21 = bool('VALUE2' in var_0.doc[var_2])
    assert var_21 is True
    var_22 = 'Members'
    var_23 = bool('Members' not in var_0.doc[var_2])
    assert var_23 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc[var_2])
    assert var_9 is True
    var_10 = 'Enums'
    var_11 = bool('Enums' not in var_0.doc[var_2])
    assert var_11 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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
    var_1 = 2
    var_2 = True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_nested_public_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_nested_private_name. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_const_in_all. Retrieved 7/9 statements.
# Partially parsed test_is_public_without_all_but_public_family. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_parent_in_all. Retrieved 7/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'pkg.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg._private_name'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.__magic__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = {var_2}
    var_4 = 'pkg.subpkg.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = {var_2}
    var_4 = 'pkg.subpkg._private_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'CONST'
    var_3 = {var_2}
    var_4 = 'pkg.CONST'
    var_5 = 'int'
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.public_name'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = {var_2}
    var_4 = 'pkg.subpkg'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True



# Parsed testcases at query #65
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = False
    var_10 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == ['Self'])
    assert var_12 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_imports_with_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'path'
    var_3 = None
    var_4 = 1
    var_5 = 'pkg.subpkg'
    var_6 = var_0.alias['pkg.subpkg.path']
    assert var_6 == 'sys.path'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_class_api_with_enum_members. Retrieved 14/22 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 12/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 9/18 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 9/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = var_0.class_api(var_1, var_2, var_12, var_13)
    var_15 = 'Bases'
    var_16 = bool('Bases' in var_0.doc[var_2])
    assert var_16 is True
    var_17 = 'BaseClass'
    var_18 = bool('BaseClass' in var_0.doc[var_2])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.EnumClass'
    var_3 = 'enum.Enum'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'MEMBER1'
    var_14 = 1
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = None
    var_20 = 'MEMBER2'
    var_21 = 2
    var_22 = []
    var_23 = 'value'
    var_24 = {var_23: var_21}
    var_25 = module_1.Constant(*var_22, **var_24)
    var_26 = 'Enums'
    var_27 = bool('Enums' in var_0.doc[var_2])
    assert var_27 is True
    var_28 = 'MEMBER1'
    var_29 = bool('MEMBER1' in var_0.doc[var_2])
    assert var_29 is True
    var_30 = 'MEMBER2'
    var_31 = bool('MEMBER2' in var_0.doc[var_2])
    assert var_31 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 'another_attr'
    var_16 = 42
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = 'Members'
    var_22 = bool('Members' in var_0.doc[var_2])
    assert var_22 is True
    var_23 = 'public_attr'
    var_24 = bool('public_attr' in var_0.doc[var_2])
    assert var_24 is True
    var_25 = 'another_attr'
    var_26 = bool('another_attr' in var_0.doc[var_2])
    assert var_26 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 'public_attr'
    var_16 = bool('public_attr' not in var_0.doc[var_2])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = None
    var_15 = 'Members'
    var_16 = bool('Members' not in var_0.doc[var_2])
    assert var_16 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test__attr_with_nonexistent_attribute. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_func_api_with_kwonlyargs. Retrieved 19/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = 'kw1'
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = 'kw2'
    var_11 = [var_10, var_5]
    var_12 = {}
    var_13 = module_1.arg(*var_11, **var_12)
    var_14 = [var_9, var_13]
    var_15 = []
    var_16 = []
    var_17 = None
    var_18 = False
    var_19 = False
    var_20 = var_0.doc[var_2]
    var_21 = '()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'
    var_22 = '.'
    var_23 = '-'



