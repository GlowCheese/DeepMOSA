####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = module_0._m(*var_1)
    assert var_2 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'urllib'
    var_1 = 'request'
    var_2 = [var_0, var_1]
    var_3 = module_0._m(*var_2)
    assert var_3 == 'urllib.request'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = ''
    var_2 = 'os'
    var_3 = [var_0, var_1, var_2, var_1]
    var_4 = module_0._m(*var_3)
    assert var_4 == 'sys.os'

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = [var_0, var_0, var_1]
    var_3 = module_0._m(*var_2)
    assert var_3 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._m(*var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = module_0._m(*var_1)
    assert var_2 == ''



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__.module.__name__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._submodule.class'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule._attribute'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_internal'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__main__.module.sub'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_imports_import_statement. Retrieved 2/10 statements.
# Partially parsed test_parser_imports_import_as_statement. Retrieved 2/10 statements.
# Partially parsed test_parser_imports_from_import_relative_level_1. Retrieved 5/9 statements.
# Partially parsed test_parser_imports_from_import_absolute. Retrieved 6/10 statements.
# Partially parsed test_parser_imports_from_import_with_asname. Retrieved 6/10 statements.
# Partially parsed test_parser_imports_from_import_deep_relative. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = var_0.alias['pkg.module.os']
    assert var_2 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = var_0.alias['pkg.module.sys_os']
    assert var_2 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = None
    var_3 = 'sibling'
    var_4 = 1
    var_5 = var_0.alias['pkg']
    assert var_5 == 'sibling'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg'
    var_3 = 'sibling'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['pkg.sibling']
    assert var_6 == 'sibling'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg'
    var_3 = 'sibling'
    var_4 = 'sib'
    var_5 = 0
    var_6 = var_0.alias['pkg.sib']
    assert var_6 == 'sibling'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a.b.c'
    var_2 = 'sub'
    var_3 = 'func'
    var_4 = None
    var_5 = 2
    var_6 = var_0.alias['a.sub.func']
    assert var_6 == 'func'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parser_globals_assign_with_type_comment. Retrieved 11/15 statements.
# Partially parsed test_parser_globals_assign_without_type_comment. Retrieved 10/12 statements.
# Partially parsed test_parser_globals_annassign. Retrieved 11/14 statements.
# Partially parsed test_parser_globals_ignores_non_uppercase_for_const. Retrieved 10/13 statements.
# Partially parsed test_parser_globals_all_filter. Retrieved 15/18 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'MY_CONST'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Name(*var_2, **var_5)
    var_7 = [var_6]
    var_8 = 10
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_0.Constant(*var_9, **var_11)
    var_13 = 'int'
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = 'type_comment'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_0.Assign(*var_14, **var_18)
    var_20 = module_1.Parser()
    var_21 = 'pkg'
    var_22 = var_20.globals(var_21, var_19)
    var_23 = 'pkg.MY_CONST'
    var_24 = bool('pkg.MY_CONST' in var_20.alias)
    assert var_24 is True
    var_25 = var_20.alias['pkg.MY_CONST']
    assert var_25 == '10'
    var_26 = var_20.const['pkg.MY_CONST']
    assert var_26 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'OTHER_CONST'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Name(*var_2, **var_5)
    var_7 = [var_6]
    var_8 = 'hello'
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_0.Constant(*var_9, **var_11)
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_7, var_15: var_12, var_16: var_1}
    var_18 = module_0.Assign(*var_13, **var_17)
    var_19 = module_1.Parser()
    var_20 = 'pkg'
    var_21 = var_19.globals(var_20, var_18)
    var_22 = 'pkg.OTHER_CONST'
    var_23 = bool('pkg.OTHER_CONST' in var_19.alias)
    assert var_23 is True
    var_24 = var_19.alias['pkg.DISCARDED']
    assert var_24 is None
    var_25 = var_19.const['pkg.OTHER_CONST']
    assert var_25 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'VAL'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Name(*var_2, **var_5)
    var_7 = 5
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_0.Constant(*var_8, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_12, var_15: var_1}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = []
    var_19 = 'target'
    var_20 = 'value'
    var_21 = 'annotation'
    var_22 = {var_19: var_6, var_20: var_11, var_21: var_17}
    var_23 = module_0.AnnAssign(*var_18, **var_22)
    var_24 = module_1.Parser()
    var_25 = 'pkg'
    var_26 = var_24.globals(var_25, var_23)
    var_27 = 'pkg.VAL'
    var_28 = bool('pkg.VAL' in var_24.alias)
    assert var_28 is True
    var_29 = var_24.alias['pkg.VAL']
    assert var_29 == '5'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'lowercase_var'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Name(*var_2, **var_5)
    var_7 = [var_6]
    var_8 = 1
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_0.Constant(*var_9, **var_11)
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_7, var_15: var_12, var_16: var_1}
    var_18 = module_0.Assign(*var_13, **var_17)
    var_19 = module_1.Parser()
    var_20 = 'pkg'
    var_21 = var_19.globals(var_20, var_18)
    var_22 = 'pkg.lowercase_var'
    var_23 = bool('pkg.lowercase_var' in var_19.alias)
    assert var_23 is True
    var_24 = 'pkg.lowercase_var'
    var_25 = bool('pkg.lowercase_var' not in var_19.const)
    assert var_25 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '__all__'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Name(*var_2, **var_5)
    var_7 = [var_6]
    var_8 = 'sub'
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_0.Constant(*var_9, **var_11)
    var_13 = 'other'
    var_14 = []
    var_15 = 'value'
    var_16 = {var_15: var_13}
    var_17 = module_0.Constant(*var_14, **var_16)
    var_18 = [var_12, var_17]
    var_19 = []
    var_20 = 'elts'
    var_21 = 'ctx'
    var_22 = {var_20: var_18, var_21: var_1}
    var_23 = module_0.Tuple(*var_19, **var_22)
    var_24 = []
    var_25 = 'targets'
    var_26 = 'value'
    var_27 = {var_25: var_7, var_26: var_23}
    var_28 = module_0.Assign(*var_24, **var_27)
    var_29 = module_1.Parser()
    var_30 = 'pkg'
    var_31 = set()
    var_32 = var_29.globals(var_30, var_28)
    var_33 = 'pkg.sub'
    var_34 = bool('pkg.sub' in var_29.imp['pkg'])
    assert var_34 is True
    var_35 = 'pkg.other'
    var_36 = bool('pkg.other' in var_29.items_found)
    assert var_36 is True
    var_37 = 'pkg.sub'
    var_38 = bool('pkg.sub' in var_29.imp['pkg'])
    assert var_38 is True



# Parsed testcases at query #5
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [' ', ' '])
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parser_globals_assignment_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_parser_globals_assignment_without_type_comment. Retrieved 6/14 statements.
# Partially parsed test_parser_globals_all_filter_updates_imports. Retrieved 9/20 statements.
# Partially parsed test_parser_globals_ignores_non_assignment_nodes. Retrieved 6/10 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'pkg.module'
    var_1 = 'MY_CONSTANT'
    var_2 = 10
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_0.Constant(*var_3, **var_5)
    var_7 = 'int'
    var_8 = module_1.Parser()
    var_9 = var_8.alias[f'{var_0}.MY_CONSTANT']
    assert var_9 == '10'
    var_10 = var_8.const[f'{var_0}.MY_CONSTANT']
    assert var_10 == 'int'
    var_11 = var_8.root[f'{var_0}.MY_CONSTANT']
    var_12 = bool(var_8.root[f'{var_0}.MY_CONSTANT'] == var_0)
    assert var_12 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'pkg.module'
    var_1 = 'OTHER_CONST'
    var_2 = 'hello'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_0.Constant(*var_3, **var_5)
    var_7 = None
    var_8 = module_1.Parser()
    var_9 = var_8.alias[f'{var_0}.OTHER_CONST']
    assert var_9 == "'hello'"
    var_10 = var_8.const[f'{var_0}.OTHER_CONST']
    assert var_10 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'pkg.module'
    var_1 = '__all__'
    var_2 = 'sub_mod'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_0.Constant(*var_3, **var_5)
    var_7 = 'other_mod'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_0.Constant(*var_8, **var_10)
    var_12 = [var_6, var_11]
    var_13 = None
    var_14 = module_1.Parser()
    var_15 = 'pkg.module.sub_mod'
    var_16 = bool('pkg.module.sub_mod' in var_14.imp[var_0])
    assert var_16 is True
    var_17 = 'pkg.module.other_mod'
    var_18 = bool('pkg.module.other_mod' in var_14.imp[var_0])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'pkg.module'
    var_1 = module_0.Parser()
    var_2 = var_1.alias
    var_3 = len(var_2)
    var_4 = var_1.alias
    var_5 = len(var_4)
    var_6 = bool(var_5 == var_3)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_func_ann_logic_basic. Retrieved 22/26 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'self'
    var_3 = 'Self'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = []
    var_13 = 'arg'
    var_14 = 'annotation'
    var_15 = {var_13: var_2, var_14: var_11}
    var_16 = module_1.arg(*var_12, **var_15)
    var_17 = 'x'
    var_18 = 'int'
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = []
    var_23 = 'id'
    var_24 = 'ctx'
    var_25 = {var_23: var_18, var_24: var_21}
    var_26 = module_1.Name(*var_22, **var_25)
    var_27 = []
    var_28 = 'arg'
    var_29 = 'annotation'
    var_30 = {var_28: var_17, var_29: var_26}
    var_31 = module_1.arg(*var_27, **var_30)
    var_32 = [var_16, var_31]
    var_33 = 'str'
    var_34 = []
    var_35 = {}
    var_36 = module_1.Load(*var_34, **var_35)
    var_37 = []
    var_38 = 'id'
    var_39 = 'ctx'
    var_40 = {var_38: var_33, var_39: var_36}
    var_41 = module_1.Name(*var_37, **var_40)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'int'
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
    var_16 = 'b'
    var_17 = None
    var_18 = []
    var_19 = 'arg'
    var_20 = 'annotation'
    var_21 = {var_19: var_16, var_20: var_17}
    var_22 = module_1.arg(*var_18, **var_21)
    var_23 = [var_15, var_22]
    var_24 = []
    var_25 = 'value'
    var_26 = {var_25: var_17}
    var_27 = module_1.Constant(*var_24, **var_26)
    var_28 = module_0.Parser()
    var_29 = 'x'
    var_30 = []
    var_31 = {}
    var_32 = module_1.Load(*var_30, **var_31)
    var_33 = []
    var_34 = 'id'
    var_35 = 'ctx'
    var_36 = {var_34: var_2, var_35: var_32}
    var_37 = module_1.Name(*var_33, **var_36)
    var_38 = []
    var_39 = 'arg'
    var_40 = 'annotation'
    var_41 = {var_39: var_29, var_40: var_37}
    var_42 = module_1.arg(*var_38, **var_41)
    var_43 = 'y'
    var_44 = []
    var_45 = 'arg'
    var_46 = 'annotation'
    var_47 = {var_45: var_43, var_46: var_17}
    var_48 = module_1.arg(*var_44, **var_47)
    var_49 = [var_42, var_48]
    var_50 = []
    var_51 = 'value'
    var_52 = {var_51: var_17}
    var_53 = module_1.Constant(*var_50, **var_52)
    var_54 = 'mod'
    var_55 = False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = True
    var_1 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.toc
    assert var_2 is True
    var_3 = var_1.link
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_parser_class_api_with_members_and_bases. Retrieved 20/37 statements.
# Partially parsed test_parser_class_api_with_enum_style. Retrieved 17/29 statements.
# Partially parsed test_parser_class_api_deletion. Retrieved 18/29 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'MyClass'
    var_4 = 'BaseClass'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = 'public_attr'
    var_17 = 'int'
    var_18 = []
    var_19 = {}
    var_20 = module_1.Load(*var_18, **var_19)
    var_21 = []
    var_22 = 'id'
    var_23 = 'ctx'
    var_24 = {var_22: var_17, var_23: var_20}
    var_25 = module_1.Name(*var_21, **var_24)
    var_26 = []
    var_27 = 'value'
    var_28 = {var_27: var_0}
    var_29 = module_1.Constant(*var_26, **var_28)
    var_30 = 'PRIVATE_ATTR'
    var_31 = 2
    var_32 = []
    var_33 = 'value'
    var_34 = {var_33: var_31}
    var_35 = module_1.Constant(*var_32, **var_34)
    var_36 = 'pkg'
    var_37 = 'pkg.'
    var_38 = 'pkg.MyClass'
    var_39 = bool('pkg.MyClass' in var_2.doc)
    assert var_39 is True
    var_40 = 'pkg.MyClass'
    var_41 = bool('pkg.MyClass' in var_2.level)
    assert var_41 is True
    var_42 = 'Members'
    var_43 = bool('Members' in var_2.doc['pkg.MyClass'])
    assert var_43 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'MyEnum'
    var_4 = 'enum'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = 'Enum'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'value'
    var_19 = 'attr'
    var_20 = 'ctx'
    var_21 = {var_18: var_12, var_19: var_13, var_20: var_16}
    var_22 = module_1.Attribute(*var_17, **var_21)
    var_23 = [var_22]
    var_24 = []
    var_25 = []
    var_26 = 'RED'
    var_27 = []
    var_28 = 'value'
    var_29 = {var_28: var_0}
    var_30 = module_1.Constant(*var_27, **var_29)
    var_31 = 'pkg'
    var_32 = 'pkg.'
    var_33 = 'Enums'
    var_34 = bool('Enums' in var_2.doc['pkg.MyEnum'])
    assert var_34 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'MyClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'temp'
    var_8 = 'int'
    var_9 = []
    var_10 = {}
    var_11 = module_1.Load(*var_9, **var_10)
    var_12 = []
    var_13 = 'id'
    var_14 = 'ctx'
    var_15 = {var_13: var_8, var_14: var_11}
    var_16 = module_1.Name(*var_12, **var_15)
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_0}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = []
    var_22 = {}
    var_23 = module_1.Load(*var_21, **var_22)
    var_24 = []
    var_25 = 'id'
    var_26 = 'ctx'
    var_27 = {var_25: var_7, var_26: var_23}
    var_28 = module_1.Name(*var_24, **var_27)
    var_29 = [var_28]
    var_30 = []
    var_31 = 'targets'
    var_32 = {var_31: var_29}
    var_33 = module_1.Delete(*var_30, **var_32)
    var_34 = 'pkg'
    var_35 = 'pkg.'
    var_36 = 'temp'
    var_37 = bool('temp' not in var_2.doc['pkg.MyClass'])
    assert var_37 is True



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass

import ast as module_0

def test_case_0():
    var_0 = '1'
    var_1 = module_0.parse(var_0)
    var_2 = None
    var_3 = '|'
    var_4 = module_0.parse(var_3)
    var_5 = [var_1, var_2, var_4]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '1'
    var_1 = module_0.parse(var_0)
    var_2 = None
    var_3 = '&'
    var_4 = module_0.parse(var_3)
    var_5 = [var_1, var_2, var_4]
    var_6 = module_0.parse(var_0)
    var_7 = module_0.parse(var_3)
    var_8 = [var_6, var_2, var_7]
    var_9 = module_1._defaults(var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['`1`', ' ', '<code>&</code>'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' '])
    assert var_4 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '|'
    var_1 = module_0.parse(var_0)
    var_2 = [var_1]
    var_3 = module_1._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['<code>&#124;</code>'])
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 22/37 statements.
# Partially parsed test_class_api_with_bases. Retrieved 15/20 statements.
# Partially parsed test_class_api_with_enums. Retrieved 16/28 statements.
# Partially parsed test_class_api_with_deletion. Retrieved 17/29 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = 'MyClass'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'name'
    var_10 = 'bases'
    var_11 = 'body'
    var_12 = 'decorator_list'
    var_13 = {var_9: var_4, var_10: var_5, var_11: var_6, var_12: var_7}
    var_14 = module_1.ClassDef(*var_8, **var_13)
    var_15 = 'ATTR_ONE'
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_3}
    var_19 = module_1.Constant(*var_16, **var_18)
    var_20 = 'int'
    var_21 = []
    var_22 = {}
    var_23 = module_1.Load(*var_21, **var_22)
    var_24 = []
    var_25 = 'id'
    var_26 = 'ctx'
    var_27 = {var_25: var_20, var_26: var_23}
    var_28 = module_1.Name(*var_24, **var_27)
    var_29 = 'ATTR_TWO'
    var_30 = 'str'
    var_31 = []
    var_32 = 'value'
    var_33 = {var_32: var_30}
    var_34 = module_1.Constant(*var_31, **var_33)
    var_35 = 'resolved'
    var_36 = 'other'
    var_37 = []
    var_38 = var_14.body
    var_39 = var_0.class_api(var_2, var_1, var_37, var_38)
    var_40 = 'pkg.MyClass'
    var_41 = bool('pkg.MyClass' in var_0.doc)
    assert var_41 is True
    var_42 = '| ATTR_ONE | `resolved` |'
    var_43 = bool('| ATTR_ONE | `resolved` |' in var_0.doc)
    assert var_43 is True
    var_44 = '| ATTR_TWO | `str` |'
    var_45 = bool('| ATTR_TWO | `str` |' in var_0.doc)
    assert var_45 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = 'BaseClass'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = 'MyClass'
    var_14 = [var_12]
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = 'name'
    var_19 = 'bases'
    var_20 = 'body'
    var_21 = 'decorator_list'
    var_22 = {var_18: var_13, var_19: var_14, var_20: var_15, var_21: var_16}
    var_23 = module_1.ClassDef(*var_17, **var_22)
    var_24 = [var_12]
    var_25 = []
    var_26 = var_0.class_api(var_2, var_1, var_24, var_25)
    var_27 = '| Bases |'
    var_28 = bool('| Bases |' in var_0.doc)
    assert var_28 is True
    var_29 = '| `BaseClass` |'
    var_30 = bool('| `BaseClass` |' in var_0.doc)
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyEnum'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = 'enum.Enum'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = 'VAL'
    var_14 = []
    var_15 = 'value'
    var_16 = {var_15: var_3}
    var_17 = module_1.Constant(*var_14, **var_16)
    var_18 = 'int'
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = []
    var_23 = 'id'
    var_24 = 'ctx'
    var_25 = {var_23: var_18, var_24: var_21}
    var_26 = module_1.Name(*var_22, **var_25)
    var_27 = 'MyEnum'
    var_28 = [var_12]
    var_29 = []
    var_30 = [var_12]
    var_31 = '| Enums |'
    var_32 = bool('| Enums |' in var_0.doc)
    assert var_32 is True
    var_33 = '| VAL |'
    var_34 = bool('| VAL |' in var_0.doc)
    assert var_34 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = 'TEMP'
    var_5 = []
    var_6 = 'value'
    var_7 = {var_6: var_3}
    var_8 = module_1.Constant(*var_5, **var_7)
    var_9 = 'int'
    var_10 = []
    var_11 = {}
    var_12 = module_1.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_1.Name(*var_13, **var_16)
    var_18 = []
    var_19 = {}
    var_20 = module_1.Load(*var_18, **var_19)
    var_21 = []
    var_22 = 'id'
    var_23 = 'ctx'
    var_24 = {var_22: var_4, var_23: var_20}
    var_25 = module_1.Name(*var_21, **var_24)
    var_26 = [var_25]
    var_27 = []
    var_28 = 'targets'
    var_29 = {var_28: var_26}
    var_30 = module_1.Delete(*var_27, **var_29)
    var_31 = 'MyClass'
    var_32 = []
    var_33 = []
    var_34 = []
    var_35 = 'TEMP'
    var_36 = bool('TEMP' not in var_0.doc['pkg.MyClass'])
    assert var_36 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_class_api_walks_body. Retrieved 13/21 statements.


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
    var_12 = 'X'
    var_13 = 1
    var_14 = []
    var_15 = 'value'
    var_16 = {var_15: var_13}
    var_17 = module_1.Constant(*var_14, **var_16)
    var_18 = None
    var_19 = 'pkg'
    var_20 = 'pkg.TestClass'
    var_21 = []
    var_22 = 'pkg.TestClass'
    var_23 = bool('pkg.TestClass' in var_0.doc)
    assert var_23 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_docstring_with_existing_docstring. Retrieved 2/8 statements.
# Partially parsed test_load_docstring_skips_unrelated_modules. Retrieved 3/8 statements.
# Partially parsed test_load_docstring_handles_none_docstrings. Retrieved 2/7 statements.
# Partially parsed test_load_docstring_with_submodules. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_mod'
    var_1 = module_0.Parser()
    var_2 = var_1.docstring['test_mod']
    assert var_2 == 'Module docstring'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'other_mod'
    var_1 = module_0.Parser()
    var_2 = 'root_mod'
    var_3 = 'other_mod'
    var_4 = bool('other_mod' not in var_1.docstring)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_mod'
    var_1 = module_0.Parser()
    var_2 = 'test_mod'
    var_3 = bool('test_mod' not in var_1.docstring)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'pkg.sub'
    var_1 = module_0.Parser()
    var_2 = 'sub'
    var_3 = 'pkg'
    var_4 = var_1.docstring['pkg.sub']
    assert var_4 == 'Sub module doc'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_visit_Constant_resolves_string_to_name_expression. Retrieved 7/8 statements.
# Partially parsed test_visit_Constant_recursively_resolves_nested_strings. Retrieved 8/9 statements.
# Partially parsed test_visit_Constant_handles_simple_name_without_alias. Retrieved 6/7 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax @#$'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.MyClass'
    var_2 = 'MyClass'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = []
    var_6 = 'value'
    var_7 = {var_6: var_1}
    var_8 = module_1.Constant(*var_5, **var_7)
    var_9 = var_4.visit_Constant(var_8)
    var_10 = var_9.id
    assert var_10 == 'MyClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.A'
    var_2 = 'pkg.B'
    var_3 = 'C'
    var_4 = {var_1: var_2, var_2: var_3}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_1}
    var_9 = module_1.Constant(*var_6, **var_8)
    var_10 = var_5.visit_Constant(var_9)
    var_11 = var_10.id
    assert var_11 == 'C'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'pkg.Unknown'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = var_8.id
    assert var_9 == 'Unknown'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_table_basic_functionality. Retrieved 10/11 statements.
# Partially parsed test_table_single_column. Retrieved 7/8 statements.
# Partially parsed test_table_with_long_titles. Retrieved 7/8 statements.
# Partially parsed test_table_with_single_string_item. Retrieved 4/5 statements.
# Partially parsed test_table_empty_items. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_2, var_3]
    var_5 = 'e'
    var_6 = 'f'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_1, var_8]
    var_10 = '| a | b |\n|:---:|:---:|\n| c | d |\n| e | f |\n\n'

def test_case_0():
    var_0 = 'col1'
    var_1 = 'val1'
    var_2 = [var_1]
    var_3 = 'val2'
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = [var_0, var_5]
    var_7 = '| col1 |\n|:---:|\n| val1 |\n| val2 |\n\n'

def test_case_0():
    var_0 = 'long_title_name'
    var_1 = 'short'
    var_2 = [var_1]
    var_3 = 'very_long_value'
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = [var_0, var_5]
    var_7 = '| long_title_name |\n|:----------:|\n| short |\n| very_long_value |\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = [var_0, var_2]
    var_4 = '| a |\n|:---:|\n| b |\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = []
    var_3 = [var_0, var_1, var_2]
    var_4 = '| a | b |\n|:---:|:---:|\n\n'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_visit_Constant_string_is_parseable. Retrieved 7/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = '1 + 1'
    var_5 = []
    var_6 = 'value'
    var_7 = {var_6: var_4}
    var_8 = module_1.Constant(*var_5, **var_7)
    var_9 = var_3.visit_Constant(var_8)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_func_api_simple. Retrieved 18/33 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 18/24 statements.
# Partially parsed test_func_api_classmethod. Retrieved 16/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'int'
    var_2 = [var_1]
    var_3 = iter(var_2)
    var_4 = 'test_func'
    var_5 = '## test_func()\n\n'
    var_6 = []
    var_7 = 'x'
    var_8 = None
    var_9 = []
    var_10 = 'arg'
    var_11 = 'annotation'
    var_12 = {var_10: var_7, var_11: var_8}
    var_13 = module_1.arg(*var_9, **var_12)
    var_14 = [var_13]
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = [var_1]
    var_20 = iter(var_19)
    var_21 = 'root'
    var_22 = False
    var_23 = 'int'
    var_24 = bool('int' in var_0.doc['test_func'])
    assert var_24 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'arg'
    var_5 = 'annotation'
    var_6 = {var_4: var_1, var_5: var_2}
    var_7 = module_1.arg(*var_3, **var_6)
    var_8 = 10
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = []
    var_14 = [var_7]
    var_15 = []
    var_16 = []
    var_17 = [var_12]
    var_18 = []
    var_19 = 'test_func'
    var_20 = '## test_func()\n\n'
    var_21 = 'int'
    var_22 = [var_21]
    var_23 = iter(var_22)
    var_24 = 'root'
    var_25 = False
    var_26 = '10'
    var_27 = bool('10' in var_0.doc['test_func'])
    assert var_27 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = []
    var_4 = 'arg'
    var_5 = 'annotation'
    var_6 = {var_4: var_1, var_5: var_2}
    var_7 = module_1.arg(*var_3, **var_6)
    var_8 = []
    var_9 = [var_7]
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = 'test_func'
    var_15 = '## test_func()\n\n'
    var_16 = 'type[Self]'
    var_17 = [var_16]
    var_18 = iter(var_17)
    var_19 = 'root'
    var_20 = True
    var_21 = 'type[Self]'
    var_22 = bool('type[Self]' in var_0.doc['test_func'])
    assert var_22 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_class_api_enums_removal. Retrieved 18/50 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = module_0.Parser()
    var_2 = 'VAL'
    var_3 = 1
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'int'
    var_9 = []
    var_10 = {}
    var_11 = module_1.Load(*var_9, **var_10)
    var_12 = []
    var_13 = 'id'
    var_14 = 'ctx'
    var_15 = {var_13: var_8, var_14: var_11}
    var_16 = module_1.Name(*var_12, **var_15)
    var_17 = []
    var_18 = {}
    var_19 = module_1.Load(*var_17, **var_18)
    var_20 = []
    var_21 = 'id'
    var_22 = 'ctx'
    var_23 = {var_21: var_2, var_22: var_19}
    var_24 = module_1.Name(*var_20, **var_23)
    var_25 = [var_24]
    var_26 = []
    var_27 = 'targets'
    var_28 = {var_27: var_25}
    var_29 = module_1.Delete(*var_26, **var_28)
    var_30 = 'enum_Base'
    var_31 = []
    var_32 = {}
    var_33 = module_1.Load(*var_31, **var_32)
    var_34 = []
    var_35 = 'id'
    var_36 = 'ctx'
    var_37 = {var_35: var_30, var_36: var_33}
    var_38 = module_1.Name(*var_34, **var_37)
    var_39 = 'pkg'
    var_40 = 'pkg.MyEnum'
    var_41 = [var_38]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_walk_body_simple_sequence. Retrieved 2/10 statements.
# Partially parsed test_walk_body_with_if_node. Retrieved 3/18 statements.
# Partially parsed test_walk_body_with_try_node. Retrieved 4/28 statements.
# Partially parsed test_walk_body_nested_structures. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'stmt2'

def test_case_0():
    var_0 = 'if_body'
    var_1 = 'if_else'
    var_2 = 'root'

def test_case_0():
    var_0 = 'try_body'
    var_1 = 'handler_body'
    var_2 = 'try_orelse'
    var_3 = 'try_final'

def test_case_0():
    var_0 = 'leaf'
    var_1 = []
    var_2 = []
    var_3 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_parser_globals_assignment_with_type_comment. Retrieved 8/14 statements.
# Partially parsed test_parser_globals_assignment_without_type_comment. Retrieved 5/11 statements.
# Partially parsed test_parser_globals_all_filter. Retrieved 10/16 statements.
# Partially parsed test_parser_globals_non_uppercase_not_constant. Retrieved 5/11 statements.
# Partially parsed test_parser_globals_annassign_no_value. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONST'
    var_2 = 10
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'int'
    var_8 = []
    var_9 = {}
    var_10 = module_1.Load(*var_8, **var_9)
    var_11 = []
    var_12 = 'id'
    var_13 = 'ctx'
    var_14 = {var_12: var_7, var_13: var_10}
    var_15 = module_1.Name(*var_11, **var_14)
    var_16 = 'pkg'
    var_17 = var_0.alias['pkg.MY_CONST']
    assert var_17 == '10'
    var_18 = var_0.const['pkg.MY_CONST']
    assert var_18 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'OTHER_CONST'
    var_2 = 'hello'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'pkg'
    var_8 = var_0.alias['pkg.OTHER_CONST']
    assert var_8 == "'hello'"
    var_9 = var_0.const['pkg.OTHER_CONST']
    assert var_9 == 'str'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'mod1'
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'mod2'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_6, var_11]
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'elts'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.List(*var_16, **var_19)
    var_21 = 'pkg'
    var_22 = 'pkg.mod1'
    var_23 = bool('pkg.mod1' in var_0.imp['pkg'])
    assert var_23 is True
    var_24 = 'pkg.mod2'
    var_25 = bool('pkg.mod2' in var_0.imp['pkg'])
    assert var_25 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'some_var'
    var_2 = 1
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'pkg'
    var_8 = 'pkg.some_var'
    var_9 = bool('pkg.some_var' not in var_0.const)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = None
    var_3 = 'int'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'pkg'
    var_13 = 'pkg.VAR'
    var_14 = bool('pkg.VAR' not in var_0.alias)
    assert var_14 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_func_api_basic_function. Retrieved 31/49 statements.
# Partially parsed test_func_api_class_method. Retrieved 31/49 statements.
# Partially parsed test_func_api_with_decorators. Retrieved 17/31 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg.func'
    var_5 = 'func'
    var_6 = '### pkg.func()\n\n*Full name:* `pkg.func`'
    var_7 = []
    var_8 = 'a'
    var_9 = 'int'
    var_10 = []
    var_11 = {}
    var_12 = module_1.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_1.Name(*var_13, **var_16)
    var_18 = []
    var_19 = 'arg'
    var_20 = 'annotation'
    var_21 = {var_19: var_8, var_20: var_17}
    var_22 = module_1.arg(*var_18, **var_21)
    var_23 = 'b'
    var_24 = 'str'
    var_25 = []
    var_26 = {}
    var_27 = module_1.Load(*var_25, **var_26)
    var_28 = []
    var_29 = 'id'
    var_30 = 'ctx'
    var_31 = {var_29: var_24, var_30: var_27}
    var_32 = module_1.Name(*var_28, **var_31)
    var_33 = []
    var_34 = 'arg'
    var_35 = 'annotation'
    var_36 = {var_34: var_23, var_35: var_32}
    var_37 = module_1.arg(*var_33, **var_36)
    var_38 = [var_22, var_37]
    var_39 = []
    var_40 = []
    var_41 = None
    var_42 = 'default'
    var_43 = []
    var_44 = 'value'
    var_45 = {var_44: var_42}
    var_46 = module_1.Constant(*var_43, **var_45)
    var_47 = [var_46]
    var_48 = []
    var_49 = 'bool'
    var_50 = []
    var_51 = {}
    var_52 = module_1.Load(*var_50, **var_51)
    var_53 = []
    var_54 = 'id'
    var_55 = 'ctx'
    var_56 = {var_54: var_49, var_55: var_52}
    var_57 = module_1.Name(*var_53, **var_56)
    var_58 = 'ANY'
    var_59 = '| a | b |'
    var_60 = ''

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg.method'
    var_5 = 'method'
    var_6 = '### pkg.method()\n\n*Full name:* `pkg.method`'
    var_7 = []
    var_8 = 'cls'
    var_9 = 'Self'
    var_10 = []
    var_11 = {}
    var_12 = module_1.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_1.Name(*var_13, **var_16)
    var_18 = []
    var_19 = 'arg'
    var_20 = 'annotation'
    var_21 = {var_19: var_8, var_20: var_17}
    var_22 = module_1.arg(*var_18, **var_21)
    var_23 = 'x'
    var_24 = 'int'
    var_25 = []
    var_26 = {}
    var_27 = module_1.Load(*var_25, **var_26)
    var_28 = []
    var_29 = 'id'
    var_30 = 'ctx'
    var_31 = {var_29: var_24, var_30: var_27}
    var_32 = module_1.Name(*var_28, **var_31)
    var_33 = []
    var_34 = 'arg'
    var_35 = 'annotation'
    var_36 = {var_34: var_23, var_35: var_32}
    var_37 = module_1.arg(*var_33, **var_36)
    var_38 = [var_22, var_37]
    var_39 = []
    var_40 = []
    var_41 = None
    var_42 = []
    var_43 = []
    var_44 = 'None'
    var_45 = []
    var_46 = {}
    var_47 = module_1.Load(*var_45, **var_46)
    var_48 = []
    var_49 = 'id'
    var_50 = 'ctx'
    var_51 = {var_49: var_44, var_50: var_47}
    var_52 = module_1.Name(*var_48, **var_51)
    var_53 = 'type[Self]'
    var_54 = 'ANY'
    var_55 = '| type[Self] | int |'
    var_56 = ''
    var_57 = '| type[Self] |'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg.f'
    var_5 = 'f'
    var_6 = '### pkg.f()\n\n*Full::name:* `pkg.f`'
    var_7 = 'decorator'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = None
    var_13 = []
    var_14 = [var_8, var_9, var_10, var_11, var_12, var_12, var_13]
    var_15 = {}
    var_16 = module_1.arguments(*var_14, **var_15)
    var_17 = var_2.func_api(var_3, var_4, var_16, var_12, has_self=var_1, cls_method=var_1)
    var_18 = ''
    var_19 = '| return |'



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "print('hello')"

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1 + 1\n2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> 1 + 1\n2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'line1\nline2'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1\n1\ntext\n>>> 2\n2'
    var_1 = '```python\n>>> 1\n1\n```\ntext\n```python\n>>> 2\n2\n```'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> start'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> start\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Intro\n>>> code\nresult\nOutro'
    var_1 = 'Intro\n```python\n>>> code\nresult\n```\nOutro'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_const_type_dict_homogeneous. Retrieved 6/12 statements.
# Partially parsed test_const_type_call_int. Retrieved 1/14 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
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
    var_0 = 1.0
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2.5
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
    assert var_15 == 'tuple[float]'

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
    assert var_15 == 'set[bool]'

import ast as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = [var_10]

def test_case_0():
    var_0 = 'int'
    assert var_0 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = module_1.const_type(var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parser_imports_import_node. Retrieved 4/9 statements.
# Partially parsed test_parser_imports_importfrom_node. Retrieved 6/11 statements.
# Partially parsed test_parser_imports_importfrom_node_no_level. Retrieved 6/11 statements.
# Partially parsed test_parser_imports_with_asname. Retrieved 4/9 statements.
# Partially parsed test_parser_imports_empty_module_name. Retrieved 5/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'os'
    var_3 = 'system'
    var_4 = var_0.alias['pkg.system']
    assert var_4 == 'os'
    var_5 = var_0.alias['pkg.os']
    assert var_5 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'sub.mod'
    var_3 = 'func'
    var_4 = 'f'
    var_5 = 1
    var_6 = var_0.alias['pkg.f']
    assert var_6 == 'pkg.sub.mod.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'other'
    var_3 = 'Class'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['pkg.Class']
    assert var_6 == 'other.Class'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'math'
    var_3 = 'm'
    var_4 = var_0.alias['pkg.m']
    assert var_4 == 'math'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = None
    var_3 = 'local'
    var_4 = 1
    var_5 = var_0.alias['pkg.local']
    assert var_5 == 'pkg.local'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 11/16 statements.
# Partially parsed test_class_api_with_bases. Retrieved 11/15 statements.
# Partially parsed test_class_api_with_enums. Retrieved 15/19 statements.
# Partially parsed test_class_api_with_deletion. Retrieved 11/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = '\nclass MyClass:\n    PUBLIC_CONST: int = 1\n    _PRIVATE_ATTR: str = "secret"\n    def method(self):\n        pass\n'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'pkg'
    var_7 = 'MyClass'
    var_8 = []
    var_9 = var_5.body
    var_10 = var_2.class_api(var_6, var_7, var_8, var_9)
    var_11 = 'PUBLIC_CONST'
    var_12 = bool('PUBLIC_CONST' in var_2.doc['pkg.MyClass'])
    assert var_12 is True
    var_13 = 'int'
    var_14 = bool('int' in var_2.doc['pkg.MyClass'])
    assert var_14 is True
    var_15 = '_PRIVATE_ATTR'
    var_16 = bool('_PRIVATE_ATTR' not in var_2.doc['pkg.MyClass'])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'class Child(BaseClass): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'pkg'
    var_7 = 'Child'
    var_8 = var_5.bases
    var_9 = []
    var_10 = var_2.class_api(var_6, var_7, var_8, var_9)
    var_11 = 'Bases'
    var_12 = bool('Bases' in var_2.doc['pkg.Child'])
    assert var_12 is True
    var_13 = '`BaseClass`'
    var_14 = bool('`BaseClass`' in var_2.doc['pkg.Child'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'from enum import Enum\nclass MyEnum(Enum):\n    VAL1 = 1\n    VAL2 = 2'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_0]
    var_6 = 'enum.Enum'
    var_7 = module_1.parse(var_6)
    var_8 = var_7.body[var_1]
    var_9 = var_8.value
    var_10 = 'pkg'
    var_11 = 'MyEnum'
    var_12 = var_5.bases
    var_13 = var_5.body
    var_14 = var_2.class_api(var_10, var_11, var_12, var_13)
    var_15 = 'Enums'
    var_16 = bool('Enums' in var_2.doc['pkg.MyEnum'])
    assert var_16 is True
    var_17 = 'VAL1'
    var_18 = bool('VAL1' in var_2.doc['pkg.MyEnum'])
    assert var_18 is True
    var_19 = 'VAL2'
    var_20 = bool('VAL2' in var_2.doc['pkg.MyEnum'])
    assert var_20 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = '\nclass MyClass:\n    TEMP_ATTR: int = 10\n    del TEMP_ATTR\n'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_1]
    var_6 = 'pkg'
    var_7 = 'MyClass'
    var_8 = []
    var_9 = var_5.body
    var_10 = var_2.class_api(var_6, var_7, var_8, var_9)
    var_11 = 'TEMP_ATTR'
    var_12 = bool('TEMP_ATTR' not in var_2.doc['pkg.MyClass'])
    assert var_12 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = True
    var_1 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.toc
    assert var_2 is True
    var_3 = var_1.link
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parser_globals_assignment_with_type_comment. Retrieved 8/28 statements.
# Partially parsed test_parser_globals_assign_constant_updates_const_dict. Retrieved 5/12 statements.
# Partially parsed test_parser_globals_assign_with_type_comment. Retrieved 8/14 statements.
# Partially parsed test_parser_globals_all_updates_imports. Retrieved 10/18 statements.
# Partially parsed test_parser_globals_ignores_non_name_targets. Retrieved 9/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONST'
    var_3 = 10
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'int'
    var_9 = []
    var_10 = {}
    var_11 = module_1.Load(*var_9, **var_10)
    var_12 = []
    var_13 = 'id'
    var_14 = 'ctx'
    var_15 = {var_13: var_8, var_14: var_11}
    var_16 = module_1.Name(*var_12, **var_15)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'X'
    var_3 = 1
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'pkg.X'
    var_9 = bool('pkg.X' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.const['pkg.X']
    assert var_10 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'Y'
    var_3 = 'hello'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'str'
    var_9 = []
    var_10 = {}
    var_11 = module_1.Load(*var_9, **var_10)
    var_12 = []
    var_13 = 'id'
    var_14 = 'ctx'
    var_15 = {var_13: var_8, var_14: var_11}
    var_16 = module_1.Name(*var_12, **var_15)
    var_17 = 'pkg.Y'
    var_18 = bool('pkg.Y' in var_0.alias)
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = '__all__'
    var_3 = 'a'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'b'
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = [var_7, var_12]
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'elts'
    var_19 = 'ctx'
    var_20 = {var_18: var_13, var_19: var_16}
    var_21 = module_1.Tuple(*var_17, **var_20)
    var_22 = 'pkg.a'
    var_23 = bool('pkg.a' in var_0.imp['pkg'])
    assert var_23 is True
    var_24 = 'pkg.b'
    var_25 = bool('pkg.b' in var_0.imp['pkg'])
    assert var_25 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'a'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = 0
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = 1
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = 'pkg.a'
    var_22 = bool('pkg.a' not in var_0.alias)
    assert var_22 is True



# Parsed testcases at query #8
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_2, var_3]
    var_5 = 'e'
    var_6 = 'f'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_1]
    var_10 = module_0.table(*var_9, items=var_8)
    assert var_10 == '| a | b |\n|:---:|:---:\n| c | d |\n| e | f |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'col1'
    var_1 = 'val1'
    var_2 = 'val2'
    var_3 = [var_1, var_2]
    var_4 = [var_0]
    var_5 = module_0.table(*var_4, items=var_3)
    assert var_5 == '| col1 |\n|:---:|\n| val1 |\n| val2 |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'long_title_name'
    var_1 = 'data'
    var_2 = [var_1]
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = module_0.table(*var_4, items=var_3)
    assert var_5 == '| long_title_name |\n|:---------------:|\n| data |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'single_str'
    var_3 = 'two'
    var_4 = 'cells'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_0, var_1]
    var_8 = module_0.table(*var_7, items=var_6)
    assert var_8 == '| a | b |\n|:---:|:---:\n| a | b |\n| single_str | b |\n| two | cells |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = []
    var_3 = [var_0, var_1]
    var_4 = module_0.table(*var_3, items=var_2)
    assert var_4 == '| a | b |\n|:---:|:---:|\n\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_const_type_call_int. Retrieved 3/5 statements.
# Partially parsed test_const_type_call_str. Retrieved 5/7 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
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
    var_0 = 1.0
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = 2.5
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
    assert var_15 == 'tuple[float]'

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
    assert var_15 == 'set[bool]'

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
    assert var_27 == 'dict[str, int]'

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
    var_12 = 'value'
    var_13 = {var_12: var_0}
    var_14 = module_0.Constant(*var_11, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = 'keys'
    var_18 = 'values'
    var_19 = {var_17: var_10, var_18: var_15}
    var_20 = module_0.Dict(*var_16, **var_19)
    var_21 = module_1.const_type(var_20)
    assert var_21 == 'dict[Any, int]'

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
    var_0 = 'builtins'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 'str'
    var_6 = []
    var_7 = 'value'
    var_8 = 'attr'
    var_9 = {var_7: var_4, var_8: var_5}
    var_10 = module_0.Attribute(*var_6, **var_9)
    var_11 = []
    var_12 = []

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_visit_Name_self_ty_replacement. Retrieved 7/10 statements.
# Partially parsed test_visit_Name_no_alias. Retrieved 8/11 statements.
# Partially parsed test_visit_Name_with_alias_replacement. Retrieved 10/13 statements.
# Partially parsed test_visit_Name_with_recursive_alias. Retrieved 11/14 statements.
# Partially parsed test_visit_Name_with_typevar_exception. Retrieved 10/13 statements.
# Partially parsed test_visit_Name_with_complex_expression_alias. Retrieved 11/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_2, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = var_3.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'Self'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'Other'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = var_3.visit_Name(var_12)
    var_14 = var_13.id
    assert var_14 == 'Other'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg.AliasClass'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyClass'
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
    assert var_16 == 'AliasClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.A'
    var_2 = 'pkg.B'
    var_3 = 'pkg.C'
    var_4 = {var_1: var_2, var_2: var_3}
    var_5 = 'T'
    var_6 = module_0.Resolver(var_0, var_4, var_5)
    var_7 = 'A'
    var_8 = []
    var_9 = {}
    var_10 = module_1.Load(*var_8, **var_9)
    var_11 = []
    var_12 = 'id'
    var_13 = 'ctx'
    var_14 = {var_12: var_7, var_13: var_10}
    var_15 = module_1.Name(*var_11, **var_14)
    var_16 = var_6.visit_Name(var_15)
    var_17 = var_16.id
    assert var_17 == 'C'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.MyVar'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyVar'
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
    assert var_16 == 'MyVar'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.A'
    var_2 = 'pkg.B | pkg.C'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'A'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_6, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = var_5.visit_Name(var_14)
    var_16 = var_15.op



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_func_api_vararg_is_none. Retrieved 8/35 statements.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = False
    assert var_7 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = True
    var_1 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.toc
    assert var_2 is True
    var_3 = var_1.link
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_globals_annassign_path. Retrieved 12/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_VAR'
    var_2 = None
    var_3 = []
    var_4 = 'id'
    var_5 = 'ctx'
    var_6 = {var_4: var_1, var_5: var_2}
    var_7 = module_1.Name(*var_3, **var_6)
    var_8 = 10
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = 'int'
    var_14 = []
    var_15 = 'id'
    var_16 = 'ctx'
    var_17 = {var_15: var_13, var_16: var_2}
    var_18 = module_1.Name(*var_14, **var_17)
    var_19 = [var_7]
    var_20 = []
    var_21 = 'targets'
    var_22 = 'value'
    var_23 = 'annotation'
    var_24 = {var_21: var_19, var_22: var_12, var_23: var_18}
    var_25 = module_1.AnnAssign(*var_20, **var_24)
    var_26 = 'pkg'
    var_27 = var_0.globals(var_26, var_25)
    var_28 = var_0.alias['pkg.MY_VAR']
    assert var_28 == '10'
    var_29 = var_0.const['pkg.MY_VAR']
    assert var_29 == 'int'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_docstring_populates_docstrings_for_existing_keys. Retrieved 6/12 statements.
# Partially parsed test_load_docstring_skips_keys_not_starting_with_root. Retrieved 6/12 statements.
# Partially parsed test_load_docstring_handles_missing_docstrings_gracefully. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that load_docstring correctly extracts and wraps docstrings from a module for keys in self.doc.'
    var_1 = module_0.Parser()
    var_2 = 'pkg'
    var_3 = 'pkg.sub'
    var_4 = '# Module `pkg`'
    var_5 = '# Submodule `pkg.sub`'
    var_6 = var_1.docstring['pkg']
    assert var_6 == 'Root docstring'
    var_7 = var_1.docstring['pkg.sub']
    assert var_7 == 'Sub docstring'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that load_docstring ignores keys in self.doc that do not start with the provided root name.'
    var_1 = module_0.Parser()
    var_2 = 'pkg'
    var_3 = 'other'
    var_4 = '# Module `pkg`'
    var_5 = '# Other'
    var_6 = var_1.docstring['pkg']
    assert var_6 == 'Root doc'
    var_7 = 'other'
    var_8 = bool('other' not in var_1.docstring)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that load_docstring does not add entries to docstring if the module attribute has no docstring.'
    var_1 = module_0.Parser()
    var_2 = 'pkg'
    var_3 = '# Module `pkg`'
    var_4 = 'pkg'
    var_5 = bool('pkg' not in var_1.docstring)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_visit_Constant_resolvable_name. Retrieved 9/12 statements.
# Partially parsed test_visit_Constant_simple_name. Retrieved 7/10 statements.
# Partially parsed test_visit_Constant_self_ty. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'import invalid syntax @@@'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = 'mod.MyClass'
    var_2 = 'mod.OtherClass'
    var_3 = 'OtherClass'
    var_4 = 'FinalClass'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_1}
    var_10 = module_1.Constant(*var_7, **var_9)
    var_11 = var_6.visit_Constant(var_10)
    var_12 = var_11.id
    assert var_12 == 'FinalClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = 'mod.Simple'
    var_2 = 'Simple'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = []
    var_6 = 'value'
    var_7 = {var_6: var_1}
    var_8 = module_1.Constant(*var_5, **var_7)
    var_9 = var_4.visit_Constant(var_8)
    var_10 = var_9.id
    assert var_10 == 'Simple'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_2}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_3.visit_Constant(var_7)
    var_9 = var_8.id
    assert var_9 == 'Self'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = True
    var_1 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.toc
    assert var_2 is True
    var_3 = var_1.link
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_visit_Name_self_ty. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_no_alias. Retrieved 8/9 statements.
# Partially parsed test_visit_Name_with_alias_simple. Retrieved 10/11 statements.
# Partially parsed test_visit_Name_with_alias_nested_expression. Retrieved 10/11 statements.
# Partially parsed test_visit_Name_with_alias_complex_expression. Retrieved 11/13 statements.
# Partially parsed test_visit_Name_typevar_protection. Retrieved 10/11 statements.
# Partially parsed test_visit_Name_no_alias_match_root_only. Retrieved 9/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_2, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = var_3.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'Self'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'Other'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = var_3.visit_Name(var_12)
    var_14 = var_13.id
    assert var_14 == 'Other'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.MyClass'
    var_2 = 'TargetClass'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'MyClass'
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
    assert var_16 == 'TargetClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.A'
    var_2 = 'pkg.B'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'A'
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
    assert var_16 == 'B'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.A'
    var_2 = 'list[int]'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'A'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_6, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = var_5.visit_Name(var_14)
    var_16 = var_15.value
    var_17 = var_15.value.id
    assert var_17 == 'list'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.TypeVar'
    var_2 = 'typing.TypeVar'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'TypeVar'
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
    assert var_16 == 'TypeVar'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.Sub'
    var_2 = 'Target'
    var_3 = {var_1: var_2}
    var_4 = 'T'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_0, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = var_5.visit_Name(var_13)
    var_15 = var_14.id
    assert var_15 == 'pkg'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_parser_class_api_with_members. Retrieved 22/43 statements.
# Partially parsed test_parser_class_api_with_bases. Retrieved 11/16 statements.
# Partially parsed test_parser_class_api_with_enums. Retrieved 17/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'ATTR_ONE'
    var_4 = None
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 10
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_10}
    var_14 = module_1.Constant(*var_11, **var_13)
    var_15 = 'int'
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_15, var_18: var_4}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = []
    var_22 = 'target'
    var_23 = 'value'
    var_24 = 'annotation'
    var_25 = {var_22: var_9, var_23: var_14, var_24: var_20}
    var_26 = module_1.AnnAssign(*var_21, **var_25)
    var_27 = 'attr_two'
    var_28 = []
    var_29 = 'id'
    var_30 = 'ctx'
    var_31 = {var_29: var_27, var_30: var_4}
    var_32 = module_1.Name(*var_28, **var_31)
    var_33 = 'str'
    var_34 = []
    var_35 = 'value'
    var_36 = {var_35: var_33}
    var_37 = module_1.Constant(*var_34, **var_36)
    var_38 = [var_32]
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_38, var_41: var_37}
    var_43 = module_1.Assign(*var_39, **var_42)
    var_44 = '_'
    var_45 = 'MyClass'
    var_46 = 'pkg'
    var_47 = 'pkg.MyClass'
    var_48 = []
    var_49 = '| `ATTR_ONE` | `int` |'
    var_50 = bool('| `ATTR_ONE` | `int` |' in var_2.doc['pkg.MyClass'])
    assert var_50 is True
    var_51 = '| `attr_two` | `str` |'
    var_52 = bool('| `attr_two` | `str` |' in var_2.doc['pkg.MyClass'])
    assert var_52 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'Base'
    var_4 = None
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 'pkg'
    var_11 = 'pkg.BaseClass'
    var_12 = [var_9]
    var_13 = []
    var_14 = var_2.class_api(var_10, var_11, var_12, var_13)
    var_15 = '| Bases |'
    var_16 = bool('| Bases |' in var_2.doc['pkg.BaseClass'])
    assert var_16 is True
    var_17 = '| `Base` |'
    var_18 = bool('| `Base` |' in var_2.doc['pkg.BaseClass'])
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'enum.Enum'
    var_4 = None
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 'VAL'
    var_11 = []
    var_12 = 'id'
    var_13 = 'ctx'
    var_14 = {var_12: var_10, var_13: var_4}
    var_15 = module_1.Name(*var_11, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_0}
    var_19 = module_1.Constant(*var_16, **var_18)
    var_20 = 'int'
    var_21 = []
    var_22 = 'id'
    var_23 = 'ctx'
    var_24 = {var_22: var_20, var_23: var_4}
    var_25 = module_1.Name(*var_21, **var_24)
    var_26 = []
    var_27 = 'target'
    var_28 = 'value'
    var_29 = 'annotation'
    var_30 = {var_27: var_15, var_28: var_19, var_29: var_25}
    var_31 = module_1.AnnAssign(*var_26, **var_30)
    var_32 = 'pkg'
    var_33 = 'pkg.MyEnum'
    var_34 = [var_9]
    var_35 = [var_31]
    var_36 = var_2.class_api(var_32, var_33, var_34, var_35)
    var_37 = '| Enums |'
    var_38 = bool('| Enums |' in var_2.doc['pkg.MyEnum'])
    assert var_38 is True
    var_39 = '| VAL |'
    var_40 = bool('| VAL |' in var_2.doc['pkg.MyEnum'])
    assert var_40 is True



# Parsed testcases at query #19
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = "print('hello')\nfoo()"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "print('hello')\nfoo()"

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1 + 1\n2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> 1 + 1\n2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> hello\nworld\n>>> next\nend'
    var_1 = '```python\n>>> hello\nworld\n```\n```python\n>>> next\nend\n```'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Intro\n>>> 1\n2\nOutro'
    var_1 = 'Intro\n```python\n>>> 1\n2\n```\nOutro'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> '
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> \n```'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_public_with_basic_public_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_exports. Retrieved 5/8 statements.
# Partially parsed test_is_public_with_not_in_all_and_private. Retrieved 6/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.module'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.module._private'
    var_4 = var_2.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.module.__init__'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.module.sub'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.module.sub'
    var_4 = 'pkg.module._hidden'
    var_5 = var_2.is_public(var_4)
    assert var_5 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_parser_class_api_with_members. Retrieved 13/61 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = '## class MyClass\n\n*Full name:* `pkg.MyClass`\n\n<a id="pkg.MyClass"></a>\n\n'
    var_5 = 'ATTR'
    var_6 = 'pkg.MyClass.ATTR'
    var_7 = 'int'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_3}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = {}
    var_14 = module_1.Load(*var_12, **var_13)
    var_15 = []
    var_16 = 'id'
    var_17 = 'ctx'
    var_18 = {var_16: var_7, var_17: var_14}
    var_19 = module_1.Name(*var_15, **var_18)
    var_20 = 'MyClass'
    var_21 = []
    var_22 = bool('pkg.MyClass.ATTR' in var_0.doc or 'ATTR' in var_0.doc)
    assert var_22 is True
    var_23 = '| ATTR |'
    var_24 = bool('| ATTR |' in var_0.doc)
    assert var_24 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_class_api_ann_assign_public_family. Retrieved 15/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = 'public_attr'
    var_4 = 'int'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = 1
    var_14 = []
    var_15 = 'value'
    var_16 = {var_15: var_13}
    var_17 = module_1.Constant(*var_14, **var_16)
    var_18 = []
    var_19 = 'classerm_name'
    var_20 = locals()
    var_21 = var_19 in var_20
    var_22 = var_0.doc[var_2]
    var_23 = len(var_22)
    var_24 = bool(var_23 > 0)
    assert var_24 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_class_api_delete_enum_member. Retrieved 35/71 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAL'
    var_2 = 1
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    var_7 = 'int'
    var_8 = []
    var_9 = {}
    var_10 = module_1.Load(*var_8, **var_9)
    var_11 = []
    var_12 = 'id'
    var_13 = 'ctx'
    var_14 = {var_12: var_7, var_13: var_10}
    var_15 = module_1.Name(*var_11, **var_14)
    var_16 = []
    var_17 = {}
    var_18 = module_1.Load(*var_16, **var_17)
    var_19 = []
    var_20 = 'id'
    var_21 = 'ctx'
    var_22 = {var_20: var_1, var_21: var_18}
    var_23 = module_1.Name(*var_19, **var_22)
    var_24 = [var_23]
    var_25 = []
    var_26 = 'targets'
    var_27 = {var_26: var_24}
    var_28 = module_1.Delete(*var_25, **var_27)
    var_29 = 'STAY'
    var_30 = 2
    var_31 = []
    var_32 = 'value'
    var_33 = {var_32: var_30}
    var_34 = module_1.Constant(*var_31, **var_33)
    var_35 = []
    var_36 = {}
    var_37 = module_1.Load(*var_35, **var_36)
    var_38 = []
    var_39 = 'id'
    var_40 = 'ctx'
    var_41 = {var_39: var_7, var_40: var_37}
    var_42 = module_1.Name(*var_38, **var_41)
    var_43 = 'enum.Enum'
    var_44 = []
    var_45 = {}
    var_46 = module_1.Load(*var_44, **var_45)
    var_47 = []
    var_48 = 'id'
    var_49 = 'ctx'
    var_50 = {var_48: var_43, var_49: var_46}
    var_51 = module_1.Name(*var_47, **var_50)
    var_52 = []
    var_53 = 'value'
    var_54 = {var_53: var_2}
    var_55 = module_1.Constant(*var_52, **var_54)
    var_56 = []
    var_57 = {}
    var_58 = module_1.Load(*var_56, **var_57)
    var_59 = []
    var_60 = 'id'
    var_61 = 'ctx'
    var_62 = {var_60: var_7, var_61: var_58}
    var_63 = module_1.Name(*var_59, **var_62)
    var_64 = []
    var_65 = {}
    var_66 = module_1.Load(*var_64, **var_65)
    var_67 = []
    var_68 = 'id'
    var_69 = 'ctx'
    var_70 = {var_68: var_1, var_69: var_66}
    var_71 = module_1.Name(*var_67, **var_70)
    var_72 = [var_71]
    var_73 = []
    var_74 = 'targets'
    var_75 = {var_74: var_72}
    var_76 = module_1.Delete(*var_73, **var_75)
    var_77 = []
    var_78 = 'value'
    var_79 = {var_78: var_30}
    var_80 = module_1.Constant(*var_77, **var_79)
    var_81 = []
    var_82 = {}
    var_83 = module_1.Load(*var_81, **var_82)
    var_84 = []
    var_85 = 'id'
    var_86 = 'ctx'
    var_87 = {var_85: var_7, var_86: var_83}
    var_88 = module_1.Name(*var_84, **var_87)
    var_89 = []
    var_90 = {}
    var_91 = module_1.Load(*var_89, **var_90)
    var_92 = []
    var_93 = 'id'
    var_94 = 'ctx'
    var_95 = {var_93: var_43, var_94: var_91}
    var_96 = module_1.Name(*var_92, **var_95)
    var_97 = [var_96]
    var_98 = 'pkg'
    var_99 = 'pkg.MyEnum'
    var_100 = 'Enums'



