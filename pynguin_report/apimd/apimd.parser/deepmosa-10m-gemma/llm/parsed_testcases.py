####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parser_imports_import_statement. Retrieved 2/11 statements.
# Partially parsed test_parser_imports_import_as_statement. Retrieved 2/9 statements.
# Partially parsed test_parser_imports_from_import_absolute. Retrieved 2/11 statements.
# Partially parsed test_parser_imports_from_import_relative. Retrieved 2/11 statements.
# Partially parsed test_parser_imports_from_import_with_asname. Retrieved 2/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = 'pkg.mod.submodule'
    var_3 = bool('pkg.mod.submodule' in var_0.alias)
    assert var_3 is True
    var_4 = var_0.alias['pkg.mod.submodule']
    assert var_4 == 'submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = 'pkg.mod.alias'
    var_3 = bool('pkg.mod.alias' in var_0.alias)
    assert var_3 is True
    var_4 = var_0.alias['pkg.mod.alias']
    assert var_4 == 'original'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = 'pkg.mod.otherpkg.func'
    var_3 = bool('pkg.mod.otherpkg.func' in var_0.alias)
    assert var_3 is True
    var_4 = var_0.alias['pkg.mod.otherpkg.func']
    assert var_4 == 'func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod.sub'
    var_2 = 'pkg.mod.sub.sibling.func'
    var_3 = bool('pkg.mod.sub.sibling.func' in var_0.alias)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = 'pkg.mod.aliased'
    var_3 = bool('pkg.mod.aliased' in var_0.alias)
    assert var_3 is True
    var_4 = var_0.alias['pkg.mod.aliased']
    assert var_4 == 'original'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_walk_body_simple_sequence. Retrieved 5/12 statements.
# Partially parsed test_walk_body_with_if_node. Retrieved 11/27 statements.
# Partially parsed test_walk_body_with_try_node. Retrieved 16/40 statements.
# Partially parsed test_walk_body_nested_structures. Retrieved 18/38 statements.


def test_case_0():
    var_0 = 'SimpleStmt'
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = 'stmt1'
    var_4 = 'stmt2'

def test_case_0():
    var_0 = 'SimpleStmt'
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = 'If'
    var_4 = 'body'
    var_5 = 'orelse'
    var_6 = [var_4, var_5]
    var_7 = 'root'
    var_8 = 'if_body'
    var_9 = 'if_else'
    var_10 = 'end'

def test_case_0():
    var_0 = 'SimpleStmt'
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = 'Handler'
    var_4 = 'body'
    var_5 = [var_4]
    var_6 = 'Try'
    var_7 = 'handlers'
    var_8 = 'orelse'
    var_9 = 'finalbody'
    var_10 = [var_4, var_7, var_8, var_9]
    var_11 = 'root'
    var_12 = 'handler_body'
    var_13 = 'try_body'
    var_14 = 'try_orelse'
    var_15 = 'try_final'

def test_case_0():
    var_0 = 'SimpleStmt'
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = 'If'
    var_4 = 'body'
    var_5 = 'orelse'
    var_6 = [var_4, var_5]
    var_7 = 'Handler'
    var_8 = [var_4]
    var_9 = 'Try'
    var_10 = 'handlers'
    var_11 = 'finalbody'
    var_12 = [var_4, var_10, var_5, var_11]
    var_13 = 'inner_if'
    var_14 = []
    var_15 = 'h_body'
    var_16 = []
    var_17 = []



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_func_api_simple_function. Retrieved 12/27 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = []
    var_15 = 'arg'
    var_16 = 'annotation'
    var_17 = {var_15: var_4, var_16: var_13}
    var_18 = module_1.arg(*var_14, **var_17)
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = []
    var_23 = 'id'
    var_24 = 'ctx'
    var_25 = {var_23: var_5, var_24: var_21}
    var_26 = module_1.Name(*var_22, **var_25)
    var_27 = 'pkg.func'
    var_28 = 'pkg.func'
    var_29 = bool('pkg.func' in var_2.doc)
    assert var_29 is True
    var_30 = 'x'
    var_31 = bool('x' in var_2.doc['pkg.func'])
    assert var_31 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_public_with_all_export_explicit_match. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_export_explicit_parent_match. Retrieved 4/8 statements.
# Partially parsed test_is_public_with_all_export_explicit_no_match. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_submodule_in_imp_via_doc_keys. Retrieved 5/8 statements.
# Partially parsed test_is_public_with_submodule_in_imp_via_const_keys. Retrieved 5/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'module.submodule.api'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'module._private_api'
    var_4 = var_2.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'module.__init__'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.exported_func'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'pkg.submodule'
    var_3 = 'pkg.submodule.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.other'
    var_4 = 'pkg.func'
    var_5 = var_2.is_public(var_4)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.sub'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.sub'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_public_standard_public_name. Retrieved 8/11 statements.
# Partially parsed test_is_public_private_name_by_convention. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_all_export. Retrieved 10/13 statements.
# Partially parsed test_is_public_with_all_export_not_in_doc. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_all_export_parent_in_all. Retrieved 9/12 statements.
# Partially parsed test_is_public_with_submodule_not_in_all. Retrieved 9/12 statements.
# Partially parsed test_is_public_with_const_in_package. Retrieved 10/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = set()
    var_5 = 'pkg.mod.func'
    var_6 = 'doc'
    var_7 = var_2.is_public(var_5)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = set()
    var_5 = 'pkg.mod._private'
    var_6 = 'doc'
    var_7 = var_2.is_public(var_5)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = 'func_a'
    var_5 = 'func_b'
    var_6 = {var_4, var_5}
    var_7 = 'pkg.mod.func_a'
    var_8 = 'doc'
    var_9 = var_2.is_public(var_7)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = 'func_a'
    var_5 = {var_4}
    var_6 = 'pkg.mod.func_a'
    var_7 = var_2.is_public(var_6)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = 'func_a'
    var_5 = {var_4}
    var_6 = 'pkg.mod.func_a.sub'
    var_7 = 'doc'
    var_8 = var_2.is_public(var_6)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = 'func_a'
    var_5 = {var_4}
    var_6 = 'pkg.mod.func_a.sub'
    var_7 = 'doc'
    var_8 = var_2.is_public(var_6)
    assert var_8 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = set()
    var_5 = 'pkg.mod.func'
    var_6 = 'doc'
    var_7 = 'pkg.mod.CONST'
    var_8 = '1'
    var_9 = var_2.is_public(var_7)
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_walk_body_evaluates_try_predicate. Retrieved 1/18 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parser_class_api_with_members. Retrieved 22/54 statements.
# Partially parsed test_parser_class_api_with_bases. Retrieved 9/19 statements.
# Partially parsed test_parser_class_api_with_enum_style. Retrieved 15/29 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, toc=var_0, level=var_1)
    var_3 = 'pkg.MyClass'
    var_4 = 'pkg'
    var_5 = 'PUBLIC_CONST'
    var_6 = 'int'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_6, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = '_private_var'
    var_16 = 10
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = 'STATUS'
    var_22 = []
    var_23 = {}
    var_24 = module_1.Load(*var_22, **var_23)
    var_25 = []
    var_26 = 'id'
    var_27 = 'ctx'
    var_28 = {var_26: var_6, var_27: var_24}
    var_29 = module_1.Name(*var_25, **var_28)
    var_30 = '\nclass MyClass:\n    PUBLIC_CONST: int = 1\n    _private_var = 10\n    STATUS: int = 1\n'
    var_31 = module_1.parse(var_30)
    var_32 = var_31.body[var_0]
    var_33 = 'pkg.MyClass.PUBLIC_CONST'
    var_34 = []
    var_35 = var_32.body
    var_36 = var_2.class_api(var_4, var_3, var_34, var_35)
    var_37 = 'pkg.MyClass'
    var_38 = bool('pkg.MyClass' in var_2.doc)
    assert var_38 is True
    var_39 = '| Members | Type |'
    var_40 = bool('| Members | Type |' in var_2.doc['pkg.MyClass'])
    assert var_40 is True
    var_41 = '| `PUBLIC_CONST` | `int` |'
    var_42 = bool('| `PUBLIC_CONST` | `int` |' in var_2.doc['pkg.MyClass'])
    assert var_42 is True
    var_43 = '_private_var'
    var_44 = bool('_private_var' not in var_2.doc['pkg.MyClass'])
    assert var_44 is True

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'pkg.Child'
    var_3 = 'pkg'
    var_4 = 'BaseClass'
    var_5 = []
    var_6 = {}
    var_7 = module_0.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_0.Name(*var_8, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = '| Bases |'
    var_16 = '| `BaseClass` |'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'pkg.MyEnum'
    var_3 = 'pkg'
    var_4 = 'enum'
    var_5 = []
    var_6 = {}
    var_7 = module_0.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_0.Name(*var_8, **var_11)
    var_13 = 'Enum'
    var_14 = []
    var_15 = {}
    var_16 = module_0.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'value'
    var_19 = 'attr'
    var_20 = 'ctx'
    var_21 = {var_18: var_12, var_19: var_13, var_20: var_16}
    var_22 = module_0.Attribute(*var_17, **var_21)
    var_23 = 'VAL'
    var_24 = 'int'
    var_25 = []
    var_26 = {}
    var_27 = module_0.Load(*var_25, **var_26)
    var_28 = []
    var_29 = 'id'
    var_30 = 'ctx'
    var_31 = {var_29: var_24, var_30: var_27}
    var_32 = module_0.Name(*var_28, **var_31)
    var_33 = [var_22]
    var_34 = '| Enums |'
    var_35 = '| `VAL` |'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_public_all_l_is_empty. Retrieved 3/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_func_api_vararg_not_none. Retrieved 4/56 statements.


def test_case_0():
    var_0 = 'root'
    var_1 = 'name'
    var_2 = None
    var_3 = False
    var_4 = 'initial'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_public_with_standard_public_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_all_filter_inclusion. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_all_filter_exclusion. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_parent_in_all_list. Retrieved 5/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod._private'
    var_4 = var_2.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod.__init__'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod.sub'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod.sub'
    var_4 = var_2.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.mod.sub'
    var_4 = var_2.is_public(var_3)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_defaults_with_special_chars. Retrieved 3/5 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' '])
    assert var_4 is True

def test_case_0():
    pass

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['`text`'])
    assert var_4 is True

def test_case_0():
    var_0 = '|'
    var_1 = '&'
    var_2 = [var_0, var_1]



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parser_globals_ann_assign_with_name_and_value. Retrieved 9/21 statements.
# Partially parsed test_parser_globals_assign_constant_upper_case. Retrieved 5/12 statements.
# Partially parsed test_parser_globals_assign_not_upper_case. Retrieved 5/12 statements.
# Partially parsed test_parser_globals_assign_with_type_comment. Retrieved 5/13 statements.
# Partially parsed test_parser_globals_all_logic_with_list_imports. Retrieved 10/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'X'
    var_3 = 'int'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 10
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'Y'
    var_3 = 20
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'pkg.Y'
    var_9 = bool('pkg.Y' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.const['pkg.Y']
    assert var_10 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'y'
    var_3 = 20
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'pkg.y'
    var_9 = bool('pkg.y' in var_0.alias)
    assert var_9 is True
    var_10 = 'pkg.y'
    var_11 = bool('pkg.y' not in var_0.const)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'Z'
    var_3 = 20
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_0.const['pkg.Z']
    assert var_8 == 'str'

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
    var_21 = module_1.List(*var_17, **var_20)
    var_22 = 'pkg.a'
    var_23 = bool('pkg.a' in var_0.imp['pkg'])
    assert var_23 is True
    var_24 = 'pkg.b'
    var_25 = bool('pkg.b' in var_0.imp['pkg'])
    assert var_25 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parser_func_api_simple_function. Retrieved 23/32 statements.
# Partially parsed test_parser_func_api_with_self_and_decorators. Retrieved 28/44 statements.
# Partially parsed test_parser_func_api_vararg_kwarg. Retrieved 17/26 statements.


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
    var_16 = 'str'
    var_17 = []
    var_18 = {}
    var_19 = module_1.Load(*var_17, **var_18)
    var_20 = []
    var_21 = 'id'
    var_22 = 'ctx'
    var_23 = {var_21: var_16, var_22: var_19}
    var_24 = module_1.Name(*var_20, **var_23)
    var_25 = []
    var_26 = [var_15]
    var_27 = None
    var_28 = []
    var_29 = []
    var_30 = 'default_val'
    var_31 = []
    var_32 = 'value'
    var_33 = {var_32: var_30}
    var_34 = module_1.Constant(*var_31, **var_33)
    var_35 = [var_34]
    var_36 = []
    var_37 = '`int`'
    var_38 = [var_37]
    var_39 = iter(var_38)
    var_40 = 'pkg'
    var_41 = 'pkg.func'
    var_42 = False
    var_43 = 'pkg.func()'
    var_44 = bool('pkg.func()' in var_0.doc['pkg.func'])
    assert var_44 is True
    var_45 = '| a |'
    var_46 = bool('| a |' in var_0.doc['pkg.func'])
    assert var_46 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'decorator'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 'self'
    var_11 = 'Self'
    var_12 = []
    var_13 = {}
    var_14 = module_1.Load(*var_12, **var_13)
    var_15 = []
    var_16 = 'id'
    var_17 = 'ctx'
    var_18 = {var_16: var_11, var_17: var_14}
    var_19 = module_1.Name(*var_15, **var_18)
    var_20 = []
    var_21 = 'arg'
    var_22 = 'annotation'
    var_23 = {var_21: var_10, var_22: var_19}
    var_24 = module_1.arg(*var_20, **var_23)
    var_25 = 'x'
    var_26 = 'int'
    var_27 = []
    var_28 = {}
    var_29 = module_1.Load(*var_27, **var_28)
    var_30 = []
    var_31 = 'id'
    var_32 = 'ctx'
    var_33 = {var_31: var_26, var_32: var_29}
    var_34 = module_1.Name(*var_30, **var_33)
    var_35 = []
    var_36 = 'arg'
    var_37 = 'annotation'
    var_38 = {var_36: var_25, var_37: var_34}
    var_39 = module_1.arg(*var_35, **var_38)
    var_40 = []
    var_41 = [var_24, var_39]
    var_42 = None
    var_43 = []
    var_44 = []
    var_45 = []
    var_46 = []
    var_47 = '`Self`'
    var_48 = '`int`'
    var_49 = [var_47, var_48]
    var_50 = iter(var_49)
    var_51 = 'pkg'
    var_52 = 'pkg.cls.method'
    var_53 = True
    var_54 = False
    var_55 = 'Decorators'
    var_56 = bool('Decorators' in var_0.doc['pkg.cls.method'])
    assert var_56 is True
    var_57 = '`@decorator`'
    var_58 = bool('`@decorator`' in var_0.doc['pkg.cls.method'])
    assert var_58 is True
    var_59 = '| self | x |'
    var_60 = bool('| self | x |' in var_0.doc['pkg.cls.method'])
    assert var_60 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'args'
    var_2 = None
    var_3 = []
    var_4 = 'arg'
    var_5 = 'annotation'
    var_6 = {var_4: var_1, var_5: var_2}
    var_7 = module_1.arg(*var_3, **var_6)
    var_8 = 'kwargs'
    var_9 = []
    var_10 = 'arg'
    var_11 = 'annotation'
    var_12 = {var_10: var_8, var_11: var_2}
    var_13 = module_1.arg(*var_9, **var_12)
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = '`ANY`'
    var_21 = [var_20]
    var_22 = iter(var_21)
    var_23 = 'pkg'
    var_24 = 'pkg.func'
    var_25 = False
    var_26 = '*args*'
    var_27 = bool('*args*' in var_0.doc['pkg.func'])
    assert var_27 is True
    var_28 = '**kwargs**'
    var_29 = bool('**kwargs**' in var_0.doc['pkg.func'])
    assert var_29 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_class_api_with_members_and_bases. Retrieved 23/41 statements.
# Partially parsed test_parser_class_api_with_enum_bases. Retrieved 22/33 statements.
# Partially parsed test_parser_class_api_with_deletion. Retrieved 16/27 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.MyClass'
    var_4 = 'pkg'
    var_5 = 'MyClass'
    var_6 = 'Base'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_6, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = 'ATTR'
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
    var_28 = 'value'
    var_29 = {var_28: var_0}
    var_30 = module_1.Constant(*var_27, **var_29)
    var_31 = 'OTHER'
    var_32 = 'string'
    var_33 = []
    var_34 = 'value'
    var_35 = {var_34: var_32}
    var_36 = module_1.Constant(*var_33, **var_35)
    var_37 = []
    var_38 = []
    var_39 = {}
    var_40 = module_1.Load(*var_38, **var_39)
    var_41 = []
    var_42 = 'id'
    var_43 = 'ctx'
    var_44 = {var_42: var_6, var_43: var_40}
    var_45 = module_1.Name(*var_41, **var_44)
    var_46 = [var_45]
    var_47 = 'Bases'
    var_48 = bool('Bases' in var_2.doc['pkg.MyClass'])
    assert var_48 is True
    var_49 = '`Base`'
    var_50 = bool('`Base`' in var_2.doc['pkg.MClass' if 'pkg.MClass' in var_2.doc else 'pkg.MyClass'])
    assert var_50 is True
    var_51 = 'Members'
    var_52 = bool('Members' in var_2.doc['pkg.MyClass'])
    assert var_52 is True
    var_53 = '`ATTR`'
    var_54 = bool('`ATTR`' in var_2.doc['pkg.MyClass'])
    assert var_54 is True
    var_55 = '`OTHER`'
    var_56 = bool('`OTHER`' in var_2.doc['pkg.MyClass'])
    assert var_56 is True
    var_57 = '`int`'
    var_58 = bool('`int`' in var_2.doc['pkg.MyClass'])
    assert var_58 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.MyEnum'
    var_4 = 'pkg'
    var_5 = 'MyEnum'
    var_6 = 'enum'
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_6, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = 'Enum'
    var_16 = []
    var_17 = {}
    var_18 = module_1.Load(*var_16, **var_17)
    var_19 = []
    var_20 = 'value'
    var_21 = 'attr'
    var_22 = 'ctx'
    var_23 = {var_20: var_14, var_21: var_15, var_22: var_18}
    var_24 = module_1.Attribute(*var_19, **var_23)
    var_25 = [var_24]
    var_26 = []
    var_27 = 'RED'
    var_28 = []
    var_29 = 'value'
    var_30 = {var_29: var_0}
    var_31 = module_1.Constant(*var_28, **var_30)
    var_32 = []
    var_33 = []
    var_34 = {}
    var_35 = module_1.Load(*var_33, **var_34)
    var_36 = []
    var_37 = 'id'
    var_38 = 'ctx'
    var_39 = {var_37: var_6, var_38: var_35}
    var_40 = module_1.Name(*var_36, **var_39)
    var_41 = []
    var_42 = {}
    var_43 = module_1.Load(*var_41, **var_42)
    var_44 = []
    var_45 = 'value'
    var_46 = 'attr'
    var_47 = 'ctx'
    var_48 = {var_45: var_40, var_46: var_15, var_47: var_43}
    var_49 = module_1.Attribute(*var_44, **var_48)
    var_50 = [var_49]
    var_51 = 'Enums'
    var_52 = bool('Enums' in var_2.doc['pkg.MyEnum'])
    assert var_52 is True
    var_53 = 'RED'
    var_54 = bool('RED' in var_2.doc['pkg.MyEnum'])
    assert var_54 is True
    var_55 = 'Members'
    var_56 = bool('Members' not in var_2.doc['pkg.MyEnum'])
    assert var_56 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg.MyClass'
    var_4 = 'pkg'
    var_5 = 'MyClass'
    var_6 = []
    var_7 = []
    var_8 = 'A'
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_0}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_8, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = 'targets'
    var_24 = {var_23: var_21}
    var_25 = module_1.Delete(*var_22, **var_24)
    var_26 = []
    var_27 = []
    var_28 = '`A`'
    var_29 = bool('`A`' not in var_2.doc['pkg.MyClass'])
    assert var_29 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.toc
    assert var_2 is True
    var_3 = var_1.link
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parser_imports_import_alias. Retrieved 4/8 statements.
# Partially parsed test_parser_imports_import_from_relative. Retrieved 6/10 statements.
# Partially parsed test_parser_imports_import_from_absolute. Retrieved 6/10 statements.
# Partially parsed test_parser_imports_import_multiple_names. Retrieved 6/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'system'
    var_3 = 'pkg'
    var_4 = var_0.alias['pkg.system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sub'
    var_2 = 'func'
    var_3 = None
    var_4 = 1
    var_5 = 'pkg'
    var_6 = var_0.alias['pkg.sub.func']
    assert var_6 == 'sub.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'math'
    var_2 = 'sqrt'
    var_3 = 's'
    var_4 = 0
    var_5 = 'pkg'
    var_6 = var_0.alias['pkg.s']
    assert var_6 == 'math.sqrt'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = None
    var_3 = 'json'
    var_4 = 'j'
    var_5 = 'pkg'
    var_6 = var_0.alias['pkg.sys']
    assert var_6 == 'sys'
    var_7 = var_0.alias['pkg.j']
    assert var_7 == 'json'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 10/12 statements.
# Partially parsed test_visit_Attribute_keeps_non_typing_attribute. Retrieved 10/12 statements.
# Partially parsed test_visit_Attribute_keeps_nested_attribute. Retrieved 14/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'List'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = 'attr'
    var_19 = 'ctx'
    var_20 = {var_17: var_11, var_18: var_12, var_19: var_15}
    var_21 = module_1.Attribute(*var_16, **var_20)
    var_22 = var_2.visit_Attribute(var_21)
    var_23 = var_22.id
    assert var_23 == 'List'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'List'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = 'attr'
    var_19 = 'ctx'
    var_20 = {var_17: var_11, var_18: var_12, var_19: var_15}
    var_21 = module_1.Attribute(*var_16, **var_20)
    var_22 = var_2.visit_Attribute(var_21)
    var_23 = var_22.value.id
    assert var_23 == 'other'
    var_24 = var_22.attr
    assert var_24 == 'List'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'sub'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = 'attr'
    var_19 = 'ctx'
    var_20 = {var_17: var_11, var_18: var_12, var_19: var_15}
    var_21 = module_1.Attribute(*var_16, **var_20)
    var_22 = 'List'
    var_23 = []
    var_24 = {}
    var_25 = module_1.Load(*var_23, **var_24)
    var_26 = []
    var_27 = 'value'
    var_28 = 'attr'
    var_29 = 'ctx'
    var_30 = {var_27: var_21, var_28: var_22, var_29: var_25}
    var_31 = module_1.Attribute(*var_26, **var_30)
    var_32 = var_2.visit_Attribute(var_31)
    var_33 = var_32.value
    var_34 = var_32.value.value.id
    assert var_34 == 'typing'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_class_api_with_bases_evaluates_true. Retrieved 13/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = 'my_class'
    var_3 = 'Initial Doc'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_1, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 'table'
    var_15 = 'TableContent'
    var_16 = 'root'
    var_17 = 'my_class'
    var_18 = var_0.class_api(var_16, var_17, var_12, var_13)
    var_19 = 'TableContent'
    var_20 = bool('TableContent' in var_0.doc['my_class'])
    assert var_20 is True



# Parsed testcases at query #8
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._e_type(*var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._e_type(*var_1)
    assert var_2 == ''

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
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = module_1._e_type(*var_5)
    assert var_6 == '[str]'

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
    var_8 = [var_3]
    var_9 = [var_7]
    var_10 = [var_8, var_9]
    var_11 = module_1._e_type(*var_10)
    assert var_11 == '[int, int]'

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
    var_8 = [var_3]
    var_9 = [var_7]
    var_10 = [var_8, var_9]
    var_11 = module_1._e_type(*var_10)
    assert var_11 == '[int, str]'

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

import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_5]
    var_7 = module_1._e_type(*var_6)
    assert var_7 == ''



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._e_type(*var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._e_type(*var_1)
    assert var_2 == ''

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
    var_0 = 'hello'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = module_1._e_type(*var_5)
    assert var_6 == '[str]'

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
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Constant(*var_9, **var_10)
    var_12 = [var_3]
    var_13 = [var_7]
    var_14 = [var_11]
    var_15 = [var_12, var_13, var_14]
    var_16 = module_1._e_type(*var_15)
    assert var_16 == '[int, int, str]'

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

import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1.5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = True
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3]
    var_9 = [var_7]
    var_10 = [var_8, var_9]
    var_11 = module_1._e_type(*var_10)
    assert var_11 == '[float, bool]'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_attr_single_level_success. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_level_success. Retrieved 2/7 statements.
# Partially parsed test_attr_missing_attribute_returns_none. Retrieved 2/5 statements.
# Partially parsed test_attr_broken_chain_returns_none. Retrieved 3/7 statements.
# Partially parsed test_attr_empty_string_returns_obj. Retrieved 2/5 statements.
# Partially parsed test_attr_deeply_nested_success. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'

def test_case_0():
    var_0 = 2
    var_1 = 'c.b'

def test_case_0():
    var_0 = 1
    var_1 = 'b'

def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'c.b'

def test_case_0():
    var_0 = 1
    var_1 = ''

def test_case_0():
    var_0 = 'found'
    var_1 = 'l2.l3.val'

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'any.path'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #11
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
    var_0 = '>>> x = 5\n>>> x\n5'
    var_1 = '```python\n>>> x = 5\n>>> x\n5\n```'
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
    var_0 = '>>> 1'
    var_1 = '```python\n>>> 1\n```'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1\n2\nMiddle\n>>> 3\n4'
    var_1 = '```python\n>>> 1\n2\n```\nMiddle\n```python\n>>> 3\n4\n```'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'my_package.module'
    var_4 = 'import os\nVERSION = 1'
    var_5 = var_2.parse(var_3, var_4)
    var_6 = 'my_package.module'
    var_7 = bool('my_package.module' in var_2.doc)
    assert var_7 is True
    var_8 = 'my_package.module'
    var_9 = bool('my_package.module' in var_2.level)
    assert var_9 is True
    var_10 = 'my_package.module'
    var_11 = bool('my_package.module' in var_2.root)
    assert var_11 is True
    var_12 = 'my_package.module'
    var_13 = bool('my_package.module' in var_2.imp)
    assert var_13 is True
    var_14 = 'my_package.os'
    var_15 = bool('my_package.os' in var_2.alias)
    assert var_15 is True
    var_16 = 'my_package.VERSION'
    var_17 = bool('my_package.VERSION' in var_2.alias)
    assert var_17 is True



