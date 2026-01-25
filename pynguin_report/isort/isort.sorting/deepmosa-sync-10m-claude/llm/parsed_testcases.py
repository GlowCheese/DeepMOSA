####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '..parent.module'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_keyword. Retrieved 2/6 statements.
# Partially parsed test_section_key_removes_import_keyword. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_complex_line. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/5 statements.
# Partially parsed test_section_key_multiple_relative_dots. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from package.subpackage import function'
    var_2 = 'B'

def test_case_0():
    var_0 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_line_20_predicate_evaluates_to_false. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class_uppercase_first. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_import_with_space. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '. module'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None
    var_4 = 'C'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_section_key_line_29_predicate_true. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'module import names'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 6/17 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = '^(\\.+)\\s*(.*)'
    var_3 = module_0.match(var_2, var_0)
    assert var_3 is None
    var_4 = False
    var_5 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 5/19 statements.


import re as module_0

def test_case_0():
    var_0 = '... utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_module_key_predicate_line_11_true. Retrieved 5/17 statements.


import re as module_0

def test_case_0():
    var_0 = '... some_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_combined_length_sort_and_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_no_sub_imports. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = '....package.module'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = [var_1]
    var_3 = 'A'
    var_4 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/12 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_remove_from_prefix. Retrieved 3/7 statements.
# Partially parsed test_section_key_remove_import_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_from_dot. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_with_different_settings. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 3/7 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 5/9 statements.
# Partially parsed test_section_key_complex_import_statement. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'Bfrom'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from package.module import ClassA, ClassB'
    var_3 = 'B'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_false. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'THIRDPARTY'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_module_key_predicate_line_11_true. Retrieved 5/19 statements.


import re as module_0

def test_case_0():
    var_0 = '...some_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative.module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = 'thirdparty'
    var_3 = ':'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_from_import_line. Retrieved 1/4 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'import django'
    var_3 = 'A'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from ..package import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ..package import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import flask'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import MyModule'
    var_2 = 'B'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_all_parameters. Retrieved 5/10 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_dot_only. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = True
    var_3 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = '6:module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..submodule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'special'
    var_3 = [var_2]
    var_4 = 'future'
    var_5 = 'special'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = False
    var_1 = '.'



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_sort_relative_in_force_sorted_sections_predicate_false.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_straight_import_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_section_name_length_sort. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'MyModule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'mymodule'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = ':'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'mymodule'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'MYCONST'
    var_2 = 'A'

def test_case_0():
    var_0 = False
    var_1 = '....module_name'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_class_prefix. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = True
    var_1 = '...'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 14/40 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'stdlib'
    var_2 = False
    var_3 = '^(\\.+)\\s*(.*)'
    var_4 = module_0.match(var_3, var_0)
    var_5 = ' '
    var_6 = '_'
    var_7 = var_5 if var_3 else var_6
    var_8 = ''
    var_9 = False
    var_10 = str(var_0)
    var_11 = str(var_0)
    var_12 = var_6 and var_2
    var_13 = str(var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 2/21 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/10 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_with_sort_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'
    var_3 = 'from'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ..package import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from . import module'
    var_1 = 'B'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_true. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_section_key_predicate_line_4. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_true. Retrieved 3/45 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'A'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_section_key_predicate_line_23_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_conditions. Retrieved 6/11 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import test'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import test'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import test'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'
    var_2 = 'path'

def test_case_0():
    var_0 = 'from . import test'
    var_1 = 'B'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = str(var_2)
    var_4 = len(var_0)
    var_5 = str(var_4)
    var_6 = ':'
    var_7 = var_5 + var_6
    var_8 = var_7 + var_0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = 'C'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.module'
    var_1 = '_'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_empty_line. Retrieved 2/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from ..package import module'
    var_1 = 'B'

def test_case_0():
    var_0 = ''
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'
    var_2 = 'B'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 1/13 statements.
# Partially parsed test_section_key_predicate_line_43_with_length_sort_true. Retrieved 1/13 statements.
# Partially parsed test_section_key_predicate_line_43_with_force_to_top. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'numpy'
    var_1 = 'B'
    var_2 = 'numpy'

def test_case_0():
    var_0 = 'numpy'
    var_1 = 'B'
    var_2 = '5'
    var_3 = 'numpy'

def test_case_0():
    var_0 = 'os'
    var_1 = 'os'
    var_2 = 'A'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'os import path'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'A'
    var_3 = []
    var_4 = 'othermodule'
    var_5 = 'B'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'A'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_remove_from_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_remove_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_returns_string. Retrieved 1/5 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_not_in_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_case_sensitive_with_honor_case. Retrieved 3/7 statements.
# Partially parsed test_section_key_empty_line. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'A'
    var_2 = 'B'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import sys'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'

def test_case_0():
    var_0 = ''
    var_1 = 'B'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 5/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_normal. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false_order_by_type_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_true_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/7 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'os'
    var_4 = len(var_3)
    var_5 = str(var_4)

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from OS import Path'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = False
    var_1 = 'from OS import Path'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'

def test_case_0():
    var_0 = 'from sys import path'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_section_key_predicate_line_43_evaluates_to_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = ''



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_relative_import_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_with_split. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_no_split. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_complex_import_statement. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'from'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = -2

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = '.'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = '.'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = 'from package.module import function'
    var_1 = 'B'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_with_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import os, sys'
    var_1 = 'B'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_section_key_length_sort_false. Retrieved 8/14 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1
    var_4 = result.split(var_2)[var_3]
    var_5 = len(var_4)
    var_6 = var_5 == var_0
    var_7 = result.split(var_2)[var_3][var_0]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_section_key_line_12_predicate. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/7 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/9 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_section_key_section_always_starts_with_letter. Retrieved 1/4 statements.
# Partially parsed test_section_key_honor_case_split_module. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from '

def test_case_0():
    var_0 = 'import os'
    var_1 = 1
    var_2 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1
    var_3 = 3

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = '  import os  '
    var_1 = 'B'

def test_case_0():
    var_0 = 'import anything'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_section_key_predicate_at_line_43_evaluates_to_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test_line'
    var_2 = len(var_1)
    var_3 = ''



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'os'
    var_2 = len(var_1)
    var_3 = ''



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.module'
    var_1 = '_'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_order_by_type_different. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_imports_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_split_module. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, getcwd'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from sys import argv'
    var_1 = 'B'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_module_key_simple_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/7 statements.
# Partially parsed test_module_key_sub_imports_false_order_by_type. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = False
    var_4 = 'A'
    var_5 = 'B'



# Parsed testcases at query #2
#--------------------------




import isort.sorting as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.naturally(var_3)
    var_5 = bool(var_4 == ['apple', 'banana', 'cherry'])
    assert var_5 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'file10.txt'
    var_1 = 'file2.txt'
    var_2 = 'file1.txt'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.naturally(var_3)
    var_5 = bool(var_4 == ['file1.txt', 'file2.txt', 'file10.txt'])
    assert var_5 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'item100'
    var_1 = 'item20'
    var_2 = 'item3'
    var_3 = 'item1'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.naturally(var_4)
    var_6 = bool(var_5 == ['item1', 'item3', 'item20', 'item100'])
    assert var_6 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'file10.txt'
    var_1 = 'file2.txt'
    var_2 = 'file1.txt'
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.naturally(var_3, reverse=var_4)
    var_6 = bool(var_5 == ['file10.txt', 'file2.txt', 'file1.txt'])
    assert var_6 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'apple10'
    var_1 = 'apple2'
    var_2 = 'apple1'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'apple'
    var_5 = ''
    var_6 = lambda x: x.replace(var_4, var_5)
    var_7 = module_0.naturally(var_3, var_6)
    var_8 = bool(var_7 == ['apple1', 'apple2', 'apple10'])
    assert var_8 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.naturally(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'single'
    var_1 = [var_0]
    var_2 = module_0.naturally(var_1)
    var_3 = bool(var_2 == ['single'])
    assert var_3 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'a1b2c3'
    var_1 = 'a1b10c3'
    var_2 = 'a1b2c10'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.naturally(var_3)
    var_5 = bool(var_4 == ['a1b2c3', 'a1b2c10', 'a1b10c3'])
    assert var_5 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = '100'
    var_1 = '20'
    var_2 = '3'
    var_3 = '1'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.naturally(var_4)
    var_6 = bool(var_5 == ['1', '3', '20', '100'])
    assert var_6 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'v1.10.0'
    var_1 = 'v1.2.0'
    var_2 = 'v1.10.1'
    var_3 = 'v1.2.1'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.naturally(var_4)
    var_6 = bool(var_5 == ['v1.2.0', 'v1.2.1', 'v1.10.0', 'v1.10.1'])
    assert var_6 is True

import isort.sorting as module_0

def test_case_0():
    var_0 = 'test10'
    var_1 = 'test2'
    var_2 = 'test1'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'test'
    var_5 = ''
    var_6 = lambda x: x.replace(var_4, var_5)
    var_7 = True
    var_8 = module_0.naturally(var_3, var_6, var_7)
    var_9 = bool(var_8 == ['test10', 'test2', 'test1'])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight_false. Retrieved 3/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_flags. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = False
    var_3 = 'os'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = '.. module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'A'
    var_5 = '2:os'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_module_key_predicate_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 2/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_reverse_relative_not_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_true. Retrieved 2/8 statements.
# Partially parsed test_section_key_length_sort_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false_order_by_type_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_split_module. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from ... import something'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_prefix. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...package.module'
    var_2 = 'package_module'

def test_case_0():
    var_0 = True
    var_1 = '...package.module'
    var_2 = 'package module'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'module'
    var_3 = '6:module'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'
    var_2 = 'C'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'A'
    var_5 = ':'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 1/6 statements.
# Partially parsed test_module_key_single_letter_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'
    var_2 = 'C'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = '..package.module'

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_section_key_predicate_line_4. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_force_to_top_predicate_true. Retrieved 31/60 statements.


import re as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'mymodule'
    var_3 = False
    var_4 = False
    var_5 = None
    var_6 = False
    var_7 = 're'
    var_8 = __import__(var_7)
    var_9 = '^(\\.+)\\s*(.*)'
    var_10 = module_0.match(var_9, var_2)
    var_11 = ' '
    var_12 = '_'
    var_13 = var_11 if var_0 else var_12
    var_14 = ''
    var_15 = str(var_2)
    var_16 = str(var_2)
    var_17 = 'A'
    var_18 = 'B'
    var_19 = 'C'
    var_20 = 'A'
    var_21 = 'B'
    var_22 = 'C'
    var_23 = str(var_5)
    var_24 = len(var_16)
    var_25 = str(var_24)
    var_26 = ':'
    var_27 = var_25 + var_26
    var_28 = var_27 + var_16
    var_29 = 'A'
    var_30 = 'B'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '...package.module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_module_key_predicate_line_11_evaluates_to_true. Retrieved 5/17 statements.


import re as module_0

def test_case_0():
    var_0 = '...some_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'B'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from .'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive_config. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class_capital. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_no_length_sort. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = '_'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = ' '

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'PI'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = '6:module'

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '.. module_name'

def test_case_0():
    var_0 = False
    var_1 = '...package.module'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_module_key_predicate_line_20_true. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_line_20_predicate_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/4 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_import_multiple_dots. Retrieved 1/4 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_combined_length_sort_and_force_to_top. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = 'C'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = '6:module'

def test_case_0():
    var_0 = '....package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = 'sys'
    var_2 = [var_1]
    var_3 = 'A'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_without_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_disabled. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'UPPERCASE'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'Capitalized'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'lowercase'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_section_key_force_to_top_predicate. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = 'A'
    var_4 = "Section should be 'A' when line.split(' ')[0] is in force_to_top"



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = False
    var_4 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_section_key_line_20_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.module'
    var_1 = '_'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 4/8 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 4/8 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 4/8 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_force_to_top_not_in_list. Retrieved 4/8 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = '_'
    var_3 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = ' '
    var_3 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'BA'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'BB'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'BC'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = 'BA'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'BB'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = 'BC'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..utils'
    var_2 = '_'
    var_3 = 'utils'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 32/56 statements.


import re as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = False
    var_3 = False
    var_4 = None
    var_5 = False
    var_6 = 're'
    var_7 = __import__(var_6)
    var_8 = '^(\\.+)\\s*(.*)'
    var_9 = module_0.match(var_8, var_1)
    var_10 = ' '
    var_11 = '_'
    var_12 = var_10 if var_0 else var_11
    var_13 = ''
    var_14 = str(var_1)
    var_15 = str(var_1)
    var_16 = 'A'
    var_17 = 'B'
    var_18 = 'C'
    var_19 = 'A'
    var_20 = 'B'
    var_21 = 'C'
    var_22 = str(var_4)
    var_23 = len(var_15)
    var_24 = str(var_23)
    var_25 = ':'
    var_26 = var_24 + var_25
    var_27 = var_26 + var_15
    var_28 = len(var_15)
    var_29 = str(var_28)
    var_30 = var_29 + var_25
    var_31 = var_30 + var_15



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_import_reverse. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 4/9 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_spaces_in_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import_multiple_dots. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'from __future__ import annotations'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from .. import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import Os'
    var_3 = 'B'
    var_4 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'B'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_lexicographical_predicate_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_23_evaluates_to_false. Retrieved 5/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 0
    var_3 = ' '
    var_4 = var_1.split(var_3)[var_2]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'from . import something'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_false. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/8 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 4/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_returns_string. Retrieved 1/5 statements.
# Partially parsed test_section_key_complex_import. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'import django'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'A'
    var_2 = 'B'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'AB0123456789'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import OS'
    var_2 = 'OS'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'A'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'from django.conf import settings'
    var_5 = 'A'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_section_key_predicate_line_20_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_class_uppercase_start. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable_lowercase. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 4/7 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_single_char_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...utils'
    var_2 = 'utils'

def test_case_0():
    var_0 = True
    var_1 = '...utils'
    var_2 = 'utils'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'BA'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'BB'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'BC'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'BA'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'BB'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = 'BC'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = 'future'
    var_4 = '6:module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..module.submodule'

def test_case_0():
    var_0 = True
    var_1 = '.module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'A'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = f'B{len(var_0)}{var_0}'
    var_2 = 'B'
    var_3 = len(var_0)
    var_4 = str(var_3)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_complex_import_line. Retrieved 2/6 statements.
# Partially parsed test_section_key_single_word_module. Retrieved 2/8 statements.
# Partially parsed test_section_key_length_sort_includes_length. Retrieved 10/16 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from '

def test_case_0():
    var_0 = 'import os'
    var_1 = 1
    var_2 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'

def test_case_0():
    var_0 = 'from django.conf import settings'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 0

def test_case_0():
    var_0 = True
    var_1 = 'import a'
    var_2 = 'import verylongname'
    var_3 = 0
    var_4 = 'a'
    var_5 = var_4.split(var_4)[var_3]
    var_6 = int(var_5)
    var_7 = 'v'
    var_8 = var_7.split(var_7)[var_3]
    var_9 = int(var_8)
    var_10 = bool(var_6 < var_9)
    assert var_10 is True

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Package import Name'
    var_3 = 'package'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_relative_imports. Retrieved 2/5 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 6/12 statements.
# Partially parsed test_section_key_lexicographical_with_from. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_split_module. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path, Other'
    var_3 = 'os'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 25/47 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'test_module'
    var_3 = False
    var_4 = False
    var_5 = None
    var_6 = False
    var_7 = None
    var_8 = ' '
    var_9 = '_'
    var_10 = ''
    var_11 = str(var_2)
    var_12 = str(var_2)
    var_13 = 'A'
    var_14 = 'B'
    var_15 = 'C'
    var_16 = 'A'
    var_17 = 'B'
    var_18 = 'C'
    var_19 = str(var_5)
    var_20 = len(var_12)
    var_21 = str(var_20)
    var_22 = ':'
    var_23 = var_21 + var_22
    var_24 = var_23 + var_12



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_sorting. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 4/9 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/11 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_deep_relative_import. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os.path'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'
    var_3 = -1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from OS import Path'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from module import Name'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = 'collections'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'from . import module'

def test_case_0():
    var_0 = 'from ... import module'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = str(var_1)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_length_sort_true_prepends_length_to_module_name. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = '10:test_module'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 3/8 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = '_'
    var_3 = True
    var_4 = ' '

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = '6:module'

def test_case_0():
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'
    var_2 = '_'
    var_3 = 'package'

def test_case_0():
    var_0 = True
    var_1 = '__future__'
    var_2 = [var_1]
    var_3 = 'CONST'
    var_4 = [var_3]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'os import path'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_case_sensitive_predicate_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_no_force_to_top_match. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_with_from. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B9'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = 'sys'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import sys'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from django import forms'
    var_2 = 'B'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_section_key_force_to_top_predicate. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'os'
    var_4 = 0
    var_5 = ' '
    var_6 = var_3.split(var_5)[var_4]



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_force_to_top_predicate_evaluates_to_false. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'B'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_section_key_predicate_line_12. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'

def test_case_0():
    var_0 = True
    var_1 = '. module'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_order_by_type_uppercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_order_by_type_class_first_letter_upper. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_import_with_spaces. Retrieved 1/5 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_combined_flags. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '. module'

def test_case_0():
    var_0 = '..module_name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = ':'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_section_key_length_sort_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'B'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_lexicographical_predicate_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = None
    var_4 = ''
    var_5 = str(var_1)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'module_name in config.force_to_top'
    var_6 = 'B'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 6/9 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 5/12 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 5/9 statements.
# Partially parsed test_section_key_import_with_spaces. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'
    var_3 = 'from'
    var_4 = result.split(var_3)[var_0]
    var_5 = var_2 not in var_4

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import something'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1
    var_4 = 3

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = 'B'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_by_first_letter. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_false_order_by_type. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = '..pkg.module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_section_key_line_12_predicate_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'from os import path'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 5/8 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'import django'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'os'
    var_4 = len(var_3)
    var_5 = str(var_4)

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import Django'
    var_2 = 'django'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Django'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from collections import OrderedDict'
    var_1 = 'collections'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 24/54 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 're'
    var_6 = __import__(var_5)
    var_7 = '^(\\.+)\\s*(.*)'
    var_8 = module_0.match(var_7, var_0)
    assert var_8 is None
    var_9 = ''
    var_10 = str(var_0)
    var_11 = str(var_0)
    var_12 = 'A'
    var_13 = 'B'
    var_14 = 'C'
    var_15 = 'A'
    var_16 = 'B'
    var_17 = 'C'
    var_18 = str(var_3)
    var_19 = len(var_11)
    var_20 = str(var_19)
    var_21 = ':'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_11



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 7/18 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = False
    var_3 = 'from . import something'
    var_4 = 'import something'
    var_5 = 'from . import something'
    var_6 = 'from . import something'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_section_key_predicate_line_4. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = 'from .'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_dots. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_with_import. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_names_lowercase. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_with_length_sort_and_force_to_top. Retrieved 5/11 statements.
# Partially parsed test_section_key_lexicographical_with_relative. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from '

def test_case_0():
    var_0 = 'import os'
    var_1 = 1
    var_2 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'path'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = [var_1]
    var_3 = 'import os'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from .module import name'
    var_2 = 'B'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 30/54 statements.


import re as module_0

def test_case_0():
    var_0 = '^from |^import '
    var_1 = module_0.compile(var_0)
    var_2 = ' import '
    var_3 = module_0.compile(var_2)
    var_4 = 'os'
    var_5 = 'B'
    var_6 = '^from (\\.+)\\s*(.*)'
    var_7 = module_0.match(var_6, var_4)
    var_8 = ' '
    var_9 = f'from {var_8.join(var_2)}'
    var_10 = 0
    var_11 = ' import '
    var_12 = 1
    var_13 = var_9.split(var_11, var_12)[var_10]
    var_14 = ''
    var_15 = '.'
    var_16 = '^from '
    var_17 = ''
    var_18 = module_0.sub(var_16, var_17, var_13)
    var_19 = '^import '
    var_20 = module_0.sub(var_19, var_17, var_18)
    var_21 = ' '
    var_22 = '_'
    var_23 = var_21 if var_16 else var_22
    var_24 = '^(\\.+)'
    var_25 = f'\\1{var_23}'
    var_26 = module_0.sub(var_24, var_25, var_20)
    var_27 = 0
    var_28 = ' '
    var_29 = var_26.split(var_28)[var_27]



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_section_key_line_23_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_predicate_at_line_41_evaluates_to_false. Retrieved 11/31 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = str(var_3)
    var_6 = len(var_0)
    var_7 = str(var_6)
    var_8 = ':'
    var_9 = var_7 + var_8
    var_10 = var_9 + var_0



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.test'
    var_1 = '_'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #86
#--------------------------

# Failed to parse test_lexicographical_predicate_evaluates_to_true.




# Parsed testcases at query #87
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_with_from_dot. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/7 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 2/5 statements.
# Partially parsed test_section_key_complex_import_line. Retrieved 5/9 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 6/12 statements.
# Partially parsed test_section_key_non_force_to_top_module. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'

def test_case_0():
    var_0 = True
    var_1 = 'django'
    var_2 = [var_1]
    var_3 = 'from django.conf import settings'
    var_4 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import sys'
    var_3 = 'B'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false_order_by_type_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_true_order_by_type_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_with_sort_relative_in_force_sorted_sections_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true_reverse_relative_true. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true_reverse_relative_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 3/7 statements.
# Partially parsed test_section_key_complex_import_line. Retrieved 4/10 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'

def test_case_0():
    var_0 = 'import os'
    var_1 = 1
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from Os import Path'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from package.submodule import ClassA, function_b'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/10 statements.


def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = 'B2'
    var_3 = False



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 2/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort_with_longer_line. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from .. import os'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'import os, sys, datetime'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_case_sensitive_predicate_evaluates_to_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 5/12 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import os'
    var_3 = 'import os'
    var_4 = 'import os'



