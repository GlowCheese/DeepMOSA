####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_order_by_type_mismatch. Retrieved 3/7 statements.
# Partially parsed test_section_key_not_order_by_type. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_combined_options. Retrieved 6/10 statements.
# Partially parsed test_section_key_empty_line_handling. Retrieved 1/5 statements.


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
    var_1 = 1
    var_2 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import os'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'import'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_combined_options. Retrieved 3/8 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = '....module_name'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative.module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '...package.submodule'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = None
    var_3 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '.module'

def test_case_0():
    var_0 = '....package.module'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_order_by_type. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import_with_sort_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_import_without_sort_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 3/7 statements.
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
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = False
    var_3 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_constants_in_config. Retrieved 3/6 statements.
# Partially parsed test_module_key_classes_in_config. Retrieved 3/6 statements.
# Partially parsed test_module_key_variables_in_config. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_module. Retrieved 2/6 statements.
# Partially parsed test_module_key_capitalized_module. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'var'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'

def test_case_0():
    var_0 = True
    var_1 = 'Module'

def test_case_0():
    var_0 = False
    var_1 = '....module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_section_key_line_20_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_module_key_predicate_line_11_true. Retrieved 5/20 statements.


import re as module_0

def test_case_0():
    var_0 = '...package.submodule'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 4/10 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 3/6 statements.
# Partially parsed test_section_key_combined_conditions. Retrieved 6/12 statements.


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

def test_case_0():
    var_0 = 'from os import path'

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
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

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
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import sys'
    var_5 = 'A'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 26/50 statements.


import re as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = False
    var_3 = False
    var_4 = None
    var_5 = False
    var_6 = '^(\\.+)\\s*(.*)'
    var_7 = module_0.match(var_6, var_1)
    var_8 = ' '
    var_9 = '_'
    var_10 = var_8 if var_0 else var_9
    var_11 = ''
    var_12 = str(var_1)
    var_13 = str(var_1)
    var_14 = 'A'
    var_15 = 'B'
    var_16 = 'C'
    var_17 = 'A'
    var_18 = 'B'
    var_19 = 'C'
    var_20 = str(var_4)
    var_21 = len(var_13)
    var_22 = str(var_21)
    var_23 = ':'
    var_24 = var_22 + var_23
    var_25 = var_24 + var_13



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_module_key_predicate_line_20_false. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 5/12 statements.


import re as module_0

def test_case_0():
    var_0 = '... some_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_imports. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_force_to_top_not_matched. Retrieved 4/8 statements.
# Partially parsed test_section_key_mixed_case_with_honor_case. Retrieved 4/8 statements.
# Partially parsed test_section_key_relative_with_reverse_relative_true. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_import_statement. Retrieved 1/5 statements.


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
    var_1 = 'AB0123456789'

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

def test_case_0():
    var_0 = True
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'AB0123456789'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

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
    var_2 = 'from os import Path'
    var_3 = 'B'

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
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import numpy as np'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 5/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1
    var_4 = var_4.split(var_1)[var_0]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_and_order_by_type_different. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_false_case_sensitive_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_lexicographical_with_relative. Retrieved 2/6 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

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
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Module'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import flask'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from ..module import name'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_case_sensitive_predicate_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'TestModule'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_section_key_length_sort_false. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = 1



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative_not_force_sorted. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_no_reverse. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_with_different_settings. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_split_module. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 5/9 statements.
# Partially parsed test_section_key_empty_line. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'import django'
    var_3 = 'A'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

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
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import flask'
    var_4 = 'A'

def test_case_0():
    var_0 = ''
    var_1 = 'B'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = 'B11:'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_false. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_space. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_underscore. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false_order_by_type_false. Retrieved 5/12 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 1/4 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 5/9 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

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
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from Os import Path'
    var_3 = 'os'
    var_4 = 'path'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'from . import module'

def test_case_0():
    var_0 = 'from .. import module'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import flask'
    var_4 = 'A'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = False
    var_3 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 6/9 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1
    var_3 = 'B'
    var_4 = result.split(var_3)[var_2]
    var_5 = len(var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 36/71 statements.


import re as module_0

def test_case_0():
    var_0 = '.module'
    var_1 = '^from |^import '
    var_2 = module_0.compile(var_1)
    var_3 = ' import '
    var_4 = module_0.compile(var_3)
    var_5 = 'B'
    var_6 = '^from (\\.+)\\s*(.*)'
    var_7 = module_0.match(var_6, var_0)
    var_8 = ' '
    var_9 = f'from {var_0.join(var_3)}'
    var_10 = 0
    var_11 = ' import '
    var_12 = 1
    var_13 = line.split(var_11, var_12)[var_10]
    var_14 = ''
    var_15 = '.'
    var_16 = module_0.sub(var_15, var_13)
    var_17 = module_0.sub(var_14, var_16)
    var_18 = '^from '
    var_19 = ''
    var_20 = module_0.sub(var_18, var_19, var_17)
    var_21 = '^import '
    var_22 = module_0.sub(var_21, var_19, var_20)
    var_23 = ' '
    var_24 = '_'
    var_25 = var_23 if var_18 else var_24
    var_26 = '^(\\.+)'
    var_27 = f'\\1{var_25}'
    var_28 = module_0.sub(var_26, var_27, var_22)
    var_29 = 'A'
    var_30 = ' import '
    var_31 = 1
    var_32 = module_0.split(var_30, var_31)
    var_33 = len(var_28)
    var_34 = ''
    var_35 = f'{var_29}{(var_33 if var_24 else var_34)}{var_28}'
    assert var_35 == 'B._module'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_module_key_simple_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

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
    var_1 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'variable'

def test_case_0():
    var_0 = False
    var_1 = '.. module'

def test_case_0():
    var_0 = True
    var_1 = '....deep.module'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/10 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_import_reverse. Retrieved 4/8 statements.
# Partially parsed test_section_key_relative_import_force_sorted. Retrieved 4/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_force_sorted_sections. Retrieved 4/9 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 2/6 statements.
# Partially parsed test_section_key_empty_config. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

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
    var_0 = False
    var_1 = 'from Os import Path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from Os import Path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from package import module1, module2'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_section_key_length_sort_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.module'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'B'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_space. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_underscore. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_complex_import. Retrieved 4/9 statements.


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
    var_2 = 'B2'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from package import module'
    var_3 = 'B'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 9/28 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 're'
    var_2 = __import__(var_1)
    var_3 = '^(\\.+)\\s*(.*)'
    var_4 = module_0.match(var_3, var_0)
    assert var_4 is None
    var_5 = ''
    var_6 = False
    var_7 = None
    var_8 = str(var_7)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 2/15 statements.


def test_case_0():
    var_0 = '.module'
    var_1 = 'B'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_section_key_line_12_predicate_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'from os import path'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = 'A'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_length_sort_false_returns_module_name. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'test_module'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from package import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_true. Retrieved 10/30 statements.


import re as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = '^import '
    var_5 = ''
    var_6 = module_0.sub(var_4, var_5, var_3)
    var_7 = 0
    var_8 = ' '
    var_9 = processed_line.split(var_8)[var_7]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_true. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module'
    var_2 = False
    var_3 = None
    var_4 = 'A'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_section_key_predicate_line_4. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/7 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/7 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/9 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 3/9 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/10 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_relative_import_reverse_relative. Retrieved 4/9 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/9 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/9 statements.
# Partially parsed test_section_key_multiple_relative_dots. Retrieved 4/9 statements.
# Partially parsed test_section_key_length_sort_with_long_line. Retrieved 3/10 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'from __future__ import annotations'
    var_3 = 'A'

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
    var_1 = 'from Os import Path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from OS import PATH'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'B'

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
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from very_long_module_name import very_long_function_name'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import x'
    var_2 = 'B'
    var_3 = 1



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_relative_imports. Retrieved 4/8 statements.
# Partially parsed test_section_key_empty_section. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort_with_short_import. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
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

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

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
    var_1 = 'from os import path, sep'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = True
    var_1 = 'import a'
    var_2 = 'B'
    var_3 = result.split(var_2)[var_0]
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 > var_5



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'os import path'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_single_letter_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'imports'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'BA'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = '..package.module'

def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = [var_1]
    var_3 = False
    var_4 = 'A'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative_with_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_complex_import. Retrieved 4/10 statements.


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
    var_1 = 'AB0123456789'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'AB0123456789'

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
    var_0 = False
    var_1 = 'import MyModule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os.path import join, exists'
    var_3 = 'B'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_import_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'variable'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

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
    var_1 = '.. module'

def test_case_0():
    var_0 = True
    var_1 = '....package.module'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_variable_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'os'

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
    var_1 = 'MYCONST'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = False
    var_1 = '....submodule'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'B'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_case_sensitive_predicate_evaluates_to_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_predicate_line_37_evaluates_to_false. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = 'thirdparty'
    var_4 = False
    var_5 = str(var_3)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/7 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_with_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_spaces_in_from_import. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

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
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = 'from os import path, sys'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_true. Retrieved 5/8 statements.


import re as module_0

def test_case_0():
    var_0 = '^(from |import )'
    var_1 = module_0.compile(var_0)
    var_2 = ' import '
    var_3 = module_0.compile(var_2)
    var_4 = True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 5/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 3/8 statements.
# Partially parsed test_section_key_lexicographical_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 4/11 statements.
# Partially parsed test_section_key_group_by_package_true. Retrieved 3/8 statements.
# Partially parsed test_section_key_reverse_relative_with_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_true. Retrieved 4/9 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_true_reverse. Retrieved 3/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/9 statements.
# Partially parsed test_section_key_honor_case_import_statement. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 4/9 statements.
# Partially parsed test_section_key_import_statement. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = module_0.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

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
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #73
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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_case_sensitive_predicate_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = None



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_section_key_predicate_line_12_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/10 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_relative_import. Retrieved 2/5 statements.


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
    var_2 = 1
    var_3 = 3

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from module import something'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from module import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Something'

def test_case_0():
    var_0 = 'from os import path'

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
    var_0 = True
    var_1 = 'from .. import module'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_variable_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_multiple_options. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..utils.helpers'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'MyModule'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_force_to_top_predicate_false. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = False
    var_2 = None
    var_3 = 'B'
    var_4 = 'A'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_line_12_predicate_evaluates_to_true. Retrieved 2/20 statements.


def test_case_0():
    var_0 = 'from package import something'
    var_1 = 'from'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_by_case. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = 'module'

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
    var_1 = 'function'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'A'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_section_key_basic. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_length_sort. Retrieved 8/12 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 4/9 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 4/8 statements.


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
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 0
    var_4 = result.split(var_2)[var_0]
    var_5 = 'os'
    var_6 = var_5.split(var_5)[var_3]
    var_7 = len(var_6)

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import MyModule'
    var_3 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, environ'
    var_2 = 'B'

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
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_imports. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_with_different_settings. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_split_module. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_length_sort_comparison. Retrieved 3/9 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ..module import func'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Django import models'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import django'
    var_4 = 'import flask'
    var_5 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'import collections'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 19/24 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = 'sort_relative_in_force_sorted_sections'
    var_2 = 'reverse_relative'
    var_3 = 'group_by_package'
    var_4 = 'lexicographical'
    var_5 = 'force_to_top'
    var_6 = 'honor_case_in_force_sorted_sections'
    var_7 = 'case_sensitive'
    var_8 = 'order_by_type'
    var_9 = 'length_sort'
    var_10 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = False
    var_12 = 'os'
    var_13 = 'sys'
    var_14 = [var_12, var_13]
    var_15 = True
    var_16 = 'import json'
    var_17 = 'B'
    var_18 = "Section should be 'B' when predicate at line 23 is False"



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 10/14 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = False
    var_3 = False
    var_4 = []
    var_5 = False
    var_6 = True
    var_7 = True
    var_8 = False
    var_9 = '.module'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_uppercase_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_relative_dots. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '..module'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = '...submodule'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 3/7 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

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
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'mymodule'

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
    var_1 = '....module.submodule'

def test_case_0():
    var_0 = False
    var_1 = '.module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'django'
    var_3 = [var_2]
    var_4 = 'Django'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

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

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'myvar'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'myvar'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = 'os'

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
    var_1 = '.. module_name'

def test_case_0():
    var_0 = False
    var_1 = '....module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = 'MyModule'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 22/55 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 're'
    var_2 = __import__(var_1)
    var_3 = '^(\\.+)\\s*(.*)'
    var_4 = module_0.match(var_3, var_0)
    var_5 = ' '
    var_6 = '_'
    var_7 = var_5 if var_1 else var_6
    var_8 = ''
    var_9 = False
    var_10 = str(var_0)
    var_11 = str(var_0)
    var_12 = False
    var_13 = 'A'
    var_14 = 'B'
    var_15 = 'C'
    var_16 = 'A'
    var_17 = 'B'
    var_18 = 'C'
    var_19 = None
    var_20 = False
    var_21 = str(var_19)



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'module'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_predicate_line_37_evaluates_to_true. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'stdlib'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_case_sensitive_predicate_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = None
    var_3 = 'testmodule'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_import_statement. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort_with_longer_line. Retrieved 7/10 statements.
# Partially parsed test_section_key_case_sensitive_honor_case. Retrieved 3/6 statements.
# Partially parsed test_section_key_empty_line_handling. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'

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
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import very_long_module_name'
    var_2 = 'B'
    var_3 = result.split(var_2)[var_0]
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 > var_5

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'

def test_case_0():
    var_0 = 'import a'



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_remove_from_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_remove_import_prefix. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_not_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_options. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'Bimport'

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
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import sys'
    var_5 = 'A'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 4/7 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/7 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 8/12 statements.
# Partially parsed test_section_key_order_by_type_false_lowercase. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_dots. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_space. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_underscore. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 1/4 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 3/7 statements.


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
    var_1 = 1
    var_2 = 'B'
    var_3 = result.split(var_2)[var_1]

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = module_0.split(var_2)
    var_4 = 1
    var_5 = var_3[var_4]
    var_6 = len(var_5)
    var_7 = var_3[var_4][var_0]

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'

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
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = 'from os import path, sys'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'import os'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_relative_imports. Retrieved 2/5 statements.
# Partially parsed test_section_key_remove_from_prefix. Retrieved 3/7 statements.
# Partially parsed test_section_key_remove_import_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 4/8 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B9'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'Bfrom'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

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
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_combined_force_to_top_and_length_sort. Retrieved 4/9 statements.
# Partially parsed test_module_key_sub_imports_false_order_by_type. Retrieved 3/7 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'

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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'

def test_case_0():
    var_0 = False
    var_1 = '.. module_name'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = False

def test_case_0():
    var_0 = ''



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = ' '
    var_1 = '_'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_line_20_predicate_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test_module'
    var_2 = False
    var_3 = None



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_default. Retrieved 2/5 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.
# Partially parsed test_module_key_order_by_type_disabled. Retrieved 3/6 statements.
# Partially parsed test_module_key_multiple_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '..module'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'test'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'

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
    var_1 = 'variable'

def test_case_0():
    var_0 = False
    var_1 = '...package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = True

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = 'A'



# Parsed testcases at query #110
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



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = '.module'



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_not_in_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_with_import. Retrieved 3/7 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

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
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = False
    var_1 = 'from Os import Path'

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
    var_2 = 'from module import Name'



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_true. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
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
# Partially parsed test_module_key_multiple_dots. Retrieved 2/5 statements.
# Partially parsed test_module_key_single_dot. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = '....utils'

def test_case_0():
    var_0 = False
    var_1 = '.module'



# Parsed testcases at query #116
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/11 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_relative_imports. Retrieved 2/5 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 5/9 statements.
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
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from .. import os'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = ''
    var_1 = 'B'



# Parsed testcases at query #117
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = 0
    var_4 = ' '
    var_5 = line.split(var_4)[var_3]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.
# Partially parsed test_module_key_multiple_options_combined. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

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
    var_1 = '..package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'MyModule'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 5/13 statements.


import re as module_0

def test_case_0():
    var_0 = '...relative.module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/5 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/5 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from package import name'

def test_case_0():
    var_0 = True
    var_1 = 'from package import name'

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
    var_2 = 'from Package import Name'

def test_case_0():
    var_0 = True
    var_1 = 'import MyModule'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'from ..package import module'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_section_name_length_sort. Retrieved 3/6 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.
# Partially parsed test_module_key_single_character. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

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

def test_case_0():
    var_0 = '....deep.module'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'a'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 4/7 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 5/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_imports_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_with_different_settings. Retrieved 3/7 statements.
# Partially parsed test_section_key_complex_line. Retrieved 4/9 statements.


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
    var_1 = 1
    var_2 = 'B'
    var_3 = result.split(var_2)[var_1]

def test_case_0():
    var_0 = 'import os'
    var_1 = 1
    var_2 = 'B'
    var_3 = result.split(var_2)[var_1]
    var_4 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Django'

def test_case_0():
    var_0 = False
    var_1 = 'import Django'

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
    var_1 = False
    var_2 = 'from Module import Function'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from package.module import ClassA, ClassB'
    var_3 = 'B'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '...relative_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_disabled. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = False
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = '....package.module'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_false_no_prefix. Retrieved 3/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_dot_module. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = False
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'variable'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = '.module'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_from_dot. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_true. Retrieved 2/7 statements.
# Partially parsed test_section_key_length_sort_false. Retrieved 4/11 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 5/9 statements.


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
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_module_key_relative_import_with_reverse_relative_true. Retrieved 1/11 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '. utils'

def test_case_0():
    var_0 = '. utils'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_section_name_length_sort. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_sub_imports_false_order_by_type_true. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'

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
    var_1 = 'UPPERCASE'

def test_case_0():
    var_0 = True
    var_1 = 'Capitalized'

def test_case_0():
    var_0 = True
    var_1 = 'lowercase'

def test_case_0():
    var_0 = 'thirdparty'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'A'
    var_2 = 'Predicate should evaluate to True when module_name is in config.force_to_top'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_length_sort_false_returns_module_name. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'test_module'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/12 statements.


import re as module_0

def test_case_0():
    var_0 = '...relative_module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_imports_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_imports_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_combined_force_to_top_and_length_sort. Retrieved 4/8 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'

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
    var_1 = 'CONST'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'variable'

def test_case_0():
    var_0 = False
    var_1 = '.. module'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = False
    var_3 = 'B'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)
    var_4 = len(var_0)
    var_5 = str(var_4)
    var_6 = ':'
    var_7 = var_5 + var_6
    var_8 = var_7 + var_0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 24/54 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 're'
    var_2 = __import__(var_1)
    var_3 = '^(\\.+)\\s*(.*)'
    var_4 = module_0.match(var_3, var_0)
    assert var_4 is None
    var_5 = ''
    var_6 = False
    var_7 = str(var_0)
    var_8 = str(var_0)
    var_9 = False
    var_10 = 'A'
    var_11 = 'B'
    var_12 = 'C'
    var_13 = 'A'
    var_14 = 'B'
    var_15 = 'C'
    var_16 = None
    var_17 = False
    var_18 = str(var_16)
    var_19 = len(var_8)
    var_20 = str(var_19)
    var_21 = ':'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_8



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/9 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_empty_section_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_with_multiline_import. Retrieved 1/5 statements.


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
    var_1 = 1
    var_2 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'

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
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import (path, sep)'



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_line_29_predicate_evaluates_to_true.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/5 statements.
# Partially parsed test_module_key_single_dot. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '..utils'

def test_case_0():
    var_0 = True
    var_1 = '..utils'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = False
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'os'

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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '.module'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'A'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/4 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 1/4 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_complex_module_name. Retrieved 1/4 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'sys'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'test'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'module'

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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = False
    var_1 = '..module'

def test_case_0():
    var_0 = 'package.subpackage.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '....module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_sensitive_order_by_type_diff. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_insensitive_order_by_type_diff. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.
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
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'Bimport'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from .. import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from package.subpackage import ClassA, ClassB'
    var_3 = 'B'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 3/18 statements.
# Partially parsed test_predicate_at_line_20_evaluates_to_false_with_sub_imports_true. Retrieved 4/19 statements.
# Partially parsed test_predicate_at_line_20_evaluates_to_false_with_order_by_type_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None

def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_true. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_relative_import_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 4/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_combined_options. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'from __future__ import annotations'
    var_3 = 'A'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

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
    var_1 = True
    var_2 = 'import Os'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import sys'
    var_5 = 'A'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_single_dot. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_import_double_dot. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_relative_import. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'module'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'

def test_case_0():
    var_0 = True
    var_1 = '.module'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = True
    var_1 = '...'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_section_length_sort. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'

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
    var_1 = 'CONST'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = '....my_module'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_imports. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_with_length_sort_includes_length. Retrieved 2/7 statements.
# Partially parsed test_section_key_multiple_relative_dots. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 4/8 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_with_from_import. Retrieved 3/7 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import Os'
    var_3 = 'B'

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
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import abc'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_relative_imports_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/5 statements.
# Partially parsed test_section_key_multiple_relative_dots. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

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
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 3/10 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_dotted_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT_VAR'
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
    var_1 = 'UPPERCASE'

def test_case_0():
    var_0 = True
    var_1 = 'ClassName'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'package.module.submodule'

def test_case_0():
    var_0 = True
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = '.. package'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = None
    var_3 = str(var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_reverse_relative. Retrieved 2/5 statements.
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
# Partially parsed test_module_key_order_by_type_class_prefix. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'

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
    var_1 = 'variable'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '. module'

def test_case_0():
    var_0 = False
    var_1 = '....package'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_true. Retrieved 4/7 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_case_sensitive_predicate_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #43
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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 5/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_removes_from_keyword. Retrieved 2/5 statements.
# Partially parsed test_section_key_removes_import_keyword. Retrieved 3/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 4/11 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_relative_imports_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_complex_import. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = module_0.split(var_2)
    var_4 = len(var_3)

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import os'

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
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = False
    var_1 = 'from package.module import name'
    var_2 = 'B'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = 'B'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 7/12 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 7/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/8 statements.
# Partially parsed test_section_key_reverse_relative_not_force_sorted. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 3/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 4/9 statements.
# Partially parsed test_section_key_case_sensitive_and_order_by_type_differ. Retrieved 4/9 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 3/8 statements.
# Partially parsed test_section_key_from_import_with_from_prefix. Retrieved 1/5 statements.
# Partially parsed test_section_key_import_prefix_removal. Retrieved 2/6 statements.


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
    var_2 = 1
    var_3 = 'B'
    var_4 = result.split(var_3)[var_2]
    var_5 = 'os'
    var_6 = var_4.split(var_5)[var_0]

import re as module_0

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = module_0.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = var_3[var_0][var_5]

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import collections'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 3/8 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_uppercase_module. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_import_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_complex_module_path. Retrieved 1/4 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = False
    var_1 = 'os'
    var_2 = ':'

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
    var_1 = 'UPPER'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '.. module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'package.subpackage.module'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 7/10 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_section_key_relative_imports. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_with_different_settings. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_returns_string. Retrieved 1/5 statements.
# Partially parsed test_section_key_with_complex_import. Retrieved 2/6 statements.


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

def test_case_0():
    var_0 = 'import os'
    var_1 = 'AB0123456789'
    var_2 = 'import'

import re as module_0

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'os'
    var_3 = module_0.split(var_2)
    var_4 = 0
    var_5 = var_3[var_4]
    var_6 = len(var_5)

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'

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
    var_2 = 'from os import Path, func'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import module'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from package.subpackage import (Class, function, constant)'
    var_1 = 'B'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_case_sensitive_predicate_evaluates_to_false. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = None



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_module_key_simple_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'test'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'sys'

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
    var_1 = 'my_var'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = False
    var_3 = True
    var_4 = 'A'

def test_case_0():
    var_0 = False
    var_1 = '.. module'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 5/50 statements.


import re as module_0

def test_case_0():
    var_0 = '^from\\s+|^import\\s+'
    var_1 = module_0.compile(var_0)
    var_2 = '\\s+import\\s+'
    var_3 = module_0.compile(var_2)
    var_4 = 'from ... import module'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_sub_imports_and_order_by_type. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_import_complex. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_all_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

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
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'imports'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test'
    var_3 = [var_2]
    var_4 = 'Test'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 2/23 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #55
#--------------------------

# Failed to parse test_lexicographical_predicate_evaluates_to_true.




# Parsed testcases at query #56
#--------------------------

# Partially parsed test_section_key_length_sort_enabled. Retrieved 5/9 statements.


def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = len(var_1)
    var_4 = str(var_3)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 32/54 statements.


import re as module_0

def test_case_0():
    var_0 = '^from .* import |^import '
    var_1 = module_0.compile(var_0)
    var_2 = ' import '
    var_3 = module_0.compile(var_2)
    var_4 = 'os'
    var_5 = 'B'
    var_6 = '^from (\\.+)\\s*(.*)'
    var_7 = module_0.match(var_6, var_4)
    var_8 = ' '
    var_9 = f'from {var_0.join(var_2)}'
    var_10 = 0
    var_11 = ' import '
    var_12 = 1
    var_13 = line.split(var_11, var_12)[var_10]
    var_14 = ''
    var_15 = '.'
    var_16 = module_0.sub(var_15, var_13)
    var_17 = module_0.sub(var_14, var_16)
    var_18 = '^from '
    var_19 = ''
    var_20 = module_0.sub(var_18, var_19, var_17)
    var_21 = '^import '
    var_22 = module_0.sub(var_21, var_19, var_20)
    var_23 = ' '
    var_24 = '_'
    var_25 = var_23 if var_18 else var_24
    var_26 = '^(\\.+)'
    var_27 = f'\\1{var_25}'
    var_28 = module_0.sub(var_26, var_27, var_22)
    var_29 = 0
    var_30 = ' '
    var_31 = line.split(var_30)[var_29]



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_by_first_letter. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_complex_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_false_no_prefix. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'mymodule'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = '.. module_name'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = [var_1]
    var_3 = False
    var_4 = 'BA'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true_reverse_relative_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = '.module'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_by_first_letter. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_all_options. Retrieved 4/8 statements.
# Partially parsed test_module_key_relative_with_content. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

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
    var_1 = 'CONSTANT_VALUE'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..utils'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'os'
    var_3 = [var_2]

def test_case_0():
    var_0 = True
    var_1 = '.. package'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_section_key_default_behavior. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 8/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 4/9 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_import_statement. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_relative_import_with_spaces. Retrieved 4/8 statements.


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
    var_2 = 0
    var_3 = 'B'
    var_4 = result.split(var_3)[var_0]
    var_5 = 'os'
    var_6 = var_4.split(var_5)[var_2]
    var_7 = len(var_6)

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Django'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'
    var_3 = 0

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
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = 'import numpy'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import django'
    var_4 = 'import flask'
    var_5 = 'A'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_lexicographical_predicate_is_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #66
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



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_section_key_predicate_at_line_43. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_relative_imports_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_class_first_letter. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_combined_options. Retrieved 4/9 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
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
    var_1 = 'UPPERCASE'
    var_2 = 'BA'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'BB'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'sys'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'CONST'
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = '. module'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 11/31 statements.


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



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 5/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 5/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 5/8 statements.
# Partially parsed test_section_key_order_by_type_false_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_dots. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 5/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_with_import. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_split_module. Retrieved 5/8 statements.
# Partially parsed test_section_key_case_sensitive_true_order_by_type_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_empty_string. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = module_0.split(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 1
    var_3 = 'B'
    var_4 = result.split(var_3)[var_2]

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 1
    var_3 = 'B'
    var_4 = result.split(var_3)[var_2]

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'
    var_3 = 'B'
    var_4 = result.split(var_3)[var_0]

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
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import PATH'
    var_3 = 'B'
    var_4 = result.split(var_3)[var_0]

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'import OS'

def test_case_0():
    var_0 = ''
    var_1 = 'B'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 5/24 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import os'
    var_3 = 'import os'
    var_4 = 'import os'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_length_sort_with_long_line. Retrieved 3/9 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

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
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from very_long_module_name import very_long_function_name'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = False
    var_2 = None
    var_3 = 'A'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_not_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_no_length_sort. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_uppercase_first_letter. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_no_sub_imports. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = False
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '.. module'

def test_case_0():
    var_0 = True
    var_1 = '....submodule'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_lexicographical_with_import. Retrieved 2/5 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

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
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'import os'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'import os'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 43 evaluates to False when length_sort is False.'
    var_1 = False
    var_2 = 'os'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_space. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_underscore. Retrieved 3/7 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_and_order_by_type. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_honor_case_mixed_conditions. Retrieved 3/6 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_empty_force_to_top. Retrieved 3/7 statements.


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
    var_1 = 'from package import module'

def test_case_0():
    var_0 = True
    var_1 = 'from package import module'

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
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from Package import Module'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = 'import package'

def test_case_0():
    var_0 = 'from package import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Package import Module'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = []
    var_1 = 'import os'
    var_2 = 'B'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = 'from . import something'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = 'B'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'numpy'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate. Retrieved 5/17 statements.


import re as module_0

def test_case_0():
    var_0 = '^from|^import'
    var_1 = module_0.compile(var_0)
    var_2 = '\\s+import\\s+'
    var_3 = module_0.compile(var_2)
    var_4 = 'from os import path'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 4/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_options. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import os'
    var_5 = 'A'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'A'
    var_2 = 'B'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 3/8 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 5/8 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_import_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_false_no_prefix. Retrieved 3/7 statements.
# Partially parsed test_module_key_all_parameters. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

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

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = ':'

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
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'CONSTANT'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'myvar'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '.. module'

def test_case_0():
    var_0 = False
    var_1 = '....package.module'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = 'stdlib'
    var_5 = 'A'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 5/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/10 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_and_order_by_type_mismatch. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/7 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.


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
    var_3 = result.split(var_2)[var_0]
    var_4 = len(var_3)

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from . import test'

def test_case_0():
    var_0 = True
    var_1 = 'from package import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import test'

def test_case_0():
    var_0 = True
    var_1 = 'from . import test'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import test'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Package import Name'

def test_case_0():
    var_0 = False
    var_1 = 'from PACKAGE import NAME'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import test'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 5/38 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_empty_like_import. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

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
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'B'

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
    var_2 = 'import Os'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from module import func1, func2'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from ..module import func'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import'
    var_1 = 'B'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 19/46 statements.


import re as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = ''
    var_4 = '.'
    var_5 = '^from '
    var_6 = ''
    var_7 = module_0.sub(var_5, var_6, var_2)
    var_8 = '^import '
    var_9 = module_0.sub(var_8, var_6, var_7)
    var_10 = ' '
    var_11 = '_'
    var_12 = var_10 if var_5 else var_11
    var_13 = '^(\\.+)'
    var_14 = f'\\1{var_12}'
    var_15 = module_0.sub(var_13, var_14, var_9)
    var_16 = 0
    var_17 = ' '
    var_18 = line.split(var_17)[var_16]



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_true. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = 'B'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_sort_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 3/6 statements.
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
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

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
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = 'from os import path'

def test_case_0():
    var_0 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = ''
    var_1 = 'B'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = False
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'

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
    var_1 = '....submodule'

def test_case_0():
    var_0 = False
    var_1 = '.module'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 21/53 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'imports'
    var_2 = '^(\\.+)\\s*(.*)'
    var_3 = module_0.match(var_2, var_0)
    var_4 = ' '
    var_5 = '_'
    var_6 = var_4 if var_2 else var_5
    var_7 = ''
    var_8 = False
    var_9 = str(var_0)
    var_10 = str(var_0)
    var_11 = False
    var_12 = 'A'
    var_13 = 'B'
    var_14 = 'C'
    var_15 = 'A'
    var_16 = 'B'
    var_17 = 'C'
    var_18 = False
    var_19 = var_5 and var_18
    var_20 = str(var_1)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = 'from . import something'



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'A'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 5/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_remove_from_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_remove_import_prefix. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_complex_line. Retrieved 4/9 statements.


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
    var_3 = result.split(var_2)[var_0]
    var_4 = len(var_3)

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .module import something'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from package import module'
    var_3 = 'B'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_section_key_predicate_line_12. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = 'B'



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative.module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



