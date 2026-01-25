####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_keyword. Retrieved 1/4 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_with_multiple_imports. Retrieved 2/6 statements.
# Partially parsed test_section_key_preserves_section_letter. Retrieved 1/4 statements.


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
    var_1 = 'from'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

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
    var_0 = False
    var_1 = 'import os'
    var_2 = 1

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_letter. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_starts_with_uppercase. Retrieved 2/5 statements.


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
    var_1 = 'module'
    var_2 = ':'

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
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'B'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 2/5 statements.


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
    var_1 = 'MyModule'
    var_2 = False

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
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'
    var_2 = 'C'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = False
    var_1 = '.module'
    var_2 = 'module'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_case_sensitive_predicate_evaluates_to_false. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 4/8 statements.


import re as module_0

def test_case_0():
    var_0 = True
    var_1 = '..package'
    var_2 = '^(\\.+)\\s*(.*)'
    var_3 = module_0.match(var_2, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_true. Retrieved 3/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from package import something'
    var_2 = 'from'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_combined_ignore_case_and_case_sensitive. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = '...module'

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
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

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
    var_0 = '....module.submodule'

def test_case_0():
    var_0 = '.module'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_line_20_evaluates_to_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_sub_imports_and_order_by_type. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_classes. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_variables. Retrieved 3/7 statements.
# Partially parsed test_module_key_uppercase_module. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_without_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_in_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_import_with_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_multiple_relative_dots. Retrieved 1/5 statements.
# Partially parsed test_module_key_straight_import_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = '..utils'

def test_case_0():
    var_0 = True
    var_1 = '..utils'

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
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = False
    var_1 = 'os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

def test_case_0():
    var_0 = '.utils'

def test_case_0():
    var_0 = '...utils'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/4 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/4 statements.
# Partially parsed test_module_key_with_ignore_case. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_force_to_top_not_in_list. Retrieved 4/8 statements.
# Partially parsed test_module_key_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_order_by_type_class_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 1/4 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '..module'
    var_1 = 'module'

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
    var_1 = 'os'
    var_2 = '2:os'

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
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_1 = 'my_module'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

def test_case_0():
    var_0 = False
    var_1 = '..module'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = '...package.module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = 'A'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_module_key_predicate_line_11. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '. relative_module'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_section_key_basic. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_case_sensitive. Retrieved 4/8 statements.
# Partially parsed test_section_key_order_by_type. Retrieved 3/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 3/7 statements.
# Partially parsed test_section_key_import_statement. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import os'
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'
    var_3 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'B'
    var_3 = 'os'

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
    var_0 = True
    var_1 = 'from ... import something'
    var_2 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'



# Parsed testcases at query #15
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
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


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
    var_0 = []
    var_1 = 'os'
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
    var_2 = 'module'
    var_3 = '6:module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = '3:sys'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'
    var_2 = ' '

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = '_'

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
    var_1 = 'test'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = ':'

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
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = '..utils.helpers'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/5 statements.
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
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

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
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'from'

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
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'os'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '... module'
    var_1 = ' '



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_module_key_relative_import_with_reverse_relative_true. Retrieved 5/17 statements.


import re as module_0

def test_case_0():
    var_0 = '... module_name'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = 'A'
    var_4 = 'A'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = len(var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 4/18 statements.


import re as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '^(\\.+)\\s*(.*)'
    var_3 = module_0.match(var_2, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_complex_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_dots. Retrieved 2/5 statements.


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
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import OS'
    var_2 = 'B'

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
    var_1 = False
    var_2 = 'from OS import path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from package.module import function'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from ..package import module'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'
    var_2 = 'B'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/7 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_sub_imports_and_order_by_type. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_sub_imports_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_sub_imports_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_length_sort. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_length_sort_straight. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/8 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/7 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/7 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_flags. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = '...module'

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
    var_1 = 'MY_CONSTANT'
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

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'A'
    var_5 = ':'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_section_key_predicate_line_12. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'
    var_2 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_module_key_predicate_line_20_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_line_42_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = False
    var_2 = None
    var_3 = 'A'
    var_4 = 'A'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/14 statements.
# Partially parsed test_module_key_relative_import. Retrieved 1/14 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 1/14 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/15 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 1/14 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 1/14 statements.
# Partially parsed test_module_key_length_sort. Retrieved 1/14 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/15 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 2/15 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 2/15 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 2/15 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 2/15 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/15 statements.
# Failed to parse test_module_key_order_by_type_capitalized.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '...module'

def test_case_0():
    var_0 = '..module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True

def test_case_0():
    var_0 = 'MyModule'

def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = True

def test_case_0():
    var_0 = 'thirdparty'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = 'MY_CONSTANT'
    var_1 = True

def test_case_0():
    var_0 = 'MyClass'
    var_1 = True

def test_case_0():
    var_0 = 'my_var'
    var_1 = True

def test_case_0():
    var_0 = 'CONSTANT'
    var_1 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/8 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import_reverse. Retrieved 4/8 statements.
# Partially parsed test_section_key_relative_import_force_sorted. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted. Retrieved 4/8 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_with_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_complex_from_import. Retrieved 4/8 statements.


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
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

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
    var_1 = False
    var_2 = 'from . import something'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from collections import OrderedDict'
    var_1 = 'B'
    var_2 = 'collections'

def test_case_0():
    var_0 = '__future__'
    var_1 = 'os'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'A'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'from .. import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from django.conf import settings'
    var_3 = 'B'
    var_4 = 'django.conf'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_remove_from_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_remove_import_prefix. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false_order_by_type_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_from_dot. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_with_split_import. Retrieved 3/7 statements.


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
    var_1 = 'from os import path'
    var_2 = 'from'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

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
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'Os import Path'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = None
    var_3 = str(var_2)
    var_4 = len(var_0)
    var_5 = str(var_4)
    var_6 = ':'
    var_7 = var_5 + var_6
    var_8 = var_7 + var_0



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 5/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_with_order_by_type. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 2/6 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 2/5 statements.


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
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import path'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'

def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 8/27 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 're'
    var_2 = __import__(var_1)
    var_3 = '^(\\.+)\\s*(.*)'
    var_4 = module_0.match(var_3, var_0)
    assert var_4 is None
    var_5 = False
    var_6 = None
    var_7 = str(var_6)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_variable_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_multiple_dots_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_single_dot_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.


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
    var_1 = 'my_variable'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'mymodule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = ':'

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = [var_0]
    var_2 = 'mymodule'
    var_3 = ':'

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
    var_1 = '....module.submodule'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '.module'
    var_2 = 'module'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_module_key_simple_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_mixed_case_class. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

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
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'A'
    var_6 = 'B'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'A'
    var_2 = 'mymodule'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_section_key_not_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_section_key_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_sensitive_true_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_sensitive_false_order_by_type_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 5/9 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.


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
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'os import path'

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
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import flask'
    var_4 = 'A'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'
    var_2 = 'B'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_true. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'os'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 4/10 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_insensitive_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_imports_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_with_different_settings. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_module_name_lowered. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_names_lowered. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_force_to_top_modules. Retrieved 6/10 statements.
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
    var_2 = 'B'
    var_3 = 2

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import something'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 're'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'import sys'
    var_3 = 'B'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = ':'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #54
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
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 5/8 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_combined_options. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

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
    var_1 = 'os'
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
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'UPPER'
    var_5 = 'A'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'Capitalized'
    var_3 = [var_2]
    var_4 = []
    var_5 = 'B'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'lowercase'
    var_5 = 'C'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = []
    var_5 = 'A'
    var_6 = ':'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_imports_reverse. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 4/10 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_not_in_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_includes_length. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.


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
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

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
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'from'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'import'

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
    var_1 = 'import a'
    var_2 = 2

def test_case_0():
    var_0 = True
    var_1 = 'import OS'
    var_2 = 'OS'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ...module import something'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 4/7 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 4/7 statements.
# Partially parsed test_section_key_length_sort. Retrieved 7/10 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 4/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_case_sensitive. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections_not_case_sensitive. Retrieved 3/6 statements.
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
    var_0 = 'from os import path'
    var_1 = 1
    var_2 = 'B'
    var_3 = result.split(var_2)[var_1]
    var_4 = 'from'
    var_5 = bool('from' not in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = 'import os'
    var_1 = 1
    var_2 = 'B'
    var_3 = result.split(var_2)[var_1]
    var_4 = 'import'
    var_5 = bool('import' not in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = result.split(var_2)[var_0]
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 > var_5

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
    var_2 = 'B'
    var_3 = result.split(var_2)[var_0]
    var_4 = 'import'
    var_5 = bool('import' not in var_3)
    assert var_5 is True

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
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = 'B'
    var_2 = 'collections'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import_reverse. Retrieved 4/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_import_with_multiple_names. Retrieved 3/7 statements.
# Partially parsed test_section_key_force_to_top_multiple_packages. Retrieved 5/9 statements.
# Partially parsed test_section_key_length_sort_with_long_import. Retrieved 3/10 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/10 statements.


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
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

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
    var_3 = 'os'

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
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path, sys'
    var_2 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from very_long_package_name import very_long_function_name'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = 'thirdparty'
    var_4 = False
    var_5 = str(var_3)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 11/31 statements.


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



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'numpy'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_case_sensitive_true_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_case_sensitive_false_order_by_type_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 1/4 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.


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
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
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
    var_0 = True
    var_1 = False
    var_2 = 'from OS import PATH'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from os import path, sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from .. import module'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_no_length_sort. Retrieved 5/10 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 4/7 statements.
# Partially parsed test_module_key_combined_options. Retrieved 6/11 statements.


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
    var_0 = 'future'
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
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'os'
    var_3 = 'stdlib'
    var_4 = ':'

def test_case_0():
    var_0 = False
    var_1 = '. module'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '....package'
    var_2 = 'package'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = [var_1]
    var_3 = False

def test_case_0():
    var_0 = False
    var_1 = '__future__'
    var_2 = [var_1]
    var_3 = True
    var_4 = 'CONST'
    var_5 = [var_4]



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_without_order_by_type. Retrieved 3/6 statements.


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
    var_1 = 'ClassName'
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
    var_0 = '__future__'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = True
    var_3 = 'module'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_section_key_honor_case_predicate. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_section_key_predicate_at_line_4. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'



# Parsed testcases at query #69
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
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_order_by_type_class_uppercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 5/8 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.


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
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

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
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'CONST'
    var_5 = 'A'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'MyClass'
    var_5 = 'B'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'my_var'
    var_5 = 'C'

def test_case_0():
    var_0 = '..module.submodule'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = False
    var_1 = '. module'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_length_sort_false_predicate. Retrieved 11/31 statements.


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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_section_key_predicate_line_4. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'A'
    var_2 = 'mymodule'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight_false. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_starts_upper. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_relative. Retrieved 1/4 statements.


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
    var_0 = True
    var_1 = 'module'
    var_2 = False

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
    var_1 = 'MY_CONSTANT'
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
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = '.module'
    var_1 = 'module'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_section_key_returns_correct_format. Retrieved 7/14 statements.


def test_case_0():
    var_0 = False
    var_1 = 'os'
    var_2 = True
    var_3 = 'os'
    var_4 = 'os'
    var_5 = [var_4]
    var_6 = 'os'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_section_key_line_29_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'module'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_lowercase_when_not_order_by_type. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 7/10 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 4/8 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 7/11 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 5/9 statements.
# Partially parsed test_section_key_import_with_as. Retrieved 2/6 statements.


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
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = result.split(var_2)[var_0]
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 > var_5

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

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
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'from OS import Path'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1
    var_4 = result.split(var_2)[var_3]
    var_5 = len(var_4)
    var_6 = var_5 > var_0

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = 'B'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_parameters. Retrieved 3/7 statements.


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
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

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

def test_case_0():
    var_0 = True
    var_1 = 'my_var'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'

def test_case_0():
    var_0 = True
    var_1 = 'variable'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'TestModule'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_group_by_package_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/10 statements.
# Partially parsed test_section_key_lexicographical_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/9 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_from_import_line. Retrieved 2/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/7 statements.


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
    var_2 = 'B9'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = True
    var_1 = 'from . import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import os'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/7 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/7 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/9 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/8 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/9 statements.
# Partially parsed test_section_key_relative_import_reverse. Retrieved 4/9 statements.
# Partially parsed test_section_key_relative_import_force_sorted. Retrieved 4/9 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_and_order_by_type_mismatch. Retrieved 4/9 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 4/10 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/9 statements.
# Partially parsed test_section_key_complex_import_statement. Retrieved 4/9 statements.


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
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'B'
    var_3 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import utils'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import utils'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from MyModule import MyClass'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from MyModule import MyClass'
    var_3 = 'B'
    var_4 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import utils'
    var_3 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'from django.conf import settings'
    var_3 = 'A'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate. Retrieved 5/18 statements.


import re as module_0

def test_case_0():
    var_0 = '^(from |import )'
    var_1 = module_0.compile(var_0)
    var_2 = ' import '
    var_3 = module_0.compile(var_2)
    var_4 = 'from os import path'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_with_dots. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_not_in_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_complex_import. Retrieved 2/7 statements.


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
    var_2 = 'B10'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = '.'

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
    var_2 = 'from os import Path'

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
    var_0 = 'from package.subpackage import Module'
    var_1 = 'B'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 26/61 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = '^(\\.+)\\s*(.*)'
    var_6 = module_0.match(var_5, var_0)
    var_7 = ' '
    var_8 = '_'
    var_9 = var_7 if var_5 else var_8
    var_10 = ''
    var_11 = str(var_0)
    var_12 = str(var_0)
    var_13 = 'A'
    var_14 = 'B'
    var_15 = 'C'
    var_16 = 'A'
    var_17 = 'B'
    var_18 = 'C'
    var_19 = var_8 and var_4
    var_20 = str(var_3)
    var_21 = len(var_12)
    var_22 = str(var_21)
    var_23 = ':'
    var_24 = var_22 + var_23
    var_25 = var_24 + var_12



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 20/39 statements.


import re as module_0

def test_case_0():
    var_0 = '.module'
    var_1 = 'B'
    var_2 = '^from (\\.+)\\s*(.*)'
    var_3 = module_0.match(var_2, var_0)
    var_4 = ' '
    var_5 = 0
    var_6 = ' import '
    var_7 = 1
    var_8 = var_0.split(var_6, var_7)[var_5]
    var_9 = '^from '
    var_10 = ''
    var_11 = 'import.*'
    var_12 = '.'
    var_13 = module_0.sub(var_11, var_12, var_8)
    var_14 = module_0.sub(var_9, var_10, var_13)
    var_15 = '^from '
    var_16 = ''
    var_17 = module_0.sub(var_15, var_16, var_14)
    var_18 = '^import '
    var_19 = module_0.sub(var_18, var_16, var_17)



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_section_key_predicate_line_12. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'from package import something'
    var_1 = 'from'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/5 statements.


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
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONSTANT'
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
    var_0 = True
    var_1 = '.. submodule'
    var_2 = 'submodule'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = '._'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_with_sort_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_simple_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_with_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 3/6 statements.


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
    var_1 = 'import OS'
    var_2 = 'os'

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
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from . import module'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from ... import module'
    var_1 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path, path'
    var_3 = 'B'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/10 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_section_key_relative_dots. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 9/17 statements.
# Partially parsed test_section_key_combined_options. Retrieved 6/12 statements.


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
    var_1 = 'import'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'from'

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
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

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
    var_1 = 'from os import path'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'django'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'import os'
    var_5 = 'import sys'
    var_6 = 'import requests'
    var_7 = 'A'
    var_8 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = 'import os'
    var_5 = 'A'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_force_to_top_predicate_true. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule'
    var_2 = 'A'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_section_key_predicate_at_line_43_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = ''



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_class_capitalized. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_all_options. Retrieved 4/8 statements.


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
    var_1 = 'MyModule'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
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
    var_1 = 'variable'
    var_2 = 'BC'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = '8:mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = '8:mymodule'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'mymodule'
    var_3 = '8:mymodule'

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
    var_1 = '..utils.helpers'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'os'
    var_3 = [var_2]



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_lexicographical_predicate_false. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'
    var_2 = 'B'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/9 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_conditions. Retrieved 6/12 statements.


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
    var_1 = 'from module import name'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'from module import name'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = 'import os'
    var_5 = 'A'



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_import_keyword. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_module_name_lowered. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.
# Partially parsed test_section_key_relative_import_with_spaces. Retrieved 3/6 statements.
# Partially parsed test_section_key_length_sort_included_in_result. Retrieved 2/7 statements.


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
    var_1 = 'from '

def test_case_0():
    var_0 = 'import os'
    var_1 = 'AB'
    var_2 = 'import '

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import Django'
    var_2 = 'django'

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
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = 'import flask'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import a'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_multiple_imports. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import_dots. Retrieved 4/8 statements.
# Partially parsed test_section_key_empty_line. Retrieved 2/6 statements.
# Partially parsed test_section_key_complex_import. Retrieved 4/8 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

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
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = 'import '
    var_1 = 'B'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = 'import sys'
    var_3 = 'A'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 3/8 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constants. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/5 statements.
# Partially parsed test_module_key_multiple_relative_dots. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 5/10 statements.
# Partially parsed test_module_key_combination_all_options. Retrieved 6/11 statements.


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
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = 'BA'

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
    var_1 = 'mymodule'
    var_2 = '8:'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = '8:'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'mymodule'
    var_3 = '8:'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'AA'

def test_case_0():
    var_0 = 'sys'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '. module'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '....module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = False
    var_4 = 'A'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'MyClass'
    var_5 = [var_4]



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_uppercase_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/6 statements.


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
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '.. module'

def test_case_0():
    var_0 = True
    var_1 = '....package.module'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = False
    var_2 = None
    var_3 = 'A'



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '. utils'
    var_1 = ' '



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 11/34 statements.


import re as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 're'
    var_2 = __import__(var_1)
    var_3 = '^(\\.+)\\s*(.*)'
    var_4 = module_0.match(var_3, var_0)
    assert var_4 is None
    var_5 = ''
    var_6 = str(var_0)
    var_7 = str(var_0)
    var_8 = False
    var_9 = None
    var_10 = str(var_9)



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/11 statements.


import re as module_0

def test_case_0():
    var_0 = '..relative.module'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = ' '
    var_4 = '_'



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_true. Retrieved 5/13 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = 'from . import something'
    var_4 = 'from .'



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = 0
    var_4 = ' '
    var_5 = var_2.split(var_4)[var_3]



# Parsed testcases at query #114
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
# Partially parsed test_module_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_disabled. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/5 statements.
# Partially parsed test_module_key_empty_relative. Retrieved 1/5 statements.


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
    var_1 = 'ab'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'ab'
    var_2 = ':'

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
    var_0 = True
    var_1 = 'myvar'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = 'CONSTANT'
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = '.. module_name'
    var_2 = 'module_name'

def test_case_0():
    var_0 = '.'



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_module_key_predicate_false. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'B'
    var_6 = 'test_module'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_regular. Retrieved 2/5 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/4 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

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
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

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
    var_1 = 'regular_module'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = '..utils'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'special'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = ':'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 4/10 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 2/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
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
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 2

def test_case_0():
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import MyModule'
    var_3 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'os'

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
    var_2 = 'from MyModule import Name'

def test_case_0():
    var_0 = 'import sys'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'os'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 13/17 statements.


def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = False
    var_7 = False
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = False
    var_12 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_relative. Retrieved 1/5 statements.


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
    var_1 = 'MyModule'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'mymodule'

def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = []
    var_1 = 'mymodule'
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

def test_case_0():
    var_0 = True
    var_1 = 'myvar'
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = 'UPPERCASE'

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
    var_1 = '.. module_name'

def test_case_0():
    var_0 = '....'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_module_key_relative_import_with_reverse_relative_false. Retrieved 5/17 statements.


import re as module_0

def test_case_0():
    var_0 = '...utils'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 2/5 statements.


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
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'mymodule'
    var_2 = ':'

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
    var_0 = False
    var_1 = '.. module'

def test_case_0():
    var_0 = True
    var_1 = '....package.module'
    var_2 = 'package'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = False
    var_3 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_section_key_line_20_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type. Retrieved 7/12 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_multiple_relative_dots. Retrieved 2/5 statements.
# Partially parsed test_module_key_empty_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_uppercase_two_char. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '.module'
    var_2 = '_'
    var_3 = 'module'

def test_case_0():
    var_0 = True
    var_1 = '.module'
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
    var_1 = 'MyModule'
    var_2 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = [var_1]
    var_3 = 'MyClass'
    var_4 = [var_3]
    var_5 = 'my_var'
    var_6 = [var_5]
    var_7 = 'A'
    var_8 = 'B'
    var_9 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

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
    var_1 = '...module.submodule'
    var_2 = '_'
    var_3 = 'module'

def test_case_0():
    var_0 = False
    var_1 = '.'

def test_case_0():
    var_0 = True
    var_1 = 'AB'
    var_2 = 'A'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



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

# Partially parsed test_predicate_at_line_20_evaluates_to_true. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.module'
    var_1 = '_'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_29_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'os import path'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 4/10 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_order_by_type_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/12 statements.
# Partially parsed test_section_key_relative_import. Retrieved 1/4 statements.
# Partially parsed test_section_key_relative_import_multiple_dots. Retrieved 1/4 statements.


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
    var_3 = 1

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
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

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
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'from . import module'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from .. import module'
    var_1 = 'B'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_module_key_reverse_relative_true. Retrieved 5/19 statements.


import re as module_0

def test_case_0():
    var_0 = '...package'
    var_1 = '^(\\.+)\\s*(.*)'
    var_2 = module_0.match(var_1, var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = ' '
    var_5 = '_'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_case_sensitive_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_underscore. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_enabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_order_by_type_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_empty_string. Retrieved 1/5 statements.
# Partially parsed test_module_key_numeric_module. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'

def test_case_0():
    var_0 = 'MyModule'
    var_1 = True
    var_2 = 'mymodule'

def test_case_0():
    var_0 = False
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
    var_1 = 'UPPERCASE'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'test'
    var_2 = 'test'

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = ':'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = ':'

def test_case_0():
    var_0 = '....package.module'

def test_case_0():
    var_0 = '.module'

def test_case_0():
    var_0 = False
    var_1 = 'Module'
    var_2 = True

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'module123'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'from . import something'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False
    var_3 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_module_key_basic_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 3/10 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_combined_options. Retrieved 4/9 statements.


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
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'module'
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
    var_1 = 'my_module'
    var_2 = 'C'

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
    var_1 = True
    var_2 = 'CONST'
    var_3 = [var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'from . import something'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_section_key_predicate_line_43_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
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
# Partially parsed test_module_key_sub_imports_with_order_by_type. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = False
    var_1 = '...module'

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
    var_1 = 'module'
    var_2 = '6:module'

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
    var_1 = 'CONSTANT'
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
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_without_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive_config. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_single_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_without_order_by_type. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
    var_1 = '...module'
    var_2 = '_'

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
    var_1 = 'my_var'
    var_2 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'imports'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

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
    var_0 = True
    var_1 = '..parent.module'

def test_case_0():
    var_0 = True
    var_1 = '.module'
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_force_to_top_predicate_evaluates_to_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import something'
    var_3 = 'from .'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #30
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
# Partially parsed test_module_key_order_by_type_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_name. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_import_with_space. Retrieved 2/5 statements.


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
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'os'
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
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '.. module'
    var_2 = 'module'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/9 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 3/9 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 3/8 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/9 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_reverse_relative_with_dots. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_in_relative_import. Retrieved 3/7 statements.
# Partially parsed test_section_key_empty_line_handling. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 8/16 statements.


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
    var_0 = 'from os import path'
    var_1 = 'from'
    var_2 = 'B'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'AB0123456789'
    var_2 = 'import'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'B0'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

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

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = ''
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'import json'
    var_6 = 'A'
    var_7 = 'B'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 23/44 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'test_module'
    var_3 = False
    var_4 = False
    var_5 = None
    var_6 = False
    var_7 = None
    var_8 = ''
    var_9 = str(var_2)
    var_10 = str(var_2)
    var_11 = 'A'
    var_12 = 'B'
    var_13 = 'C'
    var_14 = 'A'
    var_15 = 'B'
    var_16 = 'C'
    var_17 = str(var_5)
    var_18 = len(var_10)
    var_19 = str(var_18)
    var_20 = ':'
    var_21 = var_19 + var_20
    var_22 = var_21 + var_10



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'thirdparty'
    var_2 = False
    var_3 = False
    var_4 = False
    var_5 = str(var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_section_key_line_12_predicate_true. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'from package import something'
    var_1 = 'from'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_multiple_dots. Retrieved 1/5 statements.
# Partially parsed test_module_key_sub_imports_default. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

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
    var_1 = 'MY_CONSTANT'
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
    var_0 = '...package.module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = '. module'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'A'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_class_like. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_complex_relative. Retrieved 2/7 statements.
# Partially parsed test_module_key_multiple_conditions. Retrieved 5/9 statements.


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
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'MyModule'

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
    var_0 = []
    var_1 = 'os'
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
    var_1 = 'mymodule'
    var_2 = 'C'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '.. relative.module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'django'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = ':'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_module_key_basic_import. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_normal. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 3/8 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 4/8 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_combined_relative_and_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_complex_scenario. Retrieved 7/11 statements.


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
    var_2 = False
    var_3 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'

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
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'future'
    var_1 = [var_0]
    var_2 = 'os'
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
    var_1 = '..MyModule'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'CONFIG'
    var_3 = [var_2]
    var_4 = 'sys'
    var_5 = [var_4]
    var_6 = 'A'
    var_7 = ':'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_removes_import_prefix. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 4/11 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/8 statements.
# Partially parsed test_section_key_reverse_relative_with_dots. Retrieved 4/9 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_space. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_underscore. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/9 statements.
# Partially parsed test_section_key_honor_case_with_import_statement. Retrieved 4/9 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 6/10 statements.
# Partially parsed test_section_key_empty_line. Retrieved 2/6 statements.
# Partially parsed test_section_key_with_multiple_imports. Retrieved 2/7 statements.


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
    var_2 = 'import'

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
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

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
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'
    var_3 = 'B'

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
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'django'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = ''
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = 'B'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_lexicographical_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_section_key_length_sort_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = len(var_1)
    var_3 = str(var_2)
    var_4 = 'B'
    var_5 = 1



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'from'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = False
    var_3 = str(var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/10 statements.
# Partially parsed test_section_key_relative_imports. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_space. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_underscore. Retrieved 2/6 statements.
# Partially parsed test_section_key_combined_options. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'
    var_2 = [var_1]
    var_3 = 'A'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = True
    var_2 = 'path'

def test_case_0():
    var_0 = 'import os'
    var_1 = True

def test_case_0():
    var_0 = 'import OS'
    var_1 = True
    var_2 = False
    var_3 = 'os'

def test_case_0():
    var_0 = 'import Os'
    var_1 = False
    var_2 = 'os'

def test_case_0():
    var_0 = 'from . import something'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = True
    var_2 = 'import'

def test_case_0():
    var_0 = 'from os import Path'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'from ... import module'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'from .. import module'
    var_1 = True

def test_case_0():
    var_0 = 'from package import Something'
    var_1 = True
    var_2 = False
    var_3 = 'package'
    var_4 = [var_3]
    var_5 = 'A'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_section_key_predicate_line_23_evaluates_to_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'os'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_section_key_predicate_line_4. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 7/32 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = 'from .'
    var_3 = False
    var_4 = 'from . import something'
    var_5 = 'import something'
    var_6 = 'import something'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_no_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_insensitive. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_class_uppercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/7 statements.
# Partially parsed test_module_key_combined_flags. Retrieved 5/9 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
# Partially parsed test_module_key_dots_only. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = '...module'

def test_case_0():
    var_0 = False
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
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'os'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = ':'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = '...'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_42_evaluates_to_false. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = 'B'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'module_name'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top_section_a. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_import_keyword. Retrieved 1/4 statements.
# Partially parsed test_section_key_removes_from_keyword. Retrieved 1/4 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false_lowercases. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_true_preserves_case. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_not_forced. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_space. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_underscore. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_preserves_names_when_needed. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 3/6 statements.
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
    var_0 = 'import os'
    var_1 = 'os'
    var_2 = 'import'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'os'
    var_2 = 'from'

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
    var_1 = True
    var_2 = 'import Os'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import test'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from . import test'
    var_2 = 'B'

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
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from ... import test'
    var_3 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'



# Parsed testcases at query #54
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
# Partially parsed test_module_key_order_by_type_class_name. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_no_sub_imports. Retrieved 5/9 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.


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
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = ':'

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
    var_1 = 'ClassName'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = '. module'

def test_case_0():
    var_0 = True
    var_1 = 'MY_CONST'
    var_2 = [var_1]
    var_3 = False
    var_4 = 'B'

def test_case_0():
    var_0 = False
    var_1 = '....module_name'
    var_2 = 'module_name'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = ':'
    var_4 = 'B11:'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_relative_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_relative_import_reverse. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_multiple_dots. Retrieved 2/6 statements.
# Partially parsed test_section_key_empty_config_defaults. Retrieved 3/9 statements.


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
    var_0 = False
    var_1 = 'import MyModule'
    var_2 = 'B'
    var_3 = 'mymodule'

def test_case_0():
    var_0 = False
    var_1 = 'from os import Path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'
    var_3 = 'os'

def test_case_0():
    var_0 = 'from . import module'
    var_1 = 'B'

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
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = 'from ... import module'
    var_1 = 'B'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'import os'
    var_2 = 'B'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = len(var_0)
    var_3 = str(var_2)
    var_4 = 'import os'
    var_5 = f'B{len(var_0)}import os'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_section_key_lexicographical_predicate_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_relative_import_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_no_sub_imports. Retrieved 3/7 statements.
# Partially parsed test_module_key_empty_module_name. Retrieved 1/5 statements.
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
    var_2 = 'mymodule'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = 'MyModule'

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
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'sys'
    var_2 = '3:sys'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = '2:os'

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
    var_1 = '..module'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = False
    var_3 = 'module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = False
    var_1 = '....package.module'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_removes_from_keyword. Retrieved 3/8 statements.
# Partially parsed test_section_key_removes_import_keyword. Retrieved 6/10 statements.
# Partially parsed test_section_key_lexicographical_mode. Retrieved 3/7 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_reverse_relative_with_relative_import. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_dots_relative_import. Retrieved 3/7 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 6/12 statements.
# Partially parsed test_section_key_empty_line_handling. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort_with_long_line. Retrieved 6/13 statements.


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
    var_1 = 'B'
    var_2 = 'import'
    var_3 = 1
    var_4 = result.split(var_1)[var_3]
    var_5 = var_2 not in var_4

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

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
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import module'
    var_2 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = 'import unittest'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from very_long_module_name import very_long_function_name'
    var_2 = 0
    var_3 = 'B'
    var_4 = var_3.split(var_3)[var_2]
    var_5 = 2



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_sort_relative_in_force_sorted_sections_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_lexicographical_predicate_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 5/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'os'
    var_2 = 0
    var_3 = ' '
    var_4 = var_1.split(var_3)[var_2]



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_relative_import_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_space. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_underscore. Retrieved 2/6 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/7 statements.
# Partially parsed test_section_key_no_length_sort. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_order_by_type_different. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_order_by_type_same. Retrieved 2/6 statements.
# Partially parsed test_section_key_not_order_by_type. Retrieved 2/6 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_relative_dots. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_with_split_module. Retrieved 3/7 statements.
# Partially parsed test_section_key_honor_case_without_import_keyword. Retrieved 3/7 statements.
# Partially parsed test_section_key_empty_config_defaults. Retrieved 2/7 statements.


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
    var_2 = 'from . import module'
    var_3 = '.'

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
    var_1 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = 'from os import Path'

def test_case_0():
    var_0 = False
    var_1 = 'from os import Path'
    var_2 = 'path'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from ... import module'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'MyModule'

def test_case_0():
    var_0 = 'import unittest'
    var_1 = 'B'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_relative_and_spaces. Retrieved 2/6 statements.
# Partially parsed test_module_key_empty_config. Retrieved 1/5 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/5 statements.


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
    var_1 = 'module'
    var_2 = ':'

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
    var_1 = '. . module'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = True
    var_1 = '....package.module'
    var_2 = 'package'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_line_23_predicate_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'os'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'TestModule'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 24/47 statements.


import re as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'test_module'
    var_3 = True
    var_4 = False
    var_5 = 'THIRDPARTY'
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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_false. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = 'thirdparty'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort. Retrieved 2/8 statements.
# Partially parsed test_section_key_length_sort_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_order_by_type. Retrieved 2/5 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_import_statement. Retrieved 2/5 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 2/5 statements.


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
    var_2 = 'os'

def test_case_0():
    var_0 = False
    var_1 = 'from os import path'
    var_2 = 'path'

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = False
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'import Os'
    var_2 = 'Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = '.'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'from .. import module'
    var_3 = 'B'

def test_case_0():
    var_0 = False
    var_1 = 'import numpy'
    var_2 = 'numpy'

def test_case_0():
    var_0 = False
    var_1 = 'from numpy import array'
    var_2 = 'numpy'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_section_key_predicate_line_4_evaluates_to_false. Retrieved 6/15 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'
    var_2 = False
    var_3 = 'from . import something'
    var_4 = 'import something'
    var_5 = 'import something'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_section_key_predicate_at_line_4_evaluates_to_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from . import something'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_section_key_default_section_b. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_with_length_sort. Retrieved 2/6 statements.
# Partially parsed test_section_key_without_length_sort. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 1/4 statements.
# Partially parsed test_section_key_force_to_top_multiple_modules. Retrieved 6/12 statements.
# Partially parsed test_section_key_no_from_prefix_removal. Retrieved 2/5 statements.
# Partially parsed test_section_key_import_line_processing. Retrieved 3/6 statements.


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
    var_2 = 'B'

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
    var_0 = 'from .. import something'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import os'
    var_4 = 'import sys'
    var_5 = 'A'

def test_case_0():
    var_0 = False
    var_1 = 'import collections'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import sys'
    var_3 = 'sys'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'from package import something'
    var_1 = 'from'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = str(var_3)



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 5/50 statements.


import re as module_0

def test_case_0():
    var_0 = '^from |^import '
    var_1 = module_0.compile(var_0)
    var_2 = ' import '
    var_3 = module_0.compile(var_2)
    var_4 = 'from ..module import name'
    var_5 = '.._module'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_lexicographical_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_lexicographical_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_true. Retrieved 2/7 statements.
# Partially parsed test_section_key_length_sort_false. Retrieved 4/11 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_reverse_relative_true. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_reverse. Retrieved 2/6 statements.
# Partially parsed test_section_key_import_statement. Retrieved 1/4 statements.
# Partially parsed test_section_key_from_import_statement. Retrieved 1/5 statements.
# Partially parsed test_section_key_combined_options. Retrieved 6/12 statements.


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
    var_2 = 'os'

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
    var_2 = 'B'
    var_3 = 1

def test_case_0():
    var_0 = False
    var_1 = 'import OS'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = '.'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'

def test_case_0():
    var_0 = True
    var_1 = 'import OS'
    var_2 = 'OS'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from OS import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'

def test_case_0():
    var_0 = True
    var_1 = 'from .. import module'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'sys'

def test_case_0():
    var_0 = 'from sys import path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'import sys'
    var_5 = 'A'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = None
    var_3 = len(var_0)
    var_4 = str(var_3)
    var_5 = ':'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_0



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_section_key_returns_correct_format. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_section_key_predicate_line_43_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/4 statements.
# Partially parsed test_module_key_with_relative_import. Retrieved 1/4 statements.
# Partially parsed test_module_key_with_reverse_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_constants. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_classes. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_variables. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 1/4 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 1/4 statements.
# Partially parsed test_module_key_order_by_type_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = '..utils'
    var_1 = 'utils'

def test_case_0():
    var_0 = True
    var_1 = '..utils'
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
    var_2 = 'MyModule'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = '2:os'

def test_case_0():
    var_0 = True
    var_1 = 'module'
    var_2 = '6:module'

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
    var_1 = 'ClassName'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable_name'
    var_2 = 'C'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = '...package.module'

def test_case_0():
    var_0 = '.module'
    var_1 = 'module'

def test_case_0():
    var_0 = False
    var_1 = 'MyModule'
    var_2 = True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = '.module'
    var_1 = '_'
    var_2 = 'B_'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 3/8 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_with_spaces. Retrieved 2/6 statements.


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
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

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
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'A'

def test_case_0():
    var_0 = 'os'
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'CONST_VAR'
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
    var_1 = '. module'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_section_key_predicate_line_12_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = 'os'
    var_3 = 'A'
    var_4 = "Section should be 'A' when line.split(' ')[0] is in config.force_to_top"



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_length_sort_predicate_true. Retrieved 9/26 statements.


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



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_section_key_basic_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 3/9 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/7 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 4/9 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 2/7 statements.
# Partially parsed test_section_key_multiple_relative_dots. Retrieved 4/8 statements.
# Partially parsed test_section_key_force_to_top_multiple. Retrieved 5/9 statements.
# Partially parsed test_section_key_length_sort_with_long_line. Retrieved 3/9 statements.
# Partially parsed test_section_key_no_modifications. Retrieved 3/7 statements.


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
    var_0 = False
    var_1 = 'from Module import Something'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'from . import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path, sep'
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
    var_2 = 'from Module import Name'
    var_3 = 'module'

def test_case_0():
    var_0 = False
    var_1 = 'from Module import Name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from ... import module'
    var_3 = 'B'

def test_case_0():
    var_0 = '__future__'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'from sys import argv'
    var_4 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'from very_long_module_name import very_long_function_name'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 'os'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = str(var_3)



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort. Retrieved 5/8 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/6 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_reverse_relative_not_force_sorted. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_with_reverse. Retrieved 2/5 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_without_reverse. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_case_insensitive. Retrieved 3/7 statements.
# Partially parsed test_section_key_simple_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_multiple_force_to_top. Retrieved 5/9 statements.


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
    var_0 = False
    var_1 = 'import Os'
    var_2 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'from package import module'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from package import module'
    var_2 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
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
    var_2 = 'from Package import Name'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Package import Name'
    var_3 = 'package'

def test_case_0():
    var_0 = 'import sys'
    var_1 = 'B'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = 'os'
    var_1 = 'sys'
    var_2 = [var_0, var_1]
    var_3 = 'import sys'
    var_4 = 'A'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 3/8 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 2/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 2/5 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/7 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 3/6 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/6 statements.
# Partially parsed test_section_key_honor_case_with_import. Retrieved 3/7 statements.
# Partially parsed test_section_key_multiple_spaces. Retrieved 2/6 statements.
# Partially parsed test_section_key_from_import. Retrieved 2/6 statements.
# Partially parsed test_section_key_empty_line. Retrieved 2/6 statements.
# Partially parsed test_section_key_with_relative_import. Retrieved 2/5 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'import os'

def test_case_0():
    var_0 = True
    var_1 = 'from . import foo'
    var_2 = '.'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'
    var_3 = 'os'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import OS'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import foo'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from .. import bar'

def test_case_0():
    var_0 = True
    var_1 = 'from os import path'
    var_2 = 'import'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Module import Name'
    var_3 = 'module'

def test_case_0():
    var_0 = 'import  os'
    var_1 = 'B'

def test_case_0():
    var_0 = 'from os import path'
    var_1 = 'B'

def test_case_0():
    var_0 = ''
    var_1 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'from ... import something'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 7/54 statements.


import re as module_0

def test_case_0():
    var_0 = '^from|^import'
    var_1 = module_0.compile(var_0)
    var_2 = ' import '
    var_3 = module_0.compile(var_2)
    var_4 = 'import os'
    var_5 = 'A'
    var_6 = 0
    var_7 = 'os'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_case_sensitive_predicate_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'TestModule'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_sub_imports_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_capitalized_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_lowercase_variable. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 4/7 statements.
# Partially parsed test_module_key_complex_relative_import. Retrieved 2/6 statements.
# Partially parsed test_module_key_single_char_module. Retrieved 1/4 statements.


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
    var_1 = 'os'
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
    var_1 = 'my_var'
    var_2 = [var_1]
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'CONSTANT_NAME'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'
    var_2 = 'C'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = 'future'
    var_4 = ':'

def test_case_0():
    var_0 = False
    var_1 = '..package.module'

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_length_sort_predicate_evaluates_to_false. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = None
    var_3 = str(var_2)
    var_4 = len(var_0)
    var_5 = str(var_4)
    var_6 = ':'
    var_7 = var_5 + var_6
    var_8 = var_7 + var_0



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = False
    var_2 = None
    var_3 = 'A'
    var_4 = 'os'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_section_key_line_23_predicate_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = 0
    var_4 = ' '
    var_5 = var_2.split(var_4)[var_3]



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_module_key_length_sort_predicate_false. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = False
    var_2 = False
    var_3 = None
    var_4 = False
    var_5 = str(var_3)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_module_key_force_to_top_predicate. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = [var_0]
    var_2 = 'A'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_lexicographical_predicate_false. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'
    var_2 = 'B'



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports. Retrieved 1/5 statements.
# Partially parsed test_module_key_with_relative_imports_reverse. Retrieved 2/6 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_with_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_in_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_with_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_with_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_without_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_with_section_in_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_mixed_case_class. Retrieved 2/5 statements.
# Partially parsed test_module_key_lowercase_variable. Retrieved 2/5 statements.


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
    var_0 = False
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
    var_1 = 'module'
    var_2 = '6:module'

def test_case_0():
    var_0 = False
    var_1 = 'module'
    var_2 = 'module'

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
    var_0 = True
    var_1 = 'CONSTANT'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'my_variable'
    var_2 = 'C'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_section_key_line_29_predicate_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'os import path'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_line_29_predicate_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_module_key_predicate_line_20. Retrieved 4/19 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = False
    var_3 = None



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_module_key_basic. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_no_length_sort. Retrieved 3/6 statements.
# Partially parsed test_module_key_relative_with_space. Retrieved 2/6 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 3/7 statements.
# Partially parsed test_module_key_combined_options. Retrieved 4/8 statements.


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
    var_3 = 'C'

def test_case_0():
    var_0 = True
    var_1 = 'UPPERCASE'
    var_2 = 'A'

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
    var_0 = True
    var_1 = 'module'
    var_2 = ':'

def test_case_0():
    var_0 = 'stdlib'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'module'
    var_3 = ':'

def test_case_0():
    var_0 = True
    var_1 = '..module'

def test_case_0():
    var_0 = True
    var_1 = 'MyModule'
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'special'
    var_3 = [var_2]
    var_4 = ':'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'from . import something'
    var_1 = 'from .'



# Parsed testcases at query #108
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
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/7 statements.
# Partially parsed test_module_key_order_by_type_uppercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_class_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_multiple_dots_relative. Retrieved 2/6 statements.
# Partially parsed test_module_key_single_dot_relative. Retrieved 2/5 statements.
# Partially parsed test_module_key_sub_imports_false. Retrieved 5/9 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.


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
    var_1 = 'os'
    var_2 = ':'

def test_case_0():
    var_0 = False
    var_1 = 'os'

def test_case_0():
    var_0 = True
    var_1 = 'os'
    var_2 = ':'

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
    var_1 = 'UPPERCASE'
    var_2 = 'A'

def test_case_0():
    var_0 = True
    var_1 = 'MyClass'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'lowercase'
    var_2 = 'C'

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = [var_0]
    var_2 = 'os'
    var_3 = ':'

def test_case_0():
    var_0 = False
    var_1 = '....deep.module'

def test_case_0():
    var_0 = False
    var_1 = '.module'
    var_2 = 'module'

def test_case_0():
    var_0 = True
    var_1 = 'CONST'
    var_2 = [var_1]
    var_3 = False
    var_4 = 'B'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'sys'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = ':'



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_predicate_at_line_20_evaluates_to_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_module'



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_lexicographical_predicate_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'import os'



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_section_key_default_section. Retrieved 2/6 statements.
# Partially parsed test_section_key_force_to_top. Retrieved 4/8 statements.
# Partially parsed test_section_key_length_sort_disabled. Retrieved 2/5 statements.
# Partially parsed test_section_key_length_sort_enabled. Retrieved 3/7 statements.
# Partially parsed test_section_key_lexicographical. Retrieved 3/6 statements.
# Partially parsed test_section_key_reverse_relative. Retrieved 4/8 statements.
# Partially parsed test_section_key_group_by_package. Retrieved 3/6 statements.
# Partially parsed test_section_key_case_sensitive_false. Retrieved 3/6 statements.
# Partially parsed test_section_key_order_by_type_false. Retrieved 2/5 statements.
# Partially parsed test_section_key_honor_case_in_force_sorted_sections. Retrieved 3/7 statements.
# Partially parsed test_section_key_sort_relative_in_force_sorted_sections. Retrieved 4/8 statements.
# Partially parsed test_section_key_multiple_dots_relative. Retrieved 4/8 statements.


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
    var_2 = 'B9'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'
    var_3 = 'os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from . import module'
    var_3 = 'B'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from os import path'
    var_3 = 'import'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'import Os'

def test_case_0():
    var_0 = False
    var_1 = 'import Os'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'from Os import Path'
    var_3 = 'os'

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



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_module_key_simple_module. Retrieved 1/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_relative_import_reverse_relative_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_ignore_case. Retrieved 2/6 statements.
# Partially parsed test_module_key_case_sensitive_false. Retrieved 2/5 statements.
# Partially parsed test_module_key_case_sensitive_true. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_straight. Retrieved 2/5 statements.
# Partially parsed test_module_key_length_sort_sections. Retrieved 3/6 statements.
# Partially parsed test_module_key_force_to_top. Retrieved 3/7 statements.
# Partially parsed test_module_key_not_force_to_top. Retrieved 2/6 statements.
# Partially parsed test_module_key_order_by_type_constant. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_class. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_variable. Retrieved 3/6 statements.
# Partially parsed test_module_key_order_by_type_uppercase_constant. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_capitalized. Retrieved 2/5 statements.
# Partially parsed test_module_key_order_by_type_lowercase. Retrieved 2/5 statements.
# Partially parsed test_module_key_combined_options. Retrieved 5/9 statements.
# Partially parsed test_module_key_relative_with_spaces. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'

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
    var_1 = 'MyModule'
    var_2 = 'MyModule'

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
    var_1 = 'Class'
    var_2 = 'B'

def test_case_0():
    var_0 = True
    var_1 = 'variable'
    var_2 = 'C'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'django'
    var_3 = [var_2]
    var_4 = 'A'
    var_5 = ':'

def test_case_0():
    var_0 = True
    var_1 = '.. module_name'
    var_2 = 'module_name'



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_module_key_relative_import_with_reverse_relative. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '. utils'
    var_1 = ' '



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_section_key_predicate_line_43. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'B'
    var_2 = 'import os'
    var_3 = len(var_2)
    var_4 = 'import os'
    var_5 = f'{var_1}{var_3}{var_4}'



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_section_key_length_sort_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 'import os'
    var_2 = 'B'
    var_3 = 1
    var_4 = result.split(var_2)[var_3]
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



# Parsed testcases at query #116
#--------------------------

# Partially parsed test_section_key_sort_relative_in_force_sorted_sections_true. Retrieved 16/42 statements.


import re as module_0

def test_case_0():
    var_0 = '.module'
    var_1 = 'B'
    var_2 = '^from (\\.+)\\s*(.*)'
    var_3 = module_0.match(var_2, var_0)
    var_4 = ' '
    var_5 = 0
    var_6 = ' import '
    var_7 = 1
    var_8 = var_0.split(var_6, var_7)[var_5]
    var_9 = ''
    var_10 = '.'
    var_11 = '^from '
    var_12 = ''
    var_13 = module_0.sub(var_11, var_12, var_8)
    var_14 = '^import '
    var_15 = module_0.sub(var_14, var_12, var_13)



# Parsed testcases at query #117
#--------------------------

# Partially parsed test_force_to_top_predicate_evaluates_to_false. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = False
    var_2 = None
    var_3 = 'B'



