####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_unused_items_all_items_unused. Retrieved 3/10 statements.
# Partially parsed test_get_unused_items_some_items_unused. Retrieved 4/11 statements.
# Partially parsed test_get_unused_items_all_items_used. Retrieved 3/7 statements.
# Partially parsed test_get_unused_items_case_insensitive_sort. Retrieved 4/13 statements.
# Partially parsed test_get_unused_items_no_used_names. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'b'
    var_1 = 'a'
    var_2 = set()

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = {var_0, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0, var_1}

def test_case_0():
    var_0 = 'B'
    var_1 = 'a'
    var_2 = 'C'
    var_3 = set()

import vulture.core as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1, var_2}
    var_4 = module_0._get_unused_items(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'z'
    var_1 = 'y'
    var_2 = set()



# Parsed testcases at query #2
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\nprint(y)\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.get_unused_code()
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].name
    assert var_5 == 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\ndef f():\n    pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = 100
    var_4 = var_0.get_unused_code(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'b = 1\na = 2\nprint(c)\n'
    var_2 = var_0.scan(var_1)
    var_3 = False
    var_4 = var_0.get_unused_code(sort_by_size=var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_4[0].name
    assert var_6 == 'a'
    var_7 = var_4[1].name
    assert var_7 == 'b'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a = 1\nb = 2\nprint(c)\n'
    var_2 = var_0.scan(var_1)
    var_3 = True
    var_4 = var_0.get_unused_code(sort_by_size=var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_4[0].size
    var_7 = bool(var_4[0].size <= var_4[1].size)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 101
    var_2 = var_0.get_unused_code(var_1)
    var_3 = bool(False)
    assert var_3 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def f():\n    return\n    x = 1\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.get_unused_code()
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0].typ
    assert var_5 == 'unreachable_code'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\nprint(x)\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.get_unused_code()
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_visit_function_def_property_decorator. Retrieved 16/19 statements.
# Partially parsed test_visit_function_def_staticmethod_decorator. Retrieved 16/19 statements.
# Partially parsed test_visit_function_def_classmethod_decorator. Retrieved 18/21 statements.
# Partially parsed test_visit_function_def_method_with_self. Retrieved 15/18 statements.
# Partially parsed test_visit_function_def_regular_function. Retrieved 13/16 statements.
# Partially parsed test_visit_function_def_ignore_decorator. Retrieved 21/24 statements.
# Partially parsed test_visit_function_def_no_decorator_no_self. Retrieved 15/18 statements.


import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_prop'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = 'property'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = None
    var_23 = var_0.defined_props
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = var_0.defined_props[0].name
    assert var_25 == 'my_prop'
    var_26 = var_0.defined_props[0].typ
    assert var_26 == 'property'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'static_method'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = 'staticmethod'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = None
    var_23 = var_0.defined_methods
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = var_0.defined_methods[0].name
    assert var_25 == 'static_method'
    var_26 = var_0.defined_methods[0].typ
    assert var_26 == 'method'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class_method'
    var_2 = []
    var_3 = 'cls'
    var_4 = None
    var_5 = []
    var_6 = 'arg'
    var_7 = 'annotation'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.arg(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = {}
    var_17 = module_1.Pass(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = 'classmethod'
    var_20 = []
    var_21 = {}
    var_22 = module_1.Load(*var_20, **var_21)
    var_23 = []
    var_24 = 'id'
    var_25 = 'ctx'
    var_26 = {var_24: var_19, var_25: var_22}
    var_27 = module_1.Name(*var_23, **var_26)
    var_28 = [var_27]
    var_29 = var_0.defined_methods
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = var_0.defined_methods[0].name
    assert var_31 == 'class_method'
    var_32 = var_0.defined_methods[0].typ
    assert var_32 == 'method'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'instance_method'
    var_2 = []
    var_3 = 'self'
    var_4 = None
    var_5 = []
    var_6 = 'arg'
    var_7 = 'annotation'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.arg(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = {}
    var_17 = module_1.Pass(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = []
    var_20 = var_0.defined_methods
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = var_0.defined_methods[0].name
    assert var_22 == 'instance_method'
    var_23 = var_0.defined_methods[0].typ
    assert var_23 == 'method'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'regular_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = None
    var_14 = var_0.defined_funcs
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = var_0.defined_funcs[0].name
    assert var_16 == 'regular_func'
    var_17 = var_0.defined_funcs[0].typ
    assert var_17 == 'function'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'ignored_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'ignored_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = {}
    var_12 = module_1.Pass(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'id'
    var_19 = 'ctx'
    var_20 = {var_18: var_0, var_19: var_16}
    var_21 = module_1.Name(*var_17, **var_20)
    var_22 = [var_21]
    var_23 = None
    var_24 = var_2.defined_funcs
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = var_2.defined_methods
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = var_2.defined_props
    var_29 = len(var_28)
    assert var_29 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'top_level_func'
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = []
    var_6 = 'arg'
    var_7 = 'annotation'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.arg(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = {}
    var_17 = module_1.Pass(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = []
    var_20 = var_0.defined_funcs
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = var_0.defined_funcs[0].name
    assert var_22 == 'top_level_func'
    var_23 = var_0.defined_funcs[0].typ
    assert var_23 == 'function'



# Parsed testcases at query #4
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '_'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '_private'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '__name__'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '__name'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = 'normal_var'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '__x'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '_'
    var_2 = module_0._ignore_variable(var_0, var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = ''
    var_2 = module_0._ignore_variable(var_0, var_1)



# Parsed testcases at query #5
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '@property\ndef foo(self):\n    pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_FunctionDef(var_4)
    var_6 = var_0.defined_props
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_props[0].name
    assert var_8 == 'foo'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'def bar(self):\n    pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_FunctionDef(var_4)
    var_6 = var_0.defined_methods
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_methods[0].name
    assert var_8 == 'bar'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '@staticmethod\ndef baz():\n    pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_FunctionDef(var_4)
    var_6 = var_0.defined_methods
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_methods[0].name
    assert var_8 == 'baz'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '@classmethod\ndef qux(cls):\n    pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_FunctionDef(var_4)
    var_6 = var_0.defined_methods
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_methods[0].name
    assert var_8 == 'qux'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'def quux():\n    pass'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_FunctionDef(var_4)
    var_6 = var_0.defined_funcs
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_funcs[0].name
    assert var_8 == 'quux'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 0
    var_4 = '@my_decorator\ndef foo():\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_2.visit_FunctionDef(var_6)
    var_8 = var_2.defined_funcs
    var_9 = len(var_8)
    assert var_9 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 0
    var_4 = '@my_decorator\ndef bar(self):\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_2.visit_FunctionDef(var_6)
    var_8 = var_2.defined_methods
    var_9 = len(var_8)
    assert var_9 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 0
    var_4 = '@my_decorator\n@property\ndef baz(self):\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_2.visit_FunctionDef(var_6)
    var_8 = var_2.defined_props
    var_9 = len(var_8)
    assert var_9 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'ignored_func'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 0
    var_4 = 'def ignored_func():\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_2.visit_FunctionDef(var_6)
    var_8 = var_2.defined_funcs
    var_9 = len(var_8)
    assert var_9 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'ignored_method'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 0
    var_4 = 'def ignored_method(self):\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_2.visit_FunctionDef(var_6)
    var_8 = var_2.defined_methods
    var_9 = len(var_8)
    assert var_9 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'ignored_prop'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 0
    var_4 = '@property\ndef ignored_prop(self):\n    pass'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_2.visit_FunctionDef(var_6)
    var_8 = var_2.defined_props
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ignore_function_returns_true_for_pytest_function_in_test_file. Retrieved 4/9 statements.
# Partially parsed test_ignore_function_returns_true_for_test_prefixed_function_in_test_file. Retrieved 4/9 statements.
# Partially parsed test_ignore_function_returns_false_for_non_test_function_in_test_file. Retrieved 4/9 statements.
# Partially parsed test_ignore_function_returns_false_for_pytest_function_in_non_test_file. Retrieved 4/9 statements.
# Partially parsed test_ignore_function_returns_false_for_test_prefixed_function_in_non_test_file. Retrieved 4/9 statements.
# Partially parsed test_ignore_function_returns_false_for_non_test_function_in_non_test_file. Retrieved 4/9 statements.
# Partially parsed test_ignore_function_handles_test_pattern_with_underscore. Retrieved 3/8 statements.
# Partially parsed test_ignore_function_handles_test_pattern_with_hyphen. Retrieved 3/8 statements.
# Partially parsed test_ignore_function_handles_test_suffix_pattern. Retrieved 3/8 statements.
# Partially parsed test_ignore_function_handles_test_suffix_pattern_with_hyphen. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'some'
    var_1 = 'test'
    var_2 = 'file.py'
    var_3 = 'pytest_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'tests'
    var_2 = 'file.py'
    var_3 = 'test_something'

def test_case_0():
    var_0 = 'some'
    var_1 = 'test'
    var_2 = 'file.py'
    var_3 = 'regular_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'src'
    var_2 = 'file.py'
    var_3 = 'pytest_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'src'
    var_2 = 'file.py'
    var_3 = 'test_something'

def test_case_0():
    var_0 = 'some'
    var_1 = 'src'
    var_2 = 'file.py'
    var_3 = 'regular_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'test_example.py'
    var_2 = 'test_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'test-example.py'
    var_2 = 'test_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'example_test.py'
    var_2 = 'test_func'

def test_case_0():
    var_0 = 'some'
    var_1 = 'example-test.py'
    var_2 = 'test_func'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_visit_FunctionDef_regular_function. Retrieved 14/17 statements.
# Partially parsed test_visit_FunctionDef_method_with_self. Retrieved 16/19 statements.
# Partially parsed test_visit_FunctionDef_property. Retrieved 18/21 statements.
# Partially parsed test_visit_FunctionDef_staticmethod. Retrieved 16/19 statements.
# Partially parsed test_visit_FunctionDef_classmethod. Retrieved 18/21 statements.
# Partially parsed test_visit_FunctionDef_ignored_decorator. Retrieved 21/24 statements.
# Partially parsed test_visit_FunctionDef_ignored_name. Retrieved 15/18 statements.
# Partially parsed test_visit_FunctionDef_async_function. Retrieved 14/17 statements.


import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = 1
    var_14 = 3
    var_15 = var_0.defined_funcs
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = var_0.defined_funcs[0].name
    assert var_17 == 'my_func'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_method'
    var_2 = []
    var_3 = 'self'
    var_4 = []
    var_5 = 'arg'
    var_6 = {var_5: var_3}
    var_7 = module_1.arg(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = 1
    var_19 = 3
    var_20 = var_0.defined_methods
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = var_0.defined_methods[0].name
    assert var_22 == 'my_method'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_prop'
    var_2 = []
    var_3 = 'self'
    var_4 = []
    var_5 = 'arg'
    var_6 = {var_5: var_3}
    var_7 = module_1.arg(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = 'property'
    var_18 = []
    var_19 = 'id'
    var_20 = {var_19: var_17}
    var_21 = module_1.Name(*var_18, **var_20)
    var_22 = [var_21]
    var_23 = 1
    var_24 = 3
    var_25 = var_0.defined_props
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_0.defined_props[0].name
    assert var_27 == 'my_prop'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_static'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = 'staticmethod'
    var_13 = []
    var_14 = 'id'
    var_15 = {var_14: var_12}
    var_16 = module_1.Name(*var_13, **var_15)
    var_17 = [var_16]
    var_18 = 1
    var_19 = 3
    var_20 = var_0.defined_methods
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = var_0.defined_methods[0].name
    assert var_22 == 'my_static'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_classmethod'
    var_2 = []
    var_3 = 'cls'
    var_4 = []
    var_5 = 'arg'
    var_6 = {var_5: var_3}
    var_7 = module_1.arg(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = 'classmethod'
    var_18 = []
    var_19 = 'id'
    var_20 = {var_19: var_17}
    var_21 = module_1.Name(*var_18, **var_20)
    var_22 = [var_21]
    var_23 = 1
    var_24 = 3
    var_25 = var_0.defined_methods
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_0.defined_methods[0].name
    assert var_27 == 'my_classmethod'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'my_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = {}
    var_12 = module_1.Pass(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = 'id'
    var_16 = {var_15: var_0}
    var_17 = module_1.Name(*var_14, **var_16)
    var_18 = [var_17]
    var_19 = 1
    var_20 = 3
    var_21 = var_2.defined_funcs
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = var_2.defined_methods
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = var_2.defined_props
    var_26 = len(var_25)
    assert var_26 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'ignored_func'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = {}
    var_11 = module_1.Pass(*var_9, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 1
    var_15 = 3
    var_16 = var_2.defined_funcs
    var_17 = len(var_16)
    assert var_17 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'async_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = 1
    var_14 = 3
    var_15 = var_0.defined_funcs
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = var_0.defined_funcs[0].name
    assert var_17 == 'async_func'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_scavenge_valid_file. Retrieved 3/10 statements.
# Partially parsed test_scavenge_with_whitelist. Retrieved 10/11 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = []
    var_2 = 'test_exclude'
    var_3 = [var_2]
    var_4 = var_0.scavenge(var_1, var_3)
    var_5 = var_0.exit_code
    assert var_5 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'nonexistent_file.py'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)
    var_4 = var_0.exit_code
    assert var_4 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = var_0.scavenge(var_1)
    var_3 = var_0.exit_code
    assert var_3 == 0

import vulture.core as module_0
import builtins as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'ImportItem'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'os'
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}
    var_8 = module_1.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = []
    var_11 = var_0.scavenge(var_10)
    var_12 = var_0.exit_code
    assert var_12 == 0



# Parsed testcases at query #9
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'some_file.py'
    var_1 = '__init__'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'tests/test_example.py'
    var_1 = 'test_something'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'test/test_example.py'
    var_1 = 'test_another'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'tests/test_example.py'
    var_1 = 'regular_method'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is False

import vulture.core as module_0

def test_case_0():
    var_0 = 'src/main.py'
    var_1 = 'test_something'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is False

import vulture.core as module_0

def test_case_0():
    var_0 = 'src/main.py'
    var_1 = '__str__'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'test_example.py'
    var_1 = 'test_case'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'example_test.py'
    var_1 = 'test_runner'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'src/main.py'
    var_1 = 'helper'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is False

import vulture.core as module_0

def test_case_0():
    var_0 = 'tests/test_example.py'
    var_1 = 'setup_method'
    var_2 = module_0._ignore_method(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'x'
    var_6 = var_0.defined_vars[0].first_lineno
    assert var_6 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo(:'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = \x00'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # type: int'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # type: not a type'
    var_2 = var_0.scan(var_1)

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "import os\nos.path.join('a', 'b')"
    var_2 = var_0.scan(var_1)
    var_3 = var_0.used_names
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = 'os'
    var_7 = bool('os' in var_0.used_names)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def my_func():\n    pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_funcs[0].name
    assert var_5 == 'my_func'
    var_6 = var_0.defined_funcs[0].first_lineno
    assert var_6 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class MyClass:\n    pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_classes
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_classes[0].name
    assert var_5 == 'MyClass'
    var_6 = var_0.defined_classes[0].first_lineno
    assert var_6 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = 'ignored_var'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 'ignored_var = 42'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_vars
    var_6 = len(var_5)
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = '@custom_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = '@custom_decorator\ndef my_func():\n    pass'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_funcs
    var_6 = len(var_5)
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = var_0.scan(var_1)
    var_3 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # noqa: V101'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'if True:\n    dead_code = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.unreachable_code
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = var_0.scan(var_1)
    var_3 = 'y = 2'
    var_4 = var_0.scan(var_3)
    var_5 = var_0.defined_vars
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_0.defined_vars[0].name
    assert var_7 == 'y'

import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.filename
    var_5 = str(var_4)
    assert var_5 == 'test.py'
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_0.defined_vars[0].filename
    var_10 = bool(var_0.defined_vars[0].filename == var_8)
    assert var_10 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo(x: int) -> str: pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_funcs[0].name
    assert var_5 == 'foo'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'async def my_async():\n    pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_funcs[0].name
    assert var_5 == 'my_async'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class MyClass:\n    @property\n    def my_prop(self):\n        pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_props
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_props[0].name
    assert var_5 == 'my_prop'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class MyClass:\n    def my_method(self):\n        pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_methods
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_methods[0].name
    assert var_5 == 'my_method'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\ny = 2\nz = x + y'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 3
    var_5 = var_0.used_names
    var_6 = len(var_5)
    var_7 = bool(var_6 >= 2)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def outer():\n    def inner():\n        pass\n    pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = len(var_3)
    assert var_4 == 2

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "from os import path\npath.join('a', 'b')"
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_imports
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_imports[0].name
    assert var_5 == 'path'
    var_6 = 'path'
    var_7 = bool('path' in var_0.used_names)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class MyClass:\n    def __init__(self):\n        self.attr = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_attrs
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_attrs[0].name
    assert var_5 == 'attr'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo():\n    return 1\n    x = 2'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.unreachable_code
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 1)
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.code
    var_4 = bool(var_0.code == ['x = 1'])
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = \x00'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "x = 'café'"
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'match x:\n    case 1:\n        pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    var_5 = 1
    var_6 = var_4 >= var_5

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'if (x := 1):\n    pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'x'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_prepare_pattern_returns_pattern_with_wildcards_when_no_special_chars. Retrieved 5/9 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test'
    var_2 = [var_1]
    var_3 = var_0.defined_vars
    var_4 = '*test*'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_scavenge_exclude_path_true. Retrieved 8/9 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'import'
    var_2 = 'import os'
    var_3 = 'test.py'
    var_4 = var_0.scan(var_2, var_3)
    var_5 = [var_3]
    var_6 = 'test'
    var_7 = var_0.scavenge(var_5, var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_vulture_constructor_defaults. Retrieved 12/31 statements.


import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = var_0.verbose
    assert var_1 is False
    var_2 = var_0.ignore_names
    var_3 = bool(var_0.ignore_names == [])
    assert var_3 is True
    var_4 = var_0.ignore_decorators
    var_5 = bool(var_0.ignore_decorators == [])
    assert var_5 is True
    var_6 = []
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_0.filename
    var_10 = bool(var_0.filename == var_8)
    assert var_10 is True
    var_11 = var_0.code
    var_12 = bool(var_0.code == [])
    assert var_12 is True
    var_13 = var_0.exit_code
    var_14 = var_0.noqa_lines
    var_15 = bool(var_0.noqa_lines == {})
    assert var_15 is True
    var_16 = var_0.defined_attrs
    var_17 = var_0.defined_classes
    var_18 = var_0.defined_funcs
    var_19 = var_0.defined_imports
    var_20 = var_0.defined_methods
    var_21 = var_0.defined_props
    var_22 = var_0.defined_vars
    var_23 = var_0.unreachable_code
    var_24 = var_0.used_names
    var_25 = var_0.reachability

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.verbose
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = module_0.Vulture(ignore_names=var_2)
    var_4 = var_3.ignore_names
    var_5 = bool(var_3.ignore_names == ['foo', 'bar'])
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'decor1'
    var_1 = 'decor2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Vulture(ignore_decorators=var_2)
    var_4 = var_3.ignore_decorators
    var_5 = bool(var_3.ignore_decorators == ['decor1', 'decor2'])
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = var_0.verbose
    assert var_1 is False
    var_2 = var_0.ignore_names
    var_3 = bool(var_0.ignore_names == [])
    assert var_3 is True
    var_4 = var_0.ignore_decorators
    var_5 = bool(var_0.ignore_decorators == [])
    assert var_5 is True
    var_6 = []
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_0.filename
    var_10 = bool(var_0.filename == var_8)
    assert var_10 is True
    var_11 = var_0.code
    var_12 = bool(var_0.code == [])
    assert var_12 is True
    var_13 = var_0.exit_code
    var_14 = var_0.noqa_lines
    var_15 = bool(var_0.noqa_lines == {})
    assert var_15 is True

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = 'foo'
    var_2 = [var_1]
    var_3 = 'bar'
    var_4 = [var_3]
    var_5 = module_0.Vulture(var_0, var_2, var_4)
    var_6 = var_5.verbose
    assert var_6 is True
    var_7 = var_5.ignore_names
    var_8 = bool(var_5.ignore_names == ['foo'])
    assert var_8 is True
    var_9 = var_5.ignore_decorators
    var_10 = bool(var_5.ignore_decorators == ['bar'])
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_list_returns_logging_list_for_property. Retrieved 3/5 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.defined_props



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_prepare_pattern_returns_pattern_with_wildcards_when_no_wildcards_in_input. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'foo'



# Parsed testcases at query #17
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = '/path/to/file.py'
    var_3 = 10
    var_4 = 20
    var_5 = 'custom message'
    var_6 = 75
    var_7 = module_0.Item(var_0, var_1, var_2, var_3, var_4, var_5, var_6)
    var_8 = var_7.name
    assert var_8 == 'my_func'
    var_9 = var_7.typ
    assert var_9 == 'function'
    var_10 = var_7.filename
    assert var_10 == '/path/to/file.py'
    var_11 = var_7.first_lineno
    assert var_11 == 10
    var_12 = var_7.last_lineno
    assert var_12 == 20
    var_13 = var_7.message
    assert var_13 == 'custom message'
    var_14 = var_7.confidence
    assert var_14 == 75

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'variable'
    var_2 = '/path/file.py'
    var_3 = 5
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.message
    assert var_5 == "unused variable 'my_var'"

import vulture.core as module_0

def test_case_0():
    var_0 = 'MyClass'
    var_1 = 'class'
    var_2 = '/path/file.py'
    var_3 = 1
    var_4 = 10
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.confidence

import vulture.core as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'function'
    var_2 = '/path/file.py'
    var_3 = 1
    var_4 = ''
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, var_4)
    var_6 = var_5.message
    assert var_6 == "unused function 'test'"



# Parsed testcases at query #18
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'from os import path'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_ImportFrom(var_4)
    var_6 = var_0.defined_imports
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_imports[0].name
    assert var_8 == 'os'
    var_9 = var_0.defined_imports[0].confidence
    assert var_9 == 90

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'from __future__ import annotations'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_ImportFrom(var_4)
    var_6 = var_0.defined_imports
    var_7 = len(var_6)
    assert var_7 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'from os import path as p'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_ImportFrom(var_4)
    var_6 = var_0.defined_imports
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_imports[0].name
    assert var_8 == 'p'
    var_9 = var_0.defined_imports[0].confidence
    assert var_9 == 90

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'from os import path, getcwd'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_ImportFrom(var_4)
    var_6 = var_0.defined_imports
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_0.defined_imports[0].name
    assert var_8 == 'os'
    var_9 = var_0.defined_imports[1].name
    assert var_9 == 'os'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'from os import path as p'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_ImportFrom(var_4)
    var_6 = 'os'
    var_7 = bool('os' in var_0.used_names)
    assert var_7 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'from os.path import join'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_0.visit_ImportFrom(var_4)
    var_6 = var_0.defined_imports
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.defined_imports[0].name
    assert var_8 == 'os'



# Parsed testcases at query #19
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "getattr(obj, 'some_attr', default)"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "getattr(obj, 'some_attr')"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'getattr(obj)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' not in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "hasattr(obj, 'some_attr')"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'hasattr(obj)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' not in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'getattr(obj, attr_name)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'attr_name'
    var_8 = bool('attr_name' not in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '"{my_var}".format(**locals())'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'my_var'
    var_8 = bool('my_var' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '"{my_var}".format()'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'my_var'
    var_8 = bool('my_var' not in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '"{my_var}".format(**other)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'my_var'
    var_8 = bool('my_var' not in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = '"{}".format(1)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'my_var'
    var_8 = bool('my_var' not in var_0.used_names)
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.verbose
    assert var_2 is False



# Parsed testcases at query #21
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'some_attr'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'attr'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Attribute(*var_5, **var_8)
    var_10 = var_0.visit_Attribute(var_9)
    var_11 = 'some_attr'
    var_12 = bool('some_attr' in var_0.used_names)
    assert var_12 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'some_attr'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Store(*var_2, **var_3)
    var_5 = []
    var_6 = 'attr'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Attribute(*var_5, **var_8)
    var_10 = var_0.visit_Attribute(var_9)
    var_11 = var_0.defined_attrs
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_0.defined_attrs[0].name
    assert var_13 == 'some_attr'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generic_visit_with_list_of_nodes. Retrieved 11/15 statements.
# Partially parsed test_generic_visit_with_single_node. Retrieved 6/7 statements.
# Partially parsed test_generic_visit_with_empty_list. Retrieved 5/6 statements.
# Partially parsed test_generic_visit_with_nested_ast_nodes. Retrieved 9/10 statements.
# Partially parsed test_generic_visit_with_multiple_lists. Retrieved 15/19 statements.
# Partially parsed test_generic_visit_with_attribute_access. Retrieved 9/10 statements.
# Partially parsed test_generic_visit_with_none_value. Retrieved 9/10 statements.
# Partially parsed test_generic_visit_with_complex_node. Retrieved 9/12 statements.
# Partially parsed test_generic_visit_with_subscript. Retrieved 10/11 statements.
# Partially parsed test_generic_visit_with_list_node. Retrieved 10/11 statements.


import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = 'def test(): pass'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 1
    var_2 = []
    var_3 = 'value'
    var_4 = {var_3: var_1}
    var_5 = module_1.Constant(*var_2, **var_4)
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_1.Expr(*var_6, **var_8)
    var_10 = '1'
    var_11 = var_0.generic_visit(var_9)

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = []
    var_2 = []
    var_3 = 'body'
    var_4 = {var_3: var_1}
    var_5 = module_1.Module(*var_2, **var_4)
    var_6 = ''
    var_7 = var_0.generic_visit(var_5)

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 1
    var_2 = []
    var_3 = 'value'
    var_4 = {var_3: var_1}
    var_5 = module_1.Constant(*var_2, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.Add(*var_6, **var_7)
    var_9 = 2
    var_10 = []
    var_11 = 'value'
    var_12 = {var_11: var_9}
    var_13 = module_1.Constant(*var_10, **var_12)
    var_14 = []
    var_15 = 'left'
    var_16 = 'op'
    var_17 = 'right'
    var_18 = {var_15: var_5, var_16: var_8, var_17: var_13}
    var_19 = module_1.BinOp(*var_14, **var_18)
    var_20 = '1 + 2'
    var_21 = var_0.generic_visit(var_19)

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'f'
    var_2 = []
    var_3 = 'x'
    var_4 = []
    var_5 = 'arg'
    var_6 = {var_5: var_3}
    var_7 = module_1.arg(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_3, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = []
    var_22 = 'value'
    var_23 = {var_22: var_20}
    var_24 = module_1.Return(*var_21, **var_23)
    var_25 = [var_24]
    var_26 = []
    var_27 = 'def f(x): return x'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'obj'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 'attr'
    var_11 = []
    var_12 = {}
    var_13 = module_1.Load(*var_11, **var_12)
    var_14 = []
    var_15 = 'value'
    var_16 = 'attr'
    var_17 = 'ctx'
    var_18 = {var_15: var_9, var_16: var_10, var_17: var_13}
    var_19 = module_1.Attribute(*var_14, **var_18)
    var_20 = 'obj.attr'
    var_21 = var_0.generic_visit(var_19)

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = True
    var_2 = []
    var_3 = 'value'
    var_4 = {var_3: var_1}
    var_5 = module_1.Constant(*var_2, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.Pass(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = 'test'
    var_13 = 'body'
    var_14 = 'orelse'
    var_15 = {var_12: var_5, var_13: var_9, var_14: var_10}
    var_16 = module_1.If(*var_11, **var_15)
    var_17 = 'if True: pass'
    var_18 = var_0.generic_visit(var_16)

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'func'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 1
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_10}
    var_14 = module_1.Constant(*var_11, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = []
    var_18 = 'func(1)'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'lst'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = 0
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_10}
    var_14 = module_1.Constant(*var_11, **var_13)
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'value'
    var_20 = 'slice'
    var_21 = 'ctx'
    var_22 = {var_19: var_9, var_20: var_14, var_21: var_17}
    var_23 = module_1.Subscript(*var_18, **var_22)
    var_24 = 'lst[0]'
    var_25 = var_0.generic_visit(var_23)

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 1
    var_2 = []
    var_3 = 'value'
    var_4 = {var_3: var_1}
    var_5 = module_1.Constant(*var_2, **var_4)
    var_6 = 2
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_1.Constant(*var_7, **var_9)
    var_11 = [var_5, var_10]
    var_12 = []
    var_13 = {}
    var_14 = module_1.Load(*var_12, **var_13)
    var_15 = []
    var_16 = 'elts'
    var_17 = 'ctx'
    var_18 = {var_16: var_11, var_17: var_14}
    var_19 = module_1.List(*var_15, **var_18)
    var_20 = '[1, 2]'
    var_21 = var_0.generic_visit(var_19)



# Parsed testcases at query #23
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Vulture(var_0)
    var_2 = '.'
    var_3 = [var_2]
    var_4 = '*.py'
    var_5 = [var_4]
    var_6 = var_1.scavenge(var_3, var_5)
    var_7 = var_1.exit_code



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_vulture_constructor_defaults. Retrieved 13/31 statements.


import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = var_0.verbose
    assert var_1 is False
    var_2 = var_0.ignore_names
    var_3 = bool(var_0.ignore_names == [])
    assert var_3 is True
    var_4 = var_0.ignore_decorators
    var_5 = bool(var_0.ignore_decorators == [])
    assert var_5 is True
    var_6 = []
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_0.filename
    var_10 = bool(var_0.filename == var_8)
    assert var_10 is True
    var_11 = var_0.code
    var_12 = bool(var_0.code == [])
    assert var_12 is True
    var_13 = var_0.exit_code
    var_14 = var_0.noqa_lines
    var_15 = bool(var_0.noqa_lines == {})
    assert var_15 is True
    var_16 = var_0.defined_attrs
    var_17 = var_0.defined_attrs.typ
    assert var_17 == 'attribute'
    var_18 = var_0.defined_classes
    var_19 = var_0.defined_classes.typ
    assert var_19 == 'class'
    var_20 = var_0.defined_funcs
    var_21 = var_0.defined_funcs.typ
    assert var_21 == 'function'
    var_22 = var_0.defined_imports
    var_23 = var_0.defined_imports.typ
    assert var_23 == 'import'
    var_24 = var_0.defined_methods
    var_25 = var_0.defined_methods.typ
    assert var_25 == 'method'
    var_26 = var_0.defined_props
    var_27 = var_0.defined_props.typ
    assert var_27 == 'property'
    var_28 = var_0.defined_vars
    var_29 = var_0.defined_vars.typ
    assert var_29 == 'variable'
    var_30 = var_0.unreachable_code
    var_31 = var_0.unreachable_code.typ
    assert var_31 == 'unreachable_code'
    var_32 = var_0.used_names
    var_33 = var_0.used_names.typ
    assert var_33 == 'name'
    var_34 = 'reachability'
    var_35 = hasattr(var_0, var_34)
    var_36 = bool(var_35)
    assert var_36 is True

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.verbose
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = module_0.Vulture(ignore_names=var_2)
    var_4 = var_3.ignore_names
    var_5 = bool(var_3.ignore_names == ['foo', 'bar'])
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = '@deco'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = var_2.ignore_decorators
    var_4 = bool(var_2.ignore_decorators == ['@deco'])
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = 'x'
    var_2 = [var_1]
    var_3 = '@y'
    var_4 = [var_3]
    var_5 = module_0.Vulture(var_0, var_2, var_4)
    var_6 = var_5.verbose
    assert var_6 is True
    var_7 = var_5.ignore_names
    var_8 = bool(var_5.ignore_names == ['x'])
    assert var_8 is True
    var_9 = var_5.ignore_decorators
    var_10 = bool(var_5.ignore_decorators == ['@y'])
    assert var_10 is True



# Parsed testcases at query #25
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'variable'
    var_2 = 'file.py'
    var_3 = 1
    var_4 = 'custom message'
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, var_4)
    var_6 = var_5.message
    assert var_6 == 'custom message'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_unused_code_returns_list. Retrieved 5/6 statements.
# Partially parsed test_get_unused_code_returns_item_objects. Retrieved 10/20 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo():\n    pass\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 0
    var_5 = var_0.get_unused_code(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[0].name
    assert var_7 == 'foo'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo():\n    pass\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 100
    var_5 = var_0.get_unused_code(var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 0
    var_5 = var_0.get_unused_code(var_4)
    var_6 = 100
    var_7 = var_0.get_unused_code(var_6)
    var_8 = len(var_5)
    assert var_8 == 1
    var_9 = len(var_7)
    assert var_9 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo():\n    pass\n\ndef bar():\n    pass\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = False
    var_5 = var_0.get_unused_code(sort_by_size=var_4)
    var_6 = var_5[0].name
    assert var_6 == 'bar'
    var_7 = var_5[1].name
    assert var_7 == 'foo'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def foo():\n    pass\n\ndef bar():\n    pass\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = True
    var_5 = var_0.get_unused_code(sort_by_size=var_4)
    var_6 = var_5[0].size
    var_7 = bool(var_5[0].size <= var_5[1].size)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = 'name'
    var_6 = 'filename'
    var_7 = 'first_lineno'
    var_8 = 'size'
    var_9 = 'confidence'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = bool(var_4 == [])
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = -1
    var_5 = var_0.get_unused_code(var_4)
    var_6 = bool(False)
    assert var_6 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 101
    var_5 = var_0.get_unused_code(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------




import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'test.py'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'pytest'
    var_7 = module_1._ignore_function(var_5, var_6)
    assert var_7 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'test.py'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'test_myfunc'
    var_7 = module_1._ignore_function(var_5, var_6)
    assert var_7 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'test.py'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'myfunc'
    var_7 = module_1._ignore_function(var_5, var_6)
    assert var_7 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = '.py'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'pytest'
    var_7 = module_1._ignore_function(var_5, var_6)
    assert var_7 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = '.py'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'test_myfunc'
    var_7 = module_1._ignore_function(var_5, var_6)
    assert var_7 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = '.py'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'myfunc'
    var_7 = module_1._ignore_function(var_5, var_6)
    assert var_7 is False



# Parsed testcases at query #3
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = 'path/to/file.py'
    var_3 = 10
    var_4 = 20
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.get_report()
    assert var_6 == "path/to/file.py:10: unused function 'my_func' (100% confidence)"

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = 'path/to/file.py'
    var_3 = 10
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = True
    var_6 = var_4.get_report(var_5)
    assert var_6 == "path/to/file.py:10: unused function 'my_func' (100% confidence, 1 line)"

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = 'path/to/file.py'
    var_3 = 10
    var_4 = 20
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = True
    var_7 = var_5.get_report(var_6)
    assert var_7 == "path/to/file.py:10: unused function 'my_func' (100% confidence, 11 lines)"

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'variable'
    var_2 = 'path/to/file.py'
    var_3 = 5
    var_4 = 'custom message'
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, var_4)
    var_6 = var_5.get_report()
    assert var_6 == 'path/to/file.py:5: custom message (100% confidence)'

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'variable'
    var_2 = 'path/to/file.py'
    var_3 = 5
    var_4 = 75
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, confidence=var_4)
    var_6 = var_5.get_report()
    assert var_6 == "path/to/file.py:5: unused variable 'my_var' (75% confidence)"



# Parsed testcases at query #4
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = 'empty.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.code
    var_5 = bool(var_0.code == [])
    assert var_5 is True
    var_6 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.code
    var_5 = bool(var_0.code == ['x = 1'])
    assert var_5 is True
    var_6 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = '
    var_2 = 'syntax_error.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = \x00'
    var_2 = 'null_bytes.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # noqa\n'
    var_2 = 'noqa.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.code
    var_5 = bool(var_0.code == ['x = 1  # noqa'])
    assert var_5 is True
    var_6 = var_0.exit_code

import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'y = 2\n'
    var_2 = 'custom.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = [var_2]
    var_5 = {}
    var_6 = module_1.Path(*var_4, **var_5)
    var_7 = var_0.filename
    var_8 = bool(var_0.filename == var_6)
    assert var_8 is True
    var_9 = var_0.code
    var_10 = bool(var_0.code == ['y = 2'])
    assert var_10 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a = 1\nb = 2\n'
    var_2 = 'multi.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.code
    var_5 = bool(var_0.code == ['a = 1', 'b = 2'])
    assert var_5 is True
    var_6 = var_0.defined_vars
    var_7 = len(var_6)
    assert var_7 == 2

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def f():\n    pass\n'
    var_2 = 'type_comment.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\n'
    var_2 = 'first.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 'y = 2\n'
    var_5 = 'second.py'
    var_6 = var_0.scan(var_4, var_5)
    var_7 = var_0.exit_code

import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'z = 3\n'
    var_2 = ''
    var_3 = var_0.scan(var_1, var_2)
    var_4 = [var_2]
    var_5 = {}
    var_6 = module_1.Path(*var_4, **var_5)
    var_7 = var_0.filename
    var_8 = bool(var_0.filename == var_6)
    assert var_8 is True
    var_9 = var_0.code
    var_10 = bool(var_0.code == ['z = 3'])
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_scan_invalid_utf8. Retrieved 5/7 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = '
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\x00'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # type: '
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a = 1\nb = 2'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # noqa'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = 'x = 1'
    var_3 = 'test.py'
    var_4 = var_1.scan(var_2, var_3)
    var_5 = var_1.exit_code
    assert var_5 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 'x = 1'
    var_4 = 'test.py'
    var_5 = var_2.scan(var_3, var_4)
    var_6 = var_2.exit_code
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = '@decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'def func(): pass'
    var_4 = 'test.py'
    var_5 = var_2.scan(var_3, var_4)
    var_6 = var_2.exit_code
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 'test2.py'
    var_5 = var_0.scan(var_1, var_4)
    var_6 = var_0.reachability
    var_7 = bool(var_0.reachability is not None)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.code
    var_5 = bool(var_0.code == ['x = 1'])
    assert var_5 is True

import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = [var_2]
    var_5 = {}
    var_6 = module_1.Path(*var_4, **var_5)
    var_7 = var_0.filename
    var_8 = bool(var_0.filename == var_6)
    assert var_8 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "x = 'ü'"
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "x = '\\n'"
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = b'x = 1\x80'
    var_2 = 'utf-8'
    var_3 = 'replace'
    var_4 = 'test.py'
    var_5 = var_0.exit_code
    assert var_5 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = '
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 'x = 1'
    var_5 = var_0.scan(var_4, var_2)
    var_6 = var_0.exit_code
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\x00'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 'x = 1'
    var_5 = var_0.scan(var_4, var_2)
    var_6 = var_0.exit_code
    assert var_6 == 0



# Parsed testcases at query #6
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '_'
    var_2 = module_0._ignore_variable(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '_x'
    var_2 = module_0._ignore_variable(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '__init__'
    var_2 = module_0._ignore_variable(var_0, var_1)
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '__x'
    var_2 = module_0._ignore_variable(var_0, var_1)
    assert var_2 is False

import vulture.core as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'x'
    var_2 = module_0._ignore_variable(var_0, var_1)
    assert var_2 is False

import vulture.core as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '___x'
    var_2 = module_0._ignore_variable(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_visit_FunctionDef_marks_property_when_property_decorator_present. Retrieved 6/8 statements.
# Partially parsed test_visit_FunctionDef_marks_method_when_self_first_arg. Retrieved 6/8 statements.
# Partially parsed test_visit_FunctionDef_marks_method_when_staticmethod_decorator. Retrieved 6/8 statements.
# Partially parsed test_visit_FunctionDef_marks_method_when_classmethod_decorator. Retrieved 6/8 statements.
# Partially parsed test_visit_FunctionDef_marks_function_when_no_self_and_no_special_decorator. Retrieved 6/8 statements.
# Partially parsed test_visit_FunctionDef_ignores_function_with_ignored_decorator. Retrieved 7/9 statements.
# Partially parsed test_visit_FunctionDef_ignores_method_with_ignored_decorator. Retrieved 7/9 statements.
# Partially parsed test_visit_FunctionDef_ignores_property_with_ignored_decorator. Retrieved 7/9 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '\nclass MyClass:\n    @property\n    def my_prop(self):\n        return 42\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_props
    var_4 = 'my_prop'
    var_5 = 'property'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '\nclass MyClass:\n    def my_method(self, x):\n        pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_methods
    var_4 = 'my_method'
    var_5 = 'method'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '\nclass MyClass:\n    @staticmethod\n    def my_static():\n        pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_methods
    var_4 = 'my_static'
    var_5 = 'method'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '\nclass MyClass:\n    @classmethod\n    def my_classmethod(cls):\n        pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_methods
    var_4 = 'my_classmethod'
    var_5 = 'method'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '\ndef my_func():\n    pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = 'my_func'
    var_5 = 'function'

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = '\n@my_decorator\ndef my_func():\n    pass\n'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_funcs
    var_6 = 'my_func'

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = '\nclass MyClass:\n    @my_decorator\n    def my_method(self):\n        pass\n'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_methods
    var_6 = 'my_method'

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = '\nclass MyClass:\n    @my_decorator\n    @property\n    def my_prop(self):\n        return 42\n'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_props
    var_6 = 'my_prop'



# Parsed testcases at query #8
#--------------------------




import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/some/dir/__init__.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '*'
    var_5 = module_1._ignore_import(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/some/dir/__init__.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'os'
    var_5 = module_1._ignore_import(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/some/dir/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '*'
    var_5 = module_1._ignore_import(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/some/dir/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'os'
    var_5 = module_1._ignore_import(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/some/dir/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'sys'
    var_5 = module_1._ignore_import(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_ignore_class_returns_true_for_test_file_with_test_in_class_name. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_returns_false_for_non_test_file_with_test_in_class_name. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_returns_false_for_test_file_without_test_in_class_name. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_returns_false_for_non_test_file_without_test_in_class_name. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_matches_test_directory_pattern. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_matches_test_suffix_pattern. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_case_insensitive_check. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_class_name_contains_test_not_prefix. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_class_name_lowercase_test. Retrieved 7/16 statements.
# Partially parsed test_ignore_class_empty_class_name. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/tests/test_example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'TestExample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/src/example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'TestExample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/tests/test_example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'Example'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/src/example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'Example'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/test/example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'TestExample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/example_test.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'TestExample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/TESTS/test_example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'TestExample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/tests/test_example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'MyTestExample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/tests/test_example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = 'testexample'

def test_case_0():
    var_0 = 'Path'
    var_1 = 'resolve'
    var_2 = '__str__'
    var_3 = '/project/tests/test_example.py'
    var_4 = lambda self: var_3
    var_5 = {var_2: var_4}
    var_6 = ''



# Parsed testcases at query #10
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'ignored_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'TestClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_0, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = 'name'
    var_18 = 'bases'
    var_19 = 'keywords'
    var_20 = 'body'
    var_21 = 'decorator_list'
    var_22 = {var_17: var_3, var_18: var_4, var_19: var_5, var_20: var_6, var_21: var_15}
    var_23 = module_1.ClassDef(*var_16, **var_22)
    var_24 = var_2.visit_ClassDef(var_23)
    var_25 = var_2.defined_classes
    var_26 = len(var_25)
    assert var_26 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'name'
    var_8 = 'bases'
    var_9 = 'keywords'
    var_10 = 'body'
    var_11 = 'decorator_list'
    var_12 = {var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4, var_11: var_5}
    var_13 = module_1.ClassDef(*var_6, **var_12)
    var_14 = var_0.visit_ClassDef(var_13)
    var_15 = var_0.defined_classes
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = var_0.defined_classes[0].name
    assert var_17 == 'TestClass'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'other_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'TestClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'some_decorator'
    var_8 = []
    var_9 = {}
    var_10 = module_1.Load(*var_8, **var_9)
    var_11 = []
    var_12 = 'id'
    var_13 = 'ctx'
    var_14 = {var_12: var_7, var_13: var_10}
    var_15 = module_1.Name(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = 'name'
    var_19 = 'bases'
    var_20 = 'keywords'
    var_21 = 'body'
    var_22 = 'decorator_list'
    var_23 = {var_18: var_3, var_19: var_4, var_20: var_5, var_21: var_6, var_22: var_16}
    var_24 = module_1.ClassDef(*var_17, **var_23)
    var_25 = var_2.visit_ClassDef(var_24)
    var_26 = var_2.defined_classes
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = var_2.defined_classes[0].name
    assert var_28 == 'TestClass'

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'TestClass'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'name'
    var_9 = 'bases'
    var_10 = 'keywords'
    var_11 = 'body'
    var_12 = 'decorator_list'
    var_13 = {var_8: var_0, var_9: var_3, var_10: var_4, var_11: var_5, var_12: var_6}
    var_14 = module_1.ClassDef(*var_7, **var_13)
    var_15 = var_2.visit_ClassDef(var_14)
    var_16 = var_2.defined_classes
    var_17 = len(var_16)
    assert var_17 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'OtherClass'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 'TestClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'name'
    var_10 = 'bases'
    var_11 = 'keywords'
    var_12 = 'body'
    var_13 = 'decorator_list'
    var_14 = {var_9: var_3, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_15 = module_1.ClassDef(*var_8, **var_14)
    var_16 = var_2.visit_ClassDef(var_15)
    var_17 = var_2.defined_classes
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = var_2.defined_classes[0].name
    assert var_19 == 'TestClass'



# Parsed testcases at query #11
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = '*.pyc'
    var_4 = [var_3]
    var_5 = var_0.scavenge(var_2, var_4)
    var_6 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = []
    var_2 = var_0.scavenge(var_1)
    var_3 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = 'test_path'
    var_3 = [var_2]
    var_4 = var_1.scavenge(var_3)
    var_5 = var_1.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'nonexistent_file.py'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = '*'
    var_4 = [var_3]
    var_5 = var_0.scavenge(var_2, var_4)
    var_6 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_1, var_2]
    var_4 = var_0.scavenge(var_3)
    var_5 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 'test_path'
    var_4 = [var_3]
    var_5 = var_2.scavenge(var_4)
    var_6 = var_2.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = '@staticmethod'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'test_path'
    var_4 = [var_3]
    var_5 = var_2.scavenge(var_4)
    var_6 = var_2.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'path1'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)
    var_4 = 'path2'
    var_5 = [var_4]
    var_6 = var_0.scavenge(var_5)
    var_7 = var_0.exit_code



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_visit_name_store_defined_variable. Retrieved 6/8 statements.
# Partially parsed test_visit_name_param_defined_variable. Retrieved 6/8 statements.


import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_var'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = 'my_var'
    var_12 = bool('my_var' in var_0.used_names)
    assert var_12 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_var'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Del(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = 'my_var'
    var_12 = bool('my_var' in var_0.used_names)
    assert var_12 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '_'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = '_'
    var_12 = bool('_' not in var_0.used_names)
    assert var_12 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_var'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Store(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = var_0.defined_vars

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'my_var'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Param(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = var_0.defined_vars



# Parsed testcases at query #13
#--------------------------




import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'any/path/file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '__init__'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'any/path/file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '__str__'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/tests/test_module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'setup_method'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/test/test_module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_example'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/tests/test_module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'helper'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/src/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_example'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/src/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'setup_method'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/src/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'helper'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/src/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_runner'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'project/tests/test_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '__special__'
    var_5 = module_1._ignore_method(var_3, var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_ignore_function_returns_true_for_pytest_function_in_test_file. Retrieved 8/17 statements.


import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = False
    var_2 = '/tmp/test_example.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.touch()
    var_7 = 'pytest'
    var_8 = module_1._ignore_function(var_5, var_7)
    assert var_8 is True
    var_9 = var_5.unlink()

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/tmp/test_example.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'test_something'
    var_6 = module_1._ignore_function(var_3, var_5)
    assert var_6 is True
    var_7 = var_3.unlink()

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/tmp/test_example.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'helper_function'
    var_6 = module_1._ignore_function(var_3, var_5)
    assert var_6 is False
    var_7 = var_3.unlink()

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/tmp/example.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'pytest'
    var_6 = module_1._ignore_function(var_3, var_5)
    assert var_6 is False
    var_7 = var_3.unlink()

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/tmp/example.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'test_something'
    var_6 = module_1._ignore_function(var_3, var_5)
    assert var_6 is False
    var_7 = var_3.unlink()

import pathlib as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '/tmp/example.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'helper_function'
    var_6 = module_1._ignore_function(var_3, var_5)
    assert var_6 is False
    var_7 = var_3.unlink()



# Parsed testcases at query #15
#--------------------------




import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "__all__ = ['a', 'b']"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "__all__ = ('a', 'b')"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "__all__ = 'string'"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(not var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "other = ['a', 'b']"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(not var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "x = __all__ = ['a']"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "obj.__all__ = ['a']"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(not var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = '__all__ = []'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = '__all__ = ()'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 0
    var_1 = "print('test')"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = module_1._assigns_special_variable__all__(var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_scavenge_unique_imports_non_empty_when_defined_imports_present. Retrieved 11/12 statements.


import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'os'
    var_2 = 'import'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = 1
    var_7 = module_0.Item(var_1, var_2, var_5, var_6, var_6)
    var_8 = []
    var_9 = None
    var_10 = var_0.scavenge(var_8, var_9)
    var_11 = var_0.defined_imports
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_scavenge_with_valid_python_files. Retrieved 5/18 statements.
# Partially parsed test_scavenge_with_exclude_pattern. Retrieved 7/20 statements.
# Partially parsed test_scavenge_with_single_file. Retrieved 5/15 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test1.py'
    var_2 = 'x = 1\n'
    var_3 = 'test2.py'
    var_4 = 'y = 2\n'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test1.py'
    var_2 = 'x = 1\n'
    var_3 = 'test2.py'
    var_4 = 'y = 2\n'
    var_5 = 'test1*'
    var_6 = [var_5]

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '/nonexistent/path'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '.py'
    var_2 = False
    var_3 = 'w'
    var_4 = 'z = 3\n'



# Parsed testcases at query #18
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '_ = 1'
    var_2 = var_0.scan(var_1)
    var_3 = '_'
    var_4 = bool('_' not in var_0.used_names)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------




import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '__all__'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = {}
    var_13 = module_0.Load(*var_11, **var_12)
    var_14 = []
    var_15 = 'elts'
    var_16 = 'ctx'
    var_17 = {var_15: var_10, var_16: var_13}
    var_18 = module_0.List(*var_14, **var_17)
    var_19 = []
    var_20 = 'targets'
    var_21 = 'value'
    var_22 = {var_20: var_9, var_21: var_18}
    var_23 = module_0.Assign(*var_19, **var_22)
    var_24 = module_1._assigns_special_variable__all__(var_23)
    var_25 = bool(var_24)
    assert var_25 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '__all__'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = {}
    var_13 = module_0.Load(*var_11, **var_12)
    var_14 = []
    var_15 = 'elts'
    var_16 = 'ctx'
    var_17 = {var_15: var_10, var_16: var_13}
    var_18 = module_0.Tuple(*var_14, **var_17)
    var_19 = []
    var_20 = 'targets'
    var_21 = 'value'
    var_22 = {var_20: var_9, var_21: var_18}
    var_23 = module_0.Assign(*var_19, **var_22)
    var_24 = module_1._assigns_special_variable__all__(var_23)
    var_25 = bool(var_24)
    assert var_25 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.Constant(*var_1, **var_3)
    var_5 = []
    var_6 = 'value'
    var_7 = {var_6: var_4}
    var_8 = module_0.Expr(*var_5, **var_7)
    var_9 = module_1._assigns_special_variable__all__(var_8)
    var_10 = bool(not var_9)
    assert var_10 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'other'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = {}
    var_13 = module_0.Load(*var_11, **var_12)
    var_14 = []
    var_15 = 'elts'
    var_16 = 'ctx'
    var_17 = {var_15: var_10, var_16: var_13}
    var_18 = module_0.List(*var_14, **var_17)
    var_19 = []
    var_20 = 'targets'
    var_21 = 'value'
    var_22 = {var_20: var_9, var_21: var_18}
    var_23 = module_0.Assign(*var_19, **var_22)
    var_24 = module_1._assigns_special_variable__all__(var_23)
    var_25 = bool(not var_24)
    assert var_25 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = '__all__'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = [var_8]
    var_10 = 1
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_10}
    var_14 = module_0.Constant(*var_11, **var_13)
    var_15 = []
    var_16 = 'targets'
    var_17 = 'value'
    var_18 = {var_16: var_9, var_17: var_14}
    var_19 = module_0.Assign(*var_15, **var_18)
    var_20 = module_1._assigns_special_variable__all__(var_19)
    var_21 = bool(not var_20)
    assert var_21 is True

import ast as module_0
import vulture.core as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = '__all__'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Store(*var_10, **var_11)
    var_13 = []
    var_14 = 'value'
    var_15 = 'attr'
    var_16 = 'ctx'
    var_17 = {var_14: var_8, var_15: var_9, var_16: var_12}
    var_18 = module_0.Attribute(*var_13, **var_17)
    var_19 = [var_18]
    var_20 = []
    var_21 = []
    var_22 = {}
    var_23 = module_0.Load(*var_21, **var_22)
    var_24 = []
    var_25 = 'elts'
    var_26 = 'ctx'
    var_27 = {var_25: var_20, var_26: var_23}
    var_28 = module_0.List(*var_24, **var_27)
    var_29 = []
    var_30 = 'targets'
    var_31 = 'value'
    var_32 = {var_30: var_19, var_31: var_28}
    var_33 = module_0.Assign(*var_29, **var_32)
    var_34 = module_1._assigns_special_variable__all__(var_33)
    var_35 = bool(not var_34)
    assert var_35 is True



# Parsed testcases at query #20
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 1
    var_2 = var_0.__init__.__code__.co_consts[var_1]
    var_3 = 'mypattern'
    var_4 = var_2(var_3)
    assert var_4 == '*mypattern*'



# Parsed testcases at query #21
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = 'file.py'
    var_3 = 10
    var_4 = 20
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.get_whitelist_string()
    assert var_6 == 'my_func  # unused function (file.py:10)'

import vulture.core as module_0

def test_case_0():
    var_0 = 'attr1'
    var_1 = 'attribute'
    var_2 = 'file.py'
    var_3 = 5
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.get_whitelist_string()
    assert var_5 == '_.attr1  # unused attribute (file.py:5)'

import vulture.core as module_0

def test_case_0():
    var_0 = 'method1'
    var_1 = 'method'
    var_2 = 'file.py'
    var_3 = 1
    var_4 = 10
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.get_whitelist_string()
    assert var_6 == '_.method1  # unused method (file.py:1)'

import vulture.core as module_0

def test_case_0():
    var_0 = 'prop1'
    var_1 = 'property'
    var_2 = 'file.py'
    var_3 = 2
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.get_whitelist_string()
    assert var_5 == '_.prop1  # unused property (file.py:2)'

import vulture.core as module_0

def test_case_0():
    var_0 = 'code_block'
    var_1 = 'unreachable_code'
    var_2 = 'file.py'
    var_3 = 3
    var_4 = 8
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.get_whitelist_string()
    assert var_6 == "# unused unreachable_code 'code_block' (file.py:3)"

import vulture.core as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'variable'
    var_2 = 'file.py'
    var_3 = 4
    var_4 = 'custom message'
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, var_4)
    var_6 = var_5.get_whitelist_string()
    assert var_6 == 'var1  # custom message (file.py:4)'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_visit_defined_variable_on_store. Retrieved 5/7 statements.
# Partially parsed test_visit_class_def_defined_class. Retrieved 5/7 statements.
# Partially parsed test_visit_function_def_defined_function. Retrieved 5/7 statements.
# Partially parsed test_visit_function_def_defined_method. Retrieved 5/7 statements.
# Partially parsed test_visit_function_def_defined_property. Retrieved 5/7 statements.
# Partially parsed test_visit_import_adds_defined_import. Retrieved 5/7 statements.
# Partially parsed test_visit_import_from_adds_defined_import. Retrieved 5/7 statements.
# Partially parsed test_visit_import_from_future_ignored. Retrieved 5/7 statements.
# Partially parsed test_visit_attribute_as_store_adds_defined_attr. Retrieved 5/7 statements.
# Partially parsed test_visit_async_function_def_defined_function. Retrieved 5/7 statements.
# Partially parsed test_visit_arg_defined_variable. Retrieved 5/7 statements.
# Partially parsed test_visit_ignore_names_ignored. Retrieved 6/8 statements.
# Partially parsed test_visit_ignore_decorators_ignored. Retrieved 7/9 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "\n__all__ = ['foo', 'bar']\nfoo = 1\nbar = 2\n"
    var_2 = var_0.scan(var_1)
    var_3 = 'foo'
    var_4 = bool('foo' in var_0.used_names)
    assert var_4 is True
    var_5 = 'bar'
    var_6 = bool('bar' in var_0.used_names)
    assert var_6 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '_ = 1'
    var_2 = var_0.scan(var_1)
    var_3 = '_'
    var_4 = bool('_' not in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1; print(x)'
    var_2 = var_0.scan(var_1)
    var_3 = 'x'
    var_4 = bool('x' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "getattr(obj, 'some_attr')"
    var_2 = var_0.scan(var_1)
    var_3 = 'some_attr'
    var_4 = bool('some_attr' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "hasattr(obj, 'some_attr')"
    var_2 = var_0.scan(var_1)
    var_3 = 'some_attr'
    var_4 = bool('some_attr' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '"%(my_var)s" % locals()'
    var_2 = var_0.scan(var_1)
    var_3 = 'my_var'
    var_4 = bool('my_var' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '"{my_var}".format(**locals())'
    var_2 = var_0.scan(var_1)
    var_3 = 'my_var'
    var_4 = bool('my_var' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class MyClass: pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_classes
    var_4 = 'MyClass'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def my_func(): pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = 'my_func'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class A: def method(self): pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_methods
    var_4 = 'method'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class A: @property\ndef prop(self): pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_props
    var_4 = 'prop'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'import os'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_imports
    var_4 = 'os'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'from os import path'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_imports
    var_4 = 'path'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'from __future__ import annotations'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_imports
    var_4 = 'annotations'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'obj.attr = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_attrs
    var_4 = 'attr'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'print(obj.attr)'
    var_2 = var_0.scan(var_1)
    var_3 = 'attr'
    var_4 = bool('attr' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '\nmatch obj:\n    case A(x=1):\n        pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = 'x'
    var_4 = bool('x' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'async def my_async_func(): pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = 'my_async_func'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def func(arg): pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = 'arg'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ['a']\na = 1"
    var_2 = var_0.scan(var_1)
    var_3 = 'a'
    var_4 = bool('a' in var_0.used_names)
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'ignored_func'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 'def ignored_func(): pass'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_funcs

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = '@my_decorator\ndef decorated_func(): pass'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_funcs
    var_6 = 'decorated_func'



# Parsed testcases at query #23
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ['a', 'b']"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'a'
    var_7 = bool('a' in var_0.used_names)
    assert var_7 is True
    var_8 = 'b'
    var_9 = bool('b' in var_0.used_names)
    assert var_9 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ('x', 'y')"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'x'
    var_7 = bool('x' in var_0.used_names)
    assert var_7 is True
    var_8 = 'y'
    var_9 = bool('y' in var_0.used_names)
    assert var_9 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'x'
    var_7 = bool('x' not in var_0.used_names)
    assert var_7 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '__all__ = [1, 2]'
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = bool(not var_0.used_names)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ['a', 1, 'b']"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'a'
    var_7 = bool('a' in var_0.used_names)
    assert var_7 is True
    var_8 = 'b'
    var_9 = bool('b' in var_0.used_names)
    assert var_9 is True
    var_10 = '1'
    var_11 = bool('1' not in var_0.used_names)
    assert var_11 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "a, __all__ = 1, ['x', 'y']"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'x'
    var_7 = bool('x' in var_0.used_names)
    assert var_7 is True
    var_8 = 'y'
    var_9 = bool('y' in var_0.used_names)
    assert var_9 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = __all__ = ['a']"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'a'
    var_7 = bool('a' in var_0.used_names)
    assert var_7 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '__all__ = []'
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = bool(not var_0.used_names)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '__all__ = ()'
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = bool(not var_0.used_names)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "module.__all__ = ['a']"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'a'
    var_7 = bool('a' not in var_0.used_names)
    assert var_7 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "d['__all__'] = ['a']"
    var_2 = 0
    var_3 = module_1.parse(var_1)
    var_4 = var_3.body[var_2]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'a'
    var_7 = bool('a' not in var_0.used_names)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------




import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = var_0.verbose
    assert var_1 is False
    var_2 = var_0.ignore_names
    var_3 = bool(var_0.ignore_names == [])
    assert var_3 is True
    var_4 = var_0.ignore_decorators
    var_5 = bool(var_0.ignore_decorators == [])
    assert var_5 is True
    var_6 = []
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_0.filename
    var_10 = bool(var_0.filename == var_8)
    assert var_10 is True
    var_11 = var_0.code
    var_12 = bool(var_0.code == [])
    assert var_12 is True
    var_13 = var_0.exit_code
    var_14 = var_0.noqa_lines
    var_15 = bool(var_0.noqa_lines == {})
    assert var_15 is True

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.verbose
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = module_0.Vulture(ignore_names=var_2)
    var_4 = var_3.ignore_names
    var_5 = bool(var_3.ignore_names == var_2)
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'decor1'
    var_1 = 'decor2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Vulture(ignore_decorators=var_2)
    var_4 = var_3.ignore_decorators
    var_5 = bool(var_3.ignore_decorators == var_2)
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'defined_attrs'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'defined_classes'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'defined_funcs'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'defined_imports'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'defined_methods'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'defined_props'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'defined_vars'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True
    var_22 = 'unreachable_code'
    var_23 = hasattr(var_0, var_22)
    var_24 = bool(var_23)
    assert var_24 is True
    var_25 = 'used_names'
    var_26 = hasattr(var_0, var_25)
    var_27 = bool(var_26)
    assert var_27 is True
    var_28 = 'reachability'
    var_29 = hasattr(var_0, var_28)
    var_30 = bool(var_29)
    assert var_30 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_14_true. Retrieved 3/5 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.defined_props
    var_3 = var_1.defined_props.typ
    assert var_3 == 'property'



# Parsed testcases at query #26
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = var_0.ignore_names
    var_2 = bool(var_0.ignore_names == [])
    assert var_2 is True
    var_3 = var_0.ignore_decorators
    var_4 = bool(var_0.ignore_decorators == [])
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_unused_items_returns_empty_list_when_all_defined_items_are_used. Retrieved 3/7 statements.
# Partially parsed test_get_unused_items_returns_unused_items_sorted_case_insensitive. Retrieved 4/10 statements.
# Partially parsed test_get_unused_items_handles_empty_used_names. Retrieved 3/8 statements.
# Partially parsed test_get_unused_items_ignores_duplicate_items. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0, var_1}

def test_case_0():
    var_0 = 'B'
    var_1 = 'a'
    var_2 = 'C'
    var_3 = {var_1}

import vulture.core as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1, var_2}
    var_4 = module_0._get_unused_items(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = set()

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0}



# Parsed testcases at query #28
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '_'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = '_'
    var_12 = bool('_' not in var_0.used_names)
    assert var_12 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_scan_resets_reachability. Retrieved 6/7 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = '
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1\x00'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.exit_code

import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = 'module.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = [var_2]
    var_5 = {}
    var_6 = module_1.Path(*var_4, **var_5)
    var_7 = var_0.filename
    var_8 = bool(var_0.filename == var_6)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = ''
    var_2 = var_0.scan(var_1)
    var_3 = 'if True: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_0.scan(var_1)
    var_6 = var_0.reachability._current
    assert var_6 is None

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def f(x): # type: (int) -> None\n    pass'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = len(var_3)
    assert var_4 == 1

import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a = 1'
    var_2 = var_0.scan(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = var_0.filename
    var_7 = bool(var_0.filename == var_5)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1 # type: invalid'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.exit_code



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_main_valid_input. Retrieved 5/12 statements.
# Partially parsed test_main_invalid_cmdline_arguments. Retrieved 3/6 statements.
# Partially parsed test_main_verbose_mode. Retrieved 6/13 statements.
# Partially parsed test_main_with_ignore_names. Retrieved 7/14 statements.
# Partially parsed test_main_with_exclude. Retrieved 7/14 statements.
# Partially parsed test_main_with_min_confidence. Retrieved 7/14 statements.
# Partially parsed test_main_with_sort_by_size. Retrieved 6/13 statements.
# Partially parsed test_main_with_make_whitelist. Retrieved 6/13 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = 'test_file.py'
    var_2 = 'w'
    var_3 = module_0.main()
    var_4 = 'test_file.py'

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--invalid'
    var_2 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--verbose'
    var_2 = 'test_file.py'
    var_3 = 'w'
    var_4 = module_0.main()
    var_5 = 'test_file.py'

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--ignore-names'
    var_2 = 'foo'
    var_3 = 'test_file.py'
    var_4 = 'w'
    var_5 = module_0.main()
    var_6 = 'test_file.py'

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--exclude'
    var_2 = 'test_*.py'
    var_3 = 'test_file.py'
    var_4 = 'w'
    var_5 = module_0.main()
    var_6 = 'test_file.py'

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = 'test_file.py'
    var_4 = 'w'
    var_5 = module_0.main()
    var_6 = 'test_file.py'

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--sort-by-size'
    var_2 = 'test_file.py'
    var_3 = 'w'
    var_4 = module_0.main()
    var_5 = 'test_file.py'

import vulture.core as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = '--make-whitelist'
    var_2 = 'test_file.py'
    var_3 = 'w'
    var_4 = module_0.main()
    var_5 = 'test_file.py'



# Parsed testcases at query #31
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'MyClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = {}
    var_9 = module_1.Load(*var_7, **var_8)
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_0, var_12: var_9}
    var_14 = module_1.Name(*var_10, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = 'name'
    var_18 = 'bases'
    var_19 = 'keywords'
    var_20 = 'body'
    var_21 = 'decorator_list'
    var_22 = {var_17: var_3, var_18: var_4, var_19: var_5, var_20: var_6, var_21: var_15}
    var_23 = module_1.ClassDef(*var_16, **var_22)
    var_24 = var_2.visit_ClassDef(var_23)



# Parsed testcases at query #32
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = module_0.Vulture(var_0, var_1, var_2)
    var_4 = var_3.defined_attrs
    var_5 = var_4.verbose
    assert var_5 is False



# Parsed testcases at query #33
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '*pattern*'
    var_2 = [var_1]
    var_3 = []
    var_4 = var_0.scavenge(var_3, var_2)



# Parsed testcases at query #34
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '_'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = var_0.visit_Name(var_9)
    var_11 = '_'
    var_12 = bool('_' not in var_0.used_names)
    assert var_12 is True



# Parsed testcases at query #35
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = '/path/to/file.py'
    var_3 = 10
    var_4 = 20
    var_5 = 'custom message'
    var_6 = 75
    var_7 = module_0.Item(var_0, var_1, var_2, var_3, var_4, var_5, var_6)
    var_8 = var_7.name
    assert var_8 == 'my_func'
    var_9 = var_7.typ
    assert var_9 == 'function'
    var_10 = var_7.filename
    assert var_10 == '/path/to/file.py'
    var_11 = var_7.first_lineno
    assert var_11 == 10
    var_12 = var_7.last_lineno
    assert var_12 == 20
    var_13 = var_7.message
    assert var_13 == 'custom message'
    var_14 = var_7.confidence
    assert var_14 == 75

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'variable'
    var_2 = '/path/to/file.py'
    var_3 = 5
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.message
    assert var_5 == "unused variable 'my_var'"

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = '/path/to/file.py'
    var_3 = 1
    var_4 = 2
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.confidence
    assert var_6 == 75



# Parsed testcases at query #36
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = 'test'
    var_4 = var_0.scavenge(var_2, var_3)
    var_5 = var_0.exit_code



# Parsed testcases at query #37
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "getattr(obj, 'some_attr')"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = 'some_attr'
    var_5 = bool('some_attr' in var_0.used_names)
    assert var_5 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'getattr(obj)'
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = set()
    var_5 = var_0.used_names
    var_6 = bool(var_0.used_names == var_4)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "getattr(obj, 'attr', default, extra)"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = set()
    var_5 = var_0.used_names
    var_6 = bool(var_0.used_names == var_4)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "hasattr(obj, 'some_attr')"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = 'some_attr'
    var_5 = bool('some_attr' in var_0.used_names)
    assert var_5 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "hasattr(obj, 'attr', extra)"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = set()
    var_5 = var_0.used_names
    var_6 = bool(var_0.used_names == var_4)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "'{my_var}'.format(**locals())"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = 'my_var'
    var_5 = bool('my_var' in var_0.used_names)
    assert var_5 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "'{my_var}'.format(my_var=1)"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = 'my_var'
    var_5 = bool('my_var' not in var_0.used_names)
    assert var_5 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "other_func(obj, 'attr')"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = set()
    var_5 = var_0.used_names
    var_6 = bool(var_0.used_names == var_4)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'getattr(obj, attr_name)'
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = set()
    var_5 = var_0.used_names
    var_6 = bool(var_0.used_names == var_4)
    assert var_6 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "'{a} {b}'.format(**locals())"
    var_2 = module_1.parse(var_1)
    var_3 = var_0.visit(var_2)
    var_4 = bool('a' in var_0.used_names and 'b' in var_0.used_names)
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'my_decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = '\n@my_decorator\ndef my_func():\n    pass\n'
    var_4 = var_2.scan(var_3)
    var_5 = var_2.defined_funcs
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_2.unused_funcs
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_get_list_logging_verbose_false. Retrieved 3/5 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.defined_attrs



# Parsed testcases at query #40
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = 'x = 1'
    var_3 = var_1.scan(var_2)
    var_4 = var_1.code
    var_5 = bool(var_1.code == ['x = 1'])
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def f(x): # type: (int) -> None\n    pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_funcs
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_funcs[0].name
    assert var_5 == 'f'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1  # type: int\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'x'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'if True:\n    pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.unreachable_code
    var_4 = bool(var_0.unreachable_code == [])
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class A:\n    pass\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_classes
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.defined_classes[0].name
    assert var_5 == 'A'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a, b = 1, 2\n'
    var_2 = var_0.scan(var_1)
    var_3 = var_0.defined_vars
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0.defined_vars[0].name
    assert var_5 == 'a'
    var_6 = var_0.defined_vars[1].name
    assert var_6 == 'b'



# Parsed testcases at query #41
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = var_0.scan(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_get_list_returns_logging_list_with_verbose_true. Retrieved 5/10 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = '_Vulture__get_list'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'test'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_ignore_class_with_test_file_and_test_in_class_name. Retrieved 6/10 statements.
# Partially parsed test_ignore_class_with_test_file_and_no_test_in_class_name. Retrieved 6/10 statements.
# Partially parsed test_ignore_class_with_non_test_file_and_test_in_class_name. Retrieved 6/10 statements.
# Partially parsed test_ignore_class_with_non_test_file_and_no_test_in_class_name. Retrieved 6/10 statements.
# Partially parsed test_ignore_class_with_tests_directory_and_test_in_class_name. Retrieved 6/10 statements.
# Partially parsed test_ignore_class_with_test_suffix_and_test_in_class_name. Retrieved 6/10 statements.
# Partially parsed test_ignore_class_with_test_prefix_and_no_test_in_class_name. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/test/test_example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'TestExample'

def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/test/test_example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'MyClass'

def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/src/example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'TestExample'

def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/src/example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'MyClass'

def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/tests/test_example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'TestExample'

def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/test_example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'TestExample'

def test_case_0():
    var_0 = 'FakePath'
    var_1 = 'resolve'
    var_2 = '/project/test_example.py'
    var_3 = lambda self: var_2
    var_4 = {var_1: var_3}
    var_5 = 'MyClass'



# Parsed testcases at query #44
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = {'a', 'b'}"
    var_2 = var_0.scan(var_1)



# Parsed testcases at query #45
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'my_func'
    var_1 = 'function'
    var_2 = 'test.py'
    var_3 = 10
    var_4 = 20
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.name
    assert var_6 == 'my_func'
    var_7 = var_5.typ
    assert var_7 == 'function'
    var_8 = var_5.filename
    assert var_8 == 'test.py'
    var_9 = var_5.first_lineno
    assert var_9 == 10
    var_10 = var_5.last_lineno
    assert var_10 == 20
    var_11 = var_5.message
    assert var_11 == "unused function 'my_func'"
    var_12 = var_5.confidence
    assert var_12 == 80

import vulture.core as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'variable'
    var_2 = 'code.py'
    var_3 = 5
    var_4 = 'Custom message'
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, var_4)
    var_6 = var_5.name
    assert var_6 == 'my_var'
    var_7 = var_5.typ
    assert var_7 == 'variable'
    var_8 = var_5.filename
    assert var_8 == 'code.py'
    var_9 = var_5.first_lineno
    assert var_9 == 5
    var_10 = var_5.last_lineno
    assert var_10 == 5
    var_11 = var_5.message
    assert var_11 == 'Custom message'
    var_12 = var_5.confidence
    assert var_12 == 80

import vulture.core as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'attribute'
    var_2 = 'module.py'
    var_3 = 1
    var_4 = 50
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_3, confidence=var_4)
    var_6 = var_5.name
    assert var_6 == 'x'
    var_7 = var_5.typ
    assert var_7 == 'attribute'
    var_8 = var_5.filename
    assert var_8 == 'module.py'
    var_9 = var_5.first_lineno
    assert var_9 == 1
    var_10 = var_5.last_lineno
    assert var_10 == 1
    var_11 = var_5.message
    assert var_11 == "unused attribute 'x'"
    var_12 = var_5.confidence
    assert var_12 == 50



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_get_list_verbose_false. Retrieved 3/4 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Vulture(var_0)
    var_2 = 'test'



# Parsed testcases at query #47
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = module_0.Vulture(var_0, var_1, var_2)
    var_4 = var_3.ignore_names
    var_5 = bool(var_3.ignore_names == [])
    assert var_5 is True



# Parsed testcases at query #48
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.defined_attrs
    var_3 = var_2.typ
    assert var_3 == 'attribute'
    var_4 = var_2.verbose
    assert var_4 is True



# Parsed testcases at query #49
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = '*.pyc'
    var_4 = [var_3]
    var_5 = var_0.scavenge(var_2, var_4)
    var_6 = var_0.exit_code
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)
    var_4 = var_0.exit_code
    assert var_4 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = 'test_path'
    var_3 = [var_2]
    var_4 = '*test*'
    var_5 = [var_4]
    var_6 = var_1.scavenge(var_3, var_5)
    var_7 = var_1.exit_code
    assert var_7 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'nonexistent_file.py'
    var_2 = [var_1]
    var_3 = var_0.scavenge(var_2)
    var_4 = var_0.exit_code
    assert var_4 == 1

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = []
    var_2 = var_0.scavenge(var_1)
    var_3 = var_0.exit_code
    assert var_3 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_1, var_2]
    var_4 = var_0.scavenge(var_3)
    var_5 = var_0.exit_code
    assert var_5 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = '*'
    var_4 = [var_3]
    var_5 = var_0.scavenge(var_2, var_4)
    var_6 = var_0.exit_code
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = 'test_ignore'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_names=var_1)
    var_3 = 'test_path'
    var_4 = [var_3]
    var_5 = var_2.scavenge(var_4)
    var_6 = var_2.exit_code
    assert var_6 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = '@decorator'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = 'test_path'
    var_4 = [var_3]
    var_5 = var_2.scavenge(var_4)
    var_6 = var_2.exit_code
    assert var_6 == 0



# Parsed testcases at query #50
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'function'
    var_2 = 'test.py'
    var_3 = 1
    var_4 = 5
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.message
    assert var_6 == "unused function 'foo'"



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = 'unused_func'
    var_1 = 'unreachable_code'
    var_2 = '/path/to/file.py'
    var_3 = 10
    var_4 = 20
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.get_whitelist_string()
    var_7 = "# unused unreachable code 'unused_func' (/path/to/file.py:10)"
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'attr_name'
    var_1 = 'attribute'
    var_2 = '/path/to/file.py'
    var_3 = 5
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.get_whitelist_string()
    var_6 = '_.attr_name  # unused attribute (/path/to/file.py:5)'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'method_name'
    var_1 = 'method'
    var_2 = '/path/to/file.py'
    var_3 = 15
    var_4 = 25
    var_5 = module_0.Item(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.get_whitelist_string()
    var_7 = '_.method_name  # unused method (/path/to/file.py:15)'
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'prop_name'
    var_1 = 'property'
    var_2 = '/path/to/file.py'
    var_3 = 30
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.get_whitelist_string()
    var_6 = '_.prop_name  # unused property (/path/to/file.py:30)'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'var_name'
    var_1 = 'variable'
    var_2 = '/path/to/file.py'
    var_3 = 42
    var_4 = module_0.Item(var_0, var_1, var_2, var_3, var_3)
    var_5 = var_4.get_whitelist_string()
    var_6 = 'var_name  # unused variable (/path/to/file.py:42)'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_unused_code_returns_list_of_items. Retrieved 6/7 statements.
# Partially parsed test_get_unused_code_with_unused_attribute. Retrieved 7/9 statements.
# Partially parsed test_get_unused_code_with_unreachable_code. Retrieved 6/8 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'x'
    var_7 = var_4[0].typ
    assert var_7 == 'variable'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 1'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = 100
    var_5 = var_0.get_unused_code(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 200
    var_8 = var_0.get_unused_code(var_7)
    var_9 = len(var_8)
    assert var_9 == 0

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a = 1\nb = 2'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = False
    var_5 = var_0.get_unused_code(sort_by_size=var_4)
    var_6 = var_5[0].name
    assert var_6 == 'a'
    var_7 = var_5[1].name
    assert var_7 == 'b'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'a = 1\n\nb = 2'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = True
    var_5 = var_0.get_unused_code(sort_by_size=var_4)
    var_6 = var_5[0].name
    assert var_6 == 'a'
    var_7 = var_5[1].name
    assert var_7 == 'b'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class MyClass: pass'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'MyClass'
    var_7 = var_4[0].typ
    assert var_7 == 'class'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def my_func(): pass'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'my_func'
    var_7 = var_4[0].typ
    assert var_7 == 'function'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'import os'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'os'
    var_7 = var_4[0].typ
    assert var_7 == 'import'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class A:\n    def method(self): pass'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'method'
    var_7 = var_4[0].typ
    assert var_7 == 'method'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class A:\n    @property\n    def prop(self): pass'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].name
    assert var_6 == 'prop'
    var_7 = var_4[0].typ
    assert var_7 == 'property'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'class A: pass\na = A()\na.attr = 1'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = 'attribute'
    var_6 = 'attr'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'def f():\n    return 1\n    x = 2'
    var_2 = 'test.py'
    var_3 = var_0.scan(var_1, var_2)
    var_4 = var_0.get_unused_code()
    var_5 = 'unreachable_code'

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = -1
    var_2 = var_0.get_unused_code(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 101
    var_6 = var_0.get_unused_code(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'some_function(1, 2)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "getattr(obj, 'some_attr')"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'getattr(obj, attr_name)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "getattr(obj, 'some_attr', default)"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "getattr(obj, 'some_attr', default, extra)"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "hasattr(obj, 'some_attr')"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'some_attr'
    var_8 = bool('some_attr' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'hasattr(obj, attr_name)'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "hasattr(obj, 'some_attr', extra)"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "'{my_var}'.format(**locals())"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = 'my_var'
    var_8 = bool('my_var' in var_0.used_names)
    assert var_8 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "'{my_var}'.format(my_var=1)"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = 'some_var.format(**locals())'
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "'{my_var}'.replace(**locals())"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 0
    var_2 = "something.getattr(obj, 'attr')"
    var_3 = module_1.parse(var_2)
    var_4 = var_3.body[var_1]
    var_5 = var_4.value
    var_6 = var_0.visit_Call(var_5)
    var_7 = var_0.used_names
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #4
#--------------------------




import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ['func1', 'func2']"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_0.used_names)
    assert var_7 is True
    var_8 = 'func2'
    var_9 = bool('func2' in var_0.used_names)
    assert var_9 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ('func1', 'func2')"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_0.used_names)
    assert var_7 is True
    var_8 = 'func2'
    var_9 = bool('func2' in var_0.used_names)
    assert var_9 is True

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'x = 5'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = var_0.used_names
    var_7 = len(var_6)
    assert var_7 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = '__all__ = [1, 2]'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = var_0.used_names
    var_7 = len(var_6)
    assert var_7 == 0

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "__all__ = ['func1', 2, 'func3']"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = 'func1'
    var_7 = bool('func1' in var_0.used_names)
    assert var_7 is True
    var_8 = 'func3'
    var_9 = bool('func3' in var_0.used_names)
    assert var_9 is True
    var_10 = var_0.used_names
    var_11 = len(var_10)
    assert var_11 == 2

import vulture.core as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = "other_var = ['a', 'b']"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_0.visit_Assign(var_4)
    var_6 = var_0.used_names
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_vulture_constructor_defaults. Retrieved 12/31 statements.


import vulture.core as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = var_0.verbose
    assert var_1 is False
    var_2 = var_0.ignore_names
    var_3 = bool(var_0.ignore_names == [])
    assert var_3 is True
    var_4 = var_0.ignore_decorators
    var_5 = bool(var_0.ignore_decorators == [])
    assert var_5 is True
    var_6 = []
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = var_0.filename
    var_10 = bool(var_0.filename == var_8)
    assert var_10 is True
    var_11 = var_0.code
    var_12 = bool(var_0.code == [])
    assert var_12 is True
    var_13 = var_0.exit_code
    var_14 = var_0.noqa_lines
    var_15 = bool(var_0.noqa_lines == {})
    assert var_15 is True
    var_16 = var_0.defined_attrs
    var_17 = var_0.defined_attrs.typ
    assert var_17 == 'attribute'
    var_18 = var_0.defined_classes
    var_19 = var_0.defined_classes.typ
    assert var_19 == 'class'
    var_20 = var_0.defined_funcs
    var_21 = var_0.defined_funcs.typ
    assert var_21 == 'function'
    var_22 = var_0.defined_imports
    var_23 = var_0.defined_imports.typ
    assert var_23 == 'import'
    var_24 = var_0.defined_methods
    var_25 = var_0.defined_methods.typ
    assert var_25 == 'method'
    var_26 = var_0.defined_props
    var_27 = var_0.defined_props.typ
    assert var_27 == 'property'
    var_28 = var_0.defined_vars
    var_29 = var_0.defined_vars.typ
    assert var_29 == 'variable'
    var_30 = var_0.unreachable_code
    var_31 = var_0.unreachable_code.typ
    assert var_31 == 'unreachable_code'
    var_32 = var_0.used_names
    var_33 = var_0.used_names.typ
    assert var_33 == 'name'
    var_34 = var_0.reachability

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Vulture(var_0)
    var_2 = var_1.verbose
    assert var_2 is True

import vulture.core as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = module_0.Vulture(ignore_names=var_2)
    var_4 = var_3.ignore_names
    var_5 = bool(var_3.ignore_names == ['foo', 'bar'])
    assert var_5 is True

import vulture.core as module_0

def test_case_0():
    var_0 = '@deprecated'
    var_1 = [var_0]
    var_2 = module_0.Vulture(ignore_decorators=var_1)
    var_3 = var_2.ignore_decorators
    var_4 = bool(var_2.ignore_decorators == ['@deprecated'])
    assert var_4 is True

import vulture.core as module_0

def test_case_0():
    var_0 = True
    var_1 = 'x'
    var_2 = [var_1]
    var_3 = '@y'
    var_4 = [var_3]
    var_5 = module_0.Vulture(var_0, var_2, var_4)
    var_6 = var_5.verbose
    assert var_6 is True
    var_7 = var_5.ignore_names
    var_8 = bool(var_5.ignore_names == ['x'])
    assert var_8 is True
    var_9 = var_5.ignore_decorators
    var_10 = bool(var_5.ignore_decorators == ['@y'])
    assert var_10 is True



# Parsed testcases at query #6
#--------------------------




import vulture.core as module_0

def test_case_0():
    var_0 = module_0.Vulture()
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = 'test_pattern'
    var_4 = [var_3]
    var_5 = var_0.scavenge(var_2, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_main_with_valid_config_no_dead_code. Retrieved 16/23 statements.
# Partially parsed test_main_with_dead_code. Retrieved 16/26 statements.
# Partially parsed test_main_with_invalid_cmdline_arguments. Retrieved 1/8 statements.
# Partially parsed test_main_verbose_mode. Retrieved 19/26 statements.
# Partially parsed test_main_with_paths_and_exclude. Retrieved 18/25 statements.
# Partially parsed test_main_with_min_confidence_and_sort_by_size. Retrieved 18/25 statements.
# Partially parsed test_main_with_make_whitelist. Retrieved 17/24 statements.
# Partially parsed test_main_with_ignore_names. Retrieved 17/24 statements.
# Partially parsed test_main_with_ignore_decorators. Retrieved 17/24 statements.
# Partially parsed test_main_with_scavenge_raising_exception. Retrieved 17/27 statements.


import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'test_path.py'
    var_12 = [var_11]
    var_13 = None
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_13, var_5: var_8, var_6: var_8, var_7: var_8}
    var_15 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'test_path.py'
    var_12 = [var_11]
    var_13 = None
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_13, var_5: var_8, var_6: var_8, var_7: var_8}
    var_15 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = True
    var_9 = []
    var_10 = []
    var_11 = 'test_path.py'
    var_12 = [var_11]
    var_13 = None
    var_14 = 0
    var_15 = False
    var_16 = False
    var_17 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_16}
    var_18 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'path1.py'
    var_12 = 'path2.py'
    var_13 = [var_11, var_12]
    var_14 = 'test_*.py'
    var_15 = [var_14]
    var_16 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_13, var_4: var_15, var_5: var_8, var_6: var_8, var_7: var_8}
    var_17 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'test_path.py'
    var_12 = [var_11]
    var_13 = None
    var_14 = 50
    var_15 = True
    var_16 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_8}
    var_17 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'test_path.py'
    var_12 = [var_11]
    var_13 = None
    var_14 = True
    var_15 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_13, var_5: var_8, var_6: var_8, var_7: var_14}
    var_16 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = 'unused_var'
    var_10 = [var_9]
    var_11 = []
    var_12 = 'test_path.py'
    var_13 = [var_12]
    var_14 = None
    var_15 = {var_0: var_8, var_1: var_10, var_2: var_11, var_3: var_13, var_4: var_14, var_5: var_8, var_6: var_8, var_7: var_8}
    var_16 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = '@staticmethod'
    var_11 = [var_10]
    var_12 = 'test_path.py'
    var_13 = [var_12]
    var_14 = None
    var_15 = {var_0: var_8, var_1: var_9, var_2: var_11, var_3: var_13, var_4: var_14, var_5: var_8, var_6: var_8, var_7: var_8}
    var_16 = module_0.main()

import vulture.core as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = 'ignore_names'
    var_2 = 'ignore_decorators'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'make_whitelist'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'invalid_path'
    var_12 = [var_11]
    var_13 = None
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_13, var_5: var_8, var_6: var_8, var_7: var_8}
    var_15 = 'Unexpected error'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.main()



