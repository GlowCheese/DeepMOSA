####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import jinja2.environment as module_0
import cookiecutter.extensions as module_1

def test_case_0():
    var_0 = '-'
    var_1 = module_0.Environment()
    var_2 = module_1.SlugifyExtension(var_1)
    var_3 = 'slugify'
    var_4 = var_1.filters[var_3]
    var_5 = 'Test String'
    var_6 = var_4(var_5)
    assert var_6 == 'mocked-slug'
    var_7 = 'slugify'
    var_8 = bool('slugify' in var_1.filters)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import jinja2.environment as module_0
import cookiecutter.extensions as module_1
import uuid as module_2
import locale as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.UUIDExtension(var_0)
    var_2 = 'uuid4'
    var_3 = bool('uuid4' in var_0.globals)
    assert var_3 is True
    var_4 = 'uuid4'
    var_5 = var_0.globals[var_4]
    var_6 = callable(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = var_0.globals[var_4]
    var_9 = var_8()
    var_10 = 4
    var_11 = module_2.UUID(var_9, version=var_10)
    var_12 = module_3.str(var_11)
    var_13 = bool(var_12 == var_9)
    assert var_13 is True
    var_14 = bool(False)
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_constructor_initializes_environment_with_default_format.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_string_extension_constructor_adds_function_to_globals. Retrieved 12/19 statements.


import jinja2.environment as module_0
import cookiecutter.extensions as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.RandomStringExtension(var_0)
    var_2 = 'random_ascii_string'
    var_3 = bool('random_ascii_string' in var_0.globals)
    assert var_3 is True
    var_4 = 'random_ascii_string'
    var_5 = var_0.globals[var_4]
    var_6 = 10
    var_7 = False
    var_8 = var_5(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 10
    var_10 = 15
    var_11 = True
    var_12 = var_5(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 15



# Parsed testcases at query #5
#--------------------------




import jinja2.environment as module_0
import cookiecutter.extensions as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.JsonifyExtension(var_0)
    var_2 = 'jsonify'
    var_3 = bool('jsonify' in var_0.filters)
    assert var_3 is True
    var_4 = 'jsonify'
    var_5 = var_0.filters[var_4]
    var_6 = callable(var_5)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_environment_extend_datetime_format_not_called.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_extends_environment_with_default_datetime_format. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '%Y-%m-%d'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_environment_extended_with_datetime_format. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '%Y-%m-%d'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_extends_environment_with_default_datetime_format. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '%Y-%m-%d'



# Parsed testcases at query #10
#--------------------------




import jinja2.environment as module_0
import cookiecutter.extensions as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.JsonifyExtension(var_0)
    var_2 = 'jsonify'
    var_3 = bool('jsonify' in var_0.filters)
    assert var_3 is True
    var_4 = 'jsonify'
    var_5 = var_0.filters[var_4]
    var_6 = callable(var_5)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_environment_extended_with_datetime_format. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '%Y-%m-%d'



