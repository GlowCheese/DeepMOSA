####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = 'testuser'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = var_4.fields[var_1]
    var_7 = 'secret'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = 'Email'
    var_3 = False
    var_4 = module_1.String()
    var_5 = {var_1: var_4}
    var_6 = module_2.Schema(var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = var_7.render_field(field_name=var_1, field=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'optional'
    var_2 = True
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'status'
    var_2 = 'active'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = module_1.Integer()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = var_4.fields[var_1]
    var_7 = 'Invalid age'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, error=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'color'
    var_2 = 'red'
    var_3 = 'Red'
    var_4 = (var_2, var_3)
    var_5 = 'blue'
    var_6 = 'Blue'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_1.Choice(choices=var_8)
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.Form(env=var_0, schema=var_11)
    var_13 = var_11.fields[var_1]
    var_14 = var_12.render_field(field_name=var_1, field=var_13)
    var_15 = var_11.fields[var_1]
    var_16 = var_12.template_for_field(var_15)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = module_1.Boolean()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = var_4.fields[var_1]
    var_7 = var_5.render_field(field_name=var_1, field=var_6)
    var_8 = var_4.fields[var_1]
    var_9 = var_5.template_for_field(var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)
    var_9 = var_5.fields[var_1]
    var_10 = var_6.template_for_field(var_9)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'birthdate'
    var_2 = 'date'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'id'
    var_2 = True
    var_3 = module_1.Integer()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)
    assert var_8 == ''

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'user_name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = var_4.fields[var_1]
    var_7 = var_5.render_field(field_name=var_1, field=var_6)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'field1'
    var_2 = ''
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_6.render_field(field_name=var_1, field=var_7)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_template_for_field_with_choice_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_boolean_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_string_field_text_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_string_field_other_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_field_without_specialization. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_object_field_raises_assertion. Retrieved 6/10 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Choice'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Boolean'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'String'
    var_3 = ()
    var_4 = 'format'
    var_5 = 'text'
    var_6 = {var_4: var_5}
    var_7 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'String'
    var_3 = ()
    var_4 = 'format'
    var_5 = 'email'
    var_6 = {var_4: var_5}
    var_7 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Field'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Object'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.Environment()
    var_9 = 'field'
    var_10 = {var_9: var_7}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.Form(env=var_8, schema=var_11)
    var_13 = var_12.template_for_field(var_7)
    assert var_13 == 'forms/select.html'



# Parsed testcases at query #4
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = 'password'
    var_5 = module_3.String(format=var_4)
    var_6 = 'password_field'
    var_7 = 'my_password'
    var_8 = var_3.render_field(field_name=var_6, field=var_5, value=var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 28/44 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_no_values_and_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_uses_data_when_errors_exist. Retrieved 29/45 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'name'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = 'title'
    var_17 = 'allow_null'
    var_18 = 'has_default'
    var_19 = False
    var_20 = 'Name'
    var_21 = lambda : var_19
    var_22 = {var_15: var_19, var_16: var_20, var_17: var_19, var_18: var_21}
    var_23 = 'John'
    var_24 = {var_12: var_23}
    var_25 = {var_12: var_23}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with error {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'email'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Email'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = ''
    var_25 = {var_13: var_24}
    var_26 = {var_13: var_24}
    var_27 = 'Invalid email'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'id'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = True
    var_21 = 'ID'
    var_22 = False
    var_23 = lambda : var_22
    var_24 = {var_16: var_20, var_17: var_21, var_18: var_22, var_19: var_23}
    var_25 = ()
    var_26 = 'Name'
    var_27 = lambda : var_22
    var_28 = {var_16: var_22, var_17: var_26, var_18: var_22, var_19: var_27}
    var_29 = 'John'
    var_30 = {var_12: var_20, var_13: var_29}
    var_31 = {var_12: var_20, var_13: var_29}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with value {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'comment'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Comment'
    var_22 = True
    var_23 = lambda : var_20
    var_24 = {var_16: var_20, var_17: var_21, var_18: var_22, var_19: var_23}
    var_25 = None

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with value {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'age'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Age'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = 30
    var_25 = {var_13: var_24}
    var_26 = -5
    var_27 = {var_13: var_26}
    var_28 = 'Must be positive'



# Parsed testcases at query #6
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Object()
    var_4 = var_2.template_for_field(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_fields_without_validation. Retrieved 1/6 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 6/22 statements.
# Partially parsed test_render_fields_without_errors. Retrieved 5/21 statements.
# Partially parsed test_render_fields_skips_read_only. Retrieved 2/11 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = None
    var_3 = 'wrong'
    var_4 = 'Error message'
    var_5 = '<input>'

def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 'correct'
    var_3 = {var_1: var_2}
    var_4 = '<input>'

def test_case_0():
    var_0 = 'test_field'
    var_1 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 4/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = var_3.env.loader
    var_5 = var_1.FileSystemLoader
    var_6 = isinstance(var_4, var_5)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'some_package'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env.loader
    var_5 = var_1.PackageLoader
    var_6 = isinstance(var_4, var_5)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = 'some_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = var_4.env.loader
    var_6 = var_1.ChoiceLoader
    var_7 = isinstance(var_5, var_6)

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = None
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 11/17 statements.
# Partially parsed test_load_template_env_raises_assertion_error_if_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_2.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_2.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/custom/path'
    var_1 = 'mypackage'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_3.loader
    var_5 = var_4.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_4.loaders[var_7]
    var_9 = 1
    var_10 = var_4.loaders[var_9]

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_form_constructor_without_values. Retrieved 14/23 statements.
# Partially parsed test_form_constructor_with_values. Retrieved 18/27 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 15/24 statements.
# Partially parsed test_form_constructor_with_empty_values. Retrieved 16/25 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = lambda self, values: values
    var_13 = {var_11: var_12}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = 'serialized'
    var_13 = lambda self, values: {var_12: values}
    var_14 = {var_11: var_13}
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = None
    var_13 = lambda self, values: var_12
    var_14 = {var_11: var_13}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = {}
    var_13 = lambda self, values: var_12
    var_14 = {var_11: var_13}
    var_15 = {}



# Parsed testcases at query #11
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'extra'
    var_6 = 'value'
    var_7 = 'ignored'
    var_8 = {var_2: var_6, var_5: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_4, values=var_8)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_template_for_field_with_choice_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_boolean_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_string_field_text_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_string_field_other_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_generic_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_object_field_raises_assertion. Retrieved 6/10 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Choice'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Boolean'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'String'
    var_3 = ()
    var_4 = 'format'
    var_5 = 'text'
    var_6 = {var_4: var_5}
    var_7 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'String'
    var_3 = ()
    var_4 = 'format'
    var_5 = 'email'
    var_6 = {var_4: var_5}
    var_7 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Field'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Object'
    var_3 = ()
    var_4 = {}
    var_5 = module_0.Form(env=var_0, schema=var_1)



# Parsed testcases at query #13
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = 'testuser'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = 'secret'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = 'user@example.com'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = module_1.Integer()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.Integer()
    var_7 = 25
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String()
    var_7 = var_5.render_field(field_name=var_1, field=var_6)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'optional'
    var_2 = True
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'status'
    var_2 = 'active'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = 'invalid'
    var_8 = 'Invalid email'
    var_9 = var_5.render_field(field_name=var_1, field=var_6, value=var_7, error=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'color'
    var_2 = 'red'
    var_3 = 'Red'
    var_4 = (var_2, var_3)
    var_5 = 'blue'
    var_6 = 'Blue'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_1.Choice(choices=var_8)
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.Form(env=var_0, schema=var_11)
    var_13 = (var_2, var_3)
    var_14 = (var_5, var_6)
    var_15 = [var_13, var_14]
    var_16 = module_1.Choice(choices=var_15)
    var_17 = var_12.render_field(field_name=var_1, field=var_16, value=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = module_1.Boolean()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.Boolean()
    var_7 = True
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'Some text'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'token'
    var_2 = 'hidden'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'abc123'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'birthday'
    var_2 = 'date'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '2023-01-01'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'event_time'
    var_2 = 'datetime'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '2023-01-01T12:00'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'test'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'user_name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String()
    var_7 = var_5.render_field(field_name=var_1, field=var_6)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'full_name'
    var_2 = 'Full Name'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'full_name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String()
    var_7 = var_5.render_field(field_name=var_1, field=var_6)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'id'
    var_2 = True
    var_3 = module_1.Integer()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.Integer()
    var_8 = var_6.render_field(field_name=var_1, field=var_7, value=var_2)
    assert var_8 == ''



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = 'testuser'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = 'secret'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = 'test@example.com'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = module_1.Integer()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.Integer()
    var_7 = 25
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String()
    var_7 = var_5.render_field(field_name=var_1, field=var_6)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = True
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'default'
    var_3 = module_1.String()
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String()
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String()
    var_7 = ''
    var_8 = 'This field is required'
    var_9 = var_5.render_field(field_name=var_1, field=var_6, value=var_7, error=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'color'
    var_2 = 'red'
    var_3 = 'Red'
    var_4 = (var_2, var_3)
    var_5 = 'blue'
    var_6 = 'Blue'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_1.Choice(choices=var_8)
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.Form(env=var_0, schema=var_11)
    var_13 = (var_2, var_3)
    var_14 = (var_5, var_6)
    var_15 = [var_13, var_14]
    var_16 = module_1.Choice(choices=var_15)
    var_17 = var_12.render_field(field_name=var_1, field=var_16, value=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = module_1.Boolean()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.Boolean()
    var_7 = True
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'Some text'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'token'
    var_2 = 'hidden'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'abc123'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'birthday'
    var_2 = 'date'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '2023-01-01'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'start_time'
    var_2 = 'time'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '14:30'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'website'
    var_2 = 'url'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'https://example.com'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'query'
    var_2 = 'search'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = 'test'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'phone'
    var_2 = 'tel'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '1234567890'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'volume'
    var_2 = 'range'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '50'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'color'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = '#ff0000'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'month'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = '2023-01'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'week'
    var_2 = module_1.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = module_1.String(format=var_1)
    var_7 = '2023-W01'
    var_8 = var_5.render_field(field_name=var_1, field=var_6, value=var_7)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'event'
    var_2 = 'datetime'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = '2023-01-01T12:00'
    var_9 = var_6.render_field(field_name=var_1, field=var_7, value=var_8)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = module_1.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = module_1.String(format=var_2)
    var_8 = var_6.render_field(field_name=var_1, field=var_7)

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_required_false_when_field_has_default. Retrieved 13/17 statements.
# Partially parsed test_required_false_when_allow_null. Retrieved 13/17 statements.
# Partially parsed test_required_false_when_allow_blank. Retrieved 13/17 statements.
# Partially parsed test_required_false_when_has_default_and_allow_null. Retrieved 14/18 statements.
# Partially parsed test_required_false_when_has_default_and_allow_blank. Retrieved 14/18 statements.


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = 'default_value'
    var_3 = module_1.Field(default=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_7.allow_null
    var_9 = 'allow_blank'
    var_10 = False
    var_11 = getattr(var_7, var_9, var_10)
    var_12 = var_8 or var_11

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = True
    var_3 = module_1.Field(allow_null=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_7.allow_null
    var_9 = 'allow_blank'
    var_10 = False
    var_11 = getattr(var_7, var_9, var_10)
    var_12 = var_8 or var_11

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = True
    var_3 = module_1.String(allow_blank=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = module_3.Form(env=var_0, schema=var_5)
    var_7 = var_5.fields[var_1]
    var_8 = var_7.allow_null
    var_9 = 'allow_blank'
    var_10 = False
    var_11 = getattr(var_7, var_9, var_10)
    var_12 = var_8 or var_11

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = 'default'
    var_3 = True
    var_4 = module_1.Field(default=var_2, allow_null=var_3)
    var_5 = {var_1: var_4}
    var_6 = module_2.Schema(var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = var_8.allow_null
    var_10 = 'allow_blank'
    var_11 = False
    var_12 = getattr(var_8, var_10, var_11)
    var_13 = var_9 or var_12

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = 'default'
    var_3 = True
    var_4 = module_1.String(allow_blank=var_3)
    var_5 = {var_1: var_4}
    var_6 = module_2.Schema(var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = var_8.allow_null
    var_10 = 'allow_blank'
    var_11 = False
    var_12 = getattr(var_8, var_10, var_11)
    var_13 = var_9 or var_12



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 4/12 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'jinja2'



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 13/23 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'field1'
    var_5 = 'MockField'
    var_6 = ()
    var_7 = 'read_only'
    var_8 = 'title'
    var_9 = True
    var_10 = 'Field1'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {}



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 14/27 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'read_only_field'
    var_5 = 'regular_field'
    var_6 = 'MockField'
    var_7 = ()
    var_8 = 'read_only'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = ()
    var_12 = False
    var_13 = {var_8: var_12}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_form_constructor_serializes_values. Retrieved 6/9 statements.


import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'serialized_value'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = var_3.env.loader
    var_5 = var_1.FileSystemLoader
    var_6 = isinstance(var_4, var_5)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'some_package'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env.loader
    var_5 = var_1.PackageLoader
    var_6 = isinstance(var_4, var_5)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = 'some_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = var_4.env.loader
    var_6 = var_1.ChoiceLoader
    var_7 = isinstance(var_5, var_6)

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #2
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_2.Form(env=var_0, schema=var_2, values=var_5)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'test'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 10/16 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/custom/path'
    var_1 = 'mypackage'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = var_2.env.loader.loaders
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_2.env.loader.loaders[var_6]
    var_8 = 1
    var_9 = var_2.env.loader.loaders[var_8]

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_form_constructor_without_values. Retrieved 15/23 statements.
# Partially parsed test_form_constructor_with_values. Retrieved 19/27 statements.
# Partially parsed test_form_constructor_schema_serialize_called_with_values. Retrieved 16/29 statements.
# Partially parsed test_form_constructor_values_none_schema_serialize_called_with_none. Retrieved 13/26 statements.
# Partially parsed test_form_constructor_initial_state. Retrieved 16/27 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = None
    var_13 = lambda self, values: var_12
    var_14 = {var_11: var_13}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = 'key'
    var_13 = 'serialized_value'
    var_14 = {var_12: var_13}
    var_15 = lambda self, values: var_14
    var_16 = {var_11: var_15}
    var_17 = 'value'
    var_18 = {var_12: var_17}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = None
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = 'test'
    var_14 = 'data'
    var_15 = {var_13: var_14}

import builtins as module_0

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = module_0.object()
    assert var_9 is None
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'serialize'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, context: var_6
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'serialize'
    var_12 = {}
    var_13 = lambda self, values: var_12
    var_14 = {var_11: var_13}
    var_15 = 'data'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_str_with_no_errors. Retrieved 35/50 statements.
# Partially parsed test_str_with_errors. Retrieved 34/49 statements.
# Partially parsed test_str_with_read_only_field. Retrieved 35/50 statements.
# Partially parsed test_str_without_validate_called. Retrieved 34/48 statements.
# Partially parsed test_str_with_multiple_fields. Retrieved 41/58 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'autoescape'
    var_4 = 'MockTemplate'
    var_5 = ()
    var_6 = 'render'
    var_7 = 'field_name'
    var_8 = lambda self, context: f'<input name="{context[var_7]}">'
    var_9 = {var_6: var_8}
    var_10 = False
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'serialize'
    var_15 = 'validate_or_error'
    var_16 = 'name'
    var_17 = 'MockField'
    var_18 = ()
    var_19 = 'read_only'
    var_20 = 'title'
    var_21 = 'allow_null'
    var_22 = 'has_default'
    var_23 = 'format'
    var_24 = 'Name'
    var_25 = lambda : var_10
    var_26 = 'text'
    var_27 = {var_19: var_10, var_20: var_24, var_21: var_10, var_22: var_25, var_23: var_26}
    var_28 = {}
    var_29 = lambda self, values: values if values else var_28
    var_30 = None
    var_31 = lambda self, data: (data, var_30)
    var_32 = 'test'
    var_33 = {var_16: var_32}
    var_34 = {var_16: var_32}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'autoescape'
    var_4 = 'MockTemplate'
    var_5 = ()
    var_6 = 'render'
    var_7 = 'field_name'
    var_8 = lambda self, context: f'<input name="{context[var_7]}" class="error">'
    var_9 = {var_6: var_8}
    var_10 = False
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'serialize'
    var_15 = 'validate_or_error'
    var_16 = 'email'
    var_17 = 'MockField'
    var_18 = ()
    var_19 = 'read_only'
    var_20 = 'title'
    var_21 = 'allow_null'
    var_22 = 'has_default'
    var_23 = 'format'
    var_24 = 'Email'
    var_25 = lambda : var_10
    var_26 = {var_19: var_10, var_20: var_24, var_21: var_10, var_22: var_25, var_23: var_16}
    var_27 = {}
    var_28 = lambda self, values: values if values else var_27
    var_29 = 'Invalid email'
    var_30 = {var_16: var_29}
    var_31 = lambda self, data: (data, var_30)
    var_32 = 'bad'
    var_33 = {var_16: var_32}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'autoescape'
    var_4 = 'MockTemplate'
    var_5 = ()
    var_6 = 'render'
    var_7 = 'field_name'
    var_8 = lambda self, context: f'<input name="{context[var_7]}">'
    var_9 = {var_6: var_8}
    var_10 = False
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'serialize'
    var_15 = 'validate_or_error'
    var_16 = 'id'
    var_17 = 'MockField'
    var_18 = ()
    var_19 = 'read_only'
    var_20 = 'title'
    var_21 = 'allow_null'
    var_22 = 'has_default'
    var_23 = 'format'
    var_24 = True
    var_25 = 'ID'
    var_26 = lambda : var_10
    var_27 = 'number'
    var_28 = {var_19: var_24, var_20: var_25, var_21: var_10, var_22: var_26, var_23: var_27}
    var_29 = {}
    var_30 = lambda self, values: values if values else var_29
    var_31 = None
    var_32 = lambda self, data: (data, var_31)
    var_33 = {var_16: var_24}
    var_34 = {var_16: var_24}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'autoescape'
    var_4 = 'MockTemplate'
    var_5 = ()
    var_6 = 'render'
    var_7 = 'field_name'
    var_8 = lambda self, context: f'<input name="{context[var_7]}">'
    var_9 = {var_6: var_8}
    var_10 = False
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'serialize'
    var_15 = 'validate_or_error'
    var_16 = 'field'
    var_17 = 'MockField'
    var_18 = ()
    var_19 = 'read_only'
    var_20 = 'title'
    var_21 = 'allow_null'
    var_22 = 'has_default'
    var_23 = 'format'
    var_24 = 'Field'
    var_25 = lambda : var_10
    var_26 = 'text'
    var_27 = {var_19: var_10, var_20: var_24, var_21: var_10, var_22: var_25, var_23: var_26}
    var_28 = {}
    var_29 = lambda self, values: values if values else var_28
    var_30 = None
    var_31 = lambda self, data: (data, var_30)
    var_32 = 'initial'
    var_33 = {var_16: var_32}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'autoescape'
    var_4 = 'MockTemplate'
    var_5 = ()
    var_6 = 'render'
    var_7 = 'field_name'
    var_8 = lambda self, context: f'<input name="{context[var_7]}">'
    var_9 = {var_6: var_8}
    var_10 = False
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'serialize'
    var_15 = 'validate_or_error'
    var_16 = 'first'
    var_17 = 'second'
    var_18 = 'MockField'
    var_19 = ()
    var_20 = 'read_only'
    var_21 = 'title'
    var_22 = 'allow_null'
    var_23 = 'has_default'
    var_24 = 'format'
    var_25 = 'First'
    var_26 = lambda : var_10
    var_27 = 'text'
    var_28 = {var_20: var_10, var_21: var_25, var_22: var_10, var_23: var_26, var_24: var_27}
    var_29 = ()
    var_30 = 'Second'
    var_31 = lambda : var_10
    var_32 = {var_20: var_10, var_21: var_30, var_22: var_10, var_23: var_31, var_24: var_27}
    var_33 = {}
    var_34 = lambda self, values: values if values else var_33
    var_35 = None
    var_36 = lambda self, data: (data, var_35)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_16: var_37, var_17: var_38}
    var_40 = {var_16: var_37, var_17: var_38}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_directory'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_form_html_method_returns_markup. Retrieved 8/10 statements.
# Partially parsed test_form_html_method_renders_fields. Retrieved 13/15 statements.
# Partially parsed test_form_html_method_with_validation. Retrieved 16/18 statements.
# Partially parsed test_form_html_method_with_errors. Retrieved 16/18 statements.
# Partially parsed test_form_html_method_with_read_only_field. Retrieved 13/15 statements.
# Partially parsed test_form_html_method_with_different_field_types. Retrieved 31/33 statements.
# Partially parsed test_form_html_method_with_special_input_types. Retrieved 18/20 statements.
# Partially parsed test_form_html_method_with_password_field. Retrieved 12/14 statements.
# Partially parsed test_form_html_method_with_field_id_formatting. Retrieved 12/14 statements.
# Partially parsed test_form_html_method_with_required_attribute. Retrieved 12/14 statements.
# Partially parsed test_form_html_method_with_non_required_field. Retrieved 13/15 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = {}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_2, schema=var_4)
    var_6 = var_5.__html__()
    var_7 = str(var_6)
    assert var_7 == ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'username'
    var_6 = 'Username'
    var_7 = module_2.Field(title=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = var_10.__html__()
    var_12 = str(var_11)
    assert var_12 == '<input type="text" name="username" value="">'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = 'Email'
    var_7 = module_2.Field(title=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = 'test@example.com'
    var_12 = {var_5: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = var_10.__html__()
    var_15 = str(var_14)
    assert var_15 == '<input type="text" name="email" value="test@example.com">'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'age'
    var_6 = 'Age'
    var_7 = module_2.Field(title=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = 'invalid'
    var_12 = {var_5: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = var_10.__html__()
    var_15 = str(var_14)
    assert var_15 == '<input type="text" name="age" value="invalid">'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'id'
    var_6 = True
    var_7 = module_2.Field(read_only=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = var_10.__html__()
    var_12 = str(var_11)
    assert var_12 == ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text'
    var_12 = 'bool'
    var_13 = 'choice'
    var_14 = 'regular'
    var_15 = module_2.String(format=var_11)
    var_16 = module_2.Boolean()
    var_17 = 'a'
    var_18 = 'A'
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = module_2.Choice(choices=var_20)
    var_22 = module_2.Field()
    var_23 = {var_11: var_15, var_12: var_16, var_13: var_21, var_14: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_4.Form(env=var_10, schema=var_24)
    var_26 = var_25.__html__()
    var_27 = str(var_26)
    var_28 = str(var_26)
    var_29 = str(var_26)
    var_30 = str(var_26)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = 'date'
    var_7 = 'number'
    var_8 = module_2.String(format=var_5)
    var_9 = module_2.String(format=var_6)
    var_10 = module_2.String(format=var_7)
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = module_3.Schema(var_11)
    var_13 = module_4.Form(env=var_4, schema=var_12)
    var_14 = var_13.__html__()
    var_15 = str(var_14)
    var_16 = str(var_14)
    var_17 = str(var_14)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'password'
    var_6 = module_2.String(format=var_5)
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = var_9.__html__()
    var_11 = str(var_10)
    assert var_11 == '<input type="password" name="password" value="">'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'id="{{ field_id }}"'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'user_name'
    var_6 = module_2.Field()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = var_9.__html__()
    var_11 = str(var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{% if required %}required{% endif %}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'required_field'
    var_6 = module_2.Field()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = var_9.__html__()
    var_11 = str(var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{% if required %}required{% endif %}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'optional_field'
    var_6 = True
    var_7 = module_2.Field(allow_null=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = var_10.__html__()
    var_12 = str(var_11)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_sets_data_and_updates_values_and_errors. Retrieved 19/23 statements.
# Partially parsed test_validate_sets_errors_when_validation_fails. Retrieved 16/20 statements.
# Partially parsed test_validate_raises_assertion_error_if_called_twice. Retrieved 13/19 statements.
# Partially parsed test_validate_with_none_data. Retrieved 13/17 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'fields'
    var_5 = 'validate_or_error'
    var_6 = lambda self, v: v
    var_7 = {}
    var_8 = 'field'
    var_9 = 'new_value'
    var_10 = {var_8: var_9}
    var_11 = None
    var_12 = (var_10, var_11)
    var_13 = lambda self, data: var_12
    var_14 = {var_3: var_6, var_4: var_7, var_5: var_13}
    var_15 = 'old_value'
    var_16 = {var_8: var_15}
    var_17 = 'test_data'
    var_18 = {var_8: var_17}

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'fields'
    var_5 = 'validate_or_error'
    var_6 = lambda self, v: v
    var_7 = {}
    var_8 = None
    var_9 = 'field'
    var_10 = 'error'
    var_11 = {var_9: var_10}
    var_12 = (var_8, var_11)
    var_13 = lambda self, data: var_12
    var_14 = {var_3: var_6, var_4: var_7, var_5: var_13}
    var_15 = {}

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'fields'
    var_5 = 'validate_or_error'
    var_6 = lambda self, v: v
    var_7 = {}
    var_8 = {}
    var_9 = None
    var_10 = (var_8, var_9)
    var_11 = lambda self, data: var_10
    var_12 = {var_3: var_6, var_4: var_7, var_5: var_11}

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'fields'
    var_5 = 'validate_or_error'
    var_6 = lambda self, v: v
    var_7 = {}
    var_8 = {}
    var_9 = None
    var_10 = (var_8, var_9)
    var_11 = lambda self, data: var_10
    var_12 = {var_3: var_6, var_4: var_7, var_5: var_11}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_fields_without_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 19/34 statements.
# Partially parsed test_render_fields_with_no_data_and_no_errors. Retrieved 24/39 statements.
# Partially parsed test_render_fields_uses_data_when_errors_exist. Retrieved 26/41 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'name'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = 'title'
    var_17 = 'allow_null'
    var_18 = 'has_default'
    var_19 = False
    var_20 = 'Name'
    var_21 = lambda : var_19
    var_22 = {var_15: var_19, var_16: var_20, var_17: var_19, var_18: var_21}
    var_23 = 'John'
    var_24 = {var_12: var_23}
    var_25 = {var_12: var_23}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with error {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'email'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Email'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = 'invalid'
    var_25 = {var_13: var_24}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'id'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = True
    var_17 = {var_15: var_16}
    var_18 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'field1'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = 'title'
    var_17 = 'allow_null'
    var_18 = 'has_default'
    var_19 = False
    var_20 = 'Field1'
    var_21 = lambda : var_19
    var_22 = {var_15: var_19, var_16: var_20, var_17: var_19, var_18: var_21}
    var_23 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with value {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'age'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Age'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = 'not a number'
    var_25 = {var_13: var_24}



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Choice()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/select.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Boolean()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Object()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)



# Parsed testcases at query #11
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Object()
    var_4 = var_2.template_for_field(var_3)



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Object()
    var_4 = var_2.template_for_field(var_3)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Choice()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/select.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Boolean()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Object()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_raises_assertion_error_without_jinja2. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_form_constructor_serializes_values. Retrieved 8/9 statements.


import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'john'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 4/7 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 2/5 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'some_package'
    var_3 = module_0.Jinja2Forms(package=var_2)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = 'some_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_form_constructor_with_serializable_values. Retrieved 6/9 statements.


import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'serialized_value'



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_form_constructor_serializes_values. Retrieved 9/12 statements.


import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'MockObj'
    var_6 = ()
    var_7 = 'serialized_value'
    var_8 = {var_2: var_7}



# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 10/13 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 7/10 statements.
# Partially parsed test_form_constructor_with_serialized_values. Retrieved 10/13 statements.


import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_1.Form(env=var_0, schema=var_1)

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'serialized'
    var_5 = lambda self, x: {var_4: x}
    var_6 = {var_3: var_5}
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = None
    var_5 = lambda self, x: var_4
    var_6 = {var_3: var_5}

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'processed'
    var_5 = lambda self, x: {var_4: x}
    var_6 = {var_3: var_5}
    var_7 = 'raw'
    var_8 = 'data'
    var_9 = {var_7: var_8}



# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Choice()
    var_4 = var_2.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Boolean()
    var_4 = var_2.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'text'
    var_4 = module_1.String(format=var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'email'
    var_4 = module_1.String(format=var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Field()
    var_4 = var_2.template_for_field(var_3)
    assert var_4 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = module_1.Object()
    var_4 = var_2.template_for_field(var_3)



# Parsed testcases at query #32
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 8/9 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 4/5 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'serialized'
    var_6 = {var_2: var_5}
    var_7 = module_0.Form(env=var_0, schema=var_1, values=var_4)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Form(env=var_0, schema=var_1, values=var_2)



# Parsed testcases at query #34
#--------------------------






# Parsed testcases at query #35
#--------------------------

# Partially parsed test_init_asserts_jinja2_is_not_none. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Choice()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/select.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Boolean()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Object()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)



# Parsed testcases at query #37
#--------------------------






# Parsed testcases at query #38
#--------------------------

# Partially parsed test_form_constructor_without_values. Retrieved 6/9 statements.
# Partially parsed test_form_constructor_with_values. Retrieved 10/13 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 7/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = {}
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'serialized'
    var_5 = lambda self, x: {var_4: x}
    var_6 = {var_3: var_5}
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = {}
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = None



# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------






# Parsed testcases at query #41
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Choice()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/select.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Boolean()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = module_0.String(format=var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Object()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #43
#--------------------------






# Parsed testcases at query #44
#--------------------------






# Parsed testcases at query #45
#--------------------------

# Partially parsed test_form_constructor_serializes_values. Retrieved 6/9 statements.


import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_2.Schema(var_3)
    var_5 = 'serialized_value'



