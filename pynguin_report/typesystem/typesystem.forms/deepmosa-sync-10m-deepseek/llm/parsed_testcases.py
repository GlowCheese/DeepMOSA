####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_sets_values_and_errors. Retrieved 17/21 statements.
# Partially parsed test_validate_sets_errors_on_invalid_data. Retrieved 15/19 statements.
# Partially parsed test_validate_raises_assertion_if_called_twice. Retrieved 11/17 statements.
# Partially parsed test_validate_with_none_data. Retrieved 11/15 statements.
# Partially parsed test_validate_updates_data_attribute. Retrieved 14/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'validate_or_error'
    var_5 = lambda self, v: v
    var_6 = 'field'
    var_7 = 'validated_value'
    var_8 = {var_6: var_7}
    var_9 = None
    var_10 = (var_8, var_9)
    var_11 = lambda self, data: var_10
    var_12 = {var_3: var_5, var_4: var_11}
    var_13 = [var_1, var_2, var_12]
    var_14 = 'initial'
    var_15 = {var_6: var_14}
    var_16 = 'new'
    var_17 = {var_6: var_16}

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'validate_or_error'
    var_5 = lambda self, v: v
    var_6 = {}
    var_7 = 'field'
    var_8 = 'error_message'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = lambda self, data: var_10
    var_12 = {var_3: var_5, var_4: var_11}
    var_13 = [var_1, var_2, var_12]
    var_14 = 'invalid'
    var_15 = {var_7: var_14}

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'validate_or_error'
    var_5 = lambda self, v: v
    var_6 = {}
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda self, data: var_8
    var_10 = {var_3: var_5, var_4: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'validate_or_error'
    var_5 = lambda self, v: v
    var_6 = {}
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda self, data: var_8
    var_10 = {var_3: var_5, var_4: var_9}
    var_11 = [var_1, var_2, var_10]

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'validate_or_error'
    var_5 = lambda self, v: v
    var_6 = {}
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda self, data: var_8
    var_10 = {var_3: var_5, var_4: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_assertion_error_without_jinja2. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/path'])
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'some.package'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some.package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = var_2.env.loader.loaders
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_2.env.loader.loaders[var_6]
    var_8 = var_2.env.loader.loaders[0].searchpath
    var_9 = bool(var_2.env.loader.loaders[0].searchpath == ['/some/path'])
    assert var_9 is True
    var_10 = 1
    var_11 = var_2.env.loader.loaders[var_10]
    var_12 = var_2.env.loader.loaders[1].package_name
    assert var_12 == 'some.package'
    var_13 = var_2.env.loader.loaders[1].package_path
    assert var_13 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 27/43 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 19/34 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 25/40 statements.
# Partially parsed test_render_fields_with_empty_string_for_password. Retrieved 28/43 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx.get(var_6)}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Name'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'John'
    var_26 = {var_13: var_25}
    var_27 = {var_13: var_25}
    var_28 = 'Rendered name'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, ctx: f'Rendered {ctx.get(var_6)} with error {ctx.get(var_7)}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'email'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Email'
    var_23 = lambda : var_21
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_21, var_20: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = 'invalid'
    var_27 = {var_14: var_26}
    var_28 = 'Invalid email'
    var_29 = 'Rendered email with error Invalid email'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'Rendered {ctx.get(var_6)}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'id'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = True
    var_18 = {var_16: var_17}
    var_19 = [var_14, var_15, var_18]
    var_20 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx.get(var_6)} value={ctx.get(var_7)}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'field'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Field'
    var_23 = lambda : var_21
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_21, var_20: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = {}
    var_27 = 'value=None'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx.get(var_6)} value={ctx.get(var_7)}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'password'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = 'format'
    var_22 = False
    var_23 = 'Password'
    var_24 = lambda : var_22
    var_25 = {var_17: var_22, var_18: var_23, var_19: var_22, var_20: var_24, var_21: var_14}
    var_26 = [var_15, var_16, var_25]
    var_27 = 'secret'
    var_28 = {var_14: var_27}
    var_29 = {var_14: var_27}
    var_30 = 'value='



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test___str___with_no_errors_and_no_data. Retrieved 2/10 statements.
# Partially parsed test___str___with_errors_and_data. Retrieved 8/27 statements.
# Partially parsed test___str___with_read_only_field. Retrieved 4/14 statements.
# Partially parsed test___str___calls_render_fields. Retrieved 3/11 statements.


def test_case_0():
    var_0 = {}
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = {}
    var_3 = None
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = 'Error message'
    var_7 = '<input>'

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = {}
    var_2 = None
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = '<form></form>'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is not None)
    assert var_5 is True
    var_6 = var_3.env.loader
    var_7 = var_1.FileSystemLoader
    var_8 = isinstance(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'some_package'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is not None)
    assert var_5 is True
    var_6 = var_3.env.loader
    var_7 = var_1.PackageLoader
    var_8 = isinstance(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = 'some_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env is not None)
    assert var_6 is True
    var_7 = var_4.env.loader
    var_8 = var_1.ChoiceLoader
    var_9 = isinstance(var_7, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_input_type_for_field_with_all_known_formats. Retrieved 21/26 statements.


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'email'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'email'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'unknown'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Integer(**var_3)
    var_5 = var_2.input_type_for_field(var_4)
    assert var_5 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = ''
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = None
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'text'

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'color'
    var_4 = 'datetime'
    var_5 = 'date'
    var_6 = 'email'
    var_7 = 'hidden'
    var_8 = 'month'
    var_9 = 'number'
    var_10 = 'password'
    var_11 = 'range'
    var_12 = 'search'
    var_13 = 'tel'
    var_14 = 'text'
    var_15 = 'time'
    var_16 = 'url'
    var_17 = 'week'
    var_18 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'datetime'
    var_20 = 'datetime-local'



# Parsed testcases at query #8
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'title'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = 'testuser'
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    var_15 = 'testuser'
    var_16 = bool('testuser' in var_14)
    assert var_16 is True
    var_17 = 'Username'
    var_18 = bool('Username' in var_14)
    assert var_18 is True
    var_19 = 'type="text"'
    var_20 = bool('type="text"' in var_14)
    assert var_20 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'secret'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'secret'
    var_13 = bool('secret' not in var_11)
    assert var_13 is True
    var_14 = 'type="password"'
    var_15 = bool('type="password"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'test@example.com'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'test@example.com'
    var_13 = bool('test@example.com' in var_11)
    assert var_13 is True
    var_14 = 'type="email"'
    var_15 = bool('type="email"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = {}
    var_3 = module_1.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Integer(**var_8)
    var_10 = 25
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = '25'
    var_13 = bool('25' in var_11)
    assert var_13 is True
    var_14 = 'type="number"'
    var_15 = bool('type="number"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'required'
    var_12 = bool('required' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'optional'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'allow_null'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'status'
    var_2 = 'active'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'default'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'Invalid email address'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, error=var_10)
    var_12 = 'Invalid email address'
    var_13 = bool('Invalid email address' in var_11)
    assert var_13 is True

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
    var_9 = {}
    var_10 = module_1.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.Form(env=var_0, schema=var_13)
    var_15 = (var_2, var_3)
    var_16 = (var_5, var_6)
    var_17 = [var_15, var_16]
    var_18 = {}
    var_19 = module_1.Choice(choices=var_17, **var_18)
    var_20 = var_14.render_field(field_name=var_1, field=var_19, value=var_2)
    var_21 = 'Red'
    var_22 = bool('Red' in var_20)
    assert var_22 is True
    var_23 = 'Blue'
    var_24 = bool('Blue' in var_20)
    assert var_24 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Boolean(**var_8)
    var_10 = True
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'checkbox'
    var_13 = bool('checkbox' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'Some text'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'textarea'
    var_14 = bool('textarea' in var_12)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'token'
    var_2 = 'hidden'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'abc123'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'abc123'
    var_14 = bool('abc123' in var_12)
    assert var_14 is True
    var_15 = 'type="hidden"'
    var_16 = bool('type="hidden"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'id'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.Integer(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'read_only'
    var_11 = {var_10: var_2}
    var_12 = module_1.Integer(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12, value=var_2)
    assert var_13 == ''

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = var_8.render_field(field_name=var_1, field=var_10)
    var_12 = 'type="text"'
    var_13 = bool('type="text"' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'birthday'
    var_2 = 'date'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = '2023-01-01'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = '2023-01-01'
    var_14 = bool('2023-01-01' in var_12)
    assert var_14 is True
    var_15 = 'type="date"'
    var_16 = bool('type="date"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'meeting_time'
    var_2 = 'time'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = '14:30'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = '14:30'
    var_14 = bool('14:30' in var_12)
    assert var_14 is True
    var_15 = 'type="time"'
    var_16 = bool('type="time"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'website'
    var_2 = 'url'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'https://example.com'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'https://example.com'
    var_14 = bool('https://example.com' in var_12)
    assert var_14 is True
    var_15 = 'type="url"'
    var_16 = bool('type="url"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'query'
    var_2 = 'search'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'test'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'test'
    var_14 = bool('test' in var_12)
    assert var_14 is True
    var_15 = 'type="search"'
    var_16 = bool('type="search"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'phone'
    var_2 = 'tel'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = '+1234567890'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = '+1234567890'
    var_14 = bool('+1234567890' in var_12)
    assert var_14 is True
    var_15 = 'type="tel"'
    var_16 = bool('type="tel"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'expiry'
    var_2 = 'month'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = '2023-12'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = '2023-12'
    var_14 = bool('2023-12' in var_12)
    assert var_14 is True
    var_15 = 'type="month"'
    var_16 = bool('type="month"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'week'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = '2023-W01'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = '2023-W01'
    var_13 = bool('2023-W01' in var_11)
    assert var_13 is True
    var_14 = 'type="week"'
    var_15 = bool('type="week"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'volume'
    var_2 = 'range'
    var_3 = 'format'
    var_4 = {var_3: var_2}
    var_5 = module_1.Integer(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'format'
    var_11 = {var_10: var_2}
    var_12 = module_1.Integer(**var_11)
    var_13 = 50
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    var_15 = '50'
    var_16 = bool('50' in var_14)
    assert var_16 is True
    var_17 = 'type="range"'
    var_18 = bool('type="range"' in var_14)
    assert var_18 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'color'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = '#ff0000'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = '#ff0000'
    var_13 = bool('#ff0000' in var_11)
    assert var_13 is True
    var_14 = 'type="color"'
    var_15 = bool('type="color"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'event'
    var_2 = 'datetime'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_8.render



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Choice(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/select.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'

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
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 8/9 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 4/5 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is var_0)
    assert var_4 is True
    var_5 = var_2.schema
    var_6 = bool(var_2.schema is var_1)
    assert var_6 is True
    var_7 = var_2.values
    assert var_7 is None
    var_8 = var_2.errors
    assert var_8 is None
    var_9 = var_2._validate_called
    assert var_9 is False

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'serialized_value'
    var_6 = {var_2: var_5}
    var_7 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_8 = var_7.env
    var_9 = bool(var_7.env is var_0)
    assert var_9 is True
    var_10 = var_7.schema
    var_11 = bool(var_7.schema is var_1)
    assert var_11 is True
    var_12 = var_7.values
    var_13 = bool(var_7.values == var_6)
    assert var_13 is True
    var_14 = var_7.errors
    assert var_14 is None
    var_15 = var_7._validate_called
    assert var_15 is False

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Form(env=var_0, schema=var_1, values=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is var_0)
    assert var_5 is True
    var_6 = var_3.schema
    var_7 = bool(var_3.schema is var_1)
    assert var_7 is True
    var_8 = var_3.values
    assert var_8 is None
    var_9 = var_3.errors
    assert var_9 is None
    var_10 = var_3._validate_called
    assert var_10 is False



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 14/26 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'read_only_field'
    var_5 = 'normal_field'
    var_6 = 'MockField'
    var_7 = ()
    var_8 = 'read_only'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = ()
    var_13 = False
    var_14 = {var_8: var_13}
    var_15 = [var_6, var_12, var_14]
    var_16 = 'read_only_field'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 9/12 statements.
# Partially parsed test_form_constructor_schema_serialize_called. Retrieved 12/15 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is var_0)
    assert var_4 is True
    var_5 = var_2.schema
    var_6 = bool(var_2.schema is var_1)
    assert var_6 is True
    var_7 = var_2.values
    assert var_7 is None
    var_8 = var_2.errors
    assert var_8 is None
    var_9 = var_2._validate_called
    assert var_9 is False

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = None
    var_1 = 'serialized'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = 'MockSchema'
    var_5 = ()
    var_6 = 'serialize'
    var_7 = lambda self, x: var_3
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_if_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/path'])
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'mypackage'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'

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
    var_8 = var_2.env.loader.loaders[0].searchpath
    var_9 = bool(var_2.env.loader.loaders[0].searchpath == ['/custom/path'])
    assert var_9 is True
    var_10 = 1
    var_11 = var_2.env.loader.loaders[var_10]
    var_12 = var_2.env.loader.loaders[1].package_name
    assert var_12 == 'mypackage'
    var_13 = var_2.env.loader.loaders[1].package_path
    assert var_13 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 4/12 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'jinja2'



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env is var_0)
    assert var_6 is True
    var_7 = var_4.schema
    var_8 = bool(var_4.schema is var_3)
    assert var_8 is True
    var_9 = var_4.values
    assert var_9 is None
    var_10 = var_4.errors
    assert var_10 is None
    var_11 = var_4._validate_called
    assert var_11 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = module_3.Form(env=var_0, schema=var_5, values=var_7)
    var_9 = var_8.env
    var_10 = bool(var_8.env is var_0)
    assert var_10 is True
    var_11 = var_8.schema
    var_12 = bool(var_8.schema is var_5)
    assert var_12 is True
    var_13 = var_8.values
    var_14 = bool(var_8.values == {'test': 'value'})
    assert var_14 is True
    var_15 = var_8.errors
    assert var_15 is None
    var_16 = var_8._validate_called
    assert var_16 is False

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = None
    var_5 = module_2.Form(env=var_0, schema=var_3, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env is var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema is var_3)
    assert var_9 is True
    var_10 = var_5.values
    assert var_10 is None
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'extra'
    var_7 = 'value'
    var_8 = 'ignored'
    var_9 = {var_2: var_7, var_6: var_8}
    var_10 = module_3.Form(env=var_0, schema=var_5, values=var_9)
    var_11 = var_10.env
    var_12 = bool(var_10.env is var_0)
    assert var_12 is True
    var_13 = var_10.schema
    var_14 = bool(var_10.schema is var_5)
    assert var_14 is True
    var_15 = var_10.values
    var_16 = bool(var_10.values == {'test': 'value'})
    assert var_16 is True
    var_17 = var_10.errors
    assert var_17 is None
    var_18 = var_10._validate_called
    assert var_18 is False



# Parsed testcases at query #19
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Choice(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Boolean(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'text'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'email'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/input.html'

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
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Form(env=var_0, schema=var_1, values=var_2)
    var_4 = {}
    var_5 = module_1.Object(**var_4)
    var_6 = var_3.template_for_field(var_5)



# Parsed testcases at query #21
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
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.Environment()
    var_10 = 'test'
    var_11 = {var_10: var_8}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.Form(env=var_9, schema=var_13)
    var_15 = var_14.template_for_field(var_8)
    assert var_15 == 'forms/select.html'



# Parsed testcases at query #22
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env == var_0)
    assert var_6 is True
    var_7 = var_4.schema
    var_8 = bool(var_4.schema == var_3)
    assert var_8 is True
    var_9 = var_4.values
    var_10 = bool(var_4.values == {})
    assert var_10 is True
    var_11 = var_4.errors
    assert var_11 is None
    var_12 = var_4._validate_called
    assert var_12 is False

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_2.Form(env=var_0, schema=var_3, values=var_6)
    var_8 = var_7.env
    var_9 = bool(var_7.env == var_0)
    assert var_9 is True
    var_10 = var_7.schema
    var_11 = bool(var_7.schema == var_3)
    assert var_11 is True
    var_12 = var_7.values
    var_13 = bool(var_7.values == {})
    assert var_13 is True
    var_14 = var_7.errors
    assert var_14 is None
    var_15 = var_7._validate_called
    assert var_15 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'data'
    var_7 = {var_2: var_6}
    var_8 = module_3.Form(env=var_0, schema=var_5, values=var_7)
    var_9 = var_8.env
    var_10 = bool(var_8.env == var_0)
    assert var_10 is True
    var_11 = var_8.schema
    var_12 = bool(var_8.schema == var_5)
    assert var_12 is True
    var_13 = var_8.values
    var_14 = bool(var_8.values == {'test': 'data'})
    assert var_14 is True
    var_15 = var_8.errors
    assert var_15 is None
    var_16 = var_8._validate_called
    assert var_16 is False

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = None
    var_5 = module_2.Form(env=var_0, schema=var_3, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_3)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = bool(var_5.values == {})
    assert var_11 is True
    var_12 = var_5.errors
    assert var_12 is None
    var_13 = var_5._validate_called
    assert var_13 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 36/53 statements.


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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'serialize'
    var_14 = 'validate_or_error'
    var_15 = 'field1'
    var_16 = 'field2'
    var_17 = 'MockField'
    var_18 = ()
    var_19 = 'read_only'
    var_20 = 'title'
    var_21 = 'allow_null'
    var_22 = 'has_default'
    var_23 = True
    var_24 = 'Field1'
    var_25 = False
    var_26 = lambda : var_25
    var_27 = {var_19: var_23, var_20: var_24, var_21: var_25, var_22: var_26}
    var_28 = [var_17, var_18, var_27]
    var_29 = ()
    var_30 = 'Field2'
    var_31 = lambda : var_25
    var_32 = {var_19: var_25, var_20: var_30, var_21: var_25, var_22: var_31}
    var_33 = [var_17, var_29, var_32]
    var_34 = lambda self, values: values
    var_35 = None
    var_36 = lambda self, data: (data, var_35)
    var_37 = {}
    var_38 = {}
    var_39 = 'field1'



# Parsed testcases at query #24
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_fields_with_no_errors_and_no_data. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors_and_data. Retrieved 29/45 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 27/42 statements.
# Partially parsed test_render_fields_with_empty_string_value_for_password. Retrieved 28/43 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'rendered_{ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Name'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'John'
    var_26 = {var_13: var_25}
    var_27 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, ctx: f'rendered_{ctx[var_6]}_error_{ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'email'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Email'
    var_23 = lambda : var_21
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_21, var_20: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = 'test@example.com'
    var_27 = {var_14: var_26}
    var_28 = 'invalid'
    var_29 = {var_14: var_28}
    var_30 = 'Invalid email'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'rendered_{ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'id'
    var_14 = 'name'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = True
    var_22 = 'ID'
    var_23 = False
    var_24 = lambda : var_23
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = ()
    var_28 = 'Name'
    var_29 = lambda : var_23
    var_30 = {var_17: var_23, var_18: var_28, var_19: var_23, var_20: var_29}
    var_31 = [var_15, var_27, var_30]
    var_32 = 'John'
    var_33 = {var_13: var_21, var_14: var_32}
    var_34 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'rendered_{ctx[var_6]}_value_{ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'comment'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Comment'
    var_23 = True
    var_24 = lambda : var_21
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = None
    var_28 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'rendered_{ctx[var_6]}_value_{ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockField'
    var_12 = ()
    var_13 = 'read_only'
    var_14 = 'title'
    var_15 = 'allow_null'
    var_16 = 'has_default'
    var_17 = 'format'
    var_18 = False
    var_19 = 'Password'
    var_20 = lambda : var_18
    var_21 = 'password'
    var_22 = {var_13: var_18, var_14: var_19, var_15: var_18, var_16: var_20, var_17: var_21}
    var_23 = [var_11, var_12, var_22]
    var_24 = 'MockSchema'
    var_25 = ()
    var_26 = 'fields'
    var_27 = 'secret'
    var_28 = {var_21: var_27}
    var_29 = {}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 14/26 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'read_only_field'
    var_5 = 'normal_field'
    var_6 = 'MockField'
    var_7 = ()
    var_8 = 'read_only'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = ()
    var_13 = False
    var_14 = {var_8: var_13}
    var_15 = [var_6, var_12, var_14]
    var_16 = 'read_only_field'



# Parsed testcases at query #27
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Form(env=var_0, schema=var_1, values=var_2)
    var_4 = {}
    var_5 = module_1.Object(**var_4)
    var_6 = var_3.template_for_field(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 4/12 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'jinja2'



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env == var_0)
    assert var_6 is True
    var_7 = var_4.schema
    var_8 = bool(var_4.schema == var_3)
    assert var_8 is True
    var_9 = var_4.values
    var_10 = bool(var_4.values == {})
    assert var_10 is True
    var_11 = var_4.errors
    assert var_11 is None
    var_12 = var_4._validate_called
    assert var_12 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = module_3.Form(env=var_0, schema=var_5, values=var_7)
    var_9 = var_8.env
    var_10 = bool(var_8.env == var_0)
    assert var_10 is True
    var_11 = var_8.schema
    var_12 = bool(var_8.schema == var_5)
    assert var_12 is True
    var_13 = var_8.values
    var_14 = bool(var_8.values == {'test': 'value'})
    assert var_14 is True
    var_15 = var_8.errors
    assert var_15 is None
    var_16 = var_8._validate_called
    assert var_16 is False

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = None
    var_5 = module_2.Form(env=var_0, schema=var_3, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_3)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = bool(var_5.values == {})
    assert var_11 is True
    var_12 = var_5.errors
    assert var_12 is None
    var_13 = var_5._validate_called
    assert var_13 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'extra'
    var_7 = 'value'
    var_8 = 'ignored'
    var_9 = {var_2: var_7, var_6: var_8}
    var_10 = module_3.Form(env=var_0, schema=var_5, values=var_9)
    var_11 = var_10.env
    var_12 = bool(var_10.env == var_0)
    assert var_12 is True
    var_13 = var_10.schema
    var_14 = bool(var_10.schema == var_5)
    assert var_14 is True
    var_15 = var_10.values
    var_16 = bool(var_10.values == {'test': 'value'})
    assert var_16 is True
    var_17 = var_10.errors
    assert var_17 is None
    var_18 = var_10._validate_called
    assert var_18 is False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_jinja2_not_installed_raises_assertion. Retrieved 4/12 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'jinja2'



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 9/12 statements.
# Partially parsed test_form_constructor_schema_serialize_called. Retrieved 12/15 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env == var_0)
    assert var_4 is True
    var_5 = var_2.schema
    var_6 = bool(var_2.schema == var_1)
    assert var_6 is True
    var_7 = var_2.values
    assert var_7 is None
    var_8 = var_2.errors
    assert var_8 is None
    var_9 = var_2._validate_called
    assert var_9 is False

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = None
    var_1 = 'serialized'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = 'MockSchema'
    var_5 = ()
    var_6 = 'serialize'
    var_7 = lambda self, x: var_3
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}



# Parsed testcases at query #35
#--------------------------






# Parsed testcases at query #36
#--------------------------






# Parsed testcases at query #37
#--------------------------

# Partially parsed test_render_field_password_input_type_sets_value_to_empty_string. Retrieved 6/23 statements.


def test_case_0():
    var_0 = '<input>'
    var_1 = False
    var_2 = 'password'
    var_3 = 'forms/input.html'
    var_4 = 'password_field'
    var_5 = 'secret123'



# Parsed testcases at query #38
#--------------------------






# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = False
    var_4 = 'title'
    var_5 = 'allow_null'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.String(**var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.Form(env=var_0, schema=var_10)
    var_12 = var_10.fields[var_1]
    var_13 = None
    var_14 = var_11.render_field(field_name=var_1, field=var_12, value=var_13, error=var_13)
    var_15 = 'required'
    var_16 = bool('required' in var_14)
    assert var_16 is True
    var_17 = 'Username'
    var_18 = bool('Username' in var_14)
    assert var_18 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = True
    var_4 = 'title'
    var_5 = 'allow_null'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.String(**var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.Form(env=var_0, schema=var_10)
    var_12 = var_10.fields[var_1]
    var_13 = None
    var_14 = var_11.render_field(field_name=var_1, field=var_12, value=var_13, error=var_13)
    var_15 = 'required'
    var_16 = bool('required' not in var_14)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = 'guest'
    var_4 = 'title'
    var_5 = 'default'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.String(**var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.Form(env=var_0, schema=var_10)
    var_12 = var_10.fields[var_1]
    var_13 = None
    var_14 = var_11.render_field(field_name=var_1, field=var_12, value=var_13, error=var_13)
    var_15 = 'required'
    var_16 = bool('required' not in var_14)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = 'secret'
    var_10 = None
    var_11 = var_7.render_field(field_name=var_1, field=var_8, value=var_9, error=var_10)
    var_12 = 'type="password"'
    var_13 = bool('type="password"' in var_11)
    assert var_13 is True
    var_14 = 'value'
    var_15 = bool('value' not in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = 'test@example.com'
    var_10 = None
    var_11 = var_7.render_field(field_name=var_1, field=var_8, value=var_9, error=var_10)
    var_12 = 'type="email"'
    var_13 = bool('type="email"' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = 'number'
    var_3 = 'format'
    var_4 = {var_3: var_2}
    var_5 = module_1.Integer(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = 25
    var_12 = None
    var_13 = var_9.render_field(field_name=var_1, field=var_10, value=var_11, error=var_12)
    var_14 = 'type="number"'
    var_15 = bool('type="number"' in var_13)
    assert var_15 is True

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
    var_9 = {}
    var_10 = module_1.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.Form(env=var_0, schema=var_13)
    var_15 = var_13.fields[var_1]
    var_16 = None
    var_17 = var_14.render_field(field_name=var_1, field=var_15, value=var_2, error=var_16)
    var_18 = 'select'
    var_19 = bool('select' in var_17)
    assert var_19 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = True
    var_10 = None
    var_11 = var_7.render_field(field_name=var_1, field=var_8, value=var_9, error=var_10)
    var_12 = 'checkbox'
    var_13 = bool('checkbox' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = 'Some text'
    var_11 = None
    var_12 = var_8.render_field(field_name=var_1, field=var_9, value=var_10, error=var_11)
    var_13 = 'textarea'
    var_14 = bool('textarea' in var_12)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = None
    var_10 = 'Invalid username'
    var_11 = var_7.render_field(field_name=var_1, field=var_8, value=var_9, error=var_10)
    var_12 = 'Invalid username'
    var_13 = bool('Invalid username' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'user_name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = None
    var_10 = var_7.render_field(field_name=var_1, field=var_8, value=var_9, error=var_9)
    var_11 = 'id="user-name"'
    var_12 = bool('id="user-name"' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'User Name'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = None
    var_12 = var_9.render_field(field_name=var_1, field=var_10, value=var_11, error=var_11)
    var_13 = 'User Name'
    var_14 = bool('User Name' in var_12)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = None
    var_10 = var_7.render_field(field_name=var_1, field=var_8, value=var_9, error=var_9)
    var_11 = 'username'
    var_12 = bool('username' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'id'
    var_2 = 'name'
    var_3 = True
    var_4 = 'read_only'
    var_5 = {var_4: var_3}
    var_6 = module_1.Integer(**var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.Form(env=var_0, schema=var_11)
    var_13 = var_12.render_fields()
    var_14 = 'id'
    var_15 = bool('id' not in var_13)
    assert var_15 is True
    var_16 = 'name'
    var_17 = bool('name' in var_13)
    assert var_17 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = None
    var_11 = var_8.render_field(field_name=var_1, field=var_9, value=var_10, error=var_10)
    var_12 = 'type="text"'
    var_13 = bool('type="text"' in var_11)
    assert var_13 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_form_html_method_returns_markup. Retrieved 1/6 statements.
# Partially parsed test_form_html_method_returns_same_as_str. Retrieved 1/6 statements.
# Partially parsed test_form_html_method_with_values. Retrieved 4/9 statements.
# Partially parsed test_form_html_method_after_validate. Retrieved 2/8 statements.
# Partially parsed test_form_html_method_with_errors. Retrieved 4/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'invalid'
    var_3 = 'data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_form_html_method_returns_markup. Retrieved 1/6 statements.
# Partially parsed test_form_html_method_returns_same_as_str. Retrieved 1/6 statements.
# Partially parsed test_form_html_method_with_validation. Retrieved 2/8 statements.
# Partially parsed test_form_html_method_with_values. Retrieved 4/9 statements.
# Partially parsed test_form_html_method_with_errors. Retrieved 4/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'invalid'
    var_3 = 'data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___str___calls_render_fields. Retrieved 1/8 statements.
# Partially parsed test___str___returns_string. Retrieved 1/8 statements.
# Partially parsed test___str___with_no_fields. Retrieved 1/7 statements.
# Partially parsed test___str___with_fields. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'rendered html'

def test_case_0():
    var_0 = 'some html'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '<input>'



# Parsed testcases at query #4
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'title'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = 'testuser'
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    var_15 = 'testuser'
    var_16 = bool('testuser' in var_14)
    assert var_16 is True
    var_17 = 'Username'
    var_18 = bool('Username' in var_14)
    assert var_18 is True
    var_19 = 'type="text"'
    var_20 = bool('type="text"' in var_14)
    assert var_20 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'secret'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'secret'
    var_13 = bool('secret' not in var_11)
    assert var_13 is True
    var_14 = 'type="password"'
    var_15 = bool('type="password"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'user@example.com'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'user@example.com'
    var_13 = bool('user@example.com' in var_11)
    assert var_13 is True
    var_14 = 'type="email"'
    var_15 = bool('type="email"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = {}
    var_3 = module_1.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Integer(**var_8)
    var_10 = 25
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = '25'
    var_13 = bool('25' in var_11)
    assert var_13 is True
    var_14 = 'type="number"'
    var_15 = bool('type="number"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'required'
    var_12 = bool('required' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'optional'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'allow_null'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'status'
    var_2 = 'active'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'default'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'invalid'
    var_11 = 'Invalid email'
    var_12 = var_7.render_field(field_name=var_1, field=var_9, value=var_10, error=var_11)
    var_13 = 'invalid'
    var_14 = bool('invalid' in var_12)
    assert var_14 is True
    var_15 = 'Invalid email'
    var_16 = bool('Invalid email' in var_12)
    assert var_16 is True

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
    var_9 = {}
    var_10 = module_1.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.Form(env=var_0, schema=var_13)
    var_15 = (var_2, var_3)
    var_16 = (var_5, var_6)
    var_17 = [var_15, var_16]
    var_18 = {}
    var_19 = module_1.Choice(choices=var_17, **var_18)
    var_20 = var_14.render_field(field_name=var_1, field=var_19, value=var_2)
    var_21 = 'Red'
    var_22 = bool('Red' in var_20)
    assert var_22 is True
    var_23 = 'Blue'
    var_24 = bool('Blue' in var_20)
    assert var_24 is True
    var_25 = 'select'
    var_26 = bool('select' in var_20)
    assert var_26 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Boolean(**var_8)
    var_10 = True
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'checkbox'
    var_13 = bool('checkbox' in var_11)
    assert var_13 is True
    var_14 = 'checked'
    var_15 = bool('checked' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'Some text'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'textarea'
    var_14 = bool('textarea' in var_12)
    assert var_14 is True
    var_15 = 'Some text'
    var_16 = bool('Some text' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'id'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.Integer(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'read_only'
    var_11 = {var_10: var_2}
    var_12 = module_1.Integer(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12, value=var_2)
    var_14 = bool('readonly' in var_13 or 'disabled' in var_13)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'birthdate'
    var_2 = 'date'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = '2023-01-01'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = '2023-01-01'
    var_14 = bool('2023-01-01' in var_12)
    assert var_14 is True
    var_15 = 'type="date"'
    var_16 = bool('type="date"' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'test'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'type="text"'
    var_14 = bool('type="text"' in var_12)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = None
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'value=""'
    var_13 = bool('value=""' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'first_name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'id="first-name"'
    var_12 = bool('id="first-name"' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'full_name'
    var_2 = 'Full Name'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'title'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'Full Name'
    var_15 = bool('Full Name' in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email_address'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'email_address'
    var_12 = bool('email_address' in var_10)
    assert var_12 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'a'
    var_4 = 'A'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Choice(choices=var_6, **var_7)
    var_9 = var_2.template_for_field(var_8)
    assert var_9 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Boolean(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'text'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'email'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Integer(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = {}
    var_5 = module_1.Object(properties=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Choice(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Boolean(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'text'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'email'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.String(**var_3)
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
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 10/13 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 7/10 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env == var_0)
    assert var_4 is True
    var_5 = var_2.schema
    var_6 = bool(var_2.schema == var_1)
    assert var_6 is True
    var_7 = var_2.values
    assert var_7 is None
    var_8 = var_2.errors
    assert var_8 is None
    var_9 = var_2._validate_called
    assert var_9 is False

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = 'serialized'
    var_5 = lambda self, x: {var_4: x}
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = None
    var_5 = lambda self, x: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 28/44 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_no_validation_called. Retrieved 25/39 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 26/41 statements.


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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Name'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'John'
    var_26 = {var_13: var_25}
    var_27 = {var_13: var_25}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} error: {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'email'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Email'
    var_23 = lambda : var_21
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_21, var_20: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = ''
    var_27 = {var_14: var_26}
    var_28 = {var_14: var_26}
    var_29 = 'Invalid email'

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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'id'
    var_14 = 'name'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = True
    var_22 = 'ID'
    var_23 = False
    var_24 = lambda : var_23
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = ()
    var_28 = 'Name'
    var_29 = lambda : var_23
    var_30 = {var_17: var_23, var_18: var_28, var_19: var_23, var_20: var_29}
    var_31 = [var_15, var_27, var_30]
    var_32 = 'John'
    var_33 = {var_13: var_21, var_14: var_32}
    var_34 = {var_13: var_21, var_14: var_32}

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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Name'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'John'
    var_26 = {var_13: var_25}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} value: {ctx[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'comment'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Comment'
    var_23 = True
    var_24 = lambda : var_21
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 27/43 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_empty_string_for_password. Retrieved 28/43 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'Rendered {context.get(var_6)}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Name'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'John'
    var_26 = {var_13: var_25}
    var_27 = {var_13: var_25}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, context: f'Rendered {context.get(var_6)} with error {context.get(var_7)}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'email'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Email'
    var_23 = lambda : var_21
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_21, var_20: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = 'invalid'
    var_27 = {var_14: var_26}
    var_28 = 'Invalid email'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'Rendered {context.get(var_6)}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'id'
    var_14 = 'name'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = True
    var_22 = 'ID'
    var_23 = False
    var_24 = lambda : var_23
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = ()
    var_28 = 'Name'
    var_29 = lambda : var_23
    var_30 = {var_17: var_23, var_18: var_28, var_19: var_23, var_20: var_29}
    var_31 = [var_15, var_27, var_30]
    var_32 = 'John'
    var_33 = {var_13: var_21, var_14: var_32}
    var_34 = {var_14: var_32}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, context: f'Rendered {context.get(var_6)} with value {context.get(var_7)}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'comment'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Comment'
    var_23 = True
    var_24 = lambda : var_21
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, context: f"Rendered {context.get(var_6)} with value '{context.get(var_7)}'"
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'password'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = 'format'
    var_22 = False
    var_23 = 'Password'
    var_24 = lambda : var_22
    var_25 = {var_17: var_22, var_18: var_23, var_19: var_22, var_20: var_24, var_21: var_14}
    var_26 = [var_15, var_16, var_25]
    var_27 = 'secret'
    var_28 = {var_14: var_27}
    var_29 = {var_14: var_27}



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_template_for_field_asserts_not_object.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is not None)
    assert var_5 is True
    var_6 = var_3.env.loader
    var_7 = var_1.FileSystemLoader
    var_8 = isinstance(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'some_package'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is not None)
    assert var_5 is True
    var_6 = var_3.env.loader
    var_7 = var_1.PackageLoader
    var_8 = isinstance(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = 'some_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env is not None)
    assert var_6 is True
    var_7 = var_4.env.loader
    var_8 = var_1.ChoiceLoader
    var_9 = isinstance(var_7, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Choice(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Boolean(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'text'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = 'email'
    var_4 = {}
    var_5 = module_1.String(format=var_3, **var_4)
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/input.html'

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
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Choice(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/select.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'

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
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_init_with_jinja2_not_installed_raises_assertion. Retrieved 4/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is not None)
    assert var_5 is True
    var_6 = var_3.env.loader
    var_7 = var_1.FileSystemLoader
    var_8 = isinstance(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'some_package'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env
    var_5 = bool(var_3.env is not None)
    assert var_5 is True
    var_6 = var_3.env.loader
    var_7 = var_1.PackageLoader
    var_8 = isinstance(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/some/path'
    var_3 = 'some_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env is not None)
    assert var_6 is True
    var_7 = var_4.env.loader
    var_8 = var_1.ChoiceLoader
    var_9 = isinstance(var_7, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = None
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'text'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.template_for_field(var_7)
    assert var_8 == 'forms/textarea.html'



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 14/27 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'read_only_field'
    var_5 = 'normal_field'
    var_6 = 'MockField'
    var_7 = ()
    var_8 = 'read_only'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = ()
    var_13 = False
    var_14 = {var_8: var_13}
    var_15 = [var_6, var_12, var_14]
    var_16 = 'normal_field'
    var_17 = 'read_only_field'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 4/6 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2forms.jinja2'
    var_1 = None
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'a'
    var_6 = 'A'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_3.Choice(choices=var_8, **var_9)
    var_11 = var_4.template_for_field(var_10)
    assert var_11 == 'forms/select.html'



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = var_4.env
    var_6 = bool(var_4.env == var_0)
    assert var_6 is True
    var_7 = var_4.schema
    var_8 = bool(var_4.schema == var_3)
    assert var_8 is True
    var_9 = var_4.values
    assert var_9 is None
    var_10 = var_4.errors
    assert var_10 is None
    var_11 = var_4._validate_called
    assert var_11 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = module_3.Form(env=var_0, schema=var_5, values=var_7)
    var_9 = var_8.env
    var_10 = bool(var_8.env == var_0)
    assert var_10 is True
    var_11 = var_8.schema
    var_12 = bool(var_8.schema == var_5)
    assert var_12 is True
    var_13 = var_8.values
    var_14 = bool(var_8.values == {'test': 'value'})
    assert var_14 is True
    var_15 = var_8.errors
    assert var_15 is None
    var_16 = var_8._validate_called
    assert var_16 is False

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = None
    var_5 = module_2.Form(env=var_0, schema=var_3, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_3)
    assert var_9 is True
    var_10 = var_5.values
    assert var_10 is None
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = 'original'
    var_7 = {var_2: var_6}
    var_8 = module_3.Form(env=var_0, schema=var_5, values=var_7)
    var_9 = var_8.values
    var_10 = bool(var_8.values == {'test': 'original'})
    assert var_10 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = None
    var_3 = module_1.Form(env=var_2, schema=var_2)
    var_4 = var_3.template_for_field(var_1)
    assert var_4 == 'forms/checkbox.html'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_directory'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_field_password_input_type_sets_value_to_empty_string. Retrieved 6/22 statements.


def test_case_0():
    var_0 = '<input>'
    var_1 = False
    var_2 = 'password'
    var_3 = 'forms/input.html'
    var_4 = 'password_field'
    var_5 = 'secret123'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_2.searchpath
    var_4 = bool(var_2.searchpath == ['/some/directory'])
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_2.package_name
    assert var_3 == 'some_package'
    var_4 = var_2.package_path
    assert var_4 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/directory'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = var_3.loaders
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_3.loaders[var_6]
    var_8 = var_3.loaders[0].searchpath
    var_9 = bool(var_3.loaders[0].searchpath == ['/some/directory'])
    assert var_9 is True
    var_10 = 1
    var_11 = var_3.loaders[var_10]
    var_12 = var_3.loaders[1].package_name
    assert var_12 == 'some_package'
    var_13 = var_3.loaders[1].package_path
    assert var_13 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 29/47 statements.


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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = True
    var_22 = 'Field1'
    var_23 = False
    var_24 = lambda : var_23
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = ()
    var_28 = 'Field2'
    var_29 = lambda : var_23
    var_30 = {var_17: var_23, var_18: var_28, var_19: var_23, var_20: var_29}
    var_31 = [var_15, var_27, var_30]
    var_32 = 'field1'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 29/45 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 27/42 statements.
# Partially parsed test_render_fields_with_empty_values_and_errors. Retrieved 28/44 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'Rendered {context[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'name'
    var_14 = 'MockField'
    var_15 = ()
    var_16 = 'read_only'
    var_17 = 'title'
    var_18 = 'allow_null'
    var_19 = 'has_default'
    var_20 = False
    var_21 = 'Name'
    var_22 = lambda : var_20
    var_23 = {var_16: var_20, var_17: var_21, var_18: var_20, var_19: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'John'
    var_26 = {var_13: var_25}
    var_27 = {var_13: var_25}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'error'
    var_8 = lambda self, context: f'Rendered {context[var_6]} with error {context[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'email'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Email'
    var_23 = lambda : var_21
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_21, var_20: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = 'test'
    var_27 = {var_14: var_26}
    var_28 = 'invalid'
    var_29 = {var_14: var_28}
    var_30 = 'Invalid email'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'Rendered {context[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'id'
    var_14 = 'name'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = True
    var_22 = 'ID'
    var_23 = False
    var_24 = lambda : var_23
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = ()
    var_28 = 'Name'
    var_29 = lambda : var_23
    var_30 = {var_17: var_23, var_18: var_28, var_19: var_23, var_20: var_29}
    var_31 = [var_15, var_27, var_30]
    var_32 = 'John'
    var_33 = {var_13: var_21, var_14: var_32}
    var_34 = {var_13: var_21, var_14: var_32}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = lambda self, context: f'Rendered {context[var_6]} with value {context[var_7]}'
    var_9 = {var_5: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'comment'
    var_15 = 'MockField'
    var_16 = ()
    var_17 = 'read_only'
    var_18 = 'title'
    var_19 = 'allow_null'
    var_20 = 'has_default'
    var_21 = False
    var_22 = 'Comment'
    var_23 = True
    var_24 = lambda : var_21
    var_25 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_24}
    var_26 = [var_15, var_16, var_25]
    var_27 = None
    var_28 = {var_14: var_27}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = 'value'
    var_8 = 'error'
    var_9 = lambda self, context: f'Rendered {context[var_6]} with value {context[var_7]} and error {context[var_8]}'
    var_10 = {var_5: var_9}
    var_11 = [var_3, var_4, var_10]
    var_12 = 'MockSchema'
    var_13 = ()
    var_14 = 'fields'
    var_15 = 'password'
    var_16 = 'MockField'
    var_17 = ()
    var_18 = 'read_only'
    var_19 = 'title'
    var_20 = 'allow_null'
    var_21 = 'has_default'
    var_22 = False
    var_23 = 'Password'
    var_24 = lambda : var_22
    var_25 = {var_18: var_22, var_19: var_23, var_20: var_22, var_21: var_24}
    var_26 = [var_16, var_17, var_25]
    var_27 = {}
    var_28 = {}
    var_29 = 'Required field'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_when_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/dir'])
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'some.package'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/dir'
    var_1 = 'some.package'
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
    var_0 = '/some/dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_requires_jinja2_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/path'])
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'mypackage'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'

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
    var_8 = var_2.env.loader.loaders[0].searchpath
    var_9 = bool(var_2.env.loader.loaders[0].searchpath == ['/custom/path'])
    assert var_9 is True
    var_10 = 1
    var_11 = var_2.env.loader.loaders[var_10]
    var_12 = var_2.env.loader.loaders[1].package_name
    assert var_12 == 'mypackage'
    var_13 = var_2.env.loader.loaders[1].package_path
    assert var_13 == 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_render_field_password_input_type_sets_value_to_empty_string. Retrieved 6/23 statements.


def test_case_0():
    var_0 = '<input>'
    var_1 = False
    var_2 = 'password'
    var_3 = 'forms/input.html'
    var_4 = 'password_field'
    var_5 = 'secret123'



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Boolean(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'



# Parsed testcases at query #34
#--------------------------






# Parsed testcases at query #35
#--------------------------






# Parsed testcases at query #36
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_render_field_password_input_type_sets_value_to_empty_string. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'password'
    var_1 = 'forms/input.html'
    var_2 = 'password_field'
    var_3 = 'secret'
    var_4 = 0



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_form_constructor_without_values. Retrieved 16/25 statements.
# Partially parsed test_form_constructor_with_values. Retrieved 20/29 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 17/26 statements.


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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'serialize'
    var_14 = {}
    var_15 = lambda self, values: values
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = [var_10, var_11, var_16]

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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'serialize'
    var_14 = {}
    var_15 = 'serialized'
    var_16 = lambda self, values: {var_15: values}
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = [var_10, var_11, var_17]
    var_19 = 'key'
    var_20 = 'value'
    var_21 = {var_19: var_20}

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
    var_9 = [var_3, var_4, var_8]
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'fields'
    var_13 = 'serialize'
    var_14 = {}
    var_15 = None
    var_16 = lambda self, values: var_15
    var_17 = {var_12: var_14, var_13: var_16}
    var_18 = [var_10, var_11, var_17]



# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------

# Partially parsed test_template_for_field_with_choice_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_boolean_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_string_field_with_text_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_string_field_with_other_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_field_without_special_type. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_object_field_raises_assertion. Retrieved 6/10 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Choice'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Boolean'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'String'
    var_3 = ()
    var_4 = 'format'
    var_5 = 'text'
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'String'
    var_3 = ()
    var_4 = 'format'
    var_5 = 'email'
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Field'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Object'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------






# Parsed testcases at query #42
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 9/12 statements.
# Partially parsed test_form_constructor_schema_serialize_called. Retrieved 12/15 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is var_0)
    assert var_4 is True
    var_5 = var_2.schema
    var_6 = bool(var_2.schema is var_1)
    assert var_6 is True
    var_7 = var_2.values
    assert var_7 is None
    var_8 = var_2.errors
    assert var_8 is None
    var_9 = var_2._validate_called
    assert var_9 is False

def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'serialize'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = None
    var_1 = 'serialized'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = 'MockSchema'
    var_5 = ()
    var_6 = 'serialize'
    var_7 = lambda self, x: var_3
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}



# Parsed testcases at query #43
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'username'
    var_2 = 'Username'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'title'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = 'testuser'
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    var_15 = 'testuser'
    var_16 = bool('testuser' in var_14)
    assert var_16 is True
    var_17 = 'Username'
    var_18 = bool('Username' in var_14)
    assert var_18 is True
    var_19 = 'type="text"'
    var_20 = bool('type="text"' in var_14)
    assert var_20 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'secret'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'secret'
    var_13 = bool('secret' not in var_11)
    assert var_13 is True
    var_14 = 'type="password"'
    var_15 = bool('type="password"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'user@example.com'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'user@example.com'
    var_13 = bool('user@example.com' in var_11)
    assert var_13 is True
    var_14 = 'type="email"'
    var_15 = bool('type="email"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = {}
    var_3 = module_1.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Integer(**var_8)
    var_10 = 25
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = '25'
    var_13 = bool('25' in var_11)
    assert var_13 is True
    var_14 = 'type="number"'
    var_15 = bool('type="number"' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'required'
    var_12 = bool('required' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'optional'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'allow_null'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'status'
    var_2 = 'active'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'default'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'email'
    var_2 = {}
    var_3 = module_1.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(format=var_1, **var_8)
    var_10 = 'Invalid email address'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, error=var_10)
    var_12 = 'Invalid email address'
    var_13 = bool('Invalid email address' in var_11)
    assert var_13 is True

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
    var_9 = {}
    var_10 = module_1.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.Form(env=var_0, schema=var_13)
    var_15 = (var_2, var_3)
    var_16 = (var_5, var_6)
    var_17 = [var_15, var_16]
    var_18 = {}
    var_19 = module_1.Choice(choices=var_17, **var_18)
    var_20 = var_14.render_field(field_name=var_1, field=var_19, value=var_2)
    var_21 = 'select'
    var_22 = bool('select' in var_20)
    assert var_22 is True
    var_23 = 'Red'
    var_24 = bool('Red' in var_20)
    assert var_24 is True
    var_25 = 'Blue'
    var_26 = bool('Blue' in var_20)
    assert var_26 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'active'
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Boolean(**var_8)
    var_10 = True
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'checkbox'
    var_13 = bool('checkbox' in var_11)
    assert var_13 is True
    var_14 = 'checked'
    var_15 = bool('checked' in var_11)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'description'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'Some text'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'textarea'
    var_14 = bool('textarea' in var_12)
    assert var_14 is True
    var_15 = 'Some text'
    var_16 = bool('Some text' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'id'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.Integer(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'read_only'
    var_11 = {var_10: var_2}
    var_12 = module_1.Integer(**var_11)
    var_13 = 123
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    assert var_14 == ''

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'birthdate'
    var_2 = 'date'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = '2023-01-01'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = 'type="date"'
    var_14 = bool('type="date"' in var_12)
    assert var_14 is True
    var_15 = '2023-01-01'
    var_16 = bool('2023-01-01' in var_12)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'custom'
    var_2 = 'unknown'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = var_8.render_field(field_name=var_1, field=var_10)
    var_12 = 'type="text"'
    var_13 = bool('type="text"' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = None
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'value=""'
    var_13 = bool('value=""' in var_11)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'user_name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'user-name'
    var_12 = bool('user-name' in var_10)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'full_name'
    var_2 = 'Full Name'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = 'title'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = var_9.render_field(field_name=var_1, field=var_12)
    var_14 = 'Full Name'
    var_15 = bool('Full Name' in var_13)
    assert var_15 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'full_name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = var_7.render_field(field_name=var_1, field=var_9)
    var_11 = 'full_name'
    var_12 = bool('full_name' in var_10)
    assert var_12 is True



