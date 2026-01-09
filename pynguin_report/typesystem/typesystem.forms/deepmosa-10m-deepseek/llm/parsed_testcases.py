####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_str_with_no_errors. Retrieved 16/24 statements.
# Partially parsed test_str_with_errors. Retrieved 30/42 statements.
# Partially parsed test_str_with_read_only_field. Retrieved 20/32 statements.
# Partially parsed test_str_with_choice_field. Retrieved 27/39 statements.
# Partially parsed test_str_with_boolean_field. Retrieved 27/39 statements.
# Partially parsed test_str_with_textarea_field. Retrieved 29/41 statements.
# Partially parsed test_str_with_password_field. Retrieved 29/41 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'serialize'
    var_13 = {}
    var_14 = lambda self, values: values
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = [var_9, var_10, var_15]
    var_17 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Test Field'
    var_19 = lambda : var_17
    var_20 = 'text'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = 'MockSchema'
    var_24 = ()
    var_25 = 'fields'
    var_26 = 'serialize'
    var_27 = 'test_field'
    var_28 = lambda self, values: values
    var_29 = {}
    var_30 = 'test value'
    var_31 = 'An error'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = True
    var_13 = {var_11: var_12}
    var_14 = [var_9, var_10, var_13]
    var_15 = 'MockSchema'
    var_16 = ()
    var_17 = 'fields'
    var_18 = 'serialize'
    var_19 = 'test_field'
    var_20 = lambda self, values: values
    var_21 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = False
    var_17 = 'Choice Field'
    var_18 = lambda : var_16
    var_19 = {var_11: var_16, var_12: var_17, var_13: var_16, var_14: var_16, var_15: var_18}
    var_20 = [var_9, var_10, var_19]
    var_21 = 'MockSchema'
    var_22 = ()
    var_23 = 'fields'
    var_24 = 'serialize'
    var_25 = 'choice_field'
    var_26 = lambda self, values: values
    var_27 = {}
    var_28 = 'option1'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = False
    var_17 = 'Boolean Field'
    var_18 = lambda : var_16
    var_19 = {var_11: var_16, var_12: var_17, var_13: var_16, var_14: var_16, var_15: var_18}
    var_20 = [var_9, var_10, var_19]
    var_21 = 'MockSchema'
    var_22 = ()
    var_23 = 'fields'
    var_24 = 'serialize'
    var_25 = 'bool_field'
    var_26 = lambda self, values: values
    var_27 = {}
    var_28 = True

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Text Field'
    var_19 = lambda : var_17
    var_20 = 'text'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = 'MockSchema'
    var_24 = ()
    var_25 = 'fields'
    var_26 = 'serialize'
    var_27 = 'text_field'
    var_28 = lambda self, values: values
    var_29 = {}
    var_30 = 'Some text'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Password Field'
    var_19 = lambda : var_17
    var_20 = 'password'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = 'MockSchema'
    var_24 = ()
    var_25 = 'fields'
    var_26 = 'serialize'
    var_27 = 'pass_field'
    var_28 = lambda self, values: values
    var_29 = {}
    var_30 = 'secret'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_raises_assertion_error_when_jinja2_not_installed. Retrieved 2/7 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True


def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True


def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True


def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_3
import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'email'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'email'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'unknown'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'text'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.String(**var_5)
    var_7 = var_4.input_type_for_field(var_6)
    assert var_7 == 'text'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'color'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'color'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'date'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'date'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'datetime'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'datetime-local'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'month'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'month'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'number'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'number'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'password'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'password'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'range'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'range'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'search'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'search'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'tel'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'tel'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'time'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'time'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'url'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'url'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'week'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'week'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'hidden'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'hidden'


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.Boolean(**var_5)
    var_7 = var_4.input_type_for_field(var_6)
    assert var_7 == 'text'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_init_asserts_jinja2_is_not_none. Retrieved 3/8 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #5
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


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Username'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'username'
    var_10 = 'testuser'
    var_11 = None
    var_12 = var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_11)
    var_13 = 'field_id'
    var_14 = bool('field_id' in var_12)
    assert var_14 is True
    var_15 = 'field_name'
    var_16 = bool('field_name' in var_12)
    assert var_16 is True
    var_17 = 'label'
    var_18 = bool('label' in var_12)
    assert var_18 is True
    var_19 = 'required'
    var_20 = bool('required' in var_12)
    assert var_20 is True
    var_21 = 'input_type'
    var_22 = bool('input_type' in var_12)
    assert var_22 is True
    var_23 = 'value'
    var_24 = bool('value' in var_12)
    assert var_24 is True
    var_25 = 'error'
    var_26 = bool('error' in var_12)
    assert var_26 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'password'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'secret'
    var_9 = None
    var_10 = var_4.render_field(field_name=var_5, field=var_7, value=var_8, error=var_9)
    var_11 = 'password'
    var_12 = bool('password' in var_10)
    assert var_12 is True
    var_13 = bool('value' not in var_10 or 'secret' not in var_10)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Email'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'email'
    var_10 = 'invalid'
    var_11 = 'Invalid email'
    var_12 = var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_11)
    var_13 = 'error'
    var_14 = bool('error' in var_12)
    assert var_14 is True
    var_15 = 'Invalid email'
    var_16 = bool('Invalid email' in var_12)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Required Field'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'required_field'
    var_10 = None
    var_11 = var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_10)
    var_12 = 'required'
    var_13 = bool('required' in var_11)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Optional Field'
    var_6 = True
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'optional_field'
    var_12 = None
    var_13 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_12)
    var_14 = bool('required' not in var_13 or ('required' in var_13 and 'false' in var_13))
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Field with Default'
    var_6 = 'default_value'
    var_7 = 'title'
    var_8 = 'default'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'field_with_default'
    var_12 = None
    var_13 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_12)
    var_14 = bool('required' not in var_13 or ('required' in var_13 and 'false' in var_13))
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'option1'
    var_6 = 'Option 1'
    var_7 = (var_5, var_6)
    var_8 = 'option2'
    var_9 = 'Option 2'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = {}
    var_13 = module_3.Choice(choices=var_11, **var_12)
    var_14 = 'choice_field'
    var_15 = None
    var_16 = var_4.render_field(field_name=var_14, field=var_13, value=var_5, error=var_15)
    var_17 = 'select'
    var_18 = bool('select' in var_16)
    assert var_18 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Agree to terms'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_3.Boolean(**var_7)
    var_9 = 'agree'
    var_10 = True
    var_11 = None
    var_12 = var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_11)
    var_13 = 'checkbox'
    var_14 = bool('checkbox' in var_12)
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'text'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'description'
    var_9 = 'Some text'
    var_10 = None
    var_11 = var_4.render_field(field_name=var_8, field=var_7, value=var_9, error=var_10)
    var_12 = 'textarea'
    var_13 = bool('textarea' in var_11)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'email'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'test@example.com'
    var_9 = None
    var_10 = var_4.render_field(field_name=var_5, field=var_7, value=var_8, error=var_9)
    var_11 = 'email'
    var_12 = bool('email' in var_10)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'unknown'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'unknown_field'
    var_9 = 'value'
    var_10 = None
    var_11 = var_4.render_field(field_name=var_8, field=var_7, value=var_9, error=var_10)
    var_12 = 'text'
    var_13 = bool('text' in var_11)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = True
    var_6 = 'read_only'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'read_only_field'
    var_10 = 'readonly'
    var_11 = None
    var_12 = var_4.render_field(field_name=var_9, field=var_8, value=var_10, error=var_11)
    var_13 = bool('readonly' in var_12 or 'disabled' in var_12)
    assert var_13 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_assertion_error_if_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/path'])
    assert var_4 is True


def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'some_package'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'


def test_case_0():
    var_0 = '/custom/path'
    var_1 = 'some_package'
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
    assert var_12 == 'some_package'
    var_13 = var_2.env.loader.loaders[1].package_path
    assert var_13 == 'templates'


def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_assertion_error_without_jinja2_installed. Retrieved 3/7 statements.



def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_2.searchpath
    var_4 = bool(var_2.searchpath == ['/some/path'])
    assert var_4 is True


def test_case_0():
    var_0 = 'myapp'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_2.package_name
    assert var_3 == 'myapp'
    var_4 = var_2.package_path
    assert var_4 == 'templates'


def test_case_0():
    var_0 = '/custom/templates'
    var_1 = 'myapp'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = var_3.loaders
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_3.loaders[var_6]
    var_8 = var_3.loaders[0].searchpath
    var_9 = bool(var_3.loaders[0].searchpath == ['/custom/templates'])
    assert var_9 is True
    var_10 = 1
    var_11 = var_3.loaders[var_10]
    var_12 = var_3.loaders[1].package_name
    assert var_12 == 'myapp'
    var_13 = var_3.loaders[1].package_path
    assert var_13 == 'templates'


def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/path'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 3/8 statements.



def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_assertion_error_without_jinja2. Retrieved 3/7 statements.



def test_case_0():
    var_0 = '/some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/directory'])
    assert var_4 is True


def test_case_0():
    var_0 = 'some.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'some.package'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'


def test_case_0():
    var_0 = '/some/directory'
    var_1 = 'some.package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = var_2.env.loader.loaders
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_2.env.loader.loaders[var_6]
    var_8 = var_2.env.loader.loaders[0].searchpath
    var_9 = bool(var_2.env.loader.loaders[0].searchpath == ['/some/directory'])
    assert var_9 is True
    var_10 = 1
    var_11 = var_2.env.loader.loaders[var_10]
    var_12 = var_2.env.loader.loaders[1].package_name
    assert var_12 == 'some.package'
    var_13 = var_2.env.loader.loaders[1].package_path
    assert var_13 == 'templates'


def test_case_0():
    var_0 = '/some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()


def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/directory'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------




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





def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_fields_without_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 28/44 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 28/43 statements.
# Partially parsed test_render_fields_with_no_validation_called. Retrieved 12/18 statements.


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
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with error {ctx[var_7]}'
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
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with value {ctx[var_7]}'
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
    var_29 = {var_14: var_27}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, name: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 'MockSchema'
    var_8 = ()
    var_9 = 'fields'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = [var_7, var_8, var_11]
    var_13 = {}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 6/7 statements.



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
    var_1 = None
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env is var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema is var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False


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



# Parsed testcases at query #16
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.forms as module_3
import typesystem.schemas as module_2


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.String(**var_1)
    var_3 = 'name'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = 'test'
    var_8 = {var_3: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.env
    var_11 = bool(var_9.env == var_0)
    assert var_11 is True
    var_12 = var_9.schema
    var_13 = bool(var_9.schema == var_6)
    assert var_13 is True
    var_14 = var_9.values
    var_15 = bool(var_9.values == {'name': 'test'})
    assert var_15 is True
    var_16 = var_9.errors
    assert var_16 is None
    var_17 = var_9._validate_called
    assert var_17 is False

import typesystem.forms as module_2
import typesystem.schemas as module_1


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



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 25/36 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 24/35 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 18/29 statements.
# Partially parsed test_render_fields_uses_data_when_errors_exist. Retrieved 27/38 statements.
# Partially parsed test_render_fields_uses_values_when_no_errors. Retrieved 27/38 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
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
    var_23 = [var_13, var_14, var_22]
    var_24 = 'John'
    var_25 = {var_12: var_24}
    var_26 = {var_12: var_24}
    var_27 = 'Rendered forms/input.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'email'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = 'title'
    var_17 = 'allow_null'
    var_18 = 'has_default'
    var_19 = False
    var_20 = 'Email'
    var_21 = lambda : var_19
    var_22 = {var_15: var_19, var_16: var_20, var_17: var_19, var_18: var_21}
    var_23 = [var_13, var_14, var_22]
    var_24 = 'invalid'
    var_25 = {var_12: var_24}
    var_26 = 'Rendered forms/input.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'id'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = True
    var_17 = {var_15: var_16}
    var_18 = [var_13, var_14, var_17]
    var_19 = {}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name} with value {context.get(var_6)}'})()
    var_8 = {var_2: var_7}
    var_9 = [var_0, var_1, var_8]
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
    var_24 = [var_14, var_15, var_23]
    var_25 = 25
    var_26 = {var_13: var_25}
    var_27 = 'invalid'
    var_28 = {var_13: var_27}
    var_29 = 'with value invalid'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'Rendered {name} with value {context.get(var_6)}'})()
    var_8 = {var_2: var_7}
    var_9 = [var_0, var_1, var_8]
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
    var_24 = [var_14, var_15, var_23]
    var_25 = 25
    var_26 = {var_13: var_25}
    var_27 = 30
    var_28 = {var_13: var_27}
    var_29 = 'with value 30'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 3/8 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_directory'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_render_fields_uses_data_when_errors_exist. Retrieved 16/26 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = 'test_field'
    var_8 = 'MockField'
    var_9 = ()
    var_10 = 'read_only'
    var_11 = False
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]
    var_14 = {}
    var_15 = 'An error'
    var_16 = 'data_value'
    var_17 = 'values_value'
    var_18 = 'data_value'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 5/9 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 2/6 statements.



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
    var_1 = 'key'
    var_2 = 'serialized_value'
    var_3 = 'original_value'
    var_4 = {var_1: var_3}

def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_fields_uses_data_when_errors_exist. Retrieved 10/14 statements.



def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = 'test_field'
    var_4 = {var_3: var_2}
    var_5 = module_0.Form(env=var_0, schema=var_1)
    var_6 = 'Some error'
    var_7 = 'data_value'
    var_8 = 'values_value'
    var_9 = var_5.render_fields()
    var_10 = 'data_value'
    var_11 = bool('data_value' in var_9)
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_template_for_field_with_choice_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_boolean_field. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_string_field_text_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_string_field_other_format. Retrieved 8/11 statements.
# Partially parsed test_template_for_field_with_field_without_specialization. Retrieved 6/9 statements.
# Partially parsed test_template_for_field_with_object_field_raises_assertion. Retrieved 6/10 statements.



def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Choice'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Boolean'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)


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


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Field'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Form(env=var_0, schema=var_1)


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 8/9 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 4/5 statements.



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
    var_1 = None
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'serialized'
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



# Parsed testcases at query #26
#--------------------------




import jinja2.environment as module_0


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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

import typesystem.forms as module_2
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_template_for_field_asserts_not_object. Retrieved 13/21 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'serialize'
    var_5 = 'validate_or_error'
    var_6 = {}
    var_7 = lambda self, x: x
    var_8 = None
    var_9 = lambda self, x: (x, var_8)
    var_10 = {var_3: var_6, var_4: var_7, var_5: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = 'Object'
    var_13 = {}



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_3
import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Username'
    var_6 = False
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'username'
    var_12 = 'john_doe'
    var_13 = None
    var_14 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_13)
    var_15 = 'username'
    var_16 = bool('username' in var_14)
    assert var_16 is True
    var_17 = 'Username'
    var_18 = bool('Username' in var_14)
    assert var_18 is True
    var_19 = 'john_doe'
    var_20 = bool('john_doe' in var_14)
    assert var_20 is True
    var_21 = 'text'
    var_22 = bool('text' in var_14)
    assert var_22 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'password'
    var_6 = False
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_3.String(format=var_5, **var_8)
    var_10 = 'secret'
    var_11 = None
    var_12 = var_4.render_field(field_name=var_5, field=var_9, value=var_10, error=var_11)
    var_13 = 'password'
    var_14 = bool('password' in var_12)
    assert var_14 is True
    var_15 = 'secret'
    var_16 = bool('secret' not in var_12)
    assert var_16 is True
    var_17 = 'password'
    var_18 = bool('password' in var_12)
    assert var_18 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Email'
    var_6 = False
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'email'
    var_12 = ''
    var_13 = 'Invalid email'
    var_14 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_13)
    var_15 = 'email'
    var_16 = bool('email' in var_14)
    assert var_16 is True
    var_17 = 'Email'
    var_18 = bool('Email' in var_14)
    assert var_18 is True
    var_19 = 'Invalid email'
    var_20 = bool('Invalid email' in var_14)
    assert var_20 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Name'
    var_6 = False
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'name'
    var_12 = ''
    var_13 = None
    var_14 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_13)
    var_15 = 'required'
    var_16 = bool('required' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Optional'
    var_6 = True
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'optional'
    var_12 = None
    var_13 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_12)
    var_14 = 'required'
    var_15 = bool('required' not in var_13)
    assert var_15 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'With Default'
    var_6 = 'default_value'
    var_7 = False
    var_8 = 'title'
    var_9 = 'default'
    var_10 = 'allow_null'
    var_11 = {var_8: var_5, var_9: var_6, var_10: var_7}
    var_12 = module_3.String(**var_11)
    var_13 = 'with_default'
    var_14 = ''
    var_15 = None
    var_16 = var_4.render_field(field_name=var_13, field=var_12, value=var_14, error=var_15)
    var_17 = 'required'
    var_18 = bool('required' not in var_16)
    assert var_18 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = '1'
    var_6 = 'One'
    var_7 = (var_5, var_6)
    var_8 = '2'
    var_9 = 'Two'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = False
    var_13 = 'allow_null'
    var_14 = {var_13: var_12}
    var_15 = module_3.Choice(choices=var_11, **var_14)
    var_16 = 'choice'
    var_17 = None
    var_18 = var_4.render_field(field_name=var_16, field=var_15, value=var_5, error=var_17)
    var_19 = 'select'
    var_20 = bool('select' in var_18)
    assert var_20 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Agree'
    var_6 = False
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.Boolean(**var_9)
    var_11 = 'agree'
    var_12 = True
    var_13 = None
    var_14 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_13)
    var_15 = 'checkbox'
    var_16 = bool('checkbox' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'text'
    var_6 = False
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_3.String(format=var_5, **var_8)
    var_10 = 'description'
    var_11 = 'Some text'
    var_12 = None
    var_13 = var_4.render_field(field_name=var_10, field=var_9, value=var_11, error=var_12)
    var_14 = 'textarea'
    var_15 = bool('textarea' in var_13)
    assert var_15 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'email'
    var_6 = False
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_3.String(format=var_5, **var_8)
    var_10 = 'test@example.com'
    var_11 = None
    var_12 = var_4.render_field(field_name=var_5, field=var_9, value=var_10, error=var_11)
    var_13 = 'email'
    var_14 = bool('email' in var_12)
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'unknown'
    var_6 = False
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_3.String(format=var_5, **var_8)
    var_10 = 'value'
    var_11 = None
    var_12 = var_4.render_field(field_name=var_5, field=var_9, value=var_10, error=var_11)
    var_13 = 'text'
    var_14 = bool('text' in var_12)
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Read Only'
    var_6 = True
    var_7 = False
    var_8 = 'title'
    var_9 = 'read_only'
    var_10 = 'allow_null'
    var_11 = {var_8: var_5, var_9: var_6, var_10: var_7}
    var_12 = module_3.String(**var_11)
    var_13 = 'read_only'
    var_14 = 'cannot_edit'
    var_15 = None
    var_16 = var_4.render_field(field_name=var_13, field=var_12, value=var_14, error=var_15)
    var_17 = 'read_only'
    var_18 = bool('read_only' in var_16)
    assert var_18 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Field Name'
    var_6 = False
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'field_name'
    var_12 = ''
    var_13 = None
    var_14 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_13)
    var_15 = 'field-name'
    var_16 = bool('field-name' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Empty'
    var_6 = False
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'empty'
    var_12 = ''
    var_13 = None
    var_14 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_13)
    var_15 = 'value=""'
    var_16 = bool('value=""' in var_14)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'None Value'
    var_6 = True
    var_7 = 'title'
    var_8 = 'allow_null'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_3.String(**var_9)
    var_11 = 'none_value'
    var_12 = None
    var_13 = var_4.render_field(field_name=var_11, field=var_10, value=var_12, error=var_12)
    var_14 = bool('value=""' in var_13 or 'value=None' not in var_13)
    assert var_14 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 26/41 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 28/44 statements.
# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 32/49 statements.
# Partially parsed test_render_fields_with_none_values_and_no_errors. Retrieved 27/42 statements.
# Partially parsed test_render_fields_with_empty_values_and_errors. Retrieved 29/45 statements.


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
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with error {ctx[var_7]}'
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
    var_7 = 'value'
    var_8 = lambda self, ctx: f'Rendered {ctx[var_6]} with value {ctx[var_7]}'
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
    var_9 = lambda self, ctx: f'Rendered {ctx[var_6]} with value {ctx[var_7]} and error {ctx[var_8]}'
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
    var_27 = ''
    var_28 = {var_15: var_27}
    var_29 = {var_15: var_27}
    var_30 = 'Password required'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_raises_assertion_error_without_jinja2_installed. Retrieved 2/7 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True


def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True


def test_case_0():
    var_0 = 'test_templates'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_render_fields_with_no_errors. Retrieved 24/36 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 26/38 statements.
# Partially parsed test_render_fields_skips_read_only_field. Retrieved 18/30 statements.
# Partially parsed test_render_fields_with_no_data_and_no_errors. Retrieved 23/35 statements.
# Partially parsed test_render_fields_with_multiple_fields. Retrieved 30/44 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
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
    var_23 = [var_13, var_14, var_22]
    var_24 = 'John'
    var_25 = {var_12: var_24}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
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
    var_23 = [var_13, var_14, var_22]
    var_24 = 'John'
    var_25 = {var_12: var_24}
    var_26 = ''
    var_27 = 'This field is required'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'id'
    var_13 = 'MockField'
    var_14 = ()
    var_15 = 'read_only'
    var_16 = True
    var_17 = {var_15: var_16}
    var_18 = [var_13, var_14, var_17]
    var_19 = {var_12: var_16}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
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
    var_23 = [var_13, var_14, var_22]
    var_24 = None

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'name'
    var_13 = 'email'
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
    var_25 = ()
    var_26 = 'Email'
    var_27 = lambda : var_20
    var_28 = {var_16: var_20, var_17: var_26, var_18: var_20, var_19: var_27}
    var_29 = [var_14, var_25, var_28]
    var_30 = 'John'
    var_31 = 'john@example.com'
    var_32 = {var_12: var_30, var_13: var_31}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_str_with_no_validation_called. Retrieved 17/27 statements.
# Partially parsed test_str_with_validation_and_no_errors. Retrieved 33/48 statements.
# Partially parsed test_str_with_validation_and_errors. Retrieved 35/50 statements.
# Partially parsed test_str_skips_read_only_fields. Retrieved 37/54 statements.
# Partially parsed test_str_uses_data_when_errors_exist. Retrieved 40/55 statements.
# Partially parsed test_str_uses_values_when_no_errors. Retrieved 38/53 statements.


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
    var_18 = {}

def test_case_0():
    var_0 = 'MockTemplate'
    var_1 = ()
    var_2 = 'render'
    var_3 = 'rendered_field'
    var_4 = lambda self, context: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 'MockEnv'
    var_8 = ()
    var_9 = 'get_template'
    var_10 = 'MockField'
    var_11 = ()
    var_12 = 'read_only'
    var_13 = 'title'
    var_14 = 'allow_null'
    var_15 = 'allow_blank'
    var_16 = 'has_default'
    var_17 = 'format'
    var_18 = False
    var_19 = None
    var_20 = lambda : var_18
    var_21 = 'text'
    var_22 = {var_12: var_18, var_13: var_19, var_14: var_18, var_15: var_18, var_16: var_20, var_17: var_21}
    var_23 = [var_10, var_11, var_22]
    var_24 = 'MockSchema'
    var_25 = ()
    var_26 = 'fields'
    var_27 = 'serialize'
    var_28 = 'validate_or_error'
    var_29 = 'test_field'
    var_30 = lambda self, values: values
    var_31 = lambda self, data: (data, var_19)
    var_32 = {}
    var_33 = 'value'
    var_34 = {var_29: var_33}

def test_case_0():
    var_0 = 'MockTemplate'
    var_1 = ()
    var_2 = 'render'
    var_3 = 'rendered_field_with_error'
    var_4 = lambda self, context: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 'MockEnv'
    var_8 = ()
    var_9 = 'get_template'
    var_10 = 'MockField'
    var_11 = ()
    var_12 = 'read_only'
    var_13 = 'title'
    var_14 = 'allow_null'
    var_15 = 'allow_blank'
    var_16 = 'has_default'
    var_17 = 'format'
    var_18 = False
    var_19 = None
    var_20 = lambda : var_18
    var_21 = 'text'
    var_22 = {var_12: var_18, var_13: var_19, var_14: var_18, var_15: var_18, var_16: var_20, var_17: var_21}
    var_23 = [var_10, var_11, var_22]
    var_24 = 'MockSchema'
    var_25 = ()
    var_26 = 'fields'
    var_27 = 'serialize'
    var_28 = 'validate_or_error'
    var_29 = 'test_field'
    var_30 = lambda self, values: values
    var_31 = 'error'
    var_32 = {var_29: var_31}
    var_33 = lambda self, data: (data, var_32)
    var_34 = {}
    var_35 = 'value'
    var_36 = {var_29: var_35}

def test_case_0():
    var_0 = 'MockTemplate'
    var_1 = ()
    var_2 = 'render'
    var_3 = 'rendered_field'
    var_4 = lambda self, context: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 'MockEnv'
    var_8 = ()
    var_9 = 'get_template'
    var_10 = 'MockField'
    var_11 = ()
    var_12 = 'read_only'
    var_13 = 'title'
    var_14 = 'allow_null'
    var_15 = 'allow_blank'
    var_16 = 'has_default'
    var_17 = 'format'
    var_18 = True
    var_19 = None
    var_20 = False
    var_21 = lambda : var_20
    var_22 = 'text'
    var_23 = {var_12: var_18, var_13: var_19, var_14: var_20, var_15: var_20, var_16: var_21, var_17: var_22}
    var_24 = [var_10, var_11, var_23]
    var_25 = ()
    var_26 = lambda : var_20
    var_27 = {var_12: var_20, var_13: var_19, var_14: var_20, var_15: var_20, var_16: var_26, var_17: var_22}
    var_28 = [var_10, var_25, var_27]
    var_29 = 'MockSchema'
    var_30 = ()
    var_31 = 'fields'
    var_32 = 'serialize'
    var_33 = 'validate_or_error'
    var_34 = 'editable'
    var_35 = lambda self, values: values
    var_36 = lambda self, data: (data, var_19)
    var_37 = {}
    var_38 = 'value'
    var_39 = {var_34: var_38}

def test_case_0():
    var_0 = 'MockTemplate'
    var_1 = ()
    var_2 = 'render'
    var_3 = 'value'
    var_4 = ''
    var_5 = lambda self, context: context.get(var_3, var_4)
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'MockEnv'
    var_9 = ()
    var_10 = 'get_template'
    var_11 = 'MockField'
    var_12 = ()
    var_13 = 'read_only'
    var_14 = 'title'
    var_15 = 'allow_null'
    var_16 = 'allow_blank'
    var_17 = 'has_default'
    var_18 = 'format'
    var_19 = False
    var_20 = None
    var_21 = lambda : var_19
    var_22 = 'text'
    var_23 = {var_13: var_19, var_14: var_20, var_15: var_19, var_16: var_19, var_17: var_21, var_18: var_22}
    var_24 = [var_11, var_12, var_23]
    var_25 = 'MockSchema'
    var_26 = ()
    var_27 = 'fields'
    var_28 = 'serialize'
    var_29 = 'validate_or_error'
    var_30 = 'test_field'
    var_31 = lambda self, values: values
    var_32 = 'validated_value'
    var_33 = {var_30: var_32}
    var_34 = 'error'
    var_35 = {var_30: var_34}
    var_36 = (var_33, var_35)
    var_37 = lambda self, data: var_36
    var_38 = 'initial_value'
    var_39 = {var_30: var_38}
    var_40 = 'input_value'
    var_41 = {var_30: var_40}

def test_case_0():
    var_0 = 'MockTemplate'
    var_1 = ()
    var_2 = 'render'
    var_3 = 'value'
    var_4 = ''
    var_5 = lambda self, context: context.get(var_3, var_4)
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'MockEnv'
    var_9 = ()
    var_10 = 'get_template'
    var_11 = 'MockField'
    var_12 = ()
    var_13 = 'read_only'
    var_14 = 'title'
    var_15 = 'allow_null'
    var_16 = 'allow_blank'
    var_17 = 'has_default'
    var_18 = 'format'
    var_19 = False
    var_20 = None
    var_21 = lambda : var_19
    var_22 = 'text'
    var_23 = {var_13: var_19, var_14: var_20, var_15: var_19, var_16: var_19, var_17: var_21, var_18: var_22}
    var_24 = [var_11, var_12, var_23]
    var_25 = 'MockSchema'
    var_26 = ()
    var_27 = 'fields'
    var_28 = 'serialize'
    var_29 = 'validate_or_error'
    var_30 = 'test_field'
    var_31 = lambda self, values: values
    var_32 = 'validated_value'
    var_33 = {var_30: var_32}
    var_34 = (var_33, var_20)
    var_35 = lambda self, data: var_34
    var_36 = 'initial_value'
    var_37 = {var_30: var_36}
    var_38 = 'input_value'
    var_39 = {var_30: var_38}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_str_returns_render_fields_output. Retrieved 20/28 statements.
# Partially parsed test_str_with_fields_and_no_errors. Retrieved 34/46 statements.
# Partially parsed test_str_with_fields_and_errors. Retrieved 36/47 statements.
# Partially parsed test_str_with_read_only_field. Retrieved 35/47 statements.
# Partially parsed test_str_with_choice_field. Retrieved 32/44 statements.
# Partially parsed test_str_with_boolean_field. Retrieved 31/43 statements.
# Partially parsed test_str_with_password_field. Retrieved 34/46 statements.
# Partially parsed test_str_with_multiple_fields. Retrieved 42/56 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSchema'
    var_10 = ()
    var_11 = 'fields'
    var_12 = 'serialize'
    var_13 = 'validate_or_error'
    var_14 = {}
    var_15 = lambda self, values: values
    var_16 = None
    var_17 = lambda self, data: (data, var_16)
    var_18 = {var_11: var_14, var_12: var_15, var_13: var_17}
    var_19 = [var_9, var_10, var_18]
    var_20 = {}
    var_21 = ''

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Test Field'
    var_19 = lambda : var_17
    var_20 = 'text'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = 'MockSchema'
    var_24 = ()
    var_25 = 'fields'
    var_26 = 'serialize'
    var_27 = 'validate_or_error'
    var_28 = 'test_field'
    var_29 = lambda self, values: values
    var_30 = None
    var_31 = lambda self, data: (data, var_30)
    var_32 = 'initial'
    var_33 = {var_28: var_32}
    var_34 = 'new'
    var_35 = 'rendered_forms/textarea.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Test Field'
    var_19 = lambda : var_17
    var_20 = 'text'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = 'MockSchema'
    var_24 = ()
    var_25 = 'fields'
    var_26 = 'serialize'
    var_27 = 'validate_or_error'
    var_28 = 'test_field'
    var_29 = lambda self, values: values
    var_30 = 'error'
    var_31 = {var_28: var_30}
    var_32 = lambda self, data: (data, var_31)
    var_33 = 'initial'
    var_34 = {var_28: var_33}
    var_35 = 'new'
    var_36 = {var_28: var_35}
    var_37 = 'rendered_forms/textarea.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = True
    var_18 = 'Test Field'
    var_19 = False
    var_20 = lambda : var_19
    var_21 = 'text'
    var_22 = {var_11: var_17, var_12: var_18, var_13: var_19, var_14: var_19, var_15: var_20, var_16: var_21}
    var_23 = [var_9, var_10, var_22]
    var_24 = 'MockSchema'
    var_25 = ()
    var_26 = 'fields'
    var_27 = 'serialize'
    var_28 = 'validate_or_error'
    var_29 = 'test_field'
    var_30 = lambda self, values: values
    var_31 = None
    var_32 = lambda self, data: (data, var_31)
    var_33 = 'initial'
    var_34 = {var_29: var_33}
    var_35 = 'new'
    var_36 = ''

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = False
    var_17 = 'Test Field'
    var_18 = lambda : var_16
    var_19 = {var_11: var_16, var_12: var_17, var_13: var_16, var_14: var_16, var_15: var_18}
    var_20 = [var_9, var_10, var_19]
    var_21 = 'MockSchema'
    var_22 = ()
    var_23 = 'fields'
    var_24 = 'serialize'
    var_25 = 'validate_or_error'
    var_26 = 'test_field'
    var_27 = lambda self, values: values
    var_28 = None
    var_29 = lambda self, data: (data, var_28)
    var_30 = 'initial'
    var_31 = {var_26: var_30}
    var_32 = 'new'
    var_33 = 'rendered_forms/select.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = False
    var_17 = 'Test Field'
    var_18 = lambda : var_16
    var_19 = {var_11: var_16, var_12: var_17, var_13: var_16, var_14: var_16, var_15: var_18}
    var_20 = [var_9, var_10, var_19]
    var_21 = 'MockSchema'
    var_22 = ()
    var_23 = 'fields'
    var_24 = 'serialize'
    var_25 = 'validate_or_error'
    var_26 = 'test_field'
    var_27 = lambda self, values: values
    var_28 = None
    var_29 = lambda self, data: (data, var_28)
    var_30 = True
    var_31 = {var_26: var_30}
    var_32 = 'rendered_forms/checkbox.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Test Field'
    var_19 = lambda : var_17
    var_20 = 'password'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = 'MockSchema'
    var_24 = ()
    var_25 = 'fields'
    var_26 = 'serialize'
    var_27 = 'validate_or_error'
    var_28 = 'test_field'
    var_29 = lambda self, values: values
    var_30 = None
    var_31 = lambda self, data: (data, var_30)
    var_32 = 'secret'
    var_33 = {var_28: var_32}
    var_34 = 'new_secret'
    var_35 = 'rendered_forms/input.html'

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = lambda self, name: type(var_3, var_4, {var_5: lambda self, context: f'rendered_{name}'})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockField'
    var_10 = ()
    var_11 = 'read_only'
    var_12 = 'title'
    var_13 = 'allow_null'
    var_14 = 'allow_blank'
    var_15 = 'has_default'
    var_16 = 'format'
    var_17 = False
    var_18 = 'Field1'
    var_19 = lambda : var_17
    var_20 = 'text'
    var_21 = {var_11: var_17, var_12: var_18, var_13: var_17, var_14: var_17, var_15: var_19, var_16: var_20}
    var_22 = [var_9, var_10, var_21]
    var_23 = ()
    var_24 = 'Field2'
    var_25 = lambda : var_17
    var_26 = 'email'
    var_27 = {var_11: var_17, var_12: var_24, var_13: var_17, var_14: var_17, var_15: var_25, var_16: var_26}
    var_28 = [var_9, var_23, var_27]
    var_29 = 'MockSchema'
    var_30 = ()
    var_31 = 'fields'
    var_32 = 'serialize'
    var_33 = 'validate_or_error'
    var_34 = 'field1'
    var_35 = 'field2'
    var_36 = lambda self, values: values
    var_37 = None
    var_38 = lambda self, data: (data, var_37)
    var_39 = 'val1'
    var_40 = 'val2'
    var_41 = {var_34: var_39, var_35: var_40}
    var_42 = 'new1'
    var_43 = 'new2'
    var_44 = 'rendered_forms/textarea.htmlrendered_forms/input.html'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_str_returns_rendered_fields. Retrieved 34/48 statements.
# Partially parsed test_str_with_errors_uses_data_for_values. Retrieved 36/50 statements.
# Partially parsed test_str_without_errors_uses_values. Retrieved 35/49 statements.
# Partially parsed test_str_skips_read_only_fields. Retrieved 35/49 statements.
# Partially parsed test_str_without_validate_called_renders_without_errors. Retrieved 33/46 statements.
# Partially parsed test_str_with_multiple_fields_concatenates_html. Retrieved 40/56 statements.


def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'<input name="{context[var_6]}">'
    var_8 = {var_5: var_7}
    var_9 = var_8()
    var_10 = [var_3, var_4, var_9]
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
    var_24 = False
    var_25 = 'Name'
    var_26 = lambda : var_24
    var_27 = 'text'
    var_28 = {var_19: var_24, var_20: var_25, var_21: var_24, var_22: var_26, var_23: var_27}
    var_29 = [var_17, var_18, var_28]
    var_30 = lambda self, values: values
    var_31 = None
    var_32 = lambda self, data: (data, var_31)
    var_33 = 'test'
    var_34 = {var_16: var_33}
    var_35 = {var_16: var_33}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, context: f'<input value="{context[var_6]}">'
    var_8 = {var_5: var_7}
    var_9 = var_8()
    var_10 = [var_3, var_4, var_9]
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
    var_24 = False
    var_25 = 'Name'
    var_26 = lambda : var_24
    var_27 = 'text'
    var_28 = {var_19: var_24, var_20: var_25, var_21: var_24, var_22: var_26, var_23: var_27}
    var_29 = [var_17, var_18, var_28]
    var_30 = lambda self, values: values
    var_31 = 'error'
    var_32 = {var_16: var_31}
    var_33 = lambda self, data: (data, var_32)
    var_34 = 'initial'
    var_35 = {var_16: var_34}
    var_36 = 'new'
    var_37 = {var_16: var_36}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, context: f'<input value="{context[var_6]}">'
    var_8 = {var_5: var_7}
    var_9 = var_8()
    var_10 = [var_3, var_4, var_9]
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
    var_24 = False
    var_25 = 'Name'
    var_26 = lambda : var_24
    var_27 = 'text'
    var_28 = {var_19: var_24, var_20: var_25, var_21: var_24, var_22: var_26, var_23: var_27}
    var_29 = [var_17, var_18, var_28]
    var_30 = lambda self, values: values
    var_31 = None
    var_32 = lambda self, data: (data, var_31)
    var_33 = 'initial'
    var_34 = {var_16: var_33}
    var_35 = 'new'
    var_36 = {var_16: var_35}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'<input name="{context[var_6]}">'
    var_8 = {var_5: var_7}
    var_9 = var_8()
    var_10 = [var_3, var_4, var_9]
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
    var_24 = True
    var_25 = 'Name'
    var_26 = False
    var_27 = lambda : var_26
    var_28 = 'text'
    var_29 = {var_19: var_24, var_20: var_25, var_21: var_26, var_22: var_27, var_23: var_28}
    var_30 = [var_17, var_18, var_29]
    var_31 = lambda self, values: values
    var_32 = None
    var_33 = lambda self, data: (data, var_32)
    var_34 = 'test'
    var_35 = {var_16: var_34}
    var_36 = {var_16: var_34}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, context: f'<input value="{context[var_6]}">'
    var_8 = {var_5: var_7}
    var_9 = var_8()
    var_10 = [var_3, var_4, var_9]
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
    var_24 = False
    var_25 = 'Name'
    var_26 = lambda : var_24
    var_27 = 'text'
    var_28 = {var_19: var_24, var_20: var_25, var_21: var_24, var_22: var_26, var_23: var_27}
    var_29 = [var_17, var_18, var_28]
    var_30 = lambda self, values: values
    var_31 = None
    var_32 = lambda self, data: (data, var_31)
    var_33 = 'initial'
    var_34 = {var_16: var_33}

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, context: f'<input name="{context[var_6]}">'
    var_8 = {var_5: var_7}
    var_9 = var_8()
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockSchema'
    var_12 = ()
    var_13 = 'fields'
    var_14 = 'serialize'
    var_15 = 'validate_or_error'
    var_16 = 'field1'
    var_17 = 'field2'
    var_18 = 'MockField'
    var_19 = ()
    var_20 = 'read_only'
    var_21 = 'title'
    var_22 = 'allow_null'
    var_23 = 'has_default'
    var_24 = 'format'
    var_25 = False
    var_26 = 'Field1'
    var_27 = lambda : var_25
    var_28 = 'text'
    var_29 = {var_20: var_25, var_21: var_26, var_22: var_25, var_23: var_27, var_24: var_28}
    var_30 = [var_18, var_19, var_29]
    var_31 = ()
    var_32 = 'Field2'
    var_33 = lambda : var_25
    var_34 = {var_20: var_25, var_21: var_32, var_22: var_25, var_23: var_33, var_24: var_28}
    var_35 = [var_18, var_31, var_34]
    var_36 = lambda self, values: values
    var_37 = None
    var_38 = lambda self, data: (data, var_37)
    var_39 = 'val1'
    var_40 = 'val2'
    var_41 = {var_16: var_39, var_17: var_40}
    var_42 = {var_16: var_39, var_17: var_40}



# Parsed testcases at query #6
#--------------------------




import jinja2.environment as module_0


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

import typesystem.fields as module_1
import typesystem.forms as module_3
import typesystem.schemas as module_2


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

import typesystem.forms as module_2
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 3/5 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 10/16 statements.
# Partially parsed test_load_template_env_raises_if_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = '/some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.searchpath
    var_4 = bool(var_1.env.loader.searchpath == ['/some/directory'])
    assert var_4 is True


def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader.package_name
    assert var_3 == 'some_package'
    var_4 = var_1.env.loader.package_path
    assert var_4 == 'templates'


def test_case_0():
    var_0 = '/some/directory'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = var_2.env.loader.loaders
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_2.env.loader.loaders[var_6]
    var_8 = 1
    var_9 = var_2.env.loader.loaders[var_8]


def test_case_0():
    var_0 = '/some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.autoescape
    assert var_2 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()


def test_case_0():
    var_0 = 'jinja2'
    var_1 = '/some/directory'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 4/8 statements.



def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True


def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True


def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = 'jinja2'
    var_1 = None
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_with_jinja2_not_installed_raises_assertion. Retrieved 4/8 statements.



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


def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = 'jinja2'
    var_1 = None
    var_2 = '/some/path'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_3
import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Username'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'username'
    var_10 = 'testuser'
    var_11 = var_4.render_field(field_name=var_9, field=var_8, value=var_10)
    var_12 = 'testuser'
    var_13 = bool('testuser' in var_11)
    assert var_13 is True
    var_14 = 'Username'
    var_15 = bool('Username' in var_11)
    assert var_15 is True
    var_16 = 'type="text"'
    var_17 = bool('type="text"' in var_11)
    assert var_17 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'password'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'secret'
    var_9 = var_4.render_field(field_name=var_5, field=var_7, value=var_8)
    var_10 = 'secret'
    var_11 = bool('secret' not in var_9)
    assert var_11 is True
    var_12 = 'type="password"'
    var_13 = bool('type="password"' in var_9)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'email'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'user@example.com'
    var_9 = var_4.render_field(field_name=var_5, field=var_7, value=var_8)
    var_10 = 'user@example.com'
    var_11 = bool('user@example.com' in var_9)
    assert var_11 is True
    var_12 = 'type="email"'
    var_13 = bool('type="email"' in var_9)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.Integer(**var_5)
    var_7 = 'age'
    var_8 = 25
    var_9 = var_4.render_field(field_name=var_7, field=var_6, value=var_8)
    var_10 = '25'
    var_11 = bool('25' in var_9)
    assert var_11 is True
    var_12 = 'type="number"'
    var_13 = bool('type="number"' in var_9)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.String(**var_5)
    var_7 = 'name'
    var_8 = var_4.render_field(field_name=var_7, field=var_6)
    var_9 = 'required'
    var_10 = bool('required' in var_8)
    assert var_10 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'optional'
    var_10 = var_4.render_field(field_name=var_9, field=var_8)
    var_11 = 'required'
    var_12 = bool('required' not in var_10)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'default_value'
    var_6 = 'default'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'with_default'
    var_10 = var_4.render_field(field_name=var_9, field=var_8)
    var_11 = 'required'
    var_12 = bool('required' not in var_10)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.String(**var_5)
    var_7 = 'field'
    var_8 = 'Invalid value'
    var_9 = var_4.render_field(field_name=var_7, field=var_6, error=var_8)
    var_10 = 'Invalid value'
    var_11 = bool('Invalid value' in var_9)
    assert var_11 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'a'
    var_6 = 'Option A'
    var_7 = (var_5, var_6)
    var_8 = 'b'
    var_9 = 'Option B'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = {}
    var_13 = module_3.Choice(choices=var_11, **var_12)
    var_14 = 'choice'
    var_15 = var_4.render_field(field_name=var_14, field=var_13, value=var_5)
    var_16 = 'select'
    var_17 = bool('select' in var_15)
    assert var_17 is True
    var_18 = 'Option A'
    var_19 = bool('Option A' in var_15)
    assert var_19 is True
    var_20 = 'Option B'
    var_21 = bool('Option B' in var_15)
    assert var_21 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.Boolean(**var_5)
    var_7 = 'agree'
    var_8 = True
    var_9 = var_4.render_field(field_name=var_7, field=var_6, value=var_8)
    var_10 = 'checkbox'
    var_11 = bool('checkbox' in var_9)
    assert var_11 is True
    var_12 = 'checked'
    var_13 = bool('checked' in var_9)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'text'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'description'
    var_9 = 'Some text'
    var_10 = var_4.render_field(field_name=var_8, field=var_7, value=var_9)
    var_11 = 'textarea'
    var_12 = bool('textarea' in var_10)
    assert var_12 is True
    var_13 = 'Some text'
    var_14 = bool('Some text' in var_10)
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'unknown_format'
    var_6 = {}
    var_7 = module_3.String(format=var_5, **var_6)
    var_8 = 'custom'
    var_9 = var_4.render_field(field_name=var_8, field=var_7)
    var_10 = 'type="text"'
    var_11 = bool('type="text"' in var_9)
    assert var_11 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.String(**var_5)
    var_7 = 'field_name'
    var_8 = var_4.render_field(field_name=var_7, field=var_6)
    var_9 = 'field-name'
    var_10 = bool('field-name' in var_8)
    assert var_10 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'Custom Title'
    var_6 = 'title'
    var_7 = {var_6: var_5}
    var_8 = module_3.String(**var_7)
    var_9 = 'field'
    var_10 = var_4.render_field(field_name=var_9, field=var_8)
    var_11 = 'Custom Title'
    var_12 = bool('Custom Title' in var_10)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = {}
    var_6 = module_3.String(**var_5)
    var_7 = 'field_name'
    var_8 = var_4.render_field(field_name=var_7, field=var_6)
    var_9 = 'field_name'
    var_10 = bool('field_name' in var_8)
    assert var_10 is True

import typesystem.fields as module_1
import typesystem.forms as module_3
import typesystem.schemas as module_2


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'readonly'
    var_2 = 'editable'
    var_3 = True
    var_4 = 'read_only'
    var_5 = {var_4: var_3}
    var_6 = module_1.String(**var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.Form(env=var_0, schema=var_11)
    var_13 = var_12.render_fields()
    var_14 = 'readonly'
    var_15 = bool('readonly' not in var_13)
    assert var_15 is True
    var_16 = 'editable'
    var_17 = bool('editable' in var_13)
    assert var_17 is True



# Parsed testcases at query #11
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


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_1
import typesystem.forms as module_0


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Boolean(**var_3)
    var_5 = var_2.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)



# Parsed testcases at query #14
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Form(env=var_0, schema=var_1)
    var_3 = {}
    var_4 = module_1.Object(**var_3)
    var_5 = var_2.template_for_field(var_4)



# Parsed testcases at query #15
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


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_form_constructor_with_values. Retrieved 5/9 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 2/6 statements.


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
    var_1 = 'key'
    var_2 = 'serialized_value'
    var_3 = 'value'
    var_4 = {var_1: var_3}

def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_render_fields_uses_self_values_when_no_errors. Retrieved 15/26 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'test_field'
    var_5 = 'MockField'
    var_6 = ()
    var_7 = 'read_only'
    var_8 = False
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = None
    var_12 = 'initial'
    var_13 = {var_4: var_12}
    var_14 = 'new'
    var_15 = {var_4: var_14}
    var_16 = 'new'



# Parsed testcases at query #18
#--------------------------




import jinja2.environment as module_0
import typesystem.forms as module_2
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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

import typesystem.forms as module_2
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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



# Parsed testcases at query #19
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


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------




import jinja2.environment as module_0
import typesystem.forms as module_2
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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

import typesystem.forms as module_2
import typesystem.schemas as module_1


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

import typesystem.fields as module_1
import typesystem.schemas as module_2


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



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_3
import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = module_2.Form(env=var_0, schema=var_3)
    var_5 = 'a'
    var_6 = 'A'
    var_7 = (var_5, var_6)
    var_8 = 'b'
    var_9 = 'B'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = {}
    var_13 = module_3.Choice(choices=var_11, **var_12)
    var_14 = var_4.template_for_field(var_13)
    assert var_14 == 'forms/select.html'



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_template_for_field_asserts_not_object. Retrieved 12/17 statements.


import typesystem.fields as module_0


def test_case_0():
    var_0 = None
    var_1 = 'MockSchema'
    var_2 = ()
    var_3 = 'fields'
    var_4 = 'serialize'
    var_5 = 'validate_or_error'
    var_6 = {}
    var_7 = lambda self, x: x
    var_8 = None
    var_9 = lambda self, x: (x, var_8)
    var_10 = {var_3: var_6, var_4: var_7, var_5: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = {}
    var_13 = module_0.Object(**var_12)
    var_14 = bool(False)
    assert var_14 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_init_raises_assertion_error_when_jinja2_is_none. Retrieved 3/8 statements.


import typesystem.forms as module_0


def test_case_0():
    var_0 = 'jinja2'
    var_1 = 'some_directory'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_template_env_with_directory_only. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package_only. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 11/17 statements.



def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_2.loader
    var_4 = var_3.searchpath
    var_5 = bool(var_3.searchpath == ['/some/path'])
    assert var_5 is True


def test_case_0():
    var_0 = 'some_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_2.loader
    var_4 = var_3.package_name
    assert var_4 == 'some_package'
    var_5 = var_3.package_path
    assert var_5 == 'templates'


def test_case_0():
    var_0 = '/custom/path'
    var_1 = 'some_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_3.loader
    var_5 = var_4.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_4.loaders[var_7]
    var_9 = var_4.loaders[0].searchpath
    var_10 = bool(var_4.loaders[0].searchpath == ['/custom/path'])
    assert var_10 is True
    var_11 = 1
    var_12 = var_4.loaders[var_11]
    var_13 = var_4.loaders[1].package_name
    assert var_13 == 'some_package'
    var_14 = var_4.loaders[1].package_path
    assert var_14 == 'templates'


def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_2.autoescape
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'a'
    var_3 = 'A'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 'B'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_0.Choice(choices=var_8, **var_9)
    var_11 = module_1.Form(env=var_0, schema=var_1)
    var_12 = var_11.template_for_field(var_10)
    assert var_12 == 'forms/select.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Choice(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/select.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    assert var_5 == 'forms/checkbox.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/textarea.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = module_1.Form(env=var_0, schema=var_1)
    var_6 = var_5.template_for_field(var_4)
    assert var_6 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.Field()
    var_3 = module_1.Form(env=var_0, schema=var_1)
    var_4 = var_3.template_for_field(var_2)
    assert var_4 == 'forms/input.html'


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = {}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.Form(env=var_0, schema=var_1)
    var_5 = var_4.template_for_field(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------






# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------






