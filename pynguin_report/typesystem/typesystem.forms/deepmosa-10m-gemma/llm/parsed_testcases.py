####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 2/7 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 3/8 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/13 statements.
# Partially parsed test_load_template_env_raises_assertion_error_if_both_provided_invalidly. Retrieved 3/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #2
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

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
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = lambda self, name: var_12
    var_14 = {var_2: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'MockField'
    var_20 = ()
    var_21 = 'read_only'
    var_22 = 'title'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = False
    var_27 = 'Title'
    var_28 = lambda self: var_26
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_26, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = 'MockSchema'
    var_36 = ()
    var_37 = 'serialize'
    var_38 = 'validate_or_error'
    var_39 = 'fields'
    var_40 = 'name'
    var_41 = 'val'
    var_42 = {var_40: var_41}
    var_43 = lambda self, v: var_42
    var_44 = {var_40: var_41}
    var_45 = None
    var_46 = (var_44, var_45)
    var_47 = lambda self, d: var_46
    var_48 = {var_40: var_34}
    var_49 = {var_37: var_43, var_38: var_47, var_39: var_48}
    var_50 = [var_35, var_36, var_49]
    var_51 = {}
    var_52 = module_0.type(*var_50, **var_51)
    var_53 = var_52()
    var_54 = module_1.Form(env=var_18, schema=var_53)
    var_55 = {var_40: var_41}
    var_56 = var_54.validate(var_55)
    var_57 = str(var_54)
    assert var_57 == 'rendered_name'

import builtins as module_0
import typesystem.forms as module_1

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
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = lambda self, name: var_12
    var_14 = {var_2: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'MockField'
    var_20 = ()
    var_21 = 'read_only'
    var_22 = 'title'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = True
    var_27 = 'ReadOnly'
    var_28 = False
    var_29 = lambda self: var_28
    var_30 = 'text'
    var_31 = {var_21: var_26, var_22: var_27, var_23: var_28, var_24: var_29, var_25: var_30}
    var_32 = [var_19, var_20, var_31]
    var_33 = {}
    var_34 = module_0.type(*var_32, **var_33)
    var_35 = var_34()
    var_36 = ()
    var_37 = 'Active'
    var_38 = lambda self: var_28
    var_39 = {var_21: var_28, var_22: var_37, var_23: var_28, var_24: var_38, var_25: var_30}
    var_40 = [var_19, var_36, var_39]
    var_41 = {}
    var_42 = module_0.type(*var_40, **var_41)
    var_43 = var_42()
    var_44 = 'MockSchema'
    var_45 = ()
    var_46 = 'serialize'
    var_47 = 'validate_or_error'
    var_48 = 'fields'
    var_49 = {}
    var_50 = lambda self, v: var_49
    var_51 = {}
    var_52 = None
    var_53 = (var_51, var_52)
    var_54 = lambda self, d: var_53
    var_55 = 'readonly_field'
    var_56 = 'active_field'
    var_57 = {var_55: var_35, var_56: var_43}
    var_58 = {var_46: var_50, var_47: var_54, var_48: var_57}
    var_59 = [var_44, var_45, var_58]
    var_60 = {}
    var_61 = module_0.type(*var_59, **var_60)
    var_62 = var_61()
    var_63 = module_1.Form(env=var_18, schema=var_62)
    var_64 = {}
    var_65 = var_63.validate(var_64)
    var_66 = str(var_63)
    assert var_66 == 'rendered_active_field'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_field_password_clears_value. Retrieved 4/18 statements.
# Partially parsed test_render_field_text_area_template_selection. Retrieved 3/17 statements.
# Partially parsed test_render_field_input_type_mapping. Retrieved 2/16 statements.
# Partially parsed test_render_field_label_and_id_formatting. Retrieved 3/15 statements.
# Partially parsed test_render_field_required_logic. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'password_field'
    var_1 = 'password_field'
    var_2 = 'secret123'
    var_3 = {var_1: var_2}
    var_4 = 'password'

def test_case_0():
    var_0 = 'bio'
    var_1 = 'bio'
    var_2 = 'text'
    var_3 = 'hello'

def test_case_0():
    var_0 = 'email'
    var_1 = 'email'
    var_2 = 'test@example.com'

def test_case_0():
    var_0 = 'user_name'
    var_1 = 'user_name'
    var_2 = 'User Name'
    var_3 = 'admin'

def test_case_0():
    var_0 = 'req'
    var_1 = 'req'
    var_2 = 'val'
    var_3 = 'something'
    var_4 = 'def'
    var_5 = True
    var_6 = 'null'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_form_html_returns_markup_of_rendered_fields. Retrieved 7/39 statements.


def test_case_0():
    var_0 = 'Field 1'
    var_1 = True
    var_2 = 'Hidden'
    var_3 = 'f1'
    var_4 = 'f2'
    var_5 = 'val1'
    var_6 = {var_3: var_5}
    var_7 = 'rendered_forms/textarea.html'
    var_8 = 'rendered_forms/input.html'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 2/6 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 3/7 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/15 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'
    var_3 = var_1.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 'templates'
    var_4 = var_2.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_jinja2forms_init_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_both_directory_and_package. Retrieved 7/11 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader
    var_4 = var_1.env.loader.searchpath
    var_5 = bool(var_1.env.loader.searchpath == [var_0])
    assert var_5 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader
    var_4 = var_1.env.loader.searchpath
    var_5 = bool(var_1.env.loader.searchpath == [var_0])
    assert var_5 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader
    var_5 = var_2.env.loader.loaders
    var_6 = len(var_5)
    assert var_6 == 2

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = 'Should have raised AssertionError'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_input_type_for_field_returns_text_for_no_format. Retrieved 1/10 statements.
# Partially parsed test_input_type_for_field_returns_mapped_type_for_valid_format. Retrieved 2/11 statements.
# Partially parsed test_input_type_for_field_returns_text_for_unknown_format. Retrieved 2/11 statements.
# Partially parsed test_input_type_for_field_handles_all_mapped_formats. Retrieved 30/53 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'unknown_format'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'datetime'
    var_4 = {}
    var_5 = module_0.String(format=var_3, **var_4)
    var_6 = 'date'
    var_7 = {}
    var_8 = module_0.String(format=var_6, **var_7)
    var_9 = 'email'
    var_10 = {}
    var_11 = module_0.String(format=var_9, **var_10)
    var_12 = 'hidden'
    var_13 = {}
    var_14 = module_0.String(format=var_12, **var_13)
    var_15 = 'month'
    var_16 = {}
    var_17 = module_0.String(format=var_15, **var_16)
    var_18 = 'number'
    var_19 = {}
    var_20 = module_0.String(format=var_18, **var_19)
    var_21 = 'password'
    var_22 = {}
    var_23 = module_0.String(format=var_21, **var_22)
    var_24 = 'range'
    var_25 = {}
    var_26 = module_0.String(format=var_24, **var_25)
    var_27 = 'search'
    var_28 = {}
    var_29 = module_0.String(format=var_27, **var_28)
    var_30 = 'tel'
    var_31 = {}
    var_32 = module_0.String(format=var_30, **var_31)
    var_33 = 'text'
    var_34 = {}
    var_35 = module_0.String(format=var_33, **var_34)
    var_36 = 'time'
    var_37 = {}
    var_38 = module_0.String(format=var_36, **var_37)
    var_39 = 'url'
    var_40 = {}
    var_41 = module_0.String(format=var_39, **var_40)
    var_42 = 'week'
    var_43 = {}
    var_44 = module_0.String(format=var_42, **var_43)



# Parsed testcases at query #8
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = {}
    var_14 = lambda self, x: x if x is not var_3 else var_13
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = 'John'
    var_22 = {var_20: var_21}
    var_23 = module_1.Form(env=var_9, schema=var_19, values=var_22)
    var_24 = var_23.env
    var_25 = bool(var_23.env == var_9)
    assert var_25 is True
    var_26 = var_23.schema
    var_27 = bool(var_23.schema == var_19)
    assert var_27 is True
    var_28 = var_23.values
    var_29 = bool(var_23.values == var_22)
    assert var_29 is True
    var_30 = var_23.errors
    assert var_30 is None
    var_31 = var_23._validate_called
    assert var_31 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = {}
    var_14 = lambda self, x: var_13
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = module_1.Form(env=var_9, schema=var_19, values=var_3)
    var_21 = var_20.values
    var_22 = bool(var_20.values == {})
    assert var_22 is True



# Parsed testcases at query #9
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = 'name'
    var_5 = 'John'
    var_6 = {var_4: var_5}
    var_7 = lambda self, v: var_6
    var_8 = {var_4: var_5}
    var_9 = None
    var_10 = (var_8, var_9)
    var_11 = lambda self, d: var_10
    var_12 = {var_2: var_7, var_3: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = 'Env'
    var_18 = ()
    var_19 = 'get_template'
    var_20 = 'Template'
    var_21 = ()
    var_22 = 'render'
    var_23 = ''
    var_24 = lambda self, c: var_23
    var_25 = {var_22: var_24}
    var_26 = [var_20, var_21, var_25]
    var_27 = {}
    var_28 = module_0.type(*var_26, **var_27)
    var_29 = var_28()
    var_30 = lambda self, t: var_29
    var_31 = {var_19: var_30}
    var_32 = [var_17, var_18, var_31]
    var_33 = {}
    var_34 = module_0.type(*var_32, **var_33)
    var_35 = var_34()
    var_36 = 'Old'
    var_37 = {var_4: var_36}
    var_38 = module_1.Form(env=var_35, schema=var_16, values=var_37)
    var_39 = {var_4: var_5}
    var_40 = var_38.validate(var_39)
    var_41 = var_38.values
    var_42 = bool(var_38.values == {'name': 'John'})
    assert var_42 is True
    var_43 = var_38.errors
    assert var_43 is None
    var_44 = var_38.is_valid
    assert var_44 is True

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = 'name'
    var_5 = 'John'
    var_6 = {var_4: var_5}
    var_7 = lambda self, v: var_6
    var_8 = ''
    var_9 = {var_4: var_8}
    var_10 = 'Required'
    var_11 = {var_4: var_10}
    var_12 = (var_9, var_11)
    var_13 = lambda self, d: var_12
    var_14 = {var_2: var_7, var_3: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'Env'
    var_20 = ()
    var_21 = 'get_template'
    var_22 = 'Template'
    var_23 = ()
    var_24 = 'render'
    var_25 = lambda self, c: var_8
    var_26 = {var_24: var_25}
    var_27 = [var_22, var_23, var_26]
    var_28 = {}
    var_29 = module_0.type(*var_27, **var_28)
    var_30 = var_29()
    var_31 = lambda self, t: var_30
    var_32 = {var_21: var_31}
    var_33 = [var_19, var_20, var_32]
    var_34 = {}
    var_35 = module_0.type(*var_33, **var_34)
    var_36 = var_35()
    var_37 = {var_4: var_5}
    var_38 = module_1.Form(env=var_36, schema=var_18, values=var_37)
    var_39 = {var_4: var_8}
    var_40 = var_38.validate(var_39)
    var_41 = var_38.errors
    var_42 = bool(var_38.errors == {'name': 'Required'})
    assert var_42 is True
    var_43 = var_38.is_valid
    assert var_43 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = {}
    var_5 = lambda self, v: var_4
    var_6 = {}
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda self, d: var_8
    var_10 = {var_2: var_5, var_3: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = 'Env'
    var_16 = ()
    var_17 = 'get_template'
    var_18 = 'Template'
    var_19 = ()
    var_20 = 'render'
    var_21 = ''
    var_22 = lambda self, c: var_21
    var_23 = {var_20: var_22}
    var_24 = [var_18, var_19, var_23]
    var_25 = {}
    var_26 = module_0.type(*var_24, **var_25)
    var_27 = var_26()
    var_28 = lambda self, t: var_27
    var_29 = {var_17: var_28}
    var_30 = [var_15, var_16, var_29]
    var_31 = {}
    var_32 = module_0.type(*var_30, **var_31)
    var_33 = var_32()
    var_34 = module_1.Form(env=var_33, schema=var_14)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = {}
    var_38 = var_34.validate(var_37)
    var_39 = bool(False)
    assert var_39 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_field_required_logic. Retrieved 7/25 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Default Field'
    var_1 = 'some_default'
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = 'test_field'
    var_4 = 'test'
    var_5 = None
    var_6 = 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_field_basic_input. Retrieved 5/30 statements.
# Partially parsed test_render_field_password_masks_value. Retrieved 4/29 statements.


def test_case_0():
    var_0 = 'Test Label'
    var_1 = 'test_field'
    var_2 = 'hello'
    var_3 = {var_1: var_2}
    var_4 = None

def test_case_0():
    var_0 = 'password_field'
    var_1 = 'secret'
    var_2 = {var_0: var_1}
    var_3 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_form_str_renders_fields_via_render_fields. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_form_init_args_not_positional. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_field_password_value_not_empty. Retrieved 2/52 statements.


def test_case_0():
    var_0 = 'user_name'
    var_1 = 'secret_data'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_form_constructor_initializes_with_provided_values. Retrieved 3/22 statements.
# Partially parsed test_form_constructor_handles_none_values. Retrieved 2/21 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John Doe'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_form_str_returns_rendered_fields. Retrieved 5/35 statements.


def test_case_0():
    var_0 = 'user_name'
    var_1 = 'user_email'
    var_2 = True
    var_3 = 'John'
    var_4 = {var_0: var_3}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_init_raises_error_when_jinja2_is_none. Retrieved 2/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 2/6 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 3/7 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/12 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'
    var_3 = var_1.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 'templates'
    var_4 = var_2.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_form_html_returns_markup_with_rendered_fields. Retrieved 6/34 statements.


import markupsafe as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 'test_field'
    var_4 = 'rendered_test_field'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Markup(*var_5, **var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_jinja2forms_init_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_init_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = 'Should have raised AssertionError'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #4
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = 'MockSchema'
    var_8 = ()
    var_9 = 'serialize'
    var_10 = None
    var_11 = {}
    var_12 = lambda self, values: values if values is not var_10 else var_11
    var_13 = {var_9: var_12}
    var_14 = [var_7, var_8, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = 'name'
    var_19 = 'John'
    var_20 = {var_18: var_19}
    var_21 = module_1.Form(env=var_6, schema=var_17, values=var_20)
    var_22 = var_21.env
    var_23 = bool(var_21.env == var_6)
    assert var_23 is True
    var_24 = var_21.schema
    var_25 = bool(var_21.schema == var_17)
    assert var_25 is True
    var_26 = var_21.values
    var_27 = bool(var_21.values == {'name': 'John'})
    assert var_27 is True
    var_28 = var_21.errors
    assert var_28 is None
    var_29 = var_21._validate_called
    assert var_29 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = 'MockSchema'
    var_8 = ()
    var_9 = 'serialize'
    var_10 = None
    var_11 = lambda self, values: var_10
    var_12 = {var_9: var_11}
    var_13 = [var_7, var_8, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = module_1.Form(env=var_6, schema=var_16, values=var_10)
    var_18 = var_17.values
    assert var_18 is None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_template_env_else_branch_predicate_false. Retrieved 3/13 statements.
# Partially parsed test_load_template_env_trigger_line_10_failure. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None
    var_1 = 'The predicate at line 10 (or 9) was not triggered via False evaluation'
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 4/20 statements.
# Partially parsed test_render_field_generates_correct_id_and_label. Retrieved 4/20 statements.
# Partially parsed test_render_field_identifies_required_status. Retrieved 4/27 statements.
# Failed to parse test_render_field_input_type_mapping.


def test_case_0():
    var_0 = 'password_field'
    var_1 = 'password_field'
    var_2 = 'secret123'
    var_3 = {var_1: var_2}
    var_4 = None

def test_case_0():
    var_0 = 'user_name'
    var_1 = 'user_name'
    var_2 = 'admin'
    var_3 = {var_1: var_2}
    var_4 = 'Error message'

def test_case_0():
    var_0 = 'm'
    var_1 = 'o'
    var_2 = 'm'
    var_3 = 'o'
    var_4 = 'val'
    var_5 = {var_2: var_4, var_3: var_4}



# Parsed testcases at query #7
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = 'Schema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = 'field1'
    var_14 = 'value1'
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = lambda self, v: var_15 if v else var_16
    var_18 = {var_12: var_17}
    var_19 = [var_10, var_11, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = {var_13: var_14}
    var_24 = module_1.Form(env=var_9, schema=var_22, values=var_23)
    var_25 = var_24.env
    var_26 = bool(var_24.env == var_9)
    assert var_26 is True
    var_27 = var_24.schema
    var_28 = bool(var_24.schema == var_22)
    assert var_28 is True
    var_29 = var_24.values
    var_30 = bool(var_24.values == {'field1': 'value1'})
    assert var_30 is True
    var_31 = var_24.errors
    assert var_31 is None
    var_32 = var_24._validate_called
    assert var_32 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = 'Schema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = {}
    var_14 = lambda self, v: var_13
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = module_1.Form(env=var_9, schema=var_19, values=var_3)
    var_21 = var_20.values
    var_22 = bool(var_20.values == {})
    assert var_22 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_form_str_renders_fields_html. Retrieved 7/36 statements.


def test_case_0():
    var_0 = 'First Name'
    var_1 = True
    var_2 = 'first_name'
    var_3 = 'hidden_field'
    var_4 = 'John'
    var_5 = {var_2: var_4}
    var_6 = {var_2: var_4}



# Parsed testcases at query #9
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader.__class__.__name__
    assert var_2 == 'FileSystemLoader'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader.__class__.__name__
    assert var_2 == 'PackageLoader'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader.__class__.__name__
    assert var_3 == 'ChoiceLoader'

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #10
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = lambda self, v: v
    var_5 = None
    var_6 = lambda self, d: (d, var_5)
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'Env'
    var_13 = ()
    var_14 = 'get_template'
    var_15 = lambda self, n: var_5
    var_16 = {var_14: var_15}
    var_17 = [var_12, var_13, var_16]
    var_18 = {}
    var_19 = module_0.type(*var_17, **var_18)
    var_20 = var_19()
    var_21 = 'name'
    var_22 = 'test'
    var_23 = {var_21: var_22}
    var_24 = module_1.Form(env=var_20, schema=var_11, values=var_23)
    var_25 = {var_21: var_22}
    var_26 = var_24.validate(var_25)
    var_27 = var_24.values
    var_28 = bool(var_24.values == {'name': 'test'})
    assert var_28 is True
    var_29 = var_24.errors
    assert var_29 is None
    var_30 = var_24.is_valid
    assert var_30 is True
    var_31 = var_24.data
    var_32 = bool(var_24.data == {'name': 'test'})
    assert var_32 is True

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = lambda self, v: v
    var_5 = 'name'
    var_6 = 'val'
    var_7 = {var_5: var_6}
    var_8 = 'error'
    var_9 = {var_5: var_8}
    var_10 = (var_7, var_9)
    var_11 = lambda self, d: var_10
    var_12 = {var_2: var_4, var_3: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = 'Env'
    var_18 = ()
    var_19 = 'get_template'
    var_20 = None
    var_21 = lambda self, n: var_20
    var_22 = {var_19: var_21}
    var_23 = [var_17, var_18, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = 'test'
    var_28 = {var_5: var_27}
    var_29 = module_1.Form(env=var_26, schema=var_16, values=var_28)
    var_30 = 'invalid'
    var_31 = {var_5: var_30}
    var_32 = var_29.validate(var_31)
    var_33 = var_29.values
    var_34 = bool(var_29.values == {'name': 'val'})
    assert var_34 is True
    var_35 = var_29.errors
    var_36 = bool(var_29.errors == {'name': 'error'})
    assert var_36 is True
    var_37 = var_29.is_valid
    assert var_37 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = lambda self, v: v
    var_5 = None
    var_6 = lambda self, d: (d, var_5)
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = var_10()
    var_12 = 'MockEnv'
    var_13 = ()
    var_14 = 'get_template'
    var_15 = lambda self, n: var_5
    var_16 = {var_14: var_15}
    var_17 = [var_12, var_13, var_16]
    var_18 = {}
    var_19 = module_0.type(*var_17, **var_18)
    var_20 = var_19()
    var_21 = {}
    var_22 = module_1.Form(env=var_20, schema=var_11, values=var_21)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = {}
    var_26 = var_22.validate(var_25)
    var_27 = bool(False)
    assert var_27 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 6/21 statements.
# Partially parsed test_render_field_generates_correct_id. Retrieved 5/18 statements.
# Partially parsed test_render_field_includes_error_in_context. Retrieved 6/19 statements.
# Partially parsed test_render_field_uses_title_as_label. Retrieved 5/18 statements.
# Partially parsed test_render_field_uses_field_name_as_label_if_no_title. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'Password'
    var_1 = False
    var_2 = None
    var_3 = {}
    var_4 = 'password_field'
    var_5 = 'secret123'

def test_case_0():
    var_0 = 'User Name'
    var_1 = None
    var_2 = {}
    var_3 = 'user_name_input'
    var_4 = 'test'

def test_case_0():
    var_0 = 'Email'
    var_1 = None
    var_2 = {}
    var_3 = 'email'
    var_4 = 'invalid-email'
    var_5 = 'Invalid format'

def test_case_0():
    var_0 = 'Display Name'
    var_1 = None
    var_2 = {}
    var_3 = 'display_name'
    var_4 = 'test'

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = {}
    var_3 = 'username'
    var_4 = 'test'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_form_constructor_initializes_with_env_schema_and_values. Retrieved 5/14 statements.
# Partially parsed test_form_constructor_initializes_with_defaults. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John Doe'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = None



# Parsed testcases at query #13
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

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
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = lambda self, name: var_12
    var_14 = {var_2: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'Field'
    var_20 = ()
    var_21 = 'read_only'
    var_22 = 'title'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = False
    var_27 = 'Name'
    var_28 = lambda self: var_26
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_26, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = 'Schema'
    var_36 = ()
    var_37 = 'fields'
    var_38 = 'serialize'
    var_39 = 'validate_or_error'
    var_40 = 'name'
    var_41 = {var_40: var_34}
    var_42 = 'John'
    var_43 = {var_40: var_42}
    var_44 = lambda self, v: var_43
    var_45 = {var_40: var_42}
    var_46 = None
    var_47 = (var_45, var_46)
    var_48 = lambda self, d: var_47
    var_49 = {var_37: var_41, var_38: var_44, var_39: var_48}
    var_50 = [var_35, var_36, var_49]
    var_51 = {}
    var_52 = module_0.type(*var_50, **var_51)
    var_53 = var_52()
    var_54 = {var_40: var_42}
    var_55 = module_1.Form(env=var_18, schema=var_53, values=var_54)
    var_56 = {var_40: var_42}
    var_57 = var_55.validate(var_56)
    var_58 = var_55.render_fields()
    assert var_58 == 'rendered_name'

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'error_{ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = lambda self, name: var_12
    var_14 = {var_2: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'Field'
    var_20 = ()
    var_21 = 'read_only'
    var_22 = 'title'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = False
    var_27 = 'Name'
    var_28 = lambda self: var_26
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_26, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = 'Schema'
    var_36 = ()
    var_37 = 'fields'
    var_38 = 'serialize'
    var_39 = 'validate_or_error'
    var_40 = 'name'
    var_41 = {var_40: var_34}
    var_42 = ''
    var_43 = {var_40: var_42}
    var_44 = lambda self, v: var_43
    var_45 = {var_40: var_42}
    var_46 = 'Required'
    var_47 = {var_40: var_46}
    var_48 = (var_45, var_47)
    var_49 = lambda self, d: var_48
    var_50 = {var_37: var_41, var_38: var_44, var_39: var_49}
    var_51 = [var_35, var_36, var_50]
    var_52 = {}
    var_53 = module_0.type(*var_51, **var_52)
    var_54 = var_53()
    var_55 = {var_40: var_42}
    var_56 = module_1.Form(env=var_18, schema=var_54, values=var_55)
    var_57 = {var_40: var_42}
    var_58 = var_56.validate(var_57)
    var_59 = var_56.render_fields()
    assert var_59 == 'error_name'

import builtins as module_0
import typesystem.forms as module_1

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
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = lambda self, name: var_12
    var_14 = {var_2: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'Field'
    var_20 = ()
    var_21 = 'read_only'
    var_22 = 'title'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = False
    var_27 = 'R'
    var_28 = lambda self: var_26
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_26, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = ()
    var_36 = True
    var_37 = 'RO'
    var_38 = lambda self: var_26
    var_39 = {var_21: var_36, var_22: var_37, var_23: var_26, var_24: var_38, var_25: var_29}
    var_40 = [var_19, var_35, var_39]
    var_41 = {}
    var_42 = module_0.type(*var_40, **var_41)
    var_43 = var_42()
    var_44 = 'Schema'
    var_45 = ()
    var_46 = 'fields'
    var_47 = 'serialize'
    var_48 = 'validate_or_error'
    var_49 = 'r'
    var_50 = 'ro'
    var_51 = {var_49: var_34, var_50: var_43}
    var_52 = {}
    var_53 = lambda self, v: var_52
    var_54 = {}
    var_55 = None
    var_56 = (var_54, var_55)
    var_57 = lambda self, d: var_56
    var_58 = {var_46: var_51, var_47: var_53, var_48: var_57}
    var_59 = [var_44, var_45, var_58]
    var_60 = {}
    var_61 = module_0.type(*var_59, **var_60)
    var_62 = var_61()
    var_63 = {}
    var_64 = module_1.Form(env=var_18, schema=var_62, values=var_63)
    var_65 = {}
    var_66 = var_64.validate(var_65)
    var_67 = var_64.render_fields()
    assert var_67 == 'rendered_r'

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, ctx: f'val_{ctx[var_6]}'
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = lambda self, name: var_12
    var_14 = {var_2: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'Field'
    var_20 = ()
    var_21 = 'read_only'
    var_22 = 'title'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = False
    var_27 = 'Name'
    var_28 = lambda self: var_26
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_26, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = 'Schema'
    var_36 = ()
    var_37 = 'fields'
    var_38 = 'serialize'
    var_39 = 'validate_or_error'
    var_40 = 'name'
    var_41 = {var_40: var_34}
    var_42 = 'old'
    var_43 = {var_40: var_42}
    var_44 = lambda self, v: var_43
    var_45 = 'new'
    var_46 = {var_40: var_45}
    var_47 = 'err'
    var_48 = {var_40: var_47}
    var_49 = (var_46, var_48)
    var_50 = lambda self, d: var_49
    var_51 = {var_37: var_41, var_38: var_44, var_39: var_50}
    var_52 = [var_35, var_36, var_51]
    var_53 = {}
    var_54 = module_0.type(*var_52, **var_53)
    var_55 = var_54()
    var_56 = {var_40: var_42}
    var_57 = module_1.Form(env=var_18, schema=var_55, values=var_56)
    var_58 = {var_40: var_45}
    var_59 = var_57.validate(var_58)
    var_60 = var_57.render_fields()
    assert var_60 == 'val_new'



# Parsed testcases at query #14
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, name: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = {}
    var_14 = lambda self, values: values if values is not var_3 else var_13
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = 'age'
    var_22 = 'test_user'
    var_23 = 30
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_1.Form(env=var_9, schema=var_19, values=var_24)
    var_26 = var_25.env
    var_27 = bool(var_25.env == var_9)
    assert var_27 is True
    var_28 = var_25.schema
    var_29 = bool(var_25.schema == var_19)
    assert var_29 is True
    var_30 = var_25.values
    var_31 = bool(var_25.values == var_24)
    assert var_31 is True
    var_32 = var_25.errors
    assert var_32 is None
    var_33 = var_25._validate_called
    assert var_33 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = None
    var_4 = lambda self, name: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_8()
    var_10 = 'MockSchema'
    var_11 = ()
    var_12 = 'serialize'
    var_13 = {}
    var_14 = lambda self, values: var_13
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = module_1.Form(env=var_9, schema=var_19, values=var_3)
    var_21 = var_20.values
    var_22 = bool(var_20.values == {})
    assert var_22 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_template_for_field_choice. Retrieved 2/12 statements.
# Partially parsed test_template_for_field_boolean. Retrieved 2/12 statements.
# Partially parsed test_template_for_field_string_textarea. Retrieved 2/12 statements.
# Partially parsed test_template_for_field_string_input. Retrieved 2/12 statements.
# Partially parsed test_template_for_field_default_input. Retrieved 2/12 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = 'choice'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'bool_field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'text_field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'email_field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'generic_field'



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_template_for_field_does_not_support_object_fields.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_template_for_field_choice_returns_select_template. Retrieved 5/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Environment()
    var_4 = 'template'
    var_5 = var_3.from_string(var_4)
    var_6 = {}
    var_7 = 'choice_field'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_template_for_field_raises_assertion_error_on_object_field. Retrieved 4/18 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'obj_field'
    var_3 = 'AssertionError was not raised for Object field'
    var_4 = AssertionError(var_3)



