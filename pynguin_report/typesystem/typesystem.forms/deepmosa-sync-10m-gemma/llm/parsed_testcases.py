####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 3/9 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 4/10 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/15 statements.
# Partially parsed test_load_template_env_assertion_error_on_both_provided. Retrieved 2/19 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'
    var_3 = True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some/path'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = True

def test_case_0():
    var_0 = 'd'
    var_1 = 'p'



# Parsed testcases at query #2
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

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_form_html_returns_markup_with_rendered_fields. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = {var_0: var_1}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_field_basic. Retrieved 1/13 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Full Name'
    var_2 = 'title'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}



# Parsed testcases at query #5
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
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = {var_13: var_14}
    var_17 = {}
    var_18 = lambda self, v: var_16 if v == var_15 else var_17
    var_19 = {var_12: var_18}
    var_20 = [var_10, var_11, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = {var_13: var_14}
    var_25 = module_1.Form(env=var_9, schema=var_23, values=var_24)
    var_26 = var_25.env
    var_27 = bool(var_25.env == var_9)
    assert var_27 is True
    var_28 = var_25.schema
    var_29 = bool(var_25.schema == var_23)
    assert var_29 is True
    var_30 = var_25.values
    var_31 = bool(var_25.values == {'key': 'value'})
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_form_init_keyword_only_arguments. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 2/6 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 3/7 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/12 statements.
# Partially parsed test_load_template_env_raises_assertion_error_on_ambiguous_params. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = 'pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #8
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
    var_13 = 'name'
    var_14 = 'John'
    var_15 = {var_13: var_14}
    var_16 = {var_13: var_14}
    var_17 = {}
    var_18 = lambda self, v: var_16 if v == var_15 else var_17
    var_19 = {var_12: var_18}
    var_20 = [var_10, var_11, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = {var_13: var_14}
    var_25 = module_1.Form(env=var_9, schema=var_23, values=var_24)
    var_26 = var_25.env
    var_27 = bool(var_25.env == var_9)
    assert var_27 is True
    var_28 = var_25.schema
    var_29 = bool(var_25.schema == var_23)
    assert var_29 is True
    var_30 = var_25.values
    var_31 = bool(var_25.values == {'name': 'John'})
    assert var_31 is True
    var_32 = var_25.errors
    assert var_32 is None
    var_33 = var_25._validate_called
    assert var_33 is False

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



# Parsed testcases at query #9
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'Template'
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
    var_40 = 'username'
    var_41 = {var_40: var_34}
    var_42 = 'john'
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
    assert var_58 == 'rendered_username'

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'Template'
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
    var_40 = 'username'
    var_41 = {var_40: var_34}
    var_42 = {}
    var_43 = lambda self, v: var_42
    var_44 = ''
    var_45 = {var_40: var_44}
    var_46 = 'Required'
    var_47 = {var_40: var_46}
    var_48 = (var_45, var_47)
    var_49 = lambda self, d: var_48
    var_50 = {var_37: var_41, var_38: var_43, var_39: var_49}
    var_51 = [var_35, var_36, var_50]
    var_52 = {}
    var_53 = module_0.type(*var_51, **var_52)
    var_54 = var_53()
    var_55 = {}
    var_56 = module_1.Form(env=var_18, schema=var_54, values=var_55)
    var_57 = {var_40: var_44}
    var_58 = var_56.validate(var_57)
    var_59 = var_56.render_fields()
    assert var_59 == 'error_username'

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'Template'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'field_name'
    var_7 = lambda self, ctx: f'{ctx[var_6]}'
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
    var_27 = 'Active'
    var_28 = lambda self: var_26
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_26, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = ()
    var_36 = True
    var_37 = 'ReadOnly'
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
    var_49 = 'active'
    var_50 = 'readonly'
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
    assert var_67 == 'active'

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Env'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'Template'
    var_4 = ()
    var_5 = 'render'
    var_6 = 'value'
    var_7 = lambda self, ctx: f'{ctx[var_6]}'
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
    var_40 = 'username'
    var_41 = {var_40: var_34}
    var_42 = {}
    var_43 = lambda self, v: var_42
    var_44 = 'cleaned'
    var_45 = {var_40: var_44}
    var_46 = 'error'
    var_47 = {var_40: var_46}
    var_48 = (var_45, var_47)
    var_49 = lambda self, d: var_48
    var_50 = {var_37: var_41, var_38: var_43, var_39: var_49}
    var_51 = [var_35, var_36, var_50]
    var_52 = {}
    var_53 = module_0.type(*var_51, **var_52)
    var_54 = var_53()
    var_55 = {}
    var_56 = module_1.Form(env=var_18, schema=var_54, values=var_55)
    var_57 = 'raw_input'
    var_58 = {var_40: var_57}
    var_59 = var_56.validate(var_58)
    var_60 = var_56.render_fields()
    assert var_60 == 'raw_input'



# Parsed testcases at query #10
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
    var_13 = 'name'
    var_14 = 'test_value'
    var_15 = {var_13: var_14}
    var_16 = lambda self, v: var_15
    var_17 = {var_12: var_16}
    var_18 = [var_10, var_11, var_17]
    var_19 = {}
    var_20 = module_0.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'original_value'
    var_23 = {var_13: var_22}
    var_24 = module_1.Form(env=var_9, schema=var_21, values=var_23)
    var_25 = var_24.env
    var_26 = bool(var_24.env == var_9)
    assert var_26 is True
    var_27 = var_24.schema
    var_28 = bool(var_24.schema == var_21)
    assert var_28 is True
    var_29 = var_24.values
    var_30 = bool(var_24.values == {'name': 'test_value'})
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
    var_13 = lambda self, v: var_3
    var_14 = {var_12: var_13}
    var_15 = [var_10, var_11, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = module_1.Form(env=var_9, schema=var_18, values=var_3)
    var_20 = var_19.values
    assert var_20 is None



# Parsed testcases at query #11
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = var_8.input_type_for_field(var_9)
    assert var_10 == 'text'

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
    var_9 = var_7.input_type_for_field(var_8)
    assert var_9 == 'email'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'age'
    var_2 = 'number'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = var_8.input_type_for_field(var_9)
    assert var_10 == 'number'

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
    var_10 = var_8.input_type_for_field(var_9)
    assert var_10 == 'text'

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
    var_8 = var_6.fields[var_1]
    var_9 = var_7.input_type_for_field(var_8)
    assert var_9 == 'text'

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
    var_9 = var_7.fields[var_1]
    var_10 = var_8.input_type_for_field(var_9)
    assert var_10 == 'date'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'secret'
    var_2 = 'password'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = var_8.input_type_for_field(var_9)
    assert var_10 == 'password'

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
    var_9 = var_7.fields[var_1]
    var_10 = var_8.input_type_for_field(var_9)
    assert var_10 == 'url'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_form_str_returns_rendered_fields. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 5/16 statements.
# Partially parsed test_render_field_generates_correct_id. Retrieved 4/15 statements.
# Partially parsed test_render_field_includes_error_in_context. Retrieved 4/15 statements.
# Partially parsed test_render_field_uses_title_as_label. Retrieved 5/16 statements.
# Partially parsed test_render_field_uses_field_name_as_label_when_no_title. Retrieved 4/15 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'password'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'password_field'
    var_4 = 'secret123'
    var_5 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'user_name'
    var_3 = 'test'
    var_4 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'username'
    var_3 = 'Invalid input'
    var_4 = 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Full Name'
    var_1 = 'title'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'full_name'
    var_5 = 'John Doe'
    var_6 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'username'
    var_3 = 'test'
    var_4 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_template_env_directory_only. Retrieved 2/7 statements.
# Partially parsed test_load_template_env_package_only. Retrieved 3/8 statements.
# Partially parsed test_load_template_env_both_directory_and_package. Retrieved 4/13 statements.
# Partially parsed test_load_template_env_invalid_params_raises_assertion. Retrieved 4/10 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_2, package=var_2)



# Parsed testcases at query #15
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #16
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
    var_13 = 'name'
    var_14 = 'John'
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
    var_30 = bool(var_24.values == {'name': 'John'})
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_template_for_field_choice. Retrieved 4/13 statements.
# Partially parsed test_template_for_field_boolean. Retrieved 2/10 statements.
# Partially parsed test_template_for_field_string_text_format. Retrieved 3/11 statements.
# Partially parsed test_template_for_field_string_other_format. Retrieved 3/11 statements.
# Partially parsed test_template_for_field_default_case. Retrieved 2/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Boolean(**var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'text'
    var_2 = {}
    var_3 = module_0.String(format=var_1, **var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(format=var_1, **var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Field()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 4/18 statements.
# Partially parsed test_render_field_uses_correct_id_and_label. Retrieved 5/19 statements.
# Partially parsed test_render_field_determines_required_correctly. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'password_field'
    var_1 = 'password_field'
    var_2 = 'password'
    var_3 = 'secret123'
    var_4 = None

def test_case_0():
    var_0 = {}
    var_1 = 'Display Title'
    var_2 = 'test_field_name'
    var_3 = 'some_value'
    var_4 = 'some_error'

def test_case_0():
    var_0 = {}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_field_basic_input. Retrieved 5/24 statements.
# Partially parsed test_render_field_password_masks_value. Retrieved 9/23 statements.
# Failed to parse test_render_field_with_error.


def test_case_0():
    var_0 = 'MockSchema'
    var_1 = ()
    var_2 = 'fields'
    var_3 = 'username'
    var_4 = 'john_doe'

import builtins as module_0

def test_case_0():
    var_0 = 'MockSchema'
    var_1 = ()
    var_2 = 'fields'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'password'
    var_10 = 'secret123'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_template_for_field_does_not_support_object_fields. Retrieved 1/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_field_basic_string_input. Retrieved 5/19 statements.
# Partially parsed test_render_field_password_hides_value. Retrieved 5/18 statements.
# Partially parsed test_render_field_with_error. Retrieved 5/18 statements.
# Partially parsed test_render_field_id_transformation. Retrieved 4/17 statements.
# Partially parsed test_render_field_template_selection_boolean. Retrieved 5/18 statements.
# Partially parsed test_render_field_template_selection_choice. Retrieved 5/18 statements.
# Partially parsed test_render_field_required_logic. Retrieved 7/23 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'User Name'
    var_2 = 'title'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = 'username'
    var_7 = 'john_doe'
    var_8 = None
    var_9 = 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'password'
    var_1 = 'password'
    var_2 = {}
    var_3 = module_0.String(format=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = 'password'
    var_6 = 'secret123'
    var_7 = None
    var_8 = 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'age'
    var_5 = None
    var_6 = 'Must be a number'
    var_7 = 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = 'first_name'
    var_5 = 'John'
    var_6 = 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'is_active'
    var_1 = {}
    var_2 = module_0.Boolean(**var_1)
    var_3 = {var_0: var_2}
    var_4 = 'is_active'
    var_5 = True
    var_6 = None
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'status'
    var_1 = 'a'
    var_2 = 'A'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = 'status'
    var_9 = 'a'
    var_10 = None
    var_11 = 'forms/select.html'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'null_ok'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = 'something'
    var_6 = 'default'
    var_7 = {var_6: var_5}
    var_8 = module_0.String(**var_7)
    var_9 = True
    var_10 = 'allow_null'
    var_11 = {var_10: var_9}
    var_12 = module_0.String(**var_11)
    var_13 = {var_0: var_4, var_1: var_8, var_2: var_12}
    var_14 = 'req'
    var_15 = 'val'
    var_16 = 'opt'
    var_17 = 'something'
    var_18 = 'null_ok'
    var_19 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_template_for_field_boolean_returns_checkbox_template. Retrieved 5/16 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'bool_field'
    var_1 = {}
    var_2 = module_0.Boolean(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = module_0.Boolean(**var_6)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_field_label_uses_field_title. Retrieved 7/28 statements.
# Partially parsed test_render_field_label_uses_field_name_when_title_is_empty. Retrieved 7/28 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Custom Label'
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = 'template.html'
    var_4 = 'text'
    var_5 = 'test_field'
    var_6 = 'val'

import typesystem.fields as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = 'template.html'
    var_4 = 'text'
    var_5 = 'test_field'
    var_6 = 'val'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_form_init_is_not_keyword_only_for_env. Retrieved 6/12 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = True
    var_7 = False
    assert var_7 is False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_field_basic_input. Retrieved 5/16 statements.
# Partially parsed test_render_field_with_error. Retrieved 5/16 statements.
# Partially parsed test_render_field_password_hides_value. Retrieved 4/15 statements.
# Partially parsed test_render_field_id_conversion. Retrieved 4/15 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User Name'
    var_1 = 'title'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'user_name'
    var_5 = 'John Doe'
    var_6 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Email'
    var_1 = 'title'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'email'
    var_5 = ''
    var_6 = 'Invalid email'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'password'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'secret123'
    var_4 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'first_name'
    var_3 = 'John'
    var_4 = None



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 2/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 3/7 statements.
# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 4/12 statements.


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
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)



# Parsed testcases at query #2
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
    var_10 = lambda self, v: v
    var_11 = {var_9: var_10}
    var_12 = [var_7, var_8, var_11]
    var_13 = {}
    var_14 = module_0.type(*var_12, **var_13)
    var_15 = var_14()
    var_16 = 'name'
    var_17 = 'John'
    var_18 = {var_16: var_17}
    var_19 = module_1.Form(env=var_6, schema=var_15, values=var_18)
    var_20 = var_19.env
    var_21 = bool(var_19.env == var_6)
    assert var_21 is True
    var_22 = var_19.schema
    var_23 = bool(var_19.schema == var_15)
    assert var_23 is True
    var_24 = var_19.values
    var_25 = bool(var_19.values == var_18)
    assert var_25 is True
    var_26 = var_19.errors
    assert var_26 is None
    var_27 = var_19._validate_called
    assert var_27 is False

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
    var_11 = lambda self, v: var_10
    var_12 = {var_9: var_11}
    var_13 = [var_7, var_8, var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = var_15()
    var_17 = module_1.Form(env=var_6, schema=var_16, values=var_10)
    var_18 = var_17.values
    assert var_18 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_form_html_returns_markup_of_rendered_fields. Retrieved 6/21 statements.


import markupsafe as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = False
    var_3 = 'Name'
    var_4 = '<div>rendered</div>'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Markup(*var_5, **var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_updates_data_and_errors. Retrieved 5/21 statements.
# Partially parsed test_validate_sets_errors_on_failure. Retrieved 5/19 statements.
# Partially parsed test_validate_raises_error_if_called_twice. Retrieved 2/19 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'default'
    var_2 = {var_0: var_1}
    var_3 = 'valid'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 'name'
    var_1 = 'default'
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = {}
    var_1 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_form_str_returns_rendered_fields. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_template_env_predicate_at_line_10_is_false. Retrieved 4/6 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader.loaders[0].searchpath
    var_4 = bool(var_2.env.loader.loaders[0].searchpath == [var_0])
    assert var_4 is True
    var_5 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = 'pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_jinja2forms_init_with_directory. Retrieved 2/6 statements.
# Partially parsed test_jinja2forms_init_with_package. Retrieved 3/7 statements.
# Partially parsed test_jinja2forms_init_with_both. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'templates'

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dir'
    var_1 = 'pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #8
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = 'name'
    var_5 = 'test'
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
    var_20 = lambda self, t: var_9
    var_21 = {var_19: var_20}
    var_22 = [var_17, var_18, var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = var_24()
    var_26 = 'initial'
    var_27 = {var_4: var_26}
    var_28 = module_1.Form(env=var_25, schema=var_16, values=var_27)
    var_29 = 'new_value'
    var_30 = {var_4: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = var_28.data
    var_33 = bool(var_28.data == {'name': 'new_value'})
    assert var_33 is True
    var_34 = var_28.values
    var_35 = bool(var_28.values == {'name': 'test'})
    assert var_35 is True
    var_36 = var_28.errors
    assert var_36 is None
    var_37 = var_28.is_valid
    assert var_37 is True

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = lambda self, v: var_6
    var_8 = 'invalid'
    var_9 = {var_4: var_8}
    var_10 = 'error message'
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
    var_22 = None
    var_23 = lambda self, t: var_22
    var_24 = {var_21: var_23}
    var_25 = [var_19, var_20, var_24]
    var_26 = {}
    var_27 = module_0.type(*var_25, **var_26)
    var_28 = var_27()
    var_29 = 'initial'
    var_30 = {var_4: var_29}
    var_31 = module_1.Form(env=var_28, schema=var_18, values=var_30)
    var_32 = 'bad'
    var_33 = {var_4: var_32}
    var_34 = var_31.validate(var_33)
    var_35 = var_31.errors
    var_36 = bool(var_31.errors == {'name': 'error message'})
    assert var_36 is True
    var_37 = var_31.is_valid
    assert var_37 is False

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
    var_18 = lambda self, t: var_7
    var_19 = {var_17: var_18}
    var_20 = [var_15, var_16, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1.Form(env=var_23, schema=var_14)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = {}
    var_28 = var_24.validate(var_27)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_input_type_for_field_returns_text_when_no_format. Retrieved 1/20 statements.
# Partially parsed test_input_type_for_field_returns_mapped_type. Retrieved 3/26 statements.
# Partially parsed test_input_type_for_field_returns_text_for_unknown_format. Retrieved 1/20 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 'email'
    var_1 = 'number'
    var_2 = 'date'

def test_case_0():
    var_0 = 'unsupported_type'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_input_type_for_field_no_format_returns_text. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'email'
    var_2 = {var_0: var_1}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_input_type_for_field_returns_text_when_no_format. Retrieved 1/10 statements.
# Partially parsed test_input_type_for_field_returns_mapped_type. Retrieved 1/11 statements.
# Partially parsed test_input_type_for_field_returns_text_for_unmapped_format. Retrieved 1/11 statements.
# Partially parsed test_input_type_for_field_handles_numeric_format. Retrieved 1/11 statements.
# Partially parsed test_input_type_for_field_handles_date_format. Retrieved 1/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_form_html_returns_markup_of_rendered_fields.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_form_html_returns_markup_instance_of_rendered_fields. Retrieved 5/33 statements.
# Partially parsed test_form_html_calls_render_fields. Retrieved 3/26 statements.


import markupsafe as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'rendered-forms/input.html'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Markup(*var_4, **var_5)

import markupsafe as module_0

def test_case_0():
    var_0 = 'f'
    var_1 = 'tpl'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Markup(*var_2, **var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_form_html_returns_markup_of_rendered_fields. Retrieved 6/26 statements.


import markupsafe as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = '<div>rendered_content</div>'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Markup(*var_5, **var_6)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_form_init_keyword_only_args_not_at_line_3. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
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
    var_6 = ''
    var_7 = lambda self, ctx: var_6
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
    var_19 = 'MockSchema'
    var_20 = ()
    var_21 = 'serialize'
    var_22 = 'validate_or_error'
    var_23 = 'fields'
    var_24 = None
    var_25 = {}
    var_26 = lambda self, v: v if v is not var_24 else var_25
    var_27 = {}
    var_28 = (var_27, var_24)
    var_29 = lambda self, d: var_28
    var_30 = {}
    var_31 = {var_21: var_26, var_22: var_29, var_23: var_30}
    var_32 = [var_19, var_20, var_31]
    var_33 = {}
    var_34 = module_0.type(*var_32, **var_33)
    var_35 = var_34()
    var_36 = 'test'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = module_1.Form(env=var_18, schema=var_35, values=var_38)
    var_40 = {var_36: var_37}
    var_41 = var_39.validate(var_40)
    var_42 = var_39._validate_called
    assert var_42 is True



# Parsed testcases at query #17
#--------------------------




import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = None
    var_5 = {}
    var_6 = lambda self, v: v if v is not var_4 else var_5
    var_7 = lambda self, d: (d, var_4)
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'Env'
    var_14 = ()
    var_15 = 'get_template'
    var_16 = lambda self, t: var_4
    var_17 = {var_15: var_16}
    var_18 = [var_13, var_14, var_17]
    var_19 = {}
    var_20 = module_0.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'name'
    var_23 = 'test'
    var_24 = {var_22: var_23}
    var_25 = module_1.Form(env=var_21, schema=var_12, values=var_24)
    var_26 = {var_22: var_23}
    var_27 = var_25.validate(var_26)
    var_28 = var_25.values
    var_29 = bool(var_25.values == {'name': 'test'})
    assert var_29 is True
    var_30 = var_25.errors
    assert var_30 is None
    var_31 = var_25.is_valid
    assert var_31 is True
    var_32 = var_25.validated_data
    var_33 = bool(var_25.validated_data == {'name': 'test'})
    assert var_33 is True

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = None
    var_5 = {}
    var_6 = lambda self, v: v if v is not var_4 else var_5
    var_7 = 'name'
    var_8 = ''
    var_9 = {var_7: var_8}
    var_10 = 'Required'
    var_11 = {var_7: var_10}
    var_12 = (var_9, var_11)
    var_13 = lambda self, d: var_12
    var_14 = {var_2: var_6, var_3: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = 'Env'
    var_20 = ()
    var_21 = 'get_template'
    var_22 = lambda self, t: var_4
    var_23 = {var_21: var_22}
    var_24 = [var_19, var_20, var_23]
    var_25 = {}
    var_26 = module_0.type(*var_24, **var_25)
    var_27 = var_26()
    var_28 = {var_7: var_8}
    var_29 = module_1.Form(env=var_27, schema=var_18, values=var_28)
    var_30 = {var_7: var_8}
    var_31 = var_29.validate(var_30)
    var_32 = var_29.errors
    var_33 = bool(var_29.errors == {'name': 'Required'})
    assert var_33 is True
    var_34 = var_29.is_valid
    assert var_34 is False

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = 'validate_or_error'
    var_4 = None
    var_5 = {}
    var_6 = lambda self, v: v if v is not var_4 else var_5
    var_7 = lambda self, d: (d, var_4)
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'Env'
    var_14 = ()
    var_15 = 'get_template'
    var_16 = lambda self, t: var_4
    var_17 = {var_15: var_16}
    var_18 = [var_13, var_14, var_17]
    var_19 = {}
    var_20 = module_0.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = {}
    var_23 = module_1.Form(env=var_21, schema=var_12, values=var_22)
    var_24 = 'data'
    var_25 = 'val'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)
    var_28 = 'data'
    var_29 = 'another'
    var_30 = {var_28: var_29}
    var_31 = var_23.validate(var_30)
    var_32 = 'Should have raised ValueError'
    var_33 = AssertionError(var_32)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_field_basic_string_field. Retrieved 6/40 statements.
# Partially parsed test_render_field_with_error. Retrieved 5/34 statements.
# Partially parsed test_render_field_password_masks_value. Retrieved 6/35 statements.


def test_case_0():
    var_0 = 'user_name'
    var_1 = 'admin'
    var_2 = {var_0: var_1}
    var_3 = 'input.html'
    var_4 = 'text'
    var_5 = None

def test_case_0():
    var_0 = 'email'
    var_1 = 'bad'
    var_2 = {var_0: var_1}
    var_3 = 'input.html'
    var_4 = 'Invalid email'

def test_case_0():
    var_0 = 'pwd'
    var_1 = 'secret123'
    var_2 = {var_0: var_1}
    var_3 = 'input.html'
    var_4 = 'password'
    var_5 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_form_str_calls_render_fields. Retrieved 49/50 statements.


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
    var_21 = 'title'
    var_22 = 'read_only'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = 'Test'
    var_27 = False
    var_28 = lambda self: var_27
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_27, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = 'Schema'
    var_36 = ()
    var_37 = 'serialize'
    var_38 = 'validate_or_error'
    var_39 = 'fields'
    var_40 = {}
    var_41 = lambda self, v: var_40
    var_42 = {}
    var_43 = None
    var_44 = (var_42, var_43)
    var_45 = lambda self, d: var_44
    var_46 = 'name'
    var_47 = {var_46: var_34}
    var_48 = {var_37: var_41, var_38: var_45, var_39: var_47}
    var_49 = [var_35, var_36, var_48]
    var_50 = {}
    var_51 = module_0.type(*var_49, **var_50)
    var_52 = var_51()
    var_53 = module_1.Form(env=var_18, schema=var_52)
    var_54 = str(var_53)
    assert var_54 == 'rendered_name'

import builtins as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'MockEnv'
    var_1 = ()
    var_2 = 'get_template'
    var_3 = 'MockTemplate'
    var_4 = ()
    var_5 = 'render'
    var_6 = ''
    var_7 = lambda self, ctx: var_6
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
    var_21 = 'title'
    var_22 = 'read_only'
    var_23 = 'allow_null'
    var_24 = 'has_default'
    var_25 = 'format'
    var_26 = 'Test'
    var_27 = False
    var_28 = lambda self: var_27
    var_29 = 'text'
    var_30 = {var_21: var_26, var_22: var_27, var_23: var_27, var_24: var_28, var_25: var_29}
    var_31 = [var_19, var_20, var_30]
    var_32 = {}
    var_33 = module_0.type(*var_31, **var_32)
    var_34 = var_33()
    var_35 = 'Schema'
    var_36 = ()
    var_37 = 'serialize'
    var_38 = 'validate_or_error'
    var_39 = 'fields'
    var_40 = {}
    var_41 = lambda self, v: var_40
    var_42 = {}
    var_43 = None
    var_44 = (var_42, var_43)
    var_45 = lambda self, d: var_44
    var_46 = 'name'
    var_47 = {var_46: var_34}
    var_48 = {var_37: var_41, var_38: var_45, var_39: var_47}
    var_49 = [var_35, var_36, var_48]
    var_50 = {}
    var_51 = module_0.type(*var_49, **var_50)
    var_52 = var_51()
    var_53 = module_1.Form(env=var_18, schema=var_52)
    var_54 = 'value'
    var_55 = str(var_53)
    var_56 = var_53.render_fields()
    var_57 = bool(var_55 == var_56)
    assert var_57 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_init_fails_when_jinja2_is_none. Retrieved 4/10 statements.
# Partially parsed test_init_assert_jinja2_is_not_none. Retrieved 4/8 statements.


def test_case_0():
    pass

def test_case_0():
    pass

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = None
    var_2 = 'some_dir'
    var_3 = module_0.Jinja2Forms(directory=var_2)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = None
    var_2 = 'test_dir'
    var_3 = module_0.Jinja2Forms(directory=var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_field_password_masks_value. Retrieved 4/19 statements.
# Partially parsed test_render_field_input_type_mapping. Retrieved 5/19 statements.
# Partially parsed test_render_field_required_logic. Retrieved 3/26 statements.
# Partially parsed test_render_field_template_selection. Retrieved 13/34 statements.


def test_case_0():
    var_0 = 'password'
    var_1 = 'password'
    var_2 = 'secret123'
    var_3 = 0

def test_case_0():
    var_0 = 'email'
    var_1 = 'email'
    var_2 = 'user_email'
    var_3 = 'test@example.com'
    var_4 = 0

def test_case_0():
    var_0 = 'req'
    var_1 = 'val'
    var_2 = 'def_field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'txt'
    var_2 = 'bool'
    var_3 = 'choice'
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]
    var_9 = 'hi'
    var_10 = 'forms/textarea.html'
    var_11 = True
    var_12 = 'forms/checkbox.html'
    var_13 = 'forms/select.html'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_form_constructor_initializes_correctly. Retrieved 5/11 statements.
# Partially parsed test_form_constructor_with_none_values. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = None
    var_1 = None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_called_twice. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 'Expected AssertionError was not raised'
    var_10 = AssertionError(var_9)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_render_field_label_uses_field_title. Retrieved 7/28 statements.
# Partially parsed test_render_field_label_falls_back_to_field_name. Retrieved 6/25 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Custom Label'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'test_field'
    var_3 = 'val'
    var_4 = None
    var_5 = 0
    var_6 = {}
    var_7 = var_6['label']
    assert var_7 == 'Custom Label'

import typesystem.fields as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = 'test_field'
    var_4 = 'val'
    var_5 = None
    var_6 = var_2['label']
    assert var_6 == 'test_field'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_fields_valid_data. Retrieved 6/37 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 4/32 statements.


def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = 'ignored'
    var_3 = 'John'
    var_4 = {var_1: var_3}
    var_5 = {var_1: var_3}
    var_6 = 'html_name'
    var_7 = 'html_ignored'

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = 'val_John_err_Error msg'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_template_env_predicate_false. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/fake'
    var_1 = 'fake_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



