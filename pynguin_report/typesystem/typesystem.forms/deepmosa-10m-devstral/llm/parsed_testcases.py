####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_jinja2forms_constructor_with_directory. Retrieved 3/5 statements.
# Partially parsed test_jinja2forms_constructor_with_package. Retrieved 3/5 statements.
# Partially parsed test_jinja2forms_constructor_with_both_directory_and_package. Retrieved 4/6 statements.
# Partially parsed test_jinja2forms_constructor_without_jinja2_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_1.env.loader is not None)
    assert var_3 is True
    var_4 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_1.env.loader is not None)
    assert var_3 is True
    var_4 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = bool(var_2.env.loader is not None)
    assert var_4 is True
    var_5 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'path/to/templates'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #2
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
    var_5 = str(var_4)
    var_6 = var_4.render_fields()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {}
    var_8 = module_2.String(**var_7)
    var_9 = {}
    var_10 = module_2.Integer(**var_9)
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = {}
    var_13 = module_3.Schema(var_11, **var_12)
    var_14 = 'John'
    var_15 = 30
    var_16 = {var_5: var_14, var_6: var_15}
    var_17 = module_4.Form(env=var_4, schema=var_13, values=var_16)
    var_18 = {var_5: var_14, var_6: var_15}
    var_19 = var_17.validate(var_18)
    var_20 = var_17.render_fields()
    var_21 = '<input />'
    var_22 = bool('<input />' in var_20)
    assert var_22 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {}
    var_8 = module_2.String(**var_7)
    var_9 = {}
    var_10 = module_2.Integer(**var_9)
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = {}
    var_13 = module_3.Schema(var_11, **var_12)
    var_14 = 'John'
    var_15 = 'invalid'
    var_16 = {var_5: var_14, var_6: var_15}
    var_17 = module_4.Form(env=var_4, schema=var_13, values=var_16)
    var_18 = {var_5: var_14, var_6: var_15}
    var_19 = var_17.validate(var_18)
    var_20 = var_17.render_fields()
    var_21 = '<input />'
    var_22 = bool('<input />' in var_20)
    assert var_22 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'id'
    var_7 = {}
    var_8 = module_2.String(**var_7)
    var_9 = True
    var_10 = 'read_only'
    var_11 = {var_10: var_9}
    var_12 = module_2.Integer(**var_11)
    var_13 = {var_5: var_8, var_6: var_12}
    var_14 = {}
    var_15 = module_3.Schema(var_13, **var_14)
    var_16 = 'John'
    var_17 = {var_5: var_16, var_6: var_9}
    var_18 = module_4.Form(env=var_4, schema=var_15, values=var_17)
    var_19 = {var_5: var_16}
    var_20 = var_18.validate(var_19)
    var_21 = var_18.render_fields()
    var_22 = 'id'
    var_23 = bool('id' not in var_21)
    assert var_23 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {}
    var_8 = module_2.String(**var_7)
    var_9 = {}
    var_10 = module_2.Integer(**var_9)
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = {}
    var_13 = module_3.Schema(var_11, **var_12)
    var_14 = 'John'
    var_15 = 30
    var_16 = {var_5: var_14, var_6: var_15}
    var_17 = module_4.Form(env=var_4, schema=var_13, values=var_16)
    var_18 = var_17.render_fields()
    var_19 = '<input />'
    var_20 = bool('<input />' in var_18)
    assert var_20 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 11/17 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/test/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.searchpath
    var_5 = bool(var_2.loader.searchpath == ['/test/templates'])
    assert var_5 is True
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.package_name
    assert var_4 == 'test.package'
    var_5 = var_2.loader.package_path
    assert var_5 == 'templates'
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/test/templates'
    var_1 = 'test.package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader
    var_5 = var_3.loader.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_3.loader.loaders[var_7]
    var_9 = 1
    var_10 = var_3.loader.loaders[var_9]
    var_11 = var_3.loader.loaders[0].searchpath
    var_12 = bool(var_3.loader.loaders[0].searchpath == ['/test/templates'])
    assert var_12 is True
    var_13 = var_3.loader.loaders[1].package_name
    assert var_13 == 'test.package'
    var_14 = var_3.loader.loaders[1].package_path
    assert var_14 == 'templates'
    var_15 = var_3.autoescape
    assert var_15 is True



# Parsed testcases at query #6
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = {}
    var_3 = module_1.Schema(var_1, **var_2)
    var_4 = {}
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_field_with_valid_input. Retrieved 8/13 statements.
# Partially parsed test_render_field_with_error. Retrieved 8/13 statements.
# Partially parsed test_render_field_with_password_input. Retrieved 9/14 statements.
# Partially parsed test_render_field_with_checkbox_input. Retrieved 8/13 statements.
# Partially parsed test_render_field_with_select_input. Retrieved 14/19 statements.
# Partially parsed test_render_field_with_textarea_input. Retrieved 9/14 statements.
# Partially parsed test_render_field_with_read_only_field. Retrieved 9/14 statements.
# Partially parsed test_render_field_with_required_field. Retrieved 8/13 statements.
# Partially parsed test_render_field_with_optional_field. Retrieved 9/14 statements.
# Partially parsed test_render_field_with_default_value. Retrieved 9/14 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'test_value'
    var_8 = {var_1: var_7}
    var_9 = var_6.fields[var_1]
    var_10 = None
    var_11 = 'test_field'
    var_12 = 'test_value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 123
    var_8 = {var_1: var_7}
    var_9 = var_6.fields[var_1]
    var_10 = 'Must be a string.'
    var_11 = 'test_field'
    var_12 = 'Must be a string.'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'password_field'
    var_2 = 'password'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'secret'
    var_9 = {var_1: var_8}
    var_10 = var_7.fields[var_1]
    var_11 = None
    var_12 = 'password_field'
    var_13 = 'secret'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'checkbox_field'
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = True
    var_8 = {var_1: var_7}
    var_9 = var_6.fields[var_1]
    var_10 = None
    var_11 = 'checkbox_field'
    var_12 = 'checkbox'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'select_field'
    var_2 = 'a'
    var_3 = 'A'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 'B'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_0.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = {var_1: var_2}
    var_15 = var_13.fields[var_1]
    var_16 = None
    var_17 = 'select_field'
    var_18 = 'select'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'textarea_field'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'long text'
    var_9 = {var_1: var_8}
    var_10 = var_7.fields[var_1]
    var_11 = None
    var_12 = 'textarea_field'
    var_13 = 'textarea'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'read_only_field'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'value'
    var_10 = {var_1: var_9}
    var_11 = var_8.fields[var_1]
    var_12 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'required_field'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'value'
    var_8 = {var_1: var_7}
    var_9 = var_6.fields[var_1]
    var_10 = None
    var_11 = 'required'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'optional_field'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'value'
    var_10 = {var_1: var_9}
    var_11 = var_8.fields[var_1]
    var_12 = None
    var_13 = 'required'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'default_field'
    var_2 = 'default'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'value'
    var_10 = {var_1: var_9}
    var_11 = var_8.fields[var_1]
    var_12 = None
    var_13 = 'required'



# Parsed testcases at query #8
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_required_is_false_when_field_has_default_and_allow_empty. Retrieved 9/12 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = True
    var_2 = module_0.Field(default=var_0, allow_null=var_1)
    var_3 = var_2.has_default()
    var_4 = var_2.allow_null
    var_5 = 'allow_blank'
    var_6 = False
    var_7 = getattr(var_2, var_5, var_6)
    var_8 = var_4 or var_7



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_required_predicate_false. Retrieved 13/21 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'value'
    var_10 = {var_1: var_9}
    var_11 = var_8.fields[var_1]
    var_12 = var_11.allow_null
    var_13 = 'allow_blank'
    var_14 = False
    var_15 = getattr(var_11, var_13, var_14)
    var_16 = var_12 or var_15



# Parsed testcases at query #11
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0, values=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_allow_empty_with_allow_blank. Retrieved 6/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'allow_blank'
    var_3 = False
    var_4 = getattr(var_1, var_2, var_3)
    var_5 = var_4 is var_0
    var_6 = bool(var_1.allow_null or var_5)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_blank'
    var_2 = False
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = True
    var_5 = var_3 is var_4
    var_6 = bool(var_0.allow_null or var_5)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_blank'
    var_2 = False
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = var_3 is var_2
    var_5 = bool(var_0.allow_null or var_4)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_required_false_when_field_has_default_and_allow_null. Retrieved 9/12 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1, allow_null=var_0)
    var_3 = var_2.has_default()
    var_4 = var_2.allow_null
    var_5 = 'allow_blank'
    var_6 = False
    var_7 = getattr(var_2, var_5, var_6)
    var_8 = var_4 or var_7



# Parsed testcases at query #15
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #17
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
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.values
    var_11 = bool(var_9.values == {'name': 'test'})
    assert var_11 is True



# Parsed testcases at query #18
#--------------------------




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
    var_7 = 'test'
    var_8 = {var_1: var_7}
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
    var_7 = None
    var_8 = module_3.Form(env=var_0, schema=var_6, values=var_7)
    var_9 = var_8.values
    assert var_9 is None

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
    var_7 = {}
    var_8 = module_3.Form(env=var_0, schema=var_6, values=var_7)
    var_9 = var_8.values
    var_10 = bool(var_8.values == {})
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'text'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.Environment()
    var_4 = {}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = module_3.Form(env=var_3, schema=var_6, values=var_7)
    var_9 = var_8.template_for_field(var_2)
    assert var_9 == 'forms/textarea.html'



# Parsed testcases at query #20
#--------------------------




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
    var_7 = {}
    var_8 = module_3.Form(env=var_0, schema=var_6, values=var_7)
    var_9 = 'secret'
    var_10 = var_8.render_field(field_name=var_1, field=var_3, value=var_9)
    assert var_10 == ''



# Parsed testcases at query #21
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_template_for_field_with_other_field_types. Retrieved 2/4 statements.


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = []

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #23
#--------------------------




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
    var_10 = var_7.render_field(field_name=var_1, field=var_8, value=var_9)
    var_11 = '<input'
    var_12 = bool('<input' in var_10)
    assert var_12 is True
    var_13 = 'value=""'
    var_14 = bool('value=""' in var_10)
    assert var_14 is True



# Parsed testcases at query #24
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
    var_4 = {}
    var_5 = module_2.Form(env=var_0, schema=var_3, values=var_4)
    var_6 = {}
    var_7 = 'fields'
    var_8 = {var_7: var_6}
    var_9 = module_3.Object(**var_8)
    var_10 = var_5.template_for_field(var_9)



# Parsed testcases at query #25
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #27
#--------------------------




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
    var_12 = var_3.serialize(var_6)
    var_13 = var_7.values
    var_14 = bool(var_7.values == var_12)
    assert var_14 is True
    var_15 = var_7.errors
    assert var_15 is None
    var_16 = var_7._validate_called
    assert var_16 is False



# Parsed testcases at query #28
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = []
    var_3 = 'any_of'
    var_4 = {var_3: var_2}
    var_5 = module_1.Choice(**var_4)
    var_6 = var_1.template_for_field(var_5)
    assert var_6 == 'forms/select.html'



# Parsed testcases at query #29
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



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
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = 'John'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.values
    var_11 = bool(var_9.values == var_8)
    assert var_11 is True
    var_12 = var_9.errors
    assert var_12 is None
    var_13 = var_9._validate_called
    assert var_13 is False



# Parsed testcases at query #31
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Choice(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'allow_blank'
    var_3 = False
    var_4 = getattr(var_1, var_2, var_3)
    var_5 = bool(var_1.allow_null or var_4)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #34
#--------------------------




import typesystem.fields as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = []
    var_1 = 'any_of'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = module_1.Environment()
    var_5 = {}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = {}
    var_9 = module_3.Form(env=var_4, schema=var_7, values=var_8)
    var_10 = var_9.template_for_field(var_3)
    assert var_10 == 'forms/select.html'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_render_field_with_default_values. Retrieved 24/25 statements.
# Partially parsed test_render_field_with_custom_values. Retrieved 26/27 statements.
# Partially parsed test_render_field_with_password_input. Retrieved 26/27 statements.
# Partially parsed test_render_field_with_allow_null. Retrieved 25/26 statements.
# Partially parsed test_render_field_with_default_value. Retrieved 25/26 statements.
# Partially parsed test_render_field_with_choice_field. Retrieved 31/32 statements.
# Partially parsed test_render_field_with_boolean_field. Retrieved 24/25 statements.
# Partially parsed test_render_field_with_textarea. Retrieved 24/25 statements.
# Partially parsed test_render_field_with_read_only_field. Retrieved 24/25 statements.


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = var_7.render_field(field_name=var_1, field=var_8)
    var_10 = 'forms/input.html'
    var_11 = var_0.get_template(var_10)
    var_12 = 'field_id'
    var_13 = 'field_name'
    var_14 = 'field'
    var_15 = 'label'
    var_16 = 'required'
    var_17 = 'input_type'
    var_18 = 'value'
    var_19 = 'error'
    var_20 = 'test-field'
    var_21 = var_6.fields[var_1]
    var_22 = True
    var_23 = 'text'
    var_24 = None
    var_25 = {var_12: var_20, var_13: var_1, var_14: var_21, var_15: var_1, var_16: var_22, var_17: var_23, var_18: var_24, var_19: var_24}
    var_26 = [var_25]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = 'Custom Label'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = 'custom_value'
    var_12 = 'custom_error'
    var_13 = var_9.render_field(field_name=var_1, field=var_10, value=var_11, error=var_12)
    var_14 = 'forms/input.html'
    var_15 = var_0.get_template(var_14)
    var_16 = 'field_id'
    var_17 = 'field_name'
    var_18 = 'field'
    var_19 = 'label'
    var_20 = 'required'
    var_21 = 'input_type'
    var_22 = 'value'
    var_23 = 'error'
    var_24 = 'test-field'
    var_25 = var_8.fields[var_1]
    var_26 = True
    var_27 = 'text'
    var_28 = {var_16: var_24, var_17: var_1, var_18: var_25, var_19: var_2, var_20: var_26, var_21: var_27, var_22: var_11, var_23: var_12}
    var_29 = [var_28]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'password_field'
    var_2 = 'password'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = 'secret'
    var_11 = var_8.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = 'forms/input.html'
    var_13 = var_0.get_template(var_12)
    var_14 = 'field_id'
    var_15 = 'field_name'
    var_16 = 'field'
    var_17 = 'label'
    var_18 = 'required'
    var_19 = 'input_type'
    var_20 = 'value'
    var_21 = 'error'
    var_22 = 'password-field'
    var_23 = var_7.fields[var_1]
    var_24 = True
    var_25 = ''
    var_26 = None
    var_27 = {var_14: var_22, var_15: var_1, var_16: var_23, var_17: var_1, var_18: var_24, var_19: var_2, var_20: var_25, var_21: var_26}
    var_28 = [var_27]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'nullable_field'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = var_9.render_field(field_name=var_1, field=var_10)
    var_12 = 'forms/input.html'
    var_13 = var_0.get_template(var_12)
    var_14 = 'field_id'
    var_15 = 'field_name'
    var_16 = 'field'
    var_17 = 'label'
    var_18 = 'required'
    var_19 = 'input_type'
    var_20 = 'value'
    var_21 = 'error'
    var_22 = 'nullable-field'
    var_23 = var_8.fields[var_1]
    var_24 = False
    var_25 = 'text'
    var_26 = None
    var_27 = {var_14: var_22, var_15: var_1, var_16: var_23, var_17: var_1, var_18: var_24, var_19: var_25, var_20: var_26, var_21: var_26}
    var_28 = [var_27]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'default_field'
    var_2 = 'default_value'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = var_9.render_field(field_name=var_1, field=var_10)
    var_12 = 'forms/input.html'
    var_13 = var_0.get_template(var_12)
    var_14 = 'field_id'
    var_15 = 'field_name'
    var_16 = 'field'
    var_17 = 'label'
    var_18 = 'required'
    var_19 = 'input_type'
    var_20 = 'value'
    var_21 = 'error'
    var_22 = 'default-field'
    var_23 = var_8.fields[var_1]
    var_24 = False
    var_25 = 'text'
    var_26 = None
    var_27 = {var_14: var_22, var_15: var_1, var_16: var_23, var_17: var_1, var_18: var_24, var_19: var_25, var_20: var_26, var_21: var_26}
    var_28 = [var_27]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'choice_field'
    var_2 = 'a'
    var_3 = 'Option A'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 'Option B'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_1.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.Form(env=var_0, schema=var_13)
    var_15 = var_13.fields[var_1]
    var_16 = var_14.render_field(field_name=var_1, field=var_15)
    var_17 = 'forms/select.html'
    var_18 = var_0.get_template(var_17)
    var_19 = 'field_id'
    var_20 = 'field_name'
    var_21 = 'field'
    var_22 = 'label'
    var_23 = 'required'
    var_24 = 'input_type'
    var_25 = 'value'
    var_26 = 'error'
    var_27 = 'choice-field'
    var_28 = var_13.fields[var_1]
    var_29 = True
    var_30 = 'text'
    var_31 = None
    var_32 = {var_19: var_27, var_20: var_1, var_21: var_28, var_22: var_1, var_23: var_29, var_24: var_30, var_25: var_31, var_26: var_31}
    var_33 = [var_32]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'bool_field'
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_6.fields[var_1]
    var_9 = var_7.render_field(field_name=var_1, field=var_8)
    var_10 = 'forms/checkbox.html'
    var_11 = var_0.get_template(var_10)
    var_12 = 'field_id'
    var_13 = 'field_name'
    var_14 = 'field'
    var_15 = 'label'
    var_16 = 'required'
    var_17 = 'input_type'
    var_18 = 'value'
    var_19 = 'error'
    var_20 = 'bool-field'
    var_21 = var_6.fields[var_1]
    var_22 = True
    var_23 = 'text'
    var_24 = None
    var_25 = {var_12: var_20, var_13: var_1, var_14: var_21, var_15: var_1, var_16: var_22, var_17: var_23, var_18: var_24, var_19: var_24}
    var_26 = [var_25]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'text_field'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = var_7.fields[var_1]
    var_10 = var_8.render_field(field_name=var_1, field=var_9)
    var_11 = 'forms/textarea.html'
    var_12 = var_0.get_template(var_11)
    var_13 = 'field_id'
    var_14 = 'field_name'
    var_15 = 'field'
    var_16 = 'label'
    var_17 = 'required'
    var_18 = 'input_type'
    var_19 = 'value'
    var_20 = 'error'
    var_21 = 'text-field'
    var_22 = var_7.fields[var_1]
    var_23 = True
    var_24 = None
    var_25 = {var_13: var_21, var_14: var_1, var_15: var_22, var_16: var_1, var_17: var_23, var_18: var_2, var_19: var_24, var_20: var_24}
    var_26 = [var_25]

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'read_only_field'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = var_9.render_field(field_name=var_1, field=var_10)
    var_12 = 'forms/input.html'
    var_13 = var_0.get_template(var_12)
    var_14 = 'field_id'
    var_15 = 'field_name'
    var_16 = 'field'
    var_17 = 'label'
    var_18 = 'required'
    var_19 = 'input_type'
    var_20 = 'value'
    var_21 = 'error'
    var_22 = 'read-only-field'
    var_23 = var_8.fields[var_1]
    var_24 = 'text'
    var_25 = None
    var_26 = {var_14: var_22, var_15: var_1, var_16: var_23, var_17: var_1, var_18: var_2, var_19: var_24, var_20: var_25, var_21: var_25}
    var_27 = [var_26]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_render_field_password_value. Retrieved 19/25 statements.


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = 'password'
    var_5 = {}
    var_6 = module_1.String(format=var_4, **var_5)
    var_7 = 'test'
    var_8 = 'secret'
    var_9 = 'forms/input.html'
    var_10 = 'field_id'
    var_11 = 'field_name'
    var_12 = 'field'
    var_13 = 'label'
    var_14 = 'required'
    var_15 = 'input_type'
    var_16 = 'value'
    var_17 = 'error'
    var_18 = True
    var_19 = ''
    var_20 = None
    var_21 = {var_10: var_7, var_11: var_7, var_12: var_6, var_13: var_7, var_14: var_18, var_15: var_4, var_16: var_19, var_17: var_20}



# Parsed testcases at query #38
#--------------------------




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
    var_13 = bool(var_7.values == var_6)
    assert var_13 is True
    var_14 = var_7.errors
    assert var_14 is None
    var_15 = var_7._validate_called
    assert var_15 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_render_field_sets_value_to_empty_string_for_password_input_type. Retrieved 10/11 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3
import typesystem.fields as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = {}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = module_3.Form(env=var_2, schema=var_5)
    var_7 = module_4.Field()
    var_8 = 'test'
    var_9 = 'some_value'
    var_10 = var_6.render_field(field_name=var_8, field=var_7, value=var_9)
    assert var_10 == ''



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_form_init_with_keyword_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #41
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_1.String(**var_3)
    var_5 = {}
    var_6 = module_1.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = 'John'
    var_11 = 30
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = module_3.Form(env=var_0, schema=var_9, values=var_12)
    var_14 = var_13.env
    var_15 = bool(var_13.env == var_0)
    assert var_15 is True
    var_16 = var_13.schema
    var_17 = bool(var_13.schema == var_9)
    assert var_17 is True
    var_18 = var_13.values
    var_19 = bool(var_13.values == {'name': 'John', 'age': 30})
    assert var_19 is True
    var_20 = var_13.errors
    assert var_20 is None
    var_21 = var_13._validate_called
    assert var_21 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_1.String(**var_3)
    var_5 = {}
    var_6 = module_1.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.Form(env=var_0, schema=var_9)
    var_11 = var_10.env
    var_12 = bool(var_10.env == var_0)
    assert var_12 is True
    var_13 = var_10.schema
    var_14 = bool(var_10.schema == var_9)
    assert var_14 is True
    var_15 = var_10.values
    assert var_15 is None
    var_16 = var_10.errors
    assert var_16 is None
    var_17 = var_10._validate_called
    assert var_17 is False

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_1.String(**var_3)
    var_5 = {}
    var_6 = module_1.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = None
    var_11 = module_3.Form(env=var_0, schema=var_9, values=var_10)
    var_12 = var_11.env
    var_13 = bool(var_11.env == var_0)
    assert var_13 is True
    var_14 = var_11.schema
    var_15 = bool(var_11.schema == var_9)
    assert var_15 is True
    var_16 = var_11.values
    assert var_16 is None
    var_17 = var_11.errors
    assert var_17 is None
    var_18 = var_11._validate_called
    assert var_18 is False



# Parsed testcases at query #42
#--------------------------




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
    var_7 = 'test'
    var_8 = {var_1: var_7}
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_form_init_with_keyword_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_environment'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_form_init_with_keyword_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_form_init_with_kwargs_only. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #46
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
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = 'John'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.values
    var_11 = bool(var_9.values == var_8)
    assert var_11 is True

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
    var_6 = var_5.values
    assert var_6 is None



# Parsed testcases at query #47
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
    var_4 = {}
    var_5 = module_2.Form(env=var_0, schema=var_3, values=var_4)
    var_6 = {}
    var_7 = 'fields'
    var_8 = {var_7: var_6}
    var_9 = module_3.Object(**var_8)
    var_10 = var_5.template_for_field(var_9)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_jinja2forms_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #2
#--------------------------




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
    var_12 = var_3.serialize(var_6)
    var_13 = var_7.values
    var_14 = bool(var_7.values == var_12)
    assert var_14 is True
    var_15 = var_7.errors
    assert var_15 is None
    var_16 = var_7._validate_called
    assert var_16 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_form_html_method. Retrieved 8/10 statements.


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = var_7.__html__()
    var_9 = str(var_8)
    assert var_9 == ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_form_str_returns_render_fields_output. Retrieved 2/7 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_fields_with_valid_data. Retrieved 16/17 statements.
# Partially parsed test_render_fields_with_invalid_data. Retrieved 18/19 statements.
# Partially parsed test_render_fields_with_read_only_field. Retrieved 16/17 statements.
# Partially parsed test_render_fields_with_no_data. Retrieved 14/15 statements.
# Partially parsed test_render_fields_with_errors. Retrieved 18/19 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {}
    var_6 = module_2.String(**var_5)
    var_7 = {}
    var_8 = module_2.Integer(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {}
    var_11 = module_3.Schema(var_9, **var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = module_4.Form(env=var_2, schema=var_11, values=var_14)
    var_16 = {var_3: var_12, var_4: var_13}
    var_17 = var_15.validate(var_16)
    var_18 = var_15.render_fields()
    var_19 = 'name'
    var_20 = bool('name' in var_18)
    assert var_20 is True
    var_21 = 'age'
    var_22 = bool('age' in var_18)
    assert var_22 is True
    var_23 = 'John'
    var_24 = bool('John' in var_18)
    assert var_24 is True
    var_25 = '30'
    var_26 = bool('30' in var_18)
    assert var_26 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {}
    var_6 = module_2.String(**var_5)
    var_7 = {}
    var_8 = module_2.Integer(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {}
    var_11 = module_3.Schema(var_9, **var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = module_4.Form(env=var_2, schema=var_11, values=var_14)
    var_16 = ''
    var_17 = -5
    var_18 = {var_3: var_16, var_4: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = var_15.render_fields()
    var_21 = 'name'
    var_22 = bool('name' in var_20)
    assert var_22 is True
    var_23 = 'age'
    var_24 = bool('age' in var_20)
    assert var_24 is True
    var_25 = 'John'
    var_26 = bool('John' not in var_20)
    assert var_26 is True
    var_27 = '30'
    var_28 = bool('30' not in var_20)
    assert var_28 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'id'
    var_4 = 'name'
    var_5 = True
    var_6 = 'read_only'
    var_7 = {var_6: var_5}
    var_8 = module_2.Integer(**var_7)
    var_9 = {}
    var_10 = module_2.String(**var_9)
    var_11 = {var_3: var_8, var_4: var_10}
    var_12 = {}
    var_13 = module_3.Schema(var_11, **var_12)
    var_14 = 'John'
    var_15 = {var_3: var_5, var_4: var_14}
    var_16 = module_4.Form(env=var_2, schema=var_13, values=var_15)
    var_17 = {var_3: var_5, var_4: var_14}
    var_18 = var_16.validate(var_17)
    var_19 = var_16.render_fields()
    var_20 = 'id'
    var_21 = bool('id' not in var_19)
    assert var_21 is True
    var_22 = 'name'
    var_23 = bool('name' in var_19)
    assert var_23 is True
    var_24 = 'John'
    var_25 = bool('John' in var_19)
    assert var_25 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {}
    var_6 = module_2.String(**var_5)
    var_7 = {}
    var_8 = module_2.Integer(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {}
    var_11 = module_3.Schema(var_9, **var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = module_4.Form(env=var_2, schema=var_11, values=var_14)
    var_16 = var_15.render_fields()
    var_17 = 'name'
    var_18 = bool('name' in var_16)
    assert var_18 is True
    var_19 = 'age'
    var_20 = bool('age' in var_16)
    assert var_20 is True
    var_21 = 'John'
    var_22 = bool('John' in var_16)
    assert var_22 is True
    var_23 = '30'
    var_24 = bool('30' in var_16)
    assert var_24 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {}
    var_6 = module_2.String(**var_5)
    var_7 = {}
    var_8 = module_2.Integer(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {}
    var_11 = module_3.Schema(var_9, **var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = {var_3: var_12, var_4: var_13}
    var_15 = module_4.Form(env=var_2, schema=var_11, values=var_14)
    var_16 = ''
    var_17 = -5
    var_18 = {var_3: var_16, var_4: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = var_15.render_fields()
    var_21 = 'name'
    var_22 = bool('name' in var_20)
    assert var_22 is True
    var_23 = 'age'
    var_24 = bool('age' in var_20)
    assert var_24 is True
    var_25 = var_15.errors['name']
    var_26 = bool(var_15.errors['name'] in var_20)
    assert var_26 is True
    var_27 = var_15.errors['age']
    var_28 = bool(var_15.errors['age'] in var_20)
    assert var_28 is True



# Parsed testcases at query #6
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'field'
    var_2 = 'a'
    var_3 = 'A'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Choice(choices=var_5, **var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.Form(env=var_0, schema=var_10)
    var_12 = (var_2, var_3)
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Choice(choices=var_13, **var_14)
    var_16 = var_11.template_for_field(var_15)
    assert var_16 == 'forms/select.html'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'field'
    var_2 = {}
    var_3 = module_1.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Boolean(**var_8)
    var_10 = var_7.template_for_field(var_9)
    assert var_10 == 'forms/checkbox.html'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'field'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = var_8.template_for_field(var_10)
    assert var_11 == 'forms/textarea.html'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'field'
    var_2 = 'email'
    var_3 = {}
    var_4 = module_1.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = var_8.template_for_field(var_10)
    assert var_11 == 'forms/input.html'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'field'
    var_2 = {}
    var_3 = {}
    var_4 = module_1.Object(properties=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = {}
    var_10 = {}
    var_11 = module_1.Object(properties=var_9, **var_10)
    var_12 = var_8.template_for_field(var_11)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_field_with_regular_field. Retrieved 7/11 statements.
# Partially parsed test_render_field_with_checkbox_field. Retrieved 7/11 statements.
# Partially parsed test_render_field_with_select_field. Retrieved 16/20 statements.
# Partially parsed test_render_field_with_textarea_field. Retrieved 8/12 statements.
# Partially parsed test_render_field_with_password_field. Retrieved 7/11 statements.
# Partially parsed test_render_field_with_error. Retrieved 7/11 statements.
# Partially parsed test_render_field_with_none_value. Retrieved 6/10 statements.
# Partially parsed test_render_field_with_readonly_field. Retrieved 7/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = 'test'
    var_10 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'agree'
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.Boolean(**var_7)
    var_9 = True
    var_10 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'choice'
    var_2 = 'a'
    var_3 = 'A'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 'B'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_0.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = (var_2, var_3)
    var_15 = (var_5, var_6)
    var_16 = [var_14, var_15]
    var_17 = {}
    var_18 = module_0.Choice(choices=var_16, **var_17)
    var_19 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'description'
    var_2 = 'text'
    var_3 = {}
    var_4 = module_0.String(format=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {}
    var_9 = module_0.String(format=var_2, **var_8)
    var_10 = 'long text'
    var_11 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'password'
    var_2 = {}
    var_3 = module_0.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.String(format=var_1, **var_7)
    var_9 = 'secret'
    var_10 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = 'test'
    var_10 = 'Invalid value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'id'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'read_only'
    var_10 = {var_9: var_2}
    var_11 = module_0.Integer(**var_10)
    var_12 = None



# Parsed testcases at query #8
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_input_type_for_field_with_unknown_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_color_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_datetime_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_date_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_email_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_hidden_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_month_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_number_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_password_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_range_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_search_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_tel_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_text_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_time_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_url_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_week_format. Retrieved 4/5 statements.


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'color'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'datetime-local'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'date'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'email'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'hidden'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'month'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'number'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'password'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'range'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'search'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'tel'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'time'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'url'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'week'



# Parsed testcases at query #10
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_1.String(**var_3)
    var_5 = {}
    var_6 = module_1.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = 'John'
    var_11 = 30
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = module_3.Form(env=var_0, schema=var_9, values=var_12)
    var_14 = var_13.env
    var_15 = bool(var_13.env == var_0)
    assert var_15 is True
    var_16 = var_13.schema
    var_17 = bool(var_13.schema == var_9)
    assert var_17 is True
    var_18 = var_13.values
    var_19 = bool(var_13.values == {'name': 'John', 'age': 30})
    assert var_19 is True
    var_20 = var_13.errors
    assert var_20 is None
    var_21 = var_13._validate_called
    assert var_21 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_input_type_for_field_with_unknown_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_color_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_datetime_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_date_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_email_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_hidden_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_month_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_number_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_password_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_range_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_search_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_tel_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_text_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_time_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_url_format. Retrieved 4/5 statements.
# Partially parsed test_input_type_for_field_with_week_format. Retrieved 4/5 statements.


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'color'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'datetime-local'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'date'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'email'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'hidden'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'month'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'number'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'password'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'range'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'search'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'tel'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'time'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'url'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'week'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_form_init_with_keyword_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_required_false_when_field_has_default_and_allow_empty. Retrieved 9/12 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = True
    var_2 = module_0.Field(default=var_0, allow_null=var_1)
    var_3 = var_2.has_default()
    var_4 = var_2.allow_null
    var_5 = 'allow_blank'
    var_6 = False
    var_7 = getattr(var_2, var_5, var_6)
    var_8 = var_4 or var_7



# Parsed testcases at query #14
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input>'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = {}
    var_7 = module_2.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = 'test'
    var_14 = var_11.render_field(field_name=var_5, field=var_12, value=var_13)
    var_15 = '<input>'
    var_16 = bool('<input>' in var_14)
    assert var_16 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/select.html'
    var_1 = '<select>'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'choice'
    var_6 = 'a'
    var_7 = 'A'
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 'B'
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = {}
    var_14 = module_2.Choice(choices=var_12, **var_13)
    var_15 = {var_5: var_14}
    var_16 = {}
    var_17 = module_3.Schema(var_15, **var_16)
    var_18 = module_4.Form(env=var_4, schema=var_17)
    var_19 = var_17.fields[var_5]
    var_20 = var_18.render_field(field_name=var_5, field=var_19)
    var_21 = '<select>'
    var_22 = bool('<select>' in var_20)
    assert var_22 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/checkbox.html'
    var_1 = '<checkbox>'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'flag'
    var_6 = {}
    var_7 = module_2.Boolean(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = var_11.render_field(field_name=var_5, field=var_12)
    var_14 = '<checkbox>'
    var_15 = bool('<checkbox>' in var_13)
    assert var_15 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/textarea.html'
    var_1 = '<textarea>'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'text'
    var_6 = {}
    var_7 = module_2.String(format=var_5, **var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = var_11.render_field(field_name=var_5, field=var_12)
    var_14 = '<textarea>'
    var_15 = bool('<textarea>' in var_13)
    assert var_15 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'password'
    var_6 = {}
    var_7 = module_2.String(format=var_5, **var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = 'secret'
    var_14 = var_11.render_field(field_name=var_5, field=var_12, value=var_13)
    var_15 = "type='password'"
    var_16 = bool("type='password'" in var_14)
    assert var_16 is True
    var_17 = "value=''"
    var_18 = bool("value=''" in var_14)
    assert var_18 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input>{% if error %}{{ error }}{% endif %}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = {}
    var_7 = module_2.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = 'Invalid'
    var_14 = var_11.render_field(field_name=var_5, field=var_12, error=var_13)
    var_15 = 'Invalid'
    var_16 = bool('Invalid' in var_14)
    assert var_16 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input required='{{ required }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = {}
    var_7 = module_2.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = var_11.render_field(field_name=var_5, field=var_12)
    var_14 = "required='True'"
    var_15 = bool("required='True'" in var_13)
    assert var_15 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input required='{{ required }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = True
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_2.String(**var_8)
    var_10 = {var_5: var_9}
    var_11 = {}
    var_12 = module_3.Schema(var_10, **var_11)
    var_13 = module_4.Form(env=var_4, schema=var_12)
    var_14 = var_12.fields[var_5]
    var_15 = var_13.render_field(field_name=var_5, field=var_14)
    var_16 = "required='False'"
    var_17 = bool("required='False'" in var_15)
    assert var_17 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = {}
    var_7 = module_2.String(format=var_5, **var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = var_11.render_field(field_name=var_5, field=var_12)
    var_14 = "type='email'"
    var_15 = bool("type='email'" in var_13)
    assert var_15 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'default'
    var_7 = 'default'
    var_8 = {var_7: var_6}
    var_9 = module_2.String(**var_8)
    var_10 = {var_5: var_9}
    var_11 = {}
    var_12 = module_3.Schema(var_10, **var_11)
    var_13 = module_4.Form(env=var_4, schema=var_12)
    var_14 = var_12.fields[var_5]
    var_15 = var_13.render_field(field_name=var_5, field=var_14)
    var_16 = "value='default'"
    var_17 = bool("value='default'" in var_15)
    assert var_17 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 11/17 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.searchpath
    var_5 = bool(var_2.loader.searchpath == ['test_templates'])
    assert var_5 is True
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.package_name
    assert var_4 == 'test_package'
    var_5 = var_2.loader.package_path
    assert var_5 == 'templates'
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader
    var_5 = var_3.loader.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_3.loader.loaders[var_7]
    var_9 = 1
    var_10 = var_3.loader.loaders[var_9]
    var_11 = var_3.autoescape
    assert var_11 is True



# Parsed testcases at query #16
#--------------------------




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
    var_7 = 'test'
    var_8 = {var_1: var_7}
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_both. Retrieved 11/17 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.searchpath
    var_5 = bool(var_2.loader.searchpath == ['test_dir'])
    assert var_5 is True
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.package_name
    assert var_4 == 'test_package'
    var_5 = var_2.loader.package_path
    assert var_5 == 'templates'
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader
    var_5 = var_3.loader.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_3.loader.loaders[var_7]
    var_9 = 1
    var_10 = var_3.loader.loaders[var_9]
    var_11 = var_3.loader.loaders[0].searchpath
    var_12 = bool(var_3.loader.loaders[0].searchpath == ['test_dir'])
    assert var_12 is True
    var_13 = var_3.loader.loaders[1].package_name
    assert var_13 == 'test_package'
    var_14 = var_3.loader.loaders[1].package_path
    assert var_14 == 'templates'
    var_15 = var_3.autoescape
    assert var_15 is True



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
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.values
    var_11 = bool(var_9.values == var_8)
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_with_directory. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_package. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_both_directory_and_package. Retrieved 3/4 statements.
# Partially parsed test_constructor_without_directory_or_package. Retrieved 1/3 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = bool(var_1.env is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = bool(var_2.env is not None)
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------




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



# Parsed testcases at query #21
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader.loaders[0].searchpath
    var_4 = bool(var_2.env.loader.loaders[0].searchpath == ['test_dir'])
    assert var_4 is True
    var_5 = var_2.env.loader.loaders[1].package_name
    assert var_5 == 'test_pkg'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #23
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
    var_13 = bool(var_7.values == var_6)
    assert var_13 is True
    var_14 = var_7.errors
    assert var_14 is None
    var_15 = var_7._validate_called
    assert var_15 is False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 5/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader



# Parsed testcases at query #25
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
    var_13 = bool(var_7.values == var_6)
    assert var_13 is True
    var_14 = var_7.errors
    assert var_14 is None
    var_15 = var_7._validate_called
    assert var_15 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_form_init_without_keyword_arguments. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_form_init_with_keyword_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #28
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
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



# Parsed testcases at query #29
#--------------------------




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
    var_12 = var_3.serialize(var_6)
    var_13 = var_7.values
    var_14 = bool(var_7.values == var_12)
    assert var_14 is True
    var_15 = var_7.errors
    assert var_15 is None
    var_16 = var_7._validate_called
    assert var_16 is False



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
    var_4 = {}
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #32
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_2, package=var_2)



# Parsed testcases at query #33
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Form(env=var_0, schema=var_1, values=var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_form_init_with_keyword_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #36
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
    var_8 = var_3.serialize(var_6)
    var_9 = var_7.values
    var_10 = bool(var_7.values == var_8)
    assert var_10 is True

import jinja2.environment as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'not a schema'
    var_2 = module_1.Form(env=var_0, schema=var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 5/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader



# Parsed testcases at query #38
#--------------------------




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
    var_12 = var_3.serialize(var_6)
    var_13 = var_7.values
    var_14 = bool(var_7.values == var_12)
    assert var_14 is True
    var_15 = var_7.errors
    assert var_15 is None
    var_16 = var_7._validate_called
    assert var_16 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_form_init_without_keyword_arguments. Retrieved 3/5 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_jinja2forms_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_both_directory_and_package. Retrieved 5/9 statements.


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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_form_init_without_kwargs. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



# Parsed testcases at query #44
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
    var_1 = 'name'
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = 'test'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.values
    var_11 = bool(var_9.values == {'name': 'test'})
    assert var_11 is True
    var_12 = var_9.errors
    assert var_12 is None
    var_13 = var_9._validate_called
    assert var_13 is False



# Parsed testcases at query #45
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #47
#--------------------------




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
    var_12 = var_3.serialize(var_6)
    var_13 = var_7.values
    var_14 = bool(var_7.values == var_12)
    assert var_14 is True
    var_15 = var_7.errors
    assert var_15 is None
    var_16 = var_7._validate_called
    assert var_16 is False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_form_init_without_keyword_arguments. Retrieved 3/5 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_form_init_with_keyword_only_arguments. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)
    var_6 = var_5.env
    var_7 = bool(var_5.env == var_0)
    assert var_7 is True
    var_8 = var_5.schema
    var_9 = bool(var_5.schema == var_1)
    assert var_9 is True
    var_10 = var_5.values
    var_11 = var_5.errors
    assert var_11 is None
    var_12 = var_5._validate_called
    assert var_12 is False



