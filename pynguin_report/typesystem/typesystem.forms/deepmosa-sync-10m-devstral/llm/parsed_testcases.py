####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_10 = 'valid'
    var_11 = {var_1: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = bool(var_9.is_valid)
    assert var_13 is True
    var_14 = var_9.validated_data
    var_15 = bool(var_9.validated_data == {'name': 'valid'})
    assert var_15 is True
    var_16 = bool(var_9._validate_called)
    assert var_16 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 5
    var_3 = {}
    var_4 = module_1.String(min_length=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = 'test'
    var_9 = {var_1: var_8}
    var_10 = module_3.Form(env=var_0, schema=var_7, values=var_9)
    var_11 = 'bad'
    var_12 = {var_1: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = bool(not var_10.is_valid)
    assert var_14 is True
    var_15 = var_10.errors
    var_16 = bool(var_10.errors is not None)
    assert var_16 is True
    var_17 = bool(var_10._validate_called)
    assert var_17 is True

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
    var_10 = 'valid'
    var_11 = {var_1: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = 'name'
    var_14 = 'valid'
    var_15 = {var_13: var_14}
    var_16 = var_9.validate(var_15)
    var_17 = bool(False)
    assert var_17 is True

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
    var_10 = var_9.validate()
    var_11 = bool(var_9.is_valid)
    assert var_11 is True
    var_12 = var_9.validated_data
    var_13 = bool(var_9.validated_data == {'name': 'test'})
    assert var_13 is True
    var_14 = bool(var_9._validate_called)
    assert var_14 is True



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
    var_11 = bool(not var_4._validate_called)
    assert var_11 is True

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



# Parsed testcases at query #3
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
    var_3 = module_1.Integer(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = None
    var_3 = module_1.Form(env=var_2, schema=var_2)
    var_4 = var_3.template_for_field(var_1)
    assert var_4 == 'forms/select.html'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_both_directory_and_package. Retrieved 11/19 statements.


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
    var_5 = var_2.env.loader.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_2.env.loader.loaders[var_7]
    var_9 = 1
    var_10 = var_2.env.loader.loaders[var_9]

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{{ field_id }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = {}
    var_7 = module_2.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = 'test'
    var_12 = {var_5: var_11}
    var_13 = module_4.Form(env=var_4, schema=var_10, values=var_12)
    var_14 = {var_5: var_11}
    var_15 = var_13.validate(var_14)
    var_16 = var_10.fields[var_5]
    var_17 = None
    var_18 = var_13.render_field(field_name=var_5, field=var_16, value=var_11, error=var_17)
    assert var_18 == 'name'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/select.html'
    var_1 = '{{ field_id }}'
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
    var_18 = {var_5: var_6}
    var_19 = module_4.Form(env=var_4, schema=var_17, values=var_18)
    var_20 = {var_5: var_6}
    var_21 = var_19.validate(var_20)
    var_22 = var_17.fields[var_5]
    var_23 = None
    var_24 = var_19.render_field(field_name=var_5, field=var_22, value=var_6, error=var_23)
    assert var_24 == 'choice'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/checkbox.html'
    var_1 = '{{ field_id }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'flag'
    var_6 = {}
    var_7 = module_2.Boolean(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = True
    var_12 = {var_5: var_11}
    var_13 = module_4.Form(env=var_4, schema=var_10, values=var_12)
    var_14 = {var_5: var_11}
    var_15 = var_13.validate(var_14)
    var_16 = var_10.fields[var_5]
    var_17 = None
    var_18 = var_13.render_field(field_name=var_5, field=var_16, value=var_11, error=var_17)
    assert var_18 == 'flag'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/textarea.html'
    var_1 = '{{ field_id }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'description'
    var_6 = 'text'
    var_7 = {}
    var_8 = module_2.String(format=var_6, **var_7)
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = module_3.Schema(var_9, **var_10)
    var_12 = 'test'
    var_13 = {var_5: var_12}
    var_14 = module_4.Form(env=var_4, schema=var_11, values=var_13)
    var_15 = {var_5: var_12}
    var_16 = var_14.validate(var_15)
    var_17 = var_11.fields[var_5]
    var_18 = None
    var_19 = var_14.render_field(field_name=var_5, field=var_17, value=var_12, error=var_18)
    assert var_19 == 'description'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{{ error }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = {}
    var_7 = module_2.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = 'test'
    var_12 = {var_5: var_11}
    var_13 = module_4.Form(env=var_4, schema=var_10, values=var_12)
    var_14 = ''
    var_15 = {var_5: var_14}
    var_16 = var_13.validate(var_15)
    var_17 = var_10.fields[var_5]
    var_18 = 'This field is required.'
    var_19 = var_13.render_field(field_name=var_5, field=var_17, value=var_14, error=var_18)
    assert var_19 == 'This field is required.'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{{ value }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'password'
    var_6 = {}
    var_7 = module_2.String(format=var_5, **var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = 'secret'
    var_12 = {var_5: var_11}
    var_13 = module_4.Form(env=var_4, schema=var_10, values=var_12)
    var_14 = {var_5: var_11}
    var_15 = var_13.validate(var_14)
    var_16 = var_10.fields[var_5]
    var_17 = None
    var_18 = var_13.render_field(field_name=var_5, field=var_16, value=var_11, error=var_17)
    assert var_18 == ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{{ input_type }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'color'
    var_6 = {}
    var_7 = module_2.String(format=var_5, **var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = '#ff0000'
    var_12 = {var_5: var_11}
    var_13 = module_4.Form(env=var_4, schema=var_10, values=var_12)
    var_14 = {var_5: var_11}
    var_15 = var_13.validate(var_14)
    var_16 = var_10.fields[var_5]
    var_17 = None
    var_18 = var_13.render_field(field_name=var_5, field=var_16, value=var_11, error=var_17)
    assert var_18 == 'color'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '{{ input_type }}'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'unknown'
    var_6 = {}
    var_7 = module_2.String(format=var_5, **var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = 'test'
    var_12 = {var_5: var_11}
    var_13 = module_4.Form(env=var_4, schema=var_10, values=var_12)
    var_14 = {var_5: var_11}
    var_15 = var_13.validate(var_14)
    var_16 = var_10.fields[var_5]
    var_17 = None
    var_18 = var_13.render_field(field_name=var_5, field=var_16, value=var_11, error=var_17)
    assert var_18 == 'text'



# Parsed testcases at query #7
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = 'Test Label'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_8.fields[var_1]
    var_11 = var_9.render_field(field_name=var_1, field=var_10)
    var_12 = 'Test Label'
    var_13 = bool('Test Label' in var_11)
    assert var_13 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_allow_empty_with_allow_blank. Retrieved 6/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'allow_blank'
    var_3 = False
    var_4 = getattr(var_1, var_2, var_3)
    var_5 = var_4 == var_0
    var_6 = bool(var_1.allow_null or var_5)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_blank'
    var_2 = False
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = True
    var_5 = var_3 == var_4
    var_6 = bool(var_0.allow_null or var_5)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_blank'
    var_2 = False
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = var_3 == var_2
    var_5 = bool(var_0.allow_null or var_4)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_directory_and_package. Retrieved 11/17 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.searchpath
    var_5 = bool(var_2.loader.searchpath == ['/path/to/templates'])
    assert var_5 is True
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'package.name'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.package_name
    assert var_4 == 'package.name'
    var_5 = var_2.loader.package_path
    assert var_5 == 'templates'
    var_6 = var_2.autoescape
    assert var_6 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = 'package.name'
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



# Parsed testcases at query #10
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #11
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test'
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_2.Schema(var_4, **var_5)
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = {}
    var_9 = module_1.Object(**var_8)
    var_10 = var_7.template_for_field(var_9)



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 11/17 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = var_2.loader.searchpath
    var_5 = bool(var_2.loader.searchpath == ['test_templates'])
    assert var_5 is True

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
    var_11 = var_3.loader.loaders[0].searchpath
    var_12 = bool(var_3.loader.loaders[0].searchpath == ['test_templates'])
    assert var_12 is True
    var_13 = var_3.loader.loaders[1].package_name
    assert var_13 == 'test_package'
    var_14 = var_3.loader.loaders[1].package_path
    assert var_14 == 'templates'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_required_predicate_false. Retrieved 8/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = var_1.has_default()
    var_3 = var_1.allow_null
    var_4 = 'allow_blank'
    var_5 = False
    var_6 = getattr(var_1, var_4, var_5)
    var_7 = var_3 or var_6



# Parsed testcases at query #15
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
    var_5 = var_4.render_fields()
    assert var_5 == ''

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
    var_8 = 'invalid'
    var_9 = {var_1: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = var_7.render_fields()
    var_12 = var_6.fields[var_1]
    var_13 = var_7.errors[var_1]
    var_14 = var_7.render_field(field_name=var_1, field=var_12, value=var_8, error=var_13)
    var_15 = bool(var_11 == var_14)
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
    var_7 = 'valid'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.render_fields()
    var_11 = var_6.fields[var_1]
    var_12 = None
    var_13 = var_9.render_field(field_name=var_1, field=var_11, value=var_7, error=var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_9.render_fields()
    assert var_10 == ''

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
    var_11 = var_10.render_fields()
    var_12 = var_9.fields[var_1]
    var_13 = None
    var_14 = var_10.render_field(field_name=var_1, field=var_12, value=var_13, error=var_13)
    var_15 = var_9.fields[var_2]
    var_16 = var_10.render_field(field_name=var_2, field=var_15, value=var_13, error=var_13)
    var_17 = var_14 + var_16
    var_18 = bool(var_11 == var_17)
    assert var_18 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'package.name'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = 'package.name'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #17
#--------------------------




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
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = 'test_value'
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
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
    var_13 = None
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True

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
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = 'test_value'
    var_11 = 'Error message'
    var_12 = var_7.render_field(field_name=var_1, field=var_9, value=var_10, error=var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

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
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'secret'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'choice_field'
    var_2 = '1'
    var_3 = 'Option 1'
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
    var_16 = var_11.render_field(field_name=var_1, field=var_15)
    var_17 = bool(var_16 is not None)
    assert var_17 is True

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
    var_8 = {}
    var_9 = module_1.Boolean(**var_8)
    var_10 = True
    var_11 = var_7.render_field(field_name=var_1, field=var_9, value=var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True

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
    var_9 = {}
    var_10 = module_1.String(format=var_2, **var_9)
    var_11 = 'Long text'
    var_12 = var_8.render_field(field_name=var_1, field=var_10, value=var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

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
    var_10 = 'read_only'
    var_11 = {var_10: var_2}
    var_12 = module_1.String(**var_11)
    var_13 = 'read_only'
    var_14 = var_9.render_field(field_name=var_1, field=var_12, value=var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_1.env.loader is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'package.name'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_1.env.loader is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = 'package.name'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = bool(var_2.env.loader is not None)
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #21
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #22
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #23
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 5/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader



# Parsed testcases at query #27
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_form_html_method. Retrieved 11/13 statements.


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
    var_7 = 'value'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.__html__()
    var_11 = str(var_10)
    var_12 = var_9.render_fields()
    var_13 = bool(var_11 == var_12)
    assert var_13 is True



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
    var_5 = var_4.render_fields()
    assert var_5 == ''

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
    var_8 = 'invalid'
    var_9 = {var_1: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = var_7.render_fields()
    var_12 = var_6.fields[var_1]
    var_13 = var_7.render_field(field_name=var_1, field=var_12, value=var_8, error=var_8)
    var_14 = bool(var_11 == var_13)
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
    var_7 = 'valid'
    var_8 = {var_1: var_7}
    var_9 = module_3.Form(env=var_0, schema=var_6, values=var_8)
    var_10 = var_9.render_fields()
    var_11 = var_6.fields[var_1]
    var_12 = None
    var_13 = var_9.render_field(field_name=var_1, field=var_11, value=var_7, error=var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_1.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_2.Schema(var_6, **var_7)
    var_9 = module_3.Form(env=var_0, schema=var_8)
    var_10 = var_9.render_fields()
    assert var_10 == ''

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
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_1: var_11, var_2: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = var_9.fields[var_1]
    var_16 = None
    var_17 = var_10.render_field(field_name=var_1, field=var_15, value=var_11, error=var_16)
    var_18 = var_9.fields[var_2]
    var_19 = var_10.render_field(field_name=var_2, field=var_18, value=var_12, error=var_16)
    var_20 = var_17 + var_19
    var_21 = var_10.render_fields()
    var_22 = bool(var_21 == var_20)
    assert var_22 is True



# Parsed testcases at query #3
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
    var_7 = module_3.Form(env=var_0, schema=var_6)
    var_8 = 'test'
    var_9 = {var_1: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = bool(var_7.is_valid)
    assert var_11 is True
    var_12 = var_7.validated_data
    var_13 = bool(var_7.validated_data == {'name': 'test'})
    assert var_13 is True
    var_14 = bool(var_7._validate_called)
    assert var_14 is True

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 5
    var_3 = {}
    var_4 = module_1.String(min_length=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_2.Schema(var_5, **var_6)
    var_8 = module_3.Form(env=var_0, schema=var_7)
    var_9 = 'test'
    var_10 = {var_1: var_9}
    var_11 = var_8.validate(var_10)
    var_12 = bool(not var_8.is_valid)
    assert var_12 is True
    var_13 = var_8.errors
    var_14 = bool(var_8.errors is not None)
    assert var_14 is True
    var_15 = bool(var_8._validate_called)
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
    var_8 = 'test'
    var_9 = {var_1: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = 'name'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = var_7.validate(var_13)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_field_with_string_field. Retrieved 6/10 statements.
# Partially parsed test_render_field_with_password_field. Retrieved 6/10 statements.
# Partially parsed test_render_field_with_email_field. Retrieved 6/10 statements.
# Partially parsed test_render_field_with_boolean_field. Retrieved 6/12 statements.
# Partially parsed test_render_field_with_choice_field. Retrieved 12/18 statements.
# Partially parsed test_render_field_with_error. Retrieved 7/11 statements.
# Partially parsed test_render_field_with_required_field. Retrieved 6/10 statements.
# Partially parsed test_render_field_with_optional_field. Retrieved 7/11 statements.
# Partially parsed test_render_field_with_textarea. Retrieved 7/13 statements.
# Partially parsed test_render_field_with_custom_title. Retrieved 7/11 statements.


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
    var_7 = var_6.fields[var_1]
    var_8 = 'John Doe'
    var_9 = '<input'
    var_10 = 'name="name"'
    var_11 = 'type="text"'
    var_12 = 'value="John Doe"'

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
    var_7 = var_6.fields[var_1]
    var_8 = 'secret'
    var_9 = 'type="password"'
    var_10 = 'value=""'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(format=var_1, **var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = var_6.fields[var_1]
    var_8 = 'user@example.com'
    var_9 = 'type="email"'
    var_10 = 'value="user@example.com"'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'active'
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = var_6.fields[var_1]
    var_8 = True
    var_9 = 'forms/checkbox.html'
    var_10 = 'name="active"'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'status'
    var_2 = 'active'
    var_3 = 'Active'
    var_4 = (var_2, var_3)
    var_5 = 'inactive'
    var_6 = 'Inactive'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_0.Choice(choices=var_8, **var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = var_13.fields[var_1]
    var_15 = 'forms/select.html'
    var_16 = 'name="status"'
    var_17 = '<option value="active"'

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
    var_7 = var_6.fields[var_1]
    var_8 = ''
    var_9 = 'This field is required'
    var_10 = 'This field is required'

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
    var_7 = var_6.fields[var_1]
    var_8 = ''
    var_9 = 'required'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'name'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = var_8.fields[var_1]
    var_10 = ''
    var_11 = 'required'

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
    var_8 = var_7.fields[var_1]
    var_9 = 'Long text'
    var_10 = 'forms/textarea.html'
    var_11 = 'name="description"'
    var_12 = 'Long text'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = []
    var_1 = 'full_name'
    var_2 = 'Full Name'
    var_3 = 'title'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = var_8.fields[var_1]
    var_10 = 'John Doe'
    var_11 = 'Full Name'



# Parsed testcases at query #5
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

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Integer(**var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/input.html'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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

# Partially parsed test_template_for_field_with_other_field. Retrieved 2/4 statements.


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

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = []



# Parsed testcases at query #10
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
    var_7 = module_3.Boolean(**var_6)
    var_8 = var_5.template_for_field(var_7)
    assert var_8 == 'forms/checkbox.html'



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_template_env_with_directory. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_package. Retrieved 4/6 statements.
# Partially parsed test_load_template_env_with_both. Retrieved 11/17 statements.


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



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = module_1.Environment()
    var_3 = {}
    var_4 = {}
    var_5 = module_2.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = module_3.Form(env=var_2, schema=var_5, values=var_6)
    var_8 = var_1.allow_null
    var_9 = 'allow_blank'
    var_10 = False
    var_11 = getattr(var_1, var_9, var_10)
    var_12 = var_8 or var_11
    assert var_12 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_render_field_with_number_field. Retrieved 7/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"
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
    var_15 = "<input type='text' name='name' value='test'>"
    var_16 = bool("<input type='text' name='name' value='test'>" in var_14)
    assert var_16 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'age'
    var_6 = []
    var_7 = 25
    var_8 = "<input type='number' name='age' value='25'>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/checkbox.html'
    var_1 = "<input type='checkbox' name='{{ field_name }}' {{ 'checked' if value else '' }}>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'active'
    var_6 = {}
    var_7 = module_2.Boolean(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = True
    var_14 = var_11.render_field(field_name=var_5, field=var_12, value=var_13)
    var_15 = "<input type='checkbox' name='active' checked>"
    var_16 = bool("<input type='checkbox' name='active' checked>" in var_14)
    assert var_16 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/select.html'
    var_1 = "<select name='{{ field_name }}'><option value='{{ value }}'>{{ value }}</option></select>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'status'
    var_6 = 'active'
    var_7 = 'inactive'
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_2.Choice(choices=var_8, **var_9)
    var_11 = {var_5: var_10}
    var_12 = {}
    var_13 = module_3.Schema(var_11, **var_12)
    var_14 = module_4.Form(env=var_4, schema=var_13)
    var_15 = var_13.fields[var_5]
    var_16 = var_14.render_field(field_name=var_5, field=var_15, value=var_6)
    var_17 = "<select name='status'><option value='active'>active</option></select>"
    var_18 = bool("<select name='status'><option value='active'>active</option></select>" in var_16)
    assert var_18 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/textarea.html'
    var_1 = "<textarea name='{{ field_name }}'>{{ value }}</textarea>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'description'
    var_6 = 'text'
    var_7 = {}
    var_8 = module_2.String(format=var_6, **var_7)
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = module_3.Schema(var_9, **var_10)
    var_12 = module_4.Form(env=var_4, schema=var_11)
    var_13 = var_11.fields[var_5]
    var_14 = 'long text'
    var_15 = var_12.render_field(field_name=var_5, field=var_13, value=var_14)
    var_16 = "<textarea name='description'>long text</textarea>"
    var_17 = bool("<textarea name='description'>long text</textarea>" in var_15)
    assert var_17 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'><div class='error'>{{ error }}</div>"
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
    var_14 = 'Invalid name'
    var_15 = var_11.render_field(field_name=var_5, field=var_12, value=var_13, error=var_14)
    var_16 = "<div class='error'>Invalid name</div>"
    var_17 = bool("<div class='error'>Invalid name</div>" in var_15)
    assert var_17 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}'>"
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
    var_15 = "<input type='password' name='password' value=''>"
    var_16 = bool("<input type='password' name='password' value=''>" in var_14)
    assert var_16 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' required='{{ required }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = {}
    var_7 = module_2.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_3.Schema(var_8, **var_9)
    var_11 = module_4.Form(env=var_4, schema=var_10)
    var_12 = var_10.fields[var_5]
    var_13 = 'test@example.com'
    var_14 = var_11.render_field(field_name=var_5, field=var_12, value=var_13)
    var_15 = "<input type='text' name='email' value='test@example.com' required='True'>"
    var_16 = bool("<input type='text' name='email' value='test@example.com' required='True'>" in var_14)
    assert var_16 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' required='{{ required }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = ''
    var_7 = 'default'
    var_8 = {var_7: var_6}
    var_9 = module_2.String(**var_8)
    var_10 = {var_5: var_9}
    var_11 = {}
    var_12 = module_3.Schema(var_10, **var_11)
    var_13 = module_4.Form(env=var_4, schema=var_12)
    var_14 = var_12.fields[var_5]
    var_15 = 'test@example.com'
    var_16 = var_13.render_field(field_name=var_5, field=var_14, value=var_15)
    var_17 = "<input type='text' name='email' value='test@example.com' required='False'>"
    var_18 = bool("<input type='text' name='email' value='test@example.com' required='False'>" in var_16)
    assert var_18 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input type='{{ input_type }}' name='{{ field_name }}' value='{{ value }}' required='{{ required }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = True
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_2.String(**var_8)
    var_10 = {var_5: var_9}
    var_11 = {}
    var_12 = module_3.Schema(var_10, **var_11)
    var_13 = module_4.Form(env=var_4, schema=var_12)
    var_14 = var_12.fields[var_5]
    var_15 = 'test@example.com'
    var_16 = var_13.render_field(field_name=var_5, field=var_14, value=var_15)
    var_17 = "<input type='text' name='email' value='test@example.com' required='False'>"
    var_18 = bool("<input type='text' name='email' value='test@example.com' required='False'>" in var_16)
    assert var_18 is True



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some/directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some/directory'
    var_1 = 'some.package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 5/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader



# Parsed testcases at query #18
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #19
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_2 is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_2 is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



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



# Parsed testcases at query #24
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



# Parsed testcases at query #25
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #26
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(**var_2)
    var_4 = var_1.template_for_field(var_3)



# Parsed testcases at query #27
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 5/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader



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
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #32
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = bool(var_2.env.loader is not None)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_template_env_with_both_directory_and_package. Retrieved 5/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.load_template_env(directory=var_0, package=var_1)
    var_4 = var_3.loader



# Parsed testcases at query #34
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_1.env.loader is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = bool(var_1.env.loader is not None)
    assert var_3 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader
    var_4 = bool(var_2.env.loader is not None)
    assert var_4 is True

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = bool(False)
    assert var_1 is True



