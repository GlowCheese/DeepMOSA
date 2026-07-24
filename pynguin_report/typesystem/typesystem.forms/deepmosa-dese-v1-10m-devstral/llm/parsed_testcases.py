####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'username'
    var_3 = module_2.String()
    var_4 = {var_2: var_3}
    var_5 = module_3.Schema(var_4)
    var_6 = 'test'
    var_7 = {var_2: var_6}
    var_8 = module_4.Form(env=var_1, schema=var_5, values=var_7)
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    var_11 = var_5.fields[var_2]
    var_12 = None
    var_13 = var_8.render_field(field_name=var_2, field=var_11, value=var_6, error=var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'password'
    var_3 = module_2.String(format=var_2)
    var_4 = {var_2: var_3}
    var_5 = module_3.Schema(var_4)
    var_6 = 'secret'
    var_7 = {var_2: var_6}
    var_8 = module_4.Form(env=var_1, schema=var_5, values=var_7)
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    var_11 = var_5.fields[var_2]
    var_12 = None
    var_13 = var_8.render_field(field_name=var_2, field=var_11, value=var_6, error=var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'email'
    var_3 = module_2.String(format=var_2)
    var_4 = {var_2: var_3}
    var_5 = module_3.Schema(var_4)
    var_6 = 'invalid'
    var_7 = {var_2: var_6}
    var_8 = module_4.Form(env=var_1, schema=var_5, values=var_7)
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    var_11 = var_5.fields[var_2]
    var_12 = 'Enter a valid email address.'
    var_13 = var_8.render_field(field_name=var_2, field=var_11, value=var_6, error=var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'country'
    var_3 = 'US'
    var_4 = 'United States'
    var_5 = (var_3, var_4)
    var_6 = 'UK'
    var_7 = 'United Kingdom'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = module_2.Choice(choices=var_9)
    var_11 = {var_2: var_10}
    var_12 = module_3.Schema(var_11)
    var_13 = {var_2: var_3}
    var_14 = module_4.Form(env=var_1, schema=var_12, values=var_13)
    var_15 = {var_2: var_3}
    var_16 = var_14.validate(var_15)
    var_17 = var_12.fields[var_2]
    var_18 = None
    var_19 = var_14.render_field(field_name=var_2, field=var_17, value=var_3, error=var_18)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'agree'
    var_3 = module_2.Boolean()
    var_4 = {var_2: var_3}
    var_5 = module_3.Schema(var_4)
    var_6 = True
    var_7 = {var_2: var_6}
    var_8 = module_4.Form(env=var_1, schema=var_5, values=var_7)
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    var_11 = var_5.fields[var_2]
    var_12 = None
    var_13 = var_8.render_field(field_name=var_2, field=var_11, value=var_6, error=var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'description'
    var_3 = 'text'
    var_4 = module_2.String(format=var_3)
    var_5 = {var_2: var_4}
    var_6 = module_3.Schema(var_5)
    var_7 = 'Long text'
    var_8 = {var_2: var_7}
    var_9 = module_4.Form(env=var_1, schema=var_6, values=var_8)
    var_10 = {var_2: var_7}
    var_11 = var_9.validate(var_10)
    var_12 = var_6.fields[var_2]
    var_13 = None
    var_14 = var_9.render_field(field_name=var_2, field=var_12, value=var_7, error=var_13)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'required_field'
    var_3 = module_2.String()
    var_4 = {var_2: var_3}
    var_5 = module_3.Schema(var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = module_4.Form(env=var_1, schema=var_5, values=var_7)
    var_9 = {var_2: var_6}
    var_10 = var_8.validate(var_9)
    var_11 = var_5.fields[var_2]
    var_12 = None
    var_13 = var_8.render_field(field_name=var_2, field=var_11, value=var_6, error=var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'optional_field'
    var_3 = True
    var_4 = module_2.String()
    var_5 = {var_2: var_4}
    var_6 = module_3.Schema(var_5)
    var_7 = 'value'
    var_8 = {var_2: var_7}
    var_9 = module_4.Form(env=var_1, schema=var_6, values=var_8)
    var_10 = {var_2: var_7}
    var_11 = var_9.validate(var_10)
    var_12 = var_6.fields[var_2]
    var_13 = None
    var_14 = var_9.render_field(field_name=var_2, field=var_12, value=var_7, error=var_13)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'custom'
    var_3 = 'color'
    var_4 = module_2.String(format=var_3)
    var_5 = {var_2: var_4}
    var_6 = module_3.Schema(var_5)
    var_7 = '#ff0000'
    var_8 = {var_2: var_7}
    var_9 = module_4.Form(env=var_1, schema=var_6, values=var_8)
    var_10 = {var_2: var_7}
    var_11 = var_9.validate(var_10)
    var_12 = var_6.fields[var_2]
    var_13 = None
    var_14 = var_9.render_field(field_name=var_2, field=var_12, value=var_7, error=var_13)



# Parsed testcases at query #2
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = []
    var_3 = module_1.Choice()
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Boolean()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = module_1.String(format=var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = module_1.String(format=var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = {}
    var_3 = module_1.Object(properties=var_2)
    var_4 = var_1.template_for_field(var_3)

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Integer()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/input.html'



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
    var_6 = module_2.String()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = 'John'
    var_10 = {var_5: var_9}
    var_11 = module_4.Form(env=var_4, schema=var_8, values=var_10)
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    var_14 = var_11.render_fields()
    assert var_14 == '<input name="name" value="John" />'

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
    var_6 = module_2.String()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = 'John'
    var_10 = {var_5: var_9}
    var_11 = module_4.Form(env=var_4, schema=var_8, values=var_10)
    var_12 = ''
    var_13 = {var_5: var_12}
    var_14 = var_11.validate(var_13)
    var_15 = var_11.render_fields()
    assert var_15 == '<input name="name" value="" />'

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
    var_6 = True
    var_7 = module_2.String()
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = 'John'
    var_11 = {var_5: var_10}
    var_12 = module_4.Form(env=var_4, schema=var_9, values=var_11)
    var_13 = {var_5: var_10}
    var_14 = var_12.validate(var_13)
    var_15 = var_12.render_fields()
    assert var_15 == ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/select.html'
    var_1 = '<select />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'color'
    var_6 = 'red'
    var_7 = 'green'
    var_8 = 'blue'
    var_9 = [var_6, var_7, var_8]
    var_10 = module_2.Choice(choices=var_9)
    var_11 = {var_5: var_10}
    var_12 = module_3.Schema(var_11)
    var_13 = {var_5: var_6}
    var_14 = module_4.Form(env=var_4, schema=var_12, values=var_13)
    var_15 = {var_5: var_6}
    var_16 = var_14.validate(var_15)
    var_17 = var_14.render_fields()
    assert var_17 == '<select name="color" value="red" />'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/checkbox.html'
    var_1 = '<checkbox />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'active'
    var_6 = module_2.Boolean()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = True
    var_10 = {var_5: var_9}
    var_11 = module_4.Form(env=var_4, schema=var_8, values=var_10)
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    var_14 = var_11.render_fields()
    assert var_14 == '<checkbox name="active" checked />'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/textarea.html'
    var_1 = '<textarea />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'description'
    var_6 = 'text'
    var_7 = module_2.String(format=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = 'A long text'
    var_11 = {var_5: var_10}
    var_12 = module_4.Form(env=var_4, schema=var_9, values=var_11)
    var_13 = {var_5: var_10}
    var_14 = var_12.validate(var_13)
    var_15 = var_12.render_fields()
    assert var_15 == '<textarea name="description">A long text</textarea>'

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
    var_5 = 'password'
    var_6 = module_2.String(format=var_5)
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = 'secret'
    var_10 = {var_5: var_9}
    var_11 = module_4.Form(env=var_4, schema=var_8, values=var_10)
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    var_14 = var_11.render_fields()
    assert var_14 == '<input type="password" name="password" value="" />'

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
    var_5 = 'email'
    var_6 = module_2.String(format=var_5)
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = 'test@example.com'
    var_10 = {var_5: var_9}
    var_11 = module_4.Form(env=var_4, schema=var_8, values=var_10)
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    var_14 = var_11.render_fields()
    assert var_14 == '<input type="email" name="email" value="test@example.com" />'

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
    var_7 = module_2.String()
    var_8 = module_2.Integer()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_3.Schema(var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = module_4.Form(env=var_4, schema=var_10, values=var_13)
    var_15 = {var_5: var_11, var_6: var_12}
    var_16 = var_14.validate(var_15)
    var_17 = var_14.render_fields()
    assert var_17 == '<input name="name" value="John" /><input name="age" value="30" />'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_with_directory. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_package. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_both_directory_and_package. Retrieved 4/6 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = 'my_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.environment as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = 'normal_field'
    var_2 = True
    var_3 = module_0.Field(read_only=var_2)
    var_4 = module_0.Field()
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.Environment()
    var_8 = {}
    var_9 = module_3.Form(env=var_7, schema=var_6, values=var_8)
    var_10 = 'value'
    var_11 = {var_1: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = var_9.render_fields()



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = True
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1, allow_null=var_0)
    var_3 = module_1.Environment()
    var_4 = {}
    var_5 = module_2.Schema(var_4)
    var_6 = {}
    var_7 = module_3.Form(env=var_3, schema=var_5, values=var_6)
    var_8 = 'test_field'
    var_9 = var_7.render_field(field_name=var_8, field=var_2)



# Parsed testcases at query #8
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

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = 'my.package'
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



# Parsed testcases at query #9
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
    var_7 = var_2.serialize(var_5)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = None
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_form_init_with_positional_args. Retrieved 3/4 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None



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
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = 'test'
    var_6 = {var_1: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/8 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #13
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_2, package=var_2)



# Parsed testcases at query #14
#--------------------------




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
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = 'test'
    var_6 = {var_1: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)

import jinja2.environment as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'invalid_schema'
    var_2 = None
    var_3 = module_1.Form(env=var_0, schema=var_1, values=var_2)



# Parsed testcases at query #15
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Object()
    var_3 = var_1.template_for_field(var_2)



# Parsed testcases at query #16
#--------------------------




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
    var_7 = var_2.serialize(var_5)



# Parsed testcases at query #17
#--------------------------




import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Choice()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/select.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Boolean()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/checkbox.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'text'
    var_3 = module_1.String(format=var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/textarea.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = module_1.String(format=var_2)
    var_4 = var_1.template_for_field(var_3)
    assert var_4 == 'forms/input.html'

import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Object()
    var_3 = var_1.template_for_field(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_jinja2forms_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_both_directory_and_package. Retrieved 11/19 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
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
    var_0 = 'path/to/templates'
    var_1 = 'my_package'
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



# Parsed testcases at query #19
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = None
    var_2 = var_0.load_template_env(directory=var_1, package=var_1)



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



# Parsed testcases at query #2
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)



# Parsed testcases at query #3
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

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader

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
    var_4 = module_3.String()
    var_5 = 'test'
    var_6 = var_3.render_field(field_name=var_5, field=var_4)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = 'Custom Title'
    var_5 = module_3.String()
    var_6 = 'custom_field'
    var_7 = 'test_value'
    var_8 = 'test_error'
    var_9 = var_3.render_field(field_name=var_6, field=var_5, value=var_7, error=var_8)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = True
    var_5 = module_3.String()
    var_6 = 'read_only_field'
    var_7 = var_3.render_field(field_name=var_6, field=var_5)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = True
    var_5 = module_3.String()
    var_6 = 'nullable_field'
    var_7 = var_3.render_field(field_name=var_6, field=var_5)

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
    var_7 = 'secret'
    var_8 = var_3.render_field(field_name=var_6, field=var_5, value=var_7)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = 'a'
    var_5 = 'Option A'
    var_6 = (var_4, var_5)
    var_7 = 'b'
    var_8 = 'Option B'
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = module_3.Choice(choices=var_10)
    var_12 = 'choice_field'
    var_13 = var_3.render_field(field_name=var_12, field=var_11)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = module_3.Boolean()
    var_5 = 'boolean_field'
    var_6 = var_3.render_field(field_name=var_5, field=var_4)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = 'text'
    var_5 = module_3.String(format=var_4)
    var_6 = 'textarea_field'
    var_7 = var_3.render_field(field_name=var_6, field=var_5)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = 'email'
    var_5 = module_3.String(format=var_4)
    var_6 = 'email_field'
    var_7 = var_3.render_field(field_name=var_6, field=var_5)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = module_2.Form(env=var_0, schema=var_2)
    var_4 = 'unknown'
    var_5 = module_3.String(format=var_4)
    var_6 = 'unknown_field'
    var_7 = var_3.render_field(field_name=var_6, field=var_5)



# Parsed testcases at query #5
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'name'
    var_3 = 'age'
    var_4 = module_2.String()
    var_5 = module_2.Integer()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = module_4.Form(env=var_1, schema=var_7, values=var_10)
    var_12 = str(var_11)
    var_13 = var_11.render_fields()



# Parsed testcases at query #6
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
    var_7 = module_2.String()
    var_8 = module_2.Integer()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_3.Schema(var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = module_4.Form(env=var_4, schema=var_10, values=var_13)
    var_15 = {var_5: var_11, var_6: var_12}
    var_16 = var_14.validate(var_15)
    var_17 = var_14.render_fields()
    assert var_17 == '<input /><input />'

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
    var_7 = module_2.String()
    var_8 = module_2.Integer()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_3.Schema(var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = module_4.Form(env=var_4, schema=var_10, values=var_13)
    var_15 = ''
    var_16 = 'invalid'
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = var_14.validate(var_17)
    var_19 = var_14.render_fields()
    assert var_19 == '<input /><input />'

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
    var_7 = module_2.String()
    var_8 = True
    var_9 = module_2.Integer()
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = module_3.Schema(var_10)
    var_12 = 'John'
    var_13 = {var_5: var_12, var_6: var_8}
    var_14 = module_4.Form(env=var_4, schema=var_11, values=var_13)
    var_15 = {var_5: var_12, var_6: var_8}
    var_16 = var_14.validate(var_15)
    var_17 = var_14.render_fields()
    assert var_17 == '<input />'

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
    var_7 = module_2.String()
    var_8 = module_2.Integer()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_3.Schema(var_9)
    var_11 = None
    var_12 = module_4.Form(env=var_4, schema=var_10, values=var_11)
    var_13 = var_12.validate(var_11)
    var_14 = var_12.render_fields()
    assert var_14 == '<input /><input />'



# Parsed testcases at query #7
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'username'
    var_6 = module_2.String()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = 'test'
    var_11 = {var_5: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = var_8.fields[var_5]
    var_14 = None
    var_15 = var_9.render_field(field_name=var_5, field=var_13, value=var_10, error=var_14)
    assert var_15 == "<input id='username' name='username' type='text' value='test'>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'username'
    var_6 = module_2.String()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = 123
    var_11 = {var_5: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = var_8.fields[var_5]
    var_14 = 'Must be a string.'
    var_15 = var_9.render_field(field_name=var_5, field=var_13, value=var_10, error=var_14)
    assert var_15 == "<input id='username' name='username' type='text' value='123'>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'password'
    var_6 = module_2.String(format=var_5)
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = 'secret'
    var_11 = {var_5: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = var_8.fields[var_5]
    var_14 = None
    var_15 = var_9.render_field(field_name=var_5, field=var_13, value=var_10, error=var_14)
    assert var_15 == "<input id='password' name='password' type='password' value=''>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/checkbox.html'
    var_1 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'agree'
    var_6 = module_2.Boolean()
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = True
    var_11 = {var_5: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = var_8.fields[var_5]
    var_14 = None
    var_15 = var_9.render_field(field_name=var_5, field=var_13, value=var_10, error=var_14)
    assert var_15 == "<input id='agree' name='agree' type='checkbox' value='True'>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/select.html'
    var_1 = "<select id='{{ field_id }}' name='{{ field_name }}'><option value='{{ value }}'>{{ value }}</option></select>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'color'
    var_6 = 'red'
    var_7 = 'Red'
    var_8 = (var_6, var_7)
    var_9 = 'blue'
    var_10 = 'Blue'
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = module_2.Choice(choices=var_12)
    var_14 = {var_5: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_4.Form(env=var_4, schema=var_15)
    var_17 = {var_5: var_6}
    var_18 = var_16.validate(var_17)
    var_19 = var_15.fields[var_5]
    var_20 = None
    var_21 = var_16.render_field(field_name=var_5, field=var_19, value=var_6, error=var_20)
    assert var_21 == "<select id='color' name='color'><option value='red'>red</option></select>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/textarea.html'
    var_1 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'description'
    var_6 = 'text'
    var_7 = module_2.String(format=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = 'A long text'
    var_12 = {var_5: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = var_9.fields[var_5]
    var_15 = None
    var_16 = var_10.render_field(field_name=var_5, field=var_14, value=var_11, error=var_15)
    assert var_16 == "<textarea id='description' name='description'>A long text</textarea>"

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'id'
    var_6 = True
    var_7 = module_2.Integer()
    var_8 = {var_5: var_7}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = {var_5: var_6}
    var_12 = var_10.validate(var_11)
    var_13 = var_9.fields[var_5]
    var_14 = None
    var_15 = var_10.render_field(field_name=var_5, field=var_13, value=var_6, error=var_14)
    assert var_15 == ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'email'
    var_6 = module_2.String(format=var_5)
    var_7 = {var_5: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_4, schema=var_8)
    var_10 = 'test@example.com'
    var_11 = {var_5: var_10}
    var_12 = var_9.validate(var_11)
    var_13 = var_8.fields[var_5]
    var_14 = None
    var_15 = var_9.render_field(field_name=var_5, field=var_13, value=var_10, error=var_14)
    assert var_15 == "<input id='email' name='email' type='email' value='test@example.com'>"



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_render_fields_skips_read_only_fields.




# Parsed testcases at query #9
#--------------------------




import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)
    var_5 = module_3.Object()
    var_6 = var_4.template_for_field(var_5)

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)
    var_5 = 'a'
    var_6 = 'A'
    var_7 = (var_5, var_6)
    var_8 = 'b'
    var_9 = 'B'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_3.Choice(choices=var_11)
    var_13 = var_4.template_for_field(var_12)
    assert var_13 == 'forms/select.html'

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)
    var_5 = module_3.Boolean()
    var_6 = var_4.template_for_field(var_5)
    assert var_6 == 'forms/checkbox.html'

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)
    var_5 = 'text'
    var_6 = module_3.String(format=var_5)
    var_7 = var_4.template_for_field(var_6)
    assert var_7 == 'forms/textarea.html'

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)
    var_5 = 'email'
    var_6 = module_3.String(format=var_5)
    var_7 = var_4.template_for_field(var_6)
    assert var_7 == 'forms/input.html'

import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = {}
    var_4 = module_2.Form(env=var_0, schema=var_2, values=var_3)
    var_5 = module_3.Integer()
    var_6 = var_4.template_for_field(var_5)
    assert var_6 == 'forms/input.html'



# Parsed testcases at query #10
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
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = 'John'
    var_6 = {var_1: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)



# Parsed testcases at query #11
#--------------------------




import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'read_only_field'
    var_2 = 'normal_field'
    var_3 = True
    var_4 = module_1.Field(read_only=var_3)
    var_5 = module_1.Field()
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = module_2.Schema(var_6)
    var_8 = 'value1'
    var_9 = 'value2'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_3.Form(env=var_0, schema=var_7, values=var_10)
    var_12 = {var_1: var_8, var_2: var_9}
    var_13 = var_11.validate(var_12)
    var_14 = var_11.render_fields()



# Parsed testcases at query #12
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_pkg'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)



# Parsed testcases at query #13
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

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_render_fields_skips_read_only_fields. Retrieved 16/18 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'read_only_field'
    var_3 = 'normal_field'
    var_4 = True
    var_5 = module_2.Field(read_only=var_4)
    var_6 = module_2.Field()
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = 'value1'
    var_10 = 'value2'
    var_11 = {var_2: var_9, var_3: var_10}
    var_12 = module_4.Form(env=var_1, schema=var_8, values=var_11)
    var_13 = 'data1'
    var_14 = 'data2'
    var_15 = var_12.render_fields()



# Parsed testcases at query #16
#--------------------------




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
    var_5 = 'test'
    var_6 = {var_1: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)



# Parsed testcases at query #17
#--------------------------




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
    var_7 = var_2.serialize(var_5)



# Parsed testcases at query #18
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

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'my.package'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = 'my.package'
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



# Parsed testcases at query #19
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
    var_4 = module_2.Schema(var_3)
    var_5 = {}
    var_6 = module_3.Form(env=var_2, schema=var_4, values=var_5)
    var_7 = var_1.allow_null
    var_8 = 'allow_blank'
    var_9 = False
    var_10 = getattr(var_1, var_8, var_9)
    var_11 = var_7 or var_10
    assert var_11 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_form_init_without_keyword_arguments. Retrieved 3/5 statements.


import jinja2.environment as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = module_1.Schema(var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_jinja2_not_installed. Retrieved 3/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_jinja2forms_constructor_with_directory. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_package. Retrieved 4/8 statements.
# Partially parsed test_jinja2forms_constructor_with_both_directory_and_package. Retrieved 5/9 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
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
    var_0 = 'test_templates'
    var_1 = 'test_package'
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    var_3 = var_2.env
    var_4 = var_2.env.loader

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #23
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_blank'
    var_2 = False
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = True
    var_5 = var_3 is var_4

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'allow_blank'
    var_2 = False
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = var_3 is var_2



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'allow_blank'
    var_3 = False
    var_4 = getattr(var_1, var_2, var_3)



# Parsed testcases at query #25
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Form(env=var_0, schema=var_1, values=var_2)



# Parsed testcases at query #26
#--------------------------




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
    var_7 = var_2.serialize(var_5)



# Parsed testcases at query #27
#--------------------------




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
    var_7 = var_2.serialize(var_5)



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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
    var_7 = var_2.serialize(var_5)



# Parsed testcases at query #30
#--------------------------




import typesystem.schemas as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'jinja2.Environment'
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = module_1.Form(env=var_0, schema=var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_form_init_kwargs_only. Retrieved 6/7 statements.


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'dummy_env'
    var_1 = 'dummy_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.Form(env=var_0, schema=var_1, values=var_4)



# Parsed testcases at query #32
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
    var_4 = module_2.Schema(var_3)
    var_5 = {}
    var_6 = module_3.Form(env=var_2, schema=var_4, values=var_5)
    var_7 = var_1.allow_null
    var_8 = 'allow_blank'
    var_9 = False
    var_10 = getattr(var_1, var_8, var_9)
    var_11 = var_7 or var_10
    assert var_11 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = 'some_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)



# Parsed testcases at query #34
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
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = 'test'
    var_6 = {var_1: var_5}
    var_7 = module_3.Form(env=var_0, schema=var_4, values=var_6)



# Parsed testcases at query #35
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_2, package=var_2)



# Parsed testcases at query #36
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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_form_init_without_keyword_arguments. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'mock_env'
    var_1 = 'mock_schema'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #38
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/path/to/templates'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = var_3.env
    var_5 = var_1.Environment
    var_6 = isinstance(var_4, var_5)
    var_7 = var_3.env.loader
    var_8 = var_1.FileSystemLoader
    var_9 = isinstance(var_7, var_8)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = 'my_package'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env
    var_5 = var_1.Environment
    var_6 = isinstance(var_4, var_5)
    var_7 = var_3.env.loader
    var_8 = var_1.PackageLoader
    var_9 = isinstance(var_7, var_8)

import typesystem.forms as module_0

def test_case_0():
    var_0 = 'jinja2'
    var_1 = __import__(var_0)
    var_2 = '/path/to/templates'
    var_3 = 'my_package'
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = var_4.env
    var_6 = var_1.Environment
    var_7 = isinstance(var_5, var_6)
    var_8 = var_4.env.loader
    var_9 = var_1.ChoiceLoader
    var_10 = isinstance(var_8, var_9)

import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()



# Parsed testcases at query #39
#--------------------------




import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0, values=var_0)



