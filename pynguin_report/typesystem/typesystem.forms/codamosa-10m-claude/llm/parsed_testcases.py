####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = 'read_only_field'
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = module_2.String(format=var_13)
    var_19 = module_2.Boolean()
    var_20 = module_2.String()
    var_21 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = 'John'
    var_24 = 'john@example.com'
    var_25 = {var_12: var_23, var_13: var_24, var_14: var_10}
    var_26 = module_4.Form(env=var_11, schema=var_22, values=var_25)
    var_27 = {var_12: var_23, var_13: var_24, var_14: var_10}
    var_28 = var_26.validate(var_27)
    var_29 = var_26.render_fields()
    var_30 = module_4.Form(env=var_11, schema=var_22)
    var_31 = ''
    var_32 = 'invalid-email'
    var_33 = False
    var_34 = {var_12: var_31, var_13: var_32, var_14: var_33}
    var_35 = var_30.validate(var_34)
    var_36 = var_30.render_fields()
    var_37 = None
    var_38 = module_4.Form(env=var_11, schema=var_22, values=var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = var_38.render_fields()



# Parsed testcases at query #2
#--------------------------


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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'role'
    var_17 = 'Name'
    var_18 = module_2.String()
    var_19 = module_2.String(format=var_13)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'admin'
    var_24 = 'user'
    var_25 = [var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_12: var_18, var_13: var_19, var_14: var_21, var_15: var_22, var_16: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_11, schema=var_28)
    var_30 = var_28.fields[var_12]
    var_31 = 'John'
    var_32 = None
    var_33 = var_29.render_field(field_name=var_12, field=var_30, value=var_31, error=var_32)
    var_34 = var_28.fields[var_13]
    var_35 = 'test@example.com'
    var_36 = var_29.render_field(field_name=var_13, field=var_34, value=var_35, error=var_32)
    var_37 = var_28.fields[var_14]
    var_38 = 'Bio text'
    var_39 = var_29.render_field(field_name=var_14, field=var_37, value=var_38, error=var_32)
    var_40 = var_28.fields[var_15]
    var_41 = var_29.render_field(field_name=var_15, field=var_40, value=var_10, error=var_32)
    var_42 = var_28.fields[var_16]
    var_43 = var_29.render_field(field_name=var_16, field=var_42, value=var_23, error=var_32)
    var_44 = var_28.fields[var_12]
    var_45 = ''
    var_46 = 'This field is required'
    var_47 = var_29.render_field(field_name=var_12, field=var_44, value=var_45, error=var_46)
    var_48 = 'first_name'
    var_49 = module_2.String()
    var_50 = {var_48: var_49}
    var_51 = module_3.Schema(var_50)
    var_52 = module_4.Form(env=var_11, schema=var_51)
    var_53 = var_51.fields[var_48]
    var_54 = 'Jane'
    var_55 = var_52.render_field(field_name=var_48, field=var_53, value=var_54, error=var_32)
    var_56 = 'nickname'
    var_57 = module_2.String()
    var_58 = {var_56: var_57}
    var_59 = module_3.Schema(var_58)
    var_60 = module_4.Form(env=var_11, schema=var_59)
    var_61 = var_59.fields[var_56]
    var_62 = 'Nick'
    var_63 = var_60.render_field(field_name=var_56, field=var_61, value=var_62, error=var_32)
    var_64 = 'password'
    var_65 = module_2.String(format=var_64)
    var_66 = {var_64: var_65}
    var_67 = module_3.Schema(var_66)
    var_68 = module_4.Form(env=var_11, schema=var_67)
    var_69 = var_67.fields[var_64]
    var_70 = 'secret123'
    var_71 = var_68.render_field(field_name=var_64, field=var_69, value=var_70, error=var_32)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.validate() method'
    var_1 = 'name'
    var_2 = 'email'
    var_3 = 'age'
    var_4 = 100
    var_5 = module_0.String(max_length=var_4)
    var_6 = module_0.String(format=var_2)
    var_7 = module_0.Field()
    var_8 = {var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = {}
    var_11 = module_2.DictLoader(var_10)
    var_12 = module_3.Environment(loader=var_11)
    var_13 = None
    var_14 = module_4.Form(env=var_12, schema=var_9, values=var_13)
    var_15 = 'John'
    var_16 = 'john@example.com'
    var_17 = 30
    var_18 = {var_1: var_15, var_2: var_16, var_3: var_17}
    var_19 = var_14.validate(var_18)
    var_20 = var_14.validate(var_18)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.validate() with invalid data'
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = module_2.DictLoader(var_5)
    var_7 = module_3.Environment(loader=var_6)
    var_8 = None
    var_9 = module_4.Form(env=var_7, schema=var_4, values=var_8)
    var_10 = 'not-an-email'
    var_11 = {var_1: var_10}
    var_12 = var_9.validate(var_11)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.validate() with None data'
    var_1 = 'name'
    var_2 = True
    var_3 = module_0.String()
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {}
    var_7 = module_2.DictLoader(var_6)
    var_8 = module_3.Environment(loader=var_7)
    var_9 = None
    var_10 = module_4.Form(env=var_8, schema=var_5, values=var_9)
    var_11 = var_10.validate(var_9)



# Parsed testcases at query #4
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field method with various field types.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value or "" }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value or "" }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = True
    var_12 = module_1.Environment(autoescape=var_11, loader=var_10)
    var_13 = 'username'
    var_14 = 'password'
    var_15 = 'email'
    var_16 = 'bio'
    var_17 = 'is_active'
    var_18 = 'role'
    var_19 = 100
    var_20 = module_2.String(max_length=var_19)
    var_21 = module_2.String(format=var_14)
    var_22 = module_2.String(format=var_15)
    var_23 = 'text'
    var_24 = module_2.String(format=var_23)
    var_25 = module_2.Boolean()
    var_26 = 'admin'
    var_27 = 'Admin'
    var_28 = (var_26, var_27)
    var_29 = 'user'
    var_30 = 'User'
    var_31 = (var_29, var_30)
    var_32 = [var_28, var_31]
    var_33 = module_2.Choice(choices=var_32)
    var_34 = {var_13: var_20, var_14: var_21, var_15: var_22, var_16: var_24, var_17: var_25, var_18: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = None
    var_37 = module_4.Form(env=var_12, schema=var_35, values=var_36)
    var_38 = var_35.fields[var_13]
    var_39 = 'john_doe'
    var_40 = var_37.render_field(field_name=var_13, field=var_38, value=var_39, error=var_36)
    var_41 = var_35.fields[var_14]
    var_42 = 'secret123'
    var_43 = var_37.render_field(field_name=var_14, field=var_41, value=var_42, error=var_36)
    var_44 = var_35.fields[var_15]
    var_45 = 'test@example.com'
    var_46 = var_37.render_field(field_name=var_15, field=var_44, value=var_45, error=var_36)
    var_47 = var_35.fields[var_16]
    var_48 = 'My bio'
    var_49 = var_37.render_field(field_name=var_16, field=var_47, value=var_48, error=var_36)
    var_50 = var_35.fields[var_17]
    var_51 = var_37.render_field(field_name=var_17, field=var_50, value=var_11, error=var_36)
    var_52 = var_35.fields[var_13]
    var_53 = 'john'
    var_54 = 'This field is required.'
    var_55 = var_37.render_field(field_name=var_13, field=var_52, value=var_53, error=var_54)
    var_56 = 'user_name'
    var_57 = module_2.String()
    var_58 = var_37.render_field(field_name=var_56, field=var_57, value=var_36, error=var_36)
    var_59 = var_35.fields[var_18]
    var_60 = var_37.render_field(field_name=var_18, field=var_59, value=var_26, error=var_36)
    var_61 = var_35.fields[var_13]
    var_62 = var_37.render_field(field_name=var_13, field=var_61, value=var_36, error=var_36)



# Parsed testcases at query #5
#--------------------------


import posixpath as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'templates'
    var_1 = 'forms'
    var_2 = 'input.html'
    var_3 = module_0.join(var_2)
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}"{% if value %} value="{{ value }}"{% endif %}{% if required %} required{% endif %}/>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = 'checkbox.html'
    var_6 = module_0.join(var_5)
    var_7 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}/>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = 'select.html'
    var_9 = module_0.join(var_8)
    var_10 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% for choice_value, choice_label in field.choices %}<option value="{{ choice_value }}"{% if value == choice_value %} selected{% endif %}>{{ choice_label }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_11 = 'textarea.html'
    var_12 = module_0.join(var_11)
    var_13 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_14 = 'username'
    var_15 = 'password'
    var_16 = 'is_active'
    var_17 = 'status'
    var_18 = 'bio'
    var_19 = 100
    var_20 = module_1.String(max_length=var_19)
    var_21 = module_1.String(format=var_15)
    var_22 = module_1.Boolean()
    var_23 = 'active'
    var_24 = 'Active'
    var_25 = (var_23, var_24)
    var_26 = 'inactive'
    var_27 = 'Inactive'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_1.Choice(choices=var_29)
    var_31 = 'text'
    var_32 = module_1.String(format=var_31)
    var_33 = {var_14: var_20, var_15: var_21, var_16: var_22, var_17: var_30, var_18: var_32}
    var_34 = module_2.Schema(var_33)
    var_35 = None
    var_36 = var_34.fields[var_14]
    var_37 = 'john_doe'
    var_38 = var_34.fields[var_15]
    var_39 = 'secret'
    var_40 = var_34.fields[var_16]
    var_41 = True
    var_42 = var_34.fields[var_17]
    var_43 = var_34.fields[var_18]
    var_44 = 'My bio'
    var_45 = var_34.fields[var_14]
    var_46 = ''
    var_47 = 'This field is required'
    var_48 = var_34.fields[var_16]



# Parsed testcases at query #6
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = var_1.env
    var_5 = module_0.Jinja2Forms()
    var_6 = '.'
    var_7 = module_0.Jinja2Forms(directory=var_6)



# Parsed testcases at query #7
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}/>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" id="{{ field_id }}" name="{{ field_name }}"{% if value %} checked{% endif %}/>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'email'
    var_13 = module_2.String(format=var_12)
    var_14 = {var_12: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = {}
    var_17 = module_4.Form(env=var_11, schema=var_15, values=var_16)
    var_18 = var_15.fields[var_12]
    var_19 = 'test@example.com'
    var_20 = None
    var_21 = var_17.render_field(field_name=var_12, field=var_18, value=var_19, error=var_20)
    var_22 = var_15.fields[var_12]
    var_23 = 'invalid'
    var_24 = 'Invalid email'
    var_25 = var_17.render_field(field_name=var_12, field=var_22, value=var_23, error=var_24)
    var_26 = 'agree'
    var_27 = module_2.Boolean()
    var_28 = {var_26: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = {}
    var_31 = module_4.Form(env=var_11, schema=var_29, values=var_30)
    var_32 = var_29.fields[var_26]
    var_33 = var_31.render_field(field_name=var_26, field=var_32, value=var_10, error=var_20)
    var_34 = 'status'
    var_35 = 'active'
    var_36 = 'Active'
    var_37 = (var_35, var_36)
    var_38 = 'inactive'
    var_39 = 'Inactive'
    var_40 = (var_38, var_39)
    var_41 = [var_37, var_40]
    var_42 = module_2.Choice(choices=var_41)
    var_43 = {var_34: var_42}
    var_44 = module_3.Schema(var_43)
    var_45 = {}
    var_46 = module_4.Form(env=var_11, schema=var_44, values=var_45)
    var_47 = var_44.fields[var_34]
    var_48 = var_46.render_field(field_name=var_34, field=var_47, value=var_35, error=var_20)
    var_49 = 'password'
    var_50 = module_2.String(format=var_49)
    var_51 = {var_49: var_50}
    var_52 = module_3.Schema(var_51)
    var_53 = {}
    var_54 = module_4.Form(env=var_11, schema=var_52, values=var_53)
    var_55 = var_52.fields[var_49]
    var_56 = 'secret123'
    var_57 = var_54.render_field(field_name=var_49, field=var_55, value=var_56, error=var_20)
    var_58 = 'optional'
    var_59 = module_2.String()
    var_60 = {var_58: var_59}
    var_61 = module_3.Schema(var_60)
    var_62 = {}
    var_63 = module_4.Form(env=var_11, schema=var_61, values=var_62)
    var_64 = var_61.fields[var_58]
    var_65 = var_63.render_field(field_name=var_58, field=var_64, value=var_20, error=var_20)



# Parsed testcases at query #8
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = var_9.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_9.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_9.loader.loaders[var_15]
    var_17 = '/tmp/templates'
    var_18 = module_0.Jinja2Forms(directory=var_17)
    var_19 = None
    var_20 = var_18.load_template_env(directory=var_19, package=var_19)
    var_21 = 'typesystem'
    var_22 = module_0.Jinja2Forms(package=var_21)
    var_23 = None
    var_24 = var_22.load_template_env(directory=var_23, package=var_23)



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_5 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}"{% if choice[0] == value %} selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>'
    var_7 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'active'
    var_14 = 'status'
    var_15 = module_2.String()
    var_16 = module_2.String(format=var_12)
    var_17 = module_2.Boolean()
    var_18 = 'Active'
    var_19 = (var_13, var_18)
    var_20 = 'inactive'
    var_21 = 'Inactive'
    var_22 = (var_20, var_21)
    var_23 = [var_19, var_22]
    var_24 = module_2.Choice(choices=var_23)
    var_25 = {var_11: var_15, var_12: var_16, var_13: var_17, var_14: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = None
    var_28 = module_4.Form(env=var_10, schema=var_26, values=var_27)
    var_29 = 'John'
    var_30 = 'john@example.com'
    var_31 = True
    var_32 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_13}
    var_33 = var_28.validate(var_32)
    var_34 = var_28.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_5 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}"{% if choice[0] == value %} selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>'
    var_7 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = module_2.String()
    var_14 = module_2.String(format=var_12)
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = None
    var_18 = module_4.Form(env=var_10, schema=var_16, values=var_17)
    var_19 = ''
    var_20 = 'invalid'
    var_21 = {var_11: var_19, var_12: var_20}
    var_22 = var_18.validate(var_21)
    var_23 = var_18.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'id'
    var_7 = module_2.String()
    var_8 = True
    var_9 = module_2.String()
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = module_3.Schema(var_10)
    var_12 = 'John'
    var_13 = '123'
    var_14 = {var_5: var_12, var_6: var_13}
    var_15 = module_4.Form(env=var_4, schema=var_11, values=var_14)
    var_16 = {var_5: var_12}
    var_17 = var_15.validate(var_16)
    var_18 = var_15.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
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
    var_12 = var_11.render_fields()



# Parsed testcases at query #10
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = None
    var_1 = module_0.Environment()
    var_2 = 'test_choice'
    var_3 = 'test_boolean'
    var_4 = 'test_text'
    var_5 = 'test_email'
    var_6 = 'test_string'
    var_7 = 'a'
    var_8 = 'A'
    var_9 = (var_7, var_8)
    var_10 = 'b'
    var_11 = 'B'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = module_1.Choice(choices=var_13)
    var_15 = module_1.Boolean()
    var_16 = 'text'
    var_17 = module_1.String(format=var_16)
    var_18 = 'email'
    var_19 = module_1.String(format=var_18)
    var_20 = module_1.String()
    var_21 = {var_2: var_14, var_3: var_15, var_4: var_17, var_5: var_19, var_6: var_20}
    var_22 = module_2.Schema(var_21)
    var_23 = module_3.Form(env=var_1, schema=var_22)
    var_24 = var_22.fields[var_2]
    var_25 = var_23.template_for_field(var_24)
    assert var_25 == 'forms/select.html'
    var_26 = var_22.fields[var_3]
    var_27 = var_23.template_for_field(var_26)
    assert var_27 == 'forms/checkbox.html'
    var_28 = var_22.fields[var_4]
    var_29 = var_23.template_for_field(var_28)
    assert var_29 == 'forms/textarea.html'
    var_30 = var_22.fields[var_5]
    var_31 = var_23.template_for_field(var_30)
    assert var_31 == 'forms/input.html'
    var_32 = var_22.fields[var_6]
    var_33 = var_23.template_for_field(var_32)
    assert var_33 == 'forms/input.html'
    var_34 = module_1.Object()
    var_35 = var_23.template_for_field(var_34)



# Parsed testcases at query #11
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'Test Form.render_field method renders field correctly.'
    var_1 = 'typesystem'
    var_2 = module_0.Jinja2Forms(package=var_1)
    var_3 = 'name'
    var_4 = 'email'
    var_5 = 'age'
    var_6 = 'bio'
    var_7 = 'active'
    var_8 = 'country'
    var_9 = 'password'
    var_10 = 'secret123'
    var_11 = 'John Doe'
    var_12 = 'Invalid email address'
    var_13 = 'optional_field'



# Parsed testcases at query #12
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = './templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #13
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = module_2.String()
    var_14 = {var_12: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = {}
    var_17 = module_4.Form(env=var_11, schema=var_15, values=var_16)
    var_18 = 'Name'
    var_19 = module_2.String()
    var_20 = 'John'
    var_21 = None
    var_22 = var_17.render_field(field_name=var_12, field=var_19, value=var_20, error=var_21)
    var_23 = module_2.String()
    var_24 = 'Jane'
    var_25 = var_17.render_field(field_name=var_12, field=var_23, value=var_24, error=var_21)
    var_26 = 'Invalid name'
    var_27 = var_17.render_field(field_name=var_12, field=var_19, value=var_20, error=var_26)
    var_28 = module_2.Boolean()
    var_29 = 'active'
    var_30 = var_17.render_field(field_name=var_29, field=var_28, value=var_10, error=var_21)
    var_31 = 'password'
    var_32 = module_2.String(format=var_31)
    var_33 = 'secret'
    var_34 = var_17.render_field(field_name=var_31, field=var_32, value=var_33, error=var_21)
    var_35 = 'email'
    var_36 = module_2.String(format=var_35)
    var_37 = 'test@example.com'
    var_38 = var_17.render_field(field_name=var_35, field=var_36, value=var_37, error=var_21)
    var_39 = 'user_name'
    var_40 = 'test'
    var_41 = var_17.render_field(field_name=var_39, field=var_19, value=var_40, error=var_21)
    var_42 = 'Custom Label'
    var_43 = module_2.String()
    var_44 = 'custom'
    var_45 = var_17.render_field(field_name=var_44, field=var_43, value=var_21, error=var_21)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'username'
    var_1 = 100
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'john'
    var_6 = {var_0: var_5}
    var_7 = var_4.fields[var_0]
    var_8 = None
    var_9 = 'forms/input.html'
    var_10 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'user_name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = 'test'
    var_6 = None
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = 'invalid'
    var_6 = 'Invalid email format'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = 'secret123'
    var_6 = None
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'username'
    var_1 = 'User Name'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_0]
    var_6 = None
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'status'
    var_1 = 'active'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_0]
    var_6 = None
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'optional_field'
    var_1 = True
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_0]
    var_6 = None
    var_7 = 0



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'username'
    var_1 = 'Username'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'john_doe'
    var_8 = 'forms/input.html'
    var_9 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 'invalid'
    var_7 = 'Invalid email format'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 'secret123'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'first_name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 'John'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'status'
    var_1 = 'active'
    var_2 = 'inactive'
    var_3 = 'Active'
    var_4 = 'Inactive'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Choice(choices=var_5)
    var_7 = {var_0: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = None
    var_10 = var_8.fields[var_0]
    var_11 = 'forms/select.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'agree'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = True
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'description'
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'Some text'
    var_8 = 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'required_field'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'optional_field'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 0



# Parsed testcases at query #16
#--------------------------


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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'username'
    var_13 = 'password'
    var_14 = 'description'
    var_15 = 'active'
    var_16 = 'status'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = module_2.String(format=var_13)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'inactive'
    var_24 = [var_15, var_23]
    var_25 = module_2.Choice(choices=var_24)
    var_26 = {var_12: var_18, var_13: var_19, var_14: var_21, var_15: var_22, var_16: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_4.Form(env=var_11, schema=var_27)
    var_29 = var_27.fields[var_12]
    var_30 = 'john_doe'
    var_31 = None
    var_32 = var_28.render_field(field_name=var_12, field=var_29, value=var_30, error=var_31)
    var_33 = var_27.fields[var_13]
    var_34 = 'secret'
    var_35 = var_28.render_field(field_name=var_13, field=var_33, value=var_34, error=var_31)
    var_36 = var_27.fields[var_14]
    var_37 = 'Some text'
    var_38 = var_28.render_field(field_name=var_14, field=var_36, value=var_37, error=var_31)
    var_39 = var_27.fields[var_15]
    var_40 = var_28.render_field(field_name=var_15, field=var_39, value=var_10, error=var_31)
    var_41 = var_27.fields[var_16]
    var_42 = var_28.render_field(field_name=var_16, field=var_41, value=var_15, error=var_31)
    var_43 = var_27.fields[var_12]
    var_44 = 'john'
    var_45 = 'This field is required'
    var_46 = var_28.render_field(field_name=var_12, field=var_43, value=var_44, error=var_45)
    var_47 = 'user_name'
    var_48 = module_2.String()
    var_49 = var_28.render_field(field_name=var_47, field=var_48, value=var_31, error=var_31)
    var_50 = 'default_value'
    var_51 = module_2.String()
    var_52 = 'optional'
    var_53 = var_28.render_field(field_name=var_52, field=var_51, value=var_31, error=var_31)
    var_54 = module_2.String()
    var_55 = 'nullable'
    var_56 = var_28.render_field(field_name=var_55, field=var_54, value=var_31, error=var_31)



# Parsed testcases at query #17
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = 'jinja2'
    var_10 = globals()[var_9]
    var_11 = '/path/to/templates'
    var_12 = module_0.Jinja2Forms(directory=var_11)



# Parsed testcases at query #18
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = var_9.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_9.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_9.loader.loaders[var_15]



# Parsed testcases at query #19
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'active'
    var_14 = 100
    var_15 = module_2.String(max_length=var_14)
    var_16 = module_2.String(format=var_12)
    var_17 = module_2.Boolean()
    var_18 = {var_11: var_15, var_12: var_16, var_13: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = 'John'
    var_21 = 'john@example.com'
    var_22 = True
    var_23 = {var_11: var_20, var_12: var_21, var_13: var_22}
    var_24 = module_4.Form(env=var_10, schema=var_19, values=var_23)
    var_25 = {var_11: var_20, var_12: var_21, var_13: var_22}
    var_26 = var_24.validate(var_25)
    var_27 = var_24.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 5
    var_14 = module_2.String(max_length=var_13)
    var_15 = module_2.String(format=var_12)
    var_16 = {var_11: var_14, var_12: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_4.Form(env=var_10, schema=var_17)
    var_19 = 'This is too long'
    var_20 = 'invalid'
    var_21 = {var_11: var_19, var_12: var_20}
    var_22 = var_18.validate(var_21)
    var_23 = var_18.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>{% if error %}<p>{{ error }}</p>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'created_at'
    var_13 = module_2.String()
    var_14 = True
    var_15 = module_2.String()
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = 'John'
    var_19 = '2023-01-01'
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = module_4.Form(env=var_10, schema=var_17, values=var_20)
    var_22 = {var_11: var_18}
    var_23 = var_21.validate(var_22)
    var_24 = var_21.render_fields()



# Parsed testcases at query #20
#--------------------------


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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'age'
    var_15 = 'active'
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = module_2.String(format=var_13)
    var_19 = module_2.Field()
    var_20 = module_2.Boolean()
    var_21 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = None
    var_24 = module_4.Form(env=var_11, schema=var_22, values=var_23)
    var_25 = var_24.render_fields()

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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 100
    var_15 = module_2.String(max_length=var_14)
    var_16 = module_2.String(format=var_13)
    var_17 = {var_12: var_15, var_13: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = 'John'
    var_20 = 'john@example.com'
    var_21 = {var_12: var_19, var_13: var_20}
    var_22 = module_4.Form(env=var_11, schema=var_18, values=var_21)
    var_23 = var_22.render_fields()

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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span>{{ error }}</span>{% endif %}'
    var_5 = '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 100
    var_14 = module_2.String(max_length=var_13)
    var_15 = {var_12: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = None
    var_18 = module_4.Form(env=var_11, schema=var_16, values=var_17)
    var_19 = ''
    var_20 = 'This field is required'
    var_21 = var_18.render_fields()

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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}"{% if required %} required{% endif %}></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'id'
    var_14 = 100
    var_15 = module_2.String(max_length=var_14)
    var_16 = module_2.String()
    var_17 = {var_12: var_15, var_13: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = None
    var_20 = module_4.Form(env=var_11, schema=var_18, values=var_19)
    var_21 = var_20.render_fields()



# Parsed testcases at query #21
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.load_template_env(package=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = var_2.loader
    var_6 = var_1.load_template_env()
    var_7 = None
    var_8 = var_1.load_template_env(directory=var_7, package=var_7)



# Parsed testcases at query #22
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = '/tmp'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #23
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'Test Form.render_field method renders fields correctly.'
    var_1 = 'typesystem'
    var_2 = module_0.Jinja2Forms(package=var_1)
    var_3 = 'name'
    var_4 = 'John Doe'
    var_5 = None
    var_6 = 'email'
    var_7 = 'test@example.com'
    var_8 = 'password'
    var_9 = 'secret'
    var_10 = 'subscribe'
    var_11 = True
    var_12 = 'age'
    var_13 = '25'
    var_14 = 'bio'
    var_15 = 'My bio'
    var_16 = 'country'
    var_17 = 'us'
    var_18 = 'invalid'
    var_19 = 'Invalid email'
    var_20 = 'optional_field'
    var_21 = 'field_with_default'
    var_22 = 'default_value'



# Parsed testcases at query #24
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = module_0.Jinja2Forms()
    var_3 = isinstance(var_2, var_1)
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = isinstance(var_6, var_3)
    var_8 = 'typesystem'
    var_9 = var_5.env
    var_10 = isinstance(var_9, var_3)



# Parsed testcases at query #25
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = module_0.Jinja2Forms()
    var_2 = 'typesystem'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env
    var_5 = 'typesystem'
    var_6 = var_3.env



# Parsed testcases at query #26
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field() method'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'password'
    var_15 = 'bio'
    var_16 = 'active'
    var_17 = 'country'
    var_18 = 100
    var_19 = module_2.String(max_length=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = module_2.String(format=var_14)
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = module_2.Boolean()
    var_25 = 'us'
    var_26 = 'United States'
    var_27 = (var_25, var_26)
    var_28 = 'uk'
    var_29 = 'United Kingdom'
    var_30 = (var_28, var_29)
    var_31 = [var_27, var_30]
    var_32 = module_2.Choice(choices=var_31)
    var_33 = {var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_23, var_16: var_24, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_4.Form(env=var_11, schema=var_34)
    var_36 = var_34.fields[var_12]
    var_37 = 'John'
    var_38 = None
    var_39 = var_35.render_field(field_name=var_12, field=var_36, value=var_37, error=var_38)
    var_40 = var_34.fields[var_12]
    var_41 = 'Name is required'
    var_42 = var_35.render_field(field_name=var_12, field=var_40, value=var_37, error=var_41)
    var_43 = var_34.fields[var_13]
    var_44 = 'test@example.com'
    var_45 = var_35.render_field(field_name=var_13, field=var_43, value=var_44, error=var_38)
    var_46 = var_34.fields[var_14]
    var_47 = 'secret'
    var_48 = var_35.render_field(field_name=var_14, field=var_46, value=var_47, error=var_38)
    var_49 = var_34.fields[var_15]
    var_50 = 'My bio'
    var_51 = var_35.render_field(field_name=var_15, field=var_49, value=var_50, error=var_38)
    var_52 = var_34.fields[var_16]
    var_53 = True
    var_54 = var_35.render_field(field_name=var_16, field=var_52, value=var_53, error=var_38)
    var_55 = var_34.fields[var_17]
    var_56 = var_35.render_field(field_name=var_17, field=var_55, value=var_25, error=var_38)
    var_57 = 'first_name'
    var_58 = var_34.fields[var_12]
    var_59 = var_35.render_field(field_name=var_57, field=var_58, value=var_37, error=var_38)



# Parsed testcases at query #27
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice[0] }}" {% if choice[0] == value %}selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = 'status'
    var_16 = 'bio'
    var_17 = 'age'
    var_18 = 100
    var_19 = module_2.String(max_length=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = module_2.Boolean()
    var_22 = 'Active'
    var_23 = (var_14, var_22)
    var_24 = 'inactive'
    var_25 = 'Inactive'
    var_26 = (var_24, var_25)
    var_27 = [var_23, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = 'text'
    var_30 = module_2.String(format=var_29)
    var_31 = 'number'
    var_32 = module_2.String(format=var_31)
    var_33 = {var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_28, var_16: var_30, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = {}
    var_36 = module_4.Form(env=var_11, schema=var_34, values=var_35)
    var_37 = var_34.fields[var_12]
    var_38 = 'John'
    var_39 = None
    var_40 = var_36.render_field(field_name=var_12, field=var_37, value=var_38, error=var_39)
    var_41 = var_34.fields[var_13]
    var_42 = 'test@example.com'
    var_43 = var_36.render_field(field_name=var_13, field=var_41, value=var_42, error=var_39)
    var_44 = var_34.fields[var_17]
    var_45 = '25'
    var_46 = var_36.render_field(field_name=var_17, field=var_44, value=var_45, error=var_39)
    var_47 = var_34.fields[var_14]
    var_48 = var_36.render_field(field_name=var_14, field=var_47, value=var_10, error=var_39)
    var_49 = var_34.fields[var_15]
    var_50 = var_36.render_field(field_name=var_15, field=var_49, value=var_14, error=var_39)
    var_51 = var_34.fields[var_16]
    var_52 = 'My bio'
    var_53 = var_36.render_field(field_name=var_16, field=var_51, value=var_52, error=var_39)
    var_54 = var_34.fields[var_12]
    var_55 = 'Invalid name'
    var_56 = var_36.render_field(field_name=var_12, field=var_54, value=var_38, error=var_55)
    var_57 = module_2.String()
    var_58 = 'optional'
    var_59 = var_36.render_field(field_name=var_58, field=var_57, value=var_39, error=var_39)
    var_60 = 'password'
    var_61 = module_2.String(format=var_60)
    var_62 = 'secret'
    var_63 = var_36.render_field(field_name=var_60, field=var_61, value=var_62, error=var_39)
    var_64 = 'user_name'
    var_65 = var_34.fields[var_12]
    var_66 = 'test'
    var_67 = var_36.render_field(field_name=var_64, field=var_65, value=var_66, error=var_39)



# Parsed testcases at query #28
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = 'jinja2'
    var_10 = globals()[var_9]
    var_11 = '/tmp/templates'
    var_12 = module_0.Jinja2Forms(directory=var_11)



# Parsed testcases at query #29
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = '/tmp'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #30
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = var_8.load_template_env(directory=var_0)
    var_12 = module_0.Jinja2Forms(directory=var_0)
    var_13 = var_12.load_template_env()
    var_14 = module_0.Jinja2Forms(directory=var_13)
    var_15 = None
    var_16 = var_14.load_template_env(directory=var_13, package=var_15)
    var_17 = var_16.loader



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test render_field method of Form class.'
    var_1 = 'Username'
    var_2 = module_0.String()
    var_3 = 'username'
    var_4 = 'testuser'
    var_5 = None
    var_6 = 'required'
    var_7 = module_0.String()
    var_8 = 'email'
    var_9 = 'Email'
    var_10 = module_0.String()
    var_11 = 'invalid'
    var_12 = 'Invalid email format'
    var_13 = 'password'
    var_14 = 'Password'
    var_15 = module_0.String(format=var_13)
    var_16 = 'secret123'
    var_17 = True
    var_18 = 'Optional Field'
    var_19 = module_0.String()
    var_20 = 'optional'
    var_21 = 'A'
    var_22 = 'B'
    var_23 = 'C'
    var_24 = [var_21, var_22, var_23]
    var_25 = 'Select'
    var_26 = module_0.Choice(choices=var_24)
    var_27 = 'choice'
    var_28 = 'Agree'
    var_29 = module_0.Boolean()
    var_30 = 'agree'
    var_31 = 'User Name'
    var_32 = module_0.String()
    var_33 = 'user_name'
    var_34 = module_0.String()
    var_35 = 'myfield'



# Parsed testcases at query #32
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = '/tmp/templates'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #33
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_0, package=var_2)
    var_4 = var_3.loader
    var_5 = 'typesystem'
    var_6 = module_0.Jinja2Forms(package=var_5)
    var_7 = var_6.load_template_env(directory=var_2, package=var_5)
    var_8 = var_7.loader
    var_9 = module_0.Jinja2Forms(directory=var_0, package=var_5)
    var_10 = var_9.load_template_env(directory=var_0, package=var_5)
    var_11 = var_10.loader
    var_12 = var_10.loader.loaders
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 0
    var_15 = var_10.loader.loaders[var_14]
    var_16 = 1
    var_17 = var_10.loader.loaders[var_16]



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'Test Field'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_0]
    var_6 = 'test_value'
    var_7 = None
    var_8 = 'forms/input.html'
    var_9 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = None
    var_6 = 'Field is required'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = 'secret'
    var_6 = None
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'status'
    var_1 = 'active'
    var_2 = 'Active'
    var_3 = (var_1, var_2)
    var_4 = 'inactive'
    var_5 = 'Inactive'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.Choice(choices=var_7)
    var_9 = {var_0: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = var_10.fields[var_0]
    var_12 = None
    var_13 = 'forms/select.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'agree'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = True
    var_6 = None
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'description'
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_0]
    var_6 = 'Some text'
    var_7 = None
    var_8 = 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field_name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = var_3.fields[var_0]
    var_5 = None
    var_6 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'default_val'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_0]
    var_6 = None
    var_7 = 0



# Parsed testcases at query #35
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = var_1.env
    var_5 = module_0.Jinja2Forms()



# Parsed testcases at query #36
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_5 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_7 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = 100
    var_16 = module_2.String(max_length=var_15)
    var_17 = module_2.String(format=var_13)
    var_18 = module_2.Boolean()
    var_19 = {var_12: var_16, var_13: var_17, var_14: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = 'John'
    var_22 = 'john@example.com'
    var_23 = {var_12: var_21, var_13: var_22, var_14: var_10}
    var_24 = module_4.Form(env=var_11, schema=var_20, values=var_23)
    var_25 = var_24.render_fields()
    var_26 = len(var_25)
    var_27 = 'Jane'
    var_28 = 'jane@example.com'
    var_29 = False
    var_30 = {var_12: var_27, var_13: var_28, var_14: var_29}
    var_31 = var_24.validate(var_30)
    var_32 = var_24.render_fields()
    var_33 = 'Bob'
    var_34 = {var_12: var_33}
    var_35 = module_4.Form(env=var_11, schema=var_20, values=var_34)
    var_36 = ''
    var_37 = 'invalid-email'
    var_38 = {var_12: var_36, var_13: var_37, var_14: var_10}
    var_39 = var_35.validate(var_38)
    var_40 = var_35.render_fields()
    var_41 = 'id'
    var_42 = module_2.String()
    var_43 = module_2.String()
    var_44 = {var_41: var_42, var_12: var_43}
    var_45 = module_3.Schema(var_44)
    var_46 = '123'
    var_47 = 'Test'
    var_48 = {var_41: var_46, var_12: var_47}
    var_49 = module_4.Form(env=var_11, schema=var_45, values=var_48)
    var_50 = var_49.render_fields()



# Parsed testcases at query #37
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = 'jinja2'
    var_10 = globals()[var_9]
    var_11 = '/tmp/templates'
    var_12 = module_0.Jinja2Forms(directory=var_11)



# Parsed testcases at query #38
#--------------------------


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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'password'
    var_15 = 'bio'
    var_16 = 'active'
    var_17 = 'role'
    var_18 = 'Name'
    var_19 = module_2.String()
    var_20 = module_2.String(format=var_13)
    var_21 = module_2.String(format=var_14)
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = module_2.Boolean()
    var_25 = 'admin'
    var_26 = 'user'
    var_27 = [var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = {var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_23, var_16: var_24, var_17: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = {}
    var_32 = module_4.Form(env=var_11, schema=var_30, values=var_31)
    var_33 = var_30.fields[var_12]
    var_34 = 'John'
    var_35 = None
    var_36 = var_32.render_field(field_name=var_12, field=var_33, value=var_34, error=var_35)
    var_37 = var_30.fields[var_13]
    var_38 = 'test@example.com'
    var_39 = var_32.render_field(field_name=var_13, field=var_37, value=var_38, error=var_35)
    var_40 = var_30.fields[var_14]
    var_41 = 'secret'
    var_42 = var_32.render_field(field_name=var_14, field=var_40, value=var_41, error=var_35)
    var_43 = var_30.fields[var_15]
    var_44 = 'My bio'
    var_45 = var_32.render_field(field_name=var_15, field=var_43, value=var_44, error=var_35)
    var_46 = var_30.fields[var_16]
    var_47 = var_32.render_field(field_name=var_16, field=var_46, value=var_10, error=var_35)
    var_48 = var_30.fields[var_17]
    var_49 = var_32.render_field(field_name=var_17, field=var_48, value=var_25, error=var_35)
    var_50 = var_30.fields[var_12]
    var_51 = 'This field is required'
    var_52 = var_32.render_field(field_name=var_12, field=var_50, value=var_34, error=var_51)
    var_53 = 'user_name'
    var_54 = module_2.String()
    var_55 = 'test'
    var_56 = var_32.render_field(field_name=var_53, field=var_54, value=var_55, error=var_35)
    var_57 = var_30.fields[var_13]
    var_58 = var_32.render_field(field_name=var_13, field=var_57, value=var_35, error=var_35)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = 'Invalid value'
    var_6 = 0

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = 'regular_field'
    var_2 = 'ro'
    var_3 = 'val'
    var_4 = None
    var_5 = {var_1: var_3}
    var_6 = 0

def test_case_0():
    var_0 = "<input type='text' name='field1'>"
    var_1 = "<input type='email' name='field2'>"
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'value1'
    var_5 = 'test@example.com'
    var_6 = None
    var_7 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #40
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = '/path/to/templates'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = 'is_active'
    var_3 = 'read_only_field'
    var_4 = module_0.String()
    var_5 = module_0.String(format=var_1)
    var_6 = module_0.Boolean()
    var_7 = True
    var_8 = module_0.String()
    var_9 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = 'John'
    var_12 = 'john@example.com'
    var_13 = {var_0: var_11, var_1: var_12, var_2: var_7}
    var_14 = {var_0: var_11, var_1: var_12, var_2: var_7}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = 5
    var_3 = module_0.String(max_length=var_2)
    var_4 = module_0.String(format=var_1)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = None
    var_8 = 'This is too long'
    var_9 = 'invalid-email'
    var_10 = {var_0: var_8, var_1: var_9}

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = {}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = {var_0: var_7, var_1: var_8}



# Parsed testcases at query #42
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field method renders field correctly with proper attributes.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = 'username'
    var_13 = 'password'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'role'
    var_17 = 'email'
    var_18 = module_2.String()
    var_19 = module_2.String(format=var_13)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'admin'
    var_24 = 'Admin'
    var_25 = (var_23, var_24)
    var_26 = 'user'
    var_27 = 'User'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = True
    var_32 = module_2.String(format=var_17)
    var_33 = {var_12: var_18, var_13: var_19, var_14: var_21, var_15: var_22, var_16: var_30, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_4.Form(env=var_11, schema=var_34)
    var_36 = var_34.fields[var_12]
    var_37 = 'john'
    var_38 = None
    var_39 = var_35.render_field(field_name=var_12, field=var_36, value=var_37, error=var_38)
    var_40 = var_34.fields[var_13]
    var_41 = 'secret'
    var_42 = var_35.render_field(field_name=var_13, field=var_40, value=var_41, error=var_38)
    var_43 = var_34.fields[var_12]
    var_44 = 'Username is required'
    var_45 = var_35.render_field(field_name=var_12, field=var_43, value=var_37, error=var_44)
    var_46 = var_34.fields[var_15]
    var_47 = var_35.render_field(field_name=var_15, field=var_46, value=var_31, error=var_38)
    var_48 = var_34.fields[var_16]
    var_49 = var_35.render_field(field_name=var_16, field=var_48, value=var_23, error=var_38)
    var_50 = var_34.fields[var_14]
    var_51 = 'My bio'
    var_52 = var_35.render_field(field_name=var_14, field=var_50, value=var_51, error=var_38)
    var_53 = var_34.fields[var_17]
    var_54 = var_35.render_field(field_name=var_17, field=var_53, value=var_38, error=var_38)
    var_55 = 'user_name'
    var_56 = module_2.String()
    var_57 = var_35.render_field(field_name=var_55, field=var_56, value=var_38, error=var_38)



# Parsed testcases at query #43
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_fields() method'
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<textarea name="{{ field_name }}" {% if required %}required{% endif %} id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}" />{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = '<select name="{{ field_name }}" {% if required %}required{% endif %} id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice[0] }}" {% if choice[0] == value %}selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'role'
    var_17 = 'hidden_field'
    var_18 = 100
    var_19 = module_2.String(max_length=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = module_2.Boolean()
    var_24 = 'admin'
    var_25 = 'Administrator'
    var_26 = (var_24, var_25)
    var_27 = 'user'
    var_28 = 'User'
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = module_2.Choice(choices=var_30)
    var_32 = 'hidden'
    var_33 = True
    var_34 = module_2.String(format=var_32)
    var_35 = {var_12: var_19, var_13: var_20, var_14: var_22, var_15: var_23, var_16: var_31, var_17: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = 'John'
    var_38 = 'john@example.com'
    var_39 = {var_12: var_37, var_13: var_38}
    var_40 = module_4.Form(env=var_11, schema=var_36, values=var_39)
    var_41 = var_40.render_fields()
    var_42 = module_4.Form(env=var_11, schema=var_36)
    var_43 = ''
    var_44 = 'invalid'
    var_45 = {var_12: var_43, var_13: var_44, var_15: var_33}
    var_46 = var_42.validate(var_45)
    var_47 = var_42.render_fields()
    var_48 = module_4.Form(env=var_11, schema=var_36)
    var_49 = 'Jane'
    var_50 = 'jane@example.com'
    var_51 = {var_12: var_49, var_13: var_50, var_15: var_33, var_16: var_24}
    var_52 = var_48.validate(var_51)
    var_53 = var_48.render_fields()
    var_54 = None
    var_55 = module_4.Form(env=var_11, schema=var_36, values=var_54)
    var_56 = var_55.render_fields()
    var_57 = module_4.Form(env=var_11, schema=var_36)
    var_58 = 'Bob'
    var_59 = {var_12: var_58}
    var_60 = var_57.validate(var_59)
    var_61 = var_57.render_fields()



# Parsed testcases at query #44
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'name'
    var_11 = 'email'
    var_12 = 'active'
    var_13 = 'readonly_field'
    var_14 = 100
    var_15 = module_1.String(max_length=var_14)
    var_16 = module_1.String(format=var_11)
    var_17 = module_1.Boolean()
    var_18 = True
    var_19 = module_1.String()
    var_20 = {var_10: var_15, var_11: var_16, var_12: var_17, var_13: var_19}
    var_21 = module_2.Schema(var_20)
    var_22 = 'John'
    var_23 = 'john@example.com'
    var_24 = 'hidden'
    var_25 = {var_10: var_22, var_11: var_23, var_12: var_18, var_13: var_24}
    var_26 = module_3.Form(env=var_0, schema=var_21, values=var_25)
    var_27 = {var_10: var_22, var_11: var_23, var_12: var_18}
    var_28 = var_26.validate(var_27)
    var_29 = var_26.render_fields()

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'name'
    var_11 = 'email'
    var_12 = 100
    var_13 = module_1.String(max_length=var_12)
    var_14 = module_1.String(format=var_11)
    var_15 = {var_10: var_13, var_11: var_14}
    var_16 = module_2.Schema(var_15)
    var_17 = module_3.Form(env=var_0, schema=var_16)
    var_18 = ''
    var_19 = 'invalid'
    var_20 = {var_10: var_18, var_11: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = var_17.render_fields()

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_6 = '<input type="checkbox" name="{{ field_name }}">'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'name'
    var_11 = 100
    var_12 = module_1.String(max_length=var_11)
    var_13 = {var_10: var_12}
    var_14 = module_2.Schema(var_13)
    var_15 = 'Test'
    var_16 = {var_10: var_15}
    var_17 = module_3.Form(env=var_0, schema=var_14, values=var_16)
    var_18 = var_17.render_fields()

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_6 = '<input type="checkbox" name="{{ field_name }}">'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'name'
    var_11 = 'age'
    var_12 = 100
    var_13 = module_1.String(max_length=var_12)
    var_14 = True
    var_15 = module_1.String()
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = module_2.Schema(var_16)
    var_18 = module_3.Form(env=var_0, schema=var_17)
    var_19 = {}
    var_20 = var_18.validate(var_19)
    var_21 = var_18.render_fields()



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test Form.render_field() method'
    var_1 = 'test_field'
    var_2 = 'Test Field'
    var_3 = module_0.String()
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = var_5.fields[var_1]
    var_7 = 'test_value'
    var_8 = None
    var_9 = 'forms/input.html'
    var_10 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field with password field masks value'
    var_1 = 'password'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_1]
    var_6 = 'secret123'
    var_7 = None
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field with error message'
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_1]
    var_6 = 'invalid'
    var_7 = 'Invalid email format'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field converts underscores to hyphens in field_id'
    var_1 = 'user_name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_1]
    var_6 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field with Choice field uses select template'
    var_1 = 'color'
    var_2 = 'red'
    var_3 = 'blue'
    var_4 = [var_2, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = {var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = var_7.fields[var_1]
    var_9 = 'forms/select.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field with Boolean field uses checkbox template'
    var_1 = 'agree'
    var_2 = module_0.Boolean()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = var_4.fields[var_1]
    var_6 = True
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field with text format String field uses textarea template'
    var_1 = 'description'
    var_2 = 'text'
    var_3 = module_0.String(format=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = var_5.fields[var_1]
    var_7 = 'text content'
    var_8 = 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field required attribute based on field configuration'
    var_1 = 'optional'
    var_2 = 'default_value'
    var_3 = module_0.String()
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = var_5.fields[var_1]
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Test render_field required attribute for nullable field'
    var_1 = 'nullable_field'
    var_2 = True
    var_3 = module_0.String()
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = var_5.fields[var_1]
    var_7 = 0



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = 'active'
    var_3 = 100
    var_4 = module_0.String(max_length=var_3)
    var_5 = module_0.String(format=var_1)
    var_6 = module_0.Boolean()
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'forms/input.html'
    var_10 = 'forms/checkbox.html'
    var_11 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}" {% if required %}required{% endif %}>'
    var_12 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_2.DictLoader(var_13)
    var_15 = True
    var_16 = module_3.Environment(autoescape=var_15, loader=var_14)
    var_17 = 'John'
    var_18 = 'john@example.com'
    var_19 = {var_0: var_17, var_1: var_18, var_2: var_15}
    var_20 = module_4.Form(env=var_16, schema=var_8, values=var_19)
    var_21 = str(var_20)
    var_22 = len(var_21)
    var_23 = 0
    var_24 = var_22 >= var_23
    var_25 = None
    var_26 = module_4.Form(env=var_16, schema=var_8, values=var_25)
    var_27 = str(var_26)
    var_28 = {}
    var_29 = module_4.Form(env=var_16, schema=var_8, values=var_28)
    var_30 = ''
    var_31 = 'invalid'
    var_32 = {var_0: var_30, var_1: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = str(var_29)



# Parsed testcases at query #3
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
    var_5 = var_3.input_type_for_field(var_4)
    assert var_5 == 'text'
    var_6 = None
    var_7 = module_3.String(format=var_6)
    var_8 = var_3.input_type_for_field(var_7)
    assert var_8 == 'text'
    var_9 = 'color'
    var_10 = (var_9, var_9)
    var_11 = 'datetime'
    var_12 = 'datetime-local'
    var_13 = (var_11, var_12)
    var_14 = 'date'
    var_15 = (var_14, var_14)
    var_16 = 'email'
    var_17 = (var_16, var_16)
    var_18 = 'hidden'
    var_19 = (var_18, var_18)
    var_20 = 'month'
    var_21 = (var_20, var_20)
    var_22 = 'number'
    var_23 = (var_22, var_22)
    var_24 = 'password'
    var_25 = (var_24, var_24)
    var_26 = 'range'
    var_27 = (var_26, var_26)
    var_28 = 'search'
    var_29 = (var_28, var_28)
    var_30 = 'tel'
    var_31 = (var_30, var_30)
    var_32 = 'text'
    var_33 = (var_32, var_32)
    var_34 = 'time'
    var_35 = (var_34, var_34)
    var_36 = 'url'
    var_37 = (var_36, var_36)
    var_38 = 'week'
    var_39 = (var_38, var_38)
    var_40 = [var_10, var_13, var_15, var_17, var_19, var_21, var_23, var_25, var_27, var_29, var_31, var_33, var_35, var_37, var_39]
    var_41 = 'unsupported_format'
    var_42 = module_3.String(format=var_41)
    var_43 = var_3.input_type_for_field(var_42)
    assert var_43 == 'text'
    var_44 = module_3.Boolean()
    var_45 = var_3.input_type_for_field(var_44)
    assert var_45 == 'text'



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
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = module_3.Choice(choices=var_6)
    var_8 = var_3.template_for_field(var_7)
    assert var_8 == 'forms/select.html'
    var_9 = module_3.Boolean()
    var_10 = var_3.template_for_field(var_9)
    assert var_10 == 'forms/checkbox.html'
    var_11 = 'text'
    var_12 = module_3.String(format=var_11)
    var_13 = var_3.template_for_field(var_12)
    assert var_13 == 'forms/textarea.html'
    var_14 = 'email'
    var_15 = module_3.String(format=var_14)
    var_16 = var_3.template_for_field(var_15)
    assert var_16 == 'forms/input.html'
    var_17 = module_3.String()
    var_18 = var_3.template_for_field(var_17)
    assert var_18 == 'forms/input.html'
    var_19 = {}
    var_20 = module_3.Object(properties=var_19)
    var_21 = var_3.template_for_field(var_20)



# Parsed testcases at query #5
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_field'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_0, schema=var_4)
    var_6 = 'a'
    var_7 = 'Option A'
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 'Option B'
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = module_1.Choice(choices=var_12)
    var_14 = var_5.template_for_field(var_13)
    assert var_14 == 'forms/select.html'
    var_15 = module_1.Boolean()
    var_16 = var_5.template_for_field(var_15)
    assert var_16 == 'forms/checkbox.html'
    var_17 = 'text'
    var_18 = module_1.String(format=var_17)
    var_19 = var_5.template_for_field(var_18)
    assert var_19 == 'forms/textarea.html'
    var_20 = module_1.String()
    var_21 = var_5.template_for_field(var_20)
    assert var_21 == 'forms/input.html'
    var_22 = 'email'
    var_23 = module_1.String(format=var_22)
    var_24 = var_5.template_for_field(var_23)
    assert var_24 == 'forms/input.html'
    var_25 = 'nested'
    var_26 = module_1.String()
    var_27 = {var_25: var_26}
    var_28 = module_1.Object(properties=var_27)
    var_29 = var_5.template_for_field(var_28)



# Parsed testcases at query #6
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'name'
    var_3 = 'email'
    var_4 = 'active'
    var_5 = module_1.String()
    var_6 = module_1.String(format=var_3)
    var_7 = module_1.Boolean()
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = var_1.create_form(var_9)
    var_11 = 'John'
    var_12 = 'john@example.com'
    var_13 = True
    var_14 = {var_2: var_11, var_3: var_12, var_4: var_13}
    var_15 = var_1.create_form(var_9, var_14)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'templates'
    var_1 = 'forms'
    var_2 = 'input.html'
    var_3 = '<input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_4 = 'checkbox.html'
    var_5 = '<input type="checkbox" id="{{ field_id }}" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = 'select.html'
    var_7 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}"{% if choice[0] == value %} selected{% endif %}>{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = 'textarea.html'
    var_9 = '<textarea id="{{ field_id }}" name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_10 = 'name'
    var_11 = 'email'
    var_12 = module_0.String()
    var_13 = module_0.String(format=var_11)
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = 'John'
    var_17 = {var_10: var_16}
    var_18 = var_15.fields[var_10]
    var_19 = var_15.fields[var_10]
    var_20 = 'This field is required'
    var_21 = var_15.fields[var_11]
    var_22 = 'test@example.com'
    var_23 = 'password'
    var_24 = module_0.String(format=var_23)
    var_25 = {var_23: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = var_26.fields[var_23]
    var_28 = 'secret123'
    var_29 = 'nickname'
    var_30 = True
    var_31 = module_0.String()
    var_32 = {var_29: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = var_33.fields[var_29]
    var_35 = ''
    var_36 = 'active'
    var_37 = module_0.Boolean()
    var_38 = {var_36: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = var_39.fields[var_36]
    var_41 = 'status'
    var_42 = 'Active'
    var_43 = (var_36, var_42)
    var_44 = 'inactive'
    var_45 = 'Inactive'
    var_46 = (var_44, var_45)
    var_47 = [var_43, var_46]
    var_48 = module_0.Choice(choices=var_47)
    var_49 = {var_41: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = var_50.fields[var_41]



# Parsed testcases at query #8
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = '/tmp'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #9
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'name'
    var_3 = 'email'
    var_4 = module_1.String()
    var_5 = module_1.String(format=var_3)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_2.Schema(var_6)
    var_8 = var_1.create_form(var_7)

import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'name'
    var_3 = 'email'
    var_4 = module_1.String()
    var_5 = module_1.String(format=var_3)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_2.Schema(var_6)
    var_8 = 'John'
    var_9 = 'john@example.com'
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = var_1.create_form(var_7, var_10)

import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'username'
    var_3 = module_1.String()
    var_4 = {var_2: var_3}
    var_5 = module_2.Schema(var_4)
    var_6 = 'email'
    var_7 = 'age'
    var_8 = module_1.String(format=var_6)
    var_9 = 'number'
    var_10 = module_1.String(format=var_9)
    var_11 = {var_6: var_8, var_7: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = var_1.create_form(var_5)
    var_14 = var_1.create_form(var_12)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = 'Test that create_form returns a Form instance with correct initialization.'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.String()
    var_4 = 'number'
    var_5 = module_0.String(format=var_4)
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = 'typesystem'
    var_9 = module_2.Jinja2Forms(package=var_8)
    var_10 = var_9.create_form(var_7)
    var_11 = 'John'
    var_12 = '30'
    var_13 = {var_1: var_11, var_2: var_12}
    var_14 = var_9.create_form(var_7, var_13)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'username'
    var_1 = 100
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'john_doe'
    var_8 = 'forms/input.html'
    var_9 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 'invalid'
    var_7 = 'Invalid email format'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 'secret123'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'user_name'
    var_1 = 'Username'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'john'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'agree'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = True
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'color'
    var_1 = 'red'
    var_2 = 'Red'
    var_3 = (var_1, var_2)
    var_4 = 'blue'
    var_5 = 'Blue'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.Choice(choices=var_7)
    var_9 = {var_0: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = None
    var_12 = var_10.fields[var_0]
    var_13 = 'forms/select.html'



# Parsed testcases at query #12
#--------------------------


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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<textarea name="{{ field_name }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}"{% if required %} required{% endif %}>{% for choice in field.choices %}<option value="{{ choice.value }}">{{ choice.title }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = 'read_only_field'
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = module_2.String(format=var_13)
    var_19 = module_2.Boolean()
    var_20 = module_2.String()
    var_21 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = 'John'
    var_24 = 'john@example.com'
    var_25 = {var_12: var_23, var_13: var_24, var_14: var_10}
    var_26 = module_4.Form(env=var_11, schema=var_22, values=var_25)
    var_27 = {var_12: var_23, var_13: var_24, var_14: var_10}
    var_28 = var_26.validate(var_27)
    var_29 = var_26.render_fields()
    var_30 = {}
    var_31 = module_4.Form(env=var_11, schema=var_22, values=var_30)
    var_32 = ''
    var_33 = 'invalid'
    var_34 = False
    var_35 = {var_12: var_32, var_13: var_33, var_14: var_34}
    var_36 = var_31.validate(var_35)
    var_37 = var_31.render_fields()
    var_38 = None
    var_39 = module_4.Form(env=var_11, schema=var_22, values=var_38)



# Parsed testcases at query #13
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %} id="{{ field_id }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} id="{{ field_id }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice.value }}">{{ choice.label }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'active'
    var_15 = 'status'
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = module_2.String(format=var_13)
    var_19 = module_2.Boolean()
    var_20 = 'Active'
    var_21 = (var_14, var_20)
    var_22 = 'inactive'
    var_23 = 'Inactive'
    var_24 = (var_22, var_23)
    var_25 = [var_21, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = 'john@example.com'
    var_31 = {var_12: var_29, var_13: var_30, var_14: var_10, var_15: var_14}
    var_32 = module_4.Form(env=var_11, schema=var_28, values=var_31)
    var_33 = {var_12: var_29, var_13: var_30, var_14: var_10, var_15: var_14}
    var_34 = var_32.validate(var_33)
    var_35 = var_32.render_fields()
    var_36 = {}
    var_37 = module_4.Form(env=var_11, schema=var_28, values=var_36)
    var_38 = ''
    var_39 = 'invalid'
    var_40 = None
    var_41 = {var_12: var_38, var_13: var_39, var_14: var_40, var_15: var_39}
    var_42 = var_37.validate(var_41)
    var_43 = var_37.render_fields()
    var_44 = 'id'
    var_45 = module_2.String()
    var_46 = module_2.String(max_length=var_16)
    var_47 = {var_44: var_45, var_12: var_46}
    var_48 = module_3.Schema(var_47)
    var_49 = '123'
    var_50 = {var_44: var_49, var_12: var_29}
    var_51 = module_4.Form(env=var_11, schema=var_48, values=var_50)
    var_52 = {var_12: var_29}
    var_53 = var_51.validate(var_52)
    var_54 = var_51.render_fields()



# Parsed testcases at query #14
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_0, package=var_2)
    var_4 = var_3.loader
    var_5 = 'typesystem'
    var_6 = module_0.Jinja2Forms(package=var_5)
    var_7 = var_6.load_template_env(directory=var_2, package=var_5)
    var_8 = var_7.loader
    var_9 = module_0.Jinja2Forms(directory=var_0, package=var_5)
    var_10 = var_9.load_template_env(directory=var_0, package=var_5)
    var_11 = var_10.loader
    var_12 = var_10.loader.loaders
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 0
    var_15 = var_10.loader.loaders[var_14]
    var_16 = 1
    var_17 = var_10.loader.loaders[var_16]



# Parsed testcases at query #15
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = var_1.env
    var_5 = module_0.Jinja2Forms()
    var_6 = 'jinja2'
    var_7 = '.'
    var_8 = module_0.Jinja2Forms(directory=var_7)
    var_9 = 'jinja2'
    var_10 = None



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'Test Field'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = module_0.String()
    var_7 = 'test_value'
    var_8 = 'forms/input.html'
    var_9 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_0.String()
    var_6 = 'value'
    var_7 = 'This field is required'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_0.String(format=var_0)
    var_6 = 'secret123'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = module_0.String()
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = {var_0: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = None
    var_8 = [var_1, var_2]
    var_9 = module_0.Choice(choices=var_8)
    var_10 = 'forms/select.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'bool_field'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_0.Boolean()
    var_6 = True
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_0.String(format=var_0)
    var_6 = 'multiline text'
    var_7 = 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field_name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_0.String()
    var_6 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_0.String(format=var_0)
    var_6 = 'test@example.com'
    var_7 = 0



# Parsed testcases at query #17
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = var_9.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_9.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_9.loader.loaders[var_15]



# Parsed testcases at query #18
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = var_9.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_9.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_9.loader.loaders[var_15]



# Parsed testcases at query #19
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}></select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'username'
    var_12 = 'email'
    var_13 = 'bio'
    var_14 = 'is_active'
    var_15 = 'role'
    var_16 = 'age'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = module_2.String(format=var_12)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'admin'
    var_24 = 'Admin'
    var_25 = (var_23, var_24)
    var_26 = 'user'
    var_27 = 'User'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = 'number'
    var_32 = module_2.Field()
    var_33 = {var_11: var_18, var_12: var_19, var_13: var_21, var_14: var_22, var_15: var_30, var_16: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = None
    var_36 = module_4.Form(env=var_10, schema=var_34, values=var_35)
    var_37 = var_34.fields[var_11]
    var_38 = 'john_doe'
    var_39 = var_36.render_field(field_name=var_11, field=var_37, value=var_38, error=var_35)
    var_40 = var_34.fields[var_12]
    var_41 = 'test@example.com'
    var_42 = var_36.render_field(field_name=var_12, field=var_40, value=var_41, error=var_35)
    var_43 = var_34.fields[var_11]
    var_44 = 'john'
    var_45 = 'This field is required'
    var_46 = var_36.render_field(field_name=var_11, field=var_43, value=var_44, error=var_45)
    var_47 = var_34.fields[var_13]
    var_48 = 'Some bio text'
    var_49 = var_36.render_field(field_name=var_13, field=var_47, value=var_48, error=var_35)
    var_50 = var_34.fields[var_14]
    var_51 = True
    var_52 = var_36.render_field(field_name=var_14, field=var_50, value=var_51, error=var_35)
    var_53 = var_34.fields[var_15]
    var_54 = var_36.render_field(field_name=var_15, field=var_53, value=var_23, error=var_35)
    var_55 = 'user_name'
    var_56 = module_2.String()
    var_57 = var_36.render_field(field_name=var_55, field=var_56, value=var_35, error=var_35)
    var_58 = module_2.String()
    var_59 = 'optional_field'
    var_60 = var_36.render_field(field_name=var_59, field=var_58, value=var_35, error=var_35)
    var_61 = 'password'
    var_62 = module_2.String(format=var_61)
    var_63 = 'secret123'
    var_64 = var_36.render_field(field_name=var_61, field=var_62, value=var_63, error=var_35)



# Parsed testcases at query #20
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = var_1.env
    var_5 = module_0.Jinja2Forms()
    var_6 = 'jinja2'
    var_7 = globals()[var_6]
    var_8 = '.'
    var_9 = module_0.Jinja2Forms(directory=var_8)



# Parsed testcases at query #21
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = var_1.env
    var_5 = module_0.Jinja2Forms()
    var_6 = 'jinja2'
    var_7 = globals()[var_6]
    var_8 = '.'
    var_9 = module_0.Jinja2Forms(directory=var_8)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'active'
    var_3 = 'readonly_field'
    var_4 = 100
    var_5 = module_0.String(max_length=var_4)
    var_6 = module_0.Integer()
    var_7 = module_0.Boolean()
    var_8 = True
    var_9 = module_0.String()
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = 'forms/input.html'
    var_13 = 'forms/checkbox.html'
    var_14 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<p class="error">{{ error }}</p>{% endif %}'
    var_15 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<p class="error">{{ error }}</p>{% endif %}'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_2.DictLoader(var_16)
    var_18 = module_3.Environment(autoescape=var_8, loader=var_17)
    var_19 = 'John'
    var_20 = 30
    var_21 = {var_0: var_19, var_1: var_20, var_2: var_8}
    var_22 = module_4.Form(env=var_18, schema=var_11, values=var_21)
    var_23 = var_22.render_fields()
    var_24 = None
    var_25 = module_4.Form(env=var_18, schema=var_11, values=var_24)
    var_26 = ''
    var_27 = 'invalid'
    var_28 = False
    var_29 = {var_0: var_26, var_1: var_27, var_2: var_28}
    var_30 = var_25.validate(var_29)
    var_31 = var_25.render_fields()
    var_32 = module_4.Form(env=var_18, schema=var_11, values=var_24)
    var_33 = 'Jane'
    var_34 = 25
    var_35 = {var_0: var_33, var_1: var_34, var_2: var_8}
    var_36 = var_32.validate(var_35)
    var_37 = var_32.render_fields()
    var_38 = module_4.Form(env=var_18, schema=var_11, values=var_24)
    var_39 = var_38.render_fields()



# Parsed testcases at query #23
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'Test Form.render_fields() method.'
    var_1 = module_0.Environment()
    var_2 = 'forms/input.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = 'forms/textarea.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_7 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_8 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_9 = '<textarea name="{{ field_name }}">{{ value or "" }}</textarea>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'active'
    var_14 = 'read_only_field'
    var_15 = 100
    var_16 = module_1.String(max_length=var_15)
    var_17 = module_1.String(format=var_12)
    var_18 = module_1.Boolean()
    var_19 = True
    var_20 = module_1.String()
    var_21 = {var_11: var_16, var_12: var_17, var_13: var_18, var_14: var_20}
    var_22 = module_2.Schema(var_21)
    var_23 = 'John'
    var_24 = 'john@example.com'
    var_25 = {var_11: var_23, var_12: var_24, var_13: var_19}
    var_26 = module_3.Form(env=var_1, schema=var_22, values=var_25)
    var_27 = {var_11: var_23, var_12: var_24, var_13: var_19}
    var_28 = var_26.validate(var_27)
    var_29 = var_26.render_fields()

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'Test Form.render_fields() with validation errors.'
    var_1 = module_0.Environment()
    var_2 = 'forms/input.html'
    var_3 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_4 = {var_2: var_3}
    var_5 = 'name'
    var_6 = 5
    var_7 = module_1.String(max_length=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.Form(env=var_1, schema=var_9)
    var_11 = 'VeryLongNameThatExceedsLimit'
    var_12 = {var_5: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = var_10.render_fields()
    var_15 = 'error'

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'Test Form.render_fields() with no values.'
    var_1 = module_0.Environment()
    var_2 = 'forms/input.html'
    var_3 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value or "" }}"{% if required %} required{% endif %}>'
    var_4 = {var_2: var_3}
    var_5 = 'name'
    var_6 = 100
    var_7 = module_1.String(max_length=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = None
    var_11 = module_3.Form(env=var_1, schema=var_9, values=var_10)
    var_12 = var_11.validate(var_10)
    var_13 = var_11.render_fields()

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'Test Form.render_fields() with Choice field.'
    var_1 = module_0.Environment()
    var_2 = 'forms/select.html'
    var_3 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_4 = {var_2: var_3}
    var_5 = 'status'
    var_6 = 'active'
    var_7 = 'inactive'
    var_8 = [var_6, var_7]
    var_9 = module_1.Choice(choices=var_8)
    var_10 = {var_5: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_5: var_6}
    var_13 = module_3.Form(env=var_1, schema=var_11, values=var_12)
    var_14 = {var_5: var_6}
    var_15 = var_13.validate(var_14)
    var_16 = var_13.render_fields()

import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'Test Form.render_fields() with Boolean field.'
    var_1 = module_0.Environment()
    var_2 = 'forms/checkbox.html'
    var_3 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_4 = {var_2: var_3}
    var_5 = 'agree'
    var_6 = module_1.Boolean()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = True
    var_10 = {var_5: var_9}
    var_11 = module_3.Form(env=var_1, schema=var_8, values=var_10)
    var_12 = {var_5: var_9}
    var_13 = var_11.validate(var_12)
    var_14 = var_11.render_fields()



# Parsed testcases at query #24
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field() method renders field correctly.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = True
    var_12 = module_1.Environment(autoescape=var_11, loader=var_10)
    var_13 = 'username'
    var_14 = 'email'
    var_15 = 'is_active'
    var_16 = 'bio'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = module_2.String(format=var_14)
    var_20 = module_2.Boolean()
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = None
    var_26 = module_4.Form(env=var_12, schema=var_24, values=var_25)
    var_27 = var_24.fields[var_13]
    var_28 = 'john_doe'
    var_29 = var_26.render_field(field_name=var_13, field=var_27, value=var_28, error=var_25)
    var_30 = var_24.fields[var_14]
    var_31 = 'john@example.com'
    var_32 = var_26.render_field(field_name=var_14, field=var_30, value=var_31, error=var_25)
    var_33 = 'john'
    var_34 = 'Username too short'
    var_35 = var_26.render_field(field_name=var_13, field=var_27, value=var_33, error=var_34)
    var_36 = var_24.fields[var_15]
    var_37 = var_26.render_field(field_name=var_15, field=var_36, value=var_11, error=var_25)
    var_38 = var_24.fields[var_16]
    var_39 = 'My bio'
    var_40 = var_26.render_field(field_name=var_16, field=var_38, value=var_39, error=var_25)
    var_41 = 'first_name'
    var_42 = module_2.String()
    var_43 = 'John'
    var_44 = var_26.render_field(field_name=var_41, field=var_42, value=var_43, error=var_25)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Form.render_field method renders field with correct context.'
    var_1 = 'name'
    var_2 = 'John Doe'
    var_3 = None
    var_4 = ''
    var_5 = 'This field is required'
    var_6 = 'active'
    var_7 = True
    var_8 = 'choice_field'
    var_9 = 'option1'
    var_10 = 'age'
    var_11 = 'password'
    var_12 = module_0.String(format=var_11)
    var_13 = 'secret123'



# Parsed testcases at query #26
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'active'
    var_14 = 'readonly_field'
    var_15 = 100
    var_16 = module_2.String(max_length=var_15)
    var_17 = module_2.String(format=var_12)
    var_18 = module_2.Boolean()
    var_19 = True
    var_20 = module_2.String()
    var_21 = {var_11: var_16, var_12: var_17, var_13: var_18, var_14: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = 'John'
    var_24 = 'john@example.com'
    var_25 = 'hidden'
    var_26 = {var_11: var_23, var_12: var_24, var_13: var_19, var_14: var_25}
    var_27 = module_4.Form(env=var_10, schema=var_22, values=var_26)
    var_28 = {var_11: var_23, var_12: var_24, var_13: var_19}
    var_29 = var_27.validate(var_28)
    var_30 = var_27.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'email'
    var_12 = module_2.String(format=var_11)
    var_13 = {var_11: var_12}
    var_14 = module_3.Schema(var_13)
    var_15 = module_4.Form(env=var_10, schema=var_14)
    var_16 = 'invalid-email'
    var_17 = {var_11: var_16}
    var_18 = var_15.validate(var_17)
    var_19 = var_15.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_6 = '<select name="{{ field_name }}"></select>'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 100
    var_13 = module_2.String(max_length=var_12)
    var_14 = {var_11: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = None
    var_17 = module_4.Form(env=var_10, schema=var_15, values=var_16)
    var_18 = {}
    var_19 = var_17.validate(var_18)
    var_20 = var_17.render_fields()



# Parsed testcases at query #27
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'password'
    var_15 = 'agree'
    var_16 = 'choice_field'
    var_17 = 'description'
    var_18 = 'Name'
    var_19 = module_2.String()
    var_20 = module_2.String(format=var_13)
    var_21 = module_2.String(format=var_14)
    var_22 = module_2.Boolean()
    var_23 = 'option1'
    var_24 = 'option2'
    var_25 = [var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = 'text'
    var_28 = module_2.String(format=var_27)
    var_29 = {var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_22, var_16: var_26, var_17: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = {}
    var_32 = module_4.Form(env=var_11, schema=var_30, values=var_31)
    var_33 = var_30.fields[var_12]
    var_34 = 'John'
    var_35 = var_32.render_field(field_name=var_12, field=var_33, value=var_34)
    var_36 = var_30.fields[var_13]
    var_37 = 'test@example.com'
    var_38 = var_32.render_field(field_name=var_13, field=var_36, value=var_37)
    var_39 = var_30.fields[var_14]
    var_40 = 'secret'
    var_41 = var_32.render_field(field_name=var_14, field=var_39, value=var_40)
    var_42 = var_30.fields[var_15]
    var_43 = var_32.render_field(field_name=var_15, field=var_42, value=var_10)
    var_44 = var_30.fields[var_16]
    var_45 = var_32.render_field(field_name=var_16, field=var_44)
    var_46 = var_30.fields[var_17]
    var_47 = 'Some text'
    var_48 = var_32.render_field(field_name=var_17, field=var_46, value=var_47)
    var_49 = var_30.fields[var_12]
    var_50 = 'Invalid name'
    var_51 = var_32.render_field(field_name=var_12, field=var_49, value=var_34, error=var_50)
    var_52 = 'user_name'
    var_53 = module_2.String()
    var_54 = {var_52: var_53}
    var_55 = module_3.Schema(var_54)
    var_56 = {}
    var_57 = module_4.Form(env=var_11, schema=var_55, values=var_56)
    var_58 = var_55.fields[var_52]
    var_59 = var_57.render_field(field_name=var_52, field=var_58)
    var_60 = module_2.String()
    var_61 = 'optional'
    var_62 = None
    var_63 = var_32.render_field(field_name=var_61, field=var_60, value=var_62)



# Parsed testcases at query #28
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field method renders field with correct context.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/select.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = 'email'
    var_13 = module_2.String(format=var_12)
    var_14 = {var_12: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = {}
    var_17 = module_4.Form(env=var_11, schema=var_15, values=var_16)
    var_18 = var_15.fields[var_12]
    var_19 = 'test@example.com'
    var_20 = None
    var_21 = var_17.render_field(field_name=var_12, field=var_18, value=var_19, error=var_20)
    var_22 = var_15.fields[var_12]
    var_23 = 'invalid'
    var_24 = 'Invalid email format'
    var_25 = var_17.render_field(field_name=var_12, field=var_22, value=var_23, error=var_24)
    var_26 = 'password'
    var_27 = module_2.String(format=var_26)
    var_28 = {var_26: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = {}
    var_31 = module_4.Form(env=var_11, schema=var_29, values=var_30)
    var_32 = var_29.fields[var_26]
    var_33 = 'secret123'
    var_34 = var_31.render_field(field_name=var_26, field=var_32, value=var_33, error=var_20)
    var_35 = 'optional_field'
    var_36 = True
    var_37 = module_2.String()
    var_38 = {var_35: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = {}
    var_41 = module_4.Form(env=var_11, schema=var_39, values=var_40)
    var_42 = var_39.fields[var_35]
    var_43 = var_41.render_field(field_name=var_35, field=var_42, value=var_20, error=var_20)
    var_44 = 'user_name'
    var_45 = module_2.String()
    var_46 = {var_44: var_45}
    var_47 = module_3.Schema(var_46)
    var_48 = {}
    var_49 = module_4.Form(env=var_11, schema=var_47, values=var_48)
    var_50 = var_47.fields[var_44]
    var_51 = 'john'
    var_52 = var_49.render_field(field_name=var_44, field=var_50, value=var_51, error=var_20)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'Test Field'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'test_value'
    var_8 = 'forms/input.html'
    var_9 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email_field'
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'invalid'
    var_8 = 'Invalid email'
    var_9 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 'secret123'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'status'
    var_1 = 'active'
    var_2 = 'inactive'
    var_3 = [var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = {var_0: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = None
    var_8 = var_6.fields[var_0]
    var_9 = 'forms/select.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'agree'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = True
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'description'
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 'Some text'
    var_8 = 'forms/textarea.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'user_name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.fields[var_0]
    var_6 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field_with_default'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.fields[var_0]
    var_7 = 0



# Parsed testcases at query #30
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = '/tmp/templates'
    var_12 = module_0.Jinja2Forms(directory=var_11)
    var_13 = var_12.load_template_env()
    var_14 = module_0.Jinja2Forms(directory=var_11)
    var_15 = var_14.load_template_env(directory=var_11)



# Parsed testcases at query #31
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>'
    var_5 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'is_active'
    var_14 = 'status'
    var_15 = 'bio'
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = module_2.String(format=var_12)
    var_19 = module_2.Boolean()
    var_20 = 'active'
    var_21 = 'Active'
    var_22 = (var_20, var_21)
    var_23 = 'inactive'
    var_24 = 'Inactive'
    var_25 = (var_23, var_24)
    var_26 = [var_22, var_25]
    var_27 = module_2.Choice(choices=var_26)
    var_28 = 'text'
    var_29 = module_2.String(format=var_28)
    var_30 = {var_11: var_17, var_12: var_18, var_13: var_19, var_14: var_27, var_15: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = 'John'
    var_33 = 'john@example.com'
    var_34 = True
    var_35 = 'Test bio'
    var_36 = {var_11: var_32, var_12: var_33, var_13: var_34, var_14: var_20, var_15: var_35}
    var_37 = module_4.Form(env=var_10, schema=var_31, values=var_36)
    var_38 = {var_11: var_32, var_12: var_33, var_13: var_34, var_14: var_20, var_15: var_35}
    var_39 = var_37.validate(var_38)
    var_40 = var_37.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}> {% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_5 = '<input type="checkbox" name="{{ field_name }}">'
    var_6 = '<select name="{{ field_name }}"></select>'
    var_7 = '<textarea name="{{ field_name }}"></textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 100
    var_14 = module_2.String(max_length=var_13)
    var_15 = module_2.String(format=var_12)
    var_16 = {var_11: var_14, var_12: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_4.Form(env=var_10, schema=var_17)
    var_19 = 'invalid-email'
    var_20 = {var_12: var_19}
    var_21 = var_18.validate(var_20)
    var_22 = var_18.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<input type="checkbox" name="{{ field_name }}">'
    var_6 = '<select name="{{ field_name }}"></select>'
    var_7 = '<textarea name="{{ field_name }}"></textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'description'
    var_13 = module_2.String()
    var_14 = True
    var_15 = module_2.String()
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = None
    var_19 = module_4.Form(env=var_10, schema=var_17, values=var_18)
    var_20 = {}
    var_21 = var_19.validate(var_20)
    var_22 = var_19.render_fields()

import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<input type="checkbox" name="{{ field_name }}">'
    var_6 = '<select name="{{ field_name }}"></select>'
    var_7 = '<textarea name="{{ field_name }}"></textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'id'
    var_13 = module_2.String()
    var_14 = True
    var_15 = module_2.String()
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = 'John'
    var_19 = '123'
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = module_4.Form(env=var_10, schema=var_17, values=var_20)
    var_22 = {var_11: var_18}
    var_23 = var_21.validate(var_22)
    var_24 = var_21.render_fields()



# Parsed testcases at query #32
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = 'username'
    var_3 = 'email'
    var_4 = 'password'
    var_5 = 'age'
    var_6 = 'bio'
    var_7 = 'is_active'
    var_8 = 'role'
    var_9 = 'website'
    var_10 = 100
    var_11 = module_1.String(max_length=var_10)
    var_12 = module_1.String(format=var_3)
    var_13 = module_1.String(format=var_4)
    var_14 = 'number'
    var_15 = module_1.String(format=var_14)
    var_16 = 'text'
    var_17 = module_1.String(format=var_16)
    var_18 = module_1.Boolean()
    var_19 = 'admin'
    var_20 = 'user'
    var_21 = [var_19, var_20]
    var_22 = module_1.Choice(choices=var_21)
    var_23 = 'url'
    var_24 = module_1.String(format=var_23)
    var_25 = {var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_15, var_6: var_17, var_7: var_18, var_8: var_22, var_9: var_24}
    var_26 = module_2.Schema(var_25)
    var_27 = var_26.fields[var_2]
    var_28 = 'john_doe'
    var_29 = None
    var_30 = var_26.fields[var_3]
    var_31 = 'john@example.com'
    var_32 = var_26.fields[var_4]
    var_33 = 'secret'
    var_34 = var_26.fields[var_2]
    var_35 = 'john'
    var_36 = 'This field is required'
    var_37 = var_26.fields[var_5]
    var_38 = '25'
    var_39 = var_26.fields[var_6]
    var_40 = 'My bio'
    var_41 = var_26.fields[var_7]
    var_42 = True
    var_43 = var_26.fields[var_8]
    var_44 = 'user_name'
    var_45 = var_26.fields[var_2]
    var_46 = var_26.fields[var_2]
    var_47 = var_26.fields[var_2]
    var_48 = 'required'



# Parsed testcases at query #33
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = None
    var_3 = var_1.load_template_env(directory=var_0, package=var_2)
    var_4 = var_3.loader
    var_5 = 'typesystem'
    var_6 = module_0.Jinja2Forms(package=var_5)
    var_7 = var_6.load_template_env(directory=var_2, package=var_5)
    var_8 = var_7.loader
    var_9 = module_0.Jinja2Forms(directory=var_0, package=var_5)
    var_10 = var_9.load_template_env(directory=var_0, package=var_5)
    var_11 = var_10.loader
    var_12 = var_10.loader.loaders
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 0
    var_15 = var_10.loader.loaders[var_14]
    var_16 = 1
    var_17 = var_10.loader.loaders[var_16]



# Parsed testcases at query #34
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field method with various field types.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}" id="{{ field_id }}"{% if required %} required{% endif %}>{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = True
    var_12 = module_1.Environment(autoescape=var_11, loader=var_10)
    var_13 = 'test_field'
    var_14 = module_2.String()
    var_15 = {var_13: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = None
    var_18 = module_4.Form(env=var_12, schema=var_16, values=var_17)
    var_19 = 'Test Label'
    var_20 = module_2.String()
    var_21 = 'test_value'
    var_22 = var_18.render_field(field_name=var_13, field=var_20, value=var_21)
    var_23 = 'This is an error'
    var_24 = var_18.render_field(field_name=var_13, field=var_20, value=var_21, error=var_23)
    var_25 = 'password'
    var_26 = module_2.String(format=var_25)
    var_27 = 'password_field'
    var_28 = 'secret'
    var_29 = var_18.render_field(field_name=var_27, field=var_26, value=var_28)
    var_30 = 'input_type'
    var_31 = var_18.input_type_for_field(var_26)
    var_32 = var_30 in var_31
    var_33 = module_2.String()
    var_34 = 'optional_field'
    var_35 = var_18.render_field(field_name=var_34, field=var_33, value=var_17)
    var_36 = 'default_value'
    var_37 = module_2.String()
    var_38 = 'field_with_default'
    var_39 = var_18.render_field(field_name=var_38, field=var_37, value=var_36)
    var_40 = module_2.Boolean()
    var_41 = 'bool_field'
    var_42 = var_18.render_field(field_name=var_41, field=var_40)
    var_43 = 'option1'
    var_44 = 'option2'
    var_45 = [var_43, var_44]
    var_46 = module_2.Choice(choices=var_45)
    var_47 = 'choice_field'
    var_48 = var_18.render_field(field_name=var_47, field=var_46)
    var_49 = 'test_field_name'
    var_50 = var_18.render_field(field_name=var_49, field=var_20)
    var_51 = 'Custom Title'
    var_52 = module_2.String()
    var_53 = 'custom_field'
    var_54 = var_18.render_field(field_name=var_53, field=var_52)
    var_55 = False
    var_56 = module_2.String()
    var_57 = 'readonly_field'
    var_58 = var_18.render_field(field_name=var_57, field=var_56)



# Parsed testcases at query #35
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = module_0.Jinja2Forms()
    var_3 = isinstance(var_2, var_1)
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = isinstance(var_6, var_3)
    var_8 = 'typesystem'
    var_9 = var_5.env
    var_10 = isinstance(var_9, var_3)
    var_11 = var_5.env.loader



# Parsed testcases at query #36
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader
    var_11 = module_0.Jinja2Forms(directory=var_0)
    var_12 = var_11.load_template_env(directory=var_0)
    var_13 = module_0.Jinja2Forms(directory=var_0)
    var_14 = None
    var_15 = var_13.load_template_env(directory=var_14, package=var_14)
    var_16 = module_0.Jinja2Forms(directory=var_14)
    var_17 = None
    var_18 = var_16.load_template_env(directory=var_14, package=var_17)
    var_19 = var_18.loader
    var_20 = module_0.Jinja2Forms(package=var_4)
    var_21 = var_20.load_template_env(directory=var_17, package=var_4)
    var_22 = var_21.loader



# Parsed testcases at query #37
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = module_0.Jinja2Forms(directory=var_8)



# Parsed testcases at query #38
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_field method with various field types.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value or "" }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_8 = '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value or "" }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'password'
    var_15 = 'agree'
    var_16 = 'choice_field'
    var_17 = 'message'
    var_18 = 100
    var_19 = module_2.String(max_length=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = module_2.String(format=var_14)
    var_22 = module_2.Boolean()
    var_23 = 'a'
    var_24 = 'Option A'
    var_25 = (var_23, var_24)
    var_26 = 'b'
    var_27 = 'Option B'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = 'text'
    var_32 = module_2.String(format=var_31)
    var_33 = {var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_22, var_16: var_30, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = {}
    var_36 = module_4.Form(env=var_11, schema=var_34, values=var_35)
    var_37 = var_34.fields[var_12]
    var_38 = 'John'
    var_39 = None
    var_40 = var_36.render_field(field_name=var_12, field=var_37, value=var_38, error=var_39)
    var_41 = var_34.fields[var_13]
    var_42 = 'test@example.com'
    var_43 = var_36.render_field(field_name=var_13, field=var_41, value=var_42, error=var_39)
    var_44 = var_34.fields[var_14]
    var_45 = 'secret'
    var_46 = var_36.render_field(field_name=var_14, field=var_44, value=var_45, error=var_39)
    var_47 = var_34.fields[var_15]
    var_48 = True
    var_49 = var_36.render_field(field_name=var_15, field=var_47, value=var_48, error=var_39)
    var_50 = var_34.fields[var_16]
    var_51 = var_36.render_field(field_name=var_16, field=var_50, value=var_23, error=var_39)
    var_52 = var_34.fields[var_17]
    var_53 = 'Hello World'
    var_54 = var_36.render_field(field_name=var_17, field=var_52, value=var_53, error=var_39)
    var_55 = var_34.fields[var_12]
    var_56 = 'This field is required'
    var_57 = var_36.render_field(field_name=var_12, field=var_55, value=var_38, error=var_56)
    var_58 = module_2.String()
    var_59 = 'user_name'
    var_60 = 'test'
    var_61 = var_36.render_field(field_name=var_59, field=var_58, value=var_60, error=var_39)



# Parsed testcases at query #39
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Test Form.render_field() method.'
    var_1 = 'typesystem'
    var_2 = module_0.Jinja2Forms(package=var_1)
    var_3 = module_1.Boolean()
    var_4 = 'name'
    var_5 = 'John Doe'
    var_6 = None
    var_7 = 'email'
    var_8 = 'test@example.com'
    var_9 = 'age'
    var_10 = 25
    var_11 = 'password'
    var_12 = 'secret123'
    var_13 = 'bio'
    var_14 = 'My biography'
    var_15 = 'subscribe'
    var_16 = True
    var_17 = 'country'
    var_18 = 'US'
    var_19 = 'invalid'
    var_20 = 'Invalid email format'
    var_21 = module_1.String()
    var_22 = 'first_name'
    var_23 = 'John'
    var_24 = module_1.String()
    var_25 = 'id'



# Parsed testcases at query #40
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #41
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = module_0.Jinja2Forms()
    var_3 = isinstance(var_2, var_1)
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = isinstance(var_6, var_3)
    var_8 = 'typesystem'
    var_9 = var_5.env
    var_10 = isinstance(var_9, var_3)



# Parsed testcases at query #42
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'Test Form.render_fields() method.'
    var_1 = 'forms/input.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/textarea.html'
    var_5 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}"{% if required %} required{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_6 = '<input type="checkbox" name="{{ field_name }}"{% if value %} checked{% endif %}>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_7 = '<select name="{{ field_name }}">{% if error %}<span class="error">{{ error }}</span>{% endif %}</select>'
    var_8 = '<textarea name="{{ field_name }}">{{ value }}</textarea>{% if error %}<span class="error">{{ error }}</span>{% endif %}'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.DictLoader(var_9)
    var_11 = True
    var_12 = module_1.Environment(autoescape=var_11, loader=var_10)
    var_13 = 'name'
    var_14 = 'email'
    var_15 = 'active'
    var_16 = 'bio'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = module_2.String(format=var_14)
    var_20 = module_2.Boolean()
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = 'John'
    var_26 = 'john@example.com'
    var_27 = 'Test'
    var_28 = {var_13: var_25, var_14: var_26, var_15: var_11, var_16: var_27}
    var_29 = module_4.Form(env=var_12, schema=var_24, values=var_28)
    var_30 = var_29.render_fields()
    var_31 = {var_13: var_25, var_14: var_26, var_15: var_11, var_16: var_27}
    var_32 = module_4.Form(env=var_12, schema=var_24, values=var_31)
    var_33 = {var_13: var_25, var_14: var_26, var_15: var_11, var_16: var_27}
    var_34 = var_32.validate(var_33)
    var_35 = var_32.render_fields()
    var_36 = {}
    var_37 = module_4.Form(env=var_12, schema=var_24, values=var_36)
    var_38 = ''
    var_39 = 'invalid-email'
    var_40 = None
    var_41 = {var_13: var_38, var_14: var_39, var_15: var_40, var_16: var_38}
    var_42 = var_37.validate(var_41)
    var_43 = var_37.render_fields()
    var_44 = 'id'
    var_45 = module_2.String()
    var_46 = module_2.String()
    var_47 = {var_44: var_45, var_13: var_46}
    var_48 = module_3.Schema(var_47)
    var_49 = '123'
    var_50 = {var_44: var_49, var_13: var_25}
    var_51 = module_4.Form(env=var_12, schema=var_48, values=var_50)
    var_52 = {var_13: var_25}
    var_53 = var_51.validate(var_52)
    var_54 = var_51.render_fields()
    var_55 = 'Original'
    var_56 = {var_13: var_55}
    var_57 = module_4.Form(env=var_12, schema=var_24, values=var_56)
    var_58 = 'Modified'
    var_59 = 'test@example.com'
    var_60 = False
    var_61 = 'Invalid email'
    var_62 = var_57.render_fields()



# Parsed testcases at query #43
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = module_0.Jinja2Forms()
    var_3 = isinstance(var_2, var_1)
    var_4 = 'typesystem'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = isinstance(var_6, var_3)
    var_8 = 'typesystem'
    var_9 = var_5.env
    var_10 = isinstance(var_9, var_3)



# Parsed testcases at query #44
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env
    var_3 = 'typesystem'
    var_4 = var_1.env
    var_5 = module_0.Jinja2Forms()
    var_6 = 'jinja2'
    var_7 = globals()[var_6]
    var_8 = '.'
    var_9 = module_0.Jinja2Forms(directory=var_8)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'Test Field'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = module_0.String()
    var_7 = 'test_value'
    var_8 = None
    var_9 = 'forms/input.html'
    var_10 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test_field_name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = module_0.String()
    var_6 = None
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'password'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = module_0.String(format=var_0)
    var_6 = 'secret123'
    var_7 = None
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = module_0.String(format=var_0)
    var_6 = 'invalid'
    var_7 = 'Invalid email'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'status'
    var_1 = 'active'
    var_2 = 'inactive'
    var_3 = [var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = {var_0: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = {}
    var_8 = [var_1, var_2]
    var_9 = module_0.Choice(choices=var_8)
    var_10 = 'forms/select.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'agree'
    var_1 = module_0.Boolean()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = module_0.Boolean()
    var_6 = True
    var_7 = 'forms/checkbox.html'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'optional'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = module_0.String()
    var_7 = 'value'
    var_8 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'nullable'
    var_1 = True
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = module_0.String()
    var_7 = None
    var_8 = 0



