####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.Form(env=var_1, schema=var_3)
    var_5 = var_4.input_type_for_field(var_0)
    assert var_5 == 'text'
    var_6 = 'email'
    var_7 = module_0.String(format=var_6)
    var_8 = var_4.input_type_for_field(var_7)
    assert var_8 == 'email'
    var_9 = 'unknown'
    var_10 = module_0.String(format=var_9)
    var_11 = var_4.input_type_for_field(var_10)
    assert var_11 == 'text'
    var_12 = 'color'
    var_13 = 'datetime'
    var_14 = 'date'
    var_15 = 'hidden'
    var_16 = 'month'
    var_17 = 'number'
    var_18 = 'password'
    var_19 = 'range'
    var_20 = 'search'
    var_21 = 'tel'
    var_22 = 'text'
    var_23 = 'time'
    var_24 = 'url'
    var_25 = 'week'
    var_26 = 'datetime-local'
    var_27 = {var_12: var_12, var_13: var_26, var_14: var_14, var_6: var_6, var_15: var_15, var_16: var_16, var_17: var_17, var_18: var_18, var_19: var_19, var_20: var_20, var_21: var_21, var_22: var_22, var_23: var_23, var_24: var_24, var_25: var_25}
    var_28 = var_4.input_type_for_field(var_10)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = module_0.String(format=var_2)
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = {}
    var_9 = module_2.DictLoader(var_8)
    var_10 = module_3.Environment(loader=var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = 'john@example.com'
    var_14 = {var_0: var_11, var_1: var_12, var_2: var_13}
    var_15 = module_4.Form(env=var_10, schema=var_7, values=var_14)
    var_16 = 'Jane'
    var_17 = 25
    var_18 = 'jane@example.com'
    var_19 = {var_0: var_16, var_1: var_17, var_2: var_18}
    var_20 = var_15.validate(var_19)
    var_21 = ''
    var_22 = 'invalid'
    var_23 = 'not-an-email'
    var_24 = {var_0: var_21, var_1: var_22, var_2: var_23}
    var_25 = var_15.validate(var_24)
    var_26 = 'Valid'
    var_27 = 20
    var_28 = 'valid@example.com'
    var_29 = {var_0: var_26, var_1: var_27, var_2: var_28}
    var_30 = var_15.validate(var_29)
    var_31 = 'name'
    var_32 = 'age'
    var_33 = 'Should'
    var_34 = 'fail'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = var_15.validate(var_35)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.environment as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'choice_field'
    var_1 = 'boolean_field'
    var_2 = 'string_text_field'
    var_3 = 'string_default_field'
    var_4 = 'object_field'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = module_0.Choice(choices=var_7)
    var_9 = module_0.Boolean()
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = module_0.String()
    var_13 = 'nested'
    var_14 = module_0.String()
    var_15 = {var_13: var_14}
    var_16 = module_0.Object()
    var_17 = {var_0: var_8, var_1: var_9, var_2: var_11, var_3: var_12, var_4: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = module_2.Environment()
    var_20 = module_3.Form(env=var_19, schema=var_18)
    var_21 = var_18.fields[var_0]
    var_22 = var_20.template_for_field(var_21)
    assert var_22 == 'forms/select.html'
    var_23 = var_18.fields[var_1]
    var_24 = var_20.template_for_field(var_23)
    assert var_24 == 'forms/checkbox.html'
    var_25 = var_18.fields[var_2]
    var_26 = var_20.template_for_field(var_25)
    assert var_26 == 'forms/textarea.html'
    var_27 = var_18.fields[var_3]
    var_28 = var_20.template_for_field(var_27)
    assert var_28 == 'forms/input.html'
    var_29 = 'object_field'
    var_30 = var_18.fields[var_29]
    var_31 = var_20.template_for_field(var_30)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.environment as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.Environment()
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_3.Form(env=var_6, schema=var_5, values=var_9)
    var_11 = 'Jane'
    var_12 = 25
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = 'invalid'
    var_16 = {var_0: var_11, var_1: var_15}
    var_17 = var_10.validate(var_16)
    var_18 = 'name'
    var_19 = 'age'
    var_20 = 'Jane'
    var_21 = 25
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = var_10.validate(var_22)



# Parsed testcases at query #5
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'active'
    var_15 = 'gender'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'M'
    var_23 = 'F'
    var_24 = [var_22, var_23]
    var_25 = module_2.Choice(choices=var_24)
    var_26 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_21, var_15: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = 'John'
    var_29 = '30'
    var_30 = 'Test'
    var_31 = True
    var_32 = {var_11: var_28, var_12: var_29, var_13: var_30, var_14: var_31, var_15: var_22}
    var_33 = module_4.Form(env=var_10, schema=var_27, values=var_32)
    var_34 = var_33.render_fields()
    var_35 = ''
    var_36 = 'invalid'
    var_37 = 'not_bool'
    var_38 = {var_11: var_35, var_12: var_36, var_14: var_37}
    var_39 = var_33.validate(var_38)
    var_40 = var_33.render_fields()



# Parsed testcases at query #6
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'bio'
    var_14 = 'active'
    var_15 = module_2.String()
    var_16 = 'number'
    var_17 = module_2.String(format=var_16)
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = module_2.Boolean()
    var_21 = {var_11: var_15, var_12: var_17, var_13: var_19, var_14: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = 'John'
    var_24 = '30'
    var_25 = 'Developer'
    var_26 = True
    var_27 = {var_11: var_23, var_12: var_24, var_13: var_25, var_14: var_26}
    var_28 = module_4.Form(env=var_10, schema=var_22, values=var_27)
    var_29 = str(var_28)



# Parsed testcases at query #7
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #8
#--------------------------


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
    var_5 = module_2.String()
    var_6 = module_2.Integer()
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = module_4.Form(env=var_2, schema=var_8, values=var_11)
    var_13 = '<div>Mocked HTML</div>'
    var_14 = str(var_12)
    assert var_14 == '<div>Mocked HTML</div>'



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_3 = var_1.from_string(var_2)
    var_4 = 'name'
    var_5 = module_2.String()
    var_6 = {var_4: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = 'test'
    var_9 = {var_4: var_8}
    var_10 = module_4.Form(env=var_1, schema=var_7, values=var_9)
    var_11 = str(var_10)



# Parsed testcases at query #10
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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select name="{{ field_name }}"></select>'
    var_7 = '<input type="checkbox" name="{{ field_name }}">'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = module_2.String()
    var_14 = 'number'
    var_15 = module_2.String(format=var_14)
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = 'John'
    var_19 = '30'
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = module_4.Form(env=var_10, schema=var_17, values=var_20)
    var_22 = str(var_21)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'description'
    var_3 = 'agree'
    var_4 = 'gender'
    var_5 = 'read_only_field'
    var_6 = 'Name'
    var_7 = 100
    var_8 = module_0.String(max_length=var_7)
    var_9 = 'Age'
    var_10 = 'number'
    var_11 = module_0.String(format=var_10)
    var_12 = 'Description'
    var_13 = 'text'
    var_14 = module_0.String(format=var_13)
    var_15 = 'Agree'
    var_16 = module_0.Boolean()
    var_17 = 'Gender'
    var_18 = 'M'
    var_19 = 'Male'
    var_20 = (var_18, var_19)
    var_21 = 'F'
    var_22 = 'Female'
    var_23 = (var_21, var_22)
    var_24 = [var_20, var_23]
    var_25 = module_0.Choice(choices=var_24)
    var_26 = 'Read Only'
    var_27 = True
    var_28 = module_0.String()
    var_29 = {var_0: var_8, var_1: var_11, var_2: var_14, var_3: var_16, var_4: var_25, var_5: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 'forms/input.html'
    var_32 = 'forms/textarea.html'
    var_33 = 'forms/select.html'
    var_34 = 'forms/checkbox.html'
    var_35 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_36 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_37 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice[0] }}">{{ choice[1] }}</option>{% endfor %}</select>'
    var_38 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_39 = {var_31: var_35, var_32: var_36, var_33: var_37, var_34: var_38}
    var_40 = module_2.DictLoader(var_39)
    var_41 = module_3.Environment(loader=var_40)
    var_42 = 'John'
    var_43 = '30'
    var_44 = 'Test'
    var_45 = {var_0: var_42, var_1: var_43, var_2: var_44, var_3: var_27, var_4: var_18}
    var_46 = module_4.Form(env=var_41, schema=var_30, values=var_45)
    var_47 = var_46.render_fields()
    var_48 = ''
    var_49 = 'invalid'
    var_50 = 'ok'
    var_51 = False
    var_52 = 'X'
    var_53 = {var_0: var_48, var_1: var_49, var_2: var_50, var_3: var_51, var_4: var_52}
    var_54 = var_46.validate(var_53)
    var_55 = var_46.render_fields()



# Parsed testcases at query #12
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
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
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'email'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'country'
    var_17 = 'text'
    var_18 = module_2.String(format=var_17)
    var_19 = 'number'
    var_20 = module_2.String(format=var_19)
    var_21 = module_2.String(format=var_13)
    var_22 = module_2.String(format=var_17)
    var_23 = module_2.Boolean()
    var_24 = 'US'
    var_25 = 'UK'
    var_26 = 'CA'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = {var_11: var_18, var_12: var_20, var_13: var_21, var_14: var_22, var_15: var_23, var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'john@example.com'
    var_34 = 'Developer'
    var_35 = True
    var_36 = {var_11: var_31, var_12: var_32, var_13: var_33, var_14: var_34, var_15: var_35, var_16: var_24}
    var_37 = module_4.Form(env=var_10, schema=var_30, values=var_36)
    var_38 = var_37.render_fields()
    var_39 = ''
    var_40 = 'invalid'
    var_41 = False
    var_42 = 'IN'
    var_43 = {var_11: var_39, var_12: var_40, var_13: var_40, var_14: var_39, var_15: var_41, var_16: var_42}
    var_44 = var_37.validate(var_43)
    var_45 = var_37.render_fields()



# Parsed testcases at query #14
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'password'
    var_23 = module_2.String(format=var_22)
    var_24 = 'number'
    var_25 = module_2.String(format=var_24)
    var_26 = 'a'
    var_27 = 'A'
    var_28 = (var_26, var_27)
    var_29 = 'b'
    var_30 = 'B'
    var_31 = (var_29, var_30)
    var_32 = [var_28, var_31]
    var_33 = module_2.Choice(choices=var_32)
    var_34 = module_2.Boolean()
    var_35 = module_2.String(format=var_18)
    var_36 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_33, var_16: var_34, var_17: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = {}
    var_39 = module_4.Form(env=var_10, schema=var_37, values=var_38)
    var_40 = var_37.fields[var_11]
    var_41 = 'test'
    var_42 = var_39.render_field(field_name=var_11, field=var_40, value=var_41)
    var_43 = var_37.fields[var_12]
    var_44 = 'test@example.com'
    var_45 = var_39.render_field(field_name=var_12, field=var_43, value=var_44)
    var_46 = var_37.fields[var_13]
    var_47 = 'secret'
    var_48 = var_39.render_field(field_name=var_13, field=var_46, value=var_47)
    var_49 = var_37.fields[var_14]
    var_50 = '42'
    var_51 = var_39.render_field(field_name=var_14, field=var_49, value=var_50)
    var_52 = var_37.fields[var_15]
    var_53 = var_39.render_field(field_name=var_15, field=var_52, value=var_26)
    var_54 = var_37.fields[var_16]
    var_55 = True
    var_56 = var_39.render_field(field_name=var_16, field=var_54, value=var_55)
    var_57 = var_37.fields[var_17]
    var_58 = 'long text'
    var_59 = var_39.render_field(field_name=var_17, field=var_57, value=var_58)
    var_60 = var_37.fields[var_11]
    var_61 = 'Invalid'
    var_62 = var_39.render_field(field_name=var_11, field=var_60, value=var_41, error=var_61)



# Parsed testcases at query #15
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'username'
    var_12 = 'email'
    var_13 = 'age'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'country'
    var_17 = 'text'
    var_18 = module_2.String(format=var_17)
    var_19 = module_2.String(format=var_12)
    var_20 = 'number'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.String(format=var_17)
    var_23 = module_2.Boolean()
    var_24 = 'US'
    var_25 = 'UK'
    var_26 = 'CA'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = {var_11: var_18, var_12: var_19, var_13: var_21, var_14: var_22, var_15: var_23, var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_4.Form(env=var_10, schema=var_30)
    var_32 = var_30.fields[var_14]
    var_33 = 'Test bio'
    var_34 = var_31.render_field(field_name=var_14, field=var_32, value=var_33)
    var_35 = var_30.fields[var_12]
    var_36 = 'test@example.com'
    var_37 = var_31.render_field(field_name=var_12, field=var_35, value=var_36)
    var_38 = var_30.fields[var_13]
    var_39 = '25'
    var_40 = var_31.render_field(field_name=var_13, field=var_38, value=var_39)
    var_41 = var_30.fields[var_15]
    var_42 = True
    var_43 = var_31.render_field(field_name=var_15, field=var_41, value=var_42)
    var_44 = var_30.fields[var_16]
    var_45 = var_31.render_field(field_name=var_16, field=var_44, value=var_24)
    var_46 = var_30.fields[var_11]
    var_47 = ''
    var_48 = 'This field is required'
    var_49 = var_31.render_field(field_name=var_11, field=var_46, value=var_47, error=var_48)
    var_50 = 'password'
    var_51 = module_2.String(format=var_50)
    var_52 = 'secret'
    var_53 = var_31.render_field(field_name=var_50, field=var_51, value=var_52)
    var_54 = module_2.String(format=var_17)
    var_55 = 'required'
    var_56 = var_31.render_field(field_name=var_55, field=var_54, value=var_47)



# Parsed testcases at query #16
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()
    var_9 = 'test_templates'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #17
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'text_field'
    var_3 = 'choice_field'
    var_4 = 'boolean_field'
    var_5 = 'password_field'
    var_6 = module_2.String()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = [var_7, var_8]
    var_10 = module_2.Choice(choices=var_9)
    var_11 = module_2.Boolean()
    var_12 = 'password'
    var_13 = module_2.String(format=var_12)
    var_14 = {var_2: var_6, var_3: var_10, var_4: var_11, var_5: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_4.Form(env=var_1, schema=var_15)
    var_17 = 'forms/input.html'
    var_18 = var_1.get_template(var_17)
    var_19 = var_15.fields[var_2]
    var_20 = 'test'
    var_21 = None
    var_22 = var_16.render_field(field_name=var_2, field=var_19, value=var_20, error=var_21)
    var_23 = 'field_id'
    var_24 = 'field_name'
    var_25 = 'field'
    var_26 = 'label'
    var_27 = 'required'
    var_28 = 'input_type'
    var_29 = 'value'
    var_30 = 'error'
    var_31 = 'text-field'
    var_32 = var_15.fields[var_2]
    var_33 = True
    var_34 = 'text'
    var_35 = {var_23: var_31, var_24: var_2, var_25: var_32, var_26: var_2, var_27: var_33, var_28: var_34, var_29: var_20, var_30: var_21}
    var_36 = 'forms/select.html'
    var_37 = var_1.get_template(var_36)
    var_38 = var_15.fields[var_3]
    var_39 = var_16.render_field(field_name=var_3, field=var_38, value=var_7, error=var_21)
    var_40 = 'choice-field'
    var_41 = var_15.fields[var_3]
    var_42 = {var_23: var_40, var_24: var_3, var_25: var_41, var_26: var_3, var_27: var_33, var_28: var_34, var_29: var_7, var_30: var_21}
    var_43 = 'forms/checkbox.html'
    var_44 = var_1.get_template(var_43)
    var_45 = var_15.fields[var_4]
    var_46 = var_16.render_field(field_name=var_4, field=var_45, value=var_33, error=var_21)
    var_47 = 'boolean-field'
    var_48 = var_15.fields[var_4]
    var_49 = 'checkbox'
    var_50 = {var_23: var_47, var_24: var_4, var_25: var_48, var_26: var_4, var_27: var_33, var_28: var_49, var_29: var_33, var_30: var_21}
    var_51 = var_1.get_template(var_17)
    var_52 = var_15.fields[var_5]
    var_53 = 'secret'
    var_54 = var_16.render_field(field_name=var_5, field=var_52, value=var_53, error=var_21)
    var_55 = 'password-field'
    var_56 = var_15.fields[var_5]
    var_57 = ''
    var_58 = {var_23: var_55, var_24: var_5, var_25: var_56, var_26: var_5, var_27: var_33, var_28: var_12, var_29: var_57, var_30: var_21}
    var_59 = var_1.get_template(var_17)
    var_60 = var_15.fields[var_2]
    var_61 = 'Invalid value'
    var_62 = var_16.render_field(field_name=var_2, field=var_60, value=var_20, error=var_61)
    var_63 = var_15.fields[var_2]
    var_64 = {var_23: var_31, var_24: var_2, var_25: var_63, var_26: var_2, var_27: var_33, var_28: var_34, var_29: var_20, var_30: var_61}



# Parsed testcases at query #18
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader



# Parsed testcases at query #19
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = 'password'
    var_4 = 'bio'
    var_5 = 'agree'
    var_6 = 'country'
    var_7 = 'Name'
    var_8 = 'text'
    var_9 = module_0.String(format=var_8)
    var_10 = 'Age'
    var_11 = 'number'
    var_12 = module_0.String(format=var_11)
    var_13 = 'Email'
    var_14 = module_0.String(format=var_2)
    var_15 = 'Password'
    var_16 = module_0.String(format=var_3)
    var_17 = 'Bio'
    var_18 = module_0.String(format=var_8)
    var_19 = 'Agree'
    var_20 = module_0.Boolean()
    var_21 = 'Country'
    var_22 = 'US'
    var_23 = 'UK'
    var_24 = 'CA'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = {var_0: var_9, var_1: var_12, var_2: var_14, var_3: var_16, var_4: var_18, var_5: var_20, var_6: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 'forms/input.html'
    var_30 = 'forms/textarea.html'
    var_31 = 'forms/select.html'
    var_32 = 'forms/checkbox.html'
    var_33 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_34 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_35 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_36 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_37 = {var_29: var_33, var_30: var_34, var_31: var_35, var_32: var_36}
    var_38 = module_2.DictLoader(var_37)
    var_39 = module_3.Environment(loader=var_38)
    var_40 = 'John'
    var_41 = '30'
    var_42 = 'john@example.com'
    var_43 = 'secret'
    var_44 = 'Developer'
    var_45 = True
    var_46 = {var_0: var_40, var_1: var_41, var_2: var_42, var_3: var_43, var_4: var_44, var_5: var_45, var_6: var_22}
    var_47 = module_4.Form(env=var_39, schema=var_28, values=var_46)
    var_48 = var_28.fields[var_0]
    var_49 = var_47.render_field(field_name=var_0, field=var_48, value=var_40)
    var_50 = var_28.fields[var_1]
    var_51 = var_47.render_field(field_name=var_1, field=var_50, value=var_41)
    var_52 = var_28.fields[var_2]
    var_53 = var_47.render_field(field_name=var_2, field=var_52, value=var_42)
    var_54 = var_28.fields[var_3]
    var_55 = var_47.render_field(field_name=var_3, field=var_54, value=var_43)
    var_56 = var_28.fields[var_4]
    var_57 = var_47.render_field(field_name=var_4, field=var_56, value=var_44)
    var_58 = var_28.fields[var_5]
    var_59 = var_47.render_field(field_name=var_5, field=var_58, value=var_45)
    var_60 = var_28.fields[var_6]
    var_61 = var_47.render_field(field_name=var_6, field=var_60, value=var_22)
    var_62 = var_28.fields[var_6]
    var_63 = var_47.render_field(field_name=var_6, field=var_62, value=var_22)
    var_64 = var_28.fields[var_0]
    var_65 = 'Invalid name'
    var_66 = var_47.render_field(field_name=var_0, field=var_64, value=var_40, error=var_65)



# Parsed testcases at query #21
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'text_area'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'password'
    var_22 = module_2.String(format=var_21)
    var_23 = 'number'
    var_24 = module_2.String(format=var_23)
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_2.Choice(choices=var_28)
    var_30 = module_2.Boolean()
    var_31 = 'text'
    var_32 = module_2.String(format=var_31)
    var_33 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_29, var_16: var_30, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = {}
    var_36 = module_4.Form(env=var_10, schema=var_34, values=var_35)
    var_37 = var_34.fields[var_11]
    var_38 = 'test'
    var_39 = var_36.render_field(field_name=var_11, field=var_37, value=var_38)
    var_40 = var_34.fields[var_12]
    var_41 = 'test@example.com'
    var_42 = var_36.render_field(field_name=var_12, field=var_40, value=var_41)
    var_43 = var_34.fields[var_13]
    var_44 = 'secret'
    var_45 = var_36.render_field(field_name=var_13, field=var_43, value=var_44)
    var_46 = var_34.fields[var_14]
    var_47 = '123'
    var_48 = var_36.render_field(field_name=var_14, field=var_46, value=var_47)
    var_49 = var_34.fields[var_15]
    var_50 = var_36.render_field(field_name=var_15, field=var_49, value=var_25)
    var_51 = var_34.fields[var_16]
    var_52 = True
    var_53 = var_36.render_field(field_name=var_16, field=var_51, value=var_52)
    var_54 = var_34.fields[var_17]
    var_55 = 'long text'
    var_56 = var_36.render_field(field_name=var_17, field=var_54, value=var_55)
    var_57 = var_34.fields[var_11]
    var_58 = 'Invalid'
    var_59 = var_36.render_field(field_name=var_11, field=var_57, value=var_38, error=var_58)



# Parsed testcases at query #22
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'my_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #23
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'password'
    var_22 = module_2.String(format=var_21)
    var_23 = 'number'
    var_24 = module_2.String(format=var_23)
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_2.Choice(choices=var_28)
    var_30 = module_2.Boolean()
    var_31 = 'text'
    var_32 = module_2.String(format=var_31)
    var_33 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_29, var_16: var_30, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = {}
    var_36 = module_4.Form(env=var_10, schema=var_34, values=var_35)
    var_37 = var_34.fields[var_11]
    var_38 = 'test'
    var_39 = var_36.render_field(field_name=var_11, field=var_37, value=var_38)
    var_40 = var_34.fields[var_12]
    var_41 = 'test@example.com'
    var_42 = var_36.render_field(field_name=var_12, field=var_40, value=var_41)
    var_43 = var_34.fields[var_13]
    var_44 = 'secret'
    var_45 = var_36.render_field(field_name=var_13, field=var_43, value=var_44)
    var_46 = var_34.fields[var_14]
    var_47 = '42'
    var_48 = var_36.render_field(field_name=var_14, field=var_46, value=var_47)
    var_49 = var_34.fields[var_15]
    var_50 = var_36.render_field(field_name=var_15, field=var_49, value=var_25)
    var_51 = var_34.fields[var_16]
    var_52 = True
    var_53 = var_36.render_field(field_name=var_16, field=var_51, value=var_52)
    var_54 = var_34.fields[var_17]
    var_55 = 'long text'
    var_56 = var_36.render_field(field_name=var_17, field=var_54, value=var_55)
    var_57 = var_34.fields[var_11]
    var_58 = 'Invalid'
    var_59 = var_36.render_field(field_name=var_11, field=var_57, value=var_38, error=var_58)



# Parsed testcases at query #24
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #25
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader
    var_4 = 'test_package'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = var_5.env.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.env
    var_10 = var_8.env.loader
    var_11 = module_0.Jinja2Forms()



# Parsed testcases at query #26
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'text_field'
    var_3 = 'email_field'
    var_4 = 'number_field'
    var_5 = 'choice_field'
    var_6 = 'bool_field'
    var_7 = 'password_field'
    var_8 = 'text'
    var_9 = module_2.String(format=var_8)
    var_10 = 'email'
    var_11 = module_2.String(format=var_10)
    var_12 = 'number'
    var_13 = module_2.String(format=var_12)
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_2.Choice(choices=var_17)
    var_19 = module_2.Boolean()
    var_20 = 'password'
    var_21 = module_2.String(format=var_20)
    var_22 = {var_2: var_9, var_3: var_11, var_4: var_13, var_5: var_18, var_6: var_19, var_7: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = module_4.Form(env=var_1, schema=var_23)
    var_25 = var_23.fields[var_2]
    var_26 = 'test_value'
    var_27 = var_24.render_field(field_name=var_2, field=var_25, value=var_26)
    var_28 = var_23.fields[var_3]
    var_29 = 'test@example.com'
    var_30 = var_24.render_field(field_name=var_3, field=var_28, value=var_29)
    var_31 = var_23.fields[var_4]
    var_32 = '123'
    var_33 = var_24.render_field(field_name=var_4, field=var_31, value=var_32)
    var_34 = var_23.fields[var_5]
    var_35 = var_24.render_field(field_name=var_5, field=var_34, value=var_14)
    var_36 = var_23.fields[var_6]
    var_37 = True
    var_38 = var_24.render_field(field_name=var_6, field=var_36, value=var_37)
    var_39 = var_23.fields[var_7]
    var_40 = 'secret'
    var_41 = var_24.render_field(field_name=var_7, field=var_39, value=var_40)
    var_42 = var_23.fields[var_2]
    var_43 = 'Error message'
    var_44 = var_24.render_field(field_name=var_2, field=var_42, value=var_26, error=var_43)



# Parsed testcases at query #27
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #28
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
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



# Parsed testcases at query #29
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'textarea_field'
    var_16 = 'choice_field'
    var_17 = 'boolean_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'password'
    var_22 = module_2.String(format=var_21)
    var_23 = 'number'
    var_24 = module_2.String(format=var_23)
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = 'a'
    var_28 = 'b'
    var_29 = 'c'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_2.Choice(choices=var_30)
    var_32 = module_2.Boolean()
    var_33 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_26, var_16: var_31, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = {}
    var_36 = module_4.Form(env=var_10, schema=var_34, values=var_35)
    var_37 = var_34.fields[var_11]
    var_38 = 'test_value'
    var_39 = None
    var_40 = var_36.render_field(field_name=var_11, field=var_37, value=var_38, error=var_39)
    var_41 = var_34.fields[var_12]
    var_42 = 'test@example.com'
    var_43 = var_36.render_field(field_name=var_12, field=var_41, value=var_42, error=var_39)
    var_44 = var_34.fields[var_13]
    var_45 = 'secret'
    var_46 = var_36.render_field(field_name=var_13, field=var_44, value=var_45, error=var_39)
    var_47 = var_34.fields[var_14]
    var_48 = '123'
    var_49 = var_36.render_field(field_name=var_14, field=var_47, value=var_48, error=var_39)
    var_50 = var_34.fields[var_15]
    var_51 = 'multiline text'
    var_52 = var_36.render_field(field_name=var_15, field=var_50, value=var_51, error=var_39)
    var_53 = var_34.fields[var_16]
    var_54 = var_36.render_field(field_name=var_16, field=var_53, value=var_28, error=var_39)
    var_55 = var_34.fields[var_17]
    var_56 = True
    var_57 = var_36.render_field(field_name=var_17, field=var_55, value=var_56, error=var_39)
    var_58 = var_34.fields[var_17]
    var_59 = False
    var_60 = var_36.render_field(field_name=var_17, field=var_58, value=var_59, error=var_39)
    var_61 = var_34.fields[var_11]
    var_62 = ''
    var_63 = 'This field is required'
    var_64 = var_36.render_field(field_name=var_11, field=var_61, value=var_62, error=var_63)



# Parsed testcases at query #30
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'textarea_field'
    var_16 = 'choice_field'
    var_17 = 'boolean_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'number'
    var_22 = module_2.String(format=var_21)
    var_23 = 'password'
    var_24 = module_2.String(format=var_23)
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = '1'
    var_28 = 'Option 1'
    var_29 = (var_27, var_28)
    var_30 = '2'
    var_31 = 'Option 2'
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = module_2.Choice(choices=var_33)
    var_35 = module_2.Boolean()
    var_36 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_26, var_16: var_34, var_17: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_4.Form(env=var_10, schema=var_37)
    var_39 = var_37.fields[var_11]
    var_40 = 'test'
    var_41 = var_38.render_field(field_name=var_11, field=var_39, value=var_40)
    var_42 = var_37.fields[var_12]
    var_43 = 'test@example.com'
    var_44 = var_38.render_field(field_name=var_12, field=var_42, value=var_43)
    var_45 = var_37.fields[var_13]
    var_46 = '123'
    var_47 = var_38.render_field(field_name=var_13, field=var_45, value=var_46)
    var_48 = var_37.fields[var_14]
    var_49 = 'secret'
    var_50 = var_38.render_field(field_name=var_14, field=var_48, value=var_49)
    var_51 = var_37.fields[var_15]
    var_52 = 'long text'
    var_53 = var_38.render_field(field_name=var_15, field=var_51, value=var_52)
    var_54 = var_37.fields[var_16]
    var_55 = var_38.render_field(field_name=var_16, field=var_54, value=var_27)
    var_56 = var_37.fields[var_17]
    var_57 = True
    var_58 = var_38.render_field(field_name=var_17, field=var_56, value=var_57)
    var_59 = var_37.fields[var_17]
    var_60 = False
    var_61 = var_38.render_field(field_name=var_17, field=var_59, value=var_60)



# Parsed testcases at query #31
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'forms/input.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = 'forms/textarea.html'
    var_5 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_6 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_7 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_8 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'text_field'
    var_11 = 'email_field'
    var_12 = 'number_field'
    var_13 = 'choice_field'
    var_14 = 'bool_field'
    var_15 = 'password_field'
    var_16 = 'text'
    var_17 = module_1.String(format=var_16)
    var_18 = 'email'
    var_19 = module_1.String(format=var_18)
    var_20 = 'number'
    var_21 = module_1.String(format=var_20)
    var_22 = 'a'
    var_23 = 'b'
    var_24 = 'c'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_1.Choice(choices=var_25)
    var_27 = module_1.Boolean()
    var_28 = 'password'
    var_29 = module_1.String(format=var_28)
    var_30 = {var_10: var_17, var_11: var_19, var_12: var_21, var_13: var_26, var_14: var_27, var_15: var_29}
    var_31 = module_2.Schema(var_30)
    var_32 = module_3.Form(env=var_0, schema=var_31)
    var_33 = var_31.fields[var_10]
    var_34 = 'test value'
    var_35 = None
    var_36 = var_32.render_field(field_name=var_10, field=var_33, value=var_34, error=var_35)
    var_37 = var_31.fields[var_11]
    var_38 = 'test@example.com'
    var_39 = var_32.render_field(field_name=var_11, field=var_37, value=var_38, error=var_35)
    var_40 = var_31.fields[var_12]
    var_41 = '123'
    var_42 = var_32.render_field(field_name=var_12, field=var_40, value=var_41, error=var_35)
    var_43 = var_31.fields[var_13]
    var_44 = var_32.render_field(field_name=var_13, field=var_43, value=var_22, error=var_35)
    var_45 = var_31.fields[var_14]
    var_46 = True
    var_47 = var_32.render_field(field_name=var_14, field=var_45, value=var_46, error=var_35)
    var_48 = var_31.fields[var_15]
    var_49 = 'secret'
    var_50 = var_32.render_field(field_name=var_15, field=var_48, value=var_49, error=var_35)
    var_51 = var_31.fields[var_10]
    var_52 = ''
    var_53 = 'This field is required'
    var_54 = var_32.render_field(field_name=var_10, field=var_51, value=var_52, error=var_53)



# Parsed testcases at query #32
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.load_template_env(package=var_4)
    var_7 = var_6.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.load_template_env(directory=var_0, package=var_4)
    var_10 = var_9.loader



# Parsed testcases at query #33
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
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



# Parsed testcases at query #34
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'package.name'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #35
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'password'
    var_22 = module_2.String(format=var_21)
    var_23 = 'number'
    var_24 = module_2.String(format=var_23)
    var_25 = 'a'
    var_26 = 'A'
    var_27 = (var_25, var_26)
    var_28 = 'b'
    var_29 = 'B'
    var_30 = (var_28, var_29)
    var_31 = [var_27, var_30]
    var_32 = module_2.Choice(choices=var_31)
    var_33 = module_2.Boolean()
    var_34 = 'text'
    var_35 = module_2.String(format=var_34)
    var_36 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_32, var_16: var_33, var_17: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_4.Form(env=var_10, schema=var_37)
    var_39 = var_37.fields[var_11]
    var_40 = 'test'
    var_41 = var_38.render_field(field_name=var_11, field=var_39, value=var_40)
    var_42 = var_37.fields[var_12]
    var_43 = 'test@example.com'
    var_44 = var_38.render_field(field_name=var_12, field=var_42, value=var_43)
    var_45 = var_37.fields[var_13]
    var_46 = 'secret'
    var_47 = var_38.render_field(field_name=var_13, field=var_45, value=var_46)
    var_48 = var_37.fields[var_14]
    var_49 = '42'
    var_50 = var_38.render_field(field_name=var_14, field=var_48, value=var_49)
    var_51 = var_37.fields[var_15]
    var_52 = var_38.render_field(field_name=var_15, field=var_51, value=var_25)
    var_53 = var_37.fields[var_16]
    var_54 = True
    var_55 = var_38.render_field(field_name=var_16, field=var_53, value=var_54)
    var_56 = var_37.fields[var_17]
    var_57 = 'long text'
    var_58 = var_38.render_field(field_name=var_17, field=var_56, value=var_57)
    var_59 = var_37.fields[var_11]
    var_60 = 'Invalid'
    var_61 = var_38.render_field(field_name=var_11, field=var_59, value=var_40, error=var_60)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_2 = 'text_field'
    var_3 = 'choice_field'
    var_4 = 'bool_field'
    var_5 = 'text_area'
    var_6 = module_2.String()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = [var_7, var_8]
    var_10 = module_2.Choice(choices=var_9)
    var_11 = module_2.Boolean()
    var_12 = 'text'
    var_13 = module_2.String(format=var_12)
    var_14 = {var_2: var_6, var_3: var_10, var_4: var_11, var_5: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_4.Form(env=var_1, schema=var_15)
    var_17 = var_15.fields[var_2]
    var_18 = 'test_value'
    var_19 = None
    var_20 = var_16.render_field(field_name=var_2, field=var_17, value=var_18, error=var_19)
    var_21 = var_15.fields[var_3]
    var_22 = var_16.render_field(field_name=var_3, field=var_21, value=var_7, error=var_19)
    var_23 = var_15.fields[var_4]
    var_24 = True
    var_25 = var_16.render_field(field_name=var_4, field=var_23, value=var_24, error=var_19)
    var_26 = var_15.fields[var_5]
    var_27 = 'long text'
    var_28 = var_16.render_field(field_name=var_5, field=var_26, value=var_27, error=var_19)
    var_29 = var_15.fields[var_2]
    var_30 = 'Invalid value'
    var_31 = var_16.render_field(field_name=var_2, field=var_29, value=var_18, error=var_30)
    var_32 = 'password'
    var_33 = module_2.String(format=var_32)
    var_34 = 'secret'
    var_35 = var_16.render_field(field_name=var_32, field=var_33, value=var_34, error=var_19)



# Parsed testcases at query #2
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/select.html'
    var_2 = 'forms/textarea.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'active'
    var_15 = module_2.String()
    var_16 = 'number'
    var_17 = module_2.String(format=var_16)
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = module_2.Boolean()
    var_21 = {var_11: var_15, var_12: var_17, var_13: var_19, var_14: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = 'John'
    var_24 = '30'
    var_25 = 'Test'
    var_26 = True
    var_27 = {var_11: var_23, var_12: var_24, var_13: var_25, var_14: var_26}
    var_28 = module_4.Form(env=var_10, schema=var_22, values=var_27)
    var_29 = str(var_28)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = 'test'
    var_3 = {var_2: var_0}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.Form(env=var_1, schema=var_4)
    var_6 = var_5.input_type_for_field(var_0)
    assert var_6 == 'text'
    var_7 = 'email'
    var_8 = module_0.String(format=var_7)
    var_9 = {var_2: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = module_2.Form(env=var_1, schema=var_10)
    var_12 = var_11.input_type_for_field(var_8)
    assert var_12 == 'email'
    var_13 = 'unknown'
    var_14 = module_0.String(format=var_13)
    var_15 = {var_2: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = module_2.Form(env=var_1, schema=var_16)
    var_18 = var_17.input_type_for_field(var_14)
    assert var_18 == 'text'
    var_19 = 'color'
    var_20 = (var_19, var_19)
    var_21 = 'datetime'
    var_22 = 'datetime-local'
    var_23 = (var_21, var_22)
    var_24 = 'date'
    var_25 = (var_24, var_24)
    var_26 = 'month'
    var_27 = (var_26, var_26)
    var_28 = 'number'
    var_29 = (var_28, var_28)
    var_30 = 'password'
    var_31 = (var_30, var_30)
    var_32 = 'range'
    var_33 = (var_32, var_32)
    var_34 = 'search'
    var_35 = (var_34, var_34)
    var_36 = 'tel'
    var_37 = (var_36, var_36)
    var_38 = 'time'
    var_39 = (var_38, var_38)
    var_40 = 'url'
    var_41 = (var_40, var_40)
    var_42 = 'week'
    var_43 = (var_42, var_42)
    var_44 = [var_20, var_23, var_25, var_27, var_29, var_31, var_33, var_35, var_37, var_39, var_41, var_43]
    var_45 = None
    var_46 = 'test'
    var_47 = {var_46: var_14}
    var_48 = module_1.Schema(var_47)
    var_49 = module_2.Form(env=var_45, schema=var_48)
    var_50 = var_49.input_type_for_field(var_14)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = None
    var_9 = 'test'
    var_10 = {var_9: var_7}
    var_11 = module_1.Schema(var_10)
    var_12 = module_2.Form(env=var_8, schema=var_11)
    var_13 = var_12.template_for_field(var_7)
    assert var_13 == 'forms/select.html'
    var_14 = module_0.Boolean()
    var_15 = {var_9: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = module_2.Form(env=var_8, schema=var_16)
    var_18 = var_17.template_for_field(var_14)
    assert var_18 == 'forms/checkbox.html'
    var_19 = 'text'
    var_20 = module_0.String(format=var_19)
    var_21 = {var_9: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = module_2.Form(env=var_8, schema=var_22)
    var_24 = var_23.template_for_field(var_20)
    assert var_24 == 'forms/textarea.html'
    var_25 = module_0.String()
    var_26 = {var_9: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = module_2.Form(env=var_8, schema=var_27)
    var_29 = var_28.template_for_field(var_25)
    assert var_29 == 'forms/input.html'
    var_30 = module_0.Field()
    var_31 = {var_9: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = module_2.Form(env=var_8, schema=var_32)
    var_34 = var_33.template_for_field(var_30)
    assert var_34 == 'forms/input.html'
    var_35 = {}
    var_36 = module_0.Object()
    var_37 = {var_9: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = module_2.Form(env=var_8, schema=var_38)
    var_40 = var_39.template_for_field(var_36)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_0.Boolean()
    var_9 = 'text'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.String()
    var_12 = module_0.Field()
    var_13 = {}
    var_14 = module_0.Object()



# Parsed testcases at query #6
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'text_field'
    var_3 = 'choice_field'
    var_4 = 'bool_field'
    var_5 = 'password_field'
    var_6 = module_2.String()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = [var_7, var_8]
    var_10 = module_2.Choice(choices=var_9)
    var_11 = module_2.Boolean()
    var_12 = 'password'
    var_13 = module_2.String(format=var_12)
    var_14 = {var_2: var_6, var_3: var_10, var_4: var_11, var_5: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_4.Form(env=var_1, schema=var_15)
    var_17 = var_15.fields[var_2]
    var_18 = 'test_value'
    var_19 = None
    var_20 = var_16.render_field(field_name=var_2, field=var_17, value=var_18, error=var_19)
    var_21 = var_15.fields[var_3]
    var_22 = var_16.render_field(field_name=var_3, field=var_21, value=var_7, error=var_19)
    var_23 = var_15.fields[var_4]
    var_24 = True
    var_25 = var_16.render_field(field_name=var_4, field=var_23, value=var_24, error=var_19)
    var_26 = var_15.fields[var_5]
    var_27 = 'secret'
    var_28 = var_16.render_field(field_name=var_5, field=var_26, value=var_27, error=var_19)
    var_29 = var_15.fields[var_2]
    var_30 = 'Invalid value'
    var_31 = var_16.render_field(field_name=var_2, field=var_29, value=var_18, error=var_30)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = 'number'
    var_4 = module_0.String(format=var_3)
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = 'forms/input.html'
    var_8 = 'forms/textarea.html'
    var_9 = 'forms/select.html'
    var_10 = 'forms/checkbox.html'
    var_11 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_12 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_13 = '<select name="{{ field_name }}"></select>'
    var_14 = '<input type="checkbox" name="{{ field_name }}">'
    var_15 = {var_7: var_11, var_8: var_12, var_9: var_13, var_10: var_14}
    var_16 = module_2.DictLoader(var_15)
    var_17 = module_3.Environment(loader=var_16)
    var_18 = 'John'
    var_19 = '30'
    var_20 = {var_0: var_18, var_1: var_19}
    var_21 = module_4.Form(env=var_17, schema=var_6, values=var_20)
    var_22 = var_21.__html__()
    var_23 = str(var_22)
    assert var_23 == '<input type="text" name="name" value="John"><input type="number" name="age" value="30">'



# Parsed testcases at query #8
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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<select name="{{ field_name }}"></select>'
    var_6 = '<input type="checkbox" name="{{ field_name }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'agree'
    var_15 = 'country'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'us'
    var_23 = 'uk'
    var_24 = 'ca'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_21, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'Test'
    var_32 = True
    var_33 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_22}
    var_34 = module_4.Form(env=var_10, schema=var_28, values=var_33)
    var_35 = var_34.__html__()
    var_36 = str(var_35)
    assert var_36 == '<input type="text" name="name" value="John"><input type="number" name="age" value="30"><textarea name="description">Test</textarea><input type="checkbox" name="agree"><select name="country"></select>'



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'description'
    var_3 = module_0.String()
    var_4 = 'number'
    var_5 = module_0.String(format=var_4)
    var_6 = 'text'
    var_7 = module_0.String(format=var_6)
    var_8 = {var_0: var_3, var_1: var_5, var_2: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = 'forms/input.html'
    var_11 = 'forms/textarea.html'
    var_12 = 'forms/select.html'
    var_13 = 'forms/checkbox.html'
    var_14 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_15 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_16 = '<select name="{{ field_name }}"></select>'
    var_17 = '<input type="checkbox" name="{{ field_name }}">'
    var_18 = {var_10: var_14, var_11: var_15, var_12: var_16, var_13: var_17}
    var_19 = module_2.DictLoader(var_18)
    var_20 = module_3.Environment(loader=var_19)
    var_21 = 'John'
    var_22 = '30'
    var_23 = 'Test'
    var_24 = {var_0: var_21, var_1: var_22, var_2: var_23}
    var_25 = module_4.Form(env=var_20, schema=var_9, values=var_24)
    var_26 = var_25.__html__()
    var_27 = str(var_26)
    var_28 = var_25.render_fields()



# Parsed testcases at query #10
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'number_field'
    var_13 = 'password_field'
    var_14 = 'email_field'
    var_15 = 'textarea_field'
    var_16 = 'select_field'
    var_17 = 'checkbox_field'
    var_18 = module_2.String()
    var_19 = 'number'
    var_20 = module_2.String(format=var_19)
    var_21 = 'password'
    var_22 = module_2.String(format=var_21)
    var_23 = 'email'
    var_24 = module_2.String(format=var_23)
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = 'a'
    var_28 = 'b'
    var_29 = 'c'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_2.Choice(choices=var_30)
    var_32 = module_2.Boolean()
    var_33 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_26, var_16: var_31, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_4.Form(env=var_10, schema=var_34)
    var_36 = var_34.fields[var_11]
    var_37 = 'test'
    var_38 = var_35.render_field(field_name=var_11, field=var_36, value=var_37)
    var_39 = var_34.fields[var_12]
    var_40 = '123'
    var_41 = var_35.render_field(field_name=var_12, field=var_39, value=var_40)
    var_42 = var_34.fields[var_13]
    var_43 = 'secret'
    var_44 = var_35.render_field(field_name=var_13, field=var_42, value=var_43)
    var_45 = var_34.fields[var_14]
    var_46 = 'test@example.com'
    var_47 = var_35.render_field(field_name=var_14, field=var_45, value=var_46)
    var_48 = var_34.fields[var_15]
    var_49 = 'long text'
    var_50 = var_35.render_field(field_name=var_15, field=var_48, value=var_49)
    var_51 = var_34.fields[var_16]
    var_52 = var_35.render_field(field_name=var_16, field=var_51, value=var_27)
    var_53 = var_34.fields[var_17]
    var_54 = True
    var_55 = var_35.render_field(field_name=var_17, field=var_53, value=var_54)
    var_56 = var_34.fields[var_17]
    var_57 = False
    var_58 = var_35.render_field(field_name=var_17, field=var_56, value=var_57)



# Parsed testcases at query #11
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
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



# Parsed testcases at query #12
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'text_area_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'password'
    var_22 = module_2.String(format=var_21)
    var_23 = 'number'
    var_24 = module_2.String(format=var_23)
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_2.Choice(choices=var_28)
    var_30 = module_2.Boolean()
    var_31 = 'text'
    var_32 = module_2.String(format=var_31)
    var_33 = {var_11: var_18, var_12: var_20, var_13: var_22, var_14: var_24, var_15: var_29, var_16: var_30, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_4.Form(env=var_10, schema=var_34)
    var_36 = var_34.fields[var_11]
    var_37 = 'test'
    var_38 = var_35.render_field(field_name=var_11, field=var_36, value=var_37)
    var_39 = var_34.fields[var_12]
    var_40 = 'test@example.com'
    var_41 = var_35.render_field(field_name=var_12, field=var_39, value=var_40)
    var_42 = var_34.fields[var_13]
    var_43 = 'secret'
    var_44 = var_35.render_field(field_name=var_13, field=var_42, value=var_43)
    var_45 = var_34.fields[var_14]
    var_46 = '42'
    var_47 = var_35.render_field(field_name=var_14, field=var_45, value=var_46)
    var_48 = var_34.fields[var_15]
    var_49 = var_35.render_field(field_name=var_15, field=var_48, value=var_25)
    var_50 = var_34.fields[var_16]
    var_51 = True
    var_52 = var_35.render_field(field_name=var_16, field=var_50, value=var_51)
    var_53 = var_34.fields[var_17]
    var_54 = 'long text'
    var_55 = var_35.render_field(field_name=var_17, field=var_53, value=var_54)
    var_56 = var_34.fields[var_11]
    var_57 = 'Invalid'
    var_58 = var_35.render_field(field_name=var_11, field=var_56, value=var_37, error=var_57)



# Parsed testcases at query #13
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #14
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #15
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
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



# Parsed testcases at query #16
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'package.name'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #17
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.load_template_env(directory=var_0)
    var_3 = var_2.loader
    var_4 = 'test_package'
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


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = True
    var_2 = module_1.Environment(autoescape=var_1, loader=var_0)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'email'
    var_6 = 'bio'
    var_7 = 'agree'
    var_8 = 'country'
    var_9 = 'Name'
    var_10 = 'text'
    var_11 = module_2.String(format=var_10)
    var_12 = 'Age'
    var_13 = 'number'
    var_14 = module_2.String(format=var_13)
    var_15 = 'Email'
    var_16 = module_2.String(format=var_5)
    var_17 = 'Bio'
    var_18 = module_2.String(format=var_10)
    var_19 = 'Agree'
    var_20 = module_2.Boolean()
    var_21 = 'Country'
    var_22 = 'US'
    var_23 = 'UK'
    var_24 = 'CA'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_3: var_11, var_4: var_14, var_5: var_16, var_6: var_18, var_7: var_20, var_8: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_2, schema=var_28)
    var_30 = 'name'
    var_31 = var_28.fields[var_30]
    var_32 = 'Test'
    var_33 = var_29.render_field(field_name=var_30, field=var_31, value=var_32)
    var_34 = 'forms/textarea.html'
    var_35 = var_2.get_template(var_34)
    var_36 = str(var_35)
    var_37 = 'age'
    var_38 = var_28.fields[var_37]
    var_39 = '25'
    var_40 = var_29.render_field(field_name=var_37, field=var_38, value=var_39)
    var_41 = 'forms/input.html'
    var_42 = var_2.get_template(var_41)
    var_43 = str(var_42)
    var_44 = 'email'
    var_45 = var_28.fields[var_44]
    var_46 = 'test@example.com'
    var_47 = var_29.render_field(field_name=var_44, field=var_45, value=var_46)
    var_48 = var_2.get_template(var_41)
    var_49 = str(var_48)
    var_50 = 'agree'
    var_51 = var_28.fields[var_50]
    var_52 = var_29.render_field(field_name=var_50, field=var_51, value=var_1)
    var_53 = 'forms/checkbox.html'
    var_54 = var_2.get_template(var_53)
    var_55 = str(var_54)
    var_56 = 'country'
    var_57 = var_28.fields[var_56]
    var_58 = var_29.render_field(field_name=var_56, field=var_57, value=var_22)
    var_59 = 'forms/select.html'
    var_60 = var_2.get_template(var_59)
    var_61 = str(var_60)
    var_62 = 'name'
    var_63 = var_28.fields[var_62]
    var_64 = 'Invalid name'
    var_65 = var_29.render_field(field_name=var_62, field=var_63, value=var_32, error=var_64)
    var_66 = 'Password'
    var_67 = 'password'
    var_68 = module_2.String(format=var_67)
    var_69 = 'secret'
    var_70 = var_29.render_field(field_name=var_67, field=var_68, value=var_69)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = 'bio'
    var_4 = 'country'
    var_5 = 'subscribe'
    var_6 = module_0.String()
    var_7 = 'number'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.String(format=var_2)
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = 'US'
    var_13 = 'UK'
    var_14 = 'CA'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Choice(choices=var_15)
    var_17 = module_0.Boolean()
    var_18 = {var_0: var_6, var_1: var_8, var_2: var_9, var_3: var_11, var_4: var_16, var_5: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 'forms/input.html'
    var_21 = 'forms/textarea.html'
    var_22 = 'forms/select.html'
    var_23 = 'forms/checkbox.html'
    var_24 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_25 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_26 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_27 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_28 = {var_20: var_24, var_21: var_25, var_22: var_26, var_23: var_27}
    var_29 = module_2.DictLoader(var_28)
    var_30 = module_3.Environment(loader=var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'john@example.com'
    var_34 = 'Developer'
    var_35 = True
    var_36 = {var_0: var_31, var_1: var_32, var_2: var_33, var_3: var_34, var_4: var_12, var_5: var_35}
    var_37 = module_4.Form(env=var_30, schema=var_19, values=var_36)
    var_38 = var_37.render_fields()
    var_39 = ''
    var_40 = 'invalid'
    var_41 = 'INVALID'
    var_42 = {var_0: var_39, var_1: var_40, var_2: var_40, var_3: var_39, var_4: var_41, var_5: var_40}
    var_43 = var_37.validate(var_42)
    var_44 = var_37.render_fields()



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.environment as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'text_field'
    var_1 = 'email_field'
    var_2 = 'password_field'
    var_3 = 'number_field'
    var_4 = 'choice_field'
    var_5 = 'bool_field'
    var_6 = 'textarea_field'
    var_7 = module_0.String()
    var_8 = 'email'
    var_9 = module_0.String(format=var_8)
    var_10 = 'password'
    var_11 = module_0.String(format=var_10)
    var_12 = 'number'
    var_13 = module_0.String(format=var_12)
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = module_0.Boolean()
    var_20 = 'text'
    var_21 = module_0.String(format=var_20)
    var_22 = {var_0: var_7, var_1: var_9, var_2: var_11, var_3: var_13, var_4: var_18, var_5: var_19, var_6: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = module_2.Environment()
    var_25 = module_3.Form(env=var_24, schema=var_23)
    var_26 = var_23.fields[var_0]
    var_27 = 'test_value'
    var_28 = None
    var_29 = var_25.render_field(field_name=var_0, field=var_26, value=var_27, error=var_28)
    var_30 = var_23.fields[var_1]
    var_31 = 'test@example.com'
    var_32 = var_25.render_field(field_name=var_1, field=var_30, value=var_31, error=var_28)
    var_33 = var_23.fields[var_2]
    var_34 = 'secret'
    var_35 = var_25.render_field(field_name=var_2, field=var_33, value=var_34, error=var_28)
    var_36 = var_23.fields[var_3]
    var_37 = '42'
    var_38 = var_25.render_field(field_name=var_3, field=var_36, value=var_37, error=var_28)
    var_39 = var_23.fields[var_4]
    var_40 = var_25.render_field(field_name=var_4, field=var_39, value=var_15, error=var_28)
    var_41 = var_23.fields[var_5]
    var_42 = True
    var_43 = var_25.render_field(field_name=var_5, field=var_41, value=var_42, error=var_28)
    var_44 = var_23.fields[var_6]
    var_45 = 'Long text'
    var_46 = var_25.render_field(field_name=var_6, field=var_44, value=var_45, error=var_28)
    var_47 = var_23.fields[var_0]
    var_48 = 'Invalid value'
    var_49 = var_25.render_field(field_name=var_0, field=var_47, value=var_27, error=var_48)



# Parsed testcases at query #21
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'choice_field'
    var_13 = 'bool_field'
    var_14 = 'text_area'
    var_15 = module_2.String()
    var_16 = 'a'
    var_17 = 'b'
    var_18 = [var_16, var_17]
    var_19 = module_2.Choice(choices=var_18)
    var_20 = module_2.Boolean()
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = {var_11: var_15, var_12: var_19, var_13: var_20, var_14: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_4.Form(env=var_10, schema=var_24)
    var_26 = var_24.fields[var_11]
    var_27 = 'test'
    var_28 = var_25.render_field(field_name=var_11, field=var_26, value=var_27)
    var_29 = var_24.fields[var_12]
    var_30 = var_25.render_field(field_name=var_12, field=var_29, value=var_16)
    var_31 = var_24.fields[var_13]
    var_32 = True
    var_33 = var_25.render_field(field_name=var_13, field=var_31, value=var_32)
    var_34 = var_24.fields[var_14]
    var_35 = 'long text'
    var_36 = var_25.render_field(field_name=var_14, field=var_34, value=var_35)
    var_37 = var_24.fields[var_11]
    var_38 = 'Invalid'
    var_39 = var_25.render_field(field_name=var_11, field=var_37, value=var_27, error=var_38)



# Parsed testcases at query #22
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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_5 = '<select name="{{ field_name }}"></select>'
    var_6 = '<input type="checkbox" name="{{ field_name }}">'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'bio'
    var_14 = 'active'
    var_15 = 'gender'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'M'
    var_23 = 'F'
    var_24 = [var_22, var_23]
    var_25 = module_2.Choice(choices=var_24)
    var_26 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_21, var_15: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = 'John'
    var_29 = '30'
    var_30 = 'Test bio'
    var_31 = True
    var_32 = {var_11: var_28, var_12: var_29, var_13: var_30, var_14: var_31, var_15: var_22}
    var_33 = module_4.Form(env=var_10, schema=var_27, values=var_32)
    var_34 = {var_11: var_28, var_12: var_29, var_13: var_30, var_14: var_31, var_15: var_22}
    var_35 = var_33.validate(var_34)
    var_36 = var_33.render_fields()
    var_37 = ''
    var_38 = 'invalid'
    var_39 = 'not_bool'
    var_40 = 'X'
    var_41 = {var_11: var_37, var_12: var_38, var_13: var_37, var_14: var_39, var_15: var_40}
    var_42 = var_33.validate(var_41)
    var_43 = var_33.render_fields()
    var_44 = 'readonly_field'
    var_45 = module_2.String()
    var_46 = {var_44: var_45}
    var_47 = module_3.Schema(var_46)
    var_48 = module_4.Form(env=var_10, schema=var_47)
    var_49 = var_48.render_fields()



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = 'age'
    var_3 = 'password'
    var_4 = 'bio'
    var_5 = 'agree'
    var_6 = 'country'
    var_7 = 'text'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.String(format=var_1)
    var_10 = 'number'
    var_11 = module_0.String(format=var_10)
    var_12 = module_0.String(format=var_3)
    var_13 = module_0.String(format=var_7)
    var_14 = module_0.Boolean()
    var_15 = 'US'
    var_16 = 'UK'
    var_17 = 'CA'
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = {var_0: var_8, var_1: var_9, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = 'forms/input.html'
    var_23 = 'forms/textarea.html'
    var_24 = 'forms/select.html'
    var_25 = 'forms/checkbox.html'
    var_26 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_27 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_28 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_29 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_30 = {var_22: var_26, var_23: var_27, var_24: var_28, var_25: var_29}
    var_31 = module_2.DictLoader(var_30)
    var_32 = module_3.Environment(loader=var_31)
    var_33 = 'test'
    var_34 = 'test@example.com'
    var_35 = '25'
    var_36 = 'secret'
    var_37 = 'Hello'
    var_38 = True
    var_39 = {var_0: var_33, var_1: var_34, var_2: var_35, var_3: var_36, var_4: var_37, var_5: var_38, var_6: var_15}
    var_40 = module_4.Form(env=var_32, schema=var_21, values=var_39)
    var_41 = var_21.fields[var_0]
    var_42 = None
    var_43 = var_40.render_field(field_name=var_0, field=var_41, value=var_33, error=var_42)
    var_44 = var_21.fields[var_1]
    var_45 = var_40.render_field(field_name=var_1, field=var_44, value=var_34, error=var_42)
    var_46 = var_21.fields[var_2]
    var_47 = var_40.render_field(field_name=var_2, field=var_46, value=var_35, error=var_42)
    var_48 = var_21.fields[var_3]
    var_49 = var_40.render_field(field_name=var_3, field=var_48, value=var_36, error=var_42)
    var_50 = var_21.fields[var_4]
    var_51 = var_40.render_field(field_name=var_4, field=var_50, value=var_37, error=var_42)
    var_52 = var_21.fields[var_5]
    var_53 = var_40.render_field(field_name=var_5, field=var_52, value=var_38, error=var_42)
    var_54 = var_21.fields[var_6]
    var_55 = var_40.render_field(field_name=var_6, field=var_54, value=var_15, error=var_42)
    var_56 = var_21.fields[var_0]
    var_57 = 'Invalid username'
    var_58 = var_40.render_field(field_name=var_0, field=var_56, value=var_33, error=var_57)



# Parsed testcases at query #24
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'input.html'
    var_3 = 'select.html'
    var_4 = 'checkbox.html'
    var_5 = 'textarea.html'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_7 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_8 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_9 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_10 = 'name'
    var_11 = 'age'
    var_12 = 'bio'
    var_13 = 'active'
    var_14 = 'gender'
    var_15 = 100
    var_16 = module_2.String(max_length=var_15)
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'M'
    var_23 = 'F'
    var_24 = 'O'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_10: var_16, var_11: var_18, var_12: var_20, var_13: var_21, var_14: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'Developer'
    var_32 = True
    var_33 = {var_10: var_29, var_11: var_30, var_12: var_31, var_13: var_32, var_14: var_22}
    var_34 = module_4.Form(env=var_1, schema=var_28, values=var_33)
    var_35 = var_34.render_fields()
    var_36 = ''
    var_37 = 'invalid'
    var_38 = 'not_bool'
    var_39 = 'X'
    var_40 = {var_10: var_36, var_11: var_37, var_12: var_36, var_13: var_38, var_14: var_39}
    var_41 = var_34.validate(var_40)
    var_42 = var_34.render_fields()



# Parsed testcases at query #25
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'choice_field'
    var_13 = 'bool_field'
    var_14 = 'textarea_field'
    var_15 = module_2.String()
    var_16 = 'a'
    var_17 = 'b'
    var_18 = [var_16, var_17]
    var_19 = module_2.Choice(choices=var_18)
    var_20 = module_2.Boolean()
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = {var_11: var_15, var_12: var_19, var_13: var_20, var_14: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_4.Form(env=var_10, schema=var_24)
    var_26 = var_24.fields[var_11]
    var_27 = 'test'
    var_28 = var_25.render_field(field_name=var_11, field=var_26, value=var_27)
    var_29 = var_24.fields[var_12]
    var_30 = var_25.render_field(field_name=var_12, field=var_29)
    var_31 = var_24.fields[var_13]
    var_32 = True
    var_33 = var_25.render_field(field_name=var_13, field=var_31, value=var_32)
    var_34 = var_24.fields[var_14]
    var_35 = 'long text'
    var_36 = var_25.render_field(field_name=var_14, field=var_34, value=var_35)
    var_37 = var_24.fields[var_11]
    var_38 = 'Invalid'
    var_39 = var_25.render_field(field_name=var_11, field=var_37, error=var_38)
    var_40 = 'password'
    var_41 = module_2.String(format=var_40)
    var_42 = 'secret'
    var_43 = var_25.render_field(field_name=var_40, field=var_41, value=var_42)



# Parsed testcases at query #26
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
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_5 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_7 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'number_field'
    var_13 = 'password_field'
    var_14 = 'choice_field'
    var_15 = 'bool_field'
    var_16 = 'text_area'
    var_17 = module_2.String()
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = 'password'
    var_21 = module_2.String(format=var_20)
    var_22 = 'a'
    var_23 = 'b'
    var_24 = [var_22, var_23]
    var_25 = module_2.Choice(choices=var_24)
    var_26 = module_2.Boolean()
    var_27 = 'text'
    var_28 = module_2.String(format=var_27)
    var_29 = {var_11: var_17, var_12: var_19, var_13: var_21, var_14: var_25, var_15: var_26, var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_4.Form(env=var_10, schema=var_30)
    var_32 = var_30.fields[var_11]
    var_33 = 'test'
    var_34 = var_31.render_field(field_name=var_11, field=var_32, value=var_33)
    var_35 = var_30.fields[var_12]
    var_36 = '123'
    var_37 = var_31.render_field(field_name=var_12, field=var_35, value=var_36)
    var_38 = var_30.fields[var_13]
    var_39 = 'secret'
    var_40 = var_31.render_field(field_name=var_13, field=var_38, value=var_39)
    var_41 = var_30.fields[var_14]
    var_42 = var_31.render_field(field_name=var_14, field=var_41, value=var_22)
    var_43 = var_30.fields[var_15]
    var_44 = True
    var_45 = var_31.render_field(field_name=var_15, field=var_43, value=var_44)
    var_46 = var_30.fields[var_16]
    var_47 = 'long text'
    var_48 = var_31.render_field(field_name=var_16, field=var_46, value=var_47)
    var_49 = var_30.fields[var_11]
    var_50 = 'Invalid'
    var_51 = var_31.render_field(field_name=var_11, field=var_49, value=var_33, error=var_50)



