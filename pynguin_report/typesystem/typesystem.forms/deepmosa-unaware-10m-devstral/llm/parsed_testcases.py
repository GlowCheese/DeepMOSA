####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = '1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_0.Boolean()
    var_9 = 'text'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.String()
    var_12 = module_0.Field()
    var_13 = module_0.Object()



# Parsed testcases at query #2
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
    var_5 = module_3.String()
    var_6 = var_4.input_type_for_field(var_5)
    assert var_6 == 'text'
    var_7 = var_4.input_type_for_field(var_5)
    var_8 = 'unknown'
    var_9 = module_3.String(format=var_8)
    var_10 = var_4.input_type_for_field(var_9)
    assert var_10 == 'text'
    var_11 = module_3.Boolean()
    var_12 = var_4.input_type_for_field(var_11)
    assert var_12 == 'text'



# Parsed testcases at query #3
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
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for option in field.choices %}<option value="{{ option.value }}">{{ option.display_name }}</option>{% endfor %}</select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'country'
    var_15 = 'subscribe'
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = 'US'
    var_23 = 'United States'
    var_24 = (var_22, var_23)
    var_25 = 'UK'
    var_26 = 'United Kingdom'
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = module_2.Choice(choices=var_28)
    var_30 = module_2.Boolean()
    var_31 = {var_11: var_17, var_12: var_19, var_13: var_21, var_14: var_29, var_15: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = 'John'
    var_34 = '25'
    var_35 = 'Test'
    var_36 = True
    var_37 = {var_11: var_33, var_12: var_34, var_13: var_35, var_14: var_22, var_15: var_36}
    var_38 = module_4.Form(env=var_10, schema=var_32, values=var_37)
    var_39 = var_38.render_fields()
    var_40 = ''
    var_41 = 'invalid'
    var_42 = 'INVALID'
    var_43 = 'not_a_boolean'
    var_44 = {var_11: var_40, var_12: var_41, var_13: var_40, var_14: var_42, var_15: var_43}
    var_45 = var_38.validate(var_44)
    var_46 = var_38.render_fields()



# Parsed testcases at query #4
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
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'number'
    var_22 = module_2.String(format=var_21)
    var_23 = 'password'
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
    var_44 = '123'
    var_45 = var_36.render_field(field_name=var_13, field=var_43, value=var_44)
    var_46 = var_34.fields[var_14]
    var_47 = 'secret'
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
    var_58 = 'Error message'
    var_59 = var_36.render_field(field_name=var_11, field=var_57, value=var_38, error=var_58)



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_2.Schema(var_5)
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = module_3.Form(env=var_0, schema=var_6, values=var_9)
    var_11 = var_10.__html__()
    var_12 = str(var_11)
    var_13 = var_10.render_fields()



# Parsed testcases at query #7
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
    var_8 = var_7.__html__()
    var_9 = str(var_8)
    var_10 = str(var_7)



# Parsed testcases at query #8
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
    var_30 = 'subfield'
    var_31 = module_0.String()
    var_32 = {var_30: var_31}
    var_33 = module_0.Object()
    var_34 = {var_9: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = module_2.Form(env=var_8, schema=var_35)
    var_37 = var_36.template_for_field(var_33)



# Parsed testcases at query #9
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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'description'
    var_14 = 'active'
    var_15 = 'type'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'A'
    var_23 = 'B'
    var_24 = 'C'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_21, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = str(var_29)



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
    var_12 = 'email_field'
    var_13 = 'password_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'default_field'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'password'
    var_23 = module_2.String(format=var_22)
    var_24 = 'number'
    var_25 = module_2.String(format=var_24)
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = module_2.Boolean()
    var_32 = module_2.String()
    var_33 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_30, var_16: var_31, var_17: var_32}
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
    var_46 = '123'
    var_47 = var_35.render_field(field_name=var_14, field=var_45, value=var_46)
    var_48 = var_34.fields[var_15]
    var_49 = var_35.render_field(field_name=var_15, field=var_48, value=var_26)
    var_50 = var_34.fields[var_16]
    var_51 = True
    var_52 = var_35.render_field(field_name=var_16, field=var_50, value=var_51)
    var_53 = var_34.fields[var_16]
    var_54 = False
    var_55 = var_35.render_field(field_name=var_16, field=var_53, value=var_54)
    var_56 = var_34.fields[var_17]
    var_57 = 'default'
    var_58 = var_35.render_field(field_name=var_17, field=var_56, value=var_57)
    var_59 = var_34.fields[var_11]
    var_60 = 'Invalid value'
    var_61 = var_35.render_field(field_name=var_11, field=var_59, value=var_37, error=var_60)



# Parsed testcases at query #11
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
    var_30 = 'Developer'
    var_31 = True
    var_32 = {var_11: var_28, var_12: var_29, var_13: var_30, var_14: var_31, var_15: var_22}
    var_33 = module_4.Form(env=var_10, schema=var_27, values=var_32)
    var_34 = var_33.render_fields()



# Parsed testcases at query #12
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
    var_4 = 'choice_field'
    var_5 = 'bool_field'
    var_6 = 'password_field'
    var_7 = 'text'
    var_8 = module_2.String(format=var_7)
    var_9 = 'email'
    var_10 = module_2.String(format=var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_2.Choice(choices=var_14)
    var_16 = module_2.Boolean()
    var_17 = 'password'
    var_18 = module_2.String(format=var_17)
    var_19 = {var_2: var_8, var_3: var_10, var_4: var_15, var_5: var_16, var_6: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = 'test'
    var_22 = 'test@example.com'
    var_23 = True
    var_24 = 'secret'
    var_25 = {var_2: var_21, var_3: var_22, var_4: var_11, var_5: var_23, var_6: var_24}
    var_26 = module_4.Form(env=var_1, schema=var_20, values=var_25)
    var_27 = var_20.fields[var_2]
    var_28 = None
    var_29 = var_26.render_field(field_name=var_2, field=var_27, value=var_21, error=var_28)
    var_30 = var_20.fields[var_3]
    var_31 = var_26.render_field(field_name=var_3, field=var_30, value=var_22, error=var_28)
    var_32 = var_20.fields[var_4]
    var_33 = var_26.render_field(field_name=var_4, field=var_32, value=var_11, error=var_28)
    var_34 = var_20.fields[var_5]
    var_35 = var_26.render_field(field_name=var_5, field=var_34, value=var_23, error=var_28)
    var_36 = var_20.fields[var_6]
    var_37 = var_26.render_field(field_name=var_6, field=var_36, value=var_24, error=var_28)
    var_38 = var_20.fields[var_2]
    var_39 = 'Invalid value'
    var_40 = var_26.render_field(field_name=var_2, field=var_38, value=var_21, error=var_39)



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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
    var_4 = 'password_field'
    var_5 = 'number_field'
    var_6 = 'choice_field'
    var_7 = 'bool_field'
    var_8 = 'text_area_field'
    var_9 = module_2.String()
    var_10 = 'email'
    var_11 = module_2.String(format=var_10)
    var_12 = 'password'
    var_13 = module_2.String(format=var_12)
    var_14 = 'number'
    var_15 = module_2.String(format=var_14)
    var_16 = 'a'
    var_17 = 'A'
    var_18 = (var_16, var_17)
    var_19 = 'b'
    var_20 = 'B'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = module_2.Choice(choices=var_22)
    var_24 = module_2.Boolean()
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = {var_2: var_9, var_3: var_11, var_4: var_13, var_5: var_15, var_6: var_23, var_7: var_24, var_8: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_1, schema=var_28)
    var_30 = var_28.fields[var_2]
    var_31 = 'test'
    var_32 = var_29.render_field(field_name=var_2, field=var_30, value=var_31)
    var_33 = var_28.fields[var_3]
    var_34 = 'test@example.com'
    var_35 = var_29.render_field(field_name=var_3, field=var_33, value=var_34)
    var_36 = var_28.fields[var_4]
    var_37 = 'secret'
    var_38 = var_29.render_field(field_name=var_4, field=var_36, value=var_37)
    var_39 = var_28.fields[var_5]
    var_40 = 42
    var_41 = var_29.render_field(field_name=var_5, field=var_39, value=var_40)
    var_42 = var_28.fields[var_6]
    var_43 = var_29.render_field(field_name=var_6, field=var_42)
    var_44 = var_28.fields[var_7]
    var_45 = True
    var_46 = var_29.render_field(field_name=var_7, field=var_44, value=var_45)
    var_47 = var_28.fields[var_7]
    var_48 = False
    var_49 = var_29.render_field(field_name=var_7, field=var_47, value=var_48)
    var_50 = var_28.fields[var_8]
    var_51 = 'multiline text'
    var_52 = var_29.render_field(field_name=var_8, field=var_50, value=var_51)
    var_53 = var_28.fields[var_2]
    var_54 = 'error-message'
    var_55 = var_29.render_field(field_name=var_2, field=var_53, error=var_54)



# Parsed testcases at query #16
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
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {}
    var_7 = module_2.DictLoader(var_6)
    var_8 = module_3.Environment(loader=var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_4.Form(env=var_8, schema=var_5, values=var_11)
    var_13 = 'Jane'
    var_14 = 25
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = var_12.validate(var_15)
    var_17 = {var_0: var_9, var_1: var_10}
    var_18 = module_4.Form(env=var_8, schema=var_5, values=var_17)
    var_19 = ''
    var_20 = 'invalid'
    var_21 = {var_0: var_19, var_1: var_20}
    var_22 = var_18.validate(var_21)
    var_23 = {var_0: var_9, var_1: var_10}
    var_24 = module_4.Form(env=var_8, schema=var_5, values=var_23)
    var_25 = {var_0: var_13, var_1: var_14}
    var_26 = var_24.validate(var_25)
    var_27 = 'name'
    var_28 = 'age'
    var_29 = 'Jane'
    var_30 = 25
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = var_24.validate(var_31)



# Parsed testcases at query #17
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
    var_12 = module_0.Object()



# Parsed testcases at query #18
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
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.BaseLoader()
    var_7 = module_3.Environment(loader=var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = module_4.Form(env=var_7, schema=var_5, values=var_10)
    var_12 = 'Jane'
    var_13 = 25
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = {var_0: var_8, var_1: var_9}
    var_17 = module_4.Form(env=var_7, schema=var_5, values=var_16)
    var_18 = 'invalid'
    var_19 = {var_0: var_12, var_1: var_18}
    var_20 = var_17.validate(var_19)
    var_21 = {var_0: var_8, var_1: var_9}
    var_22 = module_4.Form(env=var_7, schema=var_5, values=var_21)
    var_23 = {var_0: var_12, var_1: var_13}
    var_24 = var_22.validate(var_23)
    var_25 = 'name'
    var_26 = 'age'
    var_27 = 'Jane'
    var_28 = 25
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = var_22.validate(var_29)



# Parsed testcases at query #19
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
    var_30 = 'nested'
    var_31 = module_0.String()
    var_32 = {var_30: var_31}
    var_33 = module_0.Object()
    var_34 = {var_9: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = module_2.Form(env=var_8, schema=var_35)
    var_37 = var_36.template_for_field(var_33)



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
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.BaseLoader()
    var_7 = module_3.Environment(loader=var_6)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = module_4.Form(env=var_7, schema=var_5, values=var_10)
    var_12 = 'Jane'
    var_13 = 25
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = {var_0: var_8, var_1: var_9}
    var_17 = module_4.Form(env=var_7, schema=var_5, values=var_16)
    var_18 = ''
    var_19 = 'invalid'
    var_20 = {var_0: var_18, var_1: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = {var_0: var_8, var_1: var_9}
    var_23 = module_4.Form(env=var_7, schema=var_5, values=var_22)
    var_24 = {var_0: var_12, var_1: var_13}
    var_25 = var_23.validate(var_24)
    var_26 = 'name'
    var_27 = 'age'
    var_28 = 'Jane'
    var_29 = 25
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = var_23.validate(var_30)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'text_field'
    var_2 = 'email_field'
    var_3 = 'number_field'
    var_4 = 'password_field'
    var_5 = 'choice_field'
    var_6 = 'boolean_field'
    var_7 = 'text'
    var_8 = module_1.String(format=var_7)
    var_9 = 'email'
    var_10 = module_1.String(format=var_9)
    var_11 = 'number'
    var_12 = module_1.String(format=var_11)
    var_13 = 'password'
    var_14 = module_1.String(format=var_13)
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'c'
    var_18 = [var_15, var_16, var_17]
    var_19 = module_1.Choice(choices=var_18)
    var_20 = module_1.Boolean()
    var_21 = {var_1: var_8, var_2: var_10, var_3: var_12, var_4: var_14, var_5: var_19, var_6: var_20}
    var_22 = module_2.Schema(var_21)
    var_23 = module_3.Form(env=var_0, schema=var_22)
    var_24 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">\n    '
    var_25 = var_22.fields[var_1]
    var_26 = 'test'
    var_27 = var_23.render_field(field_name=var_1, field=var_25, value=var_26)
    var_28 = var_22.fields[var_2]
    var_29 = 'test@example.com'
    var_30 = var_23.render_field(field_name=var_2, field=var_28, value=var_29)
    var_31 = var_22.fields[var_3]
    var_32 = '123'
    var_33 = var_23.render_field(field_name=var_3, field=var_31, value=var_32)
    var_34 = var_22.fields[var_4]
    var_35 = 'secret'
    var_36 = var_23.render_field(field_name=var_4, field=var_34, value=var_35)
    var_37 = '\n        <select name="{{ field_name }}" id="{{ field_id }}">\n            {% for choice in field.choices %}\n            <option value="{{ choice }}">{{ choice }}</option>\n            {% endfor %}\n        </select>\n    '
    var_38 = var_22.fields[var_5]
    var_39 = var_23.render_field(field_name=var_5, field=var_38)
    var_40 = '\n        <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>\n    '
    var_41 = var_22.fields[var_6]
    var_42 = True
    var_43 = var_23.render_field(field_name=var_6, field=var_41, value=var_42)
    var_44 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">\n        {% if error %}<div class="error">{{ error }}</div>{% endif %}\n    '
    var_45 = var_22.fields[var_1]
    var_46 = 'Invalid value'
    var_47 = var_23.render_field(field_name=var_1, field=var_45, value=var_26, error=var_46)



# Parsed testcases at query #23
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



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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
    var_18 = 'test'
    var_19 = var_16.render_field(field_name=var_2, field=var_17, value=var_18)
    var_20 = var_15.fields[var_3]
    var_21 = var_16.render_field(field_name=var_3, field=var_20)
    var_22 = var_15.fields[var_4]
    var_23 = var_16.render_field(field_name=var_4, field=var_22)
    var_24 = var_15.fields[var_5]
    var_25 = var_16.render_field(field_name=var_5, field=var_24, value=var_18)
    var_26 = 'password'
    var_27 = module_2.String(format=var_26)
    var_28 = 'secret'
    var_29 = var_16.render_field(field_name=var_26, field=var_27, value=var_28)
    var_30 = var_15.fields[var_2]
    var_31 = 'Invalid'
    var_32 = var_16.render_field(field_name=var_2, field=var_30, error=var_31)



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
    var_25 = 'test'
    var_26 = {var_11: var_25}
    var_27 = module_4.Form(env=var_10, schema=var_24, values=var_26)
    var_28 = var_24.fields[var_11]
    var_29 = var_27.render_field(field_name=var_11, field=var_28, value=var_25)
    var_30 = var_24.fields[var_12]
    var_31 = var_27.render_field(field_name=var_12, field=var_30, value=var_16)
    var_32 = var_24.fields[var_13]
    var_33 = True
    var_34 = var_27.render_field(field_name=var_13, field=var_32, value=var_33)
    var_35 = var_24.fields[var_14]
    var_36 = 'long text'
    var_37 = var_27.render_field(field_name=var_14, field=var_35, value=var_36)
    var_38 = var_24.fields[var_11]
    var_39 = 'Invalid'
    var_40 = var_27.render_field(field_name=var_11, field=var_38, value=var_25, error=var_39)
    var_41 = 'password'
    var_42 = module_2.String(format=var_41)
    var_43 = 'secret'
    var_44 = var_27.render_field(field_name=var_41, field=var_42, value=var_43)



# Parsed testcases at query #27
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
    var_12 = 'password_field'
    var_13 = 'email_field'
    var_14 = 'number_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    var_19 = 'password'
    var_20 = module_2.String(format=var_19)
    var_21 = 'email'
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
    var_40 = 'secret'
    var_41 = var_35.render_field(field_name=var_12, field=var_39, value=var_40)
    var_42 = var_34.fields[var_13]
    var_43 = 'test@example.com'
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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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
    var_47 = '42'
    var_48 = var_36.render_field(field_name=var_14, field=var_46, value=var_47)
    var_49 = var_34.fields[var_15]
    var_50 = var_36.render_field(field_name=var_15, field=var_49, value=var_25)
    var_51 = var_34.fields[var_16]
    var_52 = True
    var_53 = var_36.render_field(field_name=var_16, field=var_51, value=var_52)
    var_54 = var_34.fields[var_16]
    var_55 = False
    var_56 = var_36.render_field(field_name=var_16, field=var_54, value=var_55)
    var_57 = var_34.fields[var_17]
    var_58 = 'long text'
    var_59 = var_36.render_field(field_name=var_17, field=var_57, value=var_58)
    var_60 = var_34.fields[var_11]
    var_61 = 'Invalid'
    var_62 = var_36.render_field(field_name=var_11, field=var_60, value=var_38, error=var_61)



# Parsed testcases at query #30
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
    var_4 = 'active'
    var_5 = 'gender'
    var_6 = module_0.String()
    var_7 = 'number'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.String(format=var_2)
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = module_0.Boolean()
    var_13 = 'M'
    var_14 = 'F'
    var_15 = 'O'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = {var_0: var_6, var_1: var_8, var_2: var_9, var_3: var_11, var_4: var_12, var_5: var_17}
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
    var_36 = {var_0: var_31, var_1: var_32, var_2: var_33, var_3: var_34, var_4: var_35, var_5: var_13}
    var_37 = module_4.Form(env=var_30, schema=var_19, values=var_36)
    var_38 = {var_0: var_31, var_1: var_32, var_2: var_33, var_3: var_34, var_4: var_35, var_5: var_13}
    var_39 = var_37.validate(var_38)
    var_40 = var_37.render_fields()
    var_41 = module_4.Form(env=var_30, schema=var_19)
    var_42 = ''
    var_43 = 'invalid'
    var_44 = {var_0: var_42, var_1: var_43, var_2: var_43, var_3: var_42, var_4: var_43, var_5: var_43}
    var_45 = var_41.validate(var_44)
    var_46 = var_41.render_fields()
    var_47 = 'readonly_field'
    var_48 = module_0.String()
    var_49 = {var_47: var_48}
    var_50 = module_1.Schema(var_49)
    var_51 = module_4.Form(env=var_30, schema=var_50)
    var_52 = {}
    var_53 = var_51.validate(var_52)
    var_54 = var_51.render_fields()



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'my_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #33
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
    var_14 = 'textarea_field'
    var_15 = 'select_field'
    var_16 = 'checkbox_field'
    var_17 = module_2.String()
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = 'password'
    var_21 = module_2.String(format=var_20)
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = module_2.Boolean()
    var_30 = {var_11: var_17, var_12: var_19, var_13: var_21, var_14: var_23, var_15: var_28, var_16: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_4.Form(env=var_10, schema=var_31)
    var_33 = var_31.fields[var_11]
    var_34 = 'test'
    var_35 = var_32.render_field(field_name=var_11, field=var_33, value=var_34)
    var_36 = var_31.fields[var_12]
    var_37 = '123'
    var_38 = var_32.render_field(field_name=var_12, field=var_36, value=var_37)
    var_39 = var_31.fields[var_13]
    var_40 = 'secret'
    var_41 = var_32.render_field(field_name=var_13, field=var_39, value=var_40)
    var_42 = var_31.fields[var_14]
    var_43 = 'long text'
    var_44 = var_32.render_field(field_name=var_14, field=var_42, value=var_43)
    var_45 = var_31.fields[var_15]
    var_46 = var_32.render_field(field_name=var_15, field=var_45)
    var_47 = var_31.fields[var_16]
    var_48 = False
    var_49 = var_32.render_field(field_name=var_16, field=var_47, value=var_48)
    var_50 = var_31.fields[var_16]
    var_51 = True
    var_52 = var_32.render_field(field_name=var_16, field=var_50, value=var_51)
    var_53 = var_31.fields[var_11]
    var_54 = 'Invalid'
    var_55 = var_32.render_field(field_name=var_11, field=var_53, value=var_34, error=var_54)



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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



# Parsed testcases at query #37
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
    var_8 = 'A'
    var_9 = (var_7, var_8)
    var_10 = 'b'
    var_11 = 'B'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = module_2.Choice(choices=var_13)
    var_15 = module_2.Boolean()
    var_16 = 'text'
    var_17 = module_2.String(format=var_16)
    var_18 = {var_2: var_6, var_3: var_14, var_4: var_15, var_5: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = 'test'
    var_21 = {var_2: var_20}
    var_22 = module_4.Form(env=var_1, schema=var_19, values=var_21)
    var_23 = var_19.fields[var_2]
    var_24 = 'test_value'
    var_25 = None
    var_26 = var_22.render_field(field_name=var_2, field=var_23, value=var_24, error=var_25)
    var_27 = var_19.fields[var_3]
    var_28 = var_22.render_field(field_name=var_3, field=var_27, value=var_7, error=var_25)
    var_29 = var_19.fields[var_4]
    var_30 = True
    var_31 = var_22.render_field(field_name=var_4, field=var_29, value=var_30, error=var_25)
    var_32 = var_19.fields[var_5]
    var_33 = 'long text'
    var_34 = var_22.render_field(field_name=var_5, field=var_32, value=var_33, error=var_25)
    var_35 = var_19.fields[var_2]
    var_36 = 'Invalid value'
    var_37 = var_22.render_field(field_name=var_2, field=var_35, value=var_24, error=var_36)
    var_38 = 'password'
    var_39 = module_2.String(format=var_38)
    var_40 = 'secret'
    var_41 = var_22.render_field(field_name=var_38, field=var_39, value=var_40, error=var_25)



# Parsed testcases at query #38
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



# Parsed testcases at query #39
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



# Parsed testcases at query #40
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
    var_54 = var_34.fields[var_16]
    var_55 = False
    var_56 = var_36.render_field(field_name=var_16, field=var_54, value=var_55)
    var_57 = var_34.fields[var_17]
    var_58 = 'long text'
    var_59 = var_36.render_field(field_name=var_17, field=var_57, value=var_58)
    var_60 = var_34.fields[var_11]
    var_61 = 'Invalid'
    var_62 = var_36.render_field(field_name=var_11, field=var_60, value=var_38, error=var_61)



# Parsed testcases at query #41
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #42
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



# Parsed testcases at query #43
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



# Parsed testcases at query #44
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



# Parsed testcases at query #45
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
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = module_2.Boolean()
    var_32 = module_2.String(format=var_18)
    var_33 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_30, var_16: var_31, var_17: var_32}
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
    var_49 = var_35.render_field(field_name=var_15, field=var_48, value=var_26)
    var_50 = var_34.fields[var_16]
    var_51 = True
    var_52 = var_35.render_field(field_name=var_16, field=var_50, value=var_51)
    var_53 = var_34.fields[var_17]
    var_54 = 'long text'
    var_55 = var_35.render_field(field_name=var_17, field=var_53, value=var_54)
    var_56 = var_34.fields[var_11]
    var_57 = 'Invalid'
    var_58 = var_35.render_field(field_name=var_11, field=var_56, value=var_37, error=var_57)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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
    var_16 = 100
    var_17 = module_2.String(max_length=var_16)
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'M'
    var_24 = 'F'
    var_25 = [var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_17, var_12: var_19, var_13: var_21, var_14: var_22, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'Test'
    var_32 = True
    var_33 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_23}
    var_34 = module_4.Form(env=var_10, schema=var_28, values=var_33)
    var_35 = var_34.render_fields()
    var_36 = 'Jane'
    var_37 = '25'
    var_38 = 'New test'
    var_39 = False
    var_40 = {var_11: var_36, var_12: var_37, var_13: var_38, var_14: var_39, var_15: var_24}
    var_41 = var_34.validate(var_40)
    var_42 = var_34.render_fields()
    var_43 = ''
    var_44 = 'invalid'
    var_45 = 'X'
    var_46 = {var_11: var_43, var_12: var_44, var_13: var_31, var_14: var_32, var_15: var_45}
    var_47 = var_34.validate(var_46)
    var_48 = var_34.render_fields()



# Parsed testcases at query #48
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



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'email'
    var_14 = 'age'
    var_15 = 'bio'
    var_16 = 'active'
    var_17 = 'text'
    var_18 = module_2.String(format=var_17)
    var_19 = module_2.String(format=var_13)
    var_20 = 'number'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.String(format=var_17)
    var_23 = module_2.Boolean()
    var_24 = {var_12: var_18, var_13: var_19, var_14: var_21, var_15: var_22, var_16: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = 'John'
    var_27 = 'john@example.com'
    var_28 = '30'
    var_29 = 'Developer'
    var_30 = {var_12: var_26, var_13: var_27, var_14: var_28, var_15: var_29, var_16: var_10}
    var_31 = module_4.Form(env=var_11, schema=var_25, values=var_30)
    var_32 = var_31.render_fields()
    var_33 = ''
    var_34 = 'invalid'
    var_35 = 'abc'
    var_36 = 'not_bool'
    var_37 = {var_12: var_33, var_13: var_34, var_14: var_35, var_15: var_33, var_16: var_36}
    var_38 = var_31.validate(var_37)
    var_39 = var_31.render_fields()



# Parsed testcases at query #51
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
    var_24 = 'O'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_21, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'Test bio'
    var_32 = True
    var_33 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_22}
    var_34 = module_4.Form(env=var_10, schema=var_28, values=var_33)
    var_35 = var_34.render_fields()
    var_36 = ''
    var_37 = 'invalid'
    var_38 = 'not_bool'
    var_39 = 'X'
    var_40 = {var_11: var_36, var_12: var_37, var_13: var_36, var_14: var_38, var_15: var_39}
    var_41 = var_34.validate(var_40)
    var_42 = var_34.render_fields()
    var_43 = 'readonly_field'
    var_44 = module_2.String()
    var_45 = {var_43: var_44}
    var_46 = module_3.Schema(var_45)
    var_47 = module_4.Form(env=var_10, schema=var_46)
    var_48 = var_47.render_fields()
    assert var_48 == ''



# Parsed testcases at query #52
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



# Parsed testcases at query #53
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



# Parsed testcases at query #54
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



# Parsed testcases at query #55
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'my_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #56
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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'email'
    var_14 = 'agree'
    var_15 = 'country'
    var_16 = 'text'
    var_17 = module_2.String(format=var_16)
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = module_2.Boolean()
    var_22 = 'US'
    var_23 = 'UK'
    var_24 = 'CA'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_17, var_12: var_19, var_13: var_20, var_14: var_21, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'john@example.com'
    var_32 = True
    var_33 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_22}
    var_34 = module_4.Form(env=var_10, schema=var_28, values=var_33)
    var_35 = var_34.render_fields()
    var_36 = 'Jane'
    var_37 = '25'
    var_38 = 'jane@example.com'
    var_39 = False
    var_40 = {var_11: var_36, var_12: var_37, var_13: var_38, var_14: var_39, var_15: var_23}
    var_41 = var_34.validate(var_40)
    var_42 = var_34.render_fields()
    var_43 = ''
    var_44 = 'invalid'
    var_45 = {var_11: var_43, var_12: var_44, var_13: var_44, var_15: var_44}
    var_46 = var_34.validate(var_45)
    var_47 = var_34.render_fields()



# Parsed testcases at query #57
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



# Parsed testcases at query #58
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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'email'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'status'
    var_17 = module_2.String()
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = module_2.Boolean()
    var_24 = 'inactive'
    var_25 = [var_15, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_17, var_12: var_19, var_13: var_20, var_14: var_22, var_15: var_23, var_16: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'john@example.com'
    var_32 = 'Test bio'
    var_33 = True
    var_34 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_33, var_16: var_15}
    var_35 = module_4.Form(env=var_10, schema=var_28, values=var_34)
    var_36 = var_35.render_fields()
    var_37 = ''
    var_38 = 'invalid'
    var_39 = 'not-boolean'
    var_40 = {var_11: var_37, var_12: var_38, var_13: var_38, var_14: var_37, var_15: var_39, var_16: var_38}
    var_41 = var_35.validate(var_40)
    var_42 = var_35.render_fields()
    var_43 = 'readonly_field'
    var_44 = module_2.String()
    var_45 = {var_43: var_44}
    var_46 = module_3.Schema(var_45)
    var_47 = 'test'
    var_48 = {var_43: var_47}
    var_49 = module_4.Form(env=var_10, schema=var_46, values=var_48)
    var_50 = var_49.render_fields()



# Parsed testcases at query #59
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
    var_4 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_5 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_6 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox'>"
    var_7 = "<textarea id='{{ field_id }}' name='{{ field_name }}'></textarea>"
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'email'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'gender'
    var_17 = module_2.String()
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = module_2.String(format=var_13)
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = module_2.Boolean()
    var_24 = 'M'
    var_25 = 'F'
    var_26 = 'O'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = {var_11: var_17, var_12: var_19, var_13: var_20, var_14: var_22, var_15: var_23, var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'john@example.com'
    var_34 = 'Hello'
    var_35 = True
    var_36 = {var_11: var_31, var_12: var_32, var_13: var_33, var_14: var_34, var_15: var_35, var_16: var_24}
    var_37 = module_4.Form(env=var_10, schema=var_30, values=var_36)
    var_38 = var_37.render_fields()
    var_39 = ''
    var_40 = 'invalid'
    var_41 = 'not-boolean'
    var_42 = {var_11: var_39, var_12: var_40, var_13: var_40, var_14: var_39, var_15: var_41, var_16: var_40}
    var_43 = var_37.validate(var_42)
    var_44 = var_37.render_fields()



# Parsed testcases at query #60
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
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'choice_field'
    var_16 = 'bool_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'number'
    var_22 = module_2.String(format=var_21)
    var_23 = 'password'
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
    var_43 = '123'
    var_44 = var_35.render_field(field_name=var_13, field=var_42, value=var_43)
    var_45 = var_34.fields[var_14]
    var_46 = 'secret'
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



# Parsed testcases at query #61
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
    var_3 = 'number_field'
    var_4 = 'choice_field'
    var_5 = 'bool_field'
    var_6 = 'text_area'
    var_7 = 'password_field'
    var_8 = module_2.String()
    var_9 = 'number'
    var_10 = module_2.String(format=var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_11, var_12]
    var_14 = module_2.Choice(choices=var_13)
    var_15 = module_2.Boolean()
    var_16 = 'text'
    var_17 = module_2.String(format=var_16)
    var_18 = 'password'
    var_19 = module_2.String(format=var_18)
    var_20 = {var_2: var_8, var_3: var_10, var_4: var_14, var_5: var_15, var_6: var_17, var_7: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_4.Form(env=var_1, schema=var_21)
    var_23 = var_21.fields[var_2]
    var_24 = 'test_value'
    var_25 = var_22.render_field(field_name=var_2, field=var_23, value=var_24)
    var_26 = var_21.fields[var_3]
    var_27 = '123'
    var_28 = var_22.render_field(field_name=var_3, field=var_26, value=var_27)
    var_29 = var_21.fields[var_4]
    var_30 = var_22.render_field(field_name=var_4, field=var_29, value=var_11)
    var_31 = var_21.fields[var_5]
    var_32 = True
    var_33 = var_22.render_field(field_name=var_5, field=var_31, value=var_32)
    var_34 = var_21.fields[var_6]
    var_35 = 'long text'
    var_36 = var_22.render_field(field_name=var_6, field=var_34, value=var_35)
    var_37 = var_21.fields[var_7]
    var_38 = 'secret'
    var_39 = var_22.render_field(field_name=var_7, field=var_37, value=var_38)
    var_40 = var_21.fields[var_2]
    var_41 = 'Invalid value'
    var_42 = var_22.render_field(field_name=var_2, field=var_40, value=var_24, error=var_41)



# Parsed testcases at query #62
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
    var_4 = 'agree'
    var_5 = 'country'
    var_6 = module_0.String()
    var_7 = 'number'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.String(format=var_2)
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = module_0.Boolean()
    var_13 = 'US'
    var_14 = 'UK'
    var_15 = 'CA'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = {var_0: var_6, var_1: var_8, var_2: var_9, var_3: var_11, var_4: var_12, var_5: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 'forms/input.html'
    var_21 = 'forms/textarea.html'
    var_22 = 'forms/select.html'
    var_23 = 'forms/checkbox.html'
    var_24 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_25 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_26 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_27 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_28 = {var_20: var_24, var_21: var_25, var_22: var_26, var_23: var_27}
    var_29 = module_2.DictLoader(var_28)
    var_30 = module_3.Environment(loader=var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'john@example.com'
    var_34 = 'Hello'
    var_35 = True
    var_36 = {var_0: var_31, var_1: var_32, var_2: var_33, var_3: var_34, var_4: var_35, var_5: var_13}
    var_37 = module_4.Form(env=var_30, schema=var_19, values=var_36)
    var_38 = var_37.render_fields()
    var_39 = ''
    var_40 = 'invalid'
    var_41 = False
    var_42 = 'INVALID'
    var_43 = {var_0: var_39, var_1: var_40, var_2: var_40, var_3: var_39, var_4: var_41, var_5: var_42}
    var_44 = var_37.validate(var_43)
    var_45 = var_37.render_fields()



# Parsed testcases at query #63
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



# Parsed testcases at query #64
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
    var_11 = var_8.env.loader.loaders
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_8.env.loader.loaders[var_13]
    var_15 = 1
    var_16 = var_8.env.loader.loaders[var_15]



# Parsed testcases at query #65
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
    var_38 = 'X'
    var_39 = {var_11: var_35, var_12: var_36, var_13: var_35, var_14: var_37, var_15: var_38}
    var_40 = var_33.validate(var_39)
    var_41 = var_33.render_fields()
    var_42 = 'readonly_field'
    var_43 = module_2.String()
    var_44 = {var_42: var_43}
    var_45 = module_3.Schema(var_44)
    var_46 = module_4.Form(env=var_10, schema=var_45)
    var_47 = var_46.render_fields()
    assert var_47 == ''



# Parsed testcases at query #66
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
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'age'
    var_14 = 'bio'
    var_15 = 'country'
    var_16 = 'subscribe'
    var_17 = module_2.String()
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = 'US'
    var_23 = 'UK'
    var_24 = 'CA'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = module_2.Boolean()
    var_28 = {var_12: var_17, var_13: var_19, var_14: var_21, var_15: var_26, var_16: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = 'John'
    var_31 = '30'
    var_32 = 'Developer'
    var_33 = {var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_22, var_16: var_10}
    var_34 = module_4.Form(env=var_11, schema=var_29, values=var_33)
    var_35 = var_34.render_fields()
    var_36 = ''
    var_37 = 'invalid'
    var_38 = 'Short'
    var_39 = 'XX'
    var_40 = 'not-boolean'
    var_41 = {var_12: var_36, var_13: var_37, var_14: var_38, var_15: var_39, var_16: var_40}
    var_42 = var_34.validate(var_41)
    var_43 = var_34.render_fields()



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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



# Parsed testcases at query #70
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



# Parsed testcases at query #71
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
    var_3 = 'number_field'
    var_4 = 'password_field'
    var_5 = 'choice_field'
    var_6 = 'boolean_field'
    var_7 = 'text_area_field'
    var_8 = module_2.String()
    var_9 = 'number'
    var_10 = module_2.String(format=var_9)
    var_11 = 'password'
    var_12 = module_2.String(format=var_11)
    var_13 = '1'
    var_14 = 'Option 1'
    var_15 = (var_13, var_14)
    var_16 = '2'
    var_17 = 'Option 2'
    var_18 = (var_16, var_17)
    var_19 = [var_15, var_18]
    var_20 = module_2.Choice(choices=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = {var_2: var_8, var_3: var_10, var_4: var_12, var_5: var_20, var_6: var_21, var_7: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = module_4.Form(env=var_1, schema=var_25)
    var_27 = var_25.fields[var_2]
    var_28 = 'test'
    var_29 = var_26.render_field(field_name=var_2, field=var_27, value=var_28)
    var_30 = var_25.fields[var_3]
    var_31 = '123'
    var_32 = var_26.render_field(field_name=var_3, field=var_30, value=var_31)
    var_33 = var_25.fields[var_4]
    var_34 = 'secret'
    var_35 = var_26.render_field(field_name=var_4, field=var_33, value=var_34)
    var_36 = var_25.fields[var_5]
    var_37 = var_26.render_field(field_name=var_5, field=var_36, value=var_13)
    var_38 = var_25.fields[var_6]
    var_39 = True
    var_40 = var_26.render_field(field_name=var_6, field=var_38, value=var_39)
    var_41 = var_25.fields[var_7]
    var_42 = 'long text'
    var_43 = var_26.render_field(field_name=var_7, field=var_41, value=var_42)
    var_44 = var_25.fields[var_2]
    var_45 = 'Invalid value'
    var_46 = var_26.render_field(field_name=var_2, field=var_44, value=var_28, error=var_45)



# Parsed testcases at query #72
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



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'name'
    var_13 = 'age'
    var_14 = 'bio'
    var_15 = 'gender'
    var_16 = 'active'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = 'number'
    var_20 = module_2.String(format=var_19)
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = 'M'
    var_24 = 'F'
    var_25 = 'O'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_2.Choice(choices=var_26)
    var_28 = module_2.Boolean()
    var_29 = {var_12: var_18, var_13: var_20, var_14: var_22, var_15: var_27, var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'Developer'
    var_34 = {var_12: var_31, var_13: var_32, var_14: var_33, var_15: var_23, var_16: var_10}
    var_35 = module_4.Form(env=var_11, schema=var_30, values=var_34)
    var_36 = var_35.render_fields()



# Parsed testcases at query #75
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
    var_4 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_5 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_6 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_7 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'username'
    var_12 = 'password'
    var_13 = 'age'
    var_14 = 'bio'
    var_15 = 'country'
    var_16 = 'subscribe'
    var_17 = 'text'
    var_18 = module_2.String(format=var_17)
    var_19 = module_2.String(format=var_12)
    var_20 = 'number'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.String(format=var_17)
    var_23 = 'US'
    var_24 = 'UK'
    var_25 = 'CA'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_2.Choice(choices=var_26)
    var_28 = module_2.Boolean()
    var_29 = {var_11: var_18, var_12: var_19, var_13: var_21, var_14: var_22, var_15: var_27, var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = 'test'
    var_32 = '25'
    var_33 = 'Hello'
    var_34 = True
    var_35 = {var_11: var_31, var_13: var_32, var_14: var_33, var_15: var_23, var_16: var_34}
    var_36 = module_4.Form(env=var_10, schema=var_30, values=var_35)
    var_37 = var_30.fields[var_11]
    var_38 = var_36.render_field(field_name=var_11, field=var_37, value=var_31)
    var_39 = var_30.fields[var_12]
    var_40 = 'secret'
    var_41 = var_36.render_field(field_name=var_12, field=var_39, value=var_40)
    var_42 = var_30.fields[var_13]
    var_43 = var_36.render_field(field_name=var_13, field=var_42, value=var_32)
    var_44 = var_30.fields[var_14]
    var_45 = var_36.render_field(field_name=var_14, field=var_44, value=var_33)
    var_46 = var_30.fields[var_15]
    var_47 = var_36.render_field(field_name=var_15, field=var_46, value=var_23)
    var_48 = var_30.fields[var_16]
    var_49 = var_36.render_field(field_name=var_16, field=var_48, value=var_34)
    var_50 = var_30.fields[var_11]
    var_51 = 'Invalid username'
    var_52 = var_36.render_field(field_name=var_11, field=var_50, value=var_31, error=var_51)



# Parsed testcases at query #76
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_6 = var_3.input_type_for_field(var_4)
    var_7 = 'unsupported'
    var_8 = module_3.String(format=var_7)
    var_9 = var_3.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = 'color'
    var_11 = module_3.Boolean()
    var_12 = var_3.input_type_for_field(var_11)
    assert var_12 == 'text'



# Parsed testcases at query #2
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
    var_13 = module_0.Object()



# Parsed testcases at query #3
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
    var_15 = module_2.String()
    var_16 = 'number'
    var_17 = module_2.String(format=var_16)
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = module_2.Boolean()
    var_21 = {var_11: var_15, var_12: var_17, var_13: var_19, var_14: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = 'Test'
    var_24 = '25'
    var_25 = 'Test description'
    var_26 = True
    var_27 = {var_11: var_23, var_12: var_24, var_13: var_25, var_14: var_26}
    var_28 = module_4.Form(env=var_10, schema=var_22, values=var_27)
    var_29 = str(var_28)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'bio'
    var_3 = 'agree'
    var_4 = 'country'
    var_5 = module_0.String()
    var_6 = 'number'
    var_7 = module_0.String(format=var_6)
    var_8 = 'text'
    var_9 = module_0.String(format=var_8)
    var_10 = module_0.Boolean()
    var_11 = 'US'
    var_12 = 'UK'
    var_13 = 'CA'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.Choice(choices=var_14)
    var_16 = {var_0: var_5, var_1: var_7, var_2: var_9, var_3: var_10, var_4: var_15}
    var_17 = module_1.Schema(var_16)
    var_18 = 'forms/input.html'
    var_19 = 'forms/textarea.html'
    var_20 = 'forms/select.html'
    var_21 = 'forms/checkbox.html'
    var_22 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_23 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_24 = '<select name="{{ field_name }}"></select>'
    var_25 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_26 = {var_18: var_22, var_19: var_23, var_20: var_24, var_21: var_25}
    var_27 = module_2.DictLoader(var_26)
    var_28 = True
    var_29 = module_3.Environment(autoescape=var_28, loader=var_27)
    var_30 = 'John'
    var_31 = '30'
    var_32 = 'Developer'
    var_33 = {var_0: var_30, var_1: var_31, var_2: var_32, var_3: var_28, var_4: var_11}
    var_34 = module_4.Form(env=var_29, schema=var_17, values=var_33)
    var_35 = var_34.__html__()
    var_36 = str(var_35)
    var_37 = var_34.render_fields()



# Parsed testcases at query #5
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
    var_7 = module_2.BaseLoader()
    var_8 = module_3.Environment(loader=var_7)
    var_9 = 'John'
    var_10 = '25'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_4.Form(env=var_8, schema=var_6, values=var_11)
    var_13 = 'Jane'
    var_14 = '30'
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = var_12.validate(var_15)
    var_17 = {var_0: var_9, var_1: var_10}
    var_18 = module_4.Form(env=var_8, schema=var_6, values=var_17)
    var_19 = ''
    var_20 = {var_0: var_19}
    var_21 = var_18.validate(var_20)
    var_22 = {var_0: var_9, var_1: var_10}
    var_23 = module_4.Form(env=var_8, schema=var_6, values=var_22)
    var_24 = {var_0: var_13, var_1: var_14}
    var_25 = var_23.validate(var_24)
    var_26 = 'name'
    var_27 = 'age'
    var_28 = 'Jane'
    var_29 = '30'
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = var_23.validate(var_30)



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
    var_46 = '123'
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
    var_59 = var_34.fields[var_11]
    var_60 = 'Invalid value'
    var_61 = var_35.render_field(field_name=var_11, field=var_59, value=var_37, error=var_60)



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
    var_2 = 'email'
    var_3 = 'bio'
    var_4 = 'agree'
    var_5 = 'country'
    var_6 = module_0.String()
    var_7 = 'number'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.String(format=var_2)
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = module_0.Boolean()
    var_13 = 'US'
    var_14 = 'UK'
    var_15 = 'CA'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = {var_0: var_6, var_1: var_8, var_2: var_9, var_3: var_11, var_4: var_12, var_5: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 'forms/input.html'
    var_21 = 'forms/textarea.html'
    var_22 = 'forms/select.html'
    var_23 = 'forms/checkbox.html'
    var_24 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_25 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_26 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_27 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_28 = {var_20: var_24, var_21: var_25, var_22: var_26, var_23: var_27}
    var_29 = module_2.DictLoader(var_28)
    var_30 = module_3.Environment(loader=var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'john@example.com'
    var_34 = 'Test bio'
    var_35 = True
    var_36 = {var_0: var_31, var_1: var_32, var_2: var_33, var_3: var_34, var_4: var_35, var_5: var_13}
    var_37 = module_4.Form(env=var_30, schema=var_19, values=var_36)
    var_38 = var_37.render_fields()
    var_39 = ''
    var_40 = 'invalid'
    var_41 = False
    var_42 = {var_0: var_39, var_1: var_40, var_2: var_40, var_3: var_39, var_4: var_41, var_5: var_40}
    var_43 = var_37.validate(var_42)
    var_44 = var_37.render_fields()



# Parsed testcases at query #8
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
    var_4 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_5 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_6 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_7 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'email_field'
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'choice_field'
    var_16 = 'boolean_field'
    var_17 = 'textarea_field'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'number'
    var_23 = module_2.String(format=var_22)
    var_24 = 'password'
    var_25 = module_2.String(format=var_24)
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = module_2.Boolean()
    var_32 = module_2.String(format=var_18)
    var_33 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_30, var_16: var_31, var_17: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_4.Form(env=var_10, schema=var_34)
    var_36 = var_34.fields[var_11]
    var_37 = 'test'
    var_38 = var_35.render_field(field_name=var_11, field=var_36, value=var_37)
    var_39 = var_34.fields[var_12]
    var_40 = 'test@example.com'
    var_41 = var_35.render_field(field_name=var_12, field=var_39, value=var_40)
    var_42 = var_34.fields[var_13]
    var_43 = '123'
    var_44 = var_35.render_field(field_name=var_13, field=var_42, value=var_43)
    var_45 = var_34.fields[var_14]
    var_46 = 'secret'
    var_47 = var_35.render_field(field_name=var_14, field=var_45, value=var_46)
    var_48 = var_34.fields[var_15]
    var_49 = var_35.render_field(field_name=var_15, field=var_48, value=var_26)
    var_50 = var_34.fields[var_16]
    var_51 = True
    var_52 = var_35.render_field(field_name=var_16, field=var_50, value=var_51)
    var_53 = var_34.fields[var_17]
    var_54 = 'long text'
    var_55 = var_35.render_field(field_name=var_17, field=var_53, value=var_54)
    var_56 = var_34.fields[var_11]
    var_57 = 'Invalid'
    var_58 = var_35.render_field(field_name=var_11, field=var_56, value=var_37, error=var_57)



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
    var_2 = 'forms/input.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/textarea.html'
    var_6 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_7 = '<select name="{{ field_name }}"></select>'
    var_8 = '<input type="checkbox" name="{{ field_name }}">'
    var_9 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
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
    var_25 = module_4.Form(env=var_1, schema=var_24)
    var_26 = var_24.fields[var_11]
    var_27 = 'test'
    var_28 = var_25.render_field(field_name=var_11, field=var_26, value=var_27)
    var_29 = var_24.fields[var_12]
    var_30 = var_25.render_field(field_name=var_12, field=var_29)
    var_31 = var_24.fields[var_13]
    var_32 = var_25.render_field(field_name=var_13, field=var_31)
    var_33 = var_24.fields[var_14]
    var_34 = var_25.render_field(field_name=var_14, field=var_33, value=var_27)
    var_35 = 'password'
    var_36 = module_2.String(format=var_35)
    var_37 = 'secret'
    var_38 = var_25.render_field(field_name=var_35, field=var_36, value=var_37)
    var_39 = var_24.fields[var_11]
    var_40 = 'Error message'
    var_41 = var_25.render_field(field_name=var_11, field=var_39, error=var_40)



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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
    var_17 = 'default_field'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'password'
    var_23 = module_2.String(format=var_22)
    var_24 = 'number'
    var_25 = module_2.String(format=var_24)
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = module_2.Boolean()
    var_32 = module_2.String()
    var_33 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_30, var_16: var_31, var_17: var_32}
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
    var_49 = var_35.render_field(field_name=var_15, field=var_48, value=var_26)
    var_50 = var_34.fields[var_16]
    var_51 = True
    var_52 = var_35.render_field(field_name=var_16, field=var_50, value=var_51)
    var_53 = var_34.fields[var_17]
    var_54 = 'default'
    var_55 = var_35.render_field(field_name=var_17, field=var_53, value=var_54)
    var_56 = var_34.fields[var_11]
    var_57 = 'Invalid'
    var_58 = var_35.render_field(field_name=var_11, field=var_56, value=var_37, error=var_57)



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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
    var_3 = 'username'
    var_4 = 'email'
    var_5 = 'age'
    var_6 = 'password'
    var_7 = 'bio'
    var_8 = 'agree'
    var_9 = 'country'
    var_10 = 'text'
    var_11 = module_2.String(format=var_10)
    var_12 = module_2.String(format=var_4)
    var_13 = 'number'
    var_14 = module_2.String(format=var_13)
    var_15 = module_2.String(format=var_6)
    var_16 = module_2.String(format=var_10)
    var_17 = module_2.Boolean()
    var_18 = 'US'
    var_19 = 'UK'
    var_20 = 'CA'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_2.Choice(choices=var_21)
    var_23 = {var_3: var_11, var_4: var_12, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_4.Form(env=var_2, schema=var_24)
    var_26 = var_24.fields[var_3]
    var_27 = var_25.template_for_field(var_26)
    assert var_27 == 'forms/input.html'
    var_28 = var_24.fields[var_3]
    var_29 = var_25.input_type_for_field(var_28)
    assert var_29 == 'text'
    var_30 = var_24.fields[var_4]
    var_31 = var_25.template_for_field(var_30)
    assert var_31 == 'forms/input.html'
    var_32 = var_24.fields[var_4]
    var_33 = var_25.input_type_for_field(var_32)
    assert var_33 == 'email'
    var_34 = var_24.fields[var_5]
    var_35 = var_25.template_for_field(var_34)
    assert var_35 == 'forms/input.html'
    var_36 = var_24.fields[var_5]
    var_37 = var_25.input_type_for_field(var_36)
    assert var_37 == 'number'
    var_38 = var_24.fields[var_6]
    var_39 = var_25.template_for_field(var_38)
    assert var_39 == 'forms/input.html'
    var_40 = var_24.fields[var_6]
    var_41 = var_25.input_type_for_field(var_40)
    assert var_41 == 'password'
    var_42 = var_24.fields[var_7]
    var_43 = var_25.template_for_field(var_42)
    assert var_43 == 'forms/textarea.html'
    var_44 = var_24.fields[var_7]
    var_45 = var_25.input_type_for_field(var_44)
    assert var_45 == 'text'
    var_46 = var_24.fields[var_8]
    var_47 = var_25.template_for_field(var_46)
    assert var_47 == 'forms/checkbox.html'
    var_48 = var_24.fields[var_8]
    var_49 = var_25.input_type_for_field(var_48)
    assert var_49 == 'text'
    var_50 = var_24.fields[var_9]
    var_51 = var_25.template_for_field(var_50)
    assert var_51 == 'forms/select.html'
    var_52 = var_24.fields[var_9]
    var_53 = var_25.input_type_for_field(var_52)
    assert var_53 == 'text'
    var_54 = 'forms/input.html'
    var_55 = 'forms/textarea.html'
    var_56 = 'forms/select.html'
    var_57 = 'forms/checkbox.html'
    var_58 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_59 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_60 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_61 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {{ 'checked' if value else '' }}>"
    var_62 = {var_54: var_58, var_55: var_59, var_56: var_60, var_57: var_61}
    var_63 = module_0.DictLoader(var_62)
    var_64 = module_1.Environment(autoescape=var_1, loader=var_63)
    var_65 = module_4.Form(env=var_64, schema=var_24)
    var_66 = var_24.fields[var_3]
    var_67 = 'test'
    var_68 = var_65.render_field(field_name=var_3, field=var_66, value=var_67)
    var_69 = var_24.fields[var_4]
    var_70 = 'test@example.com'
    var_71 = var_65.render_field(field_name=var_4, field=var_69, value=var_70)
    var_72 = var_24.fields[var_6]
    var_73 = 'secret'
    var_74 = var_65.render_field(field_name=var_6, field=var_72, value=var_73)
    var_75 = var_24.fields[var_7]
    var_76 = 'Long text here'
    var_77 = var_65.render_field(field_name=var_7, field=var_75, value=var_76)
    var_78 = var_24.fields[var_8]
    var_79 = var_65.render_field(field_name=var_8, field=var_78, value=var_1)
    var_80 = var_24.fields[var_9]
    var_81 = var_65.render_field(field_name=var_9, field=var_80, value=var_18)
    var_82 = var_24.fields[var_3]
    var_83 = ''
    var_84 = 'This field is required'
    var_85 = var_65.render_field(field_name=var_3, field=var_82, value=var_83, error=var_84)



# Parsed testcases at query #14
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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'email'
    var_14 = 'description'
    var_15 = 'agree'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = module_2.String(format=var_13)
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = module_2.Boolean()
    var_23 = {var_11: var_16, var_12: var_18, var_13: var_19, var_14: var_21, var_15: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = 'John'
    var_26 = '30'
    var_27 = 'john@example.com'
    var_28 = 'Test'
    var_29 = True
    var_30 = {var_11: var_25, var_12: var_26, var_13: var_27, var_14: var_28, var_15: var_29}
    var_31 = module_4.Form(env=var_10, schema=var_24, values=var_30)
    var_32 = var_31.render_fields()
    var_33 = 'Jane'
    var_34 = '25'
    var_35 = 'jane@example.com'
    var_36 = 'Updated'
    var_37 = False
    var_38 = {var_11: var_33, var_12: var_34, var_13: var_35, var_14: var_36, var_15: var_37}
    var_39 = var_31.validate(var_38)
    var_40 = var_31.render_fields()
    var_41 = ''
    var_42 = 'invalid'
    var_43 = {var_11: var_41, var_12: var_42, var_13: var_42, var_14: var_41, var_15: var_37}
    var_44 = var_31.validate(var_43)
    var_45 = var_31.render_fields()



# Parsed testcases at query #15
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



# Parsed testcases at query #16
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
    var_13 = 'choice_field'
    var_14 = 'bool_field'
    var_15 = 'text_area'
    var_16 = module_2.String()
    var_17 = 'email'
    var_18 = module_2.String(format=var_17)
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_2.Choice(choices=var_22)
    var_24 = module_2.Boolean()
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_23, var_14: var_24, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = var_28.fields[var_11]
    var_31 = 'test'
    var_32 = var_29.render_field(field_name=var_11, field=var_30, value=var_31)
    var_33 = var_28.fields[var_12]
    var_34 = 'test@example.com'
    var_35 = var_29.render_field(field_name=var_12, field=var_33, value=var_34)
    var_36 = var_28.fields[var_13]
    var_37 = var_29.render_field(field_name=var_13, field=var_36, value=var_19)
    var_38 = var_28.fields[var_14]
    var_39 = True
    var_40 = var_29.render_field(field_name=var_14, field=var_38, value=var_39)
    var_41 = var_28.fields[var_15]
    var_42 = 'long text'
    var_43 = var_29.render_field(field_name=var_15, field=var_41, value=var_42)
    var_44 = var_28.fields[var_11]
    var_45 = 'Invalid'
    var_46 = var_29.render_field(field_name=var_11, field=var_44, value=var_31, error=var_45)



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_13 = 'choice_field'
    var_14 = 'bool_field'
    var_15 = 'text_area'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'a'
    var_20 = 'b'
    var_21 = [var_19, var_20]
    var_22 = module_2.Choice(choices=var_21)
    var_23 = module_2.Boolean()
    var_24 = 'text'
    var_25 = module_2.String(format=var_24)
    var_26 = {var_11: var_16, var_12: var_18, var_13: var_22, var_14: var_23, var_15: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_4.Form(env=var_10, schema=var_27)
    var_29 = var_27.fields[var_11]
    var_30 = 'test'
    var_31 = var_28.render_field(field_name=var_11, field=var_29, value=var_30)
    var_32 = var_27.fields[var_12]
    var_33 = '123'
    var_34 = var_28.render_field(field_name=var_12, field=var_32, value=var_33)
    var_35 = var_27.fields[var_13]
    var_36 = var_28.render_field(field_name=var_13, field=var_35, value=var_19)
    var_37 = var_27.fields[var_14]
    var_38 = True
    var_39 = var_28.render_field(field_name=var_14, field=var_37, value=var_38)
    var_40 = var_27.fields[var_15]
    var_41 = 'long text'
    var_42 = var_28.render_field(field_name=var_15, field=var_40, value=var_41)
    var_43 = var_27.fields[var_11]
    var_44 = 'Invalid'
    var_45 = var_28.render_field(field_name=var_11, field=var_43, value=var_30, error=var_44)



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
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'text_field'
    var_13 = 'choice_field'
    var_14 = 'boolean_field'
    var_15 = 'text_area_field'
    var_16 = 'password_field'
    var_17 = module_2.String()
    var_18 = 'a'
    var_19 = 'b'
    var_20 = [var_18, var_19]
    var_21 = module_2.Choice(choices=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'text'
    var_24 = module_2.String(format=var_23)
    var_25 = 'password'
    var_26 = module_2.String(format=var_25)
    var_27 = {var_12: var_17, var_13: var_21, var_14: var_22, var_15: var_24, var_16: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'test'
    var_30 = 'long text'
    var_31 = 'secret'
    var_32 = {var_12: var_29, var_13: var_18, var_14: var_10, var_15: var_30, var_16: var_31}
    var_33 = module_4.Form(env=var_11, schema=var_28, values=var_32)
    var_34 = var_28.fields[var_12]
    var_35 = var_33.render_field(field_name=var_12, field=var_34, value=var_29)
    var_36 = var_28.fields[var_13]
    var_37 = var_33.render_field(field_name=var_13, field=var_36, value=var_18)
    var_38 = var_28.fields[var_14]
    var_39 = var_33.render_field(field_name=var_14, field=var_38, value=var_10)
    var_40 = var_28.fields[var_15]
    var_41 = var_33.render_field(field_name=var_15, field=var_40, value=var_30)
    var_42 = var_28.fields[var_16]
    var_43 = var_33.render_field(field_name=var_16, field=var_42, value=var_31)
    var_44 = var_28.fields[var_12]
    var_45 = 'Invalid'
    var_46 = var_33.render_field(field_name=var_12, field=var_44, value=var_29, error=var_45)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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
    var_5 = 'textarea_field'
    var_6 = 'password_field'
    var_7 = module_2.String()
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_8, var_9]
    var_11 = module_2.Choice(choices=var_10)
    var_12 = module_2.Boolean()
    var_13 = 'text'
    var_14 = module_2.String(format=var_13)
    var_15 = 'password'
    var_16 = module_2.String(format=var_15)
    var_17 = {var_2: var_7, var_3: var_11, var_4: var_12, var_5: var_14, var_6: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_4.Form(env=var_1, schema=var_18)
    var_20 = var_18.fields[var_2]
    var_21 = 'test'
    var_22 = var_19.render_field(field_name=var_2, field=var_20, value=var_21)
    var_23 = var_18.fields[var_3]
    var_24 = var_19.render_field(field_name=var_3, field=var_23)
    var_25 = var_18.fields[var_4]
    var_26 = True
    var_27 = var_19.render_field(field_name=var_4, field=var_25, value=var_26)
    var_28 = var_18.fields[var_5]
    var_29 = var_19.render_field(field_name=var_5, field=var_28, value=var_21)
    var_30 = var_18.fields[var_6]
    var_31 = 'secret'
    var_32 = var_19.render_field(field_name=var_6, field=var_30, value=var_31)
    var_33 = var_18.fields[var_2]
    var_34 = 'Error message'
    var_35 = var_19.render_field(field_name=var_2, field=var_33, error=var_34)



# Parsed testcases at query #23
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
    var_16 = 'select_field'
    var_17 = 'checkbox_field'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'number'
    var_23 = module_2.String(format=var_22)
    var_24 = 'password'
    var_25 = module_2.String(format=var_24)
    var_26 = module_2.String(format=var_18)
    var_27 = 'a'
    var_28 = 'b'
    var_29 = 'c'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_2.Choice(choices=var_30)
    var_32 = module_2.Boolean()
    var_33 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_25, var_15: var_26, var_16: var_31, var_17: var_32}
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
    var_44 = '123'
    var_45 = var_36.render_field(field_name=var_13, field=var_43, value=var_44)
    var_46 = var_34.fields[var_14]
    var_47 = 'secret'
    var_48 = var_36.render_field(field_name=var_14, field=var_46, value=var_47)
    var_49 = var_34.fields[var_15]
    var_50 = 'long text'
    var_51 = var_36.render_field(field_name=var_15, field=var_49, value=var_50)
    var_52 = var_34.fields[var_16]
    var_53 = var_36.render_field(field_name=var_16, field=var_52, value=var_27)
    var_54 = var_34.fields[var_17]
    var_55 = True
    var_56 = var_36.render_field(field_name=var_17, field=var_54, value=var_55)
    var_57 = var_34.fields[var_17]
    var_58 = False
    var_59 = var_36.render_field(field_name=var_17, field=var_57, value=var_58)



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



# Parsed testcases at query #26
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
    var_7 = module_0.String()
    var_8 = 'number'
    var_9 = module_0.String(format=var_8)
    var_10 = module_0.String(format=var_2)
    var_11 = module_0.String(format=var_3)
    var_12 = 'text'
    var_13 = module_0.String(format=var_12)
    var_14 = module_0.Boolean()
    var_15 = 'US'
    var_16 = 'UK'
    var_17 = 'CA'
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = {var_0: var_7, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_13, var_5: var_14, var_6: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = 'forms/input.html'
    var_23 = 'forms/textarea.html'
    var_24 = 'forms/select.html'
    var_25 = 'forms/checkbox.html'
    var_26 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_27 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_28 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_29 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_30 = {var_22: var_26, var_23: var_27, var_24: var_28, var_25: var_29}
    var_31 = module_2.DictLoader(var_30)
    var_32 = module_3.Environment(loader=var_31)
    var_33 = 'John'
    var_34 = '30'
    var_35 = 'john@example.com'
    var_36 = 'secret'
    var_37 = 'Hello'
    var_38 = True
    var_39 = {var_0: var_33, var_1: var_34, var_2: var_35, var_3: var_36, var_4: var_37, var_5: var_38, var_6: var_15}
    var_40 = module_4.Form(env=var_32, schema=var_21, values=var_39)
    var_41 = var_40.render_fields()



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
    var_15 = 'password_field'
    var_16 = module_2.String()
    var_17 = 'a'
    var_18 = 'b'
    var_19 = [var_17, var_18]
    var_20 = module_2.Choice(choices=var_19)
    var_21 = module_2.Boolean()
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = 'password'
    var_25 = module_2.String(format=var_24)
    var_26 = {var_11: var_16, var_12: var_20, var_13: var_21, var_14: var_23, var_15: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_4.Form(env=var_10, schema=var_27)
    var_29 = var_27.fields[var_11]
    var_30 = 'test'
    var_31 = var_28.render_field(field_name=var_11, field=var_29, value=var_30)
    var_32 = var_27.fields[var_12]
    var_33 = var_28.render_field(field_name=var_12, field=var_32)
    var_34 = var_27.fields[var_13]
    var_35 = True
    var_36 = var_28.render_field(field_name=var_13, field=var_34, value=var_35)
    var_37 = var_27.fields[var_14]
    var_38 = var_28.render_field(field_name=var_14, field=var_37, value=var_30)
    var_39 = var_27.fields[var_15]
    var_40 = 'secret'
    var_41 = var_28.render_field(field_name=var_15, field=var_39, value=var_40)
    var_42 = var_27.fields[var_11]
    var_43 = 'Invalid'
    var_44 = var_28.render_field(field_name=var_11, field=var_42, error=var_43)



# Parsed testcases at query #29
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
    var_5 = 'textarea_field'
    var_6 = 'password_field'
    var_7 = module_2.String()
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_8, var_9]
    var_11 = module_2.Choice(choices=var_10)
    var_12 = module_2.Boolean()
    var_13 = 'text'
    var_14 = module_2.String(format=var_13)
    var_15 = 'password'
    var_16 = module_2.String(format=var_15)
    var_17 = {var_2: var_7, var_3: var_11, var_4: var_12, var_5: var_14, var_6: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_4.Form(env=var_1, schema=var_18)
    var_20 = var_18.fields[var_2]
    var_21 = 'test'
    var_22 = var_19.render_field(field_name=var_2, field=var_20, value=var_21)
    var_23 = var_18.fields[var_3]
    var_24 = var_19.render_field(field_name=var_3, field=var_23)
    var_25 = var_18.fields[var_4]
    var_26 = True
    var_27 = var_19.render_field(field_name=var_4, field=var_25, value=var_26)
    var_28 = var_18.fields[var_5]
    var_29 = 'long text'
    var_30 = var_19.render_field(field_name=var_5, field=var_28, value=var_29)
    var_31 = var_18.fields[var_6]
    var_32 = 'secret'
    var_33 = var_19.render_field(field_name=var_6, field=var_31, value=var_32)
    var_34 = var_18.fields[var_2]
    var_35 = 'Invalid'
    var_36 = var_19.render_field(field_name=var_2, field=var_34, error=var_35)



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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
    var_13 = 'textarea_field'
    var_14 = 'choice_field'
    var_15 = 'bool_field'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_2.Choice(choices=var_24)
    var_26 = module_2.Boolean()
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_25, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = var_28.fields[var_11]
    var_31 = 'test'
    var_32 = var_29.render_field(field_name=var_11, field=var_30, value=var_31)
    var_33 = var_28.fields[var_12]
    var_34 = '123'
    var_35 = var_29.render_field(field_name=var_12, field=var_33, value=var_34)
    var_36 = var_28.fields[var_13]
    var_37 = var_29.render_field(field_name=var_13, field=var_36, value=var_31)
    var_38 = var_28.fields[var_14]
    var_39 = var_29.render_field(field_name=var_14, field=var_38, value=var_21)
    var_40 = var_28.fields[var_15]
    var_41 = True
    var_42 = var_29.render_field(field_name=var_15, field=var_40, value=var_41)
    var_43 = var_28.fields[var_11]
    var_44 = 'Invalid'
    var_45 = var_29.render_field(field_name=var_11, field=var_43, value=var_31, error=var_44)
    var_46 = 'password'
    var_47 = module_2.String(format=var_46)
    var_48 = 'secret'
    var_49 = var_29.render_field(field_name=var_46, field=var_47, value=var_48)



# Parsed testcases at query #32
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



# Parsed testcases at query #33
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
    var_35 = 'test'
    var_36 = 'test@example.com'
    var_37 = 'secret'
    var_38 = '123'
    var_39 = 'long text'
    var_40 = True
    var_41 = {var_11: var_35, var_12: var_36, var_13: var_37, var_14: var_38, var_15: var_39, var_16: var_27, var_17: var_40}
    var_42 = module_4.Form(env=var_10, schema=var_34, values=var_41)
    var_43 = var_34.fields[var_11]
    var_44 = var_42.render_field(field_name=var_11, field=var_43, value=var_35)
    var_45 = var_34.fields[var_12]
    var_46 = var_42.render_field(field_name=var_12, field=var_45, value=var_36)
    var_47 = var_34.fields[var_13]
    var_48 = var_42.render_field(field_name=var_13, field=var_47, value=var_37)
    var_49 = var_34.fields[var_14]
    var_50 = var_42.render_field(field_name=var_14, field=var_49, value=var_38)
    var_51 = var_34.fields[var_15]
    var_52 = var_42.render_field(field_name=var_15, field=var_51, value=var_39)
    var_53 = var_34.fields[var_16]
    var_54 = var_42.render_field(field_name=var_16, field=var_53, value=var_27)
    var_55 = var_34.fields[var_17]
    var_56 = var_42.render_field(field_name=var_17, field=var_55, value=var_40)
    var_57 = var_34.fields[var_17]
    var_58 = False
    var_59 = var_42.render_field(field_name=var_17, field=var_57, value=var_58)
    var_60 = var_34.fields[var_11]
    var_61 = 'Invalid value'
    var_62 = var_42.render_field(field_name=var_11, field=var_60, value=var_35, error=var_61)
    var_63 = var_34.fields[var_11]
    var_64 = None
    var_65 = var_42.render_field(field_name=var_11, field=var_63, value=var_64)



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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
    var_30 = var_25.render_field(field_name=var_12, field=var_29, value=var_16)
    var_31 = var_24.fields[var_13]
    var_32 = True
    var_33 = var_25.render_field(field_name=var_13, field=var_31, value=var_32)
    var_34 = var_24.fields[var_14]
    var_35 = var_25.render_field(field_name=var_14, field=var_34, value=var_27)
    var_36 = var_24.fields[var_11]
    var_37 = 'Error message'
    var_38 = var_25.render_field(field_name=var_11, field=var_36, value=var_27, error=var_37)
    var_39 = 'password'
    var_40 = module_2.String(format=var_39)
    var_41 = 'secret'
    var_42 = var_25.render_field(field_name=var_39, field=var_40, value=var_41)



# Parsed testcases at query #36
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



# Parsed testcases at query #37
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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'bio'
    var_14 = 'active'
    var_15 = 'gender'
    var_16 = 'readonly_field'
    var_17 = 'Name'
    var_18 = module_2.String()
    var_19 = 'number'
    var_20 = 'Age'
    var_21 = module_2.String(format=var_19)
    var_22 = 'text'
    var_23 = 'Bio'
    var_24 = module_2.String(format=var_22)
    var_25 = 'Active'
    var_26 = module_2.Boolean()
    var_27 = 'M'
    var_28 = 'Male'
    var_29 = (var_27, var_28)
    var_30 = 'F'
    var_31 = 'Female'
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = 'Gender'
    var_35 = module_2.Choice(choices=var_33)
    var_36 = True
    var_37 = 'Read Only'
    var_38 = module_2.String()
    var_39 = {var_11: var_18, var_12: var_21, var_13: var_24, var_14: var_26, var_15: var_35, var_16: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = 'John'
    var_42 = '30'
    var_43 = 'Developer'
    var_44 = {var_11: var_41, var_12: var_42, var_13: var_43, var_14: var_36, var_15: var_27}
    var_45 = module_4.Form(env=var_10, schema=var_40, values=var_44)
    var_46 = var_45.render_fields()
    var_47 = ''
    var_48 = 'invalid'
    var_49 = 'X'
    var_50 = {var_11: var_47, var_12: var_48, var_13: var_47, var_14: var_47, var_15: var_49}
    var_51 = var_45.validate(var_50)
    var_52 = var_45.render_fields()



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
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_5 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_6 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_7 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'text_field'
    var_13 = 'password_field'
    var_14 = 'email_field'
    var_15 = 'number_field'
    var_16 = 'textarea_field'
    var_17 = 'choice_field'
    var_18 = 'boolean_field'
    var_19 = 'Text Field'
    var_20 = module_2.String()
    var_21 = 'password'
    var_22 = 'Password Field'
    var_23 = module_2.String(format=var_21)
    var_24 = 'email'
    var_25 = 'Email Field'
    var_26 = module_2.String(format=var_24)
    var_27 = 'number'
    var_28 = 'Number Field'
    var_29 = module_2.String(format=var_27)
    var_30 = 'text'
    var_31 = 'Text Area'
    var_32 = module_2.String(format=var_30)
    var_33 = '1'
    var_34 = 'Option 1'
    var_35 = (var_33, var_34)
    var_36 = '2'
    var_37 = 'Option 2'
    var_38 = (var_36, var_37)
    var_39 = [var_35, var_38]
    var_40 = 'Choice Field'
    var_41 = module_2.Choice(choices=var_39)
    var_42 = 'Boolean Field'
    var_43 = module_2.Boolean()
    var_44 = {var_12: var_20, var_13: var_23, var_14: var_26, var_15: var_29, var_16: var_32, var_17: var_41, var_18: var_43}
    var_45 = module_3.Schema(var_44)
    var_46 = 'test'
    var_47 = 'secret'
    var_48 = 'test@example.com'
    var_49 = '42'
    var_50 = 'long text'
    var_51 = {var_12: var_46, var_13: var_47, var_14: var_48, var_15: var_49, var_16: var_50, var_17: var_33, var_18: var_10}
    var_52 = module_4.Form(env=var_11, schema=var_45, values=var_51)
    var_53 = var_45.fields[var_12]
    var_54 = var_52.render_field(field_name=var_12, field=var_53, value=var_46)
    var_55 = var_45.fields[var_13]
    var_56 = var_52.render_field(field_name=var_13, field=var_55, value=var_47)
    var_57 = var_45.fields[var_14]
    var_58 = var_52.render_field(field_name=var_14, field=var_57, value=var_48)
    var_59 = var_45.fields[var_15]
    var_60 = var_52.render_field(field_name=var_15, field=var_59, value=var_49)
    var_61 = var_45.fields[var_16]
    var_62 = var_52.render_field(field_name=var_16, field=var_61, value=var_50)
    var_63 = var_45.fields[var_17]
    var_64 = var_52.render_field(field_name=var_17, field=var_63, value=var_33)
    var_65 = var_45.fields[var_18]
    var_66 = var_52.render_field(field_name=var_18, field=var_65, value=var_10)
    var_67 = var_45.fields[var_12]
    var_68 = 'Invalid value'
    var_69 = var_52.render_field(field_name=var_12, field=var_67, value=var_46, error=var_68)
    var_70 = var_45.fields[var_12]
    var_71 = None
    var_72 = var_52.render_field(field_name=var_12, field=var_70, value=var_71)



# Parsed testcases at query #39
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
    var_12 = 'password_field'
    var_13 = 'number_field'
    var_14 = 'choice_field'
    var_15 = 'bool_field'
    var_16 = 'textarea_field'
    var_17 = module_1.String()
    var_18 = 'email'
    var_19 = module_1.String(format=var_18)
    var_20 = 'password'
    var_21 = module_1.String(format=var_20)
    var_22 = 'number'
    var_23 = module_1.String(format=var_22)
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_1.Choice(choices=var_27)
    var_29 = module_1.Boolean()
    var_30 = 'text'
    var_31 = module_1.String(format=var_30)
    var_32 = {var_10: var_17, var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_28, var_15: var_29, var_16: var_31}
    var_33 = module_2.Schema(var_32)
    var_34 = module_3.Form(env=var_0, schema=var_33)
    var_35 = var_33.fields[var_10]
    var_36 = 'test'
    var_37 = var_34.render_field(field_name=var_10, field=var_35, value=var_36)
    var_38 = var_33.fields[var_11]
    var_39 = 'test@example.com'
    var_40 = var_34.render_field(field_name=var_11, field=var_38, value=var_39)
    var_41 = var_33.fields[var_12]
    var_42 = 'secret'
    var_43 = var_34.render_field(field_name=var_12, field=var_41, value=var_42)
    var_44 = var_33.fields[var_13]
    var_45 = '42'
    var_46 = var_34.render_field(field_name=var_13, field=var_44, value=var_45)
    var_47 = var_33.fields[var_14]
    var_48 = var_34.render_field(field_name=var_14, field=var_47, value=var_24)
    var_49 = var_33.fields[var_15]
    var_50 = True
    var_51 = var_34.render_field(field_name=var_15, field=var_49, value=var_50)
    var_52 = var_33.fields[var_16]
    var_53 = 'long text'
    var_54 = var_34.render_field(field_name=var_16, field=var_52, value=var_53)
    var_55 = var_33.fields[var_10]
    var_56 = 'Invalid'
    var_57 = var_34.render_field(field_name=var_10, field=var_55, value=var_36, error=var_56)



# Parsed testcases at query #40
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
    var_4 = 'country'
    var_5 = module_0.String()
    var_6 = 'number'
    var_7 = module_0.String(format=var_6)
    var_8 = 'text'
    var_9 = module_0.String(format=var_8)
    var_10 = module_0.Boolean()
    var_11 = 'US'
    var_12 = 'UK'
    var_13 = 'CA'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.Choice(choices=var_14)
    var_16 = {var_0: var_5, var_1: var_7, var_2: var_9, var_3: var_10, var_4: var_15}
    var_17 = module_1.Schema(var_16)
    var_18 = 'forms/input.html'
    var_19 = 'forms/textarea.html'
    var_20 = 'forms/select.html'
    var_21 = 'forms/checkbox.html'
    var_22 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_23 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_24 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_25 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_26 = {var_18: var_22, var_19: var_23, var_20: var_24, var_21: var_25}
    var_27 = module_2.DictLoader(var_26)
    var_28 = module_3.Environment(loader=var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'Test'
    var_32 = True
    var_33 = {var_0: var_29, var_1: var_30, var_2: var_31, var_3: var_32, var_4: var_11}
    var_34 = module_4.Form(env=var_28, schema=var_17, values=var_33)
    var_35 = var_34.render_fields()
    var_36 = ''
    var_37 = 'invalid'
    var_38 = False
    var_39 = 'INVALID'
    var_40 = {var_0: var_36, var_1: var_37, var_2: var_36, var_3: var_38, var_4: var_39}
    var_41 = var_34.validate(var_40)
    var_42 = var_34.render_fields()



# Parsed testcases at query #41
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
    var_3 = 'username'
    var_4 = 'password'
    var_5 = 'age'
    var_6 = 'email'
    var_7 = 'description'
    var_8 = 'agree'
    var_9 = 'country'
    var_10 = 'text'
    var_11 = module_2.String(format=var_10)
    var_12 = module_2.String(format=var_4)
    var_13 = 'number'
    var_14 = module_2.String(format=var_13)
    var_15 = module_2.String(format=var_6)
    var_16 = module_2.String(format=var_10)
    var_17 = module_2.Boolean()
    var_18 = 'US'
    var_19 = 'UK'
    var_20 = 'CA'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_2.Choice(choices=var_21)
    var_23 = {var_3: var_11, var_4: var_12, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_4.Form(env=var_2, schema=var_24)
    var_26 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}"\n               value="{{ value }}" {% if required %}required{% endif %}>\n    '
    var_27 = var_2.from_string(var_26)
    var_28 = var_24.fields[var_3]
    var_29 = 'testuser'
    var_30 = None
    var_31 = var_25.render_field(field_name=var_3, field=var_28, value=var_29, error=var_30)
    var_32 = var_24.fields[var_4]
    var_33 = 'secret'
    var_34 = var_25.render_field(field_name=var_4, field=var_32, value=var_33, error=var_30)
    var_35 = var_24.fields[var_5]
    var_36 = '25'
    var_37 = var_25.render_field(field_name=var_5, field=var_35, value=var_36, error=var_30)
    var_38 = var_24.fields[var_6]
    var_39 = 'test@example.com'
    var_40 = var_25.render_field(field_name=var_6, field=var_38, value=var_39, error=var_30)
    var_41 = '\n        <textarea name="{{ field_name }}" id="{{ field_id }}"\n                  {% if required %}required{% endif %}>{{ value }}</textarea>\n    '
    var_42 = var_2.from_string(var_41)
    var_43 = 'forms/textarea.html'
    var_44 = var_24.fields[var_7]
    var_45 = 'Test description'
    var_46 = var_25.render_field(field_name=var_7, field=var_44, value=var_45, error=var_30)
    var_47 = '\n        <input type="checkbox" name="{{ field_name }}" id="{{ field_id }}"\n               {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n    '
    var_48 = var_2.from_string(var_47)
    var_49 = 'forms/checkbox.html'
    var_50 = var_24.fields[var_8]
    var_51 = var_25.render_field(field_name=var_8, field=var_50, value=var_1, error=var_30)
    var_52 = '\n        <select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>\n            {% for choice in field.choices %}\n            <option value="{{ choice }}" {% if choice == value %}selected{% endif %}>{{ choice }}</option>\n            {% endfor %}\n        </select>\n    '
    var_53 = var_2.from_string(var_52)
    var_54 = 'forms/select.html'
    var_55 = var_24.fields[var_9]
    var_56 = var_25.render_field(field_name=var_9, field=var_55, value=var_19, error=var_30)
    var_57 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}"\n               value="{{ value }}" {% if required %}required{% endif %}>\n        {% if error %}<span class="error">{{ error }}</span>{% endif %}\n    '
    var_58 = var_2.from_string(var_57)
    var_59 = var_24.fields[var_3]
    var_60 = ''
    var_61 = 'This field is required'
    var_62 = var_25.render_field(field_name=var_3, field=var_59, value=var_60, error=var_61)
    var_63 = 'optional_field'
    var_64 = module_2.String(allow_blank=var_1)
    var_65 = {var_63: var_64}
    var_66 = module_3.Schema(var_65)
    var_67 = module_4.Form(env=var_2, schema=var_66)
    var_68 = var_66.fields[var_63]
    var_69 = 'test'
    var_70 = var_67.render_field(field_name=var_63, field=var_68, value=var_69, error=var_30)



# Parsed testcases at query #42
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
    var_24 = 'O'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_2.Choice(choices=var_25)
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_21, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = 'John'
    var_30 = '30'
    var_31 = 'Test bio'
    var_32 = True
    var_33 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_22}
    var_34 = module_4.Form(env=var_10, schema=var_28, values=var_33)
    var_35 = {var_11: var_29, var_12: var_30, var_13: var_31, var_14: var_32, var_15: var_22}
    var_36 = var_34.validate(var_35)
    var_37 = var_34.render_fields()
    var_38 = module_4.Form(env=var_10, schema=var_28)
    var_39 = ''
    var_40 = 'invalid'
    var_41 = 'not_bool'
    var_42 = 'X'
    var_43 = {var_11: var_39, var_12: var_40, var_13: var_39, var_14: var_41, var_15: var_42}
    var_44 = var_38.validate(var_43)
    var_45 = var_38.render_fields()
    var_46 = 'readonly_field'
    var_47 = module_2.String()
    var_48 = {var_46: var_47}
    var_49 = module_3.Schema(var_48)
    var_50 = module_4.Form(env=var_10, schema=var_49)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = var_50.render_fields()
    assert var_53 == ''



# Parsed testcases at query #43
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
    var_13 = 'textarea_field'
    var_14 = 'choice_field'
    var_15 = 'boolean_field'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'text'
    var_20 = module_2.String(format=var_19)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_2.Choice(choices=var_24)
    var_26 = module_2.Boolean()
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_20, var_14: var_25, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = var_28.fields[var_11]
    var_31 = 'test'
    var_32 = var_29.render_field(field_name=var_11, field=var_30, value=var_31)
    var_33 = var_28.fields[var_12]
    var_34 = '123'
    var_35 = var_29.render_field(field_name=var_12, field=var_33, value=var_34)
    var_36 = var_28.fields[var_13]
    var_37 = var_29.render_field(field_name=var_13, field=var_36, value=var_31)
    var_38 = var_28.fields[var_14]
    var_39 = var_29.render_field(field_name=var_14, field=var_38, value=var_21)
    var_40 = var_28.fields[var_15]
    var_41 = True
    var_42 = var_29.render_field(field_name=var_15, field=var_40, value=var_41)
    var_43 = var_28.fields[var_11]
    var_44 = 'Error message'
    var_45 = var_29.render_field(field_name=var_11, field=var_43, value=var_31, error=var_44)
    var_46 = 'password'
    var_47 = module_2.String(format=var_46)
    var_48 = 'password_field'
    var_49 = 'secret'
    var_50 = var_29.render_field(field_name=var_48, field=var_47, value=var_49)



# Parsed testcases at query #44
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
    var_35 = module_4.Form(env=var_10, schema=var_34)
    var_36 = var_34.fields[var_11]
    var_37 = 'test_value'
    var_38 = None
    var_39 = var_35.render_field(field_name=var_11, field=var_36, value=var_37, error=var_38)
    var_40 = var_34.fields[var_12]
    var_41 = 'test@example.com'
    var_42 = var_35.render_field(field_name=var_12, field=var_40, value=var_41, error=var_38)
    var_43 = var_34.fields[var_13]
    var_44 = 'secret'
    var_45 = var_35.render_field(field_name=var_13, field=var_43, value=var_44, error=var_38)
    var_46 = var_34.fields[var_14]
    var_47 = '123'
    var_48 = var_35.render_field(field_name=var_14, field=var_46, value=var_47, error=var_38)
    var_49 = var_34.fields[var_15]
    var_50 = 'multiline text'
    var_51 = var_35.render_field(field_name=var_15, field=var_49, value=var_50, error=var_38)
    var_52 = var_34.fields[var_16]
    var_53 = var_35.render_field(field_name=var_16, field=var_52, value=var_27, error=var_38)
    var_54 = var_34.fields[var_17]
    var_55 = True
    var_56 = var_35.render_field(field_name=var_17, field=var_54, value=var_55, error=var_38)
    var_57 = var_34.fields[var_17]
    var_58 = False
    var_59 = var_35.render_field(field_name=var_17, field=var_57, value=var_58, error=var_38)
    var_60 = var_34.fields[var_11]
    var_61 = 'Invalid value'
    var_62 = var_35.render_field(field_name=var_11, field=var_60, value=var_37, error=var_61)



# Parsed testcases at query #45
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



# Parsed testcases at query #46
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
    var_22 = 'male'
    var_23 = 'female'
    var_24 = 'other'
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
    var_35 = var_34.render_fields()
    var_36 = ''
    var_37 = 'invalid'
    var_38 = False
    var_39 = {var_11: var_36, var_12: var_37, var_13: var_36, var_14: var_38, var_15: var_37}
    var_40 = var_34.validate(var_39)
    var_41 = var_34.render_fields()
    var_42 = 'readonly_field'
    var_43 = module_2.String()
    var_44 = {var_42: var_43}
    var_45 = module_3.Schema(var_44)
    var_46 = 'value'
    var_47 = {var_42: var_46}
    var_48 = module_4.Form(env=var_10, schema=var_45, values=var_47)
    var_49 = var_48.render_fields()



# Parsed testcases at query #47
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



# Parsed testcases at query #48
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



# Parsed testcases at query #49
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
    var_6 = 'password_field'
    var_7 = module_2.String()
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_8, var_9]
    var_11 = module_2.Choice(choices=var_10)
    var_12 = module_2.Boolean()
    var_13 = 'text'
    var_14 = module_2.String(format=var_13)
    var_15 = 'password'
    var_16 = module_2.String(format=var_15)
    var_17 = {var_2: var_7, var_3: var_11, var_4: var_12, var_5: var_14, var_6: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_4.Form(env=var_1, schema=var_18)
    var_20 = var_18.fields[var_2]
    var_21 = 'test'
    var_22 = var_19.render_field(field_name=var_2, field=var_20, value=var_21)
    var_23 = var_18.fields[var_3]
    var_24 = var_19.render_field(field_name=var_3, field=var_23, value=var_8)
    var_25 = var_18.fields[var_4]
    var_26 = True
    var_27 = var_19.render_field(field_name=var_4, field=var_25, value=var_26)
    var_28 = var_18.fields[var_5]
    var_29 = 'long text'
    var_30 = var_19.render_field(field_name=var_5, field=var_28, value=var_29)
    var_31 = var_18.fields[var_6]
    var_32 = 'secret'
    var_33 = var_19.render_field(field_name=var_6, field=var_31, value=var_32)
    var_34 = var_18.fields[var_2]
    var_35 = 'Invalid'
    var_36 = var_19.render_field(field_name=var_2, field=var_34, value=var_21, error=var_35)



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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
    var_10 = True
    var_11 = module_1.Environment(autoescape=var_10, loader=var_9)
    var_12 = 'text_field'
    var_13 = 'email_field'
    var_14 = 'number_field'
    var_15 = 'password_field'
    var_16 = 'choice_field'
    var_17 = 'boolean_field'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'number'
    var_23 = module_2.String(format=var_22)
    var_24 = 'password'
    var_25 = module_2.String(format=var_24)
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'c'
    var_29 = [var_26, var_27, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = module_2.Boolean()
    var_32 = {var_12: var_19, var_13: var_21, var_14: var_23, var_15: var_25, var_16: var_30, var_17: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = {}
    var_35 = module_4.Form(env=var_11, schema=var_33, values=var_34)
    var_36 = var_33.fields[var_12]
    var_37 = 'test'
    var_38 = var_35.render_field(field_name=var_12, field=var_36, value=var_37)
    var_39 = var_33.fields[var_13]
    var_40 = 'test@example.com'
    var_41 = var_35.render_field(field_name=var_13, field=var_39, value=var_40)
    var_42 = var_33.fields[var_14]
    var_43 = '123'
    var_44 = var_35.render_field(field_name=var_14, field=var_42, value=var_43)
    var_45 = var_33.fields[var_15]
    var_46 = 'secret'
    var_47 = var_35.render_field(field_name=var_15, field=var_45, value=var_46)
    var_48 = var_33.fields[var_16]
    var_49 = var_35.render_field(field_name=var_16, field=var_48, value=var_26)
    var_50 = var_33.fields[var_17]
    var_51 = var_35.render_field(field_name=var_17, field=var_50, value=var_10)
    var_52 = var_33.fields[var_12]
    var_53 = 'Invalid value'
    var_54 = var_35.render_field(field_name=var_12, field=var_52, value=var_37, error=var_53)



# Parsed testcases at query #52
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
    var_15 = 'country'
    var_16 = 'active'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = module_2.String(format=var_12)
    var_20 = 'number'
    var_21 = module_2.String(format=var_20)
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = 'US'
    var_25 = 'UK'
    var_26 = 'CA'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = module_2.Boolean()
    var_30 = {var_11: var_18, var_12: var_19, var_13: var_21, var_14: var_23, var_15: var_28, var_16: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = 'test'
    var_33 = 'test@example.com'
    var_34 = '25'
    var_35 = 'Test bio'
    var_36 = True
    var_37 = {var_11: var_32, var_12: var_33, var_13: var_34, var_14: var_35, var_15: var_24, var_16: var_36}
    var_38 = module_4.Form(env=var_10, schema=var_31, values=var_37)
    var_39 = var_31.fields[var_11]
    var_40 = var_38.render_field(field_name=var_11, field=var_39, value=var_32)
    var_41 = var_31.fields[var_12]
    var_42 = var_38.render_field(field_name=var_12, field=var_41, value=var_33)
    var_43 = var_31.fields[var_13]
    var_44 = var_38.render_field(field_name=var_13, field=var_43, value=var_34)
    var_45 = var_31.fields[var_14]
    var_46 = var_38.render_field(field_name=var_14, field=var_45, value=var_35)
    var_47 = var_31.fields[var_15]
    var_48 = var_38.render_field(field_name=var_15, field=var_47, value=var_24)
    var_49 = var_31.fields[var_16]
    var_50 = var_38.render_field(field_name=var_16, field=var_49, value=var_36)
    var_51 = var_31.fields[var_11]
    var_52 = 'This field is required'
    var_53 = var_38.render_field(field_name=var_11, field=var_51, value=var_32, error=var_52)
    var_54 = 'password'
    var_55 = module_2.String(format=var_54)
    var_56 = 'secret'
    var_57 = var_38.render_field(field_name=var_54, field=var_55, value=var_56)



# Parsed testcases at query #53
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = module_0.BaseLoader()
    var_1 = module_1.Environment(loader=var_0)
    var_2 = 'forms/input.html'
    var_3 = 'forms/select.html'
    var_4 = 'forms/checkbox.html'
    var_5 = 'forms/textarea.html'
    var_6 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_7 = '<select id="{{ field_id }}" name="{{ field_name }}"></select>'
    var_8 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_9 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = 'text_field'
    var_12 = 'choice_field'
    var_13 = 'bool_field'
    var_14 = 'text_area_field'
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
    var_25 = 'test'
    var_26 = {var_11: var_25}
    var_27 = module_4.Form(env=var_1, schema=var_24, values=var_26)
    var_28 = var_24.fields[var_11]
    var_29 = None
    var_30 = var_27.render_field(field_name=var_11, field=var_28, value=var_25, error=var_29)
    var_31 = var_24.fields[var_12]
    var_32 = var_27.render_field(field_name=var_12, field=var_31, value=var_16, error=var_29)
    var_33 = var_24.fields[var_13]
    var_34 = True
    var_35 = var_27.render_field(field_name=var_13, field=var_33, value=var_34, error=var_29)
    var_36 = var_24.fields[var_14]
    var_37 = 'long text'
    var_38 = var_27.render_field(field_name=var_14, field=var_36, value=var_37, error=var_29)
    var_39 = var_24.fields[var_11]
    var_40 = 'Invalid value'
    var_41 = var_27.render_field(field_name=var_11, field=var_39, value=var_25, error=var_40)
    var_42 = 'password'
    var_43 = module_2.String(format=var_42)
    var_44 = 'secret'
    var_45 = var_27.render_field(field_name=var_42, field=var_43, value=var_44, error=var_29)



# Parsed testcases at query #54
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
    var_4 = 'agree'
    var_5 = 'country'
    var_6 = module_0.String()
    var_7 = 'number'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.String(format=var_2)
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = module_0.Boolean()
    var_13 = 'US'
    var_14 = 'UK'
    var_15 = 'CA'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = {var_0: var_6, var_1: var_8, var_2: var_9, var_3: var_11, var_4: var_12, var_5: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = 'forms/input.html'
    var_21 = 'forms/textarea.html'
    var_22 = 'forms/checkbox.html'
    var_23 = 'forms/select.html'
    var_24 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_25 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_26 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_27 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_28 = {var_20: var_24, var_21: var_25, var_22: var_26, var_23: var_27}
    var_29 = module_2.DictLoader(var_28)
    var_30 = module_3.Environment(loader=var_29)
    var_31 = 'John'
    var_32 = '30'
    var_33 = 'john@example.com'
    var_34 = 'Hello'
    var_35 = True
    var_36 = {var_0: var_31, var_1: var_32, var_2: var_33, var_3: var_34, var_4: var_35, var_5: var_13}
    var_37 = module_4.Form(env=var_30, schema=var_19, values=var_36)
    var_38 = var_37.render_fields()
    var_39 = ''
    var_40 = 'invalid'
    var_41 = False
    var_42 = 'INVALID'
    var_43 = {var_0: var_39, var_1: var_40, var_2: var_40, var_3: var_39, var_4: var_41, var_5: var_42}
    var_44 = var_37.validate(var_43)
    var_45 = var_37.render_fields()



# Parsed testcases at query #55
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
    var_13 = 'number_field'
    var_14 = 'password_field'
    var_15 = 'choice_field'
    var_16 = 'boolean_field'
    var_17 = 'textarea_field'
    var_18 = module_2.String()
    var_19 = 'email'
    var_20 = module_2.String(format=var_19)
    var_21 = 'number'
    var_22 = module_2.String(format=var_21)
    var_23 = 'password'
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
    var_43 = '123'
    var_44 = var_35.render_field(field_name=var_13, field=var_42, value=var_43)
    var_45 = var_34.fields[var_14]
    var_46 = 'secret'
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



# Parsed testcases at query #56
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
    var_4 = 'gender'
    var_5 = 'subscribe'
    var_6 = 100
    var_7 = module_0.String(max_length=var_6)
    var_8 = 0
    var_9 = module_0.Integer()
    var_10 = module_0.String(format=var_2)
    var_11 = 'text'
    var_12 = module_0.String(format=var_11)
    var_13 = 'M'
    var_14 = 'F'
    var_15 = 'O'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = module_0.Boolean()
    var_19 = {var_0: var_7, var_1: var_9, var_2: var_10, var_3: var_12, var_4: var_17, var_5: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 'forms/input.html'
    var_22 = 'forms/textarea.html'
    var_23 = 'forms/select.html'
    var_24 = 'forms/checkbox.html'
    var_25 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}">'
    var_26 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_27 = '<select id="{{ field_id }}" name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_28 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %}>'
    var_29 = {var_21: var_25, var_22: var_26, var_23: var_27, var_24: var_28}
    var_30 = module_2.DictLoader(var_29)
    var_31 = module_3.Environment(loader=var_30)
    var_32 = 'John'
    var_33 = 30
    var_34 = 'john@example.com'
    var_35 = 'Test bio'
    var_36 = True
    var_37 = {var_0: var_32, var_1: var_33, var_2: var_34, var_3: var_35, var_4: var_13, var_5: var_36}
    var_38 = module_4.Form(env=var_31, schema=var_20, values=var_37)
    var_39 = var_38.render_fields()
    var_40 = ''
    var_41 = -1
    var_42 = 'invalid'
    var_43 = 'X'
    var_44 = 'not_bool'
    var_45 = {var_0: var_40, var_1: var_41, var_2: var_42, var_3: var_40, var_4: var_43, var_5: var_44}
    var_46 = var_38.validate(var_45)
    var_47 = var_38.render_fields()
    var_48 = 'readonly_field'
    var_49 = 'normal_field'
    var_50 = module_0.String()
    var_51 = module_0.String()
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = 'readonly'
    var_55 = 'normal'
    var_56 = {var_48: var_54, var_49: var_55}
    var_57 = module_4.Form(env=var_31, schema=var_53, values=var_56)
    var_58 = var_57.render_fields()



# Parsed testcases at query #57
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'my_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #58
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
    var_8 = var_6.env.loader.loaders
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_6.env.loader.loaders[var_10]
    var_12 = 1
    var_13 = var_6.env.loader.loaders[var_12]
    var_14 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #59
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.loaders as module_2
import jinja2.environment as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'bio'
    var_3 = 'active'
    var_4 = 'gender'
    var_5 = 100
    var_6 = module_0.String(max_length=var_5)
    var_7 = 0
    var_8 = module_0.Integer(minimum=var_7)
    var_9 = 'text'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Boolean()
    var_12 = 'M'
    var_13 = 'F'
    var_14 = 'O'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Choice(choices=var_15)
    var_17 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11, var_4: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'forms/input.html'
    var_20 = 'forms/textarea.html'
    var_21 = 'forms/select.html'
    var_22 = 'forms/checkbox.html'
    var_23 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_24 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_25 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice }}">{{ choice }}</option>{% endfor %}</select>'
    var_26 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_27 = {var_19: var_23, var_20: var_24, var_21: var_25, var_22: var_26}
    var_28 = module_2.DictLoader(var_27)
    var_29 = module_3.Environment(loader=var_28)
    var_30 = 'John'
    var_31 = 30
    var_32 = 'Developer'
    var_33 = True
    var_34 = {var_0: var_30, var_1: var_31, var_2: var_32, var_3: var_33, var_4: var_12}
    var_35 = module_4.Form(env=var_29, schema=var_18, values=var_34)
    var_36 = var_18.fields[var_0]
    var_37 = var_35.render_field(field_name=var_0, field=var_36, value=var_30)
    var_38 = var_18.fields[var_1]
    var_39 = var_35.render_field(field_name=var_1, field=var_38, value=var_31)
    var_40 = var_18.fields[var_2]
    var_41 = var_35.render_field(field_name=var_2, field=var_40, value=var_32)
    var_42 = var_18.fields[var_3]
    var_43 = var_35.render_field(field_name=var_3, field=var_42, value=var_33)
    var_44 = var_18.fields[var_4]
    var_45 = var_35.render_field(field_name=var_4, field=var_44, value=var_12)
    var_46 = var_18.fields[var_0]
    var_47 = 'Invalid name'
    var_48 = var_35.render_field(field_name=var_0, field=var_46, value=var_30, error=var_47)
    var_49 = var_18.fields[var_0]
    var_50 = None
    var_51 = var_35.render_field(field_name=var_0, field=var_49, value=var_50)
    var_52 = 'password'
    var_53 = module_0.String(format=var_52)
    var_54 = 'secret'
    var_55 = var_35.render_field(field_name=var_52, field=var_53, value=var_54)



# Parsed testcases at query #60
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



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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



# Parsed testcases at query #63
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



# Parsed testcases at query #64
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
    var_14 = 'textarea_field'
    var_15 = 'choice_field'
    var_16 = 'boolean_field'
    var_17 = module_2.String()
    var_18 = 'number'
    var_19 = module_2.String(format=var_18)
    var_20 = 'password'
    var_21 = module_2.String(format=var_20)
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_2.Choice(choices=var_27)
    var_29 = module_2.Boolean()
    var_30 = {var_11: var_17, var_12: var_19, var_13: var_21, var_14: var_23, var_15: var_28, var_16: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_4.Form(env=var_10, schema=var_31)
    var_33 = var_31.fields[var_11]
    var_34 = 'test'
    var_35 = var_32.render_field(field_name=var_11, field=var_33, value=var_34)
    var_36 = var_31.fields[var_12]
    var_37 = '123'
    var_38 = var_32.render_field(field_name=var_12, field=var_36, value=var_37)
    var_39 = var_31.fields[var_13]
    var_40 = 'secret'
    var_41 = var_32.render_field(field_name=var_13, field=var_39, value=var_40)
    var_42 = var_31.fields[var_14]
    var_43 = 'multiline'
    var_44 = var_32.render_field(field_name=var_14, field=var_42, value=var_43)
    var_45 = var_31.fields[var_15]
    var_46 = var_32.render_field(field_name=var_15, field=var_45)
    var_47 = var_31.fields[var_16]
    var_48 = True
    var_49 = var_32.render_field(field_name=var_16, field=var_47, value=var_48)
    var_50 = var_31.fields[var_16]
    var_51 = False
    var_52 = var_32.render_field(field_name=var_16, field=var_50, value=var_51)



# Parsed testcases at query #65
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



# Parsed testcases at query #66
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
    var_9 = 'path/to/templates'
    var_10 = module_0.Jinja2Forms(directory=var_9)



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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
    var_6 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_7 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_8 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_9 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_10 = 'text_field'
    var_11 = 'email_field'
    var_12 = 'choice_field'
    var_13 = 'bool_field'
    var_14 = 'text'
    var_15 = module_2.String(format=var_14)
    var_16 = 'email'
    var_17 = module_2.String(format=var_16)
    var_18 = 'a'
    var_19 = 'b'
    var_20 = [var_18, var_19]
    var_21 = module_2.Choice(choices=var_20)
    var_22 = module_2.Boolean()
    var_23 = {var_10: var_15, var_11: var_17, var_12: var_21, var_13: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = {}
    var_26 = module_4.Form(env=var_1, schema=var_24, values=var_25)
    var_27 = var_24.fields[var_10]
    var_28 = 'test_value'
    var_29 = None
    var_30 = var_26.render_field(field_name=var_10, field=var_27, value=var_28, error=var_29)
    var_31 = var_24.fields[var_11]
    var_32 = 'test@example.com'
    var_33 = var_26.render_field(field_name=var_11, field=var_31, value=var_32, error=var_29)
    var_34 = var_24.fields[var_12]
    var_35 = var_26.render_field(field_name=var_12, field=var_34, value=var_18, error=var_29)
    var_36 = var_24.fields[var_13]
    var_37 = True
    var_38 = var_26.render_field(field_name=var_13, field=var_36, value=var_37, error=var_29)
    var_39 = 'password'
    var_40 = module_2.String(format=var_39)
    var_41 = 'secret'
    var_42 = var_26.render_field(field_name=var_39, field=var_40, value=var_41, error=var_29)
    var_43 = var_24.fields[var_10]
    var_44 = ''
    var_45 = 'This field is required'
    var_46 = var_26.render_field(field_name=var_10, field=var_43, value=var_44, error=var_45)



# Parsed testcases at query #69
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



# Parsed testcases at query #70
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



# Parsed testcases at query #71
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



# Parsed testcases at query #72
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
    var_12 = 'password_field'
    var_13 = 'email_field'
    var_14 = 'number_field'
    var_15 = 'textarea_field'
    var_16 = 'choice_field'
    var_17 = 'boolean_field'
    var_18 = module_2.String()
    var_19 = 'password'
    var_20 = module_2.String(format=var_19)
    var_21 = 'email'
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
    var_39 = var_36.render_field(field_name=var_11, field=var_37, value=var_38)
    var_40 = var_34.fields[var_12]
    var_41 = 'secret'
    var_42 = var_36.render_field(field_name=var_12, field=var_40, value=var_41)
    var_43 = var_34.fields[var_13]
    var_44 = 'test@example.com'
    var_45 = var_36.render_field(field_name=var_13, field=var_43, value=var_44)
    var_46 = var_34.fields[var_14]
    var_47 = '123'
    var_48 = var_36.render_field(field_name=var_14, field=var_46, value=var_47)
    var_49 = var_34.fields[var_15]
    var_50 = 'long text'
    var_51 = var_36.render_field(field_name=var_15, field=var_49, value=var_50)
    var_52 = var_34.fields[var_16]
    var_53 = var_36.render_field(field_name=var_16, field=var_52, value=var_27)
    var_54 = var_34.fields[var_17]
    var_55 = True
    var_56 = var_36.render_field(field_name=var_17, field=var_54, value=var_55)
    var_57 = var_34.fields[var_11]
    var_58 = 'test'
    var_59 = 'Invalid value'
    var_60 = var_36.render_field(field_name=var_11, field=var_57, value=var_58, error=var_59)



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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



# Parsed testcases at query #75
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
    var_13 = 'choice_field'
    var_14 = 'bool_field'
    var_15 = 'text_area'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = module_2.String(format=var_17)
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_2.Choice(choices=var_22)
    var_24 = module_2.Boolean()
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = {var_11: var_16, var_12: var_18, var_13: var_23, var_14: var_24, var_15: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = var_28.fields[var_11]
    var_31 = 'test'
    var_32 = var_29.render_field(field_name=var_11, field=var_30, value=var_31)
    var_33 = var_28.fields[var_12]
    var_34 = '123'
    var_35 = var_29.render_field(field_name=var_12, field=var_33, value=var_34)
    var_36 = var_28.fields[var_13]
    var_37 = var_29.render_field(field_name=var_13, field=var_36, value=var_19)
    var_38 = var_28.fields[var_14]
    var_39 = True
    var_40 = var_29.render_field(field_name=var_14, field=var_38, value=var_39)
    var_41 = var_28.fields[var_15]
    var_42 = 'long text'
    var_43 = var_29.render_field(field_name=var_15, field=var_41, value=var_42)
    var_44 = var_28.fields[var_11]
    var_45 = 'Error message'
    var_46 = var_29.render_field(field_name=var_11, field=var_44, value=var_31, error=var_45)
    var_47 = 'password'
    var_48 = module_2.String(format=var_47)
    var_49 = 'secret'
    var_50 = var_29.render_field(field_name=var_47, field=var_48, value=var_49)



