####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'
    var_4 = 'email'
    var_5 = module_1.String(format=var_4)
    var_6 = var_1.input_type_for_field(var_5)
    assert var_6 == 'email'
    var_7 = 'datetime'
    var_8 = module_1.String(format=var_7)
    var_9 = var_1.input_type_for_field(var_8)
    assert var_9 == 'datetime-local'
    var_10 = 'date'
    var_11 = module_1.String(format=var_10)
    var_12 = var_1.input_type_for_field(var_11)
    assert var_12 == 'date'
    var_13 = 'color'
    var_14 = module_1.String(format=var_13)
    var_15 = var_1.input_type_for_field(var_14)
    assert var_15 == 'color'
    var_16 = 'hidden'
    var_17 = module_1.String(format=var_16)
    var_18 = var_1.input_type_for_field(var_17)
    assert var_18 == 'hidden'
    var_19 = 'month'
    var_20 = module_1.String(format=var_19)
    var_21 = var_1.input_type_for_field(var_20)
    assert var_21 == 'month'
    var_22 = 'number'
    var_23 = module_1.String(format=var_22)
    var_24 = var_1.input_type_for_field(var_23)
    assert var_24 == 'number'
    var_25 = 'password'
    var_26 = module_1.String(format=var_25)
    var_27 = var_1.input_type_for_field(var_26)
    assert var_27 == 'password'
    var_28 = 'range'
    var_29 = module_1.String(format=var_28)
    var_30 = var_1.input_type_for_field(var_29)
    assert var_30 == 'range'
    var_31 = 'search'
    var_32 = module_1.String(format=var_31)
    var_33 = var_1.input_type_for_field(var_32)
    assert var_33 == 'search'
    var_34 = 'tel'
    var_35 = module_1.String(format=var_34)
    var_36 = var_1.input_type_for_field(var_35)
    assert var_36 == 'tel'
    var_37 = 'text'
    var_38 = module_1.String(format=var_37)
    var_39 = var_1.input_type_for_field(var_38)
    assert var_39 == 'text'
    var_40 = 'time'
    var_41 = module_1.String(format=var_40)
    var_42 = var_1.input_type_for_field(var_41)
    assert var_42 == 'time'
    var_43 = 'url'
    var_44 = module_1.String(format=var_43)
    var_45 = var_1.input_type_for_field(var_44)
    assert var_45 == 'url'
    var_46 = 'week'
    var_47 = module_1.String(format=var_46)
    var_48 = var_1.input_type_for_field(var_47)
    assert var_48 == 'week'
    var_49 = 'unknown'
    var_50 = module_1.String(format=var_49)
    var_51 = var_1.input_type_for_field(var_50)
    assert var_51 == 'text'



# Parsed testcases at query #2
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '\n        <input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" />\n        '
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'John'
    var_8 = '30'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = '300'
    var_11 = {var_5: var_7, var_6: var_10}



# Parsed testcases at query #3
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Choice()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/select.html'
    var_4 = module_1.Boolean()
    var_5 = var_1.template_for_field(var_4)
    assert var_5 == 'forms/checkbox.html'
    var_6 = 'text'
    var_7 = module_1.String(format=var_6)
    var_8 = var_1.template_for_field(var_7)
    assert var_8 == 'forms/textarea.html'
    var_9 = module_1.String()
    var_10 = var_1.template_for_field(var_9)
    assert var_10 == 'forms/input.html'



# Parsed testcases at query #4
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John Doe'
    var_6 = '30'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'thirty'
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_4: var_6}
    var_11 = 'extra'
    var_12 = 'data'
    var_13 = {var_3: var_5, var_4: var_6, var_11: var_12}
    var_14 = {}



# Parsed testcases at query #5
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input id="{{ field_id }}" name="{{ field_name }}" type="{{ input_type }}" value="{{ value }}" />'
    var_5 = '<textarea id="{{ field_id }}" name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<select id="{{ field_id }}" name="{{ field_name }}">{{ value }}</select>'
    var_7 = '<input id="{{ field_id }}" name="{{ field_name }}" type="checkbox" {% if value %}checked{% endif %} />'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'description'
    var_13 = 'active'
    var_14 = 'category'
    var_15 = 'Test'
    var_16 = 'Test description'
    var_17 = True
    var_18 = '1'
    var_19 = {var_11: var_15, var_12: var_16, var_13: var_17, var_14: var_18}
    var_20 = '<input id="name" name="name" type="text" value="Test" /><textarea id="description" name="description">Test description</textarea><input id="active" name="active" type="checkbox" checked /><select id="category" name="category">1</select>'



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = module_1.Form(env=var_1, schema=var_1)
    var_3 = var_2.input_type_for_field(var_0)
    assert var_3 == 'text'
    var_4 = 'email'
    var_5 = module_0.String(format=var_4)
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'email'
    var_7 = 'unknown'
    var_8 = module_0.String(format=var_7)
    var_9 = var_2.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = 'datetime'
    var_11 = module_0.String(format=var_10)
    var_12 = var_2.input_type_for_field(var_11)
    assert var_12 == 'datetime-local'



# Parsed testcases at query #7
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = module_1.String(format=var_2)
    var_4 = var_1.input_type_for_field(var_3)
    assert var_4 == 'email'
    var_5 = 'unknown'
    var_6 = module_1.String(format=var_5)
    var_7 = var_1.input_type_for_field(var_6)
    assert var_7 == 'text'
    var_8 = module_1.String()
    var_9 = var_1.input_type_for_field(var_8)
    assert var_9 == 'text'



# Parsed testcases at query #8
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}">'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'Test'
    var_7 = {var_5: var_6}



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import jinja2.environment as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.Environment()
    var_5 = module_3.Form(env=var_4, schema=var_3)
    var_6 = var_5.__html__()
    var_7 = True
    var_8 = module_0.String()
    var_9 = {var_0: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = module_2.Environment()
    var_12 = module_3.Form(env=var_11, schema=var_10)
    var_13 = {}
    var_14 = var_12.validate(var_13)
    var_15 = var_12.__html__()
    var_16 = 'user'
    var_17 = module_0.String()
    var_18 = {var_0: var_17}
    var_19 = module_0.Object(properties=var_18)
    var_20 = {var_16: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = module_2.Environment()
    var_23 = module_3.Form(env=var_22, schema=var_21)
    var_24 = var_23.__html__()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = '.'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'Name'
    var_5 = module_2.String()
    var_6 = {var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = module_4.Form(env=var_2, schema=var_7)
    var_9 = module_2.String()
    var_10 = 'John'
    var_11 = var_8.render_field(field_name=var_3, field=var_9, value=var_10)



# Parsed testcases at query #13
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'typesystem'
    var_1 = module_0.Jinja2Forms(package=var_0)
    var_2 = var_1.env.loader
    var_3 = var_1.env.loader
    var_4 = module_0.Jinja2Forms()



# Parsed testcases at query #14
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0, values=var_0)
    var_2 = module_1.String()
    var_3 = var_1.template_for_field(var_2)
    assert var_3 == 'forms/input.html'
    var_4 = 'text'
    var_5 = module_1.String(format=var_4)
    var_6 = var_1.template_for_field(var_5)
    assert var_6 == 'forms/textarea.html'
    var_7 = module_1.Boolean()
    var_8 = var_1.template_for_field(var_7)
    assert var_8 == 'forms/checkbox.html'
    var_9 = module_1.Choice()
    var_10 = var_1.template_for_field(var_9)
    assert var_10 == 'forms/select.html'



# Parsed testcases at query #15
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = module_0.Jinja2Forms()



# Parsed testcases at query #16
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = var_4.env.loader



# Parsed testcases at query #17
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}">'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}">{{ value }}</textarea>'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}">{{ value }}</select>'
    var_7 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'select_field'
    var_13 = 'checkbox_field'
    var_14 = True
    var_15 = 'input_field'
    var_16 = 'Test Value'



# Parsed testcases at query #18
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
    var_4 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_5 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <textarea id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_6 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <select id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>\n                {% for choice in field.choices %}\n                    <option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>\n                {% endfor %}\n            </select>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_7 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" value="true" {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'username'
    var_12 = 'password'
    var_13 = 'description'
    var_14 = 'role'
    var_15 = 'is_active'
    var_16 = 'Username'
    var_17 = module_2.String()
    var_18 = 'Password'
    var_19 = module_2.String(format=var_12)
    var_20 = 'Description'
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = 'Role'
    var_24 = 'admin'
    var_25 = 'Admin'
    var_26 = (var_24, var_25)
    var_27 = 'user'
    var_28 = 'User'
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = module_2.Choice(choices=var_30)
    var_32 = 'Is Active'
    var_33 = module_2.Boolean()
    var_34 = {var_11: var_17, var_12: var_19, var_13: var_22, var_14: var_31, var_15: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_4.Form(env=var_10, schema=var_35)
    var_37 = var_35.fields[var_11]
    var_38 = var_36.render_field(field_name=var_11, field=var_37)
    assert var_38 == '\n        <div>\n            <label for="username">Username</label>\n            <input type="text" id="username" name="username" value="" required>\n            <p></p>\n        </div>\n    '
    var_39 = var_35.fields[var_12]
    var_40 = var_36.render_field(field_name=var_12, field=var_39)
    assert var_40 == '\n        <div>\n            <label for="password">Password</label>\n            <input type="password" id="password" name="password" value="" required>\n            <p></p>\n        </div>\n    '
    var_41 = var_35.fields[var_13]
    var_42 = var_36.render_field(field_name=var_13, field=var_41)
    assert var_42 == '\n        <div>\n            <label for="description">Description</label>\n            <textarea id="description" name="description" required></textarea>\n            <p></p>\n        </div>\n    '
    var_43 = var_35.fields[var_14]
    var_44 = var_36.render_field(field_name=var_14, field=var_43)
    assert var_44 == '\n        <div>\n            <label for="role">Role</label>\n            <select id="role" name="role" required>\n                <option value="admin">Admin</option>\n                <option value="user">User</option>\n            </select>\n            <p></p>\n        </div>\n    '
    var_45 = var_35.fields[var_15]
    var_46 = var_36.render_field(field_name=var_15, field=var_45)
    assert var_46 == '\n        <div>\n            <label for="is-active">Is Active</label>\n            <input type="checkbox" id="is-active" name="is_active" value="true"  required>\n            <p></p>\n        </div>\n    '



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
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.display }}</option>{% endfor %}</select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'username'
    var_12 = 'password'
    var_13 = 'email'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'role'
    var_17 = 'Username'
    var_18 = 100
    var_19 = module_2.String(max_length=var_18)
    var_20 = 'Password'
    var_21 = module_2.String(format=var_12)
    var_22 = 'Email'
    var_23 = module_2.String(format=var_13)
    var_24 = 'Bio'
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = 'Active'
    var_28 = module_2.Boolean()
    var_29 = 'Role'
    var_30 = 'value'
    var_31 = 'display'
    var_32 = 'admin'
    var_33 = 'Administrator'
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = 'user'
    var_36 = 'User'
    var_37 = {var_30: var_35, var_31: var_36}
    var_38 = [var_34, var_37]
    var_39 = module_2.Choice(choices=var_38)
    var_40 = {var_11: var_19, var_12: var_21, var_13: var_23, var_14: var_26, var_15: var_28, var_16: var_39}
    var_41 = module_3.Schema(var_40)
    var_42 = module_4.Form(env=var_10, schema=var_41)
    var_43 = var_41.fields[var_11]
    var_44 = var_42.render_field(field_name=var_11, field=var_43)
    assert var_44 == '<input type="text" name="username" id="username" value="" required>'
    var_45 = var_41.fields[var_12]
    var_46 = var_42.render_field(field_name=var_12, field=var_45)
    assert var_46 == '<input type="password" name="password" id="password" value="" required>'
    var_47 = var_41.fields[var_13]
    var_48 = var_42.render_field(field_name=var_13, field=var_47)
    assert var_48 == '<input type="email" name="email" id="email" value="" required>'
    var_49 = var_41.fields[var_14]
    var_50 = var_42.render_field(field_name=var_14, field=var_49)
    assert var_50 == '<textarea name="bio" id="bio" required></textarea>'
    var_51 = var_41.fields[var_15]
    var_52 = var_42.render_field(field_name=var_15, field=var_51)
    assert var_52 == '<input type="checkbox" name="active" id="active">'
    var_53 = var_41.fields[var_16]
    var_54 = var_42.render_field(field_name=var_16, field=var_53)
    assert var_54 == '<select name="role" id="role" required><option value="admin">Administrator</option><option value="user">User</option></select>'



# Parsed testcases at query #20
#--------------------------


import jinja2.environment as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2
import typesystem.fields as module_3

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Schema()
    var_2 = module_2.Form(env=var_0, schema=var_1)
    var_3 = '1'
    var_4 = 'One'
    var_5 = (var_3, var_4)
    var_6 = '2'
    var_7 = 'Two'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = module_3.Choice(choices=var_9)
    var_11 = var_2.template_for_field(var_10)
    assert var_11 == 'forms/select.html'
    var_12 = module_3.Boolean()
    var_13 = var_2.template_for_field(var_12)
    assert var_13 == 'forms/checkbox.html'
    var_14 = 'text'
    var_15 = module_3.String(format=var_14)
    var_16 = var_2.template_for_field(var_15)
    assert var_16 == 'forms/textarea.html'
    var_17 = module_3.String()
    var_18 = var_2.template_for_field(var_17)
    assert var_18 == 'forms/input.html'
    var_19 = {}
    var_20 = module_3.Object(properties=var_19)
    var_21 = var_2.template_for_field(var_20)



# Parsed testcases at query #21
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader
    var_4 = 'my_package'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = var_5.env.loader
    var_8 = '/path/to/templates'
    var_9 = 'my_package'
    var_10 = module_0.Jinja2Forms(directory=var_8, package=var_9)
    var_11 = var_10.env
    var_12 = var_10.env.loader
    var_13 = var_10.env.loader.loaders
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 0
    var_16 = var_10.env.loader.loaders[var_15]
    var_17 = 1
    var_18 = var_10.env.loader.loaders[var_17]



# Parsed testcases at query #22
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = var_1.env.loader
    var_4 = 'myapp'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = var_5.env
    var_7 = var_5.env.loader
    var_8 = module_0.Jinja2Forms(directory=var_0, package=var_4)
    var_9 = var_8.env
    var_10 = var_8.env.loader
    var_11 = module_0.Jinja2Forms()



# Parsed testcases at query #23
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = 'test_dir'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'test_pkg'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = 'test_dir'
    var_6 = 'test_pkg'
    var_7 = module_0.Jinja2Forms(directory=var_5, package=var_6)



# Parsed testcases at query #24
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #25
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
    var_4 = "<input id='{{ field_id }}' name='{{ field_name }}' type='{{ input_type }}' value='{{ value }}'>"
    var_5 = "<textarea id='{{ field_id }}' name='{{ field_name }}'>{{ value }}</textarea>"
    var_6 = "<input id='{{ field_id }}' name='{{ field_name }}' type='checkbox' {% if value %}checked{% endif %}>"
    var_7 = "<select id='{{ field_id }}' name='{{ field_name }}'></select>"
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'test_field'
    var_12 = 'Test Field'
    var_13 = module_2.String()
    var_14 = {var_11: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_4.Form(env=var_10, schema=var_15)
    var_17 = 'test_field'
    var_18 = module_2.String()
    var_19 = var_16.render_field(field_name=var_17, field=var_18)
    assert var_19 == "<input id='test-field' name='test_field' type='text' value=''>"
    var_20 = 'text'
    var_21 = module_2.String(format=var_20)
    var_22 = var_16.render_field(field_name=var_17, field=var_21)
    assert var_22 == "<textarea id='test-field' name='test_field'></textarea>"
    var_23 = module_2.Boolean()
    var_24 = var_16.render_field(field_name=var_17, field=var_23)
    assert var_24 == "<input id='test-field' name='test_field' type='checkbox'>"
    var_25 = '1'
    var_26 = 'One'
    var_27 = (var_25, var_26)
    var_28 = '2'
    var_29 = 'Two'
    var_30 = (var_28, var_29)
    var_31 = [var_27, var_30]
    var_32 = module_2.Choice(choices=var_31)
    var_33 = var_16.render_field(field_name=var_17, field=var_32)
    assert var_33 == "<select id='test-field' name='test_field'></select>"



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Name'
    var_1 = module_0.String()
    var_2 = None
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = module_1.Form(env=var_2, schema=var_2, values=var_5)
    var_7 = var_6.render_field(field_name=var_3, field=var_1, value=var_4)
    var_8 = 'Age'
    var_9 = module_0.Integer()
    var_10 = 'age'
    var_11 = 25
    var_12 = {var_10: var_11}
    var_13 = module_1.Form(env=var_2, schema=var_2, values=var_12)
    var_14 = var_13.render_field(field_name=var_10, field=var_9, value=var_11)
    var_15 = 'Active'
    var_16 = module_0.Boolean()
    var_17 = 'active'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = module_1.Form(env=var_2, schema=var_2, values=var_19)
    var_21 = var_20.render_field(field_name=var_17, field=var_16, value=var_18)
    var_22 = 'Color'
    var_23 = 'red'
    var_24 = 'Red'
    var_25 = (var_23, var_24)
    var_26 = 'blue'
    var_27 = 'Blue'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = 'color'
    var_32 = {var_31: var_23}
    var_33 = module_1.Form(env=var_2, schema=var_2, values=var_32)
    var_34 = var_33.render_field(field_name=var_31, field=var_30, value=var_23)
    var_35 = module_0.String()
    var_36 = ''
    var_37 = {var_3: var_36}
    var_38 = module_1.Form(env=var_2, schema=var_2, values=var_37)
    var_39 = 'Required'
    var_40 = var_38.render_field(field_name=var_3, field=var_35, value=var_36, error=var_39)



# Parsed testcases at query #27
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'tests'
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



# Parsed testcases at query #28
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = 'templates'
    var_5 = 'tests'
    var_6 = module_0.Jinja2Forms(directory=var_4, package=var_5)
    var_7 = module_0.Jinja2Forms()



# Parsed testcases at query #29
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
    var_8 = var_6.env.loader.loaders
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_6.env.loader.loaders[var_10]
    var_12 = 1
    var_13 = var_6.env.loader.loaders[var_12]
    var_14 = module_0.Jinja2Forms()
    var_15 = module_0.Jinja2Forms(directory=var_0)
    var_16 = module_0.Jinja2Forms(directory=var_0)
    var_17 = False
    var_18 = module_0.Jinja2Forms(directory=var_0)
    var_19 = True
    var_20 = lambda x: var_19
    var_21 = module_0.Jinja2Forms(directory=var_0)
    var_22 = False
    var_23 = lambda x: var_22
    var_24 = module_0.Jinja2Forms(directory=var_0)
    var_25 = True
    var_26 = lambda x: var_25
    var_27 = module_0.Jinja2Forms(directory=var_0)
    var_28 = False
    var_29 = lambda x: var_28



# Parsed testcases at query #30
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'tests'
    var_1 = 'templates'
    var_2 = module_0.PackageLoader(var_0, var_1)
    var_3 = module_1.Environment(loader=var_2)
    var_4 = module_2.Integer()
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'test'
    var_8 = 20
    var_9 = {var_5: var_7, var_6: var_8}



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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}></select>'
    var_7 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} {% if required %}required{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'username'
    var_12 = 'password'
    var_13 = 'bio'
    var_14 = 'is_active'
    var_15 = 'role'
    var_16 = 'Username'
    var_17 = 100
    var_18 = module_2.String(max_length=var_17)
    var_19 = 'Password'
    var_20 = module_2.String(max_length=var_17, format=var_12)
    var_21 = 'Bio'
    var_22 = 'text'
    var_23 = module_2.String(format=var_22)
    var_24 = 'Is Active'
    var_25 = module_2.Boolean()
    var_26 = 'Role'
    var_27 = 'admin'
    var_28 = 'Admin'
    var_29 = (var_27, var_28)
    var_30 = 'user'
    var_31 = 'User'
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = module_2.Choice(choices=var_33)
    var_35 = {var_11: var_18, var_12: var_20, var_13: var_23, var_14: var_25, var_15: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_4.Form(env=var_10, schema=var_36)
    var_38 = var_36.fields[var_11]
    var_39 = var_37.render_field(field_name=var_11, field=var_38)
    var_40 = var_36.fields[var_12]
    var_41 = var_37.render_field(field_name=var_12, field=var_40)
    var_42 = var_36.fields[var_13]
    var_43 = var_37.render_field(field_name=var_13, field=var_42)
    var_44 = var_36.fields[var_14]
    var_45 = True
    var_46 = var_37.render_field(field_name=var_14, field=var_44, value=var_45)
    var_47 = var_36.fields[var_15]
    var_48 = var_37.render_field(field_name=var_15, field=var_47)



# Parsed testcases at query #32
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'typesystem.forms'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)



# Parsed testcases at query #33
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'tests'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = var_6.env.loader.loaders
    var_9 = 0
    var_10 = var_8[var_9]
    var_11 = 1
    var_12 = var_8[var_11]



# Parsed testcases at query #34
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'TestSchema'
    var_4 = 'test_field'
    var_5 = 'Test Field'
    var_6 = module_2.String()
    var_7 = {var_4: var_6}
    var_8 = module_3.Schema(var_7)
    var_9 = module_4.Form(env=var_2, schema=var_8)
    var_10 = module_2.String()
    var_11 = var_9.render_field(field_name=var_4, field=var_10)



# Parsed testcases at query #35
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'Name'
    var_5 = module_2.String()
    var_6 = {var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = module_4.Form(env=var_2, schema=var_7)
    var_9 = 'name'
    var_10 = module_2.String()
    var_11 = 'John Doe'
    var_12 = None
    var_13 = var_8.render_field(field_name=var_9, field=var_10, value=var_11, error=var_12)
    var_14 = 'Invalid name'
    var_15 = var_8.render_field(field_name=var_9, field=var_10, value=var_11, error=var_14)
    var_16 = '1'
    var_17 = 'One'
    var_18 = (var_16, var_17)
    var_19 = '2'
    var_20 = 'Two'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = module_2.Choice(choices=var_22)
    var_24 = var_8.render_field(field_name=var_9, field=var_23, value=var_11, error=var_14)
    var_25 = module_2.Boolean()
    var_26 = var_8.render_field(field_name=var_9, field=var_25, value=var_11, error=var_14)
    var_27 = 'text'
    var_28 = module_2.String(format=var_27)
    var_29 = var_8.render_field(field_name=var_9, field=var_28, value=var_11, error=var_14)
    var_30 = module_2.String()
    var_31 = {var_3: var_30}
    var_32 = module_2.Object(properties=var_31)
    var_33 = var_8.render_field(field_name=var_9, field=var_32, value=var_11, error=var_14)



# Parsed testcases at query #36
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #37
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'username'
    var_4 = 'Username'
    var_5 = module_2.String()
    var_6 = {var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = module_4.Form(env=var_2, schema=var_7)
    var_9 = module_2.String()
    var_10 = var_8.render_field(field_name=var_3, field=var_9)



# Parsed testcases at query #38
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/checkbox.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/textarea.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %} {% if required %}required{% endif %}>'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>{% endfor %}</select>'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'Name'
    var_13 = module_2.String()
    var_14 = 'John'
    var_15 = 'active'
    var_16 = 'Active'
    var_17 = module_2.Boolean()
    var_18 = True
    var_19 = 'a'
    var_20 = 'Option A'
    var_21 = (var_19, var_20)
    var_22 = 'b'
    var_23 = 'Option B'
    var_24 = (var_22, var_23)
    var_25 = [var_21, var_24]
    var_26 = 'choice'
    var_27 = module_2.Choice(choices=var_25)
    var_28 = 'description'
    var_29 = 'text'
    var_30 = module_2.String(format=var_29)
    var_31 = 'Test'
    var_32 = module_2.String()
    var_33 = ''
    var_34 = 'Required'
    var_35 = 'All tests passed.'
    var_36 = print(var_35)



# Parsed testcases at query #39
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = module_0.Jinja2Forms()



# Parsed testcases at query #40
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'tests'
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



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
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
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_7 = '<select name="{{ field_name }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}" {% if choice == value %}selected{% endif %}>{{ choice }}</option>{% endfor %}</select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'number_field'
    var_13 = 'checkbox_field'
    var_14 = 'select_field'
    var_15 = 'Text Field'
    var_16 = module_2.String()
    var_17 = 'number'
    var_18 = 'Number Field'
    var_19 = module_2.String(format=var_17)
    var_20 = 'Checkbox Field'
    var_21 = module_2.Boolean()
    var_22 = 'option1'
    var_23 = 'option2'
    var_24 = [var_22, var_23]
    var_25 = 'Select Field'
    var_26 = module_2.Choice(choices=var_24)
    var_27 = {var_11: var_16, var_12: var_19, var_13: var_21, var_14: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = module_2.String()
    var_31 = 'test value'
    var_32 = var_29.render_field(field_name=var_11, field=var_30, value=var_31)
    assert var_32 == '<input type="text" name="text_field" value="test value" >'
    var_33 = module_2.String(format=var_17)
    var_34 = 42
    var_35 = var_29.render_field(field_name=var_12, field=var_33, value=var_34)
    assert var_35 == '<input type="number" name="number_field" value="42" >'
    var_36 = module_2.Boolean()
    var_37 = True
    var_38 = var_29.render_field(field_name=var_13, field=var_36, value=var_37)
    assert var_38 == '<input type="checkbox" name="checkbox_field" checked>'
    var_39 = [var_22, var_23]
    var_40 = module_2.Choice(choices=var_39)
    var_41 = var_29.render_field(field_name=var_14, field=var_40, value=var_22)
    assert var_41 == '<select name="select_field" ><option value="option1" selected>option1</option><option value="option2" >option2</option></select>'



# Parsed testcases at query #2
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'email'
    var_5 = 'age'
    var_6 = 'John'
    var_7 = 'john@example.com'
    var_8 = 30
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = ''
    var_11 = 'invalid'
    var_12 = 'not a number'
    var_13 = {var_3: var_10, var_4: var_11, var_5: var_12}



# Parsed testcases at query #3
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = module_1.Field()
    var_3 = var_1.input_type_for_field(var_2)
    assert var_3 == 'text'
    var_4 = 'email'
    var_5 = module_1.String(format=var_4)
    var_6 = var_1.input_type_for_field(var_5)
    assert var_6 == 'email'
    var_7 = 'datetime'
    var_8 = module_1.String(format=var_7)
    var_9 = var_1.input_type_for_field(var_8)
    assert var_9 == 'datetime-local'
    var_10 = 'date'
    var_11 = module_1.String(format=var_10)
    var_12 = var_1.input_type_for_field(var_11)
    assert var_12 == 'date'
    var_13 = 'color'
    var_14 = module_1.String(format=var_13)
    var_15 = var_1.input_type_for_field(var_14)
    assert var_15 == 'color'
    var_16 = 'month'
    var_17 = module_1.String(format=var_16)
    var_18 = var_1.input_type_for_field(var_17)
    assert var_18 == 'month'
    var_19 = 'number'
    var_20 = module_1.String(format=var_19)
    var_21 = var_1.input_type_for_field(var_20)
    assert var_21 == 'number'
    var_22 = 'password'
    var_23 = module_1.String(format=var_22)
    var_24 = var_1.input_type_for_field(var_23)
    assert var_24 == 'password'
    var_25 = 'range'
    var_26 = module_1.String(format=var_25)
    var_27 = var_1.input_type_for_field(var_26)
    assert var_27 == 'range'
    var_28 = 'search'
    var_29 = module_1.String(format=var_28)
    var_30 = var_1.input_type_for_field(var_29)
    assert var_30 == 'search'
    var_31 = 'tel'
    var_32 = module_1.String(format=var_31)
    var_33 = var_1.input_type_for_field(var_32)
    assert var_33 == 'tel'
    var_34 = 'time'
    var_35 = module_1.String(format=var_34)
    var_36 = var_1.input_type_for_field(var_35)
    assert var_36 == 'time'
    var_37 = 'url'
    var_38 = module_1.String(format=var_37)
    var_39 = var_1.input_type_for_field(var_38)
    assert var_39 == 'url'
    var_40 = 'week'
    var_41 = module_1.String(format=var_40)
    var_42 = var_1.input_type_for_field(var_41)
    assert var_42 == 'week'
    var_43 = 'unknown'
    var_44 = module_1.String(format=var_43)
    var_45 = var_1.input_type_for_field(var_44)
    assert var_45 == 'text'



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = None
    var_9 = module_0.Boolean()
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = 'email'
    var_13 = module_0.String(format=var_12)
    var_14 = 'name'
    var_15 = module_0.String()
    var_16 = {var_14: var_15}
    var_17 = module_0.Object(properties=var_16)
    var_18 = None



# Parsed testcases at query #5
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'a'
    var_3 = 'A'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 'B'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_1.Choice(choices=var_8)
    var_10 = var_1.template_for_field(var_9)
    assert var_10 == 'forms/select.html'
    var_11 = module_1.Boolean()
    var_12 = var_1.template_for_field(var_11)
    assert var_12 == 'forms/checkbox.html'
    var_13 = 'text'
    var_14 = module_1.String(format=var_13)
    var_15 = var_1.template_for_field(var_14)
    assert var_15 == 'forms/textarea.html'
    var_16 = module_1.String()
    var_17 = var_1.template_for_field(var_16)
    assert var_17 == 'forms/input.html'
    var_18 = {}
    var_19 = module_1.Object(properties=var_18)
    var_20 = var_1.template_for_field(var_19)



# Parsed testcases at query #6
#--------------------------


import typesystem.forms as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'a'
    var_3 = 'A'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_1.Choice(choices=var_5)
    var_7 = var_1.template_for_field(var_6)
    assert var_7 == 'forms/select.html'
    var_8 = module_1.Boolean()
    var_9 = var_1.template_for_field(var_8)
    assert var_9 == 'forms/checkbox.html'
    var_10 = 'text'
    var_11 = module_1.String(format=var_10)
    var_12 = var_1.template_for_field(var_11)
    assert var_12 == 'forms/textarea.html'
    var_13 = module_1.String()
    var_14 = var_1.template_for_field(var_13)
    assert var_14 == 'forms/input.html'



# Parsed testcases at query #7
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'name'



# Parsed testcases at query #8
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)



# Parsed testcases at query #9
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = './templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = '30'
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #10
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_directory'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #11
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3
import typesystem.fields as module_4

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = module_2.Schema()
    var_4 = module_3.Form(env=var_2, schema=var_3)
    var_5 = 'option1'
    var_6 = 'Option 1'
    var_7 = (var_5, var_6)
    var_8 = 'option2'
    var_9 = 'Option 2'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_4.Choice(choices=var_11)
    var_13 = var_4.template_for_field(var_12)
    assert var_13 == 'forms/select.html'
    var_14 = module_4.Boolean()
    var_15 = var_4.template_for_field(var_14)
    assert var_15 == 'forms/checkbox.html'
    var_16 = 'text'
    var_17 = module_4.String(format=var_16)
    var_18 = var_4.template_for_field(var_17)
    assert var_18 == 'forms/textarea.html'
    var_19 = 'email'
    var_20 = module_4.String(format=var_19)
    var_21 = var_4.template_for_field(var_20)
    assert var_21 == 'forms/input.html'
    var_22 = module_4.String()
    var_23 = var_4.template_for_field(var_22)
    assert var_23 == 'forms/input.html'
    var_24 = module_4.Object()
    var_25 = var_4.template_for_field(var_24)



# Parsed testcases at query #12
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'forms'
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



# Parsed testcases at query #13
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'tests'
    var_6 = module_0.Jinja2Forms(package=var_5)



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
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_5 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <textarea id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>{{ value }}</textarea>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_6 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <input type="checkbox" id="{{ field_id }}" name="{{ field_name }}" {% if value %}checked{% endif %} {% if required %}required{% endif %}>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_7 = '\n        <div>\n            <label for="{{ field_id }}">{{ label }}</label>\n            <select id="{{ field_id }}" name="{{ field_name }}" {% if required %}required{% endif %}>\n                {% for choice in field.choices %}\n                    <option value="{{ choice }}" {% if choice == value %}selected{% endif %}>{{ choice }}</option>\n                {% endfor %}\n            </select>\n            {% if error %}<p>{{ error }}</p>{% endif %}\n        </div>\n        '
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'Name'
    var_13 = 1
    var_14 = module_2.String(min_length=var_13)
    var_15 = {var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_4.Form(env=var_10, schema=var_16)
    var_18 = module_2.String(min_length=var_13)
    var_19 = 'John'
    var_20 = None
    var_21 = var_17.render_field(field_name=var_11, field=var_18, value=var_19, error=var_20)
    var_22 = '\n        <div>\n            <label for="name">Name</label>\n            <input type="text" id="name" name="name" value="John" required>\n        </div>\n    '
    var_23 = 'bio'
    var_24 = 'Bio'
    var_25 = 'text'
    var_26 = module_2.String(format=var_25)
    var_27 = {var_23: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_4.Form(env=var_10, schema=var_28)
    var_30 = module_2.String(format=var_25)
    var_31 = 'Hello, World!'
    var_32 = var_29.render_field(field_name=var_23, field=var_30, value=var_31, error=var_20)
    var_33 = '\n        <div>\n            <label for="bio">Bio</label>\n            <textarea id="bio" name="bio" required>Hello, World!</textarea>\n        </div>\n    '
    var_34 = 'active'
    var_35 = 'Active'
    var_36 = module_2.Boolean()
    var_37 = {var_34: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_4.Form(env=var_10, schema=var_38)
    var_40 = module_2.Boolean()
    var_41 = True
    var_42 = var_39.render_field(field_name=var_34, field=var_40, value=var_41, error=var_20)
    var_43 = '\n        <div>\n            <label for="active">Active</label>\n            <input type="checkbox" id="active" name="active" checked>\n        </div>\n    '
    var_44 = 'Red'
    var_45 = 'Green'
    var_46 = 'Blue'
    var_47 = [var_44, var_45, var_46]
    var_48 = 'color'
    var_49 = 'Color'
    var_50 = module_2.Choice(choices=var_47)
    var_51 = {var_48: var_50}
    var_52 = module_3.Schema(var_51)
    var_53 = module_4.Form(env=var_10, schema=var_52)
    var_54 = module_2.Choice(choices=var_47)
    var_55 = var_53.render_field(field_name=var_48, field=var_54, value=var_45, error=var_20)
    var_56 = '\n        <div>\n            <label for="color">Color</label>\n            <select id="color" name="color" required>\n                <option value="Red">Red</option>\n                <option value="Green" selected>Green</option>\n                <option value="Blue">Blue</option>\n            </select>\n        </div>\n    '



# Parsed testcases at query #15
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)



# Parsed testcases at query #16
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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %}>'
    var_6 = '<select name="{{ field_name }}">{% for choice in field.choices %}<option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.display }}</option>{% endfor %}</select>'
    var_7 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'Test String'
    var_12 = 'text'
    var_13 = module_2.String(format=var_12)
    var_14 = 'test_field'
    var_15 = {var_14: var_13}
    var_16 = module_3.Schema(var_15)
    var_17 = module_4.Form(env=var_10, schema=var_16)
    var_18 = 'test value'
    var_19 = var_17.render_field(field_name=var_14, field=var_13, value=var_18)
    assert var_19 == '<textarea name="test_field">test value</textarea>'
    var_20 = 'Test Boolean'
    var_21 = module_2.Boolean()
    var_22 = {var_14: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = module_4.Form(env=var_10, schema=var_23)
    var_25 = True
    var_26 = var_24.render_field(field_name=var_14, field=var_21, value=var_25)
    assert var_26 == '<input type="checkbox" name="test_field" checked>'
    var_27 = 'option1'
    var_28 = 'Option 1'
    var_29 = (var_27, var_28)
    var_30 = 'option2'
    var_31 = 'Option 2'
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = module_2.Choice(choices=var_33)
    var_35 = {var_14: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_4.Form(env=var_10, schema=var_36)
    var_38 = var_37.render_field(field_name=var_14, field=var_34, value=var_27)
    assert var_38 == '<select name="test_field"><option value="option1" selected>Option 1</option><option value="option2">Option 2</option></select>'
    var_39 = 'Test Email'
    var_40 = 'email'
    var_41 = module_2.String(format=var_40)
    var_42 = {var_14: var_41}
    var_43 = module_3.Schema(var_42)
    var_44 = module_4.Form(env=var_10, schema=var_43)
    var_45 = 'test@example.com'
    var_46 = var_44.render_field(field_name=var_14, field=var_41, value=var_45)
    assert var_46 == '<input type="email" name="test_field" value="test@example.com">'
    var_47 = 'Test Error'
    var_48 = module_2.String()
    var_49 = {var_14: var_48}
    var_50 = module_3.Schema(var_49)
    var_51 = module_4.Form(env=var_10, schema=var_50)
    var_52 = ''
    var_53 = 'This field is required'
    var_54 = var_51.render_field(field_name=var_14, field=var_48, value=var_52, error=var_53)



# Parsed testcases at query #17
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import jinja2.environment as module_1
import typesystem.schemas as module_2
import typesystem.forms as module_3

def test_case_0():
    var_0 = '1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.Environment()
    var_9 = {}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.Form(env=var_8, schema=var_10)
    var_12 = var_11.template_for_field(var_7)
    assert var_12 == 'forms/select.html'
    var_13 = module_0.Boolean()
    var_14 = var_11.template_for_field(var_13)
    assert var_14 == 'forms/checkbox.html'
    var_15 = 'text'
    var_16 = module_0.String(format=var_15)
    var_17 = var_11.template_for_field(var_16)
    assert var_17 == 'forms/textarea.html'
    var_18 = module_0.String()
    var_19 = var_11.template_for_field(var_18)
    assert var_19 == 'forms/input.html'
    var_20 = {}
    var_21 = module_0.Object(properties=var_20)
    var_22 = var_11.template_for_field(var_21)



# Parsed testcases at query #19
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = var_4.env.loader



# Parsed testcases at query #20
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'username'
    var_4 = 'Username'
    var_5 = module_2.String()
    var_6 = {var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = module_4.Form(env=var_2, schema=var_7)
    var_9 = module_2.String()
    var_10 = 'testuser'
    var_11 = var_8.render_field(field_name=var_3, field=var_9, value=var_10)



# Parsed testcases at query #21
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = 'Expected AssertionError when neither directory nor package is specified'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.forms as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'tests/templates'
    var_5 = module_2.Jinja2Forms(directory=var_4)
    var_6 = var_5.create_form(var_3)
    var_7 = 'test'
    var_8 = {var_0: var_7}
    var_9 = var_5.create_form(var_3, var_8)
    var_10 = 123
    var_11 = {var_0: var_10}
    var_12 = var_5.create_form(var_3, var_11)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1

def test_case_0():
    var_0 = 'Test Field'
    var_1 = True
    var_2 = module_0.String()
    var_3 = None
    var_4 = module_1.Form(env=var_3, schema=var_3, values=var_3)
    var_5 = 'test_field'
    var_6 = var_4.render_field(field_name=var_5, field=var_2)
    var_7 = 'Test Checkbox'
    var_8 = module_0.Boolean()
    var_9 = 'test_checkbox'
    var_10 = var_4.render_field(field_name=var_9, field=var_8)
    var_11 = 'option1'
    var_12 = 'Option 1'
    var_13 = (var_11, var_12)
    var_14 = 'option2'
    var_15 = 'Option 2'
    var_16 = (var_14, var_15)
    var_17 = [var_13, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = 'test_choice'
    var_20 = var_4.render_field(field_name=var_19, field=var_18)
    var_21 = 'Test Textarea'
    var_22 = 'text'
    var_23 = module_0.String(format=var_22)
    var_24 = 'test_textarea'
    var_25 = var_4.render_field(field_name=var_24, field=var_23)
    var_26 = 'Test Email'
    var_27 = 'email'
    var_28 = module_0.String(format=var_27)
    var_29 = 'test_email'
    var_30 = var_4.render_field(field_name=var_29, field=var_28)
    var_31 = 'Test Password'
    var_32 = 'password'
    var_33 = module_0.String(format=var_32)
    var_34 = 'test_password'
    var_35 = var_4.render_field(field_name=var_34, field=var_33)
    var_36 = 'Required Field'
    var_37 = False
    var_38 = module_0.String()
    var_39 = 'required_field'
    var_40 = var_4.render_field(field_name=var_39, field=var_38)
    var_41 = 'Field with Value'
    var_42 = module_0.String()
    var_43 = 'field_with_value'
    var_44 = 'test value'
    var_45 = var_4.render_field(field_name=var_43, field=var_42, value=var_44)
    var_46 = 'Field with Error'
    var_47 = module_0.String()
    var_48 = 'field_with_error'
    var_49 = 'Test error'
    var_50 = var_4.render_field(field_name=var_48, field=var_47, error=var_49)
    var_51 = 'Custom Format Field'
    var_52 = 'color'
    var_53 = module_0.String(format=var_52)
    var_54 = 'custom_format_field'
    var_55 = var_4.render_field(field_name=var_54, field=var_53)



# Parsed testcases at query #24
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = '30'
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #25
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)



# Parsed testcases at query #26
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John'
    var_6 = 30
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = ''
    var_9 = -5
    var_10 = {var_3: var_8, var_4: var_9}



# Parsed testcases at query #27
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env.loader
    var_5 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_6 = var_5.env.loader.loaders
    var_7 = len(var_6)
    assert var_7 == 2
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
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" id="{{ field_id }}" {% if required %}required{% endif %}>'
    var_5 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice }}" {% if choice == value %}selected{% endif %}>{{ choice }}</option>{% endfor %}</select>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>'
    var_7 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'field_name'
    var_12 = 'choice_field'
    var_13 = 'bool_field'
    var_14 = 'text_field'
    var_15 = 'Test Field'
    var_16 = 'email'
    var_17 = module_2.String(format=var_16)
    var_18 = 'option1'
    var_19 = 'option2'
    var_20 = [var_18, var_19]
    var_21 = module_2.Choice(choices=var_20)
    var_22 = module_2.Boolean()
    var_23 = 'text'
    var_24 = module_2.String(format=var_23)
    var_25 = {var_11: var_17, var_12: var_21, var_13: var_22, var_14: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_4.Form(env=var_10, schema=var_26)
    var_28 = module_2.String(format=var_16)
    var_29 = 'test@example.com'
    var_30 = var_27.render_field(field_name=var_11, field=var_28, value=var_29)
    assert var_30 == '<input type="email" name="field_name" value="test@example.com" id="field-name" required>'
    var_31 = [var_18, var_19]
    var_32 = module_2.Choice(choices=var_31)
    var_33 = var_27.render_field(field_name=var_12, field=var_32, value=var_18)
    assert var_33 == '<select name="choice_field" id="choice-field" required><option value="option1" selected>option1</option><option value="option2">option2</option></select>'
    var_34 = module_2.Boolean()
    var_35 = True
    var_36 = var_27.render_field(field_name=var_13, field=var_34, value=var_35)
    assert var_36 == '<input type="checkbox" name="bool_field" id="bool-field" checked>'
    var_37 = module_2.String(format=var_23)
    var_38 = 'Hello World'
    var_39 = var_27.render_field(field_name=var_14, field=var_37, value=var_38)
    assert var_39 == '<textarea name="text_field" id="text-field" required>Hello World</textarea>'
    var_40 = module_2.String(format=var_16)
    var_41 = ''
    var_42 = 'Invalid email'
    var_43 = var_27.render_field(field_name=var_11, field=var_40, value=var_41, error=var_42)
    assert var_43 == '<input type="email" name="field_name" value="" id="field-name" required>'



# Parsed testcases at query #29
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms()



# Parsed testcases at query #30
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'input template'
    var_5 = 'textarea template'
    var_6 = 'checkbox template'
    var_7 = 'select template'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'text_field'
    var_12 = 'textarea_field'
    var_13 = 'checkbox_field'
    var_14 = 'select_field'



# Parsed testcases at query #31
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" />'
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'Test Field'
    var_6 = module_2.String()
    var_7 = 'test_field'
    var_8 = {var_7: var_6}
    var_9 = module_3.Schema(var_8)
    var_10 = module_4.Form(env=var_4, schema=var_9)
    var_11 = 'test value'
    var_12 = var_10.render_field(field_name=var_7, field=var_6, value=var_11)
    assert var_12 == '<input type="text" name="test_field" id="test-field" value="test value" />'



# Parsed testcases at query #32
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '\n            <input type="{{ input_type }}" id="{{ field_id }}" name="{{ field_name }}" value="{{ value }}">\n        '
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'email'
    var_7 = 'John'
    var_8 = 'john@example.com'
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #33
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env



# Parsed testcases at query #34
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'tests'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader



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
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = 'input template'
    var_5 = 'textarea template'
    var_6 = 'checkbox template'
    var_7 = 'select template'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'TestSchema'
    var_12 = 'field1'
    var_13 = 'field2'
    var_14 = 'field3'
    var_15 = 'field4'
    var_16 = 'field5'
    var_17 = 'Field1'
    var_18 = module_2.String()
    var_19 = 'Field2'
    var_20 = 'email'
    var_21 = module_2.String(format=var_20)
    var_22 = 'Field3'
    var_23 = module_2.Boolean()
    var_24 = 'Field4'
    var_25 = '1'
    var_26 = 'One'
    var_27 = (var_25, var_26)
    var_28 = '2'
    var_29 = 'Two'
    var_30 = (var_28, var_29)
    var_31 = [var_27, var_30]
    var_32 = module_2.Choice(choices=var_31)
    var_33 = 'Field5'
    var_34 = 'text'
    var_35 = module_2.String(format=var_34)
    var_36 = {var_12: var_18, var_13: var_21, var_14: var_23, var_15: var_32, var_16: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_4.Form(env=var_10, schema=var_37)
    var_39 = var_37.fields[var_12]
    var_40 = var_38.render_field(field_name=var_12, field=var_39)
    assert var_40 == 'input template'
    var_41 = var_37.fields[var_13]
    var_42 = var_38.render_field(field_name=var_13, field=var_41)
    assert var_42 == 'input template'
    var_43 = var_37.fields[var_14]
    var_44 = var_38.render_field(field_name=var_14, field=var_43)
    assert var_44 == 'checkbox template'
    var_45 = var_37.fields[var_15]
    var_46 = var_38.render_field(field_name=var_15, field=var_45)
    assert var_46 == 'select template'
    var_47 = var_37.fields[var_16]
    var_48 = var_38.render_field(field_name=var_16, field=var_47)
    assert var_48 == 'textarea template'



# Parsed testcases at query #36
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'package_name'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env.loader
    var_8 = 0
    var_9 = var_6.env.loader.loaders[var_8]
    var_10 = 1
    var_11 = var_6.env.loader.loaders[var_10]



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 2



# Parsed testcases at query #38
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = '<div><label for="name">Name</label><input type="text" id="name" name="name" value=""></div><div><label for="age">Age</label><input type="number" id="age" name="age" value=""></div>'



# Parsed testcases at query #39
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
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



# Parsed testcases at query #40
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = '\n            <input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>\n        '
    var_2 = {var_0: var_1}
    var_3 = module_0.DictLoader(var_2)
    var_4 = module_1.Environment(loader=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'John'
    var_8 = '30'
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #41
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env
    var_3 = 'tests'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env
    var_6 = module_0.Jinja2Forms(directory=var_0, package=var_3)
    var_7 = var_6.env
    var_8 = module_0.Jinja2Forms()



# Parsed testcases at query #42
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = module_0.Jinja2Forms()



# Parsed testcases at query #43
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'



# Parsed testcases at query #44
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3
import typesystem.forms as module_4

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'field_name'
    var_4 = 'Field Title'
    var_5 = module_2.String()
    var_6 = {var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = module_4.Form(env=var_2, schema=var_7)
    var_9 = module_2.String()
    var_10 = 'test_value'
    var_11 = 'test_error'
    var_12 = var_8.render_field(field_name=var_3, field=var_9, value=var_10, error=var_11)



# Parsed testcases at query #45
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = var_4.env.loader.loaders
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = module_0.Jinja2Forms()



# Parsed testcases at query #46
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'



# Parsed testcases at query #47
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'test_templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = var_1.env.loader
    var_3 = 'test_package'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = var_4.env.loader
    var_6 = 'test_templates'
    var_7 = 'test_package'
    var_8 = module_0.Jinja2Forms(directory=var_6, package=var_7)
    var_9 = var_8.env.loader
    var_10 = var_8.env.loader.loaders
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_8.env.loader.loaders[var_12]
    var_14 = 1
    var_15 = var_8.env.loader.loaders[var_14]
    var_16 = module_0.Jinja2Forms()



# Parsed testcases at query #48
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/select.html'
    var_3 = 'forms/checkbox.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" id="{{ field_id }}" value="{{ value }}" {% if required %}required{% endif %}>'
    var_5 = '<textarea name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{{ value }}</textarea>'
    var_6 = '<select name="{{ field_name }}" id="{{ field_id }}" {% if required %}required{% endif %}>{% for choice in field.choices %}<option value="{{ choice.value }}" {% if choice.value == value %}selected{% endif %}>{{ choice.text }}</option>{% endfor %}</select>'
    var_7 = '<input type="checkbox" name="{{ field_name }}" id="{{ field_id }}" {% if value %}checked{% endif %}>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'Name'
    var_13 = module_2.String()
    var_14 = 'John'
    var_15 = 'age'
    var_16 = 'Age'
    var_17 = module_2.Integer()
    var_18 = 30
    var_19 = 'description'
    var_20 = 'Description'
    var_21 = 'text'
    var_22 = module_2.String(format=var_21)
    var_23 = 'Test description'
    var_24 = 'option1'
    var_25 = 'Option 1'
    var_26 = (var_24, var_25)
    var_27 = 'option2'
    var_28 = 'Option 2'
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = 'choice'
    var_32 = 'Choice'
    var_33 = module_2.Choice(choices=var_30)
    var_34 = 'active'
    var_35 = 'Active'
    var_36 = module_2.Boolean()
    var_37 = True



# Parsed testcases at query #49
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = module_0.Jinja2Forms()
    var_6 = 'invalid'
    var_7 = module_0.Jinja2Forms(directory=var_6)
    var_8 = 'invalid'
    var_9 = module_0.Jinja2Forms(package=var_8)



# Parsed testcases at query #50
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'tests/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = module_0.Jinja2Forms()



