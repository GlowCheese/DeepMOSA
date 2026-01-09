####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = [var_1, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = e.messages()[0].code
    assert var_10 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 3.14
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = e.messages()[0].code
    assert var_10 == 'union'


def test_case_0():
    var_0 = 10
    var_1 = 'min_value'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = 5
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = e.messages()[0].code
    assert var_12 == 'min_value'


def test_case_0():
    var_0 = 10
    var_1 = 'min_value'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = 5
    var_5 = {}
    var_6 = module_0.String(min_length=var_4, **var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.Union(var_7, **var_8)
    var_10 = 3
    var_11 = var_9.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = e.messages()[0].code
    assert var_13 == 'union'



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = len(e.messages())
    assert var_4 == 1
    var_5 = e.messages()[0].code
    assert var_5 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = len(e.messages())
    assert var_4 == 1
    var_5 = e.messages()[0].code
    assert var_5 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'invalid_key'


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'toolongkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'invalid_property'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'min_properties'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2.validate(var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'max_properties'


def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'required'


def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'key'
    var_5 = {var_4: var_3}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 'default_value'})
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'not an integer'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '^num_'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'num_1'
    var_7 = 'num_2'
    var_8 = 42
    var_9 = 100
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'num_1': 42, 'num_2': 100})
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '^num_'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'num_1'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'type'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_property'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 42
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 42})
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'type'


def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Object(min_properties=var_2, required=var_1, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = len(e.messages())
    assert var_7 == 2
    var_8 = {msg.code for msg in e.messages()}
    var_9 = bool(var_8 == {'empty', 'required'})
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------





def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  hello  '


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hi'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello world'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'Hello123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'


def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'


def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Must be a valid email.'



# Parsed testcases at query #4
#--------------------------





def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(exc.messages())
    assert var_7 == 1
    var_8 = exc.messages()[0].code
    assert var_8 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(exc.messages())
    assert var_5 == 1
    var_6 = exc.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(exc.messages())
    assert var_7 == 1
    var_8 = exc.messages()[0].code
    assert var_8 == 'invalid_key'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'longkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(exc.messages())
    assert var_10 == 1
    var_11 = exc.messages()[0].code
    assert var_11 == 'invalid_property'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(exc.messages())
    assert var_6 == 1
    var_7 = exc.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(exc.messages())
    assert var_8 == 1
    var_9 = exc.messages()[0].code
    assert var_9 == 'min_properties'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(exc.messages())
    assert var_12 == 1
    var_13 = exc.messages()[0].code
    assert var_13 == 'max_properties'


def test_case_0():
    var_0 = 'key1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(exc.messages())
    assert var_7 == 1
    var_8 = exc.messages()[0].code
    assert var_8 == 'required'


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 'default_value'})
    assert var_10 is True


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 25
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'age': 25})
    assert var_9 is True


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'age'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(exc.messages())
    assert var_11 == 1
    var_12 = exc.messages()[0].code
    assert var_12 == 'type'


def test_case_0():
    var_0 = '^x_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'x_1'
    var_7 = 'x_2'
    var_8 = 10
    var_9 = 20
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'x_1': 10, 'x_2': 20})
    assert var_12 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(exc.messages())
    assert var_8 == 1
    var_9 = exc.messages()[0].code
    assert var_9 == 'invalid_property'


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, **var_3)
    var_5 = 'extra'
    var_6 = 'too long'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(exc.messages())
    assert var_10 == 1
    var_11 = exc.messages()[0].code
    assert var_11 == 'max_length'


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = 'name'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, required=var_5, **var_6)
    var_8 = 'age'
    var_9 = 'invalid'
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = len(exc.messages())
    assert var_13 == 2
    var_14 = {msg.code for msg in exc.messages()}
    var_15 = 'type'
    var_16 = bool('type' in var_14)
    assert var_16 is True
    var_17 = 'required'
    var_18 = bool('required' in var_14)
    assert var_18 is True



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = 'test_default'
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 'test_default'


def test_case_0():
    var_0 = 'callable_result'
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'callable_result'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None


def test_case_0():
    var_0 = None
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None


def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = False
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 1
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_item_validation_error. Retrieved 4/8 statements.
# Partially parsed test_validate_with_multiple_item_validation_errors. Retrieved 4/8 statements.
# Partially parsed test_validate_with_items_list_and_item_validation_error. Retrieved 5/11 statements.
# Partially parsed test_validate_with_additional_items_field_validation_error. Retrieved 6/10 statements.
# Partially parsed test_validate_combined_unique_and_item_validation_error. Retrieved 5/8 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'exact_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = None
    var_5 = 5
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == [None, 5])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = [var_0, var_2]
    var_4 = {}
    var_5 = module_0.Array(var_3, **var_4)
    var_6 = 10
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == [10, None])
    assert var_10 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Array(var_1, var_2, **var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'additional_items'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = [var_0]
    var_4 = {}
    var_5 = module_0.Array(var_3, var_2, **var_4)
    var_6 = None
    var_7 = 3
    var_8 = [var_1, var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == [1, None, 3])
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_3]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'unique_items'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = False
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [False, True])
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 0
    var_4 = False
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [0, False])
    assert var_7 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = [var_0, var_0]
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == [1, True])
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 15
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'minimum'
    var_7 = e.messages()[0].index
    var_8 = bool(e.messages()[0].index == [0])
    assert var_8 is True

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 7
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 2
    var_6 = e.messages()[0].code
    assert var_6 == 'minimum'
    var_7 = e.messages()[0].index
    var_8 = bool(e.messages()[0].index == [0])
    assert var_8 is True
    var_9 = e.messages()[1].code
    assert var_9 == 'minimum'
    var_10 = e.messages()[1].index
    var_11 = bool(e.messages()[1].index == [1])
    assert var_11 is True

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 5
    var_3 = 10
    var_4 = [var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 2
    var_7 = e.messages()[0].code
    assert var_7 == 'minimum'
    var_8 = e.messages()[0].index
    var_9 = bool(e.messages()[0].index == [0])
    assert var_9 is True
    var_10 = e.messages()[1].code
    assert var_10 == 'maximum'
    var_11 = e.messages()[1].index
    var_12 = bool(e.messages()[1].index == [1])
    assert var_12 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 10
    var_2 = [var_0]
    var_3 = 1
    var_4 = 5
    var_5 = [var_3, var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'minimum'
    var_9 = e.messages()[0].index
    var_10 = bool(e.messages()[0].index == [1])
    assert var_10 is True

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 5
    var_3 = [var_2, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 2
    var_6 = {msg.code for msg in e.messages()}
    var_7 = bool(var_6 == {'minimum', 'unique_items'})
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 'two'
    var_4 = True
    var_5 = None
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = var_1.validate(var_6)
    var_8 = bool(var_7 == [1, 'two', True, None])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = [var_0, var_2]
    var_4 = {}
    var_5 = module_0.Array(var_3, **var_4)
    var_6 = 10
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == [10, None])
    assert var_10 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0]
    var_3 = {}
    var_4 = module_0.Array(var_2, var_1, **var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == [1, 2, 3])
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is False



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_float_for_integer_field_raises_integer_error. Retrieved 1/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == 123


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 123.5
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 12
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.2
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.234
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.23)
    assert var_5 is True


def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = {}
    var_3 = module_0.Number(minimum=var_0, maximum=var_1, **var_2)
    var_4 = 10
    var_5 = var_3.validate(var_4)
    assert var_5 == 10


def test_case_0():
    var_0 = 5.0
    var_1 = 15.0
    var_2 = {}
    var_3 = module_0.Number(exclusive_minimum=var_0, exclusive_maximum=var_1, **var_2)
    var_4 = 10.0
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 == 10.0)
    assert var_6 is True


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '3.14'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '2.5'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 2.5)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_condition_true_when_multiple_error_messages. Retrieved 1/12 statements.
# Partially parsed test_condition_true_when_single_message_code_not_type. Retrieved 1/12 statements.
# Partially parsed test_condition_true_when_single_type_message_with_index. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_combined_errors. Retrieved 11/12 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = ''
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'invalid_property'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'max_properties'


def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'


def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'key'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'null'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'abc': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'bcd'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'bcd': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'null'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = 'missing'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, required=var_5, **var_6)
    var_8 = 'key'
    var_9 = None
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_numeric_type_int_and_float_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_with_numeric_type_int_and_integer_value. Retrieved 1/3 statements.
# Partially parsed test_validate_with_numeric_type_int_and_string_integer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5.0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123

def test_case_0():
    var_0 = 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 3.14
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True

def test_case_0():
    var_0 = '42'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_single_item_validator. Retrieved 7/8 statements.
# Partially parsed test_validate_with_list_item_validators. Retrieved 8/10 statements.
# Partially parsed test_validate_with_list_item_validators_and_additional_items_false. Retrieved 12/15 statements.
# Partially parsed test_validate_with_list_item_validators_and_additional_items_field. Retrieved 11/14 statements.
# Partially parsed test_validate_item_validation_error. Retrieved 11/15 statements.
# Partially parsed test_validate_multiple_item_validation_errors. Retrieved 13/17 statements.
# Partially parsed test_validate_combined_errors. Retrieved 14/17 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'exact_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = {}
    var_3 = module_0.Array(var_0, **var_2)
    var_4 = 1
    var_5 = 3
    var_6 = [var_4, var_1, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == [2, 4, 6])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = 2
    var_4 = [var_0, var_2]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = [var_1, var_3]
    var_8 = var_6.validate(var_7)
    var_9 = bool(var_8 == [2, 4])
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = 2
    var_4 = [var_0, var_2]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].code
    assert var_15 == 'additional_items'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = 2
    var_4 = module_0.Field()
    var_5 = [var_0, var_2]
    var_6 = {}
    var_7 = module_0.Array(var_5, var_4, **var_6)
    var_8 = 3
    var_9 = 4
    var_10 = [var_1, var_3, var_8, var_9]
    var_11 = var_7.validate(var_10)
    var_12 = bool(var_11 == [2, 4, 2, 3])
    assert var_12 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_3]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'unique_items'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.base as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Invalid'
    var_3 = 'invalid'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = {}
    var_8 = module_0.Array(var_0, **var_7)
    var_9 = 1
    var_10 = [var_9]
    var_11 = var_8.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = len(e.messages())
    assert var_13 == 1
    var_14 = e.messages()[0].code
    assert var_14 == 'invalid'
    var_15 = e.messages()[0].index
    var_16 = bool(e.messages()[0].index == [0])
    assert var_16 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Invalid'
    var_3 = 'invalid'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = [var_0, var_0]
    var_8 = {}
    var_9 = module_0.Array(var_7, **var_8)
    var_10 = 1
    var_11 = 2
    var_12 = [var_10, var_11]
    var_13 = var_9.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    assert var_15 == 2
    var_16 = e.messages()[0].code
    assert var_16 == 'invalid'
    var_17 = e.messages()[0].index
    var_18 = bool(e.messages()[0].index == [0])
    assert var_18 is True
    var_19 = e.messages()[1].code
    assert var_19 == 'invalid'
    var_20 = e.messages()[1].index
    var_21 = bool(e.messages()[1].index == [1])
    assert var_21 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Invalid'
    var_3 = 'invalid'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = True
    var_8 = 2
    var_9 = {}
    var_10 = module_0.Array(var_0, min_items=var_8, unique_items=var_7, **var_9)
    var_11 = 1
    var_12 = [var_11]
    var_13 = var_10.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = [msg.code for msg in e.messages()]
    var_16 = 'min_items'
    var_17 = bool('min_items' in var_15)
    assert var_17 is True
    var_18 = 'invalid'
    var_19 = bool('invalid' in var_15)
    assert var_19 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 'two'
    var_4 = 'three'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = var_1.validate(var_7)
    var_9 = bool(var_8 == [1, 'two', {'three': 3}])
    assert var_9 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = var_2.validate(var_8)
    var_10 = bool(var_9 == [[1, 2], [3, 4]])
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.choices
    var_7 = bool(var_1.choices == [])
    assert var_7 is True
    var_8 = var_1.coerce_types
    assert var_8 is True


def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = True
    var_3 = 'a'
    var_4 = 'A'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 'B'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = False
    var_11 = 'title'
    var_12 = 'description'
    var_13 = 'allow_null'
    var_14 = 'read_only'
    var_15 = {var_11: var_0, var_12: var_1, var_13: var_2, var_14: var_2}
    var_16 = module_0.Choice(choices=var_9, coerce_types=var_10, **var_15)
    var_17 = var_16.title
    assert var_17 == 'Test Title'
    var_18 = var_16.description
    assert var_18 == 'Test Description'
    var_19 = var_16.allow_null
    assert var_19 is True
    var_20 = var_16.read_only
    assert var_20 is True
    var_21 = var_16.choices
    var_22 = bool(var_16.choices == [('a', 'A'), ('b', 'B')])
    assert var_22 is True
    var_23 = var_16.coerce_types
    assert var_23 is False


def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.choices
    var_6 = bool(var_4.choices == [('option1', 'option1'), ('option2', 'option2')])
    assert var_6 is True


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('key1', 'value1'), ('key2', 'value2')])
    assert var_10 is True


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Choice(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Choice(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = 'default'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_integer_type_error_for_float. Retrieved 1/4 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 12
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.2
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_decimal. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_integer_decimal. Retrieved 2/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '123.456'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '789'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_multiple_errors. Retrieved 10/11 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'longkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'invalid_property'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'max_properties'


def test_case_0():
    var_0 = 'key1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'


def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'key'
    var_5 = {var_4: var_3}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 'default_value'})
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'key'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '^a.*'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'abc'
    var_7 = 'ade'
    var_8 = 123
    var_9 = 456
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'abc': 123, 'ade': 456})
    assert var_12 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 123
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 123})
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'


def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = 2
    var_3 = {}
    var_4 = module_0.String(max_length=var_2, **var_3)
    var_5 = False
    var_6 = {}
    var_7 = module_0.Object(additional_properties=var_5, property_names=var_4, required=var_1, **var_6)
    var_8 = 'longkey'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'invalid_property'
    var_14 = 'required'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_property_names_invalid. Retrieved 9/10 statements.
# Partially parsed test_validate_pattern_properties. Retrieved 9/14 statements.
# Partially parsed test_validate_additional_properties_field. Retrieved 4/8 statements.
# Partially parsed test_validate_combined_errors. Retrieved 12/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = 'valid'
    var_5 = None
    var_6 = 'ok'
    var_7 = 'invalid'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 3
    var_8 = {var_3: var_6, var_4: var_0, var_5: var_7}
    var_9 = var_2.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'max_properties'


def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'other_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'required'


def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = None
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'null'

def test_case_0():
    var_0 = 'integer'
    var_1 = '^a.*'
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 'apricot'
    var_5 = 5
    var_6 = 'not int'
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = bool(False)
    assert var_9 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'allowed'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'not allowed'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'

def test_case_0():
    var_0 = 'integer'
    var_1 = 'extra'
    var_2 = 'not an int'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = 'required'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_0, required=var_5, **var_6)
    var_8 = 'extra'
    var_9 = None
    var_10 = 'invalid'
    var_11 = {var_2: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'null'
    var_15 = 'required'
    var_16 = 'invalid_property'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_property_names_invalid. Retrieved 12/16 statements.
# Partially parsed test_validate_properties_valid. Retrieved 7/8 statements.
# Partially parsed test_validate_properties_invalid. Retrieved 15/17 statements.
# Partially parsed test_validate_pattern_properties_matching. Retrieved 8/9 statements.
# Partially parsed test_validate_pattern_properties_non_matching. Retrieved 8/9 statements.
# Partially parsed test_validate_additional_properties_field_valid. Retrieved 6/7 statements.
# Partially parsed test_validate_additional_properties_field_invalid. Retrieved 13/15 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Invalid'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = {}
    var_8 = module_0.Object(property_names=var_0, **var_7)
    var_9 = 'invalid_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].code
    assert var_15 == 'invalid_property'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'max_properties'


def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'


def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Error'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'key'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_0.Object(properties=var_9, **var_10)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = len(e.messages())
    assert var_17 == 1
    var_18 = e.messages()[0].code
    assert var_18 == 'custom'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'abc': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'bcd'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'bcd': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Error'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = {}
    var_9 = module_0.Object(additional_properties=var_0, **var_8)
    var_10 = 'extra'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    assert var_15 == 1
    var_16 = e.messages()[0].code
    assert var_16 == 'custom'


def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Object(min_properties=var_2, required=var_1, **var_3)
    var_5 = 'invalid'
    var_6 = 123
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = {msg.code for msg in e.messages()}
    var_11 = 'required'
    var_12 = bool('required' in var_10)
    assert var_12 is True
    var_13 = 'min_properties'
    var_14 = bool('min_properties' in var_10)
    assert var_14 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_native_type_for_format. Retrieved 2/5 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  hello  '


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(trim_whitespace=var_0, coerce_types=var_0, **var_2)
    var_4 = '  '
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abc'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'ab'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcde'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abcde'


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdef'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'


def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Must be a valid email.'


def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = 'test_default'
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 'test_default'


def test_case_0():
    var_0 = 'callable_result'
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'callable_result'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None


def test_case_0():
    var_0 = None
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Field(default=var_3)
    var_5 = var_4.get_default_value()
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_float_for_integer_field. Retrieved 1/4 statements.
# Partially parsed test_validate_coerce_to_int. Retrieved 1/3 statements.
# Partially parsed test_validate_coerce_to_float. Retrieved 1/3 statements.



def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.Number(coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    assert var_4 == 11


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 12
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.2
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.5)
    assert var_5 is True

def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = '3.14'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 3.14
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'a'


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 'c'
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, coerce_types=var_7, **var_9)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 is None


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = False
    var_9 = 'allow_null'
    var_10 = {var_9: var_7}
    var_11 = module_0.Choice(choices=var_6, coerce_types=var_8, **var_10)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True


def test_case_0():
    var_0 = 'key1'
    var_1 = 'Display 1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'Display 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_3)
    assert var_9 == 'key2'


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_1)
    assert var_5 == 'y'


def test_case_0():
    var_0 = True
    var_1 = 'Yes'
    var_2 = (var_0, var_1)
    var_3 = 'One'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 is True


def test_case_0():
    var_0 = False
    var_1 = 'No'
    var_2 = (var_0, var_1)
    var_3 = 'Zero'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 is False


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'List AB'
    var_4 = (var_2, var_3)
    var_5 = 'c'
    var_6 = 'C'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_0.Choice(choices=var_8, **var_9)
    var_11 = [var_0, var_1]
    var_12 = var_10.validate(var_11)
    var_13 = bool(var_12 == ['a', 'b'])
    assert var_13 is True


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Dict X'
    var_4 = (var_2, var_3)
    var_5 = 'y'
    var_6 = 'Y'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_0.Choice(choices=var_8, **var_9)
    var_11 = {var_0: var_1}
    var_12 = var_10.validate(var_11)
    var_13 = bool(var_12 == {'x': 1})
    assert var_13 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'List AB'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = 'a'
    var_9 = 'c'
    var_10 = [var_8, var_9]
    var_11 = var_7.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Dict X'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = 'x'
    var_9 = 2
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #9
#--------------------------





def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'exact_items'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'invalid'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'hello'])
    assert var_11 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 'invalid'
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 1
    var_9 = 'hello'
    var_10 = 'extra'
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].code
    assert var_15 == 'additional_items'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = 1
    var_8 = 'extra1'
    var_9 = 'extra2'
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == [1, 'extra1', 'extra2'])
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 'extra'
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = len(e.messages())
    assert var_13 == 1
    var_14 = e.messages()[0].code
    assert var_14 == 'type'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_3]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'unique_items'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = False
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [False, 0])
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = [var_0, var_0]
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == [True, 1])
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = True
    var_6 = {}
    var_7 = module_0.Array(var_4, unique_items=var_5, **var_6)
    var_8 = 'invalid'
    var_9 = 2
    var_10 = 'extra'
    var_11 = [var_8, var_9, var_10, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 3
    var_15 = [msg.code for msg in e.messages()]
    var_16 = 'type'
    var_17 = bool('type' in var_15)
    assert var_17 is True
    var_18 = 'additional_items'
    var_19 = bool('additional_items' in var_15)
    assert var_19 is True
    var_20 = 'unique_items'
    var_21 = bool('unique_items' in var_15)
    assert var_21 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_string_constructor_defaults. Retrieved 1/2 statements.
# Partially parsed test_string_constructor_with_allow_blank. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_with_allow_null. Retrieved 2/3 statements.
# Partially parsed test_string_constructor_with_default. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_with_callable_default. Retrieved 3/5 statements.
# Partially parsed test_string_constructor_with_allow_blank_and_default. Retrieved 3/5 statements.
# Partially parsed test_string_constructor_with_allow_null_and_default. Retrieved 3/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.allow_blank
    assert var_6 is False
    var_7 = var_1.trim_whitespace
    assert var_7 is True
    var_8 = var_1.max_length
    assert var_8 is None
    var_9 = var_1.min_length
    assert var_9 is None
    var_10 = var_1.pattern
    assert var_10 is None
    var_11 = var_1.pattern_regex
    assert var_11 is None
    var_12 = var_1.format
    assert var_12 is None
    var_13 = var_1.coerce_types
    assert var_13 is True


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = var_2.allow_blank
    assert var_3 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True


def test_case_0():
    var_0 = 'Name'
    var_1 = 'Full name'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Name'
    var_7 = var_5.description
    assert var_7 == 'Full name'


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = var_2.max_length
    assert var_3 == 10


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = var_2.min_length
    assert var_3 == 2


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = var_2.pattern
    assert var_3 == '^[a-z]+$'
    var_4 = var_2.pattern_regex
    var_5 = bool(var_2.pattern_regex is not None)
    assert var_5 is True

import re as module_0

import typesystem.fields as module_1


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '^[a-z]+$'
    var_5 = var_3.pattern_regex
    var_6 = bool(var_3.pattern_regex is var_1)
    assert var_6 is True

import typesystem.fields as module_0


def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = var_2.format
    assert var_3 == 'email'


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is False


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = var_2.trim_whitespace
    assert var_3 is False


def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)


def test_case_0():
    var_0 = 'callable'
    var_1 = lambda : var_0
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)


def test_case_0():
    var_0 = True
    var_1 = 'custom'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, **var_3)
    var_5 = var_4.allow_blank
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = 'not null'
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.allow_null
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = var_1.items
    assert var_2 is None
    var_3 = var_1.additional_items
    assert var_3 is False
    var_4 = var_1.min_items
    assert var_4 is None
    var_5 = var_1.max_items
    assert var_5 is None
    var_6 = var_1.unique_items
    assert var_6 is False
    var_7 = var_1.allow_null
    assert var_7 is False
    var_8 = var_1.read_only
    assert var_8 is False
    var_9 = var_1.title
    assert var_9 == ''
    var_10 = var_1.description
    assert var_10 == ''


def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    var_4 = bool(var_2.items == var_0)
    assert var_4 is True
    var_5 = var_2.additional_items
    assert var_5 is False
    var_6 = var_2.min_items
    assert var_6 is None
    var_7 = var_2.max_items
    assert var_7 is None
    var_8 = var_2.unique_items
    assert var_8 is False


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = var_4.items
    var_6 = bool(var_4.items == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.additional_items
    assert var_7 is False
    var_8 = var_4.min_items
    assert var_8 == 2
    var_9 = var_4.max_items
    assert var_9 == 2


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items == var_0)
    assert var_5 is True
    var_6 = var_3.additional_items
    var_7 = bool(var_3.additional_items == var_1)
    assert var_7 is True
    var_8 = var_3.min_items
    assert var_8 is None
    var_9 = var_3.max_items
    assert var_9 is None


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 3
    var_4 = var_2.max_items
    assert var_4 is None


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 is None
    var_4 = var_2.max_items
    assert var_4 == 5


def test_case_0():
    var_0 = 4
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 4
    var_4 = var_2.max_items
    assert var_4 == 4


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = var_2.unique_items
    assert var_3 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True


def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True


def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Array(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Title'
    var_7 = var_5.description
    assert var_7 == 'Test Description'


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.additional_items
    assert var_8 is False
    var_9 = var_5.min_items
    assert var_9 == 2
    var_10 = var_5.max_items
    assert var_10 == 2


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1]
    var_4 = {}
    var_5 = module_0.Array(var_3, var_2, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.additional_items
    var_9 = bool(var_5.additional_items == var_2)
    assert var_9 is True
    var_10 = var_5.min_items
    assert var_10 == 2
    var_11 = var_5.max_items
    assert var_11 is None


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_2, min_items=var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 == 5
    var_9 = var_5.max_items
    assert var_9 == 2


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_2, max_items=var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 == 2
    var_9 = var_5.max_items
    assert var_9 == 10


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 5
    var_5 = {}
    var_6 = module_0.Array(var_2, var_3, max_items=var_4, **var_5)
    var_7 = var_6.items
    var_8 = bool(var_6.items == [var_0, var_1])
    assert var_8 is True
    var_9 = var_6.additional_items
    assert var_9 is False
    var_10 = var_6.min_items
    assert var_10 == 2
    var_11 = var_6.max_items
    assert var_11 == 2



# Parsed testcases at query #12
#--------------------------





def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #14
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = False
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'false'
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'on'
    var_3 = var_1.validate(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'off'
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = '1'
    var_3 = var_1.validate(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = '0'
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 1
    var_3 = var_1.validate(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 0
    var_3 = var_1.validate(var_2)
    assert var_3 is False


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 1
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is False



# Parsed testcases at query #16
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.Number(coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = '123'
    var_5 = var_3.validate(var_4)
    assert var_5 == 123


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = 0
    var_5 = var_3.validate(var_4)
    assert var_5 == 0


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = -3.14
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 == -3.14)
    assert var_6 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = '99.9'
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 == 99.9)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 == ''


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 == ''


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 == ''


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 == ''


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, trim_whitespace=var_1, coerce_types=var_1, **var_3)
    var_5 = '   '
    var_6 = var_4.validate(var_5)
    assert var_6 is None


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = 'hello'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'hello'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_with_single_item_field. Retrieved 7/8 statements.
# Partially parsed test_serialize_with_list_of_item_fields. Retrieved 8/10 statements.
# Partially parsed test_serialize_with_list_of_item_fields_and_shorter_obj. Retrieved 7/9 statements.
# Partially parsed test_serialize_with_list_of_item_fields_and_longer_obj. Retrieved 9/11 statements.



def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.serialize(var_6)
    var_8 = bool(var_7 == ['serialized_1', 'serialized_2', 'serialized_3'])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = 10
    var_6 = 20
    var_7 = [var_5, var_6]
    var_8 = var_4.serialize(var_7)
    var_9 = bool(var_8 == ['a_10', 'b_20'])
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = 10
    var_6 = [var_5]
    var_7 = var_4.serialize(var_6)
    var_8 = bool(var_7 == ['a_10'])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = 10
    var_6 = 20
    var_7 = 30
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.serialize(var_8)
    var_10 = bool(var_9 == ['a_10', 'b_20', 30])
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null. Retrieved 2/11 statements.
# Partially parsed test_validate_raises_null_error_when_value_is_none_and_not_allow_null. Retrieved 1/10 statements.
# Partially parsed test_validate_returns_validated_value_from_first_matching_child. Retrieved 4/13 statements.
# Partially parsed test_validate_raises_single_candidate_error_when_one_child_has_non_type_error. Retrieved 4/24 statements.
# Partially parsed test_validate_raises_union_error_when_no_child_matches_and_no_single_candidate_error. Retrieved 3/23 statements.
# Partially parsed test_validate_raises_union_error_when_multiple_candidate_errors_exist. Retrieved 4/24 statements.
# Partially parsed test_validate_raises_error_with_index_from_child. Retrieved 5/19 statements.
# Failed to parse test_validate_allow_null_set_true_if_any_child_allows_null.


def test_case_0():
    var_0 = None
    var_1 = (var_0, var_0)

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True
    var_2 = e.messages()[0].code
    assert var_2 == 'null'

def test_case_0():
    var_0 = 'validated'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'test'

def test_case_0():
    var_0 = 'custom'
    var_1 = None
    var_2 = 'type'
    var_3 = 'test'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'type'
    var_1 = None
    var_2 = 'test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = e.messages()[0].code
    assert var_4 == 'union'

def test_case_0():
    var_0 = 'custom1'
    var_1 = None
    var_2 = 'custom2'
    var_3 = 'test'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = e.messages()[0].code
    assert var_5 == 'union'

def test_case_0():
    var_0 = 'type'
    var_1 = 0
    var_2 = [var_1]
    var_3 = None
    var_4 = 'test'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = e.messages()[0].code
    assert var_6 == 'union'



# Parsed testcases at query #22
#--------------------------





def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'longkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'invalid_property'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'


def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'


def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'max_properties'


def test_case_0():
    var_0 = 'key1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'


def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'key'
    var_5 = {var_4: var_3}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 'default_value'})
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'key'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = 'key'
    var_8 = 'toolong'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'max_length'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = '^a'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_0.Object(pattern_properties=var_4, **var_5)
    var_7 = 'a1'
    var_8 = 'b1'
    var_9 = 'val'
    var_10 = 'ignored'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(var_12 == {'a1': 'val'})
    assert var_13 is True


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = '^a'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_0.Object(pattern_properties=var_4, **var_5)
    var_7 = 'a1'
    var_8 = 'toolong'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'max_length'


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, **var_3)
    var_5 = 'extra'
    var_6 = 'val'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'extra': 'val'})
    assert var_9 is True


def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, **var_3)
    var_5 = 'extra'
    var_6 = 'toolong'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'max_length'


def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, required=var_1, **var_3)
    var_5 = 'extra'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = {msg.code for msg in e.messages()}
    var_11 = bool(var_10 == {'required', 'invalid_property'})
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------





def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = True
    var_9 = 'allow_null'
    var_10 = {var_9: var_7}
    var_11 = module_0.Choice(choices=var_6, coerce_types=var_8, **var_10)
    var_12 = 'key3'
    var_13 = var_11.validate(var_12)
    assert var_13 is None



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------





def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = True
    var_9 = 'allow_null'
    var_10 = {var_9: var_7}
    var_11 = module_0.Choice(choices=var_6, coerce_types=var_8, **var_10)
    var_12 = 'invalid'
    var_13 = var_11.validate(var_12)
    assert var_13 is None



