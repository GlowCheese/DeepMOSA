####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://google.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://google.com'
    var_3 = 'http://localhost:8080/path?query=1'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'http://localhost:8080/path?query=1'
    var_5 = 'ftp://files.server.org'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'ftp://files.server.org'
    var_7 = 'google.com'
    var_8 = var_0.validate(var_7)
    var_9 = 'https://'
    var_10 = var_0.validate(var_9)
    var_11 = ''
    var_12 = var_0.validate(var_11)
    var_13 = 'https:///path/only'
    var_14 = var_0.validate(var_13)



# Parsed testcases at query #2
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '123456@example.org'
    var_4 = '"quoted-string"@example.com'
    var_5 = 'simple@sub.domain.com'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = 'email@example..com'
    var_17 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = str(var_1)



# Parsed testcases at query #3
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '550e8400-e29b-41d4-a716-446655440000'
    var_2 = module_1.UUID(var_1)
    var_3 = 'not-a-uuid'
    var_4 = '550e8400-e29b-99d4-a716-446655440000'
    var_5 = var_0.validate(var_1)
    var_6 = var_0.validate(var_3)
    var_7 = var_0.validate(var_4)



# Parsed testcases at query #4
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_3 = 123
    var_4 = var_0.is_native_type(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = var_0.is_native_type(var_5)
    assert var_6 is False



# Parsed testcases at query #5
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = None
    var_12 = '2023-10-27T15:30:45.123456+02:00'
    var_13 = var_0.validate(var_12)
    var_14 = 2
    var_15 = module_1.timedelta()
    var_16 = 123456
    var_17 = '2023-10-27T15:30:45-0500'
    var_18 = var_0.validate(var_17)
    var_19 = -5
    var_20 = module_1.timedelta()
    var_21 = '27-10-2023 15:30:45'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-30T15:30:45'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-10-27T25:30:45'
    var_26 = var_0.validate(var_25)
    var_27 = '2023-10-27T15:30:45.1234567'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #6
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-10-27T15:30:45.123456+02:00'
    var_12 = var_0.validate(var_11)
    var_13 = 2
    var_14 = module_1.timedelta()
    var_15 = '2023-10-27T15:30:45.9+05:30'
    var_16 = var_0.validate(var_15)
    var_17 = 5
    var_18 = module_1.timedelta()
    var_19 = '27-10-2023 15:30:45'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01T15:30:45'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-10-32T15:30:45'
    var_24 = var_0.validate(var_23)
    var_25 = ''
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #7
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 14
    var_4 = 30
    var_5 = 5
    var_6 = 12
    var_7 = 0
    var_8 = 123456
    var_9 = 9
    var_10 = '14:30:05'
    var_11 = var_0.serialize(var_10)



# Parsed testcases at query #8
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '1234567890@example.com'
    var_4 = '"quoted-string"@example.com'
    var_5 = 'email@subdomain.example.com'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]



# Parsed testcases at query #9
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 10
    var_5 = 25
    var_6 = 1999
    var_7 = 1
    var_8 = '2023-10-25'
    var_9 = var_0.serialize(var_8)



# Parsed testcases at query #10
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 'https://example.com/path?query=1'
    var_4 = var_0.serialize(var_3)
    var_5 = 'http://localhost:8080'
    var_6 = var_0.serialize(var_5)
    var_7 = ''
    var_8 = var_0.serialize(var_7)
    assert var_8 == ''



# Parsed testcases at query #11
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:00'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 0
    var_5 = '23:59:59'
    var_6 = var_0.validate(var_5)
    var_7 = 23
    var_8 = 59
    var_9 = '08:30:15.123456'
    var_10 = var_0.validate(var_9)
    var_11 = 8
    var_12 = 30
    var_13 = 15
    var_14 = 123456
    var_15 = '08:30:15.12'
    var_16 = var_0.validate(var_15)
    var_17 = 120000
    var_18 = '12-00'
    var_19 = var_0.validate(var_18)
    var_20 = 'abc'
    var_21 = var_0.validate(var_20)
    var_22 = '25:00'
    var_23 = var_0.validate(var_22)
    var_24 = '12:61:00'
    var_25 = var_0.validate(var_24)



# Parsed testcases at query #12
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '2024-02-29'
    var_6 = var_0.validate(var_5)
    var_7 = 2024
    var_8 = 2
    var_9 = 29
    var_10 = '01-01-2023'
    var_11 = var_0.validate(var_10)
    var_12 = '2023/01/01'
    var_13 = var_0.validate(var_12)
    var_14 = 'not-a-date'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-13-01'
    var_17 = var_0.validate(var_16)
    var_18 = '2023-02-29'
    var_19 = var_0.validate(var_18)
    var_20 = '2023-04-31'
    var_21 = var_0.validate(var_20)



# Parsed testcases at query #13
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:00'
    var_2 = '09:30:45'
    var_3 = '23:59:59.123456'
    var_4 = '00:00:00.000000'
    var_5 = '8:5:1'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 0
    var_8 = '.'
    var_9 = var_3.split(var_8)[var_7]
    var_10 = time_str.split(var_8)[var_7]
    var_11 = '12'
    var_12 = '12:61'
    var_13 = 'abc'
    var_14 = '12:00:00:00'
    var_15 = '25:00'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = '12:00:00.1'
    var_18 = var_0.validate(var_17)
    var_19 = '00:00:00'
    var_20 = var_0.validate(var_19)
    var_21 = 0
    var_22 = '23:59:59'
    var_23 = var_0.validate(var_22)
    var_24 = 23
    var_25 = 59



# Parsed testcases at query #14
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '550e8400-e29b-41d4-a716-446655440000'
    var_2 = module_1.UUID(var_1)
    var_3 = None
    var_4 = var_0.serialize(var_3)
    assert var_4 is None
    var_5 = module_1.uuid1()
    var_6 = var_0.serialize(var_5)
    var_7 = str(var_5)
    var_8 = 'not-a-uuid-object'
    var_9 = var_0.serialize(var_8)



# Parsed testcases at query #15
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'first.last@sub.domain.org'
    var_4 = '1234567890@example.com'
    var_5 = 'email@example-one.com'
    var_6 = '_______@example.com'
    var_7 = '"quoted-string"@example.com'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'plainaddress'
    var_10 = '#@%^%#$@#$@#.com'
    var_11 = '@example.com'
    var_12 = 'Joe Smith <email@example.com>'
    var_13 = 'email.example.com'
    var_14 = 'email@example@example.com'
    var_15 = '.email@example.com'
    var_16 = 'email.@example.com'
    var_17 = 'email..email@example.com'
    var_18 = 'email@example..com'
    var_19 = [var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = str(var_1)
    var_21 = None
    var_22 = var_0.validate(var_21)



# Parsed testcases at query #16
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test@example.com'
    var_3 = 'user.name+tag@domain.co.uk'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'user.name+tag@domain.co.uk'
    var_5 = '"quoted-string"@example.com'
    var_6 = var_0.validate(var_5)
    assert var_6 == '"quoted-string"@example.com'
    var_7 = '1234567890@example.com'
    var_8 = var_0.validate(var_7)
    assert var_8 == '1234567890@example.com'
    var_9 = 'plainaddress'
    var_10 = '#@%^%#%@#@#.com'
    var_11 = '@example.com'
    var_12 = 'Joe Smith <email@example.com>'
    var_13 = 'email.example.com'
    var_14 = 'email@example@example.com'
    var_15 = '.email@example.com'
    var_16 = 'email.@example.com'
    var_17 = 'email..email@example.com'
    var_18 = 'email@example..com'
    var_19 = [var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = str(var_1)
    assert var_20 == 'Must be a valid email format.'



# Parsed testcases at query #17
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '1234567890@example.com'
    var_4 = 'email@subdomain.example.com'
    var_5 = '"quoted-local-part"@example.com'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = str(var_1)
    assert var_17 == 'Must be a valid email format.'
    var_18 = None
    var_19 = var_0.validate(var_18)



# Parsed testcases at query #18
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = '::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '127.0.0.abc'
    var_22 = var_0.validate(var_21)
    var_23 = ''
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #19
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 14
    var_4 = 30
    var_5 = 45
    var_6 = 12
    var_7 = 0
    var_8 = 123456
    var_9 = 'not a time object'
    var_10 = var_0.serialize(var_9)



# Parsed testcases at query #20
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 14
    var_4 = 30
    var_5 = 5
    var_6 = 12
    var_7 = 0
    var_8 = 123456
    var_9 = 9
    var_10 = '12:00:00'
    var_11 = var_0.serialize(var_10)
    var_12 = 12345
    var_13 = var_0.serialize(var_12)



# Parsed testcases at query #21
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2024-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2024
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '2023/01/01'
    var_18 = var_0.validate(var_17)
    var_19 = 'not-a-date'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-29'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-04-31'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #22
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:00'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 0
    var_5 = '08:30:45'
    var_6 = var_0.validate(var_5)
    var_7 = 8
    var_8 = 30
    var_9 = 45



# Parsed testcases at query #23
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2024-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2024
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '202            '
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-30'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-01-32'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #24
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = '::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '123.456.789.0'
    var_22 = var_0.validate(var_21)
    var_23 = '999.999.999.999'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #25
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '0.0.0.0'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = '::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '127.0.0.256'
    var_22 = var_0.validate(var_21)
    var_23 = '999.999.999.999'
    var_24 = var_0.validate(var_23)
    var_25 = ''
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #26
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '0.0.0.0'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = 'not-an-ip'
    var_17 = var_0.validate(var_16)
    var_18 = '999.999.999.999'
    var_19 = var_0.validate(var_18)
    var_20 = '127.0.0.1.extra'
    var_21 = var_0.validate(var_20)



# Parsed testcases at query #27
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-10-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 25
    var_6 = '2000-01-01'
    var_7 = var_0.validate(var_6)
    var_8 = 2000
    var_9 = 1
    var_10 = '25-10-2023'
    var_11 = var_0.validate(var_10)
    var_12 = '2023/10/25'
    var_13 = var_0.validate(var_12)
    var_14 = 'not-a-date'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-02-30'
    var_17 = var_0.validate(var_16)
    var_18 = '2023-13-01'
    var_19 = var_0.validate(var_18)
    var_20 = '2023-10-32'
    var_21 = var_0.validate(var_20)



# Parsed testcases at query #28
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2024-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2024
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '202le-01-01'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-30'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-01-32'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #29
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27 15:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27T15:30:45Z'
    var_10 = var_0.validate(var_9)
    var_11 = 2000
    var_12 = 1
    var_13 = 0
    var_14 = '2023-10-27 15:30:45+05:30'
    var_15 = var_0.validate(var_14)
    var_16 = 5
    var_17 = module_1.timedelta()
    var_18 = '2023-10-27 15:30:45-08:00'
    var_19 = var_0.validate(var_18)
    var_20 = -8
    var_21 = module_1.timedelta()
    var_22 = '2023-10-27 15:30:45.123456'
    var_23 = var_0.validate(var_22)
    var_24 = '2023-10'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-02-30 15:30:45'
    var_27 = var_0.validate(var_26)
    var_28 = '2023-10-27 25:00:00'
    var_29 = var_0.validate(var_28)



# Parsed testcases at query #30
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-10-25'
    var_2 = 2023
    var_3 = 10
    var_4 = 25
    var_5 = var_0.validate(var_1)
    var_6 = '2023-1-5'
    var_7 = 1
    var_8 = 5
    var_9 = var_0.validate(var_6)
    var_10 = '25-10-2023'
    var_11 = var_0.validate(var_10)
    var_12 = '2023-02-30'
    var_13 = var_0.validate(var_12)
    var_14 = None
    var_15 = var_0.validate(var_14)
    var_16 = 12345678
    var_17 = var_0.validate(var_16)



# Parsed testcases at query #31
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = '2001:db8:85a3::8a2e:370:7334'
    var_16 = module_1.IPv6Address(var_15)
    var_17 = '::1'
    var_18 = var_0.validate(var_17)
    var_19 = module_1.IPv6Address(var_17)
    var_20 = 'not-an-ip'
    var_21 = var_0.validate(var_20)
    var_22 = '999.999.999.999'
    var_23 = var_0.validate(var_22)



# Parsed testcases at query #32
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = '2023-01-01 12:00:00'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-01-01T12:00:00Z'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-01-01T12:00:00+05:30'
    var_12 = var_0.validate(var_11)
    var_13 = 5
    var_14 = 30
    var_15 = module_1.timedelta()
    var_16 = '2023-01-01T12:00:00-08'
    var_17 = var_0.validate(var_16)
    var_18 = -8
    var_19 = module_1.timedelta()
    var_20 = '2023-01-01T12:00:00.123456'
    var_21 = var_0.validate(var_20)
    var_22 = '01-01-2023 12:00:00'
    var_23 = var_0.validate(var_22)
    var_24 = '2023-02-30T12:00:00'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-01-01T25:00:00'
    var_27 = var_0.validate(var_26)
    var_28 = '01-01 12:00:00'
    var_29 = var_0.validate(var_28)



# Parsed testcases at query #33
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2000-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2000
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '2023/01/01'
    var_18 = var_0.validate(var_17)
    var_19 = 'not-a-date'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-29'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-04-31'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #34
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = '2023-05-20 15:30:45'
    var_8 = var_0.validate(var_7)
    var_9 = 5
    var_10 = 20
    var_11 = 15
    var_12 = 30
    var_13 = 45
    var_14 = '2023-01-01T12:00:00Z'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-01-01T12:00:00+05:30'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.timedelta()
    var_19 = '2023-01-01T12:00:00-07:00'
    var_20 = var_0.validate(var_19)
    var_21 = -7
    var_22 = module_1.timedelta()
    var_23 = '2023-01-01T12:00:00.123456'
    var_24 = var_0.validate(var_23)
    var_25 = 123456
    var_26 = '2023-01-01T12:00:00.123'
    var_27 = var_0.validate(var_26)
    var_28 = 123000
    var_29 = '01-01-2023 12:00:00'
    var_30 = var_0.validate(var_29)
    var_31 = 'not-a-date'
    var_32 = var_0.validate(var_31)
    var_33 = '2023-02-30T12:00:00'
    var_34 = var_0.validate(var_33)
    var_35 = '2023-13-01T12:00:00'
    var_36 = var_0.validate(var_35)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2024-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2024
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '202    '
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-29'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-01-32'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #2
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 0
    var_9 = 2
    var_10 = module_1.timedelta()
    var_11 = -5
    var_12 = -30
    var_13 = module_1.timedelta()
    var_14 = 1
    var_15 = 12
    var_16 = 123456
    var_17 = 20
    var_18 = 9
    var_19 = 15
    var_20 = '2023-10-05T14:30:00Z'
    var_21 = var_0.serialize(var_20)



# Parsed testcases at query #3
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.0.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '0.0.0.0'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = 'not-an-ip'
    var_17 = var_0.validate(var_16)
    var_18 = '127.0.0.256'
    var_19 = var_0.validate(var_18)
    var_20 = '1.2.3'
    var_21 = var_0.validate(var_20)
    var_22 = '999.999.999.999'
    var_23 = var_0.validate(var_22)



# Parsed testcases at query #4
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.0.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv6Address(var_10)
    var_13 = 'not-an-ip'
    var_14 = var_0.validate(var_13)
    var_15 = '256.256.256.256'
    var_16 = var_0.validate(var_15)
    var_17 = ''
    var_18 = var_0.validate(var_17)



# Parsed testcases at query #5
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '1234567890@example.com'
    var_4 = '"quoted-string"@example.com'
    var_5 = 'email@subdomain.example.org'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = str(var_1)



# Parsed testcases at query #6
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-01-01 12:00:00'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-10-27T15:30:45.123456'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-10-27T15:30:45+05:30'
    var_8 = var_0.validate(var_7)
    var_9 = None
    var_10 = 5
    var_11 = 30
    var_12 = module_1.timedelta()
    var_13 = '2023-10-27T15:30:45-08:00'
    var_14 = 'append_tzinfo'
    var_15 = hasattr(var_0, var_14)
    var_16 = var_0.validate(var_13)
    var_17 = -8
    var_18 = module_1.timedelta()
    var_19 = '27-10-2023 15:30:45'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-02-30T15:30:45'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-10-27T25:00:00'
    var_24 = var_0.validate(var_23)
    var_25 = 'not-a-date'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #7
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-8234-567812345678'
    var_2 = module_1.UUID(var_1)
    var_3 = var_0.serialize(var_2)
    assert var_3 == '12345678-1234-1234-8234-567812345678'
    var_4 = None
    var_5 = var_0.serialize(var_4)
    assert var_5 is None
    var_6 = module_1.uuid4()
    var_7 = var_0.serialize(var_6)
    var_8 = str(var_6)
    var_9 = 'not-a-uuid-object'
    var_10 = var_0.serialize(var_9)



# Parsed testcases at query #8
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 12
    var_4 = 30
    var_5 = 14
    var_6 = 15
    var_7 = 123456
    var_8 = 0
    var_9 = 'not a time object'
    var_10 = var_0.serialize(var_9)



# Parsed testcases at query #9
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 2
    var_10 = module_1.timedelta()
    var_11 = -5
    var_12 = -30
    var_13 = module_1.timedelta()
    var_14 = 123456
    var_15 = 'not a datetime object'
    var_16 = var_0.serialize(var_15)



# Parsed testcases at query #10
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-10-27T15:30:45Z'
    var_12 = var_0.validate(var_11)
    var_13 = 5
    var_14 = module_1.timedelta()
    var_15 = '2023-10-27T15:30:45+05:30'
    var_16 = var_0.validate(var_15)
    var_17 = -8
    var_18 = 0
    var_19 = module_1.timedelta()
    var_20 = '2023-10-27T15:30:45-08:00'
    var_21 = var_0.validate(var_20)
    var_22 = 123000
    var_23 = '2023-10-27T15:30:45.123'
    var_24 = var_0.validate(var_23)
    var_25 = '27-10-2023 15:30:45'
    var_26 = var_0.validate(var_25)
    var_27 = 'not-a-date'
    var_28 = var_0.validate(var_27)
    var_29 = '2023-02-30T15:30:45'
    var_30 = var_0.validate(var_29)
    var_31 = '2023-10-27T25:30:45'
    var_32 = var_0.validate(var_31)



# Parsed testcases at query #11
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://example.com'
    var_3 = 'http://localhost:8080/path?query=1'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'http://localhost:8080/path?query=1'
    var_5 = 'ftp://files.server.net'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'ftp://files.server.net'
    var_7 = 'example.com'
    var_8 = var_0.validate(var_7)
    var_9 = 'https://'
    var_10 = var_0.validate(var_9)
    var_11 = '/only/a/path'
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #12
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = 'not-an-ip'
    var_17 = var_0.validate(var_16)
    var_18 = '127.0.0.256'
    var_19 = var_0.validate(var_18)
    var_20 = '999.999.999.999'
    var_21 = var_0.validate(var_20)
    var_22 = ''
    var_23 = var_0.validate(var_22)



# Parsed testcases at query #13
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = 5
    var_10 = module_1.timedelta()
    var_11 = -8
    var_12 = 0
    var_13 = module_1.timedelta()
    var_14 = 123456
    var_15 = 'not a datetime object'
    var_16 = var_0.serialize(var_15)



# Parsed testcases at query #14
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '08:05:45'
    var_6 = var_0.validate(var_5)
    var_7 = 8
    var_8 = 5
    var_9 = 45
    var_10 = '23:59:59.999999'
    var_11 = var_0.validate(var_10)
    var_12 = 23
    var_13 = 59
    var_14 = 999999
    var_15 = '01:02:03.456'
    var_16 = var_0.validate(var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = 456000
    var_21 = '12-30-00'
    var_22 = var_0.validate(var_21)
    var_23 = '25:00:00'
    var_24 = var_0.validate(var_23)
    var_25 = '12:61:00'
    var_26 = var_0.validate(var_25)
    var_27 = None
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #15
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 14
    var_4 = 30
    var_5 = 5
    var_6 = 12
    var_7 = 0
    var_8 = 123456
    var_9 = 9
    var_10 = 15
    var_11 = 'not a time object'
    var_12 = var_0.serialize(var_11)



# Parsed testcases at query #16
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = '::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '127.0.0.256'
    var_22 = var_0.validate(var_21)
    var_23 = 'abc.def.ghi.jkl'
    var_24 = var_0.validate(var_23)
    var_25 = '999.999.999.999'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #17
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '09:05'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 5
    var_9 = '12:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = 45
    var_12 = '00:00:00'
    var_13 = var_0.validate(var_12)
    var_14 = 0
    var_15 = '12:30:45.123456'
    var_16 = var_0.validate(var_15)
    var_17 = 123456
    var_18 = '12:30:45.1'
    var_19 = var_0.validate(var_18)
    var_20 = 100000
    var_21 = '12:30:45.999'
    var_22 = var_0.validate(var_21)
    var_23 = 999000
    var_24 = '12-30-45'
    var_25 = var_0.validate(var_24)
    var_26 = 'abc'
    var_27 = var_0.validate(var_26)
    var_28 = '25:00:00'
    var_29 = var_0.validate(var_28)
    var_30 = '12:61:00'
    var_31 = var_0.validate(var_30)
    var_32 = ''
    var_33 = var_0.validate(var_32)



# Parsed testcases at query #18
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45+02:00'
    var_10 = var_0.validate(var_9)
    var_11 = 2
    var_12 = module_1.timedelta()
    var_13 = '2023-10-27 15:30:45-05:00'
    var_14 = var_0.validate(var_13)
    var_15 = -5
    var_16 = module_1.timedelta()
    var_17 = '2023-10-27T15:30:45.123456'
    var_18 = var_0.validate(var_17)
    var_19 = '27-10-2023 15:30:45'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-02-30T15:30:45'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-10-27T25:00:00'
    var_24 = var_0.validate(var_23)
    var_25 = ''
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #19
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 14
    var_4 = 30
    var_5 = 5
    var_6 = 123456
    var_7 = module_1.time()
    var_8 = var_0.serialize(var_7)
    assert var_8 == '14:30:05.123456'
    var_9 = 9
    var_10 = 0
    var_11 = module_1.time()
    var_12 = var_0.serialize(var_11)
    assert var_12 == '09:00:00'
    var_13 = '14:30:05'
    var_14 = var_0.serialize(var_13)



# Parsed testcases at query #20
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-10-27 15:30:45+02:00'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = 2
    var_7 = module_1.timedelta()
    var_8 = '2023-10-27 15:30:45-05:00'
    var_9 = var_0.validate(var_8)
    var_10 = -5
    var_11 = module_1.timedelta()
    var_12 = '2023-10-27T15:30:45.123456'
    var_13 = var_0.validate(var_12)
    var_14 = '27-10-2023 15:30:45'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-02-30T15:30:45'
    var_17 = var_0.validate(var_16)
    var_18 = '2023-10-27T25:00:00'
    var_19 = var_0.validate(var_18)
    var_20 = 'not-a-date'
    var_21 = var_0.validate(var_20)



# Parsed testcases at query #21
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-10-27 15:30:45+02:00'
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = 2
    var_7 = module_1.timedelta()
    var_8 = '202lag-01-01T00:00:00-05:00'
    var_9 = '2023-01-01T00:00:00-05:00'
    var_10 = var_0.validate(var_9)
    var_11 = -5
    var_12 = module_1.timedelta()
    var_13 = '2023-10-27T15:30:45.123456'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-10-26 12:00:00'
    var_16 = var_0.validate(var_15)
    var_17 = '27/10/2023 15:30'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-30T15:30:00'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-10-27T25:00:00'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-10'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #22
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '0.0.0.0'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = 'not-an-ip'
    var_17 = var_0.validate(var_16)
    var_18 = '127.0.0.256'
    var_19 = var_0.validate(var_18)
    var_20 = ''
    var_21 = var_0.validate(var_20)
    var_22 = '192.168.1.a'
    var_23 = var_0.validate(var_22)



# Parsed testcases at query #23
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '0.0.0.0'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = '::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '127.0.0.256'
    var_22 = var_0.validate(var_21)
    var_23 = '1234:5678:90ab:cdef:ghij:klmn:opqr:stuv'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #24
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-10-27T15:30:45+02:00'
    var_12 = var_0.validate(var_11)
    var_13 = 2
    var_14 = module_1.timedelta()
    var_15 = '2023-10-27T15:30:45-05:00'
    var_16 = var_0.validate(var_15)
    var_17 = -5
    var_18 = module_1.timedelta()
    var_19 = '2023-10-27T15:30:45.123456Z'
    var_20 = var_0.validate(var_19)
    var_21 = 123456
    var_22 = '2023-10-27T15:30:45.12Z'
    var_23 = var_0.validate(var_22)
    var_24 = 120000
    var_25 = None
    var_26 = '27-10-2023 15:30:45'
    var_27 = var_0.validate(var_26)
    var_28 = '2023/10/27 15:30:45'
    var_29 = var_0.validate(var_28)
    var_30 = 'not-a-date'
    var_31 = var_0.validate(var_30)
    var_32 = '2023-13-01T15:30:45'
    var_33 = var_0.validate(var_32)
    var_34 = '2023-02-30T15:30:45'
    var_35 = var_0.validate(var_34)
    var_36 = '2023-10-27T25:00:00'
    var_37 = var_0.validate(var_36)



# Parsed testcases at query #25
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27 14:30'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 14
    var_7 = 30
    var_8 = '2023-10-27 14:30:05.123'
    var_9 = var_0.validate(var_8)
    var_10 = 5
    var_11 = 123000
    var_12 = '2023-10-27T14:30:00Z'
    var_13 = var_0.validate(var_12)
    var_14 = module_1.timedelta()
    var_15 = '2023-10-27 14:30:00+05:30'
    var_16 = var_0.validate(var_15)
    var_17 = -8
    var_18 = 0
    var_19 = module_1.timedelta()
    var_20 = '2023-10-27T14:30:00-08:00'
    var_21 = var_0.validate(var_20)
    var_22 = '27-10-2023 14:30'
    var_23 = var_0.validate(var_22)
    var_24 = 'not-a-date'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-02-30 14:30'
    var_27 = var_0.validate(var_26)
    var_28 = '2023-13-01 14:30'
    var_29 = var_0.validate(var_28)
    var_30 = '2023-10-27 25:00:00'
    var_31 = var_0.validate(var_30)



# Parsed testcases at query #26
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.0.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '8.8.8.8'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv6Address(var_10)
    var_13 = 'not-an-ip'
    var_14 = var_0.validate(var_13)
    var_15 = '127.0.0.256'
    var_16 = var_0.validate(var_15)
    var_17 = '192.168.1'
    var_18 = var_0.validate(var_17)
    var_19 = ''
    var_20 = var_0.validate(var_19)



# Parsed testcases at query #27
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-10-25'
    var_2 = 2023
    var_3 = 10
    var_4 = 25
    var_5 = var_0.validate(var_1)
    var_6 = '2023-1-5'
    var_7 = 1
    var_8 = 5
    var_9 = var_0.validate(var_6)
    var_10 = '25-10-2023'
    var_11 = var_0.validate(var_10)
    var_12 = '2023-02-30'
    var_13 = var_0.validate(var_12)
    var_14 = 'YYYY-MM-DD'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-10-25 extra'
    var_17 = var_0.validate(var_16)



# Parsed testcases at query #28
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2024-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2024
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '202le-01-01'
    var_18 = var_0.validate(var_17)
    var_19 = '2023/01/01'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-29'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-04-31'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #29
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2000-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2000
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '2023/01/01'
    var_18 = var_0.validate(var_17)
    var_19 = 'not-a-date'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-02-29'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-13-01'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-01-32'
    var_26 = var_0.validate(var_25)



