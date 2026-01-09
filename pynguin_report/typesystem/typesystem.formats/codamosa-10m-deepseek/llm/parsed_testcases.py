####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0


def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2021
    var_4 = 1
    var_5 = '2021-01-01'
    var_6 = var_0.serialize(var_5)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import datetime as module_1


def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2022-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = '2022-01-01T12:00:00.123456'
    var_4 = var_0.validate(var_3)
    var_5 = '2022-01-01T12:00:00+05:30'
    var_6 = var_0.validate(var_5)
    var_7 = 5
    var_8 = 30
    var_9 = module_1.timedelta()
    var_10 = '2022-01-01T25:00:00'
    var_11 = var_0.validate(var_10)
    var_12 = '2022-01-01 12:00:00'
    var_13 = var_0.validate(var_12)
    var_14 = '2022-01-01T12:00:00Z'
    var_15 = var_0.validate(var_14)
    var_16 = 0
    var_17 = '2022-01-01T12:00:00-05:30'
    var_18 = var_0.validate(var_17)
    var_19 = -5
    var_20 = -30
    var_21 = module_1.timedelta()
    var_22 = '2022-01-01T12:00:00.123456+05:30'
    var_23 = var_0.validate(var_22)
    var_24 = module_1.timedelta()
    var_25 = '2022-01-01T12:00:00.123456Z'
    var_26 = var_0.validate(var_25)
    var_27 = '2022-01-01T12:00:00.123456-05:30'
    var_28 = var_0.validate(var_27)
    var_29 = -5
    var_30 = -30
    var_31 = module_1.timedelta()



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://example.com'
    var_2 = var_0.validate(var_1)
    var_3 = 'not-a-url'
    var_4 = var_0.validate(var_3)
    var_5 = 'example.com'
    var_6 = var_0.validate(var_5)
    var_7 = 'http://'
    var_8 = var_0.validate(var_7)
    var_9 = 'https://www.example.com/path?query=param'
    var_10 = var_0.validate(var_9)
    var_11 = 'http://example.com:8080'
    var_12 = var_0.validate(var_11)
    var_13 = 'http://example.com#section'
    var_14 = var_0.validate(var_13)
    var_15 = 'http://user:pass@example.com'
    var_16 = var_0.validate(var_15)
    var_17 = 'http://example.com?key=value'
    var_18 = var_0.validate(var_17)
    var_19 = 'http://example.com/path/to/resource'
    var_20 = var_0.validate(var_19)
    var_21 = 'http://example.com/path?query=param#fragment'
    var_22 = var_0.validate(var_21)
    var_23 = 'ftp://example.com'
    var_24 = var_0.validate(var_23)
    var_25 = 'http://'
    var_26 = var_0.validate(var_25)
    var_27 = 'example.com'
    var_28 = var_0.validate(var_27)
    var_29 = ''
    var_30 = var_0.validate(var_29)
    var_31 = None
    var_32 = var_0.validate(var_31)
    var_33 = 'http:// '
    var_34 = var_0.validate(var_33)
    var_35 = 'http://   '
    var_36 = var_0.validate(var_35)
    var_37 = 'http://example .com'
    var_38 = var_0.validate(var_37)
    var_39 = 'http://example.com:port'
    var_40 = var_0.validate(var_39)
    var_41 = 'http://example.com:99999'
    var_42 = var_0.validate(var_41)
    var_43 = 'http://example.com:-1'
    var_44 = var_0.validate(var_43)
    var_45 = 'http://example.com:0'
    var_46 = var_0.validate(var_45)
    var_47 = 'http://example.com:80'
    var_48 = var_0.validate(var_47)
    var_49 = 'http://example.com:0080'
    var_50 = var_0.validate(var_49)
    var_51 = 'http://example.com:8000'
    var_52 = var_0.validate(var_51)
    var_53 = 'http://example.com:0800'
    var_54 = var_0.validate(var_53)
    var_55 = 'http://example.com:800'
    var_56 = var_0.validate(var_55)
    var_57 = 'http://example.com:80'
    var_58 = var_0.validate(var_57)
    var_59 = 'http://example.com:8'
    var_60 = var_0.validate(var_59)
    var_61 = 'http://example.com:1'
    var_62 = var_0.validate(var_61)
    var_63 = 'http://example.com:12'
    var_64 = var_0.validate(var_63)
    var_65 = 'http://example.com:123'
    var_66 = var_0.validate(var_65)
    var_67 = 'http://example.com:1234'
    var_68 = var_0.validate(var_67)
    var_69 = 'http://example.com:12345'
    var_70 = var_0.validate(var_69)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    var_3 = 'invalid_email'
    var_4 = var_0.validate(var_3)
    var_5 = ''
    var_6 = var_0.validate(var_5)
    var_7 = None
    var_8 = var_0.validate(var_7)
    var_9 = 'test+special@example.com'
    var_10 = var_0.validate(var_9)
    var_11 = 'Test@Example.com'
    var_12 = var_0.validate(var_11)
    var_13 = 'test123@example.com'
    var_14 = var_0.validate(var_13)
    var_15 = 'test.name@example.com'
    var_16 = var_0.validate(var_15)
    var_17 = 'test-name@example.com'
    var_18 = var_0.validate(var_17)
    var_19 = 'test_name@example.com'
    var_20 = var_0.validate(var_19)
    var_21 = 'test@sub.example.com'
    var_22 = var_0.validate(var_21)
    var_23 = 'test@example.testing'
    var_24 = var_0.validate(var_23)
    var_25 = 'test@example.co'
    var_26 = var_0.validate(var_25)
    var_27 = 'test@example.c'
    var_28 = var_0.validate(var_27)
    var_29 = 'test@example.test'
    var_30 = var_0.validate(var_29)
    var_31 = 'test@example.tests'
    var_32 = var_0.validate(var_31)
    var_33 = 'test@example.testin'
    var_34 = var_0.validate(var_33)
    var_35 = 'test@example.testing'
    var_36 = var_0.validate(var_35)
    var_37 = 'test@example.testings'
    var_38 = var_0.validate(var_37)
    var_39 = 'test@example.testingss'
    var_40 = var_0.validate(var_39)
    var_41 = 'test@example.testingsss'
    var_42 = var_0.validate(var_41)
    var_43 = 'test@example.testingssss'
    var_44 = var_0.validate(var_43)
    var_45 = 'test@example.testingsssss'
    var_46 = var_0.validate(var_45)
    var_47 = 'test@example.testingssssss'
    var_48 = var_0.validate(var_47)
    var_49 = 'test@example.testingsssssss'
    var_50 = var_0.validate(var_49)
    var_51 = 'test@example.testingssssssss'
    var_52 = var_0.validate(var_51)
    var_53 = 'test@example.testingsssssssss'
    var_54 = var_0.validate(var_53)
    var_55 = 'test@example.testingssssssssss'
    var_56 = var_0.validate(var_55)
    var_57 = 'test@example.testingsssssssssss'
    var_58 = var_0.validate(var_57)
    var_59 = 'test@example.testingssssssssssss'
    var_60 = var_0.validate(var_59)
    var_61 = 'test@example.testingsssssssssssss'
    var_62 = var_0.validate(var_61)
    var_63 = 'test@example.testingssssssssssssss'
    var_64 = var_0.validate(var_63)
    var_65 = 'test@example.testingsssssssssssssss'
    var_66 = var_0.validate(var_65)
    var_67 = 'test@example.testingssssssssssssssss'
    var_68 = var_0.validate(var_67)
    var_69 = 'test@example.testingsssssssssssssssss'
    var_70 = var_0.validate(var_69)
    var_71 = 'test@example.testingssssssssssssssssss'
    var_72 = var_0.validate(var_71)
    var_73 = 'test@example.testingsssssssssssssssssss'
    var_74 = var_0.validate(var_73)
    var_75 = 'test@example.testingssssssssssssssssssss'
    var_76 = var_0.validate(var_75)
    var_77 = 'test@example.testingsssssssssssssssssssss'
    var_78 = var_0.validate(var_77)
    var_79 = 'test@example.testingssssssssssssssssssssss'
    var_80 = var_0.validate(var_79)
    var_81 = 'test@example.testingsssssssssssssssssssssss'
    var_82 = var_0.validate(var_81)



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import uuid as module_1


def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_0.serialize(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'
    var_6 = 'not a uuid'
    var_7 = var_0.serialize(var_6)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'invalid-uuid'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.uuid4()
    var_7 = str(var_6)
    var_8 = var_0.validate(var_7)
    var_9 = '12345678-1234-5678-1234-567812345678'
    var_10 = var_0.validate(var_9)
    var_11 = str(var_10)
    var_12 = '12345678123456781234567812345678'
    var_13 = var_0.validate(var_12)
    var_14 = '12345678-1234-5678-1234-56781234567g'
    var_15 = var_0.validate(var_14)
    var_16 = '12345678-1234-0678-1234-567812345678'
    var_17 = var_0.validate(var_16)
    var_18 = '12345678-1234-5678-c234-567812345678'
    var_19 = var_0.validate(var_18)
    var_20 = '12345678-1234-5678-1234-5678123456789'
    var_21 = var_0.validate(var_20)
    var_22 = '12345678_1234_5678_1234_567812345678'
    var_23 = var_0.validate(var_22)
    var_24 = '12345678-1234-5678-1234-567812345678'
    var_25 = '12345678-1234-5678-1234-567812345678-extra'
    var_26 = var_0.validate(var_25)
    var_27 = 'extra-12345678-1234-5678-1234-567812345678'
    var_28 = var_0.validate(var_27)
    var_29 = '12345678-1234-5678-1234-567812345678-'
    var_30 = var_0.validate(var_29)
    var_31 = '-12345678-1234-5678-1234-567812345678'
    var_32 = var_0.validate(var_31)
    var_33 = '12345678-1234-5678-1234-567812345678-12345678'
    var_34 = var_0.validate(var_33)
    var_35 = '12345678-1234-5678-1234-567812345678-12345678-12345678'
    var_36 = var_0.validate(var_35)
    var_37 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678'
    var_38 = var_0.validate(var_37)
    var_39 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678-12345678'
    var_40 = var_0.validate(var_39)
    var_41 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678-12345678-12345678'
    var_42 = var_0.validate(var_41)
    var_43 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678-12345678-12345678-12345678'
    var_44 = var_0.validate(var_43)
    var_45 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678-12345678-12345678-12345678-12345678'
    var_46 = var_0.validate(var_45)
    var_47 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678-12345678-12345678-12345678-12345678-12345678'
    var_48 = var_0.validate(var_47)
    var_49 = '12345678-1234-5678-1234-567812345678-12345678-12345678-12345678-12345678-12345678-12345678-12345678-12345678-12345678'



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = '192.168.0.1'
    var_1 = module_0.IPAddressFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = var_1.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'invalid'
    var_8 = var_1.validate(var_7)
    var_9 = '256.256.256.256'
    var_10 = var_1.validate(var_9)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = '192.168.0.1'
    var_1 = module_0.IPAddressFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = var_1.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'invalid'
    var_8 = var_1.validate(var_7)
    var_9 = '256.256.256.256'
    var_10 = var_1.validate(var_9)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    assert var_2 == '12:30:45'
    assert var_2 == '12:30:45.123456'
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = 123456
    var_7 = 'invalid'
    var_8 = var_0.serialize(var_7)
    var_9 = ''
    var_10 = var_0.serialize(var_9)
    var_11 = 123
    var_12 = var_0.serialize(var_11)
    var_13 = 12.34
    var_14 = var_0.serialize(var_13)
    var_15 = 12
    var_16 = 30
    var_17 = 45
    var_18 = [var_15, var_16, var_17]
    var_19 = var_0.serialize(var_18)
    var_20 = 'hour'
    var_21 = 'minute'
    var_22 = 'second'
    var_23 = 12
    var_24 = 30
    var_25 = 45
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = var_0.serialize(var_26)
    var_28 = var_0.serialize(var_20)
    var_29 = var_0.serialize(var_20)
    var_30 = 25
    var_31 = 30
    var_32 = 45
    var_33 = var_0.serialize(var_23)
    var_34 = 12
    var_35 = 60
    var_36 = 45
    var_37 = var_0.serialize(var_23)
    var_38 = 12
    var_39 = 30
    var_40 = 60
    var_41 = var_0.serialize(var_23)
    var_42 = 12
    var_43 = 30
    var_44 = 45
    var_45 = 1000000
    var_46 = var_0.serialize(var_41)
    var_47 = -1
    var_48 = 30
    var_49 = 45
    var_50 = var_0.serialize(var_45)
    var_51 = 12
    var_52 = -1
    var_53 = 45
    var_54 = var_0.serialize(var_45)
    var_55 = 12
    var_56 = 30
    var_57 = -1
    var_58 = var_0.serialize(var_45)
    var_59 = 12
    var_60 = 30
    var_61 = 45
    var_62 = -1
    var_63 = var_0.serialize(var_58)
    var_64 = '12'
    var_65 = 30
    var_66 = 45
    var_67 = var_0.serialize(var_62)
    var_68 = 12
    var_69 = '30'
    var_70 = 45
    var_71 = var_0.serialize(var_62)
    var_72 = 12
    var_73 = 30
    var_74 = '45'
    var_75 = var_0.serialize(var_62)
    var_76 = 12
    var_77 = 30
    var_78 = 45
    var_79 = '123456'
    var_80 = var_0.serialize(var_75)
    var_81 = 12.5
    var_82 = 30
    var_83 = 45
    var_84 = var_0.serialize(var_79)
    var_85 = 12
    var_86 = 30.5
    var_87 = 45
    var_88 = var_0.serialize(var_79)
    var_89 = 12
    var_90 = 30
    var_91 = 45.5
    var_92 = var_0.serialize(var_79)
    var_93 = 12
    var_94 = 30
    var_95 = 45
    var_96 = 123456.5
    var_97 = var_0.serialize(var_92)
    var_98 = 12
    var_99 = [var_98]
    var_100 = 30
    var_101 = 45
    var_102 = var_0.serialize(var_92)
    var_103 = 12
    var_104 = 30
    var_105 = [var_104]
    var_106 = 45
    var_107 = var_0.serialize(var_92)
    var_108 = 12
    var_109 = 30
    var_110 = 45
    var_111 = [var_110]
    var_112 = var_0.serialize(var_92)
    var_113 = 12
    var_114 = 30
    var_115 = 45
    var_116 = 123456
    var_117 = [var_116]
    var_118 = var_0.serialize(var_112)
    var_119 = 'hour'
    var_120 = 12
    var_121 = {var_119: var_120}
    var_122 = 30
    var_123 = 45
    var_124 = var_0.serialize(var_112)
    var_125 = 12
    var_126 = 'minute'
    var_127 = 30
    var_128 = {var_126: var_127}
    var_129 = 45
    var_130 = var_0.serialize(var_112)
    var_131 = 12
    var_132 = 30
    var_133 = 'second'
    var_134 = 45
    var_135 = {var_133: var_134}
    var_136 = var_0.serialize(var_112)
    var_137 = 12
    var_138 = 30
    var_139 = 45
    var_140 = 'microsecond'
    var_141 = 123456
    var_142 = {var_140: var_141}
    var_143 = var_0.serialize(var_136)



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import datetime as module_1


def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2022
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 5
    var_8 = 30
    var_9 = module_1.timedelta()
    var_10 = -5
    var_11 = -30
    var_12 = module_1.timedelta()
    var_13 = 123456
    var_14 = module_1.timedelta()
    var_15 = -5
    var_16 = -30
    var_17 = module_1.timedelta()
    var_18 = module_1.timedelta()
    var_19 = module_1.timedelta()
    var_20 = module_1.timedelta()
    var_21 = module_1.timedelta()
    var_22 = -5
    var_23 = -30
    var_24 = module_1.timedelta()
    var_25 = module_1.timedelta()
    var_26 = -5
    var_27 = -30
    var_28 = module_1.timedelta()
    var_29 = module_1.timedelta()
    var_30 = -5
    var_31 = -30
    var_32 = module_1.timedelta()
    var_33 = module_1.timedelta()



# Parsed testcases at query #17
#--------------------------


import ipaddress as module_3
import uuid as module_2


def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2022-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = '2022/01/01'
    var_4 = var_0.validate(var_3)
    var_5 = '2022-02-30'
    var_6 = var_0.validate(var_5)
    var_7 = '2022-01'
    var_8 = var_0.validate(var_7)
    var_9 = '2022'
    var_10 = var_0.validate(var_9)
    var_11 = ''
    var_12 = var_0.validate(var_11)
    var_13 = None
    var_14 = var_0.validate(var_13)
    var_15 = 20220101
    var_16 = var_0.validate(var_15)
    var_17 = 2022.0101
    var_18 = var_0.validate(var_17)
    var_19 = True
    var_20 = var_0.validate(var_19)
    var_21 = '2022'
    var_22 = '01'
    var_23 = [var_21, var_22, var_22]
    var_24 = var_0.validate(var_23)
    var_25 = 'year'
    var_26 = 'month'
    var_27 = 'day'
    var_28 = 2022
    var_29 = 1
    var_30 = {var_25: var_28, var_26: var_29, var_27: var_29}
    var_31 = var_0.validate(var_30)
    var_32 = (var_28, var_29, var_29)
    var_33 = var_0.validate(var_32)
    var_34 = {var_28, var_29, var_29}
    var_35 = var_0.validate(var_34)
    var_36 = {var_28, var_29, var_29}
    var_37 = frozenset(var_36)
    var_38 = var_0.validate(var_37)
    var_39 = 2023
    var_40 = range(var_28, var_39)
    var_41 = var_0.validate(var_40)
    var_42 = b'2022-01-01'
    var_43 = var_0.validate(var_42)
    var_44 = b'2022-01-01'
    var_45 = bytearray(var_44)
    var_46 = var_0.validate(var_45)
    var_47 = memoryview(var_44)
    var_48 = var_0.validate(var_47)
    var_49 = complex(var_28, var_29)
    var_50 = var_0.validate(var_49)
    var_51 = '2022.01'
    var_52 = var_0.validate(var_49)
    var_53 = var_0.validate(var_49)
    var_54 = var_0.validate(var_49)
    var_55 = module_1.timedelta()
    var_56 = var_0.validate(var_55)
    var_57 = var_0.validate(var_55)
    var_58 = module_2.uuid4()
    var_59 = var_0.validate(var_58)
    var_60 = '192.168.0.1'
    var_61 = module_3.IPv4Address(var_60)
    var_62 = var_0.validate(var_61)
    var_63 = '2001:db8::'
    var_64 = module_3.IPv6Address(var_63)
    var_65 = var_0.validate(var_64)
    var_66 = 'https://example.com'
    var_67 = var_0.validate(var_66)
    var_68 = 'test@example.com'
    var_69 = var_0.validate(var_68)
    var_70 = '192.168.0.1'
    var_71 = var_0.validate(var_70)
    var_72 = 'https://example.com'
    var_73 = var_0.validate(var_72)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    assert var_2 == '2022-01-01'
    assert var_2 == '-2022-01-01'
    assert var_2 == '0000-01-01'
    assert var_2 == '9999-12-31'
    var_3 = 2022
    var_4 = 1
    var_5 = '2022-01-01'
    var_6 = var_0.serialize(var_5)
    var_7 = 12
    var_8 = 0
    var_9 = '2022-01-01'
    var_10 = var_0.serialize(var_9)
    var_11 = 20220101
    var_12 = var_0.serialize(var_11)
    var_13 = 2022.0101
    var_14 = var_0.serialize(var_13)
    var_15 = True
    var_16 = var_0.serialize(var_15)
    var_17 = 2022
    var_18 = 1
    var_19 = [var_17, var_18, var_18]
    var_20 = var_0.serialize(var_19)
    var_21 = 'year'
    var_22 = 'month'
    var_23 = 'day'
    var_24 = 2022
    var_25 = 1
    var_26 = {var_21: var_24, var_22: var_25, var_23: var_25}
    var_27 = var_0.serialize(var_26)
    var_28 = 2022
    var_29 = 1
    var_30 = (var_28, var_29, var_29)
    var_31 = var_0.serialize(var_30)
    var_32 = 2022
    var_33 = 1
    var_34 = {var_32, var_33, var_33}
    var_35 = var_0.serialize(var_34)
    var_36 = 2022
    var_37 = 1
    var_38 = [var_36, var_37, var_37]
    var_39 = frozenset(var_38)
    var_40 = var_0.serialize(var_39)
    var_41 = 2022
    var_42 = 2023
    var_43 = range(var_41, var_42)
    var_44 = var_0.serialize(var_43)
    var_45 = b'2022-01-01'
    var_46 = var_0.serialize(var_45)
    var_47 = b'2022-01-01'
    var_48 = bytearray(var_47)
    var_49 = var_0.serialize(var_48)
    var_50 = b'2022-01-01'
    var_51 = memoryview(var_50)
    var_52 = var_0.serialize(var_51)
    var_53 = 2022
    var_54 = 1
    var_55 = complex(var_53, var_54)
    var_56 = var_0.serialize(var_55)
    var_57 = '2022.01'
    var_58 = var_0.serialize(var_54)
    var_59 = 2022
    var_60 = 1
    var_61 = var_0.serialize(var_58)
    var_62 = 1
    var_63 = module_1.timedelta()
    var_64 = var_0.serialize(var_63)
    var_65 = var_0.serialize(var_62)
    var_66 = 12
    var_67 = 0
    var_68 = var_0.serialize(var_64)
    var_69 = 123456
    var_70 = -2022
    var_71 = 9999
    var_72 = 31
    var_73 = 23
    var_74 = 59
    var_75 = 10000
    var_76 = 1
    var_77 = 12
    var_78 = 0
    var_79 = 2022
    var_80 = 13
    var_81 = 1
    var_82 = 12
    var_83 = 0
    var_84 = 2022
    var_85 = 1
    var_86 = 32
    var_87 = 12
    var_88 = 0
    var_89 = 2022
    var_90 = 1
    var_91 = 24
    var_92 = 0



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    assert var_2 == '12:30:45'
    assert var_2 == '12:30:45.123456'
    assert var_2 == '12:30:45+00:00'
    assert var_2 == '12:30:45+05:30'
    assert var_2 == '12:30:45.123456+00:00'
    assert var_2 == '12:30:45.123456+05:30'
    assert var_2 == '12:30:45.123456-05:30'
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = 123456
    var_7 = 5
    var_8 = module_1.timedelta()
    var_9 = module_1.timedelta()
    var_10 = -5
    var_11 = -30
    var_12 = module_1.timedelta()
    var_13 = 0
    var_14 = module_1.timedelta()
    var_15 = module_1.timedelta()
    var_16 = module_1.timedelta()
    var_17 = module_1.timedelta()
    var_18 = module_1.timedelta()
    var_19 = module_1.timedelta()
    var_20 = module_1.timedelta()
    var_21 = module_1.timedelta()
    var_22 = module_1.timedelta()
    var_23 = module_1.timedelta()
    var_24 = module_1.timedelta()
    var_25 = module_1.timedelta()
    var_26 = module_1.timedelta()



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    var_3 = 'invalid_email'
    var_4 = var_0.validate(var_3)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    var_3 = 'invalid-email'
    var_4 = var_0.validate(var_3)
    var_5 = ''
    var_6 = var_0.validate(var_5)
    var_7 = None
    var_8 = var_0.validate(var_7)
    var_9 = 'test+tag@example.com'
    var_10 = var_0.validate(var_9)
    var_11 = 'Test@Example.com'
    var_12 = var_0.validate(var_11)
    var_13 = 'test123@example.com'
    var_14 = var_0.validate(var_13)
    var_15 = 'test.name@example.com'
    var_16 = var_0.validate(var_15)
    var_17 = 'test-name@example.com'
    var_18 = var_0.validate(var_17)
    var_19 = 'test_name@example.com'
    var_20 = var_0.validate(var_19)
    var_21 = 'test@sub.example.com'
    var_22 = var_0.validate(var_21)
    var_23 = 'test@example.co.uk'
    var_24 = var_0.validate(var_23)
    var_25 = 'test@example.testing'
    var_26 = var_0.validate(var_25)
    var_27 = 'test@example.co'
    var_28 = var_0.validate(var_27)
    var_29 = 'test@example.com'
    var_30 = var_0.validate(var_29)
    var_31 = 'test@example.abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl'
    var_32 = var_0.validate(var_31)
    var_33 = 'test@example.'
    var_34 = 'a'
    var_35 = 64
    var_36 = var_34 * var_35
    var_37 = var_33 + var_36
    var_38 = var_0.validate(var_37)
    var_39 = 65
    var_40 = var_34 * var_39
    var_41 = var_33 + var_40
    var_42 = var_0.validate(var_41)
    var_43 = 'test@example.com-'
    var_44 = var_0.validate(var_43)
    var_45 = 'test@example.-com'
    var_46 = var_0.validate(var_45)
    var_47 = 'test@example.co--uk'
    var_48 = var_0.validate(var_47)
    var_49 = 'test@example.co-uk'
    var_50 = var_0.validate(var_49)
    var_51 = 'test@example.co-'
    var_52 = var_0.validate(var_51)
    var_53 = 'test@example.-co'
    var_54 = var_0.validate(var_53)
    var_55 = 'test@example.co-uk'
    var_56 = var_0.validate(var_55)
    var_57 = 'test@example.1com'
    var_58 = var_0.validate(var_57)
    var_59 = 'test@example.com1'
    var_60 = var_0.validate(var_59)
    var_61 = 'test@example.123'
    var_62 = var_0.validate(var_61)
    var_63 = 'test@example.abc'
    var_64 = var_0.validate(var_63)
    var_65 = 'test@example.--'
    var_66 = var_0.validate(var_65)
    var_67 = 'test@example.__'
    var_68 = var_0.validate(var_67)
    var_69 = "test@example.!#$%&'*+/=?^_`{}|~"
    var_70 = var_0.validate(var_69)
    var_71 = 'test@example.a-b1'
    var_72 = var_0.validate(var_71)
    var_73 = 'test@example.a_b1'
    var_74 = var_0.validate(var_73)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = '192.168.0.1'
    var_1 = module_0.IPAddressFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = var_1.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'invalid'
    var_8 = var_1.validate(var_7)
    var_9 = '999.999.999.999'
    var_10 = var_1.validate(var_9)
    var_11 = ''
    var_12 = var_1.validate(var_11)
    var_13 = None
    var_14 = var_1.validate(var_13)
    var_15 = ' 192.168.0.1 '
    var_16 = var_1.validate(var_15)
    var_17 = str(var_16)
    var_18 = '192.168.001.001'
    var_19 = var_1.validate(var_18)
    var_20 = str(var_19)
    var_21 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_22 = var_1.validate(var_21)
    var_23 = str(var_22)
    var_24 = '2001:db8::8a2e:370:7334'
    var_25 = var_1.validate(var_24)
    var_26 = str(var_25)
    var_27 = '2001:0DB8:85A3:0000:0000:8A2E:0370:7334'
    var_28 = var_1.validate(var_27)
    var_29 = str(var_28)
    var_30 = ' 2001:0db8:85a3:0000:0000:8a2e:0370:7334 '
    var_31 = var_1.validate(var_30)
    var_32 = str(var_31)
    var_33 = '2001:0db8:85a3:0000:0000:8a2e:0370:733g'
    var_34 = var_1.validate(var_33)
    var_35 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234'
    var_36 = var_1.validate(var_35)
    var_37 = '2001:0db8:85a3:0000:0000:8a2e:0370'
    var_38 = var_1.validate(var_37)
    var_39 = '2001:0db8:85a3:0000:0000:8a2e:0370:73345'
    var_40 = var_1.validate(var_39)
    var_41 = '2001:0db8:85a3:0000:0000:8a2e:0370:733g'
    var_42 = var_1.validate(var_41)
    var_43 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334:'
    var_44 = var_1.validate(var_43)
    var_45 = ':2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_46 = var_1.validate(var_45)
    var_47 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334::'
    var_48 = var_1.validate(var_47)
    var_49 = '::2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_50 = var_1.validate(var_49)
    var_51 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334::1234'
    var_52 = var_1.validate(var_51)
    var_53 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234::'
    var_54 = var_1.validate(var_53)
    var_55 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234:5678'
    var_56 = var_1.validate(var_55)
    var_57 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234:5678:9abc'
    var_58 = var_1.validate(var_57)
    var_59 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234:5678:9abc:def0'



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2022-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = '01-01-2022'
    var_4 = var_0.validate(var_3)
    var_5 = '2022-13-01'
    var_6 = var_0.validate(var_5)
    var_7 = 123
    var_8 = var_0.validate(var_7)
    var_9 = ''
    var_10 = var_0.validate(var_9)
    var_11 = None
    var_12 = var_0.validate(var_11)
    var_13 = '2022-01-01'
    var_14 = var_0.validate(var_13)
    var_15 = '2022-1-1'
    var_16 = var_0.validate(var_15)
    var_17 = '2022-12-31'
    var_18 = var_0.validate(var_17)
    var_19 = '2020-02-29'
    var_20 = var_0.validate(var_19)
    var_21 = '2021-02-29'
    var_22 = var_0.validate(var_21)
    var_23 = '2022-13-01'
    var_24 = var_0.validate(var_23)
    var_25 = '2022-01-32'
    var_26 = var_0.validate(var_25)
    var_27 = '0000-01-01'
    var_28 = var_0.validate(var_27)
    var_29 = '-2022-01-01'
    var_30 = var_0.validate(var_29)
    var_31 = '2022-01-01 extra'
    var_32 = var_0.validate(var_31)
    var_33 = '2022-01'
    var_34 = var_0.validate(var_33)
    var_35 = '2022-01-01-'
    var_36 = var_0.validate(var_35)
    var_37 = '2022-01-01 '
    var_38 = var_0.validate(var_37)
    var_39 = '2022-01-01T00:00:00'
    var_40 = var_0.validate(var_39)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2022-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = '2022-01-01T12:00:00.123456'
    var_4 = var_0.validate(var_3)
    var_5 = '2022-01-01T12:00:00+05:30'
    var_6 = var_0.validate(var_5)
    var_7 = 5
    var_8 = 30
    var_9 = module_1.timedelta()
    var_10 = '2022-01-01T25:00:00'
    var_11 = var_0.validate(var_10)
    var_12 = '2022-01-01 12:00:00'
    var_13 = var_0.validate(var_12)
    var_14 = '2022-01-01T12:00:00Z'
    var_15 = var_0.validate(var_14)
    var_16 = '2022-01-01T12:00:00-05:30'
    var_17 = var_0.validate(var_16)
    var_18 = -5
    var_19 = -30
    var_20 = module_1.timedelta()
    var_21 = '2022-01-01T12:00:00.123456+05:30'
    var_22 = var_0.validate(var_21)
    var_23 = module_1.timedelta()
    var_24 = '2022-01-01T12:00:00.123456Z'
    var_25 = var_0.validate(var_24)
    var_26 = '2022-01-01T12:00:00.123456-05:30'
    var_27 = var_0.validate(var_26)
    var_28 = -5
    var_29 = -30
    var_30 = module_1.timedelta()



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = '192.168.0.1'
    var_1 = module_0.IPAddressFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = var_1.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'invalid'
    var_8 = var_1.validate(var_7)
    var_9 = '999.999.999.999'
    var_10 = var_1.validate(var_9)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2022-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = '2022-01-01T12:00:00.123456'
    var_4 = var_0.validate(var_3)
    var_5 = '2022-01-01T12:00:00Z'
    var_6 = var_0.validate(var_5)
    var_7 = '2022-01-01T12:00:00+05:30'
    var_8 = var_0.validate(var_7)
    var_9 = 5
    var_10 = 30
    var_11 = module_1.timedelta()
    var_12 = '2022-01-01T12:00:00+'
    var_13 = var_0.validate(var_12)
    var_14 = '2022-13-01T12:00:00'
    var_15 = var_0.validate(var_14)
    var_16 = '2022-01-32T12:00:00'
    var_17 = var_0.validate(var_16)
    var_18 = '2022-01-01T24:00:00'
    var_19 = var_0.validate(var_18)
    var_20 = '2022-01-01T12:60:00'
    var_21 = var_0.validate(var_20)
    var_22 = '2022-01-01T12:00:60'
    var_23 = var_0.validate(var_22)
    var_24 = '2022-01-01T12:00:00.9999999'
    var_25 = var_0.validate(var_24)
    var_26 = '2022-01-01T12:00:00+25:00'
    var_27 = var_0.validate(var_26)
    var_28 = '2022-01-01T12:00:00+05:60'
    var_29 = var_0.validate(var_28)
    var_30 = '2022-01-01T12:00:00+05'
    var_31 = var_0.validate(var_30)
    var_32 = '2022-01-01T12:00:00+05:30:30'
    var_33 = var_0.validate(var_32)
    var_34 = '2022-01-01T12:00:00*05:30'
    var_35 = var_0.validate(var_34)
    var_36 = '2022-01-01T12:00:00-05:30:30'
    var_37 = var_0.validate(var_36)
    var_38 = '2022-01-01T12:00:00-05:30:30'
    var_39 = var_0.validate(var_38)
    var_40 = '2022-01-01T12:00:00-05:30:30'
    var_41 = var_0.validate(var_40)
    var_42 = '2022-01-01T12:00:00-05:30:30'
    var_43 = var_0.validate(var_42)
    var_44 = '2022-01-01T12:00:00-05:30:30'
    var_45 = var_0.validate(var_44)
    var_46 = '2022-01-01T12:00:00-05:30:30'
    var_47 = var_0.validate(var_46)
    var_48 = '2022-01-01T12:00:00-05:30:30'
    var_49 = var_0.validate(var_48)
    var_50 = '2022-01-01T12:00:00-05:30:30'
    var_51 = var_0.validate(var_50)
    var_52 = '2022-01-01T12:00:00-05:30:30'
    var_53 = var_0.validate(var_52)
    var_54 = '2022-01-01T12:00:00-05:30:30'
    var_55 = var_0.validate(var_54)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2021
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 5
    var_8 = 30
    var_9 = module_1.timedelta()
    var_10 = -5
    var_11 = -30
    var_12 = module_1.timedelta()
    var_13 = 123456
    var_14 = module_1.timedelta()
    var_15 = -5
    var_16 = -30
    var_17 = module_1.timedelta()
    var_18 = module_1.timedelta()
    var_19 = 0
    var_20 = module_1.timedelta()
    var_21 = 0
    var_22 = module_1.timedelta()
    var_23 = module_1.timedelta()
    var_24 = -30
    var_25 = module_1.timedelta()
    var_26 = module_1.timedelta()
    var_27 = -5
    var_28 = -30
    var_29 = module_1.timedelta()
    var_30 = module_1.timedelta()
    var_31 = -5
    var_32 = module_1.timedelta()
    var_33 = 45
    var_34 = module_1.timedelta()
    var_35 = -5
    var_36 = -30
    var_37 = -45
    var_38 = module_1.timedelta()



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '25:30'
    var_6 = var_0.validate(var_5)
    var_7 = '12:30:45.123456'
    var_8 = var_0.validate(var_7)
    var_9 = 45
    var_10 = 123456
    var_11 = '12:30:45.123'
    var_12 = var_0.validate(var_11)
    var_13 = 123000
    var_14 = '12:30:45.123456789'
    var_15 = var_0.validate(var_14)
    var_16 = '12'
    var_17 = var_0.validate(var_16)
    var_18 = '25:30'
    var_19 = var_0.validate(var_18)
    var_20 = '12:60'
    var_21 = var_0.validate(var_20)
    var_22 = '12:30:60'
    var_23 = var_0.validate(var_22)
    var_24 = '12:30:45.1234567'
    var_25 = var_0.validate(var_24)
    var_26 = '12:30:abc'
    var_27 = var_0.validate(var_26)
    var_28 = ''
    var_29 = var_0.validate(var_28)
    var_30 = None
    var_31 = var_0.validate(var_30)
    var_32 = 123
    var_33 = var_0.validate(var_32)
    var_34 = 12.5
    var_35 = var_0.validate(var_34)
    var_36 = 12
    var_37 = 30
    var_38 = [var_36, var_37]
    var_39 = var_0.validate(var_38)
    var_40 = 'hour'
    var_41 = 'minute'
    var_42 = 12
    var_43 = 30
    var_44 = {var_40: var_42, var_41: var_43}
    var_45 = var_0.validate(var_44)
    var_46 = 12
    var_47 = 30
    var_48 = (var_46, var_47)
    var_49 = var_0.validate(var_48)
    var_50 = 12
    var_51 = 30
    var_52 = {var_50, var_51}
    var_53 = var_0.validate(var_52)
    var_54 = True
    var_55 = var_0.validate(var_54)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = 123456
    var_7 = 5
    var_8 = module_1.timedelta()
    var_9 = -5
    var_10 = -30
    var_11 = module_1.timedelta()
    var_12 = module_1.timedelta()
    var_13 = -5
    var_14 = -45
    var_15 = module_1.timedelta()
    var_16 = 15
    var_17 = module_1.timedelta()
    var_18 = -5
    var_19 = -30
    var_20 = -15
    var_21 = module_1.timedelta()
    var_22 = module_1.timedelta()
    var_23 = -5
    var_24 = -30
    var_25 = -123456
    var_26 = module_1.timedelta()
    var_27 = module_1.timedelta()
    var_28 = -5
    var_29 = -30
    var_30 = -15
    var_31 = -123456
    var_32 = module_1.timedelta()
    var_33 = module_1.timedelta()
    var_34 = -5
    var_35 = -30
    var_36 = -123456
    var_37 = module_1.timedelta()
    var_38 = module_1.timedelta()
    var_39 = -5
    var_40 = -123456
    var_41 = module_1.timedelta()
    var_42 = module_1.timedelta()
    var_43 = -5
    var_44 = -30
    var_45 = -123456
    var_46 = module_1.timedelta()
    var_47 = module_1.timedelta()
    var_48 = -5
    var_49 = -30
    var_50 = -15
    var_51 = -123456
    var_52 = module_1.timedelta()



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2022-01-01T12:00:00+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = '2022-01-01T12:00:00'
    var_7 = var_0.validate(var_6)
    var_8 = '2022-01-01T25:00:00'
    var_9 = var_0.validate(var_8)
    var_10 = '2022-01-01 12:00:00'
    var_11 = var_0.validate(var_10)
    var_12 = '2022-01-01T12:00:00.123456'
    var_13 = var_0.validate(var_12)
    var_14 = '2022-01-01T12:00:00Z'
    var_15 = var_0.validate(var_14)
    var_16 = '2022-01-01T12:00:00-05:30'
    var_17 = var_0.validate(var_16)
    var_18 = -5
    var_19 = -30
    var_20 = module_1.timedelta()
    var_21 = '2022-01-01T12:00:00+00:00'
    var_22 = var_0.validate(var_21)
    var_23 = '2022-01-01T12:00:00-12:00'
    var_24 = var_0.validate(var_23)
    var_25 = -12
    var_26 = module_1.timedelta()
    var_27 = '2022-01-01T12:00:00+14:00'
    var_28 = var_0.validate(var_27)
    var_29 = 14
    var_30 = module_1.timedelta()
    var_31 = '2022-01-01T12:00:00.123456+05:30'
    var_32 = var_0.validate(var_31)
    var_33 = module_1.timedelta()
    var_34 = '2022-01-01T12:00:00.123456-05:30'
    var_35 = var_0.validate(var_34)
    var_36 = -5
    var_37 = -30
    var_38 = module_1.timedelta()
    var_39 = '2022-01-01T12:00:00.123456Z'
    var_40 = var_0.validate(var_39)
    var_41 = '2022-01-01T12:00:00.123456+00:00'
    var_42 = var_0.validate(var_41)
    var_43 = '2022-01-01T12:00:00.123456-12:00'
    var_44 = var_0.validate(var_43)
    var_45 = -12
    var_46 = module_1.timedelta()
    var_47 = '2022-01-01T12:00:00.123456+14:00'
    var_48 = var_0.validate(var_47)
    var_49 = module_1.timedelta()



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    assert var_2 == '2022-01-01'
    assert var_2 == '2022-12-31'
    assert var_2 == '2020-02-29'
    assert var_2 == '2021-02-28'
    assert var_2 == '0001-01-01'
    assert var_2 == '9999-12-31'
    var_3 = 2022
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = 2020
    var_8 = 2
    var_9 = 29
    var_10 = 2021
    var_11 = 28
    var_12 = 9999
    var_13 = -1
    var_14 = 1
    var_15 = 2022
    var_16 = 13
    var_17 = 1
    var_18 = 2022
    var_19 = 1
    var_20 = 32
    var_21 = 2022
    var_22 = 13
    var_23 = 32
    var_24 = -1
    var_25 = 13
    var_26 = 32
    var_27 = 10000
    var_28 = 13
    var_29 = 32
    var_30 = 10000
    var_31 = 0
    var_32 = 0
    var_33 = 10000
    var_34 = 0
    var_35 = 32
    var_36 = 10000
    var_37 = 13
    var_38 = 0
    var_39 = 0
    var_40 = 13
    var_41 = 32
    var_42 = 0
    var_43 = 32
    var_44 = 0
    var_45 = 13
    var_46 = 10000
    var_47 = 0
    var_48 = 0
    var_49 = 10000
    var_50 = 0
    var_51 = 32
    var_52 = 10000
    var_53 = 13
    var_54 = 0
    var_55 = 0
    var_56 = 13
    var_57 = 32
    var_58 = 0
    var_59 = 32
    var_60 = 0
    var_61 = 13
    var_62 = 10000
    var_63 = 0
    var_64 = 0
    var_65 = 10000
    var_66 = 0
    var_67 = 32
    var_68 = 10000
    var_69 = 13
    var_70 = 0
    var_71 = 0
    var_72 = 13
    var_73 = 32
    var_74 = 0
    var_75 = 32
    var_76 = 0
    var_77 = 13
    var_78 = 10000
    var_79 = 0
    var_80 = 0



