####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 11
    var_5 = None
    var_6 = 0



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.date as module_0


def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.timestamp()
    var_2 = 2020
    var_3 = 2022
    var_4 = var_0.timestamp()
    var_5 = 'invalid_format'
    var_6 = var_0.timestamp(var_5)
    var_7 = 2022
    var_8 = 2020
    var_9 = var_0.timestamp()
    var_10 = 'UTC'
    var_11 = var_0.timestamp()
    var_12 = len(var_11)
    assert var_12 == 20
    var_13 = len(var_11)
    assert var_13 == 26



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2010
    var_3 = 2020
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)
    var_9 = 2020
    var_10 = 2010
    var_11 = var_0.datetime(var_9, var_10)
    var_12 = '2020'
    var_13 = 2020
    var_14 = var_0.datetime(var_12, var_13)
    var_15 = -100
    var_16 = -50
    var_17 = var_0.datetime(var_15, var_16)
    var_18 = 3000
    var_19 = 4000
    var_20 = var_0.datetime(var_18, var_19)
    var_21 = 2020
    var_22 = var_0.datetime(var_21, var_21)
    var_23 = None
    var_24 = var_0.datetime(timezone=var_23)
    var_25 = ''
    var_26 = var_0.datetime(timezone=var_25)
    var_27 = 'UTC'
    var_28 = var_0.datetime(timezone=var_27)
    var_29 = 'America/New_York'
    var_30 = var_0.datetime(timezone=var_29)
    var_31 = 'Europe/London'
    var_32 = var_0.datetime(timezone=var_31)
    var_33 = 'Asia/Tokyo'
    var_34 = var_0.datetime(timezone=var_33)
    var_35 = 'Australia/Sydney'
    var_36 = var_0.datetime(timezone=var_35)
    var_37 = 'Africa/Cairo'
    var_38 = var_0.datetime(timezone=var_37)
    var_39 = 'Pacific/Honolulu'
    var_40 = var_0.datetime(timezone=var_39)
    var_41 = 'Antarctica/McMurdo'
    var_42 = var_0.datetime(timezone=var_41)
    var_43 = 'Arctic/Longyearbyen'
    var_44 = var_0.datetime(timezone=var_43)
    var_45 = 'Indian/Christmas'
    var_46 = var_0.datetime(timezone=var_45)
    var_47 = 'Etc/GMT'
    var_48 = var_0.datetime(timezone=var_47)
    var_49 = 'Etc/GMT+1'
    var_50 = var_0.datetime(timezone=var_49)
    var_51 = 'Etc/GMT-1'
    var_52 = var_0.datetime(timezone=var_51)
    var_53 = 'Etc/UTC'
    var_54 = var_0.datetime(timezone=var_53)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 5
    var_3 = 2
    var_4 = 6
    var_5 = 1
    var_6 = 0
    var_7 = None
    var_8 = 1
    var_9 = 12
    var_10 = 0
    var_11 = 500000
    var_12 = 100000
    var_13 = -1
    var_14 = 10
    var_15 = 0
    var_16 = 30
    var_17 = 7
    var_18 = 1
    var_19 = 1.5
    var_20 = 9999
    var_21 = 31
    var_22 = 1000
    var_23 = 1001
    var_24 = 9001
    var_25 = 3650000
    var_26 = 1
    var_27 = 3
    var_28 = 4



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2000
    var_3 = 2010
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)
    var_9 = 2020
    var_10 = 2010
    var_11 = var_0.datetime(var_9, var_10)
    var_12 = '2000'
    var_13 = 2010
    var_14 = var_0.datetime(var_12, var_13)
    var_15 = -100
    var_16 = -50
    var_17 = var_0.datetime(var_15, var_16)
    var_18 = 3000
    var_19 = 4000
    var_20 = var_0.datetime(var_18, var_19)
    var_21 = 2020
    var_22 = var_0.datetime(var_21, var_21)
    var_23 = ''
    var_24 = var_0.datetime(timezone=var_23)
    var_25 = None
    var_26 = var_0.datetime(timezone=var_25)
    var_27 = 'America/New_York'
    var_28 = var_0.datetime(timezone=var_27)
    var_29 = 'UTC+05:30'
    var_30 = var_0.datetime(timezone=var_29)
    var_31 = 'UTC-08:00'
    var_32 = var_0.datetime(timezone=var_31)
    var_33 = 'US/Eastern'
    var_34 = var_0.datetime(timezone=var_33)
    var_35 = 'America/Argentina/Buenos_Aires'
    var_36 = var_0.datetime(timezone=var_35)
    var_37 = 'Asia/Kolkata'
    var_38 = var_0.datetime(timezone=var_37)
    var_39 = 'America/Port-au-Prince'
    var_40 = var_0.datetime(timezone=var_39)
    var_41 = "America/St_John's"
    var_42 = var_0.datetime(timezone=var_41)
    var_43 = 'America/Argentina/ComodRivadavia'
    var_44 = var_0.datetime(timezone=var_43)
    var_45 = 'America/Argentina (Córdoba)'
    var_46 = var_0.datetime(timezone=var_45)
    var_47 = 'America/Argentina/Catamarca'
    var_48 = var_0.datetime(timezone=var_47)
    var_49 = 'America/Argentina, La Rioja'
    var_50 = var_0.datetime(timezone=var_49)
    var_51 = 'America/Argentina; Mendoza'
    var_52 = var_0.datetime(timezone=var_51)
    var_53 = 'America/Argentina: San Juan'
    var_54 = var_0.datetime(timezone=var_53)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = 6
    var_8 = 0
    var_9 = 8
    var_10 = 1
    var_11 = 0
    var_12 = None
    var_13 = 1



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 2
    var_3 = 5
    var_4 = None
    var_5 = var_0.duration(duration_unit=var_4)
    var_6 = 10
    var_7 = 5
    var_8 = var_0.duration(var_6, var_7)
    var_9 = 1.5
    var_10 = 5
    var_11 = var_0.duration(var_9, var_10)
    var_12 = 1
    var_13 = 5.5
    var_14 = var_0.duration(var_12, var_13)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = None
    var_6 = var_0.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_0.duration(var_7, var_8)
    var_10 = 1.5
    var_11 = 10
    var_12 = var_0.duration(var_10, var_11)
    var_13 = 1
    var_14 = 10.5
    var_15 = var_0.duration(var_13, var_14)
    var_16 = -5
    var_17 = -1
    var_18 = var_0.duration(var_16, var_17)
    var_19 = 0
    var_20 = var_0.duration(var_19, var_19)
    var_21 = 1000
    var_22 = 2000
    var_23 = var_0.duration(var_21, var_22)



# Parsed testcases at query #7
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 1
    var_3 = 13
    var_4 = range(var_2, var_3)
    var_5 = 32
    var_6 = range(var_2, var_5)
    var_7 = 0
    var_8 = 24
    var_9 = range(var_7, var_8)
    var_10 = 60
    var_11 = range(var_7, var_10)
    var_12 = range(var_7, var_10)
    var_13 = 1000000
    var_14 = range(var_7, var_13)
    var_15 = 2000
    var_16 = 2010
    var_17 = var_0.datetime(var_15, var_16)
    var_18 = 'UTC'
    var_19 = var_0.datetime(timezone=var_18)
    var_20 = 'Invalid/Timezone'
    var_21 = var_0.datetime(timezone=var_20)
    var_22 = 2020
    var_23 = 2010
    var_24 = var_0.datetime(var_22, var_23)
    var_25 = '2000'
    var_26 = 2010
    var_27 = var_0.datetime(var_25, var_26)
    var_28 = -1000
    var_29 = 2010
    var_30 = var_0.datetime(var_28, var_29)
    var_31 = 2000.5
    var_32 = 2010.5
    var_33 = var_0.datetime(var_31, var_32)
    var_34 = '2000'
    var_35 = '2010'
    var_36 = var_0.datetime(var_34, var_35)
    var_37 = None
    var_38 = var_0.datetime(var_37, var_37)
    var_39 = True
    var_40 = False
    var_41 = var_0.datetime(var_39, var_40)
    var_42 = 2000
    var_43 = [var_42]
    var_44 = 2010
    var_45 = [var_44]
    var_46 = var_0.datetime(var_43, var_45)
    var_47 = 2000
    var_48 = (var_47,)
    var_49 = 2010
    var_50 = (var_49,)
    var_51 = var_0.datetime(var_48, var_50)
    var_52 = 'year'
    var_53 = 2000
    var_54 = {var_52: var_53}
    var_55 = 2010
    var_56 = {var_52: var_55}
    var_57 = var_0.datetime(var_54, var_56)
    var_58 = 2000
    var_59 = {var_58}
    var_60 = 2010
    var_61 = {var_60}
    var_62 = var_0.datetime(var_59, var_61)
    var_63 = 2000
    var_64 = [var_63]
    var_65 = frozenset(var_64)
    var_66 = 2010
    var_67 = [var_66]
    var_68 = frozenset(var_67)
    var_69 = var_0.datetime(var_65, var_68)
    var_70 = b'2000'
    var_71 = b'2010'
    var_72 = var_0.datetime(var_70, var_71)
    var_73 = b'2000'
    var_74 = bytearray(var_73)
    var_75 = b'2010'
    var_76 = bytearray(var_75)
    var_77 = var_0.datetime(var_74, var_76)
    var_78 = b'2000'
    var_79 = memoryview(var_78)
    var_80 = b'2010'
    var_81 = memoryview(var_80)
    var_82 = var_0.datetime(var_79, var_81)
    var_83 = 2000
    var_84 = 0
    var_85 = complex(var_83, var_84)
    var_86 = 2010
    var_87 = complex(var_86, var_84)
    var_88 = var_0.datetime(var_85, var_87)
    var_89 = 2000
    var_90 = 2001
    var_91 = range(var_89, var_90)
    var_92 = 2010
    var_93 = 2011
    var_94 = range(var_92, var_93)
    var_95 = var_0.datetime(var_91, var_94)
    var_96 = 2000
    var_97 = 2001
    var_98 = slice(var_96, var_97)
    var_99 = 2010
    var_100 = 2011
    var_101 = slice(var_99, var_100)
    var_102 = var_0.datetime(var_98, var_101)
    var_103 = var_0.datetime(var_96, var_96)
    var_104 = module_1.object()
    var_105 = module_1.object()
    var_106 = var_0.datetime(var_104, var_105)
    var_107 = 2000
    var_108 = lambda : var_107
    var_109 = 2010
    var_110 = lambda : var_109
    var_111 = var_0.datetime(var_108, var_110)
    var_112 = 2000
    var_113 = [var_112]
    var_114 = 2010
    var_115 = [var_114]
    var_116 = var_0.datetime(var_109, var_101)
    var_117 = var_0.datetime(var_112, var_113)
    var_118 = var_0.datetime(var_112, var_113)
    var_119 = var_0.datetime(var_112, var_113)
    var_120 = var_0.datetime(var_112, var_113)
    var_121 = var_0.datetime(var_112, var_113)
    var_122 = var_0.datetime(var_112, var_113)
    var_123 = var_112.value
    var_124 = var_122.value
    var_125 = var_0.datetime(var_123, var_124)
    var_126 = var_0.datetime(var_112, var_123)
    var_127 = var_0.datetime(var_112, var_123)
    var_128 = var_112.value
    var_129 = var_127.value
    var_130 = var_0.datetime(var_128, var_129)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 10
    var_3 = 'days'
    var_4 = {var_3: var_1}
    var_5 = 2
    var_6 = 11
    var_7 = 0
    var_8 = {var_3: var_7}
    var_9 = {}
    var_10 = None
    var_11 = 'hours'
    var_12 = 12
    var_13 = {var_11: var_12}
    var_14 = 'microseconds'
    var_15 = 500000
    var_16 = {var_14: var_15}
    var_17 = 1000000
    var_18 = 'minutes'
    var_19 = 30
    var_20 = {var_18: var_19}
    var_21 = 'seconds'
    var_22 = {var_21: var_19}
    var_23 = 'weeks'
    var_24 = {var_23: var_1}
    var_25 = 15
    var_26 = 8
    var_27 = 22
    var_28 = {var_3: var_1, var_11: var_12}
    var_29 = 3
    var_30 = 5
    var_31 = {var_3: var_1}
    var_32 = 31
    var_33 = 2021
    var_34 = {var_3: var_1}
    var_35 = -1
    var_36 = {var_3: var_35}
    var_37 = {var_3: var_7}
    var_38 = 'invalid'
    var_39 = {var_38: var_1}
    var_40 = 1.5
    var_41 = {var_3: var_40}
    var_42 = '1'
    var_43 = {var_3: var_42}
    var_44 = None
    var_45 = {var_3: var_44}
    var_46 = []
    var_47 = {var_3: var_46}
    var_48 = {}
    var_49 = {var_3: var_48}
    var_50 = ()
    var_51 = {var_3: var_50}



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = None
    var_6 = var_0.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_0.duration(var_7, var_8)
    var_10 = 1.5
    var_11 = 10
    var_12 = var_0.duration(var_10, var_11)
    var_13 = 1
    var_14 = 10.5
    var_15 = var_0.duration(var_13, var_14)
    var_16 = -5
    var_17 = -1
    var_18 = var_0.duration(var_16, var_17)
    var_19 = 0
    var_20 = var_0.duration(var_19, var_19)
    var_21 = 1
    var_22 = 1000
    var_23 = var_0.duration(var_21, var_22)
    var_24 = 'All tests passed!'
    var_25 = print(var_24)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2000
    var_3 = 2010
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)
    var_9 = 2020
    var_10 = 2010
    var_11 = var_0.datetime(var_9, var_10)
    var_12 = '2000'
    var_13 = '2010'
    var_14 = var_0.datetime(var_12, var_13)
    var_15 = -100
    var_16 = -50
    var_17 = var_0.datetime(var_15, var_16)
    var_18 = 1995
    var_19 = var_0.datetime(var_18, var_18)
    var_20 = 1
    var_21 = 9999
    var_22 = var_0.datetime(var_20, var_21)
    var_23 = var_0.datetime()
    var_24 = var_0.datetime(var_20, var_21)
    var_25 = 42
    var_26 = module_0.Datetime(seed=var_25)
    var_27 = module_0.Datetime(seed=var_25)
    var_28 = var_26.datetime()
    var_29 = var_27.datetime()
    var_30 = 123
    var_31 = module_0.Datetime(seed=var_30)
    var_32 = var_31.datetime()



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 10
    var_6 = 5
    var_7 = var_0.duration(var_5, var_6)
    var_8 = 1.5
    var_9 = 10
    var_10 = var_0.duration(var_8, var_9)
    var_11 = 1
    var_12 = 10.5
    var_13 = var_0.duration(var_11, var_12)
    var_14 = None
    var_15 = var_0.duration(duration_unit=var_14)
    var_16 = 'invalid_unit'
    var_17 = var_0.duration(duration_unit=var_16)
    var_18 = 0
    var_19 = var_0.duration(var_18, var_18)
    var_20 = -5
    var_21 = -1
    var_22 = var_0.duration(var_20, var_21)
    var_23 = -5
    var_24 = 5
    var_25 = var_0.duration(var_23, var_24)
    var_26 = 1000
    var_27 = 2000
    var_28 = var_0.duration(var_26, var_27)
    var_29 = var_0.duration(var_2, var_2)
    var_30 = var_0.duration(duration_unit=var_23)
    var_31 = var_0.duration(duration_unit=var_23)
    var_32 = '1'
    var_33 = '10'
    var_34 = var_0.duration(var_32, var_33)
    var_35 = '1'
    var_36 = 10
    var_37 = var_0.duration(var_35, var_36)
    var_38 = 1
    var_39 = '10'
    var_40 = var_0.duration(var_38, var_39)
    var_41 = 1.0
    var_42 = 10.0
    var_43 = var_0.duration(var_41, var_42)
    var_44 = 1.0
    var_45 = 10
    var_46 = var_0.duration(var_44, var_45)
    var_47 = 1
    var_48 = 10.0
    var_49 = var_0.duration(var_47, var_48)
    var_50 = 1
    var_51 = var_0.duration(var_50, var_50)
    var_52 = 10
    var_53 = var_0.duration(var_52, var_52)
    var_54 = var_0.duration(var_18, var_52)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2022
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = 6
    var_8 = 0
    var_9 = 8
    var_10 = 1
    var_11 = 0
    var_12 = None
    var_13 = 1
    var_14 = 1000
    var_15 = 500
    var_16 = 1500



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = 6
    var_8 = 0
    var_9 = 4000
    var_10 = 1000
    var_11 = 2000
    var_12 = 3000
    var_13 = 5000
    var_14 = 1
    var_15 = 0
    var_16 = None
    var_17 = 1
    var_18 = 7
    var_19 = 12
    var_20 = -1
    var_21 = 0



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2010
    var_3 = 2020
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)
    var_9 = 2020
    var_10 = 2010
    var_11 = var_0.datetime(var_9, var_10)
    var_12 = '2020'
    var_13 = 2020
    var_14 = var_0.datetime(var_12, var_13)
    var_15 = -100
    var_16 = -50
    var_17 = var_0.datetime(var_15, var_16)
    var_18 = 3000
    var_19 = 4000
    var_20 = var_0.datetime(var_18, var_19)
    var_21 = 1995
    var_22 = var_0.datetime(var_21, var_21)
    var_23 = var_0.datetime()
    var_24 = 'year'
    var_25 = hasattr(var_23, var_24)
    var_26 = 'month'
    var_27 = hasattr(var_23, var_26)
    var_28 = 'day'
    var_29 = hasattr(var_23, var_28)
    var_30 = 'hour'
    var_31 = hasattr(var_23, var_30)
    var_32 = 'minute'
    var_33 = hasattr(var_23, var_32)
    var_34 = 'second'
    var_35 = hasattr(var_23, var_34)
    var_36 = 'microsecond'
    var_37 = hasattr(var_23, var_36)
    var_38 = 2020
    var_39 = 5
    var_40 = 15
    var_41 = 14
    var_42 = 30
    var_43 = 45
    var_44 = 123456
    var_45 = var_0.datetime(var_38, var_38)
    var_46 = 'All tests passed!'
    var_47 = print(var_46)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2021
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = 11
    var_6 = 0
    var_7 = 12
    var_8 = 13
    var_9 = 15
    var_10 = 1000
    var_11 = 100
    var_12 = 1100
    var_13 = 1
    var_14 = 0
    var_15 = None
    var_16 = 1
    var_17 = 5
    var_18 = 6
    var_19 = 2000
    var_20 = 31
    var_21 = 30
    var_22 = -1
    var_23 = 0
    var_24 = 8
    var_25 = 500
    var_26 = 600
    var_27 = 500000
    var_28 = 3



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'Unit test for method timestamp of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 2021
    var_4 = 'UTC'
    var_5 = var_1.timestamp(var_0)
    var_6 = 'All tests passed for Datetime.timestamp()'
    var_7 = print(var_6)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'Test method timestamp of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = '\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z'
    var_3 = '\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d+'
    var_4 = var_1.timestamp()
    var_5 = 2020
    var_6 = 2021
    var_7 = 'invalid_format'
    var_8 = var_1.timestamp(var_7)
    var_9 = 2022
    var_10 = 2021
    var_11 = var_1.timestamp(var_7)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.timestamp()
    var_2 = len(var_1)
    assert var_2 == 20
    var_3 = len(var_1)
    assert var_3 == 26
    var_4 = 2020
    var_5 = 2021
    var_6 = var_0.timestamp()
    var_7 = 'UTC'
    var_8 = var_0.timestamp()
    var_9 = 'invalid'
    var_10 = var_0.timestamp(var_9)
    var_11 = 2022
    var_12 = 2021
    var_13 = var_0.timestamp()
    var_14 = 'invalid'
    var_15 = var_0.timestamp()
    var_16 = -100
    var_17 = 100
    var_18 = var_0.timestamp()
    var_19 = 10000
    var_20 = 20000
    var_21 = var_0.timestamp()
    var_22 = var_0.timestamp()
    var_23 = 1
    var_24 = 10
    var_25 = var_0.duration(var_23, var_24)
    var_26 = 10
    var_27 = 1
    var_28 = var_0.duration(var_26, var_27)
    var_29 = 1.5
    var_30 = 10.5
    var_31 = var_0.duration(var_29, var_30)
    var_32 = 'invalid'
    var_33 = var_0.duration(duration_unit=var_32)
    var_34 = None
    var_35 = var_0.duration(duration_unit=var_34)
    var_36 = -10
    var_37 = -1
    var_38 = var_0.duration(var_36, var_37)
    var_39 = 0
    var_40 = var_0.duration(var_39, var_39)
    var_41 = 1000
    var_42 = var_0.duration(var_41, var_19)
    var_43 = 5
    var_44 = var_0.duration(var_43, var_43)
    var_45 = 'invalid'
    var_46 = var_0.duration(duration_unit=var_45)
    var_47 = var_0.duration(duration_unit=var_34)
    var_48 = -10
    var_49 = -1
    var_50 = '1'
    var_51 = '10'
    var_52 = var_0.duration(var_50, var_51, var_45)
    var_53 = 10
    var_54 = 1
    var_55 = var_0.duration(var_53, var_54, var_45)
    var_56 = -10
    var_57 = -1
    var_58 = 1.5
    var_59 = 10.5
    var_60 = var_0.duration(var_58, var_59, var_45)
    var_61 = 10
    var_62 = 1
    var_63 = var_0.duration(var_61, var_62, var_45)
    var_64 = var_0.duration



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 1990
    var_3 = 2000
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)
    var_9 = 2020
    var_10 = 2010
    var_11 = var_0.datetime(var_9, var_10)
    var_12 = '1990'
    var_13 = '2000'
    var_14 = var_0.datetime(var_12, var_13)
    var_15 = -1000
    var_16 = -500
    var_17 = var_0.datetime(var_15, var_16)
    var_18 = 1
    var_19 = 9999
    var_20 = var_0.datetime(var_18, var_19)
    var_21 = 2020
    var_22 = var_0.datetime(var_21, var_21)
    var_23 = 'America/New_York'
    var_24 = var_0.datetime(timezone=var_23)
    var_25 = 123
    var_26 = var_0.datetime(timezone=var_25)
    var_27 = ''
    var_28 = var_0.datetime(timezone=var_27)
    var_29 = None
    var_30 = var_0.datetime(timezone=var_29)
    var_31 = 0
    var_32 = 2020
    var_33 = var_0.datetime(var_31, var_32)
    var_34 = 2000
    var_35 = 10000
    var_36 = var_0.datetime(var_34, var_35)
    var_37 = 1990.5
    var_38 = 2000.5
    var_39 = var_0.datetime(var_37, var_38)
    var_40 = '1990'
    var_41 = '2000'
    var_42 = var_0.datetime(var_40, var_41)
    var_43 = -1000
    var_44 = -500
    var_45 = var_0.datetime(var_43, var_44)
    var_46 = 10000
    var_47 = 20000
    var_48 = var_0.datetime(var_46, var_47)
    var_49 = 0
    var_50 = var_0.datetime(var_49, var_49)
    var_51 = -1000
    var_52 = -1000
    var_53 = var_0.datetime(var_51, var_52)
    var_54 = 10000
    var_55 = var_0.datetime(var_54, var_54)
    var_56 = 1990.5
    var_57 = var_0.datetime(var_56, var_56)
    var_58 = '1990'
    var_59 = var_0.datetime(var_58, var_58)
    var_60 = True
    var_61 = var_0.datetime(var_60, var_60)
    var_62 = None
    var_63 = var_0.datetime(var_62, var_62)
    var_64 = 1990
    var_65 = [var_64]
    var_66 = [var_64]
    var_67 = var_0.datetime(var_65, var_66)
    var_68 = 1990
    var_69 = (var_68,)
    var_70 = (var_68,)
    var_71 = var_0.datetime(var_69, var_70)
    var_72 = 'year'
    var_73 = 1990
    var_74 = {var_72: var_73}
    var_75 = {var_72: var_73}
    var_76 = var_0.datetime(var_74, var_75)
    var_77 = 1990
    var_78 = {var_77}
    var_79 = {var_77}
    var_80 = var_0.datetime(var_78, var_79)
    var_81 = 1990
    var_82 = {var_81}
    var_83 = frozenset(var_82)
    var_84 = {var_81}
    var_85 = frozenset(var_84)
    var_86 = var_0.datetime(var_83, var_85)
    var_87 = b'1990'
    var_88 = var_0.datetime(var_87, var_87)
    var_89 = b'1990'
    var_90 = bytearray(var_89)
    var_91 = bytearray(var_89)
    var_92 = var_0.datetime(var_90, var_91)
    var_93 = b'1990'
    var_94 = memoryview(var_93)
    var_95 = memoryview(var_93)
    var_96 = var_0.datetime(var_94, var_95)
    var_97 = 1990
    var_98 = 0
    var_99 = complex(var_97, var_98)
    var_100 = complex(var_97, var_98)
    var_101 = var_0.datetime(var_99, var_100)
    var_102 = 1990
    var_103 = 1991
    var_104 = range(var_102, var_103)
    var_105 = range(var_102, var_103)
    var_106 = var_0.datetime(var_104, var_105)
    var_107 = 1990
    var_108 = 1991
    var_109 = slice(var_107, var_108)
    var_110 = slice(var_107, var_108)
    var_111 = var_0.datetime(var_109, var_110)
    var_112 = var_0.datetime(var_107, var_107)
    var_113 = module_1.object()
    var_114 = module_1.object()
    var_115 = var_0.datetime(var_113, var_114)
    var_116 = lambda x: x
    var_117 = lambda x: x
    var_118 = var_0.datetime(var_116, var_117)
    var_119 = 10
    var_120 = range(var_119)
    var_121 = range(var_119)
    var_122 = var_0.datetime(var_118, var_111)
    var_123 = var_0.datetime(var_119, var_120)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.timestamp()
    var_2 = len(var_1)
    assert var_2 == 20
    var_3 = len(var_1)
    assert var_3 == 26
    var_4 = 2020
    var_5 = 2021
    var_6 = var_0.timestamp()
    var_7 = 'UTC'
    var_8 = var_0.timestamp()
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.timestamp()
    var_11 = 2022
    var_12 = 2021
    var_13 = var_0.timestamp()
    var_14 = '2020'
    var_15 = '2021'
    var_16 = var_0.timestamp()
    var_17 = 'invalid'
    var_18 = var_0.duration(duration_unit=var_17)
    var_19 = -1
    var_20 = var_0.duration(var_19)
    var_21 = -1
    var_22 = var_0.duration(max_duration=var_21)
    var_23 = 10
    var_24 = 5
    var_25 = var_0.duration(var_23, var_24)
    var_26 = 1.5
    var_27 = var_0.duration(var_26)
    var_28 = 10.5
    var_29 = var_0.duration(max_duration=var_28)
    var_30 = None
    var_31 = var_0.duration(duration_unit=var_30)
    var_32 = 5
    var_33 = 15
    var_34 = var_0.duration(var_32, var_33)
    var_35 = 2
    var_36 = 8
    var_37 = 0
    var_38 = var_0.duration(var_37, var_32)
    var_39 = var_0.duration(var_37, var_37)
    var_40 = 1000
    var_41 = 2000
    var_42 = var_0.duration(var_40, var_41)
    var_43 = -10
    var_44 = -5
    var_45 = var_0.duration(var_43, var_44)
    var_46 = 123
    var_47 = var_0.duration(duration_unit=var_46)
    var_48 = 'invalid_unit'
    var_49 = var_0.duration(duration_unit=var_48)
    var_50 = ''
    var_51 = var_0.duration(duration_unit=var_50)
    var_52 = ' '
    var_53 = var_0.duration(duration_unit=var_52)
    var_54 = '@#$%'
    var_55 = var_0.duration(duration_unit=var_54)
    var_56 = '😀'
    var_57 = var_0.duration(duration_unit=var_56)
    var_58 = 1000000
    var_59 = 2000000
    var_60 = var_0.duration(var_58, var_59)
    var_61 = 7
    var_62 = var_0.duration(var_61, var_61)
    var_63 = '1'
    var_64 = '10'
    var_65 = var_0.duration(var_63, var_64)
    var_66 = 1.5
    var_67 = 10.5
    var_68 = var_0.duration(var_66, var_67)
    var_69 = True
    var_70 = False
    var_71 = var_0.duration(var_69, var_70)
    var_72 = None
    var_73 = var_0.duration(var_72, var_72)
    var_74 = 1
    var_75 = [var_74]
    var_76 = 10
    var_77 = [var_76]
    var_78 = var_0.duration(var_75, var_77)
    var_79 = 'min'
    var_80 = 1
    var_81 = {var_79: var_80}
    var_82 = 'max'
    var_83 = 10
    var_84 = {var_82: var_83}
    var_85 = var_0.duration(var_81, var_84)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 6
    var_6 = 0
    var_7 = 1
    var_8 = 0
    var_9 = 3
    var_10 = 4
    var_11 = -1
    var_12 = 12
    var_13 = 31
    var_14 = 30
    var_15 = 10
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 10
    var_6 = var_0.duration(var_5, var_5)
    var_7 = 15
    var_8 = 5
    var_9 = var_0.duration(var_7, var_8)
    var_10 = 1.5
    var_11 = 10
    var_12 = var_0.duration(var_10, var_11)
    var_13 = 1
    var_14 = 10.5
    var_15 = var_0.duration(var_13, var_14)
    var_16 = None
    var_17 = var_0.duration(duration_unit=var_16)
    var_18 = 'invalid'
    var_19 = var_0.duration(duration_unit=var_18)
    var_20 = -5
    var_21 = -1
    var_22 = var_0.duration(var_20, var_21)
    var_23 = 0
    var_24 = var_0.duration(var_23, var_23)
    var_25 = 1000
    var_26 = 2000
    var_27 = var_0.duration(var_25, var_26)
    var_28 = -10
    var_29 = -10
    var_30 = var_0.duration(var_28, var_29)
    var_31 = -5
    var_32 = var_0.duration(var_31, var_15)
    var_33 = 5
    var_34 = -5
    var_35 = var_0.duration(var_33, var_34)
    var_36 = var_0.duration(var_23, var_23, var_16)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.timestamp()
    var_2 = 2020
    var_3 = 2021
    var_4 = 'invalid'
    var_5 = var_0.timestamp(var_4)
    var_6 = 'UTC'
    var_7 = len(var_1)
    assert var_7 == 20
    var_8 = len(var_1)
    assert var_8 == 26
    var_9 = -100
    var_10 = -50
    var_11 = 1900
    var_12 = 2100
    var_13 = 'Invalid/Timezone'
    var_14 = var_0.timestamp(var_5)
    var_15 = module_0.Datetime()
    var_16 = '^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$'
    var_17 = '^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d+$'
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.timestamp()
    var_2 = len(var_1)
    assert var_2 == 20
    var_3 = 2020
    var_4 = 2021
    var_5 = var_0.timestamp()
    var_6 = 'UTC'
    var_7 = var_0.timestamp()
    var_8 = 'invalid'
    var_9 = var_0.timestamp(var_8)
    var_10 = 'invalid'
    var_11 = var_0.timestamp()
    var_12 = 'Invalid/Timezone'
    var_13 = var_0.timestamp()



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 'Unit test for method timestamp of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.timestamp()



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 2022
    var_1 = 1
    var_2 = 10
    var_3 = 2
    var_4 = 11
    var_5 = 1
    var_6 = -1



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = module_0.Datetime()
    var_3 = 'UTC'
    var_4 = var_2.datetime(timezone=var_3)
    var_5 = module_0.Datetime()
    var_6 = 2020
    var_7 = 2022
    var_8 = var_5.datetime(var_6, var_7)
    var_9 = module_0.Datetime()
    var_10 = 'UTC'
    var_11 = var_9.datetime(timezone=var_10)
    var_12 = module_0.Datetime()
    var_13 = var_12.datetime()
    var_14 = module_0.Datetime()
    var_15 = var_14.datetime(var_6, var_7, var_3)
    var_16 = module_0.Datetime()
    var_17 = var_16.datetime(timezone=var_3)
    var_18 = module_0.Datetime()
    var_19 = var_18.datetime(var_6, var_7, var_3)
    var_20 = module_0.Datetime()
    var_21 = None
    var_22 = var_20.datetime(var_6, var_7, var_21)
    var_23 = module_0.Datetime()
    var_24 = ''
    var_25 = var_23.datetime(var_6, var_7, var_24)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.timestamp()
    var_2 = len(var_1)
    assert var_2 == 20
    var_3 = 2020
    var_4 = 2021
    var_5 = var_0.timestamp()
    var_6 = 'UTC'
    var_7 = var_0.timestamp()
    var_8 = 'invalid'
    var_9 = var_0.timestamp(var_8)
    var_10 = 2022
    var_11 = 2021
    var_12 = var_0.timestamp()
    var_13 = 'Invalid/Timezone'
    var_14 = var_0.timestamp()
    var_15 = var_0.timestamp()
    var_16 = var_0.timestamp()
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 10
    var_4 = None
    var_5 = var_0.duration(duration_unit=var_4)
    var_6 = 10
    var_7 = 5
    var_8 = var_0.duration(var_6, var_7)
    var_9 = 1.5
    var_10 = 10
    var_11 = var_0.duration(var_9, var_10)
    var_12 = 1
    var_13 = 10.5
    var_14 = var_0.duration(var_12, var_13)
    var_15 = -5
    var_16 = -1
    var_17 = 0
    var_18 = 1000
    var_19 = 2000



# Parsed testcases at query #19
#--------------------------




