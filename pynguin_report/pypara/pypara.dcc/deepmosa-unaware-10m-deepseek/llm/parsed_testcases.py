####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.1'
    var_5 = var_0.find(var_1)
    var_6 = '30/360'
    var_7 = '30U/360'
    var_8 = 'Bond Basis'
    var_9 = {var_7, var_8}
    var_10 = set()
    var_11 = '0.2'
    var_12 = var_0.find(var_7)
    var_13 = var_0.find(var_8)
    var_14 = '  act/360  '
    var_15 = var_0.find(var_14)
    var_16 = 'AcT/360'
    var_17 = var_0.find(var_16)
    var_18 = 'NonExistent'
    var_19 = var_0.find(var_18)
    assert var_19 is None
    var_20 = '  30/360  '
    var_21 = var_0.find(var_20)
    var_22 = '  bond basis  '
    var_23 = var_0.find(var_22)
    var_24 = 'BOND BASIS'
    var_25 = var_0.find(var_24)
    var_26 = 'act/360'
    var_27 = 'Act/Act'
    var_28 = 'Actual/Actual'
    var_29 = {var_28}
    var_30 = set()
    var_31 = '0.3'
    var_32 = '  ACT/ACT  '
    var_33 = var_0.find(var_32)
    var_34 = 'actual/actual'
    var_35 = var_0.find(var_34)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'
    var_18 = 2021
    var_19 = 6
    var_20 = 365
    var_21 = 2020
    var_22 = 366
    var_23 = 0



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = 3
    var_20 = '2'
    var_21 = '12'
    var_22 = 15
    var_23 = '45'
    var_24 = '360'
    var_25 = '60'
    var_26 = 2022
    var_27 = '30'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = 3



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'
    var_18 = 2023
    var_19 = '1'
    var_20 = '365'
    var_21 = 2024
    var_22 = '366'
    var_23 = '28'
    var_24 = '31'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '100'
    var_4 = 'USD'
    var_5 = '0.1'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 30
    var_10 = 12
    var_11 = 31
    var_12 = '5'
    var_13 = '0'
    var_14 = '200'
    var_15 = 'EUR'
    var_16 = '0.05'
    var_17 = 2022
    var_18 = 2024
    var_19 = None
    var_20 = '2'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33055555555556'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08333333333333'
    var_21 = 3
    var_22 = 15
    var_23 = 20
    var_24 = '0.01388888888889'
    var_25 = 2022



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'
    var_7 = 1
    var_8 = 7
    var_9 = 2021
    var_10 = '0.24863387978142076'
    var_11 = 14
    var_12 = 12
    var_13 = 31
    var_14 = 0
    var_15 = 28
    var_16 = 8
    var_17 = '0.010810810810810811'
    var_18 = 6
    var_19 = 30
    var_20 = '0.4972677595628415'
    var_21 = 4
    var_22 = '0.08516483516483516'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 2022
    var_5 = '364'
    var_6 = '365'
    var_7 = 2020
    var_8 = '366'
    var_9 = 2019
    var_10 = 2
    var_11 = '31'
    var_12 = 28
    var_13 = 29
    var_14 = 3
    var_15 = '1'
    var_16 = 2007
    var_17 = 2008
    var_18 = '0.16939890710383'
    var_19 = 14
    var_20 = '0.17213114754098'
    var_21 = 6
    var_22 = 30
    var_23 = '2'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 31
    var_7 = '0.5'
    var_8 = 2022
    var_9 = 12
    var_10 = 2
    var_11 = '2'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'TEST'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 3
    var_6 = 10
    var_7 = 9
    var_8 = '2'
    var_9 = '9'
    var_10 = '1'
    var_11 = '0'
    var_12 = '8'
    var_13 = 2
    var_14 = 5
    var_15 = 31
    var_16 = 30
    var_17 = '4'
    var_18 = '30'
    var_19 = '3'
    var_20 = 5
    var_21 = -2
    var_22 = '-2'
    var_23 = '5'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08333333333333'
    var_21 = 3
    var_22 = 2020
    var_23 = 15
    var_24 = 6
    var_25 = '2.5'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = var_2 - var_12
    var_20 = var_4 - var_14
    var_21 = var_12 * var_20
    var_22 = var_19 + var_21
    var_23 = 360
    var_24 = var_18 - var_18
    var_25 = var_23 * var_24
    var_26 = var_22 + var_25
    var_27 = var_12 - var_12
    var_28 = var_4 - var_14
    var_29 = var_12 * var_28
    var_30 = var_27 + var_29
    var_31 = var_18 - var_18
    var_32 = var_23 * var_31
    var_33 = var_30 + var_32
    var_34 = 3
    var_35 = var_12 - var_12
    var_36 = var_34 - var_14
    var_37 = var_12 * var_36
    var_38 = var_35 + var_37
    var_39 = var_18 - var_18
    var_40 = var_23 * var_39
    var_41 = var_38 + var_40
    var_42 = 15
    var_43 = var_42 - var_42
    var_44 = var_4 - var_14
    var_45 = var_12 * var_44
    var_46 = var_43 + var_45
    var_47 = var_18 - var_18
    var_48 = var_23 * var_47
    var_49 = var_46 + var_48



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.08333333333333'
    var_20 = 15
    var_21 = 4
    var_22 = '0.25'
    var_23 = 2022
    var_24 = 3



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = 1
    var_7 = '0.5245901639'
    var_8 = '0.2622950820'
    var_9 = '0'
    var_10 = '1.0000000000'
    var_11 = 15
    var_12 = 12
    var_13 = '0.4516129032'
    var_14 = 28
    var_15 = 31
    var_16 = 4
    var_17 = '0.0322580645'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 31
    var_7 = '0.5'
    var_8 = 2022
    var_9 = 12
    var_10 = 2
    var_11 = '2'
    var_12 = 'ACT/365'
    var_13 = set()
    var_14 = set()
    var_15 = 30
    var_16 = 365
    var_17 = 'NegativeDCC'
    var_18 = set()
    var_19 = set()
    var_20 = '-0.1'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16942884946478'
    var_6 = 14
    var_7 = 29
    var_8 = '0.17216108990194'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08243131970956'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32625945055768'
    var_18 = 2020
    var_19 = '0'
    var_20 = 2021
    var_21 = '1'
    var_22 = '365'
    var_23 = '366'
    var_24 = 2019
    var_25 = '2'
    var_26 = 2022



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08611111111111'
    var_21 = 3
    var_22 = 15
    var_23 = '0'
    var_24 = 4
    var_25 = '0.08333333333333'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 2021
    var_18 = '364'
    var_19 = '365'
    var_20 = 2020



# Parsed testcases at query #20
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.1'
    var_6 = '30/360'
    var_7 = 'Bond Basis'
    var_8 = {var_7}
    var_9 = set()
    var_10 = '0.2'
    var_11 = var_0.find(var_1)
    var_12 = var_0.find(var_2)
    var_13 = var_0.find(var_6)
    var_14 = var_0.find(var_7)
    var_15 = '  act/360  '
    var_16 = var_0.find(var_15)
    var_17 = '  actual/360  '
    var_18 = var_0.find(var_17)
    var_19 = 'act/360'
    var_20 = var_0.find(var_19)
    var_21 = 'NonExistent'
    var_22 = var_0.find(var_21)
    assert var_22 is None
    var_23 = ''
    var_24 = var_0.find(var_23)
    assert var_24 is None
    var_25 = var_0.registry
    var_26 = len(var_25)
    assert var_26 == 2



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33055555555556'
    var_18 = 2020
    var_19 = 3
    var_20 = '0.08333333333333'
    var_21 = '0.08055555555556'
    var_22 = 2023
    var_23 = 15
    var_24 = 6
    var_25 = '0.25'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33055555555556'
    var_18 = 2023
    var_19 = 3
    var_20 = '0.08333333333333'
    var_21 = '0.07777777777778'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.08333333333333'
    var_20 = '0.07777777777778'
    var_21 = 15
    var_22 = '0.04166666666667'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08333333333333'
    var_21 = 3
    var_22 = 15
    var_23 = 20
    var_24 = '0.01388888888889'
    var_25 = 2022



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = 3
    var_20 = '0.1666666667'
    var_21 = 15
    var_22 = 4
    var_23 = '0.25'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 2023
    var_18 = '0'
    var_19 = 365
    var_20 = 3
    var_21 = '1'
    var_22 = '364'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '100'
    var_4 = 'USD'
    var_5 = '0.1'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 30
    var_10 = 12
    var_11 = 31
    var_12 = '5'
    var_13 = '0'
    var_14 = '200'
    var_15 = 'EUR'
    var_16 = '10'
    var_17 = 'TestDCCHalf'
    var_18 = set()
    var_19 = set()
    var_20 = '2'
    var_21 = 2024



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16666666666667'
    var_6 = 14
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '30'
    var_20 = '360'
    var_21 = 15
    var_22 = '15'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33055555555556'
    var_18 = 2020
    var_19 = 3
    var_20 = '0.08333333333333'
    var_21 = '0.08055555555556'
    var_22 = 2021
    var_23 = 15
    var_24 = 6
    var_25 = '0.25'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 2023
    var_18 = 6
    var_19 = '0.49589041095890'
    var_20 = '2'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 31
    var_7 = '0.5'
    var_8 = 2022
    var_9 = 12
    var_10 = 2
    var_11 = '2'
    var_12 = 'Actual/360'
    var_13 = set()
    var_14 = set()
    var_15 = 16
    var_16 = '15'
    var_17 = '360'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'
    var_18 = 2023
    var_19 = '0'
    var_20 = 365
    var_21 = 2020
    var_22 = 366
    var_23 = 3
    var_24 = 6
    var_25 = 4



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'
    var_7 = 1
    var_8 = 4
    var_9 = 7
    var_10 = 12
    var_11 = 31
    var_12 = 0
    var_13 = 28
    var_14 = 15



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = 'USD'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 30
    var_10 = 12
    var_11 = 31
    var_12 = '0.5'
    var_13 = '0'
    var_14 = '500'
    var_15 = 'EUR'
    var_16 = 'ZeroDCC'
    var_17 = set()
    var_18 = set()
    var_19 = 2022
    var_20 = 2024



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08333333333333'
    var_21 = 3
    var_22 = 15
    var_23 = 25
    var_24 = '0.02777777777778'
    var_25 = 2022



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08333333333333'
    var_21 = 3
    var_22 = 15
    var_23 = 25
    var_24 = '0.02777777777778'
    var_25 = 2022



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 2023
    var_18 = '0'
    var_19 = 365
    var_20 = 2020
    var_21 = 3
    var_22 = '1'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = 1
    var_7 = '0.5245901639'
    var_8 = 4
    var_9 = 7
    var_10 = '0.25'
    var_11 = '0'
    var_12 = '0.5'
    var_13 = 28
    var_14 = 15
    var_15 = '0.03125'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = '0.07777777777778'
    var_20 = '0.08611111111111'
    var_21 = 3



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '100'
    var_4 = 'USD'
    var_5 = '0.1'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 30
    var_10 = 12
    var_11 = 31
    var_12 = '5'
    var_13 = '200'
    var_14 = 'EUR'
    var_15 = '10'
    var_16 = '0'
    var_17 = '2'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16666666666667'
    var_6 = 14
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33055555555556'
    var_18 = 2023
    var_19 = '28'
    var_20 = '360'
    var_21 = 3
    var_22 = '60'
    var_23 = 2022
    var_24 = 15
    var_25 = '90'
    var_26 = '2'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 2021
    var_18 = '364'
    var_19 = '365'
    var_20 = 2020
    var_21 = 3
    var_22 = '1'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'
    var_18 = 2023
    var_19 = 360
    var_20 = 3
    var_21 = 60
    var_22 = 15
    var_23 = 25
    var_24 = 2022



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'TEST/365'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = 12
    var_7 = 31
    var_8 = 365
    var_9 = 10
    var_10 = 0
    var_11 = 2023
    var_12 = 1
    var_13 = 'ACT/ACT'
    var_14 = set()
    var_15 = set()
    var_16 = 15
    var_17 = var_8 * var_5
    var_18 = 'TEST/Leap'
    var_19 = set()
    var_20 = set()
    var_21 = 2024
    var_22 = 28
    var_23 = 29
    var_24 = 366
    var_25 = 9



