####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Different Name'
    var_7 = 'Self Parent'
    var_8 = '9999'
    var_9 = '99999'
    var_10 = 'Ghost Account'



# Parsed testcases at query #2
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'



# Parsed testcases at query #3
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of an object conforming to the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it using a mock or a concrete implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests the protocol behavior using a concrete implementation.\n    '



# Parsed testcases at query #4
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    callable object that adheres to the protocol.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '



# Parsed testcases at query #5
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __iter__ method of the COA class to ensure it correctly\n    iterates over the default root accounts and custom added accounts.\n    '
    var_1 = module_0.COA()
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = '1'
    var_5 = '5'
    var_6 = '1000'
    var_7 = '1001'
    var_8 = 'Liquidity'
    var_9 = 'Bank Account'
    var_10 = list(var_1)
    var_11 = len(var_10)
    assert var_11 == 7
    var_12 = [code for (code, acct) in var_10]

def test_case_0():
    var_0 = '\n    Tests __iter__ with a custom rootspec provided during initialization.\n    '
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = '5'



# Parsed testcases at query #6
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '1002'
    var_12 = 'Other'
    var_13 = '1002'



# Parsed testcases at query #7
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a valid implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests the __call__ method using a concrete class implementation.\n    '
    var_1 = '10'



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.\n    Since ReadChartOfAccounts is a Protocol (runtime_checkable), we test \n    a concrete implementation that follows the signature.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests that a class implementing the protocol is recognized by runtime_checkable.\n    '



# Parsed testcases at query #10
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '9999'
    var_7 = '99999'
    var_8 = 'Non-existent'
    var_9 = 'Self Parent'
    var_10 = 'Different Name'
    var_11 = '2'



# Parsed testcases at query #11
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a functional implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a real functional implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '10'



# Parsed testcases at query #12
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the nodify method of the COA class to ensure it correctly transforms\n    the flat account structure into a tree of COA.Node objects.\n    '
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = '1001'
    var_5 = 'Liquidity'
    var_6 = 'Bank Account'
    var_7 = var_1.structure
    var_8 = list(var_7)
    var_9 = None
    var_10 = 0
    var_11 = '2'
    var_12 = len(var_8)
    assert var_12 == 5



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ protocol implementation for ReadChartOfAccounts.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation\n    to verify the signature and behavior.\n    '



# Parsed testcases at query #14
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #15
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '



# Parsed testcases at query #16
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the nodify method of the COA class to ensure it correctly \n    transforms the flat account structure into a hierarchical tree of Nodes.\n    '
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = 0
    var_8 = '2'



# Parsed testcases at query #17
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.\n    '
    var_1 = module_0.COA()



# Parsed testcases at query #18
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'
    var_12 = '2000'
    var_13 = 'Liabilities Branch'
    var_14 = '2000'



# Parsed testcases at query #19
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = "\n    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.\n    Since ReadChartOfAccounts is a Protocol, we test it using a Mock \n    that conforms to the protocol's signature.\n    "
    var_1 = module_0.COA()



# Parsed testcases at query #20
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    mock object or a concrete implementation that follows the protocol.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '1'
    var_2 = '5'



# Parsed testcases at query #21
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2000'
    var_12 = 'Other Parent'
    var_13 = '2000'



# Parsed testcases at query #22
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99991'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'



# Parsed testcases at query #23
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'
    var_12 = 'Different Parent'



# Parsed testcases at query #24
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it by verifying that \n    a compatible callable returns a COA instance.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '



# Parsed testcases at query #25
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    callable object that matches its signature.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    callable object (like a function or a mock) that adheres to the protocol.\n    '



# Parsed testcases at query #27
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the nodify method of the COA class to ensure it correctly \n    converts the flat account structure into a tree of COA.Node objects.\n    '
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = var_1.structure
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = 0
    var_11 = '2'



# Parsed testcases at query #28
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a valid implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '1'
    var_2 = '5'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of an object implementing the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it using a callable mock \n    that returns a COA instance.\n    '

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '1'
    var_2 = '5'
    var_3 = '999'



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'
    var_12 = '1000'
    var_13 = 'Liquidity'



# Parsed testcases at query #4
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation\n    (a callable) that adheres to the protocol.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests the behavior of a real function implementation of the protocol.\n    '



# Parsed testcases at query #5
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of an object implementing the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test a callable that returns a COA instance.\n    '
    var_1 = module_0.COA()



# Parsed testcases at query #6
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the nodify method of the COA class to ensure it correctly \n    transforms accounts into a tree structure of COA.Node instances.\n    '
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = 0
    var_8 = '2'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it by creating \n    a compatible callable object.\n    '



# Parsed testcases at query #8
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test an implementation \n    to verify the call signature and return type.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of ReadChartOfAccounts to ensure \n    it correctly returns a COA object.\n    '



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ protocol of ReadChartOfAccounts.\n    Since ReadChartOfAccounts is a Protocol, we test it by verifying \n    that a callable object conforming to the protocol behaves as expected.\n    '



# Parsed testcases at query #10
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'



# Parsed testcases at query #11
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a functional implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests that the __call__ method can return a COA with custom root specifications.\n    '
    var_1 = '10'
    var_2 = 'Custom Assets'



# Parsed testcases at query #12
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a valid implementation.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '10'



# Parsed testcases at query #13
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99991'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'



# Parsed testcases at query #14
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test an implementation \n    of it to verify the callability and return type.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '1'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ protocol of ReadChartOfAccounts.\n    Since ReadChartOfAccounts is a Protocol, we test it by verifying \n    that a callable object conforming to the protocol behaves as expected.\n    '



# Parsed testcases at query #16
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    callable object that follows the protocol.\n    '

def test_case_0():
    var_0 = '\n    Tests a concrete implementation of the ReadChartOfAccounts protocol\n    to ensure it returns a valid COA instance.\n    '



# Parsed testcases at query #18
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ protocol implementation of ReadChartOfAccounts.\n    Since ReadChartOfAccounts is a Protocol, we test it using a mock \n    that adheres to the protocol signature.\n    '
    var_1 = module_0.COA()



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    compatible callable object.\n    '



# Parsed testcases at query #20
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '2'



# Parsed testcases at query #21
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the nodify method of the COA class to ensure it correctly\n    transforms a flat account structure into a hierarchical tree of COA.Node instances.\n    '
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = var_1.toplevel
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = var_1.structure
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 5
    var_13 = 0
    var_14 = '2'



# Parsed testcases at query #22
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.\n    '
    var_1 = module_0.COA()



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a functional implementation.\n    '



# Parsed testcases at query #24
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99999'
    var_9 = 'Ghost Account'
    var_10 = var_0.add(var_2, var_4, var_9)
    var_11 = 'Different Name'
    var_12 = '2'
    var_13 = 'Different Parent'



# Parsed testcases at query #25
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation\n    (a callable) that adheres to the protocol.\n    '
    var_1 = module_0.COA()

def test_case_0():
    var_0 = '\n    Tests a concrete function implementation of the ReadChartOfAccounts protocol.\n    '
    var_1 = '10'



# Parsed testcases at query #26
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Liquidity'
    var_7 = 'Self Parent'
    var_8 = '9999'
    var_9 = '99999'
    var_10 = 'Ghost Account'
    var_11 = 'Different Name'
    var_12 = '2'
    var_13 = 'Liquidity'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of the ReadChartOfAccounts protocol.\n    Since ReadChartOfAccounts is a Protocol, we test it using a \n    compatible callable object.\n    '



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __call__ method of a ReadChartOfAccounts implementation.\n    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.\n    '

def test_case_0():
    var_0 = '\n    Tests the __call__ method with a functional implementation \n    to ensure it correctly returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '5'



# Parsed testcases at query #29
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Self Parent'
    var_7 = '9999'
    var_8 = '99991'
    var_9 = 'Ghost Account'
    var_10 = 'Different Name'
    var_11 = '2'



# Parsed testcases at query #30
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 0



# Parsed testcases at query #31
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '10001'
    var_5 = 'Cash'
    var_6 = 'Different Name'
    var_7 = '2'
    var_8 = 'Self Parent'
    var_9 = '9999'
    var_10 = '99999'
    var_11 = 'Ghost Account'



