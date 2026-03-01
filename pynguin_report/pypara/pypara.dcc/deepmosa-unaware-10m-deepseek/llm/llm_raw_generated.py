####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    registry = DCCRegistryMachinery()
    
    # Test finding by exact name
    dcc1 = DCC("ACT/360", set(), set(), lambda s, a, e, f: Decimal("0.1"))
    registry.register(dcc1)
    assert registry.find("ACT/360") == dcc1
    
    # Test finding by alternative name
    dcc2 = DCC("30/360", {"30U/360", "Bond Basis"}, set(), lambda s, a, e, f: Decimal("0.2"))
    registry.register(dcc2)
    assert registry.find("30U/360") == dcc2
    assert registry.find("Bond Basis") == dcc2
    
    # Test case-insensitive and whitespace handling
    assert registry.find("  act/360  ") == dcc1
    assert registry.find("AcT/360") == dcc1
    
    # Test non-existent name returns None
    assert registry.find("NonExistent") is None
    
    # Test that original name still works with whitespace
    assert registry.find("  30/360  ") == dcc2
    
    # Test that alternative name with case variation works
    assert registry.find("  bond basis  ") == dcc2
    assert registry.find("BOND BASIS") == dcc2
    
    # Test that strict find works internally
    assert registry._find_strict("ACT/360") == dcc1
    assert registry._find_strict("act/360") is None  # Case sensitive
    
    # Test that find handles mixed case with whitespace
    dcc3 = DCC("Act/Act", {"Actual/Actual"}, set(), lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc3)
    assert registry.find("  ACT/ACT  ") == dcc3
    assert registry.find("actual/actual") == dcc3


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in period (2008 is leap year)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Period spanning multiple years with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32513661202186')
    assert round(result4, 14) == expected4

    # Test case 5: Period entirely within non-leap year
    start5 = datetime.date(2021, 1, 1)
    asof5 = datetime.date(2021, 6, 30)
    result5 = dcfc_act_365_a(start=start5, asof=asof5, end=asof5)
    actual_days = Decimal((asof5 - start5).days)
    expected5 = actual_days / Decimal(365)
    assert result5 == expected5

    # Test case 6: Period entirely within leap year
    start6 = datetime.date(2020, 1, 1)
    asof6 = datetime.date(2020, 6, 30)
    result6 = dcfc_act_365_a(start=start6, asof=asof6, end=asof6)
    actual_days = Decimal((asof6 - start6).days)
    expected6 = actual_days / Decimal(366)
    assert result6 == expected6

    # Test case 7: Single day calculation
    start7 = datetime.date(2020, 2, 28)
    asof7 = datetime.date(2020, 2, 29)
    result7 = dcfc_act_365_a(start=start7, asof=asof7, end=asof7)
    expected7 = Decimal(1) / Decimal(366)
    assert result7 == expected7

    # Test case 8: Zero day period
    start8 = datetime.date(2020, 1, 1)
    asof8 = datetime.date(2020, 1, 1)
    result8 = dcfc_act_365_a(start=start8, asof=asof8, end=asof8)
    assert result8 == Decimal(0)


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Basic example from docstring
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_30_360_us(start=start1, asof=asof1, end=end1)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_30_360_us(start=start2, asof=asof2, end=end2)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_30_360_us(start=start3, asof=asof3, end=end3)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Another longer period from docstring
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_30_360_us(start=start4, asof=asof4, end=end4)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Additional test: Both dates are last day of month
    start5 = datetime.date(2023, 1, 31)
    asof5 = datetime.date(2023, 3, 31)
    end5 = asof5
    result5 = dcfc_30_360_us(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('2') / Decimal('12')  # 2 months = 60/360 = 1/6
    assert round(result5, 14) == round(expected5, 14)
    
    # Additional test: Start is last day of month, asof is not
    start6 = datetime.date(2023, 1, 31)
    asof6 = datetime.date(2023, 3, 15)
    end6 = asof6
    result6 = dcfc_30_360_us(start=start6, asof=asof6, end=end6)
    # D1 becomes 30, D2 stays 15
    # Days = (15-30) + 30*(3-1) + 360*(2023-2023) = -15 + 60 = 45
    expected6 = Decimal('45') / Decimal('360')
    assert round(result6, 14) == round(expected6, 14)
    
    # Additional test: asof day is 31, start day is 30
    start7 = datetime.date(2023, 1, 30)
    asof7 = datetime.date(2023, 3, 31)
    end7 = asof7
    result7 = dcfc_30_360_us(start=start7, asof=asof7, end=end7)
    # D1=30, D2 becomes 30 (since d1=30 and d2=31)
    # Days = (30-30) + 30*(3-1) + 360*(0) = 60
    expected7 = Decimal('60') / Decimal('360')
    assert round(result7, 14) == round(expected7, 14)
    
    # Additional test: asof day is 31, start day is 31
    start8 = datetime.date(2023, 1, 31)
    asof8 = datetime.date(2023, 3, 31)
    end8 = asof8
    result8 = dcfc_30_360_us(start=start8, asof=asof8, end=end8)
    # D1 becomes 30, D2 becomes 30 (since d1=31 becomes 30, and both are last day of month)
    # Days = (30-30) + 30*(3-1) + 360*(0) = 60
    expected8 = Decimal('60') / Decimal('360')
    assert round(result8, 14) == round(expected8, 14)
    
    # Test with different year
    start9 = datetime.date(2022, 12, 31)
    asof9 = datetime.date(2023, 1, 31)
    end9 = asof9
    result9 = dcfc_30_360_us(start=start9, asof=asof9, end=end9)
    # D1=30 (last day of month), D2=30 (both last day of month)
    # Days = (30-30) + 30*(1-12) + 360*(2023-2022) = 0 + 30*(-11) + 360 = -330 + 360 = 30
    expected9 = Decimal('30') / Decimal('360')
    assert round(result9, 14) == round(expected9, 14)


# LLM-generated content at query #4
#--------------------------

```python
def test_dcfc_30_e_plus_360():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

    test_start, test_asof = datetime.date(2023, 1, 31), datetime.date(2023, 2, 28)
    assert round(dcfc_30_e_plus_360(start=test_start, asof=test_asof, end=test_asof), 14) == Decimal('0.07777777777778')

    test_start, test_asof = datetime.date(2023, 1, 31), datetime.date(2023, 3, 31)
    assert round(dcfc_30_e_plus_360(start=test_start, asof=test_asof, end=test_asof), 14) == Decimal('0.16666666666667')


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Period without leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_act_365_a(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Period with leap day (2008 is leap year)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_act_365_a(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_act_365_a(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Period crossing leap year boundary
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_act_365_a(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1.32513661202186')
    assert round(result4, 14) == expected4

    # Test case 5: Single day period in non-leap year
    start5 = datetime.date(2023, 1, 1)
    asof5 = datetime.date(2023, 1, 1)
    end5 = asof5
    result5 = dcfc_act_365_a(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('1') / Decimal('365')
    assert result5 == expected5

    # Test case 6: Single day period in leap year
    start6 = datetime.date(2024, 1, 1)
    asof6 = datetime.date(2024, 1, 1)
    end6 = asof6
    result6 = dcfc_act_365_a(start=start6, asof=asof6, end=end6)
    expected6 = Decimal('1') / Decimal('366')
    assert result6 == expected6

    # Test case 7: Period entirely within leap year
    start7 = datetime.date(2024, 2, 1)
    asof7 = datetime.date(2024, 2, 29)
    end7 = asof7
    result7 = dcfc_act_365_a(start=start7, asof=asof7, end=end7)
    expected7 = Decimal('28') / Decimal('366')
    assert result7 == expected7

    # Test case 8: Period spanning multiple years with leap day
    start8 = datetime.date(2023, 12, 31)
    asof8 = datetime.date(2024, 1, 31)
    end8 = asof8
    result8 = dcfc_act_365_a(start=start8, asof=asof8, end=end8)
    expected8 = Decimal('31') / Decimal('366')
    assert result8 == expected8


# LLM-generated content at query #6
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC with a dummy calculation method
    def dummy_calculate_fraction(start, asof, end, freq):
        # Always return 0.5 for testing
        return Decimal("0.5")
    
    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    # Test basic interest calculation
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.1")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    
    interest = dcc.interest(principal, rate, start, asof, end)
    expected = Money("5", Currencies["USD"])  # 100 * 0.1 * 0.5
    assert interest == expected
    
    # Test with asof equal to end date
    asof = end
    interest = dcc.interest(principal, rate, start, asof, end)
    assert interest == expected
    
    # Test with asof equal to start date
    asof = start
    interest = dcc.interest(principal, rate, start, asof, end)
    assert interest == expected
    
    # Test with zero principal
    principal = Money("0", Currencies["USD"])
    interest = dcc.interest(principal, rate, start, asof, end)
    assert interest == principal
    
    # Test with zero rate
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0")
    interest = dcc.interest(principal, rate, start, asof, end)
    assert interest == Money("0", Currencies["USD"])
    
    # Test with different currency
    principal = Money("200", Currencies["EUR"])
    rate = Decimal("0.05")
    interest = dcc.interest(principal, rate, start, asof, end)
    expected = Money("5", Currencies["EUR"])  # 200 * 0.05 * 0.5
    assert interest == expected
    
    # Test with asof before start (should return 0 through calculate_fraction)
    asof = datetime.date(2022, 12, 31)
    interest = dcc.interest(principal, rate, start, asof, end)
    assert interest == Money("0", Currencies["EUR"])
    
    # Test with asof after end (should return 0 through calculate_fraction)
    asof = datetime.date(2024, 1, 1)
    interest = dcc.interest(principal, rate, start, asof, end)
    assert interest == Money("0", Currencies["EUR"])
    
    # Test with end=None (should use asof as end)
    asof = datetime.date(2023, 6, 30)
    interest = dcc.interest(principal, rate, start, asof, None)
    expected = Money("5", Currencies["EUR"])  # 200 * 0.05 * 0.5
    assert interest == expected
    
    # Test with freq parameter (even though dummy method ignores it)
    freq = Decimal("2")
    interest = dcc.interest(principal, rate, start, asof, end, freq)
    assert interest == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Multi-year period from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Additional test: Start day 31 should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # Calculation: (28 - 30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')
    
    # Additional test: As-of day 31 should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(2-1) + 360*(2023-2023) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Additional test: Both start and as-of days 31 should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(2-1) + 360*(2023-2023) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Additional test: Same month, different days
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 3, 20)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # Calculation: (20 - 15) + 30*(3-3) + 360*(2023-2023) = 5 days
    # 5/360 = 0.01388888888889
    assert round(result, 14) == Decimal('0.01388888888889')
    
    # Additional test: Cross-year calculation
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # Calculation: (15 - 15) + 30*(1-12) + 360*(2023-2022) = 0 + 30*(-11) + 360 = -330 + 360 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #8
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test with frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    # Manual calculation: 182 days between start and asof, 366 days between start and end
    # 182 / 366 / 2 = 0.24863387978142076
    expected = Decimal('0.24863387978142076')
    assert round(result, 14) == round(expected, 14)

    # Test when asof equals start
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal(0)

    # Test when asof equals end
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    # 365 days between start and asof, 365 days between start and end
    # 365 / 365 = 1
    assert result == Decimal(1)

    # Test with leap year
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 8, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    # 2 days between start and asof, 185 days between start and end
    # 2 / 185 = 0.010810810810810811
    expected = Decimal('0.010810810810810811')
    assert round(result, 14) == round(expected, 14)

    # Test with frequency = 1 (annual)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    # 182 days between start and asof, 366 days between start and end
    # 182 / 366 / 1 = 0.4972677595628415
    expected = Decimal('0.4972677595628415')
    assert round(result, 14) == round(expected, 14)

    # Test with frequency = 4 (quarterly)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 4, 1)
    freq = Decimal(4)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    # 31 days between start and asof, 91 days between start and end
    # 31 / 91 / 4 = 0.08516483516483516
    expected = Decimal('0.08516483516483516')
    assert round(result, 14) == round(expected, 14)


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2021, 1, 1)
    asof1 = datetime.date(2021, 12, 31)
    end1 = datetime.date(2022, 1, 1)
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('364') / Decimal('365')
    assert result1 == expected1

    # Test case 2: Leap year period (asof in leap year)
    start2 = datetime.date(2020, 1, 1)
    asof2 = datetime.date(2020, 12, 31)
    end2 = datetime.date(2021, 1, 1)
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('365') / Decimal('366')
    assert result2 == expected2

    # Test case 3: Cross-year period with asof in non-leap year
    start3 = datetime.date(2019, 12, 31)
    asof3 = datetime.date(2020, 1, 31)
    end3 = datetime.date(2020, 2, 1)
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('31') / Decimal('366')
    assert result3 == expected3

    # Test case 4: Single day in leap year
    start4 = datetime.date(2020, 2, 28)
    asof4 = datetime.date(2020, 2, 29)
    end4 = datetime.date(2020, 3, 1)
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1') / Decimal('366')
    assert result4 == expected4

    # Test case 5: Single day in non-leap year
    start5 = datetime.date(2021, 2, 28)
    asof5 = datetime.date(2021, 3, 1)
    end5 = datetime.date(2021, 3, 2)
    result5 = dcfc_act_365_l(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('1') / Decimal('365')
    assert result5 == expected5

    # Test case 6: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result6 = dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    expected6 = Decimal('0.16939890710383')
    assert round(result6, 14) == expected6

    # Test case 7: Another example from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result7 = dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    expected7 = Decimal('0.17213114754098')
    assert round(result7, 14) == expected7

    # Test case 8: Verify freq parameter is ignored (as per implementation)
    start8 = datetime.date(2020, 1, 1)
    asof8 = datetime.date(2020, 6, 30)
    end8 = datetime.date(2020, 12, 31)
    result8_no_freq = dcfc_act_365_l(start=start8, asof=asof8, end=end8)
    result8_with_freq = dcfc_act_365_l(start=start8, asof=asof8, end=end8, freq=Decimal('2'))
    assert result8_no_freq == result8_with_freq


# LLM-generated content at query #10
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a simple DCC instance with a dummy calculation method
    def dummy_calc(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calc,
    )

    # Test normal case where start <= asof <= end
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")

    # Test edge case: asof equals start
    result = dcc.calculate_fraction(start, start, end)
    assert result == Decimal("0.5")

    # Test edge case: asof equals end
    result = dcc.calculate_fraction(start, end, end)
    assert result == Decimal("0.5")

    # Test invalid case: asof before start
    result = dcc.calculate_fraction(start, datetime.date(2022, 12, 31), end)
    assert result == ZERO

    # Test invalid case: asof after end
    result = dcc.calculate_fraction(start, datetime.date(2023, 2, 1), end)
    assert result == ZERO

    # Test invalid case: start > asof > end (impossible but check)
    result = dcc.calculate_fraction(end, asof, start)
    assert result == ZERO

    # Test with frequency parameter
    result = dcc.calculate_fraction(start, asof, end, Decimal("2"))
    assert result == Decimal("0.5")

    # Test with zero frequency
    result = dcc.calculate_fraction(start, asof, end, ZERO)
    assert result == Decimal("0.5")


# LLM-generated content at query #11
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a simple DCC with a mock calculation method
    def mock_calculate_fraction(start, asof, end, freq):
        # Simple linear day count for testing
        total_days = (end - start).days
        elapsed_days = (asof - start).days
        if total_days == 0:
            return Decimal('0')
        return Decimal(str(elapsed_days)) / Decimal(str(total_days))
    
    dcc = DCC(
        name="TEST",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    # Test 1: Normal case with multiple days
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 3)
    end = datetime.date(2023, 1, 10)
    
    # Calculate expected: fraction for day 3 minus fraction for day 2
    total_days = 9
    day3_fraction = Decimal('2') / Decimal('9')  # days 0-2 elapsed
    day2_fraction = Decimal('1') / Decimal('9')  # days 0-1 elapsed
    expected = day3_fraction - day2_fraction
    
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == expected
    
    # Test 2: First day of period
    asof = datetime.date(2023, 1, 1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # asof_minus_1 is before start, so yfact = 0
    # tfact = 0/9 = 0
    assert result == Decimal('0')
    
    # Test 3: Last day of period
    asof = datetime.date(2023, 1, 10)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # day9_fraction = 9/9 = 1, day8_fraction = 8/9
    expected = Decimal('1') - Decimal('8') / Decimal('9')
    assert result == expected
    
    # Test 4: Single day period
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # total_days = 0, so fractions are 0
    assert result == Decimal('0')
    
    # Test 5: Two day period, second day
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 1, 2)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # total_days = 1, day1_fraction = 1/1 = 1, day0_fraction = 0/1 = 0
    assert result == Decimal('1')
    
    # Test 6: With frequency parameter
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 5)
    end = datetime.date(2023, 1, 31)
    freq = Decimal('2')
    
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    # The mock method ignores freq, so same calculation
    total_days = 30
    day4_fraction = Decimal('4') / Decimal('30')
    day3_fraction = Decimal('3') / Decimal('30')
    expected = day4_fraction - day3_fraction
    assert result == expected
    
    # Test 7: Edge case where asof is before start (shouldn't happen in practice)
    # but calculate_daily_fraction doesn't check this, it just passes to calculate_fraction_method
    start = datetime.date(2023, 1, 5)
    asof = datetime.date(2023, 1, 3)
    end = datetime.date(2023, 1, 10)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # asof_minus_1 is before start, yfact = 0
    # tfact will be negative since asof < start
    total_days = 5
    elapsed_days = -2  # 3 - 5
    expected = Decimal('-2') / Decimal('5') - Decimal('0')
    assert result == expected


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Example from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Example from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Example from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test: Start day is 31, should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (28 - 30) + 30*(2 - 1) + 360*(2023 - 2023) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')

    # Additional test: Start day is 30 and asof day is 31, asof should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(2 - 1) + 360*(2023 - 2023) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')

    # Additional test: Both start and asof days are 31, both should be adjusted
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(3 - 1) + 360*(2023 - 2023) = 0 + 60 = 60 days
    # 60/360 = 0.16666666666667
    assert round(result, 14) == Decimal('0.16666666666667')

    # Additional test: Multi-year calculation
    start = datetime.date(2020, 12, 15)
    asof = datetime.date(2023, 6, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (15 - 15) + 30*(6 - 12) + 360*(2023 - 2020) = 0 + (-180) + 1080 = 900 days
    # 900/360 = 2.5
    assert result == Decimal('2.5')


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Example from docstring
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Example from docstring
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Example from docstring
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test: Start day is 31, should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal((28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

    # Additional test: Start day is 30 and asof day is 31, asof should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal((30 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

    # Additional test: Both adjustments needed
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal((30 - 30) + 30 * (3 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

    # Additional test: No adjustments needed
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 2, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal((15 - 15) + 30 * (2 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Example from docstring with leap day
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Example from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Example from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test: Start day is 31, should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal('0.08333333333333')  # (28-30 + 30*(2-1) + 360*0)/360 = 28/360
    assert round(result, 14) == round(expected, 14)

    # Additional test: Start day is 30 and asof day is 31, asof should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal('0.08333333333333')  # (30-30 + 30*(2-1) + 360*0)/360 = 30/360
    assert round(result, 14) == round(expected, 14)

    # Additional test: Same year, simple calculation
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 4, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal('0.25')  # (15-15 + 30*(4-1) + 360*0)/360 = 90/360
    assert result == expected

    # Additional test: Cross year boundary
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 3, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = Decimal('0.25')  # (15-15 + 30*(3-12+12) + 360*(2023-2022))/360 = (0 + 90 + 360)/360 = 450/360
    assert result == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=Decimal(1))
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test with different frequency
    result_freq2 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=Decimal(2))
    expected_freq2 = Decimal('0.2622950820')
    assert round(result_freq2, 10) == expected_freq2

    # Test when asof equals start
    result_same = dcfc_act_act_icma(start=ex1_start, asof=ex1_start, end=ex1_end, freq=Decimal(1))
    assert result_same == Decimal('0')

    # Test when asof equals end
    result_end = dcfc_act_act_icma(start=ex1_start, asof=ex1_end, end=ex1_end, freq=Decimal(1))
    assert round(result_end, 10) == Decimal('1.0000000000')

    # Test with partial period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 2, 1)
    result_partial = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(12))
    expected_partial = Decimal('0.4516129032')
    assert round(result_partial, 10) == expected_partial

    # Test with leap year
    start_leap = datetime.date(2020, 2, 28)
    asof_leap = datetime.date(2020, 3, 1)
    end_leap = datetime.date(2020, 3, 31)
    result_leap = dcfc_act_act_icma(start=start_leap, asof=asof_leap, end=end_leap, freq=Decimal(4))
    expected_leap = Decimal('0.0322580645')
    assert round(result_leap, 10) == expected_leap


# LLM-generated content at query #16
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a simple DCC instance with a dummy calculation method
    def dummy_calc(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calc
    )
    
    # Test normal case where start <= asof <= end
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")
    
    # Test when asof equals start
    result = dcc.calculate_fraction(start, start, end)
    assert result == Decimal("0.5")
    
    # Test when asof equals end
    result = dcc.calculate_fraction(start, end, end)
    assert result == Decimal("0.5")
    
    # Test when asof < start (should return 0)
    result = dcc.calculate_fraction(start, datetime.date(2022, 12, 31), end)
    assert result == ZERO
    
    # Test when asof > end (should return 0)
    result = dcc.calculate_fraction(start, datetime.date(2023, 2, 1), end)
    assert result == ZERO
    
    # Test when start > asof > end (should return 0)
    result = dcc.calculate_fraction(end, asof, start)
    assert result == ZERO
    
    # Test with frequency parameter
    result = dcc.calculate_fraction(start, asof, end, Decimal("2"))
    assert result == Decimal("0.5")
    
    # Test with actual calculation method (ACT/365)
    def act_365(start, asof, end, freq):
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    dcc_act365 = DCC(
        name="ACT/365",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=act_365
    )
    
    # Test ACT/365 calculation
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 12, 31)
    result = dcc_act365.calculate_fraction(start, asof, end)
    expected = Decimal(30) / Decimal(365)  # 30 days from Jan 1 to Jan 31
    assert result == expected
    
    # Test edge case with same dates
    result = dcc_act365.calculate_fraction(start, start, start)
    assert result == ZERO
    
    # Test with negative scenario where calculation method would return negative
    def negative_calc(start, asof, end, freq):
        return Decimal("-0.1")
    
    dcc_negative = DCC(
        name="NegativeDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=negative_calc
    )
    
    # Even with negative calculation method, date validation should work
    result = dcc_negative.calculate_fraction(start, asof, end)
    assert result == Decimal("-0.1")
    
    # Test with invalid date ordering but valid for calculation method
    result = dcc_negative.calculate_fraction(end, asof, start)
    assert result == ZERO  # Should be 0 due to date validation


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Example from docstring
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('0.16942884946478')
    assert round(result, 14) == expected

    # Test case 2: Leap year example from docstring
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('0.17216108990194')
    assert round(result, 14) == expected

    # Test case 3: Multi-year example from docstring
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1.08243131970956')
    assert round(result, 14) == expected

    # Test case 4: Another multi-year example from docstring
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1.32625945055768')
    assert round(result, 14) == expected

    # Test case 5: Same day should return 0
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: One day in non-leap year
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1') / Decimal('365')
    assert result == expected

    # Test case 7: One day in leap year
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1') / Decimal('366')
    assert result == expected

    # Test case 8: Cross-year boundary with leap year
    start = datetime.date(2019, 12, 31)
    asof = datetime.date(2020, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    # 1 day in 2019 (non-leap) + 2 days in 2020 (leap)
    expected = Decimal('1') / Decimal('365') + Decimal('2') / Decimal('366')
    assert result == expected

    # Test case 9: Full non-leap year
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2022, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1')

    # Test case 10: Full leap year
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1')

    # Test case 11: asof before start should be handled by calculate_fraction method
    # This test ensures the function itself doesn't break with invalid input
    start = datetime.date(2021, 1, 2)
    asof = datetime.date(2021, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    # The function will still calculate, but DCC.calculate_fraction will return 0
    # This test just ensures no exception is raised
    assert result >= Decimal('0')


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_30_e_plus_360():
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Multi-year period from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')
    
    # Additional test: Start day 31 should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # Calculation: (28 - 30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')
    
    # Additional test: As-of day 31 should be incremented to next day
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # As-of day 31 becomes Mar 1, so: (1 - 30) + 30*(3-1) + 360*(2023-2023) = -29 + 60 = 31 days
    # 31/360 = 0.08611111111111
    assert round(result, 14) == Decimal('0.08611111111111')
    
    # Additional test: Both start and asof have day 31
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # Start becomes day 30, asof becomes Apr 1
    # (1 - 30) + 30*(4-1) + 360*(2023-2023) = -29 + 90 = 61 days
    # 61/360 = 0.16944444444444
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Additional test: Same date should return 0
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0')
    
    # Additional test: One month difference with normal days
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 4, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # (15 - 15) + 30*(4-3) + 360*(2023-2023) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Period without leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Period containing Feb 29
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period crossing multiple years
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Period starting in leap year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4

    # Additional test: Period with no leap days
    start5 = datetime.date(2021, 1, 1)
    asof5 = datetime.date(2021, 12, 31)
    end5 = asof5
    result5 = dcfc_nl_365(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('364') / Decimal('365')
    assert result5 == expected5

    # Additional test: Period exactly one year with leap day
    start6 = datetime.date(2020, 1, 1)
    asof6 = datetime.date(2020, 12, 31)
    end6 = asof6
    result6 = dcfc_nl_365(start=start6, asof=asof6, end=end6)
    expected6 = Decimal('365') / Decimal('365')  # Leap day excluded
    assert result6 == expected6


# LLM-generated content at query #20
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    registry = DCCRegistryMachinery()
    
    # Create mock DCC objects for testing
    mock_dcc1 = DCC(
        name="ACT/360",
        altnames={"Actual/360"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.1")
    )
    
    mock_dcc2 = DCC(
        name="30/360",
        altnames={"Bond Basis"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.2")
    )
    
    # Register DCCs
    registry.register(mock_dcc1)
    registry.register(mock_dcc2)
    
    # Test finding by exact main name
    result = registry.find("ACT/360")
    assert result is not None
    assert result.name == "ACT/360"
    
    # Test finding by alternative name
    result = registry.find("Actual/360")
    assert result is not None
    assert result.name == "ACT/360"
    
    # Test finding by exact main name (second DCC)
    result = registry.find("30/360")
    assert result is not None
    assert result.name == "30/360"
    
    # Test finding by alternative name (second DCC)
    result = registry.find("Bond Basis")
    assert result is not None
    assert result.name == "30/360"
    
    # Test case-insensitive search with whitespace stripping
    result = registry.find("  act/360  ")
    assert result is not None
    assert result.name == "ACT/360"
    
    result = registry.find("  actual/360  ")
    assert result is not None
    assert result.name == "ACT/360"
    
    # Test uppercase conversion
    result = registry.find("act/360")
    assert result is not None
    assert result.name == "ACT/360"
    
    # Test non-existent DCC
    result = registry.find("NonExistent")
    assert result is None
    
    # Test empty string
    result = registry.find("")
    assert result is None
    
    # Test that _find_strict works as expected
    result = registry._find_strict("ACT/360")
    assert result is not None
    assert result.name == "ACT/360"
    
    result = registry._find_strict("act/360")  # Should not find with lowercase
    assert result is None
    
    # Test that registry property returns all registered DCCs
    registered_dccs = registry.registry
    assert len(registered_dccs) == 2
    assert any(dcc.name == "ACT/360" for dcc in registered_dccs)
    assert any(dcc.name == "30/360" for dcc in registered_dccs)


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_30_360_german():
    # Test case 1: Basic example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Multi-year period
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Another multi-year period
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')
    
    # Additional test: Start date is last day of February
    start = datetime.date(2020, 2, 29)  # Leap year
    asof = datetime.date(2020, 3, 31)
    end = asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # d1 = 30 (last day of Feb), d2 = 30 (31st becomes 30)
    # (30-30) + 30*(3-2) + 360*(2020-2020) = 30
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Test: asof is last day of February but not equal to end
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # d1 = 30 (31st becomes 30), d2 = 30 (last day of Feb and asof != end)
    # (30-30) + 30*(2-1) + 360*(2020-2020) = 30
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Test: asof is last day of February and equals end
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    end = asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # d1 = 30 (31st becomes 30), d2 = 29 (last day of Feb but asof == end)
    # (29-30) + 30*(2-1) + 360*(2020-2020) = 29
    # 29/360 = 0.08055555555556
    assert round(result, 14) == Decimal('0.08055555555556')
    
    # Test: Regular dates without special adjustments
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 6, 15)
    end = asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # (15-15) + 30*(6-3) + 360*(2023-2023) = 90
    # 90/360 = 0.25
    assert result == Decimal('0.25')


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_30_360_german():
    # Test case 1: Basic example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Another longer period from docstring
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')
    
    # Additional test: Start date is last day of February (non-leap year)
    start = datetime.date(2023, 2, 28)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # Calculation: d1=30 (last day of Feb), d2=30 (31st becomes 30), 
    # days = (30-30) + 30*(3-2) + 360*(2023-2023) = 30
    # fraction = 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Additional test: Start date is 31st, asof is 31st
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # Calculation: d1=30 (31st becomes 30), d2=30 (31st becomes 30),
    # days = (30-30) + 30*(3-1) + 360*(2023-2023) = 60
    # fraction = 60/360 = 0.16666666666667
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Additional test: asof is last day of February and not equal to end
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 3, 31)  # Different from asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # Calculation: d1=30, d2=30 (last day of Feb when asof != end),
    # days = (30-30) + 30*(2-1) + 360*(2023-2023) = 30
    # fraction = 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Additional test: asof is last day of February and equal to end
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = asof  # Same as asof
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # Calculation: d1=30, d2=28 (last day of Feb but asof == end, so keep actual day),
    # days = (28-30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28
    # fraction = 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Basic example from docstring
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_30_360_us(start=start1, asof=asof1, end=end1)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_30_360_us(start=start2, asof=asof2, end=end2)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_30_360_us(start=start3, asof=asof3, end=end3)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Multi-year period from docstring
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_30_360_us(start=start4, asof=asof4, end=end4)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Additional test: Both dates are last day of month
    start5 = datetime.date(2023, 1, 31)
    asof5 = datetime.date(2023, 2, 28)
    end5 = asof5
    result5 = dcfc_30_360_us(start=start5, asof=asof5, end=end5)
    # Calculation: D1=30 (last day of month), D2=30 (last day of month and D1=30)
    # (30-30) + 30*(2-1) + 360*(2023-2023) = 0 + 30 + 0 = 30
    # 30/360 = 0.08333333333333
    assert round(result5, 14) == Decimal('0.08333333333333')
    
    # Additional test: D1=31, D2=31
    start6 = datetime.date(2023, 1, 31)
    asof6 = datetime.date(2023, 2, 28)
    end6 = asof6
    result6 = dcfc_30_360_us(start=start6, asof=asof6, end=end6)
    # Same as above - both become 30
    assert round(result6, 14) == Decimal('0.08333333333333')
    
    # Additional test: D1=30, D2=31
    start7 = datetime.date(2023, 1, 30)
    asof7 = datetime.date(2023, 2, 28)
    end7 = asof7
    result7 = dcfc_30_360_us(start=start7, asof=asof7, end=end7)
    # D1=30, D2=28 (not 31, so no change)
    # (28-30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 + 0 = 28
    # 28/360 = 0.07777777777778
    assert round(result7, 14) == Decimal('0.07777777777778')
    
    # Additional test: Same month
    start8 = datetime.date(2023, 1, 15)
    asof8 = datetime.date(2023, 1, 31)
    end8 = asof8
    result8 = dcfc_30_360_us(start=start8, asof=asof8, end=end8)
    # D1=15, D2=31 -> D2 becomes 30 (since D1=15 not 30 or 31)
    # (30-15) + 30*(1-1) + 360*(2023-2023) = 15 + 0 + 0 = 15
    # 15/360 = 0.04166666666667
    assert round(result8, 14) == Decimal('0.04166666666667')


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Basic example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Another longer period from docstring
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')
    
    # Additional test: Start day is 31, should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (28 - 30) + 30*(2 - 1) + 360*(2023 - 2023) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')
    
    # Additional test: Start day is 30 and asof day is 31, asof should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(2 - 1) + 360*(2023 - 2023) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Additional test: Same month, different days
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 3, 20)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (20 - 15) + 30*(3 - 3) + 360*(2023 - 2023) = 5 days
    # 5/360 = 0.01388888888889
    assert round(result, 14) == Decimal('0.01388888888889')
    
    # Additional test: Cross-year calculation
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (15 - 15) + 30*(1 - 12) + 360*(2023 - 2022) = 0 + 30*(-11) + 360 = -330 + 360 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Basic example from docstring
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_30_360_isda(start=start1, asof=asof1, end=end1)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year adjustment
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_30_360_isda(start=start2, asof=asof2, end=end2)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_30_360_isda(start=start3, asof=asof3, end=end3)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Multi-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_30_360_isda(start=start4, asof=asof4, end=end4)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Additional test: Start day 31 adjustment
    start5 = datetime.date(2023, 1, 31)
    asof5 = datetime.date(2023, 3, 31)
    end5 = asof5
    result5 = dcfc_30_360_isda(start=start5, asof=asof5, end=end5)
    # Calculation: (30-30) + 30*(3-1) + 360*(0) = 60 days / 360 = 0.1666666667
    assert round(result5, 10) == Decimal('0.1666666667')
    
    # Additional test: Start day 30 and asof day 31 adjustment
    start6 = datetime.date(2023, 1, 30)
    asof6 = datetime.date(2023, 3, 31)
    end6 = asof6
    result6 = dcfc_30_360_isda(start=start6, asof=asof6, end=end6)
    # Calculation: (30-30) + 30*(3-1) + 360*(0) = 60 days / 360 = 0.1666666667
    assert round(result6, 10) == Decimal('0.1666666667')
    
    # Additional test: No adjustments needed
    start7 = datetime.date(2023, 1, 15)
    asof7 = datetime.date(2023, 4, 15)
    end7 = asof7
    result7 = dcfc_30_360_isda(start=start7, asof=asof7, end=end7)
    # Calculation: (15-15) + 30*(4-1) + 360*(0) = 90 days / 360 = 0.25
    assert result7 == Decimal('0.25')


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Period without leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Period containing leap day (Feb 29, 2008)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period spanning multiple years
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Period with leap year but no Feb 29 in range
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4

    # Test case 5: Single day period without leap day
    start5 = datetime.date(2023, 1, 1)
    asof5 = datetime.date(2023, 1, 1)
    end5 = asof5
    result5 = dcfc_nl_365(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('0') / Decimal(365)
    assert result5 == expected5

    # Test case 6: Period with leap day excluded from calculation
    start6 = datetime.date(2008, 2, 28)
    asof6 = datetime.date(2008, 3, 1)
    end6 = asof6
    result6 = dcfc_nl_365(start=start6, asof=asof6, end=end6)
    # Actual days = 2 (Feb 28-29), but leap day excluded = 1 day
    expected6 = Decimal('1') / Decimal(365)
    assert result6 == expected6

    # Test case 7: Period entirely within non-leap year
    start7 = datetime.date(2023, 1, 1)
    asof7 = datetime.date(2023, 12, 31)
    end7 = asof7
    result7 = dcfc_nl_365(start=start7, asof=asof7, end=end7)
    expected7 = Decimal('364') / Decimal(365)  # 364 days (exclusive of end)
    assert result7 == expected7


# LLM-generated content at query #4
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC with a dummy calculation method
    def dummy_calc(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calc,
    )

    # Test basic interest calculation
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 30)
    end = Date(2023, 12, 31)

    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money("5", Currencies["USD"])  # 100 * 0.1 * 0.5
    assert result == expected

    # Test with end date same as asof
    result = dcc.interest(principal, rate, start, asof, asof)
    assert result == expected  # Same fraction calculation applies

    # Test with zero rate
    result = dcc.interest(principal, Decimal("0"), start, asof, end)
    assert result == Money("0", Currencies["USD"])

    # Test with zero principal
    zero_principal = Money("0", Currencies["USD"])
    result = dcc.interest(zero_principal, rate, start, asof, end)
    assert result == Money("0", Currencies["USD"])

    # Test with different currency
    principal_eur = Money("200", Currencies["EUR"])
    result = dcc.interest(principal_eur, rate, start, asof, end)
    expected_eur = Money("10", Currencies["EUR"])  # 200 * 0.1 * 0.5
    assert result == expected_eur

    # Test when asof equals start date
    def calc_half(start, asof, end, freq):
        if asof == start:
            return Decimal("0")
        return Decimal("0.5")

    dcc_half = DCC(
        name="TestDCCHalf",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=calc_half,
    )
    result = dcc_half.interest(principal, rate, start, start, end)
    assert result == Money("0", Currencies["USD"])

    # Test when asof equals end date
    result = dcc_half.interest(principal, rate, start, end, end)
    assert result == Money("5", Currencies["USD"])

    # Test with frequency parameter (should be ignored by dummy calc)
    result = dcc.interest(principal, rate, start, asof, end, Decimal("2"))
    assert result == expected

    # Test with invalid date ordering (asof before start)
    # This should return zero interest because calculate_fraction returns 0
    result = dcc.interest(principal, rate, asof, start, end)
    assert result == Money("0", Currencies["USD"])

    # Test with asof after end date
    result = dcc.interest(principal, rate, start, Date(2024, 1, 1), end)
    assert result == Money("0", Currencies["USD"])


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Basic example
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

    # Test case 2: Leap year with Feb 29
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

    # Test case 3: Longer period
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

    # Test case 4: Multi-year period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

    # Additional test: Start date is last day of month
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    # Calculation: (30-30) + 30*(2-1) + 360*(2023-2023) = 30 days
    expected = Decimal('30') / Decimal('360')
    assert result == expected

    # Additional test: Both dates are last day of month
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    # Both become 30, so: (30-30) + 30*(2-1) + 360*(0) = 30 days
    expected = Decimal('30') / Decimal('360')
    assert result == expected

    # Additional test: d2=31 and d1=30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    # d2 becomes 30, so: (30-30) + 30*(2-1) + 360*(0) = 30 days
    expected = Decimal('30') / Decimal('360')
    assert result == expected

    # Additional test: d1=31 becomes 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 15)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    # d1 becomes 30, so: (15-30) + 30*(2-1) + 360*(0) = -15 + 30 = 15 days
    expected = Decimal('15') / Decimal('360')
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_30_360_german():
    # Test case 1: Basic example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: February 29 case
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Another longer period
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')
    
    # Additional test: Start date is last day of February
    start = datetime.date(2020, 2, 29)
    asof = datetime.date(2020, 3, 31)
    end = datetime.date(2020, 3, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # Calculation: d1=30 (last day of Feb), d2=30 (31st becomes 30), 
    # days = (30-30) + 30*(3-2) + 360*(2020-2020) = 30
    # fraction = 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Test: asof is last day of February but not equal to end
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # d1=30 (31st becomes 30), d2=30 (last day of Feb and asof != end)
    # days = (30-30) + 30*(2-1) + 360*(2020-2020) = 30
    # fraction = 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Test: asof is last day of February and equals end
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # d1=30 (31st becomes 30), d2=29 (last day of Feb but asof == end)
    # days = (29-30) + 30*(2-1) + 360*(2020-2020) = 29
    # fraction = 29/360 = 0.08055555555556
    assert round(result, 14) == Decimal('0.08055555555556')
    
    # Test: Regular dates with no special adjustments
    start = datetime.date(2021, 3, 15)
    asof = datetime.date(2021, 6, 15)
    end = datetime.date(2021, 6, 15)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    # days = (15-15) + 30*(6-3) + 360*(2021-2021) = 90
    # fraction = 90/360 = 0.25
    assert result == Decimal('0.25')


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Period without leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Period containing February 29 (leap day)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period spanning multiple years
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Period with leap year but no February 29 in range
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4

    # Additional test: Period entirely within non-leap year
    start5 = datetime.date(2023, 1, 1)
    asof5 = datetime.date(2023, 6, 30)
    end5 = asof5
    result5 = dcfc_nl_365(start=start5, asof=asof5, end=end5)
    # 181 days / 365 = 0.4958904109589041
    expected5 = Decimal('0.49589041095890')
    assert round(result5, 14) == expected5

    # Test with freq parameter (should be ignored according to docstring)
    start6 = datetime.date(2007, 12, 28)
    asof6 = datetime.date(2008, 2, 28)
    end6 = asof6
    result6 = dcfc_nl_365(start=start6, asof=asof6, end=end6, freq=Decimal('2'))
    expected6 = Decimal('0.16986301369863')
    assert round(result6, 14) == expected6


# LLM-generated content at query #8
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a simple DCC instance with a dummy calculation method
    def dummy_calc(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calc,
    )

    # Test normal case where start <= asof <= end
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")

    # Test edge case: asof equals start
    asof = datetime.date(2023, 1, 1)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")

    # Test edge case: asof equals end
    asof = datetime.date(2023, 1, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")

    # Test invalid case: asof before start
    asof = datetime.date(2022, 12, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO

    # Test invalid case: asof after end
    asof = datetime.date(2023, 2, 1)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO

    # Test with frequency parameter
    freq = Decimal("2")
    asof = datetime.date(2023, 1, 15)
    result = dcc.calculate_fraction(start, asof, end, freq)
    assert result == Decimal("0.5")

    # Test with another calculation method
    def actual_360_calc(start, asof, end, freq):
        days = (asof - start).days
        return Decimal(days) / Decimal(360)

    dcc2 = DCC(
        name="Actual/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=actual_360_calc,
    )

    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 16)  # 15 days difference
    end = datetime.date(2023, 1, 31)
    result = dcc2.calculate_fraction(start, asof, end)
    expected = Decimal("15") / Decimal("360")
    assert result == expected

    # Test with asof at start (0 days difference)
    asof = datetime.date(2023, 1, 1)
    result = dcc2.calculate_fraction(start, asof, end)
    assert result == ZERO

    # Test with invalid date ordering
    asof = datetime.date(2022, 12, 31)
    result = dcc2.calculate_fraction(start, asof, end)
    assert result == ZERO


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Period without leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_act_365_a(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Period with leap day (2008 is leap year)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_act_365_a(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_act_365_a(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Period crossing leap year boundary
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_act_365_a(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1.32513661202186')
    assert round(result4, 14) == expected4

    # Test case 5: Single day calculation
    start5 = datetime.date(2023, 1, 1)
    asof5 = datetime.date(2023, 1, 1)
    end5 = asof5
    result5 = dcfc_act_365_a(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('0') / Decimal(365)  # 2023 is not leap year
    assert result5 == expected5

    # Test case 6: Leap day itself in leap year
    start6 = datetime.date(2020, 2, 29)
    asof6 = datetime.date(2020, 2, 29)
    end6 = asof6
    result6 = dcfc_act_365_a(start=start6, asof=asof6, end=end6)
    expected6 = Decimal('0') / Decimal(366)  # 2020 is leap year
    assert result6 == expected6

    # Test case 7: Period entirely within leap year
    start7 = datetime.date(2020, 1, 1)
    asof7 = datetime.date(2020, 3, 1)
    end7 = asof7
    result7 = dcfc_act_365_a(start=start7, asof=asof7, end=end7)
    days = Decimal((asof7 - start7).days)
    expected7 = days / Decimal(366)
    assert result7 == expected7

    # Test case 8: Verify freq parameter is ignored (as per implementation)
    start8 = datetime.date(2023, 1, 1)
    asof8 = datetime.date(2023, 6, 30)
    end8 = asof8
    result8_without_freq = dcfc_act_365_a(start=start8, asof=asof8, end=end8)
    result8_with_freq = dcfc_act_365_a(start=start8, asof=asof8, end=end8, freq=Decimal(4))
    assert result8_without_freq == result8_with_freq


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert round(result, 10) == Decimal('0.5245901639')

    # Test with frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    expected = Decimal(_get_actual_day_count(start, asof)) / Decimal(_get_actual_day_count(start, end)) / freq
    assert result == expected

    # Test when asof equals start
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal(0)

    # Test when asof equals end
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal(_get_actual_day_count(start, asof)) / Decimal(_get_actual_day_count(start, end))
    assert result == expected

    # Test with leap year
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 3, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal(_get_actual_day_count(start, asof)) / Decimal(_get_actual_day_count(start, end))
    assert result == expected

    # Test with different year spans
    start = datetime.date(2019, 12, 31)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal(_get_actual_day_count(start, asof)) / Decimal(_get_actual_day_count(start, end))
    assert result == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC instance with a dummy calculation method
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction,
    )

    # Test basic interest calculation
    principal = Money("1000", Currencies["USD"])
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 30)
    end = Date(2023, 12, 31)

    result = dcc.interest(principal, rate, start, asof, end)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

    # Test with end date same as asof
    result2 = dcc.interest(principal, rate, start, asof, asof)
    expected2 = principal * rate * Decimal("0.5")
    assert result2 == expected2

    # Test with zero rate
    zero_rate = Decimal("0")
    result3 = dcc.interest(principal, zero_rate, start, asof, end)
    assert result3 == Money("0", Currencies["USD"])

    # Test with zero principal
    zero_principal = Money("0", Currencies["USD"])
    result4 = dcc.interest(zero_principal, rate, start, asof, end)
    assert result4 == Money("0", Currencies["USD"])

    # Test with different currency
    eur_principal = Money("500", Currencies["EUR"])
    result5 = dcc.interest(eur_principal, rate, start, asof, end)
    expected5 = eur_principal * rate * Decimal("0.5")
    assert result5 == expected5

    # Test with asof outside date range (should return zero through calculate_fraction)
    def zero_fraction_calc(start, asof, end, freq):
        if not start <= asof <= end:
            return Decimal("0")
        return Decimal("0.5")

    dcc_zero = DCC(
        name="ZeroDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=zero_fraction_calc,
    )

    # asof before start
    early_asof = Date(2022, 12, 31)
    result6 = dcc_zero.interest(principal, rate, start, early_asof, end)
    assert result6 == Money("0", Currencies["USD"])

    # asof after end
    late_asof = Date(2024, 1, 1)
    result7 = dcc_zero.interest(principal, rate, start, late_asof, end)
    assert result7 == Money("0", Currencies["USD"])


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Basic example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Multi-year period
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')
    
    # Additional test: Start day is 31st
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Manual calculation: (28 - 30) + 30*(2-1) + 360*(0) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')
    
    # Additional test: Start day 30, asof day 31
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # asof day becomes 30: (30 - 30) + 30*(2-1) + 360*(0) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')
    
    # Additional test: Same month
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 3, 25)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # (25 - 15) + 30*(0) + 360*(0) = 10 days
    # 10/360 = 0.02777777777778
    assert round(result, 14) == Decimal('0.02777777777778')
    
    # Additional test: Cross-year calculation
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # (15 - 15) + 30*(1-12) + 360*(2023-2022) = 0 + 30*(-11) + 360*1 = -330 + 360 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Leap year case from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Longer period from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Another longer period from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test: Start day is 31 should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (28 - 30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')

    # Additional test: Start day is 30 and asof day is 31 should adjust asof to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(2-1) + 360*(2023-2023) = 0 + 30 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')

    # Additional test: Same month, different days
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 3, 25)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (25 - 15) + 30*(3-3) + 360*(2023-2023) = 10 days
    # 10/360 = 0.02777777777778
    assert round(result, 14) == Decimal('0.02777777777778')

    # Additional test: Cross-year calculation
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (15 - 15) + 30*(1-12) + 360*(2023-2022) = 0 + 30*(-11) + 360 = -330 + 360 = 30 days
    # 30/360 = 0.08333333333333
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=end1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in period (Feb 29)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=end2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=end3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Period spanning multiple years
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=end4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4

    # Test case 5: Single day period without leap day
    start5 = datetime.date(2023, 1, 1)
    asof5 = datetime.date(2023, 1, 1)
    end5 = asof5
    result5 = dcfc_nl_365(start=start5, asof=asof5, end=end5)
    expected5 = Decimal('0') / Decimal(365)
    assert result5 == expected5

    # Test case 6: Period with leap day excluded
    start6 = datetime.date(2020, 2, 28)
    asof6 = datetime.date(2020, 3, 1)
    end6 = asof6
    result6 = dcfc_nl_365(start=start6, asof=asof6, end=end6)
    # Actual days = 2 (Feb 28-29), but Feb 29 is excluded
    expected6 = Decimal('1') / Decimal(365)
    assert result6 == expected6


# LLM-generated content at query #15
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test with different frequency
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    # Manual calculation: days between start and asof = 91, days between start and end = 182
    # 91 / 182 / 2 = 0.25
    assert result == Decimal('0.25')

    # Test with asof equal to start
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result == Decimal('0')

    # Test with asof equal to end
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    # Manual calculation: days between start and end = 182, days between start and end = 182
    # 182 / 182 / 2 = 0.5
    assert result == Decimal('0.5')

    # Test without frequency (should default to ONE)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    result = dcfc_act_act_icma(start, asof, end)
    # Without freq, uses ONE = 1
    # 91 / 182 / 1 = 0.5
    assert result == Decimal('0.5')

    # Test with leap year
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 3, 15)
    freq = Decimal(4)
    result = dcfc_act_act_icma(start, asof, end, freq)
    # Manual calculation: days between start and asof = 2, days between start and end = 16
    # 2 / 16 / 4 = 0.03125
    assert result == Decimal('0.03125')


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_30_e_plus_360():
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Leap year case from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Longer period from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Multi-year period from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')
    
    # Additional test: Start day 31 should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # Expected: (28 - 30) + 30*(2-1) + 360*(0) = -2 + 30 = 28 days
    # 28/360 = 0.07777777777778
    assert round(result, 14) == Decimal('0.07777777777778')
    
    # Additional test: As-of day 31 should be moved to next day (32nd)
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # As-of becomes Mar 1, so: (1 - 30) + 30*(3-1) + 360*(0) = -29 + 60 = 31 days
    # 31/360 = 0.08611111111111
    assert round(result, 14) == Decimal('0.08611111111111')
    
    # Additional test: Both start and asof have day 31
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    # Start becomes Jan 30, asof becomes Apr 1
    # (1 - 30) + 30*(4-1) + 360*(0) = -29 + 90 = 61 days
    # 61/360 = 0.16944444444444
    assert round(result, 14) == Decimal('0.16944444444444')


# LLM-generated content at query #17
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC with a dummy calculate_fraction_method
    def dummy_calculate_fraction(start, asof, end, freq):
        # Always return 0.5 for testing
        return Decimal("0.5")
    
    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    # Test basic interest calculation
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 30)
    end = Date(2023, 12, 31)
    
    interest = dcc.interest(principal, rate, start, asof, end)
    expected = Money("5", Currencies["USD"])  # 100 * 0.1 * 0.5
    assert interest == expected
    
    # Test with end date same as asof
    interest2 = dcc.interest(principal, rate, start, asof, asof)
    assert interest2 == expected
    
    # Test with different currency
    principal_eur = Money("200", Currencies["EUR"])
    interest_eur = dcc.interest(principal_eur, rate, start, asof, end)
    expected_eur = Money("10", Currencies["EUR"])
    assert interest_eur == expected_eur
    
    # Test with zero rate
    interest_zero = dcc.interest(principal, Decimal("0"), start, asof, end)
    expected_zero = Money("0", Currencies["USD"])
    assert interest_zero == expected_zero
    
    # Test with zero principal
    zero_principal = Money("0", Currencies["USD"])
    interest_zero_principal = dcc.interest(zero_principal, rate, start, asof, end)
    assert interest_zero_principal == expected_zero
    
    # Test with frequency parameter (should be ignored by dummy method)
    interest_with_freq = dcc.interest(principal, rate, start, asof, end, Decimal("2"))
    assert interest_with_freq == expected
    
    # Test when asof equals start date
    interest_start = dcc.interest(principal, rate, start, start, end)
    assert interest_start == expected
    
    # Test when asof equals end date
    interest_end = dcc.interest(principal, rate, start, end, end)
    assert interest_end == expected


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Basic example from docstring
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

    # Test case 2: Leap year case from docstring
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

    # Test case 3: Longer period from docstring
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

    # Test case 4: Multi-year period from docstring
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('1.33055555555556')
    assert round(result, 14) == expected

    # Additional test: Start day 31 should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    # Calculation: (28 - 30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28 days
    expected = Decimal('28') / Decimal('360')
    assert result == expected

    # Additional test: As-of day 31 should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    # Calculation: (30 - 30) + 30*(3-1) + 360*(2023-2023) = 0 + 60 = 60 days
    expected = Decimal('60') / Decimal('360')
    assert result == expected

    # Additional test: Both start and asof have day 31
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    # Calculation: (30 - 30) + 30*(3-1) + 360*(2023-2023) = 0 + 60 = 60 days
    expected = Decimal('60') / Decimal('360')
    assert result == expected

    # Additional test: Cross-year calculation
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 3, 15)
    end = datetime.date(2023, 3, 15)
    result = dcfc_30_e_360(start, asof, end)
    # Calculation: (15 - 15) + 30*(3-12) + 360*(2023-2022) = 0 + (-270) + 360 = 90 days
    expected = Decimal('90') / Decimal('360')
    assert result == expected

    # Test with freq parameter (should be ignored)
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, end, freq=Decimal('2'))
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Period without leap day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

    # Test case 2: Period with leap day (Feb 29)
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

    # Test case 3: Longer period spanning multiple years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('1.08219178082192')
    assert round(result, 14) == expected

    # Test case 4: Period crossing leap year
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('1.32602739726027')
    assert round(result, 14) == expected

    # Additional test: Period with no leap days
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 12, 31)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    # 364 days in 2021 (non-leap year) / 365
    expected = Decimal('364') / Decimal('365')
    assert round(result, 14) == round(expected, 14)

    # Additional test: Period containing Feb 29 in leap year
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    # 2 days total, but Feb 29 is excluded, so only 1 day counted
    expected = Decimal('1') / Decimal('365')
    assert round(result, 14) == round(expected, 14)


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Leap year case
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Longer period
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Multi-year period
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test: Start day is 31, should be adjusted to 30
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (28 - 30) + 30*(2-1) + 360*(2023-2023) = -2 + 30 = 28 days
    expected = Decimal(28) / Decimal(360)
    assert result == expected

    # Additional test: Start day is 30 and asof day is 31, asof should be adjusted to 30
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (30 - 30) + 30*(2-1) + 360*(2023-2023) = 0 + 30 = 30 days
    expected = Decimal(30) / Decimal(360)
    assert result == expected

    # Additional test: Both adjustments needed
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Start adjusted to 30, asof adjusted to 30
    # Calculation: (30 - 30) + 30*(3-1) + 360*(2023-2023) = 0 + 60 = 60 days
    expected = Decimal(60) / Decimal(360)
    assert result == expected

    # Additional test: Same year, same month
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 25)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (25 - 15) + 30*(5-5) + 360*(2023-2023) = 10 days
    expected = Decimal(10) / Decimal(360)
    assert result == expected

    # Additional test: Cross-year calculation
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    # Calculation: (15 - 15) + 30*(1-12) + 360*(2023-2022) = 0 + 30*(-11) + 360 = -330 + 360 = 30 days
    expected = Decimal(30) / Decimal(360)
    assert result == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Test 1: Basic daily fraction calculation with simple DCC
    def simple_dcc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal(365)
    
    dcc = DCC(
        name="TEST/365",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcc
    )
    
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 12, 31)
    
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal(1) / Decimal(365)
    assert daily_frac == expected
    
    # Test 2: Multiple days accumulation
    asof = datetime.date(2023, 1, 10)
    total_frac = dcc.calculate_fraction(start, asof, end)
    daily_sum = Decimal(0)
    for i in range(1, 10):
        daily_date = datetime.date(2023, 1, i)
        daily_sum += dcc.calculate_daily_fraction(start, daily_date, end)
    assert daily_sum == total_frac
    
    # Test 3: Edge case - asof equals start date
    asof = datetime.date(2023, 1, 1)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    assert daily_frac == Decimal(0)
    
    # Test 4: Edge case - asof one day after start
    asof = datetime.date(2023, 1, 2)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    assert daily_frac > Decimal(0)
    
    # Test 5: Test with ACT/ACT DCC
    def act_act_dcc(start, asof, end, freq):
        if freq is None:
            return Decimal((asof - start).days) / Decimal(365)
        return Decimal((asof - start).days) / Decimal(365 * int(freq))
    
    dcc_act = DCC(
        name="ACT/ACT",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=act_act_dcc
    )
    
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 12, 31)
    freq = Decimal(2)
    
    daily_frac = dcc_act.calculate_daily_fraction(start, asof, end, freq)
    expected = Decimal(1) / Decimal(365 * 2)
    assert daily_frac == expected
    
    # Test 6: Verify asof_minus_1 < start case
    asof = datetime.date(2023, 1, 1)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    assert daily_frac == Decimal(0)
    
    # Test 7: Test with leap year
    def leap_year_dcc(start, asof, end, freq):
        year_days = 366 if calendar.isleap(start.year) else 365
        return Decimal((asof - start).days) / Decimal(year_days)
    
    dcc_leap = DCC(
        name="TEST/Leap",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=leap_year_dcc
    )
    
    start = datetime.date(2024, 2, 28)  # Leap year
    asof = datetime.date(2024, 2, 29)
    end = datetime.date(2024, 12, 31)
    
    daily_frac = dcc_leap.calculate_daily_fraction(start, asof, end)
    expected = Decimal(1) / Decimal(366)
    assert daily_frac == expected
    
    # Test 8: Test with negative fraction (should not happen with proper date ordering)
    start = datetime.date(2023, 1, 10)
    asof = datetime.date(2023, 1, 9)
    end = datetime.date(2023, 12, 31)
    
    # This should return 0 because asof < start
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    assert daily_frac == Decimal(0)


