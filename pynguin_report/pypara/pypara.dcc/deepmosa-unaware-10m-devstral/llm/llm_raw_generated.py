####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Test1Alt"}, {"USD"}, lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC("Test2", {"Test2Alt"}, {"EUR"}, lambda s, a, e, f: Decimal(0.3))

    # Register DCCs
    registry.register(dcc1)
    registry.register(dcc2)

    # Test finding by exact name
    assert registry.find("Test1") == dcc1
    assert registry.find("Test2") == dcc2

    # Test finding by alternative name
    assert registry.find("Test1Alt") == dcc1
    assert registry.find("Test2Alt") == dcc2

    # Test finding with case insensitivity and whitespace
    assert registry.find(" test1 ") == dcc1
    assert registry.find("TEST2") == dcc2

    # Test finding non-existent DCC
    assert registry.find("NonExistent") is None
    assert registry.find("") is None

    # Test finding after partial registration failure
    with pytest.raises(TypeError):
        registry.register(DCC("Test1", set(), {"GBP"}, lambda s, a, e, f: Decimal(0.1)))
    assert registry.find("Test1") == dcc1  # Original should still be there


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_360_us():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


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


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day included
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')


# LLM-generated content at query #6
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    registry = DCCRegistryMachinery()

    # Test successful registration
    dcc1 = DCC("Test1", {"Test1Alt"}, {Currencies["USD"]}, lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    assert registry.find("Test1") == dcc1
    assert registry.find("Test1Alt") == dcc1

    # Test duplicate registration
    dcc2 = DCC("Test1", {"Test1Alt"}, {Currencies["EUR"]}, lambda s, a, e, f: Decimal(0.6))
    with pytest.raises(TypeError):
        registry.register(dcc2)

    # Test alternative name conflict
    dcc3 = DCC("Test3", {"Test1Alt"}, {Currencies["GBP"]}, lambda s, a, e, f: Decimal(0.7))
    with pytest.raises(TypeError):
        registry.register(dcc3)

    # Test case-insensitive and stripped name
    dcc4 = DCC("Test4", {"test4alt"}, {Currencies["JPY"]}, lambda s, a, e, f: Decimal(0.8))
    registry.register(dcc4)
    assert registry.find("test4") == dcc4
    assert registry.find(" TEST4ALT ") == dcc4


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period with leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('1.32513661202186')

    # Test case 5: Same day
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_a(start, asof, asof)
    assert result == Decimal('0')

    # Test case 6: Cross-leap year period
    start = datetime.date(2007, 2, 28)
    asof = datetime.date(2008, 3, 1)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('1.00547945205479')


# LLM-generated content at query #8
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16939890710383')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with non-leap year
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Longer period with leap year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32876712328767')
    assert round(result4, 14) == expected4


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Example from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 2: Full period (start to end)
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('1.0000000000')
    assert round(result, 10) == expected

    # Test case 3: Half period (leap year)
    start = datetime.date(2020, 3, 2)
    asof = datetime.date(2020, 9, 2)
    end = datetime.date(2021, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5000000000')
    assert round(result, 10) == expected

    # Test case 4: Quarter period
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 6, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.2500000000')
    assert round(result, 10) == expected

    # Test case 5: With frequency parameter
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(2))
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected

    # Test case 6: Edge case - same start and asof dates
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.0000000000')
    assert round(result, 10) == expected

    # Test case 7: Edge case - asof date equals end date
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('1.0000000000')
    assert round(result, 10) == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16939890710383')
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == expected2

    # Test case 3: Another non-leap year
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == expected3

    # Test case 4: Another leap year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32876712328767')
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == expected4


# LLM-generated content at query #12
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Setup test data
    start_date = Date(2023, 1, 1)
    asof_date = Date(2023, 6, 1)
    end_date = Date(2023, 12, 31)
    freq = Decimal('2')

    # Create a mock DCC instance with a simple calculation method
    def mock_calculation(start, asof, end, freq):
        total_days = (end - start).days
        elapsed_days = (asof - start).days
        return Decimal(elapsed_days) / Decimal(total_days)

    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculation
    )

    # Test normal case
    result = dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    expected = Decimal('161') / Decimal('364')  # 161 days from Jan 1 to Jun 1 in 2023
    assert result == expected

    # Test when asof is before start
    result = dcc.calculate_fraction(Date(2023, 2, 1), Date(2023, 1, 1), end_date, freq)
    assert result == ZERO

    # Test when asof is after end
    result = dcc.calculate_fraction(start_date, Date(2024, 1, 1), end_date, freq)
    assert result == ZERO

    # Test when asof equals start
    result = dcc.calculate_fraction(start_date, start_date, end_date, freq)
    expected = Decimal('0') / Decimal('364')
    assert result == expected

    # Test when asof equals end
    result = dcc.calculate_fraction(start_date, end_date, end_date, freq)
    expected = Decimal('364') / Decimal('364')
    assert result == ONE

    # Test with None frequency
    result = dcc.calculate_fraction(start_date, asof_date, end_date, None)
    expected = Decimal('161') / Decimal('364')
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16939890710383')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Non-leap year with longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Leap year with longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32876712328767')
    assert round(result4, 14) == expected4


# LLM-generated content at query #14
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a mock DCC instance with a simple calculation method
    def simple_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days) if (end - start).days != 0 else ZERO

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_calc
    )

    # Test case 1: Normal case where asof is between start and end
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 15)
    end = Date(2020, 1, 31)
    expected = Decimal(14) / Decimal(30)
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 2: asof is equal to start
    asof = Date(2020, 1, 1)
    expected = ZERO
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 3: asof is equal to end
    asof = Date(2020, 1, 31)
    expected = ONE
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 4: asof is outside the range (before start)
    asof = Date(2019, 12, 31)
    expected = ZERO
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 5: asof is outside the range (after end)
    asof = Date(2020, 2, 1)
    expected = ZERO
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 6: start and end are the same
    start = Date(2020, 1, 1)
    end = Date(2020, 1, 1)
    asof = Date(2020, 1, 1)
    expected = ZERO
    assert dcc.calculate_fraction(start, asof, end) == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 4, 1)
    end = Date(2020, 7, 1)
    freq = 2  # Semi-annual
    eom = None

    # Create a mock DCC instance with a simple day count fraction method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="MockDCC",
        altnames={"MOCK"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Calculate expected result manually
    prevdate = _last_payment_date(start, asof, freq, eom)
    nextdate = _next_payment_date(prevdate, freq, eom)
    expected_fraction = mock_calculate_fraction(prevdate, asof, nextdate, Decimal(freq))
    expected_interest = principal * rate * expected_fraction

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected_interest
    assert isinstance(result, Money)
    assert result.currency == principal.currency


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    expected = Decimal('0.16939890710383')
    result = dcfc_act_365_l(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 2: Leap year
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    expected = Decimal('0.17213114754098')
    result = dcfc_act_365_l(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 3: Non-leap year with longer period
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    expected = Decimal('1.08196721311475')
    result = dcfc_act_365_l(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 4: Leap year with longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    expected = Decimal('1.32876712328767')
    result = dcfc_act_365_l(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start1, asof1, asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (should subtract 1 day)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start2, asof2, asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start3, asof3, asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Period spanning multiple years
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start4, asof4, asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #18
#--------------------------

```python
def test_DCC_coupon():
    # Create a mock DCC instance for testing
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test data
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)
    freq = 2
    eom = None

    # Expected result: principal * rate * fraction * (freq / freq) = 1000 * 0.05 * 0.5 * 1 = 25
    expected = Money(25, Currencies["USD"])

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day in the period (should be subtracted)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.16986301369863')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08219178082192')

    # Test case 4: Another longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32602739726027')


# LLM-generated content at query #20
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(Currency("USD"), Decimal("1000.00"))
    rate = Decimal("0.05")  # 5% interest rate
    start = Date(2020, 1, 1)
    asof = Date(2020, 4, 1)
    end = Date(2020, 7, 1)
    freq = Decimal("2")  # Semi-annual payments
    eom = None

    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.25")  # 25% of the year

    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Call the coupon method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Expected interest: 1000 * 0.05 * 0.25 = 12.5
    expected = Money(Currency("USD"), Decimal("12.50"))

    # Assert the result
    assert result == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (should be subtracted)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day in period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    # Test case 5: Same day
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    result5 = dcfc_act_365_a(start5, asof5, asof5)
    assert result5 == Decimal('0')

    # Test case 6: Cross year boundary without leap day
    start6 = datetime.date(2019, 12, 31)
    asof6 = datetime.date(2020, 1, 1)
    result6 = dcfc_act_365_a(start6, asof6, asof6)
    assert round(result6, 14) == Decimal('0.00273972602740')


# LLM-generated content at query #23
#--------------------------

```python
def test_DCC_interest():
    # Setup
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)

    # Test with end date provided
    result_with_end = dcc.interest(principal, rate, start, asof, end)
    assert result_with_end == Money(50, Currency("USD"))

    # Test with end date as asof
    result_without_end = dcc.interest(principal, rate, start, asof)
    assert result_without_end == Money(50, Currency("USD"))

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0")
    )
    result_zero = dcc_zero.interest(principal, rate, start, asof, end)
    assert result_zero == Money(0, Currency("USD"))

    # Test with invalid date range (asof before start)
    result_invalid = dcc.interest(principal, rate, asof, start, end)
    assert result_invalid == Money(0, Currency("USD"))


# LLM-generated content at query #24
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    # Test case 5: Same start and asof dates
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    result5 = dcfc_act_365_a(start5, asof5, asof5)
    assert result5 == Decimal('0.00000000000000')

    # Test case 6: Period crossing leap day (Feb 29)
    start6 = datetime.date(2020, 2, 28)
    asof6 = datetime.date(2020, 3, 1)
    result6 = dcfc_act_365_a(start6, asof6, asof6)
    assert round(result6, 14) == Decimal('0.00273972602740')


# LLM-generated content at query #25
#--------------------------

```python
def test_DCC_coupon():
    # Create a DCC instance for testing
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test data
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None

    # Expected result
    expected_interest = principal * rate * Decimal("0.5")

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected_interest


# LLM-generated content at query #26
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    principal = Money(1000, "USD")
    rate = Decimal("0.05")  # 5%
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 12, 31)

    # Create a simple DCC instance for testing
    def simple_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal(360)

    dcc = DCC(
        name="TestDCC",
        altnames={"Test"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_calculate_fraction
    )

    # Test interest calculation
    interest = dcc.interest(principal, rate, start, asof, end)
    expected_days = (asof - start).days
    expected_fraction = Decimal(expected_days) / Decimal(360)
    expected_interest = principal * rate * expected_fraction

    assert interest == expected_interest
    assert interest.currency == principal.currency

    # Test with end date being None (should use asof)
    interest_no_end = dcc.interest(principal, rate, start, asof)
    assert interest_no_end == expected_interest

    # Test with invalid date range (should return 0)
    invalid_interest = dcc.interest(principal, rate, end, start, asof)
    assert invalid_interest == Money(0, "USD")

    # Test with different rate
    rate_10 = Decimal("0.10")  # 10%
    interest_10 = dcc.interest(principal, rate_10, start, asof, end)
    expected_interest_10 = principal * rate_10 * expected_fraction
    assert interest_10 == expected_interest_10


# LLM-generated content at query #27
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    # Setup
    registry = DCCRegistryMachinery()
    test_dcc = DCC(
        name="TestDCC",
        altnames={"AltTestDCC"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test successful registration
    registry.register(test_dcc)
    assert registry.find("TestDCC") == test_dcc
    assert registry.find("AltTestDCC") == test_dcc

    # Test duplicate registration with same name
    with pytest.raises(TypeError):
        registry.register(test_dcc)

    # Test duplicate registration with alternative name
    duplicate_dcc = DCC(
        name="AnotherDCC",
        altnames={"TestDCC"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError):
        registry.register(duplicate_dcc)

    # Test duplicate registration with alternative name in altnames
    another_duplicate = DCC(
        name="YetAnotherDCC",
        altnames={"AltTestDCC"},
        currencies={Currencies["GBP"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError):
        registry.register(another_duplicate)


# LLM-generated content at query #28
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #29
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #30
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #31
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case with no adjustments
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st, start date is 30th
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st, adjust to 30th
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Longer period with multiple years
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st, end date is 31st
    start = datetime.date(2007, 1, 31)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 6: Start date is 30th, end date is 31st (adjust end date to 30th)
    start = datetime.date(2007, 1, 30)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 7: Start date is 31st, end date is 30th
    start = datetime.date(2007, 1, 31)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 8: Start date is 31st, end date is 31st (adjust start date to 30th)
    start = datetime.date(2007, 1, 31)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 9: Start date is 30th, end date is 31st (adjust end date to 30th)
    start = datetime.date(2007, 1, 30)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 10: Start date is 31st, end date is 30th (adjust start date to 30th)
    start = datetime.date(2007, 1, 31)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')


# LLM-generated content at query #32
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Normal period without leap day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0.16942884946478')
    assert round(result, 14) == expected

    # Test case 2: Period with leap day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0.17216108990194')
    assert round(result, 14) == expected

    # Test case 3: Longer period with leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('1.08243131970956')
    assert round(result, 14) == expected

    # Test case 4: Another longer period with leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('1.32625945055768')
    assert round(result, 14) == expected

    # Test case 5: Same start and asof dates
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0')
    assert result == expected

    # Test case 6: One day period
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 2)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('1') / Decimal('365')
    assert result == expected

    # Test case 7: Period spanning multiple years with leap days
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('1461') / Decimal('365') + Decimal('366') / Decimal('366')
    assert result == expected

    # Test case 8: Period with no leap years
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2019, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('730') / Decimal('365')
    assert result == expected


# LLM-generated content at query #33
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    expected4 = Decimal('1.32513661202186')
    assert round(result4, 14) == expected4


# LLM-generated content at query #34
#--------------------------

```python
def test_dcfc_nl_365():
    # Test cases from the docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')

    # Additional test cases
    # Test with a leap day in the period
    start_leap, asof_leap = datetime.date(2020, 2, 28), datetime.date(2020, 3, 1)
    assert round(dcfc_nl_365(start=start_leap, asof=asof_leap, end=asof_leap), 14) == Decimal('0.00273972602740')

    # Test with no leap day in the period
    start_no_leap, asof_no_leap = datetime.date(2019, 2, 28), datetime.date(2019, 3, 1)
    assert round(dcfc_nl_365(start=start_no_leap, asof=asof_no_leap, end=asof_no_leap), 14) == Decimal('0.00273972602740')

    # Test with a full year period
    start_year, asof_year = datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)
    assert round(dcfc_nl_365(start=start_year, asof=asof_year, end=asof_year), 14) == Decimal('0.99726027397260')

    # Test with a full leap year period
    start_leap_year, asof_leap_year = datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)
    assert round(dcfc_nl_365(start=start_leap_year, asof=asof_leap_year, end=asof_leap_year), 14) == Decimal('0.99726027397260')


# LLM-generated content at query #35
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #36
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #37
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="MOCK",
        altnames={"MOCK_ALT"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=mock_calculate_fraction
    )

    start = Date(2020, 1, 1)
    end = Date(2020, 1, 10)
    asof = Date(2020, 1, 5)

    # Test normal case
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is start date
    result = dcc.calculate_daily_fraction(start, start, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is end date
    result = dcc.calculate_daily_fraction(start, end, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is day after start
    asof = Date(2020, 1, 2)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is day before end
    asof = Date(2020, 1, 9)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is before start (should return 0)
    asof = Date(2019, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected

    # Test when asof is after end (should return 0)
    asof = Date(2020, 1, 11)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected


# LLM-generated content at query #38
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    # Setup
    registry = DCCRegistryMachinery()
    test_dcc = DCC(
        name="TestDCC",
        altnames={"TestDCCAlt"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test successful registration
    registry.register(test_dcc)
    assert registry._find_strict("TestDCC") == test_dcc
    assert registry._find_strict("TestDCCAlt") == test_dcc

    # Test duplicate registration
    duplicate_dcc = DCC(
        name="TestDCC",
        altnames={"AnotherAlt"},
        currencies=_as_ccys({"EUR"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError, match="Day count convention 'TestDCC' is already registered"):
        registry.register(duplicate_dcc)

    # Test alternative name conflict
    conflict_dcc = DCC(
        name="ConflictDCC",
        altnames={"TestDCCAlt"},
        currencies=_as_ccys({"GBP"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError, match="Day count convention 'ConflictDCC' is already registered"):
        registry.register(conflict_dcc)

    # Test find method
    assert registry.find("TestDCC") == test_dcc
    assert registry.find("testdcc") == test_dcc
    assert registry.find("TestDCCAlt") == test_dcc
    assert registry.find("testdccalt") == test_dcc


# LLM-generated content at query #39
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


# LLM-generated content at query #40
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (but NL/365 ignores it)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #41
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")  # 5% interest rate
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 30)
    end = Date(2020, 12, 31)
    freq = Decimal("1")

    # Create a DCC instance with a simple day count fraction method
    def simple_dcf(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_dcf
    )

    # Calculate expected interest
    expected_fraction = Decimal(181) / Decimal(366)  # 2020 is a leap year
    expected_interest = principal * rate * expected_fraction

    # Test interest calculation
    result = dcc.interest(principal, rate, start, asof, end, freq)
    assert result == expected_interest

    # Test with asof equal to end
    result = dcc.interest(principal, rate, start, end, end, freq)
    assert result == principal * rate * Decimal("1")

    # Test with asof equal to start
    result = dcc.interest(principal, rate, start, start, end, freq)
    assert result == principal * rate * Decimal("0")

    # Test with no end date provided (should use asof)
    result = dcc.interest(principal, rate, start, asof)
    assert result == principal * rate * simple_dcf(start, asof, asof, None)


# LLM-generated content at query #42
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #43
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #44
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (Feb 29)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Period crossing multiple years
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #45
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #46
#--------------------------

```python
def test_DCC_interest():
    # Create a mock DCC instance with a simple day count fraction method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal('0.5')  # Simple fixed fraction for testing

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal('0.10')  # 10%
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)

    # Expected interest: 1000 * 0.10 * 0.5 = 50
    expected_interest = Money(50, Currency("USD"))

    # Test with end date provided
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected_interest

    # Test with end date as None (should use asof)
    result_no_end = dcc.interest(principal, rate, start, asof)
    assert result_no_end == expected_interest

    # Test with asof before start (should return 0)
    result_before_start = dcc.interest(principal, rate, start, datetime.date(2019, 12, 1), end)
    assert result_before_start == Money(0, Currency("USD"))

    # Test with asof after end (should return 0)
    result_after_end = dcc.interest(principal, rate, start, datetime.date(2021, 1, 1), end)
    assert result_after_end == Money(0, Currency("USD"))


# LLM-generated content at query #47
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #48
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #49
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 6: Start date is 30th and asof date is 31st
    start = datetime.date(2007, 11, 30)
    asof = datetime.date(2008, 1, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.05555555555556')


# LLM-generated content at query #50
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period with leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period without leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32513661202186')

    # Test case 5: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: Period spanning multiple years with leap day
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2021, 1, 1)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('2.00273224043716')


# LLM-generated content at query #51
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2: Example from docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Example from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Example from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

    # Additional test case: Same start and asof dates
    same_date = datetime.date(2020, 1, 15)
    assert dcfc_30_360_us(start=same_date, asof=same_date, end=same_date) == Decimal('0')

    # Additional test case: Start date is last day of month
    start_last_day = datetime.date(2020, 1, 31)
    asof_feb = datetime.date(2020, 2, 15)
    assert round(dcfc_30_360_us(start=start_last_day, asof=asof_feb, end=asof_feb), 14) == Decimal('0.13888888888889')

    # Additional test case: As-of date is last day of month
    start_mid_month = datetime.date(2020, 1, 15)
    asof_last_day = datetime.date(2020, 2, 29)  # Leap year
    assert round(dcfc_30_360_us(start=start_mid_month, asof=asof_last_day, end=asof_last_day), 14) == Decimal('0.13888888888889')

    # Additional test case: Both dates are last day of month
    start_last_day = datetime.date(2020, 1, 31)
    asof_last_day = datetime.date(2020, 2, 29)  # Leap year
    assert round(dcfc_30_360_us(start=start_last_day, asof=asof_last_day, end=asof_last_day), 14) == Decimal('0.13888888888889')

    # Additional test case: As-of date is 31st and start date is 30th or 31st
    start_30th = datetime.date(2020, 1, 30)
    asof_31st = datetime.date(2020, 3, 31)
    assert round(dcfc_30_360_us(start=start_30th, asof=asof_31st, end=asof_31st), 14) == Decimal('0.19444444444444')

    # Additional test case: As-of date is 31st and start date is not 30th or 31st
    start_15th = datetime.date(2020, 1, 15)
    asof_31st = datetime.date(2020, 3, 31)
    assert round(dcfc_30_360_us(start=start_15th, asof=asof_31st, end=asof_31st), 14) == Decimal('0.22222222222222')

    # Additional test case: Start date is 31st
    start_31st = datetime.date(2020, 1, 31)
    asof_mid_month = datetime.date(2020, 3, 15)
    assert round(dcfc_30_360_us(start=start_31st, asof=asof_mid_month, end=asof_mid_month), 14) == Decimal('0.13888888888889')


# LLM-generated content at query #52
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.1")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date provided
    result = dcc.interest(principal, rate, start, asof, end)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

    # Test with end date not provided (should use asof)
    result_no_end = dcc.interest(principal, rate, start, asof)
    assert result_no_end == expected

    # Test with different fraction calculation
    dcc_different = DCC(
        name="Test2",
        altnames={"TestAlt2"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.25")
    )
    result_different = dcc_different.interest(principal, rate, start, asof, end)
    expected_different = principal * rate * Decimal("0.25")
    assert result_different == expected_different

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames={"TestAltZero"},
        currencies={Currencies["GBP"]},
        calculate_fraction_method=lambda start, asof, end, freq: ZERO
    )
    result_zero = dcc_zero.interest(principal, rate, start, asof, end)
    assert result_zero == Money(0, Currencies["GBP"])


# LLM-generated content at query #53
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None

    # Create a mock DCC object with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal(360)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Expected result calculation
    prevdate = _last_payment_date(start, asof, freq, eom)
    nextdate = _next_payment_date(prevdate, freq, eom)
    expected_fraction = mock_calculate_fraction(prevdate, asof, nextdate, Decimal(freq))
    expected_interest = principal * rate * expected_fraction

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected_interest
    assert isinstance(result, Money)


# LLM-generated content at query #54
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"test1"}, set(), lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC("Test2", {"test2", "TEST2_ALT"}, set(), lambda s, a, e, f: Decimal(0.5))

    # Register DCCs
    registry.register(dcc1)
    registry.register(dcc2)

    # Test finding by exact name
    assert registry.find("Test1") == dcc1
    assert registry.find("Test2") == dcc2

    # Test finding by alternative name
    assert registry.find("test2_alt") == dcc2

    # Test case-insensitive and stripped name
    assert registry.find(" test1 ") == dcc1
    assert registry.find("TEST2") == dcc2

    # Test non-existent DCC
    assert registry.find("NonExistent") is None


# LLM-generated content at query #55
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    asof_date = datetime.date(2023, 6, 15)

    # Create a mock DCC instance with a simple calculation method
    def simple_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currency("USD")},
        calculate_fraction_method=simple_calc
    )

    # Test normal case
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    expected_fraction = simple_calc(start_date, asof_date, end_date) - simple_calc(start_date, asof_date - datetime.timedelta(days=1), end_date)
    assert daily_fraction == expected_fraction

    # Test when asof is start date
    daily_fraction_start = dcc.calculate_daily_fraction(start_date, start_date, end_date)
    expected_fraction_start = simple_calc(start_date, start_date, end_date) - Decimal(0)
    assert daily_fraction_start == expected_fraction_start

    # Test when asof is day after start
    daily_fraction_next_day = dcc.calculate_daily_fraction(start_date, start_date + datetime.timedelta(days=1), end_date)
    expected_fraction_next_day = simple_calc(start_date, start_date + datetime.timedelta(days=1), end_date) - simple_calc(start_date, start_date, end_date)
    assert daily_fraction_next_day == expected_fraction_next_day

    # Test when asof is end date
    daily_fraction_end = dcc.calculate_daily_fraction(start_date, end_date, end_date)
    expected_fraction_end = simple_calc(start_date, end_date, end_date) - simple_calc(start_date, end_date - datetime.timedelta(days=1), end_date)
    assert daily_fraction_end == expected_fraction_end


# LLM-generated content at query #56
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16939890710383')
    assert round(dcfc_act_365_l(start=start1, asof=asof1, end=asof1), 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    assert round(dcfc_act_365_l(start=start2, asof=asof2, end=asof2), 14) == expected2

    # Test case 3: Non-leap year with longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    assert round(dcfc_act_365_l(start=start3, asof=asof3, end=asof3), 14) == expected3

    # Test case 4: Leap year with longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32876712328767')
    assert round(dcfc_act_365_l(start=start4, asof=asof4, end=asof4), 14) == expected4


# LLM-generated content at query #57
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Normal case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Long period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.33055555555556')

    # Test case 5: Start and end dates are the same
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 1)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.00000000000000')

    # Test case 6: Start date is 31st and end date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 7: Start date is 30th and end date is 31st
    start = datetime.date(2007, 11, 30)
    asof = datetime.date(2008, 12, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.00000000000000')


# LLM-generated content at query #58
#--------------------------

```python
def test_DCC_coupon():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 4, 1)
    end = Date(2020, 7, 1)
    freq = 2  # Semi-annual
    eom = None

    # Expected values
    prevdate = Date(2020, 1, 1)
    nextdate = Date(2020, 7, 1)
    expected_fraction = Decimal(120) / Decimal(181)  # Days from 2020-01-01 to 2020-04-30 (120 days) over 181 days
    expected_interest = principal * rate * expected_fraction

    # Test
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    assert result == expected_interest
    assert dcc._last_payment_date(start, asof, freq, eom) == prevdate
    assert dcc._next_payment_date(prevdate, freq, eom) == nextdate


# LLM-generated content at query #59
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #60
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day in the period (should be excluded)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == Decimal('0.16986301369863')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == Decimal('1.08219178082192')

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == Decimal('1.32602739726027')


# LLM-generated content at query #61
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Basic test with non-leap year
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0.16942884946478')
    assert round(result, 14) == expected

    # Test case 2: Test with leap day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0.17216108990194')
    assert round(result, 14) == expected

    # Test case 3: Test with longer period
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('1.08243131970956')
    assert round(result, 14) == expected

    # Test case 4: Test with multiple years
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('1.32625945055768')
    assert round(result, 14) == expected

    # Test case 5: Test with same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0.00')
    assert round(result, 14) == expected

    # Test case 6: Test with leap year period
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    result = dcfc_act_act(start, asof, asof)
    expected = Decimal('0.00273972602740')
    assert round(result, 14) == expected


# LLM-generated content at query #62
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #63
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Period spanning multiple years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Period with leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: One day period
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.00273972602740')


# LLM-generated content at query #64
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case without day adjustment
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st when start is 30th
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3: Longer period with month crossing
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4: Multi-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st (should be adjusted to 30th)
    start5 = datetime.date(2020, 1, 31)
    asof5 = datetime.date(2020, 2, 28)
    assert round(dcfc_30_360_isda(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.08333333333333')

    # Test case 6: Both start and end are 31st
    start6 = datetime.date(2020, 1, 31)
    asof6 = datetime.date(2020, 3, 31)
    assert round(dcfc_30_360_isda(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.16666666666667')


# LLM-generated content at query #65
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Standard case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 29th
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.33055555555556')

    # Test case 5: Start and end dates are the same
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    assert dcfc_30_e_360(start=start, asof=asof, end=asof) == Decimal('0')

    # Test case 6: Start date is 31st and end date is 31st
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.02777777777778')

    # Test case 7: Start date is 30th and end date is 31st
    start = datetime.date(2020, 1, 30)
    asof = datetime.date(2020, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.02777777777778')


# LLM-generated content at query #66
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #67
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


# LLM-generated content at query #68
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period with leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period without leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_365_a(start, asof, asof)
    assert round(result, 14) == Decimal('1.32513661202186')


# LLM-generated content at query #69
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Test case 1: Check if dates are not provided properly
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    assert dcc.calculate_fraction(datetime.date(2023, 1, 1), datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)) == ZERO

    # Test case 2: Check if dates are provided properly
    assert dcc.calculate_fraction(datetime.date(2023, 1, 1), datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)) == Decimal("0.5")

    # Test case 3: Check if the method calls the underlying calculate_fraction_method correctly
    def mock_method(s, a, e, f):
        return Decimal("0.75")
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=mock_method
    )
    assert dcc.calculate_fraction(datetime.date(2023, 1, 1), datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)) == Decimal("0.75")


# LLM-generated content at query #70
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    asof_date = datetime.date(2023, 6, 15)

    # Create a mock DCC instance with a simple calculate_fraction_method
    def simple_fraction_method(start, asof, end, freq):
        total_days = (end - start).days
        if total_days == 0:
            return ZERO
        return Decimal((asof - start).days) / Decimal(total_days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_fraction_method
    )

    # Test 1: Normal case where asof is within start and end
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    expected_fraction = simple_fraction_method(start_date, asof_date, end_date, None) - \
                        simple_fraction_method(start_date, asof_date - datetime.timedelta(days=1), end_date, None)
    assert daily_fraction == expected_fraction

    # Test 2: asof is start date (should return fraction for first day)
    daily_fraction_start = dcc.calculate_daily_fraction(start_date, start_date, end_date)
    expected_fraction_start = simple_fraction_method(start_date, start_date, end_date, None)
    assert daily_fraction_start == expected_fraction_start

    # Test 3: asof is day after start date
    asof_date_next = start_date + datetime.timedelta(days=1)
    daily_fraction_next = dcc.calculate_daily_fraction(start_date, asof_date_next, end_date)
    expected_fraction_next = simple_fraction_method(start_date, asof_date_next, end_date, None) - \
                            simple_fraction_method(start_date, start_date, end_date, None)
    assert daily_fraction_next == expected_fraction_next

    # Test 4: asof is end date (should return fraction for last day)
    daily_fraction_end = dcc.calculate_daily_fraction(start_date, end_date, end_date)
    expected_fraction_end = simple_fraction_method(start_date, end_date, end_date, None) - \
                           simple_fraction_method(start_date, end_date - datetime.timedelta(days=1), end_date, None)
    assert daily_fraction_end == expected_fraction_end

    # Test 5: asof is before start date (should return 0)
    asof_before_start = start_date - datetime.timedelta(days=1)
    daily_fraction_before = dcc.calculate_daily_fraction(start_date, asof_before_start, end_date)
    assert daily_fraction_before == ZERO

    # Test 6: asof is after end date (should return 0)
    asof_after_end = end_date + datetime.timedelta(days=1)
    daily_fraction_after = dcc.calculate_daily_fraction(start_date, asof_after_end, end_date)
    assert daily_fraction_after == ZERO


# LLM-generated content at query #71
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16939890710383')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Non-leap year with longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Mixed leap and non-leap years
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32876712328767')
    assert round(result4, 14) == expected4


# LLM-generated content at query #72
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


# LLM-generated content at query #73
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Setup test data
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    asof_date = datetime.date(2020, 6, 15)
    freq = Decimal('1')

    # Create a mock DCC instance with a simple calculation method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test normal case
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date, freq)
    expected_fraction = mock_calculate_fraction(start_date, asof_date, end_date, freq) - mock_calculate_fraction(start_date, asof_date - datetime.timedelta(days=1), end_date, freq)
    assert daily_fraction == expected_fraction

    # Test when asof is start date
    daily_fraction_start = dcc.calculate_daily_fraction(start_date, start_date, end_date, freq)
    assert daily_fraction_start == mock_calculate_fraction(start_date, start_date, end_date, freq)

    # Test when asof is day after start
    daily_fraction_next_day = dcc.calculate_daily_fraction(start_date, start_date + datetime.timedelta(days=1), end_date, freq)
    expected_next_day = mock_calculate_fraction(start_date, start_date + datetime.timedelta(days=1), end_date, freq) - mock_calculate_fraction(start_date, start_date, end_date, freq)
    assert daily_fraction_next_day == expected_next_day

    # Test when asof is end date
    daily_fraction_end = dcc.calculate_daily_fraction(start_date, end_date, end_date, freq)
    assert daily_fraction_end == ZERO


# LLM-generated content at query #74
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #75
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test case 1: Normal case where asof is in the middle of the period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 31)
    freq = None

    # Expected daily fraction: (15-1)/(31-1) - (14-1)/(31-1) = 14/30 - 13/30 = 1/30
    expected = Decimal(1) / Decimal(30)
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == expected

    # Test case 2: asof is the start date
    asof = start
    # Expected daily fraction: (1-1)/(31-1) - (0-1)/(31-1) = 0 - (-1/30) = 1/30
    expected = Decimal(1) / Decimal(30)
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == expected

    # Test case 3: asof is the end date
    asof = end
    # Expected daily fraction: (31-1)/(31-1) - (30-1)/(31-1) = 1 - 29/30 = 1/30
    expected = Decimal(1) / Decimal(30)
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == expected

    # Test case 4: asof is before start (should return 0)
    asof = datetime.date(2019, 12, 31)
    expected = ZERO
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == expected

    # Test case 5: asof is after end (should return 0)
    asof = datetime.date(2020, 2, 1)
    expected = ZERO
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == expected


# LLM-generated content at query #76
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #77
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Test1Alt"}, {"USD"}, lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC("Test2", {"Test2Alt"}, {"EUR"}, lambda s, a, e, f: Decimal(0.3))

    # Register DCCs
    registry.register(dcc1)
    registry.register(dcc2)

    # Test finding by exact name
    assert registry.find("Test1") == dcc1
    assert registry.find("Test2") == dcc2

    # Test finding by alternative name
    assert registry.find("Test1Alt") == dcc1
    assert registry.find("Test2Alt") == dcc2

    # Test finding with case insensitivity and whitespace
    assert registry.find(" test1 ") == dcc1
    assert registry.find("TEST2") == dcc2

    # Test finding non-existent DCC
    assert registry.find("NonExistent") is None
    assert registry.find("") is None


# LLM-generated content at query #78
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #79
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Basic case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Leap year case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Longer period
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Another longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Test case 5: Last day of month
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.00277777777778')

    # Test case 6: Both dates are last day of month
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.00555555555556')

    # Test case 7: d2 is 31 and d1 is 30
    start = datetime.date(2020, 1, 30)
    asof = datetime.date(2020, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.02777777777778')

    # Test case 8: d1 is 31
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 1)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.00277777777778')


# LLM-generated content at query #80
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16939890710383')
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == expected3

    # Test case 4: Longer period without leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32876712328767')
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == expected4


# LLM-generated content at query #81
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day present
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == Decimal('0.16986301369863')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == Decimal('1.08219178082192')

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == Decimal('1.32602739726027')


# LLM-generated content at query #82
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Setup test data
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 6, 1)
    end_date = datetime.date(2020, 12, 31)
    freq = Decimal('1')

    # Create a mock DCC instance with a simple calculation method
    def mock_calculate_fraction_method(start, asof, end, freq):
        total_days = (end - start).days
        elapsed_days = (asof - start).days
        return Decimal(elapsed_days) / Decimal(total_days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST1", "TEST2"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction_method
    )

    # Test normal case
    result = dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    expected = Decimal('182') / Decimal('366')  # 2020 is a leap year
    assert result == expected

    # Test when asof is before start
    result = dcc.calculate_fraction(start_date, datetime.date(2019, 12, 31), end_date, freq)
    assert result == ZERO

    # Test when asof is after end
    result = dcc.calculate_fraction(start_date, datetime.date(2021, 1, 1), end_date, freq)
    assert result == ZERO

    # Test when asof equals start
    result = dcc.calculate_fraction(start_date, start_date, end_date, freq)
    expected = Decimal('0') / Decimal('366')
    assert result == expected

    # Test when asof equals end
    result = dcc.calculate_fraction(start_date, end_date, end_date, freq)
    expected = Decimal('366') / Decimal('366')
    assert result == expected


# LLM-generated content at query #83
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC instance for testing
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="SIMPLE",
        altnames={"simple"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=simple_fraction
    )

    # Test data
    principal = Money(1000, "USD")
    rate = Decimal("0.05")  # 5%
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 31)

    # Expected calculation: 1000 * 0.05 * (14/30) = 23.333...
    expected_interest = Money(Decimal("23.33333333333333333333333333"), "USD")

    # Test without end date (should use asof)
    result = dcc.interest(principal, rate, start, asof)
    assert result == expected_interest

    # Test with end date
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected_interest

    # Test with asof before start (should return 0)
    result = dcc.interest(principal, rate, start, datetime.date(2019, 12, 31))
    assert result == Money(ZERO, "USD")

    # Test with asof after end (should return 0)
    result = dcc.interest(principal, rate, start, datetime.date(2020, 2, 1), end)
    assert result == Money(ZERO, "USD")

    # Test with different frequency
    result = dcc.interest(principal, rate, start, asof, end, Decimal("12"))
    # The frequency shouldn't affect this simple fraction calculation
    assert result == expected_interest


# LLM-generated content at query #84
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in range
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in range (should subtract 1)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #85
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #86
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Create an instance of DCCRegistryMachinery
    registry = DCCRegistryMachinery()

    # Create a mock DCC object
    mock_dcc = DCC(
        name="TestDCC",
        altnames={"TestDCCAlt1", "TestDCCAlt2"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal(0.5)
    )

    # Register the mock DCC
    registry.register(mock_dcc)

    # Test finding by main name
    assert registry.find("TestDCC") == mock_dcc

    # Test finding by alternative name
    assert registry.find("TestDCCAlt1") == mock_dcc
    assert registry.find("TestDCCAlt2") == mock_dcc

    # Test finding with case insensitivity and whitespace
    assert registry.find(" testdcc ") == mock_dcc
    assert registry.find(" TESTDCCALT1 ") == mock_dcc

    # Test finding non-existent DCC
    assert registry.find("NonExistentDCC") is None

    # Test finding with empty string
    assert registry.find("") is None


# LLM-generated content at query #87
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Basic test with non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2: Test with February 29th (leap year)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3: Test with end of month dates
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4: Test with longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33055555555556')

    # Test case 5: Test with same start and asof dates
    start5 = datetime.date(2007, 12, 28)
    asof5 = datetime.date(2007, 12, 28)
    assert round(dcfc_30_e_360(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.00000000000000')

    # Test case 6: Test with start date having day 31
    start6 = datetime.date(2007, 1, 31)
    asof6 = datetime.date(2007, 2, 28)
    assert round(dcfc_30_e_360(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00000000000000')

    # Test case 7: Test with asof date having day 31
    start7 = datetime.date(2007, 1, 30)
    asof7 = datetime.date(2007, 2, 28)
    assert round(dcfc_30_e_360(start=start7, asof=asof7, end=asof7), 14) == Decimal('0.00000000000000')


# LLM-generated content at query #88
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end), 10) == Decimal('0.5245901639')

    # Test case 2: Full period (should return 1/freq)
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    asof = end
    freq = Decimal(1)
    assert dcfc_act_act_icma(start, asof, end, freq) == Decimal('1.0')

    # Test case 3: Half period (should return 0.5/freq)
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    asof = datetime.date(2020, 6, 30)
    freq = Decimal(1)
    expected = Decimal('0.5') / freq
    assert abs(dcfc_act_act_icma(start, asof, end, freq) - expected) < Decimal('0.0001')

    # Test case 4: Leap year period
    start = datetime.date(2020, 1, 1)  # Leap year
    end = datetime.date(2020, 12, 31)
    asof = datetime.date(2020, 3, 1)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result > Decimal('0.16') and result < Decimal('0.17')

    # Test case 5: Non-leap year period
    start = datetime.date(2019, 1, 1)  # Non-leap year
    end = datetime.date(2019, 12, 31)
    asof = datetime.date(2019, 3, 1)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result > Decimal('0.16') and result < Decimal('0.17')

    # Test case 6: Different frequency
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    asof = datetime.date(2020, 6, 30)
    freq = Decimal(2)
    expected = Decimal('0.25')  # 0.5 / 2
    assert abs(dcfc_act_act_icma(start, asof, end, freq) - expected) < Decimal('0.0001')

    # Test case 7: Edge case - same start and asof dates
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    asof = start
    freq = Decimal(1)
    assert dcfc_act_act_icma(start, asof, end, freq) == Decimal('0.0')

    # Test case 8: Edge case - asof date equals end date
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    asof = end
    freq = Decimal(1)
    assert dcfc_act_act_icma(start, asof, end, freq) == Decimal('1.0')

    # Test case 9: Cross-year period
    start = datetime.date(2019, 12, 1)
    end = datetime.date(2020, 11, 30)
    asof = datetime.date(2020, 6, 1)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result > Decimal('0.4') and result < Decimal('0.5')

    # Test case 10: No frequency provided (should use default 1)
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    asof = datetime.date(2020, 6, 30)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.5')
    assert abs(result - expected) < Decimal('0.0001')


# LLM-generated content at query #89
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a mock DCC instance with a simple calculation method
    def simple_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_calc
    )

    # Test case 1: Normal case where asof is between start and end
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 15)
    end = Date(2020, 1, 31)
    expected = Decimal("14") / Decimal("30")
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 2: asof is equal to start
    asof = start
    expected = Decimal("0")
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 3: asof is equal to end
    asof = end
    expected = Decimal("1")
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 4: asof is outside the range (before start)
    asof = Date(2019, 12, 31)
    expected = Decimal("0")
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 5: asof is outside the range (after end)
    asof = Date(2020, 2, 1)
    expected = Decimal("0")
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 6: With frequency parameter
    def freq_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days) * freq

    dcc_with_freq = DCC(
        name="Freq",
        altnames={"freq"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=freq_calc
    )
    freq = Decimal("2")
    expected = (Decimal("14") / Decimal("30")) * freq
    assert dcc_with_freq.calculate_fraction(start, asof, end, freq) == expected


# LLM-generated content at query #90
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #91
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Basic example from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 2: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.0')
    assert result == expected

    # Test case 3: Leap year scenario
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5') / Decimal(2)
    assert round(result, 10) == round(expected, 10)

    # Test case 4: Different frequency
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2021, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(4))
    expected = Decimal('60') / Decimal('366') / Decimal(4)
    assert round(result, 10) == round(expected, 10)

    # Test case 5: As of date equals end date
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('1.0') / Decimal(1)
    assert result == expected


# LLM-generated content at query #92
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #93
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, "USD")
    rate = Decimal("0.05")  # 5% interest rate
    start = Date(2020, 1, 1)
    asof = Date(2020, 4, 1)
    end = Date(2020, 7, 1)
    freq = 2  # Semi-annual payments
    eom = None

    # Create a mock DCC instance
    mock_dcc = DCC(
        name="MockDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.25")  # Mock fraction calculation
    )

    # Expected calculation:
    # Fraction is 0.25 (mocked)
    # Interest = principal * rate * fraction = 1000 * 0.05 * 0.25 = 12.5
    expected_interest = Money(Decimal("12.5"), "USD")

    # Call the method
    result = mock_dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected_interest
    assert result.currency == principal.currency


# LLM-generated content at query #94
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start, asof, asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start, asof, asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start, asof, asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start, asof, asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #95
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.10")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date provided
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(50, Currency("USD"))  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test with end date as asof
    result = dcc.interest(principal, rate, start, asof)
    expected = Money(50, Currency("USD"))  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames={"TestZeroAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: ZERO
    )
    result = dcc_zero.interest(principal, rate, start, asof, end)
    expected = Money(0, Currency("USD"))
    assert result == expected

    # Test with different dates (asof before start)
    result = dcc.interest(principal, rate, start, Date(2019, 12, 1), end)
    expected = Money(0, Currency("USD"))
    assert result == expected


# LLM-generated content at query #96
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Period with leap day
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Period crossing multiple years
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    # Test case 5: Same start and asof dates
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    result5 = dcfc_act_365_a(start5, asof5, asof5)
    assert result5 == Decimal('0.00000000000000')

    # Test case 6: Leap year period
    start6 = datetime.date(2020, 2, 28)
    asof6 = datetime.date(2020, 3, 1)
    result6 = dcfc_act_365_a(start6, asof6, asof6)
    assert round(result6, 14) == Decimal('0.00273224043716')


# LLM-generated content at query #97
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Period spanning multiple years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: Period with leap day
    start = datetime.date(2019, 2, 28)
    asof = datetime.date(2020, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.00273224043716')

    # Test case 7: Period with multiple leap days
    start = datetime.date(2016, 2, 29)
    asof = datetime.date(2020, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('4.00547945205479')

    # Test case 8: Period with no leap days
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.00000000000000')

    # Test case 9: Period with partial leap year
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16712328767123')

    # Test case 10: Period with partial non-leap year
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 3, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16438356164384')


# LLM-generated content at query #98
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #99
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Normal case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Both start and end dates are 31st
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.33055555555556')

    # Test case 5: Start date is 30th and end date is 31st
    start = datetime.date(2007, 11, 30)
    asof = datetime.date(2008, 12, 31)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.00000000000000')

    # Test case 6: Start date is 31st and end date is 30th
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 10, 30)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.99722222222222')


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Basic case with no adjustments
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st, start date is 30th
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st, should be adjusted to 30th
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st, asof date is 31st
    start = datetime.date(2007, 1, 31)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.0')

    # Test case 6: Start date is 30th, asof date is 31st (should adjust asof to 30th)
    start = datetime.date(2007, 1, 30)
    asof = datetime.date(2007, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.02777777777778')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case without day 31
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: Start day is 31, should be adjusted to 30
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 3: Start day is 30 and asof day is 31, asof should be adjusted to 30
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')

    # Test case 4: Start day is 31 and asof day is 31, both should be adjusted to 30
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 5: Same day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2007, 12, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.00000000000000')

    # Test case 6: Cross year boundary
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.02777777777778')


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year period (Feb 29)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Multi-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #4
#--------------------------

```python
def test_DCC_interest():
    # Setup
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.10")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)

    # Test with end date provided
    result_with_end = dcc.interest(principal, rate, start, asof, end)
    assert result_with_end == Money(50, Currency("USD"))

    # Test with end date as asof
    result_without_end = dcc.interest(principal, rate, start, asof)
    assert result_without_end == Money(50, Currency("USD"))

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames={"TestZeroAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: ZERO
    )
    result_zero = dcc_zero.interest(principal, rate, start, asof, end)
    assert result_zero == Money(0, Currency("USD"))

    # Test with invalid date range
    invalid_asof = Date(2022, 12, 31)
    result_invalid = dcc.interest(principal, rate, start, invalid_asof, end)
    assert result_invalid == Money(0, Currency("USD"))


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    assert round(dcfc_30_360_us(start=start, asof=asof, end=end), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    assert round(dcfc_30_360_us(start=start, asof=asof, end=end), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    assert round(dcfc_30_360_us(start=start, asof=asof, end=end), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    assert round(dcfc_30_360_us(start=start, asof=asof, end=end), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16939890710383')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with non-leap year
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Longer period with leap year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32876712328767')
    assert round(result4, 14) == expected4


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Basic test with example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result, 10) == Decimal('0.5245901639')

    # Test case 2: Full period (start to end)
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_end, end=ex1_end)
    assert result == Decimal('1.0')

    # Test case 3: Start date equals asof date
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_start, end=ex1_end)
    assert result == Decimal('0.0')

    # Test case 4: Different frequency
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=Decimal('2'))
    expected = Decimal('0.5245901639') / Decimal('2')
    assert round(result, 10) == round(expected, 10)

    # Test case 5: Leap year scenario
    leap_start = datetime.date(2020, 2, 28)
    leap_asof = datetime.date(2020, 8, 31)
    leap_end = datetime.date(2021, 2, 28)
    result = dcfc_act_act_icma(start=leap_start, asof=leap_asof, end=leap_end)
    assert result > Decimal('0.5') and result < Decimal('0.6')

    # Test case 6: Invalid date range (asof before start)
    result = dcfc_act_act_icma(start=ex1_asof, asof=ex1_start, end=ex1_end)
    assert result == Decimal('0.0')


# LLM-generated content at query #8
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a sample DCC instance
    def sample_calculate_fraction_method(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Sample",
        altnames={"SampleAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=sample_calculate_fraction_method
    )

    # Test case 1: Normal case
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 2)
    end = Date(2020, 1, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("1/30")
    assert result == expected

    # Test case 2: AsOf is start date
    asof = Date(2020, 1, 1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("1/30")
    assert result == expected

    # Test case 3: AsOf is end date
    asof = Date(2020, 1, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("1/30")
    assert result == expected

    # Test case 4: AsOf is before start date
    asof = Date(2019, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected

    # Test case 5: AsOf is after end date
    asof = Date(2020, 2, 1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected

    # Test case 6: With frequency parameter
    freq = Decimal("12")
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    expected = Decimal("1/30")
    assert result == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_DCC_interest():
    # Create a test DCC instance
    test_dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )

    # Test data
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.10")
    start_date = Date(2020, 1, 1)
    asof_date = Date(2020, 6, 1)
    end_date = Date(2020, 12, 31)

    # Expected result: principal * rate * fraction = 1000 * 0.10 * 0.5 = 50
    expected = Money(50, Currencies["USD"])

    # Test with end date
    result = test_dcc.interest(principal, rate, start_date, asof_date, end_date)
    assert result == expected

    # Test without end date (should use asof as end)
    result_no_end = test_dcc.interest(principal, rate, start_date, asof_date)
    assert result_no_end == expected

    # Test with different fraction
    test_dcc_2 = DCC(
        name="Test2",
        altnames={"TestAlt2"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.25")
    )
    expected_2 = Money(25, Currencies["USD"])
    result_2 = test_dcc_2.interest(principal, rate, start_date, asof_date, end_date)
    assert result_2 == expected_2

    # Test with zero fraction
    test_dcc_zero = DCC(
        name="TestZero",
        altnames={"TestAltZero"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: ZERO
    )
    expected_zero = Money(0, Currencies["USD"])
    result_zero = test_dcc_zero.interest(principal, rate, start_date, asof_date, end_date)
    assert result_zero == expected_zero


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start1, asof1, asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (should be excluded)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start2, asof2, asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start3, asof3, asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start4, asof4, asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_e_plus_360():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #14
#--------------------------

```python
def test_DCC_interest():
    # Create a mock DCC instance with a simple day count fraction calculation
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days) if (end - start).days != 0 else ZERO

    dcc = DCC(
        name="SIMPLE",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_fraction
    )

    # Test data
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 1)
    end = Date(2020, 1, 31)

    # Expected interest: 1000 * 0.05 * (30/365) ≈ 4.1096
    expected_interest = principal * rate * (Decimal(30) / Decimal(365))

    # Calculate interest
    result = dcc.interest(principal, rate, start, asof, end)

    # Assert
    assert result == expected_interest

    # Test with asof == end
    result = dcc.interest(principal, rate, start, end, end)
    assert result == principal * rate

    # Test with asof < start (should return 0)
    result = dcc.interest(principal, rate, start, Date(2019, 12, 31), end)
    assert result == ZERO

    # Test with asof > end (should return 0)
    result = dcc.interest(principal, rate, start, Date(2020, 2, 1), end)
    assert result == ZERO


# LLM-generated content at query #15
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    # Setup
    registry = DCCRegistryMachinery()
    test_dcc = DCC(
        name="TestDCC",
        altnames={"TestDCCAlt1", "TestDCCAlt2"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test successful registration
    registry.register(test_dcc)
    assert registry.find("TestDCC") == test_dcc
    assert registry.find("TestDCCAlt1") == test_dcc
    assert registry.find("TestDCCAlt2") == test_dcc

    # Test duplicate registration
    duplicate_dcc = DCC(
        name="TestDCC",
        altnames={"TestDCCAlt3"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError, match="Day count convention 'TestDCC' is already registered"):
        registry.register(duplicate_dcc)

    # Test alternative name conflict
    conflict_dcc = DCC(
        name="ConflictDCC",
        altnames={"TestDCCAlt1"},
        currencies={Currencies["GBP"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError, match="Day count convention 'ConflictDCC' is already registered"):
        registry.register(conflict_dcc)

    # Test case insensitive registration
    case_dcc = DCC(
        name="CASEDCC",
        altnames={"casedccalt"},
        currencies={Currencies["JPY"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.6")
    )
    registry.register(case_dcc)
    assert registry.find("casedcc") == case_dcc
    assert registry.find("CASEDCCALT") == case_dcc


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Basic test with exact 6 months period
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('0.5245901639')

    # Test case 2: Full year period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('0.4109589041')

    # Test case 3: Leap year period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('0.4109589041')

    # Test case 4: Short period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 31)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('0.5000000000')

    # Test case 5: With frequency parameter
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start, asof, end, Decimal(2)), 10) == Decimal('0.2622950820')

    # Test case 6: Edge case - same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('0.0000000000')

    # Test case 7: Edge case - same asof and end dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('1.0000000000')

    # Test case 8: Edge case - same start and end dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 1, 1)
    assert round(dcfc_act_act_icma(start, asof, end), 10) == Decimal('0.0000000000')


# LLM-generated content at query #17
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Test with valid dates
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)
    assert dcc.calculate_fraction(start, asof, end) == Decimal("0.5")

    # Test with asof before start
    assert dcc.calculate_fraction(start, Date(2019, 12, 31), end) == ZERO

    # Test with asof after end
    assert dcc.calculate_fraction(start, asof, Date(2020, 6, 1)) == ZERO

    # Test with asof equal to start
    assert dcc.calculate_fraction(start, start, end) == Decimal("0.5")

    # Test with asof equal to end
    assert dcc.calculate_fraction(start, end, end) == Decimal("0.5")

    # Test with freq parameter
    dcc_with_freq = DCC(
        name="TestFreq",
        altnames={"TestFreqAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.25") if f else Decimal("0.5")
    )
    assert dcc_with_freq.calculate_fraction(start, asof, end, Decimal("1")) == Decimal("0.25")


# LLM-generated content at query #18
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Create a test instance of DCCRegistryMachinery
    registry = DCCRegistryMachinery()

    # Create a test DCC object
    test_dcc = DCC(
        name="TestDCC",
        altnames={"AltTestDCC", "AnotherAltTestDCC"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Register the test DCC
    registry.register(test_dcc)

    # Test finding by exact name
    assert registry.find("TestDCC") == test_dcc

    # Test finding by alternative name
    assert registry.find("AltTestDCC") == test_dcc
    assert registry.find("AnotherAltTestDCC") == test_dcc

    # Test finding with case insensitivity and whitespace
    assert registry.find(" testdcc ") == test_dcc
    assert registry.find(" ALTTESTDCC ") == test_dcc

    # Test finding non-existent DCC
    assert registry.find("NonExistentDCC") is None
    assert registry.find("") is None

    # Test finding with partial name (should not match)
    assert registry.find("Test") is None
    assert registry.find("Alt") is None


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #21
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Test case 1: Check if dates are not in order
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    assert dcc.calculate_fraction(
        datetime.date(2023, 1, 1),
        datetime.date(2022, 12, 31),
        datetime.date(2023, 1, 2)
    ) == ZERO

    # Test case 2: Check if asof is before start
    assert dcc.calculate_fraction(
        datetime.date(2023, 1, 1),
        datetime.date(2022, 12, 31),
        datetime.date(2023, 1, 2)
    ) == ZERO

    # Test case 3: Check if asof is after end
    assert dcc.calculate_fraction(
        datetime.date(2023, 1, 1),
        datetime.date(2023, 1, 3),
        datetime.date(2023, 1, 2)
    ) == ZERO

    # Test case 4: Check if dates are in order and method is called correctly
    assert dcc.calculate_fraction(
        datetime.date(2023, 1, 1),
        datetime.date(2023, 1, 2),
        datetime.date(2023, 1, 3)
    ) == Decimal("0.5")


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16939890710383')
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == expected2

    # Test case 3: Non-leap year
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == expected3

    # Test case 4: Non-leap year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32876712328767')
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == expected4


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test1",
        altnames={"Alt1", "Alt2"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test2",
        altnames={"Alt3"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )

    # Test successful registration
    registry.register(dcc1)
    assert registry._find_strict("Test1") == dcc1
    assert registry._find_strict("Alt1") == dcc1
    assert registry._find_strict("Alt2") == dcc1

    # Test duplicate main name registration
    with pytest.raises(TypeError, match="Day count convention 'Test1' is already registered"):
        registry.register(dcc1)

    # Test duplicate alternative name registration
    dcc3 = DCC(
        name="Test3",
        altnames={"Alt1"},
        currencies={Currencies["GBP"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.2")
    )
    with pytest.raises(TypeError, match="Day count convention 'Test3' is already registered"):
        registry.register(dcc3)

    # Test successful registration of another DCC
    registry.register(dcc2)
    assert registry._find_strict("Test2") == dcc2
    assert registry._find_strict("Alt3") == dcc2

    # Test find method
    assert registry.find("test1") == dcc1
    assert registry.find("TEST2") == dcc2
    assert registry.find("alt2") == dcc1
    assert registry.find("ALT3") == dcc2


# LLM-generated content at query #26
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a mock DCC instance with a simple calculation method
    def simple_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=simple_calc
    )

    # Test case 1: Normal case where start <= asof <= end
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 15)
    end = Date(2020, 1, 31)
    result = dcc.calculate_fraction(start, asof, end)
    expected = Decimal("0.4838709677419355")
    assert result == expected

    # Test case 2: asof is equal to start
    result = dcc.calculate_fraction(start, start, end)
    expected = Decimal("0.0")
    assert result == expected

    # Test case 3: asof is equal to end
    result = dcc.calculate_fraction(start, end, end)
    expected = Decimal("1.0")
    assert result == expected

    # Test case 4: asof is before start (should return 0)
    result = dcc.calculate_fraction(start, Date(2019, 12, 31), end)
    expected = Decimal("0.0")
    assert result == expected

    # Test case 5: asof is after end (should return 0)
    result = dcc.calculate_fraction(start, Date(2020, 2, 1), end)
    expected = Decimal("0.0")
    assert result == expected

    # Test case 6: With frequency parameter
    def freq_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days) * freq

    dcc_with_freq = DCC(
        name="TEST_FREQ",
        altnames={"TEST_FREQ_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=freq_calc
    )

    result = dcc_with_freq.calculate_fraction(start, asof, end, Decimal("2"))
    expected = Decimal("0.967741935483871")
    assert result == expected


# LLM-generated content at query #27
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Setup test data
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 1, 15)
    end_date = datetime.date(2020, 1, 31)
    freq = Decimal('12')

    # Create a mock DCC instance
    mock_dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )

    # Test normal case
    result = mock_dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    expected = Decimal('14') / Decimal('360')
    assert result == expected

    # Test when asof is before start
    result = mock_dcc.calculate_fraction(start_date, datetime.date(2019, 12, 31), end_date, freq)
    assert result == ZERO

    # Test when asof is after end
    result = mock_dcc.calculate_fraction(start_date, datetime.date(2020, 2, 1), end_date, freq)
    assert result == ZERO

    # Test when asof equals start
    result = mock_dcc.calculate_fraction(start_date, start_date, end_date, freq)
    expected = Decimal('0') / Decimal('360')
    assert result == expected

    # Test when asof equals end
    result = mock_dcc.calculate_fraction(start_date, end_date, end_date, freq)
    expected = Decimal('30') / Decimal('360')
    assert result == expected


# LLM-generated content at query #28
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    # Test case 5: Same day (edge case)
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    result5 = dcfc_act_365_a(start5, asof5, asof5)
    assert result5 == Decimal('0')

    # Test case 6: Full year without leap day
    start6 = datetime.date(2019, 1, 1)
    asof6 = datetime.date(2019, 12, 31)
    result6 = dcfc_act_365_a(start6, asof6, asof6)
    assert round(result6, 14) == Decimal('1.00000000000000')

    # Test case 7: Full year with leap day
    start7 = datetime.date(2020, 1, 1)
    asof7 = datetime.date(2020, 12, 31)
    result7 = dcfc_act_365_a(start7, asof7, asof7)
    assert round(result7, 14) == Decimal('1.00000000000000')


# LLM-generated content at query #29
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Basic test with dates from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 2: Same start and asof dates
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('0.0000000000')

    # Test case 3: Same start and end dates
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2019, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('0.0000000000')

    # Test case 4: As of date is after end date
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('0.0000000000')

    # Test case 5: Test with frequency parameter
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected

    # Test case 6: Test with leap year
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 8, 28)
    end = datetime.date(2021, 2, 28)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.5000000000')
    assert round(result, 10) == expected

    # Test case 7: Test with different end date
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2021, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected


# LLM-generated content at query #30
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2014, 1, 1)
    asof = Date(2015, 1, 1)
    end = Date(2015, 7, 1)
    freq = Decimal(2)
    eom = None

    # Create a mock DCC object with a simple calculate_fraction_method
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction_method
    )

    # Call the coupon method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assert the result
    expected = principal * rate * Decimal("0.5")
    assert result == expected


# LLM-generated content at query #31
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 1, 10)

    # Test for the first day (asof = start)
    asof = start
    expected = Decimal("0.1")  # 1 day out of 10
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == expected

    # Test for a middle day
    asof = datetime.date(2020, 1, 5)
    expected = Decimal("0.1")  # 1 day out of 10
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == expected

    # Test for the last day (asof = end)
    asof = end
    expected = Decimal("0.1")  # 1 day out of 10
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == expected

    # Test when asof is before start (should return 0)
    asof = datetime.date(2019, 12, 31)
    expected = Decimal("0")
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == expected

    # Test when asof is after end (should return 0)
    asof = datetime.date(2020, 1, 11)
    expected = Decimal("0")
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == expected


# LLM-generated content at query #32
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start1, asof1, asof1)
    expected1 = Decimal('0.16939890710383')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start2, asof2, asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Non-leap year with longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start3, asof3, asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Non-leap year with longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start4, asof4, asof4)
    expected4 = Decimal('1.32876712328767')
    assert round(result4, 14) == expected4


# LLM-generated content at query #33
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Mock",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )

    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 1, 10)
    asof = datetime.date(2020, 1, 5)

    # Test normal case
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is start date
    asof = start
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is end date
    asof = end
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is before start
    asof = datetime.date(2019, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9 (since asof-1 is before start)
    assert result == expected

    # Test when asof is after end
    asof = datetime.date(2020, 1, 11)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected


# LLM-generated content at query #34
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case without day adjustments
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st, should adjust to 30th
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st, should adjust to 30th
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4: Both start and end dates require adjustment
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 30th and end date is 31st, end date should adjust to 30th
    start5 = datetime.date(2007, 4, 30)
    asof5 = datetime.date(2007, 5, 31)
    assert round(dcfc_30_360_isda(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.02777777777778')

    # Test case 6: Same start and end date
    start6 = datetime.date(2007, 12, 15)
    asof6 = datetime.date(2007, 12, 15)
    assert round(dcfc_30_360_isda(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00000000000000')


# LLM-generated content at query #35
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Test basic functionality with a mock DCC instance
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction_method
    )

    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test valid date range
    assert dcc.calculate_fraction(start, asof, end) == Decimal("0.5")

    # Test with freq parameter
    assert dcc.calculate_fraction(start, asof, end, Decimal("2")) == Decimal("0.5")

    # Test with asof before start (should return ZERO)
    assert dcc.calculate_fraction(start, Date(2019, 12, 1), end) == ZERO

    # Test with asof after end (should return ZERO)
    assert dcc.calculate_fraction(start, Date(2021, 1, 1), end) == ZERO

    # Test with asof equal to start
    assert dcc.calculate_fraction(start, start, end) == Decimal("0.5")

    # Test with asof equal to end
    assert dcc.calculate_fraction(start, end, end) == Decimal("0.5")

    # Test with all dates equal
    assert dcc.calculate_fraction(start, start, start) == Decimal("0.5")


# LLM-generated content at query #36
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16986301369863')
    assert round(dcfc_act_365_a(start=start1, asof=asof1, end=asof1), 14) == expected1

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    assert round(dcfc_act_365_a(start=start2, asof=asof2, end=asof2), 14) == expected2

    # Test case 3: Longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    assert round(dcfc_act_365_a(start=start3, asof=asof3, end=asof3), 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32513661202186')
    assert round(dcfc_act_365_a(start=start4, asof=asof4, end=asof4), 14) == expected4


# LLM-generated content at query #37
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)
    freq = Decimal("2")
    eom = None

    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test coupon calculation
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Expected result: principal * rate * fraction = 1000 * 0.05 * 0.5 = 25
    expected = Money(25, Currency("USD"))

    assert result == expected


# LLM-generated content at query #38
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.10")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date provided
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(50, Currency("USD"))  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test with end date not provided (should use asof)
    result_no_end = dcc.interest(principal, rate, start, asof)
    assert result_no_end == expected

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames={"TestZeroAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: ZERO
    )
    result_zero = dcc_zero.interest(principal, rate, start, asof, end)
    assert result_zero == Money(0, Currency("USD"))

    # Test with different dates
    start2 = Date(2021, 1, 1)
    asof2 = Date(2021, 3, 1)
    end2 = Date(2021, 6, 1)
    result2 = dcc.interest(principal, rate, start2, asof2, end2)
    assert result2 == expected

    # Test with different principal and rate
    principal3 = Money(2000, Currency("USD"))
    rate3 = Decimal("0.05")
    result3 = dcc.interest(principal3, rate3, start, asof, end)
    expected3 = Money(50, Currency("USD"))  # 2000 * 0.05 * 0.5
    assert result3 == expected3


# LLM-generated content at query #39
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Example from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 2: Full period
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('1.0')
    assert result == expected

    # Test case 3: Half period
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5')
    assert result == expected

    # Test case 4: With frequency
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected

    # Test case 5: Leap year
    start = datetime.date(2020, 3, 2)
    asof = datetime.date(2020, 9, 10)
    end = datetime.date(2021, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 6: Different end date
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2021, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected


# LLM-generated content at query #40
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a mock DCC instance with a simple calculation method
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=simple_fraction
    )

    # Test normal case
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 15)
    end = Date(2020, 1, 31)
    assert dcc.calculate_fraction(start, asof, end) == Decimal("0.4838709677419355")

    # Test when asof is before start
    assert dcc.calculate_fraction(start, Date(2019, 12, 31), end) == ZERO

    # Test when asof is after end
    assert dcc.calculate_fraction(start, Date(2020, 2, 1), end) == ZERO

    # Test when asof equals start
    assert dcc.calculate_fraction(start, start, end) == ZERO

    # Test when asof equals end
    assert dcc.calculate_fraction(start, end, end) == ONE

    # Test with frequency parameter
    def freq_fraction(start, asof, end, freq):
        if freq is None:
            return ZERO
        return Decimal((asof - start).days) * freq / Decimal(360)

    dcc_with_freq = DCC(
        name="Freq",
        altnames={"freq"},
        currencies=_as_ccys({"EUR"}),
        calculate_fraction_method=freq_fraction
    )

    assert dcc_with_freq.calculate_fraction(start, asof, end, Decimal("2")) == Decimal("0.08333333333333333333333333333")


# LLM-generated content at query #41
#--------------------------

```python
def test_DCC_interest():
    # Setup
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.1")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(50, Currency("USD"))

    # Test without end date (asof is used as end)
    result = dcc.interest(principal, rate, start, asof)
    assert result == Money(50, Currency("USD"))

    # Test with zero fraction
    dcc_zero = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: ZERO
    )
    result = dcc_zero.interest(principal, rate, start, asof, end)
    assert result == Money(0, Currency("USD"))

    # Test with invalid date range (should return zero)
    result = dcc.interest(principal, rate, asof, start, end)
    assert result == Money(0, Currency("USD"))


# LLM-generated content at query #42
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 4, 1)
    end = Date(2020, 7, 1)
    freq = 2  # Semi-annual
    eom = None

    # Create a mock DCC instance with a simple day count fraction method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal(360)

    dcc = DCC(
        name="Mock",
        altnames={"MockAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Expected calculation:
    # Previous payment date: 2020-01-01
    # Next payment date: 2020-07-01
    # Days between 2020-01-01 and 2020-04-01: 91 days
    # Fraction: 91/360
    # Interest: 1000 * 0.05 * (91/360) = 12.638888...
    expected_interest = Money(Decimal("12.63888888888888888888888889"), Currency("USD"))

    # Test the coupon method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    assert result == expected_interest


# LLM-generated content at query #43
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #44
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Setup test data
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    asof_date = datetime.date(2020, 6, 15)

    # Create a simple DCC instance for testing
    def simple_fraction(start, asof, end, freq):
        total_days = (end - start).days
        if total_days == 0:
            return ZERO
        return Decimal((asof - start).days) / Decimal(total_days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_fraction
    )

    # Test normal case
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    expected_fraction = simple_fraction(start_date, asof_date, end_date) - simple_fraction(start_date, asof_date - datetime.timedelta(days=1), end_date)
    assert daily_fraction == expected_fraction

    # Test when asof is start date
    daily_fraction_start = dcc.calculate_daily_fraction(start_date, start_date, end_date)
    assert daily_fraction_start == simple_fraction(start_date, start_date, end_date)

    # Test when asof is day after start
    daily_fraction_next_day = dcc.calculate_daily_fraction(start_date, start_date + datetime.timedelta(days=1), end_date)
    expected_next_day = simple_fraction(start_date, start_date + datetime.timedelta(days=1), end_date) - simple_fraction(start_date, start_date, end_date)
    assert daily_fraction_next_day == expected_next_day

    # Test when asof is end date
    daily_fraction_end = dcc.calculate_daily_fraction(start_date, end_date, end_date)
    expected_end = simple_fraction(start_date, end_date, end_date) - simple_fraction(start_date, end_date - datetime.timedelta(days=1), end_date)
    assert daily_fraction_end == expected_end

    # Test when asof is before start (should return 0)
    before_start = start_date - datetime.timedelta(days=1)
    daily_fraction_before = dcc.calculate_daily_fraction(start_date, before_start, end_date)
    assert daily_fraction_before == ZERO

    # Test when asof is after end (should return 0)
    after_end = end_date + datetime.timedelta(days=1)
    daily_fraction_after = dcc.calculate_daily_fraction(start_date, after_end, end_date)
    assert daily_fraction_after == ZERO


# LLM-generated content at query #45
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16939890710383')
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == expected3

    # Test case 4: Longer period without leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32876712328767')
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == expected4


# LLM-generated content at query #46
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.33055555555556')


# LLM-generated content at query #47
#--------------------------

```python
def test_DCC_coupon():
    # Create a DCC instance for testing
    dcc = DCC(
        name="TestDCC",
        altnames={"Test"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test data
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None

    # Expected result
    expected = principal * rate * Decimal("0.5")

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assert the result
    assert result == expected


# LLM-generated content at query #48
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #49
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Longer period with leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Another longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('0')

    # Test case 6: One day difference
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 2)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('0.00273972602740')

    # Test case 7: Full non-leap year
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 12, 31)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('1')

    # Test case 8: Full leap year
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2016, 12, 31)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.00273972602740')

    # Test case 9: Multiple years with leap day
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2018, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('3.00821917808219')

    # Test case 10: End date before asof date (should return 0)
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 1)
    end = datetime.date(2016, 12, 31)
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('0')


# LLM-generated content at query #50
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    start = Date(2020, 1, 1)
    end = Date(2020, 1, 10)
    asof = Date(2020, 1, 5)

    # Test daily fraction for a normal day
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)
    assert daily_fraction == Decimal("0.1")

    # Test daily fraction for the first day
    daily_fraction_first = dcc.calculate_daily_fraction(start, start, end)
    assert daily_fraction_first == Decimal("0.1")

    # Test daily fraction for the day after start
    daily_fraction_second = dcc.calculate_daily_fraction(start, Date(2020, 1, 2), end)
    assert daily_fraction_second == Decimal("0.1")

    # Test daily fraction when asof is before start
    daily_fraction_before = dcc.calculate_daily_fraction(start, Date(2019, 12, 31), end)
    assert daily_fraction_before == Decimal("0")

    # Test daily fraction when asof is after end
    daily_fraction_after = dcc.calculate_daily_fraction(start, Date(2020, 1, 11), end)
    assert daily_fraction_after == Decimal("0")


# LLM-generated content at query #51
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(50, "USD")  # 1000 * 0.10 * 0.5

    # Test without end date (asof used as end)
    result = dcc.interest(principal, rate, start, asof)
    assert result == Money(50, "USD")  # 1000 * 0.10 * 0.5

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames={"TestZeroAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda start, asof, end, freq: ZERO
    )
    result = dcc_zero.interest(principal, rate, start, asof, end)
    assert result == Money(0, "USD")

    # Test with different principal and rate
    principal_large = Money(10000, "USD")
    rate_small = Decimal("0.01")
    result = dcc.interest(principal_large, rate_small, start, asof, end)
    assert result == Money(50, "USD")  # 10000 * 0.01 * 0.5


# LLM-generated content at query #52
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


# LLM-generated content at query #53
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


# LLM-generated content at query #54
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2: Example from docstring
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Example from docstring
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Example from docstring
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

    # Additional test cases
    # Test case 5: Same start and asof dates
    same_start_asof = datetime.date(2020, 1, 1)
    assert dcfc_30_360_us(start=same_start_asof, asof=same_start_asof, end=same_start_asof) == Decimal('0')

    # Test case 6: Start date is last day of month
    start_last_day = datetime.date(2020, 1, 31)
    asof_feb = datetime.date(2020, 2, 28)
    assert round(dcfc_30_360_us(start=start_last_day, asof=asof_feb, end=asof_feb), 14) == Decimal('0.08333333333333')

    # Test case 7: As of date is last day of month
    start_jan = datetime.date(2020, 1, 15)
    asof_last_day = datetime.date(2020, 2, 29)
    assert round(dcfc_30_360_us(start=start_jan, asof=asof_last_day, end=asof_last_day), 14) == Decimal('0.13888888888889')

    # Test case 8: Both dates are last day of month
    start_last_day = datetime.date(2020, 1, 31)
    asof_last_day = datetime.date(2020, 2, 29)
    assert round(dcfc_30_360_us(start=start_last_day, asof=asof_last_day, end=asof_last_day), 14) == Decimal('0.08333333333333')

    # Test case 9: As of date is 31st and start date is 30th
    start_30th = datetime.date(2020, 1, 30)
    asof_31st = datetime.date(2020, 2, 29)
    assert round(dcfc_30_360_us(start=start_30th, asof=asof_31st, end=asof_31st), 14) == Decimal('0.08333333333333')

    # Test case 10: As of date is 31st and start date is 31st
    start_31st = datetime.date(2020, 1, 31)
    asof_31st = datetime.date(2020, 2, 29)
    assert round(dcfc_30_360_us(start=start_31st, asof=asof_31st, end=asof_31st), 14) == Decimal('0.08333333333333')


# LLM-generated content at query #55
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.16986301369863')

    # Test case 3
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08219178082192')

    # Test case 4
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32602739726027')


# LLM-generated content at query #56
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Start date is 30th and end date is 31st
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Test case 5: Start and end date are the same
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: Start date is 31st and end date is 30th
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.0')

    # Test case 7: Start date is 30th and end date is 31st (same month)
    start = datetime.date(2020, 1, 30)
    asof = datetime.date(2020, 1, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.00277777777778')


# LLM-generated content at query #57
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period without leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    expected4 = Decimal('1.32513661202186')
    assert round(result4, 14) == expected4


# LLM-generated content at query #58
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1: Standard case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: Leap year case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Long period case
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Another long period case
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test case: End of month handling
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.0')

    # Additional test case: Different end date
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 2, 15)
    end = datetime.date(2020, 3, 15)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #59
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16939890710383')
    assert round(result1, 14) == expected1

    # Test case 2: Leap year
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.17213114754098')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08196721311475')
    assert round(result3, 14) == expected3

    # Test case 4: Cross-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32876712328767')
    assert round(result4, 14) == expected4


# LLM-generated content at query #60
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


# LLM-generated content at query #61
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Basic test with dates from the docstring
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Date range with leap day
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == Decimal('0.16986301369863')

    # Test case 3: Longer date range
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == Decimal('1.08219178082192')

    # Test case 4: Another longer date range
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == Decimal('1.32602739726027')

    # Test case 5: Same start and asof date
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    result5 = dcfc_nl_365(start=start5, asof=asof5, end=asof5)
    assert result5 == Decimal('0')

    # Test case 6: Full year without leap day
    start6 = datetime.date(2019, 1, 1)
    asof6 = datetime.date(2019, 12, 31)
    result6 = dcfc_nl_365(start=start6, asof=asof6, end=asof6)
    assert result6 == Decimal('1')

    # Test case 7: Full year with leap day (should still be 1)
    start7 = datetime.date(2020, 1, 1)
    asof7 = datetime.date(2020, 12, 31)
    result7 = dcfc_nl_365(start=start7, asof=asof7, end=asof7)
    assert result7 == Decimal('1')


# LLM-generated content at query #62
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #63
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Mixed leap and non-leap years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Longer period with leap year
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same day
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('0')

    # Test case 6: Full non-leap year
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 12, 31)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('1')

    # Test case 7: Full leap year
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('1')


# LLM-generated content at query #64
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')


# LLM-generated content at query #65
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Additional test case with start day = 31
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.0')

    # Additional test case with start day = 30 and asof day = 31
    start = datetime.date(2008, 1, 30)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.02777777777778')


# LLM-generated content at query #66
#--------------------------

```python
def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Longer period without leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    # Test case 5: Same day
    start5 = datetime.date(2008, 2, 29)
    asof5 = datetime.date(2008, 2, 29)
    result5 = dcfc_act_365_a(start5, asof5, asof5)
    assert result5 == Decimal('0.00000000000000')


# LLM-generated content at query #67
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(1)

    # Create a mock DCC object with a simple calculation method
    def simple_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal(360)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=simple_calc
    )

    # Test interest calculation
    expected_interest = principal * rate * simple_calc(start, asof, end, freq)
    actual_interest = dcc.interest(principal, rate, start, asof, end, freq)

    assert actual_interest == expected_interest

    # Test with default end date (asof)
    expected_interest_default_end = principal * rate * simple_calc(start, asof, asof, freq)
    actual_interest_default_end = dcc.interest(principal, rate, start, asof, None, freq)

    assert actual_interest_default_end == expected_interest_default_end

    # Test with invalid date range (should return 0)
    invalid_asof = datetime.date(2019, 12, 31)
    assert dcc.interest(principal, rate, start, invalid_asof, end, freq) == Money(0, "USD")


# LLM-generated content at query #68
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_act(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_act(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == Decimal('0.17216108990194')

    # Test case 3: Period crossing multiple years
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_act(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == Decimal('1.08243131970956')

    # Test case 4: Longer period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_act(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    result5 = dcfc_act_act(start=start5, asof=asof5, end=asof5)
    assert result5 == Decimal('0')

    # Test case 6: Period with multiple leap years
    start6 = datetime.date(2016, 1, 1)
    asof6 = datetime.date(2020, 12, 31)
    result6 = dcfc_act_act(start=start6, asof=asof6, end=asof6)
    assert round(result6, 14) == Decimal('4.99863013698630')

    # Test case 7: Period with no leap years
    start7 = datetime.date(2017, 1, 1)
    asof7 = datetime.date(2019, 12, 31)
    result7 = dcfc_act_act(start=start7, asof=asof7, end=asof7)
    assert round(result7, 14) == Decimal('2.99863013698630')

    # Test case 8: Period with one leap day
    start8 = datetime.date(2019, 2, 28)
    asof8 = datetime.date(2020, 2, 29)
    result8 = dcfc_act_act(start=start8, asof=asof8, end=asof8)
    assert round(result8, 14) == Decimal('1.00273224043716')


# LLM-generated content at query #69
#--------------------------

```python
def test_dcfc_nl_365():
    # Test cases from the docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')

    # Additional test cases
    # Test with leap day in the period
    leap_start, leap_asof = datetime.date(2020, 2, 28), datetime.date(2020, 3, 1)
    assert round(dcfc_nl_365(start=leap_start, asof=leap_asof, end=leap_asof), 14) == Decimal('0.00273972602740')

    # Test without leap day in the period
    non_leap_start, non_leap_asof = datetime.date(2021, 2, 28), datetime.date(2021, 3, 1)
    assert round(dcfc_nl_365(start=non_leap_start, asof=non_leap_asof, end=non_leap_asof), 14) == Decimal('0.00273972602740')

    # Test with a full year period
    year_start, year_asof = datetime.date(2021, 1, 1), datetime.date(2021, 12, 31)
    assert round(dcfc_nl_365(start=year_start, asof=year_asof, end=year_asof), 14) == Decimal('1.00000000000000')

    # Test with a full leap year period
    leap_year_start, leap_year_asof = datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)
    assert round(dcfc_nl_365(start=leap_year_start, asof=leap_year_asof, end=leap_year_asof), 14) == Decimal('0.99726027397260')


# LLM-generated content at query #70
#--------------------------

```python
def test_dcfc_30_360_german():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #71
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Period spanning multiple years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Period with leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: Period with one day difference
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.00273972602740')

    # Test case 7: Period with one year difference (non-leap year)
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1.0')

    # Test case 8: Period with one year difference (leap year)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1.0')

    # Test case 9: Period with multiple leap years
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2021, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('5.0')

    # Test case 10: Period with end date before start date
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2019, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


# LLM-generated content at query #72
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #73
#--------------------------

```python
def test_dcfc_nl_365():
    # Test cases with expected results from the docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')

    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')

    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')

    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')

    # Additional test case with no leap day
    start, asof = datetime.date(2021, 1, 1), datetime.date(2021, 6, 30)
    assert round(dcfc_nl_365(start=start, asof=asof, end=asof), 14) == Decimal('0.5')

    # Additional test case with leap day
    start, asof = datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)
    assert round(dcfc_nl_365(start=start, asof=asof, end=asof), 14) == Decimal('1.0')


# LLM-generated content at query #74
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=start, asof=asof, end=end), 10) == Decimal('0.5245901639')

    # Test case 2 - same start and asof
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert dcfc_act_act_icma(start=start, asof=asof, end=end) == Decimal('0')

    # Test case 3 - same start and end
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 1, 1)
    assert dcfc_act_act_icma(start=start, asof=asof, end=end) == Decimal('0')

    # Test case 4 - leap year
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    assert round(result, 10) == Decimal('0.4123287671')

    # Test case 5 - non-leap year
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 6, 1)
    end = datetime.date(2020, 1, 1)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    assert round(result, 10) == Decimal('0.4109589041')


# LLM-generated content at query #75
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Example from docstring
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 2: Full period (start to end)
    result = dcfc_act_act_icma(start, end, end)
    expected = Decimal('1.0')
    assert result == expected

    # Test case 3: Beginning of period
    result = dcfc_act_act_icma(start, start, end)
    expected = Decimal('0.0')
    assert result == expected

    # Test case 4: Different frequency
    result = dcfc_act_act_icma(start, asof, end, Decimal(2))
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected

    # Test case 5: Leap year scenario
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.4109589041')
    assert round(result, 10) == expected

    # Test case 6: Short period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 1, 31)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.0303030303')
    assert round(result, 10) == expected


# LLM-generated content at query #76
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period but not counted
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #77
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple day count fraction method
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=simple_fraction
    )

    # Test case 1: Normal case where asof is within the period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 1, 10)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test case 2: asof is the start date
    asof = start
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test case 3: asof is the day after start
    asof = start + datetime.timedelta(days=1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test case 4: asof is the end date
    asof = end
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test case 5: asof is outside the period (before start)
    asof = start - datetime.timedelta(days=1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected

    # Test case 6: asof is outside the period (after end)
    asof = end + datetime.timedelta(days=1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0")
    assert result == expected


# LLM-generated content at query #78
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Test1_Alt"}, set(), lambda s, a, e, f: Decimal(1))
    dcc2 = DCC("Test2", {"Test2_Alt"}, set(), lambda s, a, e, f: Decimal(2))

    # Register DCCs
    registry.register(dcc1)
    registry.register(dcc2)

    # Test finding by main name
    assert registry.find("Test1") == dcc1
    assert registry.find("Test2") == dcc2

    # Test finding by alternative name
    assert registry.find("Test1_Alt") == dcc1
    assert registry.find("Test2_Alt") == dcc2

    # Test finding with stripped and uppercased name
    assert registry.find(" test1 ") == dcc1
    assert registry.find(" test2_alt ") == dcc2

    # Test finding non-existent DCC
    assert registry.find("NonExistent") is None


# LLM-generated content at query #79
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in period (but not counted)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #80
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


# LLM-generated content at query #81
#--------------------------

```python
def test_DCC_coupon():
    # Create a DCC instance for testing
    dcc = DCC(
        name="TestDCC",
        altnames={"Test"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test data
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = Decimal("2")
    eom = None

    # Expected result
    expected_interest = principal * rate * Decimal("0.5")

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected_interest

    # Test with eom parameter
    start_eom = datetime.date(2020, 1, 15)
    asof_eom = datetime.date(2020, 6, 15)
    eom = 15
    result_eom = dcc.coupon(principal, rate, start_eom, asof_eom, end, freq, eom)
    assert result_eom == expected_interest

    # Test with different frequency
    freq_4 = Decimal("4")
    result_freq4 = dcc.coupon(principal, rate, start, asof, end, freq_4, eom)
    expected_interest_freq4 = principal * rate * Decimal("0.5")
    assert result_freq4 == expected_interest_freq4


# LLM-generated content at query #82
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (should be excluded)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #83
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=start1, asof=asof1, end=asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: With leap day (but NL/365 ignores it)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=start2, asof=asof2, end=asof2)
    expected2 = Decimal('0.16986301369863')
    assert round(result2, 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=start3, asof=asof3, end=asof3)
    expected3 = Decimal('1.08219178082192')
    assert round(result3, 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #84
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


# LLM-generated content at query #85
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case with no adjustments
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st, start date is 30th
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st, adjusted to 30th
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4: Longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st, asof date is 31st (both adjusted to 30th)
    start5 = datetime.date(2020, 1, 31)
    asof5 = datetime.date(2020, 2, 29)  # Leap year, but adjusted to 30th
    assert round(dcfc_30_360_isda(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.02777777777778')

    # Test case 6: Same start and asof date
    start6 = datetime.date(2020, 3, 15)
    asof6 = datetime.date(2020, 3, 15)
    assert round(dcfc_30_360_isda(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00000000000000')


# LLM-generated content at query #86
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 10)
    asof = datetime.date(2023, 1, 5)

    # Test normal case
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal(1) / Decimal(9)  # (5-4)/9
    assert daily_fraction == expected

    # Test when asof is start date
    daily_fraction = dcc.calculate_daily_fraction(start, start, end)
    expected = Decimal(1) / Decimal(9)  # (0-(-1))/9
    assert daily_fraction == expected

    # Test when asof is end date
    daily_fraction = dcc.calculate_daily_fraction(start, end, end)
    expected = Decimal(1) / Decimal(9)  # (9-8)/9
    assert daily_fraction == expected

    # Test when asof is before start (should return 0)
    asof_before_start = datetime.date(2022, 12, 31)
    daily_fraction = dcc.calculate_daily_fraction(start, asof_before_start, end)
    assert daily_fraction == ZERO


# LLM-generated content at query #87
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.10")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(50, Currency("USD"))

    # Test without end date (asof used as end)
    result = dcc.interest(principal, rate, start, asof)
    assert result == Money(50, Currency("USD"))

    # Test with zero fraction
    dcc_zero = DCC(
        name="TestZero",
        altnames={"TestZeroAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: ZERO
    )
    result = dcc_zero.interest(principal, rate, start, asof, end)
    assert result == Money(0, Currency("USD"))

    # Test with invalid date range (should return zero)
    invalid_asof = Date(2019, 12, 31)
    result = dcc.interest(principal, rate, start, invalid_asof, end)
    assert result == Money(0, Currency("USD"))


# LLM-generated content at query #88
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Standard case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: February 29th case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: October 31st to November 30th
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33055555555556')

    # Test case 5: Same day
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: End date adjustment (31st to 30th)
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.08333333333333')

    # Test case 7: As-of date adjustment (31st to 30th)
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 3, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.20833333333333')


# LLM-generated content at query #89
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")  # 5% interest rate
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 30)
    end = Date(2020, 12, 31)
    freq = Decimal("1")  # Annual frequency

    # Create a mock DCC instance with a simple day count fraction method
    def mock_calculate_fraction(start, asof, end, freq):
        total_days = (end - start).days
        accrued_days = (asof - start).days
        return Decimal(accrued_days) / Decimal(total_days)

    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Calculate expected interest manually
    total_days = (end - start).days
    accrued_days = (asof - start).days
    expected_fraction = Decimal(accrued_days) / Decimal(total_days)
    expected_interest = principal * rate * expected_fraction

    # Test the interest method
    result = dcc.interest(principal, rate, start, asof, end, freq)

    assert result == expected_interest


# LLM-generated content at query #90
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Period spanning multiple years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Period with leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

    # Test case 6: Period with multiple leap years
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2020, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('4.0') + Decimal('1') / Decimal('366')  # 3 leap years (2016, 2020) and 2 non-leap years
    assert round(result, 14) == round(expected, 14)


# LLM-generated content at query #91
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33055555555556')


# LLM-generated content at query #92
#--------------------------

```python
def test_DCC_interest():
    # Setup
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2023, 12, 31)
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")  # 5%

    # Create a mock DCC instance with a simple day count fraction method
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="Simple",
        altnames={"simple"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=simple_fraction
    )

    # Test interest calculation
    interest = dcc.interest(principal, rate, start_date, asof_date, end_date)

    # Expected calculation: (181 days / 365 days) * 1000 * 0.05
    expected_days = (asof_date - start_date).days
    total_days = (end_date - start_date).days
    expected_fraction = Decimal(expected_days) / Decimal(total_days)
    expected_interest = principal * rate * expected_fraction

    assert interest == expected_interest

    # Test with asof_date equal to end_date
    interest_same_date = dcc.interest(principal, rate, start_date, end_date, end_date)
    assert interest_same_date == principal * rate * Decimal("1")

    # Test with asof_date before start_date (should return 0)
    interest_before_start = dcc.interest(principal, rate, start_date, datetime.date(2022, 12, 1), end_date)
    assert interest_before_start == Money(0, Currencies["USD"])

    # Test with asof_date after end_date (should return same as end_date)
    interest_after_end = dcc.interest(principal, rate, start_date, datetime.date(2024, 1, 1), end_date)
    assert interest_after_end == principal * rate * Decimal("1")


# LLM-generated content at query #93
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC instance for testing
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="SIMPLE",
        altnames={"simple"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=simple_fraction
    )

    # Test data
    principal = Money(1000, "USD")
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 1)
    end = Date(2020, 12, 31)

    # Test 1: Full year interest
    expected = principal * rate * Decimal("1")
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected

    # Test 2: Half year interest
    asof = Date(2020, 7, 1)
    expected = principal * rate * Decimal("0.5")
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected

    # Test 3: No time passed
    asof = start
    expected = principal * rate * Decimal("0")
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected

    # Test 4: Without end date (should use asof)
    asof = Date(2020, 6, 1)
    expected = principal * rate * Decimal("5/366")  # 2020 is a leap year
    result = dcc.interest(principal, rate, start, asof)
    assert result == expected

    # Test 5: Invalid date range (asof before start)
    asof = Date(2019, 12, 31)
    expected = Money(0, "USD")
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected


# LLM-generated content at query #94
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Alt1"}, set(), lambda s, a, e, f: Decimal(1))
    dcc2 = DCC("Test2", {"Alt2"}, set(), lambda s, a, e, f: Decimal(2))
    registry.register(dcc1)
    registry.register(dcc2)

    # Test exact match
    assert registry.find("Test1") == dcc1
    assert registry.find("Test2") == dcc2
    assert registry.find("Alt1") == dcc1
    assert registry.find("Alt2") == dcc2

    # Test case-insensitive and stripped match
    assert registry.find(" test1 ") == dcc1
    assert registry.find("TEST2") == dcc2
    assert registry.find("  alt1  ") == dcc1
    assert registry.find("ALT2") == dcc2

    # Test non-existent
    assert registry.find("NonExistent") is None
    assert registry.find("") is None


# LLM-generated content at query #95
#--------------------------

```python
def test_DCC_interest():
    # Create a mock DCC instance with a simple calculate_fraction_method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test case 1: Basic interest calculation
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.10")  # 10%
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(50, Currency("USD"))  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test case 2: Interest calculation with asof == end
    result = dcc.interest(principal, rate, start, asof, asof)
    expected = Money(50, Currency("USD"))  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test case 3: Interest calculation with default end (asof)
    result = dcc.interest(principal, rate, start, asof)
    expected = Money(50, Currency("USD"))  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test case 4: Zero fraction case (asof before start)
    result = dcc.interest(principal, rate, start, Date(2019, 12, 31), end)
    expected = Money(0, Currency("USD"))
    assert result == expected

    # Test case 5: Zero fraction case (asof after end)
    result = dcc.interest(principal, rate, start, Date(2021, 1, 1), end)
    expected = Money(0, Currency("USD"))
    assert result == expected


# LLM-generated content at query #96
#--------------------------

```python
def test_DCC_interest():
    # Setup test data
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal((asof - start).days) / Decimal(360)
    )
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Test with end date provided
    result = dcc.interest(principal, rate, start, asof, end)
    expected_days = (asof - start).days
    expected_fraction = Decimal(expected_days) / Decimal(360)
    expected_interest = principal * rate * expected_fraction
    assert result == expected_interest

    # Test with end date not provided (should use asof)
    result_no_end = dcc.interest(principal, rate, start, asof)
    assert result_no_end == ZERO  # Since asof == end in this case

    # Test with different dates
    asof2 = Date(2020, 3, 1)
    result2 = dcc.interest(principal, rate, start, asof2, end)
    expected_days2 = (asof2 - start).days
    expected_fraction2 = Decimal(expected_days2) / Decimal(360)
    expected_interest2 = principal * rate * expected_fraction2
    assert result2 == expected_interest2

    # Test with zero principal
    zero_principal = Money(0, Currencies["USD"])
    result_zero = dcc.interest(zero_principal, rate, start, asof, end)
    assert result_zero == ZERO

    # Test with zero rate
    result_zero_rate = dcc.interest(principal, ZERO, start, asof, end)
    assert result_zero_rate == ZERO

    # Test with invalid date order (should return zero)
    invalid_asof = Date(2019, 12, 1)
    result_invalid = dcc.interest(principal, rate, start, invalid_asof, end)
    assert result_invalid == ZERO


