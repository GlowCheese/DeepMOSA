####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start, asof, asof)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 29th Feb
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start, asof, asof)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st Oct, end date is 30th Nov
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start, asof, asof)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Start date is 1st Feb, end date is 31st May
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start, asof, asof)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st Jan, end date is 1st Mar
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 3, 1)
    result = dcfc_30_360_isda(start, asof, asof)
    assert round(result, 14) == Decimal('0.08333333333333')

    # Test case 6: Start date is 30th Jan, end date is 31st Mar
    start = datetime.date(2008, 1, 30)
    asof = datetime.date(2008, 3, 31)
    result = dcfc_30_360_isda(start, asof, asof)
    assert round(result, 14) == Decimal('0.16666666666667')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Basic case without leap year
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('0.16942884946478')
    assert round(result, 14) == expected

    # Test case 2: Case with leap year
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('0.17216108990194')
    assert round(result, 14) == expected

    # Test case 3: Longer period
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1.08243131970956')
    assert round(result, 14) == expected

    # Test case 4: Another longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1.32625945055768')
    assert round(result, 14) == expected

    # Test case 5: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('0.0')
    assert result == expected

    # Test case 6: Full year without leap day
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1.0')
    assert result == expected

    # Test case 7: Full year with leap day
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    expected = Decimal('1.0')
    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    # Setup
    registry = DCCRegistryMachinery()
    test_dcc = DCC(
        name="TestDCC",
        altnames={"TestDCCAlt1", "TestDCCAlt2"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test successful registration
    registry.register(test_dcc)
    assert registry._find_strict("TestDCC") == test_dcc
    assert registry._find_strict("TestDCCAlt1") == test_dcc
    assert registry._find_strict("TestDCCAlt2") == test_dcc

    # Test duplicate registration
    duplicate_dcc = DCC(
        name="TestDCC",
        altnames={"TestDCCAlt3"},
        currencies=_as_ccys({"EUR"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError):
        registry.register(duplicate_dcc)

    # Test alternative name conflict
    conflict_dcc = DCC(
        name="ConflictDCC",
        altnames={"TestDCCAlt1"},
        currencies=_as_ccys({"GBP"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError):
        registry.register(conflict_dcc)


# LLM-generated content at query #4
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Long period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st and end date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 6: Start date is 30th and end date is 31st
    start = datetime.date(2007, 11, 30)
    asof = datetime.date(2008, 12, 31)
    assert round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14) == Decimal('1.02777777777778')


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    # Setup
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"TEST1"}, set(), lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC("Test2", {"TEST2", "ALTERNATIVE"}, set(), lambda s, a, e, f: Decimal(0.5))

    # Register DCCs
    registry.register(dcc1)
    registry.register(dcc2)

    # Test exact match
    assert registry.find("Test1") == dcc1
    assert registry.find("Test2") == dcc2

    # Test alternative name match
    assert registry.find("ALTERNATIVE") == dcc2

    # Test case-insensitive and stripped match
    assert registry.find(" test1 ") == dcc1
    assert registry.find("  test2  ") == dcc2

    # Test non-existent DCC
    assert registry.find("NonExistent") is None
    assert registry.find("") is None


# LLM-generated content at query #7
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

    # Test case 4: Period spanning multiple years including leap day
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

    # Test case 7: Full year period (non-leap year)
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1.0')

    # Test case 8: Full year period (leap year)
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2016, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.00273972602740')

    # Test case 9: Period with frequency parameter (should not affect Act/Act)
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 6, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof, freq=Decimal(2))
    assert round(result, 14) == Decimal('0.50068493150685')

    # Test case 10: Period with None frequency parameter
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 6, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof, freq=None)
    assert round(result, 14) == Decimal('0.50068493150685')


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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


# LLM-generated content at query #10
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

    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 1, 10)
    asof = datetime.date(2020, 1, 5)

    # Test daily fraction calculation
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal(4) / Decimal(9) - Decimal(3) / Decimal(9)
    assert daily_fraction == expected

    # Test when asof is start date
    asof = start
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal(0) / Decimal(9) - Decimal(0) / Decimal(9)
    assert daily_fraction == expected

    # Test when asof is end date
    asof = end
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal(9) / Decimal(9) - Decimal(8) / Decimal(9)
    assert daily_fraction == expected

    # Test when asof is before start
    asof = datetime.date(2019, 12, 31)
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)
    expected = ZERO
    assert daily_fraction == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Setup test data
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 6, 1)
    end_date = datetime.date(2020, 12, 31)
    freq = Decimal(2)

    # Create a mock DCC instance with a simple calculation method
    def mock_calculate_method(s, a, e, f):
        return Decimal((a - s).days) / Decimal((e - s).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_method
    )

    # Test normal case
    result = dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    expected = Decimal(152) / Decimal(366)
    assert result == expected

    # Test when asof is before start
    result = dcc.calculate_fraction(start_date, datetime.date(2019, 12, 31), end_date, freq)
    assert result == ZERO

    # Test when asof is after end
    result = dcc.calculate_fraction(start_date, datetime.date(2021, 1, 1), end_date, freq)
    assert result == ZERO

    # Test when all dates are equal
    result = dcc.calculate_fraction(start_date, start_date, start_date, freq)
    assert result == ZERO

    # Test with None frequency
    result = dcc.calculate_fraction(start_date, asof_date, end_date, None)
    expected = Decimal(152) / Decimal(366)
    assert result == expected


# LLM-generated content at query #12
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


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=start1, asof=asof1, end=asof1), 14) == expected1

    # Test case 2: Leap day in the period (should be excluded)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=start2, asof=asof2, end=asof2), 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=start3, asof=asof3, end=asof3), 14) == expected3

    # Test case 4: Another longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32602739726027')
    assert round(dcfc_nl_365(start=start4, asof=asof4, end=asof4), 14) == expected4


# LLM-generated content at query #14
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC instance for testing
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=simple_fraction
    )

    # Test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 1)  # Same as start
    end = Date(2020, 12, 31)

    # Calculate expected interest (should be 0 since asof == start)
    expected = principal * rate * Decimal("0")
    result = dcc.interest(principal, rate, start, asof, end)

    assert result == expected

    # Test with asof in middle of period
    asof = Date(2020, 6, 30)
    days_passed = (asof - start).days
    total_days = (end - start).days
    expected_fraction = Decimal(days_passed) / Decimal(total_days)
    expected = principal * rate * expected_fraction
    result = dcc.interest(principal, rate, start, asof, end)

    assert result == expected

    # Test with asof == end
    asof = end
    expected_fraction = Decimal("1")
    expected = principal * rate * expected_fraction
    result = dcc.interest(principal, rate, start, asof, end)

    assert result == expected

    # Test with asof before start (should return 0)
    asof = Date(2019, 12, 31)
    expected = principal * rate * Decimal("0")
    result = dcc.interest(principal, rate, start, asof, end)

    assert result == expected

    # Test with asof after end (should return full period interest)
    asof = Date(2021, 1, 1)
    expected_fraction = Decimal("1")
    expected = principal * rate * expected_fraction
    result = dcc.interest(principal, rate, start, asof, end)

    assert result == expected


# LLM-generated content at query #15
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

    # Test duplicate registration
    duplicate_dcc = DCC(
        name="TestDCC",
        altnames={"AnotherAlt"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError):
        registry.register(duplicate_dcc)

    # Test alternative name conflict
    conflicting_dcc = DCC(
        name="NewDCC",
        altnames={"AltTestDCC"},
        currencies={Currencies["GBP"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError):
        registry.register(conflicting_dcc)


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    expected = Decimal('0.16942884946478')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 2: Leap year period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    expected = Decimal('0.17216108990194')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 3: Longer period with leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    expected = Decimal('1.08243131970956')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 4: Another longer period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    expected = Decimal('1.32625945055768')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 5: Same start and asof dates
    start = datetime.date(2010, 1, 1)
    asof = datetime.date(2010, 1, 1)
    expected = Decimal('0.00000000000000')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 6: Full year period
    start = datetime.date(2010, 1, 1)
    asof = datetime.date(2010, 12, 31)
    expected = Decimal('1.00000000000000')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 7: Full leap year period
    start = datetime.date(2012, 1, 1)
    asof = datetime.date(2012, 12, 31)
    expected = Decimal('1.00000000000000')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected

    # Test case 8: Period spanning multiple years with leap days
    start = datetime.date(2010, 1, 1)
    asof = datetime.date(2013, 1, 1)
    expected = Decimal('3.00000000000000')
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == expected


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=start1, asof=asof1, end=asof1), 14) == expected1

    # Test case 2: Leap year period (Feb 29 included)
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=start2, asof=asof2, end=asof2), 14) == expected2

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=start3, asof=asof3, end=asof3), 14) == expected3

    # Test case 4: Multi-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32602739726027')
    assert round(dcfc_nl_365(start=start4, asof=asof4, end=asof4), 14) == expected4


# LLM-generated content at query #18
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

    # Test case 4: Period spanning multiple years
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=start4, asof=asof4, end=asof4)
    expected4 = Decimal('1.32602739726027')
    assert round(result4, 14) == expected4


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Basic test with given example
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
    assert result == Decimal('0.0')

    # Test case 3: Leap year scenario
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('425') / Decimal('731') / Decimal('1')
    assert result == expected

    # Test case 4: Different frequency
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('2'))
    expected = Decimal('212') / Decimal('366') / Decimal('2')
    assert result == expected

    # Test case 5: Full period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal('1.0')

    # Test case 6: Partial period at the beginning
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected = Decimal('15') / Decimal('366')
    assert result == expected


# LLM-generated content at query #20
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
    assert result == Decimal('0.00000000000000')


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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

    # Test case 3: Period spanning multiple years
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Period spanning multiple years with leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start, asof, asof)
    assert round(result, 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('0')

    # Test case 6: Full year period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('1')

    # Test case 7: Full leap year period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    result = dcfc_act_act(start, asof, asof)
    assert result == Decimal('1')

    # Test case 8: Period with frequency parameter (should be ignored)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    result = dcfc_act_act(start, asof, asof, Decimal(2))
    assert round(result, 14) == Decimal('0.5')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Example from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16942884946478')

    # Test case 2: Example from docstring with leap day
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17216108990194')

    # Test case 3: Example from docstring spanning multiple years
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08243131970956')

    # Test case 4: Example from docstring spanning multiple years with leap year
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof dates
    same_date = datetime.date(2020, 1, 1)
    assert dcfc_act_act(start=same_date, asof=same_date, end=same_date) == Decimal('0')

    # Test case 6: One day difference
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 1, 2)
    assert round(dcfc_act_act(start=start_date, asof=asof_date, end=asof_date), 14) == Decimal('0.00273972602740')

    # Test case 7: Full non-leap year
    start_date = datetime.date(2019, 1, 1)
    asof_date = datetime.date(2019, 12, 31)
    assert round(dcfc_act_act(start=start_date, asof=asof_date, end=asof_date), 14) == Decimal('1.00000000000000')

    # Test case 8: Full leap year
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 12, 31)
    assert round(dcfc_act_act(start=start_date, asof=asof_date, end=asof_date), 14) == Decimal('1.00000000000000')

    # Test case 9: Partial leap year including Feb 29
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 3, 1)
    expected = Decimal('60') / Decimal('366') + Decimal('1') / Decimal('365')
    assert round(dcfc_act_act(start=start_date, asof=asof_date, end=asof_date), 14) == round(expected, 14)

    # Test case 10: Crossing year boundary (non-leap to leap)
    start_date = datetime.date(2019, 12, 31)
    asof_date = datetime.date(2020, 1, 1)
    expected = Decimal('1') / Decimal('365') + Decimal('1') / Decimal('366')
    assert round(dcfc_act_act(start=start_date, asof=asof_date, end=asof_date), 14) == round(expected, 14)


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Standard case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: Leap year case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Longer period
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Multi-year period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.33055555555556')

    # Test case 5: Same day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 1)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.00000000000000')

    # Test case 6: End of month adjustment
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.00000000000000')

    # Test case 7: End of month adjustment for asof
    start = datetime.date(2008, 1, 30)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.02777777777778')


# LLM-generated content at query #4
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC instance for testing
    def simple_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="simple",
        altnames={"simple"},
        currencies={Currency("USD")},
        calculate_fraction_method=simple_fraction
    )

    # Test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 1)
    end = Date(2020, 1, 31)

    # Test when asof equals start
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(0, Currency("USD"))

    # Test when asof is in the middle
    asof = Date(2020, 1, 15)
    expected = principal * rate * Decimal(14) / Decimal(30)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected

    # Test when asof equals end
    asof = Date(2020, 1, 31)
    expected = principal * rate * Decimal(30) / Decimal(30)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected

    # Test when end is None (should use asof)
    asof = Date(2020, 1, 15)
    result = dcc.interest(principal, rate, start, asof)
    expected = principal * rate * Decimal(14) / Decimal(14)
    assert result == expected

    # Test with different frequency
    freq = Decimal(2)
    result = dcc.interest(principal, rate, start, asof, end, freq)
    expected = principal * rate * simple_fraction(start, asof, end, freq)
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start1, asof1, asof1)
    expected1 = Decimal('0.16986301369863')
    assert round(result1, 14) == expected1

    # Test case 2: Leap day in the period (Feb 29)
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


# LLM-generated content at query #6
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
    assert registry.find("TestDCC") == test_dcc
    assert registry.find("TestDCCAlt") == test_dcc

    # Test duplicate registration
    duplicate_dcc = DCC(
        name="TestDCC",
        altnames={"AnotherAlt"},
        currencies=_as_ccys({"EUR"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError):
        registry.register(duplicate_dcc)

    # Test alternative name conflict
    conflict_dcc = DCC(
        name="NewDCC",
        altnames={"TestDCCAlt"},
        currencies=_as_ccys({"GBP"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError):
        registry.register(conflict_dcc)

    # Verify registry contents
    assert len(registry.registry) == 1
    assert "TestDCC" in registry.table
    assert "TestDCCAlt" in registry.table


# LLM-generated content at query #7
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


# LLM-generated content at query #8
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
    assert registry._find_strict("TestDCC") == test_dcc
    assert registry._find_strict("AltTestDCC") == test_dcc

    # Test duplicate registration raises TypeError
    duplicate_dcc = DCC(
        name="TestDCC",
        altnames={"AnotherAlt"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    with pytest.raises(TypeError):
        registry.register(duplicate_dcc)

    # Test alternative name conflict raises TypeError
    conflict_dcc = DCC(
        name="NewDCC",
        altnames={"AltTestDCC"},
        currencies={Currencies["GBP"]},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.4")
    )
    with pytest.raises(TypeError):
        registry.register(conflict_dcc)


# LLM-generated content at query #9
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
    expected = principal * rate * Decimal("0.5")
    assert result == expected

    # Test with end date as asof
    result = dcc.interest(principal, rate, start, asof)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

    # Test with different fraction
    dcc = DCC(
        name="Test2",
        altnames={"TestAlt2"},
        currencies={Currency("EUR")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.25")
    )
    result = dcc.interest(principal, rate, start, asof, end)
    expected = principal * rate * Decimal("0.25")
    assert result == expected

    # Test with zero principal
    zero_principal = Money(0, Currency("USD"))
    result = dcc.interest(zero_principal, rate, start, asof, end)
    assert result == zero_principal

    # Test with zero rate
    result = dcc.interest(principal, Decimal("0"), start, asof, end)
    assert result.amount == 0

    # Test with invalid date range (asof before start)
    result = dcc.interest(principal, rate, asof, start, end)
    assert result.amount == 0


# LLM-generated content at query #10
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
    expected2 = Decimal('0.16986301369863')  # Same as case 1 because leap day is excluded
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

    # Test case 5: Period with no leap day at all
    start5 = datetime.date(2009, 1, 1)
    asof5 = datetime.date(2009, 12, 31)
    result5 = dcfc_nl_365(start=start5, asof=asof5, end=asof5)
    expected5 = Decimal('1.0')
    assert round(result5, 14) == expected5

    # Test case 6: Period with leap day (should be excluded)
    start6 = datetime.date(2008, 1, 1)
    asof6 = datetime.date(2008, 12, 31)
    result6 = dcfc_nl_365(start=start6, asof=asof6, end=asof6)
    expected6 = Decimal('1.0')  # 365/365 because leap day is excluded
    assert round(result6, 14) == expected6


# LLM-generated content at query #11
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Setup test data
    start_date = datetime.date(2020, 1, 1)
    asof_date = datetime.date(2020, 6, 1)
    end_date = datetime.date(2020, 12, 31)
    freq = Decimal(2)

    # Create a mock DCC instance with a simple calculation method
    def mock_calculate_fraction_method(start, asof, end, freq):
        total_days = (end - start).days
        accrued_days = (asof - start).days
        return Decimal(accrued_days) / Decimal(total_days)

    dcc = DCC(
        name="ACT/ACT",
        altnames={"ACT/ACT"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=mock_calculate_fraction_method
    )

    # Test normal case
    result = dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    expected = Decimal(182) / Decimal(366)  # 2020 is a leap year
    assert result == expected

    # Test when asof is before start
    result = dcc.calculate_fraction(start_date, datetime.date(2019, 12, 31), end_date, freq)
    assert result == ZERO

    # Test when asof is after end
    result = dcc.calculate_fraction(start_date, datetime.date(2021, 1, 1), end_date, freq)
    assert result == ZERO

    # Test when asof equals start
    result = dcc.calculate_fraction(start_date, start_date, end_date, freq)
    expected = Decimal(0) / Decimal(366)
    assert result == expected

    # Test when asof equals end
    result = dcc.calculate_fraction(start_date, end_date, end_date, freq)
    expected = Decimal(366) / Decimal(366)
    assert result == ONE


# LLM-generated content at query #12
#--------------------------

```python
def test_DCC_coupon():
    # Create a sample DCC instance
    dcc = DCC(
        name="TestDCC",
        altnames={"Test"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)
    freq = 2
    eom = None

    # Expected result
    expected = Money(25, Currency("USD"))

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1: Normal case
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 29th Feb in a leap year
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st Oct, end date is 30th Nov
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('1.08333333333333')

    # Test case 4: Start date is 1st Feb, end date is 31st May next year
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('1.33333333333333')

    # Test case 5: Start date is 31st Jan, end date is 28th Feb same year
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.0')

    # Test case 6: Start date is 30th Jan, end date is 31st Mar same year
    start = datetime.date(2008, 1, 30)
    asof = datetime.date(2008, 3, 31)
    end = asof
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #14
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
    asof = datetime.date(2020, 1, 5)

    # Test normal case
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is start date
    result = dcc.calculate_daily_fraction(start, start, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is day after start
    asof = datetime.date(2020, 1, 2)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.1111111111111111111111111111")  # 1/9
    assert result == expected

    # Test when asof is end date
    result = dcc.calculate_daily_fraction(start, end, end)
    assert result == ZERO

    # Test when asof is before start (should return 0)
    asof = datetime.date(2019, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == ZERO

    # Test when asof is after end (should return 0)
    asof = datetime.date(2020, 1, 11)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == ZERO


# LLM-generated content at query #15
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Basic test with known result
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

    # Test case 2: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.0')
    assert round(result, 10) == expected

    # Test case 3: Leap year scenario
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.4109589041')
    assert round(result, 10) == expected

    # Test case 4: End date before asof date (should return 0)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 3, 1)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('0.0')
    assert result == expected

    # Test case 5: Full period test
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start, asof, end)
    expected = Decimal('1.0')
    assert result == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_DCC_calculate_fraction():
    # Create a mock DCC instance with a simple calculation method
    def simple_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="TestDCC",
        altnames={"Test"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=simple_calc
    )

    # Test case 1: Normal case where asof is between start and end
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 31)
    expected = Decimal('0.4838709677419355')
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 2: asof is equal to start
    asof = datetime.date(2020, 1, 1)
    expected = Decimal('0')
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 3: asof is equal to end
    asof = datetime.date(2020, 1, 31)
    expected = Decimal('1')
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 4: asof is before start (should return 0)
    asof = datetime.date(2019, 12, 31)
    expected = Decimal('0')
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 5: asof is after end (should return 0)
    asof = datetime.date(2020, 2, 1)
    expected = Decimal('0')
    assert dcc.calculate_fraction(start, asof, end) == expected

    # Test case 6: Test with frequency parameter
    def freq_calc(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal((end - start).days) * freq

    dcc_with_freq = DCC(
        name="TestDCCFreq",
        altnames={"TestFreq"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=freq_calc
    )

    freq = Decimal('2')
    expected = Decimal('0.967741935483871')
    assert dcc_with_freq.calculate_fraction(start, asof, end, freq) == expected


# LLM-generated content at query #17
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

    # Test case 3: Cross-year period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    assert round(dcfc_act_365_l(start=start3, asof=asof3, end=asof3), 14) == expected3

    # Test case 4: Longer period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32876712328767')
    assert round(dcfc_act_365_l(start=start4, asof=asof4, end=asof4), 14) == expected4


# LLM-generated content at query #18
#--------------------------

```python
def test_DCC_interest():
    # Create a mock DCC instance with a simple fraction calculation method
    def mock_fraction_method(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="TEST",
        altnames={"TEST_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_fraction_method
    )

    # Test case 1: Basic interest calculation
    principal = Money(1000, "USD")
    rate = Decimal("0.10")  # 10%
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(50, "USD")  # 1000 * 0.10 * 0.5
    assert result == expected

    # Test case 2: Interest calculation with asof == end
    result = dcc.interest(principal, rate, start, asof, asof)
    expected = Money(50, "USD")  # Same as above since fraction is 0.5
    assert result == expected

    # Test case 3: Interest calculation with no end date (asof used as end)
    result = dcc.interest(principal, rate, start, asof)
    expected = Money(50, "USD")  # Same as above
    assert result == expected

    # Test case 4: Zero principal
    zero_principal = Money(0, "USD")
    result = dcc.interest(zero_principal, rate, start, asof, end)
    expected = Money(0, "USD")
    assert result == expected

    # Test case 5: Zero rate
    zero_rate = Decimal("0.00")
    result = dcc.interest(principal, zero_rate, start, asof, end)
    expected = Money(0, "USD")
    assert result == expected

    # Test case 6: Different fraction method (actual/360)
    def actual_360_method(start, asof, end, freq):
        return Decimal((end - start).days) / Decimal("360")

    dcc_actual_360 = DCC(
        name="ACT/360",
        altnames={"ACTUAL/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=actual_360_method
    )

    start = Date(2020, 1, 1)
    asof = Date(2020, 1, 1)
    end = Date(2020, 7, 1)  # 181 days
    result = dcc_actual_360.interest(principal, rate, start, asof, end)
    expected = Money(50.27777777777777777777777778, "USD")  # 1000 * 0.10 * (181/360)
    assert result == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_DCC_interest():
    # Create a mock DCC instance with a simple fraction calculation method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")

    dcc = DCC(
        name="MOCK",
        altnames={"MOCK_ALT"},
        currencies={Currency("USD")},
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test data
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.10")  # 10%
    start = Date(2020, 1, 1)
    asof = Date(2020, 6, 1)
    end = Date(2020, 12, 31)

    # Expected result: 1000 * 0.10 * 0.5 = 50
    expected = Money(50, Currency("USD"))

    # Test without end date (should use asof)
    result = dcc.interest(principal, rate, start, asof)
    assert result == expected

    # Test with end date
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == expected

    # Test with different fraction
    def mock_calculate_fraction_2(start, asof, end, freq):
        return Decimal("0.25")

    dcc2 = DCC(
        name="MOCK2",
        altnames={"MOCK_ALT2"},
        currencies={Currency("EUR")},
        calculate_fraction_method=mock_calculate_fraction_2
    )

    # Expected result: 1000 * 0.10 * 0.25 = 25
    expected2 = Money(25, Currency("EUR"))
    result = dcc2.interest(principal, rate, start, asof)
    assert result == expected2

    # Test with zero fraction
    def mock_calculate_fraction_zero(start, asof, end, freq):
        return ZERO

    dcc_zero = DCC(
        name="MOCK_ZERO",
        altnames={"MOCK_ALT_ZERO"},
        currencies={Currency("GBP")},
        calculate_fraction_method=mock_calculate_fraction_zero
    )

    # Expected result: 1000 * 0.10 * 0 = 0
    expected_zero = Money(0, Currency("GBP"))
    result = dcc_zero.interest(principal, rate, start, asof)
    assert result == expected_zero

    # Test with invalid date range (should return zero)
    invalid_asof = Date(2019, 12, 31)  # Before start date
    result = dcc.interest(principal, rate, start, invalid_asof)
    assert result == Money(0, Currency("USD"))


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
def test_dcfc_act_act():
    # Test case 1: Normal period without leap day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16942884946478')

    # Test case 2: Period including leap day
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17216108990194')

    # Test case 3: Longer period with leap year
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08243131970956')

    # Test case 4: Period spanning multiple years
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

    # Test case 7: Full year period (non-leap year)
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1.0')

    # Test case 8: Full year period (leap year)
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2016, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.00273972602740')


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1: Normal case with no adjustments
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

    # Test case 2: End date is 31st, should be adjusted to 30th
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is 31st, should be adjusted to 30th
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

    # Test case 4: Both start and end dates are 31st, should be adjusted to 30th
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=asof), 14) == Decimal('1.33055555555556')

    # Test case 5: Same date
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    assert dcfc_30_e_360(start=start, asof=asof, end=asof) == Decimal('0')

    # Test case 6: One day difference
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    assert dcfc_30_e_360(start=start, asof=asof, end=asof) == Decimal('0.00277777777778')

    # Test case 7: One month difference
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 2, 1)
    assert dcfc_30_e_360(start=start, asof=asof, end=asof) == Decimal('0.08333333333333')

    # Test case 8: One year difference
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    assert dcfc_30_e_360(start=start, asof=asof, end=asof) == Decimal('1.0')


# LLM-generated content at query #23
#--------------------------

```python
def test_DCC_coupon():
    # Setup test data
    principal = Money(1000, Currencies["USD"])
    rate = Decimal("0.05")  # 5%
    start = Date(2020, 1, 1)
    asof = Date(2020, 3, 1)
    end = Date(2020, 6, 1)
    freq = Decimal(2)  # Semi-annual
    eom = None

    # Create a mock DCC object with a simple day count fraction method
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal((asof - start).days) / Decimal(360)

    dcc = DCC(
        name="MOCK",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )

    # Calculate expected coupon
    prevdate = _last_payment_date(start, asof, freq, eom)
    nextdate = _next_payment_date(prevdate, freq, eom)
    expected_days = (asof - prevdate).days
    expected_fraction = Decimal(expected_days) / Decimal(360)
    expected_coupon = principal * rate * expected_fraction

    # Call the method
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)

    # Assertions
    assert result == expected_coupon
    assert prevdate == Date(2020, 1, 1)
    assert nextdate == Date(2020, 7, 1)


# LLM-generated content at query #24
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test cases from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

    # Additional test cases
    # Test with same start and asof dates
    same_date = datetime.date(2020, 1, 1)
    assert dcfc_30_360_us(start=same_date, asof=same_date, end=same_date) == Decimal('0.0')

    # Test with start date being last day of month
    start_last_day = datetime.date(2020, 1, 31)
    asof_feb = datetime.date(2020, 2, 28)
    assert round(dcfc_30_360_us(start=start_last_day, asof=asof_feb, end=asof_feb), 14) == Decimal('0.0')

    # Test with asof date being last day of month
    start_normal = datetime.date(2020, 1, 15)
    asof_last_day = datetime.date(2020, 2, 29)  # Leap year
    assert round(dcfc_30_360_us(start=start_normal, asof=asof_last_day, end=asof_last_day), 14) == Decimal('0.14166666666667')

    # Test with both dates being last day of month
    start_last = datetime.date(2020, 1, 31)
    asof_last = datetime.date(2020, 2, 29)
    assert round(dcfc_30_360_us(start=start_last, asof=asof_last, end=asof_last), 14) == Decimal('0.0')

    # Test with d2=31 and d1 in {30, 31}
    start_30 = datetime.date(2020, 1, 30)
    asof_31 = datetime.date(2020, 2, 29)  # Not 31, but testing the logic
    assert round(dcfc_30_360_us(start=start_30, asof=asof_31, end=asof_31), 14) == Decimal('0.0')

    # Test with d1=31
    start_31 = datetime.date(2020, 1, 31)
    asof_normal = datetime.date(2020, 2, 15)
    assert round(dcfc_30_360_us(start=start_31, asof=asof_normal, end=asof_normal), 14) == Decimal('0.13888888888889')


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test_dcfc_act_365_l():
    # Test case 1: Non-leap year, no leap day in range
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_act_365_l(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16939890710383')

    # Test case 2: Non-leap year, leap day in range
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_act_365_l(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17213114754098')

    # Test case 3: Leap year, no leap day in range
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_act_365_l(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08196721311475')

    # Test case 4: Leap year, leap day in range
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_act_365_l(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32876712328767')

    # Test case 5: Leap year, asof is leap day
    start5 = datetime.date(2020, 2, 28)
    asof5 = datetime.date(2020, 2, 29)
    assert round(dcfc_act_365_l(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.00273972602740')

    # Test case 6: Non-leap year, asof is not leap day
    start6 = datetime.date(2019, 2, 28)
    asof6 = datetime.date(2019, 3, 1)
    assert round(dcfc_act_365_l(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00273224043716')


# LLM-generated content at query #27
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end), 10) == Decimal('0.5245901639')

    # Test case 2: Same start and asof dates
    assert dcfc_act_act_icma(start=ex1_start, asof=ex1_start, end=ex1_end) == ZERO

    # Test case 3: Same asof and end dates
    assert dcfc_act_act_icma(start=ex1_start, asof=ex1_end, end=ex1_end) == ONE / Decimal(1)

    # Test case 4: Different dates with frequency
    ex2_start = datetime.date(2020, 1, 1)
    ex2_asof = datetime.date(2020, 6, 30)
    ex2_end = datetime.date(2021, 1, 1)
    assert round(dcfc_act_act_icma(start=ex2_start, asof=ex2_asof, end=ex2_end, freq=Decimal(2)), 10) == Decimal('0.4958904110')

    # Test case 5: Leap year scenario
    ex3_start = datetime.date(2019, 2, 1)
    ex3_asof = datetime.date(2020, 2, 1)
    ex3_end = datetime.date(2021, 2, 1)
    assert round(dcfc_act_act_icma(start=ex3_start, asof=ex3_asof, end=ex3_end), 10) == Decimal('1.0000000000')

    # Test case 6: As of date before start date
    ex4_start = datetime.date(2020, 1, 1)
    ex4_asof = datetime.date(2019, 12, 31)
    ex4_end = datetime.date(2020, 12, 31)
    assert dcfc_act_act_icma(start=ex4_start, asof=ex4_asof, end=ex4_end) == ZERO

    # Test case 7: As of date after end date
    ex5_start = datetime.date(2020, 1, 1)
    ex5_asof = datetime.date(2021, 1, 1)
    ex5_end = datetime.date(2020, 12, 31)
    assert dcfc_act_act_icma(start=ex5_start, asof=ex5_asof, end=ex5_end) == ONE / Decimal(1)


# LLM-generated content at query #28
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

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    assert round(dcfc_act_365_a(start=start3, asof=asof3, end=asof3), 14) == expected3

    # Test case 4: Longer period without leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32513661202186')
    assert round(dcfc_act_365_a(start=start4, asof=asof4, end=asof4), 14) == expected4


# LLM-generated content at query #29
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


