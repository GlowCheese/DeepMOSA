####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_dcfc_30_360_isda():
    # Test case 1
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test case 2
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test case 3
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test case 4
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

    # Additional edge case - start day is 31
    edge_start, edge_asof = datetime.date(2020, 1, 31), datetime.date(2020, 2, 28)
    assert round(dcfc_30_360_isda(start=edge_start, asof=edge_asof, end=edge_asof), 14) == Decimal('0.08333333333333')

    # Additional edge case - asof day is 31 but start day is 30
    edge_start2, edge_asof2 = datetime.date(2020, 1, 30), datetime.date(2020, 2, 31)
    assert round(dcfc_30_360_isda(start=edge_start2, asof=edge_asof2, end=edge_asof2), 14) == Decimal('0.08333333333333')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Dates within the same year, no leap year
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 31)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.08219178082192')

    # Test case 2: Dates spanning a leap year
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2016, 12, 31)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.99726775956284')

    # Test case 3: Dates spanning multiple years, including a leap year
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2017, 1, 1)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('2.00000000000000')

    # Test case 4: Dates spanning multiple years, including multiple leap years
    start = datetime.date(2012, 1, 1)
    asof = datetime.date(2016, 1, 1)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('4.00000000000000')

    # Test case 5: Dates within a leap year
    start = datetime.date(2016, 2, 1)
    asof = datetime.date(2016, 2, 29)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.07650273224044')

    # Test case 6: Dates spanning a leap year, but not including the leap day
    start = datetime.date(2015, 12, 31)
    asof = datetime.date(2016, 12, 30)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.99726775956284')

    # Test case 7: Dates spanning multiple years, including a leap year, with end date after asof
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2016, 1, 1)
    end = datetime.date(2017, 1, 1)
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('2.00000000000000')

    # Test case 8: Dates within the same year, leap year, including leap day
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2016, 3, 1)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.16721311475410')


# LLM-generated content at query #3
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt1", "TestAlt2"},
        currencies=_as_ccys({"USD", "EUR"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    
    registry.register(dcc)
    
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt1") == dcc
    assert registry._find_strict("TestAlt2") == dcc
    
    # Test registering the same DCC again
    try:
        registry.register(dcc)
        assert False, "Expected TypeError when registering the same DCC"
    except TypeError:
        pass
    
    # Test registering a DCC with an overlapping altname
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt1"},
        currencies=_as_ccys({"GBP"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.7")
    )
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError when registering DCC with overlapping altname"
    except TypeError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_dcfc_30_360_isda():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #5
#--------------------------

def test_dcfc_act_365_a():
    # Test case 1: No leap year in the period
    start1 = Date(2007, 12, 28)
    asof1 = Date(2008, 2, 28)
    assert dcfc_act_365_a(start1, asof1, asof1) == Decimal('0.16986301369863')

    # Test case 2: Leap day included in the period
    start2 = Date(2007, 12, 28)
    asof2 = Date(2008, 2, 29)
    assert dcfc_act_365_a(start2, asof2, asof2) == Decimal('0.17213114754098')

    # Test case 3: Multiple years, no leap day
    start3 = Date(2007, 10, 31)
    asof3 = Date(2008, 11, 30)
    assert dcfc_act_365_a(start3, asof3, asof3) == Decimal('1.08196721311475')

    # Test case 4: Multiple years, leap day included
    start4 = Date(2008, 2, 1)
    asof4 = Date(2009, 5, 31)
    assert dcfc_act_365_a(start4, asof4, asof4) == Decimal('1.32513661202186')


# LLM-generated content at query #6
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    registry = DCCRegistryMachinery()
    
    # Register a sample DCC
    dcc = DCC(name="Act/Act", altnames={"Actual/Actual"}, currencies=_as_ccys({"USD"}), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    registry.register(dcc)
    
    # Test finding by main name
    assert registry.find("Act/Act") == dcc
    
    # Test finding by alternative name
    assert registry.find("Actual/Actual") == dcc
    
    # Test finding with stripped and uppercase name
    assert registry.find(" act/act ") == dcc
    
    # Test finding non-existent DCC
    assert registry.find("NonExistent") is None


# LLM-generated content at query #7
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Mock the calculate_fraction_method to return fixed values
    def mock_calculate_fraction_method(start: Date, asof: Date, end: Date, freq: Optional[Decimal] = None) -> Decimal:
        if asof == datetime.date(2023, 1, 1):
            return Decimal("0.1")
        elif asof == datetime.date(2023, 1, 2):
            return Decimal("0.3")
        else:
            return Decimal("0.0")

    # Create a DCC instance with the mock method
    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )

    # Define test dates
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 1, 3)

    # Calculate daily fraction
    daily_fraction = dcc.calculate_daily_fraction(start, asof, end)

    # Assert the result
    assert daily_fraction == Decimal("0.2")


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_act_act_icma():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal('1')

    result = dcfc_act_act_icma(start, asof, end, freq)
    expected = Decimal('0.5245901639')

    assert round(result, 10) == expected


# LLM-generated content at query #10
#--------------------------

def test_dcfc_nl_365():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_nl_365(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day in period but not counted
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_nl_365(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16986301369863')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_nl_365(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08219178082192')

    # Test case 4: Multi-year period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32602739726027')

    # Test case 5: Single day calculation
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 1, 1)
    assert round(dcfc_nl_365(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.00273972602740')

    # Test case 6: Leap day period but not crossing Feb 29
    start6 = datetime.date(2020, 1, 1)
    asof6 = datetime.date(2020, 2, 28)
    assert round(dcfc_nl_365(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.15890410958904')


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

def test_dcfc_act_365_a():
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_act_365_a(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')

    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_act_365_a(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17213114754098')

    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_act_365_a(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08196721311475')

    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_act_365_a(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32513661202186')


# LLM-generated content at query #13
#--------------------------

def test_dcfc_act_act():
    # Test case 1: Regular year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16942884946478')

    # Test case 2: Includes leap day
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17216108990194')

    # Test case 3: Multi-year period
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08243131970956')

    # Test case 4: Another multi-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32625945055768')

    # Test case 5: Single day
    start5 = datetime.date(2017, 1, 1)
    asof5 = datetime.date(2017, 1, 1)
    assert round(dcfc_act_act(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.0')

    # Test case 6: One day period
    start6 = datetime.date(2017, 1, 1)
    asof6 = datetime.date(2017, 1, 2)
    assert round(dcfc_act_act(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00273972602740')


# LLM-generated content at query #14
#--------------------------

def test_dcfc_act_act_icma():
    # Test case 1: Regular period calculation
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(1)), 10) == Decimal('0.5245901639')

    # Test case 2: Full period should return 1
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert dcfc_act_act_icma(start=start, asof=end, end=end, freq=Decimal(1)) == Decimal(1)

    # Test case 3: Single day calculation
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 12, 31)
    expected = Decimal(1) / Decimal(366) / Decimal(1)  # 2020 is a leap year
    assert dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(1)) == expected

    # Test case 4: Different frequency (semi-annual)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(2)
    actual_days = Decimal(91)  # Jan 1 to Apr 1 in 2020 (leap year)
    period_days = Decimal(182)  # Jan 1 to Jul 1 in 2020
    expected = (actual_days / period_days) / freq
    assert dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq) == expected

    # Test case 5: Start and asof same date
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(1)) == Decimal(0)

    # Test case 6: Cross-year calculation
    start = datetime.date(2019, 12, 1)
    asof = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 6, 1)
    freq = Decimal(2)
    actual_days = Decimal(62)  # Dec 1 to Feb 1 (31 + 31 + 1)
    period_days = Decimal(183)  # Dec 1 to Jun 1
    expected = (actual_days / period_days) / freq
    assert dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq) == expected


# LLM-generated content at query #15
#--------------------------

def test_dcfc_act_365_a():
    start1 = Date(2007, 12, 28)
    asof1 = Date(2008, 2, 28)
    start2 = Date(2007, 12, 28)
    asof2 = Date(2008, 2, 29)
    start3 = Date(2007, 10, 31)
    asof3 = Date(2008, 11, 30)
    start4 = Date(2008, 2, 1)
    asof4 = Date(2009, 5, 31)

    assert round(dcfc_act_365_a(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')
    assert round(dcfc_act_365_a(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17213114754098')
    assert round(dcfc_act_365_a(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08196721311475')
    assert round(dcfc_act_365_a(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32513661202186')


# LLM-generated content at query #16
#--------------------------

def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    # Test case 2: Period including leap day
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    # Test case 3: Longer period spanning non-leap year
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    # Test case 4: Period spanning leap year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    # Test case 5: Single day in non-leap year
    start5 = datetime.date(2019, 1, 1)
    asof5 = datetime.date(2019, 1, 1)
    result5 = dcfc_act_365_a(start=start5, asof=asof5, end=asof5)
    assert result5 == Decimal('1') / Decimal('365')

    # Test case 6: Single day in leap year
    start6 = datetime.date(2020, 1, 1)
    asof6 = datetime.date(2020, 1, 1)
    result6 = dcfc_act_365_a(start=start6, asof=asof6, end=asof6)
    assert result6 == Decimal('1') / Decimal('366')


# LLM-generated content at query #17
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Act/Act",
        altnames={"Actual/Actual", "Actual/Actual (ISDA)"},
        currencies=_as_ccys({"USD", "EUR"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )

    registry.register(dcc)

    assert registry.find("Act/Act") == dcc
    assert registry.find("Actual/Actual") == dcc
    assert registry.find("Actual/Actual (ISDA)") == dcc

    dcc2 = DCC(
        name="Act/Act",
        altnames={"Actual/Actual", "Actual/Actual (ISDA)"},
        currencies=_as_ccys({"USD", "EUR"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )

    try:
        registry.register(dcc2)
        assert False, "Expected TypeError when registering a duplicate DCC"
    except TypeError:
        pass

    dcc3 = DCC(
        name="30/360",
        altnames={"Act/Act"},
        currencies=_as_ccys({"USD", "EUR"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )

    try:
        registry.register(dcc3)
        assert False, "Expected TypeError when registering a DCC with conflicting altname"
    except TypeError:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_30_360_isda():
    start1, asof1 = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    start2, asof2 = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    start3, asof3 = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    start4, asof4 = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_isda(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_isda(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_isda(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_isda(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.16942884946478')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.17216108990194')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('1.08243131970956')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('1.32625945055768')

    # Edge case: same day
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    assert dcfc_act_act(start, asof, asof) == Decimal('0')

    # Edge case: start date later than asof date
    start = datetime.date(2023, 1, 2)
    asof = datetime.date(2023, 1, 1)
    assert dcfc_act_act(start, asof, asof) == Decimal('0')


# LLM-generated content at query #20
#--------------------------

def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = Date(2007, 12, 28)
    asof1 = Date(2008, 2, 28)
    assert round(dcfc_act_365_a(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')

    # Test case 2: Period including leap day
    start2 = Date(2007, 12, 28)
    asof2 = Date(2008, 2, 29)
    assert round(dcfc_act_365_a(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17213114754098')

    # Test case 3: Full year period without leap day
    start3 = Date(2007, 10, 31)
    asof3 = Date(2008, 11, 30)
    assert round(dcfc_act_365_a(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08196721311475')

    # Test case 4: Multi-year period with leap day
    start4 = Date(2008, 2, 1)
    asof4 = Date(2009, 5, 31)
    assert round(dcfc_act_365_a(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32513661202186')

    # Test case 5: Single day calculation
    start5 = Date(2020, 1, 1)
    asof5 = Date(2020, 1, 1)
    assert round(dcfc_act_365_a(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.00273224043716')

    # Test case 6: Leap year to non-leap year
    start6 = Date(2020, 2, 28)
    asof6 = Date(2021, 2, 28)
    assert round(dcfc_act_365_a(start=start6, asof=asof6, end=asof6), 14) == Decimal('1.00273224043716')


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_30_360_isda():
    # Test case 1
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_act_act_icma():
    # Test case 1
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(2)
    assert round(dcfc_act_act_icma(start, asof, end, freq), 10) == Decimal('0.5245901639')

    # Test case 2
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(2)
    assert round(dcfc_act_act_icma(start, asof, end, freq), 10) == Decimal('0.5081967213')

    # Test case 3
    start = datetime.date(2021, 7, 1)
    asof = datetime.date(2021, 12, 31)
    end = datetime.date(2022, 6, 30)
    freq = Decimal(2)
    assert round(dcfc_act_act_icma(start, asof, end, freq), 10) == Decimal('0.5081967213')

    # Test case 4: Same start and asof date
    start = datetime.date(2022, 1, 1)
    asof = datetime.date(2022, 1, 1)
    end = datetime.date(2022, 12, 31)
    freq = Decimal(1)
    assert round(dcfc_act_act_icma(start, asof, end, freq), 10) == Decimal('0.0')

    # Test case 5: Same start and end date
    start = datetime.date(2022, 1, 1)
    asof = datetime.date(2022, 1, 1)
    end = datetime.date(2022, 1, 1)
    freq = Decimal(1)
    assert round(dcfc_act_act_icma(start, asof, end, freq), 10) == Decimal('0.0')

    # Test case 6: Different frequency
    start = datetime.date(2022, 1, 1)
    asof = datetime.date(2022, 6, 30)
    end = datetime.date(2022, 12, 31)
    freq = Decimal(4)
    assert round(dcfc_act_act_icma(start, asof, end, freq), 10) == Decimal('0.2540983607')


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.16942884946478')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('0.17216108990194')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('1.08243131970956')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start, asof, asof), 14) == Decimal('1.32625945055768')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: No leap year involved
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year involved
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17216108990194')

    # Test case 3: Multiple years, no leap year
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08243131970956')

    # Test case 4: Multiple years, leap year involved
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof date
    start5 = datetime.date(2007, 12, 28)
    asof5 = datetime.date(2007, 12, 28)
    assert round(dcfc_act_act(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.00000000000000')

    # Test case 6: Single day range, leap year
    start6 = datetime.date(2008, 2, 29)
    asof6 = datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00273224043716')


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_e_360():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=end), 14) == Decimal('0.16666666666667')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=end), 14) == Decimal('0.16944444444444')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=end), 14) == Decimal('1.08333333333333')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start, asof=asof, end=end), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #4
#--------------------------

```python
def test_DCC_interest():
    # Mock DCFC function that returns a fixed fraction
    def mock_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.1")

    # Create a DCC instance with the mock DCFC function
    dcc = DCC(
        name="mock_dcc",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc,
    )

    # Create test data
    principal = Money(Decimal("1000"), Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 12, 31)

    # Expected interest calculation: principal * rate * fraction
    expected_interest = Money(Decimal("5"), Currencies["USD"])

    # Test the interest method
    calculated_interest = dcc.interest(principal, rate, start, asof, end)

    # Assert that the calculated interest matches the expected interest
    assert calculated_interest == expected_interest


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_nl_365():
    start1, asof1 = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    start2, asof2 = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    start3, asof3 = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    start4, asof4 = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_nl_365(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32602739726027')


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = round(dcfc_nl_365(start=start, asof=asof, end=end), 14)
    assert result == Decimal('0.16986301369863')

    # Test case 2: Leap day in the period
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = round(dcfc_nl_365(start=start, asof=asof, end=end), 14)
    assert result == Decimal('0.16986301369863')

    # Test case 3: Period spans multiple years with a leap day
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = round(dcfc_nl_365(start=start, asof=asof, end=end), 14)
    assert result == Decimal('1.08219178082192')

    # Test case 4: Longer period with a leap day
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = round(dcfc_nl_365(start=start, asof=asof, end=end), 14)
    assert result == Decimal('1.32602739726027')


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

def test_dcfc_30_e_plus_360():
    # Test case 1: Regular dates
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_plus_360(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2: Leap day in asof date
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_plus_360(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date has 31st day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_plus_360(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4: As of date has 31st day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_plus_360(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')

    # Test case 5: Both start and asof have 31st day
    start5 = datetime.date(2007, 1, 31)
    asof5 = datetime.date(2007, 3, 31)
    assert round(dcfc_30_e_plus_360(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.16666666666667')


# LLM-generated content at query #9
#--------------------------

```python
def test_DCC_interest():
    # Create a simple DCC instance with a dummy calculate_fraction_method
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")  # Always return 0.5 for testing
    
    dcc = DCC(
        name="test_dcc",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    # Test with basic parameters
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.1")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    
    # Expected interest = 100 * 0.1 * 0.5 = 5
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money("5", Currencies["USD"])
    assert result.ccy == Currencies["USD"]
    
    # Test with asof == start (should return 0)
    result = dcc.interest(principal, rate, start, start, end)
    assert result == Money("0", Currencies["USD"])
    
    # Test with asof == end (should use full fraction)
    result = dcc.interest(principal, rate, start, end, end)
    assert result == Money("5", Currencies["USD"])
    
    # Test with asof < start (should return 0)
    early_asof = datetime.date(2022, 12, 31)
    result = dcc.interest(principal, rate, start, early_asof, end)
    assert result == Money("0", Currencies["USD"])
    
    # Test with asof > end (should return 0)
    late_asof = datetime.date(2024, 1, 1)
    result = dcc.interest(principal, rate, start, late_asof, end)
    assert result == Money("0", Currencies["USD"])
    
    # Test with None end (should use asof as end)
    result = dcc.interest(principal, rate, start, asof, None)
    assert result == Money("5", Currencies["USD"])
    
    # Test with different currency
    principal_eur = Money("100", Currencies["EUR"])
    result = dcc.interest(principal_eur, rate, start, asof, end)
    assert result == Money("5", Currencies["EUR"])
    assert result.ccy == Currencies["EUR"]


# LLM-generated content at query #10
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Mock DCFC function that returns a fixed fraction
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal('0.1')

    # Create a DCC instance with the mock function
    dcc = DCC(name="mock", altnames=set(), currencies=set(), calculate_fraction_method=mock_calculate_fraction)

    # Define test dates
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 1, 2)
    end_date = datetime.date(2023, 1, 3)

    # Calculate daily fraction
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)

    # Assert the result
    assert daily_fraction == Decimal('0.1')

    # Test with asof_date before start_date
    asof_date_before_start = datetime.date(2022, 12, 31)
    daily_fraction_before_start = dcc.calculate_daily_fraction(start_date, asof_date_before_start, end_date)
    assert daily_fraction_before_start == Decimal('0')

    # Test with asof_date equal to start_date
    asof_date_equal_start = datetime.date(2023, 1, 1)
    daily_fraction_equal_start = dcc.calculate_daily_fraction(start_date, asof_date_equal_start, end_date)
    assert daily_fraction_equal_start == Decimal('0.1')

    # Test with asof_date equal to end_date
    asof_date_equal_end = datetime.date(2023, 1, 3)
    daily_fraction_equal_end = dcc.calculate_daily_fraction(start_date, asof_date_equal_end, end_date)
    assert daily_fraction_equal_end == Decimal('0.1')


# LLM-generated content at query #11
#--------------------------

```python
def test_DCC_interest():
    # Define a simple DCFC function for testing
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal((asof - start).days) / Decimal((end - start).days)

    # Create a DCC instance with the simple_dcfc function
    dcc = DCC(
        name="SimpleDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )

    # Define test parameters
    principal = Money(amount=Decimal("1000"), currency=Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 31)

    # Calculate expected interest
    expected_interest = principal * rate * Decimal((asof - start).days) / Decimal((end - start).days)

    # Test the interest calculation
    assert dcc.interest(principal, rate, start, asof, end) == expected_interest

    # Test with asof date outside the range
    asof_out_of_range = datetime.date(2023, 2, 1)
    assert dcc.interest(principal, rate, start, asof_out_of_range, end) == Money(amount=Decimal("0"), currency=Currencies["USD"])

    # Test with end date equal to asof date
    assert dcc.interest(principal, rate, start, end, end) == principal * rate * Decimal("1")

    # Test with start date equal to asof date
    assert dcc.interest(principal, rate, start, start, end) == Money(amount=Decimal("0"), currency=Currencies["USD"])

    # Test with end date before asof date
    asof_after_end = datetime.date(2023, 2, 1)
    end_before_asof = datetime.date(2023, 1, 30)
    assert dcc.interest(principal, rate, start, asof_after_end, end_before_asof) == Money(amount=Decimal("0"), currency=Currencies["USD"])

    # Test with negative rate
    negative_rate = Decimal("-0.05")
    assert dcc.interest(principal, negative_rate, start, asof, end) == Money(amount=Decimal("-1000") * negative_rate * Decimal((asof - start).days) / Decimal((end - start).days), currency=Currencies["USD"])


# LLM-generated content at query #12
#--------------------------

def test_dcfc_act_365_a():
    # Test case 1: Non-leap year period
    start1 = Date(2007, 12, 28)
    asof1 = Date(2008, 2, 28)
    expected1 = Decimal('0.16986301369863')
    result1 = dcfc_act_365_a(start=start1, asof=asof1, end=asof1)
    assert round(result1, 14) == expected1

    # Test case 2: Period including leap day
    start2 = Date(2007, 12, 28)
    asof2 = Date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    result2 = dcfc_act_365_a(start=start2, asof=asof2, end=asof2)
    assert round(result2, 14) == expected2

    # Test case 3: Longer period spanning multiple years
    start3 = Date(2007, 10, 31)
    asof3 = Date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    result3 = dcfc_act_365_a(start=start3, asof=asof3, end=asof3)
    assert round(result3, 14) == expected3

    # Test case 4: Period spanning leap year
    start4 = Date(2008, 2, 1)
    asof4 = Date(2009, 5, 31)
    expected4 = Decimal('1.32513661202186')
    result4 = dcfc_act_365_a(start=start4, asof=asof4, end=asof4)
    assert round(result4, 14) == expected4

    # Test case 5: Same day
    start5 = Date(2020, 1, 1)
    asof5 = Date(2020, 1, 1)
    expected5 = Decimal('0.0')
    result5 = dcfc_act_365_a(start=start5, asof=asof5, end=asof5)
    assert result5 == expected5

    # Test case 6: Period entirely within a leap year
    start6 = Date(2020, 1, 1)
    asof6 = Date(2020, 12, 31)
    expected6 = Decimal('366') / Decimal('366')
    result6 = dcfc_act_365_a(start=start6, asof=asof6, end=asof6)
    assert result6 == expected6


# LLM-generated content at query #13
#--------------------------

def test_dcfc_nl_365():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_nl_365(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day in period but not counted
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_nl_365(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16986301369863')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_nl_365(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08219178082192')

    # Test case 4: Multi-year period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32602739726027')

    # Test case 5: Single day calculation
    start5 = asof5 = datetime.date(2020, 1, 1)
    assert round(dcfc_nl_365(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.00273972602740')

    # Test case 6: Leap day period but not leap year
    start6 = datetime.date(2019, 2, 28)
    asof6 = datetime.date(2019, 3, 1)
    assert round(dcfc_nl_365(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.00273972602740')


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

def test_DCC_coupon():
    # Create a test DCC instance with a simple calculate_fraction_method
    def simple_calculate_fraction(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal((asof - start).days) / Decimal(365)

    test_dcc = DCC(
        name="TEST",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_calculate_fraction
    )

    # Test case 1: Basic coupon calculation
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 7, 1)
    end = datetime.date(2024, 1, 1)
    freq = 2
    result = test_dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal((asof - start).days) / Decimal(365)
    assert result == expected

    # Test case 2: Asof date before start date
    asof = datetime.date(2022, 12, 31)
    result = test_dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money("0", Currencies["USD"])

    # Test case 3: Asof date equals start date
    asof = datetime.date(2023, 1, 1)
    result = test_dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money("0", Currencies["USD"])

    # Test case 4: Asof date equals end date
    asof = datetime.date(2024, 1, 1)
    result = test_dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal((asof - start).days) / Decimal(365)
    assert result == expected

    # Test case 5: Different frequency
    freq = 4
    asof = datetime.date(2023, 4, 1)
    result = test_dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal((asof - start).days) / Decimal(365)
    assert result == expected

    # Test case 6: With end-of-month adjustment
    eom = 31
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 15)
    freq = 1
    result = test_dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal((asof - start).days) / Decimal(365)
    assert result == expected


# LLM-generated content at query #16
#--------------------------

def test_dcfc_30_360_german():
    # Test case 1: Regular dates
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.16666666666667')

    # Test case 2: Leap day in asof date
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.16944444444444')

    # Test case 3: 31st day in start date
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('1.08333333333333')

    # Test case 4: February end date
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('1.33055555555556')

    # Test case 5: February last day in start date
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 3, 31)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.08333333333333')

    # Test case 6: February last day in asof date (not end date)
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 3, 31)
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.08333333333333')


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

def test_dcfc_30_360_german():
    # Test case 1: Regular dates
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_german(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    # Test case 2: Leap day in asof date
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_german(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date has 31st day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_german(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    # Test case 4: Multi-year period
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_german(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33055555555556')

    # Test case 5: February end dates
    start5 = datetime.date(2008, 2, 29)  # Leap year
    asof5 = datetime.date(2009, 2, 28)   # Non-leap year
    assert round(dcfc_30_360_german(start=start5, asof=asof5, end=asof5), 14) == Decimal('0.99722222222222')

    # Test case 6: Start and asof on same day
    start6 = datetime.date(2010, 5, 15)
    asof6 = datetime.date(2010, 5, 15)
    assert round(dcfc_30_360_german(start=start6, asof=asof6, end=asof6), 14) == Decimal('0.0')


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_act_365_a():
    start1 = Date(2007, 12, 28)
    asof1 = Date(2008, 2, 28)
    assert round(dcfc_act_365_a(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')

    start2 = Date(2007, 12, 28)
    asof2 = Date(2008, 2, 29)
    assert round(dcfc_act_365_a(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.17213114754098')

    start3 = Date(2007, 10, 31)
    asof3 = Date(2008, 11, 30)
    assert round(dcfc_act_365_a(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08196721311475')

    start4 = Date(2008, 2, 1)
    asof4 = Date(2009, 5, 31)
    assert round(dcfc_act_365_a(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32513661202186')


# LLM-generated content at query #21
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Act/Act",
        altnames={"Actual/Actual"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.16942884946478"),
    )
    registry.register(dcc)

    # Test finding by main name
    assert registry.find("Act/Act") == dcc

    # Test finding by alternative name
    assert registry.find("Actual/Actual") == dcc

    # Test finding by name with leading/trailing spaces
    assert registry.find("  Act/Act  ") == dcc

    # Test finding by name with different case
    assert registry.find("ACT/ACT") == dcc

    # Test finding non-existent name
    assert registry.find("NonExistent") is None


# LLM-generated content at query #22
#--------------------------

def test_dcfc_nl_365():
    # Test case 1: No leap day in period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_nl_365(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16986301369863')

    # Test case 2: Leap day in period but not counted
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_nl_365(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16986301369863')

    # Test case 3: Longer period with leap day
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_nl_365(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08219178082192')

    # Test case 4: Multi-year period with leap day
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.32602739726027')

    # Test case 5: Period entirely within a leap year but no Feb 29
    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 3, 1)
    end5 = datetime.date(2020, 12, 31)
    # Expected: (31 + 29 + 1) / 365 = 61/365 ≈ 0.16712328767123
    assert round(dcfc_nl_365(start=start5, asof=asof5, end=end5), 14) == Decimal('0.16712328767123')

    # Test case 6: Period crossing multiple years with multiple leap days
    start6 = datetime.date(2019, 12, 31)
    asof6 = datetime.date(2021, 1, 1)
    end6 = datetime.date(2021, 12, 31)
    # Expected: (1 + 366 + 1 - 1) / 365 = 367/365 ≈ 1.00547945205479
    assert round(dcfc_nl_365(start=start6, asof=asof6, end=end6), 14) == Decimal('1.00547945205479')


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('0.16942884946478')

    # Test case 2
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('0.17216108990194')

    # Test case 3
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('1.08243131970956')

    # Test case 4
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('1.32625945055768')

    # Test case 5: Same start and asof date
    start = datetime.date(2007, 12, 28)
    asof = start
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('0.00000000000000')

    # Test case 6: Leap year example
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('1.00000000000000')

    # Test case 7: Non-leap year example
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 12, 31)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('1.00000000000000')

    # Test case 8: Spanning multiple years with leap and non-leap years
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2021, 12, 31)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('3.00000000000000')

    # Test case 9: Spanning partial years with leap day
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('0.00273224043716')

    # Test case 10: Spanning partial years without leap day
    start = datetime.date(2019, 2, 28)
    asof = datetime.date(2019, 3, 1)
    end = asof
    assert round(dcfc_act_act(start, asof, end), 14) == Decimal('0.00273972602740')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act():
    # Test case 1: Non-leap year period
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16942884946478')

    # Test case 2: Leap year period
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17216108990194')

    # Test case 3: Period spanning multiple years
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08243131970956')

    # Test case 4: Period spanning multiple years including a leap year
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32625945055768')

    # Test case 5: Single day period
    single_day_start = datetime.date(2020, 1, 1)
    single_day_asof = datetime.date(2020, 1, 1)
    assert round(dcfc_act_act(start=single_day_start, asof=single_day_asof, end=single_day_asof), 14) == Decimal('0.00273224043716')

    # Test case 6: Period spanning exactly one year
    year_start = datetime.date(2020, 1, 1)
    year_asof = datetime.date(2021, 1, 1)
    assert round(dcfc_act_act(start=year_start, asof=year_asof, end=year_asof), 14) == Decimal('1.00000000000000')

    # Test case 7: Period spanning exactly one leap year
    leap_year_start = datetime.date(2020, 2, 28)
    leap_year_asof = datetime.date(2021, 2, 28)
    assert round(dcfc_act_act(start=leap_year_start, asof=leap_year_asof, end=leap_year_asof), 14) == Decimal('1.00273972602740')


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_e_360():
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=start1, asof=asof1, end=end1), 14) == Decimal('0.16666666666667')

    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=start2, asof=asof2, end=end2), 14) == Decimal('0.16944444444444')

    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=start3, asof=asof3, end=end3), 14) == Decimal('1.08333333333333')

    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=start4, asof=asof4, end=end4), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #4
#--------------------------

def test_DCC_interest():
    # Create a test DCC instance with a simple calculation method
    def simple_calculate_fraction(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal((asof - start).days) / Decimal((end - start).days)

    test_dcc = DCC(
        name="TEST",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_calculate_fraction
    )

    # Test with normal case
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 12, 31)
    
    result = test_dcc.interest(principal, rate, start, asof, end)
    expected = principal * rate * Decimal(30)/Decimal(364)
    assert result == expected

    # Test when asof == start
    result = test_dcc.interest(principal, rate, start, start, end)
    assert result == Money("0", Currencies["USD"])

    # Test when asof == end
    result = test_dcc.interest(principal, rate, start, end, end)
    expected = principal * rate * Decimal(1)
    assert result == expected

    # Test when asof is before start
    result = test_dcc.interest(principal, rate, start, datetime.date(2022, 12, 31), end)
    assert result == Money("0", Currencies["USD"])

    # Test when asof is after end
    result = test_dcc.interest(principal, rate, start, datetime.date(2024, 1, 1), end)
    assert result == Money("0", Currencies["USD"])

    # Test with different currency
    principal_eur = Money("100", Currencies["EUR"])
    result = test_dcc.interest(principal_eur, rate, start, asof, end)
    expected = principal_eur * rate * Decimal(30)/Decimal(364)
    assert result == expected
    assert result.currency == Currencies["EUR"]


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_nl_365():
    ex1_start, ex1_asof = date(2007, 12, 28), date(2008, 2, 28)
    ex2_start, ex2_asof = date(2007, 12, 28), date(2008, 2, 29)
    ex3_start, ex3_asof = date(2007, 10, 31), date(2008, 11, 30)
    ex4_start, ex4_asof = date(2008, 2, 1), date(2009, 5, 31)

    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')


# LLM-generated content at query #6
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    registry = DCCRegistryMachinery()

    # Create a sample DCC
    dcc = DCC(
        name="Act/Act",
        altnames={"Actual/Actual"},
        currencies={Currencies["USD"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )

    # Register the DCC
    registry.register(dcc)

    # Verify that the DCC is registered
    assert registry.find("Act/Act") == dcc
    assert registry.find("Actual/Actual") == dcc

    # Attempt to register the same DCC again should raise a TypeError
    try:
        registry.register(dcc)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Attempt to register a DCC with a conflicting name should raise a TypeError
    conflicting_dcc = DCC(
        name="Act/Act",
        altnames={"SomeOtherName"},
        currencies={Currencies["EUR"]},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    try:
        registry.register(conflicting_dcc)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #7
#--------------------------

def test_dcfc_30_360_german():
    # Test case 1: Regular dates
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.16666666666667')

    # Test case 2: Leap day in asof date
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.16944444444444')

    # Test case 3: Start date is last day of February (non-leap year)
    start = datetime.date(2007, 2, 28)
    asof = datetime.date(2007, 3, 31)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.08333333333333')

    # Test case 4: Start date is last day of February (leap year)
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 3, 31)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.08333333333333')

    # Test case 5: Start date is 31st
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('1.08333333333333')

    # Test case 6: Multi-year period
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('1.33055555555556')

    # Test case 7: As of date is last day of February but not end date
    start = datetime.date(2007, 1, 15)
    asof = datetime.date(2007, 2, 28)
    end = datetime.date(2007, 3, 15)
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.13888888888889')

    # Test case 8: As of date is last day of February and is end date
    start = datetime.date(2007, 1, 15)
    asof = datetime.date(2007, 2, 28)
    end = asof
    assert round(dcfc_30_360_german(start=start, asof=asof, end=end), 14) == Decimal('0.13888888888889')


# LLM-generated content at query #8
#--------------------------

def test_DCCRegistryMachinery_register():
    # Create a test DCC
    test_dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt1", "TestAlt2"},
        currencies=_as_ccys({"USD", "EUR"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )

    # Create registry instance
    registry = DCCRegistryMachinery()

    # Test successful registration
    registry.register(test_dcc)
    assert registry._buffer_main["TestDCC"] == test_dcc
    assert registry._buffer_altn["TestAlt1"] == test_dcc
    assert registry._buffer_altn["TestAlt2"] == test_dcc

    # Test duplicate registration raises TypeError
    try:
        registry.register(test_dcc)
        assert False, "Should have raised TypeError for duplicate registration"
    except TypeError:
        pass

    # Test registration with duplicate altname raises TypeError
    test_dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt1"},  # Duplicate with first registration
        currencies=_as_ccys({"GBP"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3")
    )
    try:
        registry.register(test_dcc2)
        assert False, "Should have raised TypeError for duplicate altname"
    except TypeError:
        pass

    # Test registration with duplicate main name raises TypeError
    test_dcc3 = DCC(
        name="TestDCC",  # Duplicate with first registration
        altnames={"TestAlt3"},
        currencies=_as_ccys({"JPY"}),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.1")
    )
    try:
        registry.register(test_dcc3)
        assert False, "Should have raised TypeError for duplicate main name"
    except TypeError:
        pass


# LLM-generated content at query #9
#--------------------------

def test_DCC_interest():
    # Test with simple case - 1 day interest
    dcc = DCC(
        name="ACT/365",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(365)
    )
    principal = Money("100", Currencies["USD"])
    rate = Decimal("0.1")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    result = dcc.interest(principal, rate, start, asof)
    expected = Money("0.0273972602739726", Currencies["USD"])
    assert result == expected

    # Test with zero days
    result = dcc.interest(principal, rate, start, start)
    expected = Money("0", Currencies["USD"])
    assert result == expected

    # Test with end date different from asof
    end = datetime.date(2023, 1, 3)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money("0.0273972602739726", Currencies["USD"])
    assert result == expected

    # Test with asof before start date
    asof = datetime.date(2022, 12, 31)
    result = dcc.interest(principal, rate, start, asof)
    expected = Money("0", Currencies["USD"])
    assert result == expected

    # Test with asof after end date
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 4)
    end = datetime.date(2023, 1, 3)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money("0", Currencies["USD"])
    assert result == expected

    # Test with different currency
    principal = Money("100", Currencies["EUR"])
    result = dcc.interest(principal, rate, start, datetime.date(2023, 1, 2))
    expected = Money("0.0273972602739726", Currencies["EUR"])
    assert result == expected

    # Test with zero rate
    rate = Decimal("0")
    result = dcc.interest(principal, rate, start, datetime.date(2023, 1, 2))
    expected = Money("0", Currencies["EUR"])
    assert result == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_nl_365():
    # Test case 1: No leap day in the period
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = asof1
    result1 = round(dcfc_nl_365(start1, asof1, end1), 14)
    assert result1 == Decimal('0.16986301369863')

    # Test case 2: Leap day in the period
    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    end2 = asof2
    result2 = round(dcfc_nl_365(start2, asof2, end2), 14)
    assert result2 == Decimal('0.16986301369863')

    # Test case 3: Period spanning multiple years
    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    end3 = asof3
    result3 = round(dcfc_nl_365(start3, asof3, end3), 14)
    assert result3 == Decimal('1.08219178082192')

    # Test case 4: Period spanning more than one year
    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    end4 = asof4
    result4 = round(dcfc_nl_365(start4, asof4, end4), 14)
    assert result4 == Decimal('1.32602739726027')


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_30_360_isda():
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=start1, asof=asof1, end=asof1), 14) == Decimal('0.16666666666667')

    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=start2, asof=asof2, end=asof2), 14) == Decimal('0.16944444444444')

    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=start3, asof=asof3, end=asof3), 14) == Decimal('1.08333333333333')

    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=start4, asof=asof4, end=asof4), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #12
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Mock the calculate_fraction_method to return specific values for testing
    def mock_calculate_fraction(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        if asof == datetime.date(2023, 1, 1):
            return Decimal("0.1")
        elif asof == datetime.date(2023, 1, 2):
            return Decimal("0.2")
        else:
            return Decimal("0.0")

    # Create a DCC instance with the mocked method
    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction,
    )

    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 3)

    # Test case where asof is start_date
    asof_date = datetime.date(2023, 1, 1)
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert daily_fraction == Decimal("0.1")

    # Test case where asof is one day after start_date
    asof_date = datetime.date(2023, 1, 2)
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert daily_fraction == Decimal("0.1")

    # Test case where asof is two days after start_date
    asof_date = datetime.date(2023, 1, 3)
    daily_fraction = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert daily_fraction == Decimal("0.0")


# LLM-generated content at query #13
#--------------------------

```python
def test_DCC_interest():
    # Mock DCFC function that returns a fixed fraction
    def mock_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.5")

    # Create a DCC instance with the mock DCFC
    dcc = DCC(
        name="MockDCC",
        altnames={"mock"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=mock_dcfc,
    )

    # Define test parameters
    principal = Money(Decimal("1000"), Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)

    # Calculate interest
    interest = dcc.interest(principal, rate, start, asof, end)

    # Assert the result
    assert interest.amount == Decimal("25.0")
    assert interest.currency == Currencies["USD"]


# LLM-generated content at query #14
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


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

def test_dcfc_act_act_icma():
    # Test case 1: Regular period calculation
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.5245901639')

    # Test case 2: Same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result == Decimal(0)

    # Test case 3: Full period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result == Decimal(1)

    # Test case 4: Different frequency
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(4)
    result = dcfc_act_act_icma(start, asof, end, freq)
    expected = Decimal(91) / Decimal(366) / Decimal(4)
    assert round(result, 10) == round(expected, 10)

    # Test case 5: Leap year calculation
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2021, 2, 28)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    expected = Decimal(2) / Decimal(366)
    assert result == expected


# LLM-generated content at query #17
#--------------------------

Here's a unit test for the `calculate_daily_fraction` method of the `DCC` class:


# LLM-generated content at query #18
#--------------------------

```python
def test_DCC_interest():
    # Mock DCFC function for testing
    def mock_calculate_fraction(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal((asof - start).days) / Decimal((end - start).days)

    # Create a DCC instance with mock function
    dcc = DCC(
        name="mock_dcc",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test case 1: No interest accrued
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result.amount == Decimal("0")

    # Test case 2: Interest accrued for one day
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 1, 2)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result.amount == Decimal("1000") * Decimal("0.05") / Decimal("1")

    # Test case 3: Interest accrued for multiple days
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 3)
    end = datetime.date(2023, 1, 3)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result.amount == Decimal("1000") * Decimal("0.05") * Decimal("2") / Decimal("2")

    # Test case 4: asof date before start date
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 2)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 3)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result.amount == Decimal("0")

    # Test case 5: asof date after end date
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 4)
    end = datetime.date(2023, 1, 3)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result.amount == Decimal("1000") * Decimal("0.05") * Decimal("2") / Decimal("2")


# LLM-generated content at query #19
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    registry = DCCRegistryMachinery()
    
    # Define a DCC instance
    dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("1.0")
    )
    
    # Register the DCC
    registry.register(dcc)
    
    # Check if the DCC is registered
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt") == dcc
    
    # Attempt to register the same DCC again should raise a TypeError
    try:
        registry.register(dcc)
        assert False, "Expected TypeError when registering duplicate DCC"
    except TypeError:
        pass
    
    # Attempt to register a DCC with an already registered altname should raise a TypeError
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt"},
        currencies=_as_ccys({"EUR"}),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("2.0")
    )
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError when registering DCC with duplicate altname"
    except TypeError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_DCC_coupon():
    # Create a sample DCC instance with a dummy calculate_fraction_method
    def dummy_calculate_fraction(start: Date, asof: Date, end: Date, freq: Optional[Decimal] = None) -> Decimal:
        return Decimal((asof - start).days) / Decimal((end - start).days)

    dcc = DCC(
        name="SampleDCC",
        altnames={"Sample", "Test"},
        currencies=_as_ccys({"USD"}),
        calculate_fraction_method=dummy_calculate_fraction,
    )

    # Create sample Money instance
    principal = Money(Decimal("1000.00"), Currencies["USD"])

    # Define sample dates
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 30)
    end_date = datetime.date(2023, 12, 31)

    # Define sample rate and frequency
    rate = Decimal("0.05")
    frequency = Decimal("2")

    # Calculate expected interest manually
    prev_date = _last_payment_date(start_date, asof_date, frequency)
    next_date = _next_payment_date(prev_date, frequency)
    expected_fraction = dummy_calculate_fraction(prev_date, asof_date, next_date, frequency)
    expected_interest = principal * rate * expected_fraction

    # Test the coupon method
    calculated_interest = dcc.coupon(principal, rate, start_date, asof_date, end_date, frequency)

    assert calculated_interest == expected_interest


# LLM-generated content at query #21
#--------------------------

def test_dcfc_act_act_icma():
    # Test case 1: Regular period calculation
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.5245901639')

    # Test case 2: Same day calculation
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result == Decimal('0')

    # Test case 3: Full period calculation
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert result == Decimal('1')

    # Test case 4: Multiple frequency periods
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(4)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.5')

    # Test case 5: Leap year calculation
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 8, 28)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.0163934426')


# LLM-generated content at query #22
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    # Mock DCFC function
    def mock_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        # Simple mock that returns 0.5 for any date range
        return Decimal('0.5')

    # Create a DCC instance with the mock DCFC
    dcc = DCC(
        name="TEST",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )

    # Test case 1: asof is same as start date
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal('0.5')  # tfact (0.5) - yfact (0)

    # Test case 2: asof is one day after start
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal('0.0')  # tfact (0.5) - yfact (0.5)

    # Test case 3: asof is before start date
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2022, 12, 31)
    end = datetime.date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal('0.0')

    # Test case 4: asof is after end date
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2024, 1, 1)
    end = datetime.date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal('0.0')

    # Test case 5: multiple days with custom DCFC that returns day count
    def day_count_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal((asof - start).days)

    dcc_day_count = DCC(
        name="DAYCOUNT",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=day_count_dcfc
    )

    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 3)
    end = datetime.date(2023, 12, 31)
    result = dcc_day_count.calculate_daily_fraction(start, asof, end)
    assert result == Decimal('1')  # (2 days) - (1 day)


# LLM-generated content at query #23
#--------------------------

def test_dcfc_act_act_icma():
    # Test case 1: Regular period calculation
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.5245901639')

    # Test case 2: Full period calculation (asof == end)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(1)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('1.0')

    # Test case 3: Partial period with different frequency
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal(4)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.5')

    # Test case 4: Single day calculation
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 3, 31)
    freq = Decimal(4)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.0081967213')

    # Test case 5: Leap year calculation
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 8, 28)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.0163934426')


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


