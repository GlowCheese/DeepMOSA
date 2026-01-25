####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda():
    import datetime
    
    # Test case 1: ex1_start, ex1_asof
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: ex2_start, ex2_asof
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: ex3_start, ex3_asof
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: ex4_start, ex4_asof
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test case: Same start and asof dates
    same_start = datetime.date(2008, 3, 15)
    same_asof = datetime.date(2008, 3, 15)
    result_same = dcfc_30_360_isda(start=same_start, asof=same_asof, end=same_asof)
    assert result_same == Decimal('0')
    
    # Test case: Start date is 31st day (should be adjusted to 30th)
    start_31 = datetime.date(2008, 1, 31)
    asof_31 = datetime.date(2008, 2, 28)
    result_31 = dcfc_30_360_isda(start=start_31, asof=asof_31, end=asof_31)
    assert result_31 == Decimal('28') / Decimal(360)
    
    # Test case: Start day is 30 and asof day is 31 (asof should be adjusted to 30)
    start_30 = datetime.date(2008, 1, 30)
    asof_31_adj = datetime.date(2008, 2, 31)
    result_30_31 = dcfc_30_360_isda(start=start_30, asof=asof_31_adj, end=asof_31_adj)
    assert result_30_31 == Decimal('0')
    
    # Test case: Multiple years
    multi_year_start = datetime.date(2006, 6, 15)
    multi_year_asof = datetime.date(2008, 12, 20)
    result_multi = dcfc_30_360_isda(start=multi_year_start, asof=multi_year_asof, end=multi_year_asof)
    expected_nod = (20 - 15) + 30 * (12 - 6) + 360 * (2008 - 2006)
    assert result_multi == expected_nod / Decimal(360)


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act():
    """Unit tests for dcfc_act_act function."""
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16942884946478')
    
    # Test case 2: Leap year day (Feb 29)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17216108990194')
    
    # Test case 3: Longer period spanning leap year
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08243131970956')
    
    # Test case 4: Multi-year period
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32625945055768')
    
    # Test case 5: Same start and asof date (zero day count)
    result5 = dcfc_act_act(start=datetime.date(2020, 1, 1), asof=datetime.date(2020, 1, 1), end=datetime.date(2020, 1, 1))
    assert result5 == Decimal('0')
    
    # Test case 6: One day period in non-leap year
    result6 = dcfc_act_act(start=datetime.date(2019, 1, 1), asof=datetime.date(2019, 1, 2), end=datetime.date(2019, 1, 2))
    assert result6 == Decimal('1') / Decimal('365')
    
    # Test case 7: One day period in leap year
    result7 = dcfc_act_act(start=datetime.date(2020, 2, 29), asof=datetime.date(2020, 3, 1), end=datetime.date(2020, 3, 1))
    assert result7 == Decimal('1') / Decimal('366')
    
    # Test case 8: Period within single non-leap year
    result8 = dcfc_act_act(start=datetime.date(2019, 1, 1), asof=datetime.date(2019, 12, 31), end=datetime.date(2019, 12, 31))
    assert result8 == Decimal('364') / Decimal('365')
    
    # Test case 9: With frequency parameter (should not affect calculation)
    result9 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result9, 14) == Decimal('0.16942884946478')
    
    # Test case 10: Period spanning multiple leap and non-leap years
    result10 = dcfc_act_act(start=datetime.date(2015, 1, 1), asof=datetime.date(2017, 12, 31), end=datetime.date(2017, 12, 31))
    assert isinstance(result10, Decimal)
    assert result10 > Decimal('2')  # Should be approximately 3 years worth of days


# LLM-generated content at query #3
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    """
    Test the register method of DCCRegistryMachinery class.
    """
    # Create a new registry instance
    registry = DCCRegistryMachinery()
    
    # Create a simple day count fraction calculation function
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.5")
    
    # Create a test DCC object
    test_dcc = DCC(
        name="Test/DCC",
        altnames={"Test/Alternative", "TestAlt"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test successful registration
    registry.register(test_dcc)
    
    # Verify it's registered in the main buffer
    assert registry._find_strict("Test/DCC") == test_dcc
    
    # Verify alternative names are registered
    assert registry._find_strict("Test/Alternative") == test_dcc
    assert registry._find_strict("TestAlt") == test_dcc
    
    # Verify it appears in the registry
    assert test_dcc in registry.registry
    
    # Verify it appears in the table
    assert registry.table["Test/DCC"] == test_dcc
    assert registry.table["Test/Alternative"] == test_dcc
    assert registry.table["TestAlt"] == test_dcc


def test_DCCRegistryMachinery_register_duplicate_main_name():
    """
    Test that registering a DCC with a name that's already registered raises TypeError.
    """
    registry = DCCRegistryMachinery()
    
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.5")
    
    dcc1 = DCC(
        name="Duplicate/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    dcc2 = DCC(
        name="Duplicate/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Register the first one
    registry.register(dcc1)
    
    # Attempt to register the second one with same name should raise TypeError
    with pytest.raises(TypeError, match="Day count convention 'Duplicate/DCC' is already registered"):
        registry.register(dcc2)


def test_DCCRegistryMachinery_register_duplicate_altname():
    """
    Test that registering a DCC with an alternative name that's already registered raises TypeError.
    """
    registry = DCCRegistryMachinery()
    
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.5")
    
    dcc1 = DCC(
        name="First/DCC",
        altnames={"Shared/Alt"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    dcc2 = DCC(
        name="Second/DCC",
        altnames={"Shared/Alt"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Register the first one
    registry.register(dcc1)
    
    # Attempt to register the second one with conflicting altname should raise TypeError
    with pytest.raises(TypeError, match="Day count convention 'Second/DCC' is already registered"):
        registry.register(dcc2)


def test_DCCRegistryMachinery_register_altname_conflicts_with_main():
    """
    Test that registering a DCC with an alternative name that conflicts with an existing main name raises TypeError.
    """
    registry = DCCRegistryMachinery()
    
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.5")
    
    dcc1 = DCC(
        name="First/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    dcc2 = DCC(
        name="Second/DCC",
        altnames={"First/DCC"},  # This conflicts with dcc1's main name
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Register the first one
    registry.register(dcc1)
    
    # Attempt to register the second one with conflicting altname should raise TypeError
    with pytest.raises(TypeError, match="Day count convention 'Second/DCC' is already registered"):
        registry.register(dcc2)


def test_DCCRegistryMachinery_register_with_currencies():
    """
    Test registering a DCC with associated currencies.
    """
    registry = DCCRegistryMachinery()
    
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal("0.5")
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    
    dcc = DCC(
        name="Currency/DCC",
        altnames={"CurrencyAlt"},
        currencies={usd, eur},
        calculate_fraction_method=simple_dcfc
    )
    
    registry.register(dcc)
    
    # Verify the DCC is registered with its currencies
    registered_dcc = registry._find_strict("Currency/DCC")
    assert registered_dcc is not None
    assert usd in registered_dcc.currencies
    assert eur in registered_dcc.currencies


# LLM-generated content at query #4
#--------------------------

```python
def test_dcfc_30_360_isda():
    """Unit tests for dcfc_30_360_isda function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: 2007-12-28 to 2008-02-29
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test case 5: Same start and asof date
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_30_360_isda(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal(0)
    
    # Test case 6: Start date is 31st, should be adjusted to 30th
    start_31 = datetime.date(2008, 1, 31)
    asof_date = datetime.date(2008, 2, 15)
    result6 = dcfc_30_360_isda(start=start_31, asof=asof_date, end=asof_date)
    # Adjusted start: 2008-01-30, asof: 2008-02-15
    # nod = (15 - 30) + 30 * (2 - 1) + 360 * (2008 - 2008) = -15 + 30 + 0 = 15
    assert result6 == Decimal(15) / Decimal(360)
    
    # Test case 7: Start 30th, asof 31st, should adjust asof to 30th
    start_30 = datetime.date(2008, 1, 30)
    asof_31 = datetime.date(2008, 2, 31)
    # Note: 2008-02-31 doesn't exist, using 2008-03-31 instead
    asof_31 = datetime.date(2008, 3, 31)
    result7 = dcfc_30_360_isda(start=start_30, asof=asof_31, end=asof_31)
    # Adjusted asof: 2008-03-30
    # nod = (30 - 30) + 30 * (3 - 1) + 360 * (2008 - 2008) = 0 + 60 + 0 = 60
    assert result7 == Decimal(60) / Decimal(360)
    
    # Test case 8: Multiple years difference
    start_multi = datetime.date(2006, 6, 15)
    asof_multi = datetime.date(2008, 6, 15)
    result8 = dcfc_30_360_isda(start=start_multi, asof=asof_multi, end=asof_multi)
    # nod = (15 - 15) + 30 * (6 - 6) + 360 * (2008 - 2006) = 0 + 0 + 720 = 720
    assert result8 == Decimal(720) / Decimal(360)
    
    # Test case 9: Negative day count (earlier asof than start)
    start_later = datetime.date(2008, 6, 15)
    asof_earlier = datetime.date(2008, 3, 15)
    result9 = dcfc_30_360_isda(start=start_later, asof=asof_earlier, end=asof_earlier)
    # nod = (15 - 15) + 30 * (3 - 6) + 360 * (2008 - 2008) = 0 - 90 + 0 = -90
    assert result9 == Decimal(-90) / Decimal(360)
    
    # Test case 10: With optional freq parameter (should be ignored)
    result10 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result10, 14) == Decimal('0.16666666666667')


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_act_365_a():
    """Test the dcfc_act_365_a function with various date ranges."""
    # Test case 1: From 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: From 2007-12-28 to 2008-02-29 (leap year day)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17213114754098')
    
    # Test case 3: From 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08196721311475')
    
    # Test case 4: From 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32513661202186')
    
    # Test case 5: Same start and asof date (should return 0)
    start = datetime.date(2008, 2, 1)
    result5 = dcfc_act_365_a(start=start, asof=start, end=start)
    assert result5 == Decimal('0')
    
    # Test case 6: Test with freq parameter (should be ignored)
    result6 = dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result6, 14) == Decimal('0.16986301369863')


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_act_act_icma():
    """Unit tests for dcfc_act_act_icma function."""
    
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result, 10) == Decimal('0.5245901639')
    
    # Test case 2: Start date equals asof date (should return 0)
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal('0')
    
    # Test case 3: Asof equals end date (should return 1 / freq)
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 3, 2)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal('1')
    
    # Test case 4: With explicit frequency parameter
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(2))
    assert round(result, 10) == Decimal('0.2622950820')
    
    # Test case 5: Single day period
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 3, 3)
    end = datetime.date(2019, 3, 4)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal('0.5')
    
    # Test case 6: Leap year period
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 2)
    end = datetime.date(2020, 2, 28)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    # This should handle leap year correctly
    assert isinstance(result, Decimal)
    
    # Test case 7: Full year period
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 12, 31)
    end = datetime.date(2020, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert isinstance(result, Decimal)
    assert result > Decimal('0')
    
    # Test case 8: Multi-year period
    start = datetime.date(2018, 1, 1)
    asof = datetime.date(2019, 6, 30)
    end = datetime.date(2020, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert isinstance(result, Decimal)
    assert Decimal('0') < result < Decimal('2')
    
    # Test case 9: With frequency of 4 (quarterly)
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(4))
    expected = round(Decimal('0.5245901639') / Decimal(4), 10)
    assert round(result, 10) == expected
    
    # Test case 10: Result should be between 0 and 1/freq
    start = datetime.date(2019, 1, 15)
    asof = datetime.date(2019, 6, 15)
    end = datetime.date(2020, 1, 15)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    assert Decimal('0') <= result <= Decimal('1') / freq


# LLM-generated content at query #7
#--------------------------

```python
def test_DCC_interest():
    """Test the interest method of DCC class."""
    from decimal import Decimal
    import datetime
    
    # Create a simple DCC instance with a basic calculation method
    def simple_dcfc(start, asof, end, freq):
        """Simple day count fraction: actual/360"""
        days = (asof - start).days
        return Decimal(days) / Decimal(360)
    
    dcc = DCC(
        name="Test DCC",
        altnames={"T-DCC"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test case 1: Basic interest calculation
    principal = Money(Decimal(1000), Currencies["USD"])
    rate = Decimal("0.05")  # 5%
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 11)  # 10 days later
    end = datetime.date(2023, 12, 31)
    
    result = dcc.interest(principal, rate, start, asof, end)
    # Expected: 1000 * 0.05 * (10/360) = 1.388888...
    expected = Money(Decimal(1000) * Decimal("0.05") * (Decimal(10) / Decimal(360)), Currencies["USD"])
    assert result.amount == expected.amount
    assert result.currency == expected.currency
    
    # Test case 2: Interest at start date (0 days)
    result = dcc.interest(principal, rate, start, start, end)
    expected = Money(Decimal(0), Currencies["USD"])
    assert result.amount == expected.amount
    
    # Test case 3: Interest with end date as asof
    end_date = datetime.date(2023, 2, 1)  # 31 days from start
    result = dcc.interest(principal, rate, start, end_date, end_date)
    expected = Money(Decimal(1000) * Decimal("0.05") * (Decimal(31) / Decimal(360)), Currencies["USD"])
    assert result.amount == expected.amount
    
    # Test case 4: When end is not provided, it defaults to asof
    result = dcc.interest(principal, rate, start, asof)
    expected = Money(Decimal(1000) * Decimal("0.05") * (Decimal(10) / Decimal(360)), Currencies["USD"])
    assert result.amount == expected.amount
    
    # Test case 5: Different principal amount
    principal_large = Money(Decimal(10000), Currencies["USD"])
    result = dcc.interest(principal_large, rate, start, asof, end)
    expected = Money(Decimal(10000) * Decimal("0.05") * (Decimal(10) / Decimal(360)), Currencies["USD"])
    assert result.amount == expected.amount
    
    # Test case 6: Zero rate
    result = dcc.interest(principal, Decimal(0), start, asof, end)
    expected = Money(Decimal(0), Currencies["USD"])
    assert result.amount == expected.amount
    
    # Test case 7: With frequency parameter
    freq = Decimal(4)  # Quarterly
    result = dcc.interest(principal, rate, start, asof, end, freq)
    expected = Money(Decimal(1000) * Decimal("0.05") * (Decimal(10) / Decimal(360)), Currencies["USD"])
    assert result.amount == expected.amount


# LLM-generated content at query #8
#--------------------------

```python
def test_dcfc_act_act_icma():
    """Unit tests for dcfc_act_act_icma function."""
    
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result1 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result1, 10) == Decimal('0.5245901639')
    
    # Test case 2: Start equals asof (zero days passed)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result2 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result2 == Decimal(0)
    
    # Test case 3: Start equals asof equals end (zero days passed)
    start = datetime.date(2020, 6, 15)
    asof = datetime.date(2020, 6, 15)
    end = datetime.date(2020, 6, 15)
    result3 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result3 == Decimal(0)
    
    # Test case 4: Full period (asof equals end)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result4 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result4 == Decimal(1)
    
    # Test case 5: Half period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 2)
    end = datetime.date(2021, 1, 1)
    result5 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert Decimal('0.49') < result5 < Decimal('0.51')
    
    # Test case 6: With frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    result6 = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(2))
    assert result6 > Decimal(0)
    assert result6 < Decimal(1)
    
    # Test case 7: Leap year considerations
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2021, 2, 28)
    result7 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert Decimal(0) < result7 < Decimal(1)
    
    # Test case 8: Result is always positive
    start = datetime.date(2019, 1, 15)
    asof = datetime.date(2019, 8, 20)
    end = datetime.date(2020, 1, 15)
    result8 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result8 > Decimal(0)
    
    # Test case 9: Result is always less than or equal to 1 when asof <= end
    start = datetime.date(2020, 3, 1)
    asof = datetime.date(2020, 9, 15)
    end = datetime.date(2021, 3, 1)
    result9 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result9 <= Decimal(1)
    
    # Test case 10: Different frequency values
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 31)
    end = datetime.date(2020, 12, 31)
    result10a = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(1))
    result10b = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(2))
    result10c = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(4))
    assert result10a > result10b > result10c


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_30_e_360():
    import datetime
    from decimal import Decimal
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: Example from docstring with leap day
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: Example from docstring spanning multiple months
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: Example from docstring spanning multiple years
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test case 5: Same start and asof date
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result5 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result5 == Decimal(0)
    
    # Test case 6: One day difference
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result6 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result6 == Decimal(1) / Decimal(360)
    
    # Test case 7: Start day is 31 (should be adjusted to 30)
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    result7 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # 30 days in February (adjusted) - 30 days in January (adjusted) + 30 * 1 = 30
    assert result7 == Decimal(30) / Decimal(360)
    
    # Test case 8: Asof day is 31 (should be adjusted to 30)
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 31)
    result8 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # 30 - 15 = 15
    assert result8 == Decimal(15) / Decimal(360)
    
    # Test case 9: Both start and asof are 31 (both adjusted to 30)
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 3, 31)
    result9 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    # (30 - 30) + 30 * (3 - 1) = 0 + 60 = 60
    assert result9 == Decimal(60) / Decimal(360)
    
    # Test case 10: With freq parameter (should be ignored)
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 31)
    result10 = dcfc_30_e_360(start=start, asof=asof, end=asof, freq=Decimal(2))
    assert result10 == Decimal(30) / Decimal(360)


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_act_act():
    """Unit tests for dcfc_act_act function."""
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16942884946478')
    
    # Test case 2: Example with leap day
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17216108990194')
    
    # Test case 3: Example spanning multiple years
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08243131970956')
    
    # Test case 4: Example spanning multiple years with leap year
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32625945055768')
    
    # Test case 5: Same start and asof date (zero day count)
    same_date = datetime.date(2010, 6, 15)
    result5 = dcfc_act_act(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: One day difference
    start_date = datetime.date(2010, 6, 15)
    end_date = datetime.date(2010, 6, 16)
    result6 = dcfc_act_act(start=start_date, asof=end_date, end=end_date)
    assert result6 == Decimal('1') / Decimal('365')
    
    # Test case 7: Non-leap year span
    start_date = datetime.date(2010, 1, 1)
    asof_date = datetime.date(2010, 12, 31)
    result7 = dcfc_act_act(start=start_date, asof=asof_date, end=asof_date)
    assert result7 == Decimal('365') / Decimal('365')
    
    # Test case 8: Leap year - full year
    start_date = datetime.date(2008, 1, 1)
    asof_date = datetime.date(2008, 12, 31)
    result8 = dcfc_act_act(start=start_date, asof=asof_date, end=asof_date)
    assert result8 == Decimal('366') / Decimal('366')
    
    # Test case 9: With frequency parameter (should be ignored by function)
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result9 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal('2'))
    assert round(result9, 14) == Decimal('0.16942884946478')
    
    # Test case 10: Verify result is Decimal type
    result10 = dcfc_act_act(start=datetime.date(2015, 1, 1), asof=datetime.date(2015, 1, 2), end=datetime.date(2015, 1, 2))
    assert isinstance(result10, Decimal)


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_act_365_l():
    """Test the dcfc_act_365_l function with various date ranges."""
    import datetime
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16939890710383')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17213114754098')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08196721311475')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32876712328767')
    
    # Test case 5: Same start and asof date (zero days)
    same_date = datetime.date(2020, 6, 15)
    result5 = dcfc_act_365_l(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: One day difference in non-leap year
    start6 = datetime.date(2019, 6, 15)
    asof6 = datetime.date(2019, 6, 16)
    result6 = dcfc_act_365_l(start=start6, asof=asof6, end=asof6)
    assert result6 == Decimal('1') / Decimal('365')
    
    # Test case 7: One day difference in leap year
    start7 = datetime.date(2020, 6, 15)
    asof7 = datetime.date(2020, 6, 16)
    result7 = dcfc_act_365_l(start=start7, asof=asof7, end=asof7)
    assert result7 == Decimal('1') / Decimal('366')


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_30_360_isda():
    """Test the dcfc_30_360_isda day count fraction function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap year)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test case 5: Same start and asof dates (zero day count)
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_30_360_isda(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Start date is 31st (should be adjusted to 30th)
    start_31 = datetime.date(2008, 5, 31)
    asof_30 = datetime.date(2008, 6, 30)
    result6 = dcfc_30_360_isda(start=start_31, asof=asof_30, end=asof_30)
    assert result6 == Decimal('1') / Decimal(12)
    
    # Test case 7: Start is 30th and asof is 31st (asof should be adjusted to 30th)
    start_30 = datetime.date(2008, 5, 30)
    asof_31 = datetime.date(2008, 6, 31) if False else datetime.date(2008, 6, 30)
    result7 = dcfc_30_360_isda(start=start_30, asof=asof_31, end=asof_31)
    assert result7 == Decimal('1') / Decimal(12)
    
    # Test case 8: Multiple years span
    start_multi = datetime.date(2006, 1, 1)
    asof_multi = datetime.date(2008, 12, 31)
    result8 = dcfc_30_360_isda(start=start_multi, asof=asof_multi, end=asof_multi)
    expected_days = (31 - 1) + 30 * (12 - 1) + 360 * (2008 - 2006)
    assert result8 == Decimal(expected_days) / Decimal(360)
    
    # Test case 9: Optional freq parameter should not affect result
    result9a = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=None)
    result9b = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal('2'))
    assert result9a == result9b


# LLM-generated content at query #13
#--------------------------

```python
def test_DCC_coupon():
    """Test the coupon method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Create a simple test DCC with a basic day count fraction method
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple day count fraction: actual/360"""
        days = (asof - start).days
        return Decimal(days) / Decimal(360)
    
    dcc = DCC(
        name="Test/360",
        altnames={"T/360"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test basic coupon calculation
    principal = Money(Decimal(1000), Currencies["USD"])
    rate = Decimal("0.05")  # 5% annual rate
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 15)
    end = datetime.date(2014, 12, 31)
    freq = 2  # Semi-annual
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert isinstance(result, Money)
    assert result.currency == Currencies["USD"]
    
    # Test coupon with end-of-month handling
    start_eom = datetime.date(2014, 1, 31)
    asof_eom = datetime.date(2014, 7, 15)
    eom = 31
    
    result_eom = dcc.coupon(principal, rate, start_eom, asof_eom, end, freq, eom)
    assert isinstance(result_eom, Money)
    
    # Test coupon with asof before start (should handle gracefully)
    asof_before = datetime.date(2013, 12, 31)
    result_before = dcc.coupon(principal, rate, start, asof_before, end, freq)
    assert isinstance(result_before, Money)
    
    # Test coupon with annual frequency
    result_annual = dcc.coupon(principal, rate, start, asof, end, 1)
    assert isinstance(result_annual, Money)
    
    # Test coupon with quarterly frequency
    result_quarterly = dcc.coupon(principal, rate, start, asof, end, 4)
    assert isinstance(result_quarterly, Money)
    
    # Test coupon with zero rate
    result_zero_rate = dcc.coupon(principal, Decimal(0), start, asof, end, freq)
    assert result_zero_rate.amount == Decimal(0)
    
    # Test coupon with different principal amount
    principal_large = Money(Decimal(10000), Currencies["USD"])
    result_large = dcc.coupon(principal_large, rate, start, asof, end, freq)
    assert isinstance(result_large, Money)
    assert result_large.currency == Currencies["USD"]
    
    # Test coupon calculation consistency
    result1 = dcc.coupon(principal, rate, start, asof, end, freq)
    result2 = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result1 == result2


# LLM-generated content at query #14
#--------------------------

```python
def test_DCCRegistryMachinery_register():
    """
    Test the register method of DCCRegistryMachinery class.
    """
    registry = DCCRegistryMachinery()
    
    # Create a simple day count fraction calculation function
    def simple_dcfc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        return Decimal((asof - start).days) / Decimal(365)
    
    # Test 1: Successfully register a new DCC
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Test1", "T1"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    registry.register(dcc1)
    
    # Verify it's in the main buffer
    assert registry._find_strict("Test/DCC1") == dcc1
    # Verify alternative names are in the alternative buffer
    assert registry._find_strict("Test1") == dcc1
    assert registry._find_strict("T1") == dcc1
    
    # Test 2: Attempt to register a DCC with a name that already exists
    dcc2 = DCC(
        name="Test/DCC1",  # Same name as dcc1
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    with pytest.raises(TypeError, match="Day count convention 'Test/DCC1' is already registered"):
        registry.register(dcc2)
    
    # Test 3: Attempt to register a DCC with an alternative name that conflicts
    dcc3 = DCC(
        name="Test/DCC3",
        altnames={"Test1"},  # Conflicts with dcc1's alternative name
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    with pytest.raises(TypeError, match="Day count convention 'Test/DCC3' is already registered"):
        registry.register(dcc3)
    
    # Test 4: Successfully register multiple DCCs with non-conflicting names
    dcc4 = DCC(
        name="Test/DCC4",
        altnames={"Test4", "T4"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    registry.register(dcc4)
    
    dcc5 = DCC(
        name="Test/DCC5",
        altnames={"Test5", "T5"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    registry.register(dcc5)
    
    # Verify all are registered
    assert registry._find_strict("Test/DCC4") == dcc4
    assert registry._find_strict("Test4") == dcc4
    assert registry._find_strict("Test/DCC5") == dcc5
    assert registry._find_strict("Test5") == dcc5
    
    # Test 5: Register DCC with empty alternative names
    dcc6 = DCC(
        name="Test/DCC6",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    registry.register(dcc6)
    assert registry._find_strict("Test/DCC6") == dcc6
    
    # Test 6: Verify registry property contains registered DCCs
    registered_dccs = registry.registry
    assert dcc1 in registered_dccs
    assert dcc4 in registered_dccs
    assert dcc5 in registered_dccs
    assert dcc6 in registered_dccs
    
    # Test 7: Verify table property contains all entries
    table = registry.table
    assert "Test/DCC1" in table
    assert "Test1" in table
    assert "Test/DCC4" in table
    assert "Test4" in table


# LLM-generated content at query #15
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    """Test DCC.calculate_daily_fraction method."""
    import datetime
    from decimal import Decimal
    
    # Create a simple test DCC with a basic calculation method
    def simple_fraction_method(start, asof, end, freq):
        """Simple fraction: days from start to asof divided by 365."""
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    dcc = DCC(
        name="Test DCC",
        altnames={"Test", "TDCC"},
        currencies=set(),
        calculate_fraction_method=simple_fraction_method
    )
    
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    
    # Test case 1: Daily fraction on the first day
    asof = datetime.date(2020, 1, 2)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    # Day 1 (Jan 2): fraction should be 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 2: Daily fraction on a later day
    asof = datetime.date(2020, 1, 10)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    # Day 10: fraction from day 9 to day 10 = (9/365) - (8/365) = 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 3: Daily fraction when asof equals start
    asof = datetime.date(2020, 1, 1)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    # On start date: 0/365 - 0 = 0
    assert daily_frac == ZERO
    
    # Test case 4: Daily fraction in the middle of the year
    asof = datetime.date(2020, 7, 1)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    # From Jan 1 to Jul 1 is 182 days, from Jan 1 to Jun 30 is 181 days
    # So daily fraction = 182/365 - 181/365 = 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 5: With frequency parameter
    asof = datetime.date(2020, 3, 15)
    freq = Decimal(2)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end, freq)
    # Should still calculate the daily difference
    expected = simple_fraction_method(start, asof, end, freq) - simple_fraction_method(start, datetime.date(2020, 3, 14), end, freq)
    assert daily_frac == expected
    
    # Test case 6: Verify calculation is difference between consecutive days
    asof = datetime.date(2020, 5, 20)
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    yesterday = datetime.date(2020, 5, 19)
    today_frac = simple_fraction_method(start, asof, end, None)
    yesterday_frac = simple_fraction_method(start, yesterday, end, None)
    assert daily_frac == today_frac - yesterday_frac


# LLM-generated content at query #16
#--------------------------

```python
def test_DCCRegistryMachinery_find():
    """Test the find method of DCCRegistryMachinery class."""
    
    # Create a registry instance
    registry = DCCRegistryMachinery()
    
    # Create a sample DCC for testing
    test_dcc = DCC(
        name="Test/Convention",
        altnames={"TC", "Test Conv"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: ZERO
    )
    
    # Register the test DCC
    registry.register(test_dcc)
    
    # Test finding by exact main name
    result = registry.find("Test/Convention")
    assert result is not None
    assert result.name == "Test/Convention"
    
    # Test finding by alternative name
    result = registry.find("TC")
    assert result is not None
    assert result.name == "Test/Convention"
    
    result = registry.find("Test Conv")
    assert result is not None
    assert result.name == "Test/Convention"
    
    # Test finding with whitespace (should strip)
    result = registry.find("  Test/Convention  ")
    assert result is not None
    assert result.name == "Test/Convention"
    
    # Test finding with lowercase (should uppercase)
    result = registry.find("test/convention")
    assert result is not None
    assert result.name == "Test/Convention"
    
    # Test finding with whitespace and lowercase (should strip and uppercase)
    result = registry.find("  test/convention  ")
    assert result is not None
    assert result.name == "Test/Convention"
    
    # Test finding non-existent DCC
    result = registry.find("NonExistent/Convention")
    assert result is None
    
    # Test finding with empty string
    result = registry.find("")
    assert result is None
    
    # Test _find_strict method directly
    result = registry._find_strict("Test/Convention")
    assert result is not None
    assert result.name == "Test/Convention"
    
    # Test _find_strict with alternative name
    result = registry._find_strict("TC")
    assert result is not None
    
    # Test _find_strict with non-existent name (no stripping/uppercasing)
    result = registry._find_strict("test/convention")
    assert result is None


# LLM-generated content at query #17
#--------------------------

```python
def test_DCC_coupon():
    """Test the coupon method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple day count fraction calculation method
    def simple_dcfc(start, asof, end, freq):
        """Simple day count fraction: actual/360."""
        days = (asof - start).days
        return Decimal(days) / Decimal(360)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test DCC",
        altnames={"TDCC"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test basic coupon calculation
    principal = Money(Decimal(1000), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 4, 1)
    freq = 4
    eom = None
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert isinstance(result, Money)
    assert result.currency == principal.currency
    
    # Test coupon with end-of-month handling
    start_eom = datetime.date(2020, 1, 31)
    asof_eom = datetime.date(2020, 2, 29)
    result_eom = dcc.coupon(principal, rate, start_eom, asof_eom, end, 1, 31)
    assert isinstance(result_eom, Money)
    
    # Test that coupon amount is positive for positive rate
    assert result.amount > ZERO
    
    # Test with zero rate
    result_zero_rate = dcc.coupon(principal, ZERO, start, asof, end, freq, eom)
    assert result_zero_rate.amount == ZERO
    
    # Test with different frequencies
    result_freq2 = dcc.coupon(principal, rate, start, asof, end, 2, eom)
    assert isinstance(result_freq2, Money)
    
    # Test with different principal amounts
    principal_large = Money(Decimal(10000), "USD")
    result_large = dcc.coupon(principal_large, rate, start, asof, end, freq, eom)
    assert result_large.amount > result.amount
    
    # Test that result scales linearly with principal
    assert result_large.amount == result.amount * Decimal(10)


# LLM-generated content at query #18
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    """
    Unit tests for DCC.calculate_daily_fraction method.
    """
    import datetime
    from decimal import Decimal
    
    # Create a simple test DCC with a mock calculation method
    def mock_calculate_fraction(start: datetime.date, asof: datetime.date, end: datetime.date, freq) -> Decimal:
        """Mock method that returns fraction based on days elapsed."""
        days_elapsed = (asof - start).days
        total_days = (end - start).days
        if total_days == 0:
            return Decimal(0)
        return Decimal(days_elapsed) / Decimal(total_days)
    
    dcc = DCC(
        name="TEST",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    start = datetime.date(2017, 1, 1)
    end = datetime.date(2017, 1, 10)  # 9 days total
    
    # Test case 1: Daily fraction on second day
    asof = datetime.date(2017, 1, 2)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # Day 2: (1/9) - (0/9) = 1/9
    assert result == Decimal(1) / Decimal(9)
    
    # Test case 2: Daily fraction on third day
    asof = datetime.date(2017, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # Day 3: (2/9) - (1/9) = 1/9
    assert result == Decimal(1) / Decimal(9)
    
    # Test case 3: Daily fraction on start date
    asof = datetime.date(2017, 1, 1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # Day 1: (0/9) - 0 = 0/9 (since asof_minus_1 < start, yfact = 0)
    assert result == ZERO
    
    # Test case 4: Daily fraction on last day
    asof = datetime.date(2017, 1, 10)
    result = dcc.calculate_daily_fraction(start, asof, end)
    # Day 10: (9/9) - (8/9) = 1/9
    assert result == Decimal(1) / Decimal(9)
    
    # Test case 5: With frequency parameter
    asof = datetime.date(2017, 1, 5)
    freq = Decimal(2)
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    # Day 5: (4/9) - (3/9) = 1/9
    assert result == Decimal(1) / Decimal(9)
    
    # Test case 6: Single day period
    start_single = datetime.date(2017, 1, 1)
    end_single = datetime.date(2017, 1, 2)
    asof_single = datetime.date(2017, 1, 2)
    result = dcc.calculate_daily_fraction(start_single, asof_single, end_single)
    # (1/1) - (0/1) = 1
    assert result == Decimal(1)


# LLM-generated content at query #19
#--------------------------

def test_dcfc_nl_365():
    import datetime
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (includes leap day)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: Same start and asof date
    test_start = datetime.date(2008, 1, 1)
    test_asof = datetime.date(2008, 1, 1)
    result5 = dcfc_nl_365(start=test_start, asof=test_asof, end=test_asof)
    assert result5 == Decimal(0)
    
    # Test case 6: One day difference
    test_start = datetime.date(2008, 1, 1)
    test_asof = datetime.date(2008, 1, 2)
    result6 = dcfc_nl_365(start=test_start, asof=test_asof, end=test_asof)
    assert result6 == Decimal(1) / Decimal(365)


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_act_act_icma():
    """Unit tests for dcfc_act_act_icma function."""
    
    # Test case from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result, 10) == Decimal('0.5245901639')
    
    # Test with same start and asof dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal(0)
    
    # Test with same asof and end dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result == Decimal(1)
    
    # Test with frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    assert result > Decimal(0)
    assert result < Decimal(1)
    
    # Test with leap year
    start = datetime.date(2020, 2, 1)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2021, 2, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result > Decimal(0)
    assert result < Decimal(1)
    
    # Test with different periods
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 6, 15)
    end = datetime.date(2019, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result > Decimal(0)
    assert result < Decimal(1)
    
    # Test return type is Decimal
    start = datetime.date(2020, 3, 1)
    asof = datetime.date(2020, 9, 1)
    end = datetime.date(2021, 3, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert isinstance(result, Decimal)
    
    # Test with single year period
    start = datetime.date(2019, 6, 1)
    asof = datetime.date(2019, 9, 1)
    end = datetime.date(2019, 12, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result > Decimal(0)
    assert result < Decimal(1)
    
    # Test with frequency = 1
    start = datetime.date(2018, 5, 15)
    asof = datetime.date(2018, 11, 15)
    end = datetime.date(2019, 5, 15)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(1))
    assert result > Decimal(0)
    assert result < Decimal(1)
    
    # Test with frequency = 4 (quarterly)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 31)
    end = datetime.date(2020, 12, 31)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal(4))
    assert result > Decimal(0)


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_act_act_icma():
    """Unit tests for dcfc_act_act_icma function."""
    
    # Test case 1: Basic example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result1 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result1, 10) == Decimal('0.5245901639')
    
    # Test case 2: With explicit frequency parameter
    result2 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=Decimal(1))
    assert round(result2, 10) == Decimal('0.5245901639')
    
    # Test case 3: With different frequency
    result3 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=Decimal(2))
    assert round(result3, 10) == Decimal('0.2622950820')
    
    # Test case 4: Same start and asof dates
    result4 = dcfc_act_act_icma(start=ex1_start, asof=ex1_start, end=ex1_end)
    assert result4 == Decimal(0)
    
    # Test case 5: Same asof and end dates
    result5 = dcfc_act_act_icma(start=ex1_start, asof=ex1_end, end=ex1_end)
    assert result5 == Decimal(1)
    
    # Test case 6: Different date range
    start6 = datetime.date(2020, 1, 1)
    asof6 = datetime.date(2020, 6, 30)
    end6 = datetime.date(2020, 12, 31)
    result6 = dcfc_act_act_icma(start=start6, asof=asof6, end=end6)
    assert isinstance(result6, Decimal)
    assert result6 > Decimal(0)
    assert result6 < Decimal(1)
    
    # Test case 7: With frequency of 4 (quarterly payments)
    result7 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=Decimal(4))
    assert round(result7, 10) == Decimal('0.1311475410')
    
    # Test case 8: Leap year period
    start8 = datetime.date(2020, 2, 1)
    asof8 = datetime.date(2020, 2, 29)
    end8 = datetime.date(2020, 3, 1)
    result8 = dcfc_act_act_icma(start=start8, asof=asof8, end=end8)
    assert isinstance(result8, Decimal)
    assert result8 > Decimal(0)
    
    # Test case 9: Verify result type is Decimal
    result9 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert isinstance(result9, Decimal)
    
    # Test case 10: With None frequency (should default to ONE)
    result10 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end, freq=None)
    assert round(result10, 10) == Decimal('0.5245901639')


# LLM-generated content at query #22
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    """Test the calculate_daily_fraction method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple test day count fraction calculation method
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple DCFC that returns days between start and asof divided by 365."""
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    # Create a DCC instance for testing
    test_dcc = DCC(
        name="Test DCC",
        altnames={"TEST", "T"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test case 1: Single day difference
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 12, 31)
    
    daily_frac = test_dcc.calculate_daily_fraction(start, asof, end)
    # Expected: (1/365) - (0/365) = 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 2: Multiple days difference
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 5)
    end = datetime.date(2020, 12, 31)
    
    daily_frac = test_dcc.calculate_daily_fraction(start, asof, end)
    # Expected: (4/365) - (3/365) = 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 3: asof_minus_1 is before start (edge case)
    start = datetime.date(2020, 1, 2)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 12, 31)
    
    daily_frac = test_dcc.calculate_daily_fraction(start, asof, end)
    # Expected: (0/365) - (0/365) = 0
    assert daily_frac == Decimal(0)
    
    # Test case 4: Verify calculation is consistent with fraction method
    start = datetime.date(2020, 6, 15)
    asof = datetime.date(2020, 6, 20)
    end = datetime.date(2020, 12, 31)
    
    daily_frac = test_dcc.calculate_daily_fraction(start, asof, end)
    frac_asof = simple_dcfc(start, asof, end, None)
    frac_asof_minus_1 = simple_dcfc(start, asof - datetime.timedelta(days=1), end, None)
    expected = frac_asof - frac_asof_minus_1
    
    assert daily_frac == expected
    
    # Test case 5: Large date range
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2021, 12, 31)
    
    daily_frac = test_dcc.calculate_daily_fraction(start, asof, end)
    # 2020 is a leap year, so 365 days from Jan 1 to Dec 31
    # Expected: (365/365) - (364/365) = 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 6: With frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 15)
    end = datetime.date(2020, 12, 31)
    freq = Decimal(2)
    
    daily_frac = test_dcc.calculate_daily_fraction(start, asof, end, freq)
    frac_asof = simple_dcfc(start, asof, end, freq)
    frac_asof_minus_1 = simple_dcfc(start, asof - datetime.timedelta(days=1), end, freq)
    expected = frac_asof - frac_asof_minus_1
    
    assert daily_frac == expected


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_act_365_a():
    """Test the dcfc_act_365_a day count fraction calculator."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = round(dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14)
    assert result1 == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = round(dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14)
    assert result2 == Decimal('0.17213114754098')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = round(dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14)
    assert result3 == Decimal('1.08196721311475')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = round(dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14)
    assert result4 == Decimal('1.32513661202186')
    
    # Test case 5: Same start and asof date (zero days)
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_act_365_a(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Non-leap year period
    non_leap_start = datetime.date(2007, 1, 1)
    non_leap_asof = datetime.date(2007, 12, 31)
    result6 = dcfc_act_365_a(start=non_leap_start, asof=non_leap_asof, end=non_leap_asof)
    assert result6 == Decimal('364') / Decimal('365')
    
    # Test case 7: With freq parameter (should be ignored)
    freq_start = datetime.date(2008, 2, 1)
    freq_asof = datetime.date(2008, 5, 31)
    result7_with_freq = dcfc_act_365_a(start=freq_start, asof=freq_asof, end=freq_asof, freq=Decimal(2))
    result7_without_freq = dcfc_act_365_a(start=freq_start, asof=freq_asof, end=freq_asof)
    assert result7_with_freq == result7_without_freq


# LLM-generated content at query #24
#--------------------------

```python
def test_DCC_calculate_fraction():
    """Test DCC.calculate_fraction method with various scenarios."""
    import datetime
    from decimal import Decimal
    
    # Create a simple test DCC with a basic fraction calculation method
    def simple_fraction_method(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Decimal = None) -> Decimal:
        """Simple method: fraction of days between start and asof over total days."""
        total_days = (end - start).days
        if total_days == 0:
            return Decimal(0)
        days_elapsed = (asof - start).days
        return Decimal(days_elapsed) / Decimal(total_days)
    
    test_dcc = DCC(
        name="Test DCC",
        altnames={"TDCC", "TestDCC"},
        currencies=set(),
        calculate_fraction_method=simple_fraction_method
    )
    
    # Test 1: Valid date range - asof equals start
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 1, 10)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == Decimal(0)
    
    # Test 2: Valid date range - asof equals end
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 10)
    end = datetime.date(2020, 1, 10)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == Decimal(1)
    
    # Test 3: Valid date range - asof in the middle
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 5)
    end = datetime.date(2020, 1, 10)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == Decimal(4) / Decimal(9)
    
    # Test 4: asof before start - should return ZERO
    start = datetime.date(2020, 1, 5)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 1, 10)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == ZERO
    
    # Test 5: asof after end - should return ZERO
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 10)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == ZERO
    
    # Test 6: With frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 5)
    end = datetime.date(2020, 1, 10)
    freq = Decimal(2)
    result = test_dcc.calculate_fraction(start, asof, end, freq)
    assert result == Decimal(4) / Decimal(9)
    
    # Test 7: Same start and end dates
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 1, 1)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == ZERO
    
    # Test 8: Leap year date range
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 1)
    result = test_dcc.calculate_fraction(start, asof, end)
    assert result == Decimal(1) / Decimal(2)


# LLM-generated content at query #25
#--------------------------

```python
def test_dcfc_nl_365():
    """Unit tests for dcfc_nl_365 function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (includes leap day)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: Same day (zero day count)
    same_day_start = datetime.date(2008, 6, 15)
    result5 = dcfc_nl_365(start=same_day_start, asof=same_day_start, end=same_day_start)
    assert result5 == Decimal(0)
    
    # Test case 6: One day difference
    one_day_start = datetime.date(2008, 6, 15)
    one_day_asof = datetime.date(2008, 6, 16)
    result6 = dcfc_nl_365(start=one_day_start, asof=one_day_asof, end=one_day_asof)
    assert result6 == Decimal(1) / Decimal(365)
    
    # Test case 7: Period without leap day
    no_leap_start = datetime.date(2007, 1, 1)
    no_leap_asof = datetime.date(2007, 12, 31)
    result7 = dcfc_nl_365(start=no_leap_start, asof=no_leap_asof, end=no_leap_asof)
    assert result7 == Decimal(364) / Decimal(365)
    
    # Test case 8: With optional freq parameter
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result8 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result8, 14) == Decimal('0.16986301369863')


# LLM-generated content at query #26
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    """Test the calculate_daily_fraction method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple test day count fraction calculation method
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple DCFC that returns days elapsed / 365."""
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test DCC",
        altnames={"TEST"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test case 1: Basic daily fraction calculation
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 3)
    end = datetime.date(2020, 12, 31)
    
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    # Day 2: 2/365, Day 3: 3/365, Difference: 1/365
    expected = Decimal(1) / Decimal(365)
    assert daily_frac == expected
    
    # Test case 2: Daily fraction at the start date
    asof_start = datetime.date(2020, 1, 1)
    daily_frac_start = dcc.calculate_daily_fraction(start, asof_start, end)
    # Day 1: 1/365, Day 0 (before start): 0, Difference: 1/365
    expected_start = Decimal(1) / Decimal(365)
    assert daily_frac_start == expected_start
    
    # Test case 3: Daily fraction over multiple days
    start2 = datetime.date(2020, 1, 1)
    asof2 = datetime.date(2020, 1, 10)
    end2 = datetime.date(2020, 12, 31)
    
    daily_frac2 = dcc.calculate_daily_fraction(start2, asof2, end2)
    # Day 10: 10/365, Day 9: 9/365, Difference: 1/365
    expected2 = Decimal(1) / Decimal(365)
    assert daily_frac2 == expected2
    
    # Test case 4: Leap year consideration
    start_leap = datetime.date(2020, 2, 28)
    asof_leap = datetime.date(2020, 2, 29)
    end_leap = datetime.date(2020, 12, 31)
    
    daily_frac_leap = dcc.calculate_daily_fraction(start_leap, asof_leap, end_leap)
    # Should calculate the difference between consecutive days
    assert daily_frac_leap == Decimal(1) / Decimal(365)
    
    # Test case 5: Verify consistency with calculate_fraction
    start5 = datetime.date(2020, 6, 1)
    asof5 = datetime.date(2020, 6, 15)
    end5 = datetime.date(2020, 12, 31)
    
    daily_frac5 = dcc.calculate_daily_fraction(start5, asof5, end5)
    frac_asof = dcc.calculate_fraction_method(start5, asof5, end5, None)
    frac_asof_minus_1 = dcc.calculate_fraction_method(start5, asof5 - datetime.timedelta(days=1), end5, None)
    expected5 = frac_asof - frac_asof_minus_1
    assert daily_frac5 == expected5


# LLM-generated content at query #27
#--------------------------

```python
def test_dcfc_act_365_a():
    """Unit tests for dcfc_act_365_a function."""
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = round(dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14)
    assert result1 == Decimal('0.16986301369863')
    
    # Test case 2: Example from docstring with leap day
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = round(dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14)
    assert result2 == Decimal('0.17213114754098')
    
    # Test case 3: Example from docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = round(dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14)
    assert result3 == Decimal('1.08196721311475')
    
    # Test case 4: Example from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = round(dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14)
    assert result4 == Decimal('1.32513661202186')
    
    # Test case 5: Same day (zero day count)
    same_day = datetime.date(2020, 6, 15)
    result5 = dcfc_act_365_a(start=same_day, asof=same_day, end=same_day)
    assert result5 == Decimal('0')
    
    # Test case 6: One day difference in non-leap year
    start_date = datetime.date(2019, 6, 15)
    asof_date = datetime.date(2019, 6, 16)
    result6 = dcfc_act_365_a(start=start_date, asof=asof_date, end=asof_date)
    assert result6 == Decimal('1') / Decimal('365')
    
    # Test case 7: One day difference in leap year (with leap day in range)
    start_date_leap = datetime.date(2020, 2, 28)
    asof_date_leap = datetime.date(2020, 2, 29)
    result7 = dcfc_act_365_a(start=start_date_leap, asof=asof_date_leap, end=asof_date_leap)
    assert result7 == Decimal('1') / Decimal('366')
    
    # Test case 8: freq parameter should be ignored
    result8a = dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=None)
    result8b = dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal('2'))
    assert result8a == result8b
    
    # Test case 9: Full year in non-leap year
    start_full = datetime.date(2019, 1, 1)
    asof_full = datetime.date(2019, 12, 31)
    result9 = dcfc_act_365_a(start=start_full, asof=asof_full, end=asof_full)
    assert result9 == Decimal('364') / Decimal('365')
    
    # Test case 10: Full leap year
    start_leap_full = datetime.date(2020, 1, 1)
    asof_leap_full = datetime.date(2020, 12, 31)
    result10 = dcfc_act_365_a(start=start_leap_full, asof=asof_leap_full, end=asof_leap_full)
    assert result10 == Decimal('365') / Decimal('366')


# LLM-generated content at query #28
#--------------------------

```python
def test_DCC_interest():
    """Test the interest method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Create a simple day count fraction method for testing
    def simple_fraction(start, asof, end, freq):
        """Simple fraction: days between start and asof divided by 365."""
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test DCC",
        altnames={"TDCC", "Test"},
        currencies=set(),
        calculate_fraction_method=simple_fraction
    )
    
    # Test case 1: Basic interest calculation
    principal = Money(Decimal(1000), "USD")
    rate = Decimal("0.05")  # 5% annual rate
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 12, 31)
    
    result = dcc.interest(principal, rate, start, asof, end)
    expected = principal * rate * simple_fraction(start, asof, end, None)
    assert result == expected
    
    # Test case 2: Interest with same start and asof dates
    asof_same = datetime.date(2023, 1, 1)
    result_zero = dcc.interest(principal, rate, start, asof_same, end)
    assert result_zero == Money(Decimal(0), "USD")
    
    # Test case 3: Interest with end date not provided (defaults to asof)
    result_default_end = dcc.interest(principal, rate, start, asof)
    expected_default = principal * rate * simple_fraction(start, asof, asof, None)
    assert result_default_end == expected_default
    
    # Test case 4: Different principal amounts
    principal_large = Money(Decimal(100000), "USD")
    result_large = dcc.interest(principal_large, rate, start, asof, end)
    expected_large = principal_large * rate * simple_fraction(start, asof, end, None)
    assert result_large == expected_large
    
    # Test case 5: Different interest rates
    rate_high = Decimal("0.10")  # 10% annual rate
    result_high_rate = dcc.interest(principal, rate_high, start, asof, end)
    expected_high_rate = principal * rate_high * simple_fraction(start, asof, end, None)
    assert result_high_rate == expected_high_rate
    
    # Test case 6: With frequency parameter
    freq = Decimal(2)
    result_with_freq = dcc.interest(principal, rate, start, asof, end, freq)
    expected_with_freq = principal * rate * simple_fraction(start, asof, end, freq)
    assert result_with_freq == expected_with_freq


# LLM-generated content at query #29
#--------------------------

```python
def test_DCC_calculate_fraction():
    """Test DCC.calculate_fraction method."""
    import datetime
    from decimal import Decimal
    
    # Create a simple mock DCC instance
    def mock_calculate_fraction_method(start, asof, end, freq):
        """Mock method that returns a simple fraction."""
        return Decimal(str((asof - start).days)) / Decimal(str((end - start).days)) if (end - start).days > 0 else ZERO
    
    dcc = DCC(
        name="Test DCC",
        altnames={"Test", "DCC_Test"},
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    
    # Test case 1: Valid date range (start <= asof <= end)
    asof = datetime.date(2020, 6, 30)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("181") / Decimal("365")
    
    # Test case 2: asof equals start
    asof = datetime.date(2020, 1, 1)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO
    
    # Test case 3: asof equals end
    asof = datetime.date(2020, 12, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("365") / Decimal("365")
    
    # Test case 4: asof before start (invalid range) - should return ZERO
    asof = datetime.date(2019, 12, 31)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO
    
    # Test case 5: asof after end (invalid range) - should return ZERO
    asof = datetime.date(2021, 1, 1)
    result = dcc.calculate_fraction(start, asof, end)
    assert result == ZERO
    
    # Test case 6: With frequency parameter
    asof = datetime.date(2020, 3, 31)
    freq = Decimal("4")
    result = dcc.calculate_fraction(start, asof, end, freq)
    assert result == Decimal("90") / Decimal("365")
    
    # Test case 7: All dates are the same
    same_date = datetime.date(2020, 6, 15)
    result = dcc.calculate_fraction(same_date, same_date, same_date)
    assert result == ZERO


# LLM-generated content at query #30
#--------------------------

```python
def test_dcfc_nl_365():
    """Unit tests for dcfc_nl_365 function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day included)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: Same start and asof date (zero day count)
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 1)
    result5 = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result5 == Decimal(0)
    
    # Test case 6: One day difference
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 2)
    result6 = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result6 == Decimal(1) / Decimal(365)
    
    # Test case 7: Leap day is excluded from count
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2008, 3, 1)
    result7 = dcfc_nl_365(start=start, asof=asof, end=asof)
    # 3 days total (28, 29, 1) but 1 leap day excluded = 2 days
    assert result7 == Decimal(2) / Decimal(365)
    
    # Test case 8: Non-leap year period
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 31)
    result8 = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result8 == Decimal(30) / Decimal(365)
    
    # Test case 9: With freq parameter (should be ignored)
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 2)
    result9 = dcfc_nl_365(start=start, asof=asof, end=asof, freq=Decimal(2))
    assert result9 == Decimal(1) / Decimal(365)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_act_act():
    """Unit tests for dcfc_act_act function."""
    
    # Test case 1: Example 1 from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16942884946478')
    
    # Test case 2: Example 2 from docstring (leap year with Feb 29)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17216108990194')
    
    # Test case 3: Example 3 from docstring (longer period)
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08243131970956')
    
    # Test case 4: Example 4 from docstring (multi-year period)
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32625945055768')
    
    # Test case 5: Same start and asof date (zero days)
    same_date = datetime.date(2010, 6, 15)
    result5 = dcfc_act_act(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Single day period
    start_date = datetime.date(2010, 6, 15)
    asof_date = datetime.date(2010, 6, 16)
    result6 = dcfc_act_act(start=start_date, asof=asof_date, end=asof_date)
    assert result6 == Decimal('1') / Decimal('365')
    
    # Test case 7: Non-leap year period
    start_non_leap = datetime.date(2010, 1, 1)
    asof_non_leap = datetime.date(2010, 12, 31)
    result7 = dcfc_act_act(start=start_non_leap, asof=asof_non_leap, end=asof_non_leap)
    assert result7 == Decimal('364') / Decimal('365')
    
    # Test case 8: Period with leap year only
    start_leap = datetime.date(2008, 1, 1)
    asof_leap = datetime.date(2008, 12, 31)
    result8 = dcfc_act_act(start=start_leap, asof=asof_leap, end=asof_leap)
    assert result8 == Decimal('365') / Decimal('366')
    
    # Test case 9: Period crossing leap day
    start_cross = datetime.date(2008, 2, 28)
    asof_cross = datetime.date(2008, 3, 1)
    result9 = dcfc_act_act(start=start_cross, asof=asof_cross, end=asof_cross)
    assert result9 == Decimal('2') / Decimal('366')
    
    # Test case 10: With frequency parameter (should not affect calculation)
    result10 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result10, 14) == Decimal('0.16942884946478')


# LLM-generated content at query #2
#--------------------------

def test_dcfc_act_act():
    """Unit tests for dcfc_act_act function."""
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16942884946478')
    
    # Test case 2: Leap year date
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17216108990194')
    
    # Test case 3: Longer period across year boundary
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08243131970956')
    
    # Test case 4: Even longer period
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32625945055768')
    
    # Test case 5: Same start and asof date (single day)
    same_date = datetime.date(2020, 6, 15)
    result5 = dcfc_act_act(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Two consecutive days
    start_date = datetime.date(2020, 6, 15)
    next_date = datetime.date(2020, 6, 16)
    result6 = dcfc_act_act(start=start_date, asof=next_date, end=next_date)
    assert result6 == Decimal('1') / Decimal('366')  # 2020 is a leap year
    
    # Test case 7: Period in non-leap year
    non_leap_start = datetime.date(2019, 6, 15)
    non_leap_next = datetime.date(2019, 6, 16)
    result7 = dcfc_act_act(start=non_leap_start, asof=non_leap_next, end=non_leap_next)
    assert result7 == Decimal('1') / Decimal('365')  # 2019 is not a leap year
    
    # Test case 8: Full year in leap year
    leap_start = datetime.date(2020, 1, 1)
    leap_end = datetime.date(2020, 12, 31)
    result8 = dcfc_act_act(start=leap_start, asof=leap_end, end=leap_end)
    assert result8 == Decimal('366') / Decimal('366')
    
    # Test case 9: Full year in non-leap year
    non_leap_start = datetime.date(2019, 1, 1)
    non_leap_end = datetime.date(2019, 12, 31)
    result9 = dcfc_act_act(start=non_leap_start, asof=non_leap_end, end=non_leap_end)
    assert result9 == Decimal('365') / Decimal('365')
    
    # Test case 10: Frequency parameter (should not affect calculation for Act/Act)
    result10 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result10, 14) == Decimal('0.16942884946478')


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_e_360():
    import datetime
    from decimal import Decimal
    
    # Test case 1: From docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: From docstring
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: From docstring
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: From docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test case 5: Same day
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 1, 15)
    result5 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result5 == Decimal('0')
    
    # Test case 6: Day 31 adjustment on start date
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    result6 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result6 == Decimal('29') / Decimal('360')
    
    # Test case 7: Day 31 adjustment on asof date
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 2, 31)  # Invalid, but testing with 2020-02-29
    asof = datetime.date(2020, 2, 29)
    result7 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected7 = (29 - 15) + 30 * (2 - 1) + 360 * (2020 - 2020)
    assert result7 == Decimal(expected7) / Decimal('360')
    
    # Test case 8: Year change
    start = datetime.date(2019, 12, 15)
    asof = datetime.date(2020, 1, 15)
    result8 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected8 = (15 - 15) + 30 * (1 - 12) + 360 * (2020 - 2019)
    assert result8 == Decimal(expected8) / Decimal('360')
    
    # Test case 9: Both day 31 adjustments
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 3, 31)
    result9 = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected9 = (30 - 30) + 30 * (3 - 1) + 360 * (2020 - 2020)
    assert result9 == Decimal(expected9) / Decimal('360')
    
    # Test case 10: With freq parameter (should be ignored)
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 2, 15)
    result10_no_freq = dcfc_30_e_360(start=start, asof=asof, end=asof)
    result10_with_freq = dcfc_30_e_360(start=start, asof=asof, end=asof, freq=Decimal('2'))
    assert result10_no_freq == result10_with_freq


# LLM-generated content at query #4
#--------------------------

```python
def test_DCC_interest():
    """Test the interest method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple day count fraction method for testing
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple day count fraction: actual days / 365."""
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    # Create a test DCC instance
    test_dcc = DCC(
        name="Test DCC",
        altnames={"T-DCC", "TDCC"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test case 1: Basic interest calculation
    principal = Money(Decimal(1000), Currencies["USD"])
    rate = Decimal("0.05")  # 5% annual rate
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 2, 1)
    end = datetime.date(2023, 12, 31)
    
    result = test_dcc.interest(principal, rate, start, asof, end)
    
    # Expected: 1000 * 0.05 * (31/365) ≈ 4.25
    expected_amount = Decimal(1000) * Decimal("0.05") * (Decimal(31) / Decimal(365))
    assert result.amount == expected_amount
    assert result.currency == Currencies["USD"]
    
    # Test case 2: Interest with end date = asof (default behavior)
    result2 = test_dcc.interest(principal, rate, start, asof)
    
    # Should be same as test case 1 when end is not specified
    assert result2.amount == expected_amount
    assert result2.currency == Currencies["USD"]
    
    # Test case 3: Zero days elapsed (start = asof)
    result3 = test_dcc.interest(principal, rate, start, start, end)
    assert result3.amount == Decimal(0)
    assert result3.currency == Currencies["USD"]
    
    # Test case 4: Different principal and rate
    principal2 = Money(Decimal(5000), Currencies["EUR"])
    rate2 = Decimal("0.10")  # 10% annual rate
    start2 = datetime.date(2023, 1, 1)
    asof2 = datetime.date(2023, 4, 1)  # 90 days
    end2 = datetime.date(2023, 12, 31)
    
    result4 = test_dcc.interest(principal2, rate2, start2, asof2, end2)
    
    # Expected: 5000 * 0.10 * (90/365)
    expected_amount2 = Decimal(5000) * Decimal("0.10") * (Decimal(90) / Decimal(365))
    assert result4.amount == expected_amount2
    assert result4.currency == Currencies["EUR"]
    
    # Test case 5: Larger time period
    principal3 = Money(Decimal(10000), Currencies["GBP"])
    rate3 = Decimal("0.03")  # 3% annual rate
    start3 = datetime.date(2023, 1, 1)
    asof3 = datetime.date(2023, 7, 1)  # 181 days
    end3 = datetime.date(2023, 12, 31)
    
    result5 = test_dcc.interest(principal3, rate3, start3, asof3, end3)
    
    # Expected: 10000 * 0.03 * (181/365)
    expected_amount3 = Decimal(10000) * Decimal("0.03") * (Decimal(181) / Decimal(365))
    assert result5.amount == expected_amount3
    assert result5.currency == Currencies["GBP"]


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_nl_365():
    """Unit tests for dcfc_nl_365 function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30 (spans leap year)
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: Same start and asof dates (zero days)
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_nl_365(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Non-leap year period (no leap day adjustment needed)
    start_non_leap = datetime.date(2007, 3, 15)
    asof_non_leap = datetime.date(2007, 6, 15)
    result6 = dcfc_nl_365(start=start_non_leap, asof=asof_non_leap, end=asof_non_leap)
    # 92 days / 365
    assert result6 == Decimal(92) / Decimal(365)
    
    # Test case 7: With freq parameter (should be ignored)
    ex7_start = datetime.date(2008, 1, 1)
    ex7_asof = datetime.date(2008, 1, 31)
    result7a = dcfc_nl_365(start=ex7_start, asof=ex7_asof, end=ex7_asof, freq=None)
    result7b = dcfc_nl_365(start=ex7_start, asof=ex7_asof, end=ex7_asof, freq=Decimal(2))
    assert result7a == result7b


# LLM-generated content at query #6
#--------------------------

def test_dcfc_nl_365():
    import datetime
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day included)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: Same start and asof date (zero days)
    ex5_start = datetime.date(2008, 1, 1)
    ex5_asof = datetime.date(2008, 1, 1)
    result5 = dcfc_nl_365(start=ex5_start, asof=ex5_asof, end=ex5_asof)
    assert result5 == Decimal(0)
    
    # Test case 6: Single day difference
    ex6_start = datetime.date(2008, 1, 1)
    ex6_asof = datetime.date(2008, 1, 2)
    result6 = dcfc_nl_365(start=ex6_start, asof=ex6_asof, end=ex6_asof)
    assert result6 == Decimal(1) / Decimal(365)


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_30_360_us():
    # Test example 1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test example 2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test example 3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test example 4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test same day
    same_day = datetime.date(2008, 6, 15)
    result_same = dcfc_30_360_us(start=same_day, asof=same_day, end=same_day)
    assert result_same == Decimal('0')
    
    # Test with day 31 in start
    start_31 = datetime.date(2008, 1, 31)
    asof_30 = datetime.date(2008, 2, 29)
    result_31 = dcfc_30_360_us(start=start_31, asof=asof_30, end=asof_30)
    assert isinstance(result_31, Decimal)
    assert result_31 > 0
    
    # Test with end of month dates
    start_eom = datetime.date(2008, 1, 31)
    asof_eom = datetime.date(2008, 2, 29)
    result_eom = dcfc_30_360_us(start=start_eom, asof=asof_eom, end=asof_eom)
    assert isinstance(result_eom, Decimal)
    
    # Test with one year difference
    start_year = datetime.date(2007, 6, 15)
    asof_year = datetime.date(2008, 6, 15)
    result_year = dcfc_30_360_us(start=start_year, asof=asof_year, end=asof_year)
    assert round(result_year, 2) == Decimal('1.00')


# LLM-generated content at query #8
#--------------------------

```python
def test_DCC_calculate_daily_fraction():
    """Test the calculate_daily_fraction method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple test day count fraction calculation method
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple DCFC that counts days and divides by 365."""
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test DCC",
        altnames={"TDCC"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Test case 1: Daily fraction for a single day
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 12, 31)
    
    daily_frac = dcc.calculate_daily_fraction(start, asof, end)
    # Day 1 (2020-01-02): (1/365) - (0/365) = 1/365
    assert daily_frac == Decimal(1) / Decimal(365)
    
    # Test case 2: Daily fraction for another day
    asof2 = datetime.date(2020, 1, 5)
    daily_frac2 = dcc.calculate_daily_fraction(start, asof2, end)
    # Day 5 (2020-01-05): (4/365) - (3/365) = 1/365
    assert daily_frac2 == Decimal(1) / Decimal(365)
    
    # Test case 3: When asof_minus_1 equals start
    asof3 = datetime.date(2020, 1, 1)
    daily_frac3 = dcc.calculate_daily_fraction(start, asof3, end)
    # (0/365) - 0 = 0
    assert daily_frac3 == ZERO
    
    # Test case 4: Multiple days later
    asof4 = datetime.date(2020, 1, 11)
    daily_frac4 = dcc.calculate_daily_fraction(start, asof4, end)
    # Day 11 (2020-01-11): (10/365) - (9/365) = 1/365
    assert daily_frac4 == Decimal(1) / Decimal(365)
    
    # Test case 5: asof_minus_1 before start (edge case)
    start2 = datetime.date(2020, 1, 5)
    asof5 = datetime.date(2020, 1, 5)
    daily_frac5 = dcc.calculate_daily_fraction(start2, asof5, end)
    # (0/365) - 0 = 0
    assert daily_frac5 == ZERO
    
    # Test case 6: With frequency parameter
    freq = Decimal(2)
    daily_frac6 = dcc.calculate_daily_fraction(start, datetime.date(2020, 1, 3), end, freq)
    # (2/365) - (1/365) = 1/365
    assert daily_frac6 == Decimal(1) / Decimal(365)


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_30_360_us():
    """Unit tests for dcfc_30_360_us function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap year)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test case 5: Same start and asof date
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_30_360_us(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Start and asof on last day of February (non-leap year)
    feb_start = datetime.date(2007, 2, 28)
    feb_asof = datetime.date(2007, 3, 31)
    result6 = dcfc_30_360_us(start=feb_start, asof=feb_asof, end=feb_asof)
    assert result6 == Decimal('1') / Decimal('12')
    
    # Test case 7: Exactly one year apart
    year_start = datetime.date(2008, 1, 15)
    year_asof = datetime.date(2009, 1, 15)
    result7 = dcfc_30_360_us(start=year_start, asof=year_asof, end=year_asof)
    assert result7 == Decimal('1')
    
    # Test case 8: Both dates are 31st day
    day31_start = datetime.date(2008, 1, 31)
    day31_asof = datetime.date(2008, 3, 31)
    result8 = dcfc_30_360_us(start=day31_start, asof=day31_asof, end=day31_asof)
    # Both 31st days should be treated as 30th
    assert result8 == Decimal('60') / Decimal('360')
    
    # Test case 9: End of month handling for February
    feb_last_start = datetime.date(2008, 2, 29)
    feb_last_asof = datetime.date(2008, 3, 31)
    result9 = dcfc_30_360_us(start=feb_last_start, asof=feb_last_asof, end=feb_last_asof)
    # Feb 29 is last day of month, Mar 31 should be treated as 30
    assert result9 == Decimal('31') / Decimal('360')
    
    # Test case 10: Start date on 31st, asof on 30th
    start_31 = datetime.date(2008, 1, 31)
    asof_30 = datetime.date(2008, 2, 30)  # This will be Feb 29 in leap year
    # Adjusting to valid date
    asof_30 = datetime.date(2008, 3, 30)
    result10 = dcfc_30_360_us(start=start_31, asof=asof_30, end=asof_30)
    assert isinstance(result10, Decimal)
    assert result10 > Decimal('0')


# LLM-generated content at query #10
#--------------------------

```python
def test_DCC_coupon():
    """Test the coupon method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Create a simple DCC instance with a basic fraction calculation method
    def simple_fraction_calc(start: Date, asof: Date, end: Date, freq: Optional[Decimal]) -> Decimal:
        """Simple fraction calculation: (asof - start) / (end - start)"""
        total_days = (end - start).days
        if total_days == 0:
            return ZERO
        actual_days = (asof - start).days
        return Decimal(actual_days) / Decimal(total_days)
    
    dcc = DCC(
        name="Test DCC",
        altnames={"TEST", "T"},
        currencies=set(),
        calculate_fraction_method=simple_fraction_calc
    )
    
    # Test basic coupon calculation
    principal = Money(Decimal("1000"), Currencies["USD"])
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 15)
    end = datetime.date(2020, 12, 31)
    freq = 2  # Semi-annual
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    
    assert isinstance(result, Money)
    assert result.currency == Currencies["USD"]
    
    # Test with eom parameter
    result_eom = dcc.coupon(principal, rate, start, asof, end, freq, eom=15)
    assert isinstance(result_eom, Money)
    
    # Test with annual frequency
    result_annual = dcc.coupon(principal, rate, start, asof, end, freq=1)
    assert isinstance(result_annual, Money)
    
    # Test with different principal amounts
    principal_large = Money(Decimal("10000"), Currencies["USD"])
    result_large = dcc.coupon(principal_large, rate, start, asof, end, freq)
    assert isinstance(result_large, Money)
    assert result_large.amount > result.amount
    
    # Test with different rates
    rate_high = Decimal("0.10")
    result_high_rate = dcc.coupon(principal, rate_high, start, asof, end, freq)
    assert isinstance(result_high_rate, Money)
    assert result_high_rate.amount > result.amount
    
    # Test with quarterly frequency (freq=4)
    result_quarterly = dcc.coupon(principal, rate, start, asof, end, freq=4)
    assert isinstance(result_quarterly, Money)
    
    # Test when asof is at start date
    result_at_start = dcc.coupon(principal, rate, start, start, end, freq)
    assert isinstance(result_at_start, Money)
    assert result_at_start.amount >= ZERO
    
    # Test when asof is at end date
    result_at_end = dcc.coupon(principal, rate, start, end, end, freq)
    assert isinstance(result_at_end, Money)
    
    # Test with Decimal frequency
    result_decimal_freq = dcc.coupon(principal, rate, start, asof, end, Decimal("2"))
    assert isinstance(result_decimal_freq, Money)


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_act_act():
    """Unit tests for dcfc_act_act function."""
    
    # Test case 1: Example from docstring (2007-12-28 to 2008-02-28)
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16942884946478')
    
    # Test case 2: Example from docstring (2007-12-28 to 2008-02-29, leap year)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17216108990194')
    
    # Test case 3: Example from docstring (2007-10-31 to 2008-11-30)
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08243131970956')
    
    # Test case 4: Example from docstring (2008-02-01 to 2009-05-31)
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32625945055768')
    
    # Test case 5: Same day (zero day count)
    same_day_start = datetime.date(2020, 1, 1)
    same_day_asof = datetime.date(2020, 1, 1)
    result5 = dcfc_act_act(start=same_day_start, asof=same_day_asof, end=same_day_start)
    assert result5 == Decimal('0')
    
    # Test case 6: One day difference
    one_day_start = datetime.date(2020, 1, 1)
    one_day_asof = datetime.date(2020, 1, 2)
    result6 = dcfc_act_act(start=one_day_start, asof=one_day_asof, end=one_day_asof)
    assert result6 == Decimal('1') / Decimal('366')  # 2020 is a leap year
    
    # Test case 7: Non-leap year
    non_leap_start = datetime.date(2019, 1, 1)
    non_leap_asof = datetime.date(2019, 1, 2)
    result7 = dcfc_act_act(start=non_leap_start, asof=non_leap_asof, end=non_leap_asof)
    assert result7 == Decimal('1') / Decimal('365')  # 2019 is not a leap year
    
    # Test case 8: Full year in non-leap year
    full_year_start = datetime.date(2019, 1, 1)
    full_year_asof = datetime.date(2020, 1, 1)
    result8 = dcfc_act_act(start=full_year_start, asof=full_year_asof, end=full_year_asof)
    assert result8 == Decimal('365') / Decimal('365')
    
    # Test case 9: With frequency parameter (should not affect calculation)
    freq_start = datetime.date(2008, 1, 1)
    freq_asof = datetime.date(2008, 1, 31)
    result9a = dcfc_act_act(start=freq_start, asof=freq_asof, end=freq_asof, freq=Decimal('2'))
    result9b = dcfc_act_act(start=freq_start, asof=freq_asof, end=freq_asof, freq=None)
    assert result9a == result9b
    
    # Test case 10: Across leap day in leap year
    leap_start = datetime.date(2020, 2, 28)
    leap_asof = datetime.date(2020, 3, 1)
    result10 = dcfc_act_act(start=leap_start, asof=leap_asof, end=leap_asof)
    assert result10 == Decimal('2') / Decimal('366')  # 2020 is a leap year


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_30_360_isda():
    """Unit tests for dcfc_30_360_isda function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: 2007-12-28 to 2008-02-29
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test case 5: Start day is 31 (should be adjusted to 30)
    start_31 = datetime.date(2008, 1, 31)
    asof_test = datetime.date(2008, 2, 29)
    result5 = dcfc_30_360_isda(start=start_31, asof=asof_test, end=asof_test)
    assert isinstance(result5, Decimal)
    assert result5 > 0
    
    # Test case 6: Start day 30 and asof day 31 (asof should be adjusted to 30)
    start_30 = datetime.date(2008, 1, 30)
    asof_31 = datetime.date(2008, 2, 31)  # This is invalid, so use valid date
    asof_valid = datetime.date(2008, 3, 31)
    result6 = dcfc_30_360_isda(start=start_30, asof=asof_valid, end=asof_valid)
    assert isinstance(result6, Decimal)
    assert result6 > 0
    
    # Test case 7: Same start and asof date
    same_date = datetime.date(2008, 6, 15)
    result7 = dcfc_30_360_isda(start=same_date, asof=same_date, end=same_date)
    assert result7 == Decimal(0)
    
    # Test case 8: With optional freq parameter (should be ignored)
    result8 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal(2))
    assert round(result8, 14) == Decimal('0.16666666666667')
    
    # Test case 9: Different years
    start_2007 = datetime.date(2007, 1, 1)
    asof_2008 = datetime.date(2008, 1, 1)
    result9 = dcfc_30_360_isda(start=start_2007, asof=asof_2008, end=asof_2008)
    assert result9 == Decimal(1)
    
    # Test case 10: Verify result type is Decimal
    result10 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert isinstance(result10, Decimal)


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_360_isda():
    """Unit tests for dcfc_30_360_isda function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: 2007-12-28 to 2008-02-29
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33333333333333')
    
    # Test case 5: Same start and asof date
    same_date = datetime.date(2020, 6, 15)
    result5 = dcfc_30_360_isda(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: One day difference
    start_date = datetime.date(2020, 6, 15)
    asof_date = datetime.date(2020, 6, 16)
    result6 = dcfc_30_360_isda(start=start_date, asof=asof_date, end=asof_date)
    assert result6 == Decimal('1') / Decimal('360')
    
    # Test case 7: Start day is 31 (should be adjusted to 30)
    start_31 = datetime.date(2020, 1, 31)
    asof_31 = datetime.date(2020, 2, 29)
    result7 = dcfc_30_360_isda(start=start_31, asof=asof_31, end=asof_31)
    expected7 = (29 - 30) + 30 * (2 - 1) + 360 * (2020 - 2020)
    assert result7 == Decimal(expected7) / Decimal(360)
    
    # Test case 8: Start day is 30 and asof day is 31 (asof should be adjusted to 30)
    start_30 = datetime.date(2020, 1, 30)
    asof_31_month = datetime.date(2020, 2, 31)  # This would be invalid, use valid date
    asof_31_month = datetime.date(2020, 3, 31)
    result8 = dcfc_30_360_isda(start=start_30, asof=asof_31_month, end=asof_31_month)
    # asof should be adjusted from 31 to 30
    expected8 = (30 - 30) + 30 * (3 - 1) + 360 * (2020 - 2020)
    assert result8 == Decimal(expected8) / Decimal(360)
    
    # Test case 9: Multiple years
    start_multi = datetime.date(2018, 6, 15)
    asof_multi = datetime.date(2021, 6, 15)
    result9 = dcfc_30_360_isda(start=start_multi, asof=asof_multi, end=asof_multi)
    expected9 = (15 - 15) + 30 * (6 - 6) + 360 * (2021 - 2018)
    assert result9 == Decimal(expected9) / Decimal(360)
    
    # Test case 10: Optional freq parameter (should be ignored)
    result10 = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal('2'))
    assert round(result10, 14) == Decimal('0.16666666666667')


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_30_360_german():
    import datetime
    
    # Test case 1
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test with same start and asof dates
    same_date = datetime.date(2020, 6, 15)
    result5 = dcfc_30_360_german(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test with end of month dates
    start_eom = datetime.date(2020, 2, 29)
    asof_eom = datetime.date(2020, 3, 31)
    result6 = dcfc_30_360_german(start=start_eom, asof=asof_eom, end=asof_eom)
    assert isinstance(result6, Decimal)
    assert result6 > 0


# LLM-generated content at query #15
#--------------------------

```python
def test_DCC_coupon():
    """Test the coupon method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple day count fraction method for testing
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple day count fraction: actual/360."""
        days = (asof - start).days
        return Decimal(days) / Decimal(360)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test DCC",
        altnames={"test", "t"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Create test data
    principal = Money(Decimal(1000), "USD")
    rate = Decimal("0.05")  # 5% annual rate
    start_date = datetime.date(2014, 1, 1)
    asof_date = datetime.date(2014, 6, 15)
    end_date = datetime.date(2015, 1, 1)
    frequency = 1  # Annual payments
    eom = None
    
    # Test basic coupon calculation
    result = dcc.coupon(principal, rate, start_date, asof_date, end_date, frequency, eom)
    assert isinstance(result, Money)
    assert result.currency == "USD"
    
    # Test with different frequency (semi-annual)
    frequency_semi = 2
    result_semi = dcc.coupon(principal, rate, start_date, asof_date, end_date, frequency_semi, eom)
    assert isinstance(result_semi, Money)
    
    # Test with end-of-month flag
    eom_flag = 15
    result_eom = dcc.coupon(principal, rate, start_date, asof_date, end_date, frequency, eom_flag)
    assert isinstance(result_eom, Money)
    
    # Test with decimal frequency
    frequency_decimal = Decimal("2")
    result_decimal_freq = dcc.coupon(principal, rate, start_date, asof_date, end_date, frequency_decimal, eom)
    assert isinstance(result_decimal_freq, Money)
    
    # Test with asof_date at start
    result_at_start = dcc.coupon(principal, rate, start_date, start_date, end_date, frequency, eom)
    assert isinstance(result_at_start, Money)
    assert result_at_start.amount == Decimal(0)
    
    # Test with asof_date at end
    result_at_end = dcc.coupon(principal, rate, start_date, end_date, end_date, frequency, eom)
    assert isinstance(result_at_end, Money)
    
    # Test coupon calculation returns positive value for valid inputs
    principal_positive = Money(Decimal(10000), "EUR")
    rate_positive = Decimal("0.03")
    result_positive = dcc.coupon(
        principal_positive, 
        rate_positive, 
        datetime.date(2020, 1, 1),
        datetime.date(2020, 6, 30),
        datetime.date(2021, 1, 1),
        2,
        None
    )
    assert result_positive.amount > 0


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_act_365_l():
    """Test the dcfc_act_365_l function with various date ranges."""
    import datetime
    
    # Test case 1: Non-leap year end date
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == round(Decimal('0.16939890710383'), 14)
    
    # Test case 2: Leap year with Feb 29
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == round(Decimal('0.17213114754098'), 14)
    
    # Test case 3: Multi-year period crossing leap year
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == round(Decimal('1.08196721311475'), 14)
    
    # Test case 4: Period spanning multiple years
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == round(Decimal('1.32876712328767'), 14)
    
    # Test case 5: Same start and asof date
    ex5_start = datetime.date(2020, 1, 1)
    ex5_asof = datetime.date(2020, 1, 1)
    result5 = dcfc_act_365_l(start=ex5_start, asof=ex5_asof, end=ex5_asof)
    assert result5 == Decimal(0)
    
    # Test case 6: Single day difference in non-leap year
    ex6_start = datetime.date(2019, 1, 1)
    ex6_asof = datetime.date(2019, 1, 2)
    result6 = dcfc_act_365_l(start=ex6_start, asof=ex6_asof, end=ex6_asof)
    assert result6 == Decimal(1) / Decimal(365)
    
    # Test case 7: Single day difference in leap year
    ex7_start = datetime.date(2020, 1, 1)
    ex7_asof = datetime.date(2020, 1, 2)
    result7 = dcfc_act_365_l(start=ex7_start, asof=ex7_asof, end=ex7_asof)
    assert result7 == Decimal(1) / Decimal(366)
    
    # Test case 8: Full non-leap year
    ex8_start = datetime.date(2019, 1, 1)
    ex8_asof = datetime.date(2019, 12, 31)
    result8 = dcfc_act_365_l(start=ex8_start, asof=ex8_asof, end=ex8_asof)
    assert result8 == Decimal(364) / Decimal(365)
    
    # Test case 9: Full leap year
    ex9_start = datetime.date(2020, 1, 1)
    ex9_asof = datetime.date(2020, 12, 31)
    result9 = dcfc_act_365_l(start=ex9_start, asof=ex9_asof, end=ex9_asof)
    assert result9 == Decimal(365) / Decimal(366)
    
    # Test case 10: Verify freq parameter is ignored
    ex10_start = datetime.date(2020, 1, 1)
    ex10_asof = datetime.date(2020, 1, 2)
    result10a = dcfc_act_365_l(start=ex10_start, asof=ex10_asof, end=ex10_asof, freq=None)
    result10b = dcfc_act_365_l(start=ex10_start, asof=ex10_asof, end=ex10_asof, freq=Decimal(2))
    assert result10a == result10b


# LLM-generated content at query #17
#--------------------------

```python
def test_DCC_coupon():
    """Test the coupon method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Create a simple day count fraction calculation function
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        """Simple day count fraction: actual/360."""
        days = (asof - start).days
        return Decimal(days) / Decimal(360)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test DCC",
        altnames={"T-DCC", "TDCC"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Create test data
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")  # 5% annual rate
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 15)
    end = datetime.date(2014, 12, 31)
    freq = 2  # Semi-annual
    eom = None
    
    # Test basic coupon calculation
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert isinstance(result, Money)
    assert result.currency == principal.currency
    assert result.amount > ZERO
    
    # Test with end of month flag
    start_eom = datetime.date(2014, 1, 31)
    asof_eom = datetime.date(2014, 7, 15)
    end_eom = datetime.date(2014, 12, 31)
    eom_value = 31
    
    result_eom = dcc.coupon(principal, rate, start_eom, asof_eom, end_eom, freq, eom_value)
    assert isinstance(result_eom, Money)
    assert result_eom.amount > ZERO
    
    # Test with different frequency
    freq_quarterly = 4
    result_quarterly = dcc.coupon(principal, rate, start, asof, end, freq_quarterly, eom)
    assert isinstance(result_quarterly, Money)
    
    # Test with zero rate
    result_zero_rate = dcc.coupon(principal, Decimal("0"), start, asof, end, freq, eom)
    assert result_zero_rate.amount == ZERO
    
    # Test with different principal amounts
    principal_large = Money(Decimal("1000000"), "USD")
    result_large = dcc.coupon(principal_large, rate, start, asof, end, freq, eom)
    assert result_large.amount > result.amount
    
    # Test with higher rate
    higher_rate = Decimal("0.10")  # 10% annual rate
    result_higher_rate = dcc.coupon(principal, higher_rate, start, asof, end, freq, eom)
    assert result_higher_rate.amount > result.amount
    
    # Test return type consistency
    result1 = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    result2 = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result1.currency == result2.currency
    
    # Test with different currencies
    principal_eur = Money(Decimal("1000"), "EUR")
    result_eur = dcc.coupon(principal_eur, rate, start, asof, end, freq, eom)
    assert result_eur.currency == "EUR"


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_30_360_german():
    import datetime
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: 2007-12-28 to 2008-02-29
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test case with end date different from asof date
    test_start = datetime.date(2008, 2, 29)
    test_asof = datetime.date(2008, 2, 29)
    test_end = datetime.date(2008, 3, 31)
    result5 = dcfc_30_360_german(start=test_start, asof=test_asof, end=test_end)
    assert isinstance(result5, Decimal)
    
    # Test case with 31st day adjustment
    test_start = datetime.date(2008, 1, 31)
    test_asof = datetime.date(2008, 2, 29)
    result6 = dcfc_30_360_german(start=test_start, asof=test_asof, end=test_asof)
    assert isinstance(result6, Decimal)
    assert result6 > 0
    
    # Test case same start and asof date
    same_date = datetime.date(2008, 3, 15)
    result7 = dcfc_30_360_german(start=same_date, asof=same_date, end=same_date)
    assert result7 == Decimal(0)
    
    # Test case with optional freq parameter
    result8 = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=None)
    assert round(result8, 14) == Decimal('0.16666666666667')


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_30_360_german():
    import datetime
    
    # Test case 1: ex1_start, ex1_asof
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: ex2_start, ex2_asof
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: ex3_start, ex3_asof
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: ex4_start, ex4_asof
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test case with same start and asof date
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_30_360_german(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case with day 31 start date
    start_31 = datetime.date(2008, 1, 31)
    asof_15 = datetime.date(2008, 2, 15)
    result6 = dcfc_30_360_german(start=start_31, asof=asof_15, end=asof_15)
    assert result6 == Decimal('0.16111111111111')
    
    # Test case with February 29 (leap year)
    start_feb = datetime.date(2008, 2, 1)
    asof_feb_29 = datetime.date(2008, 2, 29)
    result7 = dcfc_30_360_german(start=start_feb, asof=asof_feb_29, end=asof_feb_29)
    assert result7 == Decimal('0.07777777777778')
    
    # Test case with end date different from asof (February last day scenario)
    start_feb2 = datetime.date(2008, 2, 1)
    asof_feb_end = datetime.date(2008, 2, 29)
    end_mar = datetime.date(2008, 3, 1)
    result8 = dcfc_30_360_german(start=start_feb2, asof=asof_feb_end, end=end_mar)
    assert result8 == Decimal('0.08055555555556')


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_act_365_l():
    """Unit tests for dcfc_act_365_l function."""
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16939890710383')
    
    # Test case 2: Example from docstring with leap day
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17213114754098')
    
    # Test case 3: Example from docstring spanning multiple years
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_l(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08196721311475')
    
    # Test case 4: Example from docstring
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_l(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32876712328767')
    
    # Test case 5: Same start and asof date (zero days)
    same_date = datetime.date(2020, 6, 15)
    result5 = dcfc_act_365_l(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal(0)
    
    # Test case 6: One day difference in non-leap year
    start_date = datetime.date(2019, 3, 15)
    asof_date = datetime.date(2019, 3, 16)
    result6 = dcfc_act_365_l(start=start_date, asof=asof_date, end=asof_date)
    assert result6 == Decimal(1) / Decimal(365)
    
    # Test case 7: One day difference in leap year (asof is in leap year)
    start_date = datetime.date(2019, 12, 31)
    asof_date = datetime.date(2020, 1, 1)
    result7 = dcfc_act_365_l(start=start_date, asof=asof_date, end=asof_date)
    assert result7 == Decimal(1) / Decimal(366)
    
    # Test case 8: With freq parameter (should be ignored)
    ex_start = datetime.date(2007, 12, 28)
    ex_asof = datetime.date(2008, 2, 28)
    result8a = dcfc_act_365_l(start=ex_start, asof=ex_asof, end=ex_asof, freq=None)
    result8b = dcfc_act_365_l(start=ex_start, asof=ex_asof, end=ex_asof, freq=Decimal(2))
    assert result8a == result8b


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_30_360_german():
    """Unit tests for dcfc_30_360_german function."""
    import datetime
    
    # Test case 1: ex1_start, ex1_asof
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test case 2: ex2_start, ex2_asof
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test case 3: ex3_start, ex3_asof
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test case 4: ex4_start, ex4_asof
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test case 5: Same start and asof date
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_30_360_german(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: Start on 31st day
    start_31 = datetime.date(2008, 1, 31)
    asof_date = datetime.date(2008, 2, 15)
    result6 = dcfc_30_360_german(start=start_31, asof=asof_date, end=asof_date)
    assert result6 > Decimal('0')
    
    # Test case 7: Asof on 31st day
    start_normal = datetime.date(2008, 1, 15)
    asof_31 = datetime.date(2008, 2, 29)
    result7 = dcfc_30_360_german(start=start_normal, asof=asof_31, end=asof_31)
    assert result7 > Decimal('0')
    
    # Test case 8: Both start and asof on 31st
    start_31_2 = datetime.date(2008, 1, 31)
    asof_31_2 = datetime.date(2008, 3, 31)
    result8 = dcfc_30_360_german(start=start_31_2, asof=asof_31_2, end=asof_31_2)
    assert result8 == Decimal('2')
    
    # Test case 9: Result should be Decimal type
    result9 = dcfc_30_360_german(start=datetime.date(2008, 1, 1), asof=datetime.date(2008, 1, 2), end=datetime.date(2008, 1, 2))
    assert isinstance(result9, Decimal)
    
    # Test case 10: Across multiple years
    start_multi_year = datetime.date(2007, 1, 1)
    asof_multi_year = datetime.date(2010, 12, 31)
    result10 = dcfc_30_360_german(start=start_multi_year, asof=asof_multi_year, end=asof_multi_year)
    assert result10 == Decimal('4')


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_nl_365():
    """Unit tests for dcfc_nl_365 function."""
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day included)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: Same start and asof date (zero days)
    same_date = datetime.date(2008, 6, 15)
    result5 = dcfc_nl_365(start=same_date, asof=same_date, end=same_date)
    assert result5 == Decimal('0')
    
    # Test case 6: One day difference
    start_date = datetime.date(2008, 6, 15)
    asof_date = datetime.date(2008, 6, 16)
    result6 = dcfc_nl_365(start=start_date, asof=asof_date, end=asof_date)
    assert result6 == Decimal('1') / Decimal(365)
    
    # Test case 7: Verify leap day is excluded (2008 is a leap year)
    leap_start = datetime.date(2008, 2, 28)
    leap_asof = datetime.date(2008, 3, 1)
    result7 = dcfc_nl_365(start=leap_start, asof=leap_asof, end=leap_asof)
    # Should be 1 day (leap day is excluded)
    assert result7 == Decimal('1') / Decimal(365)


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_nl_365():
    """Test the dcfc_nl_365 function with various date ranges."""
    import datetime
    
    # Test case 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')
    
    # Test case 2: 2007-12-28 to 2008-02-29 (leap day included)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')
    
    # Test case 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')
    
    # Test case 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')
    
    # Test case 5: same start and asof date
    test_start = datetime.date(2020, 1, 1)
    test_asof = datetime.date(2020, 1, 1)
    result5 = dcfc_nl_365(start=test_start, asof=test_asof, end=test_asof)
    assert result5 == Decimal('0')
    
    # Test case 6: one day difference
    test_start = datetime.date(2020, 1, 1)
    test_asof = datetime.date(2020, 1, 2)
    result6 = dcfc_nl_365(start=test_start, asof=test_asof, end=test_asof)
    assert round(result6, 14) == Decimal('0.00273972602740')
    
    # Test case 7: with freq parameter (should be ignored)
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result7 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof, freq=Decimal('2'))
    assert round(result7, 14) == Decimal('0.16986301369863')


# LLM-generated content at query #24
#--------------------------

```python
def test_dcfc_30_360_german():
    import datetime
    
    # Test example 1: 2007-12-28 to 2008-02-28
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16666666666667')
    
    # Test example 2: 2007-12-28 to 2008-02-29
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16944444444444')
    
    # Test example 3: 2007-10-31 to 2008-11-30
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08333333333333')
    
    # Test example 4: 2008-02-01 to 2009-05-31
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.33055555555556')
    
    # Test with same start and asof date
    same_date = datetime.date(2008, 6, 15)
    result_same = dcfc_30_360_german(start=same_date, asof=same_date, end=same_date)
    assert result_same == Decimal('0')
    
    # Test with end of month dates
    month_end_start = datetime.date(2008, 2, 29)
    month_end_asof = datetime.date(2008, 3, 31)
    result_month_end = dcfc_30_360_german(start=month_end_start, asof=month_end_asof, end=month_end_asof)
    assert isinstance(result_month_end, Decimal)
    assert result_month_end > 0
    
    # Test with 31st day dates
    day_31_start = datetime.date(2008, 1, 31)
    day_31_asof = datetime.date(2008, 2, 15)
    result_day_31 = dcfc_30_360_german(start=day_31_start, asof=day_31_asof, end=day_31_asof)
    assert isinstance(result_day_31, Decimal)
    assert result_day_31 > 0
    
    # Test where end date equals asof date (normal case)
    normal_start = datetime.date(2008, 1, 15)
    normal_asof = datetime.date(2008, 3, 20)
    result_normal = dcfc_30_360_german(start=normal_start, asof=normal_asof, end=normal_asof)
    assert isinstance(result_normal, Decimal)
    assert result_normal > 0
    
    # Test where end date differs from asof date
    diff_start = datetime.date(2008, 2, 29)
    diff_asof = datetime.date(2008, 2, 29)
    diff_end = datetime.date(2008, 3, 31)
    result_diff = dcfc_30_360_german(start=diff_start, asof=diff_asof, end=diff_end)
    assert isinstance(result_diff, Decimal)


# LLM-generated content at query #25
#--------------------------

```python
def test_DCC_interest():
    """Test the interest calculation method of DCC class."""
    import datetime
    from decimal import Decimal
    
    # Define a simple day count fraction method (Actual/365)
    def simple_dcfc(start: datetime.date, asof: datetime.date, end: datetime.date, freq: Optional[Decimal]) -> Decimal:
        days = (asof - start).days
        return Decimal(days) / Decimal(365)
    
    # Create a DCC instance
    dcc = DCC(
        name="Test/365",
        altnames={"T/365"},
        currencies=set(),
        calculate_fraction_method=simple_dcfc
    )
    
    # Create test Money objects
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")  # 5% annual rate
    
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 7, 2)  # 182 days later
    end_date = datetime.date(2023, 12, 31)
    
    # Test basic interest calculation
    result = dcc.interest(principal, rate, start_date, asof_date, end_date)
    expected_fraction = Decimal(182) / Decimal(365)
    expected_interest = principal * rate * expected_fraction
    assert result == expected_interest
    
    # Test with default end date (uses asof as end date)
    result_default_end = dcc.interest(principal, rate, start_date, asof_date)
    expected_fraction_default = Decimal(182) / Decimal(365)
    expected_interest_default = principal * rate * expected_fraction_default
    assert result_default_end == expected_interest_default
    
    # Test with same start and asof date (0 days)
    result_zero = dcc.interest(principal, rate, start_date, start_date, end_date)
    assert result_zero == Money(Decimal("0"), "USD")
    
    # Test with one day difference
    one_day_later = start_date + datetime.timedelta(days=1)
    result_one_day = dcc.interest(principal, rate, start_date, one_day_later, end_date)
    expected_one_day = principal * rate * (Decimal(1) / Decimal(365))
    assert result_one_day == expected_one_day
    
    # Test with different principal and rate
    principal_large = Money(Decimal("10000"), "USD")
    rate_high = Decimal("0.10")  # 10% annual rate
    result_large = dcc.interest(principal_large, rate_high, start_date, asof_date, end_date)
    expected_large = principal_large * rate_high * expected_fraction
    assert result_large == expected_large
    
    # Test with freq parameter
    freq = Decimal("2")
    result_with_freq = dcc.interest(principal, rate, start_date, asof_date, end_date, freq)
    assert result_with_freq == expected_interest


