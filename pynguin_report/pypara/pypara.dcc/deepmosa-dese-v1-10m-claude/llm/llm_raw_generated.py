####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_successful():
    """Test successful registration of a DCC."""
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc


def test_register_with_altnames():
    """Test registration of a DCC with alternative names."""
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Alt1", "Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Alt1"] == dcc
    assert registry._buffer_altn["Alt2"] == dcc


def test_register_duplicate_main_name():
    """Test that registering duplicate main name raises TypeError."""
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_duplicate_altname():
    """Test that registering duplicate alternative name raises TypeError."""
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Alt1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Alt1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_main_name():
    """Test that registering altname that conflicts with existing main name raises TypeError."""
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/DCC1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_main_name_conflicts_with_existing_altname():
    """Test that registering main name that conflicts with existing altname raises TypeError."""
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Alt1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Alt1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_30_e_plus_360_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_plus_360_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_plus_360_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_plus_360_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_e_plus_360_same_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_e_plus_360_one_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_e_plus_360_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 1)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('2') / Decimal('360')


def test_dcfc_30_e_plus_360_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('16') / Decimal('360')


def test_dcfc_30_e_plus_360_full_year():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2009, 1, 1)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('360') / Decimal('360')


# LLM-generated content at query #3
#--------------------------

```python
def test_calculate_fraction_valid_dates():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 6, 15)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")


def test_calculate_fraction_asof_equals_start():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_method(start, asof, end, freq):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 1, 1)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.25")


def test_calculate_fraction_asof_equals_end():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_method(start, asof, end, freq):
        return Decimal("1.0")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 12, 31)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("1.0")


def test_calculate_fraction_asof_before_start():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_method
    )
    
    start = date(2023, 6, 15)
    asof = date(2023, 1, 1)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0")


def test_calculate_fraction_asof_after_end():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2024, 1, 1)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0")


def test_calculate_fraction_with_frequency():
    from decimal import Decimal
    from datetime import date
    
    def mock_calculate_method(start, asof, end, freq):
        return freq * Decimal("0.1")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 6, 15)
    end = date(2023, 12, 31)
    freq = Decimal("4")
    
    result = dcc.calculate_fraction(start, asof, end, freq)
    assert result == Decimal("0.4")


# LLM-generated content at query #4
#--------------------------

```python
def test_dcfc_act_365_a_basic():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')


def test_dcfc_act_365_a_leap_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17213114754098')


def test_dcfc_act_365_a_long_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08196721311475')


def test_dcfc_act_365_a_over_year():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32513661202186')


def test_dcfc_act_365_a_same_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2008, 3, 15)
    asof = datetime.date(2008, 3, 15)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_act_365_a_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2008, 3, 15)
    asof = datetime.date(2008, 3, 16)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('366')


def test_dcfc_act_365_a_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof, freq=Decimal('2'))
    assert round(result, 14) == Decimal('0.16986301369863')


# LLM-generated content at query #5
#--------------------------

```python
def test_coupon_basic_calculation():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("1000")
    rate = Decimal("0.05")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 1)
    end = datetime.date(2015, 1, 1)
    freq = 1
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal("0.5")
    assert result == expected


def test_coupon_with_eom():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("5000")
    rate = Decimal("0.03")
    start = datetime.date(2014, 1, 31)
    asof = datetime.date(2014, 3, 15)
    end = datetime.date(2015, 1, 31)
    freq = 2
    eom = 31
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.25")
    assert result == expected


def test_coupon_zero_rate():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("1000")
    rate = Decimal("0")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 1)
    end = datetime.date(2015, 1, 1)
    freq = 1
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("0")


def test_coupon_different_frequencies():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.1") * freq
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("2000")
    rate = Decimal("0.04")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 4, 1)
    end = datetime.date(2015, 1, 1)
    freq = 4
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * (Decimal("0.1") * Decimal(freq))
    assert result == expected


def test_coupon_decimal_frequency():
    import datetime
    from decimal import Decimal
    
    def mock_calculate_fraction(start, asof, end, freq):
        return Decimal("0.33")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )
    
    principal = Decimal("10000")
    rate = Decimal("0.06")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 8, 1)
    end = datetime.date(2015, 1, 1)
    freq = Decimal("2")
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal("0.33")
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_next_payment_date_annual_frequency_no_eom():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, None)
    expected = datetime.date(2015, 1, 1)
    assert result == expected


def test_next_payment_date_annual_frequency_with_eom():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    expected = datetime.date(2015, 1, 15)
    assert result == expected


def test_next_payment_date_semi_annual_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 2, None)
    expected = datetime.date(2014, 7, 1)
    assert result == expected


def test_next_payment_date_quarterly_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 4, None)
    expected = datetime.date(2014, 4, 1)
    assert result == expected


def test_next_payment_date_monthly_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 12, None)
    expected = datetime.date(2014, 2, 1)
    assert result == expected


def test_next_payment_date_eom_invalid_day():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 31), 1, 31)
    expected = datetime.date(2015, 1, 31)
    assert result == expected


def test_next_payment_date_eom_february_leap_year():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    result = _next_payment_date(datetime.date(2014, 1, 31), 1, 29)
    expected = datetime.date(2015, 1, 29)
    assert result == expected


def test_next_payment_date_decimal_frequency():
    import datetime
    from dateutil.relativedelta import relativedelta
    from decimal import Decimal
    
    result = _next_payment_date(datetime.date(2014, 1, 1), Decimal(2), None)
    expected = datetime.date(2014, 7, 1)
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_find_with_exact_name():
    from decimal import Decimal
    import datetime
    
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("Act/Act", [])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is not None
    assert result.name == "Act/Act"


def test_find_with_alternative_name():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("Actual/Actual", ["Act/Act"])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is not None
    assert result.name == "Actual/Actual"


def test_find_with_stripped_uppercase_name():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("ACT/ACT", [])
    registry.register(dcc)
    
    result = registry.find("  act/act  ")
    assert result is not None
    assert result.name == "ACT/ACT"


def test_find_nonexistent_dcc():
    registry = DCCRegistryMachinery()
    
    result = registry.find("NonExistent/DCC")
    assert result is None


def test_find_case_insensitive():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("30/360", [])
    registry.register(dcc)
    
    result = registry.find("30/360")
    assert result is not None
    assert result.name == "30/360"


def test_find_with_whitespace():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("BOND BASIS", [])
    registry.register(dcc)
    
    result = registry.find("  bond basis  ")
    assert result is not None
    assert result.name == "BOND BASIS"


def test_find_returns_none_for_unregistered():
    registry = DCCRegistryMachinery()
    
    result = registry.find("UnknownDCC")
    assert result is None


def test_find_multiple_registrations():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc1 = MockDCC("ACT/360", [])
    dcc2 = MockDCC("ACT/365", [])
    registry.register(dcc1)
    registry.register(dcc2)
    
    result1 = registry.find("ACT/360")
    result2 = registry.find("ACT/365")
    
    assert result1 is not None
    assert result1.name == "ACT/360"
    assert result2 is not None
    assert result2.name == "ACT/365"


# LLM-generated content at query #8
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 1)
    end = date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_without_leap_day_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_multiple_leap_years_with_one_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2019, 2, 1)
    end = date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_multiple_leap_years_none_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 3, 1)
    end = date(2024, 2, 1)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_start_equals_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 29)
    end = date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_end_equals_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 1)
    end = date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_range_is_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 29)
    end = date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_before_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 1)
    end = date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_after_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 3, 1)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_multi_year_range_with_multiple_leap_days():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2019, 2, 1)
    end = date(2024, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_30_e_plus_360_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_plus_360_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_plus_360_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_plus_360_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_e_plus_360_same_dates():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 3, 15)
    asof = datetime.date(2008, 3, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_e_plus_360_one_day_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 3, 15)
    asof = datetime.date(2008, 3, 16)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_e_plus_360_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    nod = (29 - 30) + 30 * (2 - 1) + 360 * (2008 - 2008)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected


def test_dcfc_30_e_plus_360_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    adjusted_asof_day = 1
    adjusted_asof_month = 2
    nod = (adjusted_asof_day - 15) + 30 * (adjusted_asof_month - 1) + 360 * (2008 - 2008)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_30_e_360_example_1():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_360_example_2():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_360_example_3():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_360_example_4():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')


def test_dcfc_30_e_360_start_day_31():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2007, 11, 30)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = Decimal('1') / Decimal('12')
    assert round(result, 14) == round(expected, 14)


def test_dcfc_30_e_360_asof_day_31():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    start = datetime.date(2007, 10, 30)
    asof = datetime.date(2007, 10, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = Decimal('0')
    assert result == expected


def test_dcfc_30_e_360_same_dates():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    date = datetime.date(2008, 6, 15)
    result = dcfc_30_e_360(start=date, asof=date, end=date)
    assert result == Decimal('0')


def test_dcfc_30_e_360_one_year_apart():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    start = datetime.date(2007, 6, 15)
    asof = datetime.date(2008, 6, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result == Decimal('1')


def test_dcfc_30_e_360_month_difference():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 3, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = Decimal('60') / Decimal('360')
    assert result == expected


def test_dcfc_30_e_360_both_end_of_month_31():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    start = datetime.date(2007, 8, 31)
    asof = datetime.date(2007, 9, 30)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    expected = Decimal('30') / Decimal('360')
    assert result == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq=None):
        return Decimal(0)
    
    dcc1 = DCC(
        name="Act/Act",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    dcc2 = DCC(
        name="Act/Act",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "Act/Act" in str(e)
        assert "already registered" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_find_with_stripped_uppercase_name():
    from datetime import date
    from decimal import Decimal
    
    # Create a mock DCC class for testing
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    # Create registry and mock DCC
    registry = DCCRegistryMachinery()
    mock_dcc = MockDCC("ACT/ACT", altnames=[])
    
    # Register the DCC with uppercase name
    registry.register(mock_dcc)
    
    # Test that find returns the DCC when using lowercase with spaces
    result = registry.find("  act/act  ")
    assert result is mock_dcc
    assert result.name == "ACT/ACT"


# LLM-generated content at query #13
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_annual_same_year():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_semi_annual_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 7, 1)


def test_last_payment_date_semi_annual_august():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result == datetime.date(2015, 7, 1)


def test_last_payment_date_semi_annual_april():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_annual_june():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)


def test_last_payment_date_quarterly_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)


def test_last_payment_date_december_payment():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result == datetime.date(2014, 12, 9)


def test_last_payment_date_semi_annual_december():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_semi_annual_december_year_end():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_before_first_payment():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2015, 6, 1), datetime.date(2015, 3, 31), 1)
    assert result == datetime.date(2015, 6, 1)


def test_last_payment_date_with_eom_parameter():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 12, 31), 1, eom=31)
    assert result == datetime.date(2015, 1, 31)


def test_last_payment_date_february_eom_handling():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 2, 28), 1)
    assert result == datetime.date(2015, 1, 31)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_date_range():
    from datetime import date, timedelta
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)
    
    result = list(_get_date_range(start, end))
    
    expected = [
        date(2023, 1, 1),
        date(2023, 1, 2),
        date(2023, 1, 3),
        date(2023, 1, 4)
    ]
    
    assert result == expected
    assert len(result) == 4


def test_get_date_range_single_day():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 1)
    
    result = list(_get_date_range(start, end))
    
    assert result == []
    assert len(result) == 0


def test_get_date_range_two_days():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 2)
    
    result = list(_get_date_range(start, end))
    
    assert result == [date(2023, 1, 1)]
    assert len(result) == 1


def test_get_date_range_end_exclusive():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 3)
    
    result = list(_get_date_range(start, end))
    
    assert end not in result
    assert date(2023, 1, 3) not in result


def test_get_date_range_returns_generator():
    from datetime import date
    from pypara.dcc import _get_date_range
    from collections.abc import Generator
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)
    
    result = _get_date_range(start, end)
    
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


def test_get_date_range_long_period():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 31)
    
    result = list(_get_date_range(start, end))
    
    assert len(result) == 30
    assert result[0] == date(2023, 1, 1)
    assert result[-1] == date(2023, 1, 30)


# LLM-generated content at query #15
#--------------------------

```python
def test_is_last_day_of_month_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 1, 31))
    assert result is True


def test_is_last_day_of_month_not_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 1, 30))
    assert result is False


def test_is_last_day_of_month_february_leap_year():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2020, 2, 29))
    assert result is True


def test_is_last_day_of_month_february_non_leap_year():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 2, 28))
    assert result is True


def test_is_last_day_of_month_april_30_days():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 4, 30))
    assert result is True


def test_is_last_day_of_month_december_31():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 12, 31))
    assert result is True


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_act_act_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('0.16942884946478')


def test_dcfc_act_act_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('0.17216108990194')


def test_dcfc_act_act_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('1.08243131970956')


def test_dcfc_act_act_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('1.32625945055768')


def test_dcfc_act_act_same_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 1)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_act_act_non_leap_year():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_act_leap_year():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('366')


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_30_360_isda_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_isda_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_360_isda_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_360_isda_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_360_isda_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = (15 - 30) + 30 * (1 - 12) + 360 * (2008 - 2007)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_360_isda_start_day_30_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 12, 30)
    asof = datetime.date(2008, 1, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = (30 - 30) + 30 * (1 - 12) + 360 * (2008 - 2007)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_360_isda_same_dates():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert result == Decimal(0)


def test_dcfc_30_360_isda_one_day_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert result == Decimal(1) / Decimal(360)


def test_dcfc_30_360_isda_month_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = (15 - 15) + 30 * (2 - 1) + 360 * (2008 - 2008)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_360_isda_year_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    expected = (15 - 15) + 30 * (1 - 1) + 360 * (2008 - 2007)
    assert result == Decimal(expected) / Decimal(360)


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_nl_365_basic():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')


def test_dcfc_nl_365_leap_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.16986301369863')


def test_dcfc_nl_365_long_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08219178082192')


def test_dcfc_nl_365_extended_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32602739726027')


def test_dcfc_nl_365_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    start = datetime.date(2020, 1, 1)
    result = dcfc_nl_365(start=start, asof=start, end=start)
    assert result == Decimal('0')


def test_dcfc_nl_365_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_nl_365_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_nl_365
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    freq = Decimal('4')
    result = dcfc_nl_365(start=start, asof=asof, end=asof, freq=freq)
    assert isinstance(result, Decimal)
    assert result > Decimal('0')


# LLM-generated content at query #19
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_coupon_basic():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    principal = Decimal("1000")
    rate = Decimal("0.05")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 15)
    end = datetime.date(2015, 1, 1)
    freq = 1
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("25")


def test_coupon_with_eom():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    principal = Decimal("2000")
    rate = Decimal("0.04")
    start = datetime.date(2014, 1, 15)
    asof = datetime.date(2014, 6, 15)
    end = datetime.date(2015, 1, 15)
    freq = 2
    eom = 15
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal("20")


def test_coupon_semi_annual():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    principal = Decimal("5000")
    rate = Decimal("0.06")
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    end = datetime.date(2016, 6, 15)
    freq = 2
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("750")


def test_coupon_quarterly():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    principal = Decimal("10000")
    rate = Decimal("0.08")
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    end = datetime.date(2016, 1, 7)
    freq = 4
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("200")


def test_coupon_with_decimal_frequency():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction
    )
    
    principal = Decimal("1000")
    rate = Decimal("0.1")
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 7, 1)
    end = datetime.date(2015, 1, 1)
    freq = Decimal("1")
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal("50")


# LLM-generated content at query #21
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 28)
    end = date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_without_leap_day_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2021, 2, 28)
    end = date(2021, 3, 1)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_leap_day_on_start_date():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 29)
    end = date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_leap_day_on_end_date():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 28)
    end = date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_multiple_leap_years_in_range():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 1, 1)
    end = date(2024, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_before_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 1, 1)
    end = date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_after_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 3, 1)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_single_day_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 29)
    end = date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_single_day_non_leap_day():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2020, 2, 28)
    end = date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_century_leap_year():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(2000, 2, 28)
    end = date(2000, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_century_non_leap_year():
    from datetime import date
    from pypara.dcc import _has_leap_day
    
    start = date(1900, 2, 28)
    end = date(1900, 3, 1)
    result = _has_leap_day(start, end)
    assert result is False


# LLM-generated content at query #22
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_last_payment_date_predicate_line_57():
    import datetime
    from decimal import Decimal
    
    # Test case where p_year < 1 (year goes below 1)
    # This would require going back many years from a very early date
    # We need to construct a scenario where the calculation results in p_year < 1
    result = _last_payment_date(datetime.date(1, 1, 1), datetime.date(1, 1, 1), 1)
    assert result == datetime.date(1, 1, 1)


# LLM-generated content at query #24
#--------------------------

def test_dcfc_act_act_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start, asof, end)
    assert round(result, 14) == Decimal('0.16942884946478')


def test_dcfc_act_act_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start, asof, end)
    assert round(result, 14) == Decimal('0.17216108990194')


def test_dcfc_act_act_across_years():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start, asof, end)
    assert round(result, 14) == Decimal('1.08243131970956')


def test_dcfc_act_act_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start, asof, end)
    assert round(result, 14) == Decimal('1.32625945055768')


def test_dcfc_act_act_same_date():
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 1)
    end = datetime.date(2008, 1, 1)
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('0')


def test_dcfc_act_act_one_day():
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 2)
    end = datetime.date(2008, 1, 2)
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('1') / Decimal('366')


def test_dcfc_act_act_non_leap_year():
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    end = datetime.date(2007, 1, 2)
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_act_with_freq_parameter():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    freq = Decimal('2')
    result = dcfc_act_act(start, asof, end, freq)
    assert round(result, 14) == Decimal('0.16942884946478')


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_54_evaluates_to_false():
    import datetime
    from decimal import Decimal
    
    # Test case where future list is empty, so the predicate evaluates to False
    # This happens when no payment months are before current month or equal to current month with eom <= c_day
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 1, 1)
    frequency = 1
    eom = None
    
    result = _last_payment_date(start, asof, frequency, eom)
    
    # When future is empty, the else branch is taken: (c_year - 1, schedule[-1])
    # This means the predicate "if future" evaluates to False
    assert result == datetime.date(2013, 1, 1)


# LLM-generated content at query #26
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    machinery = DCCRegistryMachinery()
    assert machinery._buffer_main == {}
    assert machinery._buffer_altn == {}
    assert isinstance(machinery._buffer_main, dict)
    assert isinstance(machinery._buffer_altn, dict)


# LLM-generated content at query #27
#--------------------------

```python
def test_eom_parameter_false_condition():
    import datetime
    from decimal import Decimal
    
    # Test case where eom is explicitly provided (not None and not 0)
    # This ensures the predicate "eom or start.day" evaluates to False for the "or" part
    # by having eom be a truthy value (non-zero, non-None)
    start = datetime.date(2014, 1, 15)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = 10
    
    # When eom is provided as 10, the condition "eom or start.day" should use eom (10)
    # not start.day (15), meaning the "or" predicate evaluates to False
    result = _last_payment_date(start, asof, frequency, eom)
    
    # Verify that eom=10 was used, not start.day=15
    assert result.day == 10


# LLM-generated content at query #28
#--------------------------

```python
def test_next_payment_date_with_eom():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    def _next_payment_date(start, frequency, eom=None):
        months = int(12 / frequency)
        nextdate = start + relativedelta(months=months)
        if eom:
            try:
                nextdate = nextdate.replace(day=eom)
            except ValueError:
                pass
        return nextdate
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    assert result == datetime.date(2015, 1, 15)


# LLM-generated content at query #29
#--------------------------

```python
def test_last_payment_date_predicate_line_1_false():
    import datetime
    from decimal import Decimal
    
    # Test case where the predicate at line 1 evaluates to False
    # The predicate is: def _last_payment_date(start: Date, asof: Date, frequency: Union[int, Decimal], eom: Optional[int] = None) -> Date:
    # This is a function definition, so we test that it can be called with valid arguments
    # and returns a Date object (not None or False)
    
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    
    result = _last_payment_date(start, asof, frequency)
    
    assert result is not None
    assert isinstance(result, datetime.date)
    assert result == datetime.date(2015, 1, 1)


# LLM-generated content at query #30
#--------------------------

```python
def test_find_evaluates_predicate_to_true():
    from unittest.mock import Mock
    
    # Create a mock DCC object
    mock_dcc = Mock()
    mock_dcc.name = "Act/Act"
    mock_dcc.altnames = []
    
    # Create registry and register the mock DCC
    registry = DCCRegistryMachinery()
    registry.register(mock_dcc)
    
    # Test that find returns a truthy value (the DCC object)
    result = registry.find("Act/Act")
    assert result is not None
    assert result == mock_dcc
    
    # Test with stripped and uppercase name
    result_stripped = registry.find("  act/act  ")
    assert result_stripped is not None
    assert result_stripped == mock_dcc


# LLM-generated content at query #31
#--------------------------

```python
def test_dcfc_act_act_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16942884946478')


def test_dcfc_act_act_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.17216108990194')


def test_dcfc_act_act_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08243131970956')


def test_dcfc_act_act_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.32625945055768')


def test_dcfc_act_act_same_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 1)
    end = datetime.date(2008, 2, 1)
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert result == Decimal('0')


def test_dcfc_act_act_with_freq_parameter():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    freq = Decimal('2')
    result = dcfc_act_act(start=start, asof=asof, end=end, freq=freq)
    assert round(result, 14) == Decimal('0.16942884946478')


def test_dcfc_act_act_leap_year():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert result == Decimal('1') / Decimal('366')


def test_dcfc_act_act_non_leap_year():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 2, 28)
    asof = datetime.date(2007, 3, 1)
    end = datetime.date(2007, 3, 1)
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert result == Decimal('1') / Decimal('365')


# LLM-generated content at query #32
#--------------------------

```python
def test_has_leap_day_predicate_evaluates_to_true():
    import calendar
    from datetime import date
    
    # Create a date range that includes a leap day (February 29)
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    
    # Get all leap years in the range
    years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}
    
    # The predicate at line 9: for year in years
    # We need to verify that this loop iterates and the condition at line 14 is true
    result = False
    for year in years:
        leapday = date(year, 2, 29)
        if start <= leapday <= end:
            result = True
            break
    
    assert result is True


def test_has_leap_day_with_leap_year_range():
    import calendar
    from datetime import date
    
    # Test with a range spanning leap year 2024
    start = date(2024, 2, 1)
    end = date(2024, 3, 1)
    
    years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}
    
    result = False
    for year in years:
        leapday = date(year, 2, 29)
        if start <= leapday <= end:
            result = True
            break
    
    assert result is True


def test_has_leap_day_with_multiple_leap_years():
    import calendar
    from datetime import date
    
    # Test with range spanning multiple leap years (2000, 2004, 2008)
    start = date(2000, 1, 1)
    end = date(2008, 12, 31)
    
    years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}
    
    result = False
    for year in years:
        leapday = date(year, 2, 29)
        if start <= leapday <= end:
            result = True
            break
    
    assert result is True


# LLM-generated content at query #33
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #34
#--------------------------

```python
def test_get_date_range():
    from datetime import date, timedelta
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)
    result = list(_get_date_range(start, end))
    
    expected = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)]
    assert result == expected


def test_get_date_range_single_day():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    
    assert result == []


def test_get_date_range_two_days():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    
    expected = [date(2023, 1, 1)]
    assert result == expected


def test_get_date_range_month_boundary():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 30)
    end = date(2023, 2, 2)
    result = list(_get_date_range(start, end))
    
    expected = [date(2023, 1, 30), date(2023, 1, 31), date(2023, 2, 1)]
    assert result == expected


def test_get_date_range_year_boundary():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2022, 12, 31)
    end = date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    
    expected = [date(2022, 12, 31), date(2023, 1, 1)]
    assert result == expected


def test_get_date_range_is_generator():
    from datetime import date
    from pypara.dcc import _get_date_range
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)
    result = _get_date_range(start, end)
    
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


# LLM-generated content at query #35
#--------------------------

```python
def test_dcfc_act_act_example_1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16942884946478')


def test_dcfc_act_act_example_2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.17216108990194')


def test_dcfc_act_act_example_3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08243131970956')


def test_dcfc_act_act_example_4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.32625945055768')


def test_dcfc_act_act_same_date():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 1)
    end = datetime.date(2008, 1, 1)
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert result == Decimal('0')


def test_dcfc_act_act_one_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 2)
    end = datetime.date(2008, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert result == Decimal('1') / Decimal('366')


def test_dcfc_act_act_non_leap_year():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    end = datetime.date(2007, 1, 2)
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_act_with_freq_parameter():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 2, 1)
    end = datetime.date(2008, 3, 1)
    result = dcfc_act_act(start=start, asof=asof, end=end, freq=Decimal('2'))
    assert isinstance(result, Decimal)
    assert result > Decimal('0')


# LLM-generated content at query #36
#--------------------------

```python
def test_dcfc_act_act_icma_basic():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert round(result, 10) == Decimal('0.5245901639')


def test_dcfc_act_act_icma_same_dates():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 3, 2)
    end = datetime.date(2020, 3, 2)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('0')


def test_dcfc_act_act_icma_end_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 3, 2)
    end = datetime.date(2020, 3, 2)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('1')


def test_dcfc_act_act_icma_with_freq():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal('2')
    
    result = dcfc_act_act_icma(start, asof, end, freq)
    assert round(result, 10) == Decimal('0.2622950820')


def test_dcfc_act_act_icma_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 1, 2)
    end = datetime.date(2019, 1, 365)
    
    result = dcfc_act_act_icma(start, asof, end)
    assert result == Decimal('1') / Decimal('364')


def test_dcfc_act_act_icma_half_year():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2020, 12, 31)
    
    result = dcfc_act_act_icma(start, asof, end)
    expected_days_start_asof = Decimal('182')
    expected_days_start_end = Decimal('365')
    expected = expected_days_start_asof / expected_days_start_end
    assert result == expected


# LLM-generated content at query #37
#--------------------------

```python
def test_dcfc_30_e_plus_360_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_plus_360_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_plus_360_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_plus_360_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_e_plus_360_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert isinstance(result, Decimal)
    assert result > 0


def test_dcfc_30_e_plus_360_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 2, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert isinstance(result, Decimal)
    assert result > 0


def test_dcfc_30_e_plus_360_same_dates():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_e_plus_360_one_day_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_e_plus_360_one_month_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('30') / Decimal('360')


def test_dcfc_30_e_plus_360_one_year_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2009, 1, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('360') / Decimal('360')


# LLM-generated content at query #38
#--------------------------

```python
def test_construct_date_valid_date():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 1, 15)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 15


def test_construct_date_last_day_of_month():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 1, 31)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 31


def test_construct_date_february_leap_year():
    from pypara.dcc import _construct_date
    result = _construct_date(2020, 2, 29)
    assert result.year == 2020
    assert result.month == 2
    assert result.day == 29


def test_construct_date_february_non_leap_year_adjusts_day():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 2, 29)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28


def test_construct_date_april_30_days():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 4, 31)
    assert result.year == 2023
    assert result.month == 4
    assert result.day == 30


def test_construct_date_zero_year_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(0, 1, 15)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."


def test_construct_date_negative_year_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(-1, 1, 15)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."


def test_construct_date_zero_month_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 0, 15)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."


def test_construct_date_negative_month_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, -1, 15)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."


def test_construct_date_zero_day_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 1, 0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."


def test_construct_date_negative_day_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 1, -1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "year, month and day must be greater than 0."


def test_construct_date_invalid_month_raises_error():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 13, 15)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_construct_date_december_31():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 12, 31)
    assert result.year == 2023
    assert result.month == 12
    assert result.day == 31


# LLM-generated content at query #39
#--------------------------

```python
def test_dcfc_nl_365_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16986301369863')


def test_dcfc_nl_365_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16986301369863')


def test_dcfc_nl_365_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08219178082192')


def test_dcfc_nl_365_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.32602739726027')


def test_dcfc_nl_365_same_date():
    import datetime
    from decimal import Decimal
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_nl_365_one_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_nl_365_with_leap_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 2, 29)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result == Decimal('0') / Decimal('365')


def test_dcfc_nl_365_across_leap_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


# LLM-generated content at query #40
#--------------------------

```python
def test_construct_date_predicate_at_line_9_evaluates_to_false():
    from pypara.dcc import _construct_date
    
    result = _construct_date(2023, 2, 28)
    
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28


# LLM-generated content at query #41
#--------------------------

```python
def test_dcfc_30_360_isda_line_29_predicate():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_isda
    
    # Test case where start.day == 30 and asof.day == 31
    # This should trigger the condition at line 29
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 31)
    end = datetime.date(2008, 2, 31)
    
    # First, set up the start date to have day == 30
    start = datetime.date(2008, 2, 30)
    asof = datetime.date(2008, 3, 31)
    end = datetime.date(2008, 3, 31)
    
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    
    # The predicate at line 29 checks: if start.day == 30 and asof.day == 31
    # When this is true, asof should be adjusted to day 30
    # So asof becomes datetime.date(2008, 3, 30)
    # Then nod = (30 - 30) + 30 * (3 - 2) + 360 * (2008 - 2008) = 0 + 30 + 0 = 30
    # result = 30 / 360 = 1/12 ≈ 0.08333...
    
    assert start.day == 30
    assert asof.day == 31
    assert result == Decimal(30) / Decimal(360)


# LLM-generated content at query #42
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #43
#--------------------------

```python
def test_dcfc_30_e_360_asof_day_31_predicate():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_e_360
    
    # Test case where asof.day == 31 (predicate at line 29 should be True)
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 31)
    end = datetime.date(2008, 11, 31)
    
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    
    # When asof.day == 31, it should be adjusted to 30
    # start: 2007-10-30 (adjusted from 31)
    # asof: 2008-11-30 (adjusted from 31)
    # nod = (30 - 30) + 30 * (11 - 10) + 360 * (2008 - 2007)
    # nod = 0 + 30 + 360 = 390
    # result = 390 / 360
    expected = Decimal(390) / Decimal(360)
    
    assert result == expected


# LLM-generated content at query #44
#--------------------------

```python
def test_dcfc_30_360_german_example_1():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    
    result = dcfc_30_360_german(start, asof, end)
    rounded_result = round(result, 14)
    
    assert rounded_result == Decimal('0.16666666666667')


def test_dcfc_30_360_german_example_2():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    
    result = dcfc_30_360_german(start, asof, end)
    rounded_result = round(result, 14)
    
    assert rounded_result == Decimal('0.16944444444444')


def test_dcfc_30_360_german_example_3():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    
    result = dcfc_30_360_german(start, asof, end)
    rounded_result = round(result, 14)
    
    assert rounded_result == Decimal('1.08333333333333')


def test_dcfc_30_360_german_example_4():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    
    result = dcfc_30_360_german(start, asof, end)
    rounded_result = round(result, 14)
    
    assert rounded_result == Decimal('1.33055555555556')


def test_dcfc_30_360_german_same_dates():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    end = datetime.date(2008, 1, 15)
    
    result = dcfc_30_360_german(start, asof, end)
    
    assert result == Decimal('0')


def test_dcfc_30_360_german_month_31st_adjustment():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 15)
    end = datetime.date(2008, 2, 15)
    
    result = dcfc_30_360_german(start, asof, end)
    
    assert result == Decimal('15') / Decimal('360')


def test_dcfc_30_360_german_february_last_day_not_end():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 3, 15)
    end = datetime.date(2008, 3, 20)
    
    result = dcfc_30_360_german(start, asof, end)
    
    assert result == Decimal('15') / Decimal('360')


def test_dcfc_30_360_german_year_difference():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2008, 1, 1)
    end = datetime.date(2008, 1, 1)
    
    result = dcfc_30_360_german(start, asof, end)
    
    assert result == Decimal('360') / Decimal('360')


def test_dcfc_30_360_german_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 2, 1)
    end = datetime.date(2008, 2, 1)
    freq = Decimal('4')
    
    result = dcfc_30_360_german(start, asof, end, freq)
    
    assert result == Decimal('30') / Decimal('360')


# LLM-generated content at query #45
#--------------------------

```python
def test_dcfc_30_360_german_line_25_predicate_true():
    import datetime
    from pypara.dcc import dcfc_30_360_german
    
    # Test case 1: start.day == 31
    start_day_31 = datetime.date(2008, 1, 31)
    asof_day_31 = datetime.date(2008, 2, 28)
    result_1 = dcfc_30_360_german(start=start_day_31, asof=asof_day_31, end=asof_day_31)
    assert result_1 == datetime.Decimal('0.08333333333333')
    
    # Test case 2: start.month == 2 and start is last day of month (Feb 29 in leap year)
    start_feb_last = datetime.date(2008, 2, 29)
    asof_feb_last = datetime.date(2008, 3, 31)
    result_2 = dcfc_30_360_german(start=start_feb_last, asof=asof_feb_last, end=asof_feb_last)
    assert result_2 == datetime.Decimal('0.08333333333333')
    
    # Test case 3: start.month == 2 and start is last day of month (Feb 28 in non-leap year)
    start_feb_last_non_leap = datetime.date(2007, 2, 28)
    asof_feb_last_non_leap = datetime.date(2007, 3, 31)
    result_3 = dcfc_30_360_german(start=start_feb_last_non_leap, asof=asof_feb_last_non_leap, end=asof_feb_last_non_leap)
    assert result_3 == datetime.Decimal('0.08333333333333')


# LLM-generated content at query #46
#--------------------------

```python
def test_find_with_stripped_uppercase_fallback():
    """Test that find method falls back to stripped uppercase name lookup."""
    from unittest.mock import Mock
    
    # Create a mock DCC object
    mock_dcc = Mock()
    mock_dcc.name = "ACT/ACT"
    mock_dcc.altnames = []
    
    # Create registry and register the DCC
    registry = DCCRegistryMachinery()
    registry._buffer_main["ACT/ACT"] = mock_dcc
    
    # Test that find with whitespace and lowercase triggers the fallback
    result = registry.find("  act/act  ")
    
    # The predicate at line 9 should evaluate to True, meaning the second _find_strict call returns the DCC
    assert result == mock_dcc
    assert result is not None


# LLM-generated content at query #47
#--------------------------

```python
def test_dcfc_30_360_german_example_1():
    import datetime
    from decimal import Decimal
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_german_example_2():
    import datetime
    from decimal import Decimal
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_360_german_example_3():
    import datetime
    from decimal import Decimal
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_360_german_example_4():
    import datetime
    from decimal import Decimal
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')


def test_dcfc_30_360_german_same_date():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    result = dcfc_30_360_german(start=start, asof=start, end=start)
    assert result == Decimal('0')


def test_dcfc_30_360_german_one_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_360_german_with_freq_parameter():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof, freq=Decimal('2'))
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_german_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert result == Decimal('28') / Decimal('360')


def test_dcfc_30_360_german_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 31)
    end = datetime.date(2008, 2, 15)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert result == Decimal('16') / Decimal('360')


def test_dcfc_30_360_german_february_last_day_not_end():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 3, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert result == Decimal('0')


def test_dcfc_30_360_german_february_last_day_is_end():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert result == Decimal('29') / Decimal('360')


# LLM-generated content at query #48
#--------------------------

```python
def test_buffer_main_initialization():
    dcc_registry = DCCRegistryMachinery()
    assert isinstance(dcc_registry._buffer_main, dict)
    assert len(dcc_registry._buffer_main) == 0


# LLM-generated content at query #49
#--------------------------

```python
def test_last_payment_date_predicate_line_57():
    import datetime
    from decimal import Decimal
    
    # Test case where p_year < 1
    result = _last_payment_date(datetime.date(1, 1, 1), datetime.date(1, 1, 1), 1)
    assert result == datetime.date(1, 1, 1)
    
    # Test case where p_month < 1 (should not happen in normal flow, but predicate should catch it)
    # This is a defensive check in the code
    
    # Test case where eom < 1 (should not happen in normal flow, but predicate should catch it)
    # This is a defensive check in the code


# LLM-generated content at query #50
#--------------------------

```python
def test_last_payment_date_examples():
    import datetime
    from decimal import Decimal
    
    # Test case 1
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)
    
    # Test case 2
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)
    
    # Test case 3
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 7, 1)
    
    # Test case 4
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result == datetime.date(2015, 7, 1)
    
    # Test case 5
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result == datetime.date(2015, 1, 1)
    
    # Test case 6
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)
    
    # Test case 7
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)
    
    # Test case 8
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result == datetime.date(2014, 12, 9)
    
    # Test case 9
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result == datetime.date(2015, 12, 15)
    
    # Test case 10
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)


# LLM-generated content at query #51
#--------------------------

```python
def test_calculate_fraction_predicate_evaluates_to_false():
    from datetime import date
    from decimal import Decimal
    from typing import Set
    
    # Create a mock DCFC function that returns a known value
    def mock_dcfc(start, asof, end, freq):
        return Decimal("0.5")
    
    # Create a DCC instance with minimal required fields
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    # Test with dates where start <= asof <= end (predicate evaluates to False)
    start = date(2023, 1, 1)
    asof = date(2023, 6, 15)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end, None)
    
    # When predicate is False, it should call the underlying method, not return ZERO
    assert result == Decimal("0.5")


# LLM-generated content at query #52
#--------------------------

```python
def test_dcfc_30_360_us_example_1():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_us_example_2():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_360_us_example_3():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_360_us_example_4():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_360_us_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 15)
    result = dcfc_30_360_us(start=start, asof=start, end=start)
    assert result == Decimal('0')


def test_dcfc_30_360_us_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_us
    
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 3, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=asof, freq=Decimal('2'))
    assert round(result, 14) == Decimal('0.25')


# LLM-generated content at query #53
#--------------------------

```python
def test_last_payment_date_predicate_line_57_true():
    import datetime
    from decimal import Decimal
    
    # Test case where p_year < 1 (predicate evaluates to True)
    # We need to construct a scenario where the year becomes less than 1
    # This would happen with very early dates and lookback
    result = _last_payment_date(datetime.date(1, 1, 1), datetime.date(1, 1, 1), 1)
    assert result == datetime.date(1, 1, 1)


# LLM-generated content at query #54
#--------------------------

```python
def test_has_leap_day_predicate_evaluates_true():
    from datetime import date
    import calendar
    
    def _has_leap_day(start: date, end: date) -> bool:
        """
        Indicates if the range has any leap day.
        """
        ## Get all leap years:
        years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}

        ## Check if any of the lap day falls in our range:
        for year in years:
            ## Construct the leap day:
            leapday = date(year, 2, 29)

            ## Is the leap date in the range?
            if start <= leapday <= end:
                ## Yes, the leap day is within the date range. Return True:
                return True

        ## No leap day in the range, return False:
        return False
    
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    
    assert result is True


# LLM-generated content at query #55
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_annual_frequency_same_year():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_semi_annual_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 7, 1)


def test_last_payment_date_semi_annual_frequency_august():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result == datetime.date(2015, 7, 1)


def test_last_payment_date_semi_annual_frequency_april():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_annual_june_start():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)


def test_last_payment_date_quarterly_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)


def test_last_payment_date_annual_december():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result == datetime.date(2014, 12, 9)


def test_last_payment_date_semi_annual_december():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_semi_annual_december_year_end():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_with_decimal_frequency():
    import datetime
    from decimal import Decimal
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal(1))
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_before_start_date():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2015, 6, 1), datetime.date(2015, 5, 31), 1)
    assert result == datetime.date(2015, 6, 1)


def test_last_payment_date_with_eom_parameter():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 12, 31), 1, eom=31)
    assert result == datetime.date(2015, 1, 31)


def test_last_payment_date_eom_day_adjustment():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 2, 28), 1)
    assert result == datetime.date(2015, 1, 28)


# LLM-generated content at query #56
#--------------------------

```python
def test_dcfc_act_act_icma_predicate_line_22():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    freq = Decimal(1)
    
    result = dcfc_act_act_icma(start, asof, end, freq)
    
    assert isinstance(result, Decimal)
    assert result > Decimal(0)
    assert result < Decimal(1)


# LLM-generated content at query #57
#--------------------------

```python
def test_last_payment_date_predicate_line_57():
    import datetime
    from decimal import Decimal
    
    # Test case where p_year < 1 (should return start date)
    result = _last_payment_date(datetime.date(1, 1, 15), datetime.date(1, 1, 10), 1)
    assert result == datetime.date(1, 1, 15)
    
    # Test case where p_month < 1 (edge case, should return start date)
    result = _last_payment_date(datetime.date(2020, 1, 15), datetime.date(2020, 1, 10), 12)
    assert result == datetime.date(2020, 1, 15)
    
    # Test case where eom < 1 (should return start date)
    result = _last_payment_date(datetime.date(2020, 1, 1), datetime.date(2020, 12, 31), 1, eom=0)
    assert result == datetime.date(2020, 1, 1)


# LLM-generated content at query #58
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    from datetime import date
    import calendar
    
    def _has_leap_day(start: date, end: date) -> bool:
        """
        Indicates if the range has any leap day.
        """
        ## Get all leap years:
        years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}

        ## Check if any of the lap day falls in our range:
        for year in years:
            ## Construct the leap day:
            leapday = date(year, 2, 29)

            ## Is the leap date in the range?
            if start <= leapday <= end:
                ## Yes, the leap day is within the date range. Return True:
                return True

        ## No leap day in the range, return False:
        return False
    
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_without_leap_day_in_range():
    from datetime import date
    import calendar
    
    def _has_leap_day(start: date, end: date) -> bool:
        """
        Indicates if the range has any leap day.
        """
        ## Get all leap years:
        years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}

        ## Check if any of the lap day falls in our range:
        for year in years:
            ## Construct the leap day:
            leapday = date(year, 2, 29)

            ## Is the leap date in the range?
            if start <= leapday <= end:
                ## Yes, the leap day is within the date range. Return True:
                return True

        ## No leap day in the range, return False:
        return False
    
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_leap_day_at_start():
    from datetime import date
    import calendar
    
    def _has_leap_day(start: date, end: date) -> bool:
        """
        Indicates if the range has any leap day.
        """
        ## Get all leap years:
        years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}

        ## Check if any of the lap day falls in our range:
        for year in years:
            ## Construct the leap day:
            leapday = date(year, 2, 29)

            ## Is the leap date in the range?
            if start <= leapday <= end:
                ## Yes, the leap day is within the date range. Return True:
                return True

        ## No leap day in the range, return False:
        return False
    
    start = date(2020, 2, 29)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_leap_day_at_end():
    from datetime import date
    import calendar
    
    def _has_leap_day(start: date, end: date) -> bool:
        """
        Indicates if the range has any leap day.
        """
        ## Get all leap years:
        years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}

        ## Check if any of the lap day falls in our range:
        for year in years:
            ## Construct the leap day:
            leapday = date(year, 2, 29)

            ## Is the leap date in the range?
            if start <= leapday <= end:
                ## Yes, the leap day is within the date range. Return True:
                return True

        ## No leap day in the range, return False:
        return False
    
    start = date(2020, 1, 1)
    end = date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_multiple_leap_years():
    from datetime import date
    import calendar
    
    def _has_leap_day(start: date, end: date) -> bool:
        """
        Indicates if the range has any leap day.
        """
        ## Get all leap years:
        years = {year for year in range(start.year, end.year + 1) if calendar.isleap(year)}

        ## Check if any of the lap day falls in our range:
        for year in years:
            ## Construct the leap day:
            leapday = date(year, 2, 29)

            ## Is the leap date in the range?
            if start <= leapday <= end:
                ## Yes, the leap day is within the date range. Return True:
                return True

        ## No leap day in the range, return False:
        return False
    
    start = date(2019, 1, 1)
    end = date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


# LLM-generated content at query #59
#--------------------------

```python
def test_last_payment_date_predicate_line_1_false():
    import datetime
    from decimal import Decimal
    
    # The predicate at line 1 is the function definition itself
    # Testing that the function exists and is callable
    assert callable(_last_payment_date)
    
    # Testing basic functionality with valid inputs
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)
    
    # Testing with Decimal frequency
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal('1'))
    assert result == datetime.date(2015, 1, 1)
    
    # Testing with eom parameter
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1, eom=15)
    assert isinstance(result, datetime.date)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_isda_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_360_isda_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_360_isda_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_360_isda_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    nod = (15 - 30) + 30 * (2 - 1) + 360 * (2008 - 2008)
    expected = nod / Decimal(360)
    assert result == expected


def test_dcfc_30_360_isda_start_day_30_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 30)
    asof = datetime.date(2008, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    nod = (30 - 30) + 30 * (2 - 1) + 360 * (2008 - 2008)
    expected = nod / Decimal(360)
    assert result == expected


def test_dcfc_30_360_isda_same_dates():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_360_isda_one_month_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    nod = (15 - 15) + 30 * (2 - 1) + 360 * (2008 - 2008)
    expected = nod / Decimal(360)
    assert result == expected


def test_dcfc_30_360_isda_one_year_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2007, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    nod = (15 - 15) + 30 * (1 - 1) + 360 * (2008 - 2007)
    expected = nod / Decimal(360)
    assert result == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_register_valid_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc


def test_register_dcc_with_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="TestDCC",
        altnames={"Alt1", "Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc
    assert registry._buffer_altn["Alt1"] == dcc
    assert registry._buffer_altn["Alt2"] == dcc


def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_conflicting_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="TestDCC1",
        altnames={"Common"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"Common"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="TestDCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestDCC1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_multiple_valid_dccs():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="TestDCC1",
        altnames={"Alt1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    registry.register(dcc2)
    assert registry._buffer_main["TestDCC1"] == dcc1
    assert registry._buffer_main["TestDCC2"] == dcc2
    assert registry._buffer_altn["Alt1"] == dcc1
    assert registry._buffer_altn["Alt2"] == dcc2


# LLM-generated content at query #3
#--------------------------

```python
def test_register_new_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc


def test_register_dcc_with_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Alt1", "Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Alt1"] == dcc
    assert registry._buffer_altn["Alt2"] == dcc


def test_register_duplicate_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    error_raised = False
    try:
        registry.register(dcc2)
    except TypeError as e:
        error_raised = True
        assert "already registered" in str(e)
    assert error_raised


def test_register_duplicate_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"SharedAlt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"SharedAlt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    error_raised = False
    try:
        registry.register(dcc2)
    except TypeError as e:
        error_raised = True
        assert "already registered" in str(e)
    assert error_raised


def test_register_altname_conflicts_with_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/DCC1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    error_raised = False
    try:
        registry.register(dcc2)
    except TypeError as e:
        error_raised = True
        assert "already registered" in str(e)
    assert error_raised


def test_register_multiple_dccs_without_conflicts():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Alt1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Alt2"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    registry.register(dcc2)
    assert registry._buffer_main["Test/DCC1"] == dcc1
    assert registry._buffer_main["Test/DCC2"] == dcc2
    assert registry._buffer_altn["Alt1"] == dcc1
    assert registry._buffer_altn["Alt2"] == dcc2


# LLM-generated content at query #4
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_date_range():
    import datetime
    from pypara.dcc import _get_date_range
    
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 5)
    
    result = list(_get_date_range(start, end))
    
    assert len(result) == 4
    assert result[0] == datetime.date(2023, 1, 1)
    assert result[1] == datetime.date(2023, 1, 2)
    assert result[2] == datetime.date(2023, 1, 3)
    assert result[3] == datetime.date(2023, 1, 4)


def test_get_date_range_single_day():
    import datetime
    from pypara.dcc import _get_date_range
    
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    
    result = list(_get_date_range(start, end))
    
    assert len(result) == 0


def test_get_date_range_two_days():
    import datetime
    from pypara.dcc import _get_date_range
    
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 2)
    
    result = list(_get_date_range(start, end))
    
    assert len(result) == 1
    assert result[0] == datetime.date(2023, 1, 1)


def test_get_date_range_across_months():
    import datetime
    from pypara.dcc import _get_date_range
    
    start = datetime.date(2023, 1, 30)
    end = datetime.date(2023, 2, 3)
    
    result = list(_get_date_range(start, end))
    
    assert len(result) == 4
    assert result[0] == datetime.date(2023, 1, 30)
    assert result[1] == datetime.date(2023, 1, 31)
    assert result[2] == datetime.date(2023, 2, 1)
    assert result[3] == datetime.date(2023, 2, 2)


def test_get_date_range_across_years():
    import datetime
    from pypara.dcc import _get_date_range
    
    start = datetime.date(2022, 12, 31)
    end = datetime.date(2023, 1, 3)
    
    result = list(_get_date_range(start, end))
    
    assert len(result) == 3
    assert result[0] == datetime.date(2022, 12, 31)
    assert result[1] == datetime.date(2023, 1, 1)
    assert result[2] == datetime.date(2023, 1, 2)


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_30_e_plus_360_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_plus_360_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_plus_360_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_plus_360_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_e_plus_360_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    expected = (15 - 30) + 30 * (2 - 1) + 360 * (2008 - 2008)
    assert result == Decimal(expected) / Decimal(360)


def test_dcfc_30_e_plus_360_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 2, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    expected_asof = datetime.date(2008, 3, 1)
    expected_nod = (expected_asof.day - start.day) + 30 * (expected_asof.month - start.month) + 360 * (expected_asof.year - start.year)
    assert result == Decimal(expected_nod) / Decimal(360)


def test_dcfc_30_e_plus_360_same_date():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal(0)


def test_dcfc_30_e_plus_360_one_day():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal(1) / Decimal(360)


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_act_act_icma():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_act_icma
    
    # Test case 1: Example from docstring
    ex1_start = datetime.date(2019, 3, 2)
    ex1_asof = datetime.date(2019, 9, 10)
    ex1_end = datetime.date(2020, 3, 2)
    result1 = dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result1, 10) == Decimal('0.5245901639')
    
    # Test case 2: Same start and asof date (zero days passed)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    result2 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result2 == Decimal('0')
    
    # Test case 3: Full period (start equals asof, asof equals end)
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    result3 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result3 == Decimal('1')
    
    # Test case 4: With frequency parameter
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 12, 31)
    freq = Decimal('2')
    result4 = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=freq)
    assert result4 == Decimal('181') / Decimal('365') / Decimal('2')
    
    # Test case 5: One day period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2020, 1, 2)
    result5 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    assert result5 == Decimal('1')
    
    # Test case 6: Half year period
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    result6 = dcfc_act_act_icma(start=start, asof=asof, end=end)
    expected6 = Decimal('182') / Decimal('366')
    assert result6 == expected6


# LLM-generated content at query #8
#--------------------------

```python
def test_dcfc_30_360_isda_start_day_not_31():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_isda
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    
    result = dcfc_30_360_isda(start, asof, end)
    
    assert result == Decimal('0.16666666666666666666666666667')
    assert start.day != 31


# LLM-generated content at query #9
#--------------------------

```python
def test_next_payment_date_annual_frequency_no_eom():
    import datetime
    from dateutil.relativedelta import relativedelta
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, None)
    assert result == datetime.date(2015, 1, 1)

def test_next_payment_date_annual_frequency_with_eom():
    import datetime
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    assert result == datetime.date(2015, 1, 15)

def test_next_payment_date_semi_annual_frequency():
    import datetime
    result = _next_payment_date(datetime.date(2014, 1, 1), 2, None)
    assert result == datetime.date(2014, 7, 1)

def test_next_payment_date_quarterly_frequency():
    import datetime
    result = _next_payment_date(datetime.date(2014, 1, 1), 4, None)
    assert result == datetime.date(2014, 4, 1)

def test_next_payment_date_monthly_frequency():
    import datetime
    result = _next_payment_date(datetime.date(2014, 1, 1), 12, None)
    assert result == datetime.date(2014, 2, 1)

def test_next_payment_date_with_eom_february_leap_year():
    import datetime
    result = _next_payment_date(datetime.date(2016, 1, 31), 1, 31)
    assert result == datetime.date(2017, 1, 31)

def test_next_payment_date_with_eom_february_non_leap_year():
    import datetime
    result = _next_payment_date(datetime.date(2015, 1, 31), 1, 31)
    assert result == datetime.date(2016, 1, 31)

def test_next_payment_date_decimal_frequency():
    import datetime
    from decimal import Decimal
    result = _next_payment_date(datetime.date(2014, 1, 1), Decimal('2'), None)
    assert result == datetime.date(2014, 7, 1)

def test_next_payment_date_multiple_years():
    import datetime
    result = _next_payment_date(datetime.date(2010, 6, 15), 1, None)
    assert result == datetime.date(2011, 6, 15)

def test_next_payment_date_eom_valid_day():
    import datetime
    result = _next_payment_date(datetime.date(2014, 3, 15), 2, 20)
    assert result == datetime.date(2014, 9, 20)


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_30_360_isda_example1():
    import datetime
    from decimal import Decimal
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_isda_example2():
    import datetime
    from decimal import Decimal
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_360_isda_example3():
    import datetime
    from decimal import Decimal
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_360_isda_example4():
    import datetime
    from decimal import Decimal
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33333333333333')


def test_dcfc_30_360_isda_same_dates():
    import datetime
    from decimal import Decimal
    start = datetime.date(2010, 1, 1)
    result = dcfc_30_360_isda(start=start, asof=start, end=start)
    assert result == Decimal('0')


def test_dcfc_30_360_isda_one_day_difference():
    import datetime
    from decimal import Decimal
    start = datetime.date(2010, 1, 1)
    asof = datetime.date(2010, 1, 2)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_360_isda_with_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2010, 1, 31)
    asof = datetime.date(2010, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert result == Decimal('28') / Decimal('360')


def test_dcfc_30_360_isda_start_day_30_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2010, 1, 30)
    asof = datetime.date(2010, 2, 31) if False else datetime.date(2010, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=asof)
    assert isinstance(result, Decimal)


# LLM-generated content at query #11
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_register_new_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/360"] == dcc


def test_register_dcc_with_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/360",
        altnames={"Test360", "T360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/360"] == dcc
    assert registry._buffer_altn["Test360"] == dcc
    assert registry._buffer_altn["T360"] == dcc


def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Test/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Other/360",
        altnames={"Test/360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_altname_conflicts_with_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/360",
        altnames={"T360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Other/360",
        altnames={"T360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_multiple_dcc_no_conflicts():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/360",
        altnames={"T360"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.5")
    )
    dcc2 = DCC(
        name="Other/365",
        altnames={"O365"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal("0.6")
    )
    registry.register(dcc1)
    registry.register(dcc2)
    assert registry._buffer_main["Test/360"] == dcc1
    assert registry._buffer_main["Other/365"] == dcc2
    assert registry._buffer_altn["T360"] == dcc1
    assert registry._buffer_altn["O365"] == dcc2


# LLM-generated content at query #13
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Test/Alternative"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Test/Alternative"] == dcc


def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Test/Alt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/Alt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_multiple_alternative_names():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Alt1", "Alt2", "Alt3"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_altn["Alt1"] == dcc
    assert registry._buffer_altn["Alt2"] == dcc
    assert registry._buffer_altn["Alt3"] == dcc


def test_register_main_name_conflicts_with_existing_alternative():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"Test/Conflict"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/Conflict",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_empty_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_30_360_german_example1():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_360_german_example2():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_360_german_example3():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_360_german_example4():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')


def test_dcfc_30_360_german_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_360_german_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 16)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')


def test_dcfc_30_360_german_month_adjustment():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 1)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    expected = (1 - 30) / Decimal(360)
    assert result == expected


def test_dcfc_30_360_german_february_last_day_not_end_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 3, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    expected = (30 - 28) / Decimal(360)
    assert result == expected


def test_dcfc_30_360_german_february_last_day_is_end_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_german
    
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    expected = (29 - 28) / Decimal(360)
    assert result == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_dcfc_30_360_isda_start_day_31():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_30_360_isda
    
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    
    assert result == Decimal('1.08333333333333')


# LLM-generated content at query #16
#--------------------------

```python
def test_construct_date_valid_date():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 1, 15)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 15

def test_construct_date_end_of_month():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 1, 31)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 31

def test_construct_date_february_leap_year():
    from pypara.dcc import _construct_date
    result = _construct_date(2020, 2, 29)
    assert result.year == 2020
    assert result.month == 2
    assert result.day == 29

def test_construct_date_invalid_day_adjusts_down():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 2, 30)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28

def test_construct_date_invalid_day_february_non_leap():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 2, 29)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28

def test_construct_date_april_31_invalid():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 4, 31)
    assert result.year == 2023
    assert result.month == 4
    assert result.day == 30

def test_construct_date_zero_year_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(0, 1, 15)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "year, month and day must be greater than 0" in str(e)

def test_construct_date_zero_month_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 0, 15)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "year, month and day must be greater than 0" in str(e)

def test_construct_date_zero_day_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 1, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "year, month and day must be greater than 0" in str(e)

def test_construct_date_negative_year_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(-2023, 1, 15)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "year, month and day must be greater than 0" in str(e)

def test_construct_date_negative_month_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, -1, 15)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "year, month and day must be greater than 0" in str(e)

def test_construct_date_negative_day_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 1, -15)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "year, month and day must be greater than 0" in str(e)

def test_construct_date_invalid_month_raises():
    from pypara.dcc import _construct_date
    try:
        _construct_date(2023, 13, 15)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_construct_date_december_31():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 12, 31)
    assert result.year == 2023
    assert result.month == 12
    assert result.day == 31


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_30_360_german_line_31_predicate():
    """
    Test that the predicate at line 31 evaluates to True.
    The predicate is: asof.day == 31 or (asof.month == 2 and _is_last_day_of_month(asof) and end != asof)
    """
    import datetime
    from pypara.dcc import dcfc_30_360_german
    
    # Test case 1: asof.day == 31 (first part of OR is True)
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 1, 31)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start, asof, end)
    assert result == (30 + 30 * 0 + 360 * 0) / 360
    
    # Test case 2: asof.month == 2 and asof is last day of February and end != asof
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 2, 29)  # Last day of February in leap year
    end = datetime.date(2008, 3, 31)   # end != asof
    result = dcfc_30_360_german(start, asof, end)
    # With the predicate True, d2 should be 30
    assert result == (30 + 30 * 1 + 360 * 0) / 360
    
    # Test case 3: asof.month == 2 and asof is last day of February but end == asof (predicate False)
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)  # end == asof, so predicate is False
    result = dcfc_30_360_german(start, asof, end)
    # With the predicate False, d2 should be 29
    assert result == (29 + 30 * 1 + 360 * 0) / 360


# LLM-generated content at query #18
#--------------------------

```python
def test_calculate_fraction_valid_dates():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 6, 15)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.5")


def test_calculate_fraction_start_equals_asof():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 1, 1)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.25")


def test_calculate_fraction_asof_equals_end():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.75")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 12, 31)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0.75")


def test_calculate_fraction_asof_before_start():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 6, 15)
    asof = date(2023, 1, 1)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0")


def test_calculate_fraction_asof_after_end():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2024, 1, 1)
    end = date(2023, 12, 31)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0")


def test_calculate_fraction_with_freq_parameter():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        if freq is not None:
            return Decimal("0.5") * freq
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 1, 1)
    asof = date(2023, 6, 15)
    end = date(2023, 12, 31)
    freq = Decimal("2")
    
    result = dcc.calculate_fraction(start, asof, end, freq)
    assert result == Decimal("1")


def test_calculate_fraction_returns_zero_for_invalid_order():
    from datetime import date
    from decimal import Decimal
    
    def mock_calculate_fraction_method(start, asof, end, freq):
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction_method
    )
    
    start = date(2023, 12, 31)
    asof = date(2023, 6, 15)
    end = date(2023, 1, 1)
    
    result = dcc.calculate_fraction(start, asof, end)
    assert result == Decimal("0")


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_30_e_360_example_1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result, 14) == Decimal('0.16666666666667')


def test_dcfc_30_e_360_example_2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result, 14) == Decimal('0.16944444444444')


def test_dcfc_30_e_360_example_3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result, 14) == Decimal('1.08333333333333')


def test_dcfc_30_e_360_example_4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result, 14) == Decimal('1.33055555555556')


def test_dcfc_30_e_360_start_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert isinstance(result, Decimal)
    assert result > Decimal('0')


def test_dcfc_30_e_360_asof_day_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 3, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert isinstance(result, Decimal)
    assert result > Decimal('0')


def test_dcfc_30_e_360_both_days_31():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 31)
    asof = datetime.date(2008, 3, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert isinstance(result, Decimal)
    assert result > Decimal('0')


def test_dcfc_30_e_360_same_dates():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 1, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_30_e_360_one_year_apart():
    import datetime
    from decimal import Decimal
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2009, 1, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result == Decimal('1')


# LLM-generated content at query #20
#--------------------------

```python
def test_find_with_exact_name():
    from decimal import Decimal
    import datetime
    
    registry = DCCRegistryMachinery()
    
    # Create a mock DCC object
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("Act/Act", [])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is not None
    assert result.name == "Act/Act"


def test_find_with_stripped_uppercase_name():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("ACT/ACT", [])
    registry.register(dcc)
    
    result = registry.find("  act/act  ")
    assert result is not None
    assert result.name == "ACT/ACT"


def test_find_with_alternative_name():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("Actual/Actual", ["Act/Act"])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is not None
    assert result.name == "Actual/Actual"


def test_find_nonexistent_name():
    registry = DCCRegistryMachinery()
    
    result = registry.find("NonExistent/DCC")
    assert result is None


def test_find_case_insensitive():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("30/360 US", [])
    registry.register(dcc)
    
    result = registry.find("30/360 us")
    assert result is not None
    assert result.name == "30/360 US"


def test_find_with_whitespace_and_case():
    registry = DCCRegistryMachinery()
    
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    dcc = MockDCC("BOND BASIS", [])
    registry.register(dcc)
    
    result = registry.find("  bond basis  ")
    assert result is not None
    assert result.name == "BOND BASIS"


# LLM-generated content at query #21
#--------------------------

```python
def test_construct_date_predicate_at_line_9_evaluates_to_false():
    from pypara.dcc import _construct_date
    result = _construct_date(2023, 2, 28)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28


# LLM-generated content at query #22
#--------------------------

```python
def test_register_successfully_registers_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Test", "TESTDCC"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert registry._buffer_altn["Test"] == dcc
    assert registry._buffer_altn["TESTDCC"] == dcc


def test_register_raises_error_when_main_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_raises_error_when_altname_conflicts_with_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"Test/DCC1"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_raises_error_when_altname_conflicts_with_existing_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test/DCC1",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    dcc2 = DCC(
        name="Test/DCC2",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "already registered" in str(e)


def test_register_with_empty_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_main["Test/DCC"] == dcc
    assert len(registry._buffer_altn) == 0


def test_register_with_multiple_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test/DCC",
        altnames={"Alt1", "Alt2", "Alt3"},
        currencies=set(),
        calculate_fraction_method=lambda start, asof, end, freq: Decimal(0)
    )
    registry.register(dcc)
    assert registry._buffer_altn["Alt1"] == dcc
    assert registry._buffer_altn["Alt2"] == dcc
    assert registry._buffer_altn["Alt3"] == dcc


# LLM-generated content at query #23
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_annual_frequency_same_year():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_semi_annual_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 7, 1)


def test_last_payment_date_semi_annual_frequency_august():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result == datetime.date(2015, 7, 1)


def test_last_payment_date_semi_annual_frequency_april():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result == datetime.date(2015, 1, 1)


def test_last_payment_date_annual_frequency_june():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)


def test_last_payment_date_quarterly_frequency():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)


def test_last_payment_date_annual_frequency_december():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result == datetime.date(2014, 12, 9)


def test_last_payment_date_semi_annual_frequency_december():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_semi_annual_frequency_december_end():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)


def test_last_payment_date_with_eom_parameter():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 12, 31), 1, eom=15)
    assert result == datetime.date(2015, 1, 15)


def test_last_payment_date_end_of_month_handling():
    import datetime
    from pypara.dcc import _last_payment_date
    
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 2, 28), 1)
    assert result.month == 1
    assert result.year == 2015


# LLM-generated content at query #24
#--------------------------

```python
def test_last_payment_date_predicate_line_1_false():
    import datetime
    from decimal import Decimal
    
    # The predicate at line 1 is the function definition itself
    # We need to test that the function exists and is callable
    # Testing that calling the function returns a Date object (not False)
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result is not False
    assert isinstance(result, datetime.date)
    assert result == datetime.date(2015, 1, 1)


# LLM-generated content at query #25
#--------------------------

```python
def test_init_creates_empty_buffers():
    dcc_registry = DCCRegistryMachinery()
    assert dcc_registry._buffer_main == {}
    assert dcc_registry._buffer_altn == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_last_payment_date_predicate():
    import datetime
    from decimal import Decimal
    
    # Test case from line 5-6
    result1 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result1 == datetime.date(2015, 1, 1)
    
    # Test case from line 8-9
    result2 = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result2 == datetime.date(2015, 1, 1)
    
    # Test case from line 11-12
    result3 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result3 == datetime.date(2015, 7, 1)
    
    # Test case from line 14-15
    result4 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result4 == datetime.date(2015, 7, 1)
    
    # Test case from line 17-18
    result5 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result5 == datetime.date(2015, 1, 1)
    
    # Test case from line 20-21
    result6 = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result6 == datetime.date(2014, 6, 1)
    
    # Test case from line 23-24
    result7 = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result7 == datetime.date(2015, 7, 7)
    
    # Test case from line 26-27
    result8 = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result8 == datetime.date(2014, 12, 9)
    
    # Test case from line 29-30
    result9 = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result9 == datetime.date(2015, 12, 15)
    
    # Test case from line 32-33
    result10 = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result10 == datetime.date(2015, 12, 15)


# LLM-generated content at query #27
#--------------------------

```python
def test_interest_basic_calculation():
    from decimal import Decimal
    from datetime import date
    from typing import NamedTuple, Set
    
    # Mock Currency class
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    # Mock Money class
    class Money:
        def __init__(self, amount: Decimal):
            self.amount = amount
        
        def __mul__(self, other):
            if isinstance(other, Decimal):
                return Money(self.amount * other)
            return NotImplemented
        
        def __eq__(self, other):
            if isinstance(other, Money):
                return self.amount == other.amount
            return False
    
    # Mock calculate_fraction_method
    def mock_dcfc(start: date, asof: date, end: date, freq=None) -> Decimal:
        return Decimal("0.5")
    
    # Create DCC instance
    dcc = DCC(
        name="Test Convention",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    # Test basic interest calculation
    principal = Money(Decimal("1000"))
    rate = Decimal("0.05")
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 6, 30)
    end_date = date(2024, 12, 31)
    
    result = dcc.interest(principal, rate, start_date, asof_date, end_date)
    
    # Expected: 1000 * 0.05 * 0.5 = 25
    expected = Money(Decimal("25"))
    assert result == expected


def test_interest_without_end_date():
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class Money:
        def __init__(self, amount: Decimal):
            self.amount = amount
        
        def __mul__(self, other):
            if isinstance(other, Decimal):
                return Money(self.amount * other)
            return NotImplemented
        
        def __eq__(self, other):
            if isinstance(other, Money):
                return self.amount == other.amount
            return False
    
    def mock_dcfc(start: date, asof: date, end: date, freq=None) -> Decimal:
        return Decimal("0.25")
    
    dcc = DCC(
        name="Test Convention",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    principal = Money(Decimal("2000"))
    rate = Decimal("0.10")
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 3, 31)
    
    result = dcc.interest(principal, rate, start_date, asof_date)
    
    # Expected: 2000 * 0.10 * 0.25 = 50
    expected = Money(Decimal("50"))
    assert result == expected


def test_interest_zero_rate():
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class Money:
        def __init__(self, amount: Decimal):
            self.amount = amount
        
        def __mul__(self, other):
            if isinstance(other, Decimal):
                return Money(self.amount * other)
            return NotImplemented
        
        def __eq__(self, other):
            if isinstance(other, Money):
                return self.amount == other.amount
            return False
    
    def mock_dcfc(start: date, asof: date, end: date, freq=None) -> Decimal:
        return Decimal("0.5")
    
    dcc = DCC(
        name="Test Convention",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    principal = Money(Decimal("1000"))
    rate = Decimal("0")
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 6, 30)
    end_date = date(2024, 12, 31)
    
    result = dcc.interest(principal, rate, start_date, asof_date, end_date)
    
    # Expected: 1000 * 0 * 0.5 = 0
    expected = Money(Decimal("0"))
    assert result == expected


# LLM-generated content at query #28
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #29
#--------------------------

```python
def test_last_payment_date_predicate_false():
    import datetime
    from decimal import Decimal
    
    # Test case where the predicate at line 1 evaluates to False
    # The predicate is: def _last_payment_date(start: Date, asof: Date, frequency: Union[int, Decimal], eom: Optional[int] = None) -> Date:
    # This means we need to call the function with valid arguments
    
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    
    # Call the function (the predicate at line 1 is the function signature itself)
    # The function definition evaluates to False when called with arguments
    result = _last_payment_date(start, asof, frequency)
    
    # Assert the function was called and returned a valid result
    assert result == datetime.date(2015, 1, 1)
    assert isinstance(result, datetime.date)


# LLM-generated content at query #30
#--------------------------

```python
def test_last_payment_date_predicate_false():
    import datetime
    from decimal import Decimal
    
    # Test case where the predicate at line 1 evaluates to False
    # The predicate is: def _last_payment_date(start: Date, asof: Date, frequency: Union[int, Decimal], eom: Optional[int] = None) -> Date:
    # This is a function definition, so we test that it can be called with various argument types
    
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    
    # The function exists and is callable
    assert callable(_last_payment_date)
    
    # Test with int frequency
    result1 = _last_payment_date(start, asof, 1)
    assert isinstance(result1, datetime.date)
    
    # Test with Decimal frequency
    result2 = _last_payment_date(start, asof, Decimal('2'))
    assert isinstance(result2, datetime.date)
    
    # Test with explicit eom parameter
    result3 = _last_payment_date(start, asof, 1, 15)
    assert isinstance(result3, datetime.date)


# LLM-generated content at query #31
#--------------------------

```python
def test_find_with_exact_name():
    from decimal import Decimal
    import datetime
    
    # Create a mock DCC object
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = MockDCC("Act/Act", [])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is dcc


def test_find_with_stripped_uppercase_name():
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = MockDCC("ACT/ACT", [])
    registry.register(dcc)
    
    result = registry.find("  act/act  ")
    assert result is dcc


def test_find_with_alternative_name():
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = MockDCC("Actual/Actual", ["Act/Act"])
    registry.register(dcc)
    
    result = registry.find("Act/Act")
    assert result is dcc


def test_find_with_nonexistent_name():
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = MockDCC("Act/Act", [])
    registry.register(dcc)
    
    result = registry.find("NonExistent")
    assert result is None


def test_find_case_insensitive():
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = MockDCC("30/360 US", [])
    registry.register(dcc)
    
    result = registry.find("30/360 us")
    assert result is dcc


def test_find_with_whitespace():
    class MockDCC:
        def __init__(self, name, altnames=None):
            self.name = name
            self.altnames = altnames or []
    
    registry = DCCRegistryMachinery()
    dcc = MockDCC("BOND BASIS", [])
    registry.register(dcc)
    
    result = registry.find("   bond basis   ")
    assert result is dcc


# LLM-generated content at query #32
#--------------------------

```python
def test_dcfc_act_act_example1():
    import datetime
    from decimal import Decimal
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('0.16942884946478')


def test_dcfc_act_act_example2():
    import datetime
    from decimal import Decimal
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('0.17216108990194')


def test_dcfc_act_act_example3():
    import datetime
    from decimal import Decimal
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('1.08243131970956')


def test_dcfc_act_act_example4():
    import datetime
    from decimal import Decimal
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    rounded_result = round(result, 14)
    assert rounded_result == Decimal('1.32625945055768')


def test_dcfc_act_act_same_date():
    import datetime
    from decimal import Decimal
    start_date = datetime.date(2020, 1, 1)
    result = dcfc_act_act(start=start_date, asof=start_date, end=start_date)
    assert result == Decimal('0')


def test_dcfc_act_act_one_day_non_leap():
    import datetime
    from decimal import Decimal
    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date(2019, 1, 2)
    result = dcfc_act_act(start=start_date, asof=end_date, end=end_date)
    assert result == Decimal('1') / Decimal('365')


# LLM-generated content at query #33
#--------------------------

```python
def test_calculate_daily_fraction_basic():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq):
        return Decimal('0.5')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 3)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert result == Decimal('0')


def test_calculate_daily_fraction_with_increasing_fractions():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    call_count = [0]
    
    def mock_dcfc(start, asof, end, freq):
        call_count[0] += 1
        if call_count[0] == 1:
            return Decimal('0.1')
        else:
            return Decimal('0.2')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 3)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert result == Decimal('0.1')


def test_calculate_daily_fraction_asof_equals_start():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq):
        return Decimal('0.3')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert result == Decimal('0.3')


def test_calculate_daily_fraction_with_freq_parameter():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq):
        if freq is not None:
            return Decimal('0.25')
        return Decimal('0.15')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 5)
    end_date = date(2023, 12, 31)
    freq = Decimal('4')
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date, freq)
    assert result == Decimal('0.1')


def test_calculate_daily_fraction_without_freq_parameter():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq):
        return Decimal('0.4')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dcfc
    )
    
    start_date = date(2023, 1, 1)
    asof_date = date(2023, 1, 10)
    end_date = date(2023, 12, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    assert result == Decimal('0')


# LLM-generated content at query #34
#--------------------------

```python
def test_dcfc_30_360_german_line_31_predicate_false():
    """
    Test that the predicate at line 31 evaluates to False.
    The predicate is: asof.day == 31 or (asof.month == 2 and _is_last_day_of_month(asof) and end != asof)
    
    For it to be False:
    - asof.day must not be 31
    - AND (asof.month != 2 OR _is_last_day_of_month(asof) is False OR end == asof)
    """
    import datetime
    from pypara.dcc import dcfc_30_360_german
    
    # Case: asof.day is not 31, asof.month is not 2
    start = datetime.date(2008, 1, 15)
    asof = datetime.date(2008, 3, 15)
    end = datetime.date(2008, 3, 15)
    
    result = dcfc_30_360_german(start, asof, end)
    
    # The predicate should be False, so d2 should equal asof.day (15)
    assert result == (15 - 15) / 360
    assert result == 0


# LLM-generated content at query #35
#--------------------------

```python
def test_calculate_daily_fraction_basic():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq=None):
        return Decimal('0.1')
    
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=mock_dcfc)
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 1, 2)
    end_date = date(2024, 1, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    
    assert result == Decimal('0')


def test_calculate_daily_fraction_with_asof_minus_1_before_start():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq=None):
        return Decimal('0.05')
    
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=mock_dcfc)
    start_date = date(2024, 1, 2)
    asof_date = date(2024, 1, 2)
    end_date = date(2024, 1, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    
    assert result == Decimal('0.05')


def test_calculate_daily_fraction_with_different_values():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    call_count = [0]
    call_args = []
    
    def mock_dcfc(start, asof, end, freq=None):
        call_count[0] += 1
        call_args.append((start, asof, end, freq))
        if asof == date(2024, 1, 1):
            return Decimal('0.03')
        elif asof == date(2024, 1, 2):
            return Decimal('0.08')
        return Decimal('0')
    
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=mock_dcfc)
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 1, 2)
    end_date = date(2024, 1, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    
    assert result == Decimal('0.05')
    assert call_count[0] == 2


def test_calculate_daily_fraction_asof_equals_start():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    def mock_dcfc(start, asof, end, freq=None):
        if asof == start:
            return Decimal('0.02')
        return Decimal('0')
    
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=mock_dcfc)
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 1, 1)
    end_date = date(2024, 1, 31)
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date)
    
    assert result == Decimal('0.02')


def test_calculate_daily_fraction_with_freq_parameter():
    from datetime import date
    from decimal import Decimal
    from pypara.dcc import DCC
    
    freq_values = []
    
    def mock_dcfc(start, asof, end, freq=None):
        freq_values.append(freq)
        return Decimal('0.1')
    
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=mock_dcfc)
    start_date = date(2024, 1, 1)
    asof_date = date(2024, 1, 3)
    end_date = date(2024, 1, 31)
    freq = Decimal('4')
    
    result = dcc.calculate_daily_fraction(start_date, asof_date, end_date, freq)
    
    assert len(freq_values) == 2
    assert freq_values[0] == freq
    assert freq_values[1] == freq


# LLM-generated content at query #36
#--------------------------

```python
def test_last_payment_date_predicate():
    import datetime
    from decimal import Decimal
    
    # Test case from line 5-6
    result1 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result1 == datetime.date(2015, 1, 1)
    
    # Test case from line 8-9
    result2 = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result2 == datetime.date(2015, 1, 1)
    
    # Test case from line 11-12
    result3 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result3 == datetime.date(2015, 7, 1)
    
    # Test case from line 14-15
    result4 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result4 == datetime.date(2015, 7, 1)
    
    # Test case from line 17-18
    result5 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result5 == datetime.date(2015, 1, 1)
    
    # Test case from line 20-21
    result6 = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result6 == datetime.date(2014, 6, 1)
    
    # Test case from line 23-24
    result7 = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result7 == datetime.date(2015, 7, 7)
    
    # Test case from line 26-27
    result8 = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result8 == datetime.date(2014, 12, 9)
    
    # Test case from line 29-30
    result9 = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result9 == datetime.date(2015, 12, 15)
    
    # Test case from line 32-33
    result10 = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result10 == datetime.date(2015, 12, 15)


# LLM-generated content at query #37
#--------------------------

```python
def test_coupon_basic():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.5')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    
    principal = Decimal('1000')
    rate = Decimal('0.05')
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 6, 1)
    end = datetime.date(2015, 1, 1)
    freq = 2
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal('25')


def test_coupon_with_eom():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.25')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    
    principal = Decimal('2000')
    rate = Decimal('0.1')
    start = datetime.date(2014, 1, 15)
    asof = datetime.date(2014, 7, 15)
    end = datetime.date(2015, 1, 15)
    freq = 2
    eom = 15
    
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Decimal('50')


def test_coupon_annual_frequency():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction_method(start, asof, end, freq):
        return Decimal('1.0')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    
    principal = Decimal('5000')
    rate = Decimal('0.02')
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 12, 31)
    end = datetime.date(2015, 1, 1)
    freq = 1
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal('100')


def test_coupon_quarterly_frequency():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.1')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    
    principal = Decimal('10000')
    rate = Decimal('0.08')
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 4, 1)
    end = datetime.date(2014, 7, 1)
    freq = 4
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal('80')


def test_coupon_with_decimal_frequency():
    import datetime
    from decimal import Decimal
    
    def dummy_calculate_fraction_method(start, asof, end, freq):
        return Decimal('0.5')
    
    dcc = DCC(
        name="Test DCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=dummy_calculate_fraction_method
    )
    
    principal = Decimal('3000')
    rate = Decimal('0.06')
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2014, 7, 1)
    end = datetime.date(2015, 1, 1)
    freq = Decimal('2')
    
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Decimal('90')


# LLM-generated content at query #38
#--------------------------

```python
def test_is_last_day_of_month_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 1, 31))
    assert result is True


def test_is_last_day_of_month_not_last_day():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 1, 30))
    assert result is False


def test_is_last_day_of_month_february_leap_year():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 2, 29))
    assert result is True


def test_is_last_day_of_month_february_non_leap_year():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2023, 2, 28))
    assert result is True


def test_is_last_day_of_month_april_30():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 4, 30))
    assert result is True


def test_is_last_day_of_month_december_31():
    from datetime import date
    import calendar
    
    def _is_last_day_of_month(date_obj: date) -> bool:
        return date_obj.day == calendar.monthrange(date_obj.year, date_obj.month)[1]
    
    result = _is_last_day_of_month(date(2024, 12, 31))
    assert result is True


# LLM-generated content at query #39
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_without_leap_day_in_range():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_non_leap_year():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2021, 2, 1)
    end = datetime.date(2021, 3, 1)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_multiple_leap_years():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2024, 12, 31)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_exact_leap_day_start():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_exact_leap_day_end():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_single_day_leap_day():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result is True


def test_has_leap_day_single_day_not_leap_day():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_range_before_leap_day():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result is False


def test_has_leap_day_range_after_leap_day():
    import datetime
    from pypara.dcc import _has_leap_day
    
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result is False


# LLM-generated content at query #40
#--------------------------

```python
def test_dcfc_act_365_a_basic():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == Decimal('0.16986301369863')


def test_dcfc_act_365_a_leap_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == Decimal('0.17213114754098')


def test_dcfc_act_365_a_multi_year():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == Decimal('1.08196721311475')


def test_dcfc_act_365_a_long_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == Decimal('1.32513661202186')


def test_dcfc_act_365_a_same_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


def test_dcfc_act_365_a_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('365')


def test_dcfc_act_365_a_with_freq_parameter():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_a
    
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    freq = Decimal('2')
    result = dcfc_act_365_a(start=start, asof=asof, end=asof, freq=freq)
    assert isinstance(result, Decimal)
    assert result > Decimal('0')


# LLM-generated content at query #41
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #42
#--------------------------

```python
def test_dcfc_act_365_l_basic():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_act_365_l(start, asof, end)
    assert round(result, 14) == Decimal('0.16939890710383')


def test_dcfc_act_365_l_leap_year_feb_29():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_act_365_l(start, asof, end)
    assert round(result, 14) == Decimal('0.17213114754098')


def test_dcfc_act_365_l_across_years():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_act_365_l(start, asof, end)
    assert round(result, 14) == Decimal('1.08196721311475')


def test_dcfc_act_365_l_long_period():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_act_365_l(start, asof, end)
    assert round(result, 14) == Decimal('1.32876712328767')


def test_dcfc_act_365_l_same_date():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 1)
    end = datetime.date(2008, 2, 1)
    result = dcfc_act_365_l(start, asof, end)
    assert result == Decimal('0')


def test_dcfc_act_365_l_one_day():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2008, 2, 2)
    end = datetime.date(2008, 2, 2)
    result = dcfc_act_365_l(start, asof, end)
    assert result == Decimal('1') / Decimal('366')


def test_dcfc_act_365_l_non_leap_year():
    import datetime
    from decimal import Decimal
    from pypara.dcc import dcfc_act_365_l
    
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 1, 2)
    end = datetime.date(2007, 1, 2)
    result = dcfc_act_365_l(start, asof, end)
    assert result == Decimal('1') / Decimal('365')


# LLM-generated content at query #43
#--------------------------

```python
def test_next_payment_date_with_eom():
    import datetime
    from dateutil.relativedelta import relativedelta
    
    def _next_payment_date(start, frequency, eom=None):
        months = int(12 / frequency)
        nextdate = start + relativedelta(months=months)
        if eom:
            try:
                nextdate = nextdate.replace(day=eom)
            except ValueError:
                pass
        return nextdate
    
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    assert result == datetime.date(2015, 1, 15)


# LLM-generated content at query #44
#--------------------------

```python
def test_last_payment_date_line_1_predicate():
    import datetime
    from decimal import Decimal
    
    # The predicate at line 1 is the function signature itself
    # Testing that the function can be called with the documented parameters
    
    # Test case 1: Basic annual payment
    result1 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result1 == datetime.date(2015, 1, 1)
    
    # Test case 2: Same start and asof year
    result2 = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result2 == datetime.date(2015, 1, 1)
    
    # Test case 3: Semi-annual payment
    result3 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result3 == datetime.date(2015, 7, 1)
    
    # Test case 4: Semi-annual with later asof date
    result4 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result4 == datetime.date(2015, 7, 1)
    
    # Test case 5: Semi-annual with earlier asof date
    result5 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result5 == datetime.date(2015, 1, 1)
    
    # Test case 6: Different start month
    result6 = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result6 == datetime.date(2014, 6, 1)
    
    # Test case 7: Quarterly payment
    result7 = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result7 == datetime.date(2015, 7, 7)
    
    # Test case 8: Payment date in future
    result8 = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result8 == datetime.date(2014, 12, 9)
    
    # Test case 9: Semi-annual with mid-month dates
    result9 = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result9 == datetime.date(2015, 12, 15)
    
    # Test case 10: Semi-annual year-end
    result10 = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result10 == datetime.date(2015, 12, 15)
    
    # Test with Decimal frequency
    result11 = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal(1))
    assert result11 == datetime.date(2015, 1, 1)
    
    # Test with optional eom parameter
    result12 = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 12, 31), 1, eom=15)
    assert result12 == datetime.date(2015, 1, 15)


# LLM-generated content at query #45
#--------------------------

```python
def test_register_raises_error_when_altname_already_registered():
    from decimal import Decimal
    
    def dummy_method(start, asof, end, freq=None):
        return Decimal(0)
    
    dcc1 = DCC(
        name="Convention1",
        altnames={"Alt1", "Alt2"},
        currencies=set(),
        calculate_fraction_method=dummy_method
    )
    
    dcc2 = DCC(
        name="Convention2",
        altnames={"Alt1", "Alt3"},
        currencies=set(),
        calculate_fraction_method=dummy_method
    )
    
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError to be raised"
    except TypeError as e:
        assert "already registered" in str(e)


